//! Custom one-shot all-reduce over peer-mapped memory for the small messages of a decode step (`cuda/comm_kernels.cu`,
//! `fllm_comm_ar_oneshot`), the same idea as the custom all-reduce of vLLM and the one-shot strategy of TensorRT-LLM: every rank stages its
//! input in a buffer all peers can read, signals the peers with a flag, waits for their flags and sums the staged inputs in rank order, in a
//! single kernel launch. Larger messages belong to NCCL / RCCL (ring or tree); this kernel is for latency.
//!
//! All ranks must be threads of one process on one node (the staging pointers are exchanged as raw addresses and read through peer access).
//! Ranks that share a physical GPU also work (used by the self test: it exercises the flag protocol without a second GPU).
use super::cuda_ffi::{device_sync, enable_peer_access, fllm_comm_ar_oneshot, set_device, DeviceMem};
use super::{CommResult, Communicator, DType, DevPtr, StreamHandle};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Most ranks the kernel supports (the parameter block is a fixed array).
pub const MAX_RANKS: usize = 16;
/// Most thread blocks of one launch (`FLLM_AR_MAX_BLOCKS` in the kernel); each has a flag per peer.
const MAX_BLOCKS: usize = 8;
/// Default capacity of one message in bytes (`FLLM_CUSTOM_ALLREDUCE_MAX_BYTES`). Above it the caller uses NCCL / RCCL.
pub const DEFAULT_MAX_BYTES: usize = 1 << 20;

/// Whether `FLLM_CUSTOM_ALLREDUCE` asks for the custom kernel: `1`/`on` (yes), anything else (no, the default until it has been measured on
/// a multi-GPU node).
pub fn enabled_by_env() -> bool {
    matches!(std::env::var("FLLM_CUSTOM_ALLREDUCE").ok().as_deref().map(str::trim), Some("1" | "on" | "true" | "yes"))
}

pub fn max_bytes_from_env() -> usize {
    std::env::var("FLLM_CUSTOM_ALLREDUCE_MAX_BYTES").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|v| *v >= 4096).unwrap_or(DEFAULT_MAX_BYTES)
}

pub struct CustomAllReduce {
    rank: usize,
    size: usize,
    device: usize,
    cap: usize,
    /// Two halves of `cap` bytes.
    _staging: DeviceMem,
    _flags: DeviceMem,
    staging_ptrs: Vec<usize>,
    flag_ptrs: Vec<usize>,
    calls: AtomicU64,
}

// SAFETY: raw device addresses; the kernel launches are ordered by the caller's stream and the calls are SPMD.
unsafe impl Send for CustomAllReduce {}
unsafe impl Sync for CustomAllReduce {}

fn put(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

impl CustomAllReduce {
    /// Collective over the ranks of `host`. Fails (the caller then keeps NCCL) when a peer cannot be reached by peer access.
    pub fn connect(host: &Arc<dyn Communicator>, device: usize, cap_bytes: usize) -> CommResult<Self> {
        let (rank, size) = (host.rank(), host.size());
        if size > MAX_RANKS {
            return Err(format!("the custom all-reduce supports up to {MAX_RANKS} ranks, this group has {size}"));
        }
        let cap = cap_bytes.max(4096).next_multiple_of(16);
        let staging = DeviceMem::new(device, cap * 2)?;
        let flags = DeviceMem::new_flags(device, MAX_BLOCKS * size * 8)?;
        device_sync()?;
        // announce device, staging and flag addresses to everybody (root by root)
        let mut table = Vec::with_capacity(size);
        for root in 0..size {
            let mut msg = Vec::new();
            if root == rank {
                put(&mut msg, device as u64);
                put(&mut msg, staging.addr());
                put(&mut msg, flags.addr());
            }
            host.broadcast_bytes(&mut msg, root)?;
            if msg.len() != 24 {
                return Err("malformed custom all-reduce announcement".into());
            }
            let word = |i: usize| u64::from_le_bytes(msg[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]));
            table.push((word(0) as usize, word(1) as usize, word(2) as usize));
        }
        let mut problem = None;
        for (peer, (pdev, _, _)) in table.iter().enumerate() {
            if peer == rank || *pdev == device {
                continue;
            }
            let mut can = 0;
            // SAFETY: plain query with a valid out pointer.
            unsafe { super::cuda_ffi::fllm_comm_can_access_peer(device as i32, *pdev as i32, &mut can) };
            if can == 0 || enable_peer_access(device, *pdev).is_err() {
                problem = Some(format!("device {device} cannot access device {pdev} (peer access)"));
            }
        }
        // everybody must agree, or one rank would launch a kernel its peers never match
        let mut verdict = [if problem.is_some() { 1.0f32 } else { 0.0 }];
        host.all_reduce(&mut verdict)?;
        if verdict[0] > 0.0 {
            return Err(problem.unwrap_or_else(|| "a peer of this group has no peer access to another device".into()));
        }
        host.barrier()?;
        Ok(Self {
            rank,
            size,
            device,
            cap,
            _staging: staging,
            _flags: flags,
            staging_ptrs: table.iter().map(|t| t.1).collect(),
            flag_ptrs: table.iter().map(|t| t.2).collect(),
            calls: AtomicU64::new(0),
        })
    }

    pub fn max_bytes(&self) -> usize {
        self.cap
    }

    /// Whether a message of `count` elements goes through the kernel.
    pub fn handles(&self, count: usize, dtype: DType) -> bool {
        count > 0 && dtype.kernel_code().is_ok() && count * dtype.size() <= self.cap
    }

    /// In-place sum of `count` elements at `buf` (a device address on this rank's device), ordered on `stream`.
    pub fn all_reduce(&self, buf: DevPtr, count: usize, dtype: DType, stream: StreamHandle) -> CommResult<()> {
        if !self.handles(count, dtype) {
            return Err(format!("custom all-reduce: {count} x {dtype:?} does not fit {} bytes", self.cap));
        }
        let code = dtype.kernel_code()?;
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        let staging: Vec<*mut c_void> = self.staging_ptrs.iter().map(|p| *p as *mut c_void).collect();
        let flags: Vec<*mut u64> = self.flag_ptrs.iter().map(|p| *p as *mut u64).collect();
        set_device(self.device)?;
        // SAFETY: the tables hold `size` valid device addresses (peer access was checked in `connect`); `buf` holds `count` elements.
        let rc = unsafe {
            fllm_comm_ar_oneshot(
                staging.as_ptr(),
                flags.as_ptr(),
                self.rank as i32,
                self.size as i32,
                buf as *mut c_void,
                count,
                code,
                call + 1,
                (call as usize & 1) * self.cap,
                stream as *mut c_void,
            )
        };
        super::cuda_ffi::check("custom all-reduce", rc)
    }
}

impl Drop for CustomAllReduce {
    fn drop(&mut self) {
        // Peers may still read this rank's staging memory for the very last call; at the end of a run every rank has issued the same calls, so
        // waiting for this device's own work is enough. (A barrier here would hang when the other ranks are already gone, as after an abort.)
        let _ = set_device(self.device);
        let _ = device_sync();
    }
}
