//! Collective communication between the ranks of one model instance that is split over several devices
//! (tensor, pipeline and expert parallelism, see `docs/multi-gpu.md`).
//!
//! One process runs one worker thread per device. Every rank owns a [`Communicator`]; all ranks call the
//! same collectives in the same order (SPMD). Three implementations exist:
//!
//! * [`local::LocalComm`] and [`mesh::MeshComm`]: threads and queues in host memory, the reference for tests and the control plane
//!   of the other back ends. Reductions add in rank order, so every rank gets the same bits.
//! * `nccl::NcclComm` (feature `cuda`): libnccl loaded at run time.
//! * `p2p::P2pComm` (feature `cuda`): peer-to-peer copies and small reduce kernels, no NCCL needed.
//!
//! The same trait serves host slices (`f32`) and, through [`DeviceCollectives`], device buffers.

pub mod fabric;
pub mod local;
pub mod mesh;
pub mod selftest;
pub mod tcp;

#[cfg(all(feature = "cuda", not(feature = "hip")))]
pub mod cuda_ffi;
#[cfg(all(feature = "cuda", not(feature = "hip")))]
pub mod nccl;
#[cfg(all(feature = "cuda", not(feature = "hip")))]
pub mod p2p;
#[cfg(all(feature = "cuda", not(feature = "hip")))]
pub mod topology;

use std::sync::Arc;

pub type CommResult<T> = Result<T, String>;

/// Raw device pointer (CUDA `void*`) of a buffer the caller owns.
pub type DevPtr = u64;
/// Raw stream handle (CUDA `cudaStream_t`); 0 is the legacy default stream.
pub type StreamHandle = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Bf16,
    I32,
    U8,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::Bf16 => 2,
            DType::U8 => 1,
        }
    }

    /// `ncclDataType_t`
    pub fn nccl_code(self) -> i32 {
        match self {
            DType::U8 => 1,
            DType::I32 => 2,
            DType::F16 => 6,
            DType::F32 => 7,
            DType::Bf16 => 9,
        }
    }

    /// Code used by the `fllm_comm_*` kernels: 0 = f32, 1 = f16, 2 = bf16.
    pub fn kernel_code(self) -> CommResult<i32> {
        match self {
            DType::F32 => Ok(0),
            DType::F16 => Ok(1),
            DType::Bf16 => Ok(2),
            other => Err(format!("{other:?} has no reduction kernel")),
        }
    }
}

/// Collectives on device buffers. `stream` is the raw stream the operation is ordered on; the call returns once it is enqueued.
/// NCCL calls are capturable in CUDA graphs; the peer-to-peer implementation synchronizes the host and is not.
pub trait DeviceCollectives: Send + Sync {
    /// In-place sum over all ranks of `count` elements.
    fn all_reduce(&self, buf: DevPtr, count: usize, dtype: DType, stream: StreamHandle) -> CommResult<()>;
    /// `recv` holds `size * count` elements, rank order; `send` holds `count`.
    fn all_gather(&self, send: DevPtr, recv: DevPtr, count: usize, dtype: DType, stream: StreamHandle) -> CommResult<()>;
    /// `send` holds `size * count` elements; rank `r` receives the sum of everybody's block `r` into `recv` (`count`).
    fn reduce_scatter(&self, send: DevPtr, recv: DevPtr, count: usize, dtype: DType, stream: StreamHandle) -> CommResult<()>;
    fn broadcast(&self, buf: DevPtr, count: usize, dtype: DType, root: usize, stream: StreamHandle) -> CommResult<()>;
    /// Variable splits (counts in elements); the block for rank `r` starts at the sum of the counts before it.
    fn all_to_all(
        &self,
        send: DevPtr,
        send_counts: &[usize],
        recv: DevPtr,
        recv_counts: &[usize],
        dtype: DType,
        stream: StreamHandle,
    ) -> CommResult<()>;
    fn send(&self, buf: DevPtr, count: usize, dtype: DType, peer: usize, stream: StreamHandle) -> CommResult<()>;
    fn recv(&self, buf: DevPtr, count: usize, dtype: DType, peer: usize, stream: StreamHandle) -> CommResult<()>;
}

/// One rank's endpoint of a group of `size()` ranks. All methods are collective unless said otherwise: every rank of the group must
/// call them in the same order. Host-slice methods work on `f32`; control data (step descriptors, token ids) travels as bytes.
pub trait Communicator: Send + Sync {
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    fn backend(&self) -> &'static str;

    /// Sum over all ranks, in place. The result is bit-identical on every rank.
    fn all_reduce(&self, data: &mut [f32]) -> CommResult<()>;
    /// `recv.len() == size * send.len()`, blocks in rank order.
    fn all_gather(&self, send: &[f32], recv: &mut [f32]) -> CommResult<()>;
    /// `send.len() == size * recv.len()`; rank `r` receives the sum of block `r` of every rank.
    fn reduce_scatter(&self, send: &[f32], recv: &mut [f32]) -> CommResult<()>;
    fn broadcast(&self, data: &mut [f32], root: usize) -> CommResult<()>;
    /// Rank `r` sends `send_counts[r]` elements (consecutive in `send`) to rank `r`; `recv_counts[r]` arrive from rank `r`.
    fn all_to_all(&self, send: &[f32], send_counts: &[usize], recv: &mut [f32], recv_counts: &[usize]) -> CommResult<()>;
    /// Point to point (not collective). `send` never blocks on the receiver.
    fn send(&self, data: &[f32], peer: usize) -> CommResult<()>;
    fn recv(&self, data: &mut [f32], peer: usize) -> CommResult<()>;
    fn barrier(&self) -> CommResult<()>;

    /// Variable-length control data: after the call every rank holds the bytes of `root`.
    fn broadcast_bytes(&self, data: &mut Vec<u8>, root: usize) -> CommResult<()>;
    fn send_bytes(&self, data: &[u8], peer: usize) -> CommResult<()>;
    fn recv_bytes(&self, peer: usize) -> CommResult<Vec<u8>>;
    /// `result[r]` = what rank `r` announced for this rank (the counts needed to size an `all_to_all`).
    fn exchange_counts(&self, send_counts: &[usize]) -> CommResult<Vec<usize>>;

    /// Collective: ranks with the same `color` form a new group, ordered by `key` (ties by rank).
    fn split(&self, color: usize, key: usize) -> CommResult<Arc<dyn Communicator>>;

    /// Collectives on device buffers, where the back end has them.
    fn device(&self) -> Option<&dyn DeviceCollectives> {
        None
    }

    /// A rank failed: wake every rank of the world that waits for a message, with an error.
    fn abort(&self) {}
}

/// Which implementation carries the traffic (`--distributed-backend`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Host threads and queues; works with every compute backend (the CPU simulation and the control plane).
    Local,
    /// libnccl, loaded at run time.
    Nccl,
    /// Peer-to-peer copies and reduction kernels, no NCCL.
    P2p,
}

impl BackendKind {
    pub fn parse(text: &str) -> CommResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "local" | "host" => Ok(BackendKind::Local),
            "nccl" => Ok(BackendKind::Nccl),
            "p2p" => Ok(BackendKind::P2p),
            other => Err(format!("unknown --distributed-backend '{other}' (use nccl, p2p or local)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Local => "local",
            BackendKind::Nccl => "nccl",
            BackendKind::P2p => "p2p",
        }
    }
}

pub(crate) fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub(crate) fn bytes_to_f32s(bytes: &[u8], out: &mut [f32]) -> CommResult<()> {
    if bytes.len() != out.len() * 4 {
        return Err(format!("message of {} bytes does not fit {} floats (ranks disagree on the size)", bytes.len(), out.len()));
    }
    for (dst, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *dst = f32::from_le_bytes(*chunk);
    }
    Ok(())
}

pub(crate) fn encode_usizes(values: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for v in values {
        out.extend_from_slice(&(*v as u64).to_le_bytes());
    }
    out
}

pub(crate) fn decode_usizes(bytes: &[u8]) -> CommResult<Vec<usize>> {
    if !bytes.len().is_multiple_of(8) {
        return Err("malformed count message".into());
    }
    Ok(bytes.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c) as usize).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_parse() {
        assert_eq!(BackendKind::parse("NCCL").unwrap(), BackendKind::Nccl);
        assert_eq!(BackendKind::parse(" p2p ").unwrap(), BackendKind::P2p);
        assert_eq!(BackendKind::parse("local").unwrap(), BackendKind::Local);
        assert!(BackendKind::parse("mpi").is_err());
    }

    #[test]
    fn byte_codecs_round_trip() {
        let v = [1.5f32, -2.25, 0.0, f32::MAX];
        let mut back = [0.0f32; 4];
        bytes_to_f32s(&f32s_to_bytes(&v), &mut back).unwrap();
        assert_eq!(v, back);
        assert!(bytes_to_f32s(&[0u8; 7], &mut back).is_err());
        assert_eq!(decode_usizes(&encode_usizes(&[0, 7, usize::MAX >> 8])).unwrap(), vec![0, 7, usize::MAX >> 8]);
    }
}
