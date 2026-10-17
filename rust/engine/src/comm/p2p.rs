//! Peer-to-peer communicator for machines without NCCL (cudaMemcpyPeerAsync + reduce kernels).
//! Uses direct peer copies over PCIe / NVLink when peer access is supported, or host-staged copies.
use super::cuda_ffi::{
    custom_all_reduce_f32, enable_peer_access, peer_copy_async, reduce_add_f32, set_device,
    DeviceMem, RankStream,
};
use super::{
    CommResult, Communicator, DType, DeviceCollectives, DevPtr, StreamHandle,
};
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

pub struct P2pComm {
    rank: usize,
    size: usize,
    device: usize,
    stream: RankStream,
    host: Arc<dyn Communicator>,
    peer_devices: Vec<usize>,
    peer_access: Vec<bool>,
    staging: Mutex<Option<DeviceMem>>,
}

// SAFETY: pointers are GPU addresses synchronized through streams and barriers.
unsafe impl Send for P2pComm {}
unsafe impl Sync for P2pComm {}

impl P2pComm {
    pub fn connect(host: &Arc<dyn Communicator>, device: usize) -> CommResult<Arc<P2pComm>> {
        let (rank, size) = (host.rank(), host.size());
        set_device(device)?;

        // Exchange device ordinals across ranks
        let dev_info = vec![device as f32];
        let mut all_devs = vec![0.0f32; size];
        host.all_gather(&dev_info, &mut all_devs)?;
        let peer_devices: Vec<usize> = all_devs.iter().map(|&d| d as usize).collect();

        // Query and enable peer access where available
        let mut peer_access = vec![false; size];
        for (peer, &pdev) in peer_devices.iter().enumerate() {
            if peer == rank {
                peer_access[peer] = true;
                continue;
            }
            let mut can: std::ffi::c_int = 0;
            let _ = unsafe { super::cuda_ffi::fllm_comm_can_access_peer(device as i32, pdev as i32, &mut can) };
            if can != 0 && enable_peer_access(device, pdev).is_ok() {
                peer_access[peer] = true;
            }
        }

        let stream = RankStream::new(device)?;
        Ok(Arc::new(P2pComm {
            rank,
            size,
            device,
            stream,
            host: host.clone(),
            peer_devices,
            peer_access,
            staging: Mutex::new(None),
        }))
    }

    pub fn device(&self) -> usize {
        self.device
    }

    pub fn stream(&self) -> &RankStream {
        &self.stream
    }

    fn ensure_staging(&self, bytes: usize) -> CommResult<()> {
        let mut guard = self.staging.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.as_ref().is_none_or(|m| m.bytes < bytes) {
            *guard = Some(DeviceMem::new(self.device, bytes.next_power_of_two())?);
        }
        Ok(())
    }
}

impl DeviceCollectives for P2pComm {
    fn all_reduce(&self, buf: DevPtr, count: usize, dtype: DType, stream: StreamHandle) -> CommResult<()> {
        if self.size <= 1 || count == 0 {
            return Ok(());
        }
        if dtype != DType::F32 {
            return Err("P2pComm device all_reduce currently supports F32".into());
        }
        let bytes = count * dtype.size();
        self.ensure_staging(bytes)?;

        // Broadcast current buffer pointer to peers via host
        let ptr_f32 = buf as f32;
        let mut all_ptrs_f32 = vec![0.0f32; self.size];
        self.host.all_gather(&[ptr_f32], &mut all_ptrs_f32)?;
        let peer_ptrs: Vec<*const f32> = all_ptrs_f32.iter().map(|&p| (p as u64) as *const f32).collect();

        let all_p2p = self.peer_access.iter().all(|&can| can);
        if all_p2p {
            // Fast direct multi-peer custom all-reduce kernel
            custom_all_reduce_f32(buf as *mut f32, &peer_ptrs, count, stream as *mut c_void)?;
        } else {
            // Fallback: pairwise peer copies + reduce_add
            set_device(self.device)?;
            let guard = self.staging.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let staging_ptr = guard.as_ref().unwrap().ptr;
            #[allow(clippy::needless_range_loop)]
            for peer in 0..self.size {
                if peer == self.rank {
                    continue;
                }
                peer_copy_async(
                    staging_ptr,
                    self.device,
                    peer_ptrs[peer] as *const c_void,
                    self.peer_devices[peer],
                    bytes,
                    stream as *mut c_void,
                )?;
                reduce_add_f32(buf as *mut f32, staging_ptr as *const f32, count, stream as *mut c_void)?;
            }
        }
        Ok(())
    }

    fn all_gather(&self, send: DevPtr, recv: DevPtr, count: usize, dtype: DType, stream: StreamHandle) -> CommResult<()> {
        let width = dtype.size() as u64;
        let bytes = count * dtype.size();
        // Copy own slice
        unsafe {
            super::cuda_ffi::fllm_comm_peer_copy_async(
                (recv + self.rank as u64 * count as u64 * width) as *mut c_void,
                self.device as i32,
                send as *const c_void,
                self.device as i32,
                bytes,
                stream as *mut c_void,
            );
        }
        for peer in 0..self.size {
            if peer == self.rank {
                continue;
            }
            peer_copy_async(
                (recv + peer as u64 * count as u64 * width) as *mut c_void,
                self.device,
                send as *const c_void,
                self.peer_devices[peer],
                bytes,
                stream as *mut c_void,
            )?;
        }
        Ok(())
    }

    fn reduce_scatter(&self, send: DevPtr, recv: DevPtr, count: usize, dtype: DType, stream: StreamHandle) -> CommResult<()> {
        let width = dtype.size() as u64;
        let bytes = count * dtype.size();
        let my_offset = self.rank as u64 * count as u64 * width;
        // Copy own portion
        unsafe {
            super::cuda_ffi::fllm_comm_peer_copy_async(
                recv as *mut c_void,
                self.device as i32,
                (send + my_offset) as *const c_void,
                self.device as i32,
                bytes,
                stream as *mut c_void,
            );
        }
        self.ensure_staging(bytes)?;
        let guard = self.staging.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let staging_ptr = guard.as_ref().unwrap().ptr;
        for peer in 0..self.size {
            if peer == self.rank {
                continue;
            }
            peer_copy_async(
                staging_ptr,
                self.device,
                (send + my_offset) as *const c_void,
                self.peer_devices[peer],
                bytes,
                stream as *mut c_void,
            )?;
            reduce_add_f32(recv as *mut f32, staging_ptr as *const f32, count, stream as *mut c_void)?;
        }
        Ok(())
    }

    fn broadcast(&self, buf: DevPtr, count: usize, dtype: DType, root: usize, stream: StreamHandle) -> CommResult<()> {
        if self.size <= 1 || count == 0 {
            return Ok(());
        }
        let bytes = count * dtype.size();
        if self.rank != root {
            let mut root_ptr = vec![0.0f32];
            self.host.broadcast(&mut root_ptr, root)?;
            let src_ptr = (root_ptr[0] as u64) as *const c_void;
            peer_copy_async(
                buf as *mut c_void,
                self.device,
                src_ptr,
                self.peer_devices[root],
                bytes,
                stream as *mut c_void,
            )?;
        } else {
            self.host.broadcast(&mut [buf as f32], root)?;
        }
        Ok(())
    }

    fn all_to_all(
        &self,
        send: DevPtr,
        send_counts: &[usize],
        recv: DevPtr,
        recv_counts: &[usize],
        dtype: DType,
        stream: StreamHandle,
    ) -> CommResult<()> {
        let width = dtype.size() as u64;
        let mut so = 0u64;
        let mut ro = 0u64;
        for peer in 0..self.size {
            let sc = send_counts[peer];
            let rc = recv_counts[peer];
            if peer == self.rank {
                if sc > 0 {
                    unsafe {
                        super::cuda_ffi::fllm_comm_peer_copy_async(
                            (recv + ro * width) as *mut c_void,
                            self.device as i32,
                            (send + so * width) as *const c_void,
                            self.device as i32,
                            sc * dtype.size(),
                            stream as *mut c_void,
                        );
                    }
                }
            } else {
                if sc > 0 {
                    peer_copy_async(
                        (recv + ro * width) as *mut c_void,
                        self.device,
                        (send + so * width) as *const c_void,
                        self.peer_devices[peer],
                        sc * dtype.size(),
                        stream as *mut c_void,
                    )?;
                }
            }
            so += sc as u64;
            ro += rc as u64;
        }
        Ok(())
    }

    fn send(&self, buf: DevPtr, _count: usize, _dtype: DType, peer: usize, _stream: StreamHandle) -> CommResult<()> {
        self.host.send_bytes(&buf.to_le_bytes(), peer)?;
        let mut ack = [0.0f32];
        self.host.recv(&mut ack, peer)?;
        Ok(())
    }

    fn recv(&self, buf: DevPtr, count: usize, dtype: DType, peer: usize, stream: StreamHandle) -> CommResult<()> {
        let bytes = count * dtype.size();
        let raw = self.host.recv_bytes(peer)?;
        let src_buf = u64::from_le_bytes(raw.try_into().map_err(|_| "invalid address")?);
        peer_copy_async(
            buf as *mut c_void,
            self.device,
            src_buf as *const c_void,
            self.peer_devices[peer],
            bytes,
            stream as *mut c_void,
        )?;
        self.host.send(&[1.0], peer)?;
        Ok(())
    }
}

impl Communicator for P2pComm {
    fn rank(&self) -> usize {
        self.rank
    }
    fn size(&self) -> usize {
        self.size
    }
    fn backend(&self) -> &'static str {
        "p2p"
    }
    fn all_reduce(&self, data: &mut [f32]) -> CommResult<()> {
        self.host.all_reduce(data)
    }
    fn all_gather(&self, send: &[f32], recv: &mut [f32]) -> CommResult<()> {
        self.host.all_gather(send, recv)
    }
    fn reduce_scatter(&self, send: &[f32], recv: &mut [f32]) -> CommResult<()> {
        self.host.reduce_scatter(send, recv)
    }
    fn broadcast(&self, data: &mut [f32], root: usize) -> CommResult<()> {
        self.host.broadcast(data, root)
    }
    fn all_to_all(&self, send: &[f32], send_counts: &[usize], recv: &mut [f32], recv_counts: &[usize]) -> CommResult<()> {
        self.host.all_to_all(send, send_counts, recv, recv_counts)
    }
    fn send(&self, data: &[f32], peer: usize) -> CommResult<()> {
        self.host.send(data, peer)
    }
    fn recv(&self, data: &mut [f32], peer: usize) -> CommResult<()> {
        self.host.recv(data, peer)
    }
    fn barrier(&self) -> CommResult<()> {
        self.host.barrier()
    }
    fn broadcast_bytes(&self, data: &mut Vec<u8>, root: usize) -> CommResult<()> {
        self.host.broadcast_bytes(data, root)
    }
    fn send_bytes(&self, data: &[u8], peer: usize) -> CommResult<()> {
        self.host.send_bytes(data, peer)
    }
    fn recv_bytes(&self, peer: usize) -> CommResult<Vec<u8>> {
        self.host.recv_bytes(peer)
    }
    fn exchange_counts(&self, send_counts: &[usize]) -> CommResult<Vec<usize>> {
        self.host.exchange_counts(send_counts)
    }
    fn split(&self, color: usize, key: usize) -> CommResult<Arc<dyn Communicator>> {
        self.host.split(color, key)
    }
}
