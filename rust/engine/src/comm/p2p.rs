//! Peer-to-peer communicator for machines without NCCL / RCCL: host-memory collectives for control and host slices, and a device-pointer
//! all-reduce built on the custom one-shot kernel (`oneshot.rs`) that reads the peers' staging buffers directly (peer access over PCIe,
//! NVLink or Infinity Fabric). Only the all-reduce exists on the device path; the other device collectives need NCCL / RCCL and report so.
use super::cuda_ffi::{set_device, RankStream};
use super::oneshot::CustomAllReduce;
use super::{CommResult, Communicator, DType, DeviceCollectives, DevPtr, StreamHandle};
use std::sync::Arc;

pub struct P2pComm {
    rank: usize,
    size: usize,
    device: usize,
    stream: RankStream,
    host: Arc<dyn Communicator>,
    reducer: CustomAllReduce,
}

impl P2pComm {
    pub fn connect(host: &Arc<dyn Communicator>, device: usize) -> CommResult<Arc<P2pComm>> {
        let (rank, size) = (host.rank(), host.size());
        set_device(device)?;
        let reducer = CustomAllReduce::connect(host, device, super::oneshot::max_bytes_from_env())?;
        let stream = RankStream::new(device)?;
        Ok(Arc::new(P2pComm { rank, size, device, stream, host: host.clone(), reducer }))
    }

    pub fn device(&self) -> usize {
        self.device
    }

    pub fn stream(&self) -> &RankStream {
        &self.stream
    }
}

const ONLY_ALL_REDUCE: &str = "the peer-to-peer communicator implements only the device all-reduce; use --distributed-backend nccl (NCCL / RCCL) for the other device collectives";

impl DeviceCollectives for P2pComm {
    fn all_reduce(&self, buf: DevPtr, count: usize, dtype: DType, stream: StreamHandle) -> CommResult<()> {
        if self.size <= 1 || count == 0 {
            return Ok(());
        }
        // messages larger than the staging buffer go through it in pieces
        let per_chunk = (self.reducer.max_bytes() / dtype.size()).max(1);
        let mut done = 0usize;
        while done < count {
            let n = per_chunk.min(count - done);
            self.reducer.all_reduce(buf + (done * dtype.size()) as u64, n, dtype, stream)?;
            done += n;
        }
        Ok(())
    }
    fn all_gather(&self, _: DevPtr, _: DevPtr, _: usize, _: DType, _: StreamHandle) -> CommResult<()> {
        Err(ONLY_ALL_REDUCE.into())
    }
    fn reduce_scatter(&self, _: DevPtr, _: DevPtr, _: usize, _: DType, _: StreamHandle) -> CommResult<()> {
        Err(ONLY_ALL_REDUCE.into())
    }
    fn broadcast(&self, _: DevPtr, _: usize, _: DType, _: usize, _: StreamHandle) -> CommResult<()> {
        Err(ONLY_ALL_REDUCE.into())
    }
    fn all_to_all(&self, _: DevPtr, _: &[usize], _: DevPtr, _: &[usize], _: DType, _: StreamHandle) -> CommResult<()> {
        Err(ONLY_ALL_REDUCE.into())
    }
    fn send(&self, _: DevPtr, _: usize, _: DType, _: usize, _: StreamHandle) -> CommResult<()> {
        Err(ONLY_ALL_REDUCE.into())
    }
    fn recv(&self, _: DevPtr, _: usize, _: DType, _: usize, _: StreamHandle) -> CommResult<()> {
        Err(ONLY_ALL_REDUCE.into())
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
        let sub = self.host.split(color, key)?;
        Ok(P2pComm::connect(&sub, self.device)? as Arc<dyn Communicator>)
    }
    fn device(&self) -> Option<&dyn DeviceCollectives> {
        Some(self)
    }
    fn abort(&self) {
        self.host.abort();
    }
}
