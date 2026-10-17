//! Collectives built from point-to-point messages over a [`Fabric`]. Every rank sends its contribution to the others and reduces
//! locally in rank order, so all ranks compute identical bits. This is the algorithm of the in-process `LocalComm`, of the socket
//! transport used between nodes, and of the control plane (step descriptors, token ids) next to the device back ends. It is
//! O(size) messages per rank; the device back ends use NCCL or ring algorithms for bulk data.
use super::fabric::Fabric;
use super::{bytes_to_f32s, decode_usizes, encode_usizes, f32s_to_bytes, CommResult, Communicator};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub struct MeshComm {
    fabric: Arc<dyn Fabric>,
    /// World ranks of the members, in group order.
    members: Vec<usize>,
    me: usize,
    chan: u32,
    splits: AtomicU32,
    name: &'static str,
}

fn mix(a: u32, b: u32, c: u32) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for word in [a, b, c] {
        for byte in word.to_le_bytes() {
            h ^= u32::from(byte);
            h = h.wrapping_mul(0x0100_0193);
        }
    }
    h | 1
}

impl MeshComm {
    /// The group of all ranks of `fabric`.
    pub fn world(fabric: Arc<dyn Fabric>, name: &'static str) -> Arc<Self> {
        let size = fabric.world_size();
        let me = fabric.world_rank();
        Arc::new(Self { fabric, members: (0..size).collect(), me, chan: 0, splits: AtomicU32::new(0), name })
    }

    fn peer(&self, rank: usize) -> CommResult<usize> {
        self.members.get(rank).copied().ok_or_else(|| format!("rank {rank} is outside the group of {}", self.members.len()))
    }

    fn put(&self, rank: usize, data: Vec<u8>) -> CommResult<()> {
        self.fabric.send(self.peer(rank)?, self.chan, data)
    }

    fn take(&self, rank: usize) -> CommResult<Vec<u8>> {
        self.fabric.recv(self.peer(rank)?, self.chan)
    }

    /// Send `bytes` to every other rank, then collect what the others sent (own slot holds `bytes`).
    fn exchange_all(&self, bytes: Vec<u8>) -> CommResult<Vec<Vec<u8>>> {
        let n = self.members.len();
        for r in 0..n {
            if r != self.me {
                self.put(r, bytes.clone())?;
            }
        }
        let mut all = Vec::with_capacity(n);
        for r in 0..n {
            all.push(if r == self.me { bytes.clone() } else { self.take(r)? });
        }
        Ok(all)
    }

    fn sum_in_rank_order(parts: &[Vec<f32>], out: &mut [f32]) {
        out.copy_from_slice(&parts[0]);
        for part in &parts[1..] {
            for (o, p) in out.iter_mut().zip(part) {
                *o += p;
            }
        }
    }
}

impl Communicator for MeshComm {
    fn rank(&self) -> usize {
        self.me
    }
    fn size(&self) -> usize {
        self.members.len()
    }
    fn backend(&self) -> &'static str {
        self.name
    }

    fn all_reduce(&self, data: &mut [f32]) -> CommResult<()> {
        if self.members.len() == 1 {
            return Ok(());
        }
        let raw = self.exchange_all(f32s_to_bytes(data))?;
        let mut parts = Vec::with_capacity(raw.len());
        for bytes in &raw {
            let mut part = vec![0.0f32; data.len()];
            bytes_to_f32s(bytes, &mut part)?;
            parts.push(part);
        }
        Self::sum_in_rank_order(&parts, data);
        Ok(())
    }

    fn all_gather(&self, send: &[f32], recv: &mut [f32]) -> CommResult<()> {
        let n = self.members.len();
        if recv.len() != n * send.len() {
            return Err(format!("all_gather: receive buffer of {} floats, expected {} x {}", recv.len(), n, send.len()));
        }
        let raw = self.exchange_all(f32s_to_bytes(send))?;
        for (r, bytes) in raw.iter().enumerate() {
            bytes_to_f32s(bytes, &mut recv[r * send.len()..(r + 1) * send.len()])?;
        }
        Ok(())
    }

    fn reduce_scatter(&self, send: &[f32], recv: &mut [f32]) -> CommResult<()> {
        let n = self.members.len();
        if send.len() != n * recv.len() {
            return Err(format!("reduce_scatter: send buffer of {} floats, expected {} x {}", send.len(), n, recv.len()));
        }
        let block = recv.len();
        for r in 0..n {
            if r != self.me {
                self.put(r, f32s_to_bytes(&send[r * block..(r + 1) * block]))?;
            }
        }
        let mut parts = Vec::with_capacity(n);
        for r in 0..n {
            let mut part = vec![0.0f32; block];
            if r == self.me {
                part.copy_from_slice(&send[r * block..(r + 1) * block]);
            } else {
                bytes_to_f32s(&self.take(r)?, &mut part)?;
            }
            parts.push(part);
        }
        Self::sum_in_rank_order(&parts, recv);
        Ok(())
    }

    fn broadcast(&self, data: &mut [f32], root: usize) -> CommResult<()> {
        let n = self.members.len();
        if root >= n {
            return Err(format!("broadcast root {root} is outside the group of {n}"));
        }
        if self.me == root {
            let bytes = f32s_to_bytes(data);
            for r in 0..n {
                if r != root {
                    self.put(r, bytes.clone())?;
                }
            }
            Ok(())
        } else {
            let bytes = self.take(root)?;
            bytes_to_f32s(&bytes, data)
        }
    }

    fn all_to_all(&self, send: &[f32], send_counts: &[usize], recv: &mut [f32], recv_counts: &[usize]) -> CommResult<()> {
        let n = self.members.len();
        if send_counts.len() != n || recv_counts.len() != n {
            return Err(format!("all_to_all needs {n} counts per direction"));
        }
        if send_counts.iter().sum::<usize>() != send.len() || recv_counts.iter().sum::<usize>() != recv.len() {
            return Err("all_to_all: the counts do not add up to the buffer sizes".into());
        }
        let mut offset = 0;
        let mut own = Vec::new();
        for (r, &count) in send_counts.iter().enumerate() {
            let chunk = &send[offset..offset + count];
            if r == self.me {
                own = chunk.to_vec();
            } else {
                self.put(r, f32s_to_bytes(chunk))?;
            }
            offset += count;
        }
        let mut offset = 0;
        for (r, &count) in recv_counts.iter().enumerate() {
            let dst = &mut recv[offset..offset + count];
            if r == self.me {
                if own.len() != count {
                    return Err(format!("all_to_all: rank {r} sends itself {} floats but expects {count}", own.len()));
                }
                dst.copy_from_slice(&own);
            } else {
                bytes_to_f32s(&self.take(r)?, dst).map_err(|e| format!("all_to_all from rank {r}: {e}"))?;
            }
            offset += count;
        }
        Ok(())
    }

    fn send(&self, data: &[f32], peer: usize) -> CommResult<()> {
        self.put(peer, f32s_to_bytes(data))
    }

    fn recv(&self, data: &mut [f32], peer: usize) -> CommResult<()> {
        let bytes = self.take(peer)?;
        bytes_to_f32s(&bytes, data)
    }

    fn barrier(&self) -> CommResult<()> {
        if self.members.len() > 1 {
            self.exchange_all(Vec::new())?;
        }
        Ok(())
    }

    fn broadcast_bytes(&self, data: &mut Vec<u8>, root: usize) -> CommResult<()> {
        let n = self.members.len();
        if root >= n {
            return Err(format!("broadcast root {root} is outside the group of {n}"));
        }
        if self.me == root {
            for r in 0..n {
                if r != root {
                    self.put(r, data.clone())?;
                }
            }
        } else {
            *data = self.take(root)?;
        }
        Ok(())
    }

    fn send_bytes(&self, data: &[u8], peer: usize) -> CommResult<()> {
        self.put(peer, data.to_vec())
    }

    fn recv_bytes(&self, peer: usize) -> CommResult<Vec<u8>> {
        self.take(peer)
    }

    fn exchange_counts(&self, send_counts: &[usize]) -> CommResult<Vec<usize>> {
        let n = self.members.len();
        if send_counts.len() != n {
            return Err(format!("exchange_counts needs {n} counts"));
        }
        for (r, &count) in send_counts.iter().enumerate() {
            if r != self.me {
                self.put(r, encode_usizes(&[count]))?;
            }
        }
        let mut result = Vec::with_capacity(n);
        for (r, &own) in send_counts.iter().enumerate() {
            result.push(if r == self.me {
                own
            } else {
                *decode_usizes(&self.take(r)?)?.first().ok_or("empty count message")?
            });
        }
        Ok(result)
    }

    fn abort(&self) {
        self.fabric.abort();
    }

    fn split(&self, color: usize, key: usize) -> CommResult<Arc<dyn Communicator>> {
        let index = self.splits.fetch_add(1, Ordering::Relaxed);
        let raw = self.exchange_all(encode_usizes(&[color, key]))?;
        let mut group: Vec<(usize, usize, usize)> = Vec::new();
        for (rank, bytes) in raw.iter().enumerate() {
            let values = decode_usizes(bytes)?;
            if values.len() != 2 {
                return Err("malformed split message".into());
            }
            if values[0] == color {
                group.push((values[1], rank, self.members[rank]));
            }
        }
        group.sort_unstable();
        let me = group.iter().position(|&(_, rank, _)| rank == self.me).ok_or("split lost this rank")?;
        Ok(Arc::new(MeshComm {
            fabric: self.fabric.clone(),
            members: group.iter().map(|&(_, _, world)| world).collect(),
            me,
            chan: mix(self.chan, index, color as u32),
            splits: AtomicU32::new(0),
            name: self.name,
        }))
    }
}
