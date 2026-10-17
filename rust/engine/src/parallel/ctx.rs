//! What one rank of a parallel run knows: its place in the tensor / pipeline grid, its communicators, and the step descriptor that
//! rank 0 broadcasts to the others.
use super::config::ParallelConfig;
use crate::comm::Communicator;
use std::ops::Range;
use std::sync::{Arc, Mutex};

/// Errors of the collectives carry this prefix, so that a worker can tell a broken group from a rejected request.
pub const COMM_ERROR: &str = "[comm] ";

/// Turns a host-memory group into the communicator that carries tensors.
pub type DataPlane<'a> = &'a dyn Fn(&Arc<dyn Communicator>) -> Result<Arc<dyn Communicator>, String>;

pub struct ParCtx {
    pub cfg: Arc<ParallelConfig>,
    /// World rank.
    pub rank: usize,
    pub stage: usize,
    pub tp_rank: usize,
    /// Group of the `tp` ranks of this stage; this rank's index in it is `tp_rank`.
    pub tp: Arc<dyn Communicator>,
    /// Group of the ranks with the same `tp_rank` in every stage; this rank's index in it is `stage`.
    pub pp: Arc<dyn Communicator>,
    /// All ranks: step descriptors, logits on their way to rank 0, shutdown.
    pub control: Arc<dyn Communicator>,
    /// Device buffers are split with the GPU alignment rules (Marlin tiles).
    pub gpu: bool,
    /// Rank 0 only: the threads of the other local ranks, joined when the group ends.
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl ParCtx {
    /// Collective over the world: every rank builds its context at the same time. `data_plane` turns a host-memory group into the
    /// communicator that carries tensors (the same group for host-only runs).
    pub fn new(
        cfg: Arc<ParallelConfig>,
        rank: usize,
        control: Arc<dyn Communicator>,
        gpu: bool,
        data_plane: DataPlane<'_>,
    ) -> Result<Self, String> {
        let stage = cfg.stage_of(rank);
        let tp_rank = cfg.tp_rank_of(rank);
        // everybody takes part in both splits, in the same order
        let tp_host = control.split(stage, tp_rank)?;
        let pp_host = control.split(tp_rank, stage)?;
        let tp = if cfg.tp > 1 { data_plane(&tp_host)? } else { tp_host };
        let pp = if cfg.pp > 1 { data_plane(&pp_host)? } else { pp_host };
        Ok(Self { cfg, rank, stage, tp_rank, tp, pp, control, gpu, workers: Mutex::new(Vec::new()) })
    }

    /// Host-memory groups only (the CPU simulation).
    pub fn new_host(cfg: Arc<ParallelConfig>, rank: usize, control: Arc<dyn Communicator>) -> Result<Self, String> {
        Self::new(cfg, rank, control, false, &|group| Ok(group.clone()))
    }

    pub fn attach_workers(&self, handles: Vec<std::thread::JoinHandle<()>>) {
        self.workers.lock().unwrap_or_else(std::sync::PoisonError::into_inner).extend(handles);
    }

    pub fn tp_size(&self) -> usize {
        self.cfg.tp
    }
    pub fn pp_size(&self) -> usize {
        self.cfg.pp
    }
    pub fn is_leader(&self) -> bool {
        self.rank == 0
    }
    pub fn is_first_stage(&self) -> bool {
        self.stage == 0
    }
    pub fn is_last_stage(&self) -> bool {
        self.stage + 1 == self.cfg.pp
    }
    /// The rank that holds the logits of a step (last stage, first tensor rank).
    pub fn output_rank(&self) -> usize {
        (self.cfg.pp - 1) * self.cfg.tp
    }
    pub fn layer_range(&self, layers: usize) -> Range<usize> {
        self.cfg.stage_layers(layers, self.stage)
    }

    /// Sum of the partial results of the tensor-parallel group, in place; a no-op for a group of one.
    pub fn tp_reduce(&self, data: &mut [f32]) -> Result<(), String> {
        if self.cfg.tp > 1 {
            self.tp.all_reduce(data)
        } else {
            Ok(())
        }
    }

    /// Experts `start..end` of `total` held by tensor rank `rank` under expert parallelism: contiguous, the first ranks take the remainder.
    pub fn expert_range(total: usize, tp: usize, rank: usize) -> Range<usize> {
        let base = total / tp;
        let extra = total % tp;
        let start = rank * base + rank.min(extra);
        start..start + base + usize::from(rank < extra)
    }
}

impl Drop for ParCtx {
    fn drop(&mut self) {
        if self.rank != 0 {
            return;
        }
        // end of the group: tell the workers, wake any that is stuck, wait for them
        let mut stop = StepDesc::encode_stop();
        let _ = self.control.broadcast_bytes(&mut stop, 0);
        let handles = std::mem::take(&mut *self.workers.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
        if !handles.is_empty() {
            self.control.abort();
            for handle in handles {
                let _ = handle.join();
            }
        }
    }
}

/// Everything a forward step needs besides the weights and the KV cache.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepDesc {
    pub ids: Vec<i64>,
    pub lengths: Vec<i64>,
    pub tables: Vec<i64>,
    pub logit_rows: Vec<usize>,
    pub embeddings: bool,
    pub overrides: Vec<(usize, Vec<f32>)>,
}

const TAG_STEP: u8 = 1;
const TAG_STOP: u8 = 0;

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_i64s(out: &mut Vec<u8>, values: &[i64]) {
    put_u64(out, values.len() as u64);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len()).ok_or("truncated step descriptor")?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }
    fn u64(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }
    fn count(&mut self, item: usize) -> Result<usize, String> {
        let n = self.u64()? as usize;
        if n.checked_mul(item).is_none_or(|bytes| bytes > self.bytes.len() - self.at) {
            return Err("corrupt step descriptor".into());
        }
        Ok(n)
    }
    fn i64s(&mut self) -> Result<Vec<i64>, String> {
        let n = self.count(8)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.u64()? as i64);
        }
        Ok(out)
    }
}

impl StepDesc {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![TAG_STEP, u8::from(self.embeddings)];
        put_i64s(&mut out, &self.ids);
        put_i64s(&mut out, &self.lengths);
        put_i64s(&mut out, &self.tables);
        put_u64(&mut out, self.logit_rows.len() as u64);
        for r in &self.logit_rows {
            put_u64(&mut out, *r as u64);
        }
        put_u64(&mut out, self.overrides.len() as u64);
        for (row, values) in &self.overrides {
            put_u64(&mut out, *row as u64);
            put_u64(&mut out, values.len() as u64);
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// The message that ends the workers' loop.
    pub fn encode_stop() -> Vec<u8> {
        vec![TAG_STOP]
    }

    /// `None` for the stop message.
    pub fn decode(bytes: &[u8]) -> Result<Option<StepDesc>, String> {
        match bytes.first() {
            Some(&TAG_STOP) | None => return Ok(None),
            Some(&TAG_STEP) => {}
            Some(other) => return Err(format!("unknown control message {other}")),
        }
        let mut r = Reader { bytes, at: 2 };
        let embeddings = bytes.get(1) == Some(&1);
        let ids = r.i64s()?;
        let lengths = r.i64s()?;
        let tables = r.i64s()?;
        let n = r.count(8)?;
        let mut logit_rows = Vec::with_capacity(n);
        for _ in 0..n {
            logit_rows.push(r.u64()? as usize);
        }
        let n = r.count(16)?;
        let mut overrides = Vec::with_capacity(n);
        for _ in 0..n {
            let row = r.u64()? as usize;
            let len = r.count(4)?;
            let raw = r.take(len * 4)?;
            overrides.push((row, raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()));
        }
        Ok(Some(StepDesc { ids, lengths, tables, logit_rows, embeddings, overrides }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_descriptors_round_trip() {
        let d = StepDesc {
            ids: vec![5, -1, 7],
            lengths: vec![1, 2, 3],
            tables: vec![0, 1, 2, 3, 4, 5],
            logit_rows: vec![2, 0],
            embeddings: true,
            overrides: vec![(1, vec![0.5, -0.25]), (2, vec![])],
        };
        assert_eq!(StepDesc::decode(&d.encode()).unwrap(), Some(d));
        assert_eq!(StepDesc::decode(&StepDesc::encode_stop()).unwrap(), None);
        assert!(StepDesc::decode(&[9]).is_err());
        let mut bad = StepDesc::default().encode();
        bad.truncate(5);
        assert!(StepDesc::decode(&bad).is_err());
    }

    #[test]
    fn experts_are_dealt_out_contiguously() {
        let ranges: Vec<_> = (0..3).map(|r| ParCtx::expert_range(8, 3, r)).collect();
        assert_eq!(ranges, vec![0..3, 3..6, 6..8]);
        assert_eq!(ParCtx::expert_range(64, 8, 5), 40..48);
    }
}
