use crate::mem::block_allocator::{AllocatorError, BlockManager};
use crate::mem::prefix_cache::{PrefixCache, PrefixCacheError, PrefixEntry};
use std::collections::{HashMap, VecDeque};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    Allocator(AllocatorError),
    Prefix(PrefixCacheError),
    Value(String),
    Memory(String),
}
impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocator(e) => write!(f, "{e}"),
            Self::Prefix(e) => write!(f, "{e}"),
            Self::Value(s) | Self::Memory(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for SchedulerError {}
impl From<AllocatorError> for SchedulerError {
    fn from(e: AllocatorError) -> Self {
        Self::Allocator(e)
    }
}
impl From<PrefixCacheError> for SchedulerError {
    fn from(e: PrefixCacheError) -> Self {
        Self::Prefix(e)
    }
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub seq_id: i64,
    pub req_id: String,
    pub tokens: Vec<i64>,
    pub output_len: usize,
    pub max_tokens: usize,
    pub temperature: f64,
    pub sampling: crate::sampling::SamplingParams,
    pub sampling_state: crate::sampling::SamplingState,
    pub constrained: bool,
    pub has_media: bool,
    pub num_computed: usize,
    pub state: String,
    pub prefill_target: usize,
    pub preempted: bool,
    pub cached_prompt_tokens: usize,
    pub choice_index: usize,
    pub blocks: Vec<usize>,
    pub block_hashes: Vec<Option<u64>>,
    ngram_index: Option<HashMap<Vec<i64>, Vec<usize>>>,
    ngram_upto: usize,
}
impl Sequence {
    pub fn new(
        seq_id: i64,
        req_id: String,
        prompt: Vec<i64>,
        max_tokens: usize,
        temperature: f64,
    ) -> Self {
        Self::new_with_sampling(
            seq_id,
            req_id,
            prompt,
            max_tokens,
            crate::sampling::SamplingParams {
                temperature,
                ..Default::default()
            },
        )
    }

    pub fn new_with_sampling(
        seq_id: i64,
        req_id: String,
        prompt: Vec<i64>,
        max_tokens: usize,
        sampling: crate::sampling::SamplingParams,
    ) -> Self {
        let n = prompt.len();
        Self {
            seq_id,
            req_id,
            tokens: prompt,
            output_len: 0,
            max_tokens,
            temperature: sampling.temperature,
            sampling_state: crate::sampling::SamplingState::new(&sampling, seq_id),
            constrained: false,
            has_media: false,
            sampling,
            num_computed: 0,
            state: "waiting".into(),
            prefill_target: n,
            preempted: false,
            cached_prompt_tokens: 0,
            choice_index: 0,
            blocks: vec![],
            block_hashes: vec![],
            ngram_index: None,
            ngram_upto: 0,
        }
    }
    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }
    pub fn remaining_max(&self) -> usize {
        self.max_tokens.saturating_sub(self.output_len)
    }
    pub fn append_token(&mut self, t: i64) {
        self.tokens.push(t);
        self.output_len += 1;
    }
}
#[derive(Debug, Clone)]
pub struct ScheduledStep {
    pub prefill: Vec<(i64, usize, usize)>,
    pub decode: Vec<i64>,
    pub verify: Vec<(i64, Vec<i64>)>,
}
impl ScheduledStep {
    fn new() -> Self {
        Self {
            prefill: vec![],
            decode: vec![],
            verify: vec![],
        }
    }
}
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub block_size: usize,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub prefix_cache: bool,
    pub preemption: String,
    pub speculative_lookup: usize,
    pub speculative_window: usize,
}

pub struct Scheduler {
    pub cfg: SchedulerConfig,
    pub bm: BlockManager,
    pub pc: Option<PrefixCache>,
    pub max_model_len: usize,
    pub waiting: VecDeque<i64>,
    pub running: Vec<i64>,
    pub sequences: HashMap<i64, Sequence>,
    pub requests: HashMap<String, Vec<i64>>,
    pub prefix_cache_hits: u64,
    next_offset: HashMap<i64, usize>,
    pub external_drafts: HashMap<i64, Vec<i64>>,
}
impl Scheduler {
    pub fn new(
        cfg: SchedulerConfig,
        bm: BlockManager,
        pc: Option<PrefixCache>,
        max_model_len: usize,
    ) -> Self {
        Self {
            cfg,
            bm,
            pc,
            max_model_len,
            waiting: VecDeque::new(),
            running: vec![],
            sequences: HashMap::new(),
            requests: HashMap::new(),
            prefix_cache_hits: 0,
            next_offset: HashMap::new(),
            external_drafts: HashMap::new(),
        }
    }
    pub fn add_request(&mut self, req_id: String, sequences: Vec<Sequence>) {
        let ids = sequences.iter().map(|s| s.seq_id).collect::<Vec<_>>();
        for s in sequences {
            self.waiting.push_back(s.seq_id);
            self.sequences.insert(s.seq_id, s);
        }
        self.requests.insert(req_id, ids);
    }
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }
    pub fn schedule(&mut self) -> Result<Option<ScheduledStep>, SchedulerError> {
        for _ in 0..4 {
            let mut step = ScheduledStep::new();
            self.admit_waiting()?;
            if self.running.is_empty() && self.waiting.is_empty() {
                return Ok(None);
            }
            let mut budget = self.cfg.max_num_batched_tokens;
            for sid in self.running.clone() {
                if budget == 0 {
                    break;
                }
                let Some(seq) = self.sequences.get(&sid) else {
                    continue;
                };
                if seq.num_computed >= seq.prefill_target {
                    continue;
                }
                let start = *self.next_offset.get(&sid).unwrap_or(&seq.num_computed);
                let todo = seq.prefill_target.saturating_sub(start);
                if todo == 0 {
                    continue;
                }
                let chunk = todo.min(budget);
                let table = self.bm.get_table(sid)?;
                let need = (start + chunk).div_ceil(self.cfg.block_size);
                let need = need.saturating_sub(table.blocks.len());
                if need > self.bm.allocator.available() {
                    break;
                }
                step.prefill.push((sid, start, chunk));
                self.next_offset.insert(sid, start + chunk);
                budget -= chunk;
            }
            for sid in self.running.clone() {
                let Some(seq) = self.sequences.get(&sid) else {
                    continue;
                };
                if seq.state != "running"
                    || seq.num_computed < seq.prefill_target
                    || seq.remaining_max() == 0
                {
                    continue;
                }
                let remaining = seq.remaining_max();
                let mut draft = if seq.sampling.is_greedy() && !seq.constrained {
                    self.external_drafts.remove(&sid).or_else(|| {
                        if self.cfg.speculative_lookup > 0 {
                            self.lookup_draft(sid)
                        } else {
                            None
                        }
                    })
                } else {
                    None
                };
                if budget == 0 {
                    break;
                }
                if let Some(tokens) = &mut draft {
                    tokens.truncate(budget.saturating_sub(1).min(remaining.saturating_sub(1)));
                }
                if draft.as_ref().is_some_and(Vec::is_empty) {
                    draft = None;
                }
                let extra = 1 + draft.as_ref().map_or(0, Vec::len);
                if self.bm.allocator.available() < self.bm.num_blocks_needed(sid, extra as i64)? {
                    if self.evict_cache(extra)? == 0
                        && self.cfg.preemption == "recompute"
                        && self.running.len() > 1
                    {
                        let victim = *self.running.last().unwrap();
                        if victim != sid {
                            self.preempt(victim)?;
                            step.decode.retain(|x| *x != victim);
                            step.prefill.retain(|x| x.0 != victim);
                            self.evict_cache(1)?;
                        }
                    }
                    if self.bm.allocator.available()
                        < self.bm.num_blocks_needed(sid, extra as i64)?
                    {
                        continue;
                    }
                }
                self.bm.ensure_blocks(sid, extra as i64)?;
                budget = budget.saturating_sub(extra);
                if let Some(d) = draft {
                    step.verify.push((sid, d))
                } else {
                    step.decode.push(sid)
                }
            }
            if !step.prefill.is_empty() || !step.decode.is_empty() || !step.verify.is_empty() {
                return Ok(Some(step));
            }
            if self.waiting.is_empty() {
                return Ok(None);
            }
            if self.evict_cache(1)? == 0 {
                return Ok(None);
            }
        }
        Ok(None)
    }
    fn try_admit(&mut self, sid: i64) -> Result<bool, SchedulerError> {
        // Group followers already own a shared prompt table and copied final logits.
        if self
            .sequences
            .get(&sid)
            .is_some_and(|s| s.num_computed >= s.prefill_target)
            && self.bm.tables.contains_key(&sid)
        {
            return Ok(true);
        }
        let prompt = self
            .sequences
            .get(&sid)
            .ok_or_else(|| SchedulerError::Value(format!("unknown sequence {sid}")))?
            .tokens
            .clone();
        if prompt.len() > self.max_model_len {
            return Ok(false);
        }
        if !self.bm.tables.contains_key(&sid) {
            self.bm.new_table(sid)?
        }
        let n_full = prompt.len().saturating_sub(1) / self.cfg.block_size;
        let mut hashes = vec![];
        let mut toks = vec![];
        let mut prev = 0u64;
        for b in 0..n_full {
            let seg = prompt[b * self.cfg.block_size..(b + 1) * self.cfg.block_size].to_vec();
            prev = PrefixCache::hash_chain(prev, &seg);
            hashes.push(prev);
            toks.push(seg);
        }
        let mut reused = 0;
        if self.cfg.prefix_cache && self.pc.is_some() && !self.sequences[&sid].has_media {
            let (pc, bm) = (&mut self.pc, &mut self.bm);
            reused = pc
                .as_mut()
                .unwrap()
                .match_blocks(&hashes, &toks, &mut |bid| {
                    bm.claim_block_for(sid, bid).map_err(PrefixCacheError::from)
                })?;
        }
        let fresh = prompt.len().saturating_sub(reused * self.cfg.block_size);
        let need = (fresh + 1).div_ceil(self.cfg.block_size);
        if self.bm.allocator.available() < need {
            self.bm.drop_table(sid)?;
            return Ok(false);
        }
        {
            let table = self.bm.get_table_mut(sid)?;
            table.num_tokens = (reused * self.cfg.block_size) as i64;
        }
        self.bm.ensure_blocks(sid, (fresh + 1) as i64)?;
        self.bm.get_table_mut(sid)?.num_tokens = prompt.len() as i64;
        let blocks = self.bm.get_table(sid)?.blocks.clone();
        let seq = self.sequences.get_mut(&sid).unwrap();
        seq.blocks = blocks;
        seq.num_computed = reused * self.cfg.block_size;
        if !seq.preempted {
            seq.cached_prompt_tokens = seq.num_computed;
        }
        self.prefix_cache_hits += u64::from(reused > 0);
        seq.block_hashes = hashes
            .iter()
            .take(reused)
            .map(|h| Some(*h))
            .chain(std::iter::repeat_n(
                None,
                self.bm.get_table(sid)?.blocks.len().saturating_sub(reused),
            ))
            .collect();
        Ok(true)
    }
    fn admit_waiting(&mut self) -> Result<(), SchedulerError> {
        while !self.waiting.is_empty() && self.running.len() < self.cfg.max_num_seqs {
            let sid = *self.waiting.front().unwrap();
            let mut ok = self.try_admit(sid)?;
            if !ok {
                let target = self.sequences[&sid].prefill_target;
                let evicted = self.evict_cache(target.div_ceil(self.cfg.block_size) + 1)?;
                if evicted > 0 {
                    ok = self.try_admit(sid)?
                }
            }
            if !ok {
                break;
            }
            self.waiting.pop_front();
            self.sequences.get_mut(&sid).unwrap().state = "running".into();
            self.running.push(sid);
        }
        Ok(())
    }
    fn evict_cache(&mut self, need: usize) -> Result<usize, SchedulerError> {
        let Some(pc) = self.pc.as_mut() else {
            return Ok(0);
        };
        let bm = &mut self.bm;
        Ok(pc.evict_for(need, &mut |bid| {
            let i = bid as usize;
            if bm.allocator.refcounts.get(i).copied() == Some(1) {
                bm.allocator.dec_ref(bid).map_err(PrefixCacheError::from)?;
                bm.block_hashes.remove(&i);
                Ok(true)
            } else {
                Ok(false)
            }
        })?)
    }
    fn preempt(&mut self, sid: i64) -> Result<(), SchedulerError> {
        let seq = self.sequences.get_mut(&sid).unwrap();
        seq.preempted = true;
        self.bm.drop_table(sid)?;
        seq.blocks.clear();
        seq.block_hashes.clear();
        seq.num_computed = 0;
        seq.prefill_target = seq.num_tokens();
        self.next_offset.remove(&sid);
        seq.state = "waiting".into();
        if let Some(i) = self.running.iter().position(|x| *x == sid) {
            self.running.remove(i);
        }
        self.waiting.push_front(sid);
        Ok(())
    }
    pub fn commit_chunk(&mut self, sid: i64, n: usize) -> Result<(), SchedulerError> {
        let seq = self.sequences.get_mut(&sid).unwrap();
        seq.num_computed += n;
        let nt = seq.num_tokens();
        let table = self.bm.get_table_mut(sid)?;
        table.num_tokens = table.num_tokens.max(nt as i64);
        seq.blocks = table.blocks.clone();
        self.insert_full_blocks(sid)
    }
    pub fn commit_decode(&mut self, sid: i64) -> Result<(), SchedulerError> {
        let nt = self.sequences[&sid].num_tokens();
        {
            let seq = self.sequences.get_mut(&sid).unwrap();
            seq.num_computed = nt - 1;
            seq.blocks = self.bm.get_table(sid)?.blocks.clone();
        }
        self.bm.get_table_mut(sid)?.num_tokens = nt as i64;
        self.ngram_upto(sid);
        Ok(())
    }
    pub fn commit_verify(&mut self, sid: i64) -> Result<(), SchedulerError> {
        let nt = self.sequences[&sid].num_tokens();
        {
            let seq = self.sequences.get_mut(&sid).unwrap();
            seq.num_computed = nt - 1;
        }
        self.bm.get_table_mut(sid)?.num_tokens = nt as i64;
        self.trim_blocks(sid)?;
        self.sequences.get_mut(&sid).unwrap().blocks = self.bm.get_table(sid)?.blocks.clone();
        self.insert_full_blocks(sid)?;
        self.ngram_upto(sid);
        Ok(())
    }
    pub fn insert_full_blocks(&mut self, sid: i64) -> Result<(), SchedulerError> {
        if self.pc.is_none() || !self.cfg.prefix_cache || self.sequences[&sid].has_media {
            return Ok(());
        }
        let (tokens, num_computed) = {
            let s = &self.sequences[&sid];
            (s.tokens.clone(), s.num_computed)
        };
        let full = num_computed / self.cfg.block_size;
        let table = self.bm.get_table(sid)?;
        let mut entries = vec![];
        {
            let seq = self.sequences.get_mut(&sid).unwrap();
            if seq.block_hashes.len() < table.blocks.len() {
                seq.block_hashes.resize(table.blocks.len(), None)
            }
            for b in 0..full.min(table.blocks.len()) {
                let seg = tokens[b * self.cfg.block_size..(b + 1) * self.cfg.block_size].to_vec();
                let prev = if b > 0 {
                    seq.block_hashes[b - 1]
                } else {
                    Some(0)
                };
                let Some(prev) = prev else { break };
                let h = seq.block_hashes[b].unwrap_or_else(|| PrefixCache::hash_chain(prev, &seg));
                seq.block_hashes[b] = Some(h);
                entries.push(PrefixEntry {
                    chain_hash: h,
                    block_id: table.blocks[b] as i64,
                    tokens: seg,
                });
            }
        }
        if !entries.is_empty() {
            let (pc, bm) = (&mut self.pc, &mut self.bm);
            let cell = std::cell::RefCell::new(&mut *bm);
            let mut claim = |bid| {
                cell.borrow_mut()
                    .allocator
                    .inc_ref(bid)
                    .map_err(PrefixCacheError::from)
            };
            let mut release = |bid| {
                cell.borrow_mut()
                    .allocator
                    .dec_ref(bid)
                    .map_err(PrefixCacheError::from)
            };
            pc.as_mut()
                .unwrap()
                .insert_path(&entries, &mut claim, &mut release)?;
        }
        Ok(())
    }
    pub fn finish(&mut self, sid: i64) {
        if let Some(s) = self.sequences.get_mut(&sid) {
            s.state = "finished".into()
        }
        self.waiting.retain(|x| *x != sid);
        self.running.retain(|x| *x != sid);
        self.next_offset.remove(&sid);
    }
    pub fn release(&mut self, sid: i64) -> Result<(), SchedulerError> {
        self.bm.drop_table(sid)?;
        self.sequences.get_mut(&sid).unwrap().blocks.clear();
        self.finish(sid);
        Ok(())
    }
    pub fn free_request(&mut self, rid: &str) -> Result<(), SchedulerError> {
        let Some(ids) = self.requests.remove(rid) else {
            return Ok(());
        };
        for sid in ids {
            if self.bm.tables.contains_key(&sid) {
                self.release(sid)?
            }
        }
        Ok(())
    }
    fn trim_blocks(&mut self, sid: i64) -> Result<(), SchedulerError> {
        let keep = (self.sequences[&sid].num_tokens().saturating_sub(1)) / self.cfg.block_size + 1;
        let extra = {
            let table = self.bm.get_table(sid)?;
            if table.blocks.len() <= keep {
                return Ok(());
            }
            table.blocks[keep..].to_vec()
        };
        for bid in &extra {
            self.bm.block_hashes.remove(bid);
            self.bm.allocator.dec_ref(*bid as i64)?;
        }
        self.bm.get_table_mut(sid)?.blocks.truncate(keep);
        let n = self.bm.get_table(sid)?.blocks.len();
        let hashes = &mut self.sequences.get_mut(&sid).unwrap().block_hashes;
        if hashes.len() > n {
            hashes.truncate(n)
        }
        Ok(())
    }
    fn lookup_draft(&mut self, sid: i64) -> Option<Vec<i64>> {
        let k = self.cfg.speculative_lookup;
        let wmax = self.cfg.speculative_window;
        let (tokens, index_init) = {
            let s = &self.sequences[&sid];
            (s.tokens.clone(), s.ngram_index.is_some())
        };
        let n = tokens.len();
        if n < 2 {
            return None;
        }
        if !index_init {
            let s = self.sequences.get_mut(&sid).unwrap();
            s.ngram_index = Some(HashMap::new());
            s.ngram_upto = 0;
        }
        self.ngram_upto(sid);
        let s = &self.sequences[&sid];
        let idx = s.ngram_index.as_ref()?;
        let mut best: Option<Vec<i64>> = None;
        for w in (1..=wmax.min(n - 1)).rev() {
            if let Some(positions) = idx.get(&tokens[n - w..n]) {
                for pos in positions {
                    if *pos >= n - 1 {
                        continue;
                    }
                    let draft = tokens[pos + 1..(pos + 1 + k).min(n)].to_vec();
                    if !draft.is_empty() && best.as_ref().is_none_or(|b| draft.len() > b.len()) {
                        best = Some(draft)
                    }
                    break;
                }
                if best.as_ref().is_some_and(|b| b.len() >= k) {
                    break;
                }
            }
        }
        best
    }
    fn ngram_upto(&mut self, sid: i64) {
        let s = &self.sequences[&sid];
        if s.ngram_index.is_none() {
            return;
        }
        let (tokens, upto) = (s.tokens.clone(), s.ngram_upto);
        let wm = self.cfg.speculative_window;
        let s = self.sequences.get_mut(&sid).unwrap();
        let idx = s.ngram_index.as_mut().unwrap();
        for i in upto..tokens.len() {
            for w in 1..=wm.min(i + 1) {
                idx.entry(tokens[i + 1 - w..=i].to_vec())
                    .or_default()
                    .push(i)
            }
        }
        s.ngram_upto = tokens.len();
    }
}

#[cfg(test)]
mod shared_prompt_tests {
    use super::*;
    #[test]
    fn precomputed_follower_reuses_table_without_prefill() {
        let cfg = SchedulerConfig {
            block_size: 4,
            max_num_seqs: 2,
            max_num_batched_tokens: 8,
            prefix_cache: true,
            preemption: "recompute".into(),
            speculative_lookup: 0,
            speculative_window: 0,
        };
        let mut scheduler = Scheduler::new(cfg, BlockManager::new(8, 4).unwrap(), None, 32);
        let mut sequence = Sequence::new(2, "group".into(), vec![1, 2, 3], 2, 0.0);
        sequence.num_computed = 3;
        sequence.choice_index = 1;
        scheduler.bm.new_table(2).unwrap();
        let block = scheduler.bm.append_block(2).unwrap();
        scheduler.bm.get_table_mut(2).unwrap().num_tokens = 3;
        scheduler.add_request("group".into(), vec![sequence]);
        let step = scheduler.schedule().unwrap().unwrap();
        assert!(step.prefill.is_empty());
        assert_eq!(step.decode, vec![2]);
        assert_eq!(scheduler.bm.get_table(2).unwrap().blocks, vec![block]);
    }
}
