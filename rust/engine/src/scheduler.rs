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
    /// Per-request salt (`cache_salt`): its blocks never match another salt's, even for the same prompt.
    pub prefix_salt: u64,
    /// Draft length controller of this sequence (see `SchedulerConfig::speculative_adaptive`).
    pub adaptive: crate::speculative::AdaptiveDraft,
    /// Order of arrival in the scheduler (ties of equal priority are served first come, first served).
    pub arrival: u64,
    pub enqueued: std::time::Instant,
    /// The request is aborted when it has not finished by then.
    pub deadline: Option<std::time::Instant>,
    /// Blocks of this sequence that wait in host memory (preemption by swap); 0 when its KV is on the device or was dropped.
    pub swapped_blocks: usize,
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
            prefix_salt: 0,
            adaptive: crate::speculative::AdaptiveDraft::new(1, 16),
            arrival: 0,
            enqueued: std::time::Instant::now(),
            deadline: None,
            swapped_blocks: 0,
            ngram_index: None,
            ngram_upto: 0,
        }
    }
    /// Speculative decoding applies: not opted out, no mirostat (its state depends on the sampled sequence), a single choice and no
    /// image embeddings. Stochastic and grammar-constrained requests qualify (see `speculative::verify_chain`).
    pub fn can_speculate(&self) -> bool {
        self.sampling.speculative && self.sampling.mirostat == 0 && self.sampling.choices == 1 && !self.has_media
    }
    /// Smaller values are scheduled first (vLLM's convention).
    pub fn priority(&self) -> i32 {
        self.sampling.priority
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
    /// Shrink and grow the draft length per sequence from the acceptance rate.
    pub speculative_adaptive: bool,
    pub policy: SchedulingPolicy,
    /// A new sequence is admitted only when this many blocks stay free for the running ones to grow (0 when nothing runs).
    pub watermark_blocks: usize,
    /// At most this many sequences prefill in one step (0 = no limit).
    pub max_partial_prefills: usize,
    /// Prompts longer than this count as long (0 = no distinction) ...
    pub long_prefill_threshold: usize,
    /// ... and at most this many long prompts prefill in one step, so shorter prompts interleave with them.
    pub max_long_partial_prefills: usize,
    /// Blocks of host memory for preemption by swap (`preemption == "swap"`); a preempted sequence that does not fit is recomputed.
    pub swap_blocks: usize,
    /// Reserve one token of every step's budget per decoding sequence before prompt chunks are scheduled, so a long prompt cannot
    /// starve the running sequences (inter-token latency); false = the prompt chunks come first.
    pub decode_first: bool,
    /// Decode steps the runtime may run per scheduler round while the batch does not change (`--multi-step`, 1 = off).
    /// The scheduler itself ignores it; `InferenceRuntime::step` reads it.
    pub multi_step: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First come, first served.
    Fcfs,
    /// Lower `priority` first; equal priorities first come, first served. A waiting request of better priority preempts the
    /// worst running sequence when all slots are taken or the KV pool is full.
    Priority,
}

impl SchedulingPolicy {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "fcfs" => Some(Self::Fcfs),
            "priority" => Some(Self::Priority),
            _ => None,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            max_num_seqs: 16,
            max_num_batched_tokens: 512,
            prefix_cache: true,
            preemption: "recompute".into(),
            speculative_lookup: 0,
            speculative_window: 0,
            speculative_adaptive: false,
            policy: SchedulingPolicy::Fcfs,
            watermark_blocks: 0,
            max_partial_prefills: 0,
            long_prefill_threshold: 0,
            max_long_partial_prefills: 1,
            swap_blocks: 0,
            decode_first: false,
            multi_step: 1,
        }
    }
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
    /// Blocks that were just registered in the prefix cache and may be written to the disk tier. Drained by the runtime,
    /// which owns the decoder: the scheduler knows the block and its hash, but not the KV bytes.
    pub pending_spills: Vec<(u64, u64, i64, Vec<i64>)>,
    next_offset: HashMap<i64, usize>,
    pub external_drafts: HashMap<i64, Vec<i64>>,
    next_arrival: u64,
    /// Number of sequences preempted (their KV blocks dropped, to be recomputed) since the start.
    pub preemptions: u64,
    /// Seconds each admitted sequence waited in the queue; the owner drains it into its metrics.
    pub wait_times: Vec<f64>,
    /// Swap traffic the owner of the KV pool has to carry out before the next forward pass: blocks to export (their owner no longer holds
    /// them), blocks to import into, and sequences whose swapped data can be discarded.
    pending_swap_out: Vec<(i64, Vec<usize>)>,
    pending_swap_in: Vec<(i64, Vec<usize>)>,
    dropped_swaps: Vec<i64>,
    /// Blocks of host swap space in use.
    pub swap_used: usize,
}

/// Swap work for the KV pool's owner (see `Scheduler::take_swaps`).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SwapWork {
    pub out: Vec<(i64, Vec<usize>)>,
    pub into: Vec<(i64, Vec<usize>)>,
    pub dropped: Vec<i64>,
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
            pending_spills: Vec::new(),
            next_offset: HashMap::new(),
            external_drafts: HashMap::new(),
            next_arrival: 0,
            preemptions: 0,
            wait_times: Vec::new(),
            pending_swap_out: Vec::new(),
            pending_swap_in: Vec::new(),
            dropped_swaps: Vec::new(),
            swap_used: 0,
        }
    }
    pub fn add_request(&mut self, req_id: String, sequences: Vec<Sequence>) {
        let ids = sequences.iter().map(|s| s.seq_id).collect::<Vec<_>>();
        for mut s in sequences {
            s.arrival = self.next_arrival;
            self.next_arrival += 1;
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
            let decoders = self
                .running
                .iter()
                .filter(|sid| self.sequences.get(sid).is_some_and(|s| s.num_computed >= s.prefill_target && s.remaining_max() > 0))
                .count();
            // decode-first keeps one token of the budget per running decoder out of the prompt chunks' reach
            let mut prefill_budget = if self.cfg.decode_first {
                budget.saturating_sub(decoders).max(1)
            } else {
                budget
            };
            let mut long_prefills = 0;
            for sid in self.ordered_running() {
                if prefill_budget == 0 || budget == 0 {
                    break;
                }
                if self.cfg.max_partial_prefills > 0 && step.prefill.len() >= self.cfg.max_partial_prefills {
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
                let long = self.cfg.long_prefill_threshold > 0 && seq.prefill_target > self.cfg.long_prefill_threshold;
                if long && long_prefills >= self.cfg.max_long_partial_prefills.max(1) {
                    continue;
                }
                let chunk = todo.min(prefill_budget);
                let table = self.bm.get_table(sid)?;
                let need = (start + chunk).div_ceil(self.cfg.block_size);
                let need = need.saturating_sub(table.blocks.len());
                if need > self.bm.allocator.available() {
                    break;
                }
                step.prefill.push((sid, start, chunk));
                self.next_offset.insert(sid, start + chunk);
                budget -= chunk;
                prefill_budget -= chunk;
                long_prefills += usize::from(long);
            }
            for sid in self.ordered_running() {
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
                let mut draft = if seq.can_speculate() {
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
                        && matches!(self.cfg.preemption.as_str(), "recompute" | "swap")
                        && self.running.len() > 1
                    {
                        let victim = self.preemption_victim(sid);
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
    /// Re-admit a sequence whose KV blocks wait in host memory: fresh blocks for the same number of tokens, filled by the owner of the pool
    /// from the swapped data. The sequence continues where it stopped.
    fn try_admit_swapped(&mut self, sid: i64) -> Result<bool, SchedulerError> {
        let blocks_needed = self.sequences[&sid].swapped_blocks;
        let watermark = if self.running.is_empty() { 0 } else { self.cfg.watermark_blocks };
        if self.bm.allocator.available() < blocks_needed + watermark {
            return Ok(false);
        }
        self.bm.new_table(sid)?;
        let mut blocks = Vec::with_capacity(blocks_needed);
        for _ in 0..blocks_needed {
            blocks.push(self.bm.append_block(sid)?);
        }
        let seq = self.sequences.get_mut(&sid).unwrap();
        self.bm.get_table_mut(sid)?.num_tokens = seq.num_tokens() as i64;
        seq.blocks = blocks.clone();
        seq.block_hashes = vec![None; blocks.len()];
        seq.swapped_blocks = 0;
        self.swap_used = self.swap_used.saturating_sub(blocks_needed);
        self.pending_swap_in.push((sid, blocks));
        Ok(true)
    }
    fn try_admit(&mut self, sid: i64) -> Result<bool, SchedulerError> {
        if self.sequences.get(&sid).is_some_and(|s| s.swapped_blocks > 0) {
            return self.try_admit_swapped(sid);
        }
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
        // the seed separates deployments, the sequence's salt separates requests: same seed *and* same salt, shared blocks
        let mut prev = prefix_seed() ^ self.sequences[&sid].prefix_salt;
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
        let watermark = if self.running.is_empty() { 0 } else { self.cfg.watermark_blocks };
        if self.bm.allocator.available() < need + watermark {
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
    /// Scheduling key: the sequence with the smaller key goes first.
    fn key(&self, sid: i64) -> (i32, u64) {
        match self.cfg.policy {
            SchedulingPolicy::Fcfs => (0, self.sequences.get(&sid).map_or(u64::MAX, |s| s.arrival)),
            SchedulingPolicy::Priority => self.sequences.get(&sid).map_or((i32::MAX, u64::MAX), |s| (s.priority(), s.arrival)),
        }
    }
    /// Running sequences in scheduling order (arrival order under FCFS).
    fn ordered_running(&self) -> Vec<i64> {
        let mut ids = self.running.clone();
        if self.cfg.policy == SchedulingPolicy::Priority {
            ids.sort_by_key(|sid| self.key(*sid));
        }
        ids
    }
    fn pick_waiting(&self) -> Option<i64> {
        match self.cfg.policy {
            SchedulingPolicy::Fcfs => self.waiting.front().copied(),
            SchedulingPolicy::Priority => self.waiting.iter().copied().min_by_key(|sid| self.key(*sid)),
        }
    }
    /// The sequence to preempt for `needy`: the last arrival under FCFS, the worst-priority newest sequence otherwise.
    fn preemption_victim(&self, needy: i64) -> i64 {
        match self.cfg.policy {
            SchedulingPolicy::Fcfs => *self.running.last().unwrap(),
            SchedulingPolicy::Priority => self
                .running
                .iter()
                .copied()
                .filter(|sid| *sid != needy)
                .max_by_key(|sid| self.key(*sid))
                .unwrap_or(needy),
        }
    }
    fn admit_waiting(&mut self) -> Result<(), SchedulerError> {
        while let Some(sid) = self.pick_waiting() {
            if self.running.len() >= self.cfg.max_num_seqs {
                // all slots taken: a waiting request of better priority takes the slot of the worst running sequence
                let victim = (self.cfg.policy == SchedulingPolicy::Priority)
                    .then(|| self.preemption_victim(sid))
                    .filter(|v| self.key(*v).0 > self.key(sid).0);
                let Some(victim) = victim else { break };
                self.preempt(victim)?;
            }
            let mut ok = self.try_admit(sid)?;
            if !ok {
                let target = self.sequences[&sid].prefill_target;
                let evicted = self.evict_cache(target.div_ceil(self.cfg.block_size) + 1)?;
                if evicted > 0 {
                    ok = self.try_admit(sid)?
                }
            }
            if !ok && self.cfg.policy == SchedulingPolicy::Priority && !self.running.is_empty() {
                // the pool is full: free the blocks of a running sequence of worse priority
                let victim = self.preemption_victim(sid);
                if self.key(victim).0 > self.key(sid).0 {
                    self.preempt(victim)?;
                    ok = self.try_admit(sid)?;
                }
            }
            if !ok {
                break;
            }
            if let Some(position) = self.waiting.iter().position(|x| *x == sid) {
                self.waiting.remove(position);
            }
            let seq = self.sequences.get_mut(&sid).unwrap();
            seq.state = "running".into();
            self.wait_times.push(seq.enqueued.elapsed().as_secs_f64());
            self.running.push(sid);
        }
        Ok(())
    }
    /// The swap traffic accumulated by `schedule` since the last call. It must be carried out (exports first) before the forward pass of the
    /// step, even when `schedule` returned no work: the freed blocks may be rewritten by that pass.
    pub fn take_swaps(&mut self) -> SwapWork {
        SwapWork {
            out: std::mem::take(&mut self.pending_swap_out),
            into: std::mem::take(&mut self.pending_swap_in),
            dropped: std::mem::take(&mut self.dropped_swaps),
        }
    }
    /// Request ids with a sequence whose deadline has passed.
    pub fn expired_requests(&self, now: std::time::Instant) -> Vec<String> {
        let mut ids: Vec<String> = self
            .sequences
            .values()
            .filter(|s| s.deadline.is_some_and(|d| d <= now) && s.state != "finished")
            .map(|s| s.req_id.clone())
            .collect();
        ids.sort();
        ids.dedup();
        ids
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
    /// Frees up to `need` cached prefix blocks that no sequence uses; returns how many were freed.
    pub fn reclaim_cache(&mut self, need: usize) -> usize {
        self.evict_cache(need).unwrap_or(0)
    }
    /// Takes the blocks the runtime should write to the disk tier. They stay alive (the cache holds a reference), so
    /// their KV is still valid when the runtime reads it.
    pub fn take_pending_spills(&mut self) -> Vec<(u64, u64, i64, Vec<i64>)> {
        const KEEP: usize = 4096;
        if self.pending_spills.len() > KEEP {
            let drop = self.pending_spills.len() - KEEP;
            self.pending_spills.drain(..drop);
        }
        std::mem::take(&mut self.pending_spills)
    }

    /// A free block, for a block that is read back from the disk tier.
    pub fn alloc_free_block(&mut self) -> Result<i64, SchedulerError> {
        Ok(self.bm.allocator.alloc()? as i64)
    }

    /// Gives a block back that was allocated for a disk import and then not used.
    pub fn free_block(&mut self, bid: i64) {
        let _ = self.bm.allocator.dec_ref(bid);
    }

    pub fn has_prefix_hash(&self, hash: u64) -> bool {
        self.pc.as_ref().is_some_and(|pc| pc.contains(hash))
    }

    /// Registers a whole chain at once: the radix tree needs the path from its root, not single blocks side by side.
    pub fn register_prefix_chain(&mut self, chain: &[(u64, Vec<i64>, i64)]) -> Result<bool, SchedulerError> {
        let Some(pc) = self.pc.as_mut() else {
            return Ok(false);
        };
        if chain.is_empty() || pc.contains(chain[0].0) {
            return Ok(false);
        }
        let entries: Vec<PrefixEntry> = chain
            .iter()
            .map(|(hash, tokens, block)| PrefixEntry {
                chain_hash: *hash,
                block_id: *block,
                tokens: tokens.clone(),
            })
            .collect();
        let bm = &mut self.bm;
        let cell = std::cell::RefCell::new(&mut *bm);
        let mut claim = |bid: i64| {
            cell.borrow_mut()
                .allocator
                .inc_ref(bid)
                .map_err(PrefixCacheError::from)
        };
        let mut release = |bid: i64| {
            cell.borrow_mut()
                .allocator
                .dec_ref(bid)
                .map_err(PrefixCacheError::from)
        };
        pc.insert_path(&entries, &mut claim, &mut release)?;
        Ok(true)
    }

    /// Registers a block that already holds imported KV under `hash`. The parent has to be in the cache (or be `0`),
    /// otherwise the chain would claim a prefix it does not have.
    pub fn register_prefix_block(
        &mut self,
        hash: u64,
        parent: u64,
        tokens: Vec<i64>,
        block: i64,
    ) -> Result<bool, SchedulerError> {
        if parent != 0 && !self.has_prefix_hash(parent) {
            return Ok(false);
        }
        let Some(pc) = self.pc.as_mut() else {
            return Ok(false);
        };
        if pc.contains(hash) {
            return Ok(false);
        }
        let bm = &mut self.bm;
        let cell = std::cell::RefCell::new(&mut *bm);
        let mut claim = |bid: i64| {
            cell.borrow_mut()
                .allocator
                .inc_ref(bid)
                .map_err(PrefixCacheError::from)
        };
        let mut release = |bid: i64| {
            cell.borrow_mut()
                .allocator
                .dec_ref(bid)
                .map_err(PrefixCacheError::from)
        };
        pc.insert(hash, block, tokens, &mut claim, &mut release)?;
        Ok(true)
    }

    /// Empties the prefix cache and gives the blocks back to the allocator. Returns how many entries were dropped.
    pub fn clear_prefix_cache(&mut self) -> Result<usize, SchedulerError> {
        let Some(pc) = self.pc.as_mut() else {
            return Ok(0);
        };
        let before = pc.len();
        let bm = &mut self.bm;
        pc.clear(&mut |bid| {
            let i = bid as usize;
            if bm.allocator.refcounts.get(i).copied() == Some(1) {
                bm.allocator.dec_ref(bid).map_err(PrefixCacheError::from)?;
                bm.block_hashes.remove(&i);
            }
            Ok(())
        })?;
        Ok(before)
    }

    fn preempt(&mut self, sid: i64) -> Result<(), SchedulerError> {
        let seq = self.sequences.get_mut(&sid).unwrap();
        seq.preempted = true;
        self.preemptions += 1;
        if self.cfg.preemption == "swap" && seq.num_computed > 0 && !seq.has_media {
            let blocks = self.bm.get_table(sid)?.blocks.clone();
            if !blocks.is_empty() && self.swap_used + blocks.len() <= self.cfg.swap_blocks {
                // keep the progress: the data leaves with the owner's export, the blocks are freed for others
                self.swap_used += blocks.len();
                seq.swapped_blocks = blocks.len();
                self.pending_swap_out.push((sid, blocks));
                self.bm.drop_table(sid)?;
                seq.blocks.clear();
                seq.block_hashes.clear();
                self.next_offset.remove(&sid);
                seq.state = "waiting".into();
                if let Some(i) = self.running.iter().position(|x| *x == sid) {
                    self.running.remove(i);
                }
                self.waiting.push_front(sid);
                return Ok(());
            }
        }
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
    /// Jump-forward: `forced` tokens were appended behind the token just sampled. Neither has a KV entry yet, so the sequence goes
    /// back to prefill for them: the next step computes all of them in one chunk and returns the logits of the last one.
    pub fn begin_jump(&mut self, sid: i64, forced: usize) -> Result<(), SchedulerError> {
        self.bm.ensure_blocks(sid, forced as i64)?;
        let blocks = self.bm.get_table(sid)?.blocks.clone();
        let seq = self.sequences.get_mut(&sid).ok_or_else(|| SchedulerError::Value(format!("unknown sequence {sid}")))?;
        seq.blocks = blocks;
        seq.prefill_target = seq.num_tokens();
        self.next_offset.remove(&sid);
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
            // (own hash, parent hash, block, tokens) — the parent is what makes a chain rebuildable later
            let mut parent = 0u64;
            for entry in &entries {
                self.pending_spills.push((entry.chain_hash, parent, entry.block_id, entry.tokens.clone()));
                parent = entry.chain_hash;
            }
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
            } else {
                // never admitted (still queued): take it out of the queues, or the next step finds an id without a sequence
                self.finish(sid);
            }
            if let Some(seq) = self.sequences.get_mut(&sid) {
                if seq.swapped_blocks > 0 {
                    self.swap_used = self.swap_used.saturating_sub(seq.swapped_blocks);
                    seq.swapped_blocks = 0;
                    self.dropped_swaps.push(sid);
                }
            }
            self.waiting.retain(|x| *x != sid);
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
        let mut k = self.cfg.speculative_lookup;
        if self.cfg.speculative_adaptive {
            k = k.min(self.sequences[&sid].adaptive.length());
        }
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

/// The seed every prefix chain starts from. `FLLM_PREFIX_SALT` separates the block identities of two deployments, so a
/// tenant running its own server never reuses another tenant's KV blocks even when the prompt is identical.
pub fn prefix_seed() -> u64 {
    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SEED.get_or_init(|| seed_of(std::env::var("FLLM_PREFIX_SALT").unwrap_or_default().trim()))
}

/// FNV-1a over the salt; an empty salt keeps the historical seed of `0`, so a single-user setup is unchanged.
fn seed_of(salt: &str) -> u64 {
    if salt.is_empty() {
        return 0;
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in salt.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod salt_tests {
    use super::*;

    #[test]
    fn a_salt_separates_block_identities() {
        assert_eq!(seed_of(""), 0, "without a salt nothing changes");
        assert_ne!(seed_of("tenant-a"), 0);
        assert_ne!(seed_of("tenant-a"), seed_of("tenant-b"), "two tenants must not share blocks");
        assert_eq!(seed_of("tenant-a"), seed_of("tenant-a"), "the same salt is stable across restarts");
        let tokens = [1i64, 2, 3];
        let a = PrefixCache::hash_chain(seed_of("tenant-a"), &tokens);
        let b = PrefixCache::hash_chain(seed_of("tenant-b"), &tokens);
        assert_ne!(a, b, "the chains of two tenants can never meet");
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
            ..SchedulerConfig::default()
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

#[cfg(test)]
mod policy_tests {
    use super::*;
    use crate::sampling::SamplingParams;

    fn scheduler(cfg: SchedulerConfig, blocks: i64) -> Scheduler {
        Scheduler::new(cfg, BlockManager::new(blocks, 4).unwrap(), None, 256)
    }
    fn request(s: &mut Scheduler, id: i64, prompt: usize, priority: i32) {
        let sampling = SamplingParams { priority, ..Default::default() };
        let seq = Sequence::new_with_sampling(id, format!("r{id}"), (0..prompt as i64).collect(), 8, sampling);
        s.add_request(format!("r{id}"), vec![seq]);
    }
    fn cfg() -> SchedulerConfig {
        SchedulerConfig { block_size: 4, max_num_seqs: 2, max_num_batched_tokens: 16, prefix_cache: false, ..SchedulerConfig::default() }
    }

    #[test]
    fn fcfs_serves_in_arrival_order_and_priority_serves_the_smaller_value_first() {
        let mut s = scheduler(cfg(), 64);
        request(&mut s, 1, 4, 5);
        request(&mut s, 2, 4, 1);
        s.schedule().unwrap();
        assert_eq!(s.running, vec![1, 2]);
        let mut s = scheduler(SchedulerConfig { max_num_seqs: 1, policy: SchedulingPolicy::Priority, ..cfg() }, 64);
        request(&mut s, 1, 4, 5);
        request(&mut s, 2, 4, 1);
        s.schedule().unwrap();
        assert_eq!(s.running, vec![2], "the smaller priority value runs first");
    }

    #[test]
    fn a_better_priority_request_preempts_the_worst_running_sequence() {
        let mut s = scheduler(SchedulerConfig { max_num_seqs: 1, policy: SchedulingPolicy::Priority, ..cfg() }, 64);
        request(&mut s, 1, 4, 5);
        s.schedule().unwrap();
        assert_eq!(s.running, vec![1]);
        request(&mut s, 2, 4, 0);
        s.schedule().unwrap();
        assert_eq!(s.running, vec![2]);
        assert!(s.waiting.contains(&1));
        assert_eq!(s.preemptions, 1);
        assert_eq!(s.sequences[&1].num_computed, 0, "recompute preemption drops the progress");
        // an equal or worse priority never preempts
        request(&mut s, 3, 4, 0);
        s.schedule().unwrap();
        assert_eq!(s.running, vec![2]);
        assert_eq!(s.preemptions, 1);
    }

    #[test]
    fn fcfs_never_preempts_for_a_waiting_request() {
        let mut s = scheduler(SchedulerConfig { max_num_seqs: 1, ..cfg() }, 64);
        request(&mut s, 1, 4, 9);
        s.schedule().unwrap();
        request(&mut s, 2, 4, -9);
        s.schedule().unwrap();
        assert_eq!(s.running, vec![1]);
        assert_eq!(s.preemptions, 0);
    }

    #[test]
    fn a_full_pool_admits_by_priority_through_preemption() {
        // 4 blocks of 4 tokens: one 12-token prompt (3 blocks + 1) fills the pool
        let mut s = scheduler(SchedulerConfig { max_num_seqs: 4, policy: SchedulingPolicy::Priority, ..cfg() }, 4);
        request(&mut s, 1, 12, 5);
        s.schedule().unwrap();
        assert_eq!(s.running, vec![1]);
        request(&mut s, 2, 12, 0);
        s.schedule().unwrap();
        assert_eq!(s.running, vec![2]);
        assert_eq!(s.preemptions, 1);
    }

    #[test]
    fn the_watermark_keeps_blocks_free_for_running_sequences() {
        let mut s = scheduler(SchedulerConfig { max_num_seqs: 4, watermark_blocks: 3, ..cfg() }, 8);
        request(&mut s, 1, 8, 0); // 3 blocks
        s.schedule().unwrap();
        request(&mut s, 2, 8, 0); // needs 3 + watermark 3 > 5 free
        s.schedule().unwrap();
        assert_eq!(s.running, vec![1]);
        assert_eq!(s.waiting.len(), 1);
        // with nothing running the watermark does not block admission
        let mut s = scheduler(SchedulerConfig { watermark_blocks: 100, ..cfg() }, 8);
        request(&mut s, 1, 8, 0);
        s.schedule().unwrap();
        assert_eq!(s.running, vec![1]);
    }

    #[test]
    fn decode_first_leaves_budget_for_running_decoders() {
        let base = SchedulerConfig { max_num_seqs: 4, max_num_batched_tokens: 8, ..cfg() };
        for (decode_first, expected_chunk) in [(false, 8), (true, 6)] {
            let mut s = scheduler(SchedulerConfig { decode_first, ..base.clone() }, 64);
            request(&mut s, 1, 4, 0);
            request(&mut s, 2, 4, 0);
            s.schedule().unwrap();
            for sid in [1, 2] {
                s.commit_chunk(sid, 4).unwrap();
            }
            request(&mut s, 3, 40, 0);
            let step = s.schedule().unwrap().unwrap();
            let chunk = step.prefill.iter().find(|p| p.0 == 3).map(|p| p.2).unwrap();
            assert_eq!(chunk, expected_chunk, "decode_first = {decode_first}");
            assert_eq!(step.decode.len(), if decode_first { 2 } else { 0 });
        }
    }

    #[test]
    fn partial_prefill_limits_and_long_prompt_throttling() {
        let base = SchedulerConfig { max_num_seqs: 8, max_num_batched_tokens: 64, ..cfg() };
        let mut s = scheduler(SchedulerConfig { max_partial_prefills: 2, ..base.clone() }, 128);
        for id in 1..=4 {
            request(&mut s, id, 8, 0);
        }
        assert_eq!(s.schedule().unwrap().unwrap().prefill.len(), 2);
        // two long prompts and a short one: only one long prompt prefills, the short one interleaves
        let mut s = scheduler(SchedulerConfig { long_prefill_threshold: 16, max_long_partial_prefills: 1, ..base }, 128);
        request(&mut s, 1, 40, 0);
        request(&mut s, 2, 40, 0);
        request(&mut s, 3, 6, 0);
        let step = s.schedule().unwrap().unwrap();
        let ids: Vec<i64> = step.prefill.iter().map(|p| p.0).collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn deadlines_are_reported_per_request_and_waits_are_recorded() {
        let mut s = scheduler(cfg(), 64);
        request(&mut s, 1, 4, 0);
        request(&mut s, 2, 4, 0);
        s.sequences.get_mut(&1).unwrap().deadline = Some(std::time::Instant::now() - std::time::Duration::from_millis(1));
        s.sequences.get_mut(&2).unwrap().deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs(60));
        assert_eq!(s.expired_requests(std::time::Instant::now()), vec!["r1".to_string()]);
        s.schedule().unwrap();
        assert_eq!(s.wait_times.len(), 2);
    }
}

#[cfg(test)]
mod prefix_reuse_tests {
    use super::*;

    /// Agents send many requests that share a long prefix (system prompt and tool list): after the
    /// first one has computed it, the following ones reuse every full block and run together with it.
    #[test]
    fn requests_with_a_shared_prefix_reuse_full_blocks_and_run_together() {
        let cfg = SchedulerConfig {
            block_size: 4,
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            prefix_cache: true,
            preemption: "recompute".into(),
            ..SchedulerConfig::default()
        };
        let pc = PrefixCache::new(64, true, true).unwrap();
        let mut s = Scheduler::new(cfg, BlockManager::new(64, 4).unwrap(), Some(pc), 256);
        let prompt = |tail: i64| {
            let mut p: Vec<i64> = (0..10).collect();
            p.push(tail);
            p
        };
        s.add_request("r1".into(), vec![Sequence::new(1, "r1".into(), prompt(100), 8, 0.0)]);
        s.schedule().unwrap().unwrap();
        s.commit_chunk(1, 11).unwrap();
        // 11 tokens: two full blocks are cacheable (the last token is always computed)
        s.add_request("r2".into(), vec![Sequence::new(2, "r2".into(), prompt(200), 8, 0.0)]);
        s.schedule().unwrap().unwrap();
        assert_eq!(s.sequences[&2].cached_prompt_tokens, 8, "two whole blocks come from the cache");
        assert_eq!(s.prefix_cache_hits, 1);
        assert_eq!(s.sequences[&1].cached_prompt_tokens, 0);
        assert!(s.running.contains(&1) && s.running.contains(&2), "both sequences are batched");
        // a different prefix does not hit
        let other: Vec<i64> = (50..61).collect();
        s.add_request("r3".into(), vec![Sequence::new(3, "r3".into(), other, 8, 0.0)]);
        s.schedule().unwrap().unwrap();
        assert_eq!(s.sequences[&3].cached_prompt_tokens, 0);
    }
}
