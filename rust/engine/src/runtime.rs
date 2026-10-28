use crate::mem::block_allocator::BlockManager;
use crate::mem::prefix_cache::PrefixCache;
use crate::scheduler::{Scheduler, SchedulerConfig, Sequence};
use crate::GgufModel;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub struct TokenEvent {
    pub request_id: String,
    pub sequence_id: i64,
    pub token_id: i64,
    pub finished: bool,
    pub finish_reason: Option<&'static str>,
    pub logprobs: Option<crate::sampling::TokenLogprobs>,
    pub cached_prompt_tokens: usize,
    pub choice_index: usize,
    pub request_finished: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchStats {
    pub forward_calls: usize,
    pub rows_total: usize,
    pub calls_with_multiple_rows: usize,
    pub max_rows: usize,
    pub prefill_rows: usize,
    pub decode_rows: usize,
    pub draft_proposed: usize,
    pub draft_accepted: usize,
    /// Tokens the grammar forced and the engine appended without sampling (jump-forward decoding).
    pub forced_tokens: usize,
    /// Steps after which forced tokens were appended.
    pub jumps: usize,
    pub scheduler_ns: u64,
    pub batch_build_ns: u64,
    pub model_forward_ns: u64,
    pub postprocess_ns: u64,
}

struct BatchRow {
    sequence_id: i64,
    token_id: i64,
    length: i64,
    table: std::sync::Arc<Vec<i64>>,
    embedding: Option<Vec<f32>>,
}

/// Scheduler-driven inference over the memory-mapped GGUF model.
/// One shared paged KV pool backs every active sequence; each scheduler step is
/// evaluated as one ragged batch by the Rust decoder.
pub struct InferenceRuntime {
    pub model: GgufModel,
    scheduler: Scheduler,
    key: Vec<u16>,
    value: Vec<u16>,
    pending_logits: std::collections::HashMap<i64, Vec<f32>>,
    pending_choices: std::collections::HashMap<i64, Vec<Sequence>>,
    shared_choices_cpu: bool,
    constraints: std::collections::HashMap<i64, llguidance::Matcher>,
    /// Jump-forward decoding for constrained sequences (None: off).
    jump_forward: Option<std::sync::Arc<crate::structured::JumpForward>>,
    draft: Option<crate::draft::DraftEngine>,
    mtp: Option<crate::draft::MtpEngine>,
    suffix: Option<crate::speculative::SuffixCache>,
    spec_head: Option<crate::draft::SpecHeadEngine>,
    /// Distributions the draft model drew this step's draft tokens from (only for sequences with a draft model proposal).
    draft_q: std::collections::HashMap<i64, Vec<Option<crate::speculative::Dist>>>,
    spec_method: crate::speculative::SpeculativeMethod,
    /// Largest number of draft tokens per verification (`--num-speculative-tokens`).
    spec_tokens: usize,
    spec_stats: crate::speculative::SpecStats,
    /// Statistics since the last flush to the metrics.
    spec_delta: crate::speculative::SpecStats,
    decoder: Option<crate::rust_decoder::RustDecoder>,
    /// The disk tier for cached prefix blocks (off unless `FLLM_KV_DISK_DIR` is set).
    kv_tier: std::sync::Arc<crate::mem::kv_tier::KvTier>,
    /// Writes for the disk tier, off the step path: file I/O must never delay a token.
    spill_tx: Option<std::sync::mpsc::SyncSender<SpillJob>>,
    kv_warmed: bool,
    /// KV data of the sequences preempted by swap, one export per block in table order.
    swap_store: std::collections::HashMap<i64, Vec<Vec<u8>>>,
    /// n > 1 can fork sequences on this backend (host arrays, or device pools whose block copy passed its self check).
    fork_ok: bool,
    swap_out_blocks: u64,
    swap_in_blocks: u64,
    image_embeddings: std::collections::HashMap<i64, std::collections::HashMap<usize, Vec<f32>>>,
    /// Three-axis (M-RoPE) position ids of sequences whose prompt holds images or video.
    mrope_positions: std::collections::HashMap<i64, crate::vision::mrope::MropePositions>,
    /// DeepStack features of sequences with media: per prompt index, one row per decoder layer.
    deepstack: std::collections::HashMap<i64, std::collections::HashMap<usize, Vec<Vec<f32>>>>,
    /// Custom visible lengths for attention (bidirectional image attention, prefix-LM mask).
    visible_lengths: std::collections::HashMap<i64, Vec<usize>>,
    pub vision: Option<std::sync::Arc<dyn crate::vision::MultimodalEncoder>>,
    eos_ids: Vec<i64>,
    next_sequence_id: i64,
    batch_stats: BatchStats,
    metrics: Option<std::sync::Arc<crate::metrics::Metrics>>,
}

impl InferenceRuntime {
    pub fn new(
        model: GgufModel,
        max_sequences: usize,
        max_batched_tokens: usize,
        eos_ids: Vec<i64>,
    ) -> Result<Self, String> {
        if max_sequences == 0 || max_sequences > model.max_sequences() || max_batched_tokens == 0 {
            return Err(
                "scheduler capacities must be positive and within the model KV pool".into(),
            );
        }
        Self::with_speculative(model, max_sequences, max_batched_tokens, eos_ids, 0, 0)
    }

    pub fn with_speculative(
        model: GgufModel,
        max_sequences: usize,
        max_batched_tokens: usize,
        eos_ids: Vec<i64>,
        speculative_lookup: usize,
        speculative_window: usize,
    ) -> Result<Self, String> {
        if max_sequences == 0 || max_sequences > model.max_sequences() || max_batched_tokens == 0 {
            return Err(
                "scheduler capacities must be positive and within the model KV pool".into(),
            );
        }
        let block_size = model.block_size();
        let pool_blocks = model.pool_blocks();
        let mut bm =
            BlockManager::new(pool_blocks as i64, block_size as i64).map_err(|e| e.to_string())?;
        // The last block of the pool is never handed to a sequence: the padding rows of the decoder's CUDA graphs write their
        // key/value pair there (the same "null block" idea as in vLLM).
        if pool_blocks >= 2 {
            let null_block = pool_blocks - 1;
            bm.allocator.free_blocks.retain(|&b| b != null_block);
            bm.allocator.refcounts[null_block] = 1;
        }
        let stateful = model.is_stateful();
        let pc = PrefixCache::new(pool_blocks as i64, true, true).map_err(|e| e.to_string())?;
        let config = SchedulerConfig {
            block_size,
            max_num_seqs: max_sequences,
            max_num_batched_tokens: max_batched_tokens,
            // FLLM_PREFIX_CACHE=off: no shared KV blocks between requests (a tenant that must not share)
            prefix_cache: !stateful
                && !matches!(
                    std::env::var("FLLM_PREFIX_CACHE").unwrap_or_default().trim().to_ascii_lowercase().as_str(),
                    "off" | "0" | "no" | "false"
                ),
            // FLLM_PREEMPTION=swap: a preempted sequence's KV goes to host memory instead of being recomputed
            preemption: match std::env::var("FLLM_PREEMPTION").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
                "swap" => "swap".to_string(),
                _ => "recompute".to_string(),
            },
            speculative_lookup,
            speculative_window: if speculative_lookup > 0 && speculative_window == 0 {
                32
            } else {
                speculative_window
            },
            speculative_adaptive: !matches!(std::env::var("FLLM_SPEC_ADAPTIVE").as_deref(), Ok("0")),
            ..SchedulerConfig::default()
        };
        let scheduler = Scheduler::new(config, bm, Some(pc), model.max_context());
        let key_len = model.key_buffer_len()?;
        let value_len = model.value_buffer_len()?;
        // the disk tier writes on its own thread: file I/O must never sit between two tokens
        let kv_tier = std::sync::Arc::new(crate::mem::kv_tier::KvTier::from_env());
        let spill_tx = spawn_spill_writer(std::sync::Arc::clone(&kv_tier));
        Ok(Self {
            model,
            scheduler,
            key: vec![0; key_len],
            value: vec![0; value_len],
            pending_logits: std::collections::HashMap::new(),
            pending_choices: Default::default(),
            shared_choices_cpu: false,
            constraints: std::collections::HashMap::new(),
            jump_forward: None,
            draft: None,
            mtp: None,
            suffix: None,
            spec_head: None,
            draft_q: Default::default(),
            spec_method: if speculative_lookup > 0 {
                crate::speculative::SpeculativeMethod::Ngram
            } else {
                crate::speculative::SpeculativeMethod::Off
            },
            spec_tokens: speculative_lookup,
            spec_stats: Default::default(),
            spec_delta: Default::default(),
            decoder: None,
            kv_tier,
            spill_tx,
            kv_warmed: false,
            swap_store: Default::default(),
            fork_ok: false,
            swap_out_blocks: 0,
            swap_in_blocks: 0,
            image_embeddings: Default::default(),
            mrope_positions: Default::default(),
            deepstack: Default::default(),
            visible_lengths: Default::default(),
            vision: None,
            eos_ids,
            next_sequence_id: 1,
            batch_stats: BatchStats::default(),
            metrics: None,
        })
    }

    /// Copies a running sequence together with its KV, so two continuations can be explored from the same point. The copy
    /// is independent: a token appended to one never touches the other. This is the primitive beam search needs.
    pub fn fork_sequence(&mut self, source: i64) -> Result<i64, String> {
        let source_seq = self
            .scheduler
            .sequences
            .get(&source)
            .cloned()
            .ok_or_else(|| format!("unknown sequence to fork: {source}"))?;
        let blocks = self
            .scheduler
            .bm
            .get_table(source)
            .map_err(|error| error.to_string())?
            .blocks
            .clone();
        if blocks.is_empty() {
            return Err("cannot fork a sequence without KV blocks".into());
        }
        if self.available_blocks() < blocks.len() {
            return Err("not enough free KV blocks to fork".into());
        }
        let new_id = self.next_sequence_id;
        self.next_sequence_id = new_id.checked_add(1).ok_or("sequence id space exhausted")?;
        self.scheduler
            .bm
            .new_table(new_id)
            .map_err(|error| error.to_string())?;
        // copy block by block; a failure gives every block back, so a fork either happens or it does not
        let mut copied: Vec<usize> = Vec::with_capacity(blocks.len());
        for &block in &blocks {
            let free = match self.scheduler.alloc_free_block() {
                Ok(free) => free,
                Err(error) => {
                    self.discard_fork(new_id, &copied);
                    return Err(error.to_string());
                }
            };
            if let Err(error) = self.copy_prompt_block(block, free as usize) {
                self.scheduler.free_block(free);
                self.discard_fork(new_id, &copied);
                return Err(error);
            }
            if let Err(error) = self.scheduler.bm.claim_block_for(new_id, free) {
                self.scheduler.free_block(free);
                self.discard_fork(new_id, &copied);
                return Err(error.to_string());
            }
            copied.push(free as usize);
        }
        let mut sequence = source_seq;
        sequence.seq_id = new_id;
        sequence.blocks = copied.clone();
        // the copy is already computed, so admission has nothing to schedule for it
        sequence.num_computed = sequence.num_computed.max(sequence.prefill_target);
        self.scheduler.requests.entry(sequence.req_id.clone()).or_default().push(new_id);
        self.scheduler.sequences.insert(new_id, sequence);
        self.scheduler.waiting.push_back(new_id);
        Ok(new_id)
    }

    /// Gives the blocks of a failed fork back and drops its table.
    fn discard_fork(&mut self, new_id: i64, copied: &[usize]) {
        for &block in copied {
            self.scheduler.free_block(block as i64);
        }
        let _ = self.scheduler.bm.drop_table(new_id);
    }

    /// Sets the per-request prefix salt (`cache_salt`): requests with different salts never share KV blocks.
    pub fn set_request_prefix_salt(&mut self, request_id: &str, salt: u64) {
        if salt == 0 {
            return;
        }
        for sequence in self.scheduler.sequences.values_mut() {
            if sequence.req_id == request_id {
                sequence.prefix_salt = salt;
            }
        }
    }

    pub fn submit(
        &mut self,
        request_id: String,
        prompt_tokens: Vec<i64>,
        max_tokens: usize,
        temperature: f64,
    ) -> Result<i64, String> {
        self.submit_with_sampling(
            request_id,
            prompt_tokens,
            max_tokens,
            crate::sampling::SamplingParams {
                temperature,
                ..Default::default()
            },
        )
    }

    pub fn submit_with_sampling(
        &mut self,
        request_id: String,
        prompt_tokens: Vec<i64>,
        max_tokens: usize,
        sampling: crate::sampling::SamplingParams,
    ) -> Result<i64, String> {
        if request_id.is_empty() || prompt_tokens.is_empty() || max_tokens == 0 {
            return Err("request id, prompt tokens, and max_tokens must be non-empty".into());
        }
        if prompt_tokens.len().saturating_add(max_tokens) > self.model.max_context() {
            return Err("prompt plus max_tokens exceeds the configured model context".into());
        }
        if self.scheduler.requests.contains_key(&request_id) {
            return Err(format!("request id already active: {request_id}"));
        }
        if sampling.choices == 0 || sampling.choices > 16 {
            return Err("n must be between 1 and 16".into());
        }
        if sampling.choices > 1 && (!self.supports_shared_choices() || self.model.is_stateful()) {
            return Err(
                "n > 1 requires a stateless model on a backend with KV block copy and no draft/MTP engine".into(),
            );
        }
        if sampling.choices > 1
            && prompt_tokens.len().div_ceil(self.model.block_size()) + sampling.choices
                > self.model.pool_blocks()
        {
            return Err("prompt and n choices exceed the KV pool capacity".into());
        }
        if sampling.choices > 1 && self.model.kv_kind() != 1 {
            // every region of a block (codes, scales, ...) must be a whole number of words to copy the tail block
            self.kv_block_regions(true)?;
            self.kv_block_regions(false)?;
        }
        if sampling.choices > 1 && self.available_blocks() < sampling.choices - 1 {
            return Err("insufficient free KV tail blocks for n choices".into());
        }
        let seq_id = self.next_sequence_id;
        self.next_sequence_id = self
            .next_sequence_id
            .checked_add(1)
            .ok_or_else(|| "sequence id space exhausted".to_owned())?;
        let sequence = Sequence::new_with_sampling(
            seq_id,
            request_id.clone(),
            prompt_tokens,
            max_tokens,
            sampling,
        );
        let mut sequence = sequence;
        sequence.deadline = sequence
            .sampling
            .timeout_secs
            .map(|secs| Instant::now() + std::time::Duration::from_secs_f64(secs));
        let mut followers = Vec::new();
        for index in 1..sequence.sampling.choices {
            let id = self.next_sequence_id;
            self.next_sequence_id = id.checked_add(1).ok_or("sequence id space exhausted")?;
            let mut params = sequence.sampling.clone();
            params.seed = params
                .seed
                .map(|seed| seed.wrapping_add((index as u64).wrapping_mul(0x9e3779b97f4a7c15)));
            let mut follower = Sequence::new_with_sampling(
                id,
                request_id.clone(),
                sequence.tokens.clone(),
                max_tokens,
                params,
            );
            follower.choice_index = index;
            follower.deadline = sequence.deadline;
            self.scheduler.bm.new_table(id).map_err(|e| e.to_string())?;
            self.scheduler
                .bm
                .append_block(id)
                .map_err(|e| e.to_string())?;
            followers.push(follower);
        }
        if !followers.is_empty() {
            self.pending_choices.insert(seq_id, followers);
        }
        self.scheduler.add_request(request_id, vec![sequence]);
        Ok(seq_id)
    }

    pub fn supports_shared_choices(&self) -> bool {
        self.fork_ok
            && !self.model.is_stateful()
            && self.draft.is_none()
            && self.mtp.is_none()
    }
    fn finish_choice(&mut self, request: &str, sid: i64) -> Result<(), String> {
        self.scheduler.release(sid).map_err(|e| e.to_string())?;
        self.scheduler.sequences.remove(&sid);
        self.pending_logits.remove(&sid);
        self.constraints.remove(&sid);
        if let Some(ids) = self.scheduler.requests.get_mut(request) {
            ids.retain(|id| *id != sid);
        }
        if self
            .scheduler
            .requests
            .get(request)
            .is_some_and(Vec::is_empty)
        {
            self.scheduler.requests.remove(request);
        }
        Ok(())
    }
    /// Fork only after the complete prompt has been computed once. Full blocks are
    /// immutable shared cache entries; each writable tail gets its own physical block.
    fn fork_prompt_choices(&mut self, source: i64) -> Result<(), String> {
        let Some(mut followers) = self.pending_choices.remove(&source) else {
            return Ok(());
        };
        let leader = self
            .scheduler
            .sequences
            .get(&source)
            .ok_or("missing choice leader")?;
        let prompt_len = leader.prefill_target;
        let cached = leader.cached_prompt_tokens;
        let request = leader.req_id.clone();
        let block_size = self.model.block_size();
        let full = prompt_len / block_size;
        let partial = !prompt_len.is_multiple_of(block_size);
        let blocks = self
            .scheduler
            .bm
            .get_table(source)
            .map_err(|e| e.to_string())?
            .blocks
            .clone();
        let logits = self
            .pending_logits
            .get(&source)
            .ok_or("missing shared prompt logits")?
            .clone();
        for mut follower in followers.drain(..) {
            let id = follower.seq_id;
            let tail = self
                .scheduler
                .bm
                .get_table(id)
                .map_err(|e| e.to_string())?
                .blocks
                .first()
                .copied()
                .ok_or("missing reserved choice tail")?;
            self.scheduler
                .bm
                .get_table_mut(id)
                .map_err(|e| e.to_string())?
                .blocks
                .clear();
            for &block in blocks.iter().take(full) {
                self.scheduler
                    .bm
                    .claim_block_for(id, block as i64)
                    .map_err(|e| e.to_string())?;
            }
            self.scheduler
                .bm
                .get_table_mut(id)
                .map_err(|e| e.to_string())?
                .blocks
                .push(tail);
            if partial {
                let old = *blocks.get(full).ok_or("missing shared prompt tail block")?;
                self.copy_prompt_block(old, tail)?;
            }
            self.scheduler
                .bm
                .get_table_mut(id)
                .map_err(|e| e.to_string())?
                .num_tokens = prompt_len as i64;
            follower.num_computed = prompt_len;
            follower.cached_prompt_tokens = cached;
            follower.blocks = self
                .scheduler
                .bm
                .get_table(id)
                .map_err(|e| e.to_string())?
                .blocks
                .clone();
            self.pending_logits.insert(id, logits.clone());
            self.scheduler
                .requests
                .get_mut(&request)
                .ok_or("missing grouped request")?
                .push(id);
            self.scheduler.waiting.push_back(id);
            self.scheduler.sequences.insert(id, follower);
        }
        Ok(())
    }
    /// `copy_prompt_block` for models whose layers differ in KV geometry: every layer that owns a cache is copied with its own layout.
    fn copy_prompt_block_layered(&mut self, src: usize, dst: usize) -> Result<(), String> {
        let cfg = self.model.config().clone();
        let (fmt_k, fmt_v) = self.model.kv_formats();
        let layout = self.model.kv_pool_layout();
        let slots = self.model.pool_blocks() * self.model.block_size();
        let (pool_blocks, block_size) = (self.model.pool_blocks(), self.model.block_size());
        if src >= pool_blocks || dst >= pool_blocks {
            return Err("invalid shared KV block geometry".into());
        }
        for l in (0..cfg.layers).filter(|&l| cfg.layer_kv_source(l).is_none()) {
            let heads = cfg.layer_kv_heads(l);
            for (cache, fmt, dim, offset, words) in [
                (&mut self.key, fmt_k, cfg.layer_head_dim(l), layout.key_offsets[l], layout.key_words[l]),
                (&mut self.value, fmt_v, cfg.layer_value_dim(l), layout.value_offsets[l], layout.value_words[l]),
            ] {
                let regions = crate::kv_format::KvLayout::new(fmt, slots, heads, dim)
                    .block_regions(block_size)
                    .ok_or_else(|| format!("shared choices need KV blocks of whole words for {}", fmt.name()))?;
                let region = cache.get_mut(offset..offset + words).ok_or("shared KV buffer does not match layer geometry")?;
                copy_cache_regions(region, src, dst, pool_blocks, 1, words, &regions)?;
            }
        }
        Ok(())
    }

    fn copy_prompt_block(&mut self, src: usize, dst: usize) -> Result<(), String> {
        if !self.model.config().uniform_geometry() {
            return self.copy_prompt_block_layered(src, dst);
        }
        if let Some(result) = self.decoder.as_ref().and_then(|d| d.device_kv_copy(src, dst)) {
            return result;
        }
        let generic = !matches!(self.model.kv_kind(), 1 | 2);
        let region_sets = if generic {
            [self.kv_block_regions(true)?, self.kv_block_regions(false)?]
        } else {
            [Vec::new(), Vec::new()]
        };
        let cfg = self.model.config();
        let (fmt_k, fmt_v) = self.model.kv_formats();
        for (index, (cache, dim, fmt)) in [
            (&mut self.key, cfg.head_dim, fmt_k),
            (&mut self.value, cfg.value_dim, fmt_v),
        ]
        .into_iter()
        .enumerate()
        {
            let kv = cfg.kv_heads * dim;
            if generic {
                let slots = self.model.pool_blocks() * self.model.block_size();
                let layer_words = fmt.layer_words(slots, cfg.kv_heads, dim);
                let pool_blocks = self.model.pool_blocks();
                copy_cache_regions(cache, src, dst, pool_blocks, cfg.layers, layer_words, &region_sets[index])?;
                continue;
            }
            let scales = if self.model.kv_kind() == 2 {
                let group = if dim.is_multiple_of(32) { 32 } else { dim };
                self.model.block_size() * (kv / group)
            } else {
                0
            };
            copy_cache_block(
                cache,
                src,
                dst,
                self.model.pool_blocks(),
                cfg.layers,
                self.model.block_size() * kv,
                scales,
            )?;
        }
        Ok(())
    }

    /// The pieces of block `block` in the host arrays.
    fn host_block_pieces(&self, block: usize) -> Result<crate::kv_blocks::BlockPieces, String> {
        let cfg = self.model.config();
        let (fmt_k, fmt_v) = self.model.kv_formats();
        let slots = self.model.pool_blocks() * self.model.block_size();
        Ok(crate::kv_blocks::BlockPieces::new(
            cfg.layers,
            fmt_k.layer_words(slots, cfg.kv_heads, cfg.head_dim),
            &self.kv_block_regions(true)?,
            fmt_v.layer_words(slots, cfg.kv_heads, cfg.value_dim),
            &self.kv_block_regions(false)?,
            block,
        ))
    }
    /// Export the KV data of a block (all layers, keys and values) to host memory.
    fn kv_export_block(&mut self, block: usize) -> Result<Vec<u8>, String> {
        if let Some(result) = self.decoder.as_ref().and_then(|d| d.device_kv_export(block)) {
            return result;
        }
        crate::kv_blocks::host_export(&self.key, &self.value, &self.host_block_pieces(block)?)
    }
    fn kv_import_block(&mut self, block: usize, data: &[u8]) -> Result<(), String> {
        if let Some(result) = self.decoder.as_ref().and_then(|d| d.device_kv_import(block, data)) {
            return result;
        }
        let pieces = self.host_block_pieces(block)?;
        crate::kv_blocks::host_import(&mut self.key, &mut self.value, &pieces, data)
    }
    /// Carry out the swap traffic of the scheduler: exports first (the freed blocks may be rewritten by this step's forward pass), then
    /// imports into the newly allocated blocks.
    fn carry_out_swaps(&mut self) -> Result<(), String> {
        let work = self.scheduler.take_swaps();
        for (sid, blocks) in work.out {
            let mut data = Vec::with_capacity(blocks.len());
            for block in blocks {
                data.push(self.kv_export_block(block)?);
            }
            self.swap_out_blocks += data.len() as u64;
            self.swap_store.insert(sid, data);
        }
        for sid in work.dropped {
            self.swap_store.remove(&sid);
        }
        for (sid, blocks) in work.into {
            let data = self.swap_store.remove(&sid).ok_or("swapped KV data is missing")?;
            if data.len() != blocks.len() {
                return Err("swapped KV data does not match the new blocks".into());
            }
            self.swap_in_blocks += data.len() as u64;
            for (block, bytes) in blocks.into_iter().zip(data) {
                self.kv_import_block(block, &bytes)?;
            }
        }
        Ok(())
    }
    /// Block copy on an accelerator is verified once on the device itself: copy block 0 to block 1 and compare the exports. A failure
    /// (or a backend without the operations) keeps n > 1 and swapping off for this run.
    fn device_fork_self_check(&mut self) -> bool {
        if self.model.pool_blocks() < 3 {
            return false;
        }
        let check = (|| -> Result<bool, String> {
            let Some(copied) = self.decoder.as_ref().and_then(|d| d.device_kv_copy(0, 1)) else { return Ok(false) };
            copied?;
            let a = self.kv_export_block(0)?;
            let b = self.kv_export_block(1)?;
            let c = self.kv_export_block(2)?;
            self.kv_import_block(2, &a)?;
            Ok(a == b && self.kv_export_block(2)? == a && c.len() == a.len())
        })();
        match check {
            Ok(true) => true,
            Ok(false) => false,
            Err(e) => {
                eprintln!("[fllm-rs] KV block copy is not available on this device ({e}); n > 1 and swap preemption stay off");
                false
            }
        }
    }

    /// `(offset, words per block)` of the regions of a key (`true`) or value pool block.
    fn kv_block_regions(&self, key: bool) -> Result<Vec<(usize, usize)>, String> {
        let cfg = self.model.config();
        let (fmt_k, fmt_v) = self.model.kv_formats();
        let (fmt, dim) = if key { (fmt_k, cfg.head_dim) } else { (fmt_v, cfg.value_dim) };
        let slots = self.model.pool_blocks() * self.model.block_size();
        crate::kv_format::KvLayout::new(fmt, slots, cfg.kv_heads, dim)
            .block_regions(self.model.block_size())
            .ok_or_else(|| format!("shared choices need KV blocks of whole words for {}", fmt.name()))
    }

    /// Whether KV blocks can be copied and exported on this backend, which `n > 1` and preemption by swap need.
    pub fn kv_block_ops_available(&self) -> bool {
        self.fork_ok
    }
    /// Sequences preempted so far, and the KV blocks swapped out to and back in from host memory.
    /// Empties the prefix cache; returns how many entries it held.
    pub fn clear_prefix_cache(&mut self) -> Result<usize, String> {
        self.scheduler.clear_prefix_cache().map_err(|e| e.to_string())
    }

    /// What the prefix cache holds, for `GET /admin/prefix-cache`.
    pub fn prefix_cache_report(&self) -> serde_json::Value {
        let hits = self.scheduler.prefix_cache_hits;
        match self.scheduler.pc.as_ref() {
            None => serde_json::json!({"enabled": false, "hits": hits}),
            Some(pc) => serde_json::json!({
                "enabled": self.scheduler.cfg.prefix_cache,
                "entries": pc.len(),
                "hits": hits,
                "blocks": pc.entries_lru().iter().map(|entry| serde_json::json!({
                    "hash": format!("{:016x}", entry.chain_hash),
                    "tokens": entry.tokens.len(),
                })).collect::<Vec<_>>(),
                "tree": pc.tree_snapshot(),
            }),
        }
    }

    pub fn preemption_stats(&self) -> (u64, u64, u64) {
        (self.scheduler.preemptions, self.swap_out_blocks, self.swap_in_blocks)
    }

    /// Scheduler settings (policy, watermark, prefill limits); change them before the first request.
    pub fn scheduler_config_mut(&mut self) -> &mut SchedulerConfig {
        &mut self.scheduler.cfg
    }

    /// Abort the requests whose `timeout` has passed and return their ids; their KV blocks are released.
    pub fn expire_requests(&mut self) -> Vec<String> {
        let expired = self.scheduler.expired_requests(Instant::now());
        for id in &expired {
            let _ = self.cancel_request(id);
            if let Some(metrics) = &self.metrics {
                metrics.timeout();
            }
        }
        expired
    }

    pub fn active_requests(&self) -> usize {
        self.scheduler.requests.len()
    }

    /// Switch jump-forward decoding on (`Some`) or off (`None`). It applies to grammar-constrained sequences only; see
    /// `structured::JumpForward` for the rules and `docs/structured-output.md` for what it does and does not change.
    pub fn set_jump_forward(&mut self, jump: Option<std::sync::Arc<crate::structured::JumpForward>>) {
        self.jump_forward = jump;
    }

    /// After a sampled token of a constrained sequence: if the grammar forces the next bytes, append them as tokens (events like
    /// sampled ones) and send the sequence back to prefill so that one forward pass computes the KV entries of the sampled token and
    /// all forced tokens and returns the logits for the token after them. Nothing changes for sequences that cannot jump safely.
    fn jump_forward_after(&mut self, sid: i64, last: i64, events: &mut Vec<TokenEvent>) -> Result<(), String> {
        /// Tokens worth of KV blocks that must be free before a jump is tried.
        const RESERVE: usize = 1024;
        let Some(jump) = self.jump_forward.clone() else {
            return Ok(());
        };
        // Draft models, MTP and speculative heads keep per-token state of their own that forced tokens would bypass.
        if self.draft.is_some() || self.mtp.is_some() || self.spec_head.is_some() {
            return Ok(());
        }
        let Some(sequence) = self.scheduler.sequences.get(&sid) else {
            return Ok(());
        };
        let s = &sequence.sampling;
        // Settings whose per-token effect is not a pure function of the one allowed token decide by plain decoding.
        if !sequence.constrained
            || sequence.has_media
            || s.choices != 1
            || s.mirostat != 0
            || s.logprobs.is_some()
            || !s.logit_bias.is_empty()
            || !s.allowed_token_ids.is_empty()
            || !s.bad_word_ids.is_empty()
            || sequence.output_len < s.min_tokens
        {
            return Ok(());
        }
        let remaining = sequence.remaining_max();
        if remaining == 0 {
            return Ok(());
        }
        let bm = &self.scheduler.bm;
        let needed = bm.num_blocks_needed(sid, remaining.min(RESERVE) as i64).map_err(|e| e.to_string())?;
        if needed > bm.allocator.available() {
            return Ok(());
        }
        let Some(matcher) = self.constraints.get_mut(&sid) else {
            return Ok(());
        };
        let forced = jump.plan(matcher, last as u32, remaining);
        if forced.is_empty() {
            return Ok(());
        }
        let sequence = self.scheduler.sequences.get_mut(&sid).ok_or("missing jump sequence")?;
        let request_id = sequence.req_id.clone();
        let choice_index = sequence.choice_index;
        let cached_prompt_tokens = sequence.cached_prompt_tokens;
        let stop_at = forced.iter().position(|&t| sequence.sampling.stops_on_token(i64::from(t), &self.eos_ids));
        let mut appended = 0usize;
        let mut finished = false;
        for (j, &token) in forced.iter().enumerate() {
            let token = i64::from(token);
            sequence.append_token(token);
            appended += 1;
            let hit_eos = stop_at == Some(j);
            let hit_limit = sequence.remaining_max() == 0;
            finished = hit_eos || hit_limit;
            events.push(TokenEvent {
                request_id: request_id.clone(),
                sequence_id: sid,
                token_id: token,
                finished,
                finish_reason: if hit_eos {
                    Some("stop")
                } else if hit_limit {
                    Some("length")
                } else {
                    None
                },
                logprobs: None,
                cached_prompt_tokens,
                choice_index,
                request_finished: false,
            });
            if finished {
                break;
            }
        }
        self.batch_stats.forced_tokens += appended;
        self.batch_stats.jumps += 1;
        if let Some(metrics) = &self.metrics {
            metrics.jump_forward(appended);
        }
        if finished {
            self.finish_choice(&request_id, sid)?;
            let done = !self.scheduler.requests.contains_key(&request_id);
            if let Some(event) = events.last_mut() {
                event.request_finished = done;
            }
        } else {
            self.scheduler.begin_jump(sid, appended).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn set_constraint(
        &mut self,
        sequence_id: i64,
        matcher: llguidance::Matcher,
    ) -> Result<(), String> {
        let sequence = self
            .scheduler
            .sequences
            .get_mut(&sequence_id)
            .ok_or("unknown constrained sequence")?;
        if sequence.sampling.choices > 1 {
            return Err("shared choices cannot use grammar constraints".into());
        }
        sequence.constrained = true;
        self.constraints.insert(sequence_id, matcher);
        Ok(())
    }

    /// Cancel an active request and release its scheduler sequence and KV blocks.
    pub fn cancel_request(&mut self, request_id: &str) -> Result<(), String> {
        let sequence_ids = self
            .scheduler
            .requests
            .get(request_id)
            .cloned()
            .unwrap_or_default();
        self.scheduler
            .free_request(request_id)
            .map_err(|e| e.to_string())?;
        for sequence_id in sequence_ids {
            self.scheduler.sequences.remove(&sequence_id);
            self.swap_store.remove(&sequence_id);
            self.pending_logits.remove(&sequence_id);
            if let Some(followers) = self.pending_choices.remove(&sequence_id) {
                for follower in followers {
                    self.scheduler
                        .bm
                        .drop_table(follower.seq_id)
                        .map_err(|e| e.to_string())?;
                }
            }
            self.constraints.remove(&sequence_id);
            self.image_embeddings.remove(&sequence_id);
        }
        Ok(())
    }

    pub fn available_blocks(&self) -> usize {
        self.scheduler.bm.allocator.available()
    }

    pub fn metrics_snapshot(&self) -> (usize, usize, usize, usize, u64) {
        (
            self.scheduler.running.len(),
            self.scheduler.waiting.len(),
            self.available_blocks(),
            self.model.pool_blocks(),
            self.scheduler.prefix_cache_hits,
        )
    }

    pub fn attach_metrics(&mut self, metrics: std::sync::Arc<crate::metrics::Metrics>) {
        self.metrics = Some(metrics);
    }

    /// Pool contextual hidden states using isolated KV blocks. Existing generation
    /// requests and their prefix cache never share these temporary allocations.
    pub fn embedding(&mut self, tokens: &[i64], pooling: &str) -> Result<Vec<f32>, String> {
        if tokens.is_empty() || tokens.len() > self.model.max_context() {
            return Err("embedding input must fit the model context".into());
        }
        if !matches!(pooling, "mean" | "last") {
            return Err("pooling must be mean or last".into());
        }
        let sid = self.next_sequence_id;
        self.next_sequence_id = sid.checked_add(1).ok_or("sequence id space exhausted")?;
        self.scheduler
            .bm
            .new_table(sid)
            .map_err(|e| e.to_string())?;
        let result = (|| {
            self.scheduler
                .bm
                .ensure_blocks(sid, tokens.len() as i64)
                .map_err(|e| e.to_string())?;
            let table = self.table_row(sid)?;
            let hidden = self.model.hidden_size();
            let mut output = vec![0.0f32; hidden];
            let chunk_size = self.scheduler.cfg.max_num_batched_tokens;
            for start in (0..tokens.len()).step_by(chunk_size) {
                let end = (start + chunk_size).min(tokens.len());
                let lengths: Vec<i64> = (start + 1..=end).map(|n| n as i64).collect();
                let tables = table.repeat(end - start);
                let states = Self::decoder(&self.decoder)?.forward(
                    &tokens[start..end],
                    &lengths,
                    &tables,
                    &mut self.key,
                    &mut self.value,
                    &[],
                    true,
                )?;
                for row in states.chunks_exact(hidden) {
                    if pooling == "last" {
                        output.copy_from_slice(row);
                    } else {
                        for (dst, value) in output.iter_mut().zip(row) {
                            *dst += value / tokens.len() as f32;
                        }
                    }
                }
            }
            let norm = output
                .iter()
                .map(|n| (*n as f64).powi(2))
                .sum::<f64>()
                .sqrt() as f32;
            if norm > 0.0 {
                for value in &mut output {
                    *value /= norm;
                }
            }
            if output.iter().any(|v| !v.is_finite()) {
                return Err("embedding produced a non-finite value".into());
            }
            Ok(output)
        })();
        self.scheduler
            .bm
            .drop_table(sid)
            .map_err(|e| e.to_string())?;
        result
    }

    pub fn batch_stats(&self) -> BatchStats {
        self.batch_stats
    }

    /// Speculation method in use, the draft length it was configured with and the cumulative statistics.
    pub fn speculative_status(&self) -> (crate::speculative::SpeculativeMethod, usize, crate::speculative::SpecStats) {
        (self.spec_method, self.spec_tokens, self.spec_stats)
    }

    /// Suffix-match speculation across the conversation and recent generations (`--speculative suffix`).
    pub fn enable_suffix(&mut self, tokens: usize, config: crate::speculative::SuffixConfig) -> Result<(), String> {
        if self.draft.is_some() || self.mtp.is_some() || self.suffix.is_some() || self.spec_head.is_some() || self.scheduler.cfg.speculative_lookup > 0 {
            return Err("speculative methods are mutually exclusive".into());
        }
        if tokens == 0 || tokens > 32 {
            return Err("the number of speculative tokens must be between 1 and 32".into());
        }
        self.suffix = Some(crate::speculative::SuffixCache::new(config));
        self.spec_method = crate::speculative::SpeculativeMethod::Suffix;
        self.spec_tokens = tokens;
        Ok(())
    }

    pub fn enable_mtp(&mut self) -> Result<(), String> {
        if self.draft.is_some() || self.suffix.is_some() || self.spec_head.is_some() || self.scheduler.cfg.speculative_lookup > 0 {
            return Err("speculative methods are mutually exclusive".into());
        }
        if self.decoder.is_some() {
            return Err("enable MTP before attaching compute".into());
        }
        self.mtp = Some(crate::draft::MtpEngine::new(&self.model)?);
        self.spec_method = crate::speculative::SpeculativeMethod::Mtp;
        self.spec_tokens = 1;
        // Main-model prefix hits cannot restore the corresponding MTP KV and
        // residual states. Warm both paths from the prompt until that is supported.
        self.scheduler.cfg.prefix_cache = false;
        self.scheduler.pc = None;
        Ok(())
    }
    pub fn enable_draft(&mut self, model: GgufModel, tokens: usize) -> Result<(), String> {
        let target = &self.model.mapping().index().metadata;
        let draft = &model.mapping().index().metadata;
        for key in [
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
        ] {
            if target.get(key) != draft.get(key) {
                return Err(format!(
                    "draft model must use the same token IDs as the target ({key})"
                ));
            }
        }
        if self.draft.is_some() || self.mtp.is_some() || self.suffix.is_some() || self.spec_head.is_some() || self.scheduler.cfg.speculative_lookup > 0 {
            return Err("speculative methods are mutually exclusive".into());
        }
        self.draft = Some(crate::draft::DraftEngine::new(model, tokens)?);
        self.spec_method = crate::speculative::SpeculativeMethod::Draft;
        self.spec_tokens = tokens;
        Ok(())
    }

    /// Enable Medusa, EAGLE, or EAGLE-3 speculative draft head.
    pub fn enable_spec_head(
        &mut self,
        model: GgufModel,
        method: crate::speculative::SpeculativeMethod,
        tokens: usize,
    ) -> Result<(), String> {
        if self.draft.is_some() || self.mtp.is_some() || self.suffix.is_some() || self.spec_head.is_some() || self.scheduler.cfg.speculative_lookup > 0 {
            return Err("speculative methods are mutually exclusive".into());
        }
        if tokens == 0 || tokens > 32 {
            return Err("the number of speculative tokens must be between 1 and 32".into());
        }
        self.spec_head = Some(crate::draft::SpecHeadEngine::new(&model, method)?);
        self.spec_method = method;
        self.spec_tokens = tokens;
        Ok(())
    }
    /// Load the decoder weights for `device` ("cpu" or a GPU backend). Must be called once before
    /// any request is processed.
    pub fn attach_compute(&mut self, device: &str) -> Result<(), String> {
        let Some((key_format, value_format)) = crate::kv_format::unpack_kind(self.model.kv_kind()) else {
            return Err("unknown KV cache format".into());
        };
        for l in 0..self.model.config().layers {
            key_format.check_dim(self.model.config().layer_head_dim(l))?;
            value_format.check_dim(self.model.config().layer_value_dim(l))?;
        }
        self.decoder = Some(crate::rust_decoder::RustDecoder::load(&self.model, device)?);
        // the host CPU runs the model unless an accelerator backend took it (`auto` falls back to the CPU on machines without a usable GPU)
        self.shared_choices_cpu = self.decoder.as_ref().is_some_and(|decoder| !decoder.uses_accelerator());
        if self.model.mla_latent() && self.shared_choices_cpu {
            return Err("FLLM_MLA=latent runs on the CUDA backend only, and the model is on the host; unset FLLM_MLA or use FLLM_MLA=expanded".into());
        }
        self.fork_ok = self.shared_choices_cpu || self.device_fork_self_check();
        if let Some(draft) = &mut self.draft {
            draft.attach_compute(device)?;
        }
        if let Some(mtp) = &mut self.mtp {
            mtp.attach_compute(&self.model, device)?;
        }
        Ok(())
    }
    fn decoder(
        slot: &Option<crate::rust_decoder::RustDecoder>,
    ) -> Result<&crate::rust_decoder::RustDecoder, String> {
        slot.as_ref()
            .ok_or_else(|| "model compute is not initialized (call attach_compute)".to_owned())
    }
    pub fn attach_image_embeddings(
        &mut self,
        sid: i64,
        embeddings: Vec<(usize, Vec<f32>)>,
    ) -> Result<(), String> {
        if embeddings.is_empty() {
            return Ok(());
        }
        Self::decoder(&self.decoder)?;
        let seq = self
            .scheduler
            .sequences
            .get_mut(&sid)
            .ok_or("unknown media sequence")?;
        if embeddings.iter().any(|(p, x)| {
            *p >= seq.prefill_target
                || x.len() != self.model.hidden_size()
                || x.iter().any(|v| !v.is_finite())
        }) {
            return Err("invalid image embeddings".into());
        }
        if seq.sampling.choices > 1 {
            return Err("shared choices cannot use image embeddings".into());
        }
        seq.has_media = true;
        seq.constrained = true;
        self.image_embeddings
            .insert(sid, embeddings.into_iter().collect());
        Ok(())
    }

    /// Attaches the embeddings and position ids of a prompt with media to a submitted sequence.
    pub fn attach_media(
        &mut self,
        sid: i64,
        inputs: crate::vision::pipeline::MediaInputs,
    ) -> Result<(), String> {
        if let Some(positions) = &inputs.mrope {
            if !Self::decoder(&self.decoder)?.supports_mrope() {
                return Err("this model has no multimodal rotary (M-RoPE) sections; re-convert the checkpoint with this version of fllm-rs".into());
            }
            let length = self.scheduler.sequences.get(&sid).map(|s| s.prefill_target);
            if length != Some(positions.prompt.len()) {
                return Err("multimodal positions do not match the prompt".into());
            }
        }
        if !inputs.deepstack.is_empty() {
            let layers = Self::decoder(&self.decoder)?.layer_count();
            let hidden = self.model.hidden_size();
            if inputs.deepstack.len() > layers
                || inputs.deepstack.iter().flatten().any(|(_, v)| v.len() != hidden || v.iter().any(|x| !x.is_finite()))
            {
                return Err("invalid DeepStack features".into());
            }
        }
        self.attach_image_embeddings(sid, inputs.overrides)?;
        if let Some(positions) = inputs.mrope {
            self.mrope_positions.insert(sid, positions);
        }
        if let Some(vis) = inputs.visible_lengths {
            self.visible_lengths.insert(sid, vis);
        }
        if !inputs.deepstack.is_empty() {
            let mut by_position: std::collections::HashMap<usize, Vec<Vec<f32>>> = Default::default();
            for (level, rows) in inputs.deepstack.into_iter().enumerate() {
                for (position, values) in rows {
                    let entry = by_position.entry(position).or_default();
                    entry.resize(level + 1, Vec::new());
                    entry[level] = values;
                }
            }
            self.deepstack.insert(sid, by_position);
        }
        Ok(())
    }

    /// What the disk tier holds, for `GET /admin/kv-tier`.
    pub fn kv_tier_stats(&self) -> serde_json::Value {
        self.kv_tier.stats()
    }

    /// Empties the disk tier; returns how many blocks were removed.
    pub fn clear_kv_tier(&self) -> usize {
        self.kv_tier.clear()
    }

    /// Reads cached blocks back from the disk tier and registers them in the prefix cache, parents before children, so a
    /// prompt can hit the cache right after a restart. Returns how many blocks were taken.
    pub fn warm_prefix_cache(&mut self, limit: usize) -> Result<usize, String> {
        if !self.kv_tier.enabled() {
            return Ok(0);
        }
        let stored = self.kv_tier.entries();
        // children by parent, so a chain can be walked from its root
        let mut children: std::collections::HashMap<u64, Vec<crate::mem::kv_tier::Stored>> = std::collections::HashMap::new();
        for block in stored {
            children.entry(block.parent).or_default().push(block);
        }
        for list in children.values_mut() {
            list.sort_by_key(|block| block.hash);
        }
        let mut registered = 0usize;
        let mut roots: Vec<u64> = children.keys().copied().filter(|parent| *parent == 0).collect();
        if roots.is_empty() {
            // a tier that only holds a middle of a chain cannot be rebuilt: without the root the hash chain is unverifiable
            return Ok(0);
        }
        roots.sort_unstable();
        for root in roots {
            let mut chain: Vec<crate::mem::kv_tier::Stored> = Vec::new();
            let mut cursor = root;
            while let Some(next) = children.get(&cursor).and_then(|list| list.first()).cloned() {
                chain.push(next.clone());
                cursor = next.hash;
                if chain.len() >= limit {
                    break;
                }
            }
            if chain.is_empty() {
                continue;
            }
            // blocks first: a block has to hold its KV before the tree may point at it
            let mut blocks: Vec<(u64, Vec<i64>, i64)> = Vec::new();
            let mut failed = false;
            for block in &chain {
                let Some(bytes) = self.kv_tier.load(block.hash) else {
                    failed = true;
                    break;
                };
                let Ok(bid) = self.scheduler.alloc_free_block() else {
                    failed = true;
                    break;
                };
                if self.kv_import_block(bid as usize, &bytes).is_err() {
                    self.scheduler.free_block(bid);
                    failed = true;
                    break;
                }
                blocks.push((block.hash, block.tokens.clone(), bid));
            }
            if failed || blocks.is_empty() {
                for (_, _, bid) in &blocks {
                    self.scheduler.free_block(*bid);
                }
                continue;
            }
            match self.scheduler.register_prefix_chain(&blocks) {
                Ok(true) => registered += blocks.len(),
                _ => {
                    for (_, _, bid) in &blocks {
                        self.scheduler.free_block(*bid);
                    }
                }
            }
            if registered >= limit {
                break;
            }
        }
        Ok(registered)
    }

    /// Writes the blocks the scheduler just registered in the prefix cache to the disk tier, when one is configured.
    /// They are alive (the cache holds a reference), so their KV is still exactly what belongs to their hash.
    fn spill_pending(&mut self) {
        // export on this thread (it owns the decoder), write on another (it owns the file I/O)
        let Some(tx) = self.spill_tx.clone() else {
            self.scheduler.take_pending_spills();
            return;
        };
        for (hash, parent, block, tokens) in self.scheduler.take_pending_spills() {
            if block < 0 {
                continue;
            }
            let bytes = match self.kv_export_block(block as usize) {
                Ok(bytes) => bytes,
                Err(error) => {
                    tracing::warn!("could not export block {block} for the disk tier: {error}");
                    continue;
                }
            };
            // a full queue drops the block: losing a cache entry is better than delaying a token
            let _ = tx.try_send(SpillJob { hash, parent, tokens, bytes });
        }
    }

    pub fn step(&mut self) -> Result<Vec<TokenEvent>, String> {
        self.spill_pending();
        if !self.kv_warmed {
            self.kv_warmed = true;
            match self.warm_prefix_cache(128) {
                Ok(0) => {}
                Ok(count) => tracing::info!("warmed {count} prefix blocks from the disk tier"),
                Err(error) => tracing::warn!("could not warm the prefix cache from disk: {error}"),
            }
        }
        self.image_embeddings
            .retain(|sid, _| self.scheduler.sequences.contains_key(sid));
        self.draft_q.clear();
        let adaptive = self.scheduler.cfg.speculative_adaptive;
        self.mrope_positions
            .retain(|sid, _| self.scheduler.sequences.contains_key(sid));
        self.deepstack
            .retain(|sid, _| self.scheduler.sequences.contains_key(sid));
        self.visible_lengths
            .retain(|sid, _| self.scheduler.sequences.contains_key(sid));
        if let Some(draft) = &mut self.draft {
            let started = Instant::now();
            let active = self.scheduler.sequences.keys().copied().collect();
            draft.retain(&active)?;
            self.scheduler.external_drafts.clear();
            for &sid in &self.scheduler.running {
                let sequence = &self.scheduler.sequences[&sid];
                if sequence.num_computed >= sequence.prefill_target
                    && sequence.can_speculate()
                    && !self.pending_logits.contains_key(&sid)
                {
                    let limit = if adaptive { sequence.adaptive.length() } else { usize::MAX };
                    let (tokens, q) = draft.propose(sid, &sequence.tokens, sequence.remaining_max(), limit, &sequence.sampling)?;
                    if !tokens.is_empty() {
                        self.scheduler.external_drafts.insert(sid, tokens);
                        self.draft_q.insert(sid, q);
                    }
                }
            }
            self.spec_delta.draft_ns += started.elapsed().as_nanos() as u64;
        }
        if let Some(suffix) = &mut self.suffix {
            let started = Instant::now();
            let live = self.scheduler.sequences.keys().copied().collect();
            suffix.retain_live(&live);
            self.scheduler.external_drafts.clear();
            for &sid in &self.scheduler.running {
                let sequence = &self.scheduler.sequences[&sid];
                suffix.observe(sid, &sequence.tokens);
                if sequence.num_computed >= sequence.prefill_target
                    && sequence.can_speculate()
                    && !self.pending_logits.contains_key(&sid)
                {
                    let limit = if adaptive { sequence.adaptive.length() } else { usize::MAX };
                    let count = self.spec_tokens.min(limit).min(sequence.remaining_max().saturating_sub(1));
                    let tokens = suffix.propose(sid, &sequence.tokens, count);
                    if !tokens.is_empty() {
                        self.scheduler.external_drafts.insert(sid, tokens);
                    }
                }
            }
            self.spec_delta.draft_ns += started.elapsed().as_nanos() as u64;
        }
        let schedule_started = Instant::now();
        if let Some(mtp) = &mut self.mtp {
            let active = self.scheduler.sequences.keys().copied().collect();
            mtp.retain(&active)?;
            self.scheduler.external_drafts.clear();
            for &sid in &self.scheduler.running {
                let sequence = &self.scheduler.sequences[&sid];
                if sequence.num_computed >= sequence.prefill_target
                    && sequence.can_speculate()
                    && !self.pending_logits.contains_key(&sid)
                {
                    let tokens = mtp.propose(sid, &sequence.tokens, sequence.remaining_max());
                    if !tokens.is_empty() {
                        self.scheduler.external_drafts.insert(sid, tokens);
                    }
                }
            }
        }
        if let Some(spec_head) = &mut self.spec_head {
            let started = Instant::now();
            let active = self.scheduler.sequences.keys().copied().collect();
            spec_head.retain(&active)?;
            self.scheduler.external_drafts.clear();
            for &sid in &self.scheduler.running {
                let sequence = &self.scheduler.sequences[&sid];
                if sequence.num_computed >= sequence.prefill_target
                    && sequence.can_speculate()
                    && !self.pending_logits.contains_key(&sid)
                {
                    let limit = if adaptive { sequence.adaptive.length() } else { usize::MAX };
                    let count = self.spec_tokens.min(limit).min(sequence.remaining_max().saturating_sub(1));
                    let tokens = spec_head.propose(sid, count);
                    if !tokens.is_empty() {
                        self.scheduler.external_drafts.insert(sid, tokens);
                    }
                }
            }
            self.spec_delta.draft_ns += started.elapsed().as_nanos() as u64;
        }
        let scheduled = self.scheduler.schedule().map_err(|e| e.to_string())?;
        self.carry_out_swaps()?;
        if let Some(metrics) = &self.metrics {
            let (running, waiting, free, total, hits) = self.metrics_snapshot();
            metrics.snapshot(running, waiting, free, total, hits);
        }
        self.batch_stats.scheduler_ns += schedule_started.elapsed().as_nanos() as u64;
        if let Some(metrics) = &self.metrics {
            for wait in self.scheduler.wait_times.drain(..) {
                metrics.queue_wait(wait);
            }
            metrics.set_preemptions(self.scheduler.preemptions);
        } else {
            self.scheduler.wait_times.clear();
        }
        let Some(mut step) = scheduled else {
            return Ok(Vec::new());
        };
        // A sequence whose prompt logits are still pending takes its next token from them: it is a plain decode step, whatever draft
        // the lookup source attached.
        let pending: Vec<i64> = step.verify.iter().map(|v| v.0).filter(|sid| self.pending_logits.contains_key(sid)).collect();
        if !pending.is_empty() {
            step.verify.retain(|v| !pending.contains(&v.0));
            step.decode.extend(pending);
        }
        let cached_counts: std::collections::HashMap<_, _> = self
            .scheduler
            .sequences
            .iter()
            .map(|(&id, s)| (id, s.cached_prompt_tokens))
            .collect();
        let build_started = Instant::now();
        struct VerifyCandidate {
            sid: i64,
            draft: Vec<i64>,
            q: Vec<Option<crate::speculative::Dist>>,
            base_row: usize,
            draft_rows: Vec<usize>,
            token_id: i64,
            position: usize,
            table: std::sync::Arc<Vec<i64>>,
            state_snapshot: Option<crate::rust_decoder::SequenceSnapshot>,
            first_block: Option<i64>,
        }
        let mut verify_candidates = Vec::<VerifyCandidate>::new();
        let mut rows = Vec::<BatchRow>::new();
        let mut prefill_commits = Vec::<(i64, usize, Option<usize>)>::new();
        let mut decode_row = std::collections::HashMap::<i64, usize>::new();
        let mut logit_rows = Vec::<usize>::new();
        let mut prefill_rows = 0usize;
        let mut decode_rows = 0usize;

        for (sid, start, count) in &step.prefill {
            let sequence = self
                .scheduler
                .sequences
                .get(sid)
                .ok_or_else(|| format!("scheduler returned unknown sequence {sid}"))?;
            let end = start
                .checked_add(*count)
                .ok_or("prefill position overflow")?;
            if *count == 0 || end > sequence.prefill_target || end > sequence.tokens.len() {
                return Err(format!("invalid prefill slice for sequence {sid}"));
            }
            let table = std::sync::Arc::new(self.table_row(*sid)?);
            for position in *start..end {
                let completes_prompt = end == sequence.prefill_target && position + 1 == end;
                let row = rows.len();
                rows.push(BatchRow {
                    sequence_id: *sid,
                    token_id: sequence.tokens[position],
                    length: (position + 1) as i64,
                    table: table.clone(),
                    embedding: self
                        .image_embeddings
                        .get(sid)
                        .and_then(|values| values.get(&position))
                        .cloned(),
                });
                prefill_rows += 1;
                if completes_prompt {
                    prefill_commits.push((*sid, *count, Some(row)));
                    logit_rows.push(row);
                }
            }
            if !prefill_commits.iter().any(|item| item.0 == *sid) {
                prefill_commits.push((*sid, *count, None));
            }
        }

        for sid in &step.decode {
            if self.pending_logits.contains_key(sid) {
                continue;
            }
            let sequence = self
                .scheduler
                .sequences
                .get(sid)
                .ok_or_else(|| format!("scheduler returned unknown sequence {sid}"))?;
            let position = sequence.num_computed;
            let token_id = *sequence.tokens.get(position).ok_or_else(|| {
                format!("decode position {position} is missing for sequence {sid}")
            })?;
            let row = rows.len();
            rows.push(BatchRow {
                sequence_id: *sid,
                token_id,
                length: (position + 1) as i64,
                table: std::sync::Arc::new(self.table_row(*sid)?),
                embedding: None,
            });
            decode_rows += 1;
            decode_row.insert(*sid, row);
            logit_rows.push(row);
        }

        for (sid, draft) in &step.verify {
            let sequence = self
                .scheduler
                .sequences
                .get(sid)
                .ok_or_else(|| format!("scheduler returned unknown sequence {sid}"))?;
            let position = sequence.num_computed;
            let token_id = *sequence.tokens.get(position).ok_or_else(|| {
                format!("verify position {position} is missing for sequence {sid}")
            })?;
            let table = std::sync::Arc::new(self.table_row(*sid)?);
            let base_row = rows.len();
            rows.push(BatchRow {
                sequence_id: *sid,
                token_id,
                length: (position + 1) as i64,
                table: table.clone(),
                embedding: None,
            });
            logit_rows.push(base_row);
            decode_rows += 1;

            let mut d_rows = Vec::with_capacity(draft.len());
            for (idx, &d_tok) in draft.iter().enumerate() {
                let d_row = rows.len();
                rows.push(BatchRow {
                    sequence_id: *sid,
                    token_id: d_tok,
                    length: (position + 2 + idx) as i64,
                    table: table.clone(),
                    embedding: None,
                });
                logit_rows.push(d_row);
                d_rows.push(d_row);
                decode_rows += 1;
            }
            let first_block = sequence.blocks.first().copied().map(|b| b as i64);
            let state_snapshot = first_block.and_then(|first| {
                self.decoder.as_ref().and_then(|d| d.snapshot_state(first).ok().flatten())
            });
            let mut q = self.draft_q.remove(sid).unwrap_or_default();
            q.resize(draft.len(), None);
            verify_candidates.push(VerifyCandidate {
                sid: *sid,
                draft: draft.clone(),
                q,
                base_row,
                draft_rows: d_rows,
                token_id,
                position,
                table: table.clone(),
                state_snapshot,
                first_block,
            });
            self.batch_stats.draft_proposed += draft.len();
            if self.mtp.is_some() {
                if let Some(metrics) = &self.metrics {
                    metrics.mtp_proposed(draft.len());
                }
            }
        }

        if !rows.is_empty() {
            self.batch_stats.forward_calls += 1;
            self.batch_stats.rows_total += rows.len();
            self.batch_stats.calls_with_multiple_rows += usize::from(rows.len() > 1);
            self.batch_stats.max_rows = self.batch_stats.max_rows.max(rows.len());
        }
        self.batch_stats.prefill_rows += prefill_rows;
        self.batch_stats.decode_rows += decode_rows;
        self.batch_stats.batch_build_ns += build_started.elapsed().as_nanos() as u64;
        let forward_started = Instant::now();
        let mut hidden_capture = Vec::new();
        // A decode-only step in which every sequence just takes the largest logit needs no logits on the host: the device returns
        // the ids (vLLM's greedy_sample is an argmax on the GPU as well).
        let use_greedy_ids = !rows.is_empty()
            && !logit_rows.is_empty()
            && verify_candidates.is_empty()
            && self.mtp.is_none()
            && logit_rows.len() == decode_row.len()
            && prefill_commits.iter().all(|(_, _, final_row)| final_row.is_none())
            && step.decode.iter().all(|sid| {
                !self.pending_logits.contains_key(sid)
                    && !self.constraints.contains_key(sid)
                    && self.scheduler.sequences.get(sid).is_some_and(|s| {
                        s.sampling.argmax_only() && s.sampling.logprobs.is_none() && s.output_len >= s.sampling.min_tokens
                    })
            });
        // Sampled decode rows can be sampled on the device as well (FLLM_DEVICE_SAMPLING=1, CUDA): the same eligibility as above, but any
        // sampler setting except mirostat / DRY is allowed, because the penalties, bias, controls and log-probabilities run on the device.
        let device_job = if !use_greedy_ids
            && crate::rust_decoder::device_sampling_enabled()
            && !rows.is_empty()
            && !logit_rows.is_empty()
            && verify_candidates.is_empty()
            && self.mtp.is_none()
            && logit_rows.len() == decode_row.len()
            && prefill_commits.iter().all(|(_, _, final_row)| final_row.is_none())
            && step.decode.iter().all(|sid| !self.pending_logits.contains_key(sid) && !self.constraints.contains_key(sid))
        {
            self.device_sample_job(&step.decode)
        } else {
            None
        };
        if use_greedy_ids {
            crate::rust_decoder::RustDecoder::request_greedy_ids();
        }
        let device_job_rows = device_job.as_ref().map(|j| j.rows.len());
        if let Some(job) = device_job {
            crate::rust_decoder::RustDecoder::request_device_sample(job);
        }
        let batch_logits = if rows.is_empty() {
            Vec::new()
        } else {
            let mut ids = Vec::with_capacity(rows.len());
            let mut lengths = Vec::with_capacity(rows.len());
            let mut table = Vec::with_capacity(rows.len() * self.model.table_stride());
            for row in &rows {
                ids.push(row.token_id);
                lengths.push(row.length);
                table.extend_from_slice(&row.table);
            }
            let overrides = rows
                .iter()
                .enumerate()
                .filter_map(|(r, row)| row.embedding.as_ref().map(|x| (r, x.clone())))
                .collect::<Vec<_>>();
            // sequences with media use three-axis positions; the other rows of the batch get the ordinary ones on every axis
            let positions3 = (!self.mrope_positions.is_empty()
                && rows.iter().any(|row| self.mrope_positions.contains_key(&row.sequence_id)))
            .then(|| {
                rows.iter()
                    .map(|row| {
                        let index = (row.length - 1) as usize;
                        self.mrope_positions
                            .get(&row.sequence_id)
                            .map_or([index as u32; 3], |m| m.at(index))
                    })
                    .collect::<Vec<_>>()
            });
            // DeepStack features of the rows of this batch, grouped by decoder layer
            let mut levels: Vec<Vec<(usize, Vec<f32>)>> = Vec::new();
            if !self.deepstack.is_empty() {
                for (r, row) in rows.iter().enumerate() {
                    if let Some(per_level) = self
                        .deepstack
                        .get(&row.sequence_id)
                        .and_then(|m| m.get(&(row.length as usize - 1)))
                    {
                        for (level, values) in per_level.iter().enumerate().filter(|(_, v)| !v.is_empty()) {
                            if levels.len() <= level {
                                levels.resize(level + 1, Vec::new());
                            }
                            levels[level].push((r, values.clone()));
                        }
                    }
                }
            }
            let visible_lengths = (!self.visible_lengths.is_empty()
                && rows.iter().any(|row| self.visible_lengths.contains_key(&row.sequence_id)))
            .then(|| {
                rows.iter()
                    .map(|row| {
                        let index = (row.length - 1) as usize;
                        self.visible_lengths
                            .get(&row.sequence_id)
                            .and_then(|v| v.get(index).copied())
                            .unwrap_or(row.length as usize)
                    })
                    .collect::<Vec<_>>()
            });
            let decoder = Self::decoder(&self.decoder)?;
            decoder.set_row_positions3(positions3);
            decoder.set_row_deepstack((!levels.is_empty()).then_some(levels));
            decoder.set_row_visible_lengths(visible_lengths);
            let forwarded = decoder.forward_with_hidden(
                &ids,
                &lengths,
                &table,
                &mut self.key,
                &mut self.value,
                &logit_rows,
                false,
                &overrides,
                if self.mtp.is_some() || self.spec_head.is_some() {
                    Some(&mut hidden_capture)
                } else {
                    None
                },
            );
            decoder.set_row_positions3(None);
            decoder.set_row_deepstack(None);
            decoder.set_row_visible_lengths(None);
            forwarded?
        };
        let greedy_ids = if use_greedy_ids {
            crate::rust_decoder::RustDecoder::take_greedy_ids()
        } else {
            None
        };
        let device_sampled = if device_job_rows.is_some() {
            crate::rust_decoder::RustDecoder::take_device_sample()
        } else {
            None
        };
        self.batch_stats.model_forward_ns += forward_started.elapsed().as_nanos() as u64;
        let postprocess_started = Instant::now();

        for (sid, count, final_row) in prefill_commits {
            self.scheduler
                .commit_chunk(sid, count)
                .map_err(|e| e.to_string())?;
            if let Some(row) = final_row {
                let compact_row = logit_rows
                    .binary_search(&row)
                    .map_err(|_| "missing prompt LM-head result")?;
                self.pending_logits
                    .insert(sid, self.logit_row(&batch_logits, compact_row));
                self.fork_prompt_choices(sid)?;
            }
        }

        let vocab = self.model.vocab_size();
        let mut events = Vec::new();
        for sid in step.decode {
            let (sampling, tokens_history) = self
                .scheduler
                .sequences
                .get(&sid)
                .map(|s| (s.sampling.clone(), s.tokens.clone()))
                .unwrap_or_else(|| (crate::sampling::SamplingParams::default(), Vec::new()));
            let greedy_token = greedy_ids.as_ref().and_then(|ids| {
                let row = decode_row.get(&sid)?;
                let compact_row = logit_rows.binary_search(row).ok()?;
                ids.get(compact_row).map(|&id| i64::from(id))
            });
            let device_pick = device_sampled.as_ref().and_then(|out| {
                let row = decode_row.get(&sid)?;
                let compact = logit_rows.binary_search(row).ok()?;
                Some((out, compact))
            });
            let mut processed_dist: Option<Vec<(usize, f32)>> = None;
            let (token, original_logits) = if let Some((out, compact)) = device_pick {
                let id = out.ids[compact];
                if id < 0 {
                    return Err("generation controls leave no valid next token".into());
                }
                if let Some(sequence) = self.scheduler.sequences.get_mut(&sid) {
                    // the device draws number `draws` of the request's stream for every sampled (non-greedy) row
                    if sampling.temperature > 1e-5 {
                        sequence.sampling_state.draws += 1;
                    }
                }
                (i64::from(id), None)
            } else if let Some(token) = greedy_token {
                (token, None)
            } else {
            // borrowed from the batch result unless a stop-token, grammar or logprob step has to change or keep a copy
            let mut logits: std::borrow::Cow<'_, [f32]> = if let Some(logits) = self.pending_logits.remove(&sid) {
                if logits.len() != vocab {
                    return Err("LM head returned an invalid vocabulary row".into());
                }
                std::borrow::Cow::Owned(logits)
            } else {
                let row = *decode_row
                    .get(&sid)
                    .ok_or_else(|| format!("missing decode result for {sid}"))?;
                let compact_row = logit_rows
                    .binary_search(&row)
                    .map_err(|_| format!("missing decode LM-head result for {sid}"))?;
                let offset = compact_row * vocab;
                if offset + vocab > batch_logits.len() {
                    return Err("LM head returned an invalid vocabulary row".into());
                }
                std::borrow::Cow::Borrowed(&batch_logits[offset..offset + vocab])
            };
            let original_logits = sampling.logprobs.filter(|_| !sampling.logprobs_processed).map(|_| logits.to_vec());
            let generated = self
                .scheduler
                .sequences
                .get(&sid)
                .map_or(0, |s| s.output_len);
            if generated < sampling.min_tokens || !sampling.allowed_token_ids.is_empty() || !sampling.bad_word_ids.is_empty() {
                crate::sampling::apply_generation_controls(logits.to_mut(), &sampling, &self.eos_ids, generated, &tokens_history);
            }
            if !logits.iter().any(|v| v.is_finite()) {
                return Err("generation controls leave no valid next token".into());
            }
            if let Some(matcher) = self.constraints.get_mut(&sid) {
                let mask = matcher
                    .compute_mask_or_eos()
                    .map_err(|e| format!("grammar mask failed: {e}"))?;
                for (id, logit) in logits.to_mut().iter_mut().enumerate() {
                    if !mask.is_allowed(id as u32) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
                if !logits.iter().any(|v| v.is_finite()) {
                    return Err("grammar has no valid next token".into());
                }
            }
            let sequence = self
                .scheduler
                .sequences
                .get_mut(&sid)
                .ok_or("missing sampling sequence")?;
            if sampling.logprobs.is_some() && sampling.logprobs_processed {
                processed_dist = Some(crate::sampling::processed_distribution(
                    &logits,
                    &tokens_history,
                    &sampling,
                    sequence.sampling_state.mirostat_mu,
                ));
            }
            let token = crate::sampling::sample_with_state(
                &logits,
                &tokens_history,
                &sampling,
                &mut sequence.sampling_state,
            );
            (token, original_logits)
            };
            if let Some(matcher) = self.constraints.get_mut(&sid) {
                // An accepting grammar ends on EOS. The parser has no byte
                // extension for that control token and must not consume it.
                if !self.eos_ids.contains(&token) {
                    matcher
                        .consume_token(token as u32)
                        .map_err(|e| format!("grammar token rejected: {e}"))?;
                }
            }
            let sequence = self
                .scheduler
                .sequences
                .get_mut(&sid)
                .ok_or_else(|| format!("unknown sequence {sid}"))?;
            let request_id = sequence.req_id.clone();
            let choice_index = sequence.choice_index;
            sequence.append_token(token);
            let hit_eos = sampling.stops_on_token(token, &self.eos_ids);
            let cached_prompt_tokens = sequence.cached_prompt_tokens;
            let logprobs = if let (Some(out), Some(row)) = (device_sampled.as_ref(), decode_row.get(&sid)) {
                // log-probabilities computed on the device, from the raw logits
                match (sampling.logprobs, logit_rows.binary_search(row)) {
                    (Some(_), Ok(compact)) => {
                        let stride = out.top_ids.len() / out.ids.len().max(1);
                        Some(crate::sampling::TokenLogprobs {
                            logprob: f64::from(out.logprobs[compact]),
                            top: (0..stride)
                                .map(|j| (out.top_ids[compact * stride + j], out.top_logprobs[compact * stride + j]))
                                .filter(|(id, _)| *id >= 0)
                                .map(|(id, lp)| (id as u32, f64::from(lp)))
                                .collect(),
                        })
                    }
                    _ => None,
                }
            } else if let Some(dist) = &processed_dist {
                Some(crate::sampling::processed_logprobs(dist, token as usize, sampling.logprobs.unwrap_or(0)))
            } else {
                original_logits
                    .as_ref()
                    .map(|logits| {
                        crate::sampling::raw_logprobs(
                            logits,
                            token as usize,
                            sampling.logprobs.unwrap_or(0),
                        )
                    })
                    .transpose()?
            };
            let hit_limit = sequence.remaining_max() == 0;
            self.scheduler
                .commit_decode(sid)
                .map_err(|e| e.to_string())?;
            let finished = hit_eos || hit_limit;
            let finish_reason = if hit_eos {
                Some("stop")
            } else if hit_limit {
                Some("length")
            } else {
                None
            };
            if finished {
                self.finish_choice(&request_id, sid)?;
            }
            events.push(TokenEvent {
                request_id: request_id.clone(),
                sequence_id: sid,
                token_id: token,
                finished,
                finish_reason,
                logprobs,
                cached_prompt_tokens,
                choice_index,
                request_finished: finished && !self.scheduler.requests.contains_key(&request_id),
            });
            if !finished {
                self.jump_forward_after(sid, token, &mut events)?;
            }
        }

        let had_verify = !verify_candidates.is_empty();
        for cand in verify_candidates {
            let sid = cand.sid;
            let (sampling, mut sstate, history, generated, request_id, choice_index) = {
                let s = self
                    .scheduler
                    .sequences
                    .get(&sid)
                    .ok_or_else(|| format!("unknown sequence {sid}"))?;
                (s.sampling.clone(), s.sampling_state.clone(), s.tokens.clone(), s.output_len, s.req_id.clone(), s.choice_index)
            };
            let logits_at = |i: usize| -> Result<&[f32], String> {
                let row = if i == 0 { cand.base_row } else { cand.draft_rows[i - 1] };
                let compact = logit_rows
                    .binary_search(&row)
                    .map_err(|_| format!("missing verify LM-head result for {sid}"))?;
                batch_logits
                    .get(compact * vocab..(compact + 1) * vocab)
                    .ok_or_else(|| "LM head returned an invalid vocabulary row".to_owned())
            };
            // The target distribution at draft position i is the one plain decoding would sample from after the accepted draft
            // tokens: same hard controls, grammar mask, penalties and filters. The grammar advances with every accepted token.
            let mut failure: Option<String> = None;
            // the processed distributions per position, kept when the request wants processed log-probabilities
            let capture = sampling.logprobs.is_some() && sampling.logprobs_processed;
            let mut captured: Vec<Vec<(usize, f32)>> = Vec::new();
            let mut hist = history;
            let mut matcher = self.constraints.get_mut(&sid);
            let eos_ids = &self.eos_ids;
            let outcome = crate::speculative::verify_chain(
                &cand.draft,
                &cand.q,
                |i| {
                    if i > 0 {
                        let token = cand.draft[i - 1];
                        hist.push(token);
                        if let Some(m) = matcher.as_mut() {
                            if !eos_ids.contains(&token) {
                                if let Err(e) = m.consume_token(token as u32) {
                                    failure.get_or_insert(format!("grammar token rejected: {e}"));
                                }
                            }
                        }
                    }
                    let mut logits = match logits_at(i) {
                        Ok(row) => row.to_vec(),
                        Err(e) => {
                            failure.get_or_insert(e);
                            return vec![(0, 1.0)];
                        }
                    };
                    crate::sampling::apply_generation_controls(&mut logits, &sampling, eos_ids, generated + i, &hist);
                    if let Some(m) = matcher.as_mut() {
                        match m.compute_mask_or_eos() {
                            Ok(mask) => {
                                for (id, logit) in logits.iter_mut().enumerate() {
                                    if !mask.is_allowed(id as u32) {
                                        *logit = f32::NEG_INFINITY;
                                    }
                                }
                            }
                            Err(e) => {
                                failure.get_or_insert(format!("grammar mask failed: {e}"));
                            }
                        }
                    }
                    if !logits.iter().any(|v| v.is_finite()) {
                        failure.get_or_insert("generation controls leave no valid next token".to_owned());
                        return vec![(0, 1.0)];
                    }
                    let dist = crate::sampling::processed_distribution(&logits, &hist, &sampling, 0.0);
                    if capture {
                        captured.push(dist.clone());
                    }
                    dist
                },
                || sstate.next_uniform(),
            );
            if let Some(e) = failure {
                return Err(e);
            }
            if let Some(m) = matcher.as_mut() {
                if !eos_ids.contains(&outcome.next) {
                    m.consume_token(outcome.next as u32)
                        .map_err(|e| format!("grammar token rejected: {e}"))?;
                }
            }
            let accepted = outcome.accepted;
            self.batch_stats.draft_accepted += accepted;
            if self.mtp.is_some() {
                if let Some(metrics) = &self.metrics {
                    metrics.mtp_accepted(accepted);
                }
            }
            self.spec_delta.record(cand.draft.len(), accepted);
            let sequence = self
                .scheduler
                .sequences
                .get_mut(&sid)
                .ok_or_else(|| format!("unknown sequence {sid}"))?;
            sequence.sampling_state = sstate;
            sequence.adaptive.update(cand.draft.len(), accepted);
            let mut emitted: Vec<i64> = cand.draft[..accepted].to_vec();
            emitted.push(outcome.next);
            let mut finished = false;
            for (j, &token) in emitted.iter().enumerate() {
                sequence.append_token(token);
                let hit_eos = sampling.stops_on_token(token, eos_ids);
                let hit_limit = sequence.remaining_max() == 0;
                // logprobs describe the target model at the emitted token's position, never the draft
                let logprobs = match sampling.logprobs {
                    Some(top) if sampling.logprobs_processed => {
                        captured.get(j).map(|dist| crate::sampling::processed_logprobs(dist, token as usize, top))
                    }
                    Some(top) => Some(crate::sampling::raw_logprobs(logits_at(j)?, token as usize, top)?),
                    None => None,
                };
                finished = hit_eos || hit_limit;
                events.push(TokenEvent {
                    request_id: request_id.clone(),
                    sequence_id: sid,
                    token_id: token,
                    finished,
                    finish_reason: if hit_eos {
                        Some("stop")
                    } else if hit_limit {
                        Some("length")
                    } else {
                        None
                    },
                    logprobs,
                    cached_prompt_tokens: 0,
                    choice_index,
                    request_finished: false,
                });
                if finished {
                    break;
                }
            }
            if accepted < cand.draft.len() {
                if let (Some(snap), Some(first)) = (&cand.state_snapshot, cand.first_block) {
                    if let Some(decoder) = &self.decoder {
                        decoder.restore_state(first, snap)?;
                        let mut adv_ids = Vec::with_capacity(1 + accepted);
                        let mut adv_lens = Vec::with_capacity(1 + accepted);
                        adv_ids.push(cand.token_id);
                        adv_lens.push((cand.position + 1) as i64);
                        for (idx, &d_tok) in cand.draft[..accepted].iter().enumerate() {
                            adv_ids.push(d_tok);
                            adv_lens.push((cand.position + 2 + idx) as i64);
                        }
                        let adv_table = cand.table.repeat(adv_ids.len());
                        let _ = decoder.forward_with_hidden(
                            &adv_ids,
                            &adv_lens,
                            &adv_table,
                            &mut self.key,
                            &mut self.value,
                            &[],
                            false,
                            &[],
                            None,
                        );
                    }
                }
            }
            self.scheduler
                .commit_verify(sid)
                .map_err(|e| e.to_string())?;
            if finished {
                self.finish_choice(&request_id, sid)?;
                let done = !self.scheduler.requests.contains_key(&request_id);
                if let Some(last) = events.last_mut() {
                    last.request_finished = done;
                }
            }
        }
        if had_verify {
            self.spec_delta.verify_ns += forward_started.elapsed().as_nanos() as u64;
        }
        if let Some(spec_head) = &mut self.spec_head {
            let width = spec_head.hidden_size;
            if hidden_capture.len() == rows.len() * width {
                for (index, row) in rows.iter().enumerate().rev() {
                    if let Some(sequence) = self.scheduler.sequences.get(&row.sequence_id) {
                        let position = row.length as usize - 1;
                        if position < sequence.num_computed
                            && sequence.tokens.get(position) == Some(&row.token_id)
                        {
                            spec_head.observe(
                                row.sequence_id,
                                &hidden_capture[index * width..(index + 1) * width],
                            );
                            break;
                        }
                    }
                }
            }
        }
        if let Some(mtp) = &mut self.mtp {
            let width = self.model.hidden_size();
            if hidden_capture.len() != rows.len() * width {
                return Err("invalid target hidden capture for MTP".into());
            }
            let mut captures = std::collections::HashMap::<i64, Vec<(usize, Vec<f32>)>>::new();
            for (index, row) in rows.iter().enumerate() {
                if let Some(sequence) = self.scheduler.sequences.get(&row.sequence_id) {
                    let position = row.length as usize - 1;
                    // Keep only target rows whose tokens survived verification.
                    if position < sequence.num_computed
                        && sequence.tokens.get(position) == Some(&row.token_id)
                    {
                        captures.entry(row.sequence_id).or_default().push((
                            position,
                            hidden_capture[index * width..(index + 1) * width].to_vec(),
                        ));
                    }
                }
            }
            for (&sid, sequence) in &self.scheduler.sequences {
                mtp.observe(
                    sid,
                    &sequence.tokens,
                    sequence.num_computed,
                    captures.remove(&sid).unwrap_or_default(),
                )?;
            }
        }
        if self.spec_delta != crate::speculative::SpecStats::default() {
            self.spec_stats.add(&self.spec_delta);
            if let Some(metrics) = &self.metrics {
                metrics.speculative(&self.spec_delta);
            }
            self.spec_delta = Default::default();
        }
        self.batch_stats.postprocess_ns += postprocess_started.elapsed().as_nanos() as u64;
        self.image_embeddings
            .retain(|sid, _| self.scheduler.sequences.contains_key(sid));
        for event in &mut events {
            event.cached_prompt_tokens =
                cached_counts.get(&event.sequence_id).copied().unwrap_or(0);
        }
        Ok(events)
    }

    pub fn run_until_request_done(&mut self, request_id: &str) -> Result<Vec<i64>, String> {
        let mut tokens = Vec::new();
        while self.scheduler.requests.contains_key(request_id) {
            let events = self.step()?;
            if events.is_empty() && self.scheduler.requests.contains_key(request_id) {
                return Err("scheduler could not make progress with the available KV pool".into());
            }
            for event in events.into_iter().filter(|e| e.request_id == request_id) {
                tokens.push(event.token_id);
            }
        }
        Ok(tokens)
    }

    /// The device sampling job of a decode-only step, or `None` when a sequence needs something the device does not do (mirostat, DRY, the
    /// xorshift generator). Rows follow the order of `decode`, which is the order of the logit rows.
    pub(crate) fn device_sample_job(&self, decode: &[i64]) -> Option<crate::rust_decoder::DeviceSampleJob> {
        let vocab = self.model.vocab_size();
        let words = vocab.div_ceil(32);
        let mut job = crate::rust_decoder::DeviceSampleJob { mask_words: words, ..Default::default() };
        for sid in decode {
            let s = self.scheduler.sequences.get(sid)?;
            let p = &s.sampling;
            if !p.device_samplable() || s.sampling_state.kind != crate::sampling::RngKind::Philox {
                return None;
            }
            let mut row = p.device_row(s.sampling_state.rng, s.sampling_state.draws);
            row.pen_off = job.pen.len() as i32;
            if (p.repetition_penalty - 1.0).abs() >= 1e-5 || p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 {
                job.pen.extend(crate::sampling::penalty_pairs(&s.tokens, p.repeat_last_n, vocab));
            }
            row.pen_len = job.pen.len() as i32 - row.pen_off;
            // bias, the controls that forbid tokens (as -inf) and single-token bad words, merged per token
            let mut bias = std::collections::BTreeMap::<i32, f32>::new();
            for &(id, value) in &p.logit_bias {
                if id < vocab {
                    *bias.entry(id as i32).or_default() += value;
                }
            }
            let mut forbid = |id: i64| {
                if id >= 0 && (id as usize) < vocab {
                    bias.insert(id as i32, f32::NEG_INFINITY);
                }
            };
            if s.output_len < p.min_tokens {
                self.eos_ids.iter().chain(&p.stop_token_ids).for_each(|&id| forbid(id));
            }
            for word in &p.bad_word_ids {
                if let Some((&last, prefix)) = word.split_last() {
                    if s.tokens.len() >= prefix.len() && s.tokens[s.tokens.len() - prefix.len()..] == *prefix {
                        forbid(last);
                    }
                }
            }
            row.bias_off = job.bias.len() as i32;
            job.bias.extend(bias);
            row.bias_len = job.bias.len() as i32 - row.bias_off;
            if !p.allowed_token_ids.is_empty() {
                row.mask_row = (job.mask.len() / words) as i32;
                let mut bits = vec![0u32; words];
                for &id in &p.allowed_token_ids {
                    if id < vocab {
                        bits[id >> 5] |= 1 << (id & 31);
                    }
                }
                job.mask.extend(bits);
            }
            if let Some(top) = p.logprobs {
                row.logprobs_mode = if p.logprobs_processed { 2 } else { 1 };
                row.top_n = top.min(20) as i32;
                job.top_stride = job.top_stride.max(top.min(20));
            }
            job.rows.push(row);
        }
        Some(job)
    }

    fn table_row(&self, sid: i64) -> Result<Vec<i64>, String> {
        let blocks = &self
            .scheduler
            .bm
            .get_table(sid)
            .map_err(|e| e.to_string())?
            .blocks;
        if blocks.len() > self.model.table_stride() {
            return Err(format!("sequence {sid} has too many KV blocks"));
        }
        let mut row = vec![-1; self.model.table_stride()];
        for (i, bid) in blocks.iter().enumerate() {
            row[i] = i64::try_from(*bid).map_err(|_| "KV block id exceeds C ABI range")?;
        }
        Ok(row)
    }

    fn logit_row(&self, logits: &[f32], row: usize) -> Vec<f32> {
        let start = row * self.model.vocab_size();
        logits[start..start + self.model.vocab_size()].to_vec()
    }
}

/// Copy slot-major data and, for q8, the separate scale plane of every layer.
fn copy_cache_block(
    cache: &mut [u16],
    src: usize,
    dst: usize,
    blocks: usize,
    layers: usize,
    block_values: usize,
    scales: usize,
) -> Result<(), String> {
    if src >= blocks
        || dst >= blocks
        || layers == 0
        || block_values == 0
        || scales > 0 && !block_values.is_multiple_of(2)
    {
        return Err("invalid shared KV block geometry".into());
    }
    let data = if scales > 0 {
        block_values / 2
    } else {
        block_values
    };
    let words = data
        .checked_add(scales)
        .and_then(|n| n.checked_mul(blocks))
        .ok_or("shared KV block size overflow")?;
    let total = words
        .checked_mul(layers)
        .ok_or("shared KV buffer size overflow")?;
    if cache.len() != total {
        return Err("shared KV buffer does not match layer geometry".into());
    }
    for layer in 0..layers {
        let offset = layer * words;
        cache.copy_within(
            offset + src * data..offset + (src + 1) * data,
            offset + dst * data,
        );
        if scales > 0 {
            let offset = offset + blocks * data;
            cache.copy_within(
                offset + src * scales..offset + (src + 1) * scales,
                offset + dst * scales,
            );
        }
    }
    Ok(())
}

/// Copy block `src` to `dst` in every region of every layer (`regions`: offset and words per block of each region).
fn copy_cache_regions(
    cache: &mut [u16],
    src: usize,
    dst: usize,
    blocks: usize,
    layers: usize,
    layer_words: usize,
    regions: &[(usize, usize)],
) -> Result<(), String> {
    if src >= blocks || dst >= blocks || layers == 0 || cache.len() != layer_words * layers {
        return Err("invalid shared KV block geometry".into());
    }
    for layer in 0..layers {
        for &(offset, words) in regions {
            let base = layer * layer_words + offset;
            if base + blocks * words > (layer + 1) * layer_words {
                return Err("shared KV region exceeds its layer".into());
            }
            cache.copy_within(base + src * words..base + (src + 1) * words, base + dst * words);
        }
    }
    Ok(())
}

#[cfg(test)]
fn argmax(logits: &[f32]) -> usize {
    crate::sampling::argmax(logits)
}

/// One block on its way to the disk tier.
struct SpillJob {
    hash: u64,
    parent: u64,
    tokens: Vec<i64>,
    bytes: Vec<u8>,
}

/// Starts the writer thread and returns the sending half. The queue is small on purpose: a block that cannot be queued is
/// dropped, because a cache entry is worth less than a token.
fn spawn_spill_writer(tier: std::sync::Arc<crate::mem::kv_tier::KvTier>) -> Option<std::sync::mpsc::SyncSender<SpillJob>> {
    if !tier.enabled() {
        return None;
    }
    let (tx, rx) = std::sync::mpsc::sync_channel::<SpillJob>(256);
    std::thread::Builder::new()
        .name("fllm-kv-disk".into())
        .spawn(move || {
            for job in rx {
                tier.spill(job.hash, job.parent, &job.tokens, &job.bytes);
            }
        })
        .ok()?;
    Some(tx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_f16_tail_copy_preserves_other_blocks_and_layers() {
        let mut cache = (0..24).collect::<Vec<u16>>();
        let before = cache.clone();
        let mut expected = before.clone();
        expected[8..12].copy_from_slice(&before[0..4]);
        expected[20..24].copy_from_slice(&before[12..16]);
        copy_cache_block(&mut cache, 0, 2, 3, 2, 4, 0).unwrap();
        assert_eq!(cache, expected);
    }
    #[test]
    fn shared_q8_tail_copy_includes_the_scale_plane() {
        let mut cache = (0..36).collect::<Vec<u16>>();
        let before = cache.clone();
        let mut expected = before.clone();
        expected[8..12].copy_from_slice(&before[4..8]);
        expected[16..18].copy_from_slice(&before[14..16]);
        expected[26..30].copy_from_slice(&before[22..26]);
        expected[34..36].copy_from_slice(&before[32..34]);
        copy_cache_block(&mut cache, 1, 2, 3, 2, 8, 2).unwrap();
        assert_eq!(cache, expected);
        assert!(copy_cache_block(&mut cache, 1, 3, 3, 2, 8, 2).is_err());
        assert!(copy_cache_block(&mut cache, 1, 2, 3, 2, 7, 2).is_err());
        assert!(copy_cache_block(&mut cache, 1, 2, 3, 3, 8, 2).is_err());
    }
    #[test]
    fn shared_tail_copy_works_for_every_kv_format() {
        use crate::kv_format::{KvFormat, KvLayout};
        let (blocks, block_size, heads, dim, layers) = (3usize, 16usize, 2usize, 64usize, 2usize);
        for &format in &KvFormat::ALL {
            let layout = KvLayout::new(format, blocks * block_size, heads, dim);
            let regions = layout.block_regions(block_size).expect("whole-word blocks");
            let mut cache = vec![0u16; layout.words * layers];
            for layer in 0..layers {
                for slot in 0..block_size {
                    let row: Vec<f32> = (0..heads * dim)
                        .map(|c| ((layer * 31 + slot * 7 + c * 3) % 17) as f32 - 8.0)
                        .collect();
                    layout.encode_slot(&mut cache[layer * layout.words..(layer + 1) * layout.words], slot, &row);
                }
            }
            copy_cache_regions(&mut cache, 0, 2, blocks, layers, layout.words, &regions).unwrap();
            for layer in 0..layers {
                let words = &cache[layer * layout.words..(layer + 1) * layout.words];
                for slot in 0..block_size {
                    let (mut a, mut b) = (vec![0.0f32; heads * dim], vec![0.0f32; heads * dim]);
                    layout.decode_slot(words, slot, &mut a);
                    layout.decode_slot(words, 2 * block_size + slot, &mut b);
                    assert_eq!(a, b, "{} layer {layer} slot {slot}", format.name());
                }
            }
        }
    }
    #[test]
    fn argmax_keeps_the_first_index_when_maxima_tie() {
        assert_eq!(argmax(&[1.0, 5.0, 5.0, 3.0]), 1);
        assert_eq!(argmax(&[5.0, 4.0, 5.0]), 0);
    }

    #[test]
    fn scheduler_config_accepts_speculative_lookahead() {
        let cfg = SchedulerConfig {
            block_size: 16,
            max_num_seqs: 4,
            max_num_batched_tokens: 128,
            prefix_cache: true,
            preemption: "recompute".into(),
            speculative_lookup: 3,
            speculative_window: 32,
            speculative_adaptive: false,
            ..SchedulerConfig::default()
        };
        assert_eq!(cfg.speculative_lookup, 3);
        assert_eq!(cfg.speculative_window, 32);
    }
}
