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
    pub scheduler_ns: u64,
    pub batch_build_ns: u64,
    pub model_forward_ns: u64,
    pub postprocess_ns: u64,
}

struct BatchRow {
    sequence_id: i64,
    token_id: i64,
    length: i64,
    table: Vec<i64>,
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
    draft: Option<crate::draft::DraftEngine>,
    mtp: Option<crate::draft::MtpEngine>,
    decoder: Option<crate::rust_decoder::RustDecoder>,
    image_embeddings: std::collections::HashMap<i64, std::collections::HashMap<usize, Vec<f32>>>,
    pub vision: Option<std::sync::Arc<crate::vision::VisionEncoder>>,
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
        let speculative_lookup = if stateful { 0 } else { speculative_lookup };
        let pc = PrefixCache::new(pool_blocks as i64, true, true).map_err(|e| e.to_string())?;
        let config = SchedulerConfig {
            block_size,
            max_num_seqs: max_sequences,
            max_num_batched_tokens: max_batched_tokens,
            prefix_cache: !stateful,
            preemption: "recompute".into(),
            speculative_lookup,
            speculative_window: if speculative_lookup > 0 && speculative_window == 0 {
                32
            } else {
                speculative_window
            },
        };
        let scheduler = Scheduler::new(config, bm, Some(pc), model.max_context());
        let key_len = model.key_buffer_len()?;
        let value_len = model.value_buffer_len()?;
        Ok(Self {
            model,
            scheduler,
            key: vec![0; key_len],
            value: vec![0; value_len],
            pending_logits: std::collections::HashMap::new(),
            pending_choices: Default::default(),
            shared_choices_cpu: false,
            constraints: std::collections::HashMap::new(),
            draft: None,
            mtp: None,
            decoder: None,
            image_embeddings: Default::default(),
            vision: None,
            eos_ids,
            next_sequence_id: 1,
            batch_stats: BatchStats::default(),
            metrics: None,
        })
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
                "n > 1 requires a stateless model with CPU compute and no draft/MTP engine".into(),
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
        self.shared_choices_cpu
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
    fn copy_prompt_block(&mut self, src: usize, dst: usize) -> Result<(), String> {
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

    pub fn active_requests(&self) -> usize {
        self.scheduler.requests.len()
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

    pub fn enable_mtp(&mut self) -> Result<(), String> {
        if self.draft.is_some() || self.scheduler.cfg.speculative_lookup > 0 {
            return Err(
                "MTP, an external draft model and lookup speculation are mutually exclusive".into(),
            );
        }
        if self.decoder.is_some() {
            return Err("enable MTP before attaching compute".into());
        }
        self.mtp = Some(crate::draft::MtpEngine::new(&self.model)?);
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
        if self.mtp.is_some() {
            return Err("MTP and an external draft model are mutually exclusive".into());
        }
        self.draft = Some(crate::draft::DraftEngine::new(model, tokens)?);
        Ok(())
    }
    /// Load the decoder weights for `device` ("cpu" or a GPU backend). Must be called once before
    /// any request is processed.
    pub fn attach_compute(&mut self, device: &str) -> Result<(), String> {
        let Some((key_format, value_format)) = crate::kv_format::unpack_kind(self.model.kv_kind()) else {
            return Err("unknown KV cache format".into());
        };
        key_format.check_dim(self.model.config().head_dim)?;
        value_format.check_dim(self.model.config().value_dim)?;
        self.decoder = Some(crate::rust_decoder::RustDecoder::load(&self.model, device)?);
        // the host CPU runs the model unless an accelerator backend took it (`auto` falls back to the CPU on machines without a usable GPU)
        self.shared_choices_cpu = self.decoder.as_ref().is_some_and(|decoder| !decoder.uses_accelerator());
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

    pub fn step(&mut self) -> Result<Vec<TokenEvent>, String> {
        self.image_embeddings
            .retain(|sid, _| self.scheduler.sequences.contains_key(sid));
        if let Some(draft) = &mut self.draft {
            let active = self.scheduler.sequences.keys().copied().collect();
            draft.retain(&active)?;
            self.scheduler.external_drafts.clear();
            for &sid in &self.scheduler.running {
                let sequence = &self.scheduler.sequences[&sid];
                if sequence.num_computed >= sequence.prefill_target
                    && sequence.sampling.is_greedy()
                    && !sequence.constrained
                    && !self.pending_logits.contains_key(&sid)
                {
                    let tokens = draft.propose(sid, &sequence.tokens, sequence.remaining_max())?;
                    if !tokens.is_empty() {
                        self.scheduler.external_drafts.insert(sid, tokens);
                    }
                }
            }
        }
        let schedule_started = Instant::now();
        if let Some(mtp) = &mut self.mtp {
            let active = self.scheduler.sequences.keys().copied().collect();
            mtp.retain(&active)?;
            self.scheduler.external_drafts.clear();
            for &sid in &self.scheduler.running {
                let sequence = &self.scheduler.sequences[&sid];
                if sequence.num_computed >= sequence.prefill_target
                    && sequence.sampling.is_greedy()
                    && !sequence.constrained
                    && !self.pending_logits.contains_key(&sid)
                {
                    let tokens = mtp.propose(sid, &sequence.tokens, sequence.remaining_max());
                    if !tokens.is_empty() {
                        self.scheduler.external_drafts.insert(sid, tokens);
                    }
                }
            }
        }
        let scheduled = self.scheduler.schedule().map_err(|e| e.to_string())?;
        if let Some(metrics) = &self.metrics {
            let (running, waiting, free, total, hits) = self.metrics_snapshot();
            metrics.snapshot(running, waiting, free, total, hits);
        }
        self.batch_stats.scheduler_ns += schedule_started.elapsed().as_nanos() as u64;
        let Some(step) = scheduled else {
            return Ok(Vec::new());
        };
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
            base_row: usize,
            draft_rows: Vec<usize>,
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
            let table = self.table_row(*sid)?;
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
                table: self.table_row(*sid)?,
                embedding: None,
            });
            decode_rows += 1;
            decode_row.insert(*sid, row);
            logit_rows.push(row);
        }

        let mut pending_verify_events = Vec::new();
        for (sid, draft) in &step.verify {
            if let Some(logits) = self.pending_logits.remove(sid) {
                let token = argmax(&logits) as i64;
                let sequence = self
                    .scheduler
                    .sequences
                    .get_mut(sid)
                    .ok_or_else(|| format!("unknown sequence {sid}"))?;
                let request_id = sequence.req_id.clone();
                sequence.append_token(token);
                let hit_eos = self.eos_ids.contains(&token);
                let hit_limit = sequence.remaining_max() == 0;
                self.scheduler
                    .commit_decode(*sid)
                    .map_err(|e| e.to_string())?;
                let finished = hit_eos || hit_limit;
                let finish_reason = if hit_eos {
                    Some("stop")
                } else if hit_limit {
                    Some("length")
                } else {
                    None
                };
                pending_verify_events.push(TokenEvent {
                    request_id: request_id.clone(),
                    sequence_id: *sid,
                    token_id: token,
                    finished,
                    finish_reason,
                    logprobs: None,
                    cached_prompt_tokens: 0,
                    choice_index: 0,
                    request_finished: finished,
                });
                if finished {
                    self.scheduler
                        .free_request(&request_id)
                        .map_err(|e| e.to_string())?;
                    self.scheduler.sequences.remove(sid);
                }
                continue;
            }
            let sequence = self
                .scheduler
                .sequences
                .get(sid)
                .ok_or_else(|| format!("scheduler returned unknown sequence {sid}"))?;
            let position = sequence.num_computed;
            let token_id = *sequence.tokens.get(position).ok_or_else(|| {
                format!("verify position {position} is missing for sequence {sid}")
            })?;
            let table = self.table_row(*sid)?;
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
            verify_candidates.push(VerifyCandidate {
                sid: *sid,
                draft: draft.clone(),
                base_row,
                draft_rows: d_rows,
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
        if use_greedy_ids {
            crate::rust_decoder::RustDecoder::request_greedy_ids();
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
            Self::decoder(&self.decoder)?.forward_with_hidden(
                &ids,
                &lengths,
                &table,
                &mut self.key,
                &mut self.value,
                &logit_rows,
                false,
                &overrides,
                if self.mtp.is_some() {
                    Some(&mut hidden_capture)
                } else {
                    None
                },
            )?
        };
        let greedy_ids = if use_greedy_ids {
            crate::rust_decoder::RustDecoder::take_greedy_ids()
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
        let mut events = pending_verify_events;
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
            let (token, original_logits) = if let Some(token) = greedy_token {
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
            let original_logits = sampling.logprobs.map(|_| logits.to_vec());
            let generated = self
                .scheduler
                .sequences
                .get(&sid)
                .map_or(0, |s| s.output_len);
            if generated < sampling.min_tokens {
                for &id in self.eos_ids.iter().chain(&sampling.stop_token_ids) {
                    if let Some(value) = logits.to_mut().get_mut(id as usize) {
                        *value = f32::NEG_INFINITY;
                    }
                }
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
            let logprobs = original_logits
                .as_ref()
                .map(|logits| {
                    crate::sampling::raw_logprobs(
                        logits,
                        token as usize,
                        sampling.logprobs.unwrap_or(0),
                    )
                })
                .transpose()?;
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
        }

        for cand in verify_candidates {
            let sid = cand.sid;
            let sequence = self
                .scheduler
                .sequences
                .get_mut(&sid)
                .ok_or_else(|| format!("unknown sequence {sid}"))?;
            let request_id = sequence.req_id.clone();

            let compact_base = logit_rows
                .binary_search(&cand.base_row)
                .map_err(|_| format!("missing verify base LM-head result for {sid}"))?;
            let offset = compact_base * vocab;
            if offset + vocab > batch_logits.len() {
                return Err("LM head returned an invalid vocabulary row".into());
            }
            let mut pred_token = argmax(&batch_logits[offset..offset + vocab]) as i64;

            let mut all_drafts_accepted = true;
            let mut finished = false;

            for (idx, &draft_tok) in cand.draft.iter().enumerate() {
                if pred_token == draft_tok {
                    self.batch_stats.draft_accepted += 1;
                    if self.mtp.is_some() {
                        if let Some(metrics) = &self.metrics {
                            metrics.mtp_accepted(1);
                        }
                    }
                    sequence.append_token(draft_tok);
                    let hit_eos = self.eos_ids.contains(&draft_tok);
                    let hit_limit = sequence.remaining_max() == 0;
                    finished = hit_eos || hit_limit;
                    let finish_reason = if hit_eos {
                        Some("stop")
                    } else if hit_limit {
                        Some("length")
                    } else {
                        None
                    };
                    events.push(TokenEvent {
                        request_id: request_id.clone(),
                        sequence_id: sid,
                        token_id: draft_tok,
                        finished,
                        finish_reason,
                        logprobs: None,
                        cached_prompt_tokens: 0,
                        choice_index: 0,
                        request_finished: finished,
                    });
                    if finished {
                        all_drafts_accepted = false;
                        break;
                    }
                    let compact_d = logit_rows
                        .binary_search(&cand.draft_rows[idx])
                        .map_err(|_| format!("missing verify draft LM-head result for {sid}"))?;
                    let d_offset = compact_d * vocab;
                    if d_offset + vocab > batch_logits.len() {
                        return Err("LM head returned an invalid vocabulary row".into());
                    }
                    pred_token = argmax(&batch_logits[d_offset..d_offset + vocab]) as i64;
                } else {
                    sequence.append_token(pred_token);
                    let hit_eos = self.eos_ids.contains(&pred_token);
                    let hit_limit = sequence.remaining_max() == 0;
                    finished = hit_eos || hit_limit;
                    let finish_reason = if hit_eos {
                        Some("stop")
                    } else if hit_limit {
                        Some("length")
                    } else {
                        None
                    };
                    events.push(TokenEvent {
                        request_id: request_id.clone(),
                        sequence_id: sid,
                        token_id: pred_token,
                        finished,
                        finish_reason,
                        logprobs: None,
                        cached_prompt_tokens: 0,
                        choice_index: 0,
                        request_finished: finished,
                    });
                    all_drafts_accepted = false;
                    break;
                }
            }

            if all_drafts_accepted && !finished {
                sequence.append_token(pred_token);
                let hit_eos = self.eos_ids.contains(&pred_token);
                let hit_limit = sequence.remaining_max() == 0;
                finished = hit_eos || hit_limit;
                let finish_reason = if hit_eos {
                    Some("stop")
                } else if hit_limit {
                    Some("length")
                } else {
                    None
                };
                events.push(TokenEvent {
                    request_id: request_id.clone(),
                    sequence_id: sid,
                    token_id: pred_token,
                    finished,
                    finish_reason,
                    logprobs: None,
                    cached_prompt_tokens: 0,
                    choice_index: 0,
                    request_finished: finished,
                });
            }

            self.scheduler
                .commit_verify(sid)
                .map_err(|e| e.to_string())?;

            if finished {
                self.scheduler
                    .free_request(&request_id)
                    .map_err(|e| e.to_string())?;
                self.scheduler.sequences.remove(&sid);
                self.pending_logits.remove(&sid);
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

fn argmax(logits: &[f32]) -> usize {
    crate::sampling::argmax(logits)
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
        };
        assert_eq!(cfg.speculative_lookup, 3);
        assert_eq!(cfg.speculative_window, 32);
    }
}
