//! A separate, cached draft model proposes tokens; the target verifies every proposal.
use crate::{
    mem::block_allocator::BlockManager,
    sampling::{argmax, philox_uniform, processed_distribution, select_from, SamplingParams},
    speculative::Dist,
    GgufModel,
};
use std::collections::{HashMap, HashSet};

struct DraftSequence {
    tokens: Vec<i64>,
    logits: Vec<f32>,
}
pub struct DraftEngine {
    model: GgufModel,
    blocks: BlockManager,
    key: Vec<u16>,
    value: Vec<u16>,
    sequences: HashMap<i64, DraftSequence>,
    /// Number of uniform draws taken from the draft's own random stream, per target sequence.
    draws: HashMap<i64, u64>,
    pub tokens: usize,
    decoder: Option<crate::rust_decoder::RustDecoder>,
}

impl DraftEngine {
    pub fn new(model: GgufModel, tokens: usize) -> Result<Self, String> {
        if tokens == 0 || tokens > 32 {
            return Err("draft token count must be between 1 and 32".into());
        }
        let blocks = BlockManager::new(model.pool_blocks() as i64, model.block_size() as i64)
            .map_err(|e| e.to_string())?;
        let key_len = model.key_buffer_len()?;
        let value_len = model.value_buffer_len()?;
        Ok(Self {
            model,
            blocks,
            key: vec![0; key_len],
            value: vec![0; value_len],
            sequences: HashMap::new(),
            draws: HashMap::new(),
            tokens,
            decoder: None,
        })
    }
    pub fn attach_compute(&mut self, device: &str) -> Result<(), String> {
        self.decoder = Some(crate::rust_decoder::RustDecoder::load(&self.model, device)?);
        Ok(())
    }

    pub fn retain(&mut self, active: &HashSet<i64>) -> Result<(), String> {
        let expired: Vec<i64> = self
            .sequences
            .keys()
            .filter(|id| !active.contains(id))
            .copied()
            .collect();
        for id in expired {
            self.sequences.remove(&id);
            self.draws.remove(&id);
            self.blocks.drop_table(id).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Draft up to `limit` (at most `self.tokens`) tokens. The tokens are drawn from the draft model's distribution after the same
    /// filters the request applies to the target (`sampling`), and that distribution is returned with each token (`None` for a point
    /// mass, i.e. a greedy request) so the verification can use the speculative sampling rule.
    pub fn propose(
        &mut self,
        sid: i64,
        history: &[i64],
        remaining: usize,
        limit: usize,
        sampling: &SamplingParams,
    ) -> Result<(Vec<i64>, Vec<Option<Dist>>), String> {
        let count = self
            .tokens
            .min(limit)
            .min(remaining.saturating_sub(1))
            .min(self.model.max_context().saturating_sub(history.len()));
        if count == 0 || history.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        if !self.sequences.contains_key(&sid) {
            self.blocks.new_table(sid).map_err(|e| e.to_string())?;
            self.sequences.insert(
                sid,
                DraftSequence {
                    tokens: Vec::new(),
                    logits: Vec::new(),
                },
            );
        }
        let decoder = self
            .decoder
            .as_ref()
            .ok_or("draft compute is not initialized (call attach_compute)")?;
        let sequence = self.sequences.get_mut(&sid).unwrap();
        let matched = common_prefix(&sequence.tokens, history);
        // The last matching row needs to be evaluated again when its saved logits
        // were produced from a later, rejected draft continuation.
        let start = if matched == history.len()
            && matched == sequence.tokens.len()
            && !sequence.logits.is_empty()
        {
            matched
        } else {
            matched.saturating_sub(1)
        };
        sequence.tokens.truncate(start);
        self.blocks
            .ensure_blocks(sid, (history.len() + count) as i64)
            .map_err(|e| e.to_string())?;
        let blocks = &self
            .blocks
            .get_table(sid)
            .map_err(|e| e.to_string())?
            .blocks;
        let mut table = vec![-1; self.model.table_stride()];
        for (slot, id) in table.iter_mut().zip(blocks) {
            *slot = *id as i64;
        }
        for chunk_start in (start..history.len()).step_by(128) {
            let end = (chunk_start + 128).min(history.len());
            let lengths: Vec<i64> = (chunk_start + 1..=end).map(|n| n as i64).collect();
            let logits = decoder.forward(
                &history[chunk_start..end],
                &lengths,
                &table.repeat(end - chunk_start),
                &mut self.key,
                &mut self.value,
                &[end - chunk_start - 1],
                false,
            )?;
            sequence.logits = logits;
            sequence
                .tokens
                .extend_from_slice(&history[chunk_start..end]);
        }
        let mut draft = Vec::with_capacity(count);
        let mut q = Vec::with_capacity(count);
        let mut context = history.to_vec();
        let key = sampling.seed.unwrap_or(0x2545_f491_4f6c_dd1d ^ sid as u64) ^ 0xd6e8_feb8_6659_fd93;
        for _ in 0..count {
            let token = if sampling.is_greedy() {
                q.push(None);
                argmax(&sequence.logits) as i64
            } else {
                let dist = processed_distribution(&sequence.logits, &context, sampling, 0.0);
                let token = if dist.len() == 1 {
                    dist[0].0
                } else {
                    let draws = self.draws.entry(sid).or_default();
                    let u = philox_uniform(key, *draws, 0);
                    *draws += 1;
                    select_from(&dist, u).0
                };
                q.push(if dist.len() == 1 { None } else { Some(dist) });
                token as i64
            };
            draft.push(token);
            context.push(token);
            let length = sequence.tokens.len() + 1;
            sequence.logits = decoder.forward(
                &[token],
                &[length as i64],
                &table,
                &mut self.key,
                &mut self.value,
                &[0],
                false,
            )?;
            sequence.tokens.push(token);
        }
        Ok((draft, q))
    }
}

fn common_prefix(a: &[i64], b: &[i64]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

/// The first trained NextN block proposes one token. Only committed target
/// hidden states warm its separate KV cache; rejected target rows never enter it.
pub(crate) struct MtpEngine {
    decoder: Option<crate::rust_decoder::RustDecoder>,
    blocks: BlockManager,
    key: Vec<u16>,
    value: Vec<u16>,
    stride: usize,
    hidden: usize,
    sequences: HashMap<i64, MtpSequence>,
}
#[derive(Default)]
struct MtpSequence {
    length: usize,
    token: Option<i64>,
    logits: Vec<f32>,
    pending: Option<(usize, Vec<f32>)>,
}

impl MtpEngine {
    pub(crate) fn new(model: &GgufModel) -> Result<Self, String> {
        if model.is_stateful() {
            return Err("MTP is unavailable for stateful DeltaNet/Mamba models".into());
        }
        // Validate all weights at compute attachment, without running a model.
        Ok(Self {
            decoder: None,
            blocks: BlockManager::new(model.pool_blocks() as i64, model.block_size() as i64)
                .map_err(|e| e.to_string())?,
            key: vec![0; model.kv_words_per_layer(model.kv_formats().0, model.config().head_dim)],
            value: vec![0; model.kv_words_per_layer(model.kv_formats().1, model.config().value_dim)],
            stride: model.table_stride(),
            hidden: model.hidden_size(),
            sequences: HashMap::new(),
        })
    }
    pub(crate) fn attach_compute(&mut self, model: &GgufModel, device: &str) -> Result<(), String> {
        self.decoder = Some(crate::rust_decoder::RustDecoder::load_mtp(model, device)?);
        Ok(())
    }
    pub(crate) fn retain(&mut self, active: &HashSet<i64>) -> Result<(), String> {
        let expired: Vec<_> = self
            .sequences
            .keys()
            .filter(|sid| !active.contains(sid))
            .copied()
            .collect();
        for sid in expired {
            self.sequences.remove(&sid);
            self.blocks.drop_table(sid).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub(crate) fn propose(&self, sid: i64, history: &[i64], remaining: usize) -> Vec<i64> {
        let Some(s) = self.sequences.get(&sid) else {
            return Vec::new();
        };
        if remaining <= 1
            || s.length + 1 != history.len()
            || s.token != history.last().copied()
            || s.logits.is_empty()
        {
            return Vec::new();
        }
        vec![argmax(&s.logits) as i64]
    }
    /// Positions are target positions p; the MTP input is token[p+1]. Logical
    /// cache slot p corresponds to RoPE position p+1, so no artificial zero KV
    /// entry is exposed at the beginning of the shifted sequence.
    pub(crate) fn observe(
        &mut self,
        sid: i64,
        history: &[i64],
        computed: usize,
        mut rows: Vec<(usize, Vec<f32>)>,
    ) -> Result<(), String> {
        if !self.sequences.contains_key(&sid) {
            self.blocks.new_table(sid).map_err(|e| e.to_string())?;
            self.sequences.insert(sid, MtpSequence::default());
        }
        let s = self.sequences.get_mut(&sid).ok_or("missing MTP sequence")?;
        if let Some(pending) = s.pending.take() {
            rows.push(pending);
        }
        rows.sort_by_key(|r| r.0);
        rows.dedup_by_key(|r| r.0);
        let mut ids = Vec::new();
        let mut hidden = Vec::new();
        let mut lengths = Vec::new();
        for (position, h) in rows {
            if position >= computed {
                continue;
            }
            if h.len() != self.hidden {
                return Err("invalid captured MTP hidden state".into());
            }
            if position + 1 >= history.len() {
                s.pending = Some((position, h));
                continue;
            }
            if ids.is_empty() && position > s.length {
                return Err(format!(
                    "MTP sequence {sid}: missing committed hidden state at {}",
                    s.length
                ));
            }
            if position < s.length && !ids.is_empty() {
                return Err("non-contiguous MTP warmup".into());
            }
            if ids.is_empty() {
                s.length = position;
            }
            if position != s.length + ids.len() {
                return Err("non-contiguous MTP hidden rows".into());
            }
            ids.push(history[position + 1]);
            hidden.extend(h);
            lengths.push((position + 1) as i64);
        }
        if ids.is_empty() {
            return Ok(());
        }
        self.blocks
            .ensure_blocks(sid, *lengths.last().ok_or("empty MTP batch")?)
            .map_err(|e| e.to_string())?;
        let blocks = &self
            .blocks
            .get_table(sid)
            .map_err(|e| e.to_string())?
            .blocks;
        let mut table = vec![-1; self.stride];
        for (slot, id) in table.iter_mut().zip(blocks) {
            *slot = *id as i64;
        }
        s.logits = self
            .decoder
            .as_ref()
            .ok_or("MTP compute is not initialized")?
            .forward_mtp(
                &ids,
                &hidden,
                &lengths,
                &table.repeat(ids.len()),
                &mut self.key,
                &mut self.value,
            )?;
        s.length += ids.len();
        s.token = ids.last().copied();
        Ok(())
    }
}


/// Speculative draft engine for Medusa, EAGLE, and EAGLE-3 heads.
pub struct SpecHeadEngine {
    pub method: crate::speculative::SpeculativeMethod,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub heads: Vec<crate::compute::Linear>,
    pub sequences: HashMap<i64, Vec<f32>>,
}

impl SpecHeadEngine {
    pub fn new(model: &GgufModel, method: crate::speculative::SpeculativeMethod) -> Result<Self, String> {
        let mapping = model.mapping();
        let meta = &mapping.index().metadata;
        let hidden_size = meta
            .get("spec_head.hidden_size")
            .and_then(serde_json::Value::as_u64)
            .map(|x| x as usize)
            .unwrap_or_else(|| model.hidden_size());
        let vocab_size = meta
            .get("spec_head.vocab_size")
            .and_then(serde_json::Value::as_u64)
            .map(|x| x as usize)
            .unwrap_or_else(|| model.vocab_size());
        let mut heads = Vec::new();
        match method {
            crate::speculative::SpeculativeMethod::Medusa => {
                let num_heads = meta
                    .get("spec_head.num_heads")
                    .and_then(serde_json::Value::as_u64)
                    .map(|x| x as usize)
                    .unwrap_or(4);
                for i in 0..num_heads {
                    let name = format!("spec_head.medusa.{i}");
                    if let Ok(l) = crate::compute::Linear::load(mapping, &name, vocab_size, hidden_size) {
                        heads.push(l);
                    } else if let Ok(l) = crate::compute::Linear::load(mapping, &format!("spec_head.{i}"), vocab_size, hidden_size) {
                        heads.push(l);
                    } else if let Ok(l) = crate::compute::Linear::load(mapping, &format!("medusa_heads.{i}"), vocab_size, hidden_size) {
                        heads.push(l);
                    }
                }
            }
            crate::speculative::SpeculativeMethod::Eagle | crate::speculative::SpeculativeMethod::Eagle3 => {
                let num_heads = meta
                    .get("spec_head.num_heads")
                    .and_then(serde_json::Value::as_u64)
                    .map(|x| x as usize)
                    .unwrap_or(1);
                for i in 0..num_heads {
                    let name = format!("spec_head.eagle.{i}");
                    if let Ok(l) = crate::compute::Linear::load(mapping, &name, vocab_size, hidden_size) {
                        heads.push(l);
                    }
                }
                if heads.is_empty() {
                    if let Ok(l) = crate::compute::Linear::load(mapping, "spec_head.head", vocab_size, hidden_size) {
                        heads.push(l);
                    } else if let Ok(l) = crate::compute::Linear::load(mapping, "lm_head", vocab_size, hidden_size) {
                        heads.push(l);
                    }
                }
            }
            _ => return Err("unsupported spec-head method".into()),
        }
        Ok(Self {
            method,
            hidden_size,
            vocab_size,
            heads,
            sequences: HashMap::new(),
        })
    }

    pub fn retain(&mut self, active: &HashSet<i64>) -> Result<(), String> {
        self.sequences.retain(|sid, _| active.contains(sid));
        Ok(())
    }

    pub fn observe(&mut self, sid: i64, hidden: &[f32]) {
        self.sequences.insert(sid, hidden.to_vec());
    }

    pub fn propose(&self, sid: i64, max_tokens: usize) -> Vec<i64> {
        let Some(h) = self.sequences.get(&sid) else {
            return Vec::new();
        };
        if self.heads.is_empty() || h.is_empty() {
            return Vec::new();
        }
        let mut tokens = Vec::new();
        let limit = max_tokens.min(self.heads.len());
        for head in &self.heads[..limit] {
            if let Ok(logits) = head.forward(h) {
                tokens.push(argmax(&logits) as i64);
            }
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rollback_stops_at_the_first_rejected_draft() {
        assert_eq!(common_prefix(&[1, 2, 3, 9, 8], &[1, 2, 3, 4]), 3);
    }
    #[test]
    fn mtp_prompt_warmup_collects_contiguous_rows_before_compute() {
        let mut engine = MtpEngine {
            decoder: None,
            blocks: BlockManager::new(8, 4).unwrap(),
            key: Vec::new(),
            value: Vec::new(),
            stride: 2,
            hidden: 2,
            sequences: HashMap::new(),
        };
        let rows = vec![
            (0, vec![1.0, 2.0]),
            (1, vec![3.0, 4.0]),
            (2, vec![5.0, 6.0]),
        ];
        let error = engine.observe(1, &[4, 5, 6, 7], 3, rows).unwrap_err();
        // Validation must reach compute initialization, rather than treating
        // the second contiguous prompt row as a missing hidden state.
        assert_eq!(error, "MTP compute is not initialized");
        let error = engine
            .observe(2, &[4, 5, 6, 7], 3, vec![(1, vec![1.0, 2.0])])
            .unwrap_err();
        assert!(error.contains("missing committed hidden state at 0"));
    }

    #[test]
    fn mtp_seed_requires_current_token_and_capacity_for_verification() {
        let mut engine = MtpEngine {
            decoder: None,
            blocks: BlockManager::new(8, 4).unwrap(),
            key: Vec::new(),
            value: Vec::new(),
            stride: 2,
            hidden: 2,
            sequences: HashMap::new(),
        };
        engine.sequences.insert(
            1,
            MtpSequence {
                length: 2,
                token: Some(9),
                logits: vec![0.0, 2.0],
                pending: None,
            },
        );
        assert_eq!(engine.propose(1, &[4, 5, 9], 3), vec![1]);
        assert!(engine.propose(1, &[4, 5, 8], 3).is_empty());
        assert!(engine.propose(1, &[4, 5, 9], 1).is_empty());
        assert!(engine.propose(1, &[4, 5, 9, 2], 3).is_empty());
    }

    #[test]
    fn spec_head_engine_propose_returns_expected_tokens() {
        let mut sequences = HashMap::new();
        sequences.insert(1, vec![1.0, 2.0]);
        // Engine with no heads should safely return empty
        let engine = SpecHeadEngine {
            method: crate::speculative::SpeculativeMethod::Medusa,
            hidden_size: 2,
            vocab_size: 10,
            heads: Vec::new(),
            sequences,
        };
        assert!(engine.propose(1, 4).is_empty());
        assert!(engine.propose(2, 4).is_empty());
    }
}