//! A separate, cached draft model proposes tokens; the target verifies every proposal.
use crate::{mem::block_allocator::BlockManager, sampling::argmax, GgufModel};
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
            self.blocks.drop_table(id).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn propose(
        &mut self,
        sid: i64,
        history: &[i64],
        remaining: usize,
    ) -> Result<Vec<i64>, String> {
        let count = self
            .tokens
            .min(remaining.saturating_sub(1))
            .min(self.model.max_context().saturating_sub(history.len()));
        if count == 0 || history.is_empty() {
            return Ok(Vec::new());
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
        for _ in 0..count {
            let token = argmax(&sequence.logits) as i64;
            draft.push(token);
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
        Ok(draft)
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
}
