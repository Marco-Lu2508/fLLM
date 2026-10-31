//! Qwen3.8-Flash-Next on the host (see `rust/checkpoint/src/qwen4_exp.rs` for the model and the tensor names).
//!
//! The layers are those of Qwen3.5: gated delta layers (`delta_forward`) and gated softmax attention over the ordinary KV pools, and the
//! softmax MoE (`ffn_forward`). Around them this model keeps `hc` residual streams per token. Instead of a norm in front of a block, a gate
//! reads the normalised streams and mixes them into the block input; the block output is added to every stream with its own weight.
//! Delta layers with an n-gram embedding first add a gated, convolved lexical feature to every stream; attention layers attend only to
//! the best blocks of tokens that a small indexer picks (and to the unfinished last block).
//!
//! What a sequence keeps besides the KV pools and the delta states lives in `Qwen4Model::seqs`, keyed like the recurrent states by the first
//! KV block: the last token ids and the convolution history of every n-gram layer, and the indexer keys of every attention layer.
use super::*;
use std::collections::HashMap;

mod device;
pub(super) use device::{DeviceQwen4, Q4Step};
use std::sync::Mutex;

/// Gate of a block: reads the normalised streams, mixes them into one input and weights the injection of the block output.
struct Gate {
    norm: Vec<f32>,
    down: Linear,
    up: Linear,
    inject: Option<Linear>,
}

struct Indexer {
    qk: Linear,
    /// The query rows and the key row of `qk` as two projections (the device keeps them apart).
    split: (Linear, Linear),
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

/// Per-layer hashed n-gram embedding added to the streams.
struct Ple {
    embedding: Embedding,
    /// Head vocabulary sizes and their offsets in the table.
    sizes: Vec<u64>,
    offsets: Vec<u64>,
    multipliers: Vec<i64>,
    key: Linear,
    value: Linear,
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    /// `channels x kernel`
    conv: Vec<f32>,
}

struct Q4Layer {
    attn: Gate,
    ffn: Gate,
    indexer: Option<Indexer>,
    ple: Option<Ple>,
}

struct PleState {
    len: usize,
    /// The last `ngram - 1` token ids (the end-of-sequence id before the first token).
    ids: Vec<i64>,
    /// The last `(kernel - 1) * dilation` inputs of the convolution, oldest first.
    hist: Vec<f32>,
}

#[derive(Default)]
struct IndexState {
    len: usize,
    /// Raw indexer key of every token.
    raw: Vec<f32>,
    /// Pooled, normalised and rotated key of every finished block.
    blocks: Vec<f32>,
}

struct SeqState {
    used: u64,
    ple: Vec<Option<PleState>>,
    index: Vec<Option<IndexState>>,
}

#[derive(Default)]
struct Store {
    clock: u64,
    map: HashMap<i64, SeqState>,
}

pub(super) struct Qwen4Model {
    layers: Vec<Q4Layer>,
    head: Gate,
    seqs: Mutex<Store>,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// RMS norm over every group of `group` values of each row of `x` (rows of `weight.len()` values), with the per-value `weight`.
fn group_rms(x: &[f32], weight: &[f32], group: usize, eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    for row in out.chunks_exact_mut(weight.len()) {
        for (g, chunk) in row.chunks_exact_mut(group).enumerate() {
            let inv = (chunk.iter().map(|v| v * v).sum::<f32>() / group as f32 + eps).sqrt().recip();
            for (j, v) in chunk.iter_mut().enumerate() {
                *v *= inv * weight[g * group + j];
            }
        }
    }
    out
}

fn splitmix64(value: u64) -> u64 {
    let mut v = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

/// The `count`-th prime after `start`.
fn nth_prime_after(start: u64, count: u64) -> u64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

/// Odd multipliers of the hash of the PLE layer `ordinal` (the position among the PLE layers).
fn layer_multipliers(vocab: u64, ngram: usize, ordinal: u64, seed: u64) -> Vec<i64> {
    let max_long = i64::MAX as u64;
    let half_bound = (max_long / vocab.max(1) / 2).max(1);
    let base_seed = seed + 10007 * ordinal;
    (0..ngram as u64)
        .map(|index| {
            let value = base_seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(index + 1));
            (2 * (splitmix64(value) % half_bound) + 1) as i64
        })
        .collect()
}

impl Gate {
    fn load(model: &GgufModel, p: &str, hc: usize, lowrank: usize, hidden: usize, inject: bool) -> Result<Self, String> {
        let mapping = model.mapping();
        let load = |name: &str, rows: usize, cols: usize| -> Result<Linear, String> {
            let mut l = Linear::load(mapping, name, rows, cols)?;
            l.replace_weights(model.adapted_projection(&format!("{name}.weight")))?;
            l.enable_fast_cpu();
            Ok(l)
        };
        let width = hc * hidden;
        Ok(Self {
            norm: vector(mapping, &format!("{p}_norm.weight"), width)?,
            down: load(&format!("{p}_down"), lowrank, width)?,
            up: load(&format!("{p}_up"), width, lowrank)?,
            inject: if inject { Some(load(&format!("{p}_inject"), hc, width)?) } else { None },
        })
    }

    /// `(block input, injection weights)` for the streams (rows x hc x hidden): the input is the mean over the streams of the normalised
    /// streams times their sigmoid gates (rows x hidden); the injection weights are `2 * sigmoid(...)` per stream (rows x hc).
    fn mix(&self, streams: &[f32], hc: usize, hidden: usize, eps: f32) -> Result<(Vec<f32>, Vec<f32>), String> {
        let rows = streams.len() / (hc * hidden);
        let normed = group_rms(streams, &self.norm, hidden, eps);
        let mut low = self.down.forward(&normed)?;
        low.iter_mut().for_each(|v| *v = silu(*v / hc as f32));
        let gates = self.up.forward(&low)?;
        let mut input = vec![0.0f32; rows * hidden];
        for r in 0..rows {
            let out = &mut input[r * hidden..(r + 1) * hidden];
            for n in 0..hc {
                let at = (r * hc + n) * hidden;
                for d in 0..hidden {
                    out[d] += sigmoid(gates[at + d]) * normed[at + d] / hc as f32;
                }
            }
        }
        let inject = match &self.inject {
            Some(l) => {
                let mut w = l.forward(&normed)?;
                w.iter_mut().for_each(|v| *v = 2.0 * sigmoid(*v / hc as f32));
                w
            }
            None => Vec::new(),
        };
        Ok((input, inject))
    }
}

/// `streams[r][n] += branch[r] * weights[r][n]`
fn inject(streams: &mut [f32], branch: &[f32], weights: &[f32], hc: usize, hidden: usize) {
    for (r, row) in branch.chunks_exact(hidden).enumerate() {
        for n in 0..hc {
            let w = weights[r * hc + n];
            let at = (r * hc + n) * hidden;
            streams[at..at + hidden].iter_mut().zip(row).for_each(|(s, b)| *s += b * w);
        }
    }
}

impl Qwen4Model {
    pub(super) fn load(model: &GgufModel, cfg: &ModelConfig) -> Result<Self, String> {
        let qc = cfg.qwen4.as_ref().ok_or("not a Qwen3.8-Flash-Next model")?;
        let mapping = model.mapping();
        let (hidden, hc) = (cfg.hidden, qc.hc);
        let width = hc * hidden;
        let mut layers = Vec::with_capacity(cfg.layers);
        let mut ordinal = 0u64;
        for i in 0..cfg.layers {
            let p = format!("blk.{i}");
            let load = |name: &str, rows: usize, cols: usize| -> Result<Linear, String> {
                let mut l = Linear::load(mapping, &format!("{p}.{name}"), rows, cols)?;
                l.replace_weights(model.adapted_projection(&format!("{p}.{name}.weight")))?;
                l.enable_fast_cpu();
                Ok(l)
            };
            let indexer = match qc.indexer {
                Some((heads, dim, _, _)) if has_tensor(mapping, &format!("{p}.indexer.qk.weight")) => {
                    let name = format!("{p}.indexer.qk.weight");
                    let adapted = model.adapted_projection(&name);
                    let part = |first: usize, rows: usize| -> Result<Linear, String> {
                        let mut l = Linear::load_rows(mapping, &name, None, first, rows, hidden)?;
                        l.replace_weights(adapted.as_ref().map(|all| all[first * hidden..(first + rows) * hidden].to_vec()))?;
                        l.enable_fast_cpu();
                        Ok(l)
                    };
                    Some(Indexer {
                        qk: load("indexer.qk", (heads + 1) * dim, hidden)?,
                        split: (part(0, heads * dim)?, part(heads * dim, dim)?),
                        q_norm: vector(mapping, &format!("{p}.indexer.q_norm.weight"), dim)?,
                        k_norm: vector(mapping, &format!("{p}.indexer.k_norm.weight"), dim)?,
                    })
                }
                _ => None,
            };
            let ple = if qc.ple.get(i).copied().unwrap_or(false) {
                let heads = (qc.ngram - 1) * qc.heads_per_ngram;
                let head_dim = qc.ple_embed / heads;
                let (mut sizes, mut offsets, mut total) = (Vec::with_capacity(heads), Vec::with_capacity(heads), 0u64);
                for h in 0..heads as u64 {
                    let size = nth_prime_after(qc.vocab_base - 1, ordinal * heads as u64 + h + 1);
                    sizes.push(size);
                    offsets.push(total);
                    total += size;
                }
                let embedding = Embedding::load_with_cols(model, &format!("{p}.ple_embd.weight"), head_dim)?;
                if (embedding.rows as u64) < total {
                    return Err(format!("layer {i}: the n-gram embedding has {} rows, the hash needs {total}", embedding.rows));
                }
                let state = Ple {
                    embedding,
                    sizes,
                    offsets,
                    multipliers: layer_multipliers(qc.vocab, qc.ngram, ordinal, qc.seed),
                    key: load("ple_key", width, qc.ple_embed)?,
                    value: load("ple_value", hidden, qc.ple_embed)?,
                    norm_key: vector(mapping, &format!("{p}.ple_norm_key.weight"), width)?,
                    norm_query: vector(mapping, &format!("{p}.ple_norm_query.weight"), width)?,
                    norm_conv: vector(mapping, &format!("{p}.ple_norm_conv.weight"), width)?,
                    conv: vector(mapping, &format!("{p}.ple_conv.weight"), width * qc.ple_kernel)?,
                };
                ordinal += 1;
                Some(state)
            } else {
                None
            };
            layers.push(Q4Layer {
                attn: Gate::load(model, &format!("{p}.hc_attn"), hc, qc.lowrank, hidden, true)?,
                ffn: Gate::load(model, &format!("{p}.hc_ffn"), hc, qc.lowrank, hidden, true)?,
                indexer,
                ple,
            });
        }
        Ok(Self { layers, head: Gate::load(model, "output_hc", hc, qc.lowrank, hidden, false)?, seqs: Mutex::new(Store::default()) })
    }

    /// Forgets what a finished sequence kept.
    pub(super) fn forget(&self, first: i64) {
        if let Ok(mut store) = self.seqs.lock() {
            store.map.remove(&first);
        }
    }
}

impl RustDecoder {
    /// The runs of consecutive tokens of each sequence: `(first row, end row, first KV block, position of the first row)`.
    fn q4_runs(&self, tables: &[i64], lengths: &[i64]) -> Vec<(usize, usize, i64, usize)> {
        let rows = lengths.len();
        let mut out = Vec::new();
        let mut start = 0;
        while start < rows {
            let first = tables[start * self.stride];
            let position = lengths[start] as usize - 1;
            let mut end = start + 1;
            while end < rows && tables[end * self.stride] == first && lengths[end] == lengths[end - 1] + 1 {
                end += 1;
            }
            out.push((start, end, first, position));
            start = end;
        }
        out
    }

    /// The state of a sequence (created on demand, the oldest one is dropped when the store is full).
    fn q4_seq<'a>(&self, store: &'a mut Store, first: i64) -> &'a mut SeqState {
        store.clock += 1;
        let clock = store.clock;
        if !store.map.contains_key(&first) && store.map.len() >= self.state_capacity {
            if let Some(oldest) = store.map.iter().min_by_key(|(_, s)| s.used).map(|(k, _)| *k) {
                store.map.remove(&oldest);
            }
        }
        let layers = self.cfg.layers;
        let seq = store.map.entry(first).or_insert_with(|| SeqState { used: clock, ple: (0..layers).map(|_| None).collect(), index: (0..layers).map(|_| None).collect() });
        seq.used = clock;
        seq
    }

    /// All layers of a Qwen3.8-Flash-Next model for a batch: `x` holds the embeddings (rows x hidden); the result is the collapsed stream.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen4_layers(&self, x: &[f32], ids: &[i64], positions: &[usize], tables: &[i64], lengths: &[i64], key: &mut [u16], value: &mut [u16]) -> Result<Vec<f32>, String> {
        let (cfg, model) = (&self.cfg, self.qwen4.as_deref().ok_or("Qwen3.8-Flash-Next weights missing")?);
        let qc = cfg.qwen4.as_ref().ok_or("Qwen3.8-Flash-Next configuration missing")?;
        let (rows, hidden, hc) = (positions.len(), cfg.hidden, qc.hc);
        let mut streams = Vec::with_capacity(rows * hc * hidden);
        for r in 0..rows {
            for _ in 0..hc {
                streams.extend_from_slice(&x[r * hidden..(r + 1) * hidden]);
            }
        }
        let mut workspace = compute::ProjectionGroupWorkspace::default();
        for (i, (layer, ql)) in self.layers.iter().zip(&model.layers).enumerate() {
            if let Some(ple) = &ql.ple {
                let add = self.q4_ple(model, qc, i, ple, &streams, ids, tables, lengths)?;
                dump_layer("ple", i, &add);
                streams.iter_mut().zip(&add).for_each(|(s, a)| *s += a);
            }
            let (normed, weights) = ql.attn.mix(&streams, hc, hidden, cfg.eps)?;
            let branch = match &layer.attn {
                Attn::Linear(delta) => self.delta_forward(delta, &layer.out, i, &normed, tables, lengths)?,
                Attn::Std { .. } => self.q4_attention(model, qc, i, layer, ql, &normed, positions, tables, lengths, key, value)?,
                _ => return Err("a Qwen3.8-Flash-Next layer that is neither delta nor attention".into()),
            };
            dump_layer("branch", i, &branch);
            inject(&mut streams, &branch, &weights, hc, hidden);
            let (normed, weights) = ql.ffn.mix(&streams, hc, hidden, cfg.eps)?;
            let mlp = self.ffn_forward(layer, &normed, &mut workspace)?;
            inject(&mut streams, &mlp, &weights, hc, hidden);
            dump_layer("ffn", i, &streams);
        }
        Ok(model.head.mix(&streams, hc, hidden, cfg.eps)?.0)
    }

    /// Gated softmax attention of one layer over the selected tokens; returns the projected output (rows x hidden).
    #[allow(clippy::too_many_arguments)]
    fn q4_attention(&self, model: &Qwen4Model, qc: &crate::config::Qwen4Config, i: usize, layer: &Layer, ql: &Q4Layer, normed: &[f32], positions: &[usize], tables: &[i64], lengths: &[i64], key: &mut [u16], value: &mut [u16]) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let Attn::Std { q: qp, k: kp, v: vp } = &layer.attn else {
            return Err("not an attention layer".into());
        };
        let rows = positions.len();
        let (dk, dv) = (cfg.layer_head_dim(i), cfg.layer_value_dim(i));
        let (heads, kv_heads) = (cfg.layer_heads(i), cfg.layer_kv_heads(i));
        let (kv_k, kv_v) = (kv_heads * dk, kv_heads * dv);
        let rope_dims = cfg.layer_rope_dims(i);
        let table = &self.rope[layer.rope_table];
        let mut qkv = Linear::forward_group(&[qp, kp, vp], normed)?.into_iter();
        let (mut q, mut k, v) = qkv.next().zip(qkv.next()).zip(qkv.next()).map(|((q, k), v)| (q, k, v)).ok_or("attention projection group")?;
        // all queries of a token, then all gates (the converter regroups the checkpoint's per-head pairs)
        let mut gate = Vec::new();
        if layer.output_gate {
            let width = heads * dk;
            let (mut plain, mut gates) = (Vec::with_capacity(rows * width), Vec::with_capacity(rows * width));
            for row in q.chunks_exact(2 * width) {
                plain.extend_from_slice(&row[..width]);
                gates.extend_from_slice(&row[width..]);
            }
            q = plain;
            gate = gates;
        }
        qk_norm_cpu(&mut q, layer.q_norm.as_deref(), layer.q_norm_bias.as_deref(), cfg.eps);
        qk_norm_cpu(&mut k, layer.k_norm.as_deref(), layer.k_norm_bias.as_deref(), cfg.eps);
        if cfg.layer_rotates(i) {
            apply_rope(&mut q, heads, dk, rope_dims, cfg.rope.neox, positions, table);
            apply_rope(&mut k, kv_heads, dk, rope_dims, cfg.rope.neox, positions, table);
        }
        let (fmt_k, fmt_v) = self.kv_formats();
        let (key_region, value_region) = (self.key_region(i), self.value_region(i));
        for (r, &position) in positions.iter().enumerate() {
            let block = tables[r * self.stride + position / self.block_size] as usize;
            let slot = block * self.block_size + position % self.block_size;
            self.write_cache(key, key_region, slot, &k[r * kv_k..(r + 1) * kv_k], fmt_k);
            self.write_cache(value, value_region, slot, &v[r * kv_v..(r + 1) * kv_v], fmt_v);
        }
        // `FLLM_DSA=dense` attends to every key (an approximation: the model selects blocks)
        let selection = match (&ql.indexer, qc.indexer) {
            (Some(ix), Some(shape)) if !self.dsa_dense => Some(self.q4_select(model, ix, shape, i, layer, normed, positions, tables, lengths)?),
            _ => None,
        };
        let mut attn = self.attention_selected(i, &q, key, value, tables, lengths, selection.as_deref());
        if !gate.is_empty() {
            attn.iter_mut().zip(&gate).for_each(|(a, g)| *a *= cfg.attn_gate.apply(*g));
        }
        layer.out.forward(&attn)
    }

    /// The tokens every query row attends to: the tokens of the best `budget / ratio` finished blocks (scored by the indexer: the sum over
    /// its heads of `relu(query . block key)`, ties to the earlier block) plus the tokens of the unfinished last block. `None` while all
    /// finished blocks fit.
    #[allow(clippy::too_many_arguments)]
    fn q4_select(&self, model: &Qwen4Model, ix: &Indexer, shape: (usize, usize, usize, usize), i: usize, layer: &Layer, normed: &[f32], positions: &[usize], tables: &[i64], lengths: &[i64]) -> Result<Vec<Option<Vec<u32>>>, String> {
        let cfg = &self.cfg;
        let (heads, dim, budget, ratio) = shape;
        let rows = positions.len();
        let rope_dims = cfg.layer_rope_dims(i);
        let table = &self.rope[layer.rope_table];
        let projected = ix.qk.forward(normed)?;
        let width = (heads + 1) * dim;
        let (mut queries, mut raw) = (Vec::with_capacity(rows * heads * dim), Vec::with_capacity(rows * dim));
        for row in projected.chunks_exact(width) {
            queries.extend_from_slice(&row[..heads * dim]);
            raw.extend_from_slice(&row[heads * dim..]);
        }
        rms(&mut queries, &ix.q_norm, cfg.eps);
        apply_rope(&mut queries, heads, dim, rope_dims, cfg.rope.neox, positions, table);
        let topk = budget / ratio;
        let mut selection: Vec<Option<Vec<u32>>> = vec![None; rows];
        let mut store = model.seqs.lock().map_err(|_| "state lock poisoned")?;
        for (start, end, first, position) in self.q4_runs(tables, lengths) {
            let seq = self.q4_seq(&mut store, first);
            if position == 0 {
                seq.index[i] = None;
            }
            let state = seq.index[i].get_or_insert_with(IndexState::default);
            if state.len != position {
                return Err(format!("token indexer state out of sync (holds {} tokens, got position {position})", state.len));
            }
            for t in start..end {
                state.raw.extend_from_slice(&raw[t * dim..(t + 1) * dim]);
                state.len += 1;
                let n = state.len;
                if n.is_multiple_of(ratio) {
                    let mut pooled = vec![0.0f32; dim];
                    for token in n - ratio..n {
                        pooled.iter_mut().zip(&state.raw[token * dim..(token + 1) * dim]).for_each(|(p, v)| *p += v);
                    }
                    pooled.iter_mut().for_each(|p| *p /= ratio as f32);
                    rms(&mut pooled, &ix.k_norm, cfg.eps);
                    apply_rope(&mut pooled, 1, dim, rope_dims, cfg.rope.neox, &[n - ratio], table);
                    state.blocks.extend_from_slice(&pooled);
                }
                let complete = n / ratio;
                if complete <= topk {
                    continue;
                }
                let scale = (dim as f32).sqrt().recip();
                let score: Vec<f32> = (0..complete)
                    .map(|b| {
                        let block = &state.blocks[b * dim..(b + 1) * dim];
                        (0..heads).map(|h| compute::dot(&queries[(t * heads + h) * dim..(t * heads + h + 1) * dim], block).max(0.0)).sum::<f32>() * scale
                    })
                    .collect();
                let mut order: Vec<usize> = (0..complete).collect();
                order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
                order.truncate(topk);
                order.sort_unstable();
                let mut tokens: Vec<u32> = order.iter().flat_map(|&b| (b * ratio..(b + 1) * ratio).map(|p| p as u32)).collect();
                tokens.extend((complete * ratio..n).map(|p| p as u32));
                selection[t] = Some(tokens);
            }
        }
        Ok(selection)
    }

    /// The rows of the n-gram table that every head of every token reads (`rows x heads` ids, token by token), hashed from the last `ngram` token ids of
    /// the sequence; advances the token history of PLE layer `i`. `history`: the state also keeps the convolution history (the host path; the device keeps
    /// that in its state buffers).
    #[allow(clippy::too_many_arguments)]
    fn q4_ple_ids(&self, model: &Qwen4Model, qc: &crate::config::Qwen4Config, i: usize, ple: &Ple, ids: &[i64], tables: &[i64], lengths: &[i64], history: bool) -> Result<Vec<i64>, String> {
        let width = qc.hc * self.cfg.hidden;
        let context = qc.ngram - 1;
        let heads = context * qc.heads_per_ngram;
        let eos = qc.eos;
        let hist_len = if history { (qc.ple_kernel - 1) * qc.ngram * width } else { 0 };
        let mut store = model.seqs.lock().map_err(|_| "state lock poisoned")?;
        let mut rows_of = vec![0i64; lengths.len() * heads];
        for (start, end, first, position) in self.q4_runs(tables, lengths) {
            let seq = self.q4_seq(&mut store, first);
            if position == 0 {
                seq.ple[i] = None;
            }
            let state = seq.ple[i].get_or_insert_with(|| PleState { len: 0, ids: vec![eos; context], hist: vec![0.0; hist_len] });
            if state.len != position {
                return Err(format!("n-gram state out of sync (holds {} tokens, got position {position})", state.len));
            }
            let mut tokens = state.ids.clone();
            tokens.extend_from_slice(&ids[start..end]);
            for t in 0..end - start {
                let at = context + t;
                // a token only sees the ones after the last end-of-sequence token
                let shifted: Vec<i64> = (0..qc.ngram)
                    .map(|s| if s == 0 || !tokens[at - s..at].contains(&eos) { tokens[at - s] } else { eos })
                    .collect();
                for n in 2..=qc.ngram {
                    let mut mixed = shifted[0].wrapping_mul(ple.multipliers[0]);
                    for (token, multiplier) in shifted.iter().zip(&ple.multipliers).take(n).skip(1) {
                        mixed ^= token.wrapping_mul(*multiplier);
                    }
                    for h in 0..qc.heads_per_ngram {
                        let head = (n - 2) * qc.heads_per_ngram + h;
                        rows_of[(start + t) * heads + head] = (mixed.rem_euclid(ple.sizes[head] as i64) as u64 + ple.offsets[head]) as i64;
                    }
                }
            }
            state.len += end - start;
            state.ids = tokens[tokens.len() - context..].to_vec();
        }
        Ok(rows_of)
    }

    /// The feature of the hashed n-gram embedding of PLE layer `i` for every stream (rows x hc x hidden).
    #[allow(clippy::too_many_arguments)]
    fn q4_ple(&self, model: &Qwen4Model, qc: &crate::config::Qwen4Config, i: usize, ple: &Ple, streams: &[f32], ids: &[i64], tables: &[i64], lengths: &[i64]) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let (rows, hidden, hc) = (lengths.len(), cfg.hidden, qc.hc);
        let width = hc * hidden;
        let heads = (qc.ngram - 1) * qc.heads_per_ngram;
        let head_dim = qc.ple_embed / heads;
        let dilation = qc.ngram;
        let runs = self.q4_runs(tables, lengths);
        // the hashed embedding rows of every token
        let row_ids = self.q4_ple_ids(model, qc, i, ple, ids, tables, lengths, true)?;
        let mut embeddings = vec![0.0f32; rows * qc.ple_embed];
        for (slot, &id) in row_ids.iter().enumerate() {
            let row = ple.embedding.row(id as usize)?;
            embeddings[slot * head_dim..(slot + 1) * head_dim].copy_from_slice(&row);
        }
        let mut store = model.seqs.lock().map_err(|_| "state lock poisoned")?;
        let key = group_rms(&ple.key.forward(&embeddings)?, &ple.norm_key, hidden, cfg.eps);
        let value = ple.value.forward(&embeddings)?;
        let query = group_rms(streams, &ple.norm_query, hidden, cfg.eps);
        let mut gated = vec![0.0f32; rows * width];
        for r in 0..rows {
            for n in 0..hc {
                let at = (r * hc + n) * hidden;
                let dot: f32 = key[at..at + hidden].iter().zip(&query[at..at + hidden]).map(|(a, b)| a * b).sum();
                let gate = dot / (hidden as f32).sqrt();
                let gate = gate.abs().max(1e-6).sqrt() * if gate == 0.0 { 0.0 } else { gate.signum() };
                let g = sigmoid(gate);
                for d in 0..hidden {
                    gated[at + d] = g * value[r * hidden + d];
                }
            }
        }
        let normed = group_rms(&gated, &ple.norm_conv, hidden, cfg.eps);
        let kernel = qc.ple_kernel;
        let mut out = gated;
        for &(start, end, first, position) in &runs {
            let seq = self.q4_seq(&mut store, first);
            let state = seq.ple[i].as_mut().ok_or("n-gram state vanished")?;
            // `state.len` was advanced above: the history belongs to the tokens before this run
            let _ = position;
            let n = end - start;
            let mut ext = state.hist.clone();
            ext.extend_from_slice(&normed[start * width..end * width]);
            for t in 0..n {
                for c in 0..width {
                    let mut sum = 0.0f32;
                    for j in 0..kernel {
                        sum += ple.conv[c * kernel + j] * ext[(t + j * dilation) * width + c];
                    }
                    out[(start + t) * width + c] += silu(sum);
                }
            }
            state.hist = ext[n * width..].to_vec();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes_follow_the_start_value() {
        assert_eq!(nth_prime_after(99, 1), 101);
        assert_eq!(nth_prime_after(99, 3), 107);
        assert_eq!(nth_prime_after(1, 1), 2);
    }

    #[test]
    fn multipliers_are_odd_and_bounded() {
        let m = layer_multipliers(1000, 3, 2, 1234);
        assert_eq!(m.len(), 3);
        assert!(m.iter().all(|&v| v % 2 == 1 && v > 0 && (v as u128) * 1000 <= i64::MAX as u128));
        assert_ne!(layer_multipliers(1000, 3, 0, 1234), m);
    }

    #[test]
    fn group_norm_scales_each_group_on_its_own() {
        let out = group_rms(&[3.0, 4.0, 0.0, 10.0], &[1.0; 4], 2, 0.0);
        let first = (12.5f32).sqrt().recip();
        assert!((out[0] - 3.0 * first).abs() < 1e-6 && (out[1] - 4.0 * first).abs() < 1e-6);
        assert!((out[3] - std::f32::consts::SQRT_2).abs() < 1e-5 && out[2] == 0.0);
    }

    #[test]
    fn injection_adds_the_weighted_branch_to_every_stream() {
        let mut streams = vec![1.0f32; 4];
        inject(&mut streams, &[2.0, 4.0], &[0.5, 2.0, 1.0, 0.0], 2, 1);
        assert_eq!(streams, [2.0, 5.0, 5.0, 1.0]);
    }
}
