//! What a model needs from a parallel layout: which sizes divide its dimensions, what each rank holds, and error messages that name
//! the tensor, its size and the sizes that would work.
use super::config::ParallelConfig;
use super::shard::{alignment, valid_parts};
use crate::config::{ModelConfig, QkNorm};
use fllm_gguf_probe::MappedGguf;
use std::ops::Range;

/// How the attention heads are dealt out to the `tp` ranks of a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnSplit {
    pub tp: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub heads_local: usize,
    pub kv_local: usize,
    /// Ranks that hold the same key / value head (more than one when there are fewer KV heads than ranks).
    pub kv_replicas: usize,
}

/// Tensor-parallel sizes up to `max` that divide the head counts.
pub fn attention_sizes(heads: usize, kv_heads: usize, max: usize) -> Vec<usize> {
    (1..=max).filter(|&t| heads.is_multiple_of(t) && (kv_heads.is_multiple_of(t) || t.is_multiple_of(kv_heads))).collect()
}

pub fn attention_split(heads: usize, kv_heads: usize, tp: usize) -> Result<AttnSplit, String> {
    let sizes = || attention_sizes(heads, kv_heads, 64).iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
    if tp == 0 || !heads.is_multiple_of(tp) {
        return Err(format!(
            "the model has {heads} attention heads, which {tp} tensor-parallel ranks cannot share equally; sizes that work: {}",
            sizes()
        ));
    }
    let (kv_local, kv_replicas) = if kv_heads.is_multiple_of(tp) {
        (kv_heads / tp, 1)
    } else if tp.is_multiple_of(kv_heads) {
        (1, tp / kv_heads)
    } else {
        return Err(format!(
            "the model has {kv_heads} key/value heads: {tp} tensor-parallel ranks neither divide them nor can replicate them evenly; sizes that work: {}",
            sizes()
        ));
    };
    Ok(AttnSplit { tp, heads, kv_heads, heads_local: heads / tp, kv_local, kv_replicas })
}

impl AttnSplit {
    pub fn q_range(&self, rank: usize) -> Range<usize> {
        rank * self.heads_local..(rank + 1) * self.heads_local
    }
    pub fn kv_range(&self, rank: usize) -> Range<usize> {
        let first = if self.kv_replicas > 1 { rank / self.kv_replicas } else { rank * self.kv_local };
        first..first + self.kv_local
    }
}

/// Rows of the (padded) vocabulary slice every rank computes; the last ranks may hold fewer real rows.
pub fn vocab_chunk(vocab: usize, tp: usize, row_align: usize) -> usize {
    vocab.div_ceil(tp).next_multiple_of(row_align.max(1))
}

/// The real rows `start..end` of the vocabulary that tensor rank `rank` computes.
pub fn vocab_range(vocab: usize, tp: usize, row_align: usize, rank: usize) -> Range<usize> {
    let chunk = vocab_chunk(vocab, tp, row_align);
    let start = (rank * chunk).min(vocab);
    start..(start + chunk).min(vocab)
}

/// Names the first reason a model cannot run with `par`, or `Ok` with notes worth printing.
#[derive(Debug, Default)]
pub struct ModelCheck {
    pub notes: Vec<String>,
}

fn dims(mapping: &MappedGguf, name: &str) -> Option<(u32, Vec<usize>)> {
    mapping.index().tensors.iter().find(|t| t.name == name).map(|t| (t.type_id, t.dimensions.iter().map(|&d| d as usize).collect()))
}

/// Check, without reading weights, that `par` fits the model: layer types that cannot be split, head counts, and for every
/// projection of every layer the size of the piece against its block / group / tile size.
pub fn check_model(mapping: &MappedGguf, cfg: &ModelConfig, par: &ParallelConfig, gpu: bool) -> Result<ModelCheck, String> {
    let mut problems: Vec<String> = Vec::new();
    let mut check = ModelCheck::default();
    let tp = par.tp;
    if par.pp > cfg.layers {
        problems.push(format!("--pipeline-parallel-size {} is larger than the {} layers of the model", par.pp, cfg.layers));
    }
    if tp > 1 {
        let unsupported = if cfg.mla.is_some() {
            Some("multi-head latent attention")
        } else if cfg.delta.is_some() {
            Some("Gated DeltaNet layers")
        } else if cfg.mamba.is_some() || cfg.mamba1.is_some() || cfg.rwkv.is_some() {
            Some("state-space layers")
        } else if cfg.shortconv.as_ref().is_some_and(|s| s.layers.iter().any(|&c| c)) {
            Some("short-convolution layers")
        } else if cfg.value_experts.is_some() {
            Some("value experts")
        } else if cfg.qk_norm == QkNorm::Full {
            Some("a query/key norm over all heads")
        } else {
            None
        };
        if let Some(what) = unsupported {
            problems.push(format!(
                "{} uses {what}, which cannot be split across tensor-parallel ranks; use --pipeline-parallel-size to spread its layers instead",
                cfg.arch
            ));
        } else if let Err(e) = attention_split(cfg.heads, cfg.kv_heads, tp) {
            problems.push(e);
        }
        if let Some(moe) = &cfg.moe {
            if moe.latent.is_some() {
                problems.push(format!("{}: latent mixture-of-experts layers are not supported with tensor parallelism", cfg.arch));
            }
            if par.expert_parallel && moe.experts < tp {
                problems.push(format!("--expert-parallel needs at least one expert per rank: the model has {} experts for {tp} ranks", moe.experts));
            }
        } else if par.expert_parallel {
            problems.push("--expert-parallel was given but the model has no experts".into());
        }
    }
    if tp > 1 && problems.is_empty() {
        let split_k = |problems: &mut Vec<String>, name: &str, k_values: usize, parts: usize| {
            let Some((kind, _)) = dims(mapping, name) else { return };
            match alignment(kind, gpu) {
                Ok(align) => {
                    if !k_values.is_multiple_of(parts) || !(k_values / parts).is_multiple_of(align.k) {
                        let working = valid_parts(k_values, align.k, 64);
                        problems.push(format!(
                            "{name}: input width {k_values} split {parts} ways is {} values per rank, which is not a multiple of the {}-value block/tile of this format; sizes that work for this tensor: {}",
                            k_values as f64 / parts as f64,
                            align.k,
                            if working.is_empty() { "none".to_owned() } else { working.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ") }
                        ));
                    }
                }
                Err(e) => problems.push(format!("{name}: {e}")),
            }
        };
        let split_n = |problems: &mut Vec<String>, name: &str, n_values: usize, parts: usize| {
            let Some((kind, _)) = dims(mapping, name) else { return };
            if let Ok(align) = alignment(kind, gpu) {
                if !n_values.is_multiple_of(parts) || !(n_values / parts).is_multiple_of(align.n) {
                    let working = valid_parts(n_values, align.n, 64);
                    problems.push(format!(
                        "{name}: {n_values} output rows split {parts} ways must be a multiple of {} per rank; sizes that work for this tensor: {}",
                        align.n,
                        if working.is_empty() { "none".to_owned() } else { working.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ") }
                    ));
                }
            }
        };
        let dv = cfg.value_dim;
        for layer in 0..cfg.layers {
            let p = format!("blk.{layer}");
            split_k(&mut problems, &format!("{p}.attn_output.weight"), cfg.heads * dv, tp);
            for name in ["attn_q", "ffn_up", "ffn_gate"] {
                let full = format!("{p}.{name}.weight");
                if let Some((_, d)) = dims(mapping, &full) {
                    split_n(&mut problems, &full, d.get(1).copied().unwrap_or(0), tp);
                }
            }
            if let Some((_, d)) = dims(mapping, &format!("{p}.ffn_down.weight")) {
                split_k(&mut problems, &format!("{p}.ffn_down.weight"), d[0], tp);
            }
            if !par.expert_parallel {
                if let Some((_, d)) = dims(mapping, &format!("{p}.ffn_down_exps.weight")) {
                    split_k(&mut problems, &format!("{p}.ffn_down_exps.weight"), d[0], tp);
                }
            }
            if let Some((_, d)) = dims(mapping, &format!("{p}.ffn_down_shexp.weight")) {
                // the shared expert falls back to rank 0 when it does not divide, which is correct but slower
                if !d[0].is_multiple_of(tp) {
                    check.notes.push(format!("{p}.ffn_down_shexp.weight: width {} is not divisible by {tp}; the shared expert runs on rank 0 only", d[0]));
                }
            }
            if problems.len() >= 6 {
                break;
            }
        }
    }
    if problems.is_empty() {
        Ok(check)
    } else {
        problems.dedup();
        Err(format!("this model cannot run with {}:\n  - {}", par.describe(), problems.join("\n  - ")))
    }
}

/// Bytes one rank holds on its device.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RankMemory {
    pub weight_bytes: f64,
    pub layers: usize,
    /// Key + value bytes of one cached token across the rank's layers.
    pub kv_bytes_per_token: f64,
}

/// Estimate of the largest rank (stage 0 is the heaviest when layers do not divide; the last stage adds the output head).
/// `kv_bytes_per_value` is 2.0 for the f16 cache. The embedding table stays in host memory (memory-mapped) and is not counted.
pub fn rank_memory(mapping: &MappedGguf, cfg: &ModelConfig, par: &ParallelConfig, kv_bytes_per_value: f64) -> RankMemory {
    let tp = par.tp as f64;
    let split = attention_split(cfg.heads, cfg.kv_heads, par.tp).ok();
    let kv_local = split.map_or(cfg.kv_heads, |s| s.kv_local) as f64;
    let mut worst = RankMemory::default();
    for stage in 0..par.pp {
        let range = par.stage_layers(cfg.layers, stage);
        let mut bytes = 0.0f64;
        for t in &mapping.index().tensors {
            let size = t.payload_len as f64;
            if let Some(rest) = t.name.strip_prefix("blk.") {
                let Some((index, name)) = rest.split_once('.') else { continue };
                let Ok(index) = index.parse::<usize>() else { continue };
                if !range.contains(&index) {
                    continue;
                }
                bytes += if name.starts_with("attn_k.") || name.starts_with("attn_v.") {
                    size / cfg.kv_heads as f64 * kv_local
                } else if name.starts_with("attn_q.")
                    || name.starts_with("attn_output.")
                    || name.starts_with("attn_qkv.")
                    || name.starts_with("attn_gate.")
                    || name.starts_with("ffn_gate.")
                    || name.starts_with("ffn_up.")
                    || name.starts_with("ffn_down.")
                    || name.contains("_shexp.")
                    || name.contains("_exps.")
                {
                    size / tp
                } else {
                    size
                };
            } else if (t.name == "output.weight" || (t.name == "token_embd.weight" && cfg.tied_embeddings)) && stage + 1 == par.pp {
                bytes += size / tp;
            }
        }
        let kv = range.len() as f64 * kv_local * (cfg.head_dim + cfg.value_dim) as f64 * kv_bytes_per_value;
        if bytes + kv > worst.weight_bytes + worst.kv_bytes_per_token {
            worst = RankMemory { weight_bytes: bytes, layers: range.len(), kv_bytes_per_token: kv };
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heads_are_dealt_out_and_kv_heads_replicated_when_ranks_outnumber_them() {
        // Llama-3 8B: 32 query heads, 8 KV heads
        let s = attention_split(32, 8, 4).unwrap();
        assert_eq!((s.heads_local, s.kv_local, s.kv_replicas), (8, 2, 1));
        assert_eq!(s.kv_range(3), 6..8);
        let s = attention_split(32, 8, 16).unwrap();
        assert_eq!((s.heads_local, s.kv_local, s.kv_replicas), (2, 1, 2));
        assert_eq!((s.kv_range(0), s.kv_range(1), s.kv_range(2), s.kv_range(15)), (0..1, 0..1, 1..2, 7..8));
        // multi-query attention: every rank keeps the single KV head
        let s = attention_split(16, 1, 8).unwrap();
        assert_eq!((s.heads_local, s.kv_local, s.kv_replicas), (2, 1, 8));
        assert_eq!(s.kv_range(5), 0..1);
        // every query head finds its group's KV head on its own rank
        for (heads, kv, tp) in [(32usize, 8usize, 16usize), (16, 1, 8), (32, 8, 2), (64, 8, 8)] {
            let s = attention_split(heads, kv, tp).unwrap();
            for rank in 0..tp {
                for h in s.q_range(rank) {
                    let wanted = h / (heads / kv);
                    assert!(s.kv_range(rank).contains(&wanted), "heads {heads} kv {kv} tp {tp} rank {rank} head {h}");
                }
            }
        }
    }

    #[test]
    fn impossible_head_splits_list_the_sizes_that_work() {
        let e = attention_split(14, 2, 4).unwrap_err();
        assert!(e.contains("14 attention heads") && e.contains("1, 2"), "{e}");
        let e = attention_split(24, 6, 4).unwrap_err();
        assert!(e.contains("key/value heads") && e.contains("1, 2, 3, 6"), "{e}");
    }

    #[test]
    fn the_vocabulary_is_padded_to_equal_chunks() {
        assert_eq!(vocab_chunk(32000, 4, 1), 8000);
        assert_eq!(vocab_chunk(151_936, 8, 64), 19_008);
        assert_eq!(vocab_range(10, 4, 1, 0), 0..3);
        assert_eq!(vocab_range(10, 4, 1, 3), 9..10);
        assert_eq!(vocab_range(9, 4, 1, 3), 9..9);
    }
}
