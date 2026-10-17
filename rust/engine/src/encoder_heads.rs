//! Output heads of the encoder beyond pooled embeddings: SPLADE-style sparse vectors from an MLM
//! head, BGE-M3 lexical weights, ColBERT multi-vector outputs with MaxSim scoring and token
//! classification. The tensors and metadata are written by `rust/checkpoint/src/enc_heads.rs`.
use crate::compute::{self, LayerNorm, Linear};
use fllm_gguf_probe::MappedGguf;
use rayon::prelude::*;
use serde_json::Value;

/// A sparse vector as ascending `(token id, weight)` pairs without zero weights.
pub type SparseVector = Vec<(u32, f32)>;

/// Which side of a late-interaction pair a text belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InputKind {
    Query,
    Document,
}

impl InputKind {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "query" => Ok(Self::Query),
            "document" | "doc" | "passage" => Ok(Self::Document),
            _ => Err("input_type must be query or document".into()),
        }
    }
}

struct Mlm {
    dense: Linear,
    norm: LayerNorm,
    bias: Vec<f32>,
    /// Untied decoder matrix (`vocab` rows of `hidden`); `None` ties to the token embeddings.
    decoder: Option<Vec<f32>>,
}

#[derive(Clone, Copy)]
struct SparseConfig {
    log1p: bool,
    sum: bool,
}

struct ColbertMarkers {
    query: u32,
    doc: u32,
    mask: u32,
    query_length: usize,
    doc_length: usize,
    skip: Vec<u32>,
}

struct Colbert {
    linear: Linear,
    dim: usize,
    skip: usize,
    mean_score: bool,
    markers: Option<ColbertMarkers>,
}

pub(crate) struct Heads {
    /// Whether a pooled dense embedding is available.
    pub dense: bool,
    mlm: Option<Mlm>,
    splade: Option<SparseConfig>,
    sparse_linear: Option<Linear>,
    colbert: Option<Colbert>,
    token_cls: Option<Linear>,
    pub token_labels: Vec<String>,
    /// Tokens that never carry lexical weight (CLS, SEP, PAD, UNK).
    special: Vec<u32>,
}

fn tensor_exists(model: &MappedGguf, name: &str) -> bool {
    model.index().tensors.iter().any(|t| t.name == name)
}

impl Heads {
    pub(crate) fn load(
        model: &MappedGguf,
        hidden: usize,
        vocab: usize,
        eps: f32,
        special: Vec<u32>,
    ) -> Result<Self, String> {
        let meta = &model.index().metadata;
        let flag = |key: &str| meta.get(key).and_then(Value::as_bool);
        let text = |key: &str| meta.get(key).and_then(Value::as_str);
        let number = |key: &str| {
            meta.get(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
        };
        let mut mlm = None;
        let mut splade = None;
        if flag("bert.head.mlm") == Some(true) {
            let decoder = if tensor_exists(model, "mlm.decoder.weight") {
                let t = compute::Tensor::load(model, "mlm.decoder.weight")?;
                if t.shape != [hidden, vocab] {
                    return Err("invalid MLM decoder shape".into());
                }
                Some(t.data)
            } else {
                None
            };
            let bias = compute::Tensor::load(model, "mlm.bias")?;
            if bias.shape != [vocab] {
                return Err("invalid MLM bias shape".into());
            }
            mlm = Some(Mlm {
                dense: Linear::load(model, "mlm.dense", hidden, hidden)?,
                norm: LayerNorm::load(model, "mlm.norm", hidden, eps)?,
                bias: bias.data,
                decoder,
            });
            splade = Some(SparseConfig {
                log1p: text("bert.sparse.activation").unwrap_or("log1p_relu") == "log1p_relu",
                sum: text("bert.sparse.pooling").unwrap_or("max") == "sum",
            });
        }
        let sparse_linear = if flag("bert.head.sparse_linear") == Some(true) {
            Some(Linear::load(model, "sparse", 1, hidden)?)
        } else {
            None
        };
        let colbert = if flag("bert.head.colbert") == Some(true) {
            let dim = number("bert.colbert.dim")
                .filter(|&d| d > 0 && d <= 8192)
                .ok_or("invalid ColBERT dimension")?;
            let markers = match number("bert.colbert.query_marker") {
                Some(query) => Some(ColbertMarkers {
                    query: query as u32,
                    doc: number("bert.colbert.doc_marker").ok_or("missing ColBERT doc marker")? as u32,
                    mask: number("bert.colbert.mask_id").ok_or("missing ColBERT mask id")? as u32,
                    query_length: number("bert.colbert.query_length").unwrap_or(32).max(4),
                    doc_length: number("bert.colbert.doc_length").unwrap_or(180).max(4),
                    skip: meta
                        .get("bert.colbert.skip_ids")
                        .and_then(Value::as_array)
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()))
                                .collect()
                        })
                        .unwrap_or_default(),
                }),
                None => None,
            };
            Some(Colbert {
                linear: Linear::load(model, "colbert", dim, hidden)?,
                dim,
                skip: number("bert.colbert.skip").unwrap_or(0),
                mean_score: flag("bert.colbert.mean_score") == Some(true),
                markers,
            })
        } else {
            None
        };
        let (token_cls, token_labels) = match meta
            .get("bert.token_classifier.labels")
            .and_then(Value::as_array)
        {
            Some(labels) => {
                let names = labels
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| "token label must be a string".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if names.is_empty() || names.len() > 4096 {
                    return Err("invalid token classification label count".into());
                }
                (
                    Some(Linear::load(model, "token_cls", names.len(), hidden)?),
                    names,
                )
            }
            None => (None, Vec::new()),
        };
        Ok(Self {
            dense: flag("bert.head.dense") != Some(false),
            mlm,
            splade,
            sparse_linear,
            colbert,
            token_cls,
            token_labels,
            special,
        })
    }

    pub(crate) fn has_sparse(&self) -> bool {
        self.splade.is_some() || self.sparse_linear.is_some()
    }
    pub(crate) fn has_sparse_mlm(&self) -> bool {
        self.splade.is_some()
    }
    pub(crate) fn has_multi_vector(&self) -> bool {
        self.colbert.is_some()
    }
    pub(crate) fn has_token_classifier(&self) -> bool {
        self.token_cls.is_some()
    }
    pub(crate) fn multi_vector_dim(&self) -> Option<usize> {
        self.colbert.as_ref().map(|c| c.dim)
    }
    /// BGE-M3 divides the MaxSim sum by the number of query vectors.
    pub(crate) fn mean_maxsim(&self) -> bool {
        self.colbert.as_ref().is_some_and(|c| c.mean_score)
    }

    /// Token ids the ColBERT head feeds the encoder: markers after CLS and query padding with
    /// mask tokens when the checkpoint defines them, otherwise the ordinary ids.
    pub(crate) fn colbert_ids(&self, ids: &[u32], kind: InputKind) -> Vec<u32> {
        let Some(markers) = self.colbert.as_ref().and_then(|c| c.markers.as_ref()) else {
            return ids.to_vec();
        };
        if ids.len() < 2 {
            return ids.to_vec();
        }
        let (cls, sep) = (ids[0], ids[ids.len() - 1]);
        let body = &ids[1..ids.len() - 1];
        let (marker, limit) = match kind {
            InputKind::Query => (markers.query, markers.query_length),
            InputKind::Document => (markers.doc, markers.doc_length),
        };
        let keep = body.len().min(limit.saturating_sub(3));
        let mut out = Vec::with_capacity(limit.max(keep + 3));
        out.push(cls);
        out.push(marker);
        out.extend_from_slice(&body[..keep]);
        out.push(sep);
        if kind == InputKind::Query {
            while out.len() < limit {
                out.push(markers.mask);
            }
        }
        out
    }

    /// SPLADE vector from final token states: transform, tied decoder, activation, pooling over tokens.
    pub(crate) fn splade(
        &self,
        x: &[f32],
        hidden: usize,
        vocab: usize,
        tied: &[f32],
    ) -> Result<SparseVector, String> {
        let (mlm, config) = match (&self.mlm, self.splade) {
            (Some(m), Some(c)) => (m, c),
            _ => return Err("model has no SPLADE MLM head".into()),
        };
        let rows = x.len() / hidden;
        let mut t = mlm.dense.forward(x)?;
        for v in &mut t {
            *v = compute::gelu(*v);
        }
        mlm.norm.apply(&mut t);
        let decoder = mlm.decoder.as_deref().unwrap_or(tied);
        if decoder.len() != vocab * hidden {
            return Err("MLM decoder does not match the vocabulary".into());
        }
        let mut weights = vec![0.0f32; vocab];
        weights.par_iter_mut().enumerate().for_each(|(v, slot)| {
            let column = &decoder[v * hidden..(v + 1) * hidden];
            let mut acc = 0.0f32;
            for r in 0..rows {
                let logit = compute::dot(&t[r * hidden..(r + 1) * hidden], column) + mlm.bias[v];
                let relu = logit.max(0.0);
                let w = if config.log1p { relu.ln_1p() } else { relu };
                if config.sum {
                    acc += w;
                } else {
                    acc = acc.max(w);
                }
            }
            *slot = acc;
        });
        Ok(weights
            .into_iter()
            .enumerate()
            .filter(|(_, w)| *w > 0.0)
            .map(|(i, w)| (i as u32, w))
            .collect())
    }

    /// BGE-M3 lexical weights: `relu(sparse_linear(h))` per token, maximum per distinct token id,
    /// without the special tokens.
    pub(crate) fn lexical(&self, x: &[f32], ids: &[u32]) -> Result<SparseVector, String> {
        let linear = self
            .sparse_linear
            .as_ref()
            .ok_or("model has no lexical weight head")?;
        let weights = linear.forward(x)?;
        let mut best = std::collections::BTreeMap::<u32, f32>::new();
        for (&id, &w) in ids.iter().zip(weights.iter()) {
            let w = w.max(0.0);
            if w > 0.0 && !self.special.contains(&id) {
                let slot = best.entry(id).or_insert(0.0);
                *slot = slot.max(w);
            }
        }
        Ok(best.into_iter().collect())
    }

    /// Normalised per-token vectors of the ColBERT head. `ids` are the ids the states were computed for.
    pub(crate) fn multi_vector(
        &self,
        x: &[f32],
        hidden: usize,
        ids: &[u32],
        kind: InputKind,
    ) -> Result<Vec<Vec<f32>>, String> {
        let colbert = self.colbert.as_ref().ok_or("model has no multi-vector head")?;
        let rows = x.len() / hidden;
        if colbert.skip >= rows {
            return Err("input is too short for the multi-vector head".into());
        }
        let projected = colbert.linear.forward(&x[colbert.skip * hidden..])?;
        let mut out = Vec::new();
        for (row, vector) in projected.chunks_exact(colbert.dim).enumerate() {
            let id = ids.get(row + colbert.skip).copied();
            if kind == InputKind::Document {
                if let (Some(markers), Some(id)) = (colbert.markers.as_ref(), id) {
                    if markers.skip.contains(&id) {
                        continue;
                    }
                }
            }
            let mut v = vector.to_vec();
            compute::normalize(&mut v)?;
            out.push(v);
        }
        Ok(out)
    }

    /// Per-token class probabilities (softmax over the labels).
    pub(crate) fn token_probabilities(&self, x: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        let head = self
            .token_cls
            .as_ref()
            .ok_or("model has no token classification head")?;
        let labels = self.token_labels.len();
        let logits = head.forward(x)?;
        logits
            .chunks_exact(labels)
            .map(|row| {
                if row.iter().any(|v| !v.is_finite()) {
                    return Err("invalid token classifier logits".to_string());
                }
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut p = row.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
                let sum: f32 = p.iter().sum();
                for v in &mut p {
                    *v /= sum;
                }
                Ok(p)
            })
            .collect()
    }
}

/// Late-interaction score: the sum over query vectors of the best dot product with a document vector.
pub fn max_sim(query: &[Vec<f32>], document: &[Vec<f32>], mean: bool) -> f32 {
    if query.is_empty() {
        return 0.0;
    }
    let sum: f32 = query
        .iter()
        .map(|q| {
            document
                .iter()
                .map(|d| compute::dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .sum();
    if mean {
        sum / query.len() as f32
    } else {
        sum
    }
}

/// Dot product over shared token ids of two ascending sparse vectors.
pub fn sparse_dot(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let (mut i, mut j, mut sum) = (0, 0, 0.0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                sum += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_sim_sums_best_matches() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let d = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
        assert_eq!(max_sim(&q, &d, false), 1.5);
        assert_eq!(max_sim(&q, &d, true), 0.75);
        assert_eq!(max_sim(&[], &d, false), 0.0);
        assert_eq!(max_sim(&q, &[], false), 0.0);
    }

    #[test]
    fn sparse_dot_matches_shared_ids() {
        let a = [(1, 2.0), (5, 1.0), (9, 3.0)];
        let b = [(5, 4.0), (9, 0.5), (11, 1.0)];
        assert_eq!(sparse_dot(&a, &b), 5.5);
        assert_eq!(sparse_dot(&a, &[]), 0.0);
    }

    #[test]
    fn input_kind_names() {
        assert_eq!(InputKind::parse("query").unwrap(), InputKind::Query);
        assert_eq!(InputKind::parse("passage").unwrap(), InputKind::Document);
        assert!(InputKind::parse("x").is_err());
    }
}
