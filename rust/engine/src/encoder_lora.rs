//! Task LoRA adapters of Jina embeddings v3: low-rank updates `W + scaling * B A` of the attention
//! and feed-forward projections plus the word and token-type embeddings, one set per task
//! (`retrieval.query`, `retrieval.passage`, `separation`, `classification`, `text-matching`).
//!
//! The adapters stay separate from the base weights and are applied to the activations
//! (`y += scaling * B (A x)`), so one loaded model serves every task. Tensors are written by
//! `rust/checkpoint/src/jina_xlmr.rs`.
use crate::compute::{self, Tensor};
use fllm_gguf_probe::MappedGguf;
use rayon::prelude::*;
use serde_json::Value;

/// Low-rank factors of one linear layer for every adapter: `a` is `[adapters][rank][inp]`,
/// `b` is `[adapters][out][rank]`.
pub(crate) struct LoraLinear {
    a: Vec<f32>,
    b: Vec<f32>,
    inp: usize,
    out: usize,
}

impl LoraLinear {
    fn load(model: &MappedGguf, base: &str, adapters: usize, rank: usize) -> Result<Self, String> {
        let a = Tensor::load(model, &format!("{base}.lora_a"))?;
        let b = Tensor::load(model, &format!("{base}.lora_b"))?;
        // GGUF shapes are reversed: a is [inp, rank, adapters], b is [rank, out, adapters].
        let well_formed = a.shape.len() == 3
            && b.shape.len() == 3
            && a.shape[1] == rank
            && a.shape[2] == adapters
            && b.shape[0] == rank
            && b.shape[2] == adapters;
        if !well_formed {
            return Err(format!("invalid LoRA shapes for {base}"));
        }
        Ok(Self {
            inp: a.shape[0],
            out: b.shape[1],
            a: a.data,
            b: b.data,
        })
    }

    /// Adds the update for `adapter` to the rows of `y` given the rows of `x`. `y` has `rows * out`
    /// values; with `split > 1` the output columns are scattered over `split` equally wide buffers
    /// (the fused QKV projection feeding separate Q, K and V buffers).
    pub(crate) fn add(
        &self,
        adapter: usize,
        rank: usize,
        scaling: f32,
        x: &[f32],
        ys: &mut [&mut Vec<f32>],
    ) {
        let split = ys.len();
        let width = self.out / split;
        let a = &self.a[adapter * rank * self.inp..(adapter + 1) * rank * self.inp];
        let b = &self.b[adapter * self.out * rank..(adapter + 1) * self.out * rank];
        let rows = x.len() / self.inp;
        // compute all deltas first (parallel over rows), then scatter
        let deltas: Vec<Vec<f32>> = (0..rows)
            .into_par_iter()
            .map(|i| {
                let xi = &x[i * self.inp..(i + 1) * self.inp];
                let t: Vec<f32> = (0..rank)
                    .map(|k| compute::dot(&a[k * self.inp..(k + 1) * self.inp], xi))
                    .collect();
                (0..self.out)
                    .map(|o| scaling * compute::dot(&b[o * rank..(o + 1) * rank], &t))
                    .collect()
            })
            .collect();
        for (i, delta) in deltas.iter().enumerate() {
            for (o, d) in delta.iter().enumerate() {
                ys[o / width][i * width + o % width] += d;
            }
        }
    }
}

struct LoraEmbedding {
    /// `[adapters][vocab][rank]`
    a: Vec<f32>,
    /// `[adapters][rank][hidden]`
    b: Vec<f32>,
    vocab: usize,
    hidden: usize,
}

impl LoraEmbedding {
    fn load(model: &MappedGguf, base: &str, adapters: usize, rank: usize) -> Result<Self, String> {
        let a = Tensor::load(model, &format!("{base}.lora_a"))?;
        let b = Tensor::load(model, &format!("{base}.lora_b"))?;
        // a is [rank, vocab, adapters], b is [hidden, rank, adapters] in GGUF order.
        let well_formed = a.shape.len() == 3
            && b.shape.len() == 3
            && a.shape[0] == rank
            && a.shape[2] == adapters
            && b.shape[1] == rank
            && b.shape[2] == adapters;
        if !well_formed {
            return Err(format!("invalid LoRA embedding shapes for {base}"));
        }
        Ok(Self {
            vocab: a.shape[1],
            hidden: b.shape[0],
            a: a.data,
            b: b.data,
        })
    }
    fn add_row(&self, adapter: usize, rank: usize, scaling: f32, id: usize, out: &mut [f32]) {
        if id >= self.vocab {
            return;
        }
        let a = &self.a[(adapter * self.vocab + id) * rank..(adapter * self.vocab + id + 1) * rank];
        let b = &self.b[adapter * rank * self.hidden..(adapter + 1) * rank * self.hidden];
        for (k, ak) in a.iter().enumerate() {
            let row = &b[k * self.hidden..(k + 1) * self.hidden];
            for (o, bv) in out.iter_mut().zip(row) {
                *o += scaling * ak * bv;
            }
        }
    }
}

pub(crate) struct LayerLora {
    pub(crate) qkv: LoraLinear,
    pub(crate) out: LoraLinear,
    pub(crate) up: LoraLinear,
    pub(crate) down: LoraLinear,
}

pub(crate) struct Lora {
    pub(crate) names: Vec<String>,
    pub(crate) instructions: Vec<String>,
    pub(crate) rank: usize,
    pub(crate) scaling: f32,
    word: LoraEmbedding,
    types: Option<LoraEmbedding>,
    pub(crate) layers: Vec<LayerLora>,
}

impl Lora {
    /// Reads the adapters when the checkpoint carries `bert.lora.names`.
    pub(crate) fn load(model: &MappedGguf, layer_count: usize) -> Result<Option<Self>, String> {
        let meta = &model.index().metadata;
        let Some(names) = meta.get("bert.lora.names").and_then(Value::as_array) else {
            return Ok(None);
        };
        let names = names
            .iter()
            .map(|v| v.as_str().map(str::to_owned).ok_or("invalid LoRA adapter name"))
            .collect::<Result<Vec<_>, _>>()?;
        let rank = meta
            .get("bert.lora.rank")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0 && n <= 256)
            .ok_or("invalid LoRA rank")?;
        let alpha = meta
            .get("bert.lora.alpha")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32;
        let instructions = match meta.get("bert.lora.instructions").and_then(Value::as_array) {
            Some(list) => list
                .iter()
                .map(|v| v.as_str().map(str::to_owned).ok_or("invalid LoRA instruction"))
                .collect::<Result<Vec<_>, _>>()?,
            None => vec![String::new(); names.len()],
        };
        if names.is_empty() || instructions.len() != names.len() {
            return Err("LoRA names and instructions disagree".into());
        }
        let n = names.len();
        let has_types = model
            .index()
            .tensors
            .iter()
            .any(|t| t.name == "token_types.lora_a");
        let mut layers = Vec::with_capacity(layer_count);
        for i in 0..layer_count {
            let p = format!("blk.{i}");
            layers.push(LayerLora {
                qkv: LoraLinear::load(model, &format!("{p}.attn_qkv"), n, rank)?,
                out: LoraLinear::load(model, &format!("{p}.attn_output"), n, rank)?,
                up: LoraLinear::load(model, &format!("{p}.ffn_up"), n, rank)?,
                down: LoraLinear::load(model, &format!("{p}.ffn_down"), n, rank)?,
            });
        }
        Ok(Some(Self {
            names,
            instructions,
            rank,
            scaling: alpha / rank as f32,
            word: LoraEmbedding::load(model, "token_embd", n, rank)?,
            types: if has_types {
                Some(LoraEmbedding::load(model, "token_types", n, rank)?)
            } else {
                None
            },
            layers,
        }))
    }

    pub(crate) fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// Adds the embedding updates of `adapter` for token `id` and segment `segment` to `row`.
    pub(crate) fn add_embedding(&self, adapter: usize, id: usize, segment: usize, row: &mut [f32]) {
        self.word.add_row(adapter, self.rank, self.scaling, id, row);
        if let Some(types) = &self.types {
            types.add_row(adapter, self.rank, self.scaling, segment, row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_update_matches_a_dense_product() {
        // two adapters, rank 1, 2 inputs, 4 outputs split over two buffers
        let lora = LoraLinear {
            a: vec![1.0, 2.0, 0.0, 1.0],
            b: vec![1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 3.0],
            inp: 2,
            out: 4,
        };
        let x = vec![3.0, 4.0];
        let mut first = vec![0.0; 2];
        let mut second = vec![0.0; 2];
        lora.add(1, 1, 0.5, &x, &mut [&mut first, &mut second]);
        // adapter 1: t = 0*3 + 1*4 = 4; b = [1, 1, 0, 3]; scaled 0.5 -> [2, 2, 0, 6]
        assert_eq!(first, vec![2.0, 2.0]);
        assert_eq!(second, vec![0.0, 6.0]);
    }

    #[test]
    fn embedding_update_is_a_row_times_b() {
        let emb = LoraEmbedding {
            a: vec![1.0, 2.0, 3.0, 4.0],
            b: vec![1.0, 0.0, 0.0, 1.0],
            vocab: 2,
            hidden: 2,
        };
        // adapter 0, rank 2, token 1: a = [3, 4], b = identity -> [3, 4] scaled by 0.5
        let mut row = vec![1.0, 1.0];
        emb.add_row(0, 2, 0.5, 1, &mut row);
        assert_eq!(row, vec![2.5, 3.0]);
    }
}
