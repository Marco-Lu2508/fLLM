//! Decoder-architecture encoders: Llama / Qwen / Gemma style transformer stacks used as embedding,
//! reranking and sequence-classification models (EmbeddingGemma, llama-embed-nemotron,
//! Qwen3-Reranker, reward models). The stack is read from `blk.<i>.*` tensors written by
//! `rust/checkpoint/src/decoder_enc.rs`; attention is causal or bidirectional as the checkpoint says.
use crate::compute::{self, Linear, Tensor};
use fllm_gguf_probe::{MappedGguf, TensorBytes};
use rayon::prelude::*;
use serde_json::Value;

/// RMS norm over rows of `weight.len()` values; Gemma's `1 + weight` is folded in by the converter.
pub(crate) struct RmsNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    fn load(model: &MappedGguf, base: &str, width: usize, eps: f32) -> Result<Self, String> {
        let weight = Tensor::load(model, &format!("{base}.weight"))?;
        if weight.shape != [width] {
            return Err(format!("invalid norm dimensions: {base}"));
        }
        Ok(Self {
            weight: weight.data,
            eps,
        })
    }
    fn apply(&self, x: &mut [f32]) {
        let width = self.weight.len();
        x.par_chunks_mut(width).for_each(|row| {
            let mean = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
            let scale = 1.0 / (mean + self.eps).sqrt();
            for (v, w) in row.iter_mut().zip(&self.weight) {
                *v = *v * scale * w;
            }
        });
    }
}

#[derive(Clone, Copy)]
enum Activation {
    Silu,
    GeluTanh,
}

impl Activation {
    fn apply(self, v: f32) -> f32 {
        match self {
            Self::Silu => v / (1.0 + (-v).exp()),
            Self::GeluTanh => 0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044_715 * v * v * v)).tanh()),
        }
    }
}

struct Rope {
    half: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    fn new(len: usize, inv_freq: &[f64]) -> Self {
        let half = inv_freq.len();
        let mut cos = vec![0.0; len * half];
        let mut sin = vec![0.0; len * half];
        for p in 0..len {
            for (i, f) in inv_freq.iter().enumerate() {
                let (s, c) = (p as f64 * f).sin_cos();
                cos[p * half + i] = c as f32;
                sin[p * half + i] = s as f32;
            }
        }
        Self { half, cos, sin }
    }
    /// Rotate-half rotary embedding of `heads` heads of `2 * half` values per row.
    fn apply(&self, x: &mut [f32], heads: usize) {
        let dim = self.half * 2;
        let width = heads * dim;
        x.par_chunks_mut(width).enumerate().for_each(|(pos, row)| {
            let cos = &self.cos[pos * self.half..(pos + 1) * self.half];
            let sin = &self.sin[pos * self.half..(pos + 1) * self.half];
            for head in row.chunks_exact_mut(dim) {
                for i in 0..self.half {
                    let (x0, x1) = (head[i], head[i + self.half]);
                    head[i] = x0 * cos[i] - x1 * sin[i];
                    head[i + self.half] = x1 * cos[i] + x0 * sin[i];
                }
            }
        });
    }
}

struct Layer {
    attn_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    post_attn_norm: Option<RmsNorm>,
    ffn_norm: RmsNorm,
    post_ffn_norm: Option<RmsNorm>,
    gate: Linear,
    up: Linear,
    down: Linear,
    /// Sliding-window layer (Gemma 3 local attention) with its own rotary base.
    local: bool,
}

/// Token embedding rows read on demand from the memory-mapped tensor.
pub(crate) struct EmbedTable {
    bytes: TensorBytes,
    kind: u32,
    hidden: usize,
    vocab: usize,
    row_bytes: usize,
    scale: f32,
}

impl EmbedTable {
    pub(crate) fn lookup(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(ids.len() * self.hidden);
        for &id in ids {
            let id = id as usize;
            if id >= self.vocab {
                return Err("token id outside the embedding table".into());
            }
            let row = &self.bytes[id * self.row_bytes..(id + 1) * self.row_bytes];
            out.extend(compute::dequantize(row, self.kind, self.hidden)?);
        }
        if self.scale != 1.0 {
            for v in &mut out {
                *v *= self.scale;
            }
        }
        Ok(out)
    }
}

pub(crate) struct DecoderStack {
    layers: Vec<Layer>,
    output_norm: RmsNorm,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    activation: Activation,
    causal: bool,
    /// Sliding window of the local layers (half width for bidirectional attention).
    window: usize,
    rope_global: Rope,
    rope_local: Option<Rope>,
    pub(crate) embed: EmbedTable,
}

fn llama3_inv_freq(base: &mut [f64], meta: &serde_json::Map<String, Value>) -> Result<(), String> {
    let get = |key: &str| {
        meta.get(key)
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("missing {key}"))
    };
    let factor = get("bert.dec.rope_factor")?;
    let low = get("bert.dec.rope_low_freq_factor")?;
    let high = get("bert.dec.rope_high_freq_factor")?;
    let original = get("bert.dec.rope_original_context")?;
    if factor < 1.0 || low <= 0.0 || high <= low {
        return Err("invalid llama3 rotary scaling".into());
    }
    let low_wavelen = original / low;
    let high_wavelen = original / high;
    for f in base.iter_mut() {
        let wavelen = 2.0 * std::f64::consts::PI / *f;
        if wavelen > low_wavelen {
            *f /= factor;
        } else if wavelen >= high_wavelen {
            let smooth = (original / wavelen - low) / (high - low);
            *f = (1.0 - smooth) * *f / factor + smooth * *f;
        }
    }
    Ok(())
}

impl DecoderStack {
    pub(crate) fn load(
        model: &MappedGguf,
        hidden: usize,
        heads: usize,
        count: usize,
        inter: usize,
        context: usize,
    ) -> Result<Self, String> {
        let meta = &model.index().metadata;
        let number = |key: &str| {
            meta.get(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
        };
        let float = |key: &str, default: f64| meta.get(key).and_then(Value::as_f64).unwrap_or(default);
        let flag = |key: &str| meta.get(key).and_then(Value::as_bool).unwrap_or(false);
        let kv_heads = number("bert.dec.kv_head_count").unwrap_or(heads);
        let head_dim = number("bert.dec.head_dim").unwrap_or(hidden / heads);
        if heads == 0
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || head_dim == 0
            || !head_dim.is_multiple_of(2)
        {
            return Err("invalid attention geometry".into());
        }
        let eps = float("bert.dec.rms_eps", 1e-6) as f32;
        if !eps.is_finite() || eps <= 0.0 {
            return Err("invalid RMS norm epsilon".into());
        }
        let activation = match meta
            .get("bert.dec.activation")
            .and_then(Value::as_str)
            .unwrap_or("silu")
        {
            "silu" => Activation::Silu,
            "gelu_tanh" => Activation::GeluTanh,
            other => return Err(format!("unsupported decoder-encoder activation {other}")),
        };
        let post_norms = flag("bert.dec.post_norms");
        let qk_norm = flag("bert.dec.qk_norm");
        let local_flags = meta
            .get("bert.dec.layer_local")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| v.as_bool().unwrap_or(false)).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![false; count]);
        if local_flags.len() != count {
            return Err("layer_local flags disagree with block_count".into());
        }
        let qdim = heads * head_dim;
        let kvdim = kv_heads * head_dim;
        let mut layers = Vec::with_capacity(count);
        for (i, &local) in local_flags.iter().enumerate() {
            let p = format!("blk.{i}");
            let norm = |name: &str, width: usize| RmsNorm::load(model, &format!("{p}.{name}"), width, eps);
            layers.push(Layer {
                attn_norm: norm("attn_norm", hidden)?,
                q: Linear::load(model, &format!("{p}.attn_q"), qdim, hidden)?,
                k: Linear::load(model, &format!("{p}.attn_k"), kvdim, hidden)?,
                v: Linear::load(model, &format!("{p}.attn_v"), kvdim, hidden)?,
                out: Linear::load(model, &format!("{p}.attn_output"), hidden, qdim)?,
                q_norm: if qk_norm { Some(norm("attn_q_norm", head_dim)?) } else { None },
                k_norm: if qk_norm { Some(norm("attn_k_norm", head_dim)?) } else { None },
                post_attn_norm: if post_norms { Some(norm("post_attn_norm", hidden)?) } else { None },
                ffn_norm: norm("ffn_norm", hidden)?,
                post_ffn_norm: if post_norms { Some(norm("post_ffn_norm", hidden)?) } else { None },
                gate: Linear::load(model, &format!("{p}.ffn_gate"), inter, hidden)?,
                up: Linear::load(model, &format!("{p}.ffn_up"), inter, hidden)?,
                down: Linear::load(model, &format!("{p}.ffn_down"), hidden, inter)?,
                local,
            });
        }
        let theta = float("bert.dec.rope_theta", 10000.0);
        let theta_local = float("bert.dec.rope_theta_local", 0.0);
        if !theta.is_finite() || theta <= 1.0 {
            return Err("invalid rotary base".into());
        }
        let inv_freq = |theta: f64| -> Vec<f64> {
            (0..head_dim / 2)
                .map(|i| 1.0 / theta.powf(i as f64 * 2.0 / head_dim as f64))
                .collect()
        };
        let mut global = inv_freq(theta);
        match meta
            .get("bert.dec.rope_scaling")
            .and_then(Value::as_str)
            .unwrap_or("none")
        {
            "none" => {}
            "llama3" => llama3_inv_freq(&mut global, meta)?,
            other => return Err(format!("unsupported rotary scaling {other}")),
        }
        let rope_global = Rope::new(context, &global);
        let rope_local = if theta_local > 1.0 {
            Some(Rope::new(context, &inv_freq(theta_local)))
        } else {
            None
        };
        let causal = flag("bert.dec.causal");
        let window = number("bert.dec.sliding_window").unwrap_or(0);
        let attn_scale = float("bert.dec.attn_scale", 0.0) as f32;
        let info = model
            .index()
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .ok_or("missing tensor token_embd.weight")?;
        let vocab = info
            .dimensions
            .get(1)
            .and_then(|&n| usize::try_from(n).ok())
            .ok_or("invalid embedding shape")?;
        if info.dimensions.first().copied() != Some(hidden as u64) {
            return Err("embedding width disagrees with the hidden size".into());
        }
        let kind = info.type_id;
        let (block, width) = match kind {
            0 => (1, 4),
            1 | 30 => (1, 2),
            _ => return Err("decoder-encoder embeddings must be stored as F32, F16 or BF16".into()),
        };
        let row_bytes = hidden / block * width;
        let bytes = model.owned_tensor_bytes("token_embd.weight")?;
        if bytes.len() != vocab * row_bytes {
            return Err("invalid embedding tensor size".into());
        }
        Ok(Self {
            layers,
            output_norm: RmsNorm::load(model, "output_norm", hidden, eps)?,
            hidden,
            heads,
            kv_heads,
            head_dim,
            scale: if attn_scale > 0.0 { attn_scale } else { 1.0 / (head_dim as f32).sqrt() },
            activation,
            causal,
            window,
            rope_global,
            rope_local,
            embed: EmbedTable {
                bytes,
                kind,
                hidden,
                vocab,
                row_bytes,
                scale: float("bert.dec.embed_scale", 1.0) as f32,
            },
        })
    }

    pub(crate) fn vocab(&self) -> usize {
        self.embed.vocab
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], local: bool) -> Vec<f32> {
        let n = q.len() / (self.heads * self.head_dim);
        let qdim = self.heads * self.head_dim;
        let kvdim = self.kv_heads * self.head_dim;
        let group = self.heads / self.kv_heads;
        let hd = self.head_dim;
        let mut out = vec![0.0f32; n * qdim];
        out.par_chunks_mut(qdim).enumerate().for_each(|(i, row)| {
            let (mut lo, mut hi) = (0, if self.causal { i } else { n - 1 });
            if local && self.window > 0 {
                if self.causal {
                    // Causal sliding window: keys with i - j < window.
                    lo = i.saturating_sub(self.window - 1);
                } else {
                    // Bidirectional window: |i - j| <= window.
                    lo = i.saturating_sub(self.window);
                    hi = hi.min(i + self.window);
                }
            }
            let mut scores = vec![0.0f32; hi - lo + 1];
            for h in 0..self.heads {
                let kv = h / group;
                let qv = &q[i * qdim + h * hd..i * qdim + (h + 1) * hd];
                for (s, j) in scores.iter_mut().zip(lo..=hi) {
                    *s = compute::dot(qv, &k[j * kvdim + kv * hd..j * kvdim + (kv + 1) * hd]) * self.scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let dst = &mut row[h * hd..(h + 1) * hd];
                for (s, j) in scores.iter().zip(lo..=hi) {
                    let w = s / sum;
                    let src = &v[j * kvdim + kv * hd..j * kvdim + (kv + 1) * hd];
                    for d in 0..hd {
                        dst[d] += w * src[d];
                    }
                }
            }
        });
        out
    }

    /// Final-norm hidden states of the embedded tokens (`n * hidden` values).
        #[cfg(feature = "rust-gpu")]
    pub fn enable_gpu(&mut self, context: &std::sync::Arc<crate::gpu::GpuContext>) -> Result<(), String> {
        for layer in &mut self.layers {
            for linear in [
                &mut layer.q,
                &mut layer.k,
                &mut layer.v,
                &mut layer.out,
                &mut layer.gate,
                &mut layer.up,
                &mut layer.down,
            ] {
                linear.enable_gpu(context)?;
            }
        }
        Ok(())
    }

pub(crate) fn forward(&self, mut x: Vec<f32>) -> Result<Vec<f32>, String> {
        let n = x.len() / self.hidden;
        if n > self.rope_global.cos.len() / self.rope_global.half {
            return Err("encoder position exceeds rotary table".into());
        }
        for layer in &self.layers {
            let mut h = x.clone();
            layer.attn_norm.apply(&mut h);
            let mut q = layer.q.forward(&h)?;
            let mut k = layer.k.forward(&h)?;
            let v = layer.v.forward(&h)?;
            if let (Some(qn), Some(kn)) = (&layer.q_norm, &layer.k_norm) {
                qn.apply(&mut q);
                kn.apply(&mut k);
            }
            let rope = match (&self.rope_local, layer.local) {
                (Some(local), true) => local,
                _ => &self.rope_global,
            };
            rope.apply(&mut q, self.heads);
            rope.apply(&mut k, self.kv_heads);
            let a = self.attention(&q, &k, &v, layer.local);
            let mut o = layer.out.forward(&a)?;
            if let Some(norm) = &layer.post_attn_norm {
                norm.apply(&mut o);
            }
            compute::add(&mut x, &o);
            let mut m = x.clone();
            layer.ffn_norm.apply(&mut m);
            let gate = layer.gate.forward(&m)?;
            let up = layer.up.forward(&m)?;
            let act = self.activation;
            let f = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| act.apply(*g) * u)
                .collect::<Vec<_>>();
            let mut d = layer.down.forward(&f)?;
            if let Some(norm) = &layer.post_ffn_norm {
                norm.apply(&mut d);
            }
            compute::add(&mut x, &d);
            crate::encoder::apply_act_dtype(&mut x);
        }
        self.output_norm.apply(&mut x);
        Ok(x)
    }
}

/// Reads the optional `bert.pair_template.*` keys used to render reranker prompts.
pub(crate) struct PairTemplate {
    prefix: String,
    body: String,
    suffix: String,
    pub instruction: String,
}

impl PairTemplate {
    pub(crate) fn load(model: &MappedGguf) -> Option<Self> {
        let meta = &model.index().metadata;
        let text = |key: &str| meta.get(key).and_then(Value::as_str).map(str::to_owned);
        Some(Self {
            prefix: text("bert.pair_template.prefix")?,
            body: text("bert.pair_template.body")?,
            suffix: text("bert.pair_template.suffix")?,
            instruction: text("bert.pair_template.instruction").unwrap_or_default(),
        })
    }
    pub(crate) fn render(&self, query: &str, document: &str, instruction: Option<&str>) -> String {
        let body = self
            .body
            .replace("{instruction}", instruction.unwrap_or(&self.instruction))
            .replace("{query}", query)
            .replace("{document}", document);
        format!("{}{}{}", self.prefix, body, self.suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_rotates_pairs_and_keeps_position_zero() {
        let rope = Rope::new(4, &[1.0, 0.1]);
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let before = x.clone();
        rope.apply(&mut x, 1);
        // position 0 is the identity rotation; position 1 rotates (x0,x2) by 1 rad and (x1,x3) by 0.1 rad
        assert_eq!(&x[..4], &before[..4]);
        let (s, c) = 1.0f32.sin_cos();
        assert!((x[4] - (1.0 * c - 3.0 * s)).abs() < 1e-6);
        assert!((x[6] - (3.0 * c + 1.0 * s)).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_scales_rows() {
        let norm = RmsNorm {
            weight: vec![2.0, 2.0],
            eps: 0.0,
        };
        let mut x = vec![3.0, 4.0];
        norm.apply(&mut x);
        let rms = ((9.0f32 + 16.0) / 2.0).sqrt();
        assert!((x[0] - 2.0 * 3.0 / rms).abs() < 1e-6);
        assert!((x[1] - 2.0 * 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn pair_template_substitutes_fields() {
        let t = PairTemplate {
            prefix: "<p>".into(),
            body: "I:{instruction} Q:{query} D:{document}".into(),
            suffix: "<s>".into(),
            instruction: "default".into(),
        };
        assert_eq!(t.render("q", "d", None), "<p>I:default Q:q D:d<s>");
        assert_eq!(t.render("q", "d", Some("x")), "<p>I:x Q:q D:d<s>");
    }

    #[test]
    fn llama3_scaling_leaves_high_frequencies_and_divides_low_ones() {
        let mut meta = serde_json::Map::new();
        for (k, v) in [
            ("bert.dec.rope_factor", 8.0),
            ("bert.dec.rope_low_freq_factor", 1.0),
            ("bert.dec.rope_high_freq_factor", 4.0),
            ("bert.dec.rope_original_context", 8192.0),
        ] {
            meta.insert(k.into(), serde_json::json!(v));
        }
        // wavelength 2*pi (high frequency, kept), wavelength ~ 62832 (low frequency, divided by 8)
        let mut f = vec![1.0, 0.0001];
        llama3_inv_freq(&mut f, &meta).unwrap();
        assert_eq!(f[0], 1.0);
        assert!((f[1] - 0.0001 / 8.0).abs() < 1e-12);
    }
}
