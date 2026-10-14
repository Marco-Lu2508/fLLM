//! BERT inference implemented with fLLM's Rust tensor operations.
use crate::compute::{self, LayerNorm, Linear, Tensor};
use fllm_gguf_probe::MappedGguf;
use serde_json::Value;
use tokenizers::{
    decoders::wordpiece::WordPiece as WordPieceDecoder, models::wordpiece::WordPiece,
    normalizers::bert::BertNormalizer, pre_tokenizers::bert::BertPreTokenizer, Tokenizer,
};

enum EncoderTokenizer {
    WordPiece(Tokenizer),
    Other(crate::tokenizer::TokenizerBundle),
}

struct BertLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    attn_norm: LayerNorm,
    up: Linear,
    gate: Option<Linear>,
    down: Linear,
    out_norm: LayerNorm,
}

/// Bias-free layer norm (ModernBERT stores no norm bias anywhere).
struct PlainNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl PlainNorm {
    fn load(model: &MappedGguf, base: &str, hidden: usize, eps: f32) -> Result<Self, String> {
        if model
            .index()
            .tensors
            .iter()
            .any(|t| t.name == format!("{base}.bias"))
        {
            return Err(format!("{base} must not carry a norm bias"));
        }
        let weight = Tensor::load(model, &format!("{base}.weight"))?;
        if weight.shape != vec![hidden] {
            return Err(format!("invalid {base} dimensions"));
        }
        if weight.data.iter().any(|v| !v.is_finite()) {
            return Err(format!("non-finite {base} weight"));
        }
        Ok(Self {
            weight: weight.data,
            eps,
        })
    }

    fn has_tensor(model: &MappedGguf, base: &str) -> bool {
        model
            .index()
            .tensors
            .iter()
            .any(|t| t.name == format!("{base}.weight"))
    }

    fn apply(&self, x: &mut [f32]) {
        use rayon::prelude::*;
        let hidden = self.weight.len();
        x.par_chunks_mut(hidden).for_each(|row| {
            let mean = row.iter().sum::<f32>() / hidden as f32;
            let var = row.iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / hidden as f32;
            let scale = 1.0 / (var + self.eps).sqrt();
            for (value, weight) in row.iter_mut().zip(self.weight.iter()) {
                *value = (*value - mean) * scale * weight;
            }
        });
    }
}

/// Either biased (BERT family) or bias-free (ModernBERT) norms.
enum EncoderNorm {
    Biased(LayerNorm),
    Plain(PlainNorm),
}

impl EncoderNorm {
    fn apply(&self, x: &mut [f32]) {
        match self {
            EncoderNorm::Biased(norm) => norm.apply(x),
            EncoderNorm::Plain(norm) => norm.apply(x),
        }
    }
}

/// Rotary tables for one ModernBERT attention flavour (global or local theta).
struct ModernRopeTables {
    len: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

fn build_rope_tables(len: usize, dim: usize, theta: f32) -> ModernRopeTables {
    let half = dim / 2;
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        inv_freq.push(1.0 / theta.powf(i as f32 * 2.0 / dim as f32));
    }
    let mut cos = vec![0.0; len * dim];
    let mut sin = vec![0.0; len * dim];
    for p in 0..len {
        for i in 0..half {
            let (s, c) = (p as f32 * inv_freq[i]).sin_cos();
            cos[p * dim + i] = c;
            cos[p * dim + i + half] = c;
            sin[p * dim + i] = s;
            sin[p * dim + i + half] = s;
        }
    }
    ModernRopeTables { len, cos, sin }
}

struct ModernRope {
    global: ModernRopeTables,
    local: ModernRopeTables,
}

fn apply_modern_rope(
    x: &mut [f32],
    positions: &[usize],
    heads: usize,
    head_dim: usize,
    tables: &ModernRopeTables,
) {
    use rayon::prelude::*;
    let hidden = heads * head_dim;
    let half = head_dim / 2;
    x.par_chunks_mut(hidden)
        .enumerate()
        .for_each(|(row, values)| {
            let base = positions[row] * head_dim;
            for head in 0..heads {
                let offset = head * head_dim;
                for i in 0..half {
                    let c = tables.cos[base + i];
                    let s = tables.sin[base + i];
                    let x0 = values[offset + i];
                    let x1 = values[offset + i + half];
                    values[offset + i] = x0 * c - x1 * s;
                    values[offset + i + half] = x0 * s + x1 * c;
                }
            }
        });
}

/// Bidirectional attention with an optional sliding window over the original
/// (pre-filter) positions; mirrors [`compute::attention`] plus masking.
#[allow(clippy::too_many_arguments)]
fn modern_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    positions: &[usize],
    hidden: usize,
    heads: usize,
    global: bool,
    half: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let count = positions.len();
    let head_dim = hidden / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    (0..count)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![0.0; hidden];
            for head in 0..heads {
                let base = head * head_dim;
                let mut scores = vec![f32::NEG_INFINITY; count];
                for j in 0..count {
                    if !global && positions[i].abs_diff(positions[j]) > half {
                        continue;
                    }
                    scores[j] = compute::dot(
                        &q[i * hidden + base..i * hidden + base + head_dim],
                        &k[j * hidden + base..j * hidden + base + head_dim],
                    ) * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let out = &mut row[base..base + head_dim];
                for j in 0..count {
                    let weight = scores[j] / sum;
                    let source = &v[j * hidden + base..j * hidden + base + head_dim];
                    for d in 0..head_dim {
                        out[d] += weight * source[d];
                    }
                }
            }
            row
        })
        .collect::<Vec<_>>()
        .concat()
}

/// tanh-approximated GELU used by the ModernBERT GeGLU feed-forward block.
fn gelu_tanh(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.7978846 * (value + 0.044715 * value.powi(3))).tanh())
}

struct ModernBertLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    attn_norm: Option<PlainNorm>,
    up: Linear,
    gate: Linear,
    down: Linear,
    ffn_norm: PlainNorm,
    global: bool,
}
pub struct BertEncoder {
    tokenizer: EncoderTokenizer,
    token: Tensor,
    position: Option<Tensor>,
    types: Option<Tensor>,
    projection: Option<Linear>,
    classifier: Option<Linear>,
    classifier_dense: Option<Linear>,
    roberta: bool,
    modern: bool,
    nomic: bool,
    pub labels: Vec<String>,
    position_offset: usize,
    pad: Option<u32>,
    activation: String,
    embedding_norm: EncoderNorm,
    layers: Vec<BertLayer>,
    modern_layers: Vec<ModernBertLayer>,
    rope: Option<ModernRope>,
    rope_single: Option<ModernRopeTables>,
    sliding_half: usize,
    output_norm: Option<PlainNorm>,
    classifier_norm: Option<PlainNorm>,
    classifier_pooling: String,
    pub hidden: usize,
    pub context: usize,
    pub vocab: usize,
    heads: usize,
    cls: u32,
    sep: u32,
    pool: rayon::ThreadPool,
    pub default_pooling: String,
}
impl BertEncoder {
    pub fn load(path: &str, threads: usize, pooling: Option<String>) -> Result<Self, String> {
        let model = MappedGguf::open(path).map_err(|e| e.to_string())?;
        let meta = &model.index().metadata;
        if meta.get("general.architecture").and_then(Value::as_str) != Some("bert") {
            return Err("this independent encoder supports the BERT GGUF architecture".into());
        }
        if meta.get("bert.attention.causal").and_then(Value::as_bool) == Some(true) {
            return Err("BERT encoder requires bidirectional attention".into());
        }
        let integer = |key: &str| -> Result<usize, String> {
            meta.get(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("invalid/missing {key}"))
        };
        let hidden = integer("bert.embedding_length")?;
        let context = integer("bert.context_length")?;
        let heads = integer("bert.attention.head_count")?;
        let intermediate = integer("bert.feed_forward_length")?;
        let count = integer("bert.block_count")?;
        if hidden % heads != 0 || hidden > 8192 || count > 128 || context > 8192 {
            return Err("unsupported BERT dimensions".into());
        }
        let eps = meta
            .get("bert.attention.layer_norm_epsilon")
            .and_then(Value::as_f64)
            .unwrap_or(1e-12) as f32;
        if !eps.is_finite() || eps <= 0.0 {
            return Err("invalid BERT layer norm epsilon".into());
        }
        let tokens = meta
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or("missing WordPiece vocabulary")?;
        let mut vocabulary = ahash::AHashMap::new();
        for (i, token) in tokens.iter().enumerate() {
            vocabulary.insert(
                token.as_str().ok_or("invalid WordPiece token")?.to_owned(),
                i as u32,
            );
        }
        let special_id = |key: &str, fallback: &str| -> Result<u32, String> {
            meta.get(key)
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
                .or_else(|| vocabulary.get(fallback).copied())
                .filter(|&n| (n as usize) < tokens.len())
                .ok_or_else(|| format!("invalid/missing {key}"))
        };
        let wordpiece = meta
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .unwrap_or("bert")
            == "bert";
        let cls = special_id(
            "tokenizer.ggml.cls_token_id",
            if wordpiece { "[CLS]" } else { "<s>" },
        )?;
        let sep = special_id(
            "tokenizer.ggml.separator_token_id",
            if wordpiece { "[SEP]" } else { "</s>" },
        )?;
        let pad = meta
            .get("tokenizer.ggml.padding_token_id")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok());
        if pad.is_some_and(|n| n as usize >= tokens.len()) {
            return Err("invalid padding token ID".into());
        }
        let lowercase = meta
            .get("tokenizer.ggml.do_lower_case")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let tokenizer = if !wordpiece {
            EncoderTokenizer::Other(crate::tokenizer::TokenizerBundle::from_gguf(&model)?)
        } else if let Some(raw) = meta
            .get("tokenizer.huggingface.json")
            .and_then(Value::as_str)
        {
            let tokenizer = Tokenizer::from_bytes(raw.as_bytes()).map_err(|e| e.to_string())?;
            for (token, id) in tokenizer.get_vocab(true) {
                if tokens.get(id as usize).and_then(Value::as_str) != Some(token.as_str()) {
                    return Err("HF tokenizer vocabulary does not match GGUF".into());
                }
            }
            EncoderTokenizer::WordPiece(tokenizer)
        } else {
            let unk = special_id("tokenizer.ggml.unknown_token_id", "[UNK]")?;
            let wp = WordPiece::builder()
                .vocab(vocabulary)
                .unk_token(
                    tokens[unk as usize]
                        .as_str()
                        .ok_or("invalid unknown token")?
                        .into(),
                )
                .build()
                .map_err(|e| e.to_string())?;
            let mut tokenizer = Tokenizer::new(wp);
            tokenizer
                .with_normalizer(Some(BertNormalizer::new(
                    true,
                    true,
                    Some(lowercase),
                    lowercase,
                )))
                .map_err(|e| e.to_string())?;
            tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
            tokenizer.with_decoder(Some(WordPieceDecoder::new("##".into(), true)));
            EncoderTokenizer::WordPiece(tokenizer)
        };
        let modern = meta
            .get("bert.model_type")
            .and_then(Value::as_str)
            .is_some_and(|v| v == "modernbert");
        let nomic = meta
            .get("bert.model_type")
            .and_then(Value::as_str)
            .is_some_and(|v| v == "nomic_bert");
        let token = Tensor::load(&model, "token_embd.weight")?;
        let embedding = token
            .shape
            .first()
            .copied()
            .filter(|&n| n > 0 && n <= 8192)
            .ok_or("invalid embedding dimensions")?;
        token.matrix(tokens.len(), embedding, "token embeddings")?;
        if (modern || nomic)
            && model
                .index()
                .tensors
                .iter()
                .any(|t| t.name == "position_embd.weight")
        {
            return Err(if modern {
                "modernbert must not carry learned position embeddings".into()
            } else {
                "nomic must not carry learned position embeddings".into()
            });
        }
        let position = if modern || nomic {
            None
        } else {
            Some(Tensor::load(&model, "position_embd.weight")?)
        };
        let position_offset = match meta.get("bert.position_offset") {
            Some(v) => v
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or("invalid position offset")?,
            None => 0,
        };
        if let Some(position) = &position {
            if position.shape.len() != 2
                || position.shape[0] != embedding
                || position_offset
                    .checked_add(context)
                    .is_none_or(|n| n > position.shape[1])
            {
                return Err("invalid position embeddings or offset".into());
            }
        }
        if modern
            && model
                .index()
                .tensors
                .iter()
                .any(|t| t.name == "token_types.weight")
        {
            return Err("modernbert must not carry token type embeddings".into());
        }
        let types = if !modern
            && model
                .index()
                .tensors
                .iter()
                .any(|t| t.name == "token_types.weight")
        {
            let types = Tensor::load(&model, "token_types.weight")?;
            if types.shape.len() != 2 || types.shape[0] != embedding || types.shape[1] < 1 {
                return Err("invalid token type embeddings".into());
            }
            Some(types)
        } else {
            None
        };
        if modern && embedding != hidden {
            return Err("modernbert requires matching embedding and hidden sizes".into());
        }
        let projection = if embedding != hidden {
            Some(Linear::load(
                &model,
                "token_embd_project",
                hidden,
                embedding,
            )?)
        } else {
            None
        };
        let roberta = meta
            .get("bert.model_type")
            .and_then(Value::as_str)
            .is_some_and(|v| matches!(v, "roberta" | "xlm-roberta"))
            || position_offset > 0;
        let num_labels = meta
            .get("bert.classifier.label_count")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        if num_labels > 4096 {
            return Err("classifier has too many labels".into());
        }
        let classifier = if num_labels > 0 {
            Some(Linear::load(&model, "classifier", num_labels, hidden)?)
        } else {
            None
        };
        let classifier_dense = if num_labels > 0 {
            Some(Linear::load(
                &model,
                if roberta || modern {
                    "classifier_dense"
                } else {
                    "pooler"
                },
                hidden,
                hidden,
            )?)
        } else {
            None
        };
        let (classifier_norm, classifier_pooling) = if modern && num_labels > 0 {
            let norm = Some(PlainNorm::load(&model, "classifier_norm", hidden, eps)?);
            let pooling = meta
                .get("bert.classifier.pooling")
                .and_then(Value::as_str)
                .unwrap_or("cls")
                .to_owned();
            if !matches!(pooling.as_str(), "cls" | "mean") {
                return Err("modernbert classifier pooling must be cls or mean".into());
            }
            (norm, pooling)
        } else {
            (None, "cls".to_owned())
        };
        let labels = if num_labels > 0 {
            let labels = meta
                .get("bert.classifier.labels")
                .and_then(Value::as_array)
                .ok_or("missing classifier labels")?;
            if labels.len() != num_labels {
                return Err("classifier labels disagree with label_count".into());
            }
            labels
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| "classifier label must be a string".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let embedding_norm = if modern {
            EncoderNorm::Plain(PlainNorm::load(&model, "token_embd_norm", embedding, eps)?)
        } else {
            EncoderNorm::Biased(LayerNorm::load(&model, "token_embd_norm", embedding, eps)?)
        };
        let activation = meta
            .get("bert.activation")
            .and_then(Value::as_str)
            .unwrap_or("gelu")
            .to_owned();
        if !matches!(
            activation.as_str(),
            "gelu" | "gelu_new" | "gelu_fast" | "relu" | "silu" | "swish" | "geglu" | "swiglu"
        ) {
            return Err(format!("unsupported encoder activation {activation}"));
        }
        let mut layers = Vec::new();
        let mut modern_layers = Vec::new();
        if modern {
            let layer_global = match meta.get("bert.layer_global").and_then(Value::as_array) {
                Some(flags) => flags
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .map(|n| n != 0)
                            .ok_or_else(|| "invalid layer_global flags".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                None => (0..count).map(|i| i.is_multiple_of(3)).collect(),
            };
            if layer_global.len() != count {
                return Err("layer_global flags disagree with block_count".into());
            }
            for (i, g) in layer_global.iter().enumerate().take(count) {
                let prefix = format!("blk.{i}");
                modern_layers.push(ModernBertLayer {
                    q: Linear::load_rows(
                        &model,
                        &format!("{prefix}.attn_qkv.weight"),
                        None,
                        0,
                        hidden,
                        hidden,
                    )?,
                    k: Linear::load_rows(
                        &model,
                        &format!("{prefix}.attn_qkv.weight"),
                        None,
                        hidden,
                        hidden,
                        hidden,
                    )?,
                    v: Linear::load_rows(
                        &model,
                        &format!("{prefix}.attn_qkv.weight"),
                        None,
                        2 * hidden,
                        hidden,
                        hidden,
                    )?,
                    out: Linear::load(&model, &format!("{prefix}.attn_output"), hidden, hidden)?,
                    attn_norm: if PlainNorm::has_tensor(
                        &model,
                        &format!("{prefix}.attn_output_norm"),
                    ) {
                        Some(PlainNorm::load(
                            &model,
                            &format!("{prefix}.attn_output_norm"),
                            hidden,
                            eps,
                        )?)
                    } else {
                        None
                    },
                    up: Linear::load_rows(
                        &model,
                        &format!("{prefix}.ffn_up.weight"),
                        None,
                        0,
                        intermediate,
                        hidden,
                    )?,
                    gate: Linear::load_rows(
                        &model,
                        &format!("{prefix}.ffn_up.weight"),
                        None,
                        intermediate,
                        intermediate,
                        hidden,
                    )?,
                    down: Linear::load(
                        &model,
                        &format!("{prefix}.ffn_down"),
                        hidden,
                        intermediate,
                    )?,
                    ffn_norm: PlainNorm::load(
                        &model,
                        &format!("{prefix}.ffn_norm"),
                        hidden,
                        eps,
                    )?,
                    global: *g,
                });
            }
        } else {
            for i in 0..count {
                let prefix = format!("blk.{i}");
                let gate_name = format!("{prefix}.ffn_gate.weight");
                let has_gate = model
                    .index()
                    .tensors
                    .iter()
                    .any(|t| t.name == gate_name);
                if has_gate != nomic {
                    return Err(if nomic {
                        format!("nomic is missing {gate_name}")
                    } else {
                        format!("{prefix} must not carry a gated feed-forward tensor")
                    });
                }
                // Nomic legacy checkpoints fuse QKV (as ModernBERT does);
                // freshly built ones carry separate projections.
                let qkv_name = format!("{prefix}.attn_qkv.weight");
                let fused = model
                    .index()
                    .tensors
                    .iter()
                    .any(|t| t.name == qkv_name);
                let separate = ["attn_q", "attn_k", "attn_v"].iter().all(|n| {
                    model
                        .index()
                        .tensors
                        .iter()
                        .any(|t| t.name == format!("{prefix}.{n}.weight"))
                });
                if nomic && fused == separate {
                    return Err(format!(
                        "{prefix} requires either fused attn_qkv or separate q/k/v projections"
                    ));
                }
                let (q, k, v) = if fused {
                    (
                        Linear::load_rows(&model, &qkv_name, None, 0, hidden, hidden)?,
                        Linear::load_rows(&model, &qkv_name, None, hidden, hidden, hidden)?,
                        Linear::load_rows(&model, &qkv_name, None, 2 * hidden, hidden, hidden)?,
                    )
                } else {
                    (
                        Linear::load(&model, &format!("{prefix}.attn_q"), hidden, hidden)?,
                        Linear::load(&model, &format!("{prefix}.attn_k"), hidden, hidden)?,
                        Linear::load(&model, &format!("{prefix}.attn_v"), hidden, hidden)?,
                    )
                };
                layers.push(BertLayer {
                    q,
                    k,
                    v,
                    out: Linear::load(&model, &format!("{prefix}.attn_output"), hidden, hidden)?,
                    attn_norm: LayerNorm::load(
                        &model,
                        &format!("{prefix}.attn_output_norm"),
                        hidden,
                        eps,
                    )?,
                    up: Linear::load(&model, &format!("{prefix}.ffn_up"), intermediate, hidden)?,
                    gate: if nomic {
                        Some(Linear::load(
                            &model,
                            &format!("{prefix}.ffn_gate"),
                            intermediate,
                            hidden,
                        )?)
                    } else {
                        None
                    },
                    down: Linear::load(&model, &format!("{prefix}.ffn_down"), hidden, intermediate)?,
                    out_norm: LayerNorm::load(
                        &model,
                        &format!("{prefix}.layer_output_norm"),
                        hidden,
                        eps,
                    )?,
                });
            }
        }
        let (rope, sliding_half, output_norm, rope_single) = if modern {
            let theta = |key: &str, default: f64| {
                meta.get(key).and_then(Value::as_f64).unwrap_or(default) as f32
            };
            let theta_global = theta("bert.rope_theta_global", 160_000.0);
            let theta_local = theta("bert.rope_theta_local", 10_000.0);
            if !theta_global.is_finite()
                || theta_global <= 0.0
                || !theta_local.is_finite()
                || theta_local <= 0.0
            {
                return Err("invalid modernbert rotary theta".into());
            }
            let local_attention = meta
                .get("bert.local_attention")
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(128);
            if local_attention == 0 || local_attention % 2 != 0 {
                return Err("invalid modernbert local attention window".into());
            }
            let head_dim = hidden / heads;
            (
                Some(ModernRope {
                    global: build_rope_tables(context, head_dim, theta_global),
                    local: build_rope_tables(context, head_dim, theta_local),
                }),
                local_attention / 2,
                Some(PlainNorm::load(&model, "output_norm", hidden, eps)?),
                None,
            )
        } else if nomic {
            let theta = meta
                .get("bert.rope_theta")
                .and_then(Value::as_f64)
                .unwrap_or(1000.0) as f32;
            if !theta.is_finite() || theta <= 0.0 {
                return Err("invalid nomic rotary theta".into());
            }
            let head_dim = hidden / heads;
            (
                None,
                0,
                None,
                Some(build_rope_tables(context, head_dim, theta)),
            )
        } else {
            (None, 0, None, None)
        };
        let default_pooling = pooling.unwrap_or_else(|| {
            match meta.get("bert.pooling_type").and_then(Value::as_u64) {
                Some(2) => "cls",
                Some(3) => "last",
                _ => "mean",
            }
            .into()
        });
        validate_pooling(&default_pooling)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.clamp(1, 256))
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            tokenizer,
            token,
            position,
            types,
            projection,
            classifier,
            classifier_dense,
            roberta,
            modern,
            nomic,
            labels,
            position_offset,
            pad,
            activation,
            embedding_norm,
            layers,
            modern_layers,
            rope,
            rope_single,
            sliding_half,
            output_norm,
            classifier_norm,
            classifier_pooling,
            hidden,
            context,
            vocab: tokens.len(),
            heads,
            cls,
            sep,
            pool,
            default_pooling,
        })
    }
    pub fn tokenize(&self, text: &str, special: bool) -> Result<Vec<u32>, String> {
        let encoding = match &self.tokenizer {
            EncoderTokenizer::WordPiece(t) => t
                .encode(text, false)
                .map_err(|e| e.to_string())?
                .get_ids()
                .to_vec(),
            EncoderTokenizer::Other(t) => t.encode_without_bos(text)?,
        };
        let mut ids = Vec::new();
        if special {
            ids.push(self.cls);
        }
        ids.extend_from_slice(&encoding);
        if special {
            ids.push(self.sep);
        }
        if special || !ids.is_empty() {
            self.validate_tokens(&ids)?;
        }
        Ok(ids)
    }
    pub fn detokenize(&self, ids: &[u32]) -> Result<String, String> {
        self.validate_tokens(ids)?;
        let ids = ids
            .iter()
            .copied()
            .filter(|&id| id != self.cls && id != self.sep)
            .collect::<Vec<_>>();
        match &self.tokenizer {
            EncoderTokenizer::WordPiece(t) => t.decode(&ids, false).map_err(|e| e.to_string()),
            EncoderTokenizer::Other(t) => t.decode(&ids, false),
        }
    }
    pub fn validate_tokens(&self, ids: &[u32]) -> Result<(), String> {
        if ids.is_empty()
            || ids.len() > self.context
            || ids.iter().any(|&id| id as usize >= self.vocab)
        {
            return Err("encoder input must contain valid token IDs and fit model context".into());
        }
        Ok(())
    }
    /// Exact pair boundaries and segment IDs, without padding or truncating silently.
    pub fn tokenize_pair(
        &self,
        query: &str,
        document: &str,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        let query = self.tokenize(query, false)?;
        let document = self.tokenize(document, false)?;
        let (ids, types) =
            pair_tokens(self.cls, self.sep, self.roberta || self.modern, &query, &document);
        self.validate_tokens(&ids)?;
        if !self.roberta && !self.modern && self.types.as_ref().is_none_or(|t| t.shape[1] < 2) {
            return Err("BERT pair inputs require two segment embedding rows".into());
        }
        Ok((ids, types))
    }
    pub fn classify_batch(&self, batch: &[(Vec<u32>, Vec<u32>)]) -> Result<Vec<Vec<f32>>, String> {
        use rayon::prelude::*;
        // Ragged sequence batches avoid padding work entirely; each sequence's
        // bidirectional attention sees only its own tokens, within the same pool.
        self.pool.install(|| {
            batch
                .par_iter()
                .map(|(ids, types)| self.classify(ids, types))
                .collect()
        })
    }
    pub fn pool_batch(&self, batch: &[Vec<u32>], pooling: &str) -> Result<Vec<Vec<f32>>, String> {
        use rayon::prelude::*;
        self.pool.install(|| {
            batch
                .par_iter()
                .map(|ids| self.embed(ids, pooling))
                .collect()
        })
    }
    pub fn has_classifier(&self) -> bool {
        self.classifier.is_some()
    }
    fn hidden_states(&self, input: &[u32], segments: &[u32]) -> Result<Vec<f32>, String> {
        if self.modern {
            return self.modern_hidden_states(input, segments);
        }
        if self.nomic {
            return self.nomic_hidden_states(input, segments);
        }
        self.validate_tokens(input)?;
        if segments.len() != input.len() {
            return Err("token_type_ids length must match input_ids".into());
        }
        let type_count = self.types.as_ref().map_or(1, |t| t.shape[1]);
        if segments.iter().any(|v| *v as usize >= type_count) {
            return Err("token_type_ids outside segment embedding table".into());
        }
        let ids = input
            .iter()
            .copied()
            .zip(segments.iter().copied())
            .enumerate()
            .filter(|(_, (id, _))| Some(*id) != self.pad)
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return Err("encoder input contains only padding".into());
        }
        self.pool.install(|| {
            let embedding = self.token.shape[0];
            let mut x = Vec::with_capacity(ids.len() * embedding);
            for (i, &(original, (id, segment))) in ids.iter().enumerate() {
                let position = if self.roberta {
                    i + self.position_offset
                } else {
                    original + self.position_offset
                };
                let table = self
                    .position
                    .as_ref()
                    .ok_or("encoder is missing position embeddings")?;
                if position >= table.shape[1] {
                    return Err("encoder position exceeds embedding table".into());
                }
                for j in 0..embedding {
                    x.push(
                        self.token.data[id as usize * embedding + j]
                            + table.data[position * embedding + j]
                            + self
                                .types
                                .as_ref()
                                .map_or(0.0, |t| t.data[segment as usize * embedding + j]),
                    );
                }
            }
            self.embedding_norm.apply(&mut x);
            if let Some(projection) = &self.projection {
                x = projection.forward(&x)?;
            }
            for layer in &self.layers {
                let q = layer.q.forward(&x)?;
                let k = layer.k.forward(&x)?;
                let v = layer.v.forward(&x)?;
                let a = compute::attention(&q, &k, &v, self.hidden, self.heads);
                compute::add(&mut x, &layer.out.forward(&a)?);
                layer.attn_norm.apply(&mut x);
                let mut f = layer.up.forward(&x)?;
                for v in &mut f {
                    *v = match self.activation.as_str() {
                        "relu" => v.max(0.0),
                        "silu" | "swish" => *v / (1.0 + (-*v).exp()),
                        "gelu_new" | "gelu_fast" => {
                            0.5 * *v * (1.0 + (0.7978846 * (*v + 0.044715 * v.powi(3))).tanh())
                        }
                        _ => compute::gelu(*v),
                    };
                }
                compute::add(&mut x, &layer.down.forward(&f)?);
                layer.out_norm.apply(&mut x);
            }
            Ok(x)
        })
    }
    /// NomicBERT: BERT-style post-norm flow with token-type embeddings, no
    /// learned positions, a single RoPE theta on Q/K and a SwiGLU block
    /// (`silu(gate) * up`).
    fn nomic_hidden_states(&self, input: &[u32], segments: &[u32]) -> Result<Vec<f32>, String> {
        self.validate_tokens(input)?;
        if segments.len() != input.len() {
            return Err("token_type_ids length must match input_ids".into());
        }
        let type_count = self.types.as_ref().map_or(1, |t| t.shape[1]);
        if segments.iter().any(|v| *v as usize >= type_count) {
            return Err("token_type_ids outside segment embedding table".into());
        }
        let rope = self
            .rope_single
            .as_ref()
            .ok_or("nomic requires rotary tables")?;
        if input.len() > rope.len {
            return Err("encoder position exceeds rotary table".into());
        }
        let ids = input
            .iter()
            .copied()
            .zip(segments.iter().copied())
            .enumerate()
            .filter(|(_, (id, _))| Some(*id) != self.pad)
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return Err("encoder input contains only padding".into());
        }
        self.pool.install(|| {
            let embedding = self.token.shape[0];
            let head_dim = self.hidden / self.heads;
            let mut x = Vec::with_capacity(ids.len() * embedding);
            for &(_, (id, segment)) in &ids {
                for j in 0..embedding {
                    x.push(
                        self.token.data[id as usize * embedding + j]
                            + self
                                .types
                                .as_ref()
                                .map_or(0.0, |t| t.data[segment as usize * embedding + j]),
                    );
                }
            }
            self.embedding_norm.apply(&mut x);
            if let Some(projection) = &self.projection {
                x = projection.forward(&x)?;
            }
            let positions = ids.iter().map(|(original, _)| *original).collect::<Vec<_>>();
            for layer in &self.layers {
                let gate = layer
                    .gate
                    .as_ref()
                    .ok_or("nomic is missing its gated feed-forward tensor")?;
                let mut q = layer.q.forward(&x)?;
                let mut k = layer.k.forward(&x)?;
                let v = layer.v.forward(&x)?;
                apply_modern_rope(&mut q, &positions, self.heads, head_dim, rope);
                apply_modern_rope(&mut k, &positions, self.heads, head_dim, rope);
                let a = compute::attention(&q, &k, &v, self.hidden, self.heads);
                compute::add(&mut x, &layer.out.forward(&a)?);
                layer.attn_norm.apply(&mut x);
                let up = layer.up.forward(&x)?;
                let gated = gate.forward(&x)?;
                let mut f = Vec::with_capacity(up.len());
                for (u, g) in up.iter().zip(gated.iter()) {
                    f.push((g / (1.0 + (-*g).exp())) * u);
                }
                compute::add(&mut x, &layer.down.forward(&f)?);
                layer.out_norm.apply(&mut x);
            }
            Ok(x)
        })
    }
    fn modern_hidden_states(&self, input: &[u32], segments: &[u32]) -> Result<Vec<f32>, String> {
        self.validate_tokens(input)?;
        if segments.len() != input.len() {
            return Err("token_type_ids length must match input_ids".into());
        }
        if segments.iter().any(|&v| v != 0) {
            return Err("modernbert has no segment embeddings".into());
        }
        let rope = self
            .rope
            .as_ref()
            .ok_or("modernbert requires rotary tables")?;
        let output_norm = self
            .output_norm
            .as_ref()
            .ok_or("modernbert requires a final norm")?;
        if input.len() > rope.global.len {
            return Err("encoder position exceeds rotary table".into());
        }
        let ids = input
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, id)| Some(*id) != self.pad)
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return Err("encoder input contains only padding".into());
        }
        self.pool.install(|| {
            let hidden = self.hidden;
            let head_dim = hidden / self.heads;
            let mut x = Vec::with_capacity(ids.len() * hidden);
            for &(_, id) in &ids {
                x.extend_from_slice(
                    &self.token.data[id as usize * hidden..(id as usize + 1) * hidden],
                );
            }
            self.embedding_norm.apply(&mut x);
            let positions = ids.iter().map(|(original, _)| *original).collect::<Vec<_>>();
            for layer in &self.modern_layers {
                let mut h = x.clone();
                if let Some(norm) = &layer.attn_norm {
                    norm.apply(&mut h);
                }
                let mut q = layer.q.forward(&h)?;
                let mut k = layer.k.forward(&h)?;
                let v = layer.v.forward(&h)?;
                let tables = if layer.global {
                    &rope.global
                } else {
                    &rope.local
                };
                apply_modern_rope(&mut q, &positions, self.heads, head_dim, tables);
                apply_modern_rope(&mut k, &positions, self.heads, head_dim, tables);
                let a = modern_attention(
                    &q,
                    &k,
                    &v,
                    &positions,
                    hidden,
                    self.heads,
                    layer.global,
                    self.sliding_half,
                );
                compute::add(&mut x, &layer.out.forward(&a)?);
                let mut m = x.clone();
                layer.ffn_norm.apply(&mut m);
                let up = layer.up.forward(&m)?;
                let gate = layer.gate.forward(&m)?;
                let mut f = Vec::with_capacity(up.len());
                for (u, g) in up.iter().zip(gate.iter()) {
                    f.push(gelu_tanh(*u) * g);
                }
                compute::add(&mut x, &layer.down.forward(&f)?);
            }
            let mut out = x;
            output_norm.apply(&mut out);
            Ok(out)
        })
    }
    pub fn embed(&self, ids: &[u32], pooling: &str) -> Result<Vec<f32>, String> {
        validate_pooling(pooling)?;
        let x = self.hidden_states(ids, &vec![0; ids.len()])?;
        let mut vector = match pooling {
            "cls" => x[..self.hidden].to_vec(),
            "last" => x[x.len() - self.hidden..].to_vec(),
            "mean" => {
                let mut sum = vec![0.0; self.hidden];
                for row in x.chunks_exact(self.hidden) {
                    compute::add(&mut sum, row);
                }
                for v in &mut sum {
                    *v /= (x.len() / self.hidden) as f32;
                }
                sum
            }
            _ => return Err("invalid pooling mode".into()),
        };
        compute::normalize(&mut vector)?;
        Ok(vector)
    }
    pub fn classify(&self, ids: &[u32], segments: &[u32]) -> Result<Vec<f32>, String> {
        let head = self
            .classifier
            .as_ref()
            .ok_or("model has no sequence classification head")?;
        let dense = self
            .classifier_dense
            .as_ref()
            .ok_or("missing classification dense/pooler tensor")?;
        let x = self.hidden_states(ids, segments)?;
        if self.modern {
            let norm = self
                .classifier_norm
                .as_ref()
                .ok_or("missing modernbert classifier norm")?;
            return self.pool.install(|| {
                let pooled = match self.classifier_pooling.as_str() {
                    "mean" => {
                        let mut sum = vec![0.0; self.hidden];
                        for row in x.chunks_exact(self.hidden) {
                            compute::add(&mut sum, row);
                        }
                        for v in &mut sum {
                            *v /= (x.len() / self.hidden) as f32;
                        }
                        sum
                    }
                    _ => x[..self.hidden].to_vec(),
                };
                let mut dense_out = dense.forward(&pooled)?;
                for v in &mut dense_out {
                    *v = gelu_tanh(*v);
                }
                let mut normalized = dense_out;
                norm.apply(&mut normalized);
                let logits = head.forward(&normalized)?;
                classification_probabilities(&logits)
            });
        }
        self.pool.install(|| {
            let mut pooled = dense.forward(&x[..self.hidden])?;
            for v in &mut pooled {
                *v = v.tanh();
            }
            let logits = head.forward(&pooled)?;
            classification_probabilities(&logits)
        })
    }
    pub fn set_device(&mut self, device: &str) -> Result<(), String> {
        if matches!(device, "cpu" | "auto") {
            return Ok(());
        }
        if self.modern {
            return Err("modernbert encoder runs on the CPU; use --device cpu".into());
        }
        #[cfg(feature = "rust-gpu")]
        {
            let context = crate::gpu::GpuContext::new(device)?;
            for layer in &mut self.layers {
                for linear in [
                    &mut layer.q,
                    &mut layer.k,
                    &mut layer.v,
                    &mut layer.out,
                    &mut layer.up,
                    &mut layer.down,
                ] {
                    linear.enable_gpu(&context)?;
                }
                if let Some(gate) = layer.gate.as_mut() {
                    gate.enable_gpu(&context)?;
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rust-gpu"))]
        Err("build with --features rust-gpu to enable the independent Rust GPU backend".into())
    }
}
pub fn validate_pooling(pooling: &str) -> Result<(), String> {
    if !matches!(pooling, "cls" | "mean" | "last") {
        return Err("pooling must be cls, mean or last".into());
    }
    Ok(())
}

fn pair_tokens(
    cls: u32,
    sep: u32,
    roberta: bool,
    query: &[u32],
    document: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut ids = vec![cls];
    ids.extend_from_slice(query);
    ids.push(sep);
    if roberta {
        ids.push(sep);
    }
    let boundary = ids.len();
    ids.extend_from_slice(document);
    ids.push(sep);
    let mut types = vec![0; ids.len()];
    if !roberta {
        types[boundary..].fill(1);
    }
    (ids, types)
}
fn classification_probabilities(logits: &[f32]) -> Result<Vec<f32>, String> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err("invalid classifier logits".into());
    }
    if logits.len() == 1 {
        let x = logits[0];
        return Ok(vec![if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }]);
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = logits.iter().map(|v| (*v - max).exp()).collect::<Vec<_>>();
    let sum: f32 = probabilities.iter().sum();
    for value in &mut probabilities {
        *value /= sum;
    }
    Ok(probabilities)
}
#[cfg(test)]
mod classification_tests {
    use super::*;
    #[test]
    fn pair_layout_uses_family_specific_boundaries() {
        assert_eq!(
            pair_tokens(101, 102, false, &[4, 5], &[6]),
            (vec![101, 4, 5, 102, 6, 102], vec![0, 0, 0, 0, 1, 1])
        );
        assert_eq!(
            pair_tokens(0, 2, true, &[4, 5], &[6]),
            (vec![0, 4, 5, 2, 2, 6, 2], vec![0; 7])
        );
        assert_eq!(
            pair_tokens(101, 102, false, &[], &[]),
            (vec![101, 102, 102], vec![0, 0, 1])
        );
    }
    #[test]
    fn stable_sigmoid_and_softmax() {
        assert_eq!(classification_probabilities(&[0.0]).unwrap(), vec![0.5]);
        assert_eq!(classification_probabilities(&[-1000.0]).unwrap(), vec![0.0]);
        assert_eq!(classification_probabilities(&[1000.0]).unwrap(), vec![1.0]);
        let p = classification_probabilities(&[1000.0, 1001.0, 999.0]).unwrap();
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[1] > p[0] && p[0] > p[2]);
        assert!(classification_probabilities(&[f32::NAN]).is_err());
    }
}
