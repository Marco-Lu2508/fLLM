//! BERT inference implemented with fLLM's Rust tensor operations.
use crate::compute::{self, LayerNorm, Linear, Tensor};
use crate::encoder_decoder::{DecoderStack, PairTemplate};
use crate::encoder_heads::{Heads, InputKind, SparseVector};
use crate::encoder_lora::Lora;
use fllm_gguf_probe::MappedGguf;
use serde_json::Value;
use tokenizers::{
    decoders::wordpiece::WordPiece as WordPieceDecoder, models::wordpiece::WordPiece,
    normalizers::bert::BertNormalizer, pre_tokenizers::bert::BertPreTokenizer, Tokenizer,
};

pub(crate) fn apply_act_dtype(values: &mut [f32]) {
    static DTYPE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let dtype = DTYPE.get_or_init(|| {
        std::env::var("FLLM_ACT_DTYPE")
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
    });
    if let Some(dt) = dtype {
        match dt.as_str() {
            "f16" | "fp16" => {
                for v in values.iter_mut() {
                    *v = half::f16::from_f32(*v).to_f32();
                }
            }
            "bf16" | "bfloat16" => {
                for v in values.iter_mut() {
                    *v = half::bf16::from_f32(*v).to_f32();
                }
            }
            _ => {}
        }
    }
}

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

/// NTK-scaled rotary tables as in the GTE remote code: base * factor and every frequency
/// divided by factor^(2/dim).
fn build_ntk_rope_tables(len: usize, dim: usize, theta: f64, factor: f64) -> ModernRopeTables {
    let half = dim / 2;
    let base = theta * factor;
    let shrink = factor.powf(2.0 / dim as f64);
    let inv_freq = (0..half)
        .map(|i| 1.0 / base.powf(i as f64 * 2.0 / dim as f64) / shrink)
        .collect::<Vec<_>>();
    let mut cos = vec![0.0; len * dim];
    let mut sin = vec![0.0; len * dim];
    for p in 0..len {
        for (i, f) in inv_freq.iter().enumerate() {
            let (s, c) = (p as f64 * f).sin_cos();
            cos[p * dim + i] = c as f32;
            cos[p * dim + i + half] = c as f32;
            sin[p * dim + i] = s as f32;
            sin[p * dim + i + half] = s as f32;
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
    /// sentence-transformers modules after pooling (Dense layers, Normalize).
    post: Vec<PostOp>,
    /// True when the checkpoint carried a sentence-transformers `modules.json` pipeline.
    has_pipeline: bool,
    /// Width of the pooled output after the post modules.
    pub embedding_dim: usize,
    extra: Heads,
    /// Llama / Qwen / Gemma style stack (embedding, reranker and reward models).
    decoder: Option<DecoderStack>,
    lora: Option<Lora>,
    pair_template: Option<PairTemplate>,
    /// Ids placed before and after the tokenised text (`[CLS]` / `[SEP]` for BERT).
    prefix: Vec<u32>,
    suffix: Vec<u32>,
}

/// Which outputs one forward pass should produce.
#[derive(Default, Clone)]
pub struct Want {
    /// Pooling mode and optional normalisation override of the dense embedding.
    pub dense: Option<(String, Option<bool>)>,
    /// Sparse lexical weights (SPLADE or BGE-M3).
    pub sparse: bool,
    /// Multi-vector (ColBERT) output for a query or a document.
    pub multi: Option<InputKind>,
    /// Index of the task LoRA adapter to apply (Jina embeddings v3); `None` uses the base weights.
    pub adapter: Option<usize>,
}

/// Outputs of [`BertEncoder::encode_outputs`].
#[derive(Default)]
pub struct Outputs {
    pub dense: Option<Vec<f32>>,
    pub sparse: Option<SparseVector>,
    pub multi: Option<Vec<Vec<f32>>>,
}

/// What a loaded encoder can produce.
#[derive(Clone, Copy, Debug, Default)]
pub struct Capabilities {
    pub dense: bool,
    pub sparse: bool,
    pub multi_vector: bool,
    pub token_classification: bool,
    pub sequence_classification: bool,
}

/// One sentence-transformers module that runs after pooling.
enum PostOp {
    Dense {
        linear: Linear,
        activation: PostActivation,
    },
    Normalize,
}

#[derive(Clone, Copy)]
enum PostActivation {
    Identity,
    Tanh,
    Relu,
    Gelu,
    Sigmoid,
    Silu,
}

impl PostActivation {
    fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "identity" => Self::Identity,
            "tanh" => Self::Tanh,
            "relu" => Self::Relu,
            "gelu" => Self::Gelu,
            "sigmoid" => Self::Sigmoid,
            "silu" => Self::Silu,
            other => return Err(format!("unsupported post-pooling activation {other}")),
        })
    }
    fn apply(self, x: &mut [f32]) {
        for v in x {
            *v = match self {
                Self::Identity => *v,
                Self::Tanh => v.tanh(),
                Self::Relu => v.max(0.0),
                Self::Gelu => compute::gelu(*v),
                Self::Sigmoid => 1.0 / (1.0 + (-*v).exp()),
                Self::Silu => *v / (1.0 + (-*v).exp()),
            };
        }
    }
}

/// Reads the `bert.post.modules` list and the matching `post.<i>.*` tensors.
fn load_post_ops(model: &MappedGguf, hidden: usize) -> Result<(Vec<PostOp>, bool, usize), String> {
    let Some(list) = model
        .index()
        .metadata
        .get("bert.post.modules")
        .and_then(Value::as_array)
    else {
        return Ok((Vec::new(), false, hidden));
    };
    let mut ops = Vec::new();
    let mut width = hidden;
    for (index, entry) in list.iter().enumerate() {
        let entry = entry.as_str().ok_or("invalid post-pooling module")?;
        if entry == "normalize" {
            ops.push(PostOp::Normalize);
        } else if let Some(act) = entry.strip_prefix("dense:").or_else(|| entry.strip_prefix("token_dense:")).or_else(|| entry.strip_prefix("pre_dense:")) {
            let prefix = if entry.starts_with("dense:") { "post" } else { "pre" };
            let name = format!("{prefix}.{index}.weight");
            let info = model
                .index()
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| format!("missing tensor {name}"))?;
            let rows = info
                .dimensions
                .get(1)
                .and_then(|&n| usize::try_from(n).ok())
                .filter(|&n| n > 0 && n <= 65536)
                .ok_or("invalid Dense module shape")?;
            ops.push(PostOp::Dense {
                linear: Linear::load(model, &format!("{prefix}.{index}"), rows, width)?,
                activation: PostActivation::parse(act)?,
            });
            width = rows;
        } else {
            return Err(format!("unsupported post-pooling module {entry}"));
        }
    }
    Ok((ops, true, width))
}

impl BertEncoder {
    pub fn load(path: &str, threads: usize, pooling: Option<String>) -> Result<Self, String> {
        let model = MappedGguf::open(path).map_err(|e| e.to_string())?;
        let meta = &model.index().metadata;
        if meta.get("general.architecture").and_then(Value::as_str) != Some("bert") {
            return Err("this independent encoder supports the BERT GGUF architecture".into());
        }
        if meta.get("bert.model_type").and_then(Value::as_str) == Some("decoder_style") {
            return Self::load_decoder(model, threads, pooling);
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
        let model_type = meta.get("bert.model_type").and_then(Value::as_str);
        let modern = model_type == Some("modernbert");
        // GTE "new" shares the NomicBERT flow (rotary, gated MLP, post-norm) with biased,
        // fused projections; both run through `nomic_hidden_states`.
        let gte = model_type == Some("gte_new");
        // Jina XLM-R with rotary positions: the same post-norm flow with an ungated MLP.
        let jina = model_type == Some("jina_xlmr");
        let nomic = gte || jina || model_type == Some("nomic_bert");
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
            || jina
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
                if !gte && !jina && has_gate != nomic {
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
                let qkv_bias = format!("{prefix}.attn_qkv.bias");
                let qkv_bias = Some(qkv_bias.as_str());
                let (q, k, v) = if fused {
                    (
                        Linear::load_rows(&model, &qkv_name, qkv_bias, 0, hidden, hidden)?,
                        Linear::load_rows(&model, &qkv_name, qkv_bias, hidden, hidden, hidden)?,
                        Linear::load_rows(&model, &qkv_name, qkv_bias, 2 * hidden, hidden, hidden)?,
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
                    up: if gte {
                        Linear::load_rows(
                            &model,
                            &format!("{prefix}.ffn_up.weight"),
                            None,
                            0,
                            intermediate,
                            hidden,
                        )?
                    } else {
                        Linear::load(&model, &format!("{prefix}.ffn_up"), intermediate, hidden)?
                    },
                    gate: if gte {
                        Some(Linear::load_rows(
                            &model,
                            &format!("{prefix}.ffn_up.weight"),
                            None,
                            intermediate,
                            intermediate,
                            hidden,
                        )?)
                    } else if nomic && !jina {
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
            let factor = meta
                .get("bert.rope_scaling_factor")
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            if !factor.is_finite() || !(1.0..=64.0).contains(&factor) {
                return Err("invalid rotary scaling factor".into());
            }
            let tables = if factor > 1.0 {
                build_ntk_rope_tables(
                    (context as f64 * factor).ceil() as usize,
                    head_dim,
                    f64::from(theta),
                    factor,
                )
            } else {
                build_rope_tables(context, head_dim, theta)
            };
            (None, 0, None, Some(tables))
        } else {
            (None, 0, None, None)
        };
        let default_pooling = pooling.unwrap_or_else(|| {
            match meta.get("bert.pooling_type").and_then(Value::as_u64) {
                Some(2) => "cls",
                Some(3) => "last",
                Some(4) => "max",
                Some(5) => "weightedmean",
                Some(6) => "mean_sqrt_len",
                _ => "mean",
            }
            .into()
        });
        validate_pooling(&default_pooling)?;
        let (post, has_pipeline, embedding_dim) = load_post_ops(&model, hidden)?;
        let lora = Lora::load(&model, count)?;
        let mut special = vec![cls, sep];
        special.extend(pad);
        if let Some(unk) = meta
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
        {
            special.push(unk);
        }
        let extra = Heads::load(&model, hidden, tokens.len(), eps, special)?;
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
            post,
            has_pipeline,
            embedding_dim,
            extra,
            decoder: None,
            lora,
            pair_template: None,
            prefix: vec![cls],
            suffix: vec![sep],
        })
    }
    /// Feed-forward activation named by `bert.activation`.
    fn activate(&self, v: f32) -> f32 {
        match self.activation.as_str() {
            "relu" => v.max(0.0),
            "silu" | "swish" | "swiglu" => v / (1.0 + (-v).exp()),
            "gelu_new" | "gelu_fast" => gelu_tanh(v),
            _ => compute::gelu(v),
        }
    }
    pub fn tokenize(&self, text: &str, special: bool) -> Result<Vec<u32>, String> {
        let ids = self.tokenize_raw(text, special)?;
        if special || !ids.is_empty() {
            self.validate_tokens(&ids)?;
        }
        Ok(ids)
    }
    /// Tokenises without checking the length against the model context (for truncation).
    pub fn tokenize_raw(&self, text: &str, special: bool) -> Result<Vec<u32>, String> {
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
            ids.extend_from_slice(&self.prefix);
        }
        ids.extend_from_slice(&encoding);
        if special {
            ids.extend_from_slice(&self.suffix);
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
        self.tokenize_pair_with(query, document, None)
    }
    /// Like [`Self::tokenize_pair`]; `instruction` fills the `{instruction}` slot of decoder reranker prompts.
    pub fn tokenize_pair_with(
        &self,
        query: &str,
        document: &str,
        instruction: Option<&str>,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        if let Some(template) = &self.pair_template {
            let ids = self.tokenize(&template.render(query, document, instruction), true)?;
            self.validate_tokens(&ids)?;
            let types = vec![0; ids.len()];
            return Ok((ids, types));
        }
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
    pub fn capabilities(&self) -> Capabilities {
        Capabilities {
            dense: self.extra.dense,
            sparse: self.extra.has_sparse(),
            multi_vector: self.extra.has_multi_vector(),
            token_classification: self.extra.has_token_classifier(),
            sequence_classification: self.classifier.is_some(),
        }
    }
    /// Names of the task LoRA adapters (empty without adapters).
    pub fn adapter_names(&self) -> Vec<String> {
        self.lora.as_ref().map_or_else(Vec::new, |l| l.names.clone())
    }
    /// Index of the adapter called `name`.
    pub fn adapter_index(&self, name: &str) -> Option<usize> {
        self.lora.as_ref().and_then(|l| l.index_of(name))
    }
    /// Text prepended to inputs of the adapter (Jina task instructions).
    pub fn adapter_instruction(&self, index: usize) -> &str {
        self.lora
            .as_ref()
            .and_then(|l| l.instructions.get(index))
            .map_or("", String::as_str)
    }
    /// Dimension of the multi-vector head, if any.
    pub fn multi_vector_dim(&self) -> Option<usize> {
        self.extra.multi_vector_dim()
    }
    /// Whether MaxSim scores are averaged over the query vectors (BGE-M3) instead of summed.
    pub fn mean_maxsim(&self) -> bool {
        self.extra.mean_maxsim()
    }
    pub fn token_labels(&self) -> &[String] {
        &self.extra.token_labels
    }
    /// Dense, sparse and multi-vector outputs of one input in a single pass (a second pass only
    /// when the multi-vector head rewrites the ids with markers).
    pub fn encode_outputs(&self, ids: &[u32], want: &Want) -> Result<Outputs, String> {
        let zeros = |n: usize| vec![0u32; n];
        let kept = |ids: &[u32]| {
            ids.iter()
                .copied()
                .filter(|id| Some(*id) != self.pad)
                .collect::<Vec<_>>()
        };
        let mut out = Outputs::default();
        let mut states: Option<(Vec<u32>, Vec<f32>)> = None;
        if want.dense.is_some() || want.sparse {
            let x = self.hidden_states_with(ids, &zeros(ids.len()), want.adapter)?;
            states = Some((ids.to_vec(), x));
        }
        if let (Some((pooling, normalize)), Some((_, x))) = (&want.dense, &states) {
            if !self.extra.dense {
                return Err("this model has no pooled dense embedding output".into());
            }
            let pooled = self.pool_states(x, pooling)?;
            let mut vector = self.pool.install(|| self.apply_post(pooled))?;
            if normalize.unwrap_or_else(|| self.default_normalize()) {
                compute::normalize(&mut vector)?;
            }
            out.dense = Some(vector);
        }
        if want.sparse {
            let (used, x) = states.as_ref().ok_or("missing encoder states")?;
            out.sparse = Some(self.pool.install(|| {
                if self.extra.has_sparse_mlm() {
                    self.extra.splade(x, self.hidden, self.vocab, &self.token.data)
                } else {
                    self.extra.lexical(x, &kept(used))
                }
            })?);
        }
        if let Some(kind) = want.multi {
            let cids = self.extra.colbert_ids(ids, kind);
            self.validate_tokens(&cids)?;
            let reuse = states.as_ref().filter(|(used, _)| *used == cids);
            let owned;
            let x = match reuse {
                Some((_, x)) => x,
                None => {
                    owned = self.hidden_states(&cids, &zeros(cids.len()))?;
                    &owned
                }
            };
            out.multi = Some(self.pool.install(|| {
                self.extra
                    .multi_vector(x, self.hidden, &kept(&cids), kind)
            })?);
        }
        Ok(out)
    }
    /// Per-token class probabilities for a token classification model, with the kept token ids.
    pub fn classify_tokens(&self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let x = self.hidden_states(ids, &vec![0; ids.len()])?;
        self.pool.install(|| self.extra.token_probabilities(&x))
    }
    /// Runs `f` over `items` on the encoder's thread pool.
    pub fn parallel<T: Sync, R: Send>(&self, items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
        use rayon::prelude::*;
        self.pool.install(|| items.par_iter().map(f).collect())
    }
    /// Shortens `ids` to at most `limit` tokens, keeping a trailing separator token.
    pub fn truncate_ids(&self, mut ids: Vec<u32>, limit: usize) -> Vec<u32> {
        let limit = limit.clamp(2, self.context.max(2));
        if ids.len() <= limit {
            return ids;
        }
        let keep_suffix = !self.suffix.is_empty() && ids.ends_with(&self.suffix);
        if keep_suffix {
            let room = limit.saturating_sub(self.suffix.len());
            ids.truncate(room);
            ids.extend_from_slice(&self.suffix);
        } else {
            ids.truncate(limit);
        }
        ids
    }
    /// The ids that produce token states (padding is dropped).
    pub fn kept_ids(&self, ids: &[u32]) -> Vec<u32> {
        ids.iter()
            .copied()
            .filter(|id| Some(*id) != self.pad)
            .collect()
    }
    pub fn has_classifier(&self) -> bool {
        self.classifier.is_some()
    }
    fn hidden_states(&self, input: &[u32], segments: &[u32]) -> Result<Vec<f32>, String> {
        self.hidden_states_with(input, segments, None)
    }
    fn hidden_states_with(
        &self,
        input: &[u32],
        segments: &[u32],
        adapter: Option<usize>,
    ) -> Result<Vec<f32>, String> {
        if let Some(a) = adapter {
            if self.lora.as_ref().is_none_or(|l| a >= l.names.len()) || !self.nomic {
                return Err("this model has no such LoRA adapter".into());
            }
        }
        if let Some(decoder) = &self.decoder {
            self.validate_tokens(input)?;
            return self
                .pool
                .install(|| decoder.forward(decoder.embed.lookup(input)?));
        }
        if self.modern {
            return self.modern_hidden_states(input, segments);
        }
        if self.nomic {
            return self.nomic_hidden_states(input, segments, adapter);
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
                apply_act_dtype(&mut x);
                layer.out_norm.apply(&mut x);
            }
            Ok(x)
        })
    }
    /// NomicBERT: BERT-style post-norm flow with token-type embeddings, no
    /// learned positions, a single RoPE theta on Q/K and a SwiGLU block
    /// (`silu(gate) * up`).
    fn nomic_hidden_states(
        &self,
        input: &[u32],
        segments: &[u32],
        adapter: Option<usize>,
    ) -> Result<Vec<f32>, String> {
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
                let start = x.len();
                for j in 0..embedding {
                    x.push(
                        self.token.data[id as usize * embedding + j]
                            + self
                                .types
                                .as_ref()
                                .map_or(0.0, |t| t.data[segment as usize * embedding + j]),
                    );
                }
                if let (Some(a), Some(lora)) = (adapter, &self.lora) {
                    lora.add_embedding(a, id as usize, segment as usize, &mut x[start..]);
                }
            }
            self.embedding_norm.apply(&mut x);
            if let Some(projection) = &self.projection {
                x = projection.forward(&x)?;
            }
            let positions = ids.iter().map(|(original, _)| *original).collect::<Vec<_>>();
            let lora = adapter.zip(self.lora.as_ref());
            for (li, layer) in self.layers.iter().enumerate() {
                let mut q = layer.q.forward(&x)?;
                let mut k = layer.k.forward(&x)?;
                let mut v = layer.v.forward(&x)?;
                if let Some((a, l)) = lora {
                    l.layers[li]
                        .qkv
                        .add(a, l.rank, l.scaling, &x, &mut [&mut q, &mut k, &mut v]);
                }
                apply_modern_rope(&mut q, &positions, self.heads, head_dim, rope);
                apply_modern_rope(&mut k, &positions, self.heads, head_dim, rope);
                let attended = compute::attention(&q, &k, &v, self.hidden, self.heads);
                let mut o = layer.out.forward(&attended)?;
                if let Some((a, l)) = lora {
                    l.layers[li]
                        .out
                        .add(a, l.rank, l.scaling, &attended, &mut [&mut o]);
                }
                compute::add(&mut x, &o);
                layer.attn_norm.apply(&mut x);
                let mut up = layer.up.forward(&x)?;
                if let Some((a, l)) = lora {
                    l.layers[li]
                        .up
                        .add(a, l.rank, l.scaling, &x, &mut [&mut up]);
                }
                let f = match &layer.gate {
                    // Gated feed-forward (NomicBERT, GTE): act(gate) * up.
                    Some(gate) => {
                        let gated = gate.forward(&x)?;
                        up.iter()
                            .zip(gated.iter())
                            .map(|(u, g)| self.activate(*g) * u)
                            .collect::<Vec<_>>()
                    }
                    // Plain feed-forward (Jina XLM-R): act(up).
                    None => up.iter().map(|u| self.activate(*u)).collect::<Vec<_>>(),
                };
                let mut d = layer.down.forward(&f)?;
                if let Some((a, l)) = lora {
                    l.layers[li]
                        .down
                        .add(a, l.rank, l.scaling, &f, &mut [&mut d]);
                }
                compute::add(&mut x, &d);
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
                apply_act_dtype(&mut x);
            }
            let mut out = x;
            output_norm.apply(&mut out);
            Ok(out)
        })
    }
    /// Pools the token states of one sequence (`rows * hidden` values) with the named mode.
    pub fn pool_states(&self, x: &[f32], pooling: &str) -> Result<Vec<f32>, String> {
        validate_pooling(pooling)?;
        let hidden = self.hidden;
        let rows = x.len() / hidden;
        if rows == 0 {
            return Err("encoder produced no token states".into());
        }
        Ok(match pooling {
            "cls" => x[..hidden].to_vec(),
            "last" => x[x.len() - hidden..].to_vec(),
            "mean" | "mean_sqrt_len" => {
                let mut sum = vec![0.0; hidden];
                for row in x.chunks_exact(hidden) {
                    compute::add(&mut sum, row);
                }
                let divisor = if pooling == "mean" {
                    rows as f32
                } else {
                    (rows as f32).sqrt()
                };
                for v in &mut sum {
                    *v /= divisor;
                }
                sum
            }
            "max" => {
                let mut best = x[..hidden].to_vec();
                for row in x.chunks_exact(hidden).skip(1) {
                    for (b, v) in best.iter_mut().zip(row) {
                        *b = b.max(*v);
                    }
                }
                best
            }
            "weightedmean" => {
                // sentence-transformers weights token i (1-based) by i.
                let mut sum = vec![0.0; hidden];
                for (i, row) in x.chunks_exact(hidden).enumerate() {
                    let w = (i + 1) as f32;
                    for (s, v) in sum.iter_mut().zip(row) {
                        *s += w * v;
                    }
                }
                let total = (rows * (rows + 1) / 2) as f32;
                for v in &mut sum {
                    *v /= total;
                }
                sum
            }
            _ => return Err("invalid pooling mode".into()),
        })
    }
    /// Whether outputs are L2-normalised when the request does not say.
    pub fn default_normalize(&self) -> bool {
        if !self.has_pipeline || std::env::var_os("FLLM_ENCODER_ALWAYS_NORMALIZE").is_some() {
            return true;
        }
        self.post.iter().any(|op| matches!(op, PostOp::Normalize))
    }
    /// Runs the Dense modules on a pooled vector.
    fn apply_post(&self, mut vector: Vec<f32>) -> Result<Vec<f32>, String> {
        for op in &self.post {
            if let PostOp::Dense { linear, activation } = op {
                vector = linear.forward(&vector)?;
                activation.apply(&mut vector);
            }
        }
        Ok(vector)
    }
    pub fn embed(&self, ids: &[u32], pooling: &str) -> Result<Vec<f32>, String> {
        self.embed_with(ids, pooling, None)
    }
    /// Pooled embedding; `normalize` overrides the model default when given.
    pub fn embed_with(
        &self,
        ids: &[u32],
        pooling: &str,
        normalize: Option<bool>,
    ) -> Result<Vec<f32>, String> {
        validate_pooling(pooling)?;
        let x = self.hidden_states(ids, &vec![0; ids.len()])?;
        let pooled = self.pool_states(&x, pooling)?;
        let mut vector = self.pool.install(|| self.apply_post(pooled))?;
        if normalize.unwrap_or_else(|| self.default_normalize()) {
            compute::normalize(&mut vector)?;
        }
        Ok(vector)
    }
    /// Class probabilities (sigmoid for one label, softmax otherwise).
    pub fn classify(&self, ids: &[u32], segments: &[u32]) -> Result<Vec<f32>, String> {
        classification_probabilities(&self.classify_logits(ids, segments)?)
    }
    /// Probabilities and raw logits of the classification head.
    pub fn classify_with_logits(
        &self,
        ids: &[u32],
        segments: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let logits = self.classify_logits(ids, segments)?;
        Ok((classification_probabilities(&logits)?, logits))
    }
    /// Raw logits of the sequence classification head.
    pub fn classify_logits(&self, ids: &[u32], segments: &[u32]) -> Result<Vec<f32>, String> {
        let head = self
            .classifier
            .as_ref()
            .ok_or("model has no sequence classification head")?;
        if self.decoder.is_some() {
            // Decoder models classify the hidden state of the last token (after the final norm).
            let x = self.hidden_states(ids, segments)?;
            return self
                .pool
                .install(|| head.forward(&x[x.len() - self.hidden..]));
        }
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
                head.forward(&normalized)
            });
        }
        self.pool.install(|| {
            let mut pooled = dense.forward(&x[..self.hidden])?;
            for v in &mut pooled {
                *v = v.tanh();
            }
            head.forward(&pooled)
        })
    }
    pub fn set_device(&mut self, device: &str) -> Result<(), String> {
        if matches!(device, "cpu" | "auto") {
            return Ok(());
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
            for layer in &mut self.modern_layers {
                for linear in [
                    &mut layer.q,
                    &mut layer.k,
                    &mut layer.v,
                    &mut layer.out,
                    &mut layer.up,
                    &mut layer.gate,
                    &mut layer.down,
                ] {
                    linear.enable_gpu(&context)?;
                }
            }
            if let Some(dec) = &mut self.decoder {
                dec.enable_gpu(&context)?;
            }
            if let Some(p) = &mut self.projection {
                p.enable_gpu(&context)?;
            }
            if let Some(c) = &mut self.classifier {
                c.enable_gpu(&context)?;
            }
            if let Some(cd) = &mut self.classifier_dense {
                cd.enable_gpu(&context)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "rust-gpu"))]
        Err("build with --features rust-gpu to enable the independent Rust GPU backend".into())
    }
}
impl BertEncoder {
    /// Loads a Llama / Qwen / Gemma style stack written by the decoder-encoder converter.
    fn load_decoder(
        model: MappedGguf,
        threads: usize,
        pooling: Option<String>,
    ) -> Result<Self, String> {
        let meta = &model.index().metadata;
        let integer = |key: &str| -> Result<usize, String> {
            meta.get(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("invalid/missing {key}"))
        };
        let hidden = integer("bert.embedding_length")?;
        let heads = integer("bert.attention.head_count")?;
        let inter = integer("bert.feed_forward_length")?;
        let count = integer("bert.block_count")?;
        let context = integer("bert.context_length")?;
        if hidden > 32768 || count > 256 || context > 262_144 {
            return Err("unsupported decoder-encoder dimensions".into());
        }
        let decoder = DecoderStack::load(&model, hidden, heads, count, inter, context)?;
        let vocab = decoder.vocab();
        let tokenizer =
            EncoderTokenizer::Other(crate::tokenizer::TokenizerBundle::from_gguf(&model)?);
        let ids = |key: &str| -> Vec<u32> {
            meta.get(key)
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let prefix = ids("bert.prefix_ids");
        let suffix = ids("bert.suffix_ids");
        if prefix.iter().chain(&suffix).any(|&id| id as usize >= vocab) {
            return Err("prefix/suffix token id outside the vocabulary".into());
        }
        let num_labels = meta
            .get("bert.classifier.label_count")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let (classifier, labels) = if num_labels > 0 {
            let names = meta
                .get("bert.classifier.labels")
                .and_then(Value::as_array)
                .ok_or("missing classifier labels")?
                .iter()
                .map(|v| v.as_str().map(str::to_owned).ok_or("label must be a string"))
                .collect::<Result<Vec<_>, _>>()?;
            if names.len() != num_labels {
                return Err("classifier labels disagree with label_count".into());
            }
            (
                Some(Linear::load(&model, "classifier", num_labels, hidden)?),
                names,
            )
        } else {
            (None, Vec::new())
        };
        let default_pooling = pooling.unwrap_or_else(|| {
            match meta.get("bert.pooling_type").and_then(Value::as_u64) {
                Some(2) => "cls",
                Some(1) => "mean",
                Some(4) => "max",
                Some(5) => "weightedmean",
                Some(6) => "mean_sqrt_len",
                _ => "last",
            }
            .into()
        });
        validate_pooling(&default_pooling)?;
        let (post, has_pipeline, embedding_dim) = load_post_ops(&model, hidden)?;
        let extra = Heads::load(&model, hidden, vocab, 1e-6, Vec::new())?;
        let pair_template = PairTemplate::load(&model);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.clamp(1, 256))
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            tokenizer,
            token: Tensor {
                shape: vec![hidden, vocab],
                data: Vec::new(),
            },
            position: None,
            types: None,
            projection: None,
            classifier,
            classifier_dense: None,
            roberta: false,
            modern: false,
            nomic: false,
            labels,
            position_offset: 0,
            pad: None,
            activation: "silu".into(),
            embedding_norm: EncoderNorm::Plain(PlainNorm {
                weight: vec![1.0; hidden],
                eps: 1e-6,
            }),
            layers: Vec::new(),
            modern_layers: Vec::new(),
            rope: None,
            rope_single: None,
            sliding_half: 0,
            output_norm: None,
            classifier_norm: None,
            classifier_pooling: "cls".into(),
            hidden,
            context,
            vocab,
            heads,
            cls: u32::MAX,
            sep: u32::MAX,
            pool,
            default_pooling,
            post,
            has_pipeline,
            embedding_dim,
            extra,
            decoder: Some(decoder),
            lora: None,
            pair_template,
            prefix,
            suffix,
        })
    }
}
pub fn validate_pooling(pooling: &str) -> Result<(), String> {
    if !matches!(
        pooling,
        "cls" | "mean" | "last" | "max" | "weightedmean" | "mean_sqrt_len"
    ) {
        return Err("pooling must be cls, mean, last, max, weightedmean or mean_sqrt_len".into());
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
