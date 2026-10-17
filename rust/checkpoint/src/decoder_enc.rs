//! Decoder-architecture checkpoints used as encoders, exported in the BERT GGUF container
//! (`bert.model_type = decoder_style`) so the encoder server can run them:
//!
//! * `Gemma3TextModel` with bidirectional attention (EmbeddingGemma), with the
//!   sentence-transformers Dense modules from `modules.json`;
//! * `LlamaBidirectionalModel` (llama-embed-nemotron) and `Qwen3Model` embedders;
//! * `Qwen2/Qwen3/Llama ForSequenceClassification` (reward and classification models);
//! * Qwen3-Reranker style `Qwen3ForCausalLM` checkpoints scored by the yes/no logits
//!   (selected by a "reranker" checkpoint name or `FLLM_CONVERT_AS=reranker`).
//!
//! The engine side is `rust/engine/src/encoder_decoder.rs`.
use crate::{
    gguf::{Meta, OutTensor},
    safetensors::{self, Payload, Tensor},
    st, CheckpointError, ConvertOptions,
};
use half::{bf16, f16};
use serde_json::Value;
use std::{collections::HashMap, path::Path, sync::Arc};

const RERANK_PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
const RERANK_BODY: &str = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}";
const RERANK_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const RERANK_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Embedder,
    SequenceClassifier,
    Reranker,
}

fn archs(cfg: &Value) -> Vec<&str> {
    cfg["architectures"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

fn text_config(cfg: &Value) -> &Value {
    if cfg["text_config"].is_object() {
        &cfg["text_config"]
    } else {
        cfg
    }
}

fn reranker_requested(root: &Path, cfg: &Value) -> bool {
    if std::env::var("FLLM_CONVERT_AS").is_ok_and(|v| v == "reranker") {
        return true;
    }
    let name = format!(
        "{} {}",
        root.display(),
        cfg["_name_or_path"].as_str().unwrap_or_default()
    )
    .to_lowercase();
    name.contains("reranker")
}

fn classify(root: &Path, cfg: &Value) -> Option<Kind> {
    let archs = archs(cfg);
    let tc = text_config(cfg);
    let model_type = tc["model_type"].as_str().unwrap_or_default();
    if archs
        .iter()
        .any(|a| matches!(*a, "Gemma3TextModel" | "LlamaBidirectionalModel" | "Qwen3Model" | "ColPali" | "ColQwen" | "ColPaliForRetrieval" | "ColQwen2ForRetrieval"))
        || (model_type == "gemma3_text" && tc["use_bidirectional_attention"].as_bool() == Some(true))
    {
        return Some(Kind::Embedder);
    }
    if archs.iter().any(|a| {
        matches!(
            *a,
            "Qwen2ForSequenceClassification"
                | "Qwen3ForSequenceClassification"
                | "LlamaForSequenceClassification"
        )
    }) {
        return Some(Kind::SequenceClassifier);
    }
    if archs.contains(&"Qwen3ForCausalLM") && reranker_requested(root, cfg) {
        return Some(Kind::Reranker);
    }
    None
}

pub(crate) fn supports(root: &Path, cfg: &Value) -> bool {
    classify(root, cfg).is_some()
}

fn number(cfg: &Value, key: &str) -> Result<u32, CheckpointError> {
    cfg[key]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .filter(|&v| v > 0)
        .ok_or_else(|| format!("config.json requires positive {key}").into())
}

fn decode_row(t: &Tensor, row: usize, width: usize) -> Result<Vec<f32>, CheckpointError> {
    let bytes = t.payload.as_slice();
    let size = match t.dtype.as_str() {
        "F32" => 4,
        "F16" | "BF16" => 2,
        other => return Err(format!("unsupported language-model head dtype {other}").into()),
    };
    let chunk = bytes
        .get(row * width * size..(row + 1) * width * size)
        .ok_or("language-model head row out of range")?;
    Ok(chunk
        .chunks_exact(size)
        .map(|b| match t.dtype.as_str() {
            "F32" => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            "F16" => f16::from_le_bytes([b[0], b[1]]).to_f32(),
            _ => bf16::from_le_bytes([b[0], b[1]]).to_f32(),
        })
        .collect())
}

/// Ids placed around the text by the tokenizer's post-processor or its add_bos/add_eos flags.
fn affixes(root: &Path, meta: &[(String, Meta)]) -> Result<(Vec<u32>, Vec<u32>), CheckpointError> {
    let tj = st::json(&root.join("tokenizer.json")).unwrap_or(Value::Null);
    let template = |p: &Value| -> Option<(Vec<u32>, Vec<u32>)> {
        let single = p["single"].as_array()?;
        let (mut before, mut after) = (Vec::new(), Vec::new());
        let mut seen = false;
        for piece in single {
            if piece.get("Sequence").is_some() {
                seen = true;
            } else if let Some(special) = piece.get("SpecialToken") {
                let id = special["id"].as_str()?;
                let ids = p["special_tokens"][id]["ids"].as_array()?;
                let target = if seen { &mut after } else { &mut before };
                target.extend(ids.iter().filter_map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok())));
            }
        }
        Some((before, after))
    };
    let post = &tj["post_processor"];
    let from_post = match post["type"].as_str() {
        Some("TemplateProcessing") => template(post),
        Some("Sequence") => post["processors"]
            .as_array()
            .and_then(|a| a.iter().find(|p| p["type"].as_str() == Some("TemplateProcessing")))
            .and_then(template),
        _ => None,
    };
    if let Some(found) = from_post {
        return Ok(found);
    }
    let tc = st::json(&root.join("tokenizer_config.json")).unwrap_or(Value::Null);
    let id = |key: &str| {
        meta.iter().find_map(|(k, v)| match v {
            Meta::U32(n) if k == key => Some(*n),
            _ => None,
        })
    };
    let mut before = Vec::new();
    let mut after = Vec::new();
    if tc["add_bos_token"].as_bool() == Some(true) {
        before.extend(id("tokenizer.ggml.bos_token_id"));
    }
    if tc["add_eos_token"].as_bool() == Some(true) {
        after.extend(id("tokenizer.ggml.eos_token_id"));
    }
    Ok((before, after))
}

fn theta_of(cfg: &Value, layer_type: &str) -> Option<f64> {
    cfg["rope_parameters"][layer_type]["rope_theta"]
        .as_f64()
        .or_else(|| cfg["rope_parameters"]["rope_theta"].as_f64())
}

pub(crate) fn prepare(
    input: &Path,
    options: ConvertOptions,
    cfg: &Value,
) -> Result<crate::enc_out::Prepared, CheckpointError> {
    let root = if input.is_dir() {
        input
    } else {
        input.parent().ok_or("checkpoint has no parent directory")?
    };
    let kind = classify(root, cfg).ok_or("not a decoder-encoder checkpoint")?;
    let tc = text_config(cfg);
    let model_type = tc["model_type"].as_str().unwrap_or_default();
    let gemma = model_type.starts_with("gemma3");
    let llama = model_type == "llama";
    let qwen3 = model_type == "qwen3";
    if !(gemma || llama || qwen3 || model_type == "qwen2") {
        return Err(format!("unsupported decoder-encoder model_type {model_type:?}").into());
    }
    let hidden = number(tc, "hidden_size")?;
    let heads = number(tc, "num_attention_heads")?;
    let kv_heads = tc["num_key_value_heads"].as_u64().map_or(heads, |v| v as u32);
    let head_dim = tc["head_dim"].as_u64().map_or(hidden / heads, |v| v as u32);
    let layers = number(tc, "num_hidden_layers")?;
    let inter = number(tc, "intermediate_size")?;
    let vocab = number(tc, "vocab_size")?;
    let positions = number(tc, "max_position_embeddings")?;
    if kv_heads == 0 || !heads.is_multiple_of(kv_heads) || !head_dim.is_multiple_of(2) || layers > 256 {
        return Err("unsupported attention geometry".into());
    }
    let eps = tc["rms_norm_eps"].as_f64().unwrap_or(1e-6);
    let act_name = tc["hidden_activation"]
        .as_str()
        .or_else(|| tc["hidden_act"].as_str())
        .unwrap_or("silu");
    let activation = match act_name {
        "silu" | "swish" => "silu",
        "gelu_pytorch_tanh" | "gelu_new" | "gelu_tanh" => "gelu_tanh",
        other => return Err(format!("unsupported activation {other}").into()),
    };
    let bidirectional = tc["use_bidirectional_attention"].as_bool() == Some(true)
        || archs(cfg).contains(&"LlamaBidirectionalModel");
    let causal = !bidirectional;
    let theta = theta_of(tc, "full_attention")
        .or_else(|| tc["rope_theta"].as_f64())
        .unwrap_or(if gemma { 1_000_000.0 } else { 10_000.0 });
    let theta_local = if gemma {
        theta_of(tc, "sliding_attention")
            .or_else(|| tc["rope_local_base_freq"].as_f64())
            .unwrap_or(10_000.0)
    } else {
        0.0
    };
    let context = positions.min(131_072);
    let mut meta = vec![
        ("general.architecture".into(), Meta::Str("bert".into())),
        (
            "general.name".into(),
            Meta::Str(root.file_name().unwrap_or_default().to_string_lossy().into()),
        ),
        ("bert.model_type".into(), Meta::Str("decoder_style".into())),
        ("bert.activation".into(), Meta::Str(activation.into())),
        ("bert.attention.causal".into(), Meta::Bool(causal)),
        ("bert.attention.layer_norm_epsilon".into(), Meta::F32(eps as f32)),
    ];
    for (key, n) in [
        ("embedding_length", hidden),
        ("context_length", context),
        ("attention.head_count", heads),
        ("feed_forward_length", inter),
        ("block_count", layers),
        ("dec.kv_head_count", kv_heads),
        ("dec.head_dim", head_dim),
    ] {
        meta.push((format!("bert.{key}"), Meta::U32(n)));
    }
    meta.push(("bert.dec.rms_eps".into(), Meta::F32(eps as f32)));
    meta.push(("bert.dec.causal".into(), Meta::Bool(causal)));
    meta.push(("bert.dec.activation".into(), Meta::Str(activation.into())));
    meta.push(("bert.dec.post_norms".into(), Meta::Bool(gemma)));
    meta.push(("bert.dec.qk_norm".into(), Meta::Bool(gemma || qwen3)));
    meta.push(("bert.dec.rope_theta".into(), Meta::F32(theta as f32)));
    meta.push(("bert.dec.rope_theta_local".into(), Meta::F32(theta_local as f32)));
    if gemma {
        let scalar = tc["query_pre_attn_scalar"].as_f64().unwrap_or(f64::from(head_dim));
        meta.push(("bert.dec.attn_scale".into(), Meta::F32((1.0 / scalar.sqrt()) as f32)));
        meta.push((
            "bert.dec.embed_scale".into(),
            Meta::F32((f64::from(hidden)).sqrt() as f32),
        ));
        let window = tc["sliding_window"].as_u64().unwrap_or(0) as u32;
        // Bidirectional Gemma 3 attends to keys within half the configured window.
        meta.push((
            "bert.dec.sliding_window".into(),
            Meta::U32(if bidirectional { window / 2 } else { window }),
        ));
        let pattern = tc["sliding_window_pattern"].as_u64().unwrap_or(6) as usize;
        let local = (0..layers as usize)
            .map(|i| match tc["layer_types"].as_array() {
                Some(types) => types.get(i).and_then(Value::as_str) == Some("sliding_attention"),
                None => (i + 1) % pattern != 0,
            })
            .collect::<Vec<_>>();
        meta.push(("bert.dec.layer_local".into(), Meta::Bools(local)));
    }
    let scaling = tc["rope_scaling"]["rope_type"]
        .as_str()
        .or_else(|| tc["rope_scaling"]["type"].as_str())
        .or_else(|| tc["rope_parameters"]["rope_type"].as_str());
    match scaling {
        None | Some("default") => {}
        Some("llama3") => {
            let s = if tc["rope_scaling"].is_object() {
                &tc["rope_scaling"]
            } else {
                &tc["rope_parameters"]
            };
            for (key, field) in [
                ("rope_factor", "factor"),
                ("rope_low_freq_factor", "low_freq_factor"),
                ("rope_high_freq_factor", "high_freq_factor"),
                ("rope_original_context", "original_max_position_embeddings"),
            ] {
                let v = s[field].as_f64().ok_or_else(|| format!("rope scaling lacks {field}"))?;
                meta.push((format!("bert.dec.{key}"), Meta::F32(v as f32)));
            }
            meta.push(("bert.dec.rope_scaling".into(), Meta::Str("llama3".into())));
        }
        Some(other) => return Err(format!("unsupported rotary scaling {other}").into()),
    }
    let pooling = match kind {
        Kind::Embedder => st::pooling_or(root, || {
            match cfg["pooling"].as_str() {
                Some("avg" | "mean") => st::MEAN,
                Some("cls") => st::CLS,
                Some("last") => st::LAST,
                _ if bidirectional => st::MEAN,
                _ => st::LAST,
            }
        })?,
        _ => st::LAST,
    };
    meta.push(("bert.pooling_type".into(), Meta::U32(pooling)));
    let (_, token_meta) = crate::convert::tokenizer(root, u64::from(vocab), tc)?;
    meta.extend(token_meta);
    let (prefix, suffix) = if kind == Kind::Reranker {
        (Vec::new(), Vec::new())
    } else {
        affixes(root, &meta)?
    };
    meta.push(("bert.prefix_ids".into(), Meta::U32s(prefix)));
    meta.push(("bert.suffix_ids".into(), Meta::U32s(suffix)));

    let mut sources: HashMap<String, Tensor> = safetensors::read_checkpoint(input)?
        .into_iter()
        .map(|t| (t.name.strip_prefix("model.").unwrap_or(&t.name).to_owned(), t))
        .collect();
    let mut tensors = Vec::new();
    let mut take = |sources: &mut HashMap<String, Tensor>,
                    src: &str,
                    dst: &str,
                    shape: &[u64],
                    add: f32,
                    required: bool|
     -> Result<(), CheckpointError> {
        match sources.remove(src) {
            Some(t) => {
                if t.shape != shape {
                    return Err(format!("tensor {src} has shape {:?}, expected {shape:?}", t.shape).into());
                }
                let mut out = st::out_tensor(t, dst.to_owned(), &options)?;
                out.add = add;
                tensors.push(out);
                Ok(())
            }
            None if required => Err(format!("missing required tensor {src}").into()),
            None => Ok(()),
        }
    };
    let h = u64::from(hidden);
    let qdim = u64::from(heads * head_dim);
    let kvdim = u64::from(kv_heads * head_dim);
    let one = if gemma { 1.0 } else { 0.0 };
    // Qwen3-Reranker: score = softmax([no, yes])[yes] = sigmoid((w_yes - w_no) . h) with the tied or
    // separate language-model head; keep the difference of the two rows as a single classifier row.
    let reranker_row = if kind == Kind::Reranker {
        let tokens = tokenizer_tokens(&meta);
        let yes = token_id(&tokens, "yes")?;
        let no = token_id(&tokens, "no")?;
        let head = sources
            .get("lm_head.weight")
            .or_else(|| sources.get("embed_tokens.weight"))
            .ok_or("reranker checkpoint has neither lm_head.weight nor embed_tokens.weight")?;
        if head.shape != [u64::from(vocab), h] {
            return Err("language-model head has an unexpected shape".into());
        }
        let y = decode_row(head, yes, hidden as usize)?;
        let n = decode_row(head, no, hidden as usize)?;
        Some(y.iter().zip(&n).map(|(a, b)| a - b).collect::<Vec<f32>>())
    } else {
        None
    };
    take(&mut sources, "embed_tokens.weight", "token_embd.weight", &[u64::from(vocab), h], 0.0, true)?;
    take(&mut sources, "norm.weight", "output_norm.weight", &[h], one, true)?;
    for i in 0..layers {
        let p = format!("layers.{i}");
        let b = format!("blk.{i}");
        take(&mut sources, &format!("{p}.input_layernorm.weight"), &format!("{b}.attn_norm.weight"), &[h], one, true)?;
        for (name, dst, rows) in [("q_proj", "attn_q", qdim), ("k_proj", "attn_k", kvdim), ("v_proj", "attn_v", kvdim)] {
            take(&mut sources, &format!("{p}.self_attn.{name}.weight"), &format!("{b}.{dst}.weight"), &[rows, h], 0.0, true)?;
            take(&mut sources, &format!("{p}.self_attn.{name}.bias"), &format!("{b}.{dst}.bias"), &[rows], 0.0, false)?;
        }
        take(&mut sources, &format!("{p}.self_attn.o_proj.weight"), &format!("{b}.attn_output.weight"), &[h, qdim], 0.0, true)?;
        if gemma || qwen3 {
            for (name, dst) in [("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")] {
                take(&mut sources, &format!("{p}.self_attn.{name}.weight"), &format!("{b}.{dst}.weight"), &[u64::from(head_dim)], one, true)?;
            }
        }
        if gemma {
            take(&mut sources, &format!("{p}.post_attention_layernorm.weight"), &format!("{b}.post_attn_norm.weight"), &[h], 1.0, true)?;
            take(&mut sources, &format!("{p}.pre_feedforward_layernorm.weight"), &format!("{b}.ffn_norm.weight"), &[h], 1.0, true)?;
            take(&mut sources, &format!("{p}.post_feedforward_layernorm.weight"), &format!("{b}.post_ffn_norm.weight"), &[h], 1.0, true)?;
        } else {
            take(&mut sources, &format!("{p}.post_attention_layernorm.weight"), &format!("{b}.ffn_norm.weight"), &[h], 0.0, true)?;
        }
        for (name, dst, shape) in [
            ("gate_proj", "ffn_gate", [u64::from(inter), h]),
            ("up_proj", "ffn_up", [u64::from(inter), h]),
            ("down_proj", "ffn_down", [h, u64::from(inter)]),
        ] {
            take(&mut sources, &format!("{p}.mlp.{name}.weight"), &format!("{b}.{dst}.weight"), &shape, 0.0, true)?;
        }
    }
    match kind {
        Kind::Embedder => {}
        Kind::SequenceClassifier => {
            let id2label = tc["id2label"].as_object();
            let labels = sources
                .get("score.weight")
                .and_then(|t| t.shape.first().copied())
                .ok_or("missing score.weight classification head")?;
            if labels == 0 || labels > 4096 {
                return Err("invalid classifier label count".into());
            }
            take(&mut sources, "score.weight", "classifier.weight", &[labels, h], 0.0, true)?;
            take(&mut sources, "score.bias", "classifier.bias", &[labels], 0.0, false)?;
            let names = (0..labels)
                .map(|i| {
                    id2label
                        .and_then(|m| m.get(&i.to_string()))
                        .and_then(Value::as_str)
                        .map_or_else(|| format!("LABEL_{i}"), str::to_owned)
                })
                .collect::<Vec<_>>();
            meta.push(("bert.classifier.label_count".into(), Meta::U32(labels as u32)));
            meta.push(("bert.classifier.labels".into(), Meta::Strs(names)));
        }
        Kind::Reranker => {
            let row = reranker_row.ok_or("missing reranker head")?;
            let bytes = row.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
            let len = bytes.len();
            tensors.push(OutTensor {
                name: "classifier.weight".into(),
                dims: vec![h, 1],
                source_dtype: "F32".into(),
                dtype: "F32".into(),
                data: Payload::Owned(Arc::new(bytes.clone()), 0..len),
                packed: None,
                data_len: len as u64,
                source_for_output: None,
                add: 0.0,
                owned: Some(bytes),
                parts: Vec::new(),
            });
            meta.push(("bert.classifier.label_count".into(), Meta::U32(1)));
            meta.push(("bert.classifier.labels".into(), Meta::Strs(vec!["relevance".into()])));
            for (key, value) in reranker_template() {
                meta.push((key.into(), Meta::Str(value.into())));
            }
        }
    }
    let skipped = sources.into_keys().collect::<Vec<_>>();
    st::extend(root, hidden, &options, &mut meta, &mut tensors)?;
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((meta, tensors, skipped))
}

fn tokenizer_tokens(meta: &[(String, Meta)]) -> Vec<String> {
    meta.iter()
        .find_map(|(k, v)| match v {
            Meta::Strs(s) if k == "tokenizer.ggml.tokens" => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn token_id(tokens: &[String], text: &str) -> Result<usize, CheckpointError> {
    tokens
        .iter()
        .position(|t| t == text)
        .ok_or_else(|| format!("tokenizer has no {text:?} token").into())
}

fn reranker_template() -> [(&'static str, &'static str); 4] {
    [
        ("bert.pair_template.prefix", RERANK_PREFIX),
        ("bert.pair_template.body", RERANK_BODY),
        ("bert.pair_template.suffix", RERANK_SUFFIX),
        ("bert.pair_template.instruction", RERANK_INSTRUCTION),
    ]
}
