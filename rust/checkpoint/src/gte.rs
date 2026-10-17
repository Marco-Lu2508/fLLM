//! GTE "new" encoders (`model_type: new`, remote code `NewModel`) exported in the BERT GGUF layout.
//!
//! Alibaba-NLP/gte-base-en-v1.5, gte-large-en-v1.5 and gte-multilingual-base are
//! BERT-style post-norm encoders with rotary positions (optionally NTK-scaled),
//! biased fused QKV, a gated feed-forward block (`act(gate) * up` with a fused
//! `up_gate_proj`) and an embedding LayerNorm. The loader in
//! `rust/engine/src/encoder.rs` reads `bert.model_type = gte_new`.
use crate::{
    bert,
    gguf::Meta,
    safetensors, st, CheckpointError, ConvertOptions,
};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

pub(crate) fn supports(cfg: &Value) -> bool {
    matches!(cfg["model_type"].as_str(), Some("new" | "gte"))
        || cfg["architectures"].as_array().is_some_and(|a| {
            a.iter().any(|v| {
                matches!(
                    v.as_str(),
                    Some("NewModel" | "GteModel" | "GteNewModel" | "NewForSequenceClassification")
                )
            })
        })
}

fn number(cfg: &Value, key: &str) -> Result<u32, CheckpointError> {
    cfg[key]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .filter(|&v| v > 0)
        .ok_or_else(|| format!("config.json requires positive {key}").into())
}

fn strip(name: &str) -> &str {
    name.strip_prefix("new.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name)
}

/// True when `tokenizer.json` describes a WordPiece model (English GTE) rather than SentencePiece.
fn wordpiece(root: &Path) -> bool {
    let path = root.join("tokenizer.json");
    if let Ok(bytes) = fs::read(&path) {
        if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
            return v["model"]["type"].as_str() == Some("WordPiece");
        }
    }
    root.join("vocab.txt").is_file()
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
    if cfg["position_embedding_type"].as_str().unwrap_or("rope") != "rope" {
        return Err("GTE conversion requires position_embedding_type rope".into());
    }
    if cfg["layer_norm_type"].as_str().unwrap_or("layer_norm") != "layer_norm" {
        return Err("GTE conversion requires layer_norm_type layer_norm".into());
    }
    if cfg["logn_attention_scale"].as_bool() == Some(true)
        || cfg["logn_attention_clip1"].as_bool() == Some(true)
    {
        return Err("GTE log-n attention scaling is not supported".into());
    }
    let hidden = number(cfg, "hidden_size")?;
    let heads = number(cfg, "num_attention_heads")?;
    let layers = number(cfg, "num_hidden_layers")?;
    let inter = number(cfg, "intermediate_size")?;
    let positions = number(cfg, "max_position_embeddings")?;
    let vocab = number(cfg, "vocab_size")?;
    if hidden % heads != 0
        || !(hidden / heads).is_multiple_of(2)
        || hidden > 8192
        || layers > 128
        || positions > 65536
        || vocab > 10_000_000
    {
        return Err("unsupported encoder dimensions".into());
    }
    let type_vocab = cfg["type_vocab_size"]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(1);
    if type_vocab > 16 {
        return Err("unsupported type vocabulary".into());
    }
    let eps = cfg["layer_norm_eps"].as_f64().unwrap_or(1e-12) as f32;
    if !eps.is_finite() || eps <= 0.0 {
        return Err("invalid layer norm epsilon".into());
    }
    let act = cfg["hidden_act"].as_str().unwrap_or("gelu");
    if !matches!(act, "gelu" | "gelu_new" | "gelu_fast" | "relu" | "silu" | "swish") {
        return Err(format!("unsupported GTE activation {act}").into());
    }
    let theta = cfg["rope_theta"].as_f64().unwrap_or(10000.0);
    if !theta.is_finite() || theta <= 0.0 {
        return Err("invalid rope theta".into());
    }
    let mut scaling = 1.0f32;
    if !cfg["rope_scaling"].is_null() {
        let kind = cfg["rope_scaling"]["type"].as_str().unwrap_or_default();
        if kind != "ntk" || !cfg["rope_scaling"]["mixed_b"].is_null() {
            return Err(format!("unsupported GTE rope scaling {kind} (only plain ntk)").into());
        }
        scaling = cfg["rope_scaling"]["factor"]
            .as_f64()
            .filter(|f| f.is_finite() && *f >= 1.0 && *f <= 64.0)
            .ok_or("invalid rope scaling factor")? as f32;
    }
    let pack = cfg["pack_qkv"].as_bool().unwrap_or(true);
    let mut meta = vec![
        ("general.architecture".into(), Meta::Str("bert".into())),
        (
            "general.name".into(),
            Meta::Str(
                root.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into(),
            ),
        ),
        ("bert.attention.causal".into(), Meta::Bool(false)),
        ("bert.attention.layer_norm_epsilon".into(), Meta::F32(eps)),
        ("bert.activation".into(), Meta::Str(act.into())),
    ];
    for (key, n) in [
        ("embedding_length", hidden),
        ("context_length", positions),
        ("attention.head_count", heads),
        ("feed_forward_length", inter),
        ("block_count", layers),
        ("position_offset", 0),
        ("token_type_count", type_vocab),
        (
            "pooling_type",
            st::pooling_or(root, || st::CLS)?,
        ),
    ] {
        meta.push((format!("bert.{key}"), Meta::U32(n)));
    }
    meta.push(("bert.rope_theta".into(), Meta::F32(theta as f32)));
    meta.push(("bert.rope_scaling_factor".into(), Meta::F32(scaling)));
    meta.extend(bert::tokenizer(root, vocab, cfg, wordpiece(root))?);
    meta.push(("bert.model_type".into(), Meta::Str("gte_new".into())));

    let sources = safetensors::read_checkpoint(input)?;
    let h = u64::from(hidden);
    let mut expected = HashMap::<String, (String, Vec<u64>)>::new();
    expected.insert(
        "embeddings.word_embeddings.weight".into(),
        ("token_embd.weight".into(), vec![u64::from(vocab), h]),
    );
    if type_vocab > 0 {
        expected.insert(
            "embeddings.token_type_embeddings.weight".into(),
            ("token_types.weight".into(), vec![u64::from(type_vocab), h]),
        );
    }
    for suffix in ["weight", "bias"] {
        expected.insert(
            format!("embeddings.LayerNorm.{suffix}"),
            (format!("token_embd_norm.{suffix}"), vec![h]),
        );
    }
    for i in 0..layers {
        let pre = format!("encoder.layer.{i}");
        let mut add = |src: String, dst: String, shape: Vec<u64>| {
            expected.insert(src, (dst, shape));
        };
        if pack {
            add(
                format!("{pre}.attention.qkv_proj.weight"),
                format!("blk.{i}.attn_qkv.weight"),
                vec![3 * h, h],
            );
            add(
                format!("{pre}.attention.qkv_proj.bias"),
                format!("blk.{i}.attn_qkv.bias"),
                vec![3 * h],
            );
        } else {
            for (p, d) in [("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v")] {
                add(
                    format!("{pre}.attention.{p}.weight"),
                    format!("blk.{i}.{d}.weight"),
                    vec![h, h],
                );
                add(
                    format!("{pre}.attention.{p}.bias"),
                    format!("blk.{i}.{d}.bias"),
                    vec![h],
                );
            }
        }
        add(
            format!("{pre}.attention.o_proj.weight"),
            format!("blk.{i}.attn_output.weight"),
            vec![h, h],
        );
        add(
            format!("{pre}.attention.o_proj.bias"),
            format!("blk.{i}.attn_output.bias"),
            vec![h],
        );
        add(
            format!("{pre}.mlp.up_gate_proj.weight"),
            format!("blk.{i}.ffn_up.weight"),
            vec![2 * u64::from(inter), h],
        );
        add(
            format!("{pre}.mlp.down_proj.weight"),
            format!("blk.{i}.ffn_down.weight"),
            vec![h, u64::from(inter)],
        );
        add(
            format!("{pre}.mlp.down_proj.bias"),
            format!("blk.{i}.ffn_down.bias"),
            vec![h],
        );
        for (src, dst) in [("attn_ln", "attn_output_norm"), ("mlp_ln", "layer_output_norm")] {
            for suffix in ["weight", "bias"] {
                add(
                    format!("{pre}.{src}.{suffix}"),
                    format!("blk.{i}.{dst}.{suffix}"),
                    vec![h],
                );
            }
        }
    }
    let mut tensors = Vec::new();
    let mut skipped = Vec::new();
    let mut seen = HashSet::new();
    for t in sources {
        let source = strip(&t.name).to_owned();
        let Some((name, shape)) = expected.get(&source) else {
            skipped.push(t.name);
            continue;
        };
        if &t.shape != shape {
            return Err(format!(
                "tensor {} has shape {:?}, expected {shape:?}",
                t.name, t.shape
            )
            .into());
        }
        if !seen.insert(name.clone()) {
            return Err(format!("duplicate encoder tensor {name}").into());
        }
        let name = name.clone();
        tensors.push(st::out_tensor(t, name, &options)?);
    }
    for (name, _) in expected.values() {
        if !seen.contains(name) {
            return Err(format!("missing required encoder tensor {name}").into());
        }
    }
    st::extend(root, hidden, &options, &mut meta, &mut tensors)?;
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((meta, tensors, skipped))
}
