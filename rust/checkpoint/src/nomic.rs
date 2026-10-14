//! NomicBERT encoder checkpoints exported in the BERT GGUF tensor layout.
//!
//! NomicBERT (`nomic_bert`, e.g. nomic-embed-text-v1/v1.5) is a BERT-vocabulary
//! encoder with a ModernBERT-style block: no learned position table (single
//! RoPE theta instead), separate bias-free Q/K/V/O projections, a SwiGLU
//! feed-forward block (`silu(gate) * up`, all bias-free) and biased LayerNorms
//! in post-norm arrangement (embedding norm plus two per layer). There is no
//! MoE in these checkpoints. The loader (`rust/engine/src/encoder.rs`) reads
//! the `bert.*` keys written here; `bert.rope_theta` carries the single RoPE
//! theta and `bert.model_type` is `nomic_bert`.
use crate::{
    bert,
    gguf::{self, Meta, OutTensor},
    safetensors, CheckpointError, ConversionReport, ConvertOptions, OutputDType,
};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

pub(crate) fn supports(cfg: &Value) -> bool {
    cfg["model_type"].as_str() == Some("nomic_bert")
}

fn number(cfg: &Value, key: &str) -> Result<u32, CheckpointError> {
    cfg[key]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .filter(|&v| v > 0)
        .ok_or_else(|| format!("config.json requires positive {key}").into())
}

fn json(path: &Path) -> Result<Value, CheckpointError> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|e| format!("invalid {}: {e}", path.display()).into())
}

fn pooling(root: &Path) -> Result<u32, CheckpointError> {
    let path = root.join("1_Pooling/config.json");
    if path.is_file() {
        let pool = json(&path)?;
        let cls = pool["pooling_mode_cls_token"].as_bool().unwrap_or(false);
        let mean = pool["pooling_mode_mean_tokens"].as_bool().unwrap_or(false);
        let unsupported = [
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens",
            "pooling_mode_lasttoken",
        ]
        .iter()
        .any(|k| pool[*k].as_bool() == Some(true));
        if unsupported || cls == mean {
            return Err("encoder conversion requires exactly one CLS or mean pooling mode".into());
        }
        return Ok(if cls { 2 } else { 1 });
    }
    // Nomic embedding checkpoints use mean pooling.
    Ok(1)
}

fn theta(cfg: &Value) -> Result<f32, CheckpointError> {
    let value = cfg["rope_parameters"]["rope_theta"]
        .as_f64()
        .or_else(|| cfg["rope_theta"].as_f64())
        .or_else(|| cfg["rotary_emb_base"].as_f64())
        .or_else(|| cfg["default_theta"].as_f64())
        .unwrap_or(1000.0);
    if !value.is_finite() || value <= 0.0 {
        return Err("invalid nomic rope theta".into());
    }
    Ok(value as f32)
}

fn strip(name: &str) -> &str {
    name.strip_prefix("nomic_bert.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name)
}

pub(crate) fn convert(
    input: &Path,
    output: &Path,
    options: ConvertOptions,
    cfg: &Value,
) -> Result<ConversionReport, CheckpointError> {
    let root = if input.is_dir() {
        input
    } else {
        input.parent().ok_or("checkpoint has no parent directory")?
    };
    if cfg["is_decoder"].as_bool() == Some(true)
        || cfg["add_cross_attention"].as_bool() == Some(true)
    {
        return Err(
            "encoder conversion requires bidirectional attention without cross-attention".into(),
        );
    }
    let hidden = number(cfg, "hidden_size")?;
    let heads = number(cfg, "num_attention_heads")?;
    let layers = number(cfg, "num_hidden_layers")?;
    let inter = number(cfg, "intermediate_size")?;
    let positions = number(cfg, "max_position_embeddings")?;
    let vocab = number(cfg, "vocab_size")?;
    let head_dim = cfg["head_dim"]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(hidden / heads);
    if hidden % heads != 0
        || head_dim * heads != hidden
        || !head_dim.is_multiple_of(2)
        || hidden > 8192
        || layers > 128
        || positions > 8192
        || vocab > 10_000_000
    {
        return Err("unsupported encoder dimensions".into());
    }
    let type_vocab = cfg["type_vocab_size"]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(2);
    if type_vocab > 16 {
        return Err("unsupported nomic type vocabulary".into());
    }
    let eps = cfg["layer_norm_eps"]
        .as_f64()
        .or_else(|| cfg["layer_norm_epsilon"].as_f64())
        .unwrap_or(1e-12) as f32;
    if !eps.is_finite() || eps <= 0.0 {
        return Err("invalid layer norm epsilon".into());
    }
    let act = cfg["hidden_act"].as_str().unwrap_or("silu");
    if !matches!(act, "silu" | "swiglu") {
        return Err(format!("unsupported nomic activation {act}").into());
    }
    let rope_theta = theta(cfg)?;
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
        ("bert.activation".into(), Meta::Str("silu".into())),
    ];
    for (key, n) in [
        ("embedding_length", hidden),
        ("context_length", positions),
        ("attention.head_count", heads),
        ("feed_forward_length", inter),
        ("block_count", layers),
        ("position_offset", 0),
        ("token_type_count", type_vocab),
        ("pooling_type", pooling(root)?),
    ] {
        meta.push((format!("bert.{key}"), Meta::U32(n)));
    }
    meta.push(("bert.rope_theta".into(), Meta::F32(rope_theta)));
    meta.extend(bert::tokenizer(root, vocab, cfg, true)?);
    meta.push(("bert.model_type".into(), Meta::Str("nomic_bert".into())));
    let sources = safetensors::read_checkpoint(input)?;
    // Real Nomic checkpoints (nomic-embed-text v1/v1.5) use the legacy
    // mosaic-style names; transformers converts them on load/save
    // (see `conversion_mapping.py`, key `nomic_bert`): `encoder.layers`
    // holds a fused `attn.Wqkv`, `mlp.fc11` is the up projection,
    // `mlp.fc12` the gate projection, `norm1/2` the post norms and `emb_ln`
    // the embedding norm. Freshly built models use the new names instead;
    // accept both, the `seen` set below guarantees exactly one variant wins.
    for t in &sources {
        let name = strip(&t.name);
        let legacy = name
            .strip_prefix("encoder.layers.")
            .unwrap_or(name);
        if legacy.starts_with("attn.Wqkv.bias")
            || legacy.starts_with("attn.out_proj.bias")
            || legacy.starts_with("mlp.fc11.bias")
            || legacy.starts_with("mlp.fc12.bias")
            || legacy.starts_with("mlp.fc2.bias")
            || name.starts_with("embeddings.position_embeddings.")
            || name.contains("self_attn") && name.ends_with("_proj.bias")
            || name.contains("mlp.")
                && (name.ends_with("_proj.bias") || name.ends_with("down_proj.bias"))
        {
            return Err(format!("unexpected nomic tensor {}", t.name).into());
        }
    }
    let mut expected = HashMap::<String, (String, Vec<u64>)>::new();
    // Embeddings: identical in both namings except the embedding norm.
    expected.insert(
        "embeddings.word_embeddings.weight".into(),
        (
            "token_embd.weight".into(),
            vec![vocab as u64, hidden as u64],
        ),
    );
    if type_vocab > 0 {
        expected.insert(
            "embeddings.token_type_embeddings.weight".into(),
            (
                "token_types.weight".into(),
                vec![type_vocab as u64, hidden as u64],
            ),
        );
    }
    for suffix in ["weight", "bias"] {
        expected.insert(
            format!("embeddings.LayerNorm.{suffix}"),
            (format!("token_embd_norm.{suffix}"), vec![hidden as u64]),
        );
        expected.insert(
            format!("emb_ln.{suffix}"),
            (format!("token_embd_norm.{suffix}"), vec![hidden as u64]),
        );
    }
    for i in 0..layers {
        // Fused legacy QKV, split by the loader (as with ModernBERT).
        expected.insert(
            format!("encoder.layers.{i}.attn.Wqkv.weight"),
            (
                format!("blk.{i}.attn_qkv.weight"),
                vec![3 * hidden as u64, hidden as u64],
            ),
        );
        // Separate projections of freshly built models.
        for (proj, dst) in [
            ("q_proj", "attn_q"),
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("o_proj", "attn_output"),
        ] {
            expected.insert(
                format!("layers.{i}.self_attn.{proj}.weight"),
                (
                    format!("blk.{i}.{dst}.weight"),
                    vec![hidden as u64, hidden as u64],
                ),
            );
        }
        expected.insert(
            format!("encoder.layers.{i}.attn.out_proj.weight"),
            (
                format!("blk.{i}.attn_output.weight"),
                vec![hidden as u64, hidden as u64],
            ),
        );
        // fc11 is the up projection, fc12 the gate projection.
        for (legacy, proj, dst) in [
            ("fc11", "up_proj", "ffn_up"),
            ("fc12", "gate_proj", "ffn_gate"),
            ("fc2", "down_proj", "ffn_down"),
        ] {
            let (rows, cols) = if dst == "ffn_down" {
                (hidden, inter)
            } else {
                (inter, hidden)
            };
            expected.insert(
                format!("encoder.layers.{i}.mlp.{legacy}.weight"),
                (
                    format!("blk.{i}.{dst}.weight"),
                    vec![rows as u64, cols as u64],
                ),
            );
            expected.insert(
                format!("layers.{i}.mlp.{proj}.weight"),
                (
                    format!("blk.{i}.{dst}.weight"),
                    vec![rows as u64, cols as u64],
                ),
            );
        }
        for (legacy, src, dst) in [
            ("norm1", "post_attention_layernorm", "attn_output_norm"),
            ("norm2", "post_mlp_layernorm", "layer_output_norm"),
        ] {
            for suffix in ["weight", "bias"] {
                expected.insert(
                    format!("encoder.layers.{i}.{legacy}.{suffix}"),
                    (
                        format!("blk.{i}.{dst}.{suffix}"),
                        vec![hidden as u64],
                    ),
                );
                expected.insert(
                    format!("layers.{i}.{src}.{suffix}"),
                    (
                        format!("blk.{i}.{dst}.{suffix}"),
                        vec![hidden as u64],
                    ),
                );
            }
        }
    }
    // Exactly one QKV variant may be present; the loader carries the same rule.
    let stripped: HashSet<String> = sources
        .iter()
        .map(|t| strip(&t.name).to_owned())
        .collect();
    for i in 0..layers {
        let fused = stripped.contains(&format!("encoder.layers.{i}.attn.Wqkv.weight"));
        let separate = ["q_proj", "k_proj", "v_proj"].iter().all(|proj| {
            stripped.contains(&format!("layers.{i}.self_attn.{proj}.weight"))
        });
        if fused == separate {
            return Err(format!(
                "nomic layer {i} requires either fused attn.Wqkv or separate q/k/v projections"
            )
            .into());
        }
        // Drop the losing variant so the required-tensor check below only
        // enforces the names of the variant actually present.
        if fused {
            for proj in ["q_proj", "k_proj", "v_proj"] {
                expected.remove(&format!("layers.{i}.self_attn.{proj}.weight"));
            }
        } else {
            expected.remove(&format!("encoder.layers.{i}.attn.Wqkv.weight"));
        }
    }
    let mut tensors = Vec::new();
    let mut skipped = Vec::new();
    let mut seen = HashSet::new();
    for t in sources {
        let source = strip(&t.name);
        let Some((name, shape)) = expected.get(source) else {
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
        if !matches!(t.dtype.as_str(), "F32" | "F16" | "BF16") {
            return Err(format!("unsupported encoder dtype {}", t.dtype).into());
        }
        let dtype = match options.output_dtype {
            OutputDType::F32 => "F32",
            OutputDType::F16 => "F16",
            OutputDType::Preserve => {
                if t.dtype == "BF16" {
                    "F16"
                } else {
                    &t.dtype
                }
            }
        }
        .to_owned();
        let count = t
            .shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .ok_or("encoder tensor size overflow")?;
        let data_len = count
            .checked_mul(if dtype == "F32" { 4 } else { 2 })
            .ok_or("encoder tensor byte size overflow")?;
        tensors.push(OutTensor {
            name: name.clone(),
            dims: t.shape.iter().rev().copied().collect(),
            source_dtype: t.dtype,
            dtype,
            data: t.payload,
            packed: None,
            data_len,
            source_for_output: None,
            add: 0.0,
            owned: None,
            parts: Vec::new(),
        });
    }
    for (name, _) in expected.values() {
        if !seen.contains(name) {
            return Err(format!("missing required encoder tensor {name}").into());
        }
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    if output.exists() && !options.overwrite {
        return Err(format!("output already exists: {}", output.display()).into());
    }
    if input.is_file() && fs::canonicalize(input).ok() == fs::canonicalize(output).ok() {
        return Err("input and output must differ".into());
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut tmp = output.as_os_str().to_owned();
    tmp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let tmp = std::path::PathBuf::from(tmp);
    let result = (|| -> Result<u64, CheckpointError> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        let size = gguf::write_to(&mut file, &meta, &tensors)?;
        file.sync_all()?;
        drop(file);
        if options.overwrite {
            fs::rename(&tmp, output)?;
        } else {
            fs::hard_link(&tmp, output)?;
            fs::remove_file(&tmp)?;
        }
        Ok(size)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    Ok(ConversionReport {
        architecture: "bert".into(),
        tensors_written: tensors.len(),
        bytes_written: result?,
        skipped_tensors: skipped,
        output: output.to_owned(),
    })
}
