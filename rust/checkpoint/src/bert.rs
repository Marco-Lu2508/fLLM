//! HF bidirectional encoders exported in the BERT GGUF tensor layout.
use crate::{
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
    matches!(
        cfg["model_type"].as_str(),
        Some("bert" | "roberta" | "xlm-roberta" | "distilbert" | "electra")
    )
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

fn pooling(root: &Path, cfg: &Value) -> Result<u32, CheckpointError> {
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
    let name = format!(
        "{} {}",
        root.display(),
        cfg["_name_or_path"].as_str().unwrap_or_default()
    )
    .to_lowercase();
    Ok(if name.contains("bge") {
        2
    } else if ["e5", "gte", "minilm"].iter().any(|s| name.contains(s)) {
        1
    } else {
        2
    })
}

pub(crate) fn tokenizer(
    root: &Path,
    vocab: u32,
    cfg: &Value,
    wordpiece: bool,
) -> Result<Vec<(String, Meta)>, CheckpointError> {
    if !wordpiece {
        let (_, mut meta) = crate::convert::tokenizer(root, vocab.into(), cfg)?;
        special_tokens(root, &mut meta, false)?;
        return Ok(meta);
    }
    let path = root.join("tokenizer.json");
    let tc = if root.join("tokenizer_config.json").is_file() {
        json(&root.join("tokenizer_config.json"))?
    } else {
        Value::Null
    };
    let mut tokens = vec![String::new(); vocab as usize];
    let mut types = vec![5; vocab as usize];
    let mut raw = None;
    let mut lower = tc["do_lower_case"].as_bool().unwrap_or(true);
    if path.is_file() {
        let text = fs::read_to_string(&path)?;
        let t: Value =
            serde_json::from_str(&text).map_err(|e| format!("invalid tokenizer.json: {e}"))?;
        if t["model"]["type"].as_str() != Some("WordPiece") {
            return Err("BERT tokenizer must use WordPiece".into());
        }
        fn lowercase(v: &Value) -> Option<bool> {
            v.get("lowercase").and_then(Value::as_bool).or_else(|| {
                v.get("normalizers")
                    .and_then(Value::as_array)
                    .and_then(|a| a.iter().find_map(lowercase))
            })
        }
        lower = lowercase(&t["normalizer"]).unwrap_or(lower);
        for (s, id) in t["model"]["vocab"]
            .as_object()
            .ok_or("missing WordPiece vocabulary")?
        {
            let id = id
                .as_u64()
                .filter(|&v| v < vocab as u64)
                .ok_or("WordPiece ID exceeds embedding vocabulary")? as usize;
            if types[id] != 5 {
                return Err("duplicate WordPiece ID".into());
            }
            tokens[id] = s.clone();
            types[id] = 1;
        }
        if let Some(added) = t["added_tokens"].as_array() {
            for a in added {
                let id = a["id"]
                    .as_u64()
                    .filter(|&v| v < vocab as u64)
                    .ok_or("added token ID exceeds embedding vocabulary")?
                    as usize;
                let s = a["content"].as_str().ok_or("invalid added token")?;
                if types[id] != 5 && tokens[id] != s {
                    return Err("added token conflicts with WordPiece vocabulary".into());
                }
                tokens[id] = s.into();
                types[id] = if a["special"].as_bool() == Some(true) {
                    3
                } else {
                    4
                };
            }
        }
        raw = Some(text);
    } else {
        for (id, s) in fs::read_to_string(root.join("vocab.txt"))?
            .lines()
            .enumerate()
        {
            if id >= tokens.len() {
                return Err("vocab.txt exceeds embedding vocabulary".into());
            }
            tokens[id] = s.into();
            types[id] = 1;
        }
    }
    let mut unique: HashSet<String> = tokens.iter().filter(|s| !s.is_empty()).cloned().collect();
    for (id, s) in tokens.iter_mut().enumerate() {
        if types[id] == 5 {
            *s = format!("<|fllm_unused_{id}|>");
            while unique.contains(s) {
                s.push('_');
            }
            unique.insert(s.clone());
        } else if s.is_empty() {
            return Err("empty WordPiece vocabulary token".into());
        }
    }
    if tokens.iter().collect::<HashSet<_>>().len() != tokens.len() {
        return Err("duplicate WordPiece vocabulary token".into());
    }
    let mut meta = vec![
        ("tokenizer.ggml.model".into(), Meta::Str("bert".into())),
        ("tokenizer.ggml.tokens".into(), Meta::Strs(tokens)),
        ("tokenizer.ggml.token_type".into(), Meta::U32s(types)),
        ("tokenizer.ggml.do_lower_case".into(), Meta::Bool(lower)),
    ];
    if let Some(raw) = raw {
        meta.push(("tokenizer.huggingface.json".into(), Meta::Str(raw)));
    }
    special_tokens(root, &mut meta, true)?;
    Ok(meta)
}

fn special_tokens(
    root: &Path,
    meta: &mut Vec<(String, Meta)>,
    wp: bool,
) -> Result<(), CheckpointError> {
    let mut tc = if root.join("tokenizer_config.json").is_file() {
        json(&root.join("tokenizer_config.json"))?
    } else {
        serde_json::json!({})
    };
    if root.join("special_tokens_map.json").is_file() {
        let sm = json(&root.join("special_tokens_map.json"))?;
        if let (Some(dst), Some(src)) = (tc.as_object_mut(), sm.as_object()) {
            for (k, v) in src {
                dst.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
    let tokens = meta
        .iter()
        .find_map(|(k, v)| match v {
            Meta::Strs(s) if k == "tokenizer.ggml.tokens" => Some(s.clone()),
            _ => None,
        })
        .ok_or("missing tokenizer vocabulary")?;
    for (field, key, default) in [
        (
            "cls_token",
            "cls_token_id",
            if wp { "[CLS]" } else { "<s>" },
        ),
        (
            "sep_token",
            "separator_token_id",
            if wp { "[SEP]" } else { "</s>" },
        ),
        (
            "pad_token",
            "padding_token_id",
            if wp { "[PAD]" } else { "<pad>" },
        ),
        (
            "unk_token",
            "unknown_token_id",
            if wp { "[UNK]" } else { "<unk>" },
        ),
    ] {
        let text = tc[field]
            .as_str()
            .or_else(|| tc[field]["content"].as_str())
            .unwrap_or(default);
        let id = tokens
            .iter()
            .position(|s| s == text)
            .ok_or_else(|| format!("missing special token {text}"))? as u32;
        let key = format!("tokenizer.ggml.{key}");
        meta.retain(|(k, _)| k != &key);
        meta.push((key, Meta::U32(id)));
        for (k, value) in meta.iter_mut() {
            if k == "tokenizer.ggml.token_type" {
                if let Meta::U32s(types) = value {
                    types[id as usize] = if field == "unk_token" { 2 } else { 3 };
                }
            }
        }
    }
    Ok(())
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
    let model_type = cfg["model_type"].as_str().ok_or("missing model_type")?;
    if cfg["is_decoder"].as_bool() == Some(true)
        || cfg["add_cross_attention"].as_bool() == Some(true)
    {
        return Err(
            "encoder conversion requires bidirectional attention without cross-attention".into(),
        );
    }
    let distil = model_type == "distilbert";
    let roberta = matches!(model_type, "roberta" | "xlm-roberta");
    let hidden = number(cfg, if distil { "dim" } else { "hidden_size" })?;
    let heads = number(
        cfg,
        if distil {
            "n_heads"
        } else {
            "num_attention_heads"
        },
    )?;
    let layers = number(
        cfg,
        if distil {
            "n_layers"
        } else {
            "num_hidden_layers"
        },
    )?;
    let inter = number(
        cfg,
        if distil {
            "hidden_dim"
        } else {
            "intermediate_size"
        },
    )?;
    let positions = number(cfg, "max_position_embeddings")?;
    let vocab = number(cfg, "vocab_size")?;
    if hidden % heads != 0 || hidden > 8192 || layers > 128 || vocab > 10_000_000 {
        return Err("unsupported encoder dimensions".into());
    }
    let emb = cfg["embedding_size"]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(hidden);
    if emb == 0 || (emb != hidden && model_type != "electra") {
        return Err("unsupported encoder embedding size".into());
    }
    let offset = if roberta {
        cfg["pad_token_id"]
            .as_u64()
            .unwrap_or(1)
            .checked_add(1)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or("position offset overflow")?
    } else {
        0
    };
    let context = positions
        .checked_sub(offset)
        .filter(|&n| n > 0)
        .ok_or("position offset exceeds position vocabulary")?;
    // BGE-M3 carries 8194 position rows for 8192 context positions (offset 2);
    // gate the usable context, not the raw table length.
    if context > 8192 {
        return Err("unsupported encoder dimensions".into());
    }
    let type_vocab = if distil {
        0
    } else {
        cfg["type_vocab_size"]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(if roberta { 1 } else { 2 })
    };
    let eps = cfg["layer_norm_eps"].as_f64().unwrap_or(1e-12) as f32;
    if !eps.is_finite() || eps <= 0.0 {
        return Err("invalid layer norm epsilon".into());
    }
    let act = cfg[if distil { "activation" } else { "hidden_act" }]
        .as_str()
        .unwrap_or("gelu");
    if !matches!(
        act,
        "gelu" | "gelu_new" | "gelu_fast" | "relu" | "silu" | "swish"
    ) {
        return Err(format!("unsupported encoder activation {act}").into());
    }
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
        ("context_length", context),
        ("attention.head_count", heads),
        ("feed_forward_length", inter),
        ("block_count", layers),
        ("position_offset", offset),
        ("token_type_count", type_vocab),
        ("pooling_type", pooling(root, cfg)?),
    ] {
        meta.push((format!("bert.{key}"), Meta::U32(n)));
    }
    meta.extend(tokenizer(root, vocab, cfg, !roberta)?);
    meta.push(("bert.model_type".into(), Meta::Str(model_type.into())));
    let sources = safetensors::read_checkpoint(input)?;
    let head_name = if roberta {
        "classifier.out_proj.weight"
    } else {
        "classifier.weight"
    };
    let head = sources.iter().find(|t| t.name == head_name);
    let wants_head = cfg["architectures"].as_array().is_some_and(|a| {
        a.iter().any(|v| {
            v.as_str()
                .is_some_and(|v| v.ends_with("ForSequenceClassification"))
        })
    });
    if (head.is_some() || wants_head) && !matches!(model_type, "bert" | "roberta" | "xlm-roberta") {
        return Err(format!(
            "sequence classification conversion is not implemented for {model_type}"
        )
        .into());
    }
    let num_labels = if let Some(t) = head {
        let labels = t
            .shape
            .first()
            .copied()
            .filter(|v| *v > 0 && *v <= 4096)
            .ok_or("invalid classifier label count")?;
        if cfg
            .get("num_labels")
            .is_some_and(|v| v.as_u64() != Some(labels))
            || cfg.get("id2label").is_some_and(|v| {
                !v.is_null() && v.as_object().is_none_or(|m| m.len() as u64 != labels)
            })
        {
            return Err("classifier label count disagrees with config.json".into());
        }
        Some(labels)
    } else if wants_head {
        return Err(format!("missing required classifier tensor {head_name}").into());
    } else {
        None
    };
    let mut expected = HashMap::<String, (String, Vec<u64>)>::new();
    let pooler = sources
        .iter()
        .any(|t| t.name.ends_with("pooler.dense.weight"));
    if pooler || num_labels.is_some() && !roberta {
        for suffix in ["weight", "bias"] {
            expected.insert(
                format!("pooler.dense.{suffix}"),
                (
                    format!("pooler.{suffix}"),
                    if suffix == "weight" {
                        vec![hidden as u64, hidden as u64]
                    } else {
                        vec![hidden as u64]
                    },
                ),
            );
        }
    }
    if let Some(labels) = num_labels {
        meta.push((
            "bert.classifier.label_count".into(),
            Meta::U32(labels as u32),
        ));
        meta.push((
            "bert.classifier.kind".into(),
            Meta::Str(if roberta { "roberta" } else { "bert" }.into()),
        ));
        let names = (0..labels)
            .map(|id| {
                if cfg["id2label"].is_null() {
                    Ok(format!("LABEL_{id}"))
                } else {
                    cfg["id2label"][id.to_string()]
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| format!("config.json id2label is missing string label {id}"))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        meta.push(("bert.classifier.labels".into(), Meta::Strs(names)));
        for suffix in ["weight", "bias"] {
            let src = if roberta {
                format!("classifier.out_proj.{suffix}")
            } else {
                format!("classifier.{suffix}")
            };
            expected.insert(
                src,
                (
                    format!("classifier.{suffix}"),
                    if suffix == "weight" {
                        vec![labels, hidden as u64]
                    } else {
                        vec![labels]
                    },
                ),
            );
            if roberta {
                expected.insert(
                    format!("classifier.dense.{suffix}"),
                    (
                        format!("classifier_dense.{suffix}"),
                        if suffix == "weight" {
                            vec![hidden as u64, hidden as u64]
                        } else {
                            vec![hidden as u64]
                        },
                    ),
                );
            }
        }
    }

    for (src, dst, shape) in [
        (
            "word_embeddings.weight",
            "token_embd.weight",
            vec![vocab as u64, emb as u64],
        ),
        (
            "position_embeddings.weight",
            "position_embd.weight",
            vec![positions as u64, emb as u64],
        ),
        (
            "LayerNorm.weight",
            "token_embd_norm.weight",
            vec![emb as u64],
        ),
        ("LayerNorm.bias", "token_embd_norm.bias", vec![emb as u64]),
    ] {
        expected.insert(format!("embeddings.{src}"), (dst.into(), shape));
    }
    if type_vocab > 0 {
        expected.insert(
            "embeddings.token_type_embeddings.weight".into(),
            (
                "token_types.weight".into(),
                vec![type_vocab as u64, emb as u64],
            ),
        );
    }
    if emb != hidden {
        for suffix in ["weight", "bias"] {
            expected.insert(
                format!("embeddings_project.{suffix}"),
                (
                    format!("token_embd_project.{suffix}"),
                    if suffix == "weight" {
                        vec![hidden as u64, emb as u64]
                    } else {
                        vec![hidden as u64]
                    },
                ),
            );
        }
    }
    let pairs = if distil {
        vec![
            ("attention.q_lin", "attn_q"),
            ("attention.k_lin", "attn_k"),
            ("attention.v_lin", "attn_v"),
            ("attention.out_lin", "attn_output"),
            ("sa_layer_norm", "attn_output_norm"),
            ("ffn.lin1", "ffn_up"),
            ("ffn.lin2", "ffn_down"),
            ("output_layer_norm", "layer_output_norm"),
        ]
    } else {
        vec![
            ("attention.self.query", "attn_q"),
            ("attention.self.key", "attn_k"),
            ("attention.self.value", "attn_v"),
            ("attention.output.dense", "attn_output"),
            ("attention.output.LayerNorm", "attn_output_norm"),
            ("intermediate.dense", "ffn_up"),
            ("output.dense", "ffn_down"),
            ("output.LayerNorm", "layer_output_norm"),
        ]
    };
    for i in 0..layers {
        for (src, dst) in &pairs {
            for suffix in ["weight", "bias"] {
                let rows = if *dst == "ffn_up" { inter } else { hidden } as u64;
                let cols = if *dst == "ffn_down" { inter } else { hidden } as u64;
                let shape = if suffix == "bias" || dst.ends_with("norm") {
                    vec![rows]
                } else {
                    vec![rows, cols]
                };
                expected.insert(
                    format!(
                        "{}.layer.{i}.{src}.{suffix}",
                        if distil { "transformer" } else { "encoder" }
                    ),
                    (format!("blk.{i}.{dst}.{suffix}"), shape),
                );
            }
        }
    }
    let mut tensors = Vec::new();
    let mut skipped = Vec::new();
    let mut seen = HashSet::new();
    for t in sources {
        let source = t
            .name
            .strip_prefix(&format!("{model_type}."))
            .or_else(|| t.name.strip_prefix("roberta."))
            .unwrap_or(&t.name);
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
