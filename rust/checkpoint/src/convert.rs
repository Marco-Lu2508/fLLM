use crate::{
    gguf::{self, Meta, OutTensor},
    packed_quant::{Fp8Format, PackedData, ScaleType},
    plan::{Family, Mapped, Plan},
    safetensors::{self, Tensor},
    CheckpointError,
};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputDType {
    Preserve,
    F16,
    F32,
}
#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub output_dtype: OutputDType,
    pub overwrite: bool,
}
impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            output_dtype: OutputDType::Preserve,
            overwrite: false,
        }
    }
}
#[derive(Clone, Debug)]
pub struct ConversionReport {
    pub architecture: String,
    pub tensors_written: usize,
    pub bytes_written: u64,
    pub skipped_tensors: Vec<String>,
    pub output: PathBuf,
}
#[derive(Clone, Debug)]
pub struct TensorSummary {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
}
#[derive(Clone, Debug)]
pub struct CheckpointInventory {
    pub tensors: Vec<TensorSummary>,
    pub formats: Vec<String>,
}

/// List safe dense tensor headers without loading tensor payloads into heap buffers.
pub fn inspect_checkpoint(input: &Path) -> Result<CheckpointInventory, CheckpointError> {
    let tensors = safetensors::read_checkpoint(input)?;
    let safetensors_present = if input.is_file() {
        input.extension().is_some_and(|x| x == "safetensors")
    } else {
        input.join("model.safetensors").is_file()
            || input.join("model.safetensors.index.json").is_file()
            || std::fs::read_dir(input)
                .ok()
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
    };
    let format = if safetensors_present {
        "safetensors"
    } else {
        "pytorch_zip"
    };
    Ok(CheckpointInventory {
        tensors: tensors
            .into_iter()
            .map(|t| TensorSummary {
                name: t.name,
                dtype: t.dtype,
                shape: t.shape,
            })
            .collect(),
        formats: vec![format.into()],
    })
}

/// What `check_config` found out about a checkpoint without reading any weights.
#[derive(Clone, Debug)]
pub struct ConfigCheck {
    /// GGUF architecture name the checkpoint converts to (`llama`, `qwen3`, ...).
    pub architecture: String,
    /// `none` for unquantized checkpoints, else `awq`, `gptq`, `fp8`, `nvfp4`, `int4`, `int8`, `exl2` or `exl3`.
    pub quantization: String,
}

/// Read only `config.json` (and `hf_quant_config.json`) of a checkpoint directory and report whether `convert` supports
/// its architecture and quantization. Fails with the reason when it does not, before any weights are fetched.
pub fn check_config(root: &Path) -> Result<ConfigCheck, CheckpointError> {
    let cfg_path = root.join("config.json");
    let mut cfg: Value =
        serde_json::from_slice(&fs::read(&cfg_path).map_err(|e| format!("read config.json: {e}"))?)
            .map_err(|e| format!("invalid config.json: {e}"))?;
    if let Some(q) = modelopt_quant_config(root)? {
        if cfg.get("quantization_config").is_none_or(Value::is_null) {
            cfg["quantization_config"] = q;
        }
    }
    let quantization = match quantization(&cfg)? {
        Quantization::Dense => "none",
        Quantization::Exl2 => "exl2",
        Quantization::Exl3 => "exl3",
        Quantization::Gptq { .. } => "gptq",
        Quantization::Awq { .. } => "awq",
        Quantization::Fp8 => "fp8",
        Quantization::Nvfp4 => "nvfp4",
        Quantization::CtPacked { bits: 4, .. } => "int4",
        Quantization::CtPacked { .. } | Quantization::CtInt8 { .. } => "int8",
    }
    .to_owned();
    let architecture = if crate::bert::supports(&cfg) {
        "bert".to_owned()
    } else if crate::modernbert::supports(&cfg) {
        "modernbert".to_owned()
    } else if crate::nomic::supports(&cfg) {
        "nomic-bert".to_owned()
    } else {
        Plan::from_config(&flatten_text_config(cfg))?.arch
    };
    Ok(ConfigCheck {
        architecture,
        quantization,
    })
}

pub fn convert_checkpoint(
    input: &Path,
    output: &Path,
    options: ConvertOptions,
) -> Result<ConversionReport, CheckpointError> {
    let root = if input.is_dir() {
        input
    } else {
        input
            .parent()
            .ok_or("checkpoint file has no parent directory")?
    };
    let cfg_path = root.join("config.json");
    let mut cfg: Value =
        serde_json::from_slice(&fs::read(&cfg_path).map_err(|e| format!("read config.json: {e}"))?)
            .map_err(|e| format!("invalid config.json: {e}"))?;
    if let Some(q) = modelopt_quant_config(root)? {
        if cfg.get("quantization_config").is_none_or(Value::is_null) {
            cfg["quantization_config"] = q;
        }
    }
    if crate::bert::supports(&cfg) {
        return crate::bert::convert(input, output, options, &cfg);
    }
    if crate::modernbert::supports(&cfg) {
        return crate::modernbert::convert(input, output, options, &cfg);
    }
    if crate::nomic::supports(&cfg) {
        return crate::nomic::convert(input, output, options, &cfg);
    }
    let quant = quantization(&cfg)?;
    let cfg = flatten_text_config(cfg);
    let plan = Plan::from_config(&cfg)?;
    let arch = plan.arch.as_str();
    let cfg = if let Some(classic) = &plan.classic {
        // standard keys for the shape checks, original keys for the tokenizer
        let mut merged = cfg.clone();
        if let (Some(m), Some(n)) = (merged.as_object_mut(), classic.normalized.as_object()) {
            for (k, v) in n {
                m.insert(k.clone(), v.clone());
            }
        }
        merged
    } else if matches!(arch, "lfm2" | "lfm2moe") {
        crate::plan::lfm2_config(&cfg)?
    } else if arch == "dbrx" {
        crate::plan::dbrx_config(&cfg)?
    } else if matches!(plan.family, Family::Mamba | Family::Nemotron) {
        // no attention or feed-forward sizes: give the shape checks harmless values
        let mut patched = cfg.clone();
        for key in [
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
        ] {
            if patched.get(key).is_none() {
                patched[key] = 1.into();
            }
        }
        patched["num_hidden_layers"] = plan.layers.into();
        if patched.get("head_dim").is_none() {
            patched["head_dim"] = cfg["hidden_size"].clone();
        }
        patched
    } else {
        cfg
    };
    let hidden = num(&cfg, "hidden_size", None)?;
    let heads = num(&cfg, "num_attention_heads", None)?;
    let layers = num(&cfg, "num_hidden_layers", None)?;
    let inter = num(&cfg, "intermediate_size", None)?;
    let inter = if arch == "llama4" {
        num(&cfg, "intermediate_size_mlp", Some(inter))?
    } else {
        inter
    };
    let inter = if arch == "granitehybrid" && !plan.moe {
        // the shared MLP is the only feed-forward of a dense Granite hybrid
        num(&cfg, "shared_intermediate_size", Some(inter))?
    } else {
        inter
    };
    let kv_heads = num(&cfg, "num_key_value_heads", Some(heads))?;
    let vocab = num(&cfg, "vocab_size", Some(32000))?;
    let head_dim = num(&cfg, "head_dim", Some(hidden / heads))?;
    if vocab > u32::MAX as u64 {
        return Err("config dimension vocab_size exceeds GGUF's u32 metadata range".into());
    }
    let mut meta = plan.meta.clone();
    if let Some(q) = cfg.get("quantization_config").filter(|q| !q.is_null()) {
        meta.push((
            "fllm.source.quantization_config".into(),
            Meta::Str(q.to_string()),
        ));
    }
    let tie = cfg
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let (_real_token_count, tokenmeta) = tokenizer(root, vocab, &cfg)?;
    meta.extend(tokenmeta);
    let all = safetensors::read_checkpoint(input)?;
    let fp8_block = fp8_block_size(&cfg);
    let (dense, packed_groups) = split_packed(all, quant, fp8_block, &|n| {
        Ok(match plan.map(n)? {
            Mapped::Skip => None,
            Mapped::Tensor { name, .. } => Some(name),
            Mapped::Transposed { .. } => {
                return Err(format!(
                    "packed tensor {n}: transposed architecture mappings are unsupported"
                ))
            }
            Mapped::Expert { name, index } => Some(format!("{name}.exl_expert.{index}")),
            Mapped::Fused { .. } => None,
        })
    })?;
    let expected = |name: &str| {
        if plan.family == Family::Classic {
            return Ok(None);
        }
        // Phi-3 fuses gate and up into one `ffn_up`
        if matches!(arch, "phi3" | "glm" | "glm4") && name.ends_with(".ffn_up.weight") {
            return Ok(None);
        }
        if name.contains(".nextn.") {
            return Ok(None);
        }
        // MLA attention has its own projection shapes
        if matches!(
            arch,
            "deepseek2" | "minicpm3" | "qwen3next" | "g9v3" | "mamba2" | "nemotron_h" | "nemotron_h_moe"
        ) && name.contains(".attn_")
        {
            return Ok(None);
        }
        expected_shape(name, hidden, inter, heads, kv_heads, head_dim, vocab)
    };
    let mut seen = HashSet::new();
    let mut outs = Vec::new();
    let mut skipped = Vec::new();
    let mut experts: std::collections::BTreeMap<String, Vec<(usize, Tensor)>> = Default::default();
    let mut fused: std::collections::BTreeMap<u64, Vec<(String, Tensor)>> = Default::default();
    for mut t in dense {
        if arch == "lfm2moe"
            && t.name.contains(".feed_forward.experts.")
            && (t.name.ends_with("gate_up_proj") || t.name.ends_with("down_proj"))
        {
            let e = num(&cfg, "num_experts", None)?;
            let f = num(&cfg, "moe_intermediate_size", None)?;
            let expected = if t.name.ends_with("gate_up_proj") {
                vec![
                    e,
                    f.checked_mul(2).ok_or("LFM2-MoE expert width overflow")?,
                    hidden,
                ]
            } else if t.name.ends_with("down_proj") {
                vec![e, hidden, f]
            } else {
                return Err(format!("unrecognized LFM2-MoE expert tensor {}", t.name).into());
            };
            if t.shape != expected {
                return Err(format!(
                    "LFM2-MoE tensor {} has shape {:?}, expected {:?}",
                    t.name, t.shape, expected
                )
                .into());
            }
        }
        if matches!(arch, "lfm2" | "lfm2moe") && t.name.ends_with(".conv.conv.weight") {
            let expected = vec![hidden, 1, num(&cfg, "conv_L_cache", Some(3))?];
            if t.shape != expected {
                return Err(format!(
                    "LFM2 tensor {} has shape {:?}, expected {:?}",
                    t.name, t.shape, expected
                )
                .into());
            }
        }
        if arch == "dbrx" && t.name.contains(".ffn.experts.mlp.") {
            let experts = num(&cfg, "num_experts", None)?;
            let expected = vec![
                experts
                    .checked_mul(inter)
                    .ok_or("DBRX expert size overflow")?,
                hidden,
            ];
            if t.shape != expected {
                return Err(format!(
                    "DBRX tensor {} has shape {:?}, expected {:?}",
                    t.name, t.shape, expected
                )
                .into());
            }
            // Safetensors [E*F,H] is contiguous [E,F,H]; w2 is transposed by split_chunked.
            t.shape = vec![experts, inter, hidden];
        }
        let (name, add) = match plan.map(&t.name)? {
            Mapped::Skip => {
                skipped.push(t.name);
                continue;
            }
            Mapped::Tensor { name, add } => (name, add),
            Mapped::Transposed { name } => {
                if tie && name == "output.weight" {
                    skipped.push(t.name);
                    continue;
                }
                let dtype = output_dtype(&t.dtype, options.output_dtype)?;
                let [rows, cols] = t.shape[..] else {
                    return Err(format!("{} is not a matrix", t.name).into());
                };
                let (rows, cols) = (rows as usize, cols as usize);
                let w = match t.dtype.as_str() {
                    "F32" => 4,
                    "F16" | "BF16" => 2,
                    other => return Err(format!("unsupported dtype {other} for {}", t.name).into()),
                };
                let data = t.payload.as_slice();
                let mut bytes =
                    Vec::with_capacity(rows * cols * if dtype == "F32" { 4 } else { 2 });
                for j in 0..cols {
                    for i in 0..rows {
                        let at = (i * cols + j) * w;
                        let v = crate::gguf::to_f32(&t.dtype, &data[at..at + w])
                            .map_err(|e| e.to_string())?;
                        crate::gguf::append_converted(&mut bytes, &dtype, v)
                            .map_err(|e| e.to_string())?;
                    }
                }
                if !seen.insert(name.clone()) {
                    return Err(format!("multiple HF tensors map to GGUF name {name}").into());
                }
                outs.push(OutTensor {
                    name,
                    dims: vec![rows as u64, cols as u64],
                    source_dtype: dtype.clone(),
                    dtype,
                    data: safetensors::Payload::Owned(std::sync::Arc::new(Vec::new()), 0..0),
                    packed: None,
                    data_len: bytes.len() as u64,
                    source_for_output: None,
                    add: 0.0,
                    owned: Some(bytes),
                    parts: Vec::new(),
                });
                continue;
            }
            Mapped::Expert { name, index } => {
                experts.entry(name).or_default().push((index, t));
                continue;
            }
            Mapped::Fused { layer, part } => {
                fused.entry(layer).or_default().push((part, t));
                continue;
            }
        };
        if tie && name == "output.weight" {
            skipped.push(t.name);
            continue;
        }
        if !seen.insert(name.clone()) {
            return Err(format!("multiple HF tensors map to GGUF name {name}").into());
        }
        let expected = expected(&name)?;
        if let Some(e) = expected {
            if t.shape != e {
                return Err(format!(
                    "tensor {} has shape {:?}, expected {:?}",
                    t.name, t.shape, e
                )
                .into());
            }
        }
        if matches!(t.dtype.as_str(), "F8_E4M3" | "F8_E5M2") {
            if !name.ends_with(".weight") {
                return Err(format!("FP8 tensor {} is not a weight matrix", t.name).into());
            }
            let count = usize::try_from(element_count(&t.shape, &t.name)?)
                .map_err(|_| "FP8 element count does not fit usize")?;
            let format = if t.dtype == "F8_E4M3" {
                Fp8Format::E4M3Fn
            } else {
                Fp8Format::E5M2
            };
            let packed = PackedData::fp8(t.payload.clone(), format, count)?;
            let out_dtype = match options.output_dtype {
                OutputDType::F32 => "F32",
                _ => "F16",
            }
            .to_string();
            let bytes = (count as u64)
                .checked_mul(if out_dtype == "F32" { 4 } else { 2 })
                .ok_or("FP8 output size overflow")?;
            let source_for_output = row_permutation(arch, &name, heads, kv_heads, head_dim)?;
            let dims = t.shape.iter().rev().copied().collect();
            outs.push(OutTensor {
                name,
                dims,
                source_dtype: "F32".into(),
                dtype: out_dtype,
                data: t.payload,
                packed: Some(std::sync::Arc::new(packed)),
                data_len: bytes,
                source_for_output,
                add: 0.0,
                owned: None,
                parts: Vec::new(),
            });
            continue;
        }
        let source_dtype = t.dtype.clone();
        let dtype = output_dtype(&source_dtype, options.output_dtype)?;
        let count = t.shape.iter().try_fold(1u64, |n, &d| {
            n.checked_mul(d).ok_or_else(|| {
                CheckpointError::Invalid(format!("tensor {} shape overflow", t.name))
            })
        })?;
        let width = match dtype.as_str() {
            "F32" => 4u64,
            "F16" | "BF16" => 2u64,
            _ => return Err(format!("unsupported output dtype {dtype}").into()),
        };
        let data_len = count.checked_mul(width).ok_or_else(|| {
            CheckpointError::Invalid(format!("tensor {} output byte count overflow", t.name))
        })?;
        let source_width = match source_dtype.as_str() {
            "F32" => 4u64,
            "F16" | "BF16" => 2u64,
            _ => return Err(format!("unsupported source dtype {source_dtype}").into()),
        };
        let source_len = count.checked_mul(source_width).ok_or_else(|| {
            CheckpointError::Invalid(format!("tensor {} source byte count overflow", t.name))
        })?;
        if t.payload.as_slice().len() as u64 != source_len {
            return Err(format!(
                "tensor {} payload length does not match shape and dtype",
                t.name
            )
            .into());
        }
        let source_for_output = match &plan.classic {
            Some(c) if name.contains(".attn_qkv.") => crate::classic::qkv_order(
                c.qkv,
                heads as usize,
                kv_heads as usize,
                head_dim as usize,
            ),
            _ => row_permutation(arch, &name, heads, kv_heads, head_dim)?,
        };
        let mut dims: Vec<u64> = t.shape.iter().rev().copied().collect();
        if name.ends_with("exp_probs_b.bias") {
            // some checkpoints store the selection bias as [1, experts]
            dims = vec![count];
        }
        outs.push(OutTensor {
            name,
            dims,
            source_dtype,
            dtype,
            data: t.payload,
            packed: None,
            data_len,
            source_for_output,
            add,
            owned: None,
            parts: Vec::new(),
        });
    }
    for (layer, parts) in fused {
        let dims = plan.linear.unwrap_or(crate::plan::LinearDims {
            k_heads: 1,
            v_heads: 1,
            k_dim: 1,
            v_dim: 1,
        });
        let (xielu, parts): (Vec<_>, Vec<_>) = parts
            .into_iter()
            .partition(|(p, _)| p.starts_with("xielu."));
        let (decay, rest): (Vec<_>, Vec<_>) = parts
            .into_iter()
            .partition(|(p, _)| p == "A_log" || p.starts_with("lin_") || p.starts_with("conv_"));
        let mut split = Vec::new();
        if !xielu.is_empty() {
            split.extend(crate::gptoss::split_xielu(layer, xielu)?);
        }
        if !decay.is_empty() {
            split.extend(crate::deltanet::split_linear(
                layer,
                decay,
                dims,
                options.output_dtype,
            )?);
        }
        if !rest.is_empty() {
            split.extend(match plan.family {
                Family::Qwen3Next => {
                    crate::deltanet::split_linear(layer, rest, dims, options.output_dtype)?
                }
                Family::GptOss => crate::gptoss::split_experts(layer, rest, options.output_dtype)?,
                _ => crate::gptoss::split_chunked(layer, rest, options.output_dtype)?,
            });
        }
        for out in split {
            seen.insert(out.name.clone());
            outs.push(out);
        }
    }
    for (name, mut list) in experts {
        list.sort_by_key(|(i, _)| *i);
        if list.iter().enumerate().any(|(n, (i, _))| n != *i) {
            return Err(format!("experts of {name} are not numbered 0..{}", list.len()).into());
        }
        let first = &list[0].1;
        if list
            .iter()
            .any(|(_, t)| t.shape != first.shape || t.dtype != first.dtype)
        {
            return Err(format!("experts of {name} differ in shape or dtype").into());
        }
        let source_dtype = first.dtype.clone();
        let dtype = output_dtype(&source_dtype, options.output_dtype)?;
        let count = first.shape.iter().product::<u64>();
        let width = if dtype == "F32" { 4 } else { 2 };
        let mut dims: Vec<u64> = first.shape.iter().rev().copied().collect();
        dims.push(list.len() as u64);
        let data_len = count * width * list.len() as u64;
        let mut payloads = list.into_iter().map(|(_, t)| t.payload);
        let data = payloads.next().expect("at least one expert");
        seen.insert(name.clone());
        outs.push(OutTensor {
            name,
            dims,
            source_dtype,
            dtype,
            data,
            packed: None,
            data_len,
            source_for_output: None,
            add: 0.0,
            owned: None,
            parts: payloads.collect(),
        });
    }
    let mut regular = Vec::new();
    let mut packed_experts = std::collections::BTreeMap::<
        String,
        Vec<(usize, PackedData, Vec<u64>, safetensors::Payload)>,
    >::new();
    for (name, data, dims, payload) in packed_groups {
        if let Some((base, index)) = name.rsplit_once(".exl_expert.") {
            let index = index
                .parse::<usize>()
                .map_err(|_| "invalid packed expert index")?;
            packed_experts
                .entry(base.into())
                .or_default()
                .push((index, data, dims, payload));
        } else {
            regular.push((name, data, dims, payload));
        }
    }
    for (name, mut parts) in packed_experts {
        parts.sort_by_key(|p| p.0);
        let mut dims = parts.first().ok_or("empty packed expert group")?.2.clone();
        if parts
            .iter()
            .enumerate()
            .any(|(i, p)| p.0 != i || p.2 != dims)
        {
            return Err(format!(
                "packed experts of {name} must have matching shapes and contiguous indices"
            )
            .into());
        }
        let elements = parts
            .iter()
            .try_fold(0usize, |n, p| n.checked_add(p.1.element_count()))
            .ok_or("packed expert size overflow")?;
        dims.insert(0, parts.len() as u64);
        let payload = parts[0].3.clone();
        regular.push((
            name,
            PackedData::Concat {
                parts: parts.into_iter().map(|p| p.1).collect(),
                elements,
            },
            dims,
            payload,
        ));
    }
    for (name, packed, dims, payload) in regular {
        if tie && name == "output.weight" {
            continue;
        }
        if !seen.insert(name.clone()) {
            return Err(format!("multiple HF tensors map to GGUF name {name}").into());
        }
        let expected = expected(&name)?;
        if let Some(e) = expected {
            if dims != e {
                return Err(
                    format!("packed tensor {name} has shape {dims:?}, expected {e:?}").into(),
                );
            }
        }
        let count = packed.element_count();
        // By default 4-bit AWQ / GPTQ weights stay 4-bit (native Q4Z/Q4ZP rows); `--dtype f16|f32` expands them.
        let q4z = if options.output_dtype == OutputDType::Preserve {
            packed.q4z_group()
        } else {
            None
        };
        let q4zp = if options.output_dtype == OutputDType::Preserve && q4z.is_none() {
            packed.q4zp_group()
        } else {
            None
        };
        let preserve = options.output_dtype == OutputDType::Preserve;
        let fp8m = if preserve && q4z.is_none() && q4zp.is_none() {
            packed.is_fp8m()
        } else {
            false
        };
        // NVFP4 stays GGUF type 40 when the rows fit 64-value blocks, 8-bit integer codes stay Q8_0
        let nvfp4 = preserve && packed.nvfp4().is_some_and(|t| t.fits_gguf_blocks());
        let q8_0 = preserve && packed.sym_int().is_some_and(|t| t.q8_0_ok());
        let in_features = dims.get(1).copied().unwrap_or(0) as usize;
        let rows_out = dims.first().copied().unwrap_or(0) as usize;
        let out_dtype = match (q4z, q4zp, fp8m, options.output_dtype) {
            _ if nvfp4 => "NVFP4".to_string(),
            _ if q8_0 => "Q8_0".to_string(),
            (Some(g), _, _, _) => format!("Q4Z{g}"),
            (None, Some(g), _, _) => format!("Q4ZP{g}"),
            (None, None, true, _) => "FP8M".to_string(),
            (None, None, false, OutputDType::F32) => "F32".to_string(),
            (None, None, false, _) => "F16".to_string(),
        };
        let data_len = match (q4z, q4zp, fp8m) {
            _ if nvfp4 => (rows_out * (in_features / 64 * 36)) as u64,
            _ if q8_0 => (rows_out * (in_features / 32 * 34)) as u64,
            (Some(g), _, _) => (count / g * (g / 2 + 3)) as u64,
            (None, Some(g), _) => (count / g * (g / 2 + 3) + in_features * 4) as u64,
            (None, None, true) => (count + rows_out * 2) as u64,
            (None, None, false) => (count as u64)
                .checked_mul(if out_dtype == "F32" { 4 } else { 2 })
                .ok_or("packed output size overflow")?,
        };
        // the per-tensor factor and the activation scale of an NVFP4 weight travel as F32 scalars `<name>.scale`
        // and `<name>.input_scale` (the llama.cpp names the engine reads)
        let nvfp4_scalars: Vec<(String, f32)> = match packed.nvfp4().filter(|_| nvfp4) {
            Some(t) => {
                let stem = name.strip_suffix(".weight").unwrap_or(&name).to_owned();
                let mut v = vec![(format!("{stem}.scale"), t.global_scale())];
                if let Some(i) = t.input_scale() {
                    v.push((format!("{stem}.input_scale"), i));
                }
                v
            }
            None => Vec::new(),
        };
        let source_for_output = row_permutation(arch, &name, heads, kv_heads, head_dim)?;
        outs.push(OutTensor {
            name,
            dims: dims.iter().rev().copied().collect(),
            source_dtype: "F32".into(),
            dtype: out_dtype,
            data: payload,
            packed: Some(std::sync::Arc::new(packed)),
            data_len,
            source_for_output,
            add: 0.0,
            owned: None,
            parts: Vec::new(),
        });
        for (scalar_name, value) in nvfp4_scalars {
            if !seen.insert(scalar_name.clone()) {
                return Err(format!("multiple HF tensors map to GGUF name {scalar_name}").into());
            }
            outs.push(OutTensor {
                name: scalar_name,
                dims: vec![1],
                source_dtype: "F32".into(),
                dtype: "F32".into(),
                data: safetensors::Payload::Owned(std::sync::Arc::new(Vec::new()), 0..0),
                packed: None,
                data_len: 4,
                source_for_output: None,
                add: 0.0,
                owned: Some(value.to_le_bytes().to_vec()),
                parts: Vec::new(),
            });
        }
    }
    for (name, _) in &plan.synthesized {
        seen.insert(name.clone());
    }
    for required in ["token_embd.weight", "output_norm.weight"] {
        if !seen.contains(required) {
            return Err(format!("missing required tensor {required}").into());
        }
    }
    if !tie && !seen.contains("output.weight") {
        return Err("missing lm_head.weight while tie_word_embeddings is false".into());
    }
    for (name, values) in &plan.synthesized {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        outs.push(OutTensor {
            name: name.clone(),
            dims: vec![values.len() as u64],
            source_dtype: "F32".into(),
            dtype: "F32".into(),
            data: safetensors::Payload::Owned(std::sync::Arc::new(Vec::new()), 0..0),
            packed: None,
            data_len: bytes.len() as u64,
            source_for_output: None,
            add: 0.0,
            owned: Some(bytes),
            parts: Vec::new(),
        });
    }
    let byname: HashMap<_, _> = outs.iter().map(|t| (t.name.as_str(), t)).collect();
    let sandwich = plan.family == Family::Gemma && arch != "gemma";
    for i in 0..layers {
        let mut required = vec!["attn_output.weight"];
        let moe_here = plan.moe && plan.moe_layers[i as usize];
        if plan.family == Family::Classic {
            continue;
        }
        if plan.family == Family::Nemotron {
            let kind = plan.kinds[i as usize];
            let mut need = vec!["attn_norm.weight"];
            match kind {
                b'M' => need.extend([
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_d",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ]),
                b'*' => need.extend([
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                ]),
                b'-' => need.extend(["ffn_up.weight", "ffn_down.weight"]),
                _ => need.extend([
                    "ffn_gate_inp.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                ]),
            }
            for n in need {
                let key = format!("blk.{i}.{n}");
                if !byname.contains_key(key.as_str()) {
                    return Err(format!("missing required tensor {key}").into());
                }
            }
            continue;
        }
        if plan.family == Family::Mamba {
            let tensors: &[&str] = if arch == "mamba" {
                &[
                    "attn_norm.weight",
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_x.weight",
                    "ssm_dt.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_d",
                    "ssm_out.weight",
                ]
            } else {
                &[
                    "attn_norm.weight",
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_d",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ]
            };
            for n in tensors {
                let key = format!("blk.{i}.{n}");
                if !byname.contains_key(key.as_str()) {
                    return Err(format!("missing required tensor {key}").into());
                }
            }
            continue;
        }
        if arch == "olmo_hybrid" {
            let linear = plan.kinds.get(i as usize) == Some(&b'L');
            let mut need = vec!["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"];
            if linear {
                need.extend([
                    "attn_norm.weight",
                    "ffn_norm.weight",
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_beta.weight",
                    "ssm_alpha.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ]);
            } else {
                need.extend([
                    "post_attention_norm.weight",
                    "post_ffw_norm.weight",
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                ]);
            }
            for n in need {
                let key = format!("blk.{i}.{n}");
                if !byname.contains_key(key.as_str()) {
                    return Err(format!("missing required tensor {key}").into());
                }
            }
            continue;
        }
        if matches!(arch, "lfm2" | "lfm2moe") {
            required.extend(["attn_norm.weight", "ffn_norm.weight"]);
            if plan.kinds.get(i as usize) == Some(&b'C') {
                required.retain(|&name| name != "attn_output.weight");
                required.extend([
                    "shortconv_in.weight",
                    "shortconv_conv.weight",
                    "shortconv_out.weight",
                ]);
                if cfg.get("conv_bias").and_then(Value::as_bool) == Some(true) {
                    required.extend([
                        "shortconv_in.bias",
                        "shortconv_conv.bias",
                        "shortconv_out.bias",
                    ]);
                }
            } else {
                required.extend(["attn_q_norm.weight", "attn_k_norm.weight"]);
            }
        }
        let shortconv_layer =
            matches!(arch, "lfm2" | "lfm2moe") && plan.kinds.get(i as usize) == Some(&b'C');
        let mamba_layer = matches!(arch, "granitehybrid" | "bamba")
            && byname.contains_key(format!("blk.{i}.ssm_in.weight").as_str());
        let linear_layer = plan.family == Family::Qwen3Next
            && byname.contains_key(format!("blk.{i}.attn_qkv.weight").as_str());
        if shortconv_layer {
            // This layer has no Q/K/V projections; its required convolution tensors are above.
        } else if mamba_layer {
            required.clear();
            required.extend([
                "ssm_in.weight",
                "ssm_conv1d.weight",
                "ssm_dt.bias",
                "ssm_a",
                "ssm_d",
                "ssm_norm.weight",
                "ssm_out.weight",
            ]);
        } else if linear_layer {
            required.clear();
            required.extend([
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_beta.weight",
                "ssm_alpha.weight",
                "ssm_conv1d.weight",
                "ssm_dt.bias",
                "ssm_a",
                "ssm_norm.weight",
                "ssm_out.weight",
            ]);
        } else if matches!(plan.family, Family::Phi3 | Family::Neox)
            || matches!(arch, "dbrx" | "internlm2" | "persimmon")
        {
            required.push("attn_qkv.weight");
        } else if plan.family == Family::Deepseek {
            required.extend([
                "attn_kv_a_mqa.weight",
                "attn_kv_a_norm.weight",
                "attn_kv_b.weight",
            ]);
            if plan.q_lora {
                required.extend(["attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight"]);
            } else {
                required.push("attn_q.weight");
            }
        } else if arch == "k2_horizon" && moe_here && plan.meta.iter().any(|(k, _)| k.ends_with(".attention.value_expert_count")) {
            required.extend(["attn_q.weight", "attn_k.weight", "attn_v_exps.weight", "attn_v_gate.weight"]);
        } else {
            required.extend(["attn_q.weight", "attn_k.weight", "attn_v.weight"]);
        }
        if moe_here {
            required.extend([
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ]);
        } else {
            if plan.gated {
                required.push("ffn_gate.weight");
            }
            required.extend(["ffn_up.weight", "ffn_down.weight"]);
        }
        match plan.family {
            Family::Cohere => required.push("attn_norm.weight"),
            Family::Glm if arch == "glm4" => required.extend([
                "attn_norm.weight",
                "post_attention_norm.weight",
                "ffn_norm.weight",
                "post_ffw_norm.weight",
            ]),
            Family::Olmo2 => {
                required.extend(["post_attention_norm.weight", "post_ffw_norm.weight"])
            }
            Family::Gemma if sandwich => required.extend([
                "attn_norm.weight",
                "post_attention_norm.weight",
                "ffn_norm.weight",
                "post_ffw_norm.weight",
            ]),
            _ if plan.meta.iter().any(|(k, v)| {
                k.ends_with(".use_parallel_residual") && matches!(v, Meta::Bool(true))
            }) =>
            {
                // parallel blocks may share one norm for both branches
                required.push("attn_norm.weight")
            }
            _ => required.extend(["attn_norm.weight", "ffn_norm.weight"]),
        }
        if matches!(
            arch,
            "qwen3"
                | "gemma3"
                | "muse-glimmer"
                | "olmo2"
                | "exaone4"
                | "olmo3"
                | "hunyuan-dense"
                | "hunyuan-moe"
                | "dots1"
                | "minimax-m2"
                | "hy_v3"
                | "exaone_moe"
                | "afmoe"
                | "persimmon"
        ) {
            required.extend(["attn_q_norm.weight", "attn_k_norm.weight"]);
        }
        for n in required {
            let key = format!("blk.{i}.{n}");
            if !byname.contains_key(key.as_str()) {
                return Err(format!("missing required tensor {key}").into());
            }
        }
    }
    if output.exists() && !options.overwrite {
        return Err(format!("output already exists: {}", output.display()).into());
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let tmp = temp_path(output);
    let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
    let result = (|| -> std::io::Result<u64> {
        let size = gguf::write_to(&mut f, &meta, &outs)?;
        f.sync_all()?;
        drop(f);
        commit_output(&tmp, output, options.overwrite)?;
        Ok(size)
    })();
    let size = match result {
        Ok(n) => n,
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            return Err(e.into());
        }
    };
    Ok(ConversionReport {
        architecture: arch.into(),
        tensors_written: outs.len(),
        bytes_written: size,
        skipped_tensors: skipped,
        output: output.to_path_buf(),
    })
}
fn temp_path(out: &Path) -> PathBuf {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut p = out.as_os_str().to_owned();
    p.push(format!(".{}.{}.tmp", std::process::id(), n));
    PathBuf::from(p)
}
fn commit_output(tmp: &Path, out: &Path, overwrite: bool) -> std::io::Result<()> {
    if !overwrite {
        fs::hard_link(tmp, out)?;
        fs::remove_file(tmp)?;
        return Ok(());
    }
    if !out.exists() {
        return fs::rename(tmp, out);
    }
    let backup = temp_path(out);
    fs::rename(out, &backup)?;
    match fs::rename(tmp, out) {
        Ok(()) => {
            let _ = fs::remove_file(backup);
            Ok(())
        }
        Err(e) => {
            let _ = fs::rename(&backup, out);
            Err(e)
        }
    }
}
fn num(v: &Value, k: &str, d: Option<u64>) -> Result<u64, String> {
    v.get(k)
        .and_then(Value::as_u64)
        .or(d)
        .ok_or_else(|| format!("config.json requires valid {k}"))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Quantization {
    Dense,
    Exl2,
    Exl3,
    Gptq { group: u64 },
    Awq { group: u64 },
    Fp8,
    /// NVIDIA ModelOpt `NVFP4` (E2M1 codes, E4M3 block scales per 16 inputs, per-tensor factor).
    Nvfp4,
    /// `compressed-tensors` `pack-quantized`, symmetric; `group` inputs per scale (0: one scale per output row).
    CtPacked { bits: u64, group: u64 },
    /// `compressed-tensors` `int-quantized` int8, symmetric; `group` as above.
    CtInt8 { group: u64 },
}

/// `hf_quant_config.json` of ModelOpt exports as a `quantization_config` object (`quant_method: modelopt`).
fn modelopt_quant_config(root: &Path) -> Result<Option<Value>, String> {
    let path = root.join("hf_quant_config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read(&path).map_err(|e| format!("read hf_quant_config.json: {e}"))?;
    let v: Value = serde_json::from_slice(&text).map_err(|e| format!("invalid hf_quant_config.json: {e}"))?;
    let Some(mut q) = v.get("quantization").filter(|q| q.is_object()).cloned() else {
        return Err("hf_quant_config.json has no quantization object".into());
    };
    if q.get("quant_method").is_none() {
        q["quant_method"] = "modelopt".into();
    }
    Ok(Some(q))
}

/// Tile size of block-wise FP8 weights (`weight_block_size` or compressed-tensors `block_structure`).
fn fp8_block_size(cfg: &Value) -> Option<(usize, usize)> {
    let q = cfg.get("quantization_config")?;
    let from_array = |a: &Value| -> Option<(usize, usize)> {
        let a = a.as_array()?;
        match (a.first()?.as_u64()?, a.get(1)?.as_u64()?) {
            (r, c) if r > 0 && c > 0 && a.len() == 2 => Some((r as usize, c as usize)),
            _ => None,
        }
    };
    if let Some(b) = q.get("weight_block_size").and_then(from_array) {
        return Some(b);
    }
    let weights = ct_weights(q).ok()??;
    let b = weights.get("block_structure")?;
    from_array(b).or_else(|| {
        let (r, c) = b.as_str()?.split_once('x')?;
        Some((r.trim().parse().ok()?, c.trim().parse().ok()?))
    })
}

/// The `weights` scheme shared by every `config_groups` entry of a compressed-tensors config.
fn ct_weights(q: &Value) -> Result<Option<&Value>, String> {
    let Some(groups) = q.get("config_groups").and_then(Value::as_object) else {
        return Ok(None);
    };
    let mut first: Option<&Value> = None;
    for group in groups.values() {
        let Some(w) = group.get("weights").filter(|w| !w.is_null()) else {
            continue;
        };
        match first {
            None => first = Some(w),
            Some(f) if f == w => {}
            Some(_) => {
                return Err("compressed-tensors config_groups use different weight schemes; mixed-precision checkpoints are not supported".into())
            }
        }
    }
    Ok(first)
}

fn compressed_tensors(q: &Value) -> Result<Quantization, String> {
    const HINT: &str = "supported compressed-tensors checkpoints: FP8 (float-quantized), symmetric int4/int8 weights (pack-quantized) and symmetric int8 (int-quantized); convert other formats to GGUF with llama.cpp or pick another repository";
    let format = q.get("format").and_then(Value::as_str).unwrap_or("");
    if q.get("sparsity_config")
        .and_then(|s| s.get("format"))
        .and_then(Value::as_str)
        .is_some_and(|f| f != "dense")
    {
        return Err(format!("compressed-tensors sparse checkpoints are not supported; {HINT}"));
    }
    let w = ct_weights(q)?.ok_or_else(|| format!("compressed-tensors config has no weight quantization scheme; {HINT}"))?;
    let bits = w.get("num_bits").and_then(Value::as_u64).unwrap_or(0);
    let kind = w.get("type").and_then(Value::as_str).unwrap_or("int");
    let strategy = w.get("strategy").and_then(Value::as_str).unwrap_or("tensor");
    if format.contains("nvfp4") || (kind == "float" && bits == 4) {
        return Err(format!(
            "compressed-tensors NVFP4 (format {format:?}) is not supported; use an NVIDIA ModelOpt NVFP4 checkpoint (quant_algo NVFP4) or a GGUF; {HINT}"
        ));
    }
    if kind == "float" {
        if bits == 8 {
            return Ok(Quantization::Fp8);
        }
        return Err(format!("compressed-tensors float weights with {bits} bits are not supported; {HINT}"));
    }
    if w.get("symmetric").and_then(Value::as_bool) == Some(false) {
        return Err(format!("compressed-tensors asymmetric (zero-point) integer weights are not supported; {HINT}"));
    }
    if w.get("actorder").is_some_and(|a| !a.is_null() && a != "null") {
        return Err(format!("compressed-tensors weights with activation ordering (actorder) are not supported; {HINT}"));
    }
    let group = match strategy {
        "group" => w
            .get("group_size")
            .and_then(Value::as_u64)
            .filter(|&g| g > 0)
            .ok_or("compressed-tensors group quantization requires a positive group_size")?,
        "channel" | "tensor" => 0,
        other => {
            return Err(format!("compressed-tensors weight strategy {other:?} is not supported; {HINT}"))
        }
    };
    match (format, bits) {
        ("int-quantized", 8) => Ok(Quantization::CtInt8 { group }),
        ("pack-quantized", 4 | 8) | ("", 4) => Ok(Quantization::CtPacked { bits, group }),
        ("", 8) => Ok(Quantization::CtInt8 { group }),
        _ => Err(format!(
            "compressed-tensors format {format:?} with {bits}-bit integer weights is not supported; {HINT}"
        )),
    }
}
fn quantization(cfg: &Value) -> Result<Quantization, String> {
    let Some(q) = cfg.get("quantization_config").filter(|v| !v.is_null()) else {
        return Ok(Quantization::Dense);
    };
    let method = q
        .get("quant_method")
        .and_then(Value::as_str)
        .ok_or("quantization_config requires quant_method")?
        .to_ascii_lowercase();
    match method.as_str() {
        "exl2" => Ok(Quantization::Exl2),
        "exl3" => Ok(Quantization::Exl3),
        "gptq" => {
            if q.get("bits").and_then(Value::as_u64) != Some(4) {
                return Err("only GPTQ 4-bit legacy packing is supported".into());
            }
            let group = q
                .get("group_size")
                .and_then(Value::as_u64)
                .filter(|&x| x > 0)
                .ok_or("GPTQ requires positive group_size")?;
            if q.get("checkpoint_format")
                .and_then(Value::as_str)
                .is_some_and(|x| !x.eq_ignore_ascii_case("gptq"))
            {
                return Err("only GPTQ checkpoint_format is supported".into());
            }
            Ok(Quantization::Gptq { group })
        }
        "awq" => {
            if q.get("bits").and_then(Value::as_u64) != Some(4) {
                return Err("only AWQ 4-bit GEMM packing is supported".into());
            }
            if q.get("version")
                .and_then(Value::as_str)
                .is_none_or(|x| !x.eq_ignore_ascii_case("gemm"))
            {
                return Err("only explicitly declared AutoAWQ GEMM packing is supported".into());
            }
            if q.get("zero_point").and_then(Value::as_bool) != Some(true) {
                return Err(
                    "only AutoAWQ checkpoints with explicit zero points are supported".into(),
                );
            }
            let group = q
                .get("group_size")
                .and_then(Value::as_u64)
                .filter(|&x| x > 0)
                .ok_or("AWQ requires positive group_size")?;
            Ok(Quantization::Awq { group })
        }
        "fp8" => Ok(Quantization::Fp8),
        "compressed-tensors" => compressed_tensors(q),
        "modelopt" => {
            let algo = q
                .get("quant_algo")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_ascii_uppercase();
            match algo.as_str() {
                "NVFP4" => {
                    let group = q.get("group_size").and_then(Value::as_u64).unwrap_or(16);
                    if group != 16 {
                        return Err(format!("ModelOpt NVFP4 with group_size {group} is not supported (16 expected)"));
                    }
                    Ok(Quantization::Nvfp4)
                }
                "FP8" => Ok(Quantization::Fp8),
                other => Err(format!(
                    "unsupported ModelOpt quant_algo {other:?}: only NVFP4 and FP8 checkpoints can be imported"
                )),
            }
        }
        _ => Err(format!(
            "unsupported quantization_config quant_method {method:?} (supported: gptq, awq, fp8, compressed-tensors, modelopt NVFP4/FP8, exl2, exl3)"
        )),
    }
}
fn element_count(shape: &[u64], name: &str) -> Result<u64, String> {
    shape.iter().try_fold(1u64, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| format!("tensor {name} shape overflow"))
    })
}
pub(crate) type PackedGroup = (String, PackedData, Vec<u64>, safetensors::Payload);
pub(crate) type PackedSplit = (Vec<Tensor>, Vec<PackedGroup>);
fn parse_fp8_scales(t: &Tensor, rows: usize) -> Result<Vec<u16>, String> {
    let bytes = t.payload.as_slice();
    let num_scales = t.shape.iter().product::<u64>() as usize;
    if num_scales != 1 && num_scales != rows {
        return Err(format!(
            "scale tensor {} has {} elements, expected 1 or {rows}",
            t.name, num_scales
        ));
    }
    let mut out_f32 = Vec::with_capacity(num_scales);
    match t.dtype.as_str() {
        "BF16" => {
            if bytes.len() != num_scales * 2 {
                return Err(format!("scale tensor {} length mismatch for BF16", t.name));
            }
            for chunk in bytes.as_chunks::<2>().0 {
                let u = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f = f32::from_bits((u as u32) << 16);
                out_f32.push(f);
            }
        }
        "F16" => {
            if bytes.len() != num_scales * 2 {
                return Err(format!("scale tensor {} length mismatch for F16", t.name));
            }
            for chunk in bytes.as_chunks::<2>().0 {
                let u = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f = half::f16::from_bits(u).to_f32();
                out_f32.push(f);
            }
        }
        "F32" => {
            if bytes.len() != num_scales * 4 {
                return Err(format!("scale tensor {} length mismatch for F32", t.name));
            }
            for chunk in bytes.as_chunks::<4>().0 {
                let u = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let f = f32::from_bits(u);
                out_f32.push(f);
            }
        }
        other => return Err(format!("unsupported scale dtype {other} for {}", t.name)),
    }
    if num_scales == 1 {
        let s = half::f16::from_f32(out_f32[0]).to_bits();
        Ok(vec![s; rows])
    } else {
        Ok(out_f32.into_iter().map(|f| half::f16::from_f32(f).to_bits()).collect())
    }
}

/// All scalars of a float scale tensor as f32.
fn scale_values(t: &Tensor) -> Result<Vec<f32>, String> {
    let bytes = t.payload.as_slice();
    match t.dtype.as_str() {
        "BF16" => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()),
        "F16" => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        "F32" => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        other => Err(format!("unsupported scale dtype {other} for {}", t.name)),
    }
}

fn single_f32(t: &Tensor) -> Result<f32, String> {
    match scale_values(t)?.as_slice() {
        [v] => Ok(*v),
        other => Err(format!("{} must hold one value, found {}", t.name, other.len())),
    }
}

fn matrix_dims(t: &Tensor) -> Result<(usize, usize), String> {
    match t.shape[..] {
        [r, c] => Ok((
            usize::try_from(r).map_err(|_| "dimension too large")?,
            usize::try_from(c).map_err(|_| "dimension too large")?,
        )),
        _ => Err(format!("{} must have rank 2, found shape {:?}", t.name, t.shape)),
    }
}

/// ModelOpt NVFP4: `U8` weights with `weight_scale` (E4M3), `weight_scale_2` and `input_scale` per linear.
fn split_nvfp4(
    all: Vec<Tensor>,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<PackedSplit, String> {
    let mut weights = Vec::new();
    let mut scales = HashMap::new();
    let mut globals = HashMap::new();
    let mut inputs = HashMap::new();
    let mut dense = Vec::new();
    for t in all {
        if let Some(base) = t.name.strip_suffix(".weight_scale_2") {
            globals.insert(base.to_owned(), t);
        } else if let Some(base) = t.name.strip_suffix(".weight_scale") {
            scales.insert(base.to_owned(), t);
        } else if let Some(base) = t.name.strip_suffix(".input_scale") {
            inputs.insert(base.to_owned(), t);
        } else if t.name.ends_with(".k_scale") || t.name.ends_with(".v_scale") {
            // FP8 KV-cache scales: the engine chooses its own KV format
        } else if t.name.ends_with(".weight") && t.dtype == "U8" {
            weights.push(t);
        } else {
            dense.push(t);
        }
    }
    let mut packed = Vec::new();
    for w in weights {
        let base = w.name.strip_suffix(".weight").unwrap_or(&w.name).to_owned();
        let scale = scales
            .remove(&base)
            .ok_or_else(|| format!("NVFP4 weight {} has no {base}.weight_scale", w.name))?;
        let global = globals
            .remove(&base)
            .ok_or_else(|| format!("NVFP4 weight {} has no {base}.weight_scale_2", w.name))?;
        let (rows, half_cols) = matrix_dims(&w)?;
        let cols = half_cols * 2;
        if scale.dtype != "F8_E4M3" || scale.shape != [rows as u64, (cols / 16) as u64] {
            return Err(format!(
                "{} must be F8_E4M3 [{rows}, {}], found {} {:?}",
                scale.name,
                cols / 16,
                scale.dtype,
                scale.shape
            ));
        }
        if global.dtype != "F32" {
            return Err(format!("{} must be F32, found {}", global.name, global.dtype));
        }
        let input = inputs.remove(&base).map(|t| single_f32(&t)).transpose()?;
        let t = crate::nvfp4::Nvfp4::new(w.payload.clone(), scale.payload, rows, cols, single_f32(&global)?, input)
            .map_err(|e| format!("{}: {e}", w.name))?;
        let mapped = map(&w.name)?
            .ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
        packed.push((mapped, PackedData::Nvfp4(t), vec![rows as u64, cols as u64], w.payload));
    }
    if let Some(left) = scales.keys().chain(globals.keys()).next() {
        return Err(format!("NVFP4 scale tensors of {left} have no U8 weight"));
    }
    if packed.is_empty() {
        return Err("quantization_config declares NVFP4 but no U8 weights with weight_scale were found".into());
    }
    Ok((dense, packed))
}

/// compressed-tensors symmetric integer weights: `pack-quantized` (`weight_packed`) or int8 `weight`.
fn split_ct_int(
    all: Vec<Tensor>,
    quant: Quantization,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<PackedSplit, String> {
    use crate::symint::{SymCodes, SymInt, SymScale};
    let packed_format = matches!(quant, Quantization::CtPacked { .. });
    let mut weights = Vec::new();
    let mut scales = HashMap::new();
    let mut shapes = HashMap::new();
    let mut dense = Vec::new();
    for t in all {
        let n = t.name.as_str();
        if let Some(base) = n.strip_suffix(".weight_scale") {
            scales.insert(base.to_owned(), t);
        } else if let Some(base) = n.strip_suffix(".weight_shape") {
            shapes.insert(base.to_owned(), t);
        } else if n.ends_with(".weight_zero_point") || n.ends_with(".weight_g_idx") {
            return Err(format!(
                "{n}: compressed-tensors asymmetric or activation-ordered weights are not supported"
            ));
        } else if n.ends_with(".input_scale") || n.ends_with(".input_zero_point") {
            // activations are quantized dynamically by the engine
        } else if (packed_format && n.ends_with(".weight_packed"))
            || (!packed_format && n.ends_with(".weight") && t.dtype == "I8")
        {
            weights.push(t);
        } else {
            dense.push(t);
        }
    }
    let mut packed = Vec::new();
    for w in weights {
        let base = w
            .name
            .strip_suffix(".weight_packed")
            .or_else(|| w.name.strip_suffix(".weight"))
            .unwrap_or(&w.name)
            .to_owned();
        let scale = scales
            .remove(&base)
            .ok_or_else(|| format!("integer weight {} has no {base}.weight_scale", w.name))?;
        let scale_type = SymScale::from_dtype(&scale.dtype)
            .ok_or_else(|| format!("{} has unsupported dtype {}", scale.name, scale.dtype))?;
        let (rows, stored_cols) = matrix_dims(&w)?;
        let (kind, cols, group_cfg) = match quant {
            Quantization::CtPacked { bits: 4, group } => {
                if w.dtype != "I32" {
                    return Err(format!("{} must be I32, found {}", w.name, w.dtype));
                }
                (SymCodes::Packed4, stored_cols * 8, group)
            }
            Quantization::CtPacked { bits: 8, group } => {
                if w.dtype != "I32" {
                    return Err(format!("{} must be I32, found {}", w.name, w.dtype));
                }
                (SymCodes::Packed8, stored_cols * 4, group)
            }
            Quantization::CtInt8 { group } => (SymCodes::Int8, stored_cols, group),
            _ => return Err("not a compressed-tensors integer format".into()),
        };
        if let Some(shape) = shapes.remove(&base) {
            let dims: Vec<u64> = shape
                .payload
                .as_slice()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect();
            if dims != [rows as u64, cols as u64] {
                return Err(format!(
                    "{}.weight_shape {dims:?} does not match the packed weight [{rows}, {cols}]",
                    base
                ));
            }
        }
        let group = if group_cfg == 0 { cols } else { group_cfg as usize };
        let t = SymInt::new(w.payload.clone(), kind, scale.payload, scale_type, rows, cols, group)
            .map_err(|e| format!("{}: {e}", w.name))?;
        let mapped = map(&format!("{base}.weight"))?
            .ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
        packed.push((mapped, PackedData::SymInt(t), vec![rows as u64, cols as u64], w.payload));
    }
    if let Some(left) = scales.keys().next() {
        return Err(format!("{left}.weight_scale has no matching integer weight"));
    }
    if packed.is_empty() {
        return Err("quantization_config declares compressed-tensors integer weights but none were found".into());
    }
    Ok((dense, packed))
}

pub(crate) fn split_packed(
    all: Vec<Tensor>,
    quant: Quantization,
    fp8_block: Option<(usize, usize)>,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<PackedSplit, String> {
    use std::collections::BTreeMap;
    if quant == Quantization::Nvfp4 {
        return split_nvfp4(all, map);
    }
    if matches!(quant, Quantization::CtPacked { .. } | Quantization::CtInt8 { .. }) {
        return split_ct_int(all, quant, map);
    }
    let has_exl = all
        .iter()
        .any(|t| t.name.ends_with(".q_weight") || t.name.ends_with(".trellis"));
    if has_exl || matches!(quant, Quantization::Exl2 | Quantization::Exl3) {
        let declared = match quant {
            Quantization::Exl2 => Some(false),
            Quantization::Exl3 => Some(true),
            Quantization::Dense => None,
            _ => return Err("EXL tensors conflict with declared quantization method".into()),
        };
        return crate::exl2::split(all, declared, map);
    }
    if quant == Quantization::Fp8 {
        let mut scales = HashMap::new();
        let mut fp8_weights = Vec::new();
        let mut dense = Vec::new();
        for t in all {
            if t.name.ends_with(".input_scale") {
                continue;
            }
            if t.name.ends_with(".weight_scale") {
                let base = t.name.strip_suffix(".weight_scale").unwrap();
                scales.insert(base.to_string(), t);
            } else if t.name.ends_with(".weight_scale_inv") {
                let base = t.name.strip_suffix(".weight_scale_inv").unwrap();
                scales.insert(base.to_string(), t);
            } else if t.dtype == "F8_E4M3" && t.name.ends_with(".weight") {
                fp8_weights.push(t);
            } else {
                dense.push(t);
            }
        }
        if !fp8_weights.is_empty() && !scales.is_empty() {
            let mut packed = Vec::new();
            for w in fp8_weights {
                let base = w.name.strip_suffix(".weight").unwrap();
                let scale_tensor = scales.remove(base).ok_or_else(|| {
                    format!("missing weight_scale for FP8 weight {}", w.name)
                })?;
                if w.shape.len() != 2 {
                    return Err(format!("FP8 weight {} must have rank 2", w.name));
                }
                let rows = w.shape[0] as usize;
                let cols = w.shape[1] as usize;
                let scale_count = scale_tensor.shape.iter().product::<u64>() as usize;
                let tile_scales = scale_tensor.shape.len() == 2 && scale_count != 1 && scale_count != rows;
                if tile_scales {
                    let (br, bc) = fp8_block.ok_or_else(|| {
                        format!(
                            "{} has the 2-D scale shape {:?} but the quantization_config declares no weight_block_size",
                            w.name, scale_tensor.shape
                        )
                    })?;
                    let t = crate::symint::Fp8Block::new(w.payload.clone(), scale_values(&scale_tensor)?, rows, cols, br, bc)
                        .map_err(|e| format!("{}: {e}", w.name))?;
                    let mapped = map(&w.name)?
                        .ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
                    packed.push((mapped, PackedData::Fp8Block(t), w.shape.clone(), w.payload));
                    continue;
                }
                let parsed_scales = parse_fp8_scales(&scale_tensor, rows)?;
                let p = PackedData::Fp8Marlin {
                    weight: w.payload.clone(),
                    scales: parsed_scales,
                    rows,
                    cols,
                };
                let mapped = map(&w.name)?
                    .ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
                packed.push((mapped, p, w.shape.clone(), w.payload));
            }
            if !scales.is_empty() {
                return Err(format!(
                    "unmatched weight_scales in FP8 checkpoint: {:?}",
                    scales.keys().collect::<Vec<_>>()
                ));
            }
            return Ok((dense, packed));
        } else {
            dense.extend(fp8_weights);
            return Ok((dense, Vec::new()));
        }
    }
    let mut dense = Vec::new();
    let mut groups: BTreeMap<String, HashMap<String, Tensor>> = BTreeMap::new();
    for t in all {
        let suffix = if t.name.ends_with(".qweight") {
            Some("qweight")
        } else if t.name.ends_with(".qzeros") {
            Some("qzeros")
        } else if t.name.ends_with(".scales") {
            Some("scales")
        } else if t.name.ends_with(".g_idx") {
            Some("g_idx")
        } else {
            None
        };
        if let Some(suffix) = suffix {
            if matches!(quant, Quantization::Dense | Quantization::Fp8) {
                return Err(format!(
                    "packed quantization tensor {} has no matching supported quantization_config",
                    t.name
                ));
            }
            let base = &t.name[..t.name.len() - suffix.len()];
            let base = base
                .strip_suffix('.')
                .ok_or("malformed packed quantization tensor name")?
                .to_string();
            if groups
                .entry(base)
                .or_default()
                .insert(suffix.into(), t)
                .is_some()
            {
                return Err("duplicate packed quantization component".into());
            }
        } else {
            dense.push(t);
        }
    }
    let mut packed = Vec::new();
    for (base, mut g) in groups {
        let get = |g: &mut HashMap<String, Tensor>, k: &str| {
            g.remove(k)
                .ok_or_else(|| format!("packed tensor group {base:?} is missing {k}"))
        };
        let qw = get(&mut g, "qweight")?;
        let qz = get(&mut g, "qzeros")?;
        let scales = get(&mut g, "scales")?;
        let source_name = format!("{base}.weight");
        let (data, dims) = match quant {
            Quantization::Gptq { group } => {
                let in_features = qw
                    .shape
                    .first()
                    .copied()
                    .ok_or("GPTQ qweight must be rank 2")?
                    .checked_mul(8)
                    .ok_or("GPTQ input dimension overflow")?;
                let out_features = *qw.shape.get(1).ok_or("GPTQ qweight must be rank 2")?;
                if qw.shape.len() != 2
                    || qw.dtype != "I32"
                    || qz.dtype != "I32"
                    || !matches!(scales.dtype.as_str(), "F16" | "F32")
                {
                    return Err(format!(
                        "GPTQ components for {base:?} have unsupported rank/dtype"
                    ));
                }
                let expected_qz = vec![in_features.div_ceil(group), out_features / 8];
                if qz.shape != expected_qz
                    || scales.shape != vec![expected_qz[0], out_features]
                    || !out_features.is_multiple_of(8)
                {
                    return Err(format!(
                        "GPTQ component shapes for {base:?} do not match 4-bit GEMM layout"
                    ));
                }
                let gi = g.remove("g_idx");
                let gi_payload = if let Some(gi) = gi {
                    if gi.shape != vec![in_features] || gi.dtype != "I32" {
                        return Err(format!(
                            "GPTQ g_idx for {base:?} must be I32[{in_features}]"
                        ));
                    }
                    Some(gi.payload)
                } else {
                    None
                };
                let typ = if scales.dtype == "F16" {
                    ScaleType::F16
                } else {
                    ScaleType::F32
                };
                let p = PackedData::gptq_v1(
                    qw.payload.clone(),
                    qz.payload,
                    scales.payload,
                    gi_payload,
                    usize::try_from(in_features).map_err(|_| "GPTQ input dimension too large")?,
                    usize::try_from(out_features).map_err(|_| "GPTQ output dimension too large")?,
                    usize::try_from(group).map_err(|_| "GPTQ group size too large")?,
                    typ,
                )?;
                (p, vec![out_features, in_features])
            }
            Quantization::Awq { group } => {
                let in_features = *qw.shape.first().ok_or("AWQ qweight must be rank 2")?;
                let out_features = qw
                    .shape
                    .get(1)
                    .copied()
                    .ok_or("AWQ qweight must be rank 2")?
                    .checked_mul(8)
                    .ok_or("AWQ output dimension overflow")?;
                if qw.shape.len() != 2
                    || qw.dtype != "I32"
                    || qz.dtype != "I32"
                    || !matches!(scales.dtype.as_str(), "F16" | "F32")
                {
                    return Err(format!(
                        "AWQ components for {base:?} have unsupported rank/dtype"
                    ));
                }
                let groups = in_features / group;
                if !in_features.is_multiple_of(group)
                    || qz.shape != vec![groups, out_features / 8]
                    || scales.shape != vec![groups, out_features]
                {
                    return Err(format!(
                        "AWQ component shapes for {base:?} do not match GEMM layout"
                    ));
                }
                let typ = if scales.dtype == "F16" {
                    ScaleType::F16
                } else {
                    ScaleType::F32
                };
                let p = PackedData::awq_gemm(
                    qw.payload.clone(),
                    qz.payload,
                    scales.payload,
                    usize::try_from(in_features).map_err(|_| "AWQ input dimension too large")?,
                    usize::try_from(out_features).map_err(|_| "AWQ output dimension too large")?,
                    usize::try_from(group).map_err(|_| "AWQ group size too large")?,
                    typ,
                )?;
                (p, vec![out_features, in_features])
            }
            _ => return Err("packed int4 tensors require GPTQ or AWQ quantization_config".into()),
        };
        if !g.is_empty() {
            return Err(format!(
                "unsupported quantization component(s) for {base:?}: {:?}",
                g.keys().collect::<Vec<_>>()
            ));
        }
        let mapped = map(&source_name)?
            .ok_or_else(|| format!("packed tensor {source_name:?} maps to a skipped tensor"))?;
        packed.push((mapped, data, dims, qw.payload));
    }
    if matches!(quant, Quantization::Gptq { .. } | Quantization::Awq { .. }) && packed.is_empty() {
        return Err("quantization_config declares GPTQ/AWQ but no supported packed tensor groups were found".into());
    }
    if matches!(quant, Quantization::Fp8)
        && !dense
            .iter()
            .any(|t| matches!(t.dtype.as_str(), "F8_E4M3" | "F8_E5M2"))
    {
        return Err(
            "quantization_config declares FP8 but no raw unscaled FP8 tensors were found".into(),
        );
    }
    Ok((dense, packed))
}
fn expected_shape(
    n: &str,
    h: u64,
    inter: u64,
    heads: u64,
    kv: u64,
    hd: u64,
    vocab: u64,
) -> Result<Option<Vec<u64>>, String> {
    let s = match n {
        "token_embd.weight" => vec![vocab, h],
        "output_norm.weight" | "output_norm.bias" => vec![h],
        "output.bias" => vec![vocab],
        "output.weight" => vec![vocab, h],
        _ => {
            let tail = n.strip_prefix("blk.").ok_or("bad mapped tensor")?;
            let (_, p) = tail.split_once('.').ok_or("bad layer tensor")?;
            if p.starts_with("nextn.") {
                return Ok(None);
            }
            match p {
                "attn_norm.weight"
                | "attn_norm.bias"
                | "ffn_norm.bias"
                | "ffn_norm.weight"
                | "post_attention_norm.weight"
                | "post_ffw_norm.weight" => vec![h],
                // per-head or full-width depending on the architecture
                "attn_q_norm.weight" | "attn_k_norm.weight" => return Ok(None),
                "attn_q.weight" => vec![heads * hd, h],
                "attn_k.weight" | "attn_v.weight" => vec![kv * hd, h],
                "attn_output.weight" => vec![h, heads * hd],
                "ffn_gate.weight" | "ffn_up.weight" => vec![inter, h],
                "ffn_down.weight" => vec![h, inter],
                "attn_q.bias" => vec![heads * hd],
                "attn_k.bias" | "attn_v.bias" => vec![kv * hd],
                "attn_output.bias" | "ffn_down.bias" => vec![h],
                "ffn_gate.bias" | "ffn_up.bias" => vec![inter],
                _ => return Ok(None),
            }
        }
    };
    Ok(Some(s))
}
fn output_dtype(source: &str, requested: OutputDType) -> Result<String, String> {
    match requested {
        OutputDType::Preserve => Ok(source.into()),
        OutputDType::F16 | OutputDType::F32 if matches!(source, "F32" | "F16" | "BF16") => {
            Ok(if requested == OutputDType::F16 {
                "F16"
            } else {
                "F32"
            }
            .into())
        }
        _ => Err(format!("unsupported source dtype {source}")),
    }
}
/// Multimodal configs keep the language model in `text_config`; use that as the model config.
fn flatten_text_config(cfg: Value) -> Value {
    let Some(mut text) = cfg.get("text_config").filter(|v| v.is_object()).cloned() else {
        return cfg;
    };
    for key in [
        "tie_word_embeddings",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "vocab_size",
    ] {
        if text.get(key).is_none() {
            if let Some(v) = cfg.get(key) {
                text[key] = v.clone();
            }
        }
    }
    text
}

pub(crate) fn row_permutation(
    arch: &str,
    name: &str,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
) -> Result<Option<Vec<usize>>, String> {
    if matches!(arch, "qwen3next" | "g9v3") && name.ends_with("attn_q.weight") {
        // rows are stored per head as [query; gate]; GGUF wants all queries, then all gates
        let (h, d) = (heads as usize, head_dim as usize);
        let mut order = vec![0usize; 2 * h * d];
        for head in 0..h {
            for part in 0..2 {
                for i in 0..d {
                    order[part * h * d + head * d + i] = head * 2 * d + part * d + i;
                }
            }
        }
        return Ok(Some(order));
    }
    if arch == "gptneox" && name.contains(".attn_qkv.") {
        // fused rows are stored per head as [q; k; v]; GGUF wants all q rows, then k, then v
        let (h, d) = (heads as usize, head_dim as usize);
        let mut order = vec![0usize; 3 * h * d];
        for head in 0..h {
            for part in 0..3 {
                for i in 0..d {
                    order[part * h * d + head * d + i] = head * 3 * d + part * d + i;
                }
            }
        }
        return Ok(Some(order));
    }
    if arch == "internlm2" && (name.ends_with("attn_qkv.weight") || name.ends_with("attn_qkv.bias")) {
        // Rows are grouped per KV head as [q .. q, k, v] (`heads / kv_heads` query heads). GGUF wants all
        // queries, then all keys, then all values, with the Q/K rows of every head interleaved
        // (rotary pairs adjacent, the layout of llama GGUFs).
        let (h, kv, d) = (heads as usize, kv_heads as usize, head_dim as usize);
        if kv == 0 || h % kv != 0 || d % 2 != 0 {
            return Err("InternLM2 needs heads divisible by kv_heads and an even head_dim".into());
        }
        let group = h / kv;
        let source = |kind: usize, head: usize, row: usize| -> usize {
            // kind 0 = q (head index over all query heads), 1 = k, 2 = v (head index over KV heads)
            let (g, slot) = match kind {
                0 => (head / group, head % group),
                1 => (head, group),
                _ => (head, group + 1),
            };
            (g * (group + 2) + slot) * d + row
        };
        let half = d / 2;
        let mut order = Vec::with_capacity((h + 2 * kv) * d);
        for (kind, count) in [(0usize, h), (1, kv), (2, kv)] {
            for head in 0..count {
                if kind == 2 {
                    order.extend((0..d).map(|row| source(kind, head, row)));
                } else {
                    let mut chunk = vec![0usize; d];
                    for src in 0..d {
                        let local = if src < half { 2 * src } else { 2 * (src - half) + 1 };
                        chunk[local] = source(kind, head, src);
                    }
                    order.extend(chunk);
                }
            }
        }
        return Ok(Some(order));
    }
    if !matches!(arch, "llama" | "mistral" | "granite" | "muse-glimmer") {
        return Ok(None);
    }
    let count = if name.ends_with("attn_q.weight") || name.ends_with("attn_q.bias") {
        Some(heads)
    } else if name.ends_with("attn_k.weight") || name.ends_with("attn_k.bias") {
        Some(kv_heads)
    } else {
        None
    };
    let Some(count) = count else { return Ok(None) };
    if !head_dim.is_multiple_of(2) {
        return Err("Llama/Mistral RoPE requires an even head_dim for Q/K row conversion".into());
    }
    let rows = count
        .checked_mul(head_dim)
        .ok_or("Q/K row count overflow")?;
    let rows = usize::try_from(rows).map_err(|_| "Q/K row count does not fit usize")?;
    let hd = usize::try_from(head_dim).map_err(|_| "head_dim does not fit usize")?;
    let half = hd / 2;
    let mut order = vec![0usize; rows];
    for h in 0..usize::try_from(count).map_err(|_| "head count does not fit usize")? {
        let base = h * hd;
        for src in 0..hd {
            let local = if src < half {
                2 * src
            } else {
                2 * (src - half) + 1
            };
            order[base + local] = base + src;
        }
    }
    Ok(Some(order))
}

pub(crate) fn tokenizer(
    dir: &Path,
    model_vocab: u64,
    model_config: &Value,
) -> Result<(usize, Vec<(String, Meta)>), String> {
    let p = dir.join("tokenizer.json");
    if !p.is_file() {
        return Err("tokenizer.json is required for tokenizer import".into());
    }
    let raw = fs::read(p).map_err(|e| format!("read tokenizer.json: {e}"))?;
    let raw_text =
        String::from_utf8(raw.clone()).map_err(|e| format!("tokenizer.json is not UTF-8: {e}"))?;
    let t: Value =
        serde_json::from_slice(&raw).map_err(|e| format!("invalid tokenizer.json: {e}"))?;
    let model = t.get("model").ok_or("tokenizer.json missing model")?;
    let model_type = model.get("type").and_then(Value::as_str).unwrap_or("BPE");
    if !matches!(model_type, "BPE" | "Unigram") {
        return Err(format!("unsupported tokenizer model type {model_type:?}"));
    }
    let mut pieces = Vec::<String>::new();
    let mut occupied = Vec::<bool>::new();
    let mut scores = Vec::<f32>::new();
    let mut merges = Vec::new();
    if let Some(v) = model.get("vocab").and_then(Value::as_object) {
        let max = v.values().filter_map(Value::as_u64).max().unwrap_or(0);
        if max > 10_000_000 {
            return Err("tokenizer vocabulary exceeds safety limit".into());
        }
        pieces.resize(max as usize + 1, String::new());
        occupied.resize(pieces.len(), false);
        for (s, id) in v {
            let id = usize::try_from(id.as_u64().ok_or("invalid tokenizer vocabulary id")?)
                .map_err(|_| "token id does not fit usize")?;
            if occupied[id] {
                return Err("duplicate tokenizer vocabulary id".into());
            }
            pieces[id] = s.clone();
            occupied[id] = true;
        }
    } else if let Some(v) = model.get("vocab").and_then(Value::as_array) {
        for item in v {
            let pair = item.as_array().ok_or("invalid Unigram vocabulary entry")?;
            pieces.push(
                pair.first()
                    .and_then(Value::as_str)
                    .ok_or("invalid Unigram vocabulary token")?
                    .to_string(),
            );
            occupied.push(true);
            let score = pair.get(1).and_then(Value::as_f64).unwrap_or(0.0) as f32;
            if !score.is_finite() {
                return Err("non-finite Unigram tokenizer score".into());
            }
            scores.push(score);
        }
    } else {
        return Err("unsupported tokenizer.json model vocabulary".into());
    }
    if let Some(ms) = model.get("merges").and_then(Value::as_array) {
        for m in ms {
            merges.push(match m {
                Value::String(s) => s.clone(),
                Value::Array(a) if a.len() == 2 => format!(
                    "{} {}",
                    a.first().and_then(Value::as_str).ok_or("invalid merge")?,
                    a.get(1).and_then(Value::as_str).ok_or("invalid merge")?
                ),
                _ => return Err("invalid tokenizer merge".into()),
            });
        }
    }
    if model_type == "Unigram" && scores.len() != pieces.len() {
        return Err("Unigram tokenizer scores do not match vocabulary".into());
    }
    let mut types = vec![1u32; pieces.len()];
    let cfgp = dir.join("tokenizer_config.json");
    let mut tc: Value = if cfgp.exists() {
        serde_json::from_slice(
            &fs::read(cfgp).map_err(|e| format!("read tokenizer_config.json: {e}"))?,
        )
        .map_err(|e| format!("invalid tokenizer_config.json: {e}"))?
    } else {
        Value::Null
    };
    let smp = dir.join("special_tokens_map.json");
    if smp.is_file() {
        let sm: Value = serde_json::from_slice(
            &fs::read(smp).map_err(|e| format!("read special_tokens_map.json: {e}"))?,
        )
        .map_err(|e| format!("invalid special_tokens_map.json: {e}"))?;
        if let Some(dst) = tc.as_object_mut() {
            if let Some(src) = sm.as_object() {
                for k in ["bos_token", "eos_token", "unk_token"] {
                    if !dst.contains_key(k) {
                        if let Some(v) = src.get(k) {
                            dst.insert(k.into(), v.clone());
                        }
                    }
                }
            }
        } else {
            tc = sm;
        }
    }
    let mut ids = HashMap::<String, u32>::new();
    for (i, s) in pieces.iter().enumerate() {
        ids.insert(s.clone(), i as u32);
    }
    if let Some(added) = t.get("added_tokens").and_then(Value::as_array) {
        for a in added {
            let id = usize::try_from(
                a.get("id")
                    .and_then(Value::as_u64)
                    .ok_or("invalid added token id")?,
            )
            .map_err(|_| "added token id does not fit usize")?;
            if id > 10_000_000 {
                return Err("tokenizer vocabulary exceeds safety limit".into());
            }
            let content = a
                .get("content")
                .and_then(Value::as_str)
                .ok_or("invalid added token content")?;
            if id >= pieces.len() {
                pieces.resize(id + 1, String::new());
                occupied.resize(id + 1, false);
                types.resize(id + 1, 1);
                if model_type == "Unigram" {
                    scores.resize(id + 1, 0.0);
                }
            }
            if occupied[id] && pieces[id] != content {
                return Err("added token id conflicts with model vocabulary".into());
            }
            pieces[id] = content.into();
            occupied[id] = true;
            types[id] = if a.get("special").and_then(Value::as_bool) == Some(true) {
                3
            } else {
                4
            };
            ids.insert(content.into(), id as u32);
        }
    }
    if occupied.iter().any(|present| !present) {
        return Err(
            "tokenizer vocabulary contains missing ids inside the declared tokenizer vocabulary"
                .into(),
        );
    }
    let real_vocab = pieces.len();
    let target =
        usize::try_from(model_vocab).map_err(|_| "config vocab_size does not fit usize")?;
    if real_vocab > target {
        return Err(format!(
            "tokenizer has {real_vocab} real tokens, exceeding config vocab_size {model_vocab}"
        ));
    }
    if target > 10_000_000 {
        return Err("config vocab_size exceeds tokenizer safety limit".into());
    }
    let mut unique: HashSet<String> = pieces.iter().cloned().collect();
    for id in real_vocab..target {
        let mut unused = format!("<|fllm_unused_{id}|>");
        while unique.contains(&unused) {
            unused.push('_');
        }
        unique.insert(unused.clone());
        pieces.push(unused);
        types.push(5);
        if model_type == "Unigram" {
            scores.push(-1.0e9);
        }
    }
    let ggml_model = if model_type == "Unigram" {
        "llama"
    } else {
        "gpt2"
    };
    let mut meta = vec![
        ("tokenizer.huggingface.json".into(), Meta::Str(raw_text)),
        ("tokenizer.ggml.model".into(), Meta::Str(ggml_model.into())),
        ("tokenizer.ggml.pre".into(), Meta::Str("default".into())),
        ("tokenizer.ggml.tokens".into(), Meta::Strs(pieces.clone())),
        ("tokenizer.ggml.token_type".into(), Meta::U32s(types)),
        ("tokenizer.ggml.merges".into(), Meta::Strs(merges)),
    ];
    if model_type == "Unigram" {
        meta.push(("tokenizer.ggml.scores".into(), Meta::F32s(scores)));
    }
    for (key, field, config_key) in [
        ("eos_token_id", "eos_token", "eos_token_id"),
        ("eot_token_id", "eos_token", "eos_token_id"),
        ("bos_token_id", "bos_token", "bos_token_id"),
        ("unknown_token_id", "unk_token", "unk_token_id"),
    ] {
        let from_token = tc.get(field).and_then(|v| {
            let token = v
                .get("content")
                .and_then(Value::as_str)
                .or_else(|| v.as_str());
            token
                .and_then(|s| ids.get(s).copied())
                .map(u64::from)
                .or_else(|| json_id(v))
        });
        let id = from_token.or_else(|| model_config.get(config_key).and_then(json_id));
        if let Some(id) = id {
            if id >= real_vocab as u64 {
                return Err(format!("special token {key} id {id} is outside tokenizer's real vocabulary of {real_vocab} tokens"));
            }
            let id = u32::try_from(id)
                .map_err(|_| format!("special token {key} id exceeds GGUF range"))?;
            meta.push((format!("tokenizer.ggml.{key}"), Meta::U32(id)));
        }
    }
    if let Some(values) = model_config.get("eos_token_id").and_then(Value::as_array) {
        if values.is_empty() {
            return Err("config eos_token_id array is empty".into());
        }
        let mut eos_ids = Vec::with_capacity(values.len());
        for value in values {
            let id = value
                .as_u64()
                .ok_or("config eos_token_id array must contain only unsigned integer IDs")?;
            if id >= real_vocab as u64 {
                return Err(format!("special token eos_token_id {id} is outside tokenizer's real vocabulary of {real_vocab} tokens"));
            }
            eos_ids.push(
                u32::try_from(id).map_err(|_| "special token eos_token_id exceeds GGUF range")?,
            );
        }
        meta.push((
            "tokenizer.huggingface.eos_token_ids".into(),
            Meta::U32s(eos_ids),
        ));
    }
    if let Some(v) = tc.get("add_bos_token").and_then(Value::as_bool) {
        meta.push(("tokenizer.ggml.add_bos_token".into(), Meta::Bool(v)));
    }
    if let Some(v) = tc.get("add_eos_token").and_then(Value::as_bool) {
        meta.push(("tokenizer.ggml.add_eos_token".into(), Meta::Bool(v)));
    }
    if let Some(chat) = tc.get("chat_template") {
        let s = if let Some(s) = chat.as_str() {
            Some(s)
        } else {
            chat.as_array().and_then(|a| {
                a.iter()
                    .find(|v| v.get("name").and_then(Value::as_str) == Some("default"))
                    .and_then(|v| v.get("template"))
                    .and_then(Value::as_str)
            })
        };
        if let Some(s) = s {
            meta.push(("tokenizer.chat_template".into(), Meta::Str(s.into())));
        }
    }
    Ok((real_vocab, meta))
}
fn json_id(v: &Value) -> Option<u64> {
    if let Some(x) = v.as_u64() {
        Some(x)
    } else if let Some(a) = v.as_array() {
        a.first().and_then(Value::as_u64)
    } else {
        v.get("id").and_then(Value::as_u64)
    }
}

#[cfg(test)]
mod quantization_config_tests {
    use super::{check_config, compressed_tensors, fp8_block_size, quantization, Quantization};
    use serde_json::json;

    fn ct(weights: serde_json::Value, format: &str) -> serde_json::Value {
        json!({"quant_method": "compressed-tensors", "format": format, "config_groups": {"group_0": {"targets": ["Linear"], "weights": weights}}})
    }

    #[test]
    fn modelopt_methods_and_algorithms() {
        let nv = json!({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16}});
        assert_eq!(quantization(&nv).unwrap(), Quantization::Nvfp4);
        let fp8 = json!({"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}});
        assert_eq!(quantization(&fp8).unwrap(), Quantization::Fp8);
        let bad = json!({"quantization_config": {"quant_method": "modelopt", "quant_algo": "INT8_SQ"}});
        assert!(quantization(&bad).unwrap_err().contains("only NVFP4 and FP8"));
        let other = json!({"quantization_config": {"quant_method": "bitsandbytes"}});
        assert!(quantization(&other).unwrap_err().contains("supported: gptq, awq"));
        assert_eq!(quantization(&json!({})).unwrap(), Quantization::Dense);
    }

    #[test]
    fn compressed_tensors_schemes() {
        let int4 = ct(json!({"num_bits": 4, "type": "int", "symmetric": true, "strategy": "group", "group_size": 128}), "pack-quantized");
        assert_eq!(compressed_tensors(&int4).unwrap(), Quantization::CtPacked { bits: 4, group: 128 });
        let w8 = ct(json!({"num_bits": 8, "type": "int", "symmetric": true, "strategy": "channel"}), "int-quantized");
        assert_eq!(compressed_tensors(&w8).unwrap(), Quantization::CtInt8 { group: 0 });
        let fp8 = ct(json!({"num_bits": 8, "type": "float", "strategy": "tensor"}), "float-quantized");
        assert_eq!(compressed_tensors(&fp8).unwrap(), Quantization::Fp8);
        let asym = ct(json!({"num_bits": 4, "type": "int", "symmetric": false, "strategy": "group", "group_size": 32}), "pack-quantized");
        assert!(compressed_tensors(&asym).unwrap_err().contains("asymmetric"));
        let order = ct(json!({"num_bits": 4, "type": "int", "symmetric": true, "strategy": "group", "group_size": 32, "actorder": "group"}), "pack-quantized");
        assert!(compressed_tensors(&order).unwrap_err().contains("actorder"));
        let fp4 = ct(json!({"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16}), "nvfp4-pack-quantized");
        let err = compressed_tensors(&fp4).unwrap_err();
        assert!(err.contains("compressed-tensors NVFP4") && err.contains("ModelOpt"), "{err}");
        let mixed = json!({"quant_method": "compressed-tensors", "config_groups": {
            "a": {"weights": {"num_bits": 4, "type": "int", "strategy": "group", "group_size": 32}},
            "b": {"weights": {"num_bits": 8, "type": "int", "strategy": "channel"}}}});
        assert!(compressed_tensors(&mixed).unwrap_err().contains("mixed-precision"));
        let sparse = json!({"quant_method": "compressed-tensors", "sparsity_config": {"format": "sparse-24-bitmask"}, "config_groups": {}});
        assert!(compressed_tensors(&sparse).unwrap_err().contains("sparse"));
    }

    #[test]
    fn fp8_block_sizes_come_from_either_convention() {
        assert_eq!(fp8_block_size(&json!({"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}})), Some((128, 128)));
        let ctb = json!({"quantization_config": ct(json!({"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [64, 32]}), "float-quantized")});
        assert_eq!(fp8_block_size(&ctb), Some((64, 32)));
        let text = json!({"quantization_config": ct(json!({"num_bits": 8, "type": "float", "strategy": "block", "block_structure": "128x128"}), "float-quantized")});
        assert_eq!(fp8_block_size(&text), Some((128, 128)));
        assert_eq!(fp8_block_size(&json!({"quantization_config": {"quant_method": "fp8"}})), None);
    }

    #[test]
    fn check_config_reports_architecture_quantization_and_unsupported_types() {
        let dir = std::env::temp_dir().join(format!("fllm-check-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |v: serde_json::Value| std::fs::write(dir.join("config.json"), v.to_string()).unwrap();
        write(json!({"model_type": "llama", "hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2,
            "intermediate_size": 128, "vocab_size": 64}));
        let c = check_config(&dir).unwrap();
        assert_eq!((c.architecture.as_str(), c.quantization.as_str()), ("llama", "none"));
        write(json!({"model_type": "llama", "hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2,
            "intermediate_size": 128, "vocab_size": 64, "quantization_config": {"quant_method": "awq", "bits": 4, "version": "gemm", "zero_point": true, "group_size": 128}}));
        assert_eq!(check_config(&dir).unwrap().quantization, "awq");
        write(json!({"model_type": "not_a_model"}));
        assert!(check_config(&dir).unwrap_err().to_string().contains("unsupported model_type"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
