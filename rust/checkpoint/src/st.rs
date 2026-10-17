//! sentence-transformers pipelines (`modules.json`) for the encoder converters.
//!
//! A pipeline is `Transformer -> Pooling -> [Dense ...] -> [Normalize]`. The
//! pooling mode is stored in `bert.pooling_type` (1 mean, 2 cls, 3 last, 4 max,
//! 5 weighted mean, 6 mean over sqrt(length)); the modules after pooling are
//! stored as `bert.post.modules` (`dense:<activation>` or `normalize`) with the
//! Dense weights as `post.<i>.weight` / `post.<i>.bias`.
use crate::{
    gguf::{Meta, OutTensor},
    safetensors, CheckpointError, ConvertOptions, OutputDType,
};
use serde_json::Value;
use std::{fs, path::Path};

pub(crate) const MEAN: u32 = 1;
pub(crate) const CLS: u32 = 2;
pub(crate) const LAST: u32 = 3;
pub(crate) const MAX: u32 = 4;
pub(crate) const WEIGHTED_MEAN: u32 = 5;
pub(crate) const MEAN_SQRT_LEN: u32 = 6;

pub(crate) fn json(path: &Path) -> Result<Value, CheckpointError> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|e| format!("invalid {}: {e}", path.display()).into())
}

/// Pooling type of a sentence-transformers `1_Pooling/config.json`.
pub(crate) fn pooling_mode(pool: &Value) -> Result<u32, CheckpointError> {
    // sentence-transformers 5+ writes a single `pooling_mode` string.
    if let Some(mode) = pool["pooling_mode"].as_str() {
        return match mode {
            "cls" => Ok(CLS),
            "mean" => Ok(MEAN),
            "max" => Ok(MAX),
            "mean_sqrt_len_tokens" | "mean_sqrt_len" => Ok(MEAN_SQRT_LEN),
            "weightedmean" => Ok(WEIGHTED_MEAN),
            "lasttoken" | "last" => Ok(LAST),
            other => Err(format!("unsupported sentence-transformers pooling mode {other}").into()),
        };
    }
    let modes = [
        ("pooling_mode_cls_token", CLS),
        ("pooling_mode_mean_tokens", MEAN),
        ("pooling_mode_max_tokens", MAX),
        ("pooling_mode_mean_sqrt_len_tokens", MEAN_SQRT_LEN),
        ("pooling_mode_weightedmean_tokens", WEIGHTED_MEAN),
        ("pooling_mode_lasttoken", LAST),
    ];
    let active = modes
        .iter()
        .filter(|(key, _)| pool[*key].as_bool() == Some(true))
        .map(|(_, id)| *id)
        .collect::<Vec<_>>();
    match active.as_slice() {
        [one] => Ok(*one),
        [] => Err("sentence-transformers pooling config enables no pooling mode".into()),
        _ => Err("concatenating several pooling modes is not supported; enable exactly one".into()),
    }
}

/// The pooling type recorded next to the checkpoint, or `fallback()` without a pooling config.
pub(crate) fn pooling_or(
    root: &Path,
    fallback: impl FnOnce() -> u32,
) -> Result<u32, CheckpointError> {
    let path = root.join("1_Pooling/config.json");
    if path.is_file() {
        return pooling_mode(&json(&path)?);
    }
    Ok(fallback())
}

fn activation(name: &str) -> Result<&'static str, CheckpointError> {
    let lower = name.to_ascii_lowercase();
    let last = lower.rsplit('.').next().unwrap_or("");
    Ok(match last {
        "" | "identity" => "identity",
        "tanh" => "tanh",
        "relu" => "relu",
        "gelu" => "gelu",
        "sigmoid" => "sigmoid",
        "silu" => "silu",
        other => return Err(format!("unsupported Dense activation {other}").into()),
    })
}

/// Modules that follow the pooling step, as metadata plus renamed tensors.
pub(crate) struct Post {
    pub meta: Vec<(String, Meta)>,
    pub tensors: Vec<(String, safetensors::Tensor)>,
}

/// Parses `modules.json`. `Ok(None)` when the checkpoint has no pipeline definition.
pub(crate) fn read_post(root: &Path, hidden: u32) -> Result<Option<Post>, CheckpointError> {
    let path = root.join("modules.json");
    if !path.is_file() {
        return Ok(None);
    }
    let modules = json(&path)?;
    let list = modules.as_array().ok_or("modules.json must be an array")?;
    let mut ops = Vec::new();
    let mut tensors = Vec::new();
    let mut width = hidden;
    let mut seen_pooling = false;
    for module in list {
        let kind = module["type"].as_str().unwrap_or_default();
        let kind = kind.rsplit('.').next().unwrap_or(kind);
        let dir = module["path"].as_str().unwrap_or_default();
        if dir.contains("..") || Path::new(dir).is_absolute() {
            return Err("modules.json path must stay inside the checkpoint".into());
        }
        match kind {
            "Transformer" | "MLMTransformer" => {
                if seen_pooling || !ops.is_empty() {
                    return Err("modules.json: Transformer must come first".into());
                }
            }
            "Pooling" | "SpladePooling" => seen_pooling = true,
            "Dense" => {
                let dir = root.join(dir);
                let cfg = json(&dir.join("config.json"))?;
                let input = cfg["in_features"].as_u64().ok_or("Dense config lacks in_features")?;
                let output = cfg["out_features"].as_u64().ok_or("Dense config lacks out_features")?;
                if input != u64::from(width) || output == 0 || output > 65536 {
                    return Err(format!(
                        "Dense module expects {input} inputs but the previous module produces {width}"
                    )
                    .into());
                }
                let act = activation(cfg["activation_function"].as_str().unwrap_or("identity"))?;
                let index = ops.len();
                let weights = safetensors::read_checkpoint(&dir)?;
                let find = |suffix: &str| {
                    weights
                        .iter()
                        .position(|t| t.name == format!("linear.{suffix}") || t.name == suffix)
                };
                let wi = find("weight").ok_or("Dense module has no weight tensor")?;
                if weights[wi].shape != [output, input] {
                    return Err("Dense weight shape disagrees with its config".into());
                }
                let has_bias = cfg["bias"].as_bool().unwrap_or(true);
                let bi = if has_bias { find("bias") } else { None };
                if has_bias && bi.is_none() {
                    return Err("Dense config requests a bias but none was found".into());
                }
                let mut weights = weights.into_iter().map(Some).collect::<Vec<_>>();
                let prefix = if seen_pooling { "post" } else { "pre" };
                tensors.push((format!("{prefix}.{index}.weight"), weights[wi].take().unwrap()));
                if let Some(bi) = bi {
                    tensors.push((format!("{prefix}.{index}.bias"), weights[bi].take().unwrap()));
                }
                ops.push(if seen_pooling {
                    format!("dense:{act}")
                } else {
                    format!("token_dense:{act}")
                });
                width = output as u32;
            }
            "Normalize" => ops.push("normalize".to_owned()),
            other => {
                return Err(format!("unsupported sentence-transformers module {other}").into())
            }
        }
    }
    let meta = vec![("bert.post.modules".to_owned(), Meta::Strs(ops))];
    Ok(Some(Post { meta, tensors }))
}

/// Converts one source tensor into an output tensor under `name` (F32/F16/BF16 sources only).
pub(crate) fn out_tensor(
    t: safetensors::Tensor,
    name: String,
    options: &ConvertOptions,
) -> Result<OutTensor, CheckpointError> {
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
    Ok(OutTensor {
        name,
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
    })
}

/// Appends the pipeline metadata and tensors found next to the checkpoint to a converter's output.
pub(crate) fn extend(
    root: &Path,
    hidden: u32,
    options: &ConvertOptions,
    meta: &mut Vec<(String, Meta)>,
    tensors: &mut Vec<OutTensor>,
) -> Result<(), CheckpointError> {
    if let Some(post) = read_post(root, hidden)? {
        meta.extend(post.meta);
        for (name, t) in post.tensors {
            tensors.push(out_tensor(t, name, options)?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pooling_modes_require_exactly_one() {
        assert_eq!(pooling_mode(&json!({"pooling_mode_cls_token": true})).unwrap(), CLS);
        assert_eq!(
            pooling_mode(&json!({"pooling_mode_cls_token": false, "pooling_mode_max_tokens": true}))
                .unwrap(),
            MAX
        );
        assert_eq!(
            pooling_mode(&json!({"pooling_mode_weightedmean_tokens": true})).unwrap(),
            WEIGHTED_MEAN
        );
        assert_eq!(pooling_mode(&json!({"pooling_mode": "lasttoken"})).unwrap(), LAST);
        assert!(pooling_mode(&json!({"pooling_mode": "median"})).is_err());
        assert!(pooling_mode(&json!({})).is_err());
        assert!(pooling_mode(
            &json!({"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": true})
        )
        .is_err());
    }

    #[test]
    fn activation_names() {
        assert_eq!(activation("torch.nn.modules.activation.Tanh").unwrap(), "tanh");
        assert_eq!(activation("torch.nn.modules.linear.Identity").unwrap(), "identity");
        assert!(activation("torch.nn.modules.activation.Mish").is_err());
    }
}
