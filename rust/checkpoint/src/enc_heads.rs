//! Output heads of encoder checkpoints beyond the pooled embedding.
//!
//! * MLM head used by SPLADE-style sparse encoders (`mlm.*`, `bert.head.mlm`).
//! * Token classification head (`token_cls.*`, `bert.token_classifier.labels`).
//! * BGE-M3 `colbert_linear` / `sparse_linear` heads and the `linear` projection of
//!   `HF_ColBERT` checkpoints (`colbert.*`, `sparse.*`).
//!
//! The loader in `rust/engine/src/encoder_heads.rs` reads the keys written here.
use crate::{
    gguf::{Meta, OutTensor},
    safetensors::{self, Tensor},
    st, CheckpointError, ConvertOptions,
};
use serde_json::Value;
use std::path::Path;

fn archs(cfg: &Value) -> Vec<String> {
    cfg["architectures"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default()
}

pub(crate) fn is_token_classification(cfg: &Value) -> bool {
    archs(cfg).iter().any(|a| a.ends_with("ForTokenClassification"))
}

fn take(sources: &mut Vec<Tensor>, names: &[&str]) -> Option<Tensor> {
    let pos = sources.iter().position(|t| {
        let n = t.name.strip_prefix("model.").unwrap_or(&t.name);
        names.contains(&n)
    })?;
    Some(sources.remove(pos))
}

fn require(
    sources: &mut Vec<Tensor>,
    names: &[&str],
    shape: &[u64],
    what: &str,
) -> Result<Tensor, CheckpointError> {
    let t = take(sources, names).ok_or_else(|| format!("missing {what} tensor {}", names[0]))?;
    if t.shape != shape {
        return Err(format!(
            "{what} tensor {} has shape {:?}, expected {shape:?}",
            t.name, t.shape
        )
        .into());
    }
    Ok(t)
}

/// SPLADE detection: sentence-transformers sparse modules or a name that says so.
fn splade_config(root: &Path, cfg: &Value) -> Result<Option<(String, String)>, CheckpointError> {
    let default = ("log1p_relu".to_owned(), "max".to_owned());
    let modules = root.join("modules.json");
    if modules.is_file() {
        let list = st::json(&modules)?;
        let mut found = false;
        let mut result = default.clone();
        for m in list.as_array().into_iter().flatten() {
            let kind = m["type"].as_str().unwrap_or_default();
            if kind.ends_with("MLMTransformer") {
                found = true;
            }
            if kind.ends_with("SpladePooling") {
                found = true;
                let dir = root.join(m["path"].as_str().unwrap_or_default());
                if dir.join("config.json").is_file() {
                    let c = st::json(&dir.join("config.json"))?;
                    if let Some(a) = c["activation_function"].as_str() {
                        result.0 = a.to_owned();
                    }
                    if let Some(p) = c["pooling_strategy"].as_str() {
                        result.1 = p.to_owned();
                    }
                }
            }
        }
        if found {
            if !matches!(result.0.as_str(), "relu" | "log1p_relu") || !matches!(result.1.as_str(), "max" | "sum") {
                return Err(format!(
                    "unsupported SPLADE pooling {}/{}",
                    result.0, result.1
                )
                .into());
            }
            return Ok(Some(result));
        }
    }
    let name = format!(
        "{} {}",
        root.display(),
        cfg["_name_or_path"].as_str().unwrap_or_default()
    )
    .to_lowercase();
    let masked = archs(cfg).iter().any(|a| a.ends_with("ForMaskedLM"));
    if masked && (name.contains("splade") || name.contains("neural-sparse")) {
        return Ok(Some(default));
    }
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn collect(
    root: &Path,
    cfg: &Value,
    model_type: &str,
    sources: &mut Vec<Tensor>,
    hidden: u32,
    vocab: u32,
    options: &ConvertOptions,
    meta: &mut Vec<(String, Meta)>,
    tensors: &mut Vec<OutTensor>,
) -> Result<(), CheckpointError> {
    let h = u64::from(hidden);
    let mut push = |t: Tensor, name: &str| -> Result<(), CheckpointError> {
        tensors.push(st::out_tensor(t, name.to_owned(), options)?);
        Ok(())
    };
    // MLM head (SPLADE).
    if let Some((activation, pooling)) = splade_config(root, cfg)? {
        let (dense, norm, bias, decoder): (&str, &str, &[&str], &str) = match model_type {
            "distilbert" => (
                "vocab_transform",
                "vocab_layer_norm",
                &["vocab_projector.bias"],
                "vocab_projector.weight",
            ),
            "roberta" | "xlm-roberta" => (
                "lm_head.dense",
                "lm_head.layer_norm",
                &["lm_head.bias", "lm_head.decoder.bias"],
                "lm_head.decoder.weight",
            ),
            _ => (
                "cls.predictions.transform.dense",
                "cls.predictions.transform.LayerNorm",
                &["cls.predictions.bias", "cls.predictions.decoder.bias"],
                "cls.predictions.decoder.weight",
            ),
        };
        let v = u64::from(vocab);
        let w = require(sources, &[&format!("{dense}.weight")], &[h, h], "MLM")?;
        push(w, "mlm.dense.weight")?;
        let b = require(sources, &[&format!("{dense}.bias")], &[h], "MLM")?;
        push(b, "mlm.dense.bias")?;
        for suffix in ["weight", "bias"] {
            // BERT names its LayerNorm weight "weight"/"bias"; older checkpoints use gamma/beta.
            let alt = if suffix == "weight" { "gamma" } else { "beta" };
            let t = require(
                sources,
                &[&format!("{norm}.{suffix}"), &format!("{norm}.{alt}")],
                &[h],
                "MLM",
            )?;
            push(t, &format!("mlm.norm.{suffix}"))?;
        }
        let b = require(sources, bias, &[v], "MLM")?;
        push(b, "mlm.bias")?;
        // The decoder is tied to the word embeddings in every SPLADE release; keep it only if it differs.
        if let Some(dec) = take(sources, &[decoder]) {
            let tied = sources
                .iter()
                .find(|t| t.name.ends_with("embeddings.word_embeddings.weight"))
                .is_some_and(|e| e.payload.as_slice() == dec.payload.as_slice());
            if !tied {
                if dec.shape != [v, h] {
                    return Err("MLM decoder has an unexpected shape".into());
                }
                push(dec, "mlm.decoder.weight")?;
            }
        }
        meta.push(("bert.head.dense".into(), Meta::Bool(false)));
        meta.push(("bert.head.mlm".into(), Meta::Bool(true)));
        meta.push(("bert.sparse.kind".into(), Meta::Str("splade".into())));
        meta.push(("bert.sparse.activation".into(), Meta::Str(activation)));
        meta.push(("bert.sparse.pooling".into(), Meta::Str(pooling)));
    }
    // Token classification head.
    if is_token_classification(cfg) {
        let id2label = cfg["id2label"].as_object().ok_or("token classification needs id2label")?;
        let labels = id2label.len() as u64;
        if labels == 0 || labels > 4096 {
            return Err("invalid token classification label count".into());
        }
        let w = require(sources, &["classifier.weight"], &[labels, h], "token classifier")?;
        push(w, "token_cls.weight")?;
        let b = require(sources, &["classifier.bias"], &[labels], "token classifier")?;
        push(b, "token_cls.bias")?;
        let names = (0..labels)
            .map(|id| {
                id2label
                    .get(&id.to_string())
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| format!("config.json id2label is missing string label {id}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        meta.push(("bert.token_classifier.labels".into(), Meta::Strs(names)));
    }
    // BGE-M3 sparse and ColBERT heads (separate torch files next to the checkpoint).
    let load_pt = |name: &str| -> Result<Option<Vec<Tensor>>, CheckpointError> {
        let path = root.join(name);
        if path.is_file() {
            Ok(Some(safetensors::read_checkpoint(&path)?))
        } else {
            Ok(None)
        }
    };
    if let Some(mut sparse) = load_pt("sparse_linear.pt")? {
        let w = require(&mut sparse, &["weight"], &[1, h], "sparse_linear")?;
        push(w, "sparse.weight")?;
        let b = require(&mut sparse, &["bias"], &[1], "sparse_linear")?;
        push(b, "sparse.bias")?;
        meta.push(("bert.head.sparse_linear".into(), Meta::Bool(true)));
    }
    if let Some(mut colbert) = load_pt("colbert_linear.pt")? {
        let dim = colbert
            .iter()
            .find(|t| t.name == "weight")
            .and_then(|t| t.shape.first().copied())
            .ok_or("colbert_linear has no weight")?;
        let w = require(&mut colbert, &["weight"], &[dim, h], "colbert_linear")?;
        push(w, "colbert.weight")?;
        let b = require(&mut colbert, &["bias"], &[dim], "colbert_linear")?;
        push(b, "colbert.bias")?;
        meta.push(("bert.head.colbert".into(), Meta::Bool(true)));
        meta.push(("bert.colbert.dim".into(), Meta::U32(dim as u32)));
        // FlagEmbedding drops the CLS vector and averages the MaxSim sum over the query tokens.
        meta.push(("bert.colbert.skip".into(), Meta::U32(1)));
        meta.push(("bert.colbert.mean_score".into(), Meta::Bool(true)));
    }
    // HF_ColBERT (colbert-ir/colbertv2.0): "linear.weight" without bias.
    let is_colbert = archs(cfg).iter().any(|a| a.contains("ColBERT"));
    if is_colbert && !meta.iter().any(|(k, _)| k == "bert.head.colbert") {
        let dim = sources
            .iter()
            .find(|t| t.name == "linear.weight")
            .and_then(|t| t.shape.first().copied())
            .ok_or("ColBERT checkpoint has no linear.weight projection")?;
        let w = require(sources, &["linear.weight"], &[dim, h], "ColBERT projection")?;
        push(w, "colbert.weight")?;
        meta.push(("bert.head.dense".into(), Meta::Bool(false)));
        meta.push(("bert.head.colbert".into(), Meta::Bool(true)));
        meta.push(("bert.colbert.dim".into(), Meta::U32(dim as u32)));
        meta.push(("bert.colbert.skip".into(), Meta::U32(0)));
        meta.push(("bert.colbert.mean_score".into(), Meta::Bool(false)));
        colbert_markers(meta)?;
    }
    Ok(())
}

/// Query/document markers, query augmentation and the punctuation skip list of the original ColBERT.
fn colbert_markers(meta: &mut Vec<(String, Meta)>) -> Result<(), CheckpointError> {
    let tokens = meta
        .iter()
        .find_map(|(k, v)| match v {
            Meta::Strs(s) if k == "tokenizer.ggml.tokens" => Some(s.clone()),
            _ => None,
        })
        .ok_or("missing tokenizer vocabulary")?;
    let find = |s: &str| tokens.iter().position(|t| t == s).map(|p| p as u32);
    let (Some(q), Some(d), Some(mask)) = (find("[unused0]"), find("[unused1]"), find("[MASK]")) else {
        return Err("ColBERT conversion needs [unused0], [unused1] and [MASK] in the vocabulary".into());
    };
    let skip = r##"!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"##
        .chars()
        .filter_map(|c| find(&c.to_string()))
        .collect::<Vec<_>>();
    meta.push(("bert.colbert.query_marker".into(), Meta::U32(q)));
    meta.push(("bert.colbert.doc_marker".into(), Meta::U32(d)));
    meta.push(("bert.colbert.mask_id".into(), Meta::U32(mask)));
    meta.push(("bert.colbert.query_length".into(), Meta::U32(32)));
    meta.push(("bert.colbert.doc_length".into(), Meta::U32(180)));
    meta.push(("bert.colbert.skip_ids".into(), Meta::U32s(skip)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn token_classification_is_detected_from_architectures() {
        assert!(is_token_classification(
            &json!({"architectures": ["BertForTokenClassification"]})
        ));
        assert!(!is_token_classification(
            &json!({"architectures": ["BertForSequenceClassification"]})
        ));
        assert!(!is_token_classification(&json!({})));
    }

    #[test]
    fn colbert_markers_need_the_special_vocabulary() {
        let mut meta = vec![(
            "tokenizer.ggml.tokens".to_owned(),
            Meta::Strs(vec!["[unused0]".into(), "[unused1]".into(), "[MASK]".into(), ".".into(), "a".into()]),
        )];
        colbert_markers(&mut meta).unwrap();
        let skip = meta.iter().find(|(k, _)| k == "bert.colbert.skip_ids").unwrap();
        assert!(matches!(&skip.1, Meta::U32s(v) if v == &[3]));
        let mut bad = vec![("tokenizer.ggml.tokens".to_owned(), Meta::Strs(vec!["a".into()]))];
        assert!(colbert_markers(&mut bad).is_err());
    }
}
