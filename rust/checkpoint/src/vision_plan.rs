//! Mapping of Hugging Face vision tower tensors to engine-side GGUF tensors.
//!
//! fLLM stores the vision tower inside the model GGUF (next to the language model) under the
//! prefix `v.hf.`. The tensors keep their Hugging Face names minus the root prefix of their family,
//! and no per-family renaming table has to be kept in sync on both sides. Rank-3-and-higher
//! tensors (the Conv3d patch embedding) are flattened to `[out, rest]`, which is how the engine
//! applies them (as a matrix over flattened patches).
//!
//! Metadata written (only when a tower is imported):
//! * `fllm.vision.family`: engine-side family name (`qwen2_vl`, `qwen2_5_vl`, ...);
//! * `fllm.vision.config`: JSON text with `vision` (the `vision_config` of `config.json`),
//!   `preprocessor` (`preprocessor_config.json`, or `{}`), `video_preprocessor`
//!   (`video_preprocessor_config.json`, or `{}`) and `tokens` (the special token ids);
//! * `fllm.rope.mrope_sections` / `fllm.rope.mrope_interleaved`: multimodal rotary sections of the
//!   language model (Qwen2-VL family).
use crate::gguf::Meta;
use serde_json::{json, Value};
use std::path::Path;

/// Tensor name prefix of imported vision tower weights.
pub const PREFIX: &str = "v.hf.";

struct Family {
    /// `model_type` values of `config.json` that select the family.
    model_types: &'static [&'static str],
    name: &'static str,
    /// The language model alternates the position axes lane by lane (Qwen3-VL).
    interleaved_mrope: bool,
    /// Module prefixes (tried in order) that identify tower tensors in the checkpoint.
    prefixes: &'static [&'static str],
}

const FAMILIES: &[Family] = &[
    Family {
        model_types: &["qwen2_vl"],
        name: "qwen2_vl",
        interleaved_mrope: false,
        prefixes: &["model.visual.", "visual."],
    },
    Family {
        model_types: &["qwen2_5_vl"],
        name: "qwen2_5_vl",
        interleaved_mrope: false,
        prefixes: &["model.visual.", "visual."],
    },
    Family {
        model_types: &["qwen3_vl", "qwen3_vl_moe"],
        name: "qwen3_vl",
        interleaved_mrope: true,
        prefixes: &["model.visual.", "visual."],
    },
    Family {
        model_types: &["paligemma", "paligemma2"],
        name: "paligemma",
        interleaved_mrope: false,
        prefixes: &[
            "model.vision_tower.vision_model.",
            "vision_tower.vision_model.",
            "model.vision_tower.",
            "vision_tower.",
        ],
    },
    Family {
        model_types: &["gemma3", "gemma3_vision"],
        name: "gemma3",
        interleaved_mrope: false,
        prefixes: &[
            "model.vision_tower.vision_model.",
            "vision_tower.vision_model.",
            "model.vision_tower.",
            "vision_tower.",
        ],
    },
    Family {
        model_types: &["mllama", "llama4"],
        name: "mllama",
        interleaved_mrope: false,
        prefixes: &[
            "model.vision_model.",
            "vision_model.",
        ],
    },
    Family {
        model_types: &["llava", "llava_next"],
        name: "llava",
        interleaved_mrope: false,
        prefixes: &[
            "model.vision_tower.",
            "vision_tower.",
        ],
    },
    Family {
        model_types: &["whisper"],
        name: "whisper",
        interleaved_mrope: false,
        prefixes: &[
            "model.encoder.",
            "encoder.",
        ],
    },
];

pub struct VisionPlan {
    family: &'static Family,
    pub meta: Vec<(String, Meta)>,
}

impl VisionPlan {
    /// The plan for a multimodal `config.json` (the full file, before `text_config` is flattened),
    /// or `None` when the model is text-only or its tower is not supported by the engine.
    pub fn from_config(cfg: &Value, root: &Path) -> Result<Option<Self>, String> {
        let Some(model_type) = cfg.get("model_type").and_then(Value::as_str) else {
            return Ok(None);
        };
        let Some(family) = FAMILIES.iter().find(|f| f.model_types.contains(&model_type)) else {
            return Ok(None);
        };
        if std::env::var("FLLM_CONVERT_VISION").is_ok_and(|v| matches!(v.as_str(), "0" | "off" | "false")) {
            return Ok(None);
        }
        let vision = if family.name == "whisper" {
            cfg.get("encoder_config")
                .or_else(|| cfg.get("vision_config"))
                .filter(|v| v.is_object())
                .unwrap_or(cfg)
        } else {
            cfg.get("vision_config")
                .filter(|v| v.is_object())
                .ok_or_else(|| format!("{model_type} config.json has no vision_config"))?
        };
        let preprocessor = match std::fs::read(root.join("preprocessor_config.json")) {
            Ok(bytes) => serde_json::from_slice::<Value>(&bytes)
                .map_err(|e| format!("invalid preprocessor_config.json: {e}"))?,
            Err(_) => json!({}),
        };
        let video = match std::fs::read(root.join("video_preprocessor_config.json")) {
            Ok(bytes) => serde_json::from_slice::<Value>(&bytes)
                .map_err(|e| format!("invalid video_preprocessor_config.json: {e}"))?,
            Err(_) => json!({}),
        };
        let mut tokens = serde_json::Map::new();
        for key in [
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ] {
            if let Some(v) = cfg.get(key).filter(|v| v.is_u64()) {
                tokens.insert(key.to_owned(), v.clone());
            }
        }
        let config = json!({"vision": vision, "preprocessor": preprocessor, "video_preprocessor": video, "tokens": tokens});
        let mut meta = vec![
            ("fllm.vision.family".to_owned(), Meta::Str(family.name.to_owned())),
            ("fllm.vision.config".to_owned(), Meta::Str(config.to_string())),
        ];
        if let Some((sections, interleaved)) = mrope_sections(cfg) {
            meta.push(("fllm.rope.mrope_sections".to_owned(), Meta::U32s(sections)));
            meta.push((
                "fllm.rope.mrope_interleaved".to_owned(),
                Meta::Bool(interleaved || family.interleaved_mrope),
            ));
        }
        Ok(Some(Self { family, meta }))
    }

    /// GGUF tensor name for a checkpoint tensor that belongs to the vision tower.
    pub fn tensor_name(&self, hf: &str) -> Option<String> {
        for proj_prefix in [
            "model.multi_modal_projector.",
            "multi_modal_projector.",
            "model.mm_projector.",
            "mm_projector.",
        ] {
            if let Some(rest) = hf.strip_prefix(proj_prefix) {
                return Some(format!("{PREFIX}projector.{rest}"));
            }
        }
        self.family
            .prefixes
            .iter()
            .find_map(|p| hf.strip_prefix(p))
            .map(|rest| format!("{PREFIX}{rest}"))
    }
}

/// `mrope_section` of the language model: from `rope_parameters` (Transformers 5), `rope_scaling`
/// (older files), in the top level or in `text_config`. The second value is Qwen3-VL's
/// `mrope_interleaved`.
pub fn mrope_sections(cfg: &Value) -> Option<(Vec<u32>, bool)> {
    let scopes = [Some(cfg), cfg.get("text_config")];
    for scope in scopes.into_iter().flatten() {
        for key in ["rope_parameters", "rope_scaling"] {
            let Some(p) = scope.get(key).filter(|v| v.is_object()) else {
                continue;
            };
            let Some(sections) = p.get("mrope_section").and_then(Value::as_array) else {
                continue;
            };
            let sections: Option<Vec<u32>> = sections
                .iter()
                .map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()))
                .collect();
            if let Some(sections) = sections.filter(|s| !s.is_empty()) {
                let interleaved = p.get("mrope_interleaved").and_then(Value::as_bool).unwrap_or(false);
                return Some((sections, interleaved));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_qwen_tower_tensors_only() {
        let cfg = json!({"model_type": "qwen2_vl", "vision_config": {"depth": 2}, "image_token_id": 7,
            "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]}});
        let plan = VisionPlan::from_config(&cfg, Path::new("/nonexistent")).unwrap().unwrap();
        assert_eq!(
            plan.tensor_name("visual.blocks.0.attn.qkv.weight").as_deref(),
            Some("v.hf.blocks.0.attn.qkv.weight")
        );
        assert_eq!(
            plan.tensor_name("model.visual.merger.mlp.2.bias").as_deref(),
            Some("v.hf.merger.mlp.2.bias")
        );
        assert_eq!(plan.tensor_name("model.layers.0.mlp.up_proj.weight"), None);
        let keys: Vec<_> = plan.meta.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"fllm.vision.config") && keys.contains(&"fllm.rope.mrope_sections"));
        let text = plan.meta.iter().find_map(|(k, v)| match v {
            Meta::Str(s) if k == "fllm.vision.config" => Some(s.clone()),
            _ => None,
        });
        let parsed: Value = serde_json::from_str(&text.unwrap()).unwrap();
        assert_eq!(parsed["tokens"]["image_token_id"], 7);
        assert_eq!(parsed["vision"]["depth"], 2);
    }

    #[test]
    fn text_only_and_unsupported_models_have_no_plan() {
        let none = |cfg: Value| VisionPlan::from_config(&cfg, Path::new("/nonexistent")).unwrap().is_none();
        assert!(none(json!({"model_type": "llama"})));
        assert!(none(json!({"model_type": "deepseek", "vision_config": {}})));
        assert!(VisionPlan::from_config(&json!({"model_type": "qwen2_vl"}), Path::new("/x")).is_err());
    }

    #[test]
    fn maps_multimodal_projector_and_paligemma_tensors() {
        let cfg = json!({"model_type": "paligemma", "vision_config": {"hidden_size": 1152}, "image_token_id": 257152});
        let plan = VisionPlan::from_config(&cfg, Path::new("/nonexistent")).unwrap().unwrap();
        assert_eq!(
            plan.tensor_name("multi_modal_projector.linear.weight").as_deref(),
            Some("v.hf.projector.linear.weight")
        );
        assert_eq!(
            plan.tensor_name("vision_tower.vision_model.embeddings.patch_embedding.weight").as_deref(),
            Some("v.hf.embeddings.patch_embedding.weight")
        );
    }

    #[test]
    fn finds_mrope_sections_in_both_layouts() {
        let new = json!({"text_config": {"rope_parameters": {"mrope_section": [2, 3, 3], "mrope_interleaved": true}}});
        assert_eq!(mrope_sections(&new), Some((vec![2, 3, 3], true)));
        let old = json!({"rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]}});
        assert_eq!(mrope_sections(&old), Some((vec![16, 24, 24], false)));
        assert_eq!(mrope_sections(&json!({"rope_theta": 1.0})), None);
    }
}
