//! Sampling defaults that ship with a model (`--generation-config`).
//!
//! Hugging Face checkpoints carry them in `generation_config.json`; the checkpoint converter copies the
//! fields below into the GGUF as `general.sampling.*` (the key family GGUF files use for sampling defaults), and GGUF files
//! from other tools may already have them. They apply when a request leaves the parameter out, after
//! the command-line defaults (`--temp`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`).
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationDefaults {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// GGUF metadata, else `generation_config.json` next to the model file.
    Auto,
    /// No model defaults: the OpenAI/vLLM defaults apply (temperature 1.0, no truncation).
    Vllm,
    /// A directory containing `generation_config.json`, or the JSON file itself.
    Path(PathBuf),
}

impl Source {
    pub fn parse(text: &str) -> Self {
        match text.trim() {
            "auto" => Self::Auto,
            "vllm" => Self::Vllm,
            path => Self::Path(PathBuf::from(path)),
        }
    }
}

fn six_digits(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

impl GenerationDefaults {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(v) = self.temperature {
            parts.push(format!("temperature={v}"));
        }
        if let Some(v) = self.top_p {
            parts.push(format!("top_p={v}"));
        }
        if let Some(v) = self.top_k {
            parts.push(format!("top_k={v}"));
        }
        if let Some(v) = self.min_p {
            parts.push(format!("min_p={v}"));
        }
        if let Some(v) = self.repetition_penalty {
            parts.push(format!("repetition_penalty={v}"));
        }
        parts.join(", ")
    }

    fn sanitized(mut self) -> Self {
        // values a sampler cannot use are dropped instead of failing every request
        self.temperature = self.temperature.filter(|v| v.is_finite() && (0.0..=100.0).contains(v)).map(six_digits);
        self.top_p = self.top_p.filter(|v| v.is_finite() && (0.0..=1.0).contains(v)).map(six_digits);
        self.min_p = self.min_p.filter(|v| v.is_finite() && (0.0..=1.0).contains(v)).map(six_digits);
        self.repetition_penalty = self
            .repetition_penalty
            .filter(|v| v.is_finite() && (0.0001..=100.0).contains(v))
            .map(six_digits);
        self
    }

    /// The `general.sampling.*` keys of a GGUF.
    pub fn from_gguf(metadata: &Map<String, Value>) -> Self {
        let number = |key: &str| metadata.get(key).and_then(Value::as_f64);
        Self {
            temperature: number("general.sampling.temp"),
            top_p: number("general.sampling.top_p"),
            top_k: metadata.get("general.sampling.top_k").and_then(Value::as_i64).and_then(|v| usize::try_from(v).ok()),
            min_p: number("general.sampling.min_p"),
            repetition_penalty: number("general.sampling.penalty_repeat"),
        }
        .sanitized()
    }

    /// A Hugging Face `generation_config.json`. Like vLLM it takes the values whether or not `do_sample` is set;
    /// `top_k` of 0 or below means "off" there and here.
    pub fn from_hf_json(value: &Value) -> Self {
        Self {
            temperature: value.get("temperature").and_then(Value::as_f64),
            top_p: value.get("top_p").and_then(Value::as_f64),
            top_k: value.get("top_k").and_then(Value::as_i64).and_then(|v| usize::try_from(v).ok()),
            min_p: value.get("min_p").and_then(Value::as_f64),
            repetition_penalty: value.get("repetition_penalty").and_then(Value::as_f64),
        }
        .sanitized()
    }

    /// Resolves `--generation-config`. Returns the defaults and a short description of where they came from.
    pub fn resolve(source: &Source, metadata: &Map<String, Value>, model_file: &Path) -> Result<(Self, String), String> {
        let read_file = |file: &Path| -> Result<Self, String> {
            let text = std::fs::read_to_string(file).map_err(|e| format!("cannot read {}: {e}", file.display()))?;
            let value: Value = serde_json::from_str(&text).map_err(|e| format!("invalid {}: {e}", file.display()))?;
            Ok(Self::from_hf_json(&value))
        };
        match source {
            Source::Vllm => Ok((Self::default(), "none (--generation-config vllm)".into())),
            Source::Path(path) => {
                let file = if path.is_dir() { path.join("generation_config.json") } else { path.clone() };
                Ok((read_file(&file)?, file.display().to_string()))
            }
            Source::Auto => {
                let from_model = Self::from_gguf(metadata);
                if !from_model.is_empty() {
                    return Ok((from_model, "GGUF metadata".into()));
                }
                let beside = model_file.parent().map(|d| d.join("generation_config.json"));
                if let Some(file) = beside.filter(|f| f.is_file()) {
                    if let Ok(defaults) = read_file(&file) {
                        if !defaults.is_empty() {
                            return Ok((defaults, file.display().to_string()));
                        }
                    }
                }
                Ok((Self::default(), "none in the model".into()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hf_json_values_are_read_and_bad_ones_dropped() {
        let d = GenerationDefaults::from_hf_json(&json!({
            "do_sample": true, "temperature": 0.6000000238, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05, "min_p": 7.0
        }));
        assert_eq!(d.temperature, Some(0.6));
        assert_eq!(d.top_p, Some(0.95));
        assert_eq!(d.top_k, Some(20));
        assert_eq!(d.repetition_penalty, Some(1.05));
        assert_eq!(d.min_p, None, "min_p above 1 is not usable");
        assert!(GenerationDefaults::from_hf_json(&json!({"top_k": -1})).top_k.is_none());
    }

    #[test]
    fn gguf_keys_use_the_llama_cpp_names() {
        let meta = json!({"general.sampling.temp": 0.7, "general.sampling.top_k": 40, "general.sampling.penalty_repeat": 1.1})
            .as_object()
            .unwrap()
            .clone();
        let d = GenerationDefaults::from_gguf(&meta);
        assert_eq!((d.temperature, d.top_k, d.repetition_penalty), (Some(0.7), Some(40), Some(1.1)));
        assert_eq!(d.describe(), "temperature=0.7, top_k=40, repetition_penalty=1.1");
    }

    #[test]
    fn sources_resolve_in_the_documented_order() {
        let dir = std::env::temp_dir().join(format!("fllm-gencfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("generation_config.json"), r#"{"temperature": 0.3, "top_p": 0.8}"#).unwrap();
        let model = dir.join("model.gguf");
        let empty = Map::new();
        let (d, origin) = GenerationDefaults::resolve(&Source::Auto, &empty, &model).unwrap();
        assert_eq!((d.temperature, d.top_p), (Some(0.3), Some(0.8)));
        assert!(origin.contains("generation_config.json"));
        let meta = json!({"general.sampling.temp": 0.9}).as_object().unwrap().clone();
        assert_eq!(GenerationDefaults::resolve(&Source::Auto, &meta, &model).unwrap().0.temperature, Some(0.9));
        assert!(GenerationDefaults::resolve(&Source::Vllm, &meta, &model).unwrap().0.is_empty());
        let explicit = GenerationDefaults::resolve(&Source::Path(dir.clone()), &meta, &model).unwrap().0;
        assert_eq!(explicit.temperature, Some(0.3));
        assert!(GenerationDefaults::resolve(&Source::Path(dir.join("missing.json")), &meta, &model).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
