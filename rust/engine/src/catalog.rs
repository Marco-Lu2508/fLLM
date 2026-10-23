//! Model catalog: short aliases such as `qwen3.5:9b` mapped to Hugging Face GGUF
//! repositories, with user-extensible TOML files and automatic quantization
//! selection based on available device memory.

use crate::devices::{list_devices, DeviceType};
use crate::hub;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogEntry {
    pub name: String,
    pub repo: String,
    pub default_quant: String,
    pub quants: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Catalog {
    pub version: u32,
    pub models: Vec<CatalogEntry>,
}

pub const DEFAULT_QUANTS: &[&str] = &[
    "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M", "Q2_K",
];

/// Default tag for a family when the user gives a bare family name (e.g. `llama3.1` -> `llama3.1:8b`).
const FAMILY_DEFAULTS: &[(&str, &str)] = &[
    ("llama3", "llama3:8b"),
    ("llama3.1", "llama3.1:8b"),
    ("llama3.2", "llama3.2:3b"),
    ("llama3.3", "llama3.3:70b"),
    ("llama4", "llama4:scout"),
    ("qwen2.5", "qwen2.5:7b"),
    ("qwen2.5-coder", "qwen2.5-coder:7b"),
    ("qwq", "qwq:32b"),
    ("qwen3", "qwen3:8b"),
    ("qwen3.5", "qwen3.5:9b"),
    ("qwen3.6", "qwen3.6:35b-a3b"),
    ("gemma2", "gemma2:9b"),
    ("gemma3", "gemma3:4b"),
    ("mistral", "mistral:7b"),
    ("mistral-nemo", "mistral-nemo:12b"),
    ("mistral-small", "mistral-small:24b"),
    ("mistral-large", "mistral-large:123b"),
    ("mixtral", "mixtral:8x7b"),
    ("codestral", "codestral:22b"),
    ("devstral", "devstral:24b"),
    ("ministral", "ministral:8b"),
    ("deepseek-r1", "deepseek-r1:8b"),
    ("deepseek-v3", "deepseek-v3:671b"),
    ("deepseek-coder-v2", "deepseek-coder-v2:16b"),
    ("phi3", "phi3:3.8b"),
    ("phi3.5", "phi3.5:3.8b"),
    ("phi4", "phi4:14b"),
    ("phi4-mini", "phi4-mini:3.8b"),
    ("phi4-reasoning", "phi4-reasoning:14b"),
    ("smollm2", "smollm2:1.7b"),
    ("granite3", "granite3:8b"),
    ("granite3.1", "granite3.1:8b"),
    ("granite3.2", "granite3.2:8b"),
    ("command-r", "command-r:35b"),
    ("command-r-plus", "command-r-plus:104b"),
    ("command-r7b", "command-r7b:7b"),
    ("aya-expanse", "aya-expanse:8b"),
    ("internlm2.5", "internlm2.5:7b"),
    ("internlm", "internlm3:8b"),
    ("internlm3", "internlm3:8b"),
    ("starcoder2", "starcoder2:7b"),
    ("yi", "yi:9b"),
    ("yi-coder", "yi-coder:9b"),
    ("olmo2", "olmo2:7b"),
    ("glm4", "glm4:9b"),
    ("nemotron-mini", "nemotron-mini:4b"),
];

pub fn catalog() -> Vec<CatalogEntry> {
    load().models
}

pub fn find(name: &str) -> Option<CatalogEntry> {
    let normalized = name.trim().to_ascii_lowercase();
    let stripped = normalized.strip_suffix(":latest").unwrap_or(&normalized);
    let target = FAMILY_DEFAULTS
        .iter()
        .find_map(|(family, canonical)| (*family == stripped).then_some(*canonical))
        .unwrap_or(stripped);
    catalog()
        .into_iter()
        .find(|entry| entry.name.eq_ignore_ascii_case(target) || entry.name.eq_ignore_ascii_case(&normalized))
}

/// Memory a model can use: the free memory of the best discrete GPU, else (integrated GPU, CPU, or a GPU that reports nothing) the free
/// memory of the computer, which both of them share.
pub fn available_memory_bytes() -> u64 {
    let discrete = list_devices()
        .iter()
        .filter(|d| d.dev_type == DeviceType::Gpu)
        .map(|d| d.free_memory_bytes)
        .max()
        .unwrap_or(0);
    if discrete > 0 {
        discrete
    } else {
        crate::devices::host_free_memory_bytes()
    }
}

pub fn load() -> Catalog {
    let mut merged: BTreeMap<String, CatalogEntry> = BTreeMap::new();
    let mut version = 1;
    for entry in parse_str(include_str!("catalog.toml"), &mut version) {
        merged.insert(entry.name.clone(), entry);
    }
    if let Some(path) = user_catalog_path() {
        if let Ok(text) = fs::read_to_string(&path) {
            for entry in parse_str(&text, &mut version) {
                merged.insert(entry.name.clone(), entry);
            }
        }
    }
    Catalog {
        version,
        models: merged.into_values().collect(),
    }
}

fn user_catalog_path() -> Option<PathBuf> {
    let base = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("APPDATA").map(PathBuf::from))
        .or_else(|| {
            env::var_os("HOME")
                .or_else(|| env::var_os("USERPROFILE"))
                .map(|home| PathBuf::from(home).join(".config"))
        })?;
    Some(base.join("fllm").join("catalog.toml"))
}

fn parse_str(text: &str, _version: &mut u32) -> Vec<CatalogEntry> {
    let mut models = Vec::new();
    let mut current: Option<CatalogEntry> = None;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[models]]" || line == "[[model]]" {
            if let Some(entry) = current.take() {
                if !entry.name.is_empty() {
                    models.push(entry);
                }
            }
            current = Some(CatalogEntry {
                name: String::new(),
                repo: String::new(),
                default_quant: "Q4_K_M".to_owned(),
                quants: Vec::new(),
            });
            continue;
        }
        if line.starts_with('[') {
            if let Some(entry) = current.take() {
                if !entry.name.is_empty() {
                    models.push(entry);
                }
            }
            current = None;
            continue;
        }
        if line.starts_with("version") {
            // Accept a top-level version = N line.
            if let Some((_, value)) = line.split_once('=') {
                let value = value.trim().trim_matches('"');
                // ignore parse errors; keep default
                let _ = value;
            }
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        if current.is_none() {
            current = Some(CatalogEntry {
                name: String::new(),
                repo: String::new(),
                default_quant: "Q4_K_M".to_owned(),
                quants: Vec::new(),
            });
        }
        let entry = current.as_mut().unwrap();
        match key {
            "name" => entry.name = unquote(value),
            "repo" | "repository" => entry.repo = unquote(value),
            "default" | "default_quant" | "default-quant" => {
                entry.default_quant = unquote(value).to_ascii_uppercase();
            }
            "quants" | "quantizations" => {
                entry.quants = parse_string_array(value);
            }
            _ => {}
        }
    }
    if let Some(entry) = current.take() {
        if !entry.name.is_empty() {
            models.push(entry);
        }
    }
    models
}

fn unquote(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_owned()
}

fn parse_string_array(value: &str) -> Vec<String> {
    let value = value.trim();
    if !value.starts_with('[') {
        return vec![unquote(value)];
    }
    value
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|item| unquote(item).to_ascii_uppercase())
        .filter(|item| !item.is_empty())
        .collect()
}

/// Infers approximate total parameter count (in billions) from a catalog model tag such as `llama3.1:8b`, `smollm2:135m`,
/// `mixtral:8x7b`, `qwen3.5:35b-a3b`, or `llama4:scout`.
pub fn infer_parameters_billion(name: &str) -> Option<f64> {
    let tag = name.split_once(':').map_or(name, |(_, t)| t).to_ascii_lowercase();
    match tag.as_str() {
        "scout" => return Some(109.0),
        "maverick" => return Some(400.0),
        _ => {}
    }
    let primary = tag.split('-').next().unwrap_or(&tag);
    if let Some((experts, per_expert)) = primary.split_once('x') {
        if let Ok(e) = experts.parse::<f64>() {
            if let Some(b) = per_expert.strip_suffix('b').and_then(|v| v.parse::<f64>().ok()) {
                // MoE shared parameters reduce total slightly below e * b
                return Some(e * b * 0.85);
            }
        }
    }
    if let Some(b) = primary.strip_suffix('b') {
        if let Ok(val) = b.parse::<f64>() {
            return Some(val);
        }
    }
    if let Some(m) = primary.strip_suffix('m') {
        if let Ok(val) = m.parse::<f64>() {
            return Some(val / 1000.0);
        }
    }
    None
}

/// Estimates GGUF file size in bytes for a model of `params_billion` billion parameters at `quant`.
pub fn estimate_quant_bytes(params_billion: f64, quant: &str) -> u64 {
    let bytes_per_param = match quant.to_ascii_uppercase().as_str() {
        "F32" => 4.0,
        "F16" | "BF16" => 2.0,
        "Q8_0" => 1.08,
        "Q6_K" => 0.82,
        "Q5_K_M" | "Q5_K_S" | "Q5_0" | "Q5_1" => 0.70,
        "Q4_K_M" | "Q4_K_S" | "Q4_0" | "Q4_1" => 0.58,
        "IQ4_XS" | "IQ4_NL" => 0.53,
        "Q3_K_L" | "Q3_K_M" | "Q3_K_S" | "IQ3_M" | "IQ3_XS" => 0.45,
        "Q2_K" | "IQ2_M" | "IQ2_S" | "IQ2_XS" => 0.36,
        _ => 0.60,
    };
    let overhead = 256.0 * 1024.0 * 1024.0; // tokenizer + metadata + unquantized norms/embeddings margin
    ((params_billion * 1e9 * bytes_per_param) + overhead) as u64
}

/// Offline VRAM-aware quantization recommendation for `entry` given `available_memory` bytes.
pub fn recommend_quant(entry: &CatalogEntry, available_memory: u64) -> String {
    let Some(params_b) = infer_parameters_billion(&entry.name) else {
        return entry.default_quant.clone();
    };
    let budget = available_memory.saturating_sub(1024 * 1024 * 1024) / 10 * 8;
    let pool: Vec<String> = if entry.quants.is_empty() {
        DEFAULT_QUANTS.iter().map(|q| (*q).to_owned()).collect()
    } else {
        entry.quants.clone()
    };
    let variants: Vec<(String, u64)> = pool
        .into_iter()
        .map(|q| {
            let est = estimate_quant_bytes(params_b, &q);
            (q, est)
        })
        .collect();
    choose_from_variants(&variants, budget).unwrap_or_else(|| entry.default_quant.clone())
}

/// Returns `Some(normalized hf:// reference)` when `input` is a catalog alias.
pub fn expand_reference(input: &str, memory: Option<u64>) -> Result<Option<String>, String> {
    let input = input.trim();
    if input.is_empty() {
        return Ok(None);
    }
    let Some(entry) = find(input) else {
        return Ok(None);
    };
    if entry.repo.is_empty() {
        return Err(format!("catalog entry {} has no repository", entry.name));
    }
    let quant = match memory {
        Some(bytes) if bytes > 0 => choose_quant(&entry.repo, bytes)
            .unwrap_or_else(|_| recommend_quant(&entry, bytes)),
        _ => entry.default_quant.clone(),
    };
    if !quant.is_empty() && memory.is_some() {
        eprintln!(
            "[fllm-rs] selected {} ({} of available memory)",
            quant,
            human_bytes(memory.unwrap_or(0))
        );
    }
    Ok(Some(format!("hf://{}:{}", entry.repo, quant)))
}

fn human_bytes(bytes: u64) -> String {
    if bytes >= 1024_u64.pow(4) {
        format!("{:.1} TiB", bytes as f64 / 1024_u64.pow(4) as f64)
    } else if bytes >= 1024_u64.pow(3) {
        format!("{:.1} GiB", bytes as f64 / 1024_u64.pow(3) as f64)
    } else if bytes >= 1024_u64.pow(2) {
        format!("{:.1} MiB", bytes as f64 / 1024_u64.pow(2) as f64)
    } else {
        format!("{bytes} B")
    }
}

const QUANT_PRIORITY: &[&str] = &[
    "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1", "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
    "Q3_K_M", "Q3_K_S", "Q3_K_L", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_XS", "IQ2_M",
    "IQ2_S", "IQ2_XS", "BF16", "F16", "F32",
];

pub fn choose_quant(repo: &str, available_memory: u64) -> Result<String, String> {
    let files = hub::repository_files(repo)?;
    let budget = available_memory.saturating_sub(1024 * 1024 * 1024) / 10 * 8; // 80% after 1 GiB reserve
    let mut candidates: Vec<(String, u64)> = Vec::new();
    for (name, size) in files {
        let upper = name.to_ascii_uppercase();
        if !upper.ends_with(".GGUF") {
            continue;
        }
        let Some(quant) = quant_from_filename(&upper) else {
            continue;
        };
        candidates.push((quant, size));
    }
    if candidates.is_empty() {
        return Err(format!("no GGUF variants found for {repo}"));
    }
    // Deduplicate by quant, keeping the smallest file for that quant.
    let mut by_quant: BTreeMap<String, u64> = BTreeMap::new();
    for (quant, size) in candidates {
        by_quant
            .entry(quant)
            .and_modify(|old| {
                if size < *old {
                    *old = size;
                }
            })
            .or_insert(size);
    }
    let fitting: Vec<(&String, &u64)> = by_quant
        .iter()
        .filter(|(_, size)| **size <= budget || budget == 0)
        .collect();
    let chosen = if !fitting.is_empty() {
        fitting.into_iter().max_by_key(|(quant, _)| priority(quant)).map(|(q, _)| q.clone())
    } else {
        by_quant
            .iter()
            .min_by_key(|(_, size)| **size)
            .map(|(q, _)| q.clone())
    };
    chosen.ok_or_else(|| format!("could not choose a quantization for {repo}"))
}

pub fn quant_from_filename(name: &str) -> Option<String> {
    let upper = name.to_ascii_uppercase();
    QUANT_PRIORITY
        .iter()
        .filter(|q| upper.contains(**q))
        .max_by_key(|q| q.len())
        .map(|q| (*q).to_owned())
}

fn priority(quant: &str) -> usize {
    QUANT_PRIORITY
        .iter()
        .position(|candidate| *candidate == quant)
        .unwrap_or(usize::MAX)
}

/// Pure helper used by unit tests: pick the highest-priority quantum from
/// (quant, bytes) pairs that fits into `budget` bytes.
pub fn choose_from_variants(variants: &[(String, u64)], budget: u64) -> Option<String> {
    let mut fitting: Vec<&str> = variants
        .iter()
        .filter(|(_, size)| *size <= budget)
        .map(|(quant, _)| quant.as_str())
        .collect();
    if fitting.is_empty() {
        return variants
            .iter()
            .min_by_key(|(_, size)| *size)
            .map(|(quant, _)| quant.clone());
    }
    fitting.sort_by_key(|q| priority(q));
    fitting.first().map(|q| (*q).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_catalog_fixture() {
        let text = r#"
version = 1
[[models]]
name = "demo:1b"
repo = "owner/demo-GGUF"
default = "Q4_K_M"
quants = ["Q8_0", "Q4_K_M"]
"#;
        let mut v = 1;
        let entries = parse_str(text, &mut v);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "demo:1b");
        assert_eq!(entries[0].repo, "owner/demo-GGUF");
        assert_eq!(entries[0].default_quant, "Q4_K_M");
        assert_eq!(entries[0].quants, vec!["Q8_0", "Q4_K_M"]);
    }

    #[test]
    fn quant_priority_order() {
        assert!(priority("Q8_0") < priority("Q4_K_M"));
    }

    #[test]
    fn choose_from_variants_prefers_largest_fit() {
        let variants = vec![
            ("Q8_0".to_owned(), 8_000_000_000),
            ("Q4_K_M".to_owned(), 4_000_000_000),
        ];
        assert_eq!(
            choose_from_variants(&variants, 5_000_000_000),
            Some("Q4_K_M".to_owned())
        );
        assert_eq!(
            choose_from_variants(&variants, 9_000_000_000),
            Some("Q8_0".to_owned())
        );
    }

    #[test]
    fn quant_from_filename_detects() {
        assert_eq!(quant_from_filename("MODEL-Q4_K_M.GGUF"), Some("Q4_K_M".to_owned()));
        assert_eq!(quant_from_filename("MODEL-F16.GGUF"), Some("F16".to_owned()));
    }

    #[test]
    fn expands_catalog_alias_to_default() {
        let all = catalog();
        assert!(all.len() >= 60, "expected rich model catalog, got {}", all.len());
        for family in [
            "llama3:8b", "llama3.1:8b", "llama3.2:3b", "llama3.3:70b", "llama4:scout",
            "qwen2.5:7b", "qwen3:8b", "qwen3.5:9b", "gemma2:9b", "gemma3:4b",
            "mistral:7b", "mixtral:8x7b", "devstral:24b", "codestral:22b",
            "deepseek-r1:8b", "deepseek-v3:671b", "phi3:3.8b", "phi4:14b",
            "smollm2:1.7b", "granite3:8b", "command-r:35b", "internlm3:8b",
        ] {
            assert!(find(family).is_some(), "missing catalog family {family}");
        }
        assert_eq!(
            expand_reference("llama3.1:8b", None).unwrap(),
            Some("hf://unsloth/Llama-3.1-8B-Instruct-GGUF:Q4_K_M".to_owned())
        );
        assert_eq!(
            expand_reference("llama3.1", None).unwrap(),
            Some("hf://unsloth/Llama-3.1-8B-Instruct-GGUF:Q4_K_M".to_owned())
        );
    }

    #[test]
    fn vram_aware_quantization_recommendations() {
        let entry = find("llama3.1:8b").expect("llama3.1:8b");
        // 24 GiB VRAM easily fits Q8_0 for an 8B model
        assert_eq!(recommend_quant(&entry, 24 * (1 << 30)), "Q8_0");
        // 8 GiB VRAM (budget ~5.6 GiB) fits Q4_K_M / Q5_K_M for an 8B model
        let q8g = recommend_quant(&entry, 8 * (1 << 30));
        assert!(matches!(q8g.as_str(), "Q4_K_M" | "Q5_K_M"), "got {q8g}");
        // 4 GiB VRAM (budget ~2.4 GiB) picks Q2_K for an 8B model
        assert_eq!(recommend_quant(&entry, 4 * (1 << 30)), "Q2_K");
    }
}
