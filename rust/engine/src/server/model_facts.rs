//! Facts about the loaded GGUF that model listings report (Ollama `/api/tags`, `/api/ps`, `/api/show`,
//! `/v1/models`): file size, a digest, parameter count and quantization.
use fllm_gguf_probe::MappedGguf;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

#[derive(Clone, Debug, Default)]
pub struct ModelFacts {
    pub size_bytes: u64,
    /// SHA-256 over the file length, its first 16 MiB (header, metadata, tensor table) and last MiB. A full hash of a
    /// multi-gigabyte file would delay the start; the fingerprint still changes whenever the file is replaced.
    pub digest: String,
    pub parameters: u64,
    pub quantization: String,
    /// Context length the model was trained for (`<arch>.context_length`), 0 when the file does not say.
    pub trained_context: usize,
}

/// llama.cpp `LLAMA_FTYPE` names (`general.file_type`).
fn file_type_name(id: u64) -> Option<&'static str> {
    Some(match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        _ => return None,
    })
}

/// ggml tensor type ids that matter for naming the dominant storage type.
fn tensor_type_name(id: u32) -> String {
    match id {
        0 => "F32".into(),
        1 => "F16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        6 => "Q5_0".into(),
        7 => "Q5_1".into(),
        8 => "Q8_0".into(),
        10 => "Q2_K".into(),
        11 => "Q3_K".into(),
        12 => "Q4_K".into(),
        13 => "Q5_K".into(),
        14 => "Q6_K".into(),
        30 => "BF16".into(),
        other => format!("type{other}"),
    }
}

/// Parameter count as Ollama prints it: `7.6B`, `494M`.
pub fn parameter_label(parameters: u64) -> String {
    if parameters == 0 {
        String::new()
    } else if parameters >= 1_000_000_000 {
        format!("{:.1}B", parameters as f64 / 1e9)
    } else if parameters >= 1_000_000 {
        format!("{:.0}M", parameters as f64 / 1e6)
    } else {
        format!("{:.0}K", parameters as f64 / 1e3)
    }
}

fn fingerprint(path: &Path, len: u64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(len.to_le_bytes());
    let mut read = || -> std::io::Result<()> {
        let mut file = std::fs::File::open(path)?;
        let mut head = vec![0u8; len.min(16 << 20) as usize];
        file.read_exact(&mut head)?;
        hasher.update(&head);
        if len > (16 << 20) {
            let tail_len = (1u64 << 20).min(len - (16 << 20));
            file.seek(SeekFrom::Start(len - tail_len))?;
            let mut tail = vec![0u8; tail_len as usize];
            file.read_exact(&mut tail)?;
            hasher.update(&tail);
        }
        Ok(())
    };
    if read().is_err() {
        return String::new();
    }
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

impl ModelFacts {
    pub fn from_gguf(path: &Path, mapping: &MappedGguf) -> Self {
        let index = mapping.index();
        let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(index.file_len);
        let parameters = index
            .tensors
            .iter()
            .map(|t| t.dimensions.iter().product::<u64>())
            .sum();
        let metadata = &index.metadata;
        let quantization = metadata
            .get("general.file_type")
            .and_then(Value::as_u64)
            .and_then(file_type_name)
            .map(str::to_owned)
            .unwrap_or_else(|| {
                // no file type: the storage type that holds most of the weights
                let mut by_type: BTreeMap<u32, u64> = BTreeMap::new();
                for tensor in index.tensors.iter().filter(|t| t.dimensions.len() >= 2) {
                    *by_type.entry(tensor.type_id).or_default() += tensor.dimensions.iter().product::<u64>();
                }
                by_type
                    .into_iter()
                    .max_by_key(|(_, weights)| *weights)
                    .map(|(id, _)| tensor_type_name(id))
                    .unwrap_or_default()
            });
        let trained_context = metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .and_then(|arch| metadata.get(&format!("{arch}.context_length")))
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(0);
        Self {
            size_bytes,
            digest: fingerprint(path, size_bytes),
            parameters,
            quantization,
            trained_context,
        }
    }

    pub fn parameter_size(&self) -> String {
        parameter_label(self.parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_follow_ollama() {
        assert_eq!(parameter_label(7_615_616_512), "7.6B");
        assert_eq!(parameter_label(494_032_768), "494M");
        assert_eq!(parameter_label(0), "");
    }

    #[test]
    fn file_types_are_named() {
        assert_eq!(file_type_name(15), Some("Q4_K_M"));
        assert_eq!(file_type_name(7), Some("Q8_0"));
        assert_eq!(file_type_name(999), None);
    }

    #[test]
    fn fingerprint_changes_with_the_file() {
        let path = std::env::temp_dir().join(format!("fllm-facts-{}.bin", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        let a = fingerprint(&path, 3);
        std::fs::write(&path, b"abd").unwrap();
        assert_ne!(a, fingerprint(&path, 3));
        assert_eq!(a.len(), 64);
        let _ = std::fs::remove_file(path);
    }
}
