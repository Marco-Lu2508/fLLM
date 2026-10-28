//! Safe, mmap-backed Hugging Face safetensors to GGUF conversion.
mod bert;
mod classic;
mod convert;
mod dequant;
mod dequant_bits;
mod exl2;
mod exl3;
mod decoder_enc;
mod deltanet;
mod enc_heads;
mod enc_out;
mod gemma;
mod gemma3n;
pub mod gguf;
pub mod gguf_tools;
mod gte;
mod jina_xlmr;
mod gptoss;
mod mistral_native;
mod hybrid;
mod kimi_linear;
mod modernbert;
mod nomic;
mod nvfp4;
mod packed_quant;
mod plan;
mod quantize;
mod requant;
pub mod safetensors;
mod st;
mod symint;
mod tokenizer_import;
mod torch_zip;
mod zamba2;
mod vision_plan;

pub use convert::{
    check_config, convert_checkpoint, inspect_checkpoint, load_checkpoint_as_gguf, load_checkpoint_quantized, CheckpointInventory, ConversionReport,
    ConvertOptions, ConfigCheck, OutputDType, SpecHeadKind, TensorSummary,
};

pub use quantize::{quantize_row, QType};
pub use tokenizer_import::tokenizer_json_from_dir;

/// `config.json` of a checkpoint directory, or the one derived from Mistral's `params.json` for the native layout.
pub fn read_checkpoint_config(root: &std::path::Path) -> Result<serde_json::Value, String> {
    mistral_native::read_config(root)
}
pub use requant::{quantize_gguf, Preset, QuantizeReport};

use std::{fmt, io};

#[derive(Debug)]
pub enum CheckpointError {
    Io(io::Error),
    Invalid(String),
    /// A direct load would have to build more tensor data in memory than the caller allows.
    OverBudget { needed: u64, limit: u64 },
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Invalid(e) => write!(f, "{e}"),
            Self::OverBudget { needed, limit } => write!(f, "loading directly needs {needed} bytes of memory for converted tensors, more than the {limit} allowed"),
        }
    }
}
impl std::error::Error for CheckpointError {}
impl From<io::Error> for CheckpointError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<String> for CheckpointError {
    fn from(e: String) -> Self {
        Self::Invalid(e)
    }
}
impl From<&str> for CheckpointError {
    fn from(e: &str) -> Self {
        Self::Invalid(e.to_owned())
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod direct_tests;
#[cfg(test)]
mod direct_format_tests;
