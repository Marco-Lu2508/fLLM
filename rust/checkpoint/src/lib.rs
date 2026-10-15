//! Safe, mmap-backed Hugging Face safetensors to GGUF conversion.
mod bert;
mod classic;
mod convert;
mod exl2;
mod exl3;
mod deltanet;
mod gguf;
pub mod gguf_tools;
mod gptoss;
mod modernbert;
mod nomic;
mod nvfp4;
mod packed_quant;
mod plan;
mod quantize;
mod requant;
mod safetensors;
mod symint;
mod torch_zip;

pub use convert::{
    convert_checkpoint, inspect_checkpoint, CheckpointInventory, ConversionReport, ConvertOptions,
    OutputDType, TensorSummary,
};

pub use quantize::{quantize_row, QType};
pub use requant::{quantize_gguf, Preset, QuantizeReport};

use std::{fmt, io};

#[derive(Debug)]
pub enum CheckpointError {
    Io(io::Error),
    Invalid(String),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Invalid(e) => write!(f, "{e}"),
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
