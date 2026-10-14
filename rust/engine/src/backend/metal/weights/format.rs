//! Table of the weight formats the Metal quantized-linear kernels support.

use crate::backend::QuantType;

/// Kernel file family of a format (one Metal library per family and operation, see `metal/q_gen.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL
    Legacy,
    /// Q2_K .. Q6_K, IQ4_XS
    KQuant,
    /// MXFP4, NVFP4, Q4Z*, floats, FP8
    Fp,
}

impl Family {
    /// Suffix of the kernel file names: `q_mv_<suffix>.metal`, `q_mm_<suffix>.metal`.
    pub fn file_suffix(self) -> &'static str {
        match self {
            Family::Legacy => "legacy",
            Family::KQuant => "kquant",
            Family::Fp => "fp",
        }
    }
}

/// Which re-sorting [`super::pack_weight`] applies to the bytes of the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repack {
    /// The device image is the file layout (the kernels read the GGUF blocks).
    None,
    /// MXFP4: 17-byte blocks (E8M0 scale byte, 16 code bytes) into a plane of codes and a plane of scale bytes.
    Mxfp4,
    /// Q4Z / Q4ZP: blocks of `group / 2` code bytes, an f16 scale and a zero-point byte into a plane of codes and a plane of
    /// words `scale | zero << 16`.
    Q4z,
}

/// Geometry of one format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatInfo {
    pub quant: QuantType,
    /// Kernel tag: kernels are called `fllm_q_mv_<tag>_<variant>` and `fllm_q_mm_<tag>_<variant>`.
    pub tag: &'static str,
    pub family: Family,
    /// Values per block as the kernel sees it (8 for the dense and FP8 formats, which have no blocks).
    pub qk: usize,
    /// Bytes of one such block in the device image (plane 0 only for the repacked formats).
    pub block_bytes: usize,
    /// Values per block in the file, bytes per block in the file (for the dense formats the same 8-value pseudo block).
    pub src_qk: usize,
    pub src_block_bytes: usize,
    pub repack: Repack,
    /// Act-order GPTQ: the file holds a permutation table (one u32 per column) after the blocks.
    pub permuted: bool,
    /// FP8M: one f16 scale per row follows the E4M3 bytes of the file (and of the device image).
    pub row_scales: bool,
}

const fn raw(
    quant: QuantType,
    tag: &'static str,
    family: Family,
    qk: usize,
    bytes: usize,
) -> FormatInfo {
    FormatInfo {
        quant,
        tag,
        family,
        qk,
        block_bytes: bytes,
        src_qk: qk,
        src_block_bytes: bytes,
        repack: Repack::None,
        permuted: false,
        row_scales: false,
    }
}

const fn q4z(quant: QuantType, tag: &'static str, group: usize, permuted: bool) -> FormatInfo {
    FormatInfo {
        quant,
        tag,
        family: Family::Fp,
        qk: group,
        block_bytes: group / 2,
        src_qk: group,
        src_block_bytes: group / 2 + 3,
        repack: Repack::Q4z,
        permuted,
        row_scales: false,
    }
}

const fn dense(quant: QuantType, tag: &'static str, value_bytes: usize) -> FormatInfo {
    raw(quant, tag, Family::Fp, 8, 8 * value_bytes)
}

static FORMATS: &[FormatInfo] = &[
    raw(QuantType::Q4_0, "q4_0", Family::Legacy, 32, 18),
    raw(QuantType::Q4_1, "q4_1", Family::Legacy, 32, 20),
    raw(QuantType::Q5_0, "q5_0", Family::Legacy, 32, 22),
    raw(QuantType::Q5_1, "q5_1", Family::Legacy, 32, 24),
    raw(QuantType::Q8_0, "q8_0", Family::Legacy, 32, 34),
    raw(QuantType::IQ4_NL, "iq4_nl", Family::Legacy, 32, 18),
    raw(QuantType::Q2_K, "q2_k", Family::KQuant, 256, 84),
    raw(QuantType::Q3_K, "q3_k", Family::KQuant, 256, 110),
    raw(QuantType::Q4_K, "q4_k", Family::KQuant, 256, 144),
    raw(QuantType::Q5_K, "q5_k", Family::KQuant, 256, 176),
    raw(QuantType::Q6_K, "q6_k", Family::KQuant, 256, 210),
    raw(QuantType::IQ4_XS, "iq4_xs", Family::KQuant, 256, 136),
    FormatInfo {
        quant: QuantType::Mxfp4,
        tag: "mxfp4",
        family: Family::Fp,
        qk: 32,
        block_bytes: 16,
        src_qk: 32,
        src_block_bytes: 17,
        repack: Repack::Mxfp4,
        permuted: false,
        row_scales: false,
    },
    raw(QuantType::NvFp4, "nvfp4", Family::Fp, 64, 36),
    q4z(QuantType::Q4Z32, "q4z32", 32, false),
    q4z(QuantType::Q4Z64, "q4z64", 64, false),
    q4z(QuantType::Q4Z128, "q4z128", 128, false),
    q4z(QuantType::Q4ZP32, "q4z32", 32, true),
    q4z(QuantType::Q4ZP64, "q4z64", 64, true),
    q4z(QuantType::Q4ZP128, "q4z128", 128, true),
    dense(QuantType::F32, "f32", 4),
    dense(QuantType::F16, "f16", 2),
    dense(QuantType::BF16, "bf16", 2),
    dense(QuantType::Fp8E4m3, "fp8e4m3", 1),
    dense(QuantType::Fp8E5m2, "fp8e5m2", 1),
    FormatInfo {
        row_scales: true,
        ..dense(QuantType::Fp8Marlin, "fp8m", 1)
    },
];

/// Geometry of `quant`, or `None` if the Metal kernels do not implement it.
pub fn format_info(quant: QuantType) -> Option<FormatInfo> {
    FORMATS.iter().find(|f| f.quant == quant).copied()
}

/// All formats `format_info` knows, in kernel-file order.
pub fn supported_formats() -> Vec<QuantType> {
    FORMATS.iter().map(|f| f.quant).collect()
}
