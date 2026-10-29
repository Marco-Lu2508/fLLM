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
    /// MLX affine group quantization (the `MLXA` types)
    Mlx,
    /// Block-wise FP8 (the `FP8B` types, every tile shape)
    Fp8b,
    /// bitsandbytes NF4, FP4 and LLM.int8() (the `NF4B*`, `FP4B*`, `INT8R` types)
    Bnb,
}

impl Family {
    /// Suffix of the kernel file names: `q_mv_<suffix>.metal`, `q_mm_<suffix>.metal`.
    pub fn file_suffix(self) -> &'static str {
        match self {
            Family::Legacy => "legacy",
            Family::KQuant => "kquant",
            Family::Fp => "fp",
            Family::Mlx => "mlx",
            Family::Fp8b => "fp8b",
            Family::Bnb => "bnb",
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
    /// MLXA (MLX affine): blocks of `[scale: 2][bias: 2][group * bits / 8 code bytes]` into a plane of codes and a plane of words
    /// `scale | bias << 16` (the 16-bit floats as they are; bfloat16 scales are flagged by bit 63 of the plane-1 offset in `q_params`).
    Mlx,
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
    /// FP8B: the E4M3 codes are followed by one f32 scale per tile and the 16-byte trailer (`fllm_gguf_probe::Fp8bLayout`; the tile shape is part of
    /// the type id). The device image is the file; `q_params::plane1` is the offset of the scale table and `q_params::pad` the tile shape.
    pub tiled: bool,
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
        tiled: false,
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
        tiled: false,
    }
}

/// GGUF type id of an MLX affine type (`fllm_gguf_probe::MlxAffine::type_id`; a unit test checks that they agree).
const fn mlx_id(bits: u32, group: u32, bf16: bool) -> u32 {
    0x8001_0000 | ((bf16 as u32) << 12) | ((group / 32) << 4) | bits
}

/// An MLX affine format; `tag` names its kernels (`mlx<bits>g<group>`), shared by the f16 and the bf16 variant.
const fn mlx(bits: u32, group: u32, bf16: bool, tag: &'static str) -> FormatInfo {
    FormatInfo {
        quant: QuantType::Other(mlx_id(bits, group, bf16)),
        tag,
        family: Family::Mlx,
        qk: group as usize,
        block_bytes: (group * bits / 8) as usize,
        src_qk: group as usize,
        src_block_bytes: (group * bits / 8 + 4) as usize,
        repack: Repack::Mlx,
        permuted: false,
        row_scales: false,
        tiled: false,
    }
}

/// A block-wise FP8 format: the kernels see 8-code chunks (`qk` 8) and look the tile scale up from the row and column of the chunk.
const fn fp8b(quant: QuantType) -> FormatInfo {
    FormatInfo {
        tiled: true,
        ..raw(quant, "fp8b", Family::Fp8b, 8, 8)
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
        tiled: false,
    },
    raw(QuantType::NvFp4, "nvfp4", Family::Fp, 64, 36),
    q4z(QuantType::Q4Z32, "q4z32", 32, false),
    q4z(QuantType::Q4Z64, "q4z64", 64, false),
    q4z(QuantType::Q4Z128, "q4z128", 128, false),
    q4z(QuantType::Q4ZP32, "q4z32", 32, true),
    q4z(QuantType::Q4ZP64, "q4z64", 64, true),
    q4z(QuantType::Q4ZP128, "q4z128", 128, true),
    mlx(2, 32, false, "mlx2g32"),
    mlx(2, 32, true, "mlx2g32"),
    mlx(2, 64, false, "mlx2g64"),
    mlx(2, 64, true, "mlx2g64"),
    mlx(2, 128, false, "mlx2g128"),
    mlx(2, 128, true, "mlx2g128"),
    mlx(3, 32, false, "mlx3g32"),
    mlx(3, 32, true, "mlx3g32"),
    mlx(3, 64, false, "mlx3g64"),
    mlx(3, 64, true, "mlx3g64"),
    mlx(3, 128, false, "mlx3g128"),
    mlx(3, 128, true, "mlx3g128"),
    mlx(4, 32, false, "mlx4g32"),
    mlx(4, 32, true, "mlx4g32"),
    mlx(4, 64, false, "mlx4g64"),
    mlx(4, 64, true, "mlx4g64"),
    mlx(4, 128, false, "mlx4g128"),
    mlx(4, 128, true, "mlx4g128"),
    mlx(5, 32, false, "mlx5g32"),
    mlx(5, 32, true, "mlx5g32"),
    mlx(5, 64, false, "mlx5g64"),
    mlx(5, 64, true, "mlx5g64"),
    mlx(5, 128, false, "mlx5g128"),
    mlx(5, 128, true, "mlx5g128"),
    mlx(6, 32, false, "mlx6g32"),
    mlx(6, 32, true, "mlx6g32"),
    mlx(6, 64, false, "mlx6g64"),
    mlx(6, 64, true, "mlx6g64"),
    mlx(6, 128, false, "mlx6g128"),
    mlx(6, 128, true, "mlx6g128"),
    mlx(8, 32, false, "mlx8g32"),
    mlx(8, 32, true, "mlx8g32"),
    mlx(8, 64, false, "mlx8g64"),
    mlx(8, 64, true, "mlx8g64"),
    mlx(8, 128, false, "mlx8g128"),
    mlx(8, 128, true, "mlx8g128"),
    dense(QuantType::F32, "f32", 4),
    dense(QuantType::F16, "f16", 2),
    dense(QuantType::BF16, "bf16", 2),
    dense(QuantType::Fp8E4m3, "fp8e4m3", 1),
    dense(QuantType::Fp8E5m2, "fp8e5m2", 1),
    FormatInfo {
        row_scales: true,
        ..dense(QuantType::Fp8Marlin, "fp8m", 1)
    },
    fp8b(QuantType::Fp8Block),
    raw(QuantType::Nf4B64, "nf4b64", Family::Bnb, 64, 36),
    raw(QuantType::Nf4B128, "nf4b128", Family::Bnb, 128, 68),
    raw(QuantType::Fp4B64, "fp4b64", Family::Bnb, 64, 36),
    raw(QuantType::Fp4B128, "fp4b128", Family::Bnb, 128, 68),
    raw(QuantType::Int8Row, "int8r", Family::Bnb, 64, 68),
];

/// Geometry of `quant`, or `None` if the Metal kernels do not implement it.
pub fn format_info(quant: QuantType) -> Option<FormatInfo> {
    // every tile shape of the block-wise FP8 format is one kernel (the shape travels in `q_params::pad`): the shapes other than 128 x 128 are
    // `Other` types and are not in the table
    if let QuantType::Other(id) = quant {
        if fllm_gguf_probe::fp8b_type(id) {
            return Some(fp8b(quant));
        }
    }
    FORMATS.iter().find(|f| f.quant == quant).copied()
}

/// The tile shapes (rows, columns) besides 128 x 128 whose formats the hardware check and the emulation dump run: partial tiles in both
/// directions, tiles narrower than a chunk of 8 codes (one scale per code), a tile wider than the matrix, one tile per row (the shape of
/// `FP8M` weights widened to FP8B).
pub const FP8B_EXTRA_TILES: [(usize, usize); 6] = [(16, 32), (1, 8), (2, 4), (64, 1), (1, 512), (32, 16)];

/// `supported_formats` plus the extra FP8B tile shapes ([`FP8B_EXTRA_TILES`]): what the hardware check runs.
pub fn check_formats() -> Vec<QuantType> {
    let mut all = supported_formats();
    all.extend(FP8B_EXTRA_TILES.iter().filter_map(|&(r, c)| fllm_gguf_probe::fp8b_type_id(r, c)).map(QuantType::from_gguf_type_id));
    all
}

/// All formats `format_info` knows, in kernel-file order.
pub fn supported_formats() -> Vec<QuantType> {
    FORMATS.iter().map(|f| f.quant).collect()
}
