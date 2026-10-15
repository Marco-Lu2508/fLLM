//! Bounded-memory readers for common Hugging Face packed linear weights.
//!
//! This module deliberately implements narrow checkpoint layouts, not a general
//! GPTQ/AWQ importer. It supports AutoGPTQ's legacy 4-bit packing and AutoAWQ's
//! 4-bit GEMM layout with zero points, plus unscaled SafeTensors FP8 element
//! decoding for F8_E4M3 (float8_e4m3fn) and F8_E5M2 (float8_e5m2).

use crate::safetensors::Payload;
use half::f16;

const AWQ_SLOT_FOR_LOGICAL: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleType {
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp8Format {
    /// SafeTensors F8_E4M3 / PyTorch float8_e4m3fn, not E4M3FNUZ.
    E4M3Fn,
    /// SafeTensors F8_E5M2 / PyTorch float8_e5m2, not E5M2FNUZ.
    E5M2,
}

#[derive(Clone, Debug)]
pub struct PackedInt4 {
    qweight: Payload,
    qzeros: Payload,
    scales: Payload,
    g_idx: Option<Payload>,
    in_features: usize,
    out_features: usize,
    group_size: usize,
    group_count: usize,
    elements: usize,
    scale_type: ScaleType,
    layout: Int4Layout,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Int4Layout {
    /// AutoGPTQ legacy int4: qweight [in/8, out], qzeros [groups, out/8].
    /// qzeros store zero_point - 1; dequantization adds one modulo 16.
    GptqV1,
    /// AutoAWQ GEMM int4: qweight/qzeros [in-or-groups, out/8] with the
    /// interleaved nibble order [0,2,4,6,1,3,5,7]. Requires zero points.
    AwqGemm,
}

#[derive(Clone, Debug)]
pub enum PackedData {
    Exl2(crate::exl2::Exl2),
    Exl3(crate::exl3::Exl3),
    Concat {
        parts: Vec<PackedData>,
        elements: usize,
    },
    Int4(PackedInt4),
    Fp8 {
        bytes: Payload,
        format: Fp8Format,
        elements: usize,
    },
    Fp8Marlin {
        weight: Payload,
        scales: Vec<u16>,
        rows: usize,
        cols: usize,
    },
    /// FP8 weights with one scale per tile (block-wise FP8); written as f16.
    Fp8Block(crate::symint::Fp8Block),
    /// NVIDIA ModelOpt NVFP4 (GGUF type 40 when the rows fit 64-value blocks).
    Nvfp4(crate::nvfp4::Nvfp4),
    /// Symmetric integer weights of `compressed-tensors` checkpoints.
    SymInt(crate::symint::SymInt),
}

impl PackedInt4 {
    /// Construct a supported legacy AutoGPTQ 4-bit tensor.
    ///
    /// `qweight` is row-major int32 `[in_features/8, out_features]`; `qzeros`
    /// is row-major int32 `[groups, out_features/8]` with stored zero points
    /// biased by -1; `scales` is row-major `[groups, out_features]`. If `g_idx`
    /// is absent, input rows map to `input_index / group_size`.
    #[allow(clippy::too_many_arguments)]
    pub fn gptq_v1(
        qweight: Payload,
        qzeros: Payload,
        scales: Payload,
        g_idx: Option<Payload>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        scale_type: ScaleType,
    ) -> Result<Self, String> {
        Self::new(
            qweight,
            qzeros,
            scales,
            g_idx,
            in_features,
            out_features,
            group_size,
            scale_type,
            Int4Layout::GptqV1,
        )
    }

    /// Construct an AutoAWQ 4-bit GEMM tensor with zero points enabled.
    ///
    /// `qweight` is row-major int32 `[in_features, out_features/8]`; `qzeros`
    /// is row-major int32 `[groups, out_features/8]`; scales are `[groups,
    /// out_features]`. AWQ stores direct zero points and uses the interleaved
    /// nibble ordering documented by AutoAWQ's WQLinear_GEMM packer.
    pub fn awq_gemm(
        qweight: Payload,
        qzeros: Payload,
        scales: Payload,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        scale_type: ScaleType,
    ) -> Result<Self, String> {
        Self::new(
            qweight,
            qzeros,
            scales,
            None,
            in_features,
            out_features,
            group_size,
            scale_type,
            Int4Layout::AwqGemm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        qweight: Payload,
        qzeros: Payload,
        scales: Payload,
        g_idx: Option<Payload>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        scale_type: ScaleType,
        layout: Int4Layout,
    ) -> Result<Self, String> {
        if in_features == 0 || out_features == 0 || group_size == 0 {
            return Err("packed int4 dimensions and group_size must be nonzero".into());
        }
        if !in_features.is_multiple_of(8) || !out_features.is_multiple_of(8) {
            return Err(
                "packed int4 helper requires in_features and out_features divisible by 8".into(),
            );
        }
        let elements = in_features
            .checked_mul(out_features)
            .ok_or("packed int4 element count overflow")?;
        if layout == Int4Layout::AwqGemm && !in_features.is_multiple_of(group_size) {
            return Err("AutoAWQ GEMM requires in_features divisible by group_size".into());
        }
        let group_count = match layout {
            Int4Layout::GptqV1 => in_features.div_ceil(group_size),
            Int4Layout::AwqGemm => in_features / group_size,
        };
        let qweight_words = match layout {
            Int4Layout::GptqV1 => (in_features / 8).checked_mul(out_features),
            Int4Layout::AwqGemm => in_features.checked_mul(out_features / 8),
        }
        .ok_or("packed qweight size overflow")?;
        let qzero_words = group_count
            .checked_mul(out_features / 8)
            .ok_or("packed qzeros size overflow")?;
        let scale_width = match scale_type {
            ScaleType::F16 => 2,
            ScaleType::F32 => 4,
        };
        let scale_values = group_count
            .checked_mul(out_features)
            .ok_or("packed scales size overflow")?;
        require_len(
            "qweight",
            qweight.as_slice(),
            qweight_words
                .checked_mul(4)
                .ok_or("packed qweight byte size overflow")?,
        )?;
        require_len(
            "qzeros",
            qzeros.as_slice(),
            qzero_words
                .checked_mul(4)
                .ok_or("packed qzeros byte size overflow")?,
        )?;
        require_len(
            "scales",
            scales.as_slice(),
            scale_values
                .checked_mul(scale_width)
                .ok_or("packed scales byte size overflow")?,
        )?;
        if let Some(indices) = &g_idx {
            if layout != Int4Layout::GptqV1 {
                return Err("g_idx is supported only for GPTQ v1".into());
            }
            require_len(
                "g_idx",
                indices.as_slice(),
                in_features
                    .checked_mul(4)
                    .ok_or("g_idx byte size overflow")?,
            )?;
        }
        Ok(Self {
            qweight,
            qzeros,
            scales,
            g_idx,
            in_features,
            out_features,
            group_size,
            group_count,
            elements,
            scale_type,
            layout,
        })
    }

    pub fn element_count(&self) -> usize {
        self.elements
    }

    /// Group size when the tensor can be stored as native `Q4Z` rows (see `encode_q4z_row`), else `None`.
    fn q4z_group(&self) -> Option<usize> {
        let g = self.group_size;
        // a g_idx that maps input `k` to group `k / g` (written by GPTQ without act-order) changes nothing
        let sequential = match &self.g_idx {
            None => true,
            Some(indices) => {
                let bytes = indices.as_slice();
                (0..self.in_features).all(|k| read_u32(bytes, k * 4).is_ok_and(|v| v as usize == k / g))
            }
        };
        (matches!(g, 32 | 64 | 128) && sequential && self.in_features.is_multiple_of(g) && self.scale_type == ScaleType::F16).then_some(g)
    }

    /// Column permutation table for `Q4ZP` tensors (GPTQ with act-order `desc_act = true`), or `None` if not applicable.
    pub fn q4zp_perm(&self) -> Option<Vec<u32>> {
        let g = self.group_size;
        if !matches!(g, 32 | 64 | 128) || !self.in_features.is_multiple_of(g) || self.scale_type != ScaleType::F16 {
            return None;
        }
        let indices = self.g_idx.as_ref()?;
        let bytes = indices.as_slice();
        let g_idx: Vec<u32> = (0..self.in_features)
            .map(|k| read_u32(bytes, k * 4).ok())
            .collect::<Option<Vec<_>>>()?;
        let mut perm: Vec<u32> = (0..self.in_features as u32).collect();
        perm.sort_by_key(|&k| g_idx[k as usize]);
        for (i, &k) in perm.iter().enumerate() {
            if g_idx[k as usize] as usize != i / g {
                return None;
            }
        }
        Some(perm)
    }

    /// Group size when the tensor can be stored as native `Q4ZP` rows with permutation table, else `None`.
    pub fn q4zp_group(&self) -> Option<usize> {
        if self.q4z_group().is_some() {
            return None;
        }
        if self.q4zp_perm().is_some() {
            Some(self.group_size)
        } else {
            None
        }
    }

    /// Append the `Q4ZP` blocks of output row `out_idx`: same block format as `Q4Z`, but taking input channels
    /// in the order dictated by `perm`.
    pub fn encode_q4zp_row(&self, out_idx: usize, perm: &[u32], dst: &mut Vec<u8>) -> Result<(), String> {
        self.encode_q4z_row_with_perm(out_idx, Some(perm), dst)
    }

    /// Append the `Q4Z` blocks of output row `out_idx`: for every group of `group_size` inputs, `group_size / 2` bytes of 4-bit
    /// codes (input `2j` in the low nibble of byte `j`), the fp16 scale and the zero point as one byte. No value changes: the
    /// dequantized weight stays `(code - zero) * scale`.
    fn encode_q4z_row(&self, out_idx: usize, dst: &mut Vec<u8>) -> Result<(), String> {
        self.encode_q4z_row_with_perm(out_idx, None, dst)
    }

    fn encode_q4z_row_with_perm(&self, out_idx: usize, perm: Option<&[u32]>, dst: &mut Vec<u8>) -> Result<(), String> {
        let g = self.group_size;
        if out_idx >= self.out_features {
            return Err(format!("output row {out_idx} out of range"));
        }
        if let Some(p) = perm {
            if p.len() != self.in_features {
                return Err("perm length does not match in_features".into());
            }
        }
        let qw = self.qweight.as_slice();
        let qz = self.qzeros.as_slice();
        let sc = self.scales.as_slice();
        let slot = match self.layout {
            Int4Layout::GptqV1 => out_idx % 8,
            Int4Layout::AwqGemm => AWQ_SLOT_FOR_LOGICAL[out_idx % 8],
        };
        let code = |k: usize| -> Result<u8, String> {
            let (word_index, nibble) = match self.layout {
                Int4Layout::GptqV1 => ((k / 8) * self.out_features + out_idx, k % 8),
                Int4Layout::AwqGemm => (k * (self.out_features / 8) + out_idx / 8, slot),
            };
            Ok(((read_u32(qw, word_index * 4)? >> (4 * nibble)) & 0x0f) as u8)
        };
        for group in 0..self.in_features / g {
            let zero_word = read_u32(qz, (group * (self.out_features / 8) + out_idx / 8) * 4)?;
            let stored = ((zero_word >> (4 * slot)) & 0x0f) as u8;
            let zero = match self.layout {
                Int4Layout::GptqV1 => (stored + 1) & 0x0f,
                Int4Layout::AwqGemm => stored,
            };
            for j in 0..g / 2 {
                let k0 = match perm {
                    Some(p) => p[group * g + 2 * j] as usize,
                    None => group * g + 2 * j,
                };
                let k1 = match perm {
                    Some(p) => p[group * g + 2 * j + 1] as usize,
                    None => group * g + 2 * j + 1,
                };
                dst.push(code(k0)? | (code(k1)? << 4));
            }
            let at = (group * self.out_features + out_idx) * 2;
            dst.extend_from_slice(sc.get(at..at + 2).ok_or("scale read past tensor end")?);
            dst.push(zero);
        }
        Ok(())
    }

    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let end = checked_range(start, dst.len(), self.element_count())?;
        let qw = self.qweight.as_slice();
        let qz = self.qzeros.as_slice();
        let scale_bytes = self.scales.as_slice();
        let scale_width = match self.scale_type {
            ScaleType::F16 => 2,
            ScaleType::F32 => 4,
        };
        let indices = self.g_idx.as_ref().map(Payload::as_slice);
        for (slot, flat) in (start..end).enumerate() {
            let out_idx = flat / self.in_features;
            let in_idx = flat % self.in_features;
            let group = match indices {
                Some(bytes) => read_u32(bytes, in_idx * 4)? as usize,
                None => in_idx / self.group_size,
            };
            if group >= self.group_count {
                return Err(format!(
                    "packed int4 g_idx[{in_idx}]={group} exceeds {} groups",
                    self.group_count
                ));
            }
            let (word_index, nibble) = match self.layout {
                Int4Layout::GptqV1 => ((in_idx / 8) * self.out_features + out_idx, in_idx % 8),
                Int4Layout::AwqGemm => (
                    in_idx * (self.out_features / 8) + out_idx / 8,
                    AWQ_SLOT_FOR_LOGICAL[out_idx % 8],
                ),
            };
            let q_word = read_u32(
                qw,
                word_index.checked_mul(4).ok_or("qweight offset overflow")?,
            )?;
            let q = ((q_word >> (4 * nibble)) & 0x0f) as i32;

            let zero_word_index = group * (self.out_features / 8) + out_idx / 8;
            let zero_word = read_u32(
                qz,
                zero_word_index
                    .checked_mul(4)
                    .ok_or("qzeros offset overflow")?,
            )?;
            let stored_zero = ((zero_word
                >> (4 * match self.layout {
                    Int4Layout::GptqV1 => out_idx % 8,
                    Int4Layout::AwqGemm => AWQ_SLOT_FOR_LOGICAL[out_idx % 8],
                }))
                & 0x0f) as i32;
            let zero = match self.layout {
                Int4Layout::GptqV1 => (stored_zero + 1) & 0x0f,
                Int4Layout::AwqGemm => stored_zero,
            };
            let scale_index = group
                .checked_mul(self.out_features)
                .and_then(|n| n.checked_add(out_idx))
                .ok_or("scale offset overflow")?;
            let scale_offset = scale_index
                .checked_mul(scale_width)
                .ok_or("scale byte offset overflow")?;
            let scale = match self.scale_type {
                ScaleType::F16 => {
                    f16::from_le_bytes([scale_bytes[scale_offset], scale_bytes[scale_offset + 1]])
                        .to_f32()
                }
                ScaleType::F32 => f32::from_le_bytes(
                    scale_bytes[scale_offset..scale_offset + 4]
                        .try_into()
                        .unwrap(),
                ),
            };
            dst[slot] = (q - zero) as f32 * scale;
        }
        Ok(())
    }
}

impl PackedData {
    #[allow(clippy::too_many_arguments)]
    pub fn gptq_v1(
        qweight: Payload,
        qzeros: Payload,
        scales: Payload,
        g_idx: Option<Payload>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        scale_type: ScaleType,
    ) -> Result<Self, String> {
        Ok(Self::Int4(PackedInt4::gptq_v1(
            qweight,
            qzeros,
            scales,
            g_idx,
            in_features,
            out_features,
            group_size,
            scale_type,
        )?))
    }

    pub fn awq_gemm(
        qweight: Payload,
        qzeros: Payload,
        scales: Payload,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        scale_type: ScaleType,
    ) -> Result<Self, String> {
        Ok(Self::Int4(PackedInt4::awq_gemm(
            qweight,
            qzeros,
            scales,
            in_features,
            out_features,
            group_size,
            scale_type,
        )?))
    }

    pub fn fp8(bytes: Payload, format: Fp8Format, elements: usize) -> Result<Self, String> {
        require_len("FP8 weight", bytes.as_slice(), elements)?;
        Ok(Self::Fp8 {
            bytes,
            format,
            elements,
        })
    }

    /// Group size of a 4-bit AWQ / GPTQ tensor that can be written as native `Q4Z` rows without changing any value.
    pub fn q4z_group(&self) -> Option<usize> {
        match self {
            Self::Int4(t) => t.q4z_group(),
            Self::SymInt(t) => t.q4z_group(),
            _ => None,
        }
    }

    /// Group size of a 4-bit GPTQ tensor with act-order that can be written as native `Q4ZP` rows.
    pub fn q4zp_group(&self) -> Option<usize> {
        match self {
            Self::Int4(t) => t.q4zp_group(),
            _ => None,
        }
    }

    /// Column permutation table for `Q4ZP` tensors.
    pub fn q4zp_perm(&self) -> Option<Vec<u32>> {
        match self {
            Self::Int4(t) => t.q4zp_perm(),
            _ => None,
        }
    }

    /// The NVFP4 tensor, if this is one.
    pub fn nvfp4(&self) -> Option<&crate::nvfp4::Nvfp4> {
        match self {
            Self::Nvfp4(t) => Some(t),
            _ => None,
        }
    }

    /// The symmetric integer tensor, if this is one.
    pub fn sym_int(&self) -> Option<&crate::symint::SymInt> {
        match self {
            Self::SymInt(t) => Some(t),
            _ => None,
        }
    }

    /// Whether this tensor is in the FP8 Marlin format (W8A16 with per-channel scales).
    pub fn is_fp8m(&self) -> bool {
        matches!(self, Self::Fp8Marlin { .. })
    }

    /// Payload and scales for an FP8 Marlin tensor.
    #[allow(dead_code)]
    pub fn fp8m_data(&self) -> Option<(&Payload, &[u16])> {
        match self {
            Self::Fp8Marlin { weight, scales, .. } => Some((weight, scales.as_slice())),
            _ => None,
        }
    }

    /// Payload, scales, rows, cols for an FP8 Marlin tensor.
    pub fn fp8m_info(&self) -> Option<(&Payload, &[u16], usize, usize)> {
        match self {
            Self::Fp8Marlin { weight, scales, rows, cols } => Some((weight, scales.as_slice(), *rows, *cols)),
            _ => None,
        }
    }

    /// Append the `Q4Z` blocks of output row `out_idx` (see `PackedInt4::encode_q4z_row`).
    pub fn encode_q4z_row(&self, out_idx: usize, dst: &mut Vec<u8>) -> Result<(), String> {
        match self {
            Self::Int4(t) => t.encode_q4z_row(out_idx, dst),
            Self::SymInt(t) => t.encode_q4z_row(out_idx, dst),
            _ => Err("only 4-bit AWQ / GPTQ / compressed-tensors tensors can be written as Q4Z".into()),
        }
    }

    /// Append the `Q4ZP` blocks of output row `out_idx` (see `PackedInt4::encode_q4zp_row`).
    pub fn encode_q4zp_row(&self, out_idx: usize, perm: &[u32], dst: &mut Vec<u8>) -> Result<(), String> {
        match self {
            Self::Int4(t) => t.encode_q4zp_row(out_idx, perm, dst),
            _ => Err("only 4-bit GPTQ tensors with act-order can be written as Q4ZP".into()),
        }
    }

    pub fn element_count(&self) -> usize {
        match self {
            Self::Exl2(t) => t.element_count(),
            Self::Exl3(t) => t.element_count(),
            Self::Concat { elements, .. } => *elements,
            Self::Int4(t) => t.element_count(),
            Self::Fp8 { elements, .. } => *elements,
            Self::Fp8Marlin { rows, cols, .. } => rows * cols,
            Self::Fp8Block(t) => t.element_count(),
            Self::Nvfp4(t) => t.element_count(),
            Self::SymInt(t) => t.element_count(),
        }
    }

    /// Decode a bounded flat output slice in row-major `[out_features, in_features]` order.
    pub fn decode_f32_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        match self {
            Self::Exl2(t) => t.decode_chunk(start, dst),
            Self::Exl3(t) => t.decode_chunk(start, dst),
            Self::Concat { parts, elements } => {
                checked_range(start, dst.len(), *elements)?;
                let mut offset = 0;
                for part in parts {
                    let lo = start.max(offset);
                    let hi = (start + dst.len()).min(offset + part.element_count());
                    if lo < hi {
                        part.decode_f32_chunk(lo - offset, &mut dst[lo - start..hi - start])?;
                    }
                    offset += part.element_count();
                }
                Ok(())
            }
            Self::Int4(t) => t.decode_chunk(start, dst),
            Self::Fp8Block(t) => t.decode_chunk(start, dst),
            Self::Nvfp4(t) => t.decode_chunk(start, dst),
            Self::SymInt(t) => t.decode_chunk(start, dst),
            Self::Fp8 {
                bytes,
                format,
                elements,
            } => decode_fp8_chunk(bytes.as_slice(), *format, start, dst, *elements),
            Self::Fp8Marlin {
                weight,
                scales,
                rows,
                cols,
            } => {
                checked_range(start, dst.len(), rows * cols)?;
                let bytes = weight.as_slice();
                for (i, out) in dst.iter_mut().enumerate() {
                    let idx = start + i;
                    let r = idx / cols;
                    let c = idx % cols;
                    let byte = bytes[r * cols + c];
                    let scale = half::f16::from_bits(scales[r]).to_f32();
                    *out = e4m3fn_to_f32(byte) * scale;
                }
                Ok(())
            }
        }
    }
}

/// Convert a bounded slice of raw SafeTensors FP8 bytes to f32.
/// NaN and infinity encodings are preserved as their IEEE f32 counterparts;
/// callers that require finite model weights should reject them after decoding.
pub fn decode_fp8_chunk(
    bytes: &[u8],
    format: Fp8Format,
    start: usize,
    dst: &mut [f32],
    elements: usize,
) -> Result<(), String> {
    require_len("FP8 weight", bytes, elements)?;
    let end = checked_range(start, dst.len(), elements)?;
    for (dst_slot, &bits) in dst.iter_mut().zip(&bytes[start..end]) {
        *dst_slot = match format {
            Fp8Format::E4M3Fn => e4m3fn_to_f32(bits),
            Fp8Format::E5M2 => e5m2_to_f32(bits),
        };
    }
    Ok(())
}

pub(crate) fn e4m3fn_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 3) & 0x0f;
    let mant = bits & 0x07;
    if exp == 0 {
        sign * (mant as f32) * 2f32.powi(-9)
    } else if exp == 0x0f && mant == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    }
}

fn e5m2_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 2) & 0x1f;
    let mant = bits & 0x03;
    if exp == 0 {
        sign * (mant as f32) * 2f32.powi(-16)
    } else if exp == 0x1f && mant == 0 {
        sign * f32::INFINITY
    } else if exp == 0x1f {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 4.0) * 2f32.powi(exp as i32 - 15)
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let end = offset.checked_add(4).ok_or("packed word offset overflow")?;
    let word = bytes
        .get(offset..end)
        .ok_or("packed word read past tensor end")?;
    Ok(u32::from_le_bytes(word.try_into().unwrap()))
}

fn checked_range(start: usize, len: usize, total: usize) -> Result<usize, String> {
    let end = start
        .checked_add(len)
        .ok_or("packed output range overflow")?;
    if end > total {
        return Err(format!(
            "packed output range {start}..{end} exceeds {total} elements"
        ));
    }
    Ok(end)
}

fn require_len(name: &str, bytes: &[u8], expected: usize) -> Result<(), String> {
    if bytes.len() != expected {
        return Err(format!(
            "{name} has {} bytes, expected {expected}",
            bytes.len()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{decode_fp8_chunk, Fp8Format, PackedData, PackedInt4, ScaleType};
    use crate::safetensors::Payload;
    use std::sync::Arc;

    fn owned(data: Vec<u8>) -> Payload {
        let len = data.len();
        Payload::Owned(Arc::new(data), 0..len)
    }

    #[test]
    fn fp8_known_encodings() {
        let e4 = [0x01, 0x38, 0xb8, 0x7e, 0x7f, 0xff];
        let mut got = [0.0; 6];
        decode_fp8_chunk(&e4, Fp8Format::E4M3Fn, 0, &mut got, 6).unwrap();
        assert_eq!(got[0], 2f32.powi(-9));
        assert_eq!(got[1], 1.0);
        assert_eq!(got[2], -1.0);
        assert_eq!(got[3], 448.0);
        assert!(got[4].is_nan() && got[5].is_nan());

        let e5 = [0x01, 0x3c, 0xbc, 0x7c, 0xfc, 0x7d];
        decode_fp8_chunk(&e5, Fp8Format::E5M2, 0, &mut got, 6).unwrap();
        assert_eq!(got[0], 2f32.powi(-16));
        assert_eq!(got[1], 1.0);
        assert_eq!(got[2], -1.0);
        assert_eq!(got[3], f32::INFINITY);
        assert_eq!(got[4], f32::NEG_INFINITY);
        assert!(got[5].is_nan());
        assert!(decode_fp8_chunk(&e5, Fp8Format::E5M2, 5, &mut got[..2], 6).is_err());
    }

    #[test]
    fn fp8_all_byte_patterns_match_pytorch_golden_bits() {
        let golden: std::collections::HashMap<String, Vec<u32>> =
            serde_json::from_str(include_str!("../testdata/fp8_torch_bits.json"))
                .expect("valid PyTorch FP8 golden JSON");
        let raw: Vec<u8> = (0..=u8::MAX).collect();
        for (key, format) in [("e4m3fn", Fp8Format::E4M3Fn), ("e5m2", Fp8Format::E5M2)] {
            let expected = golden
                .get(key)
                .unwrap_or_else(|| panic!("missing {key} golden values"));
            assert_eq!(expected.len(), 256, "{key} golden must cover every byte");
            let packed = PackedData::fp8(owned(raw.clone()), format, 256).unwrap();
            let mut actual = [0.0f32; 256];
            packed.decode_f32_chunk(0, &mut actual).unwrap();
            for (bits, (&got, &expected_bits)) in raw.iter().zip(actual.iter().zip(expected)) {
                let byte = *bits;
                let nan = match format {
                    Fp8Format::E4M3Fn => ((byte >> 3) & 0x0f) == 0x0f && (byte & 0x07) == 0x07,
                    Fp8Format::E5M2 => ((byte >> 2) & 0x1f) == 0x1f && (byte & 0x03) != 0,
                };
                let reference = f32::from_bits(expected_bits);
                if nan {
                    assert!(got.is_nan(), "{key} byte {byte:#04x} should decode as NaN");
                    assert!(
                        reference.is_nan(),
                        "{key} PyTorch golden for {byte:#04x} is not NaN"
                    );
                } else {
                    assert_eq!(got.to_bits(), expected_bits, "{key} byte {byte:#04x}");
                }
            }
        }
    }

    #[test]
    fn gptq_v1_decodes_input_packed_codes_and_minus_one_zeros_in_chunks() {
        let mut qweight = vec![0u8; 8 * 4];
        for out in 0..8 {
            let word = (1u32..=8)
                .enumerate()
                .fold(0u32, |acc, (i, q)| acc | (q << (4 * i)));
            qweight[out * 4..out * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        let qzeros = (0x4444_4444u32).to_le_bytes().to_vec(); // stored 4 decodes to zero point 5
        let mut scales = Vec::new();
        for _ in 0..8 {
            scales.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        }
        let tensor = PackedInt4::gptq_v1(
            owned(qweight),
            owned(qzeros),
            owned(scales),
            None,
            8,
            8,
            8,
            ScaleType::F16,
        )
        .unwrap();
        let mut got = [0.0; 8];
        tensor.decode_chunk(3, &mut got).unwrap();
        assert_eq!(got, [-1.0, 0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0]);
    }

    #[test]
    fn awq_gemm_inverts_packer_nibble_order() {
        let order = [0usize, 2, 4, 6, 1, 3, 5, 7];
        let mut qweight = Vec::new();
        for _input in 0..8 {
            let mut word = 0u32;
            for (slot, &logical) in order.iter().enumerate() {
                word |= (logical as u32) << (4 * slot);
            }
            qweight.extend_from_slice(&word.to_le_bytes());
        }
        let qzeros = 0x2222_2222u32.to_le_bytes().to_vec();
        let mut scales = Vec::new();
        for _ in 0..8 {
            scales.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        }
        let tensor = PackedInt4::awq_gemm(
            owned(qweight),
            owned(qzeros),
            owned(scales),
            8,
            8,
            8,
            ScaleType::F16,
        )
        .unwrap();
        let mut got = [0.0; 16];
        tensor.decode_chunk(0, &mut got).unwrap();
        assert_eq!(&got[..8], &[-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]);
        assert_eq!(&got[8..], &[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn q4z_rows_dequantize_to_the_values_of_the_dense_decode() {
        let (inp, out, group) = (64usize, 16usize, 32usize);
        let mut seed = 99u32;
        let mut bytes = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 24) as u8
                })
                .collect()
        };
        let mut scales = Vec::new();
        for i in 0..inp / group * out {
            scales.extend_from_slice(&half::f16::from_f32(0.01 * (1 + i % 7) as f32).to_bits().to_le_bytes());
        }
        let awq = PackedInt4::awq_gemm(
            owned(bytes(inp * out / 2)),
            owned(bytes(inp / group * out / 2)),
            owned(scales.clone()),
            inp,
            out,
            group,
            ScaleType::F16,
        )
        .unwrap();
        let gptq = PackedInt4::gptq_v1(
            owned(bytes(inp * out / 2)),
            owned(bytes(inp / group * out / 2)),
            owned(scales),
            None,
            inp,
            out,
            group,
            ScaleType::F16,
        )
        .unwrap();
        for tensor in [awq, gptq] {
            assert_eq!(tensor.q4z_group(), Some(group));
            let mut dense = vec![0.0f32; inp * out];
            tensor.decode_chunk(0, &mut dense).unwrap();
            for row in 0..out {
                let mut blocks = Vec::new();
                tensor.encode_q4z_row(row, &mut blocks).unwrap();
                let mut got = Vec::new();
                for b in blocks.chunks_exact(group / 2 + 3) {
                    let scale = half::f16::from_le_bytes([b[group / 2], b[group / 2 + 1]]).to_f32();
                    let zero = (b[group / 2 + 2] & 15) as f32;
                    for &q in &b[..group / 2] {
                        got.push(((q & 15) as f32 - zero) * scale);
                        got.push(((q >> 4) as f32 - zero) * scale);
                    }
                }
                assert_eq!(got, dense[row * inp..(row + 1) * inp], "row {row}");
            }
        }
      }

    #[test]
    fn q4zp_rows_with_act_order_dequantize_to_dense_decode() {
        let (inp, out, group) = (64usize, 16usize, 32usize);
        let mut seed = 99u32;
        let mut bytes = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 24) as u8
                })
                .collect()
        };
        let mut scales = Vec::new();
        for i in 0..inp / group * out {
            scales.extend_from_slice(&half::f16::from_f32(0.01 * (1 + i % 7) as f32).to_bits().to_le_bytes());
        }
        let mut count0 = 0;
        let mut count1 = 0;
        let mut g_idx_fixed = Vec::new();
        for i in 0..inp {
            let g = if (i % 2 == 0 && count0 < 32) || count1 >= 32 {
                count0 += 1;
                0u32
            } else {
                count1 += 1;
                1u32
            };
            g_idx_fixed.extend_from_slice(&g.to_le_bytes());
        }
        let gptq = PackedInt4::gptq_v1(
            owned(bytes(inp * out / 2)),
            owned(bytes(inp / group * out / 2)),
            owned(scales),
            Some(owned(g_idx_fixed)),
            inp,
            out,
            group,
            ScaleType::F16,
        )
        .unwrap();
        assert_eq!(gptq.q4z_group(), None);
        assert_eq!(gptq.q4zp_group(), Some(group));
        let perm = gptq.q4zp_perm().expect("valid perm");
        assert_eq!(perm.len(), inp);
        let mut dense = vec![0.0f32; inp * out];
        gptq.decode_chunk(0, &mut dense).unwrap();
        for row in 0..out {
            let mut blocks = Vec::new();
            gptq.encode_q4zp_row(row, &perm, &mut blocks).unwrap();
            let mut got = Vec::new();
            for b in blocks.chunks_exact(group / 2 + 3) {
                let scale = half::f16::from_le_bytes([b[group / 2], b[group / 2 + 1]]).to_f32();
                let zero = (b[group / 2 + 2] & 15) as f32;
                for &q in &b[..group / 2] {
                    got.push(((q & 15) as f32 - zero) * scale);
                    got.push(((q >> 4) as f32 - zero) * scale);
                }
            }
            let mut unpermuted = vec![0.0f32; inp];
            for c in 0..inp {
                unpermuted[perm[c] as usize] = got[c];
            }
            assert_eq!(unpermuted, dense[row * inp..(row + 1) * inp], "row {row}");
        }
    }
}
