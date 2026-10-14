//! Device layouts of quantized weights for the Vulkan shaders (`shaders/vulkan/_q_fmt_*.glsl`) and the CPU side of the upload.
//!
//! The GGUF blocks are split into *planes* (struct of arrays), each plane starting at a 16-byte aligned word offset of one storage buffer,
//! so every shader reads aligned 32-bit words and needs neither 8-bit nor 16-bit storage. Some formats are decoded on the host
//! (K-quant scale bytes, f16 -> f32 super-block scales, nibble order of Q4Z). The exact plane tables are in `INTERFACE_QUANT.md`.

use crate::backend::{BackendError, QuantType};
use half::f16;

/// Static description of a supported weight format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatInfo {
    /// Shader name fragment (`mv_<tag>_s`, ...).
    pub tag: &'static str,
    /// Weights per block (the shader constant `QK`).
    pub qk: usize,
    /// Bytes of one block in the GGUF file.
    pub src_block_bytes: usize,
    /// Bytes per block in each of the (up to four) planes of the device layout; 0 = plane unused.
    pub plane_bytes: [usize; 4],
}

/// Description of `quant`, `None` for formats without a Vulkan shader.
pub fn format_info(quant: QuantType) -> Option<FormatInfo> {
    let (tag, qk, src, planes): (&'static str, usize, usize, [usize; 4]) = match quant {
        QuantType::Q4_0 => ("q4_0", 32, 18, [16, 2, 0, 0]),
        QuantType::Q4_1 => ("q4_1", 32, 20, [16, 4, 0, 0]),
        QuantType::Q5_0 => ("q5_0", 32, 22, [16, 4, 2, 0]),
        QuantType::Q5_1 => ("q5_1", 32, 24, [16, 4, 4, 0]),
        QuantType::Q8_0 => ("q8_0", 32, 34, [32, 2, 0, 0]),
        QuantType::Q2_K => ("q2_k", 256, 84, [64, 16, 4, 0]),
        QuantType::Q3_K => ("q3_k", 256, 110, [64, 32, 16, 4]),
        QuantType::Q4_K => ("q4_k", 256, 144, [128, 16, 4, 0]),
        QuantType::Q5_K => ("q5_k", 256, 176, [128, 32, 16, 4]),
        QuantType::Q6_K => ("q6_k", 256, 210, [128, 64, 16, 4]),
        QuantType::IQ4_NL => ("iq4_nl", 32, 18, [16, 2, 0, 0]),
        QuantType::Mxfp4 => ("mxfp4", 32, 17, [16, 1, 0, 0]),
        QuantType::NvFp4 => ("nvfp4", 64, 36, [32, 4, 0, 0]),
        QuantType::Q4Z32 | QuantType::Q4ZP32 => ("q4z32", 32, 19, [16, 4, 0, 0]),
        QuantType::Q4Z64 | QuantType::Q4ZP64 => ("q4z64", 64, 35, [32, 4, 0, 0]),
        QuantType::Q4Z128 | QuantType::Q4ZP128 => ("q4z128", 128, 67, [64, 4, 0, 0]),
        _ => return None,
    };
    Some(FormatInfo {
        tag,
        qk,
        src_block_bytes: src,
        plane_bytes: planes,
    })
}

/// Formats the quantized-linear shaders of this package handle.
pub fn supported_formats() -> &'static [QuantType] {
    &[
        QuantType::Q4_0,
        QuantType::Q4_1,
        QuantType::Q5_0,
        QuantType::Q5_1,
        QuantType::Q8_0,
        QuantType::Q2_K,
        QuantType::Q3_K,
        QuantType::Q4_K,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::IQ4_NL,
        QuantType::Mxfp4,
        QuantType::NvFp4,
        QuantType::Q4Z32,
        QuantType::Q4Z64,
        QuantType::Q4Z128,
        QuantType::Q4ZP32,
        QuantType::Q4ZP64,
        QuantType::Q4ZP128,
    ]
}

/// Where the planes of a `rows x cols` weight lie in the device buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceLayout {
    pub info: FormatInfo,
    pub rows: usize,
    pub cols: usize,
    /// `cols / qk`
    pub blocks_per_row: usize,
    /// Word (4 byte) offset of every plane; plane 0 is at 0, unused planes repeat the end of the buffer.
    pub plane_word_offset: [u32; 4],
    /// Size of the device buffer (a multiple of 16).
    pub total_bytes: usize,
}

fn invalid(msg: String) -> BackendError {
    BackendError::InvalidShape(msg)
}

impl DeviceLayout {
    pub fn new(quant: QuantType, rows: usize, cols: usize) -> Result<Self, BackendError> {
        let info = format_info(quant).ok_or(BackendError::UnsupportedQuant(quant))?;
        if rows == 0 || cols == 0 {
            return Err(invalid(format!("empty {quant:?} weight {rows} x {cols}")));
        }
        if !cols.is_multiple_of(info.qk) {
            return Err(invalid(format!(
                "{quant:?} needs a multiple of {} columns, got {cols}",
                info.qk
            )));
        }
        let blocks_per_row = cols / info.qk;
        let blocks = rows
            .checked_mul(blocks_per_row)
            .ok_or_else(|| invalid("weight block count overflows".into()))?;
        let mut offset = 0usize;
        let mut plane_word_offset = [0u32; 4];
        for (i, &bytes) in info.plane_bytes.iter().enumerate() {
            plane_word_offset[i] = u32::try_from(offset / 4)
                .map_err(|_| invalid("weight larger than 16 GiB".into()))?;
            let size = blocks
                .checked_mul(bytes)
                .ok_or_else(|| invalid("weight size overflows".into()))?
                .next_multiple_of(16);
            offset = offset
                .checked_add(size)
                .ok_or_else(|| invalid("weight size overflows".into()))?;
        }
        if offset / 4 > u32::MAX as usize {
            return Err(invalid(format!(
                "{rows} x {cols} weight exceeds 32-bit word indexing"
            )));
        }
        Ok(Self {
            info,
            rows,
            cols,
            blocks_per_row,
            plane_word_offset,
            total_bytes: offset,
        })
    }

    /// Number of blocks (`rows * blocks_per_row`).
    pub fn blocks(&self) -> usize {
        self.rows * self.blocks_per_row
    }

    /// Byte offset of plane `i` in the device buffer.
    pub fn plane_byte_offset(&self, i: usize) -> usize {
        self.plane_word_offset[i] as usize * 4
    }
}

/// A weight in device layout, ready to be copied into a storage buffer.
#[derive(Debug, Clone)]
pub struct PackedWeight {
    pub layout: DeviceLayout,
    /// The planes (`layout.total_bytes` bytes).
    pub bytes: Vec<u8>,
    /// Q4ZP: the column permutation (`x'[c] = x[perm[c]]`), to be uploaded as `DeviceWeight::perm`; `None` otherwise.
    pub perm: Option<Vec<u32>>,
}

/// Largest number of rows (a multiple of 64) of `cols`-wide weights of `quant` whose device buffer stays within `max_range` bytes
/// (`maxStorageBufferRange`); weights with more rows are uploaded as several slices dispatched with `row_base`. At least 64.
pub fn rows_per_slice(
    quant: QuantType,
    cols: usize,
    max_range: usize,
) -> Result<usize, BackendError> {
    let info = format_info(quant).ok_or(BackendError::UnsupportedQuant(quant))?;
    if cols == 0 || !cols.is_multiple_of(info.qk) {
        return Err(invalid(format!(
            "{quant:?} needs a multiple of {} columns, got {cols}",
            info.qk
        )));
    }
    let per_row: usize = info.plane_bytes.iter().sum::<usize>() * (cols / info.qk);
    // planes are padded to 16 bytes each: keep a margin of 4 x 16 bytes
    let rows = max_range.saturating_sub(64) / per_row.max(1);
    Ok((rows / 64 * 64).max(64))
}

fn put(out: &mut [u8], at: usize, data: &[u8]) {
    out[at..at + data.len()].copy_from_slice(data);
}

fn f16_to_f32_bytes(b: &[u8]) -> [u8; 4] {
    f16::from_le_bytes([b[0], b[1]]).to_f32().to_le_bytes()
}

/// The 16 sub-block scales of a Q3_K block as int8 (`scale - 32`) from its 12 packed bytes.
pub fn q3k_scales(sc: &[u8]) -> [i8; 16] {
    let mut s = [0i8; 16];
    for i in 0..4 {
        let (hi, lo, packed) = (sc[i], sc[4 + i], sc[8 + i]);
        s[i] = (((hi & 15) | ((packed & 3) << 4)) as i32 - 32) as i8;
        s[i + 4] = (((lo & 15) | (((packed >> 2) & 3) << 4)) as i32 - 32) as i8;
        s[i + 8] = (((hi >> 4) | (((packed >> 4) & 3) << 4)) as i32 - 32) as i8;
        s[i + 12] = (((lo >> 4) | (((packed >> 6) & 3) << 4)) as i32 - 32) as i8;
    }
    s
}

/// The eight 6-bit scales and eight 6-bit mins of a Q4_K / Q5_K block from its 12 packed bytes (`get_scale_min_k4`).
pub fn q4k_scales_mins(sc: &[u8]) -> ([u8; 8], [u8; 8]) {
    let (mut s, mut m) = ([0u8; 8], [0u8; 8]);
    for j in 0..8 {
        if j < 4 {
            s[j] = sc[j] & 63;
            m[j] = sc[j + 4] & 63;
        } else {
            s[j] = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4);
            m[j] = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
        }
    }
    (s, m)
}

/// Rewrites the `group / 2` code bytes of a Q4Z block so that every chunk of 8 values is one word: byte `k` holds value `k` in the low and
/// value `k + 4` in the high nibble (the file stores value `2j` in the low and `2j + 1` in the high nibble of byte `j`).
fn q4z_codes(src: &[u8], dst: &mut [u8]) {
    for (chunk, out) in src
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<4>().0.iter_mut())
    {
        let mut v = [0u8; 8];
        for (k, &b) in chunk.iter().enumerate() {
            v[2 * k] = b & 15;
            v[2 * k + 1] = b >> 4;
        }
        for k in 0..4 {
            out[k] = v[k] | (v[k + 4] << 4);
        }
    }
}

/// Converts the raw GGUF bytes of a `rows x cols` weight to the device layout. For Q4ZP the permutation table that follows the blocks in the
/// file is returned separately.
pub fn pack_weight(
    quant: QuantType,
    raw: &[u8],
    rows: usize,
    cols: usize,
) -> Result<PackedWeight, BackendError> {
    let layout = DeviceLayout::new(quant, rows, cols)?;
    let info = layout.info;
    let blocks = layout.blocks();
    let block_len = blocks * info.src_block_bytes;
    let perm_len = if quant.is_q4zp() { cols * 4 } else { 0 };
    if raw.len() != block_len + perm_len {
        return Err(invalid(format!(
            "{quant:?} {rows} x {cols} needs {} bytes, got {}",
            block_len + perm_len,
            raw.len()
        )));
    }
    let perm = if quant.is_q4zp() {
        let table: Vec<u32> = raw[block_len..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        if table.iter().any(|&p| p as usize >= cols) {
            return Err(invalid("Q4ZP permutation index outside the row".into()));
        }
        Some(table)
    } else {
        None
    };
    let mut out = vec![0u8; layout.total_bytes];
    let at = |plane: usize, g: usize| layout.plane_byte_offset(plane) + g * info.plane_bytes[plane];
    for (g, b) in raw[..block_len]
        .chunks_exact(info.src_block_bytes)
        .enumerate()
    {
        match quant {
            QuantType::Q4_0 | QuantType::IQ4_NL => {
                put(&mut out, at(0, g), &b[2..18]);
                put(&mut out, at(1, g), &b[0..2]);
            }
            QuantType::Q4_1 => {
                put(&mut out, at(0, g), &b[4..20]);
                put(&mut out, at(1, g), &b[0..4]);
            }
            QuantType::Q5_0 => {
                put(&mut out, at(0, g), &b[6..22]);
                put(&mut out, at(1, g), &b[2..6]);
                put(&mut out, at(2, g), &b[0..2]);
            }
            QuantType::Q5_1 => {
                put(&mut out, at(0, g), &b[8..24]);
                put(&mut out, at(1, g), &b[4..8]);
                put(&mut out, at(2, g), &b[0..4]);
            }
            QuantType::Q8_0 => {
                put(&mut out, at(0, g), &b[2..34]);
                put(&mut out, at(1, g), &b[0..2]);
            }
            QuantType::Q2_K => {
                put(&mut out, at(0, g), &b[16..80]);
                put(&mut out, at(1, g), &b[0..16]);
                put(&mut out, at(2, g), &b[80..84]);
            }
            QuantType::Q3_K => {
                put(&mut out, at(0, g), &b[32..96]);
                put(&mut out, at(1, g), &b[0..32]);
                let scales = q3k_scales(&b[96..108]);
                let bytes: Vec<u8> = scales.iter().map(|&s| s as u8).collect();
                put(&mut out, at(2, g), &bytes);
                put(&mut out, at(3, g), &f16_to_f32_bytes(&b[108..110]));
            }
            QuantType::Q4_K => {
                put(&mut out, at(0, g), &b[16..144]);
                let (s, m) = q4k_scales_mins(&b[4..16]);
                put(&mut out, at(1, g), &s);
                put(&mut out, at(1, g) + 8, &m);
                put(&mut out, at(2, g), &b[0..4]);
            }
            QuantType::Q5_K => {
                put(&mut out, at(0, g), &b[48..176]);
                put(&mut out, at(1, g), &b[16..48]);
                let (s, m) = q4k_scales_mins(&b[4..16]);
                put(&mut out, at(2, g), &s);
                put(&mut out, at(2, g) + 8, &m);
                put(&mut out, at(3, g), &b[0..4]);
            }
            QuantType::Q6_K => {
                put(&mut out, at(0, g), &b[0..128]);
                put(&mut out, at(1, g), &b[128..192]);
                put(&mut out, at(2, g), &b[192..208]);
                put(&mut out, at(3, g), &f16_to_f32_bytes(&b[208..210]));
            }
            QuantType::Mxfp4 => {
                put(&mut out, at(0, g), &b[1..17]);
                out[at(1, g)] = b[0];
            }
            QuantType::NvFp4 => {
                put(&mut out, at(0, g), &b[4..36]);
                put(&mut out, at(1, g), &b[0..4]);
            }
            QuantType::Q4Z32
            | QuantType::Q4ZP32
            | QuantType::Q4Z64
            | QuantType::Q4ZP64
            | QuantType::Q4Z128
            | QuantType::Q4ZP128 => {
                let half = info.qk / 2;
                let dst = at(0, g);
                q4z_codes(&b[..half], &mut out[dst..dst + half]);
                let zero = (b[half + 2] & 15) as u32;
                let d = u16::from_le_bytes([b[half], b[half + 1]]) as u32;
                put(&mut out, at(1, g), &(d | (zero << 16)).to_le_bytes());
            }
            _ => return Err(BackendError::UnsupportedQuant(quant)),
        }
    }
    Ok(PackedWeight {
        layout,
        bytes: out,
        perm,
    })
}

/// A group of consecutive rows of a weight that did not fit one storage buffer (`maxStorageBufferRange`), see [`pack_weight_slices`].
#[derive(Debug, Clone)]
pub struct WeightSlice {
    /// Index of the first row of the slice in the whole weight: the `row_base` of its dispatch.
    pub row_base: usize,
    pub packed: PackedWeight,
}

/// Like [`pack_weight`] for a weight that is uploaded as several buffers of at most `rows_per_slice` rows (use [`rows_per_slice`]). Every slice
/// is dispatched with its own `row_base` and with `out_stride` = the total number of rows. Q4ZP slices all carry the (shared) permutation table.
pub fn pack_weight_slices(
    quant: QuantType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    rows_per_slice: usize,
) -> Result<Vec<WeightSlice>, BackendError> {
    let layout = DeviceLayout::new(quant, rows, cols)?;
    let row_bytes = layout.blocks_per_row * layout.info.src_block_bytes;
    let block_len = rows * row_bytes;
    let perm_len = if quant.is_q4zp() { cols * 4 } else { 0 };
    if raw.len() != block_len + perm_len {
        return Err(invalid(format!(
            "{quant:?} {rows} x {cols} needs {} bytes, got {}",
            block_len + perm_len,
            raw.len()
        )));
    }
    let step = rows_per_slice.max(1);
    let mut slices = Vec::new();
    let mut first = 0;
    while first < rows {
        let n = step.min(rows - first);
        let part = &raw[first * row_bytes..(first + n) * row_bytes];
        let packed = if perm_len == 0 {
            pack_weight(quant, part, n, cols)?
        } else {
            let mut with_perm = part.to_vec();
            with_perm.extend_from_slice(&raw[block_len..]);
            pack_weight(quant, &with_perm, n, cols)?
        };
        slices.push(WeightSlice {
            row_base: first,
            packed,
        });
        first += n;
    }
    Ok(slices)
}
