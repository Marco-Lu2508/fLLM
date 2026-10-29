//! Device layouts of quantized weights for the Vulkan shaders (`shaders/vulkan/_q_fmt_*.glsl`) and the CPU side of the upload.
//!
//! The GGUF blocks are split into *planes* (struct of arrays), each plane starting at a 16-byte aligned word offset of one storage buffer,
//! so every shader reads aligned 32-bit words and needs neither 8-bit nor 16-bit storage. Some formats are decoded on the host
//! (K-quant scale bytes, f16 -> f32 super-block scales, nibble order of Q4Z). The exact plane tables are in `INTERFACE_QUANT.md`.

use crate::backend::{BackendError, QuantType};
use half::f16;

/// How the file layout of a format maps to the device layout (what `pack_weight` does).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Geometry {
    /// Fixed blocks of the file split into planes, one device block per file block.
    Planar,
    /// MLX affine group quantization: device blocks of 32 codes, one scale/bias word per group of `group` codes (plane 1).
    Mlx { bits: u32, group: u32, bf16: bool },
    /// Block-wise FP8: codes row-major (plane 0), the scale of the tile repeated for every row of the tile (plane 1, one f32 per row and tile column).
    Fp8b { block_rows: usize, block_cols: usize },
}

/// Static description of a supported weight format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatInfo {
    /// Shader name fragment (`mv_<tag>_s`, ...).
    pub tag: &'static str,
    /// Weights per device block (the shader constant `QK`).
    pub qk: usize,
    /// Weights per block of the file (`qk` except for MLX, whose file block is a whole group).
    pub src_qk: usize,
    /// Bytes of one block in the GGUF file.
    pub src_block_bytes: usize,
    /// Bytes per block in each of the (up to four) planes of the device layout; 0 = plane unused.
    pub plane_bytes: [usize; 4],
    pub geometry: Geometry,
    /// The decoder yields floats (`wchunk_f`) instead of small integers: only the f32 shaders exist (FP8, NF4, FP4).
    pub float_chunk: bool,
}

fn mlx_tag(bits: u32) -> Option<&'static str> {
    Some(match bits {
        2 => "mlx2",
        3 => "mlx3",
        4 => "mlx4",
        5 => "mlx5",
        6 => "mlx6",
        8 => "mlx8",
        _ => return None,
    })
}

/// Description of `quant`, `None` for formats without a Vulkan shader.
pub fn format_info(quant: QuantType) -> Option<FormatInfo> {
    let id = quant.to_gguf_type_id();
    if let Some((block_rows, block_cols)) = fllm_gguf_probe::fp8b_shape(id) {
        return Some(FormatInfo {
            tag: "fp8b",
            qk: 8,
            src_qk: 1,
            src_block_bytes: 1,
            plane_bytes: [8, 0, 0, 0],
            geometry: Geometry::Fp8b { block_rows, block_cols },
            float_chunk: true,
        });
    }
    if let Some(m) = fllm_gguf_probe::mlx_affine(id) {
        return Some(FormatInfo {
            tag: mlx_tag(m.bits)?,
            qk: 32,
            src_qk: m.group as usize,
            src_block_bytes: m.block_bytes() as usize,
            plane_bytes: [4 * m.bits as usize, 4, 0, 0],
            geometry: Geometry::Mlx { bits: m.bits, group: m.group, bf16: m.bf16 },
            float_chunk: false,
        });
    }
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
        QuantType::Nf4B64 => ("nf4b64", 64, 36, [32, 4, 0, 0]),
        QuantType::Nf4B128 => ("nf4b128", 128, 68, [64, 4, 0, 0]),
        QuantType::Fp4B64 => ("fp4b64", 64, 36, [32, 4, 0, 0]),
        QuantType::Fp4B128 => ("fp4b128", 128, 68, [64, 4, 0, 0]),
        QuantType::Int8Row => ("int8r", 64, 68, [64, 4, 0, 0]),
        _ => return None,
    };
    Some(FormatInfo {
        tag,
        qk,
        src_qk: qk,
        src_block_bytes: src,
        plane_bytes: planes,
        geometry: Geometry::Planar,
        float_chunk: matches!(quant, QuantType::Nf4B64 | QuantType::Nf4B128 | QuantType::Fp4B64 | QuantType::Fp4B128),
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

/// Formats added for the native formats of the other backends: bitsandbytes NF4 / FP4 / int8, block-wise FP8 (any tile shape is one entry of the capability
/// list, see `BackendCapabilities::supports_quant`) and the MLX affine types with a group of 32, 64 or 128.
pub fn extended_formats() -> Vec<QuantType> {
    let mut all = vec![
        QuantType::Nf4B64,
        QuantType::Nf4B128,
        QuantType::Fp4B64,
        QuantType::Fp4B128,
        QuantType::Int8Row,
        QuantType::Fp8Block,
    ];
    all.extend(QuantType::mlx_affine_common());
    all
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
    /// Format parameter the shader reads as push constant `off2` (0 = none, the offset of plane 2 is passed instead): MLX `group / 32 | bf16 << 16`,
    /// block-wise FP8 the number of columns of a tile.
    pub param: u32,
}

fn invalid(msg: String) -> BackendError {
    BackendError::InvalidShape(msg)
}

/// Bytes of plane `i` of a `rows x cols` weight (before the padding to 16 bytes).
fn plane_len(info: &FormatInfo, rows: usize, cols: usize, i: usize) -> Option<usize> {
    let blocks = rows.checked_mul(cols / info.qk)?;
    match info.geometry {
        Geometry::Planar => blocks.checked_mul(info.plane_bytes[i]),
        Geometry::Mlx { group, .. } => match i {
            0 => blocks.checked_mul(info.plane_bytes[0]),
            1 => rows.checked_mul(cols / group as usize)?.checked_mul(4),
            _ => Some(0),
        },
        Geometry::Fp8b { block_cols, .. } => match i {
            0 => rows.checked_mul(cols),
            1 => rows.checked_mul(cols.div_ceil(block_cols))?.checked_mul(4),
            _ => Some(0),
        },
    }
}

impl DeviceLayout {
    pub fn new(quant: QuantType, rows: usize, cols: usize) -> Result<Self, BackendError> {
        let info = format_info(quant).ok_or(BackendError::UnsupportedQuant(quant))?;
        if rows == 0 || cols == 0 {
            return Err(invalid(format!("empty {quant:?} weight {rows} x {cols}")));
        }
        if !cols.is_multiple_of(info.qk) || !cols.is_multiple_of(info.src_qk) {
            return Err(invalid(format!(
                "{quant:?} needs a multiple of {} columns, got {cols}",
                info.qk.max(info.src_qk)
            )));
        }
        // a chunk of 8 codes must not straddle two tiles
        if let Geometry::Fp8b { block_cols, .. } = info.geometry {
            if !block_cols.is_multiple_of(8) {
                return Err(invalid(format!("{quant:?}: tiles of {block_cols} columns are narrower than a chunk of 8")));
            }
        }
        let blocks_per_row = cols / info.qk;
        rows.checked_mul(blocks_per_row)
            .ok_or_else(|| invalid("weight block count overflows".into()))?;
        let mut offset = 0usize;
        let mut plane_word_offset = [0u32; 4];
        for (i, slot) in plane_word_offset.iter_mut().enumerate() {
            *slot = u32::try_from(offset / 4)
                .map_err(|_| invalid("weight larger than 16 GiB".into()))?;
            let size = plane_len(&info, rows, cols, i)
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
        let param = match info.geometry {
            Geometry::Planar => 0,
            Geometry::Mlx { group, bf16, .. } => (group / 32) | (u32::from(bf16) << 16),
            Geometry::Fp8b { block_cols, .. } => block_cols as u32,
        };
        Ok(Self {
            info,
            rows,
            cols,
            blocks_per_row,
            plane_word_offset,
            total_bytes: offset,
            param,
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
    if cols == 0 || !cols.is_multiple_of(info.qk) || !cols.is_multiple_of(info.src_qk) {
        return Err(invalid(format!(
            "{quant:?} needs a multiple of {} columns, got {cols}",
            info.qk.max(info.src_qk)
        )));
    }
    let per_row: usize = match info.geometry {
        Geometry::Planar => info.plane_bytes.iter().sum::<usize>() * (cols / info.qk),
        _ => (0..4).map(|i| plane_len(&info, 1, cols, i).unwrap_or(0)).sum(),
    };
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

/// Rows `first .. first + n` of a block-wise FP8 weight (`raw` = the whole tensor with codes, tile scales and trailer, one matrix) in the device layout: the codes
/// of the rows, and one f32 scale per row and tile column (the scale of the tile, repeated for the rows of the tile).
fn pack_fp8b(quant: QuantType, raw: &[u8], rows: usize, cols: usize, first: usize, n: usize) -> Result<PackedWeight, BackendError> {
    let kind = quant.to_gguf_type_id();
    let tiles = fllm_gguf_probe::Fp8bLayout::parse(kind, raw).map_err(invalid)?;
    if tiles.slabs != 1 || tiles.rows != rows || tiles.cols != cols {
        return Err(invalid(format!(
            "{quant:?} weight holds {} matrices of {} x {}, one {rows} x {cols} matrix was expected",
            tiles.slabs, tiles.rows, tiles.cols
        )));
    }
    if n == 0 || first + n > rows {
        return Err(invalid(format!("rows {first}..{} outside the {rows} rows of the weight", first + n)));
    }
    let layout = DeviceLayout::new(quant, n, cols)?;
    let mut out = vec![0u8; layout.total_bytes];
    put(&mut out, 0, &raw[first * cols..(first + n) * cols]);
    let scales = layout.plane_byte_offset(1);
    let tile_cols = tiles.tile_cols();
    for r in 0..n {
        for t in 0..tile_cols {
            let s = tiles.tile_index(0, first + r, t * tiles.block_cols);
            let at = tiles.scale_offset() + s * 4;
            put(&mut out, scales + (r * tile_cols + t) * 4, &raw[at..at + 4]);
        }
    }
    Ok(PackedWeight { layout, bytes: out, perm: None })
}

/// [`pack_fp8b`] for the rows `first .. first + n` when `quant` is a block-wise FP8 type (`None` for every other format: the caller slices the file rows itself).
pub fn pack_fp8b_slice(quant: QuantType, raw: &[u8], rows: usize, cols: usize, first: usize, n: usize) -> Option<Result<PackedWeight, BackendError>> {
    matches!(format_info(quant)?.geometry, Geometry::Fp8b { .. }).then(|| pack_fp8b(quant, raw, rows, cols, first, n))
}

/// An MLX affine weight (`[scale][bias][codes]` per group, file order) in the device layout: the codes of all groups back to back (a group of `group` codes is
/// `group / 32` device blocks of `bits` words), and one word `scale | bias << 16` per group.
fn pack_mlx(quant: QuantType, raw: &[u8], rows: usize, cols: usize) -> Result<PackedWeight, BackendError> {
    let layout = DeviceLayout::new(quant, rows, cols)?;
    let info = layout.info;
    let Geometry::Mlx { bits, group, .. } = info.geometry else {
        return Err(BackendError::UnsupportedQuant(quant));
    };
    let groups = rows * (cols / group as usize);
    if raw.len() != groups * info.src_block_bytes {
        return Err(invalid(format!("{quant:?} {rows} x {cols} needs {} bytes, got {}", groups * info.src_block_bytes, raw.len())));
    }
    let code_bytes = group as usize * bits as usize / 8;
    let mut out = vec![0u8; layout.total_bytes];
    let (plane0, plane1) = (layout.plane_byte_offset(0), layout.plane_byte_offset(1));
    for (g, b) in raw.chunks_exact(info.src_block_bytes).enumerate() {
        put(&mut out, plane0 + g * code_bytes, &b[4..]);
        put(&mut out, plane1 + g * 4, &b[0..4]);
    }
    Ok(PackedWeight { layout, bytes: out, perm: None })
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
    match info.geometry {
        Geometry::Fp8b { .. } => return pack_fp8b(quant, raw, rows, cols, 0, rows),
        Geometry::Mlx { .. } => return pack_mlx(quant, raw, rows, cols),
        Geometry::Planar => {}
    }
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
            // bitsandbytes: `[f32 scale][codes]` into a plane of codes and a plane of scales
            QuantType::Nf4B64 | QuantType::Nf4B128 | QuantType::Fp4B64 | QuantType::Fp4B128 | QuantType::Int8Row => {
                put(&mut out, at(0, g), &b[4..]);
                put(&mut out, at(1, g), &b[0..4]);
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
    if matches!(layout.info.geometry, Geometry::Fp8b { .. }) {
        let step = rows_per_slice.max(1);
        return (0..rows)
            .step_by(step)
            .map(|first| Ok(WeightSlice { row_base: first, packed: pack_fp8b(quant, raw, rows, cols, first, step.min(rows - first))? }))
            .collect();
    }
    let row_bytes = (cols / layout.info.src_qk) * layout.info.src_block_bytes;
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
