//! Device images of the weights: [`pack_weight`] turns the bytes of a GGUF tensor into what the Metal kernels read.
//!
//! Most formats are used as stored in the file (`DeviceLayout::Raw`): a memory-mapped model needs no copy, the device image is a
//! borrowed slice of the file bytes (with `Cow::Borrowed`; the caller can bind the mapped pages directly, see `buffers.rs`).
//! MXFP4 and the Q4Z formats are re-sorted into two planes (`DeviceLayout::Planar`), see [`super::format::Repack`].
//!
//! Alignment the kernels rely on: the buffer offset of the image is a multiple of 4 (16 for dense F32/F16/BF16 rows, whose
//! activation loads are 16 bytes wide); the blocks of Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K and Q6_K are only 2-byte aligned and are read
//! with 16 bit loads.

use super::format::{format_info, FormatInfo, Repack};
use crate::backend::QuantType;
use fllm_gguf_probe::exl::Exl2View;
use fllm_gguf_probe::Fp8bLayout;
use rayon::prelude::*;
use std::borrow::Cow;
use std::fmt;

/// How the device image relates to the bytes of the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceLayout {
    /// Identical to the file (without the permutation table of Q4ZP; FP8M keeps its row scales behind the E4M3 bytes).
    Raw,
    /// Plane 0 at byte 0, plane 1 at [`Planes::plane1`] (a multiple of 16).
    Planar,
    /// A format-specific image with its own header (EXL2, see `q_fmt_exl2` in common.metal); `plane1` is 0.
    Image,
}

/// Plane geometry of a weight image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Planes {
    pub layout: DeviceLayout,
    /// Byte offset of plane 1 (0 if the format has none); passed to the kernels as `q_params::plane1`.
    pub plane1: u64,
    /// Size of the device image in bytes.
    pub total_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightError {
    /// The format has no Metal kernel.
    UnsupportedFormat(QuantType),
    /// `rows`, `cols` do not fit the block geometry of the format.
    BadShape {
        quant: QuantType,
        rows: usize,
        cols: usize,
        reason: &'static str,
    },
    /// The tensor has the wrong number of bytes.
    BadLength {
        quant: QuantType,
        expected: usize,
        got: usize,
    },
    /// The permutation table of an act-order weight is not a permutation of `0..cols`.
    BadPermutation(String),
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightError::UnsupportedFormat(q) => {
                write!(f, "metal: no kernel for weight format {q:?}")
            }
            WeightError::BadShape {
                quant,
                rows,
                cols,
                reason,
            } => {
                write!(
                    f,
                    "metal: {rows} x {cols} weight of format {quant:?}: {reason}"
                )
            }
            WeightError::BadLength {
                quant,
                expected,
                got,
            } => {
                write!(
                    f,
                    "metal: {quant:?} weight has {got} bytes, expected {expected}"
                )
            }
            WeightError::BadPermutation(why) => {
                write!(f, "metal: invalid permutation table: {why}")
            }
        }
    }
}

impl std::error::Error for WeightError {}

/// A weight ready for upload.
#[derive(Clone, Debug)]
pub struct PackedWeight<'a> {
    pub quant: QuantType,
    pub rows: usize,
    pub cols: usize,
    pub planes: Planes,
    /// The device image. Borrowed from the input for [`DeviceLayout::Raw`].
    pub bytes: Cow<'a, [u8]>,
    /// Q4ZP: the permutation table, one entry per column (`x'[c] = x[perm[c]]`); upload it as `DeviceWeight::perm`.
    pub perm: Option<Vec<u32>>,
}

fn round_up(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

/// The shape rules a blob format has without looking at the blob.
pub(super) fn validate_blob_shape(info: &FormatInfo, rows: usize, cols: usize) -> Result<(), WeightError> {
    check_shape(info, rows, cols)
}

fn check_shape(info: &FormatInfo, rows: usize, cols: usize) -> Result<(), WeightError> {
    let bad = |reason| WeightError::BadShape {
        quant: info.quant,
        rows,
        cols,
        reason,
    };
    if rows == 0 || cols == 0 {
        return Err(bad("empty matrix"));
    }
    if !cols.is_multiple_of(info.qk) {
        return Err(bad("cols is not a multiple of the block size"));
    }
    if !cols.is_multiple_of(8) {
        return Err(bad("cols is not a multiple of 8"));
    }
    if u32::try_from(rows).is_err() || u32::try_from(cols).is_err() {
        return Err(bad("dimension exceeds 32 bits"));
    }
    rows.checked_mul(cols)
        .ok_or_else(|| bad("rows * cols overflows"))?;
    Ok(())
}

/// Layout of a block-wise FP8 weight of one `rows` x `cols` matrix (the shape of the tiles is in the type id).
fn fp8b_layout(quant: QuantType, rows: usize, cols: usize) -> Fp8bLayout {
    Fp8bLayout::new(quant.to_gguf_type_id(), cols, rows, 1).expect("an FP8B type id")
}

/// Bytes of the part of the file that holds the blocks (without permutation table).
fn block_bytes_len(info: &FormatInfo, rows: usize, cols: usize) -> usize {
    if info.tiled {
        // codes, f32 tile scales, trailer
        fp8b_layout(info.quant, rows, cols).total_len()
    } else if info.row_scales {
        rows * cols
    } else {
        rows * cols / info.src_qk * info.src_block_bytes
    }
}

/// The error of the functions that would need the bytes of a self-describing blob (EXL2) to answer.
fn self_describing(quant: QuantType, rows: usize, cols: usize) -> WeightError {
    WeightError::BadShape {
        quant,
        rows,
        cols,
        reason: "the layout of this format is described by the header of the tensor, not by its shape",
    }
}

/// Number of bytes the tensor `quant` of `rows` x `cols` has in the GGUF file (the Q4ZP table and the FP8M row scales included).
pub fn expected_raw_len(quant: QuantType, rows: usize, cols: usize) -> Result<usize, WeightError> {
    let info = format_info(quant).ok_or(WeightError::UnsupportedFormat(quant))?;
    check_shape(&info, rows, cols)?;
    if info.blob {
        return Err(self_describing(quant, rows, cols));
    }
    let mut len = block_bytes_len(&info, rows, cols);
    if info.permuted {
        len += cols * 4;
    }
    if info.row_scales {
        len += rows * 2;
    }
    Ok(len)
}

/// Geometry of the device image of a `rows` x `cols` weight without looking at its bytes.
pub fn device_layout(quant: QuantType, rows: usize, cols: usize) -> Result<Planes, WeightError> {
    let info = format_info(quant).ok_or(WeightError::UnsupportedFormat(quant))?;
    check_shape(&info, rows, cols)?;
    if info.blob {
        return Err(self_describing(quant, rows, cols));
    }
    let blocks = rows * cols / info.qk;
    Ok(match info.repack {
        Repack::None => {
            let mut total = block_bytes_len(&info, rows, cols);
            let mut plane1 = 0;
            if info.tiled {
                // the scale table follows the codes; the trailer behind it stays in the image and is not read
                plane1 = (rows * cols) as u64;
            } else if info.row_scales {
                plane1 = total as u64;
                total += rows * 2;
            }
            Planes {
                layout: DeviceLayout::Raw,
                plane1,
                total_bytes: total as u64,
            }
        }
        Repack::Mxfp4 => {
            let plane1 = round_up(blocks * info.block_bytes, 16);
            Planes {
                layout: DeviceLayout::Planar,
                plane1: plane1 as u64,
                total_bytes: round_up(plane1 + blocks, 16) as u64,
            }
        }
        Repack::Exl2 => unreachable!("blob formats return above"),
        Repack::Q4z | Repack::Mlx => {
            let plane1 = round_up(blocks * info.block_bytes, 16);
            Planes {
                layout: DeviceLayout::Planar,
                plane1: plane1 as u64,
                total_bytes: round_up(plane1 + blocks * 4, 16) as u64,
            }
        }
    })
}

fn parse_permutation(table: &[u8], cols: usize) -> Result<Vec<u32>, WeightError> {
    debug_assert_eq!(table.len(), cols * 4);
    let perm: Vec<u32> = table
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let mut seen = vec![false; cols];
    for &p in &perm {
        let slot = seen
            .get_mut(p as usize)
            .ok_or_else(|| WeightError::BadPermutation(format!("entry {p} >= {cols}")))?;
        if *slot {
            return Err(WeightError::BadPermutation(format!(
                "entry {p} appears twice"
            )));
        }
        *slot = true;
    }
    Ok(perm)
}

/// Builds the device image of one weight from the `raw` bytes of its GGUF tensor (`rows` x `cols`).
pub fn pack_weight(
    quant: QuantType,
    rows: usize,
    cols: usize,
    raw: &[u8],
) -> Result<PackedWeight<'_>, WeightError> {
    let info = format_info(quant).ok_or(WeightError::UnsupportedFormat(quant))?;
    if info.blob {
        return pack_exl2(&info, rows, cols, raw);
    }
    let expected = expected_raw_len(quant, rows, cols)?;
    if raw.len() != expected {
        return Err(WeightError::BadLength {
            quant,
            expected,
            got: raw.len(),
        });
    }
    let planes = device_layout(quant, rows, cols)?;
    if info.tiled {
        // the trailer of the tensor must describe exactly one matrix of this shape (a stack of experts is not one weight)
        match Fp8bLayout::parse(quant.to_gguf_type_id(), raw) {
            Ok(l) if l.slabs == 1 && l.rows == rows && l.cols == cols => {}
            _ => {
                return Err(WeightError::BadShape {
                    quant,
                    rows,
                    cols,
                    reason: "the FP8B trailer does not describe one matrix of this shape",
                })
            }
        }
    }
    let blocks_len = block_bytes_len(&info, rows, cols);
    let perm = if info.permuted {
        Some(parse_permutation(&raw[blocks_len..], cols)?)
    } else {
        None
    };
    let blocks = rows * cols / info.qk;
    let bytes: Cow<'_, [u8]> = match info.repack {
        Repack::None => Cow::Borrowed(&raw[..planes.total_bytes as usize]),
        Repack::Mxfp4 => {
            let mut out = vec![0u8; planes.total_bytes as usize];
            let (codes, scales) = out.split_at_mut(planes.plane1 as usize);
            codes
                .par_chunks_mut(16 * 64)
                .zip(scales.par_chunks_mut(64))
                .enumerate()
                .for_each(|(i, (c, s))| {
                    for (k, scale) in s.iter_mut().enumerate() {
                        let block = i * 64 + k;
                        if block >= blocks {
                            break;
                        }
                        let src = &raw[block * 17..block * 17 + 17];
                        *scale = src[0];
                        c[k * 16..k * 16 + 16].copy_from_slice(&src[1..17]);
                    }
                });
            Cow::Owned(out)
        }
        Repack::Q4z => {
            let half = info.block_bytes;
            let mut out = vec![0u8; planes.total_bytes as usize];
            let (codes, params) = out.split_at_mut(planes.plane1 as usize);
            codes
                .par_chunks_mut(half * 64)
                .zip(params.par_chunks_mut(4 * 64))
                .enumerate()
                .for_each(|(i, (c, p))| {
                    for k in 0..p.len() / 4 {
                        let block = i * 64 + k;
                        if block >= blocks {
                            break;
                        }
                        let src = &raw[block * (half + 3)..(block + 1) * (half + 3)];
                        c[k * half..(k + 1) * half].copy_from_slice(&src[..half]);
                        p[k * 4..k * 4 + 2].copy_from_slice(&src[half..half + 2]);
                        p[k * 4 + 2] = src[half + 2] & 15;
                        p[k * 4 + 3] = 0;
                    }
                });
            Cow::Owned(out)
        }
        Repack::Exl2 => unreachable!("blob formats return above"),
        Repack::Mlx => {
            // [scale: 2][bias: 2][codes] per group -> codes plane + one word per group (scale in the low half, bias in the high half)
            let codes_len = info.block_bytes;
            let mut out = vec![0u8; planes.total_bytes as usize];
            let (codes, params) = out.split_at_mut(planes.plane1 as usize);
            codes
                .par_chunks_mut(codes_len * 64)
                .zip(params.par_chunks_mut(4 * 64))
                .enumerate()
                .for_each(|(i, (c, p))| {
                    for k in 0..p.len() / 4 {
                        let block = i * 64 + k;
                        if block >= blocks {
                            break;
                        }
                        let src = &raw[block * (codes_len + 4)..(block + 1) * (codes_len + 4)];
                        p[k * 4..k * 4 + 4].copy_from_slice(&src[..4]);
                        c[k * codes_len..(k + 1) * codes_len].copy_from_slice(&src[4..]);
                    }
                });
            Cow::Owned(out)
        }
    };
    Ok(PackedWeight {
        quant,
        rows,
        cols,
        planes,
        bytes,
        perm,
    })
}

/// The device image of an EXL2 blob (layout in `q_fmt_exl2`, common.metal): a 16-byte header, the rows of the blob, the per-chunk table of
/// (byte offset of the codes inside the row, group | bits << 24) and the group maxima divided by 256; the permutation is returned separately.
/// Refused (the caller expands the weight to f16): groups that are not a multiple of 8 inputs, an invalid blob, another shape.
fn pack_exl2<'a>(info: &FormatInfo, rows: usize, cols: usize, raw: &'a [u8]) -> Result<PackedWeight<'a>, WeightError> {
    let quant = info.quant;
    let bad = |reason| WeightError::BadShape { quant, rows, cols, reason };
    check_shape(info, rows, cols)?;
    let view = Exl2View::validate(raw).map_err(|_| bad("not a valid EXL2 tensor"))?;
    if (view.rows(), view.cols()) != (rows, cols) {
        return Err(bad("the EXL2 tensor holds another shape"));
    }
    let h = view.header;
    if view.groups() >= 1 << 24 {
        return Err(bad("too many EXL2 groups"));
    }
    let groups: Vec<_> = (0..view.groups()).map(|g| view.group(g)).collect();
    if groups.iter().any(|e| !e.n.is_multiple_of(8)) {
        return Err(bad("an EXL2 group is not a multiple of 8 inputs"));
    }
    let rows_end = 16 + rows * h.row_bytes;
    let table = round_up(rows_end, 16);
    let maxima = table + cols / 8 * 8;
    let total = round_up(maxima + groups.len() * 4, 16);
    if u32::try_from(total).is_err() {
        return Err(bad("the EXL2 image exceeds 4 GiB"));
    }
    let scale_words = h.row_bytes / 4 - groups.len().div_ceil(8);
    let mut out = vec![0u8; total];
    for (i, v) in [h.row_bytes, table, scale_words * 4, groups.len()].into_iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&(v as u32).to_le_bytes());
    }
    out[16..rows_end].copy_from_slice(&raw[h.data_off..h.data_off + rows * h.row_bytes]);
    for (g, e) in groups.iter().enumerate() {
        for j in 0..e.n / 8 {
            let at = table + (e.start / 8 + j) * 8;
            out[at..at + 4].copy_from_slice(&((e.woff * 4 + j * e.bits) as u32).to_le_bytes());
            out[at + 4..at + 8].copy_from_slice(&((g | e.bits << 24) as u32).to_le_bytes());
        }
        out[maxima + 4 * g..maxima + 4 * g + 4].copy_from_slice(&e.max_scaled().to_le_bytes());
    }
    let perm: Vec<u32> = (0..cols).map(|k| view.perm(k) as u32).collect();
    Ok(PackedWeight {
        quant,
        rows,
        cols,
        planes: Planes { layout: DeviceLayout::Image, plane1: 0, total_bytes: total as u64 },
        bytes: Cow::Owned(out),
        perm: Some(perm),
    })
}
