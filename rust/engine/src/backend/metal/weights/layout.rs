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

/// Bytes of the part of the file that holds the blocks (without permutation table).
fn block_bytes_len(info: &FormatInfo, rows: usize, cols: usize) -> usize {
    if info.row_scales {
        rows * cols
    } else {
        rows * cols / info.src_qk * info.src_block_bytes
    }
}

/// Number of bytes the tensor `quant` of `rows` x `cols` has in the GGUF file (the Q4ZP table and the FP8M row scales included).
pub fn expected_raw_len(quant: QuantType, rows: usize, cols: usize) -> Result<usize, WeightError> {
    let info = format_info(quant).ok_or(WeightError::UnsupportedFormat(quant))?;
    check_shape(&info, rows, cols)?;
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
    let blocks = rows * cols / info.qk;
    Ok(match info.repack {
        Repack::None => {
            let mut total = block_bytes_len(&info, rows, cols);
            let mut plane1 = 0;
            if info.row_scales {
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
    let expected = expected_raw_len(quant, rows, cols)?;
    if raw.len() != expected {
        return Err(WeightError::BadLength {
            quant,
            expected,
            got: raw.len(),
        });
    }
    let planes = device_layout(quant, rows, cols)?;
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
