//! Packs GGUF 4-bit weights (Q4_0, Q4_1, Q4_K) into the tile stream that the dequantizing NPU program reads
//! (`rust/engine/xdna/kernel_src/fllm_dq.cc`, `whole_array_dq.py`, host reference `dq_pack.py`).
//!
//! A weight `W[n][k]` is cut into tiles of 64 columns of `n` and 64 values of `k`. One tile is a payload of
//! [`PAYLOAD`] bytes: bf16 scales `[2 groups][64]`, bf16 minima `[2][64]`, then the 4-bit codes ordered
//! `[n / 16][k][n % 16]` (element `2i` in the low nibble). The NPU computes `min + scale * code` per weight in the compute tile.
//! The payloads of one dispatch are stored `[k step][n tile]`.
//!
//! The group size is 32 along `k`, which is the block of Q4_0 and Q4_1 and the sub-block of Q4_K (whose scale is `d * sc` and
//! whose minimum is `-dmin * m`; Q4_0 has the minimum `-8 d`).

use half::{bf16, f16};
use rayon::prelude::*;

pub const TILE_K: usize = 64;
pub const TILE_N: usize = 64;
pub const GROUP: usize = 32;
/// Bytes of one packed (64 x 64) tile.
pub const PAYLOAD: usize = 4 * (TILE_K / GROUP) * TILE_N + TILE_K * TILE_N / 2;
/// Tiles along `n` in one dispatch (1024 weight rows).
pub const NTILES: usize = 16;

/// GGUF types this packer reads.
pub fn supported(kind: u32) -> bool {
    matches!(kind, 2 | 3 | 12)
}

fn h(b: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
}

/// Decodes one row of `cols` weights (`cols` a multiple of the block size) into per-group `(scale, min)` and the codes 0..15.
pub fn decode_row(kind: u32, row: &[u8], cols: usize, scales: &mut [f32], mins: &mut [f32], codes: &mut [u8]) {
    match kind {
        2 => {
            for (b, blk) in row.as_chunks::<18>().0.iter().enumerate() {
                let d = h(blk);
                scales[b] = d;
                mins[b] = -8.0 * d;
                for l in 0..16 {
                    codes[b * 32 + l] = blk[2 + l] & 15;
                    codes[b * 32 + 16 + l] = blk[2 + l] >> 4;
                }
            }
        }
        3 => {
            for (b, blk) in row.as_chunks::<20>().0.iter().enumerate() {
                scales[b] = h(blk);
                mins[b] = h(&blk[2..]);
                for l in 0..16 {
                    codes[b * 32 + l] = blk[4 + l] & 15;
                    codes[b * 32 + 16 + l] = blk[4 + l] >> 4;
                }
            }
        }
        12 => {
            for (b, blk) in row.as_chunks::<144>().0.iter().enumerate() {
                let d = h(blk);
                let dmin = h(&blk[2..]);
                let sc = &blk[4..16];
                let qs = &blk[16..144];
                for j in 0..4 {
                    for half in 0..2 {
                        let i = 2 * j + half;
                        let (s, m) = if i < 4 {
                            (sc[i] & 63, sc[i + 4] & 63)
                        } else {
                            ((sc[i + 4] & 15) | ((sc[i - 4] >> 6) << 4), (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4))
                        };
                        let g = b * 8 + i;
                        scales[g] = d * s as f32;
                        mins[g] = -dmin * m as f32;
                        for l in 0..32 {
                            let q = qs[32 * j + l];
                            codes[g * 32 + l] = if half == 0 { q & 15 } else { q >> 4 };
                        }
                    }
                }
            }
        }
        _ => unreachable!("dqpack::supported() says which formats are handled"),
    }
    let _ = cols;
}

/// Bytes of one row of `cols` values.
pub fn row_bytes(kind: u32, cols: usize) -> Option<usize> {
    match kind {
        2 if cols.is_multiple_of(32) => Some(cols / 32 * 18),
        3 if cols.is_multiple_of(32) => Some(cols / 32 * 20),
        12 if cols.is_multiple_of(256) => Some(cols / 256 * 144),
        _ => None,
    }
}

/// Packs the weight rows `n0..n0 + 1024` (rows past `rows` are zero), depth `k0..k0 + kt` (columns past `cols` are zero) into `dst`
/// (`kt / 64 * 16 * PAYLOAD` bytes, `[k step][n tile]`).
#[allow(clippy::too_many_arguments)]
pub fn pack_group(kind: u32, raw: &[u8], rows: usize, cols: usize, n0: usize, k0: usize, kt: usize, dst: &mut [u8]) {
    let rb = row_bytes(kind, cols).expect("checked by the caller");
    let steps = kt / TILE_K;
    assert_eq!(dst.len(), steps * NTILES * PAYLOAD);
    let groups = cols / GROUP;
    // decode the 1024 rows once: codes [row][k], scales/mins [row][group]
    let nrows = NTILES * TILE_N;
    let mut codes = vec![0u8; nrows * cols];
    let mut scales = vec![0f32; nrows * groups];
    let mut mins = vec![0f32; nrows * groups];
    codes
        .par_chunks_mut(cols)
        .zip(scales.par_chunks_mut(groups))
        .zip(mins.par_chunks_mut(groups))
        .enumerate()
        .for_each(|(r, ((c, s), m))| {
            if n0 + r < rows {
                decode_row(kind, &raw[(n0 + r) * rb..(n0 + r + 1) * rb], cols, s, m, c);
            }
        });
    // payloads: one per (k step, n tile)
    dst.par_chunks_mut(PAYLOAD).enumerate().for_each(|(idx, pay)| {
        let (step, nt) = (idx / NTILES, idx % NTILES);
        let (kk0, nn0) = (k0 + step * TILE_K, nt * TILE_N);
        let (sc_area, rest) = pay.split_at_mut(2 * TILE_N * 2);
        let (mn_area, nib) = rest.split_at_mut(2 * TILE_N * 2);
        for g in 0..TILE_K / GROUP {
            let grp = (kk0 + g * GROUP) / GROUP;
            for n in 0..TILE_N {
                let (s, m) = if kk0 + g * GROUP < cols { (scales[(nn0 + n) * groups + grp], mins[(nn0 + n) * groups + grp]) } else { (0.0, 0.0) };
                let o = (g * TILE_N + n) * 2;
                sc_area[o..o + 2].copy_from_slice(&bf16::from_f32(s).to_bits().to_le_bytes());
                mn_area[o..o + 2].copy_from_slice(&bf16::from_f32(m).to_bits().to_le_bytes());
            }
        }
        // codes [n/16][k][n%16]: element index e = (a * 64 + kk) * 16 + c, two elements per byte
        let code = |a: usize, kk: usize, c: usize| -> u8 {
            let k = kk0 + kk;
            if k < cols {
                codes[(nn0 + a * 16 + c) * cols + k]
            } else {
                0
            }
        };
        for a in 0..TILE_N / 16 {
            for kk in 0..TILE_K {
                let base = (a * TILE_K + kk) * 8;
                for c in 0..8 {
                    nib[base + c] = code(a, kk, 2 * c) | (code(a, kk, 2 * c + 1) << 4);
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_payload_size_matches_the_kernel() {
        assert_eq!(PAYLOAD, 2560);
    }

    /// Q4_0: block of one scale and 16 bytes, element l in the low nibble of byte l, element 16 + l in the high one.
    #[test]
    fn q4_0_rows_decode_to_scale_min_and_codes() {
        let mut blk = vec![0u8; 18];
        blk[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        for l in 0..16 {
            blk[2 + l] = (l as u8) | (((15 - l) as u8) << 4);
        }
        let (mut s, mut m, mut c) = (vec![0f32; 1], vec![0f32; 1], vec![0u8; 32]);
        decode_row(2, &blk, 32, &mut s, &mut m, &mut c);
        assert_eq!((s[0], m[0]), (0.5, -4.0));
        assert_eq!(c[3], 3);
        assert_eq!(c[16 + 3], 12);
    }

    /// Each decoded weight must equal the generic dequantizer's value (scale * code + min).
    #[test]
    fn decoded_groups_reproduce_the_generic_dequantizer() {
        use crate::cpu_selftest::{random_weights, Lcg};
        for kind in [2u32, 3, 12] {
            let cols = 512;
            let mut rng = Lcg(u64::from(kind) * 7 + 1);
            let raw = random_weights(kind, 3, cols, &mut rng);
            let want = crate::compute::dequantize(&raw, kind, 3 * cols).unwrap();
            let rb = row_bytes(kind, cols).unwrap();
            for r in 0..3 {
                let (mut s, mut m, mut c) = (vec![0f32; cols / 32], vec![0f32; cols / 32], vec![0u8; cols]);
                decode_row(kind, &raw[r * rb..(r + 1) * rb], cols, &mut s, &mut m, &mut c);
                for k in 0..cols {
                    let got = s[k / 32] * c[k] as f32 + m[k / 32];
                    let w = want[r * cols + k];
                    assert!((got - w).abs() <= 1e-4 * w.abs().max(s[k / 32].abs() * 4.0) + 1e-6, "kind {kind} row {r} k {k}: {got} vs {w}");
                }
            }
        }
    }

    #[test]
    fn a_packed_group_has_the_expected_size_and_pads_with_zero() {
        use crate::cpu_selftest::{random_weights, Lcg};
        let (rows, cols) = (70usize, 256usize);
        let mut rng = Lcg(5);
        let raw = random_weights(12, rows, cols, &mut rng);
        let mut dst = vec![0xAAu8; 4 * NTILES * PAYLOAD];
        pack_group(12, &raw, rows, cols, 0, 0, 256, &mut dst);
        // a tile entirely past the 70 rows is all zero (scales, minima and codes)
        let last = &dst[(NTILES - 1) * PAYLOAD..NTILES * PAYLOAD];
        assert!(last.iter().all(|b| *b == 0));
        // the first tile is not
        assert!(dst[..PAYLOAD].iter().any(|b| *b != 0));
    }
}
