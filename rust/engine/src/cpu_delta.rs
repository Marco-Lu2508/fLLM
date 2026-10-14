//! Gated DeltaNet recurrence for a run of consecutive tokens of one sequence.
//!
//! The recurrent state (`k_dim x v_dim` per value head) stays hot while one task walks all tokens
//! of the run for a block of its columns; the arithmetic order per element is the same as the
//! token-at-a-time reference, so results are bit-identical to it.
#![allow(clippy::needless_range_loop)]
use rayon::prelude::*;

/// Raw output pointer that tasks write to at disjoint offsets.
#[derive(Clone, Copy)]
struct Shared(*mut f32);
// SAFETY: every task writes only elements no other task touches (documented at each use).
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}
impl Shared {
    fn ptr(self) -> *mut f32 {
        self.0
    }
}

/// Static shape and weights of a DeltaNet layer.
pub(crate) struct Layer<'a> {
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
    pub beta_scale: f32,
    pub eps: f32,
    pub tiled_heads: bool,
    /// `conv_dim x kernel`
    pub conv: &'a [f32],
    pub dt_bias: &'a [f32],
    pub decay: &'a [f32],
    pub norm: &'a [f32],
}

/// Convolution channels one task handles at a time.
const CONV_BLOCK: usize = 128;

/// In-place SiLU, `x / (1 + exp(-x))` (the libm exponential keeps results identical to the
/// token-at-a-time reference).
fn silu(values: &mut [f32]) {
    for v in values {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// Columns of the value dimension one task handles at a time.
const CHUNK: usize = 32;

/// Run `n` consecutive tokens. `qkv` is `n x conv_dim`, `z` `n x value_dim`, `beta`/`alpha`
/// `n x v_heads`; `out` (`n x value_dim`) receives the gated, normalised head outputs.
pub(crate) fn run(
    l: &Layer<'_>,
    history: &mut [f32],
    rec: &mut [f32],
    (qkv, z, beta, alpha): (&[f32], &[f32], &[f32], &[f32]),
    out: &mut [f32],
    n: usize,
) {
    let (kh, vh, hk, hv, ksz) = (l.k_heads, l.v_heads, l.k_dim, l.v_dim, l.conv_kernel);
    let key_dim = kh * hk;
    let value_dim = vh * hv;
    let conv_dim = 2 * key_dim + value_dim;
    let ratio = vh / kh.max(1);
    let parallel = n >= 4;

    // causal convolution + SiLU, channel blocks in parallel
    let mut mixed = vec![0.0f32; n * conv_dim];
    {
        let dst = Shared(mixed.as_mut_ptr());
        let hist = Shared(history.as_mut_ptr());
        let block = |b: usize| {
            let (cs, ce) = (b * CONV_BLOCK, ((b + 1) * CONV_BLOCK).min(conv_dim));
            let cw = ce - cs;
            // weights and history of the block, transposed to `[tap][channel]`
            let mut wt = vec![0.0f32; ksz * cw];
            let mut ht = vec![0.0f32; (ksz - 1) * cw];
            for c in 0..cw {
                for k in 0..ksz {
                    wt[k * cw + c] = l.conv[(cs + c) * ksz + k];
                }
                for k in 0..ksz - 1 {
                    // SAFETY: this block owns the history of its channels.
                    ht[k * cw + c] = unsafe { *hist.ptr().add((cs + c) * (ksz - 1) + k) };
                }
            }
            for t in 0..n {
                // SAFETY: this block owns columns cs..ce of every row.
                let row = unsafe { std::slice::from_raw_parts_mut(dst.ptr().add(t * conv_dim + cs), cw) };
                let current = &qkv[t * conv_dim + cs..t * conv_dim + ce];
                for c in 0..cw {
                    row[c] = wt[(ksz - 1) * cw + c] * current[c];
                }
                for k in 0..ksz - 1 {
                    // tap k meets the input `ksz - 1 - k` tokens back
                    let back = t + k;
                    if back >= ksz - 1 {
                        let at = (back - (ksz - 1)) * conv_dim + cs;
                        let previous = &qkv[at..at + cw];
                        for c in 0..cw {
                            row[c] += wt[k * cw + c] * previous[c];
                        }
                    } else {
                        for c in 0..cw {
                            row[c] += wt[k * cw + c] * ht[back * cw + c];
                        }
                    }
                }
                silu(row);
            }
            for c in 0..cw {
                for k in 0..ksz - 1 {
                    let back = n + k;
                    let value = if back >= ksz - 1 {
                        qkv[(back - (ksz - 1)) * conv_dim + cs + c]
                    } else {
                        ht[back * cw + c]
                    };
                    // SAFETY: as above.
                    unsafe { *hist.ptr().add((cs + c) * (ksz - 1) + k) = value };
                }
            }
        };
        let blocks = conv_dim.div_ceil(CONV_BLOCK);
        if parallel {
            (0..blocks).into_par_iter().for_each(block);
        } else {
            (0..blocks).for_each(block);
        }
    }

    // normalised queries/keys per token and key head, decay and write strength per value head
    let query_scale = (hk as f32).powf(-0.5);
    let mut qn = vec![0.0f32; n * key_dim];
    let mut kn = vec![0.0f32; n * key_dim];
    let mut forget = vec![0.0f32; n * vh];
    let mut sigma = vec![0.0f32; n * vh];
    let prep = |t: usize, qn: &mut [f32], kn: &mut [f32], forget: &mut [f32], sigma: &mut [f32]| {
        let row = &mixed[t * conv_dim..(t + 1) * conv_dim];
        let unit = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
        for h in 0..kh {
            let q = &row[h * hk..(h + 1) * hk];
            let k = &row[key_dim + h * hk..key_dim + (h + 1) * hk];
            let (nq, nk) = (unit(q), unit(k));
            for i in 0..hk {
                qn[h * hk + i] = q[i] * nq * query_scale;
                kn[h * hk + i] = k[i] * nk;
            }
        }
        for j in 0..vh {
            sigma[j] = l.beta_scale / (1.0 + (-beta[t * vh + j]).exp());
            let raw = alpha[t * vh + j] + l.dt_bias[j];
            let softplus = if raw > 20.0 { raw } else { raw.exp().ln_1p() };
            forget[j] = (l.decay[j] * softplus).exp();
        }
    };
    if parallel {
        qn.par_chunks_mut(key_dim)
            .zip(kn.par_chunks_mut(key_dim))
            .zip(forget.par_chunks_mut(vh).zip(sigma.par_chunks_mut(vh)))
            .enumerate()
            .for_each(|(t, ((q, k), (f, s)))| prep(t, q, k, f, s));
    } else {
        for t in 0..n {
            prep(
                t,
                &mut qn[t * key_dim..(t + 1) * key_dim],
                &mut kn[t * key_dim..(t + 1) * key_dim],
                &mut forget[t * vh..(t + 1) * vh],
                &mut sigma[t * vh..(t + 1) * vh],
            );
        }
    }

    // recurrence: one task per (value head, column block), tokens in order
    let chunks = hv.div_ceil(CHUNK);
    {
        let state = Shared(rec.as_mut_ptr());
        let dst = Shared(out.as_mut_ptr());
        let task = |index: usize| {
            let (j, c0) = (index / chunks, (index % chunks) * CHUNK);
            let width = CHUNK.min(hv - c0);
            let source = if l.tiled_heads {
                j % kh.max(1)
            } else {
                j / ratio.max(1)
            };
            // SAFETY: the task owns columns c0..c0+width of head j in the state and the output.
            unsafe {
                // the state of a column block is stored contiguously, `hk` rows of `width` values
                let s = state.ptr().add(j * hk * hv + c0 * hk);
                for t in 0..n {
                    let k = &kn[t * key_dim + source * hk..t * key_dim + (source + 1) * hk];
                    let q = &qn[t * key_dim + source * hk..t * key_dim + (source + 1) * hk];
                    let v = &mixed[t * conv_dim + 2 * key_dim + j * hv + c0..];
                    let o = dst.ptr().add(t * value_dim + j * hv + c0);
                    let scalars = (forget[t * vh + j], sigma[t * vh + j]);
                    step(s, (hk, width), (k, q, v), scalars, o);
                }
            }
        };
        let tasks = vh * chunks;
        if parallel || n * vh * hk * hv > 1 << 16 {
            (0..tasks).into_par_iter().for_each(task);
        } else {
            (0..tasks).for_each(task);
        }
    }

    // gated RMS norm: norm(o) * weight * silu(z)
    let finish = |t: usize, o: &mut [f32]| {
        let mut gates = z[t * value_dim..(t + 1) * value_dim].to_vec();
        silu(&mut gates);
        for j in 0..vh {
            let o = &mut o[j * hv..(j + 1) * hv];
            let inv = (o.iter().map(|x| x * x).sum::<f32>() / hv as f32 + l.eps)
                .sqrt()
                .recip();
            for (col, o) in o.iter_mut().enumerate() {
                *o = *o * inv * l.norm[col] * gates[j * hv + col];
            }
        }
    };
    if parallel {
        out.par_chunks_mut(value_dim)
            .enumerate()
            .for_each(|(t, o)| finish(t, o));
    } else {
        for (t, o) in out.chunks_mut(value_dim).enumerate() {
            finish(t, o);
        }
    }
}

type Dims = (usize, usize);
type Vectors<'a> = (&'a [f32], &'a [f32], &'a [f32]);

/// One token for one block of `width` columns of one head: decay, delta rule update, readout.
///
/// # Safety
/// `s` points at the block's `hk x width` state, `o` at its first output column; no other thread
/// touches these. `v` holds at least `width` values.
unsafe fn step(
    s: *mut f32,
    dims: Dims,
    vectors: Vectors<'_>,
    scalars: (f32, f32),
    o: *mut f32,
) {
    // 512-bit state update for widths of 16..=64 (the `avx512` feature); bit-identical to AVX2.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if crate::cpu_avx512::delta_step(s, dims, vectors, scalars, o) {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dims.1.is_multiple_of(8) && crate::cpu_kernels::isa() != "scalar" {
        match dims.1 / 8 {
            4 => return step_avx2::<4>(s, dims, vectors, scalars, o),
            3 => return step_avx2::<3>(s, dims, vectors, scalars, o),
            2 => return step_avx2::<2>(s, dims, vectors, scalars, o),
            1 => return step_avx2::<1>(s, dims, vectors, scalars, o),
            _ => {}
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if dims.1.is_multiple_of(4) && crate::cpu_kernels::neon() {
        use crate::cpu_arm::delta::step_neon;
        match dims.1 / 4 {
            8 => return step_neon::<8>(s, dims, vectors, scalars, o),
            7 => return step_neon::<7>(s, dims, vectors, scalars, o),
            6 => return step_neon::<6>(s, dims, vectors, scalars, o),
            5 => return step_neon::<5>(s, dims, vectors, scalars, o),
            4 => return step_neon::<4>(s, dims, vectors, scalars, o),
            3 => return step_neon::<3>(s, dims, vectors, scalars, o),
            2 => return step_neon::<2>(s, dims, vectors, scalars, o),
            1 => return step_neon::<1>(s, dims, vectors, scalars, o),
            _ => {}
        }
    }
    step_scalar(s, dims, vectors, scalars, o);
}

unsafe fn step_scalar(
    s: *mut f32,
    (hk, width): Dims,
    (k, q, v): Vectors<'_>,
    (forget, sigma): (f32, f32),
    o: *mut f32,
) {
    for col in 0..width {
        let at = |i: usize| s.add(i * width + col);
        let mut memory = 0.0f32;
        for (i, &ki) in k.iter().enumerate().take(hk) {
            *at(i) *= forget;
            memory += *at(i) * ki;
        }
        let delta = (v[col] - memory) * sigma;
        let mut acc = 0.0f32;
        for i in 0..hk {
            *at(i) += k[i] * delta;
            acc += *at(i) * q[i];
        }
        *o.add(col) = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn step_avx2<const NV: usize>(
    s: *mut f32,
    (hk, width): Dims,
    (k, q, v): Vectors<'_>,
    (forget, sigma): (f32, f32),
    o: *mut f32,
) {
    use std::arch::x86_64::*;
    let (k, q) = (&k[..hk], &q[..hk]);
    let fb = _mm256_set1_ps(forget);
    let mut memory = [_mm256_setzero_ps(); NV];
    for (i, &ki) in k.iter().enumerate() {
        let kb = _mm256_set1_ps(ki);
        let row = s.add(i * width);
        for (u, m) in memory.iter_mut().enumerate() {
            let value = _mm256_mul_ps(_mm256_loadu_ps(row.add(8 * u)), fb);
            _mm256_storeu_ps(row.add(8 * u), value);
            *m = _mm256_add_ps(*m, _mm256_mul_ps(value, kb));
        }
    }
    let sb = _mm256_set1_ps(sigma);
    let mut delta = [_mm256_setzero_ps(); NV];
    for (u, d) in delta.iter_mut().enumerate() {
        let diff = _mm256_sub_ps(_mm256_loadu_ps(v.as_ptr().add(8 * u)), memory[u]);
        *d = _mm256_mul_ps(diff, sb);
    }
    let mut acc = [_mm256_setzero_ps(); NV];
    for (i, (&ki, &qi)) in k.iter().zip(q).enumerate() {
        let kb = _mm256_set1_ps(ki);
        let qb = _mm256_set1_ps(qi);
        let row = s.add(i * width);
        for u in 0..NV {
            let old = _mm256_loadu_ps(row.add(8 * u));
            let value = _mm256_add_ps(old, _mm256_mul_ps(kb, delta[u]));
            _mm256_storeu_ps(row.add(8 * u), value);
            acc[u] = _mm256_add_ps(acc[u], _mm256_mul_ps(value, qb));
        }
    }
    for (u, a) in acc.iter().enumerate() {
        _mm256_storeu_ps(o.add(8 * u), *a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_step_matches_scalar() {
        let (hk, hv) = (16usize, 64usize);
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        let state: Vec<f32> = (0..hk * 32).map(|_| rnd()).collect();
        let k: Vec<f32> = (0..hk).map(|_| rnd()).collect();
        let q: Vec<f32> = (0..hk).map(|_| rnd()).collect();
        let v: Vec<f32> = (0..hv).map(|_| rnd()).collect();
        for width in [8usize, 16, 24, 32, 5, 4, 12, 20, 28] {
            let (mut a, mut b) = (state.clone(), state.clone());
            let (mut oa, mut ob) = (vec![0.0f32; 32], vec![0.0f32; 32]);
            unsafe {
                step(a.as_mut_ptr(), (hk, width), (&k, &q, &v), (0.9, 0.7), oa.as_mut_ptr());
                step_scalar(b.as_mut_ptr(), (hk, width), (&k, &q, &v), (0.9, 0.7), ob.as_mut_ptr());
            }
            assert_eq!(a, b);
            assert_eq!(oa, ob);
        }
    }
}
