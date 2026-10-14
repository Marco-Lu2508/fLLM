//! The DeltaNet state update (`cpu_delta::step`) on 512-bit vectors.
//!
//! The AVX2 kernel (`step_avx2`) handles widths that are multiples of 8 up to 32 with separate
//! multiply and add (no FMA), so the result does not depend on the vector width. This kernel does
//! the same operations on 16 lanes per vector, widths 16, 32, 48 and 64, and therefore produces
//! bit-identical state and output.
use std::arch::x86_64::*;

/// Whether `step` handles blocks of `width` columns: a multiple of 16, at most 64.
pub(crate) fn supports(width: usize) -> bool {
    width.is_multiple_of(16) && (16..=64).contains(&width)
}

/// One token for one block of `width` columns of one head: decay, delta rule update, readout.
///
/// # Safety
/// AVX-512F must be available and `supports(width)`; `s` points at the block's `hk x width`
/// state, `o` at its first output column, `v` holds at least `width` values and `k`/`q` at least
/// `hk`; no other thread touches the state.
#[allow(clippy::needless_range_loop)]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn step<const NV: usize>(
    s: *mut f32,
    (hk, width): (usize, usize),
    (k, q, v): (&[f32], &[f32], &[f32]),
    (forget, sigma): (f32, f32),
    o: *mut f32,
) {
    debug_assert!(NV * 16 == width);
    let (k, q) = (&k[..hk], &q[..hk]);
    let fb = _mm512_set1_ps(forget);
    let mut memory = [_mm512_setzero_ps(); NV];
    for (i, &ki) in k.iter().enumerate() {
        let kb = _mm512_set1_ps(ki);
        let row = s.add(i * width);
        for (u, m) in memory.iter_mut().enumerate() {
            let value = _mm512_mul_ps(_mm512_loadu_ps(row.add(16 * u)), fb);
            _mm512_storeu_ps(row.add(16 * u), value);
            *m = _mm512_add_ps(*m, _mm512_mul_ps(value, kb));
        }
    }
    let sb = _mm512_set1_ps(sigma);
    let mut delta = [_mm512_setzero_ps(); NV];
    for (u, d) in delta.iter_mut().enumerate() {
        let diff = _mm512_sub_ps(_mm512_loadu_ps(v.as_ptr().add(16 * u)), memory[u]);
        *d = _mm512_mul_ps(diff, sb);
    }
    let mut acc = [_mm512_setzero_ps(); NV];
    for (i, (&ki, &qi)) in k.iter().zip(q).enumerate() {
        let kb = _mm512_set1_ps(ki);
        let qb = _mm512_set1_ps(qi);
        let row = s.add(i * width);
        for u in 0..NV {
            let old = _mm512_loadu_ps(row.add(16 * u));
            let value = _mm512_add_ps(old, _mm512_mul_ps(kb, delta[u]));
            _mm512_storeu_ps(row.add(16 * u), value);
            acc[u] = _mm512_add_ps(acc[u], _mm512_mul_ps(value, qb));
        }
    }
    for (u, a) in acc.iter().enumerate() {
        _mm512_storeu_ps(o.add(16 * u), *a);
    }
}

/// Dispatch on the number of vectors; `false` if `width` is not supported.
///
/// # Safety
/// As `step`.
pub(crate) unsafe fn step_dispatch(
    s: *mut f32,
    dims: (usize, usize),
    vectors: (&[f32], &[f32], &[f32]),
    scalars: (f32, f32),
    o: *mut f32,
) -> bool {
    match dims.1 {
        64 => step::<4>(s, dims, vectors, scalars, o),
        48 => step::<3>(s, dims, vectors, scalars, o),
        32 => step::<2>(s, dims, vectors, scalars, o),
        16 => step::<1>(s, dims, vectors, scalars, o),
        _ => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scalar update of `cpu_delta::step_scalar`, restated so this module does not depend on
    /// a private function.
    #[allow(clippy::needless_range_loop)]
    unsafe fn scalar(
        s: *mut f32,
        (hk, width): (usize, usize),
        (k, q, v): (&[f32], &[f32], &[f32]),
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

    #[test]
    fn avx512_step_is_bit_identical_to_the_scalar_update() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        let (hk, hv) = (16usize, 64usize);
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        let state: Vec<f32> = (0..hk * 64).map(|_| rnd()).collect();
        let k: Vec<f32> = (0..hk).map(|_| rnd()).collect();
        let q: Vec<f32> = (0..hk).map(|_| rnd()).collect();
        let v: Vec<f32> = (0..hv).map(|_| rnd()).collect();
        for width in [16usize, 32, 48, 64] {
            assert!(supports(width));
            let (mut a, mut b) = (state.clone(), state.clone());
            let (mut oa, mut ob) = (vec![0.0f32; 64], vec![0.0f32; 64]);
            unsafe {
                assert!(step_dispatch(
                    a.as_mut_ptr(),
                    (hk, width),
                    (&k, &q, &v),
                    (0.9, 0.7),
                    oa.as_mut_ptr()
                ));
                scalar(
                    b.as_mut_ptr(),
                    (hk, width),
                    (&k, &q, &v),
                    (0.9, 0.7),
                    ob.as_mut_ptr(),
                );
            }
            assert_eq!(a, b, "width {width}");
            assert_eq!(oa, ob, "width {width}");
        }
        for width in [8usize, 24, 80, 5] {
            assert!(!supports(width));
        }
    }
}
