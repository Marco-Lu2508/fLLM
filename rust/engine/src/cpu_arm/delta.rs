//! NEON step of the gated DeltaNet recurrence (`cpu_delta::step`).
//!
//! One token for one block of `width` columns (a multiple of four, at most 32) of one head. Every
//! column is a lane, and each lane performs the scalar loop's operations in the scalar loop's
//! order with separate multiplies and adds (no fused multiply-add, which the scalar code does
//! not use either), so the result is bit-identical to `step_scalar`; the AVX2 version works the
//! same way.
use std::arch::aarch64::*;

/// Decay, delta-rule update and readout of one token.
///
/// # Safety
/// NEON must be available. `s` points at the block's `hk x width` state (row `i` at
/// `s + i * width`) and `o` at its first output column, with `width == 4 * NV`; no other thread
/// touches them. `k` and `q` hold at least `hk` values and `v` at least `width`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn step_neon<const NV: usize>(
    s: *mut f32,
    (hk, width): (usize, usize),
    (k, q, v): (&[f32], &[f32], &[f32]),
    (forget, sigma): (f32, f32),
    o: *mut f32,
) {
    debug_assert_eq!(width, 4 * NV);
    let (k, q, v) = (&k[..hk], &q[..hk], &v[..width]);
    let fb = vdupq_n_f32(forget);
    let mut memory = [vdupq_n_f32(0.0); NV];
    for (i, &ki) in k.iter().enumerate() {
        let kb = vdupq_n_f32(ki);
        let row = s.add(i * width);
        for (u, m) in memory.iter_mut().enumerate() {
            let value = vmulq_f32(vld1q_f32(row.add(4 * u)), fb);
            vst1q_f32(row.add(4 * u), value);
            *m = vaddq_f32(*m, vmulq_f32(value, kb));
        }
    }
    let sb = vdupq_n_f32(sigma);
    let mut delta = [vdupq_n_f32(0.0); NV];
    for (u, d) in delta.iter_mut().enumerate() {
        let diff = vsubq_f32(vld1q_f32(v.as_ptr().add(4 * u)), memory[u]);
        *d = vmulq_f32(diff, sb);
    }
    let mut acc = [vdupq_n_f32(0.0); NV];
    for (i, (&ki, &qi)) in k.iter().zip(q).enumerate() {
        let kb = vdupq_n_f32(ki);
        let qb = vdupq_n_f32(qi);
        let row = s.add(i * width);
        for u in 0..NV {
            let old = vld1q_f32(row.add(4 * u));
            let value = vaddq_f32(old, vmulq_f32(kb, delta[u]));
            vst1q_f32(row.add(4 * u), value);
            acc[u] = vaddq_f32(acc[u], vmulq_f32(value, qb));
        }
    }
    for (u, a) in acc.iter().enumerate() {
        vst1q_f32(o.add(4 * u), *a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scalar recurrence of `cpu_delta::step_scalar`, restated for the test.
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
    fn neon_step_is_bit_identical_to_the_scalar_recurrence() {
        if !crate::cpu_kernels::neon() {
            return;
        }
        let mut rng = crate::cpu_selftest::Lcg(12345);
        for hk in [1usize, 16, 33, 128] {
            let state = rng.vector(hk * 32, 0.5);
            let k = rng.vector(hk, 0.5);
            let q = rng.vector(hk, 0.5);
            let v = rng.vector(32, 0.5);
            for width in [4usize, 8, 12, 16, 20, 24, 28, 32] {
                let (mut a, mut b) = (state.clone(), state.clone());
                let (mut oa, mut ob) = (vec![0.0f32; 32], vec![0.0f32; 32]);
                // SAFETY: NEON detected; `a` and `b` hold `hk * 32 >= hk * width` values.
                unsafe {
                    macro_rules! neon {
                        ($nv:literal) => {
                            step_neon::<$nv>(
                                a.as_mut_ptr(),
                                (hk, width),
                                (&k, &q, &v),
                                (0.9, 0.7),
                                oa.as_mut_ptr(),
                            )
                        };
                    }
                    match width / 4 {
                        1 => neon!(1),
                        2 => neon!(2),
                        3 => neon!(3),
                        4 => neon!(4),
                        5 => neon!(5),
                        6 => neon!(6),
                        7 => neon!(7),
                        _ => neon!(8),
                    }
                    scalar(
                        b.as_mut_ptr(),
                        (hk, width),
                        (&k, &q, &v),
                        (0.9, 0.7),
                        ob.as_mut_ptr(),
                    );
                }
                assert_eq!(a, b, "state, hk {hk} width {width}");
                assert_eq!(oa, ob, "output, hk {hk} width {width}");
            }
        }
    }
}
