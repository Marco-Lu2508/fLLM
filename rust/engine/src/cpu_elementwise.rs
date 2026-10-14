//! Own SIMD exponential and parallel SwiGLU; no native math ABI calls.
use rayon::prelude::*;

pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    gate.par_chunks_mut(SWIGLU_CHUNK)
        .zip(up.par_chunks(SWIGLU_CHUNK))
        .for_each(|(g, u)| swiglu_chunk(g, u));
}

/// Same arithmetic as [`swiglu`] on the calling thread. Chunks match the parallel version, so
/// SIMD tails fall at the same elements and the results are bit for bit identical. Used between
/// the fused expert passes, where the vectors are too short to be worth a fork/join.
pub(crate) fn swiglu_serial(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    gate.chunks_mut(SWIGLU_CHUNK)
        .zip(up.chunks(SWIGLU_CHUNK))
        .for_each(|(g, u)| swiglu_chunk(g, u));
}

const SWIGLU_CHUNK: usize = 1024;

fn swiglu_chunk(g: &mut [f32], u: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu_kernels::isa() != "scalar" {
        // SAFETY: `isa()` confirmed AVX2/FMA.
        unsafe {
            x86::swiglu(g, u);
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if crate::cpu_kernels::neon() {
        // SAFETY: NEON was detected.
        unsafe {
            neon::swiglu(g, u);
        }
        return;
    }
    for (g, &u) in g.iter_mut().zip(u) {
        *g = *g / (1.0 + (-*g).exp()) * u;
    }
}

pub(crate) fn exp_shifted(scores: &mut [f32], maximum: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu_kernels::avx512().is_some() {
        // SAFETY: `avx512()` confirmed AVX-512F alongside AVX2/FMA.
        return unsafe { x86::exp_shifted512(scores, maximum) };
    }
    #[cfg(target_arch = "x86_64")]
    if crate::cpu_kernels::isa() != "scalar" {
        // SAFETY: `isa()` confirmed AVX2/FMA.
        return unsafe { x86::exp_shifted(scores, maximum) };
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if crate::cpu_kernels::neon() {
        // SAFETY: NEON was detected.
        return unsafe { neon::exp_shifted(scores, maximum) };
    }
    for v in scores.iter_mut() {
        *v = (*v - maximum).max(-80.0).exp();
    }
    scores.iter().sum()
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp(v: __m256) -> __m256 {
        // Range reduction x=n*ln(2)+r with |r|<=ln(2)/2. Splitting ln(2)
        // preserves precision for large |x|; degree-seven Taylor error is
        // below f32 rounding on this interval. Normal exponent bits construct
        // 2^n without scalar exp calls. Extreme sigmoid tails are masked.
        let x = _mm256_min_ps(
            _mm256_max_ps(v, _mm256_set1_ps(-87.0)),
            _mm256_set1_ps(88.0),
        );
        let n = _mm256_round_ps(
            _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
            8,
        );
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.69314575), x);
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.0000014286068), r);
        let mut p = _mm256_set1_ps(1.0 / 5040.0);
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 720.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 120.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 24.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 6.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
        let bits = _mm256_slli_epi32(
            _mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127)),
            23,
        );
        let y = _mm256_mul_ps(p, _mm256_castsi256_ps(bits));
        let y = _mm256_andnot_ps(_mm256_cmp_ps(v, _mm256_set1_ps(-87.0), _CMP_LT_OQ), y);
        let y = _mm256_blendv_ps(
            y,
            _mm256_set1_ps(f32::INFINITY),
            _mm256_cmp_ps(v, _mm256_set1_ps(88.0), _CMP_GT_OQ),
        );
        _mm256_blendv_ps(y, v, _mm256_cmp_ps(v, v, _CMP_UNORD_Q))
    }
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn swiglu(gate: &mut [f32], up: &[f32]) {
        let end = gate.len() / 8 * 8;
        for j in (0..end).step_by(8) {
            let g = _mm256_loadu_ps(gate.as_ptr().add(j));
            let u = _mm256_loadu_ps(up.as_ptr().add(j));
            let e = exp(_mm256_sub_ps(_mm256_setzero_ps(), g));
            let y = _mm256_mul_ps(_mm256_div_ps(g, _mm256_add_ps(_mm256_set1_ps(1.0), e)), u);
            _mm256_storeu_ps(gate.as_mut_ptr().add(j), y);
        }
        for (g, &u) in gate[end..].iter_mut().zip(&up[end..]) {
            *g = *g / (1.0 + (-*g).exp()) * u;
        }
    }
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn exp_shifted(scores: &mut [f32], maximum: f32) -> f32 {
        let end = scores.len() / 8 * 8;
        let mut sum = _mm256_setzero_ps();
        for j in (0..end).step_by(8) {
            let x = _mm256_max_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(scores.as_ptr().add(j)),
                    _mm256_set1_ps(maximum),
                ),
                _mm256_set1_ps(-80.0),
            );
            let y = exp(x);
            _mm256_storeu_ps(scores.as_mut_ptr().add(j), y);
            sum = _mm256_add_ps(sum, y);
        }
        let t = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        let t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        let mut sum = _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55)));
        for s in &mut scores[end..] {
            *s = (*s - maximum).max(-80.0).exp();
            sum += *s;
        }
        sum
    }
    /// `exp` on 512-bit vectors with the operation sequence of the AVX2 `exp` (clamps, range
    /// reduction, polynomial, fused multiply-adds). Lanes are expected to agree with AVX2; this
    /// has not been verified on AVX-512 hardware.
    #[inline]
    #[target_feature(enable = "avx512f,avx2,fma")]
    unsafe fn exp512(v: __m512) -> __m512 {
        let x = _mm512_min_ps(
            _mm512_max_ps(v, _mm512_set1_ps(-87.0)),
            _mm512_set1_ps(88.0),
        );
        // Round to nearest even, exceptions suppressed: `_mm256_round_ps(.., 8)`.
        let n =
            _mm512_roundscale_ps::<8>(_mm512_mul_ps(x, _mm512_set1_ps(std::f32::consts::LOG2_E)));
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.69314575), x);
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.0000014286068), r);
        let mut p = _mm512_set1_ps(1.0 / 5040.0);
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 720.0));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 120.0));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 24.0));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0 / 6.0));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));
        let bits = _mm512_slli_epi32::<23>(_mm512_add_epi32(
            _mm512_cvttps_epi32(n),
            _mm512_set1_epi32(127),
        ));
        let y = _mm512_mul_ps(p, _mm512_castsi512_ps(bits));
        let low = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(v, _mm512_set1_ps(-87.0));
        let y = _mm512_mask_blend_ps(low, y, _mm512_setzero_ps());
        let high = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(v, _mm512_set1_ps(88.0));
        let y = _mm512_mask_blend_ps(high, y, _mm512_set1_ps(f32::INFINITY));
        let nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(v, v);
        _mm512_mask_blend_ps(nan, y, v)
    }
    /// The final partial vector runs masked, so short score rows avoid the scalar tail.
    #[target_feature(enable = "avx512f,avx2,fma")]
    pub(super) unsafe fn exp_shifted512(scores: &mut [f32], maximum: f32) -> f32 {
        let mut sum = _mm512_setzero_ps();
        for j in (0..scores.len()).step_by(16) {
            let mask = if scores.len() - j >= 16 {
                0xffff
            } else {
                (1u16 << (scores.len() - j)) - 1
            };
            let p = scores.as_mut_ptr().add(j);
            let x = _mm512_max_ps(
                _mm512_sub_ps(_mm512_maskz_loadu_ps(mask, p), _mm512_set1_ps(maximum)),
                _mm512_set1_ps(-80.0),
            );
            let y = _mm512_maskz_mov_ps(mask, exp512(x));
            _mm512_mask_storeu_ps(p, mask, y);
            sum = _mm512_add_ps(sum, y);
        }
        _mm512_reduce_add_ps(sum)
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon {
    use std::arch::aarch64::*;
    /// The operation sequence of the AVX2 `exp` (clamps, range reduction, polynomial, fused
    /// multiply-adds) on float32x4. Note that the scalar fallback uses `f32::exp` instead.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn exp(v: float32x4_t) -> float32x4_t {
        let x = vminq_f32(vmaxq_f32(v, vdupq_n_f32(-87.0)), vdupq_n_f32(88.0));
        let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)));
        let r = vfmsq_f32(x, n, vdupq_n_f32(0.69314575));
        let r = vfmsq_f32(r, n, vdupq_n_f32(0.0000014286068));
        let mut p = vdupq_n_f32(1.0 / 5040.0);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 720.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 120.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
        let bits = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)));
        let y = vmulq_f32(p, vreinterpretq_f32_s32(bits));
        let y = vbslq_f32(vcltq_f32(v, vdupq_n_f32(-87.0)), vdupq_n_f32(0.0), y);
        let y = vbslq_f32(
            vcgtq_f32(v, vdupq_n_f32(88.0)),
            vdupq_n_f32(f32::INFINITY),
            y,
        );
        // NaN lanes return the input.
        vbslq_f32(vceqq_f32(v, v), y, v)
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn swiglu(gate: &mut [f32], up: &[f32]) {
        let end = gate.len() / 4 * 4;
        for j in (0..end).step_by(4) {
            let g = vld1q_f32(gate.as_ptr().add(j));
            let u = vld1q_f32(up.as_ptr().add(j));
            let e = exp(vsubq_f32(vdupq_n_f32(0.0), g));
            let y = vmulq_f32(vdivq_f32(g, vaddq_f32(vdupq_n_f32(1.0), e)), u);
            vst1q_f32(gate.as_mut_ptr().add(j), y);
        }
        for (g, &u) in gate[end..].iter_mut().zip(&up[end..]) {
            *g = *g / (1.0 + (-*g).exp()) * u;
        }
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn exp_shifted(scores: &mut [f32], maximum: f32) -> f32 {
        let end = scores.len() / 4 * 4;
        let mut sum = vdupq_n_f32(0.0);
        for j in (0..end).step_by(4) {
            // `vmaxnm` replaces NaN by the clamp, like the scalar `max(-80.0)`.
            let x = vmaxnmq_f32(
                vsubq_f32(vld1q_f32(scores.as_ptr().add(j)), vdupq_n_f32(maximum)),
                vdupq_n_f32(-80.0),
            );
            let y = exp(x);
            vst1q_f32(scores.as_mut_ptr().add(j), y);
            sum = vaddq_f32(sum, y);
        }
        let mut sum = vaddvq_f32(sum);
        for s in &mut scores[end..] {
            *s = (*s - maximum).max(-80.0).exp();
            sum += *s;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    // The masked AVX-512 exp: every tail length of the final partial vector.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_exp_shifted_handles_every_tail_length() {
        if !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        for len in 1..=70 {
            let mut scores = (0..len)
                .map(|j| j as f32 * -1.2345 + 0.25)
                .collect::<Vec<_>>();
            let reference = scores
                .iter()
                .map(|&s| f64::from((s - 0.5).max(-80.0)).exp())
                .collect::<Vec<_>>();
            // SAFETY: AVX-512F, AVX2 and FMA detected above.
            let sum = unsafe { super::x86::exp_shifted512(&mut scores, 0.5) };
            for (&a, &b) in scores.iter().zip(&reference) {
                assert!((f64::from(a) - b).abs() <= 1e-6 * b, "len {len}: {a}/{b}");
            }
            let total = reference.iter().sum::<f64>();
            assert!((f64::from(sum) - total).abs() <= 1e-6 * total, "len {len}");
        }
    }
    #[test]
    fn serial_swiglu_is_bitwise_equal_to_parallel() {
        for n in [1usize, 7, 768, 1024, 1025, 2051] {
            let up: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.07)
                .collect();
            let gate: Vec<f32> = (0..n)
                .map(|i| ((i * 53 % 89) as f32 - 44.0) * 0.11)
                .collect();
            let (mut a, mut b) = (gate.clone(), gate);
            super::swiglu(&mut a, &up);
            super::swiglu_serial(&mut b, &up);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "n={n}"
            );
        }
    }

    #[test]
    fn swiglu_matches_scalar_across_tails_and_dynamic_range() {
        let input = (0..1031)
            .map(|i| (i as f32 - 515.0) / 4.0)
            .collect::<Vec<_>>();
        let up = (0..1031).map(|i| (i % 17) as f32 - 8.0).collect::<Vec<_>>();
        let reference = input
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect::<Vec<_>>();
        let mut actual = input;
        super::swiglu(&mut actual, &up);
        for (&a, &b) in actual.iter().zip(&reference) {
            assert!((a - b).abs() <= 1e-5 + 2e-7 * b.abs(), "{a}/{b}");
        }
        let mut scores = (0..65).map(|j| j as f32 * -1.2345).collect::<Vec<_>>();
        let reference = scores.iter().map(|v| v.exp()).collect::<Vec<_>>();
        let sum = super::exp_shifted(&mut scores, 0.0);
        assert!((sum - reference.iter().sum::<f32>()).abs() < 1e-6);
        for (&a, &b) in scores.iter().zip(&reference) {
            assert!((a - b).abs() <= 2e-7 * b.abs(), "{a}/{b}");
        }
    }
}
