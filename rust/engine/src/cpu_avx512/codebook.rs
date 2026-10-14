//! IQ4_NL and MXFP4 projections on wide integer units.
//!
//! The AVX2 codebook kernel (`cpu_quantized`) needs `vpsignb` + `vpmaddubsw` for the signed
//! codebook values and runs on every CPU. These kernels use the dot-product instructions instead:
//!
//! - 512-bit AVX-512 VNNI: the 16-entry table holds `value + 128` (unsigned), `vpdpbusd` adds the
//!   product with the activations and the activation records' `init` word carries the bias
//!   (`-128 * sum(x)`, spread over the lanes of each 16-value half exactly like the Q8_0 tiles).
//! - 512-bit AVX-512 BW without VNNI: signed table and the sign-transfer `dot512_signed`.
//! - 256-bit AVX-VNNI (VEX `vpdpbusd`, Alder Lake and later without AVX-512): biased as above.
//! - 256-bit AVX-VNNI-INT8 (`vpdpbssd`, Arrow Lake, Lunar Lake): signed table, no bias at all.
//!
//! All variants read the same 18-byte (IQ4_NL, fp16 scale) or 17-byte (MXFP4, E8M0 scale) blocks
//! and the Q8_0 activation records as `codebook_avx2`, and produce the same integer dots; only the
//! float summation order differs. Two consecutive blocks of a row share one 512-bit vector:
//! 128-bit lanes `[a low nibbles, b low nibbles, a high nibbles, b high nibbles]` against the
//! activation records arranged the same way.
use super::amx_decode::{e8m0_half, MXFP4_VALUES};
use crate::cpu_kernels::x86::{dot512_signed, half_to_float, join256};
use crate::cpu_kquant::IQ4_VALUES;
use crate::cpu_quantized::X;
use crate::cpu_schedule::Rows;
use std::arch::x86_64::*;

/// Bytes per block and header bytes before the nibbles: IQ4_NL (kind 20) and MXFP4 (39).
const fn block_layout(kind: u32) -> (usize, usize) {
    if kind == 20 {
        (18, 2)
    } else {
        (17, 1)
    }
}

/// The 16-entry codebook of `kind`, optionally biased to unsigned (`value + 128`).
fn table_bytes(kind: u32, biased: bool) -> [u8; 16] {
    let values = if kind == 20 { IQ4_VALUES } else { MXFP4_VALUES };
    values.map(|v| {
        if biased {
            (i32::from(v) + 128) as u8
        } else {
            v as u8
        }
    })
}

/// Scale of the block at `base`.
#[inline]
#[target_feature(enable = "f16c")]
unsafe fn block_scale(kind: u32, base: *const u8) -> f32 {
    if kind == 20 {
        half_to_float(std::ptr::read_unaligned(base.cast::<u16>()))
    } else {
        e8m0_half(*base)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: __m256) -> f32 {
    let t = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let t = _mm_add_ps(t, _mm_movehl_ps(t, t));
    _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55)))
}

macro_rules! dispatch_rows {
    ($body:ident, $kind:expr, $rows:expr, $max_rows:literal, $w:expr, $first:expr, $x:expr, $nb:expr, $batch:expr, $out:expr) => {{
        let (w, first, x, nb, batch, out) = ($w, $first, $x, $nb, $batch, $out);
        macro_rules! by_rows {
            ($kind_lit:literal) => {{
                let mut token = 0;
                while token < batch {
                    let tokens = if $rows > 4 { 1 } else { (batch - token).min(2) };
                    match ($rows, tokens) {
                        (1, 1) => $body::<$kind_lit, 1, 1>(w, first, x, nb, token, out),
                        (2, 1) => $body::<$kind_lit, 2, 1>(w, first, x, nb, token, out),
                        (3, 1) => $body::<$kind_lit, 3, 1>(w, first, x, nb, token, out),
                        (4, 1) => $body::<$kind_lit, 4, 1>(w, first, x, nb, token, out),
                        (5, 1) if $max_rows >= 8 => {
                            $body::<$kind_lit, 5, 1>(w, first, x, nb, token, out)
                        }
                        (6, 1) if $max_rows >= 8 => {
                            $body::<$kind_lit, 6, 1>(w, first, x, nb, token, out)
                        }
                        (7, 1) if $max_rows >= 8 => {
                            $body::<$kind_lit, 7, 1>(w, first, x, nb, token, out)
                        }
                        (8, 1) if $max_rows >= 8 => {
                            $body::<$kind_lit, 8, 1>(w, first, x, nb, token, out)
                        }
                        (1, _) => $body::<$kind_lit, 1, 2>(w, first, x, nb, token, out),
                        (2, _) => $body::<$kind_lit, 2, 2>(w, first, x, nb, token, out),
                        (3, _) => $body::<$kind_lit, 3, 2>(w, first, x, nb, token, out),
                        (4, _) => $body::<$kind_lit, 4, 2>(w, first, x, nb, token, out),
                        _ => unreachable!("panel has 1 to {} rows", $max_rows),
                    }
                    token += tokens;
                }
            }};
        }
        match $kind {
            20 => by_rows!(20),
            39 => by_rows!(39),
            _ => unreachable!("not a codebook format"),
        }
    }};
}

// 512-bit kernels. `$biased` selects the unsigned table + `vpdpbusd` (VNNI) over the signed table
// + sign-transfer dot (BW).
macro_rules! codebook512 {
    ($name:ident, $body:ident, $features:literal, $biased:literal) => {
        /// `rows` (1..=8) consecutive IQ4_NL / MXFP4 rows from `first` for every token; panels of
        /// more than four rows run one token at a time.
        ///
        /// # Safety
        /// The CPU must support the enabled features; `w` holds `first + rows` complete rows of
        /// `nb` blocks of `kind` and `x` holds `batch * nb` Q8_0 activation records.
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $features)]
        pub(crate) unsafe fn $name(
            w: &[u8],
            kind: u32,
            first: usize,
            rows: usize,
            x: &[X],
            nb: usize,
            batch: usize,
            out: &mut Rows<'_>,
        ) {
            dispatch_rows!($body, kind, rows, 8, w, first, x, nb, batch, out);
        }

        #[inline]
        #[target_feature(enable = $features)]
        unsafe fn $body<const KIND: u32, const R: usize, const T: usize>(
            w: &[u8],
            first: usize,
            x: &[X],
            nb: usize,
            token: usize,
            out: &mut Rows<'_>,
        ) {
            let (width, header) = block_layout(KIND);
            let stride = nb * width;
            assert!(w.len() >= (first + R) * stride);
            let table =
                _mm512_broadcast_i32x4(_mm_loadu_si128(table_bytes(KIND, $biased).as_ptr().cast()));
            let mask = _mm256_set1_epi8(15);
            let mut acc = [[_mm512_setzero_ps(); T]; R];
            let mut kb = 0;
            while kb < nb {
                let pair = kb + 1 < nb;
                // Activations of the block pair in the weight layout, shared by every row.
                let mut qx = [_mm512_setzero_si512(); T];
                let mut init = [_mm512_setzero_si512(); T];
                let mut dx = [(0.0f32, 0.0f32); T];
                for t in 0..T {
                    let xa = &x[(token + t) * nb + kb];
                    let va = _mm256_load_si256(xa.q.as_ptr().cast());
                    let (vb, ib, db) = if pair {
                        let xb = &x[(token + t) * nb + kb + 1];
                        (_mm256_load_si256(xb.q.as_ptr().cast()), xb.init, xb.d)
                    } else {
                        // An odd final block runs with zero upper activations, bias and scale.
                        (_mm256_setzero_si256(), [0, 0], 0.0)
                    };
                    let joined = join256(va, vb);
                    // lanes [a.lo16, a.hi16, b.lo16, b.hi16] -> [a.lo16, b.lo16, a.hi16, b.hi16]
                    qx[t] = _mm512_shuffle_i64x2::<0xD8>(joined, joined);
                    init[t] = _mm512_setr_epi32(
                        xa.init[0], xa.init[0], xa.init[0], xa.init[0], ib[0], ib[0], ib[0], ib[0],
                        xa.init[1], xa.init[1], xa.init[1], xa.init[1], ib[1], ib[1], ib[1], ib[1],
                    );
                    dx[t] = (xa.d, db);
                }
                for r in 0..R {
                    let base = w.as_ptr().add((first + r) * stride + kb * width);
                    let pa = _mm_loadu_si128(base.add(header).cast());
                    let (pb, sb) = if pair {
                        (
                            _mm_loadu_si128(base.add(width + header).cast()),
                            block_scale(KIND, base.add(width)),
                        )
                    } else {
                        (pa, 0.0)
                    };
                    let sa = block_scale(KIND, base);
                    let both = _mm256_inserti128_si256(_mm256_castsi128_si256(pa), pb, 1);
                    let low = _mm256_and_si256(both, mask);
                    let high = _mm256_and_si256(_mm256_srli_epi16(both, 4), mask);
                    let index = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(low), high);
                    let weights = _mm512_shuffle_epi8(table, index);
                    for t in 0..T {
                        let dot = if $biased {
                            _mm512_dpbusd_epi32(init[t], weights, qx[t])
                        } else {
                            dot512_signed(weights, qx[t])
                        };
                        // dword lanes 0-3 and 8-11 belong to block a, 4-7 and 12-15 to block b
                        let scale = _mm512_mask_blend_ps(
                            0xF0F0,
                            _mm512_set1_ps(sa * dx[t].0),
                            _mm512_set1_ps(sb * dx[t].1),
                        );
                        acc[r][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), scale, acc[r][t]);
                    }
                }
                kb += 2;
            }
            for t in 0..T {
                for (r, y) in out.row(token + t).iter_mut().take(R).enumerate() {
                    *y = _mm512_reduce_add_ps(acc[r][t]);
                }
            }
        }
    };
}

codebook512!(
    codebook512_vnni,
    codebook512_vnni_body,
    "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni",
    true
);
codebook512!(
    codebook512_bw,
    codebook512_bw_body,
    "avx2,fma,f16c,avx512f,avx512bw,avx512vl",
    false
);

// 256-bit kernels, one block per step. `$biased`: unsigned table + `vpdpbusd` (AVX-VNNI);
// otherwise signed table + `vpdpbssd` (AVX-VNNI-INT8).
macro_rules! codebook256 {
    ($name:ident, $body:ident, $features:literal, $biased:literal) => {
        /// `rows` (1..=4) consecutive IQ4_NL / MXFP4 rows from `first` for every token.
        ///
        /// # Safety
        /// The CPU must support the enabled features; `w` holds `first + rows` complete rows of
        /// `nb` blocks of `kind` and `x` holds `batch * nb` Q8_0 activation records.
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = $features)]
        pub(crate) unsafe fn $name(
            w: &[u8],
            kind: u32,
            first: usize,
            rows: usize,
            x: &[X],
            nb: usize,
            batch: usize,
            out: &mut Rows<'_>,
        ) {
            dispatch_rows!($body, kind, rows, 4, w, first, x, nb, batch, out);
        }

        #[inline]
        #[target_feature(enable = $features)]
        unsafe fn $body<const KIND: u32, const R: usize, const T: usize>(
            w: &[u8],
            first: usize,
            x: &[X],
            nb: usize,
            token: usize,
            out: &mut Rows<'_>,
        ) {
            let (width, header) = block_layout(KIND);
            let stride = nb * width;
            assert!(w.len() >= (first + R) * stride);
            let table = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                table_bytes(KIND, $biased).as_ptr().cast(),
            ));
            let mask = _mm_set1_epi8(15);
            let mut acc = [[_mm256_setzero_ps(); T]; R];
            for kb in 0..nb {
                let mut xs = [&x[token * nb + kb]; T];
                let mut qx = [_mm256_setzero_si256(); T];
                let mut init = [_mm256_setzero_si256(); T];
                for t in 0..T {
                    xs[t] = &x[(token + t) * nb + kb];
                    qx[t] = _mm256_load_si256(xs[t].q.as_ptr().cast());
                    init[t] = _mm256_setr_epi32(
                        xs[t].init[0],
                        xs[t].init[0],
                        xs[t].init[0],
                        xs[t].init[0],
                        xs[t].init[1],
                        xs[t].init[1],
                        xs[t].init[1],
                        xs[t].init[1],
                    );
                }
                for r in 0..R {
                    let base = w.as_ptr().add((first + r) * stride + kb * width);
                    let d = block_scale(KIND, base);
                    let packed = _mm_loadu_si128(base.add(header).cast());
                    let index = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(_mm_and_si128(packed, mask)),
                        _mm_and_si128(_mm_srli_epi16(packed, 4), mask),
                        1,
                    );
                    let weights = _mm256_shuffle_epi8(table, index);
                    for t in 0..T {
                        let dot = if $biased {
                            _mm256_dpbusd_avx_epi32(init[t], weights, qx[t])
                        } else {
                            _mm256_dpbssd_epi32(_mm256_setzero_si256(), weights, qx[t])
                        };
                        acc[r][t] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(dot),
                            _mm256_set1_ps(d * xs[t].d),
                            acc[r][t],
                        );
                    }
                }
            }
            for t in 0..T {
                for (r, y) in out.row(token + t).iter_mut().take(R).enumerate() {
                    *y = hsum256(acc[r][t]);
                }
            }
        }
    };
}

codebook256!(
    codebook_avx_vnni,
    codebook_avx_vnni_body,
    "avx2,fma,f16c,avxvnni",
    true
);
codebook256!(
    codebook_avx_vnni_int8,
    codebook_avx_vnni_int8_body,
    "avx2,fma,f16c,avxvnniint8",
    false
);

/// Rows per panel the wide codebook kernels like: eight rows for decode (one token) when a
/// 512-bit kernel is available, four otherwise.
pub(crate) fn panel_rows(batch: usize) -> usize {
    if batch == 1 && crate::cpu_kernels::avx512().is_some() {
        8
    } else {
        4
    }
}

/// Run the best codebook kernel the active tier allows on a panel of `rows` rows (at most
/// `panel_rows(batch)`). Returns `false` if only the AVX2 kernel applies, which the caller then
/// runs.
///
/// # Safety
/// `w` holds `first + rows` complete rows of `nb` blocks of `kind` (20 or 39), `x` holds
/// `batch * nb` Q8_0 records, `out` is this panel's exclusive view.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn panel(
    w: &[u8],
    kind: u32,
    first: usize,
    rows: usize,
    x: &[X],
    nb: usize,
    batch: usize,
    out: &mut Rows<'_>,
) -> bool {
    use crate::cpu_kernels::Avx512;
    match crate::cpu_kernels::avx512() {
        Some(Avx512::Vnni) => codebook512_vnni(w, kind, first, rows, x, nb, batch, out),
        Some(Avx512::Bw) => codebook512_bw(w, kind, first, rows, x, nb, batch, out),
        None if crate::cpu_kernels::isa() == "avx-vnni" => {
            // A panel of 4 rows is what the caller asked for without a 512-bit kernel.
            if super::ident::features().avx_vnni_int8 {
                codebook_avx_vnni_int8(w, kind, first, rows, x, nb, batch, out)
            } else {
                codebook_avx_vnni(w, kind, first, rows, x, nb, batch, out)
            }
        }
        None => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::{compare, random_weights, reference_packed, Lcg};

    type Kernel = unsafe fn(&[u8], u32, usize, usize, &[X], usize, usize, &mut Rows<'_>);

    /// All rows of a projection through `kernel` in panels of `size` rows.
    #[allow(clippy::too_many_arguments)]
    fn run(
        kernel: Kernel,
        size: usize,
        w: &[u8],
        kind: u32,
        rows: usize,
        packed: &[X],
        nb: usize,
        batch: usize,
    ) -> Vec<f32> {
        let mut output = vec![vec![0.0f32; rows * batch]];
        crate::cpu_schedule::row_panels(
            &mut output,
            &[rows],
            batch,
            size,
            || (),
            |_, _, first, out| {
                let valid = (rows - first).min(size);
                // SAFETY: the caller checked the CPU features; `w` and `packed` have the shapes
                // the scheduler's panels stay inside.
                unsafe { kernel(w, kind, first, valid, packed, nb, batch, out) };
            },
        );
        output.pop().unwrap()
    }

    fn check(kernel: Kernel, size: usize, label: &str) {
        let mut rng = Lcg(0xc0de_b00c ^ size as u64);
        for kind in [20u32, 39] {
            // Odd and even block counts (the odd one exercises the single-block tail of the
            // 512-bit kernels), every row count of a panel, odd token counts.
            for nb in [1usize, 2, 7, 12] {
                let cols = nb * 32;
                for rows in [1usize, 3, 4, 5, 8, 13] {
                    let w = random_weights(kind, rows, cols, &mut rng);
                    for batch in [1usize, 2, 3, 5] {
                        let x = rng.vector(batch * cols, 3.0);
                        let packed = crate::cpu_quantized::prepare(&x, cols, 2);
                        let have = run(kernel, size, &w, kind, rows, &packed, nb, batch);
                        let (want, scale) = reference_packed(&w, kind, rows, cols, &packed);
                        let (error, ok) = compare(&have, &want, &scale);
                        assert!(
                            ok,
                            "{label} kind {kind} nb {nb} rows {rows} batch {batch}: {error:e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn avx_vnni_256_codebook_matches_the_reference() {
        if !(std::arch::is_x86_feature_detected!("avxvnni")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c"))
        {
            return;
        }
        check(codebook_avx_vnni, 4, "avx-vnni");
    }

    #[test]
    fn avx_vnni_int8_256_codebook_matches_the_reference() {
        if !(std::arch::is_x86_feature_detected!("avxvnniint8")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c"))
        {
            return;
        }
        check(codebook_avx_vnni_int8, 4, "avx-vnni-int8");
    }

    fn has_512() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
    }

    #[test]
    fn avx512_bw_codebook_matches_the_reference() {
        if !has_512() {
            return;
        }
        check(codebook512_bw, 4, "avx512-bw/4");
        check(codebook512_bw, 8, "avx512-bw/8");
    }

    #[test]
    fn avx512_vnni_codebook_matches_the_reference() {
        if !(has_512() && std::arch::is_x86_feature_detected!("avx512vnni")) {
            return;
        }
        check(codebook512_vnni, 4, "avx512-vnni/4");
        check(codebook512_vnni, 8, "avx512-vnni/8");
    }

    /// The tables behind the biased and signed variants describe the same weights.
    #[test]
    fn biased_tables_are_the_signed_tables_plus_128() {
        for kind in [20u32, 39] {
            let signed = table_bytes(kind, false);
            let biased = table_bytes(kind, true);
            for i in 0..16 {
                assert_eq!(
                    i32::from(biased[i]),
                    i32::from(signed[i] as i8) + 128,
                    "kind {kind} entry {i}"
                );
            }
        }
    }

    /// Scalar model of the 512-bit lane arrangement: weights `[a.lo, b.lo, a.hi, b.hi]` against
    /// activations arranged by `shuffle_i64x2::<0xD8>` give the two blocks' dot products.
    /// Scalar model of the 512-bit kernel's data flow for one block pair of one row and one
    /// token, written with the operations of `model` in the order the kernel issues them: the
    /// nibble arrangement `[a low, b low, a high, b high]`, the table lookup, the activation
    /// shuffle, the bias in `init`, and the scale blend. Returns the sum over the 16 dword lanes.
    fn model_pair(
        kind: u32,
        biased: bool,
        a: &[u8],
        b: Option<&[u8]>,
        xa: &X,
        xb: Option<&X>,
    ) -> f32 {
        use crate::cpu_avx512::model::*;
        let (_, header) = block_layout(kind);
        let table = broadcast128(&table_bytes(kind, biased));
        let nibbles = |block: &[u8]| -> [u8; 16] { block[header..header + 16].try_into().unwrap() };
        let pa = nibbles(a);
        let pb = b.map_or(pa, nibbles);
        let mut both = [0u8; 32];
        both[..16].copy_from_slice(&pa);
        both[16..].copy_from_slice(&pb);
        let low = both.map(|v| v & 15);
        let high = both.map(|v| (v >> 4) & 15);
        let weights = shuffle_epi8(&table, &join256(&low, &high));
        let bytes = |x: &X| -> [u8; 32] { x.q.map(|v| v as u8) };
        let joined = join256(&bytes(xa), &xb.map_or([0; 32], bytes));
        let qx = shuffle_i64x2(&joined, &joined, 0xD8);
        let ib = xb.map_or([0, 0], |x| x.init);
        let mut init = [0i32; 16];
        for i in 0..4 {
            init[i] = xa.init[0];
            init[4 + i] = ib[0];
            init[8 + i] = xa.init[1];
            init[12 + i] = ib[1];
        }
        let dot = if biased {
            dpbusd(&init, &weights, &qx)
        } else {
            dot_signed(&weights, &qx)
        };
        // SAFETY: F16C is checked by the callers of this model.
        let scale_of = |block: &[u8]| unsafe { block_scale(kind, block.as_ptr()) };
        let sa = scale_of(a) * xa.d;
        let sb = b.map_or(0.0, scale_of) * xb.map_or(0.0, |x| x.d);
        let scale = blend_ps(0xF0F0, &[sa; 16], &[sb; 16]);
        (0..16).map(|i| dot[i] as f32 * scale[i]).sum()
    }

    /// The lane arrangement of the 512-bit codebook kernels (both flavours) gives the reference
    /// result for even and odd block counts.
    #[test]
    fn modelled_lane_arrangement_matches_the_reference() {
        if !std::arch::is_x86_feature_detected!("f16c") {
            return;
        }
        let mut rng = Lcg(0x70de_0001);
        for kind in [20u32, 39] {
            let (width, _) = block_layout(kind);
            for nb in [1usize, 2, 3, 8] {
                let cols = nb * 32;
                let w = random_weights(kind, 1, cols, &mut rng);
                let x = rng.vector(cols, 3.0);
                let packed = crate::cpu_quantized::prepare(&x, cols, 2);
                let (want, scale) = reference_packed(&w, kind, 1, cols, &packed);
                for biased in [true, false] {
                    let mut total = 0.0f32;
                    let mut kb = 0;
                    while kb < nb {
                        let pair = kb + 1 < nb;
                        let a = &w[kb * width..(kb + 1) * width];
                        let b = pair.then(|| &w[(kb + 1) * width..(kb + 2) * width]);
                        let xb = pair.then(|| &packed[kb + 1]);
                        total += model_pair(kind, biased, a, b, &packed[kb], xb);
                        kb += 2;
                    }
                    let (error, ok) = compare(&[total], &want, &scale);
                    assert!(ok, "kind {kind} nb {nb} biased {biased}: {error:e}");
                }
            }
        }
    }

    #[test]
    fn block_pair_lane_arrangement_covers_each_value_once() {
        // 128-bit lane L of the index vector holds values: L0 a[0..16], L1 b[0..16], L2 a[16..32],
        // L3 b[16..32]. The joined activation vector holds a[0..16], a[16..32], b[0..16], b[16..32]
        // and 0xD8 selects lanes (0, 2, 1, 3) of it.
        let joined: [&str; 4] = ["a.lo", "a.hi", "b.lo", "b.hi"];
        let imm = 0xD8u32;
        let picked: Vec<&str> = (0..4)
            .map(|lane| joined[((imm >> (2 * lane)) & 3) as usize])
            .collect();
        assert_eq!(picked, ["a.lo", "b.lo", "a.hi", "b.hi"]);
        // The scale blend: lanes (dwords) 4..8 and 12..16 are block b.
        let mask = 0xF0F0u32;
        for dword in 0..16 {
            let lane = dword / 4;
            let is_b = mask >> dword & 1 == 1;
            assert_eq!(is_b, lane == 1 || lane == 3, "dword {dword}");
        }
    }
}
