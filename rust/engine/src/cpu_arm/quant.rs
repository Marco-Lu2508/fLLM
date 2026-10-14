//! NEON `sdot` / `smmla` projections for Q2_K, Q3_K, IQ4_NL, IQ4_XS and MXFP4.
//!
//! The structure is that of the proven kernels in `cpu_quantized.rs` (`neon` module): the
//! activations are the 8-bit records `cpu_quantized::prepare` produces (Q8_0-style for IQ4_NL
//! and MXFP4, Q8_K-style for the 256-value formats), one decode token runs a four-row `sdot`
//! kernel that decodes each weight group into registers, and a prompt claims a four-row
//! panel, unpacks it once into [`Group`] records (signed 8-bit weights plus the scales and the
//! minimum coefficient, see `formats.rs`) and runs every token pair over it, with `smmla` when
//! the CPU has FEAT_I8MM. The dataflow of every function here is written once more with scalar
//! models of the instructions in `model.rs`, and that copy is tested against the f64 reference
//! on x86.
//!
//! Safety convention: functions marked `unsafe` need the CPU features of their
//! `target_feature` attribute; the callers check `cpu_kernels::neon_dotprod()` /
//! `neon_i8mm()`. Shapes are validated by `cpu_quantized::validate` before any kernel runs,
//! and every slice below is indexed with bounds checks except the 16-byte vector loads inside a
//! block whose length was asserted on entry.
use super::formats::{
    block_bytes, groups, has_min, iq4xs_scale, q3k_scales, split_scales, Group, IQ4_NL, IQ4_XS,
    KVALUES_IQ4NL, MXFP4, MXFP4_VALUES, Q2_K, Q3_K,
};
use crate::cpu_kernels::neon::{half_to_float, smmla, zip_hi, zip_lo};
use crate::cpu_quantized::X;
use crate::cpu_schedule::Rows;
use std::arch::aarch64::*;

/// Codebooks as statics, so a pointer to them stays valid.
static IQ4NL_TABLE: [i8; 16] = KVALUES_IQ4NL;
static MXFP4_TABLE: [i8; 16] = MXFP4_VALUES;

/// GGUF types this module has kernels for.
pub(crate) const fn handles(kind: u32) -> bool {
    matches!(kind, Q2_K | Q3_K | IQ4_NL | IQ4_XS | MXFP4)
}

/// Reusable prompt panel: four rows of unpacked groups.
pub(crate) type Panel = Vec<Group>;

macro_rules! by_kind {
    ($kind:expr, $run:ident) => {
        match $kind {
            10 => $run!(10),
            11 => $run!(11),
            20 => $run!(20),
            23 => $run!(23),
            39 => $run!(39),
            other => unreachable!("no ARM kernel for GGUF type {other}"),
        }
    };
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn half(raw: &[u8], at: usize) -> f32 {
    half_to_float(u16::from_le_bytes([raw[at], raw[at + 1]]))
}

/// Block-level scales, decoded once per row and block.
#[derive(Clone, Copy)]
struct Head {
    /// `d` of the block (E8M0 scale for MXFP4).
    d: f32,
    /// `dmin` (Q2_K).
    min: f32,
    /// The sixteen signed 6-bit scales of Q3_K.
    scales: [i8; 16],
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn head<const KIND: u32>(raw: &[u8]) -> Head {
    assert_eq!(raw.len(), block_bytes(KIND));
    let mut h = Head {
        d: 0.0,
        min: 0.0,
        scales: [0; 16],
    };
    match KIND {
        Q2_K => {
            h.d = half(raw, 80);
            h.min = half(raw, 82);
        }
        Q3_K => {
            h.d = half(raw, 108);
            h.scales = q3k_scales(raw);
        }
        IQ4_NL | IQ4_XS => h.d = half(raw, 0),
        _ => h.d = super::formats::e8m0_half(raw[0]),
    }
    h
}

/// One decoded group in registers (the NEON form of [`Group`]).
#[derive(Clone, Copy)]
struct Regs {
    q: [int8x16_t; 2],
    d: [f32; 2],
    m: [f32; 2],
}

/// Group `g` of the block `raw`: the values `formats::decode_group` defines.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn group<const KIND: u32>(raw: &[u8], h: &Head, g: usize) -> Regs {
    let p = raw.as_ptr();
    match KIND {
        Q2_K => {
            let (n, shift) = (g / 4, 2 * (g % 4));
            let qs = p.add(16 + n * 32);
            let shift = vdupq_n_s8(-(shift as i8));
            let three = vdupq_n_u8(3);
            let a = vandq_u8(vshlq_u8(vld1q_u8(qs), shift), three);
            let b = vandq_u8(vshlq_u8(vld1q_u8(qs.add(16)), shift), three);
            let (s0, s1) = (raw[2 * g], raw[2 * g + 1]);
            Regs {
                q: [vreinterpretq_s8_u8(a), vreinterpretq_s8_u8(b)],
                d: [h.d * f32::from(s0 & 15), h.d * f32::from(s1 & 15)],
                m: [-h.min * f32::from(s0 >> 4), -h.min * f32::from(s1 >> 4)],
            }
        }
        Q3_K => {
            let (outer, shift) = (g / 4, 2 * (g % 4));
            let qs = p.add(32 + outer * 32);
            let shift = vdupq_n_s8(-(shift as i8));
            let bit = vdupq_n_s8(-(g as i8));
            let (three, one, four) = (vdupq_n_u8(3), vdupq_n_u8(1), vdupq_n_s8(4));
            // value = (low two bits | hmask bit << 2) - 4
            let decode = |q: uint8x16_t, hm: uint8x16_t| {
                let low = vandq_u8(vshlq_u8(q, shift), three);
                let high = vshlq_n_u8::<2>(vandq_u8(vshlq_u8(hm, bit), one));
                vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(low, high)), four)
            };
            Regs {
                q: [
                    decode(vld1q_u8(qs), vld1q_u8(p)),
                    decode(vld1q_u8(qs.add(16)), vld1q_u8(p.add(16))),
                ],
                d: [
                    h.d * f32::from(h.scales[2 * g]),
                    h.d * f32::from(h.scales[2 * g + 1]),
                ],
                m: [0.0; 2],
            }
        }
        _ => {
            // IQ4_NL, IQ4_XS and MXFP4: sixteen bytes of nibbles through a 16-entry table.
            let (table, at, scale) = match KIND {
                IQ4_NL => (IQ4NL_TABLE.as_ptr(), 2, h.d),
                IQ4_XS => (
                    IQ4NL_TABLE.as_ptr(),
                    8 + g * 16,
                    h.d * iq4xs_scale(raw, g) as f32,
                ),
                _ => (MXFP4_TABLE.as_ptr(), 1, h.d),
            };
            debug_assert!(at + 16 <= raw.len());
            let table = vld1q_s8(table);
            let nibbles = vld1q_u8(p.add(at));
            Regs {
                q: [
                    vqtbl1q_s8(table, vandq_u8(nibbles, vdupq_n_u8(15))),
                    vqtbl1q_s8(table, vshrq_n_u8::<4>(nibbles)),
                ],
                d: [scale; 2],
                m: [0.0; 2],
            }
        }
    }
}

/// Minimum correction of one group (Q2_K), in the order of the Q4_K kernels.
#[inline]
fn offset<const KIND: u32>(acc: &mut f32, m: [f32; 2], xr: &X) {
    if has_min(KIND) {
        *acc = m[0].mul_add(xr.sum[0], m[1].mul_add(xr.sum[1], *acc));
    }
}

/// One group times one activation record into a four-lane accumulator.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn accumulate<const KIND: u32>(
    acc: &mut float32x4_t,
    q: [int8x16_t; 2],
    d: [f32; 2],
    xq: [int8x16_t; 2],
    xd: f32,
) {
    let zero = vdupq_n_s32(0);
    if split_scales(KIND) {
        let lo = vdotq_s32(zero, q[0], xq[0]);
        let hi = vdotq_s32(zero, q[1], xq[1]);
        *acc = vfmaq_f32(*acc, vcvtq_f32_s32(lo), vdupq_n_f32(d[0] * xd));
        *acc = vfmaq_f32(*acc, vcvtq_f32_s32(hi), vdupq_n_f32(d[1] * xd));
    } else {
        let dot = vdotq_s32(vdotq_s32(zero, q[0], xq[0]), q[1], xq[1]);
        *acc = vfmaq_f32(*acc, vcvtq_f32_s32(dot), vdupq_n_f32(d[0] * xd));
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn activation(xr: &X) -> [int8x16_t; 2] {
    [vld1q_s8(xr.q.as_ptr()), vld1q_s8(xr.q.as_ptr().add(16))]
}

/// Unpack one row of `kind` blocks into groups (the prompt panel row).
///
/// # Safety
/// NEON must be available; `raw` holds whole blocks and `out` one record per 32 values.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn unpack_row(kind: u32, raw: &[u8], out: &mut [Group]) {
    macro_rules! run {
        ($kind:literal) => {
            unpack_kind::<$kind>(raw, out)
        };
    }
    by_kind!(kind, run)
}

// The block width is a function of `KIND`, which stable Rust cannot use as a const generic size.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_kind<const KIND: u32>(raw: &[u8], out: &mut [Group]) {
    let (width, per_block) = (block_bytes(KIND), groups(KIND));
    assert_eq!(raw.len() % width, 0);
    assert_eq!(out.len(), raw.len() / width * per_block);
    for (block, dst) in raw.chunks_exact(width).zip(out.chunks_exact_mut(per_block)) {
        let h = head::<KIND>(block);
        for (g, record) in dst.iter_mut().enumerate() {
            let regs = group::<KIND>(block, &h, g);
            vst1q_s8(record.q.as_mut_ptr(), regs.q[0]);
            vst1q_s8(record.q.as_mut_ptr().add(16), regs.q[1]);
            record.d = regs.d;
            record.m = regs.m;
        }
    }
}

/// `out.len()` (1..=4) rows from `first` for one decode token, decoding straight into registers.
///
/// # Safety
/// The CPU must support NEON and dotprod; `w` holds complete rows of `kind` blocks covering
/// `x.len()` records through row `first + out.len()`.
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn gemv(w: &[u8], kind: u32, x: &[X], first: usize, out: &mut [f32]) {
    macro_rules! run {
        ($kind:literal) => {
            match out.len() {
                1 => gemv_rows::<$kind, 1>(w, x, first, out),
                2 => gemv_rows::<$kind, 2>(w, x, first, out),
                3 => gemv_rows::<$kind, 3>(w, x, first, out),
                4 => gemv_rows::<$kind, 4>(w, x, first, out),
                _ => unreachable!(),
            }
        };
    }
    by_kind!(kind, run)
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemv_rows<const KIND: u32, const ROWS: usize>(
    w: &[u8],
    x: &[X],
    first: usize,
    out: &mut [f32],
) {
    let (width, per_block) = (block_bytes(KIND), groups(KIND));
    let blocks = x.len() / per_block;
    let stride = blocks * width;
    assert!(w.len() >= (first + ROWS) * stride);
    let mut acc = [vdupq_n_f32(0.0); ROWS];
    let mut offsets = [0.0_f32; ROWS];
    for block in 0..blocks {
        let raws: [&[u8]; ROWS] = std::array::from_fn(|r| {
            let start = (first + r) * stride + block * width;
            &w[start..start + width]
        });
        let heads: [Head; ROWS] = std::array::from_fn(|r| head::<KIND>(raws[r]));
        for g in 0..per_block {
            let xr = &x[block * per_block + g];
            let xq = activation(xr);
            for r in 0..ROWS {
                let regs = group::<KIND>(raws[r], &heads[r], g);
                accumulate::<KIND>(&mut acc[r], regs.q, regs.d, xq, xr.d);
                offset::<KIND>(&mut offsets[r], regs.m, xr);
            }
        }
    }
    for (r, y) in out.iter_mut().enumerate() {
        *y = vaddvq_f32(acc[r]) + offsets[r];
    }
}

/// Four-row panel by two tokens per pass with `sdot`.
///
/// # Safety
/// The CPU must support NEON and dotprod; `panel` holds four rows of `nb` groups and `x` holds
/// `batch * nb` activation records.
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn tile(
    kind: u32,
    panel: &[Group],
    x: &[X],
    nb: usize,
    batch: usize,
    out: &mut Rows<'_>,
) {
    macro_rules! run {
        ($kind:literal) => {
            tile_kind::<$kind>(panel, x, nb, batch, out)
        };
    }
    by_kind!(kind, run)
}

#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn tile_kind<const KIND: u32>(
    panel: &[Group],
    x: &[X],
    nb: usize,
    batch: usize,
    out: &mut Rows<'_>,
) {
    assert!(panel.len() >= 4 * nb && x.len() >= batch * nb);
    for b in (0..batch).step_by(2) {
        let mut acc = [[vdupq_n_f32(0.0); 4]; 2];
        let mut offsets = [[0.0_f32; 4]; 2];
        for kb in 0..nb {
            let xs = [&x[b * nb + kb], &x[(b + 1).min(batch - 1) * nb + kb]];
            let xq = [activation(xs[0]), activation(xs[1])];
            for r in 0..4 {
                let record = &panel[r * nb + kb];
                let q = [
                    vld1q_s8(record.q.as_ptr()),
                    vld1q_s8(record.q.as_ptr().add(16)),
                ];
                for t in 0..2 {
                    accumulate::<KIND>(&mut acc[t][r], q, record.d, xq[t], xs[t].d);
                    offset::<KIND>(&mut offsets[t][r], record.m, xs[t]);
                }
            }
        }
        for j in 0..(batch - b).min(2) {
            for (r, y) in out.row(b + j).iter_mut().enumerate() {
                *y = vaddvq_f32(acc[j][r]) + offsets[j][r];
            }
        }
    }
}

/// `tile` with `smmla`: each row pair meets the token pair in four instructions per 32-value
/// group (two per 16-value half when the halves have their own scale).
///
/// # Safety
/// As `tile`, and the CPU must support i8mm.
#[target_feature(enable = "neon,dotprod,i8mm")]
pub(crate) unsafe fn tile_i8mm(
    kind: u32,
    panel: &[Group],
    x: &[X],
    nb: usize,
    batch: usize,
    out: &mut Rows<'_>,
) {
    macro_rules! run {
        ($kind:literal) => {
            tile_i8mm_kind::<$kind>(panel, x, nb, batch, out)
        };
    }
    by_kind!(kind, run)
}

#[inline]
#[target_feature(enable = "neon,dotprod,i8mm")]
unsafe fn tile_i8mm_kind<const KIND: u32>(
    panel: &[Group],
    x: &[X],
    nb: usize,
    batch: usize,
    out: &mut Rows<'_>,
) {
    assert!(panel.len() >= 4 * nb && x.len() >= batch * nb);
    let zero = vdupq_n_s32(0);
    for b in (0..batch).step_by(2) {
        // Lanes: [row 2k token 0, row 2k token 1, row 2k+1 token 0, row 2k+1 token 1].
        let mut acc = [vdupq_n_f32(0.0); 2];
        let mut offsets = [[0.0_f32; 4]; 2];
        for kb in 0..nb {
            let xs = [&x[b * nb + kb], &x[(b + 1).min(batch - 1) * nb + kb]];
            let (a, c) = (activation(xs[0]), activation(xs[1]));
            let xm = [
                zip_lo(a[0], c[0]),
                zip_hi(a[0], c[0]),
                zip_lo(a[1], c[1]),
                zip_hi(a[1], c[1]),
            ];
            let xd = [xs[0].d, xs[1].d, xs[0].d, xs[1].d];
            let xd = vld1q_f32(xd.as_ptr());
            for (k, acc) in acc.iter_mut().enumerate() {
                let (wa, wb) = (&panel[2 * k * nb + kb], &panel[(2 * k + 1) * nb + kb]);
                let (qa0, qa1) = (vld1q_s8(wa.q.as_ptr()), vld1q_s8(wa.q.as_ptr().add(16)));
                let (qb0, qb1) = (vld1q_s8(wb.q.as_ptr()), vld1q_s8(wb.q.as_ptr().add(16)));
                let low = smmla(
                    smmla(zero, zip_lo(qa0, qb0), xm[0]),
                    zip_hi(qa0, qb0),
                    xm[1],
                );
                let high = smmla(
                    smmla(zero, zip_lo(qa1, qb1), xm[2]),
                    zip_hi(qa1, qb1),
                    xm[3],
                );
                let d0 = [wa.d[0], wa.d[0], wb.d[0], wb.d[0]];
                let scale0 = vmulq_f32(vld1q_f32(d0.as_ptr()), xd);
                if split_scales(KIND) {
                    let d1 = [wa.d[1], wa.d[1], wb.d[1], wb.d[1]];
                    let scale1 = vmulq_f32(vld1q_f32(d1.as_ptr()), xd);
                    *acc = vfmaq_f32(*acc, vcvtq_f32_s32(low), scale0);
                    *acc = vfmaq_f32(*acc, vcvtq_f32_s32(high), scale1);
                } else {
                    *acc = vfmaq_f32(*acc, vcvtq_f32_s32(vaddq_s32(low, high)), scale0);
                }
                for (t, xr) in xs.into_iter().enumerate() {
                    offset::<KIND>(&mut offsets[t][2 * k], wa.m, xr);
                    offset::<KIND>(&mut offsets[t][2 * k + 1], wb.m, xr);
                }
            }
        }
        let mut lanes = [[0.0_f32; 4]; 2];
        for (lanes, acc) in lanes.iter_mut().zip(acc) {
            vst1q_f32(lanes.as_mut_ptr(), acc);
        }
        for j in 0..(batch - b).min(2) {
            for (r, y) in out.row(b + j).iter_mut().enumerate() {
                *y = lanes[r / 2][(r % 2) * 2 + j] + offsets[j][r];
            }
        }
    }
}

/// Unpack the panel of rows `first..first + valid` (at most four, `width` bytes per row) and run
/// every token of the prompt over it.
///
/// # Safety
/// The CPU must support NEON and dotprod (and i8mm when `i8mm` is set); `w` holds rows
/// `first..first + valid` of `width` bytes, `x` holds `batch * nb` activation records.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn prompt_panel(
    kind: u32,
    w: &[u8],
    width: usize,
    (first, valid): (usize, usize),
    x: &[X],
    (nb, batch): (usize, usize),
    i8mm: bool,
    panel: &mut Panel,
    out: &mut Rows<'_>,
) {
    debug_assert!((1..=4).contains(&valid));
    panel.resize(4 * nb, Group::default());
    for r in 0..valid {
        unpack_row(
            kind,
            &w[(first + r) * width..][..width],
            &mut panel[r * nb..(r + 1) * nb],
        );
    }
    for r in valid..4 {
        panel[r * nb..(r + 1) * nb].fill(Group::default());
    }
    if i8mm {
        tile_i8mm(kind, panel, x, nb, batch, out);
    } else {
        tile(kind, panel, x, nb, batch, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_arm::formats::{decode_row, Format};
    use crate::cpu_selftest::{compare, random_weights, reference_packed, Lcg};

    fn dotprod() -> bool {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    fn i8mm() -> bool {
        dotprod() && std::arch::is_aarch64_feature_detected!("i8mm")
    }

    #[test]
    fn neon_decoders_equal_the_scalar_decoders() {
        let mut rng = Lcg(0x4e_0a);
        for format in Format::ALL {
            let kind = format.kind();
            let (values, _) = crate::cpu_selftest::layout(kind);
            let blocks = 5;
            let raw = random_weights(kind, 1, values * blocks, &mut rng);
            let count = blocks * format.groups();
            let mut want = vec![Group::default(); count];
            decode_row(format, &raw, &mut want);
            let mut have = vec![Group::default(); count];
            // SAFETY: NEON is the AArch64 baseline; `raw` holds whole blocks.
            unsafe { unpack_row(kind, &raw, &mut have) };
            for (i, (a, b)) in have.iter().zip(&want).enumerate() {
                assert_eq!(a.q, b.q, "{format:?} group {i} weights");
                assert_eq!(
                    a.d.map(f32::to_bits),
                    b.d.map(f32::to_bits),
                    "{format:?} group {i} d"
                );
                assert_eq!(
                    a.m.map(f32::to_bits),
                    b.m.map(f32::to_bits),
                    "{format:?} group {i} m"
                );
            }
        }
    }

    /// Prompt kernels over a panel, scheduled the way `linear_neon` does.
    fn prompt(
        kind: u32,
        w: &[u8],
        rows: usize,
        cols: usize,
        x: &[X],
        batch: usize,
        i8mm: bool,
    ) -> Vec<f32> {
        let nb = cols / 32;
        let (values, bytes) = crate::cpu_selftest::layout(kind);
        let width = cols / values * bytes;
        let mut output = vec![vec![0.0; batch * rows]];
        crate::cpu_schedule::row_panels(
            &mut output,
            &[rows],
            batch,
            4,
            Panel::new,
            |panel, _, first, out| {
                let valid = (rows - first).min(4);
                // SAFETY: the features were checked by the caller; shapes follow from the arguments.
                unsafe {
                    prompt_panel(
                        kind,
                        w,
                        width,
                        (first, valid),
                        x,
                        (nb, batch),
                        i8mm,
                        panel,
                        out,
                    )
                };
            },
        );
        output.pop().unwrap()
    }

    #[test]
    fn kernels_match_the_reference() {
        if !dotprod() {
            return;
        }
        let mut rng = Lcg(0x77_01);
        for format in Format::ALL {
            let kind = format.kind();
            let cols = if format.groups() == 8 { 768 } else { 352 };
            for rows in [1usize, 3, 4, 6, 13] {
                let w = random_weights(kind, rows, cols, &mut rng);
                for batch in [1usize, 2, 3, 5, 17] {
                    let x = rng.vector(batch * cols, 3.0);
                    let packed = crate::cpu_quantized::prepare(&x, cols, kind);
                    let (want, scale) = reference_packed(&w, kind, rows, cols, &packed);
                    let name = format!("{format:?} rows {rows} batch {batch}");
                    // Decode kernel, panels of up to four rows.
                    if batch == 1 {
                        let mut have = vec![0.0f32; rows];
                        let mut first = 0;
                        while first < rows {
                            let n = (rows - first).min(4);
                            // SAFETY: dotprod detected; rows lie inside `w`.
                            unsafe { gemv(&w, kind, &packed, first, &mut have[first..first + n]) };
                            first += n;
                        }
                        let (error, ok) = compare(&have, &want, &scale);
                        assert!(ok, "gemv {name}: {error:e}");
                    }
                    for use_i8mm in [false, true] {
                        if use_i8mm && !i8mm() {
                            continue;
                        }
                        let have = prompt(kind, &w, rows, cols, &packed, batch, use_i8mm);
                        let (error, ok) = compare(&have, &want, &scale);
                        assert!(ok, "prompt i8mm={use_i8mm} {name}: {error:e}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_public_projection_uses_the_new_kernels() {
        if !crate::cpu_kernels::neon_dotprod() {
            return;
        }
        let mut rng = Lcg(0x91);
        for format in Format::ALL {
            let kind = format.kind();
            let (rows, cols, batch) = (9, 768, 4);
            let w = random_weights(kind, rows, cols, &mut rng);
            let x = rng.vector(batch * cols, 3.0);
            let have = crate::cpu_quantized::linear(&w, kind, &x, rows, cols, None)
                .expect("kernel present at this tier");
            let packed = crate::cpu_quantized::prepare(&x, cols, kind);
            let (want, scale) = reference_packed(&w, kind, rows, cols, &packed);
            let (error, ok) = compare(&have, &want, &scale);
            assert!(ok, "{format:?}: {error:e}");
        }
    }
}
