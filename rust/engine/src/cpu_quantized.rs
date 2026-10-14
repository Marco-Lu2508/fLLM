//! Packed integer projections for Q4_0, Q4_1, Q5_0, Q5_1, Q2_K..Q6_K, IQ4_NL, IQ4_XS and MXFP4.
//!
//! `linear_mixed` packs the input once into 32-value 8-bit records (`X`): Q8_0-style records with
//! an fp16 scale for the 32-value formats (plus the fp16 Q8_1 sum for Q4_1/Q5_1), Q8_K-style
//! records sharing one f32 scale per 256 values for the K formats. Every member of a group then
//! runs in one work queue: the K formats in `cpu_kquant`, IQ4_NL/MXFP4 in a codebook kernel, the
//! classic formats in a direct-decode GEMV (one token) or an unpacked W-panel tile (prompts).
//!
//! Accuracy: these kernels quantize the activations to 8 bits, like llama.cpp's CPU kernels.
//! Machines without a kernel (no AVX2/FMA/F16C on x86-64, no dotprod on AArch64) fall back to the
//! f32 row decode in `compute.rs`, which keeps full-precision activations, so the two paths do
//! not agree bit for bit; see docs/cpu-kernels.md.
use half::f16;
use rayon::prelude::*;

#[repr(C, align(32))]
#[derive(Clone, Default)]
pub(crate) struct X {
    pub(crate) q: [i8; 32],
    pub(crate) d: f32,
    pub(crate) init: [i32; 2],
    pub(crate) sum: [f32; 2],
}
#[repr(C, align(16))]
#[derive(Clone, Default)]
struct W {
    q: [u8; 32],
    d: [f32; 2],
    m: [f32; 2],
}

pub(crate) fn prepare(x: &[f32], cols: usize, kind: u32) -> Vec<X> {
    let mut packed = Vec::new();
    prepare_into(x, cols, kind, &mut packed);
    packed
}

/// `prepare` into a caller-owned buffer, so decode loops reuse one allocation.
pub(crate) fn prepare_into(x: &[f32], cols: usize, kind: u32, packed: &mut Vec<X>) {
    packed.clear();
    packed.resize(x.len() / 32, X::default());
    #[cfg(target_arch = "x86_64")]
    let fast = crate::cpu_kernels::isa() != "scalar";
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    let fast = crate::cpu_kernels::neon();
    let pack_row = |dst: &mut [X], row: &[f32]| {
        let k = matches!(kind, 10..=14 | 23);
        let group = if k { 256 } else { 32 };
        for (chunk, out) in row.chunks_exact(group).zip(dst.chunks_mut(group / 32)) {
            #[cfg(target_arch = "x86_64")]
            if fast && !k {
                let dst = &mut out[0];
                // SAFETY: AVX2/F16C confirmed (`fast`); `chunk` is one 32-value block.
                let (d, exact, sums) =
                    unsafe { crate::cpu_kernels::x86::pack_q8_block(chunk, &mut dst.q, false) };
                dst.d = d;
                dst.init = [-32 * sums[0], -32 * sums[1]];
                dst.sum = if matches!(kind, 3 | 7) {
                    // SAFETY: F16C confirmed (`fast`).
                    let sum = unsafe {
                        crate::cpu_kernels::x86::half_to_float(
                            crate::cpu_kernels::x86::float_to_half(
                                (sums[0] + sums[1]) as f32 * exact,
                            ),
                        )
                    };
                    [sum, 0.0]
                } else {
                    [sums[0] as f32 * d, sums[1] as f32 * d]
                };
                continue;
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            if fast && !k {
                let dst = &mut out[0];
                // SAFETY: NEON detected (`fast`); `chunk` is one 32-value block.
                let (d, exact, sums) =
                    unsafe { crate::cpu_kernels::neon::pack_q8_block(chunk, &mut dst.q, false) };
                dst.d = d;
                dst.init = [-32 * sums[0], -32 * sums[1]];
                dst.sum = if matches!(kind, 3 | 7) {
                    [
                        f16::from_f32((sums[0] + sums[1]) as f32 * exact).to_f32(),
                        0.0,
                    ]
                } else {
                    [sums[0] as f32 * d, sums[1] as f32 * d]
                };
                continue;
            }
            let mut amax = 0.0;
            let mut signed_max = 0.0;
            #[cfg(target_arch = "x86_64")]
            let use_simd = fast && k;
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            let use_simd = fast && k;
            #[cfg(not(any(
                target_arch = "x86_64",
                all(target_arch = "aarch64", target_endian = "little")
            )))]
            let use_simd = false;
            #[cfg(target_arch = "x86_64")]
            if use_simd {
                // K activation blocks require the sign of the FIRST absolute
                // maximum. A vector lane reduction alone can change ties.
                // SAFETY: AVX2 confirmed (`fast`); `chunk` is a whole 256-value block.
                (amax, signed_max) = unsafe { x86::first_signed_max(chunk) };
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            if use_simd {
                // SAFETY: NEON detected (`fast`); `chunk` is a whole 256-value block.
                (amax, signed_max) = unsafe { neon::first_signed_max(chunk) };
            }
            if !use_simd {
                for &v in chunk {
                    if v.abs() > amax {
                        amax = v.abs();
                        signed_max = v;
                    }
                }
            }
            let inv = if amax == 0.0 {
                0.0
            } else if k {
                -127.0 / signed_max
            } else {
                127.0 / amax
            };
            let exact_d = if inv == 0.0 {
                0.0
            } else if k {
                inv.recip()
            } else {
                amax / 127.0
            };
            let d = if k {
                exact_d
            } else {
                f16::from_f32(exact_d).to_f32()
            };
            for (src, dst) in chunk.as_chunks::<32>().0.iter().zip(out) {
                dst.d = d;
                #[cfg(target_arch = "x86_64")]
                if fast {
                    // SAFETY: AVX2 confirmed (`fast`); `src` is one 32-value block.
                    let sums =
                        unsafe { crate::cpu_kernels::x86::quantize_i8_block(src, inv, &mut dst.q) };
                    dst.init = [-32 * sums[0], -32 * sums[1]];
                    dst.sum = [sums[0] as f32 * d, sums[1] as f32 * d];
                    continue;
                }
                #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
                if fast {
                    // SAFETY: NEON detected (`fast`); `src` is one 32-value block.
                    let sums = unsafe {
                        crate::cpu_kernels::neon::quantize_i8_block(src, inv, &mut dst.q)
                    };
                    dst.init = [-32 * sums[0], -32 * sums[1]];
                    dst.sum = [sums[0] as f32 * d, sums[1] as f32 * d];
                    continue;
                }
                for (q, &v) in dst.q.iter_mut().zip(src) {
                    *q = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
                }
                let sums = [
                    dst.q[..16].iter().map(|&q| q as i32).sum::<i32>(),
                    dst.q[16..].iter().map(|&q| q as i32).sum::<i32>(),
                ];
                dst.init = [-32 * sums[0], -32 * sums[1]];
                dst.sum = if matches!(kind, 3 | 7) {
                    // Q8_1 stores its sum separately as fp16, using the exact
                    // scale before d itself rounds to fp16.
                    [
                        f16::from_f32((sums[0] + sums[1]) as f32 * exact_d).to_f32(),
                        0.0,
                    ]
                } else {
                    [sums[0] as f32 * d, sums[1] as f32 * d]
                };
            }
        }
    };
    // Decode has only one row; Rayon cannot divide it between workers. Small
    // prompt vectors likewise do not justify waking the pool just to pack.
    if x.len() == cols || x.len() <= 16_384 {
        for (dst, row) in packed.chunks_mut(cols / 32).zip(x.chunks(cols)) {
            pack_row(dst, row);
        }
    } else {
        packed
            .par_chunks_mut(cols / 32)
            .zip(x.par_chunks(cols))
            .for_each(|(dst, row)| pack_row(dst, row));
    }
}
/// Activation records a weight format consumes. One packing serves every member of a group
/// with the same records; the K formats all take Q8_K, Q4_1/Q5_1 need the fp16 Q8_1 sum.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Activation {
    Q8_0,
    Q8_1,
    Q8K,
}

fn activation(kind: u32) -> Option<Activation> {
    match kind {
        2 | 6 | 20 | 39 => Some(Activation::Q8_0),
        3 | 7 => Some(Activation::Q8_1),
        10..=14 | 23 => Some(Activation::Q8K),
        _ => None,
    }
}

/// (values, bytes) per block of the packed formats that have CPU kernels.
fn layout(kind: u32) -> Option<(usize, usize)> {
    Some(match kind {
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        20 => (32, 18),
        23 => (256, 136),
        39 => (32, 17),
        _ => return None,
    })
}

/// Whether `kinds` may share one activation packing in `linear_mixed`: K formats (Q2_K..Q6_K,
/// IQ4_XS) mix freely, every other format only with itself.
pub fn groupable(kinds: &[u32]) -> bool {
    let Some(&first) = kinds.first() else {
        return false;
    };
    match activation(first) {
        Some(Activation::Q8K) => kinds
            .iter()
            .all(|&kind| activation(kind) == Some(Activation::Q8K)),
        Some(_) => kinds.iter().all(|&kind| kind == first),
        None => false,
    }
}

/// Shape checks shared by every architecture; returns the token count.
fn validate(projections: &[Projection<'_>], kinds: &[u32], x: &[f32], cols: usize) -> usize {
    assert_eq!(projections.len(), kinds.len());
    assert!(cols > 0 && x.len().is_multiple_of(cols));
    for (p, &kind) in projections.iter().zip(kinds) {
        check_projection(p, kind, cols);
    }
    x.len() / cols
}

/// Shape checks of one projection against its input width.
fn check_projection(p: &Projection<'_>, kind: u32, cols: usize) {
    let (block, width) = layout(kind).expect("format has a CPU kernel");
    assert!(cols.is_multiple_of(block));
    assert!(p.rows > 0);
    assert_eq!(p.weights.len(), p.rows * (cols / block * width));
    assert!(p.bias.is_none_or(|bias| bias.len() == p.rows));
}

fn add_bias(
    out: &mut crate::cpu_schedule::Rows<'_>,
    bias: Option<&[f32]>,
    first: usize,
    batch: usize,
) {
    if let Some(bias) = bias {
        for token in 0..batch {
            for (r, y) in out.row(token).iter_mut().enumerate() {
                *y += bias[first + r];
            }
        }
    }
}

pub fn linear(
    w: &[u8],
    kind: u32,
    x: &[f32],
    rows: usize,
    cols: usize,
    bias: Option<&[f32]>,
) -> Option<Vec<f32>> {
    linear_group(
        &[Projection {
            weights: w,
            rows,
            bias,
        }],
        kind,
        x,
        cols,
    )
    .map(|mut output| output.pop().unwrap())
}

pub struct Projection<'a> {
    pub weights: &'a [u8],
    pub rows: usize,
    pub bias: Option<&'a [f32]>,
}

/// Pack once and claim sibling projection rows in a single joined work queue.
pub fn linear_group(
    projections: &[Projection<'_>],
    kind: u32,
    x: &[f32],
    cols: usize,
) -> Option<Vec<Vec<f32>>> {
    if projections.is_empty() {
        return (crate::cpu_kernels::packed_simd() && layout(kind).is_some()).then(Vec::new);
    }
    linear_mixed(projections, &vec![kind; projections.len()], x, cols)
}

/// `linear_group` for siblings whose formats share an activation packing (`groupable`), such
/// as the Q4_K and Q6_K members of one Q/K/V group of a Q4_K_M model: the input is quantized
/// once and all rows run in one work queue. `None` when this CPU has no kernel for a member.
pub fn linear_mixed(
    projections: &[Projection<'_>],
    kinds: &[u32],
    x: &[f32],
    cols: usize,
) -> Option<Vec<Vec<f32>>> {
    if !groupable(kinds) || kinds.len() != projections.len() {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let isa = crate::cpu_kernels::isa();
        if isa != "scalar" {
            return Some(linear_x86(projections, kinds, x, cols, isa == "avx-vnni"));
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if crate::cpu_kernels::neon_dotprod()
        && kinds
            .iter()
            .all(|&kind| {
                matches!(kind, 2 | 3 | 6 | 7 | 12..=14) || crate::cpu_arm::quant::handles(kind)
            })
    {
        return Some(linear_neon(projections, kinds, x, cols));
    }
    let _ = (x, cols);
    None
}

/// The three projections of one gated expert FFN and their weight formats.
pub struct ExpertProjections<'a> {
    pub gate: Projection<'a>,
    pub up: Projection<'a>,
    pub down: Projection<'a>,
    /// Weight formats of gate, up and down.
    pub kinds: [u32; 3],
}

/// Buffers `linear_experts` reuses between calls so a decode step allocates nothing per expert.
#[derive(Default)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub struct ExpertScratch {
    input: Vec<X>,
    hidden: Vec<X>,
    gate_up: Vec<Vec<f32>>,
    mid: Vec<f32>,
    down: Vec<Vec<f32>>,
    rows: Vec<usize>,
}

/// The expert FFNs `down(activate(gate(x), up(x)))` of one token, one result per expert.
///
/// Instead of a fork/join for every projection of every expert, all gate and up projections
/// of the active experts run in one work queue and all down projections in a second one, with
/// the serial `activate` in between. Every output row is computed by the same kernel, from the
/// same activation records, as `linear_mixed` / `linear` would use for the projection alone,
/// so the results are bit for bit those of the per-projection path.
///
/// `None` when the fused path does not apply (an empty list, differing or non-K formats,
/// no x86-64 kernels); the caller then runs the experts one by one. Shape errors panic like
/// the other entry points.
pub fn linear_experts<'s>(
    experts: &[ExpertProjections<'_>],
    input: &[f32],
    cols: usize,
    activate: &dyn Fn(&mut [f32], &[f32]),
    scratch: &'s mut ExpertScratch,
) -> Option<&'s [Vec<f32>]> {
    #[cfg(target_arch = "x86_64")]
    {
        let isa = crate::cpu_kernels::isa();
        if isa != "scalar" {
            return experts_x86(experts, input, cols, activate, scratch, isa == "avx-vnni");
        }
    }
    let _ = (experts, input, cols, activate, scratch);
    None
}

/// Smallest prompt for which a worker decodes its weight panel once: two token pairs.
#[cfg(target_arch = "x86_64")]
const PREDECODE_MIN_BATCH: usize = 4;

/// K formats whose prompt path runs on decoded panels.
#[cfg(target_arch = "x86_64")]
const PREDECODED_KINDS: &[u32] = &[14, 10, 13, 12, 11];

/// Formats whose 512-bit decode kernels take eight-row panels. Q2_K, Q3_K and IQ4_XS have such
/// kernels only in builds with the `avx512` feature (`cpu_avx512::kextra`).
#[cfg(target_arch = "x86_64")]
fn eight_row_kind(kind: u32) -> bool {
    matches!(kind, 12..=14) || (cfg!(feature = "avx512") && matches!(kind, 10 | 11 | 23))
}

#[cfg(target_arch = "x86_64")]
fn linear_x86(
    projections: &[Projection<'_>],
    kinds: &[u32],
    x: &[f32],
    cols: usize,
    vnni: bool,
) -> Vec<Vec<f32>> {
    use crate::cpu_kernels::Avx512;
    let batch = validate(projections, kinds, x, cols);
    let rows = projections.iter().map(|p| p.rows).collect::<Vec<_>>();
    let mut output = rows
        .iter()
        .map(|&r| vec![0.0; batch * r])
        .collect::<Vec<_>>();
    if batch == 0 {
        return output;
    }
    // Prompts on AMX tiles or the 16-row VNNI GEMM (the `avx512` feature).
    #[cfg(feature = "avx512")]
    if let Some(prompt) =
        crate::cpu_avx512::prompt_linear_group(projections, kinds, x, cols, batch)
    {
        return prompt;
    }
    let nb = cols / 32;
    let kind = kinds[0];
    // Prompts through Q4_0 and Q5_0: repack each four-row panel into the Q8_0 tile layout.
    if batch >= 2 && matches!(kind, 2 | 6) && crate::cpu_kernels::repack_enabled() {
        classic_q8_tiles(projections, kind, x, cols, vnni, &rows, batch, &mut output);
        return output;
    }
    let packed = prepare(x, cols, kind);
    let wide = crate::cpu_kernels::avx512();
    if activation(kind) == Some(Activation::Q8K) {
        // Prompts through Q4_K, Q5_K and Q6_K: eight-row panels with the rows in the vector lanes.
        if batch >= PREDECODE_MIN_BATCH
            && wide.is_none()
            && crate::cpu_kernels::repack_enabled()
            && !r8_disabled()
            && kinds.iter().all(|&k| crate::cpu_kquant::r8_kind(k))
        {
            r8_panels(projections, kinds, &packed, (nb, batch), vnni, &rows, &mut output);
            return output;
        }
        // Wide decode panels take eight rows when every member has an AVX-512 K kernel;
        // otherwise four rows, two tokens per pass.
        let eight = wide.is_some() && batch == 1 && kinds.iter().all(|&k| eight_row_kind(k));
        let size = if eight { 8 } else { 4 };
        // Prompts: a worker decodes its four-row panel once and every token pair reuses it.
        // The AVX-512 tier keeps its own streaming kernels.
        let predecode =
            batch >= PREDECODE_MIN_BATCH && wide.is_none() && crate::cpu_kernels::repack_enabled();
        let decoded = |kind: u32| predecode && PREDECODED_KINDS.contains(&kind);
        let sums = |half: bool| {
            kinds
                .iter()
                .any(|&k| decoded(k) && crate::cpu_kquant::half_sums(k) == half)
                .then(|| crate::cpu_kquant::token_sums(if half { 10 } else { 12 }, &packed))
        };
        let (half_sums, group_sums) = (sums(true), sums(false));
        let blocks = nb / 8;
        crate::cpu_schedule::row_panels(
            &mut output,
            &rows,
            batch,
            size,
            || {
                use crate::cpu_kquant::{KBlock, PANEL_BLOCKS};
                if !kinds.iter().any(|&k| decoded(k)) {
                    return (Vec::new(), Vec::new());
                }
                // A row longer than one chunk keeps its partial sums between chunks.
                let partial = if blocks > PANEL_BLOCKS {
                    vec![[0.0f32; 8]; 4 * batch]
                } else {
                    Vec::new()
                };
                (vec![KBlock::default(); 4 * PANEL_BLOCKS], partial)
            },
            |(panel, partial), member, first, out| {
                let p = &projections[member];
                let (w, kind) = (p.weights, kinds[member]);
                let valid = (p.rows - first).min(size);
                if decoded(kind) {
                    let sums = if crate::cpu_kquant::half_sums(kind) {
                        half_sums.as_deref()
                    } else {
                        group_sums.as_deref()
                    }
                    .expect("sums were made for every predecoded format");
                    // SAFETY: as below; the panel holds four rows of `blocks` decoded blocks and
                    // `sums` one entry per token and block.
                    unsafe {
                        use crate::cpu_kquant as k;
                        for blk0 in (0..blocks).step_by(k::PANEL_BLOCKS) {
                            let count = (blocks - blk0).min(k::PANEL_BLOCKS);
                            k::decode_panel(w, kind, (first, valid), blocks, (blk0, count), panel);
                            let chunk = (blk0, count);
                            if vnni {
                                k::predecoded_vnni(
                                    kind, panel, valid, &packed, sums, nb, batch, chunk, partial,
                                    out,
                                )
                            } else {
                                k::predecoded_avx2(
                                    kind, panel, valid, &packed, sums, nb, batch, chunk, partial,
                                    out,
                                )
                            }
                        }
                    }
                    add_bias(out, p.bias, first, batch);
                    return;
                }
                // SAFETY: `validate` bounds every weight row and activation record the kernels
                // read, `isa`/`avx512` checked their features, and the scheduler gives `out`
                // these output columns exclusively until the joined queue completes.
                unsafe {
                    k_stream_panel(w, kind, wide, vnni, first, valid, &packed, nb, batch, out);
                }
                add_bias(out, p.bias, first, batch);
            },
        );
        return output;
    }
    // (The codebook values are stored as value + 128; `maddubs` would saturate on them, so the
    // plain AVX2 tier keeps the streaming kernel.)
    if kind == 20
        && vnni
        && batch >= PREDECODE_MIN_BATCH
        && cols.is_multiple_of(256)
        && wide.is_none()
        && crate::cpu_kernels::repack_enabled()
        && !r8_disabled()
        && kinds.iter().all(|&k| k == 20)
    {
        r8_panels(projections, kinds, &packed, (nb, batch), vnni, &rows, &mut output);
        return output;
    }
    if matches!(kind, 20 | 39) {
        // IQ4_NL and MXFP4: 512-bit and AVX-VNNI codebook kernels in builds with the `avx512`
        // feature (eight-row decode panels when a 512-bit kernel exists), AVX2 otherwise.
        #[cfg(feature = "avx512")]
        let size = crate::cpu_avx512::codebook::panel_rows(batch);
        #[cfg(not(feature = "avx512"))]
        let size = 4;
        crate::cpu_schedule::row_panels(
            &mut output,
            &rows,
            batch,
            size,
            || (),
            |_, member, first, out| {
                let p = &projections[member];
                let valid = (p.rows - first).min(size);
                // SAFETY: as above; the codebook kernels need AVX2/FMA/F16C at least and the
                // wide ones check their tier.
                unsafe {
                    #[cfg(feature = "avx512")]
                    let done = crate::cpu_avx512::codebook::panel(
                        p.weights, kind, first, valid, &packed, nb, batch, out,
                    );
                    #[cfg(not(feature = "avx512"))]
                    let done = false;
                    if !done {
                        x86::codebook_avx2(p.weights, kind, first, valid, &packed, nb, batch, out)
                    }
                };
                add_bias(out, p.bias, first, batch);
            },
        );
        return output;
    }
    let width = cols / 32 * layout(kind).expect("validated").1;
    if batch == 1 {
        let size = if wide.is_some() { 8 } else { 4 };
        crate::cpu_schedule::row_panels(
            &mut output,
            &rows,
            batch,
            size,
            || (),
            |_, member, first, out| {
                let p = &projections[member];
                // SAFETY: `validate` bounds the weight rows and activation records; the
                // scheduler gives `out` these output columns exclusively.
                unsafe {
                    match wide {
                        Some(Avx512::Vnni) => {
                            x86::gemv512_vnni(p.weights, kind, &packed, first, out.row(0))
                        }
                        Some(Avx512::Bw) => {
                            x86::gemv512_bw(p.weights, kind, &packed, first, out.row(0))
                        }
                        None if vnni => x86::gemv_vnni(p.weights, kind, &packed, first, out.row(0)),
                        None => x86::gemv_avx2(p.weights, kind, &packed, first, out.row(0)),
                    }
                }
                add_bias(out, p.bias, first, batch);
            },
        );
        return output;
    }
    crate::cpu_schedule::row_panels(
        &mut output,
        &rows,
        batch,
        4,
        || vec![W::default(); nb * 4],
        |panel, member, first, out| {
            let p = &projections[member];
            let valid = (p.rows - first).min(4);
            for r in 0..valid {
                // SAFETY: AVX2/F16C were checked by `isa`; the row slice holds `nb` blocks.
                unsafe {
                    x86::unpack_avx2(
                        &p.weights[(first + r) * width..][..width],
                        kind,
                        &mut panel[r * nb..(r + 1) * nb],
                    )
                };
            }
            for r in valid..4 {
                panel[r * nb..(r + 1) * nb].fill(W::default());
            }
            // SAFETY: the panel holds four rows of `nb` records and `packed` `batch * nb`
            // records; `out` owns these output columns.
            unsafe {
                macro_rules! run {
                    ($kind:literal) => {
                        match wide {
                            Some(Avx512::Vnni) => {
                                x86::tile512_vnni::<$kind>(panel, &packed, nb, batch, out)
                            }
                            Some(Avx512::Bw) => {
                                x86::tile512_bw::<$kind>(panel, &packed, nb, batch, out)
                            }
                            None if vnni => x86::tile_vnni::<$kind>(panel, &packed, nb, batch, out),
                            None => x86::tile_avx2::<$kind>(panel, &packed, nb, batch, out),
                        }
                    };
                }
                match kind {
                    2 | 6 => run!(2),
                    3 | 7 => run!(3),
                    _ => unreachable!("W panels hold classic formats only"),
                }
            }
            add_bias(out, p.bias, first, batch);
        },
    );
    output
}

/// `FLLM_NO_R8` turns the eight-row prompt kernels off (for comparisons).
#[cfg(target_arch = "x86_64")]
fn r8_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FLLM_NO_R8").is_some_and(|v| !v.is_empty() && v != "0"))
}

/// Packed activations of one call for the eight-row kernels, per activation layout.
#[cfg(target_arch = "x86_64")]
struct ActSet {
    half: Option<crate::cpu_kquant::Acts>,
    group: Option<crate::cpu_kquant::Acts>,
    q80: Option<crate::cpu_kquant::Acts>,
}

#[cfg(target_arch = "x86_64")]
impl ActSet {
    fn new(kinds: &[u32], packed: &[X], nb: usize) -> Self {
        use crate::cpu_kquant as k;
        let make = |half: bool| {
            kinds
                .iter()
                .any(|&kind| kind != 20 && k::half_sums(kind) == half)
                .then(|| {
                    k::pack_acts(packed, &k::token_sums(if half { 10 } else { 12 }, packed), nb)
                })
        };
        Self {
            half: make(true),
            group: make(false),
            q80: kinds.contains(&20).then(|| k::pack_acts_q8(packed, nb)),
        }
    }

    fn of(&self, kind: u32) -> &crate::cpu_kquant::Acts {
        let acts = if kind == 20 {
            self.q80.as_ref()
        } else if crate::cpu_kquant::half_sums(kind) {
            self.half.as_ref()
        } else {
            self.group.as_ref()
        };
        acts.expect("activations were packed for every format")
    }
}

/// One eight-row panel of `p` against every token of `acts`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn r8_panel(
    p: &Projection<'_>,
    kind: u32,
    acts: &crate::cpu_kquant::Acts,
    (nb, vnni): (usize, bool),
    first: usize,
    scratch: &mut (Vec<crate::cpu_kquant::R8Block>, Vec<[f32; 8]>),
    out: &mut crate::cpu_schedule::Rows<'_>,
    batch: usize,
) {
    use crate::cpu_kquant as k;
    let blocks = nb / 8;
    let (panel, total) = scratch;
    let total = &mut total[..batch];
    let valid = (p.rows - first).min(8);
    total.fill([0.0; 8]);
    // SAFETY: AVX2/F16C (and VNNI when `vnni`) were detected by the caller; `p.weights` holds
    // complete rows through `first + valid` (checked by `validate`) and `acts` one entry per
    // token of `total`.
    unsafe {
        for blk0 in (0..blocks).step_by(k::R8_BLOCKS) {
            let chunk = (blk0, (blocks - blk0).min(k::R8_BLOCKS));
            k::decode_r8(p.weights, kind, (first, valid), blocks, chunk, panel);
            if vnni {
                k::r8_vnni(kind, panel, acts, chunk, total);
            } else {
                k::r8_avx2(kind, panel, acts, chunk, total);
            }
        }
    }
    for (token, lanes) in total.iter().enumerate() {
        for (y, v) in out.row(token).iter_mut().zip(lanes) {
            *y = *v;
        }
    }
    add_bias(out, p.bias, first, batch);
}

/// K-format prompts on eight-row panels: every worker decodes a panel chunk by chunk and runs
/// all tokens through it.
#[cfg(target_arch = "x86_64")]
fn r8_panels(
    projections: &[Projection<'_>],
    kinds: &[u32],
    packed: &[X],
    (nb, batch): (usize, usize),
    vnni: bool,
    rows: &[usize],
    output: &mut [Vec<f32>],
) {
    use crate::cpu_kquant as k;
    let acts = ActSet::new(kinds, packed, nb);
    crate::cpu_schedule::row_panels(
        output,
        rows,
        batch,
        8,
        || {
            (
                vec![k::R8Block::default(); k::R8_BLOCKS],
                vec![[0.0f32; 8]; batch],
            )
        },
        |scratch, member, first, out| {
            let kind = kinds[member];
            r8_panel(
                &projections[member],
                kind,
                acts.of(kind),
                (nb, vnni),
                first,
                scratch,
                out,
                batch,
            );
        },
    );
}

/// One expert of a prompt batch: its three projections and the token rows routed to it.
pub struct ExpertRows<'a> {
    pub experts: ExpertProjections<'a>,
    /// `tokens x cols` inputs of this expert.
    pub input: &'a [f32],
}

/// The gated FFNs of all experts that received tokens in a prompt batch in two joined work
/// queues over every expert's panels (gate and up rows, then the down rows), instead of a
/// fork/join per expert and projection. Outputs are one `tokens x rows` matrix per expert and
/// equal what the per-expert path computes. `None` when an expert does not qualify (formats
/// without an eight-row kernel, AVX-512 hosts, inconsistent shapes); nothing is computed then.
pub fn linear_experts_prompt(
    experts: &[ExpertRows<'_>],
    cols: usize,
    activate: &(dyn Fn(&mut [f32], &[f32]) + Sync),
) -> Option<Vec<Vec<f32>>> {
    #[cfg(target_arch = "x86_64")]
    {
        linear_experts_prompt_x86(experts, cols, activate)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (experts, cols, activate);
        None
    }
}

#[cfg(target_arch = "x86_64")]
fn linear_experts_prompt_x86(
    experts: &[ExpertRows<'_>],
    cols: usize,
    activate: &(dyn Fn(&mut [f32], &[f32]) + Sync),
) -> Option<Vec<Vec<f32>>> {
    use crate::cpu_kquant as k;
    let isa = crate::cpu_kernels::isa();
    if experts.is_empty()
        || isa == "scalar"
        || crate::cpu_kernels::avx512().is_some()
        || !crate::cpu_kernels::repack_enabled()
        || r8_disabled()
        || cols == 0
        || !cols.is_multiple_of(256)
    {
        return None;
    }
    let vnni = isa == "avx-vnni";
    let ff = experts[0].experts.gate.rows;
    for e in experts {
        let [g, u, d] = e.experts.kinds;
        let ok = e.input.len() >= cols
            && e.input.len().is_multiple_of(cols)
            && ff.is_multiple_of(256)
            && e.experts.gate.rows == ff
            && e.experts.up.rows == ff
            && e.experts.down.rows > 0
            && g == u
            && [g, d].iter().all(|&kind| k::r8_kind(kind) && (kind != 20 || vnni))
            && e.experts.gate.bias.is_none()
            && e.experts.up.bias.is_none();
        if !ok {
            return None;
        }
        check_projection(&e.experts.gate, g, cols);
        check_projection(&e.experts.up, u, cols);
        check_projection(&e.experts.down, d, ff);
    }
    let batches = experts
        .iter()
        .map(|e| e.input.len() / cols)
        .collect::<Vec<_>>();
    let widest = batches.iter().copied().max().unwrap_or(1);
    let scratch = || {
        (
            vec![k::R8Block::default(); k::R8_BLOCKS],
            vec![[0.0f32; 8]; widest],
        )
    };
    // gate and up of every expert: activations packed once per expert
    let gate_acts = experts
        .par_iter()
        .map(|e| {
            let packed = prepare(e.input, cols, e.experts.kinds[0]);
            ActSet::new(&e.experts.kinds[..1], &packed, cols / 32)
        })
        .collect::<Vec<_>>();
    let mut hidden = Vec::with_capacity(2 * experts.len());
    let mut rows = Vec::with_capacity(2 * experts.len());
    let mut members = Vec::with_capacity(2 * experts.len());
    let mut member_batches = Vec::with_capacity(2 * experts.len());
    for (i, e) in experts.iter().enumerate() {
        for projection in [&e.experts.gate, &e.experts.up] {
            hidden.push(vec![0.0f32; batches[i] * ff]);
            rows.push(ff);
            members.push((i, projection));
            member_batches.push(batches[i]);
        }
    }
    crate::cpu_schedule::row_panels_ragged(
        &mut hidden,
        &rows,
        &member_batches,
        8,
        scratch,
        |state, member, first, out| {
            let (i, projection) = members[member];
            let kind = experts[i].experts.kinds[0];
            r8_panel(
                projection,
                kind,
                gate_acts[i].of(kind),
                (cols / 32, vnni),
                first,
                state,
                out,
                batches[i],
            );
        },
    );
    // activation per expert: gate = act(gate) * up
    let mut pairs = hidden.chunks_mut(2).collect::<Vec<_>>();
    pairs.par_iter_mut().for_each(|pair| {
        let (gate, up) = pair.split_at_mut(1);
        activate(&mut gate[0], &up[0]);
    });
    let activated = hidden.into_iter().step_by(2).collect::<Vec<_>>();
    // down projections
    let down_acts = experts
        .par_iter()
        .zip(activated.par_iter())
        .map(|(e, act)| {
            let packed = prepare(act, ff, e.experts.kinds[2]);
            ActSet::new(&e.experts.kinds[2..], &packed, ff / 32)
        })
        .collect::<Vec<_>>();
    let mut output = experts
        .iter()
        .enumerate()
        .map(|(i, e)| vec![0.0f32; batches[i] * e.experts.down.rows])
        .collect::<Vec<_>>();
    let rows = experts.iter().map(|e| e.experts.down.rows).collect::<Vec<_>>();
    crate::cpu_schedule::row_panels_ragged(&mut output, &rows, &batches, 8, scratch, |state, i, first, out| {
        let kind = experts[i].experts.kinds[2];
        r8_panel(
            &experts[i].experts.down,
            kind,
            down_acts[i].of(kind),
            (ff / 32, vnni),
            first,
            state,
            out,
            batches[i],
        );
    });
    Some(output)
}

/// One streaming K-format panel (`valid` rows from `first`, every token of `x`) on the widest
/// kernel the CPU has.
///
/// # Safety
/// `w` must hold `first + valid` complete rows of `nb / 8` blocks, `x` must hold `batch * nb`
/// activation records, and `out` must be this panel's exclusive output view.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn k_stream_panel(
    w: &[u8],
    kind: u32,
    wide: Option<crate::cpu_kernels::Avx512>,
    vnni: bool,
    first: usize,
    valid: usize,
    x: &[X],
    nb: usize,
    batch: usize,
    out: &mut crate::cpu_schedule::Rows<'_>,
) {
    use crate::cpu_kernels::Avx512;
    use crate::cpu_kquant as k;
    // Q2_K, Q3_K and IQ4_XS on 512-bit vectors (the `avx512` feature), ahead of the 256-bit AVX-VNNI
    // kernel they would otherwise take on CPUs that have both.
    #[cfg(feature = "avx512")]
    if matches!(kind, 10 | 11 | 23)
        && crate::cpu_avx512::kextra::panel(w, kind, first, valid, x, nb, batch, out)
    {
        return;
    }
    match (kind, wide) {
        (12..=14, Some(Avx512::Vnni)) => k::panel512_vnni(w, kind, first, valid, x, nb, batch, out),
        (12..=14, Some(Avx512::Bw)) => k::panel512_bw(w, kind, first, valid, x, nb, batch, out),
        (12..=14, None) if vnni => k::panel_vnni(w, kind, first, valid, x, nb, batch, out),
        (12..=14, None) => k::panel_avx2(w, kind, first, valid, x, nb, batch, out),
        _ if vnni => k::panel_extra_vnni(w, kind, first, valid, x, nb, batch, out),
        (_, Some(Avx512::Vnni)) => k::panel_extra_vnni512(w, kind, first, valid, x, nb, batch, out),
        _ => k::panel_extra_avx2(w, kind, first, valid, x, nb, batch, out),
    }
}

/// Rows a worker claims at once in the expert passes: long contiguous blocks of one expert's
/// weights per core instead of the 16-row claims of the dense decode.
#[cfg(target_arch = "x86_64")]
const EXPERT_CLAIM_ROWS: usize = 64;

/// Software-prefetch the weights of `rows` rows starting at `first_row` (at most 8 KiB).
///
/// Expert rows are short (a Q4_K down row is 432 bytes), so the hardware prefetcher barely
/// ramps up inside one row before the panel moves on; asking for the next panel while the
/// current one is computed hides most of that latency. Out-of-range requests are clipped.
#[cfg(target_arch = "x86_64")]
#[inline]
fn prefetch_rows(weights: &[u8], first_row: usize, rows: usize, row_bytes: usize) {
    use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
    const MAX_BYTES: usize = 8192;
    let start = first_row.saturating_mul(row_bytes);
    if start >= weights.len() {
        return;
    }
    let end = first_row
        .saturating_add(rows)
        .saturating_mul(row_bytes)
        .min(weights.len())
        .min(start + MAX_BYTES);
    let mut offset = start;
    while offset < end {
        // SAFETY: `offset < weights.len()`; a prefetch is only a hint and cannot fault.
        unsafe { _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(offset).cast::<i8>()) };
        offset += 64;
    }
}

/// One work queue over `count` K-format members at decode (one token). Member `m` reads the
/// activation record row `m` of `packed` (`distinct_inputs`) or row 0 (all members share the
/// input), and its output goes to `output[m]`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn expert_pass<'p, F>(
    member: F,
    count: usize,
    packed: &[X],
    nb: usize,
    distinct_inputs: bool,
    (vnni, size): (bool, usize),
    rows: &mut Vec<usize>,
    output: &mut Vec<Vec<f32>>,
) where
    F: Fn(usize) -> (&'p Projection<'p>, u32) + Sync,
{
    let wide = crate::cpu_kernels::avx512();
    rows.clear();
    rows.extend((0..count).map(|m| member(m).0.rows));
    if output.len() < count {
        output.resize_with(count, Vec::new);
    }
    // The kernels assign every output element; stale values are never read.
    for (out, &stride) in output.iter_mut().zip(rows.iter()) {
        out.resize(stride, 0.0);
    }
    crate::cpu_schedule::row_panels_claim(
        &mut output[..count],
        rows,
        1,
        size,
        EXPERT_CLAIM_ROWS,
        || (),
        |_, m, first, out| {
            let (p, kind) = member(m);
            let valid = (p.rows - first).min(size);
            let record = if distinct_inputs { m } else { 0 };
            let row_bytes = nb / 8 * layout(kind).expect("validated").1;
            prefetch_rows(p.weights, first + size, size, row_bytes);
            // SAFETY: `check_projection` bounded every weight row, `packed` holds `nb` records
            // for each record row used, and the scheduler gives `out` these columns exclusively.
            unsafe {
                k_stream_panel(
                    p.weights,
                    kind,
                    wide,
                    vnni,
                    first,
                    valid,
                    &packed[record * nb..][..nb],
                    nb,
                    1,
                    out,
                );
            }
            add_bias(out, p.bias, first, 1);
        },
    );
}

#[cfg(target_arch = "x86_64")]
fn experts_x86<'s>(
    experts: &[ExpertProjections<'_>],
    input: &[f32],
    cols: usize,
    activate: &dyn Fn(&mut [f32], &[f32]),
    scratch: &'s mut ExpertScratch,
    vnni: bool,
) -> Option<&'s [Vec<f32>]> {
    let count = experts.len();
    let kinds = experts.first()?.kinds;
    let [gate_kind, up_kind, down_kind] = kinds;
    if experts.iter().any(|e| e.kinds != kinds)
        || !kinds
            .iter()
            .all(|&kind| activation(kind) == Some(Activation::Q8K))
    {
        return None;
    }
    assert_eq!(input.len(), cols);
    let ff = experts[0].gate.rows;
    for e in experts {
        assert_eq!(e.up.rows, ff);
        check_projection(&e.gate, gate_kind, cols);
        check_projection(&e.up, up_kind, cols);
        check_projection(&e.down, down_kind, ff);
    }
    // Same panel height as `linear_x86` picks for these members.
    let wide = crate::cpu_kernels::avx512().is_some();
    let size = |kinds: &[u32]| {
        if wide && kinds.iter().all(|k| matches!(k, 12..=14)) {
            8
        } else {
            4
        }
    };
    let ExpertScratch {
        input: packed_input,
        hidden,
        gate_up,
        mid,
        down,
        rows,
    } = scratch;
    // Gate and up of every expert share the input: one packing, one queue.
    prepare_into(input, cols, gate_kind, packed_input);
    expert_pass(
        |m| {
            let e = &experts[m / 2];
            if m % 2 == 0 {
                (&e.gate, gate_kind)
            } else {
                (&e.up, up_kind)
            }
        },
        2 * count,
        packed_input,
        cols / 32,
        false,
        (vnni, size(&[gate_kind, up_kind])),
        rows,
        gate_up,
    );
    // The activations are small (an expert's FFN width); a fork/join per expert would cost
    // more than the arithmetic.
    mid.clear();
    for [gate, up] in gate_up[..2 * count].as_chunks_mut::<2>().0 {
        activate(gate, up);
        mid.extend_from_slice(gate);
    }
    // Every expert's down projection reads its own activated row.
    prepare_into(mid, ff, down_kind, hidden);
    expert_pass(
        |m| (&experts[m].down, down_kind),
        count,
        hidden,
        ff / 32,
        true,
        (vnni, size(&[down_kind])),
        rows,
        down,
    );
    Some(&down[..count])
}

/// Q4_0 and Q5_0 projections of a prompt with the Q8_0 tile kernels.
///
/// Both formats store `d * (q - c)` with a small unsigned `q`, which is `d * w` with a signed
/// byte `w` after the offset: exactly what a Q8_0 weight is. A worker therefore unpacks a
/// four-row panel into the `WPanel` records of the Q8_0 tiles once, and every token pair of
/// the prompt runs the tuned Q8_0 4x2 (AVX2, AVX-VNNI) or row-pair (AVX-512) tile on it, with
/// the Q8_0 activation records (`Q8Input`), which are packed bit-identically to `prepare`.
/// The accumulation order is that of the tiles the classic W-panel path used.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn classic_q8_tiles(
    projections: &[Projection<'_>],
    kind: u32,
    x: &[f32],
    cols: usize,
    vnni: bool,
    rows: &[usize],
    batch: usize,
    output: &mut [Vec<f32>],
) {
    use crate::cpu_kernels::{Avx512, Q8Input, Q8PanelScratch};
    let nb = cols / 32;
    let width = nb * layout(kind).expect("validated").1;
    let input = Q8Input::new(x, cols);
    let wide = crate::cpu_kernels::avx512();
    crate::cpu_schedule::row_panels(
        output,
        rows,
        batch,
        4,
        || Q8PanelScratch::new(nb),
        |panel, member, first, out| {
            let p = &projections[member];
            let valid = (p.rows - first).min(4);
            // SAFETY: `isa`/`avx512` confirmed the features; `validate` bounds every weight row;
            // `Q8Input` holds `batch.div_ceil(2) * nb` pairs; `out` owns these output columns.
            unsafe {
                x86::pack_classic_panel(&p.weights[first * width..], width, kind, valid, panel);
                match wide {
                    Some(Avx512::Vnni) => {
                        crate::cpu_kernels::x86::tile512_vnni(panel, input.packed(), batch, out)
                    }
                    Some(Avx512::Bw) => {
                        crate::cpu_kernels::x86::tile512_bw(panel, input.packed(), batch, out)
                    }
                    None if vnni => {
                        crate::cpu_kernels::x86::tile_vnni(panel, input.packed(), batch, out)
                    }
                    None => crate::cpu_kernels::x86::tile_avx2(panel, input.packed(), batch, out),
                }
            }
            add_bias(out, p.bias, first, batch);
        },
    );
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
fn linear_neon(
    projections: &[Projection<'_>],
    kinds: &[u32],
    x: &[f32],
    cols: usize,
) -> Vec<Vec<f32>> {
    let batch = validate(projections, kinds, x, cols);
    let rows = projections.iter().map(|p| p.rows).collect::<Vec<_>>();
    let mut output = rows
        .iter()
        .map(|&r| vec![0.0; batch * r])
        .collect::<Vec<_>>();
    if batch == 0 {
        return output;
    }
    let nb = cols / 32;
    let packed = prepare(x, cols, kinds[0]);
    let i8mm = crate::cpu_kernels::neon_i8mm();
    if batch == 1 {
        crate::cpu_schedule::row_panels(
            &mut output,
            &rows,
            batch,
            4,
            || (),
            |_, member, first, out| {
                let p = &projections[member];
                let kind = kinds[member];
                // SAFETY: `validate` bounds the weight rows and activation records, dotprod
                // was detected, and `out` owns these output columns until the queue joins.
                unsafe {
                    if crate::cpu_arm::quant::handles(kind) {
                        crate::cpu_arm::quant::gemv(p.weights, kind, &packed, first, out.row(0));
                    } else {
                        neon::gemv(p.weights, kind, &packed, first, out.row(0));
                    }
                };
                add_bias(out, p.bias, first, batch);
            },
        );
        return output;
    }
    crate::cpu_schedule::row_panels(
        &mut output,
        &rows,
        batch,
        4,
        || (vec![W::default(); nb * 4], crate::cpu_arm::quant::Panel::new()),
        |(panel, groups), member, first, out| {
            let p = &projections[member];
            let kind = kinds[member];
            let (block, bytes) = layout(kind).expect("validated");
            let width = cols / block * bytes;
            let valid = (p.rows - first).min(4);
            if crate::cpu_arm::quant::handles(kind) {
                // SAFETY: dotprod (and i8mm for `i8mm`) were detected; `validate` bounds the
                // weight rows and the `batch * nb` activation records; `out` owns these columns.
                unsafe {
                    crate::cpu_arm::quant::prompt_panel(
                        kind,
                        p.weights,
                        width,
                        (first, valid),
                        &packed,
                        (nb, batch),
                        i8mm,
                        groups,
                        out,
                    );
                }
                add_bias(out, p.bias, first, batch);
                return;
            }
            for r in 0..valid {
                // SAFETY: the row slice holds whole blocks and the panel `nb` records per row.
                unsafe {
                    neon::unpack(
                        &p.weights[(first + r) * width..][..width],
                        kind,
                        &mut panel[r * nb..(r + 1) * nb],
                    );
                }
            }
            for r in valid..4 {
                panel[r * nb..(r + 1) * nb].fill(W::default());
            }
            // SAFETY: four panel rows of `nb` records, `batch * nb` activation records, features
            // detected above, `out` owns these columns.
            unsafe {
                if i8mm {
                    neon::tile_i8mm(kind, panel, &packed, nb, batch, out);
                } else {
                    neon::tile(kind, panel, &packed, nb, batch, out);
                }
            }
            add_bias(out, p.bias, first, batch);
        },
    );
    output
}
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{W, X};
    use crate::cpu_kernels::x86::{dot_biased512_bw, dot_biased512_vnni};
    use std::arch::x86_64::*;

    /// Caller supplies a complete 256-value K block and has checked AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn first_signed_max(src: &[f32]) -> (f32, f32) {
        debug_assert_eq!(src.len(), 256);
        let sign = _mm256_set1_ps(-0.0);
        let mut best = _mm256_setzero_ps();
        for values in src.as_chunks::<8>().0 {
            let values = _mm256_loadu_ps(values.as_ptr());
            // MAXPS returns its second operand on NaN, matching the scalar
            // comparison that ignores a NaN instead of replacing the maximum.
            best = _mm256_max_ps(_mm256_andnot_ps(sign, values), best);
        }
        let best = _mm_max_ps(_mm256_castps256_ps128(best), _mm256_extractf128_ps(best, 1));
        let best = _mm_max_ps(best, _mm_movehl_ps(best, best));
        let amax = _mm_cvtss_f32(_mm_max_ss(best, _mm_shuffle_ps(best, best, 0x55)));
        if amax == 0.0 {
            return (0.0, 0.0);
        }
        let wanted = _mm256_set1_ps(amax);
        for values in src.as_chunks::<8>().0 {
            let absolute = _mm256_andnot_ps(sign, _mm256_loadu_ps(values.as_ptr()));
            let mask = _mm256_movemask_ps(_mm256_cmp_ps(absolute, wanted, _CMP_EQ_OQ)) as u32;
            if mask != 0 {
                return (amax, values[mask.trailing_zeros() as usize]);
            }
        }
        unreachable!("nonzero maximum must have a first matching element");
    }
    /// Unpack one row of classic blocks into W records (unsigned weights + 128).
    ///
    /// # Safety
    /// The CPU must support AVX2/F16C; `raw` holds whole blocks of `kind` (Q4_0, Q4_1, Q5_0 or
    /// Q5_1) and `groups` one record per block.
    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    pub(super) unsafe fn unpack_avx2(raw: &[u8], kind: u32, groups: &mut [W]) {
        let width = match kind {
            2 => 18,
            3 => 20,
            6 => 22,
            7 => 24,
            _ => unreachable!("W panels hold classic formats only"),
        };
        let affine = matches!(kind, 3 | 7);
        let bias = if affine {
            -128
        } else if matches!(kind, 6 | 7) {
            112
        } else {
            120
        };
        for (b, dst) in raw.chunks_exact(width).zip(groups.iter_mut()) {
            let (q, d, m) = decode_classic_group(b, kind);
            _mm256_storeu_si256(
                dst.q.as_mut_ptr().cast(),
                _mm256_add_epi8(q, _mm256_set1_epi8(bias)),
            );
            dst.d = [d; 2];
            dst.m = [m, 0.0];
        }
    }
    /// Unpack `valid` (1..=4) rows of Q4_0 / Q5_0 blocks, `stride` bytes apart, into the Q8_0
    /// tile records: signed weight plus 128 as a byte, and the block scale. Missing rows become
    /// zero weights.
    ///
    /// # Safety
    /// The CPU must support AVX2/F16C; `rows` holds `valid` rows of `panel.len()` blocks of
    /// `kind` (Q4_0 or Q5_0), `stride` bytes apart.
    #[target_feature(enable = "avx2,f16c")]
    pub(super) unsafe fn pack_classic_panel(
        rows: &[u8],
        stride: usize,
        kind: u32,
        valid: usize,
        panel: &mut [crate::cpu_kernels::WPanel],
    ) {
        let (width, bias) = match kind {
            2 => (18, 120),
            6 => (22, 112),
            _ => unreachable!("Q4_0 and Q5_0 only"),
        };
        for (kb, record) in panel.iter_mut().enumerate() {
            for r in 0..valid {
                let (q, d, _) =
                    decode_classic_group(&rows[r * stride + kb * width..][..width], kind);
                _mm256_store_si256(
                    record.q[r].as_mut_ptr().cast(),
                    _mm256_add_epi8(q, _mm256_set1_epi8(bias)),
                );
                record.d[r] = d;
            }
            for r in valid..4 {
                record.q[r] = [128; 32];
                record.d[r] = 0.0;
            }
        }
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum(v: __m256) -> f32 {
        let t = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55)))
    }
    // Constant format classes eliminate the minimum correction for symmetric weights.
    macro_rules! add_row {
        ($dot:ident, $r:literal, $a:expr, $offset:expr, $wr:expr, $xr:expr, $q:expr, $init:expr, $kind:expr) => {{
            let wr = $wr;
            let xr = $xr;
            let scale = _mm256_set1_ps(wr.d[0] * xr.d);
            let dot =
                crate::cpu_kernels::x86::$dot($init, _mm256_loadu_si256(wr.q.as_ptr().cast()), $q);
            $a[$r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), scale, $a[$r]);
            if $kind == 3 || $kind == 7 {
                $offset[$r] = wr.m[0].mul_add(xr.sum[0], $offset[$r]);
            }
        }};
    }
    macro_rules! tile {
        ($name:ident, $features:literal, $dot:ident) => {
            #[target_feature(enable = $features)]
            pub(super) unsafe fn $name<const KIND: u32>(
                w: &[W],
                x: &[X],
                nb: usize,
                batch: usize,
                out: &mut crate::cpu_schedule::Rows<'_>,
            ) {
                for b in (0..batch).step_by(2) {
                    let mut a = [[_mm256_setzero_ps(); 4]; 2];
                    let mut offsets = [[0.0_f32; 4]; 2];
                    for kb in 0..nb {
                        let x0 = &x[b * nb + kb];
                        let x1 = &x[(b + 1).min(batch - 1) * nb + kb];
                        let q0 = _mm256_load_si256(x0.q.as_ptr().cast());
                        let q1 = _mm256_load_si256(x1.q.as_ptr().cast());
                        let i0 = _mm256_setr_epi32(
                            x0.init[0], x0.init[0], x0.init[0], x0.init[0], x0.init[1], x0.init[1],
                            x0.init[1], x0.init[1],
                        );
                        let i1 = _mm256_setr_epi32(
                            x1.init[0], x1.init[0], x1.init[0], x1.init[0], x1.init[1], x1.init[1],
                            x1.init[1], x1.init[1],
                        );
                        macro_rules! row {
                            ($r:literal) => {
                                let wr = &w[$r * nb + kb];
                                add_row!($dot, $r, a[0], offsets[0], wr, x0, q0, i0, KIND);
                                add_row!($dot, $r, a[1], offsets[1], wr, x1, q1, i1, KIND);
                            };
                        }
                        row!(0);
                        row!(1);
                        row!(2);
                        row!(3);
                    }
                    for j in 0..(batch - b).min(2) {
                        for (r, y) in out.row(b + j).iter_mut().enumerate() {
                            *y = hsum(a[j][r]) + offsets[j][r];
                        }
                    }
                }
            }
        };
    }
    tile!(tile_avx2, "avx2,fma", dot_centered_avx2);
    tile!(tile_vnni, "avx2,fma,avxvnni", dot_vnni);

    /// Raw unsigned classic nibbles (below 32) by signed activations: a maddubs pair is at most
    /// 2 * 31 * 127 and cannot saturate.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_small_avx2(init: __m256i, u: __m256i, s: __m256i) -> __m256i {
        let pairs = _mm256_maddubs_epi16(u, s);
        _mm256_add_epi32(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)), init)
    }

    /// Symmetric raw quants have an offset. Build its bias per four-value dot lane;
    /// X.init stores whole 16-value sums, so it cannot be used as lane init.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn signed_lane_bias<const SHIFT: i32>(qx: __m256i) -> __m256i {
        let pairs = _mm256_maddubs_epi16(_mm256_set1_epi8(1), qx);
        let groups = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
        _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32::<SHIFT>(groups))
    }

    // Decode directly into a register. The raw unsigned nibble values
    // permit one maddubs/madd dot without signed-weight conversion or a W panel.
    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn decode_classic_group(b: &[u8], kind: u32) -> (__m256i, f32, f32) {
        let affine = matches!(kind, 3 | 7);
        let five = matches!(kind, 6 | 7);
        let off = if affine { 4 } else { 2 };
        let qp = b.as_ptr().add(off + if five { 4 } else { 0 });
        let packed = _mm_loadu_si128(qp.cast());
        let lo = _mm_and_si128(packed, _mm_set1_epi8(15));
        let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(15));
        let mut q = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        if five {
            let bits = _mm_cvtsi32_si128(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()));
            let lo = _mm_shuffle_epi8(
                bits,
                _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1),
            );
            let hi = _mm_shuffle_epi8(
                bits,
                _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3),
            );
            let bits = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
            let masks = _mm256_setr_epi8(
                1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64,
                -128, 1, 2, 4, 8, 16, 32, 64, -128,
            );
            let high = _mm256_and_si256(
                _mm256_cmpeq_epi8(_mm256_and_si256(bits, masks), masks),
                _mm256_set1_epi8(16),
            );
            q = _mm256_or_si256(q, high);
        }
        let d = crate::cpu_kernels::x86::half_to_float(u16::from_le_bytes([b[0], b[1]]));
        let m = if affine {
            crate::cpu_kernels::x86::half_to_float(u16::from_le_bytes([b[2], b[3]]))
        } else {
            0.0
        };
        (q, d, m)
    }

    macro_rules! gemv {
        ($name:ident, $body:ident, $features:literal, $small:ident) => {
            /// `out.len()` (1..=4) classic-format rows from `first` for one decode token.
            ///
            /// # Safety
            /// The CPU must support the enabled features; `w` holds complete rows of `x.len()`
            /// blocks of `kind` through row `first + out.len()`.
            #[target_feature(enable = $features)]
            pub(super) unsafe fn $name(
                w: &[u8],
                kind: u32,
                x: &[X],
                first: usize,
                out: &mut [f32],
            ) {
                macro_rules! run {
                    ($kind:literal) => {
                        match out.len() {
                            1 => $body::<$kind, 1>(w, x, first, out),
                            2 => $body::<$kind, 2>(w, x, first, out),
                            3 => $body::<$kind, 3>(w, x, first, out),
                            4 => $body::<$kind, 4>(w, x, first, out),
                            _ => unreachable!(),
                        }
                    };
                }
                match kind {
                    2 => run!(2),
                    3 => run!(3),
                    6 => run!(6),
                    7 => run!(7),
                    _ => unreachable!("classic formats only"),
                }
            }
            #[inline]
            #[target_feature(enable = $features)]
            unsafe fn $body<const KIND: u32, const ROWS: usize>(
                w: &[u8],
                x: &[X],
                first: usize,
                out: &mut [f32],
            ) {
                let width = match KIND {
                    2 => 18,
                    3 => 20,
                    6 => 22,
                    _ => 24,
                };
                let blocks = x.len();
                let stride = blocks * width;
                let mut a = [_mm256_setzero_ps(); 4];
                let mut offsets = [0.0_f32; 4];
                for (block, xr) in x.iter().enumerate() {
                    // Raw unsigned nibbles go straight into the integer dot; the offset of the
                    // symmetric formats enters as the dot's initial value.
                    let qx = _mm256_load_si256(xr.q.as_ptr().cast());
                    let init = if KIND == 2 {
                        signed_lane_bias::<3>(qx)
                    } else if KIND == 6 {
                        signed_lane_bias::<4>(qx)
                    } else {
                        _mm256_setzero_si256()
                    };
                    for r in 0..ROWS {
                        let start = (first + r) * stride + block * width;
                        let (wq, d, m) = decode_classic_group(&w[start..start + width], KIND);
                        let dot = $small(init, wq, qx);
                        a[r] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(dot),
                            _mm256_set1_ps(d * xr.d),
                            a[r],
                        );
                        if KIND == 3 || KIND == 7 {
                            offsets[r] = m.mul_add(xr.sum[0], offsets[r]);
                        }
                    }
                }
                for (r, y) in out.iter_mut().enumerate() {
                    *y = hsum(a[r]) + offsets[r];
                }
            }
        };
    }
    /// `vpdpbusd` on raw unsigned nibbles (below 32) and signed activations: the products are
    /// summed in 32 bits, so nothing saturates.
    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn dot_small_vnni(init: __m256i, u: __m256i, s: __m256i) -> __m256i {
        _mm256_dpbusd_avx_epi32(init, u, s)
    }
    gemv!(gemv_avx2, gemv_avx2_format, "avx2,fma,f16c", dot_small_avx2);
    gemv!(
        gemv_vnni,
        gemv_vnni_format,
        "avx2,fma,f16c,avxvnni",
        dot_small_vnni
    );
    /// Unsigned weights below 32 by signed activations; pairs stay far below `vpmaddubsw`
    /// saturation (2 * 31 * 127).
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn dot_small512_bw(init: __m512i, u: __m512i, s: __m512i) -> __m512i {
        let pairs = _mm512_maddubs_epi16(u, s);
        _mm512_add_epi32(_mm512_madd_epi16(pairs, _mm512_set1_epi16(1)), init)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn dot_small512_vnni(init: __m512i, u: __m512i, s: __m512i) -> __m512i {
        _mm512_dpbusd_epi32(init, u, s)
    }
    /// 512-bit counterpart of `signed_lane_bias`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn signed_lane_bias512<const SHIFT: u32>(qx: __m512i) -> __m512i {
        let pairs = _mm512_maddubs_epi16(_mm512_set1_epi8(1), qx);
        let groups = _mm512_madd_epi16(pairs, _mm512_set1_epi16(1));
        _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_slli_epi32::<SHIFT>(groups))
    }

    // Wide decode for Q4_0/Q4_1/Q5_0/Q5_1: two consecutive blocks of a row share one 512-bit
    // dot, and up to eight rows reuse each activation pair. Weights stay raw unsigned nibbles;
    // the symmetric formats fold their offset into the lane bias as on AVX2.
    macro_rules! gemv512 {
        ($name:ident, $body:ident, $features:literal, $dot:ident) => {
            /// `out.len()` (1..=8) classic-format rows from `first` for one decode token.
            ///
            /// # Safety
            /// The CPU must support the enabled features; `w` holds complete rows of `x.len()`
            /// blocks of `kind` through row `first + out.len()`.
            #[target_feature(enable = $features)]
            pub(super) unsafe fn $name(
                w: &[u8],
                kind: u32,
                x: &[X],
                first: usize,
                out: &mut [f32],
            ) {
                macro_rules! run {
                    ($kind:literal) => {
                        match out.len() {
                            1 => $body::<$kind, 1>(w, x, first, out),
                            2 => $body::<$kind, 2>(w, x, first, out),
                            3 => $body::<$kind, 3>(w, x, first, out),
                            4 => $body::<$kind, 4>(w, x, first, out),
                            5 => $body::<$kind, 5>(w, x, first, out),
                            6 => $body::<$kind, 6>(w, x, first, out),
                            7 => $body::<$kind, 7>(w, x, first, out),
                            8 => $body::<$kind, 8>(w, x, first, out),
                            _ => unreachable!(),
                        }
                    };
                }
                match kind {
                    2 => run!(2),
                    3 => run!(3),
                    6 => run!(6),
                    7 => run!(7),
                    _ => unreachable!(),
                }
            }
            #[inline]
            #[target_feature(enable = $features)]
            unsafe fn $body<const KIND: u32, const ROWS: usize>(
                w: &[u8],
                x: &[X],
                first: usize,
                out: &mut [f32],
            ) {
                let width = match KIND {
                    2 => 18,
                    3 => 20,
                    6 => 22,
                    _ => 24,
                };
                let blocks = x.len();
                let stride = blocks * width;
                let affine = KIND == 3 || KIND == 7;
                let mut acc = [_mm512_setzero_ps(); ROWS];
                let mut offsets = [0.0_f32; ROWS];
                for block in (0..blocks).step_by(2) {
                    let pair = block + 1 < blocks;
                    let xa = &x[block];
                    let xb = &x[block + usize::from(pair)];
                    let low = _mm256_load_si256(xa.q.as_ptr().cast());
                    // An odd final block runs with zero upper activations and scale.
                    let qx = if pair {
                        crate::cpu_kernels::x86::join256(
                            low,
                            _mm256_load_si256(xb.q.as_ptr().cast()),
                        )
                    } else {
                        _mm512_zextsi256_si512(low)
                    };
                    let init = if KIND == 2 {
                        signed_lane_bias512::<3>(qx)
                    } else if KIND == 6 {
                        signed_lane_bias512::<4>(qx)
                    } else {
                        _mm512_setzero_si512()
                    };
                    for r in 0..ROWS {
                        let start = (first + r) * stride + block * width;
                        let (q0, d0, m0) = decode_classic_group(&w[start..start + width], KIND);
                        let (wq, d1, m1) = if pair {
                            let next = start + width;
                            let (q1, d1, m1) = decode_classic_group(&w[next..next + width], KIND);
                            (crate::cpu_kernels::x86::join256(q0, q1), d1, m1)
                        } else {
                            (_mm512_zextsi256_si512(q0), 0.0, 0.0)
                        };
                        let scale = crate::cpu_kernels::x86::halves_ps(d0 * xa.d, d1 * xb.d);
                        acc[r] =
                            _mm512_fmadd_ps(_mm512_cvtepi32_ps($dot(init, wq, qx)), scale, acc[r]);
                        if affine {
                            offsets[r] = m0.mul_add(xa.sum[0], offsets[r]);
                            if pair {
                                offsets[r] = m1.mul_add(xb.sum[0], offsets[r]);
                            }
                        }
                    }
                }
                for (r, y) in out.iter_mut().enumerate() {
                    *y = _mm512_reduce_add_ps(acc[r]) + offsets[r];
                }
            }
        };
    }
    gemv512!(
        gemv512_bw,
        gemv512_bw_format,
        "avx2,fma,f16c,avx512f,avx512bw,avx512vl",
        dot_small512_bw
    );
    gemv512!(
        gemv512_vnni,
        gemv512_vnni_format,
        "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni",
        dot_small512_vnni
    );

    // Wide prompt tile over an unpacked W panel: rows pair up into 512-bit vectors against
    // each token's activation record broadcast to both halves. Only the classic formats reach
    // this tile (K formats use `cpu_kquant`), so the half-scales of a record are equal.
    macro_rules! tile512 {
        ($name:ident, $features:literal, $dot:ident) => {
            /// # Safety
            /// The CPU must support the enabled features; `w` holds a four-row panel of `nb`
            /// records per row and `x` holds `batch * nb` activation records.
            #[target_feature(enable = $features)]
            pub(super) unsafe fn $name<const KIND: u32>(
                w: &[W],
                x: &[X],
                nb: usize,
                batch: usize,
                out: &mut crate::cpu_schedule::Rows<'_>,
            ) {
                debug_assert!(KIND == 2 || KIND == 3);
                for b in (0..batch).step_by(2) {
                    let mut a = [[_mm512_setzero_ps(); 2]; 2];
                    let mut offsets = [[0.0_f32; 4]; 2];
                    for kb in 0..nb {
                        let xs = [&x[b * nb + kb], &x[(b + 1).min(batch - 1) * nb + kb]];
                        let wr = [&w[kb], &w[nb + kb], &w[2 * nb + kb], &w[3 * nb + kb]];
                        let weights = [
                            crate::cpu_kernels::x86::join256(
                                _mm256_load_si256(wr[0].q.as_ptr().cast()),
                                _mm256_load_si256(wr[1].q.as_ptr().cast()),
                            ),
                            crate::cpu_kernels::x86::join256(
                                _mm256_load_si256(wr[2].q.as_ptr().cast()),
                                _mm256_load_si256(wr[3].q.as_ptr().cast()),
                            ),
                        ];
                        for t in 0..2 {
                            let xr = xs[t];
                            let qx =
                                _mm512_broadcast_i64x4(_mm256_load_si256(xr.q.as_ptr().cast()));
                            let half_init = _mm256_setr_epi32(
                                xr.init[0], xr.init[0], xr.init[0], xr.init[0], xr.init[1],
                                xr.init[1], xr.init[1], xr.init[1],
                            );
                            let init = _mm512_broadcast_i64x4(half_init);
                            for k in 0..2 {
                                let scale = crate::cpu_kernels::x86::halves_ps(
                                    wr[2 * k].d[0] * xr.d,
                                    wr[2 * k + 1].d[0] * xr.d,
                                );
                                a[t][k] = _mm512_fmadd_ps(
                                    _mm512_cvtepi32_ps($dot(init, weights[k], qx)),
                                    scale,
                                    a[t][k],
                                );
                            }
                            if KIND == 3 {
                                for r in 0..4 {
                                    offsets[t][r] = wr[r].m[0].mul_add(xr.sum[0], offsets[t][r]);
                                }
                            }
                        }
                    }
                    for j in 0..(batch - b).min(2) {
                        let (s0, s1) = crate::cpu_kernels::x86::hsum_halves(a[j][0]);
                        let (s2, s3) = crate::cpu_kernels::x86::hsum_halves(a[j][1]);
                        for (r, y) in out.row(b + j).iter_mut().enumerate() {
                            *y = [s0, s1, s2, s3][r] + offsets[j][r];
                        }
                    }
                }
            }
        };
    }
    tile512!(
        tile512_bw,
        "avx2,fma,f16c,avx512f,avx512bw,avx512vl",
        dot_biased512_bw
    );
    tile512!(
        tile512_vnni,
        "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni",
        dot_biased512_vnni
    );

    /// Signed MXFP4 (E2M1) values at twice their magnitude; `e8m0_half` halves the scale.
    const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    /// E8M0 exponent as 2^(x - 128), the half scale matching `MXFP4_VALUES` (as `quant_extra`).
    #[inline]
    fn e8m0_half(x: u8) -> f32 {
        f32::from_bits(if x < 2 {
            0x0020_0000u32 << x
        } else {
            (u32::from(x) - 1) << 23
        })
    }

    /// IQ4_NL (fp16 scale) and MXFP4 (E8M0 scale) projections: 32-value blocks of 4-bit
    /// codebook indices, low nibbles first, against Q8_0 activation records. Each row's block
    /// is decoded once per token pair with one in-lane table lookup.
    ///
    /// # Safety
    /// The CPU must support AVX2/FMA/F16C; `w` holds `first + rows` (1..=4) complete rows of
    /// `nb` blocks and `x` holds `batch * nb` activation records.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn codebook_avx2(
        w: &[u8],
        kind: u32,
        first: usize,
        rows: usize,
        x: &[X],
        nb: usize,
        batch: usize,
        out: &mut crate::cpu_schedule::Rows<'_>,
    ) {
        macro_rules! by_rows {
            ($kind:literal) => {{
                let mut token = 0;
                while token < batch {
                    let tokens = (batch - token).min(2);
                    match (rows, tokens) {
                        (1, 1) => codebook_rows::<$kind, 1, 1>(w, first, x, nb, token, out),
                        (2, 1) => codebook_rows::<$kind, 2, 1>(w, first, x, nb, token, out),
                        (3, 1) => codebook_rows::<$kind, 3, 1>(w, first, x, nb, token, out),
                        (4, 1) => codebook_rows::<$kind, 4, 1>(w, first, x, nb, token, out),
                        (1, _) => codebook_rows::<$kind, 1, 2>(w, first, x, nb, token, out),
                        (2, _) => codebook_rows::<$kind, 2, 2>(w, first, x, nb, token, out),
                        (3, _) => codebook_rows::<$kind, 3, 2>(w, first, x, nb, token, out),
                        (4, _) => codebook_rows::<$kind, 4, 2>(w, first, x, nb, token, out),
                        _ => unreachable!("panel has 1 to 4 rows"),
                    }
                    token += tokens;
                }
            }};
        }
        match kind {
            20 => by_rows!(20),
            39 => by_rows!(39),
            _ => unreachable!("not a codebook format"),
        }
    }
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn codebook_rows<const KIND: u32, const R: usize, const T: usize>(
        w: &[u8],
        first: usize,
        x: &[X],
        nb: usize,
        token: usize,
        out: &mut crate::cpu_schedule::Rows<'_>,
    ) {
        let width = if KIND == 20 { 18 } else { 17 };
        let stride = nb * width;
        assert!(w.len() >= (first + R) * stride);
        let values = if KIND == 20 {
            crate::cpu_kquant::IQ4_VALUES
        } else {
            MXFP4_VALUES
        };
        let table = _mm256_broadcastsi128_si256(_mm_loadu_si128(values.as_ptr().cast()));
        let mask = _mm_set1_epi8(15);
        let ones = _mm256_set1_epi16(1);
        let mut acc = [[_mm256_setzero_ps(); T]; R];
        for kb in 0..nb {
            let mut xs = [&x[token * nb + kb]; T];
            let mut qx = [_mm256_setzero_si256(); T];
            for t in 0..T {
                xs[t] = &x[(token + t) * nb + kb];
                qx[t] = _mm256_load_si256(xs[t].q.as_ptr().cast());
            }
            for (r, acc) in acc.iter_mut().enumerate() {
                let b = w.as_ptr().add((first + r) * stride + kb * width);
                let (packed, d) = if KIND == 20 {
                    (
                        _mm_loadu_si128(b.add(2).cast()),
                        crate::cpu_kernels::x86::half_to_float(std::ptr::read_unaligned(b.cast())),
                    )
                } else {
                    (_mm_loadu_si128(b.add(1).cast()), e8m0_half(*b))
                };
                let index = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(_mm_and_si128(packed, mask)),
                    _mm_and_si128(_mm_srli_epi16(packed, 4), mask),
                    1,
                );
                let q = _mm256_shuffle_epi8(table, index);
                // Signed by signed: |w| <= 127 and |x| <= 127 keep maddubs pairs unsaturated.
                let magnitude = _mm256_sign_epi8(q, q);
                for ((acc, &qx), xr) in acc.iter_mut().zip(&qx).zip(xs) {
                    let dot = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(magnitude, _mm256_sign_epi8(qx, q)),
                        ones,
                    );
                    *acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), _mm256_set1_ps(d * xr.d), *acc);
                }
            }
        }
        for (r, acc) in acc.iter().enumerate() {
            for (t, &acc) in acc.iter().enumerate() {
                out.row(token + t)[r] = hsum(acc);
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon {
    use super::{W, X};
    use crate::cpu_kernels::neon::{half_to_float, smmla, zip_hi, zip_lo};
    use crate::cpu_schedule::Rows;
    use std::arch::aarch64::*;

    /// Caller supplies a complete 256-value K block. Returns the first absolute maximum and its
    /// signed value, ignoring NaN like the scalar loop.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn first_signed_max(src: &[f32]) -> (f32, f32) {
        debug_assert_eq!(src.len(), 256);
        let mut best = vdupq_n_f32(0.0);
        for values in src.as_chunks::<4>().0 {
            best = vmaxnmq_f32(best, vabsq_f32(vld1q_f32(values.as_ptr())));
        }
        let amax = vmaxnmvq_f32(best);
        if amax == 0.0 {
            return (0.0, 0.0);
        }
        let wanted = vdupq_n_f32(amax);
        for values in src.as_chunks::<4>().0 {
            let hit = vceqq_f32(vabsq_f32(vld1q_f32(values.as_ptr())), wanted);
            if vmaxvq_u32(hit) != 0 {
                if let Some(&v) = values.iter().find(|v| v.abs() == amax) {
                    return (amax, v);
                }
            }
        }
        unreachable!("nonzero maximum must have a first matching element");
    }

    const fn block_width(kind: u32) -> usize {
        match kind {
            2 => 18,
            3 => 20,
            6 => 22,
            7 => 24,
            12 => 144,
            13 => 176,
            _ => 210,
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn half(raw: &[u8], at: usize) -> f32 {
        half_to_float(u16::from_le_bytes([raw[at], raw[at + 1]]))
    }

    /// Super-block scales, decoded once per row block: `d`, and the minimum scale (Q4_K, Q5_K)
    /// or the affine offset (Q4_1, Q5_1).
    #[derive(Clone, Copy)]
    struct Header {
        d: f32,
        min: f32,
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn header<const KIND: u32>(raw: &[u8]) -> Header {
        match KIND {
            14 => Header {
                d: half(raw, 208),
                min: 0.0,
            },
            3 | 7 | 12 | 13 => Header {
                d: half(raw, 0),
                min: half(raw, 2),
            },
            _ => Header {
                d: half(raw, 0),
                min: 0.0,
            },
        }
    }

    /// One 32-value group: signed weights (values 0..16, 16..32), the two 16-value scales and
    /// the coefficients of the activation half sums: the values `compute::dequantize` defines.
    #[derive(Clone, Copy)]
    struct Group {
        q: [int8x16_t; 2],
        d: [f32; 2],
        m: [f32; 2],
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn group<const KIND: u32>(raw: &[u8], h: Header, g: usize) -> Group {
        let p = raw.as_ptr();
        let low4 = vdupq_n_u8(15);
        let signed =
            |v: uint8x16_t, offset: u8| vreinterpretq_s8_u8(vsubq_u8(v, vdupq_n_u8(offset)));
        match KIND {
            2 | 3 | 6 | 7 => {
                let affine = KIND == 3 || KIND == 7;
                let five = KIND == 6 || KIND == 7;
                let off = if affine { 4 } else { 2 };
                let packed = vld1q_u8(p.add(off + if five { 4 } else { 0 }));
                let mut lo = vandq_u8(packed, low4);
                let mut hi = vshrq_n_u8::<4>(packed);
                if five {
                    // Bit j of the 32-bit word is the fifth bit of value j: replicate byte j / 8
                    // over eight lanes and test one bit per lane.
                    let bits =
                        u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                    let bytes = vreinterpretq_u8_u32(vdupq_n_u32(bits));
                    let masks = vreinterpretq_u8_u64(vdupq_n_u64(0x8040_2010_0804_0201));
                    let sixteen = vdupq_n_u8(16);
                    let select_lo = vcombine_u8(vdup_n_u8(0), vdup_n_u8(1));
                    let select_hi = vcombine_u8(vdup_n_u8(2), vdup_n_u8(3));
                    let high_lo = vtstq_u8(vqtbl1q_u8(bytes, select_lo), masks);
                    let high_hi = vtstq_u8(vqtbl1q_u8(bytes, select_hi), masks);
                    lo = vorrq_u8(lo, vandq_u8(high_lo, sixteen));
                    hi = vorrq_u8(hi, vandq_u8(high_hi, sixteen));
                }
                let offset = if affine {
                    0
                } else if five {
                    16
                } else {
                    8
                };
                Group {
                    q: [signed(lo, offset), signed(hi, offset)],
                    d: [h.d; 2],
                    m: [if affine { h.min } else { 0.0 }, 0.0],
                }
            }
            12 | 13 => {
                let scales = &raw[4..16];
                let (s, m) = if g < 4 {
                    (scales[g] & 63, scales[g + 4] & 63)
                } else {
                    (
                        (scales[g + 4] & 15) | ((scales[g - 4] >> 6) << 4),
                        (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4),
                    )
                };
                let qs = p.add(if KIND == 13 { 48 } else { 16 } + g / 2 * 32);
                let (a, b) = (vld1q_u8(qs), vld1q_u8(qs.add(16)));
                let (mut lo, mut hi) = if g.is_multiple_of(2) {
                    (vandq_u8(a, low4), vandq_u8(b, low4))
                } else {
                    (vshrq_n_u8::<4>(a), vshrq_n_u8::<4>(b))
                };
                if KIND == 13 {
                    let shift = vdupq_n_s8(-(g as i8));
                    let one = vdupq_n_u8(1);
                    let bit = |v: uint8x16_t| vshlq_n_u8::<4>(vandq_u8(vshlq_u8(v, shift), one));
                    lo = vorrq_u8(lo, bit(vld1q_u8(p.add(16))));
                    hi = vorrq_u8(hi, bit(vld1q_u8(p.add(32))));
                }
                Group {
                    q: [signed(lo, 0), signed(hi, 0)],
                    d: [h.d * s as f32; 2],
                    m: [-h.min * m as f32; 2],
                }
            }
            _ => {
                let (n, sub) = (g / 4, g % 4);
                let lp = p.add(n * 64 + (sub % 2) * 32);
                let (a, b) = (vld1q_u8(lp), vld1q_u8(lp.add(16)));
                let (lo, hi) = if sub < 2 {
                    (vandq_u8(a, low4), vandq_u8(b, low4))
                } else {
                    (vshrq_n_u8::<4>(a), vshrq_n_u8::<4>(b))
                };
                let hp = p.add(128 + n * 32);
                let shift = vdupq_n_s8(-((2 * sub) as i8));
                let three = vdupq_n_u8(3);
                let bits = |v: uint8x16_t| vshlq_n_u8::<4>(vandq_u8(vshlq_u8(v, shift), three));
                let lo = vorrq_u8(lo, bits(vld1q_u8(hp)));
                let hi = vorrq_u8(hi, bits(vld1q_u8(hp.add(16))));
                let at = 192 + n * 8 + sub * 2;
                Group {
                    q: [signed(lo, 32), signed(hi, 32)],
                    d: [h.d * raw[at] as i8 as f32, h.d * raw[at + 1] as i8 as f32],
                    m: [0.0; 2],
                }
            }
        }
    }

    /// Minimum / affine correction of one group, in the order of the x86 kernels.
    #[inline]
    fn offset<const KIND: u32>(acc: &mut f32, m: [f32; 2], xr: &X) {
        if KIND == 3 || KIND == 7 {
            *acc = m[0].mul_add(xr.sum[0], *acc);
        } else if KIND == 12 || KIND == 13 {
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
        if KIND == 14 {
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

    /// W records hold unsigned weights + 128; flip them back to signed values.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn panel_weights(wr: &W) -> [int8x16_t; 2] {
        let flip = vdupq_n_u8(0x80);
        [
            vreinterpretq_s8_u8(veorq_u8(vld1q_u8(wr.q.as_ptr()), flip)),
            vreinterpretq_s8_u8(veorq_u8(vld1q_u8(wr.q.as_ptr().add(16)), flip)),
        ]
    }

    macro_rules! by_kind {
        ($kind:expr, $run:ident) => {
            match $kind {
                2 => $run!(2),
                3 => $run!(3),
                6 => $run!(6),
                7 => $run!(7),
                12 => $run!(12),
                13 => $run!(13),
                14 => $run!(14),
                _ => unreachable!(),
            }
        };
    }

    /// Unpack one row of `kind` blocks into W records (unsigned weights + 128, as on x86).
    ///
    /// # Safety
    /// `raw` holds whole blocks and `out` one record per 32 values.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn unpack(raw: &[u8], kind: u32, out: &mut [W]) {
        macro_rules! run {
            ($kind:literal) => {
                unpack_kind::<$kind>(raw, out)
            };
        }
        by_kind!(kind, run)
    }
    // `as_chunks` needs the block width as a constant expression of `KIND`, which stable
    // Rust cannot name.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn unpack_kind<const KIND: u32>(raw: &[u8], out: &mut [W]) {
        let groups = if KIND >= 12 { 8 } else { 1 };
        let flip = vdupq_n_u8(0x80);
        for (block, dst) in raw
            .chunks_exact(block_width(KIND))
            .zip(out.chunks_mut(groups))
        {
            let h = header::<KIND>(block);
            for (g, w) in dst.iter_mut().enumerate() {
                let group = group::<KIND>(block, h, g);
                for (i, q) in group.q.into_iter().enumerate() {
                    vst1q_u8(
                        w.q.as_mut_ptr().add(16 * i),
                        veorq_u8(vreinterpretq_u8_s8(q), flip),
                    );
                }
                w.d = group.d;
                w.m = group.m;
            }
        }
    }

    /// `out.len()` (1..=4) rows from `first` for one decode token, decoding straight into
    /// registers.
    ///
    /// # Safety
    /// The CPU must support NEON and dotprod; `w` holds complete rows of `kind` blocks covering
    /// `x.len()` records through row `first + out.len()`.
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn gemv(w: &[u8], kind: u32, x: &[X], first: usize, out: &mut [f32]) {
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
        let groups = if KIND >= 12 { 8 } else { 1 };
        let width = block_width(KIND);
        let blocks = x.len() / groups;
        let stride = blocks * width;
        let mut acc = [vdupq_n_f32(0.0); ROWS];
        let mut offsets = [0.0_f32; ROWS];
        for block in 0..blocks {
            let raws: [&[u8]; ROWS] = std::array::from_fn(|r| {
                let start = (first + r) * stride + block * width;
                &w[start..start + width]
            });
            let mut headers = [Header { d: 0.0, min: 0.0 }; ROWS];
            for (h, raw) in headers.iter_mut().zip(raws) {
                *h = header::<KIND>(raw);
            }
            for g in 0..groups {
                let xr = &x[block * groups + g];
                let xq = activation(xr);
                for r in 0..ROWS {
                    let group = group::<KIND>(raws[r], headers[r], g);
                    accumulate::<KIND>(&mut acc[r], group.q, group.d, xq, xr.d);
                    offset::<KIND>(&mut offsets[r], group.m, xr);
                }
            }
        }
        for (r, y) in out.iter_mut().enumerate() {
            *y = vaddvq_f32(acc[r]) + offsets[r];
        }
    }

    /// Four-row W panel by two tokens per pass with `sdot`.
    ///
    /// # Safety
    /// The CPU must support NEON and dotprod; `w` holds four rows of `nb` records and `x`
    /// holds `batch * nb` activation records.
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn tile(
        kind: u32,
        w: &[W],
        x: &[X],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        macro_rules! run {
            ($kind:literal) => {
                tile_kind::<$kind>(w, x, nb, batch, out)
            };
        }
        by_kind!(kind, run)
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn tile_kind<const KIND: u32>(
        w: &[W],
        x: &[X],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        assert!(w.len() >= 4 * nb && x.len() >= batch * nb);
        for b in (0..batch).step_by(2) {
            let mut acc = [[vdupq_n_f32(0.0); 4]; 2];
            let mut offsets = [[0.0_f32; 4]; 2];
            for kb in 0..nb {
                let xs = [&x[b * nb + kb], &x[(b + 1).min(batch - 1) * nb + kb]];
                let xq = [activation(xs[0]), activation(xs[1])];
                for r in 0..4 {
                    let wr = &w[r * nb + kb];
                    let q = panel_weights(wr);
                    for t in 0..2 {
                        accumulate::<KIND>(&mut acc[t][r], q, wr.d, xq[t], xs[t].d);
                        offset::<KIND>(&mut offsets[t][r], wr.m, xs[t]);
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

    /// `tile` with `smmla`: each row pair meets the token pair in four instructions per group
    /// (two per 16-value half for Q6_K's separate scales).
    ///
    /// # Safety
    /// As `tile`, and the CPU must support i8mm.
    #[target_feature(enable = "neon,dotprod,i8mm")]
    pub(super) unsafe fn tile_i8mm(
        kind: u32,
        w: &[W],
        x: &[X],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        macro_rules! run {
            ($kind:literal) => {
                tile_i8mm_kind::<$kind>(w, x, nb, batch, out)
            };
        }
        by_kind!(kind, run)
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod,i8mm")]
    unsafe fn tile_i8mm_kind<const KIND: u32>(
        w: &[W],
        x: &[X],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        assert!(w.len() >= 4 * nb && x.len() >= batch * nb);
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
                    let (wa, wb) = (&w[2 * k * nb + kb], &w[(2 * k + 1) * nb + kb]);
                    let (qa, qb) = (panel_weights(wa), panel_weights(wb));
                    let low = smmla(
                        smmla(zero, zip_lo(qa[0], qb[0]), xm[0]),
                        zip_hi(qa[0], qb[0]),
                        xm[1],
                    );
                    let high = smmla(
                        smmla(zero, zip_lo(qa[1], qb[1]), xm[2]),
                        zip_hi(qa[1], qb[1]),
                        xm[3],
                    );
                    let d0 = [wa.d[0], wa.d[0], wb.d[0], wb.d[0]];
                    let scale0 = vmulq_f32(vld1q_f32(d0.as_ptr()), xd);
                    if KIND == 14 {
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
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::cpu_selftest::{compare, random_weights, reference_packed, Lcg};

    fn has_avx512() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && crate::cpu_kernels::isa() != "scalar"
    }

    // The fused expert path against the per-projection calls it replaces: grouped gate+up,
    // SwiGLU, then the down projection. Same kernels and records, so equal bit for bit.
    #[test]
    fn fused_experts_equal_the_per_projection_path_bit_for_bit() {
        let mut rng = Lcg(77);
        let mut scratch = ExpertScratch::default();
        let swiglu = |g: &mut [f32], u: &[f32]| crate::cpu_elementwise::swiglu_serial(g, u);
        // (formats of gate/up/down, hidden width, expert FFN width, output rows, biases)
        let configs = [
            ([12u32, 12, 12], 256usize, 256usize, 8usize, false),
            ([12, 12, 14], 512, 768, 37, false),
            ([12, 14, 12], 768, 512, 100, true),
            ([14, 14, 14], 512, 1024, 63, true),
            ([13, 11, 10], 512, 256, 21, false),
        ];
        for (kinds, cols, ff, out_rows, bias) in configs {
            let weights = (0..8)
                .map(|_| {
                    (
                        random_weights(kinds[0], ff, cols, &mut rng),
                        random_weights(kinds[1], ff, cols, &mut rng),
                        random_weights(kinds[2], out_rows, ff, &mut rng),
                    )
                })
                .collect::<Vec<_>>();
            let (gate_bias, up_bias, down_bias) = (
                rng.vector(ff, 0.5),
                rng.vector(ff, 0.5),
                rng.vector(out_rows, 0.5),
            );
            fn pick(on: bool, v: &[f32]) -> Option<&[f32]> {
                on.then_some(v)
            }
            // Differing counts reuse one scratch: stale rows from the larger call must not leak.
            for count in [8, 1, 3, 8, 5] {
                let x = rng.vector(cols, 2.0);
                let experts = weights[..count]
                    .iter()
                    .map(|(g, u, d)| ExpertProjections {
                        gate: Projection {
                            weights: g,
                            rows: ff,
                            bias: pick(bias, &gate_bias),
                        },
                        up: Projection {
                            weights: u,
                            rows: ff,
                            bias: pick(bias, &up_bias),
                        },
                        down: Projection {
                            weights: d,
                            rows: out_rows,
                            bias: pick(bias, &down_bias),
                        },
                        kinds,
                    })
                    .collect::<Vec<_>>();
                let Some(fused) = linear_experts(&experts, &x, cols, &swiglu, &mut scratch) else {
                    return;
                };
                assert_eq!(fused.len(), count);
                for (e, have) in experts.iter().zip(fused) {
                    let mut pair = linear_mixed(
                        &[
                            Projection {
                                weights: e.gate.weights,
                                rows: ff,
                                bias: e.gate.bias,
                            },
                            Projection {
                                weights: e.up.weights,
                                rows: ff,
                                bias: e.up.bias,
                            },
                        ],
                        &kinds[..2],
                        &x,
                        cols,
                    )
                    .unwrap();
                    let up = pair.pop().unwrap();
                    let mut gate = pair.pop().unwrap();
                    crate::cpu_elementwise::swiglu(&mut gate, &up);
                    let want =
                        linear(e.down.weights, kinds[2], &gate, out_rows, ff, e.down.bias).unwrap();
                    assert_eq!(have.len(), want.len());
                    assert!(
                        have.iter()
                            .zip(&want)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "kinds {kinds:?} cols {cols} ff {ff} rows {out_rows} count {count}"
                    );
                }
            }
        }
    }

    #[test]
    fn fused_experts_decline_what_they_cannot_run() {
        let mut scratch = ExpertScratch::default();
        let nothing = |_: &mut [f32], _: &[f32]| {};
        assert!(linear_experts(&[], &[], 256, &nothing, &mut scratch).is_none());
        // Q4_0 is not a K format; the caller falls back to the per-expert path.
        let w = [0u8; 18];
        let projection = || Projection {
            weights: &w,
            rows: 1,
            bias: None,
        };
        let classic = [ExpertProjections {
            gate: projection(),
            up: projection(),
            down: projection(),
            kinds: [2, 2, 2],
        }];
        assert!(linear_experts(&classic, &[0.0; 32], 32, &nothing, &mut scratch).is_none());
    }

    #[test]
    fn prefetch_clips_to_the_weights() {
        let weights = vec![1u8; 1000];
        prefetch_rows(&weights, 0, 4, 432);
        prefetch_rows(&weights, 2, 8, 432);
        prefetch_rows(&weights, 5, 8, 432);
        prefetch_rows(&weights, usize::MAX, usize::MAX, usize::MAX);
        prefetch_rows(&[], 0, 1, 1);
    }

    // The AVX-512 classic kernels against the f64 reference: odd block counts reach the zero
    // upper half of the decode pair, every decode row count at a nonzero offset, odd token
    // counts in the W-panel tile.
    #[test]
    fn avx512_classic_kernels_match_the_scalar_reference_at_odd_shapes() {
        if !has_avx512() {
            return;
        }
        let vnni = std::arch::is_x86_feature_detected!("avx512vnni");
        let mut rng = Lcg(23);
        for kind in [2, 3, 6, 7] {
            let width = layout(kind).unwrap().1;
            for blocks in [1, 2, 3, 5, 11] {
                let cols = blocks * 32;
                let rows = 8;
                let weights = random_weights(kind, rows, cols, &mut rng);
                let x = rng.vector(cols, 3.0);
                let packed = prepare(&x, cols, kind);
                let (want, scale) = reference_packed(&weights, kind, rows, cols, &packed);
                for n in 1..=8 {
                    let first = rows - n;
                    for use_vnni in [false, true] {
                        if use_vnni && !vnni {
                            continue;
                        }
                        let mut have = vec![0.0; n];
                        // SAFETY: features detected above; eight complete rows of `blocks`.
                        unsafe {
                            if use_vnni {
                                x86::gemv512_vnni(&weights, kind, &packed, first, &mut have);
                            } else {
                                x86::gemv512_bw(&weights, kind, &packed, first, &mut have);
                            }
                        }
                        let (error, ok) = compare(&have, &want[first..], &scale[first..]);
                        assert!(
                            ok,
                            "kind {kind} vnni={use_vnni} blocks {blocks} rows {n}: {error:e}"
                        );
                    }
                }
                for batch in [2, 3, 5] {
                    let x = rng.vector(batch * cols, 3.0);
                    let packed = prepare(&x, cols, kind);
                    let rows = 7;
                    let (want, scale) = reference_packed(&weights, kind, rows, cols, &packed);
                    for use_vnni in [false, true] {
                        if use_vnni && !vnni {
                            continue;
                        }
                        let mut output = vec![vec![0.0; batch * rows]];
                        crate::cpu_schedule::row_panels(
                            &mut output,
                            &[rows],
                            batch,
                            4,
                            || vec![W::default(); blocks * 4],
                            |panel, _, first, out| {
                                let valid = (rows - first).min(4);
                                for r in 0..valid {
                                    // SAFETY: AVX2/F16C present; the row holds `blocks` blocks.
                                    unsafe {
                                        x86::unpack_avx2(
                                            &weights[(first + r) * blocks * width..]
                                                [..blocks * width],
                                            kind,
                                            &mut panel[r * blocks..(r + 1) * blocks],
                                        )
                                    };
                                }
                                for r in valid..4 {
                                    panel[r * blocks..(r + 1) * blocks].fill(W::default());
                                }
                                // SAFETY: features detected above; four panel rows and
                                // `batch * blocks` activation records.
                                unsafe {
                                    match (kind, use_vnni) {
                                        (2 | 6, true) => x86::tile512_vnni::<2>(
                                            panel, &packed, blocks, batch, out,
                                        ),
                                        (2 | 6, false) => {
                                            x86::tile512_bw::<2>(panel, &packed, blocks, batch, out)
                                        }
                                        (_, true) => x86::tile512_vnni::<3>(
                                            panel, &packed, blocks, batch, out,
                                        ),
                                        (_, false) => {
                                            x86::tile512_bw::<3>(panel, &packed, blocks, batch, out)
                                        }
                                    }
                                }
                            },
                        );
                        let (error, ok) = compare(&output[0], &want, &scale);
                        assert!(
                            ok,
                            "kind {kind} vnni={use_vnni} blocks {blocks} batch {batch}: {error:e}"
                        );
                    }
                }
            }
        }
    }

    // IQ4_NL and MXFP4 against the f64 reference for every row count and odd token counts.
    #[test]
    fn codebook_kernels_match_the_scalar_reference() {
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        let mut rng = Lcg(29);
        for kind in [20, 39] {
            for blocks in [1, 3, 11] {
                let cols = blocks * 32;
                let rows = 7;
                let weights = random_weights(kind, rows, cols, &mut rng);
                for batch in [1, 2, 5] {
                    let x = rng.vector(batch * cols, 3.0);
                    let packed = prepare(&x, cols, kind);
                    let (want, scale) = reference_packed(&weights, kind, rows, cols, &packed);
                    let mut output = vec![vec![0.0; batch * rows]];
                    crate::cpu_schedule::row_panels(
                        &mut output,
                        &[rows],
                        batch,
                        4,
                        || (),
                        |_, _, first, out| {
                            let valid = (rows - first).min(4);
                            // SAFETY: AVX2/FMA/F16C present; shapes from `random_weights`.
                            unsafe {
                                x86::codebook_avx2(
                                    &weights, kind, first, valid, &packed, blocks, batch, out,
                                )
                            };
                        },
                    );
                    let (error, ok) = compare(&output[0], &want, &scale);
                    assert!(ok, "kind {kind} blocks {blocks} batch {batch}: {error:e}");
                }
            }
        }
    }

    // Mixed K members share one packing and each matches its own f64 reference.
    #[test]
    fn mixed_k_group_matches_single_format_projections() {
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        let mut rng = Lcg(31);
        let cols = 512;
        let kinds = [12, 14, 13, 10, 11, 23];
        let rows = [13, 5, 9, 4, 7, 3];
        let weights = kinds
            .iter()
            .zip(rows)
            .map(|(&kind, rows)| random_weights(kind, rows, cols, &mut rng))
            .collect::<Vec<_>>();
        for batch in [1, 3] {
            let x = rng.vector(batch * cols, 3.0);
            let members = weights
                .iter()
                .zip(rows)
                .map(|(weights, rows)| Projection {
                    weights,
                    rows,
                    bias: None,
                })
                .collect::<Vec<_>>();
            let grouped = linear_mixed(&members, &kinds, &x, cols).unwrap();
            let packed = prepare(&x, cols, 12);
            for ((output, &kind), (weights, rows)) in
                grouped.iter().zip(&kinds).zip(weights.iter().zip(rows))
            {
                let (want, scale) = reference_packed(weights, kind, rows, cols, &packed);
                let (error, ok) = compare(output, &want, &scale);
                assert!(ok, "kind {kind} batch {batch}: {error:e}");
            }
        }
        assert!(!groupable(&[12, 2]));
        assert!(!groupable(&[2, 6]));
        assert!(groupable(&[10, 11, 12, 13, 14, 23]));
        assert!(groupable(&[3, 3]));
    }
    #[test]
    fn classic_direct_decode_preserves_integer_lanes_and_fma_order() {
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        for kind in [2, 3, 6, 7] {
            let width = match kind {
                2 => 18,
                3 => 20,
                6 => 22,
                7 => 24,
                _ => unreachable!(),
            };
            for blocks in [1, 3, 17] {
                let cols = blocks * 32;
                let input = (0..cols)
                    .map(|i| ((i * 19 % 255) as f32 - 127.0) / 37.0)
                    .collect::<Vec<_>>();
                let packed = prepare(&input, cols, kind);
                let mut weights = (0..7 * blocks * width)
                    .map(|i| (i * 37 + 11) as u8)
                    .collect::<Vec<_>>();
                for (i, raw) in weights.chunks_mut(width).enumerate() {
                    let d = f16::from_f32((i % 9 + 1) as f32 / 256.0).to_le_bytes();
                    raw[..2].copy_from_slice(&d);
                    if kind == 3 || kind == 7 {
                        raw[2..4].copy_from_slice(
                            &f16::from_f32((i % 7) as f32 / 64.0 - 0.0625).to_le_bytes(),
                        );
                    }
                }
                for rows in 1..=4 {
                    let first = 2;
                    let mut output = vec![0.0; rows];
                    // SAFETY: guarded by the scalar check at the top of the test.
                    unsafe {
                        x86::gemv_avx2(&weights, kind, &packed, first, &mut output);
                    }
                    if is_x86_feature_detected!("avxvnni") {
                        // `vpdpbusd` sums the same integers as `maddubs` + `madd`: bit-identical.
                        let mut vnni = vec![0.0; rows];
                        // SAFETY: AVX-VNNI was detected just above.
                        unsafe { x86::gemv_vnni(&weights, kind, &packed, first, &mut vnni) };
                        assert_eq!(
                            vnni.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "vnni kind {kind} blocks {blocks} rows {rows}"
                        );
                    }
                    for (r, actual) in output.into_iter().enumerate() {
                        let mut a = [0.0_f32; 8];
                        let mut offset = 0.0_f32;
                        for (b, xr) in packed.iter().enumerate() {
                            let start = ((first + r) * blocks + b) * width;
                            let mut panel = [W::default()];
                            // SAFETY: guarded by the scalar check at the top of the test.
                            unsafe {
                                x86::unpack_avx2(&weights[start..start + width], kind, &mut panel);
                            }
                            let wr = &panel[0];
                            for (lane, acc) in a.iter_mut().enumerate() {
                                let dot = (lane * 4..lane * 4 + 4)
                                    .map(|i| ((wr.q[i] ^ 128) as i8 as i32) * xr.q[i] as i32)
                                    .sum::<i32>();
                                *acc = (dot as f32).mul_add(wr.d[0] * xr.d, *acc);
                            }
                            if kind == 3 || kind == 7 {
                                offset = wr.m[0].mul_add(xr.sum[0], offset);
                            }
                        }
                        let expected = (((a[0] + a[4]) + (a[2] + a[6]))
                            + ((a[1] + a[5]) + (a[3] + a[7])))
                            + offset;
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "kind {kind} blocks {blocks} rows {rows} row {r}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn quantized_siblings_preserve_bias_and_unequal_row_tails() {
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        let rows = [7, 1, 2, 3, 13];
        let cols = 512;
        for kind in [2, 3, 6, 7, 12, 13, 14] {
            let block_size = if kind >= 12 { 256 } else { 32 };
            let block_width = match kind {
                2 => 18,
                3 => 20,
                6 => 22,
                7 => 24,
                12 => 144,
                13 => 176,
                14 => 210,
                _ => unreachable!(),
            };
            let weights = rows
                .iter()
                .enumerate()
                .map(|(member, &r)| {
                    let mut weights = (0..r * cols / block_size * block_width)
                        .map(|i| ((i * 37 + member * 11) % 256) as u8)
                        .collect::<Vec<_>>();
                    for block in weights.chunks_mut(block_width) {
                        let d = f16::from_f32(0.03125).to_le_bytes();
                        if kind == 14 {
                            block[208..210].copy_from_slice(&d);
                        } else {
                            block[..2].copy_from_slice(&d);
                            if matches!(kind, 3 | 7 | 12 | 13) {
                                block[2..4].copy_from_slice(&d);
                            }
                        }
                    }
                    weights
                })
                .collect::<Vec<_>>();
            let bias = (0..rows[0]).map(|i| i as f32 / 11.0).collect::<Vec<_>>();
            let members = weights
                .iter()
                .enumerate()
                .map(|(i, w)| Projection {
                    weights: w,
                    rows: rows[i],
                    bias: (i == 0).then_some(bias.as_slice()),
                })
                .collect::<Vec<_>>();
            for threads in [1, 3, 10] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                for batch in [0, 1, 3] {
                    let input = (0..batch * cols)
                        .map(|i| (i % 71) as f32 / 13.0 - 2.0)
                        .collect::<Vec<_>>();
                    pool.install(|| {
                        let grouped = linear_group(&members, kind, &input, cols).unwrap();
                        for (p, output) in members.iter().zip(grouped) {
                            let individual =
                                linear(p.weights, kind, &input, p.rows, cols, p.bias).unwrap();
                            assert_eq!(
                                output, individual,
                                "kind {kind} batch {batch} threads {threads}"
                            );
                        }
                    });
                }
            }
        }
    }

    #[test]
    fn k_maximum_preserves_first_sign_across_vector_lanes() {
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        for first in 0..256 {
            for value in [-16.0, 16.0] {
                let mut input = [0.25; 256];
                input[first] = value;
                if first + 1 < input.len() {
                    input[first + 1] = -value;
                }
                input[0] = if first == 0 { value } else { f32::NAN };
                // SAFETY: guarded by the scalar check at the top of the test.
                let (maximum, signed) = unsafe { x86::first_signed_max(&input) };
                assert_eq!(maximum.to_bits(), 16.0_f32.to_bits());
                assert_eq!(signed.to_bits(), value.to_bits(), "first index {first}");
            }
        }
        for input in [[0.0; 256], [-0.0; 256], [f32::NAN; 256]] {
            // SAFETY: guarded by the scalar check at the top of the test.
            let (maximum, signed) = unsafe { x86::first_signed_max(&input) };
            assert_eq!((maximum.to_bits(), signed.to_bits()), (0, 0));
        }
    }

    #[test]
    fn k_activation_packing_preserves_scalar_rounding_and_sums() {
        if crate::cpu_kernels::isa() == "scalar" {
            return;
        }
        // One decode row, a small serial prompt, and a parallel prompt.
        for batch in [1, 3, 65] {
            let input = (0..batch * 256)
                .map(|i| {
                    if i % 256 == 31 {
                        -7.0
                    } else if i % 256 == 32 {
                        7.0
                    } else {
                        ((i * 37 % 257) as f32 - 128.0) / 64.0
                    }
                })
                .collect::<Vec<_>>();
            for kind in [12, 13, 14] {
                let packed = prepare(&input, 256, kind);
                for (block, actual) in input
                    .as_chunks::<256>()
                    .0
                    .iter()
                    .zip(packed.as_chunks::<8>().0)
                {
                    let inverse = -127.0 / -7.0;
                    let d = 1.0_f32 / inverse;
                    for (part, actual) in block.as_chunks::<32>().0.iter().zip(actual) {
                        let expected = std::array::from_fn::<_, 32, _>(|i| {
                            (part[i] * inverse).round_ties_even() as i8
                        });
                        assert_eq!(actual.q, expected);
                        assert_eq!(actual.d.to_bits(), d.to_bits());
                        for lane in 0..2 {
                            let sum = expected[lane * 16..lane * 16 + 16]
                                .iter()
                                .map(|v| *v as i32)
                                .sum::<i32>();
                            assert_eq!(actual.init[lane], -32 * sum);
                            assert_eq!(actual.sum[lane].to_bits(), (sum as f32 * d).to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prompt_experts_match_the_per_expert_path() {
        use crate::cpu_selftest::{random_weights, Lcg};
        if crate::cpu_kernels::isa() == "scalar" || crate::cpu_kernels::avx512().is_some() {
            return;
        }
        let mut rng = Lcg(5);
        let (cols, ff, hidden) = (512usize, 256usize, 768usize);
        let activate = |g: &mut [f32], u: &[f32]| {
            for (g, u) in g.iter_mut().zip(u) {
                *g = *g * *u * 0.05;
            }
        };
        for (gate_up, down) in [(12u32, 14u32), (11, 12), (10, 13)] {
            let weights: Vec<_> = (0..3)
                .map(|_| {
                    (
                        random_weights(gate_up, ff, cols, &mut rng),
                        random_weights(gate_up, ff, cols, &mut rng),
                        random_weights(down, hidden, ff, &mut rng),
                    )
                })
                .collect();
            let tokens = [5usize, 17, 1];
            let inputs: Vec<Vec<f32>> = tokens.iter().map(|&t| rng.vector(t * cols, 1.0)).collect();
            let projection = |weights: &'static [u8], rows| Projection { weights, rows, bias: None };
            let leak = |v: &Vec<u8>| -> &'static [u8] { Box::leak(v.clone().into_boxed_slice()) };
            let experts: Vec<ExpertRows<'_>> = weights
                .iter()
                .zip(&inputs)
                .map(|((g, u, d), input)| ExpertRows {
                    experts: ExpertProjections {
                        gate: projection(leak(g), ff),
                        up: projection(leak(u), ff),
                        down: projection(leak(d), hidden),
                        kinds: [gate_up, gate_up, down],
                    },
                    input,
                })
                .collect();
            let Some(got) = linear_experts_prompt(&experts, cols, &activate) else {
                panic!("the eight-row expert path declined kinds {gate_up}/{down}");
            };
            for (e, (rows, input)) in experts.iter().zip(&inputs).enumerate() {
                let run = |p: &Projection<'_>, kind, x: &[f32], k| {
                    linear_group(std::slice::from_ref(p), kind, x, k).unwrap().pop().unwrap()
                };
                let mut g = run(&rows.experts.gate, gate_up, input, cols);
                let u = run(&rows.experts.up, gate_up, input, cols);
                activate(&mut g, &u);
                let want = run(&rows.experts.down, down, &g, ff);
                let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
                assert_eq!(got[e].len(), want.len());
                for (i, (a, b)) in got[e].iter().zip(&want).enumerate() {
                    assert!(
                        (a - b).abs() <= 2e-3 * peak,
                        "kinds {gate_up}/{down} expert {e} cell {i}: {a} vs {b}"
                    );
                }
            }
        }
    }
}
