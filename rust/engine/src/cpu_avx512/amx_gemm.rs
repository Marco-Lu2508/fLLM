//! AMX-INT8 / AMX-BF16 prompt GEMMs.
//!
//! Layout (int8, weights as the `A` operand, activations as `B`, tokens in the tile columns):
//! - A tile: 16 weight rows x `G` bytes (one scale group), loaded from a decoded `AmxPanel`
//!   (`amx_decode`), row stride = the panel row length.
//! - B tile: `G / 4` rows of 16 tokens x 4 bytes (the VNNI interleave); `AmxActs` builds these once
//!   per call, so every weight panel reuses them.
//! - C tile: 16 x 16 `i32`, `C[m][n] = sum_k A[m][k] * B[k][n]` for one scale group. A group's tile
//!   is stored, converted to float and added by the epilogue to the accumulator as
//!   `C * scale[m][group] * dx[n][group]`, minus `min[m][group] * sum[n][group]` for formats with a
//!   minimum, so every format keeps its own scale structure (Q8_0 scale per 32 values, Q6_K scale
//!   per 16, Q4_K scale and minimum per 32, ...).
//!
//! The drivers are generic over `TileOps` (hardware or the scalar emulation of the unit tests) and
//! `Epilogue` (AVX-512 or scalar), so the unit tests run the very same loops that drive the
//! hardware, except for the tile instructions and the AVX-512 epilogue.
use super::amx::{TileConfig, TileOps, TileShape};
use super::amx_decode::{self as dec, AmxPanel, PANEL_ROWS};
use crate::cpu_schedule::Rows;
use rayon::prelude::*;
use std::arch::x86_64::*;

/// Tokens per tile column block.
const TOKENS: usize = 16;

/// One activation record: 32 quantized values, their scale and the two 16-value sums (already
/// multiplied by the scale, like `cpu_quantized::X::sum`).
pub(crate) struct Rec<'a> {
    pub q: &'a [i8; 32],
    pub d: f32,
    pub sum: [f32; 2],
}

/// Activations of one call in the B-tile layout for scale groups of `group` (16 or 32) values.
pub(crate) struct AmxActs {
    pub batch: usize,
    pub group: usize,
    pub groups: usize,
    pub tiles: usize,
    /// `[tile][group][group / 4 rows][16 tokens][4 bytes]`; tokens past the batch are zero.
    pub b: Vec<u8>,
    /// `[tile][group][16 tokens]` activation scale.
    pub dx: Vec<f32>,
    /// `[tile][group][16 tokens]` sum of the group's values times the scale (for formats with a
    /// minimum).
    pub sum: Vec<f32>,
}

impl AmxActs {
    /// `rec(token, block)` returns the 32-value record `block` of `token`; `nb` blocks per token.
    pub(crate) fn build<'a, F>(batch: usize, nb: usize, group: usize, rec: F) -> Self
    where
        F: Fn(usize, usize) -> Rec<'a> + Sync,
    {
        assert!(group == 16 || group == 32);
        let groups = nb * 32 / group;
        let tiles = batch.div_ceil(TOKENS);
        let tile_b = groups * 16 * group;
        let tile_f = groups * TOKENS;
        let mut b = vec![0u8; tiles * tile_b];
        let mut dx = vec![0f32; tiles * tile_f];
        let mut sum = vec![0f32; tiles * tile_f];
        let fill = |t: usize, bt: &mut [u8], dxt: &mut [f32], sumt: &mut [f32]| {
            for n in 0..(batch - TOKENS * t).min(TOKENS) {
                let token = TOKENS * t + n;
                for gi in 0..groups {
                    let (block, half) = if group == 32 {
                        (gi, 0)
                    } else {
                        (gi / 2, gi % 2)
                    };
                    let r = rec(token, block);
                    let q = if group == 32 {
                        &r.q[..]
                    } else {
                        &r.q[16 * half..16 * half + 16]
                    };
                    let base = gi * 16 * group;
                    for kk in 0..group / 4 {
                        let at = base + (kk * TOKENS + n) * 4;
                        for j in 0..4 {
                            bt[at + j] = q[4 * kk + j] as u8;
                        }
                    }
                    dxt[gi * TOKENS + n] = r.d;
                    sumt[gi * TOKENS + n] = if group == 32 {
                        r.sum[0] + r.sum[1]
                    } else {
                        r.sum[half]
                    };
                }
            }
        };
        if batch * nb * 32 >= 1 << 16 {
            b.par_chunks_mut(tile_b)
                .zip(dx.par_chunks_mut(tile_f))
                .zip(sum.par_chunks_mut(tile_f))
                .enumerate()
                .for_each(|(t, ((bt, dxt), st))| fill(t, bt, dxt, st));
        } else {
            for (t, ((bt, dxt), st)) in b
                .chunks_mut(tile_b)
                .zip(dx.chunks_mut(tile_f))
                .zip(sum.chunks_mut(tile_f))
                .enumerate()
            {
                fill(t, bt, dxt, st);
            }
        }
        Self {
            batch,
            group,
            groups,
            tiles,
            b,
            dx,
            sum,
        }
    }
}

/// Pointers to the per-group data the epilogue reads.
pub(crate) struct GroupPtrs {
    /// 16 token scales.
    pub dx: *const f32,
    /// 16 token sums.
    pub sum: *const f32,
    /// 16 row scales.
    pub scale: *const f32,
    /// 16 row minimums.
    pub min: *const f32,
}

/// Turns the `i32` tile of one group into float accumulators.
pub(crate) trait Epilogue {
    /// 16 x 16 float accumulators (row = weight row, lane = token); all-zero bits are 0.0.
    type Acc;
    /// # Safety
    /// `c` points at a 16 x 16 row-major `i32` tile; `p` at valid group data (`sum`/`min` only
    /// read when `MIN`).
    unsafe fn add<const MIN: bool>(acc: &mut Self::Acc, c: *const i32, p: &GroupPtrs);
    /// # Safety
    /// `out` points at 256 writable floats; row-major `[row][token]`.
    unsafe fn store(acc: &Self::Acc, out: *mut f32);
}

/// Portable epilogue (unit tests, and the reference for the AVX-512 one).
#[cfg(test)]
pub(crate) struct ScalarEpilogue;

#[cfg(test)]
impl Epilogue for ScalarEpilogue {
    type Acc = [f32; 256];
    #[inline(always)]
    unsafe fn add<const MIN: bool>(acc: &mut [f32; 256], c: *const i32, p: &GroupPtrs) {
        for m in 0..16 {
            let scale = *p.scale.add(m);
            let min = if MIN { *p.min.add(m) } else { 0.0 };
            for n in 0..16 {
                let value = *c.add(m * 16 + n) as f32;
                acc[m * 16 + n] += value * (*p.dx.add(n) * scale);
                if MIN {
                    acc[m * 16 + n] -= min * *p.sum.add(n);
                }
            }
        }
    }
    #[inline(always)]
    unsafe fn store(acc: &[f32; 256], out: *mut f32) {
        std::ptr::copy_nonoverlapping(acc.as_ptr(), out, 256);
    }
}

/// AVX-512 epilogue: one zmm per weight row, the 16 token lanes across.
pub(crate) struct Avx512Epilogue;

impl Epilogue for Avx512Epilogue {
    type Acc = [__m512; 16];
    #[inline(always)]
    unsafe fn add<const MIN: bool>(acc: &mut [__m512; 16], c: *const i32, p: &GroupPtrs) {
        let dx = _mm512_loadu_ps(p.dx);
        let sum = if MIN {
            _mm512_loadu_ps(p.sum)
        } else {
            _mm512_setzero_ps()
        };
        for (m, acc) in acc.iter_mut().enumerate() {
            let value = _mm512_cvtepi32_ps(_mm512_loadu_si512(c.add(m * 16).cast()));
            let scale = _mm512_mul_ps(dx, _mm512_set1_ps(*p.scale.add(m)));
            *acc = _mm512_fmadd_ps(value, scale, *acc);
            if MIN {
                *acc = _mm512_fnmadd_ps(sum, _mm512_set1_ps(*p.min.add(m)), *acc);
            }
        }
    }
    #[inline(always)]
    unsafe fn store(acc: &[__m512; 16], out: *mut f32) {
        for (m, acc) in acc.iter().enumerate() {
            _mm512_storeu_ps(out.add(m * 16), *acc);
        }
    }
}

/// Scratch tiles of one panel run, 64-byte aligned for the tile stores.
#[repr(C, align(64))]
struct Scratch {
    c0: [i32; 256],
    c1: [i32; 256],
    out: [f32; 256],
}

/// One 16-row weight panel against every token. `G` is the group size of the format, `SIGNED`
/// selects `tdpbssd` over `tdpbusd`, `MIN` the minimum correction. `panel_rows` rows are valid;
/// `first` is the absolute index of the panel's first row (for the bias).
///
/// # Safety
/// The tile instructions of `T` and the vector instructions of `E` must be available; `panel` and
/// `acts` are consistent with `G`, `out` is this panel's exclusive view of `panel_rows` columns.
#[inline(always)]
pub(crate) unsafe fn int8_panel<
    T: TileOps,
    E: Epilogue,
    const G: usize,
    const SIGNED: bool,
    const MIN: bool,
>(
    ops: &mut T,
    panel: &AmxPanel,
    acts: &AmxActs,
    panel_rows: usize,
    first: usize,
    bias: Option<&[f32]>,
    out: &mut Rows<'_>,
) {
    debug_assert!(panel.group == G && acts.group == G && panel.cols / G == acts.groups);
    let (cols, groups) = (panel.cols, acts.groups);
    let c_shape = TileShape {
        rows: 16,
        colsb: 64,
    };
    let a_shape = TileShape {
        rows: 16,
        colsb: G as u16,
    };
    let b_shape = TileShape {
        rows: (G / 4) as u8,
        colsb: 64,
    };
    // Two independent slots (tiles 0-2 and 3-5) keep two dot chains in flight.
    ops.configure(&TileConfig::palette1(&[
        c_shape, a_shape, b_shape, c_shape, a_shape, b_shape,
    ]));
    let mut scratch = Scratch {
        c0: [0; 256],
        c1: [0; 256],
        out: [0.0; 256],
    };
    let a_base = panel.q.as_ptr();
    for t in 0..acts.tiles {
        let b_tile = acts.b.as_ptr().add(t * groups * 16 * G);
        let dx_tile = acts.dx.as_ptr().add(t * groups * TOKENS);
        let sum_tile = acts.sum.as_ptr().add(t * groups * TOKENS);
        let mut acc: E::Acc = std::mem::zeroed();
        let params = |gi: usize| GroupPtrs {
            dx: dx_tile.add(gi * TOKENS),
            sum: sum_tile.add(gi * TOKENS),
            scale: panel.scale.as_ptr().add(gi * PANEL_ROWS),
            min: panel.min.as_ptr().add(gi * PANEL_ROWS),
        };
        let mut gi = 0;
        while gi < groups {
            let two = gi + 1 < groups;
            ops.zero::<0>();
            ops.load::<1>(a_base.add(gi * G), cols);
            ops.load::<2>(b_tile.add(gi * 16 * G), 64);
            if SIGNED {
                ops.dpbssd::<0, 1, 2>();
            } else {
                ops.dpbusd::<0, 1, 2>();
            }
            if two {
                ops.zero::<3>();
                ops.load::<4>(a_base.add((gi + 1) * G), cols);
                ops.load::<5>(b_tile.add((gi + 1) * 16 * G), 64);
                if SIGNED {
                    ops.dpbssd::<3, 4, 5>();
                } else {
                    ops.dpbusd::<3, 4, 5>();
                }
            }
            ops.store::<0>(scratch.c0.as_mut_ptr().cast(), 64);
            E::add::<MIN>(&mut acc, scratch.c0.as_ptr(), &params(gi));
            if two {
                ops.store::<3>(scratch.c1.as_mut_ptr().cast(), 64);
                E::add::<MIN>(&mut acc, scratch.c1.as_ptr(), &params(gi + 1));
            }
            gi += 2;
        }
        E::store(&acc, scratch.out.as_mut_ptr());
        for n in 0..(acts.batch - TOKENS * t).min(TOKENS) {
            let row = out.row(TOKENS * t + n);
            for (m, y) in row.iter_mut().enumerate().take(panel_rows) {
                *y = scratch.out[m * 16 + n] + bias.map_or(0.0, |b| b[first + m]);
            }
        }
    }
    ops.release();
}

/// A member projection of a call.
pub(crate) struct Member<'a> {
    pub weights: &'a [u8],
    pub rows: usize,
    pub bias: Option<&'a [f32]>,
    pub kind: u32,
}

/// The activations of a call, per group size in use.
pub(crate) struct ActSets {
    pub g16: Option<AmxActs>,
    pub g32: Option<AmxActs>,
}

/// Per-worker state of `run_members`: the decoded panels (one per group size) and the tile unit.
pub(crate) struct State<T> {
    p16: Option<AmxPanel>,
    p32: Option<AmxPanel>,
    ops: T,
}

/// One panel of one member: decode, then run. The job is a function pointer because the
/// closure handed to the scheduler cannot carry target features, whereas the hardware job
/// (`panel_job_hw`) is compiled with them, so the epilogue vectorizes.
pub(crate) type Job<T> =
    unsafe fn(&mut State<T>, &Member<'_>, &ActSets, usize, usize, &mut Rows<'_>);

/// Body of a panel job, inlined into `panel_job_hw` (hardware) or the test job (emulation).
///
/// # Safety
/// As `int8_panel`; AVX2 + F16C for the decoders; `m.weights` holds `m.rows` complete rows of
/// `m.kind` with `cols` columns.
#[inline(always)]
unsafe fn panel_job<T: TileOps, E: Epilogue>(
    state: &mut State<T>,
    m: &Member<'_>,
    acts: &ActSets,
    cols: usize,
    first: usize,
    out: &mut Rows<'_>,
) {
    let info = dec::kind_info(m.kind).expect("kind has an AMX decoder");
    let valid = (m.rows - first).min(PANEL_ROWS);
    let slot = if info.group == 16 {
        &mut state.p16
    } else {
        &mut state.p32
    };
    let panel = slot.get_or_insert_with(|| AmxPanel::new(cols, info.group));
    dec::decode_panel(m.kind, m.weights, cols, first, valid, panel);
    let a = if info.group == 16 {
        acts.g16.as_ref()
    } else {
        acts.g32.as_ref()
    }
    .expect("activations exist for every group size in use");
    let ops = &mut state.ops;
    match (info.group, info.signed, info.has_min) {
        (32, true, false) => {
            int8_panel::<T, E, 32, true, false>(ops, panel, a, valid, first, m.bias, out)
        }
        (32, false, true) => {
            int8_panel::<T, E, 32, false, true>(ops, panel, a, valid, first, m.bias, out)
        }
        (16, true, false) => {
            int8_panel::<T, E, 16, true, false>(ops, panel, a, valid, first, m.bias, out)
        }
        (16, false, true) => {
            int8_panel::<T, E, 16, false, true>(ops, panel, a, valid, first, m.bias, out)
        }
        _ => unreachable!("every AMX format is signed without minimum or unsigned with one"),
    }
}

/// The hardware panel job: AMX tiles and the AVX-512 epilogue.
///
/// # Safety
/// The CPU must have AMX-TILE/INT8, AVX-512 (F, BW, VL), AVX2, FMA and F16C and the process the
/// tile permission (`ident::amx_ready`); other conditions as `panel_job`.
#[target_feature(enable = "avx2,fma,f16c,avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn panel_job_hw(
    state: &mut State<super::amx::RealTiles>,
    m: &Member<'_>,
    acts: &ActSets,
    cols: usize,
    first: usize,
    out: &mut Rows<'_>,
) {
    panel_job::<super::amx::RealTiles, Avx512Epilogue>(state, m, acts, cols, first, out);
}

/// Run all members over the shared activations with `job`.
///
/// # Safety
/// As `panel_job` for the job's tile and vector instructions; `output[i]` holds
/// `members[i].rows * batch` floats.
pub(crate) unsafe fn run_members<T: TileOps + Default>(
    job: Job<T>,
    members: &[Member<'_>],
    cols: usize,
    batch: usize,
    acts: &ActSets,
    output: &mut [Vec<f32>],
) {
    let rows: Vec<usize> = members.iter().map(|m| m.rows).collect();
    crate::cpu_schedule::row_panels(
        output,
        &rows,
        batch,
        PANEL_ROWS,
        || State {
            p16: None,
            p32: None,
            ops: T::default(),
        },
        |state, member, first, out| job(state, &members[member], acts, cols, first, out),
    );
}

/// Activation sets for the groups sizes the members need, built from `rec`.
pub(crate) fn act_sets<'a, F>(members: &[Member<'_>], batch: usize, nb: usize, rec: F) -> ActSets
where
    F: Fn(usize, usize) -> Rec<'a> + Sync,
{
    let need = |group: usize| {
        members
            .iter()
            .any(|m| dec::kind_info(m.kind).is_some_and(|i| i.group == group))
    };
    ActSets {
        g16: need(16).then(|| AmxActs::build(batch, nb, 16, &rec)),
        g32: need(32).then(|| AmxActs::build(batch, nb, 32, &rec)),
    }
}

// ---------------------------------------------------------------------------------------------
// bf16: weights stay in place (no decode), accumulation over the whole row in the C tiles.
// ---------------------------------------------------------------------------------------------

/// Elements per `tdpbf16ps` step: a 16 x 64-byte A tile.
const BF16_CHUNK: usize = 32;

/// bf16 activations in B-tile layout: `[tile][chunk][16 pair rows][16 tokens][2 x bf16]`.
pub(crate) struct Bf16Acts {
    pub batch: usize,
    pub tiles: usize,
    pub chunks: usize,
    pub b: Vec<u8>,
}

impl Bf16Acts {
    /// `x` holds `batch * cols` values, `cols` a multiple of 32. Values are rounded to bf16 (they
    /// already are when `cpu_float` rounded the input to the weight type).
    pub(crate) fn build(x: &[f32], batch: usize, cols: usize) -> Self {
        assert!(cols.is_multiple_of(BF16_CHUNK) && x.len() == batch * cols);
        let (tiles, chunks) = (batch.div_ceil(TOKENS), cols / BF16_CHUNK);
        let tile_bytes = chunks * 1024;
        let mut b = vec![0u8; tiles * tile_bytes];
        let fill = |t: usize, bt: &mut [u8]| {
            for n in 0..(batch - TOKENS * t).min(TOKENS) {
                let src = &x[(TOKENS * t + n) * cols..(TOKENS * t + n + 1) * cols];
                for ch in 0..chunks {
                    for kp in 0..16 {
                        let at = ch * 1024 + (kp * TOKENS + n) * 4;
                        let lo = half::bf16::from_f32(src[ch * BF16_CHUNK + 2 * kp]).to_bits();
                        let hi = half::bf16::from_f32(src[ch * BF16_CHUNK + 2 * kp + 1]).to_bits();
                        bt[at..at + 2].copy_from_slice(&lo.to_le_bytes());
                        bt[at + 2..at + 4].copy_from_slice(&hi.to_le_bytes());
                    }
                }
            }
        };
        if batch * cols >= 1 << 16 {
            b.par_chunks_mut(tile_bytes)
                .enumerate()
                .for_each(|(t, bt)| fill(t, bt));
        } else {
            for (t, bt) in b.chunks_mut(tile_bytes).enumerate() {
                fill(t, bt);
            }
        }
        Self {
            batch,
            tiles,
            chunks,
            b,
        }
    }
}

/// Up to 32 bf16 weight rows (`valid`, from row `first`) against every token with 2 x 2 tile
/// blocking: tiles 0-3 are C (weight rows x token tiles), 4-5 weights, 6-7 activations.
///
/// # Safety
/// AMX-BF16 (for `T` = hardware); `weights` holds complete bf16 rows of `cols` elements through
/// row `first + valid`; `out` is the exclusive view of `valid` columns.
#[inline(always)]
pub(crate) unsafe fn bf16_panel<T: TileOps>(
    ops: &mut T,
    weights: &[u8],
    cols: usize,
    (first, valid): (usize, usize),
    acts: &Bf16Acts,
    bias: Option<&[f32]>,
    out: &mut Rows<'_>,
) {
    debug_assert!((1..=2 * PANEL_ROWS).contains(&valid));
    let ldw = cols * 2;
    let (m0, m1) = (valid.min(PANEL_ROWS), valid.saturating_sub(PANEL_ROWS));
    let w0 = weights.as_ptr().add(first * ldw);
    let w1 = weights.as_ptr().add((first + PANEL_ROWS) * ldw);
    // 64-byte aligned scratch for the four C tiles.
    #[repr(C, align(64))]
    struct Cs([f32; 256]);
    let mut cs = [
        Cs([0.0; 256]),
        Cs([0.0; 256]),
        Cs([0.0; 256]),
        Cs([0.0; 256]),
    ];
    for t in (0..acts.tiles).step_by(2) {
        let n0 = (acts.batch - TOKENS * t).min(TOKENS);
        let n1 = if t + 1 < acts.tiles {
            (acts.batch - TOKENS * (t + 1)).min(TOKENS)
        } else {
            0
        };
        let shape = |rows: usize, tokens: usize| TileShape {
            rows: rows as u8,
            colsb: (4 * tokens) as u16,
        };
        let a_shape = |rows: usize| TileShape {
            rows: rows as u8,
            colsb: 64,
        };
        let none = TileShape { rows: 0, colsb: 0 };
        let used = |rows: usize, tokens: usize| {
            if rows > 0 && tokens > 0 {
                shape(rows, tokens)
            } else {
                none
            }
        };
        ops.configure(&TileConfig::palette1(&[
            used(m0, n0),
            used(m0, n1),
            used(m1, n0),
            used(m1, n1),
            if m0 > 0 { a_shape(m0) } else { none },
            if m1 > 0 { a_shape(m1) } else { none },
            TileShape {
                rows: 16,
                colsb: (4 * n0) as u16,
            },
            if n1 > 0 {
                TileShape {
                    rows: 16,
                    colsb: (4 * n1) as u16,
                }
            } else {
                none
            },
        ]));
        ops.zero::<0>();
        if n1 > 0 {
            ops.zero::<1>();
        }
        if m1 > 0 {
            ops.zero::<2>();
            if n1 > 0 {
                ops.zero::<3>();
            }
        }
        let b0 = acts.b.as_ptr().add(t * acts.chunks * 1024);
        let b1 = acts.b.as_ptr().add((t + 1) * acts.chunks * 1024);
        for ch in 0..acts.chunks {
            ops.load::<4>(w0.add(ch * 64), ldw);
            ops.load::<6>(b0.add(ch * 1024), 64);
            if m1 > 0 {
                ops.load::<5>(w1.add(ch * 64), ldw);
            }
            if n1 > 0 {
                ops.load::<7>(b1.add(ch * 1024), 64);
            }
            ops.dpbf16ps::<0, 4, 6>();
            if n1 > 0 {
                ops.dpbf16ps::<1, 4, 7>();
            }
            if m1 > 0 {
                ops.dpbf16ps::<2, 5, 6>();
                if n1 > 0 {
                    ops.dpbf16ps::<3, 5, 7>();
                }
            }
        }
        ops.store::<0>(cs[0].0.as_mut_ptr().cast(), 64);
        if n1 > 0 {
            ops.store::<1>(cs[1].0.as_mut_ptr().cast(), 64);
        }
        if m1 > 0 {
            ops.store::<2>(cs[2].0.as_mut_ptr().cast(), 64);
            if n1 > 0 {
                ops.store::<3>(cs[3].0.as_mut_ptr().cast(), 64);
            }
        }
        ops.release();
        // (tile index, first row, rows, first token, tokens)
        let blocks = [
            (0, 0, m0, TOKENS * t, n0),
            (1, 0, m0, TOKENS * (t + 1), n1),
            (2, PANEL_ROWS, m1, TOKENS * t, n0),
            (3, PANEL_ROWS, m1, TOKENS * (t + 1), n1),
        ];
        for (c, row0, rows, token0, tokens) in blocks {
            for n in 0..tokens {
                let row = out.row(token0 + n);
                for m in 0..rows {
                    row[row0 + m] = cs[c].0[m * 16 + n] + bias.map_or(0.0, |b| b[first + row0 + m]);
                }
            }
        }
    }
}

/// A bf16 member projection of a call.
pub(crate) struct Bf16Member<'a> {
    pub weights: &'a [u8],
    pub rows: usize,
    pub bias: Option<&'a [f32]>,
}

/// Run all bf16 members over the shared activations. Generic so the tests use the emulation.
///
/// # Safety
/// As `bf16_panel`; `output[i]` holds `rows * batch` floats.
#[inline(always)]
pub(crate) unsafe fn run_bf16_members<T: TileOps + Default>(
    members: &[Bf16Member<'_>],
    cols: usize,
    batch: usize,
    acts: &Bf16Acts,
    output: &mut [Vec<f32>],
) {
    let rows: Vec<usize> = members.iter().map(|m| m.rows).collect();
    crate::cpu_schedule::row_panels(
        output,
        &rows,
        batch,
        2 * PANEL_ROWS,
        T::default,
        |ops, member, first, out| {
            let m = &members[member];
            let valid = (m.rows - first).min(2 * PANEL_ROWS);
            bf16_panel::<T>(ops, m.weights, cols, (first, valid), acts, m.bias, out);
        },
    );
}

// ---------------------------------------------------------------------------------------------
// Entry points used by the dispatch hunks in cpu_kernels / cpu_quantized / cpu_float.
// ---------------------------------------------------------------------------------------------

/// Whether the AMX prompt kernels should run for `batch` tokens now: the `amx` tier is enabled
/// (CPU support, tile permission, `FLLM_CPU_ISA` cap), the prompt reaches the policy's minimum
/// (`FLLM_AMX_MIN_BATCH`, default 8) and prompt repacking is not switched off (`--no-repack`).
pub(crate) fn wanted(batch: usize) -> bool {
    batch >= super::ident::policy().amx_min_batch
        && crate::cpu_kernels::repack_enabled()
        && crate::cpu_kernels::tier_enabled(crate::cpu_kernels::CpuTier::Amx)
}

/// Whether `kinds` are all formats with an AMX decoder.
pub(crate) fn supports(kinds: &[u32]) -> bool {
    !kinds.is_empty() && kinds.iter().all(|&k| dec::kind_info(k).is_some())
}

/// `cpu_quantized::linear_mixed` for prompts: Q4_0..Q5_1, IQ4_NL, MXFP4, Q2_K..Q6_K, IQ4_XS.
/// `None` if a member has no AMX decoder. The caller has validated shapes and `wanted(batch)`.
pub(crate) fn linear_group(
    projections: &[crate::cpu_quantized::Projection<'_>],
    kinds: &[u32],
    x: &[f32],
    cols: usize,
    batch: usize,
) -> Option<Vec<Vec<f32>>> {
    if !supports(kinds) || projections.len() != kinds.len() {
        return None;
    }
    // One representative kind selects the activation records of the whole group.
    let prepare_kind = match dec::act_kind(kinds[0])? {
        dec::ActKind::Q8_0 => 2,
        dec::ActKind::Q8_1 => 3,
        dec::ActKind::Q8K => 12,
    };
    let records = crate::cpu_quantized::prepare(x, cols, prepare_kind);
    let nb = cols / 32;
    let members: Vec<Member<'_>> = projections
        .iter()
        .zip(kinds)
        .map(|(p, &kind)| Member {
            weights: p.weights,
            rows: p.rows,
            bias: p.bias,
            kind,
        })
        .collect();
    let sets = act_sets(&members, batch, nb, |token, block| {
        let r = &records[token * nb + block];
        Rec {
            q: &r.q,
            d: r.d,
            sum: r.sum,
        }
    });
    let mut output: Vec<Vec<f32>> = members.iter().map(|m| vec![0.0; m.rows * batch]).collect();
    // SAFETY: the `amx` tier is enabled (`wanted`), which implies AMX, AVX-512 F/BW/VL, AVX2, FMA
    // and F16C and the tile permission; shapes were validated by the caller.
    unsafe {
        run_members::<super::amx::RealTiles>(
            panel_job_hw,
            &members,
            cols,
            batch,
            &sets,
            &mut output,
        )
    };
    Some(output)
}

/// Q8_0 projections of `cpu_kernels::q8_linear_group_into` for prompts; `packed` holds the
/// activation pairs of `Q8Input` (`batch.div_ceil(2) * nb`). `output[i]` is resized.
pub(crate) fn q8_group(
    projections: &[crate::cpu_kernels::Q8Projection<'_>],
    packed: &[crate::cpu_kernels::XPair],
    nb: usize,
    batch: usize,
    output: &mut [Vec<f32>],
) {
    let members: Vec<Member<'_>> = projections
        .iter()
        .map(|p| Member {
            weights: p.weights,
            rows: p.rows,
            bias: p.bias,
            kind: dec::Q8_0,
        })
        .collect();
    let sets = act_sets(&members, batch, nb, |token, block| {
        let pair = &packed[(token / 2) * nb + block];
        Rec {
            q: &pair.q[token % 2],
            d: pair.d[token % 2],
            sum: [0.0; 2],
        }
    });
    for (out, m) in output.iter_mut().zip(&members) {
        out.clear();
        out.resize(m.rows * batch, 0.0);
    }
    // SAFETY: as in `linear_group`; the caller asserted the weight shapes.
    unsafe {
        run_members::<super::amx::RealTiles>(panel_job_hw, &members, nb * 32, batch, &sets, output)
    };
}

/// BF16 projections of `cpu_float::linear_group` for prompts. `x` holds the input already
/// rounded to bf16. Declines (false) when AMX-BF16 is missing or `cols` is not a multiple of 32.
pub(crate) fn bf16_group(
    projections: &[crate::cpu_float::Projection<'_>],
    x: &[f32],
    cols: usize,
    batch: usize,
    output: &mut [Vec<f32>],
) -> bool {
    if !cols.is_multiple_of(BF16_CHUNK) || !super::ident::features().amx_bf16 {
        return false;
    }
    let acts = Bf16Acts::build(x, batch, cols);
    let members: Vec<Bf16Member<'_>> = projections
        .iter()
        .map(|p| Bf16Member {
            weights: p.weights,
            rows: p.rows,
            bias: p.bias,
        })
        .collect();
    // SAFETY: the `amx` tier is enabled (`wanted`) and AMX-BF16 was checked above; the caller
    // asserted the weight shapes and sized `output`.
    unsafe { run_bf16_members::<super::amx::RealTiles>(&members, cols, batch, &acts, output) };
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_avx512::amx::EmuTiles;
    use crate::cpu_quantized::X;
    use crate::cpu_selftest::{compare, random_weights, reference_packed, Lcg};

    fn has_avx2() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("f16c")
    }

    /// The emulated panel job: same loops as the hardware job, scalar tiles and epilogue.
    unsafe fn emu_job(
        state: &mut State<EmuTiles>,
        m: &Member<'_>,
        acts: &ActSets,
        cols: usize,
        first: usize,
        out: &mut Rows<'_>,
    ) {
        panel_job::<EmuTiles, ScalarEpilogue>(state, m, acts, cols, first, out);
    }

    /// Every weight format against the f64 reference, through the emulated tiles and the scalar
    /// epilogue: decoders, activation transposition, tile shapes, group loops, scales, minimums,
    /// row/token tails and bias.
    #[test]
    fn emulated_int8_gemm_matches_the_reference_for_every_format() {
        if !has_avx2() {
            return;
        }
        let mut rng = Lcg(0x0a77_c0de);
        let kinds = [8u32, 2, 6, 3, 7, 20, 39, 10, 11, 12, 13, 14, 23];
        for kind in kinds {
            let info = dec::kind_info(kind).unwrap();
            let cols = if info.block == 256 { 768 } else { 352 };
            for (rows, batch) in [(13usize, 5usize), (16, 16), (21, 17), (33, 40)] {
                let weights = random_weights(kind, rows, cols, &mut rng);
                let x = rng.vector(batch * cols, 3.0);
                let bias: Vec<f32> = rng.vector(rows, 1.0);
                // Activation records exactly as the kernels consume them.
                let act_kind = match dec::act_kind(kind).unwrap() {
                    dec::ActKind::Q8_0 => 2,
                    dec::ActKind::Q8_1 => 3,
                    dec::ActKind::Q8K => 12,
                };
                let records: Vec<X> = crate::cpu_quantized::prepare(&x, cols, act_kind);
                let nb = cols / 32;
                let members = [Member {
                    weights: &weights,
                    rows,
                    bias: Some(&bias),
                    kind,
                }];
                let sets = act_sets(&members, batch, nb, |token, block| {
                    let r = &records[token * nb + block];
                    Rec {
                        q: &r.q,
                        d: r.d,
                        sum: r.sum,
                    }
                });
                let mut output = vec![vec![0.0f32; rows * batch]];
                unsafe {
                    run_members::<EmuTiles>(emu_job, &members, cols, batch, &sets, &mut output);
                }
                let (mut want, scale) = reference_packed(&weights, kind, rows, cols, &records);
                for t in 0..batch {
                    for r in 0..rows {
                        want[t * rows + r] += f64::from(bias[r]);
                    }
                }
                let (error, ok) = compare(&output[0], &want, &scale);
                assert!(
                    ok,
                    "kind {kind} rows {rows} batch {batch}: max error {error:e}"
                );
            }
        }
    }

    /// Siblings of different group sizes (Q4_K and Q6_K in one Q/K/V group) share one call.
    #[test]
    fn emulated_mixed_group_sizes_share_one_call() {
        if !has_avx2() {
            return;
        }
        let mut rng = Lcg(0x31_41_59);
        let (cols, batch) = (768usize, 20usize);
        let kinds = [12u32, 14, 13];
        let rows = [13usize, 5, 9];
        let weights: Vec<Vec<u8>> = kinds
            .iter()
            .zip(rows)
            .map(|(&k, r)| random_weights(k, r, cols, &mut rng))
            .collect();
        let x = rng.vector(batch * cols, 3.0);
        let records = crate::cpu_quantized::prepare(&x, cols, 12);
        let nb = cols / 32;
        let members: Vec<Member<'_>> = kinds
            .iter()
            .zip(rows)
            .zip(&weights)
            .map(|((&kind, rows), weights)| Member {
                weights,
                rows,
                bias: None,
                kind,
            })
            .collect();
        let sets = act_sets(&members, batch, nb, |token, block| {
            let r = &records[token * nb + block];
            Rec {
                q: &r.q,
                d: r.d,
                sum: r.sum,
            }
        });
        assert!(sets.g16.is_some() && sets.g32.is_some());
        let mut output: Vec<Vec<f32>> = rows.iter().map(|&r| vec![0.0; r * batch]).collect();
        unsafe { run_members::<EmuTiles>(emu_job, &members, cols, batch, &sets, &mut output) };
        for (i, &kind) in kinds.iter().enumerate() {
            let (want, scale) = reference_packed(&weights[i], kind, rows[i], cols, &records);
            let (error, ok) = compare(&output[i], &want, &scale);
            assert!(ok, "member {i} (kind {kind}): {error:e}");
        }
    }

    #[test]
    fn activation_tiles_use_the_vnni_interleave_and_pad_with_zeros() {
        let (batch, nb) = (19usize, 3usize);
        let mut rng = Lcg(5);
        let x = rng.vector(batch * nb * 32, 2.0);
        let records = crate::cpu_quantized::prepare(&x, nb * 32, 2);
        for group in [32usize, 16] {
            let acts = AmxActs::build(batch, nb, group, |t, b| {
                let r = &records[t * nb + b];
                Rec {
                    q: &r.q,
                    d: r.d,
                    sum: r.sum,
                }
            });
            assert_eq!(acts.tiles, 2);
            assert_eq!(acts.groups, nb * 32 / group);
            for t in 0..acts.tiles {
                for n in 0..16 {
                    let token = 16 * t + n;
                    for gi in 0..acts.groups {
                        let base = (t * acts.groups + gi) * 16 * group;
                        for k in 0..group {
                            let at = base + ((k / 4) * 16 + n) * 4 + k % 4;
                            let want = if token < batch {
                                let value = nb_value(&records, nb, token, gi * group + k);
                                value as u8
                            } else {
                                0
                            };
                            assert_eq!(acts.b[at], want, "t{t} n{n} g{gi} k{k}");
                        }
                        let dx = acts.dx[(t * acts.groups + gi) * 16 + n];
                        if token < batch {
                            assert_eq!(dx, records[token * nb + gi * group / 32].d);
                        } else {
                            assert_eq!(dx, 0.0);
                        }
                    }
                }
            }
        }
    }

    fn nb_value(records: &[X], nb: usize, token: usize, k: usize) -> i8 {
        records[token * nb + k / 32].q[k % 32]
    }

    #[test]
    fn emulated_bf16_gemm_matches_the_float_reference() {
        let mut rng = Lcg(0xbf16_5eed);
        for (rows, cols, batch) in [
            (13usize, 64usize, 5usize),
            (32, 96, 33),
            (47, 128, 17),
            (16, 32, 16),
        ] {
            let w: Vec<f32> = (0..rows * cols).map(|_| rng.unit()).collect();
            let wbytes: Vec<u8> = w
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                .collect();
            let x: Vec<f32> = rng
                .vector(batch * cols, 2.0)
                .into_iter()
                .map(|v| half::bf16::from_f32(v).to_f32())
                .collect();
            let bias = rng.vector(rows, 1.0);
            let acts = Bf16Acts::build(&x, batch, cols);
            let members = [Bf16Member {
                weights: &wbytes,
                rows,
                bias: Some(&bias),
            }];
            let mut output = vec![vec![0.0f32; rows * batch]];
            unsafe { run_bf16_members::<EmuTiles>(&members, cols, batch, &acts, &mut output) };
            let mut want = vec![0.0f64; rows * batch];
            let mut scale = vec![0.0f64; rows * batch];
            for t in 0..batch {
                for r in 0..rows {
                    let mut sum = f64::from(bias[r]);
                    let mut mag = f64::from(bias[r]).abs();
                    for k in 0..cols {
                        let term = f64::from(half::bf16::from_f32(w[r * cols + k]).to_f32())
                            * f64::from(x[t * cols + k]);
                        sum += term;
                        mag += term.abs();
                    }
                    want[t * rows + r] = sum;
                    scale[t * rows + r] = mag;
                }
            }
            let (error, ok) = compare(&output[0], &want, &scale);
            assert!(ok, "rows {rows} cols {cols} batch {batch}: {error:e}");
        }
    }
}
