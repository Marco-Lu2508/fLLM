//! Prompt GEMM on 16-row weight panels with the rows in the vector lanes (AVX-512 VNNI).
//!
//! The streaming K-format kernels decode every weight block again for each token pair and reduce
//! horizontally after every dot. For prompts a worker instead decodes its 16-row panel once
//! (`amx_decode`, the same decoders the AMX kernels use), transposes it so that one 64-byte vector
//! holds the same four-value slice of all 16 rows (`lanes[k / 4][row][4 bytes]`), and then one
//! `vpdpbusd` against a broadcast four-value slice of a token advances the dot products of 16 rows
//! at once. There are no horizontal sums and no pack/madd scale step: every scale group (16 or 32
//! values) is converted to float once and scaled there, with the minimum subtracted using the
//! group's activation sum. This is the arrangement of `cpu_kquant::r8_*` (eight rows, 256-bit),
//! widened to 16 rows and generalised to every format with an AMX decoder.
//!
//! Signed formats (Q8_0, Q4_0, Q6_K, codebooks, ...) are stored as `q + 128` (unsigned, as
//! `vpdpbusd` needs) and the offset becomes a minimum of `128 * scale` against the group's
//! activation sum, so all formats share one kernel: `weight = scale * u - min`.
//!
//! The kernel body is one macro instantiated for 512-bit vectors (production) and for 256-bit
//! AVX-VNNI vectors (8 rows per vector, two passes; used by the unit tests, which can run it on any
//! AVX-VNNI CPU and so exercise everything but the 512-bit intrinsics).
use super::amx_decode::{self as dec, AmxPanel, PANEL_ROWS};
use super::amx_gemm::{act_sets, ActSets, AmxActs, Member, Rec};
use crate::cpu_avx512::float::transpose8;
use crate::cpu_schedule::Rows;
use std::arch::x86_64::*;

/// Formats the prompt path takes by default: those whose existing AVX-512 prompt kernels decode
/// each block again per token pair (K formats, IQ4_XS, IQ4_NL, MXFP4). Q8_0, Q4_0 and Q5_0 keep
/// their tuned 4x2 tiles, Q4_1/Q5_1 their W-panel tile. `FLLM_VNNI_GEMM=all` takes every format.
pub(crate) const DEFAULT_KINDS: &[u32] = &[10, 11, 12, 13, 14, 23, 20, 39];

/// Smallest prompt that decodes a panel once for all tokens: two token pairs, as the AVX2
/// predecoded K panels (`cpu_quantized::PREDECODE_MIN_BATCH`).
const MIN_BATCH: usize = 4;

/// A transposed panel and the per-group parameters, as the kernels read them.
pub(crate) struct PanelRefs<'a> {
    /// `[k / 4][16 rows][4 bytes]`, unsigned.
    pub lanes: &'a [u8],
    /// `[group][16 rows]` scales.
    pub scale: &'a [f32],
    /// `[group][16 rows]` minimums including the `128 * scale` offset of signed formats.
    pub min: &'a [f32],
    pub group: usize,
    pub groups: usize,
}

/// A kernel: tokens `n0..n0 + out.len()` of `tile` against the panel; `out[t][row]`.
pub(crate) type Kernel = unsafe fn(&PanelRefs<'_>, &AmxActs, usize, usize, &mut [[f32; 16]]);

/// A wide and a narrow kernel (tokens per pass) for the last few tokens of a tile.
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub wide: Kernel,
    pub wide_tokens: usize,
    pub narrow: Kernel,
    pub narrow_tokens: usize,
}

macro_rules! vnni_kernel {
    (
        $name:ident, $tokens:literal, $lanes:literal, $features:literal,
        ($vi:ty, $zero_si:ident, $loadu_si:ident, $set1_epi32:ident, $dpbusd:ident, $cvt:ident),
        ($zero_ps:ident, $loadu_ps:ident, $storeu_ps:ident, $set1_ps:ident, $mul:ident, $fmadd:ident, $fnmadd:ident)
    ) => {
        /// # Safety
        /// The CPU must support the enabled features; `p` holds a transposed panel of
        /// `p.groups` groups of `p.group` values, `acts` the B-tile activations of the same
        /// width, `tile`/`n0` select tokens `n0..n0 + out.len()` inside the 16 tokens of the tile.
        #[allow(clippy::needless_range_loop)]
        #[target_feature(enable = $features)]
        unsafe fn $name(
            p: &PanelRefs<'_>,
            acts: &AmxActs,
            tile: usize,
            n0: usize,
            out: &mut [[f32; 16]],
        ) {
            const T: usize = $tokens;
            debug_assert!(out.len() == T && n0 + T <= 16);
            let steps = p.group / 4;
            let tile_groups = tile * p.groups;
            // 16 rows are `16 / lanes` vectors of `lanes` rows.
            for part in 0..16 / $lanes {
                let mut acc = [$zero_ps(); T];
                for gi in 0..p.groups {
                    let lane = p.lanes.as_ptr().add(gi * steps * 64 + part * $lanes * 4);
                    let b = acts.b.as_ptr().add((tile_groups + gi) * 16 * p.group);
                    let mut dots: [$vi; T] = [$zero_si(); T];
                    for kk in 0..steps {
                        let w = $loadu_si(lane.add(kk * 64).cast());
                        for t in 0..T {
                            let x = std::ptr::read_unaligned(
                                b.add((kk * 16 + n0 + t) * 4).cast::<i32>(),
                            );
                            dots[t] = $dpbusd(dots[t], w, $set1_epi32(x));
                        }
                    }
                    let scale = $loadu_ps(p.scale.as_ptr().add(gi * 16 + part * $lanes));
                    let min = $loadu_ps(p.min.as_ptr().add(gi * 16 + part * $lanes));
                    for t in 0..T {
                        let at = (tile_groups + gi) * 16 + n0 + t;
                        let scaled = $mul(scale, $set1_ps(acts.dx[at]));
                        acc[t] = $fmadd($cvt(dots[t]), scaled, acc[t]);
                        acc[t] = $fnmadd(min, $set1_ps(acts.sum[at]), acc[t]);
                    }
                }
                for t in 0..T {
                    $storeu_ps(out[t].as_mut_ptr().add(part * $lanes), acc[t]);
                }
            }
        }
    };
}

vnni_kernel!(
    kernel512_t8,
    8,
    16,
    "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni",
    (
        __m512i,
        _mm512_setzero_si512,
        _mm512_loadu_si512,
        _mm512_set1_epi32,
        _mm512_dpbusd_epi32,
        _mm512_cvtepi32_ps
    ),
    (
        _mm512_setzero_ps,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_set1_ps,
        _mm512_mul_ps,
        _mm512_fmadd_ps,
        _mm512_fnmadd_ps
    )
);
vnni_kernel!(
    kernel512_t4,
    4,
    16,
    "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni",
    (
        __m512i,
        _mm512_setzero_si512,
        _mm512_loadu_si512,
        _mm512_set1_epi32,
        _mm512_dpbusd_epi32,
        _mm512_cvtepi32_ps
    ),
    (
        _mm512_setzero_ps,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_set1_ps,
        _mm512_mul_ps,
        _mm512_fmadd_ps,
        _mm512_fnmadd_ps
    )
);
#[cfg(test)]
vnni_kernel!(
    kernel256_t4,
    4,
    8,
    "avx2,fma,avxvnni",
    (
        __m256i,
        _mm256_setzero_si256,
        _mm256_loadu_si256,
        _mm256_set1_epi32,
        _mm256_dpbusd_avx_epi32,
        _mm256_cvtepi32_ps
    ),
    (
        _mm256_setzero_ps,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_set1_ps,
        _mm256_mul_ps,
        _mm256_fmadd_ps,
        _mm256_fnmadd_ps
    )
);
#[cfg(test)]
vnni_kernel!(
    kernel256_t2,
    2,
    8,
    "avx2,fma,avxvnni",
    (
        __m256i,
        _mm256_setzero_si256,
        _mm256_loadu_si256,
        _mm256_set1_epi32,
        _mm256_dpbusd_avx_epi32,
        _mm256_cvtepi32_ps
    ),
    (
        _mm256_setzero_ps,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_set1_ps,
        _mm256_mul_ps,
        _mm256_fmadd_ps,
        _mm256_fnmadd_ps
    )
);

/// The 512-bit kernels.
pub(crate) const KERNELS_512: Kernels = Kernels {
    wide: kernel512_t8,
    wide_tokens: 8,
    narrow: kernel512_t4,
    narrow_tokens: 4,
};

/// The 256-bit AVX-VNNI instantiation of the same kernel body (tests).
#[cfg(test)]
pub(crate) const KERNELS_256: Kernels = Kernels {
    wide: kernel256_t4,
    wide_tokens: 4,
    narrow: kernel256_t2,
    narrow_tokens: 2,
};

/// Transpose a decoded panel into `lanes[k / 4][row][4]` and make the values unsigned (`xor 0x80`
/// for signed formats, i.e. `q + 128`).
///
/// # Safety
/// AVX2 must be available; `lanes` holds `16 * panel.cols` bytes.
#[target_feature(enable = "avx2")]
unsafe fn to_lanes(panel: &AmxPanel, signed: bool, lanes: &mut [u8]) {
    let cols = panel.cols;
    debug_assert!(cols.is_multiple_of(32) && lanes.len() == PANEL_ROWS * cols);
    let flip = _mm256_set1_epi8(if signed { -128 } else { 0 });
    for k in (0..cols).step_by(32) {
        for half in 0..2 {
            // eight rows of eight dwords (32 weights); the dwords move as bit patterns
            let rows: [__m256; 8] = std::array::from_fn(|r| {
                _mm256_castsi256_ps(_mm256_loadu_si256(
                    panel.q.as_ptr().add((half * 8 + r) * cols + k).cast(),
                ))
            });
            let columns = transpose8(rows);
            for (j, column) in columns.iter().enumerate() {
                let bits = _mm256_xor_si256(_mm256_castps_si256(*column), flip);
                _mm256_storeu_si256(
                    lanes.as_mut_ptr().add((k / 4 + j) * 64 + half * 32).cast(),
                    bits,
                );
            }
        }
    }
}

/// Per-worker scratch: the decoded panel (per group size), its lanes and the adjusted minimums.
struct State {
    p16: Option<AmxPanel>,
    p32: Option<AmxPanel>,
    lanes: Vec<u8>,
    min: Vec<f32>,
    tmp: [[f32; 16]; 8],
}

/// One panel of one member against every token.
///
/// # Safety
/// AVX2 and F16C (decoders); the kernels' features; `m.weights` holds `m.rows` complete rows of
/// `m.kind` with `cols` columns.
unsafe fn panel_job(
    kernels: Kernels,
    state: &mut State,
    m: &Member<'_>,
    acts: &ActSets,
    cols: usize,
    first: usize,
    out: &mut Rows<'_>,
) {
    let info = dec::kind_info(m.kind).expect("kind has a decoder");
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
    state.lanes.resize(PANEL_ROWS * cols, 0);
    to_lanes(panel, info.signed, &mut state.lanes);
    state.min.clear();
    state
        .min
        .extend(panel.min.iter().zip(&panel.scale).map(|(&min, &scale)| {
            if info.signed {
                min + 128.0 * scale
            } else {
                min
            }
        }));
    let refs = PanelRefs {
        lanes: &state.lanes,
        scale: &panel.scale,
        min: &state.min,
        group: info.group,
        groups: panel.groups(),
    };
    for tile in 0..a.tiles {
        let in_tile = (a.batch - 16 * tile).min(16);
        let mut n0 = 0;
        while n0 < in_tile {
            let (kernel, tokens) = if in_tile - n0 > kernels.narrow_tokens {
                (kernels.wide, kernels.wide_tokens)
            } else {
                (kernels.narrow, kernels.narrow_tokens)
            };
            // The chunk may reach past the batch: those lanes hold zero activations and are not
            // stored.
            kernel(&refs, a, tile, n0, &mut state.tmp[..tokens]);
            for t in 0..tokens.min(in_tile - n0) {
                let row = out.row(16 * tile + n0 + t);
                for (r, y) in row.iter_mut().enumerate().take(valid) {
                    *y = state.tmp[t][r] + m.bias.map_or(0.0, |b| b[first + r]);
                }
            }
            n0 += tokens;
        }
    }
}

/// Run all members over the shared activations.
///
/// # Safety
/// As `panel_job`; `output[i]` holds `members[i].rows * batch` floats.
unsafe fn run_members(
    kernels: Kernels,
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
            lanes: Vec::new(),
            min: Vec::new(),
            tmp: [[0.0; 16]; 8],
        },
        |state, member, first, out| {
            panel_job(kernels, state, &members[member], acts, cols, first, out)
        },
    );
}

/// `FLLM_VNNI_GEMM`: `0`/`off` disables the path, `all` takes every format with a decoder.
fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("FLLM_VNNI_GEMM") {
        Ok(v)
            if matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "false" | "no"
            ) =>
        {
            0
        }
        Ok(v) if v.trim().eq_ignore_ascii_case("all") => 2,
        _ => 1,
    })
}

/// Whether the prompt path of this GEMM applies: AVX-512 VNNI tier enabled, wide integer kernels
/// allowed by the policy, repacking not switched off, a prompt of at least four tokens and only
/// formats of the default list (every decodable format with `FLLM_VNNI_GEMM=all`).
pub(crate) fn wanted(batch: usize, kinds: &[u32]) -> bool {
    let formats = match mode() {
        0 => return false,
        2 => kinds.iter().all(|&k| dec::kind_info(k).is_some()),
        _ => kinds.iter().all(|k| DEFAULT_KINDS.contains(k)),
    };
    formats
        && !kinds.is_empty()
        && batch >= MIN_BATCH
        && crate::cpu_kernels::repack_enabled()
        && super::ident::policy().wide_int
        && crate::cpu_kernels::tier_enabled(crate::cpu_kernels::CpuTier::Avx512Vnni)
}

/// Build the member list and shared activations of a `cpu_quantized::linear_mixed` call.
fn prepare<'a>(
    projections: &'a [crate::cpu_quantized::Projection<'a>],
    kinds: &[u32],
    cols: usize,
    batch: usize,
    records: &'a [crate::cpu_quantized::X],
) -> (Vec<Member<'a>>, ActSets) {
    let nb = cols / 32;
    let members: Vec<Member<'a>> = projections
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
    (members, sets)
}

/// `cpu_quantized::linear_mixed` for prompts on 512-bit VNNI. `None` if a member has no decoder.
/// The caller has validated shapes and `wanted(batch, kinds)`.
pub(crate) fn linear_group(
    projections: &[crate::cpu_quantized::Projection<'_>],
    kinds: &[u32],
    x: &[f32],
    cols: usize,
    batch: usize,
) -> Option<Vec<Vec<f32>>> {
    if projections.len() != kinds.len() || kinds.iter().any(|&k| dec::kind_info(k).is_none()) {
        return None;
    }
    let prepare_kind = match dec::act_kind(kinds[0])? {
        dec::ActKind::Q8_0 => 2,
        dec::ActKind::Q8_1 => 3,
        dec::ActKind::Q8K => 12,
    };
    let records = crate::cpu_quantized::prepare(x, cols, prepare_kind);
    let (members, sets) = prepare(projections, kinds, cols, batch, &records);
    let mut output: Vec<Vec<f32>> = members.iter().map(|m| vec![0.0; m.rows * batch]).collect();
    // SAFETY: `wanted` checked the `avx512-vnni` tier, which implies AVX-512 F/BW/VL/VNNI, AVX2,
    // FMA and F16C; shapes were validated by the caller.
    unsafe { run_members(KERNELS_512, &members, cols, batch, &sets, &mut output) };
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::{compare, random_weights, reference_packed, Lcg};

    fn has_avx_vnni() -> bool {
        std::arch::is_x86_feature_detected!("avxvnni")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
    }

    fn has_512_vnni() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
    }

    fn check(kernels: Kernels, label: &str) {
        let mut rng = Lcg(0x91_6e_6e ^ label.len() as u64);
        // every format with a decoder: the kernel is format-agnostic, the formats differ only in
        // group size, signedness and minimum
        for kind in [8u32, 2, 6, 3, 7, 20, 39, 10, 11, 12, 13, 14, 23] {
            let info = dec::kind_info(kind).unwrap();
            let cols = if info.block == 256 { 768 } else { 352 };
            for (rows, batch) in [(13usize, 4usize), (16, 5), (21, 16), (33, 17), (5, 40)] {
                let weights = random_weights(kind, rows, cols, &mut rng);
                let x = rng.vector(batch * cols, 3.0);
                let bias = rng.vector(rows, 1.0);
                let projections = [crate::cpu_quantized::Projection {
                    weights: &weights,
                    rows,
                    bias: Some(&bias),
                }];
                let prepare_kind = match dec::act_kind(kind).unwrap() {
                    dec::ActKind::Q8_0 => 2,
                    dec::ActKind::Q8_1 => 3,
                    dec::ActKind::Q8K => 12,
                };
                let records = crate::cpu_quantized::prepare(&x, cols, prepare_kind);
                let (members, sets) = prepare(&projections, &[kind], cols, batch, &records);
                let mut output = vec![vec![0.0f32; rows * batch]];
                unsafe { run_members(kernels, &members, cols, batch, &sets, &mut output) };
                let (mut want, scale) = reference_packed(&weights, kind, rows, cols, &records);
                for t in 0..batch {
                    for r in 0..rows {
                        want[t * rows + r] += f64::from(bias[r]);
                    }
                }
                let (error, ok) = compare(&output[0], &want, &scale);
                assert!(
                    ok,
                    "{label} kind {kind} rows {rows} batch {batch}: {error:e}"
                );
            }
        }
    }

    /// The kernel body on 256-bit AVX-VNNI vectors (the same macro text as the 512-bit one):
    /// transposition, offsets, minimums, group loops, token chunks and tails.
    #[test]
    fn avx_vnni_256_instantiation_matches_the_reference_for_every_format() {
        if has_avx_vnni() {
            check(KERNELS_256, "vnni256");
        }
    }

    #[test]
    fn avx512_vnni_gemm_matches_the_reference_for_every_format() {
        if has_512_vnni() {
            check(KERNELS_512, "vnni512");
        }
    }

    /// `to_lanes`: byte `j` of row `r` at column `k` lands at `lanes[(k / 4) * 64 + r * 4 + k % 4]`,
    /// signed values offset by 128.
    #[test]
    fn lanes_hold_four_columns_of_all_sixteen_rows() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let cols = 96;
        let mut panel = AmxPanel::new(cols, 32);
        let mut rng = Lcg(3);
        for b in panel.q.iter_mut() {
            *b = rng.next() as u8;
        }
        for signed in [false, true] {
            let mut lanes = vec![0u8; PANEL_ROWS * cols];
            unsafe { to_lanes(&panel, signed, &mut lanes) };
            for r in 0..PANEL_ROWS {
                for k in 0..cols {
                    let want = panel.q[r * cols + k] ^ if signed { 0x80 } else { 0 };
                    assert_eq!(lanes[(k / 4) * 64 + r * 4 + k % 4], want, "row {r} col {k}");
                }
            }
        }
    }

    #[test]
    fn default_formats_are_the_ones_without_a_tuned_prompt_tile() {
        for kind in [2u32, 3, 6, 7, 8] {
            assert!(!DEFAULT_KINDS.contains(&kind));
        }
        for kind in [10u32, 11, 12, 13, 14, 20, 23, 39] {
            assert!(DEFAULT_KINDS.contains(&kind));
            assert!(dec::kind_info(kind).is_some());
        }
    }
}
