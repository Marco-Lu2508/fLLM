//! Plans of the chunk-parallel scan of the recurrent layers (`scan_*` shaders, see `shaders/vulkan/_scan_common.glsl`).
//!
//! Gated DeltaNet, Kimi Delta Attention, RWKV-7, RWKV-6 and Mamba-2 run a long prompt run through the chunked form of their recurrence
//! (`rust/engine/src/chunked_scan.rs` proves it equal to the token-by-token form): a per-architecture `*_prep` shader brings the inputs of a window of the
//! run into the layout of the scan, `scan_cum`, `scan_pairs`, `scan_solve`, `scan_out` work on all chunks of 64 tokens at once, `scan_state` walks the
//! chunks in order, a `*_post` shader applies the output norm and gate. The serial shaders (`delta_rec_*`, `kda_rec_*`, `rwkv*_wkv_*`, `mamba_scan2_*`)
//! stay for decode and for runs shorter than [`min_rows`]; `FLLM_CHUNKED_SCAN=0` keeps them for everything, `FLLM_CHUNKED_SCAN_MIN=<rows>` moves the
//! threshold (default 64).
//!
//! A run is cut into windows of at most [`window_rows`] rows (the scratch of a window must stay small and a dispatch short: the state carries over
//! from window to window in the sequence's state buffer). The scratch of a window is slot [`SCAN_SCRATCH_ID`] of the operation.

use super::recurrent::{Mamba2Shape, RunSpec};
use super::*;
use crate::backend::{DeltaShape, KdaShape, RwkvShape};

/// Tokens per chunk.
pub const SCAN_CHUNK: usize = 64;
/// Value columns of the state one workgroup of `scan_state` handles.
const SCAN_COLUMNS: usize = 32;
/// Scratch id of the scan (ids 0 and 1 hold the convolution output of the delta / KDA and Mamba plans).
pub const SCAN_SCRATCH_ID: u8 = 3;
/// Shared memory the largest scan shader needs (`scan_pairs`), bytes.
pub const SCAN_SHARED_BYTES: u32 = 26_624;
/// Most chunks of a window.
const MAX_WINDOW_CHUNKS: usize = 8;
/// Scratch of a window, floats (96 MiB).
const BUDGET_FLOATS: usize = 24 << 20;

pub const RANK1: usize = 1;
pub const BSAME: usize = 2;
pub const LWV: usize = 4;
pub const QPREV: usize = 8;
pub const HASC: usize = 16;

/// Is the chunk-parallel scan allowed (`FLLM_CHUNKED_SCAN` is not `0`, `off` or `false`)?
pub fn scan_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("FLLM_CHUNKED_SCAN").is_ok_and(|v| matches!(v.trim(), "0" | "off" | "false")))
}

/// Rows of a run from which the chunked scan is used (`FLLM_CHUNKED_SCAN_MIN`, default 64).
pub fn min_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| std::env::var("FLLM_CHUNKED_SCAN_MIN").ok().and_then(|v| v.trim().parse().ok()).filter(|&v| v > 0).unwrap_or(64))
}

fn up4(x: usize) -> usize {
    (x + 3) & !3
}

/// Floats of scratch for `slots` chunk slots of a head set (the same arithmetic as `scan_layout` in `_scan_common.glsl`).
pub fn scan_floats(hk: usize, hv: usize, heads: usize, flags: usize, slots: usize) -> usize {
    let r1 = flags & RANK1 != 0;
    let gw = if flags & LWV != 0 { hk } else { 1 };
    let r = |x: usize| if r1 { x } else { 0 };
    let chunk = up4(SCAN_CHUNK * gw)
        + 3 * up4(r(SCAN_CHUNK * SCAN_CHUNK))
        + up4(SCAN_CHUNK * SCAN_CHUNK)
        + up4(r(SCAN_CHUNK * hv))
        + up4(r(SCAN_CHUNK * hk))
        + up4(SCAN_CHUNK * hk)
        + up4(SCAN_CHUNK * hv)
        + up4(SCAN_CHUNK * hk)
        + up4(r(SCAN_CHUNK * hk))
        + up4(hk);
    let rh = slots * SCAN_CHUNK * heads;
    let inputs = 2 * up4(rh * hk)
        + up4(r(rh * hk))
        + up4(if r1 && flags & BSAME == 0 { rh * hk } else { 0 })
        + up4(rh * gw)
        + up4(rh)
        + 2 * up4(rh * hv);
    inputs + slots * heads * chunk
}

/// Can the shaders take heads of this size on a device with `max_shared` bytes of shared memory per workgroup?
pub fn scan_usable(hk: usize, hv: usize, heads: usize, max_shared: u32) -> bool {
    hk >= 16 && hk <= 128 && hk.is_multiple_of(16) && (1..=256).contains(&hv) && heads > 0 && max_shared >= SCAN_SHARED_BYTES + 2048
}

/// Rows of one window of a run for these head sizes.
pub fn window_rows(hk: usize, hv: usize, heads: usize, flags: usize) -> usize {
    let per_slot = scan_floats(hk, hv, heads, flags, 1).max(1);
    (BUDGET_FLOATS / per_slot).clamp(1, MAX_WINDOW_CHUNKS) * SCAN_CHUNK
}

/// Should this run take the chunked scan?
pub fn wanted(max_shared: Option<u32>, rows: usize, hk: usize, hv: usize, heads: usize) -> bool {
    max_shared.is_some_and(|m| rows >= min_rows() && scan_usable(hk, hv, heads, m))
}

/// Head sizes and recurrence flags of one scan.
#[derive(Clone, Copy, Debug)]
pub struct ScanSpec {
    pub hk: usize,
    pub hv: usize,
    pub heads: usize,
    pub flags: usize,
}

/// Where the state of a head is in the run's state buffer: element (head h, key i, value j) at float `off + h * hstride + i * kstride + j * vstride`.
#[derive(Clone, Copy, Debug)]
pub struct ScanState {
    pub off: usize,
    pub hstride: usize,
    pub kstride: usize,
    pub vstride: usize,
}

fn scratch_slots(spec: &ScanSpec, n: usize) -> Slots {
    let slots = div_up(n, SCAN_CHUNK);
    Slots::new().scratch("scratch", SCAN_SCRATCH_ID, scan_floats(spec.hk, spec.hv, spec.heads, spec.flags, slots) * 4, Access::ReadWrite)
}

fn core_push(spec: &ScanSpec, n: usize, reset: bool) -> Push {
    Push::new()
        .u("hk", spec.hk)
        .u("hv", spec.hv)
        .u("heads", spec.heads)
        .u("flags", spec.flags)
        .u("n", n)
        .b("reset", reset)
        .u("slots", div_up(n, SCAN_CHUNK))
}

/// The five shaders of the chunk algebra for one window of `n` rows (`scan_cum`, `scan_pairs`, [`scan_solve`], `scan_out`, `scan_state`).
pub fn plan_scan_core(spec: &ScanSpec, n: usize, reset: bool, run_index: usize, state: &ScanState) -> Result<Vec<OpPlan>, BackendError> {
    if n == 0 || spec.heads > MAX_GROUPS_PER_DIM as usize {
        return Err(invalid("scan of an empty window"));
    }
    let slots = div_up(n, SCAN_CHUNK);
    let r1 = spec.flags & RANK1 != 0;
    let mut plans = Vec::with_capacity(5);
    let grid = [slots as u32, spec.heads as u32, 1];
    plans.push(plan("scan_cum", core_push(spec, n, reset), grid, scratch_slots(spec, n))?);
    plans.push(plan("scan_pairs", core_push(spec, n, reset), grid, scratch_slots(spec, n))?);
    if r1 {
        let cb = div_up(spec.hv + spec.hk, 64);
        plans.push(plan("scan_solve", core_push(spec, n, reset), [slots as u32, (spec.heads * cb) as u32, 1], scratch_slots(spec, n))?);
    }
    let cb = div_up(if r1 { spec.hk + spec.hv } else { spec.hv }, 64);
    plans.push(plan("scan_out", core_push(spec, n, reset), [slots as u32, (spec.heads * cb) as u32, 1], scratch_slots(spec, n))?);
    let push = core_push(spec, n, reset).u("state_off", state.off).u("hstride", state.hstride).u("kstride", state.kstride).u("vstride", state.vstride);
    plans.push(plan(
        "scan_state",
        push,
        [spec.heads as u32, div_up(spec.hv, SCAN_COLUMNS) as u32, 1],
        scratch_slots(spec, n).run_state("state", run_index),
    )?);
    Ok(plans)
}

/// Windows `(first row, rows, reset)` of a run.
fn windows(run: &RunSpec, window: usize) -> Vec<(usize, usize, bool)> {
    (0..run.rows.div_ceil(window))
        .map(|k| (run.first_row + k * window, (run.rows - k * window).min(window), run.reset && k == 0))
        .collect()
}

fn check_offset(state_offset: usize) -> Result<(), BackendError> {
    if !state_offset.is_multiple_of(4) {
        return Err(invalid("scan state offset must be a multiple of 4 bytes"));
    }
    Ok(())
}

/// Gated DeltaNet: the plans after `delta_conv` (whose output is scratch id 0 of `mixed_bytes`) for a long run.
/// Slots: `beta`, `alpha`, `dt_bias`, `decay`, `z`, `norm`, `state`, `out`.
pub fn plan_scan_delta_run(shape: &DeltaShape, run_index: usize, run: &RunSpec, state_offset: usize, mixed_bytes: usize) -> Result<Vec<OpPlan>, BackendError> {
    check_offset(state_offset)?;
    let flags = RANK1 | BSAME;
    let spec = ScanSpec { hk: shape.k_dim, hv: shape.v_dim, heads: shape.v_heads, flags };
    let state = ScanState { off: state_offset / 4, hstride: shape.k_dim * shape.v_dim, kstride: shape.v_dim, vstride: 1 };
    let mut plans = Vec::new();
    for (row0, n, reset) in windows(run, window_rows(spec.hk, spec.hv, spec.heads, flags)) {
        let slots = div_up(n, SCAN_CHUNK);
        let prep = Push::new()
            .u("kh", shape.k_heads)
            .u("vh", shape.v_heads)
            .u("hk", shape.k_dim)
            .u("hv", shape.v_dim)
            .u("conv_dim", shape.conv_dim())
            .b("tiled", shape.tiled_heads)
            .u("row0", row0)
            .u("n", n)
            .u("slots", slots)
            .f("beta_scale", shape.beta_scale);
        let prep_slots = Slots::new()
            .scratch("mixed", 0, mixed_bytes, Access::Read)
            .read("beta")
            .read("alpha")
            .read("dt_bias")
            .read("decay")
            .scratch("scratch", SCAN_SCRATCH_ID, scan_floats(spec.hk, spec.hv, spec.heads, flags, slots) * 4, Access::ReadWrite);
        plans.push(plan("scan_delta_prep", prep, [n as u32, 1, 1], prep_slots)?);
        plans.extend(plan_scan_core(&spec, n, reset, run_index, &state)?);
        let post = Push::new().u("vh", shape.v_heads).u("hk", shape.k_dim).u("hv", shape.v_dim).u("row0", row0).u("n", n).u("slots", slots).f("eps", shape.eps);
        let post_slots = Slots::new()
            .read("z")
            .read("norm")
            .scratch("scratch", SCAN_SCRATCH_ID, scan_floats(spec.hk, spec.hv, spec.heads, flags, slots) * 4, Access::ReadWrite)
            .write("out");
        plans.push(plan("scan_delta_post", post, [shape.v_heads as u32, n as u32, 1], post_slots)?);
    }
    Ok(plans)
}

/// Kimi Delta Attention: the plans after `delta_conv` for a long run. Slots: `forget`, `beta`, `dt_bias`, `rate`, `gate`, `norm`, `state`, `out`.
pub fn plan_scan_kda_run(shape: &KdaShape, run_index: usize, run: &RunSpec, state_offset: usize, mixed_bytes: usize) -> Result<Vec<OpPlan>, BackendError> {
    check_offset(state_offset)?;
    let (heads, hd) = (shape.heads, shape.head_dim);
    let flags = RANK1 | BSAME | LWV;
    let spec = ScanSpec { hk: hd, hv: hd, heads, flags };
    let state = ScanState { off: state_offset / 4, hstride: hd * hd, kstride: hd, vstride: 1 };
    let mut plans = Vec::new();
    for (row0, n, reset) in windows(run, window_rows(hd, hd, heads, flags)) {
        let slots = div_up(n, SCAN_CHUNK);
        let bytes = scan_floats(hd, hd, heads, flags, slots) * 4;
        let prep = Push::new()
            .u("heads", heads)
            .u("hd", hd)
            .u("conv_dim", shape.conv_dim())
            .u("row0", row0)
            .u("n", n)
            .u("slots", slots)
            .b("bounded", shape.lower_bound.is_some())
            .f("lower_bound", shape.lower_bound.unwrap_or(0.0));
        let prep_slots = Slots::new()
            .scratch("mixed", 0, mixed_bytes, Access::Read)
            .read("forget")
            .read("beta")
            .read("dt_bias")
            .read("rate")
            .scratch("scratch", SCAN_SCRATCH_ID, bytes, Access::ReadWrite);
        plans.push(plan("scan_kda_prep", prep, [n as u32, 1, 1], prep_slots)?);
        plans.extend(plan_scan_core(&spec, n, reset, run_index, &state)?);
        let post = Push::new().u("heads", heads).u("hd", hd).u("row0", row0).u("n", n).u("slots", slots).f("eps", shape.eps);
        let post_slots = Slots::new().read("gate").read("norm").scratch("scratch", SCAN_SCRATCH_ID, bytes, Access::ReadWrite).write("out");
        plans.push(plan("scan_kda_post", post, [heads as u32, n as u32, 1], post_slots)?);
    }
    Ok(plans)
}

/// RWKV-7: the plans of a long run. Slots: `r`, `k`, `v`, `w`, `a`, `vres` (dummy without), `k_k`, `k_a`, `r_k`, `v_first`, `gate`, `norm_w`, `norm_b`, `state`, `out`.
pub fn plan_scan_rwkv7_run(shape: &RwkvShape, run_index: usize, run: &RunSpec, state_offset: usize, has_vres: bool) -> Result<Vec<OpPlan>, BackendError> {
    check_offset(state_offset)?;
    let (n_head, heads) = (shape.head_size, shape.heads());
    let flags = RANK1 | LWV;
    let spec = ScanSpec { hk: n_head, hv: n_head, heads, flags };
    let state = ScanState { off: state_offset / 4, hstride: n_head * n_head, kstride: 1, vstride: n_head };
    let mut plans = Vec::new();
    for (row0, n, reset) in windows(run, window_rows(n_head, n_head, heads, flags)) {
        let slots = div_up(n, SCAN_CHUNK);
        let bytes = scan_floats(n_head, n_head, heads, flags, slots) * 4;
        let prep = Push::new().u("hidden", shape.hidden).u("heads", heads).u("hn", n_head).u("row0", row0).u("n", n).u("slots", slots).b("has_vres", has_vres);
        let prep_slots = Slots::new()
            .read("r")
            .read("k")
            .read("v")
            .read("w")
            .read("a")
            .read_opt("vres", has_vres)
            .read("k_k")
            .read("k_a")
            .read("r_k")
            .rw("v_first")
            .scratch("scratch", SCAN_SCRATCH_ID, bytes, Access::ReadWrite);
        plans.push(plan("scan_rwkv7_prep", prep, [heads as u32, n as u32, 1], prep_slots)?);
        plans.extend(plan_scan_core(&spec, n, reset, run_index, &state)?);
        let post = Push::new().u("hidden", shape.hidden).u("heads", heads).u("hn", n_head).u("row0", row0).u("n", n).u("slots", slots).f("eps", shape.group_eps);
        let post_slots = Slots::new().read("gate").read("norm_w").read("norm_b").scratch("scratch", SCAN_SCRATCH_ID, bytes, Access::ReadWrite).write("out");
        plans.push(plan("scan_rwkv7_post", post, [heads as u32, n as u32, 1], post_slots)?);
    }
    Ok(plans)
}

/// RWKV-6: the plans of a long run. Slots: `r`, `k`, `v`, `w`, `first`, `gate`, `norm_w`, `norm_b`, `state`, `out`.
pub fn plan_scan_rwkv6_run(shape: &RwkvShape, run_index: usize, run: &RunSpec, state_offset: usize) -> Result<Vec<OpPlan>, BackendError> {
    check_offset(state_offset)?;
    let (n_head, heads) = (shape.head_size, shape.heads());
    let flags = LWV | QPREV | HASC;
    let spec = ScanSpec { hk: n_head, hv: n_head, heads, flags };
    let state = ScanState { off: state_offset / 4, hstride: n_head * n_head, kstride: n_head, vstride: 1 };
    let mut plans = Vec::new();
    for (row0, n, reset) in windows(run, window_rows(n_head, n_head, heads, flags)) {
        let slots = div_up(n, SCAN_CHUNK);
        let bytes = scan_floats(n_head, n_head, heads, flags, slots) * 4;
        let prep = Push::new().u("hidden", shape.hidden).u("heads", heads).u("hn", n_head).u("row0", row0).u("n", n).u("slots", slots);
        let prep_slots = Slots::new().read("r").read("k").read("v").read("w").read("first").scratch("scratch", SCAN_SCRATCH_ID, bytes, Access::ReadWrite);
        plans.push(plan("scan_rwkv6_prep", prep, [heads as u32, n as u32, 1], prep_slots)?);
        plans.extend(plan_scan_core(&spec, n, reset, run_index, &state)?);
        let post = Push::new().u("hidden", shape.hidden).u("heads", heads).u("hn", n_head).u("row0", row0).u("n", n).u("slots", slots).f("eps", shape.group_eps);
        let post_slots = Slots::new().read("gate").read("norm_w").read("norm_b").scratch("scratch", SCAN_SCRATCH_ID, bytes, Access::ReadWrite).write("out");
        plans.push(plan("scan_rwkv6_post", post, [heads as u32, n as u32, 1], post_slots)?);
    }
    Ok(plans)
}

/// Mamba-2: the plans after `mamba_conv` (scratch `mixed_scratch` = (id, bytes)) for a long run; `mamba_gate_norm` follows for all rows. Slots: `projected`,
/// `dt_bias`, `decay`, `skip`, `state`, `out`. `rec_off`: float offset of the recurrent state in the run's state buffer.
pub fn plan_scan_mamba2_run(
    shape: &Mamba2Shape,
    run_index: usize,
    run: &RunSpec,
    proj_stride: usize,
    proj_dt_off: usize,
    rec_off: usize,
    mixed_scratch: (u8, usize),
) -> Result<Vec<OpPlan>, BackendError> {
    let (hd, ns, heads) = (shape.head_dim(), shape.state, shape.heads);
    let flags = HASC;
    let spec = ScanSpec { hk: ns, hv: hd, heads, flags };
    let state = ScanState { off: rec_off, hstride: hd * ns, kstride: 1, vstride: ns };
    let mut plans = Vec::new();
    for (row0, n, reset) in windows(run, window_rows(ns, hd, heads, flags)) {
        let slots = div_up(n, SCAN_CHUNK);
        let bytes = scan_floats(ns, hd, heads, flags, slots) * 4;
        let prep = Push::new()
            .u("inner", shape.inner)
            .u("ns", ns)
            .u("heads", heads)
            .u("groups", shape.groups)
            .u("head_dim", hd)
            .u("conv_dim", shape.conv_dim())
            .u("proj_stride", proj_stride)
            .u("proj_dt_off", proj_dt_off)
            .u("row0", row0)
            .u("n", n)
            .u("slots", slots)
            .f("step_min", shape.step_min);
        let prep_slots = Slots::new()
            .scratch("mixed", mixed_scratch.0, mixed_scratch.1, Access::Read)
            .read("projected")
            .read("dt_bias")
            .read("decay")
            .read("skip")
            .scratch("scratch", SCAN_SCRATCH_ID, bytes, Access::ReadWrite);
        plans.push(plan("scan_mamba2_prep", prep, [n as u32, 1, 1], prep_slots)?);
        plans.extend(plan_scan_core(&spec, n, reset, run_index, &state)?);
        let post = Push::new().u("ns", ns).u("head_dim", hd).u("heads", heads).u("inner", shape.inner).u("row0", row0).u("n", n).u("slots", slots);
        plans.push(plan("scan_copy_out", post, [n as u32, 1, 1], Slots::new().scratch("scratch", SCAN_SCRATCH_ID, bytes, Access::ReadWrite).write("out"))?);
    }
    Ok(plans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_grows_with_the_slots_and_windows_cover_the_run() {
        for flags in [RANK1 | BSAME, RANK1 | BSAME | LWV, RANK1 | LWV, LWV | QPREV | HASC, HASC] {
            let one = scan_floats(64, 64, 8, flags, 1);
            let four = scan_floats(64, 64, 8, flags, 4);
            assert_eq!(four, 4 * one, "flags {flags}: the scratch is linear in the slots");
        }
        let run = RunSpec { first_row: 10, rows: 1000, reset: true };
        for window in [64usize, 192, 512] {
            let w = windows(&run, window);
            assert_eq!(w.iter().map(|x| x.1).sum::<usize>(), 1000);
            assert_eq!(w[0], (10, window, true));
            assert!(w[1..].iter().all(|x| !x.2));
            assert!(w.windows(2).all(|p| p[1].0 == p[0].0 + p[0].1));
            assert!(w.iter().all(|x| x.1 <= window));
        }
        // the window of a big model stays within the scratch budget
        let rows = window_rows(128, 128, 32, RANK1 | BSAME);
        assert!(rows % SCAN_CHUNK == 0 && rows >= SCAN_CHUNK && rows <= MAX_WINDOW_CHUNKS * SCAN_CHUNK);
        assert!(scan_floats(128, 128, 32, RANK1 | BSAME, rows / SCAN_CHUNK) <= BUDGET_FLOATS || rows == SCAN_CHUNK);
    }

    #[test]
    fn short_runs_and_small_devices_keep_the_serial_shaders() {
        assert!(!wanted(None, 1000, 64, 64, 8));
        assert!(!wanted(Some(65536), 63, 64, 64, 8));
        assert!(wanted(Some(65536), 64, 64, 64, 8));
        assert!(!wanted(Some(16384), 1000, 64, 64, 8));
        assert!(!wanted(Some(65536), 1000, 256, 64, 8));
        assert!(!wanted(Some(65536), 1000, 24, 64, 8));
        assert!(scan_usable(96, 96, 4, 32768));
    }

    #[test]
    fn a_window_is_five_or_six_core_dispatches_with_the_state_last() {
        let spec = ScanSpec { hk: 64, hv: 64, heads: 4, flags: RANK1 | BSAME };
        let state = ScanState { off: 64, hstride: 4096, kstride: 64, vstride: 1 };
        let plans = plan_scan_core(&spec, 130, true, 2, &state).unwrap();
        let names: Vec<&str> = plans.iter().map(|p| p.shader_name).collect();
        assert_eq!(names, ["scan_cum", "scan_pairs", "scan_solve", "scan_out", "scan_state"]);
        assert_eq!(plans[0].groups, [3, 4, 1]);
        assert_eq!(plans[2].groups, [3, 4 * 2, 1]);
        assert_eq!(plans[4].groups, [4, 2, 1]);
        let plain = plan_scan_core(&ScanSpec { flags: HASC, ..spec }, 64, false, 0, &state).unwrap();
        assert_eq!(plain.iter().map(|p| p.shader_name).collect::<Vec<_>>(), ["scan_cum", "scan_pairs", "scan_out", "scan_state"]);
    }
}
