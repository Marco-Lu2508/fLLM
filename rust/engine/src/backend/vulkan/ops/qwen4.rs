//! Plans of the Qwen3.8-Flash-Next operations (`Stream::q4_act`, `q4_gate_mix`, `q4_ple_gate`, `q4_ple_conv`, `q4_pool_keys`). Shaders: `misc_q4_act`,
//! `misc_q4_gate_mix`, `misc_q4_ple_gate`, `misc_q4_ple_conv`, `misc_q4_pool_keys`; layouts in `shaders/vulkan/INTERFACE_OPS.md`.

use super::recurrent::RunSpec;
use super::*;

/// Most residual streams the shaders take.
pub const MAX_STREAMS: usize = 8;
/// Longest history of the n-gram convolution (`(kernel - 1) * dilation` values per channel, kept in registers).
pub const MAX_HISTORY: usize = 32;
/// Widest indexer key one workgroup of `misc_q4_pool_keys` normalises and rotates.
pub const MAX_POOL_DIM: usize = 256;
/// Storage buffers of the widest shader (`misc_q4_pool_keys`).
pub const MAX_BINDINGS: u32 = 7;
/// Tokens of one run that a single `misc_q4_ple_conv` dispatch walks (each invocation owns a channel and loops over them); longer runs are submitted in pieces.
pub const CONV_PIECE_ROWS: usize = 1024;

/// Whether the shaders take `hc` streams and a convolution of `kernel` taps `dilation` rows apart.
pub fn qwen4_supported(hc: usize, kernel: usize, dilation: usize) -> bool {
    (1..=MAX_STREAMS).contains(&hc) && kernel >= 1 && dilation >= 1 && (kernel - 1) * dilation <= MAX_HISTORY
}

/// `q4_act` (`misc_q4_act`): `x = factor * act(x * scale)` over `count` floats, `mode` 0 SiLU, 1 sigmoid. Slot: `x` (rw).
pub fn plan_q4_act(count: usize, mode: i32, scale: f32, factor: f32) -> Result<OpPlan, BackendError> {
    if count == 0 || !(0..=1).contains(&mode) {
        return Err(invalid("q4_act: empty tensor or unknown activation"));
    }
    plan("misc_q4_act", Push::new().u("count", count).u("mode", mode as usize).f("scale", scale).f("factor", factor), groups_1d(count, 256), Slots::new().rw("x"))
}

/// `q4_gate_mix` (`misc_q4_gate_mix`): one invocation per (row, d). Slots: `out`, `normed`, `gates`.
pub fn plan_q4_gate_mix(rows: usize, hc: usize, hidden: usize) -> Result<OpPlan, BackendError> {
    if rows == 0 || hidden == 0 || !(1..=MAX_STREAMS).contains(&hc) {
        return Err(invalid(format!("q4_gate_mix: {rows} rows, {hc} streams of {hidden}")));
    }
    plan("misc_q4_gate_mix", Push::new().u("count", rows * hidden).u("hc", hc).u("hidden", hidden), groups_1d(rows * hidden, 256), Slots::new().write("out").read("normed").read("gates"))
}

/// `q4_ple_gate` (`misc_q4_ple_gate`): one workgroup per (row, stream). Slots: `out`, `key`, `query`, `value`.
pub fn plan_q4_ple_gate(rows: usize, hc: usize, hidden: usize) -> Result<OpPlan, BackendError> {
    if rows == 0 || hidden == 0 || !(1..=MAX_STREAMS).contains(&hc) {
        return Err(invalid(format!("q4_ple_gate: {rows} rows, {hc} streams of {hidden}")));
    }
    plan("misc_q4_ple_gate", Push::new().u("pairs", rows * hc).u("hc", hc).u("hidden", hidden), groups_rows(rows * hc), Slots::new().write("out").read("key").read("query").read("value"))
}

/// `q4_ple_conv` (`misc_q4_ple_conv`) for one run. Slots: `gated`, `normed`, `conv`, `state` (the run's state buffer), `out`; `state_offset` is the byte offset of the
/// layer's history inside the state buffer.
pub fn plan_q4_ple_conv(run_index: usize, run: &RunSpec, width: usize, kernel: usize, dilation: usize, state_offset: usize) -> Result<OpPlan, BackendError> {
    if !qwen4_supported(1, kernel, dilation) || kernel < 2 || !state_offset.is_multiple_of(4) || run.rows == 0 || width == 0 {
        return Err(invalid("n-gram convolution: unsupported kernel, offset or empty run"));
    }
    let push = Push::new().u("width", width).u("ksz", kernel).u("dilation", dilation).u("row0", run.first_row).u("n", run.rows).b("reset", run.reset).u("state_off", state_offset / 4);
    plan("misc_q4_ple_conv", push, [div_up(width, 128) as u32, 1, 1], Slots::new().read("gated").read("normed").read("conv").run_state("state", run_index).write("out"))
}

/// `q4_pool_keys` (`misc_q4_pool_keys`): one workgroup per row. Slots: `out`, `k`, `store`, `norm`, `cos`, `sin`; `meta` is a host array of `ratio + 1` ints per row.
#[allow(clippy::too_many_arguments)]
pub fn plan_q4_pool_keys(meta: &[i32], dim: usize, width: usize, ratio: usize, rope_half: usize, neox: bool, eps: f32) -> Result<OpPlan, BackendError> {
    if ratio == 0 || meta.is_empty() || !meta.len().is_multiple_of(ratio + 1) || !(2..=MAX_POOL_DIM).contains(&dim) || width < dim || 2 * rope_half > dim {
        return Err(invalid("q4_pool_keys: metadata that is not whole rows, or a key width the shader does not take"));
    }
    let rows = meta.len() / (ratio + 1);
    let bytes: Vec<u8> = meta.iter().flat_map(|v| v.to_le_bytes()).collect();
    let push = Push::new().u("rows", rows).u("dim", dim).u("width", width).u("ratio", ratio).u("rope_half", rope_half).b("neox", neox).f("eps", eps);
    plan(
        "misc_q4_pool_keys",
        push,
        groups_rows(rows),
        Slots::new().write("out").read("k").read("store").read("norm").read("cos").read("sin").host("meta", Arc::new(bytes)),
    )
}
