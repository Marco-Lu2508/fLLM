//! Plans of the hyper-connection operations (`Stream::hc_mix`, `Stream::hc_combine`). Shaders: `misc_hc_mix`, `misc_hc_combine`; layouts in
//! `shaders/vulkan/INTERFACE_OPS.md`.

use super::*;
use crate::backend::HcMode;

/// Most residual streams the shaders take (private arrays of the Sinkhorn projection).
pub const MAX_STREAMS: usize = 8;

/// `rotate_heads` (`misc_rotate_heads`): `out[g][(d + shift) % dim] = x[g][d]` over `count` floats of groups of `dim`.
pub fn plan_rotate_heads(count: usize, dim: usize, shift: usize) -> Result<OpPlan, BackendError> {
    if count == 0 || dim == 0 || shift >= dim || !count.is_multiple_of(dim) {
        return Err(invalid("head rotation: empty tensor, or a shift outside the head"));
    }
    plan("misc_rotate_heads", Push::new().u("count", count).u("dim", dim).u("shift", shift), groups_1d(count, 256), Slots::new().read("x").write("out"))
}

/// `hc_mix`: one workgroup per row. Slots: `streams`, `mixes`, `base`, `scale` (the last three are dummies for [`HcMode::Mean`]), `post`, `comb`, `collapsed`.
pub fn plan_hc_mix(rows: usize, hc: usize, hidden: usize, mode: HcMode, eps: f32) -> Result<OpPlan, BackendError> {
    if rows == 0 || hidden == 0 || hc == 0 || hc > MAX_STREAMS {
        return Err(invalid(format!("hyper-connection mix: {rows} rows, {hc} streams of {hidden}")));
    }
    let (code, iters, magnitude) = match mode {
        HcMode::Independent { magnitude } => (0, 0, magnitude),
        HcMode::Sinkhorn { iters } => (1, iters.max(1), 0.0),
        HcMode::Head => (2, 0, 0.0),
        HcMode::Mean => (3, 0, 0.0),
    };
    let weights = !matches!(mode, HcMode::Mean);
    plan(
        "misc_hc_mix",
        Push::new().u("hc", hc).u("hidden", hidden).u("mode", code).u("iters", iters).f("eps", eps).f("magnitude", magnitude),
        groups_rows(rows),
        Slots::new().read("streams").read_opt("mixes", weights).read_opt("base", weights).read_opt("scale", weights).rw("post").rw("comb").write("collapsed"),
    )
}

/// `hc_combine`: `rows * hidden` invocations, each updating its `hc` values. Slots: `streams` (rw), `post`, `comb` (a dummy without), `block`.
pub fn plan_hc_combine(rows: usize, hc: usize, hidden: usize, has_comb: bool) -> Result<OpPlan, BackendError> {
    if rows == 0 || hidden == 0 || hc == 0 || hc > MAX_STREAMS {
        return Err(invalid(format!("hyper-connection update: {rows} rows, {hc} streams of {hidden}")));
    }
    plan(
        "misc_hc_combine",
        Push::new().u("count", rows * hidden).u("hc", hc).u("hidden", hidden).b("has_comb", has_comb),
        groups_1d(rows * hidden, 256),
        Slots::new().rw("streams").read("post").read_opt("comb", has_comb).read("block"),
    )
}
