//! Plans of the recurrent mixers: Gated DeltaNet (`Stream::delta_layer`) and Mamba-1 / Mamba-2 (no `Stream` method yet: the pieces are planned here so
//! that the stream can offer one, see INTERFACE_OPS.md). Shaders: `delta_conv`, `delta_rec_hk<N>`, `mamba_conv`, `mamba_scan1_*`, `mamba_scan2_*`,
//! `mamba_gate_norm`, plus `norm_rms` for the Mamba-1 B / C normalisation.

use super::*;
use crate::backend::DeltaShape;

/// Tokens per workgroup column of the convolution shaders (must be >= conv_kernel - 1).
pub const CONV_CHUNK: usize = 16;

/// Consecutive rows of one sequence (mirror of [`crate::backend::DeltaRun`] without the buffer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunSpec {
    pub first_row: usize,
    pub rows: usize,
    /// The sequence starts with this run: the stored state is ignored.
    pub reset: bool,
}

/// Whether the DeltaNet shaders take a layer of this shape (key width 32 / 64 / 96 / 128 with an `delta_rec_hk*` variant, value width up to 128, kernel 2..8).
pub fn kda_supported(shape: &crate::backend::KdaShape) -> bool {
    matches!(shape.head_dim, 16 | 32 | 64 | 128)
        && shape.heads > 0
        && (2..=8).contains(&shape.conv_kernel)
}

pub fn delta_supported(shape: &DeltaShape) -> bool {
    matches!(shape.k_dim, 32 | 64 | 96 | 128)
        && (1..=128).contains(&shape.v_dim)
        && shape.k_heads > 0
        && shape.v_heads > 0
        && shape.v_heads.is_multiple_of(shape.k_heads)
        && (2..=8).contains(&shape.conv_kernel)
}

fn delta_rec_shader(hk: usize, cols: bool) -> Option<&'static str> {
    Some(match (hk, cols) {
        (32, false) => "delta_rec_hk32",
        (64, false) => "delta_rec_hk64",
        (96, false) => "delta_rec_hk96",
        (128, false) => "delta_rec_hk128",
        (32, true) => "delta_recc_hk32",
        (64, true) => "delta_recc_hk64",
        (96, true) => "delta_recc_hk96",
        (128, true) => "delta_recc_hk128",
        _ => return None,
    })
}

/// Value columns per workgroup of `delta_recc_hk*`.
const RECC_COLUMNS: usize = 16;

/// Whether the column-split recurrence (`delta_recc_hk*` + `delta_gnorm`) can run: it needs subgroup shuffles and subgroups of at least 4 lanes.
/// `FLLM_VK_DELTA_COLS=0` keeps the one-workgroup-per-head shader.
pub fn delta_cols_usable(shuffle: bool, min_subgroup: u32) -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let off = *OFF.get_or_init(|| std::env::var("FLLM_VK_DELTA_COLS").is_ok_and(|v| matches!(v.trim(), "0" | "off" | "false")));
    shuffle && min_subgroup >= 4 && !off
}

/// Plans of `Stream::delta_layer`: for every non-empty run a `delta_conv` and a `delta_rec_hk*` dispatch (in this order, barrier in between; the runs are
/// independent of each other as every run has its own state buffer, slot `state` = [`SlotSource::RunState`]). `state_offset` is the byte offset of the
/// layer's state inside each run's state buffer (the recurrent matrices `v_heads x k_dim x v_dim` f32, then the convolution history `conv_dim x (kernel - 1)`
/// f32 at `rec_bytes()`); it must be a multiple of 4. Slots of `delta_conv`: `qkv`, `conv`, `state`, `mixed` (scratch id 0, `rows x conv_dim` floats;
/// the caller's `mixed` argument), of `delta_rec`: `mixed`, `z`, `beta`, `alpha`, `dt_bias`, `decay`, `norm`, `state`, `out`.
pub fn plan_delta_layer(
    shape: &DeltaShape,
    runs: &[RunSpec],
    state_offset: usize,
) -> Result<Vec<OpPlan>, BackendError> {
    plan_delta_layer_with(shape, runs, state_offset, false)
}

/// [`plan_delta_layer`]; `cols`: the recurrence as `delta_recc_hk*` over `ceil(v_dim / 16)` workgroups per value head followed by the gated norm
/// `delta_gnorm` (slots `z`, `norm`, `out`), see [`delta_cols_usable`].
pub fn plan_delta_layer_with(
    shape: &DeltaShape,
    runs: &[RunSpec],
    state_offset: usize,
    cols: bool,
) -> Result<Vec<OpPlan>, BackendError> {
    plan_delta_layer_scan(shape, runs, state_offset, cols, None)
}

/// [`plan_delta_layer_with`]; `scan`: the shared memory of the device when the runs of [`super::scan::min_rows`] rows and more may take the chunk-parallel scan
/// (`scan_delta_prep`, `scan_cum` .. `scan_state`, `scan_delta_post` after the `delta_conv` of the run, see [`super::scan`]).
pub fn plan_delta_layer_scan(
    shape: &DeltaShape,
    runs: &[RunSpec],
    state_offset: usize,
    cols: bool,
    scan: Option<u32>,
) -> Result<Vec<OpPlan>, BackendError> {
    if !delta_supported(shape) {
        return Err(BackendError::UnsupportedOperation(format!(
            "DeltaNet shape {}/{} heads {}x{} kernel {} is not supported by the Vulkan shaders",
            shape.k_heads, shape.v_heads, shape.k_dim, shape.v_dim, shape.conv_kernel
        )));
    }
    if !state_offset.is_multiple_of(4) {
        return Err(invalid(
            "DeltaNet state offset must be a multiple of 4 bytes",
        ));
    }
    let rec_name = delta_rec_shader(shape.k_dim, cols).ok_or_else(|| invalid("DeltaNet key width"))?;
    let (kh, vh, hk, hv, ksz) = (
        shape.k_heads,
        shape.v_heads,
        shape.k_dim,
        shape.v_dim,
        shape.conv_kernel,
    );
    let conv_dim = shape.conv_dim();
    let state_off = state_offset / 4;
    let hist_off = (state_offset + shape.rec_bytes()) / 4;
    let qscale = (hk as f32).powf(-0.5);
    let value_blocks = div_up(shape.value_dim(), 128);
    let mut plans = Vec::with_capacity(runs.len() * 3);
    for (i, run) in runs.iter().enumerate() {
        if run.rows == 0 {
            continue;
        }
        let mixed_bytes = (run.first_row + run.rows) * conv_dim * 4;
        let conv_groups_x = 2 * kh + value_blocks;
        let conv_groups_y = div_up(run.rows, CONV_CHUNK);
        if conv_groups_x > MAX_GROUPS_PER_DIM as usize
            || conv_groups_y > MAX_GROUPS_PER_DIM as usize
        {
            return Err(BackendError::UnsupportedOperation(
                "DeltaNet run too long for one dispatch".into(),
            ));
        }
        let conv_push = Push::new()
            .u("conv_dim", conv_dim)
            .u("kh", kh)
            .u("hk", hk)
            .u("ksz", ksz)
            .u("row0", run.first_row)
            .u("n", run.rows)
            .b("reset", run.reset)
            .u("hist_off", hist_off)
            .f("qscale", qscale);
        let conv_slots = Slots::new()
            .read("qkv")
            .read("conv")
            .run_state("state", i)
            .scratch("mixed", 0, mixed_bytes, Access::ReadWrite);
        plans.push(plan(
            "delta_conv",
            conv_push,
            [conv_groups_x as u32, conv_groups_y as u32, 1],
            conv_slots,
        )?);

        if super::scan::wanted(scan, run.rows, hk, hv, vh) {
            plans.extend(super::scan::plan_scan_delta_run(shape, i, run, state_offset, mixed_bytes)?);
            continue;
        }
        let rec_push = Push::new()
            .u("kh", kh)
            .u("vh", vh)
            .u("hv", hv)
            .u("conv_dim", conv_dim)
            .b("tiled", shape.tiled_heads)
            .u("row0", run.first_row)
            .u("n", run.rows)
            .b("reset", run.reset)
            .u("state_off", state_off)
            .f("beta_scale", shape.beta_scale)
            .f("eps", shape.eps);
        let rec_slots = Slots::new()
            .scratch("mixed", 0, mixed_bytes, Access::Read)
            .read("z")
            .read("beta")
            .read("alpha")
            .read("dt_bias")
            .read("decay")
            .read("norm")
            .run_state("state", i)
            .write("out");
        if vh > MAX_GROUPS_PER_DIM as usize {
            return Err(BackendError::UnsupportedOperation(
                "too many DeltaNet value heads".into(),
            ));
        }
        if !cols {
            plans.push(plan(rec_name, rec_push, [vh as u32, 1, 1], rec_slots)?);
            continue;
        }
        let recc_slots = Slots::new()
            .scratch("mixed", 0, mixed_bytes, Access::Read)
            .read("beta")
            .read("alpha")
            .read("dt_bias")
            .read("decay")
            .run_state("state", i)
            .write("out");
        plans.push(plan(rec_name, rec_push, [vh as u32, div_up(hv, RECC_COLUMNS) as u32, 1], recc_slots)?);
        if run.rows > MAX_GROUPS_PER_DIM as usize {
            return Err(BackendError::UnsupportedOperation("DeltaNet run too long for one dispatch".into()));
        }
        let norm_push = Push::new().u("vh", vh).u("hv", hv).u("row0", run.first_row).f("eps", shape.eps);
        plans.push(plan("delta_gnorm", norm_push, [vh as u32, run.rows as u32, 1], Slots::new().read("z").read("norm").write("out"))?);
    }
    Ok(plans)
}

/// Plans of `Stream::kda_layer`: for every non-empty run the `delta_conv` dispatch of the Gated DeltaNet (the convolution, SiLU and L2 normalisation over
/// `[q | k | v]` are the same with `k_heads = v_heads = heads` and equal key and value widths; the history sits behind the `heads x head_dim x head_dim`
/// matrices exactly as there) and one `kda_rec_hk<head_dim>` dispatch (barrier in between). Slots of `delta_conv`: `qkv`, `conv`, `state`, `mixed` (scratch
/// id 0), of `kda_rec`: `mixed`, `forget`, `beta`, `gate`, `dt_bias`, `rate`, `norm`, `state`, `out`.
pub fn plan_kda_layer(shape: &crate::backend::KdaShape, runs: &[RunSpec], state_offset: usize) -> Result<Vec<OpPlan>, BackendError> {
    plan_kda_layer_scan(shape, runs, state_offset, None)
}

/// [`plan_kda_layer`]; `scan`: the shared memory of the device when long runs may take the chunk-parallel scan ([`super::scan`]).
pub fn plan_kda_layer_scan(shape: &crate::backend::KdaShape, runs: &[RunSpec], state_offset: usize, scan: Option<u32>) -> Result<Vec<OpPlan>, BackendError> {
    if !kda_supported(shape) {
        return Err(BackendError::UnsupportedOperation(format!(
            "KDA shape {} heads of {} kernel {} is not supported by the Vulkan shaders",
            shape.heads, shape.head_dim, shape.conv_kernel
        )));
    }
    if !state_offset.is_multiple_of(4) {
        return Err(invalid("KDA state offset must be a multiple of 4 bytes"));
    }
    let rec_name = match shape.head_dim {
        16 => "kda_rec_hk16",
        32 => "kda_rec_hk32",
        64 => "kda_rec_hk64",
        _ => "kda_rec_hk128",
    };
    let (heads, hd, ksz) = (shape.heads, shape.head_dim, shape.conv_kernel);
    let conv_dim = shape.conv_dim();
    let state_off = state_offset / 4;
    let hist_off = (state_offset + shape.rec_bytes()) / 4;
    let qscale = (hd as f32).powf(-0.5);
    let value_blocks = div_up(heads * hd, 128);
    let mut plans = Vec::with_capacity(runs.len() * 2);
    for (i, run) in runs.iter().enumerate() {
        if run.rows == 0 {
            continue;
        }
        let mixed_bytes = (run.first_row + run.rows) * conv_dim * 4;
        let (gx, gy) = (2 * heads + value_blocks, div_up(run.rows, CONV_CHUNK));
        if gx > MAX_GROUPS_PER_DIM as usize || gy > MAX_GROUPS_PER_DIM as usize || heads > MAX_GROUPS_PER_DIM as usize {
            return Err(BackendError::UnsupportedOperation("KDA run too long for one dispatch".into()));
        }
        let conv_push = Push::new()
            .u("conv_dim", conv_dim)
            .u("kh", heads)
            .u("hk", hd)
            .u("ksz", ksz)
            .u("row0", run.first_row)
            .u("n", run.rows)
            .b("reset", run.reset)
            .u("hist_off", hist_off)
            .f("qscale", qscale);
        let conv_slots = Slots::new().read("qkv").read("conv").run_state("state", i).scratch("mixed", 0, mixed_bytes, Access::ReadWrite);
        plans.push(plan("delta_conv", conv_push, [gx as u32, gy as u32, 1], conv_slots)?);
        if super::scan::wanted(scan, run.rows, hd, hd, heads) {
            plans.extend(super::scan::plan_scan_kda_run(shape, i, run, state_offset, mixed_bytes)?);
            continue;
        }
        let rec_push = Push::new()
            .u("heads", heads)
            .u("conv_dim", conv_dim)
            .u("row0", run.first_row)
            .u("n", run.rows)
            .b("reset", run.reset)
            .u("state_off", state_off)
            .b("bounded", shape.lower_bound.is_some())
            .f("lower_bound", shape.lower_bound.unwrap_or(0.0))
            .f("eps", shape.eps);
        let rec_slots = Slots::new()
            .scratch("mixed", 0, mixed_bytes, Access::Read)
            .read("forget")
            .read("beta")
            .read("gate")
            .read("dt_bias")
            .read("rate")
            .read("norm")
            .run_state("state", i)
            .write("out");
        plans.push(plan(rec_name, rec_push, [heads as u32, 1, 1], rec_slots)?);
    }
    Ok(plans)
}

/// Whether the `shortconv` shader takes this convolution (kernel 2..8).
pub fn shortconv_supported(hidden: usize, kernel: usize) -> bool {
    hidden > 0 && (2..=8).contains(&kernel)
}

/// `Stream::shortconv_layer` for one run (`shortconv`, slots `proj`, `conv`, `bias`, `state`, `out`); `state_offset` is the byte offset of the layer's history.
pub fn plan_shortconv(run_index: usize, run: &RunSpec, hidden: usize, kernel: usize, state_offset: usize, has_bias: bool) -> Result<OpPlan, BackendError> {
    if !shortconv_supported(hidden, kernel) || !state_offset.is_multiple_of(4) || run.rows == 0 {
        return Err(invalid("ShortConv: unsupported kernel, offset or empty run"));
    }
    let push = Push::new()
        .u("hidden", hidden)
        .u("ksz", kernel)
        .u("row0", run.first_row)
        .u("n", run.rows)
        .b("reset", run.reset)
        .u("state_off", state_offset / 4)
        .b("has_bias", has_bias);
    plan(
        "shortconv",
        push,
        [div_up(hidden, 128) as u32, 1, 1],
        Slots::new().read("proj").read("conv").read_opt("bias", has_bias).run_state("state", run_index).write("out"),
    )
}

/// `Stream::copy_columns` (`misc_copy_cols`): `dst[r * dst_stride + dst_col + c] = src[r * width + c]` for `rows` rows.
pub fn plan_copy_cols(rows: usize, width: usize, dst_stride: usize, dst_col: usize) -> Result<OpPlan, BackendError> {
    if rows == 0 || width == 0 || dst_col + width > dst_stride {
        return Err(invalid("copy_columns: empty block or block wider than the destination row"));
    }
    plan(
        "misc_copy_cols",
        Push::new().u("count", rows * width).u("width", width).u("dst_stride", dst_stride).u("dst_col", dst_col),
        groups_1d(rows * width, 256),
        Slots::new().read("src").rw("dst"),
    )
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// RWKV-6 / RWKV-7
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// Whether the RWKV shaders take a layer with this head size (`rwkv7_wkv_n*`, `rwkv6_wkv_n*`).
pub fn rwkv_supported(shape: &crate::backend::RwkvShape) -> bool {
    matches!(shape.head_size, 16 | 32 | 64 | 128) && shape.hidden > 0 && shape.hidden.is_multiple_of(shape.head_size)
}

/// Storage buffers the widest RWKV shader binds (`rwkv7_wkv`).
pub const RWKV_BINDINGS: u32 = 15;

/// `Stream::rwkv_shift` for one run (`rwkv_shift`, slots `x`, `shift`, `state`); `shift_offset` is the byte offset of the stored vector inside the run's state buffer.
pub fn plan_rwkv_shift(run_index: usize, run: &RunSpec, hidden: usize, shift_offset: usize) -> Result<OpPlan, BackendError> {
    if !shift_offset.is_multiple_of(4) || run.rows == 0 || hidden == 0 {
        return Err(invalid("RWKV shift: bad offset or empty run"));
    }
    let push = Push::new().u("hidden", hidden).u("row0", run.first_row).u("n", run.rows).b("reset", run.reset).u("off", shift_offset / 4);
    plan("rwkv_shift", push, groups_1d(run.rows * hidden, 256), Slots::new().read("x").write("shift").run_state("state", run_index))
}

/// `Stream::rwkv_lerp` (`rwkv_lerp`): `out = x + (shift - x) * (mix + delta)` over `count` floats.
pub fn plan_rwkv_lerp(count: usize, hidden: usize, has_delta: bool) -> Result<OpPlan, BackendError> {
    if count == 0 || hidden == 0 {
        return Err(invalid("RWKV interpolation of an empty tensor"));
    }
    plan(
        "rwkv_lerp",
        Push::new().u("count", count).u("hidden", hidden).b("has_delta", has_delta),
        groups_1d(count, 256),
        Slots::new().read("x").read("shift").read("mix").read_opt("delta", has_delta).write("out"),
    )
}

/// `Stream::rwkv_act` (`rwkv_act`): in-place activation of `count` floats, `mode` as documented there.
pub fn plan_rwkv_act(count: usize, mode: i32, has_gate: bool) -> Result<OpPlan, BackendError> {
    if count == 0 || !(0..=5).contains(&mode) || (mode == 4 && !has_gate) {
        return Err(invalid("RWKV activation: empty tensor, unknown mode or missing gate"));
    }
    plan("rwkv_act", Push::new().u("count", count).u("mode", mode as usize), groups_1d(count, 256), Slots::new().rw("x").read_opt("gate", has_gate))
}

fn rwkv_wkv_shader(v7: bool, head_size: usize) -> Option<&'static str> {
    Some(match (v7, head_size) {
        (true, 16) => "rwkv7_wkv_n16",
        (true, 32) => "rwkv7_wkv_n32",
        (true, 64) => "rwkv7_wkv_n64",
        (true, 128) => "rwkv7_wkv_n128",
        (false, 16) => "rwkv6_wkv_n16",
        (false, 32) => "rwkv6_wkv_n32",
        (false, 64) => "rwkv6_wkv_n64",
        (false, 128) => "rwkv6_wkv_n128",
        _ => return None,
    })
}

/// `Stream::rwkv7_wkv` for one run (`rwkv7_wkv_n*`): one workgroup per head. `state_offset` is the byte offset of the WKV matrices inside the run's state buffer.
pub fn plan_rwkv7_wkv(run_index: usize, run: &RunSpec, shape: &crate::backend::RwkvShape, state_offset: usize, has_vres: bool) -> Result<OpPlan, BackendError> {
    let name = rwkv_wkv_shader(true, shape.head_size).filter(|_| rwkv_supported(shape) && state_offset.is_multiple_of(4) && run.rows > 0);
    let name = name.ok_or_else(|| invalid("RWKV-7: unsupported head size, offset or empty run"))?;
    let push = Push::new()
        .u("hidden", shape.hidden)
        .u("row0", run.first_row)
        .u("n", run.rows)
        .b("reset", run.reset)
        .u("state_off", state_offset / 4)
        .b("has_vres", has_vres)
        .f("eps", shape.group_eps);
    let slots = Slots::new()
        .read("r")
        .read("k")
        .read("v")
        .read("w")
        .read("a")
        .read_opt("vres", has_vres)
        .read("gate")
        .read("norm_w")
        .read("norm_b")
        .read("k_k")
        .read("k_a")
        .read("r_k")
        .rw("v_first")
        .run_state("state", run_index)
        .write("out");
    plan(name, push, [shape.heads() as u32, 1, 1], slots)
}

/// `Stream::rwkv6_wkv` for one run (`rwkv6_wkv_n*`).
pub fn plan_rwkv6_wkv(run_index: usize, run: &RunSpec, shape: &crate::backend::RwkvShape, state_offset: usize) -> Result<OpPlan, BackendError> {
    let name = rwkv_wkv_shader(false, shape.head_size).filter(|_| rwkv_supported(shape) && state_offset.is_multiple_of(4) && run.rows > 0);
    let name = name.ok_or_else(|| invalid("RWKV-6: unsupported head size, offset or empty run"))?;
    let push = Push::new()
        .u("hidden", shape.hidden)
        .u("row0", run.first_row)
        .u("n", run.rows)
        .b("reset", run.reset)
        .u("state_off", state_offset / 4)
        .f("eps", shape.group_eps);
    let slots = Slots::new()
        .read("r")
        .read("k")
        .read("v")
        .read("w")
        .read("gate")
        .read("norm_w")
        .read("norm_b")
        .read("first")
        .run_state("state", run_index)
        .write("out");
    plan(name, push, [shape.heads() as u32, 1, 1], slots)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Mamba
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// Shape of a Mamba-2 mixer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mamba2Shape {
    pub inner: usize,
    pub state: usize,
    pub heads: usize,
    pub groups: usize,
    pub conv_kernel: usize,
    pub norm_groups: usize,
    pub step_min: f32,
    /// Epsilon of the group RMS norm (the model's `rms_norm_eps`).
    pub eps: f32,
    /// 0 `norm(y * silu(z))`, 1 `y * silu(z)`, 2 `norm(y) * silu(z)` (see `MambaShape::norm_mode`).
    pub norm_mode: u8,
}

impl Mamba2Shape {
    /// Channels of the convolution: `[x (inner) | B (groups * state) | C (groups * state)]`.
    pub fn conv_dim(&self) -> usize {
        self.inner + 2 * self.groups * self.state
    }

    pub fn head_dim(&self) -> usize {
        self.inner / self.heads.max(1)
    }

    /// Floats of one row of the input projection: `[gate (inner) | xBC (conv_dim) | dt (heads)]`.
    pub fn proj_width(&self) -> usize {
        self.inner + self.conv_dim() + self.heads
    }

    /// Floats of the convolution history and of the recurrent state of one layer of one sequence.
    pub fn state_floats(&self) -> (usize, usize) {
        (
            self.conv_dim() * (self.conv_kernel - 1),
            self.heads * self.head_dim() * self.state,
        )
    }

    /// Bytes of one layer in a state buffer (history first, then the recurrent state), rounded up to 256.
    pub fn layer_state_bytes(&self) -> usize {
        let (h, r) = self.state_floats();
        ((h + r) * 4).next_multiple_of(256)
    }

    fn check(&self) -> Result<(), BackendError> {
        if self.inner == 0
            || self.heads == 0
            || self.groups == 0
            || self.state == 0
            || !self.inner.is_multiple_of(self.heads)
            || !self.heads.is_multiple_of(self.groups)
        {
            return Err(invalid("inconsistent Mamba-2 shape"));
        }
        if !(2..=8).contains(&self.conv_kernel) || self.state > 256 {
            return Err(BackendError::UnsupportedOperation(
                "Mamba-2 conv kernel must be 2..8 and the state size <= 256".into(),
            ));
        }
        if self.norm_groups == 0 || !self.inner.is_multiple_of(self.norm_groups) {
            return Err(invalid("Mamba-2 norm groups do not divide the inner width"));
        }
        Ok(())
    }
}

/// Shape of a Mamba-1 mixer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mamba1Shape {
    pub inner: usize,
    pub state: usize,
    /// Rank of dt (`x_proj` outputs `rank + 2 * state` values per token).
    pub rank: usize,
    pub conv_kernel: usize,
    /// Some(eps): dt, B and C are RMS-normalised (without weight) after `x_proj`.
    pub bc_rms_eps: Option<f32>,
}

impl Mamba1Shape {
    /// Floats of the convolution history and of the recurrent state of one layer of one sequence.
    pub fn state_floats(&self) -> (usize, usize) {
        (self.inner * (self.conv_kernel - 1), self.inner * self.state)
    }

    pub fn layer_state_bytes(&self) -> usize {
        let (h, r) = self.state_floats();
        ((h + r) * 4).next_multiple_of(256)
    }

    /// Floats of one row of the `x_proj` output.
    pub fn split_width(&self) -> usize {
        self.rank + 2 * self.state
    }

    fn check(&self) -> Result<(), BackendError> {
        if self.inner == 0 || self.state == 0 || self.rank == 0 {
            return Err(invalid("empty Mamba-1 shape"));
        }
        if !(2..=8).contains(&self.conv_kernel) || self.state > 256 {
            return Err(BackendError::UnsupportedOperation(
                "Mamba-1 conv kernel must be 2..8 and the state size <= 256".into(),
            ));
        }
        Ok(())
    }
}

/// `mamba_conv` for one run: depthwise causal convolution + SiLU of `channels` channels of rows `row0 ..` of `x` (row stride `in_stride`, first channel at
/// `in_off`), history at float `hist_off` of the run's state buffer. Slots: `x`, `convw` (`channels x ksz`), `convb` (dummy without bias), `state`, `mixed`.
#[allow(clippy::too_many_arguments)]
pub fn plan_mamba_conv(
    run_index: usize,
    run: &RunSpec,
    channels: usize,
    ksz: usize,
    in_stride: usize,
    in_off: usize,
    hist_off: usize,
    has_bias: bool,
    mixed_scratch: Option<(u8, usize)>,
) -> Result<OpPlan, BackendError> {
    if run.rows == 0 || channels == 0 {
        return Err(invalid("mamba conv of an empty run"));
    }
    if !(2..=8).contains(&ksz) {
        return Err(BackendError::UnsupportedOperation(format!(
            "Mamba conv kernel {ksz} (2..8)"
        )));
    }
    let gy = div_up(run.rows, CONV_CHUNK);
    let gx = div_up(channels, 64);
    if gx > MAX_GROUPS_PER_DIM as usize || gy > MAX_GROUPS_PER_DIM as usize {
        return Err(BackendError::UnsupportedOperation(
            "Mamba run too long for one dispatch".into(),
        ));
    }
    let push = Push::new()
        .u("channels", channels)
        .u("ksz", ksz)
        .u("in_stride", in_stride)
        .u("in_off", in_off)
        .u("row0", run.first_row)
        .u("n", run.rows)
        .b("reset", run.reset)
        .u("hist_off", hist_off)
        .b("has_bias", has_bias);
    let slots = Slots::new()
        .read("x")
        .read("convw")
        .read_opt("convb", has_bias)
        .run_state("state", run_index);
    let slots = match mixed_scratch {
        Some((id, bytes)) => slots.scratch("mixed", id, bytes, Access::ReadWrite),
        None => slots.rw("mixed"),
    };
    plan("mamba_conv", push, [gx as u32, gy as u32, 1], slots)
}

fn scan2_shader(ns: usize) -> &'static str {
    match ns {
        16 => "mamba_scan2_ns16",
        32 => "mamba_scan2_ns32",
        64 => "mamba_scan2_ns64",
        128 => "mamba_scan2_ns128",
        _ => "mamba_scan2_gen",
    }
}

fn scan1_shader(ns: usize) -> &'static str {
    match ns {
        8 => "mamba_scan1_ns8",
        16 => "mamba_scan1_ns16",
        32 => "mamba_scan1_ns32",
        _ => "mamba_scan1_gen",
    }
}

/// `mamba_scan2_*` for one run: selective state update of a Mamba-2 layer. `mixed` = output of `mamba_conv` (`conv_dim` floats per row), `projected` = the
/// input projection (`proj_stride` floats per row, dt at `proj_dt_off`), `state_off` = float offset of the recurrent state in the run's state buffer.
/// Slots: `mixed`, `projected`, `dt_bias`, `decay`, `skip`, `state`, `out` (`rows x inner`, before the gated norm).
pub fn plan_mamba2_scan(
    run_index: usize,
    run: &RunSpec,
    shape: &Mamba2Shape,
    proj_stride: usize,
    proj_dt_off: usize,
    state_off: usize,
    mixed_scratch: Option<(u8, usize)>,
) -> Result<OpPlan, BackendError> {
    shape.check()?;
    if run.rows == 0 {
        return Err(invalid("mamba scan of an empty run"));
    }
    let head_dim = shape.head_dim();
    let push = Push::new()
        .u("inner", shape.inner)
        .u("ns", shape.state)
        .u("heads", shape.heads)
        .u("groups", shape.groups)
        .u("head_dim", head_dim)
        .u("conv_dim", shape.conv_dim())
        .u("proj_stride", proj_stride)
        .u("proj_dt_off", proj_dt_off)
        .u("row0", run.first_row)
        .u("n", run.rows)
        .b("reset", run.reset)
        .u("state_off", state_off)
        .f("step_min", shape.step_min);
    let slots = match mixed_scratch {
        Some((id, bytes)) => Slots::new().scratch("mixed", id, bytes, Access::Read),
        None => Slots::new().read("mixed"),
    };
    let slots = slots
        .read("projected")
        .read("dt_bias")
        .read("decay")
        .read("skip")
        .run_state("state", run_index)
        .write("out");
    plan(
        scan2_shader(shape.state),
        push,
        [shape.heads as u32, div_up(head_dim, 64) as u32, 1],
        slots,
    )
}

/// `mamba_scan1_*` for one run: selective scan of a Mamba-1 layer (see the shader for the operands). Slots: `mixed`, `dt`, `split`, `projected`, `decay`,
/// `skip`, `state`, `out` (`rows x inner`, final: already gated).
#[allow(clippy::too_many_arguments)]
pub fn plan_mamba1_scan(
    run_index: usize,
    run: &RunSpec,
    shape: &Mamba1Shape,
    proj_stride: usize,
    proj_gate_off: usize,
    state_off: usize,
) -> Result<OpPlan, BackendError> {
    shape.check()?;
    if run.rows == 0 {
        return Err(invalid("mamba scan of an empty run"));
    }
    let push = Push::new()
        .u("inner", shape.inner)
        .u("ns", shape.state)
        .u("rank", shape.rank)
        .u("split_stride", shape.split_width())
        .u("proj_stride", proj_stride)
        .u("proj_gate_off", proj_gate_off)
        .u("row0", run.first_row)
        .u("n", run.rows)
        .b("reset", run.reset)
        .u("state_off", state_off);
    let slots = Slots::new()
        .read("mixed")
        .read("dt")
        .read("split")
        .read("projected")
        .read("decay")
        .read("skip")
        .run_state("state", run_index)
        .write("out");
    plan(
        scan1_shader(shape.state),
        push,
        [div_up(shape.inner, 64) as u32, 1, 1],
        slots,
    )
}

/// `mamba_gate_norm` for all `rows` of the batch: in place on `y`. Slots: `y`, `projected`, `norm`.
pub fn plan_mamba_gate_norm(
    shape: &Mamba2Shape,
    rows: usize,
    proj_stride: usize,
    proj_gate_off: usize,
) -> Result<OpPlan, BackendError> {
    shape.check()?;
    if rows == 0 {
        return Err(invalid("mamba gate norm of an empty batch"));
    }
    let push = Push::new()
        .u("inner", shape.inner)
        .u("norm_groups", shape.norm_groups)
        .u("rows", rows)
        .u("proj_stride", proj_stride)
        .u("proj_gate_off", proj_gate_off)
        .u("mode", usize::from(shape.norm_mode))
        .f("eps", shape.eps);
    plan(
        "mamba_gate_norm",
        push,
        groups_2d(shape.norm_groups, rows)?,
        Slots::new().rw("y").read("projected").read("norm"),
    )
}

/// Grid `[x, y, 1]` with both checked against the dispatch limit.
fn groups_2d(x: usize, y: usize) -> Result<[u32; 3], BackendError> {
    if x > MAX_GROUPS_PER_DIM as usize || y > MAX_GROUPS_PER_DIM as usize {
        return Err(BackendError::UnsupportedOperation(
            "dispatch grid larger than 65535".into(),
        ));
    }
    Ok([x as u32, y as u32, 1])
}

/// The complete Mamba-2 mixer after the input projection, for the runs of a batch: per run `mamba_conv` + `mamba_scan2_*`, then `mamba_gate_norm` for
/// all rows. The caller applies the input projection before (`projected`: `rows x proj_width`) and the output projection after (on `out`, `rows x inner`).
/// The conv output `mixed` is scratch id 1. The state of a layer in a run's state buffer starts at byte `state_offset` (history, then recurrent state).
pub fn plan_mamba2_mixer(
    shape: &Mamba2Shape,
    runs: &[RunSpec],
    rows: usize,
    state_offset: usize,
) -> Result<Vec<OpPlan>, BackendError> {
    plan_mamba2_mixer_scan(shape, runs, rows, state_offset, None)
}

/// [`plan_mamba2_mixer`]; `scan`: the shared memory of the device when long runs may take the chunk-parallel scan ([`super::scan`]).
pub fn plan_mamba2_mixer_scan(
    shape: &Mamba2Shape,
    runs: &[RunSpec],
    rows: usize,
    state_offset: usize,
    scan: Option<u32>,
) -> Result<Vec<OpPlan>, BackendError> {
    shape.check()?;
    if !state_offset.is_multiple_of(4) {
        return Err(invalid("Mamba state offset must be a multiple of 4 bytes"));
    }
    let (hist, _rec) = shape.state_floats();
    let hist_off = state_offset / 4;
    let rec_off = hist_off + hist;
    let width = shape.proj_width();
    let scratch = Some((1u8, rows * shape.conv_dim() * 4));
    let mut plans = Vec::new();
    for (i, run) in runs.iter().enumerate() {
        if run.rows == 0 {
            continue;
        }
        plans.push(plan_mamba_conv(
            i,
            run,
            shape.conv_dim(),
            shape.conv_kernel,
            width,
            shape.inner,
            hist_off,
            true,
            scratch,
        )?);
        if super::scan::wanted(scan, run.rows, shape.state, shape.head_dim(), shape.heads) {
            let mixed = (1u8, rows * shape.conv_dim() * 4);
            plans.extend(super::scan::plan_scan_mamba2_run(shape, i, run, width, shape.inner + shape.conv_dim(), rec_off, mixed)?);
            continue;
        }
        plans.push(plan_mamba2_scan(
            i,
            run,
            shape,
            width,
            shape.inner + shape.conv_dim(),
            rec_off,
            scratch,
        )?);
    }
    plans.push(plan_mamba_gate_norm(shape, rows, width, 0)?);
    Ok(plans)
}

/// The RMS norms (no weight) of the three parts of an `x_proj` row of a Mamba-1 layer (dt, B, C), in place on `split` (`rows x (rank + 2 state)`).
/// Slots as `plan_rms_norm`: `x`, `weight` (dummy), `out` (the same buffer as `x`).
pub fn plan_mamba1_bc_norms(shape: &Mamba1Shape, rows: usize) -> Result<Vec<OpPlan>, BackendError> {
    plan_mamba1_bc_norms_with(shape, rows, false)
}

/// [`plan_mamba1_bc_norms`]; `weighted`: every norm has a learned weight (Jamba), bound as `weight` of its plan.
pub fn plan_mamba1_bc_norms_with(shape: &Mamba1Shape, rows: usize, weighted: bool) -> Result<Vec<OpPlan>, BackendError> {
    shape.check()?;
    let Some(eps) = shape.bc_rms_eps else {
        return Ok(Vec::new());
    };
    let stride = shape.split_width();
    [
        (0, shape.rank),
        (shape.rank, shape.state),
        (shape.rank + shape.state, shape.state),
    ]
    .into_iter()
    .map(|(off, cols)| {
        plan_rms_norm_strided(
            rows,
            cols,
            eps,
            weighted,
            RowLayout {
                x_stride: stride,
                x_off: off,
                o_stride: stride,
                o_off: off,
            },
        )
    })
    .collect()
}
