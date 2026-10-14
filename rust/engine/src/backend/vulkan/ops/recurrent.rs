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
pub fn delta_supported(shape: &DeltaShape) -> bool {
    matches!(shape.k_dim, 32 | 64 | 96 | 128)
        && (1..=128).contains(&shape.v_dim)
        && shape.k_heads > 0
        && shape.v_heads > 0
        && shape.v_heads.is_multiple_of(shape.k_heads)
        && (2..=8).contains(&shape.conv_kernel)
}

fn delta_rec_shader(hk: usize) -> Option<&'static str> {
    Some(match hk {
        32 => "delta_rec_hk32",
        64 => "delta_rec_hk64",
        96 => "delta_rec_hk96",
        128 => "delta_rec_hk128",
        _ => return None,
    })
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
    let rec_name = delta_rec_shader(shape.k_dim).ok_or_else(|| invalid("DeltaNet key width"))?;
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
    let mut plans = Vec::with_capacity(runs.len() * 2);
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
        plans.push(plan(rec_name, rec_push, [vh as u32, 1, 1], rec_slots)?);
    }
    Ok(plans)
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
            false,
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
