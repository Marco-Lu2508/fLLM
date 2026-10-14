//! Dispatch planning of gated attention (`split_q_gate`, `sigmoid_gate_mul`), of the Gated DeltaNet mixer (`Stream::delta_layer`) and of the Mamba-1 / Mamba-2
//! mixers (no `Stream` method yet: the pieces are planned here so that the stream can offer one). Owner: package 6c (kernels `gate_split_q`,
//! `gate_sigmoid_mul`, `delta_conv`, `delta_rec_hk<N>`, `mamba_conv`, `mamba_scan1_*`, `mamba_scan2_*`, `mamba_gate_norm`, `mamba_bc_norm`).
//!
//! The conventions are those of [`super::moe_plan`] (constants struct at buffer 0, slots from buffer 1 on, accesses declared for the hazard tracking); the
//! kernel side is documented in the `// FLLM-KERNEL:` headers and in `rust/engine/metal/INTERFACE_RECURRENT.md`. Level of verification: **L0**.

use super::moe_plan::{dim, div_up, groups_1d, invalid, plan, Consts, Slots, ELEMENTWISE_THREADS};
use super::{Access, OpPlan};
use crate::backend::{BackendError, DeltaShape};

/// Tokens per threadgroup row of the convolution kernels (must be >= conv_kernel - 1).
pub const CONV_CHUNK: usize = 16;

// ---------------------------------------------------------------------------------------------------------------------------------------------
// gated attention
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// `Stream::supports_attention_gate`: the two gate kernels have no restriction of their own.
pub fn attention_gate_supported() -> bool {
    true
}

/// `Stream::split_q_gate` (`gate_split_q`): the `rows x heads x 2*dk` projection into queries and gates. Slots: `projected`, `q`, `gate`.
pub fn plan_split_q_gate(
    rows: usize,
    heads: usize,
    dk: usize,
    per_head: bool,
) -> Result<OpPlan, BackendError> {
    if rows == 0 || heads == 0 || dk == 0 {
        return Err(invalid("split_q_gate of an empty tensor"));
    }
    let total = rows
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(dk))
        .ok_or_else(|| invalid("split_q_gate size overflows"))?;
    let consts = Consts::new()
        .u("total", total)
        .u("heads", heads)
        .u("dk", dk)
        .b("per_head", per_head);
    plan(
        "gate_split_q",
        consts,
        groups_1d(total, ELEMENTWISE_THREADS as usize),
        [ELEMENTWISE_THREADS, 1, 1],
        Slots::new().read("projected").write("q").write("gate"),
    )
}

/// `Stream::sigmoid_gate_mul` (`gate_sigmoid_mul`): `x[i] *= sigmoid(gate[i])` in place over `count` floats. Slots: `x` (read-write), `gate`.
pub fn plan_sigmoid_gate_mul(count: usize) -> Result<OpPlan, BackendError> {
    if count == 0 {
        return Err(invalid("sigmoid_gate_mul of an empty tensor"));
    }
    let consts = Consts::new().u("count", count);
    plan(
        "gate_sigmoid_mul",
        consts,
        groups_1d(count, ELEMENTWISE_THREADS as usize),
        [ELEMENTWISE_THREADS, 1, 1],
        Slots::new().rw("x").read("gate"),
    )
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Gated DeltaNet
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// Consecutive rows of one sequence (mirror of [`crate::backend::DeltaRun`] without the buffer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunSpec {
    pub first_row: usize,
    pub rows: usize,
    /// The sequence starts with this run: the stored state is ignored.
    pub reset: bool,
}

/// Basis of `Stream::supports_delta` for the Metal stream: the DeltaNet kernels take a layer when
///  * the key width `k_dim` is 32, 64, 96 or 128 (one `delta_rec_hk<N>` kernel each; the key half of a value column lives in registers),
///  * the value width `v_dim` is 1..=128 (two threads per value column, 256 threads per threadgroup),
///  * the head counts are positive and `v_heads` is a multiple of `k_heads` (the key head of value head `j` is `j % k_heads` or `j / (v_heads / k_heads)`),
///  * the convolution kernel is 2..=8 (`conv_kernel - 1` history values per channel are kept in registers),
///  * `conv_dim` and the head counts fit the 32-bit indices and the grid.
///
/// The stream additionally needs the 9 weight/activation buffers of [`plan_delta_layer`] to be f32 device buffers and the state buffer of every run to
/// hold `state_offset + layer_state_bytes()` bytes (not checkable here).
pub fn delta_supported(shape: &DeltaShape) -> bool {
    matches!(shape.k_dim, 32 | 64 | 96 | 128)
        && (1..=128).contains(&shape.v_dim)
        && shape.k_heads > 0
        && shape.v_heads > 0
        && shape.v_heads.is_multiple_of(shape.k_heads)
        && (2..=8).contains(&shape.conv_kernel)
        && shape.conv_dim() <= u32::MAX as usize / 4
        && shape.v_heads <= u16::MAX as usize
}

fn delta_rec_kernel(hk: usize) -> Option<&'static str> {
    Some(match hk {
        32 => "delta_rec_hk32",
        64 => "delta_rec_hk64",
        96 => "delta_rec_hk96",
        128 => "delta_rec_hk128",
        _ => return None,
    })
}

/// Plans of `Stream::delta_layer`: for every non-empty run a `delta_conv` and a `delta_rec_hk*` dispatch (in this order; the second reads what the first
/// wrote, which the stream's hazard tracking sees through the declared accesses; the runs are independent of each other as every run has its own state
/// buffer, slot `state` = `SlotSource::RunState`). `state_offset` is the byte offset of the layer's state inside each run's state buffer (the recurrent
/// matrices `v_heads x k_dim x v_dim` f32, then the convolution history `conv_dim x (kernel - 1)` f32 at `rec_bytes()`); it must be a multiple of 4.
/// Slots of `delta_conv`: `qkv`, `conv`, `state`, `mixed` (the caller's `mixed` scratch of `rows x conv_dim` floats); of `delta_rec`: `mixed`, `z`, `beta`,
/// `alpha`, `dt_bias`, `decay`, `norm`, `state`, `out`.
pub fn plan_delta_layer(
    shape: &DeltaShape,
    runs: &[RunSpec],
    state_offset: usize,
) -> Result<Vec<OpPlan>, BackendError> {
    if !delta_supported(shape) {
        return Err(BackendError::UnsupportedOperation(format!(
            "DeltaNet shape {}/{} heads of {}x{} with kernel {} is not supported by the Metal kernels",
            shape.k_heads, shape.v_heads, shape.k_dim, shape.v_dim, shape.conv_kernel
        )));
    }
    if !state_offset.is_multiple_of(4) {
        return Err(invalid(
            "DeltaNet state offset must be a multiple of 4 bytes",
        ));
    }
    let rec_name = delta_rec_kernel(shape.k_dim).ok_or_else(|| invalid("DeltaNet key width"))?;
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
        let conv_consts = Consts::new()
            .u("conv_dim", conv_dim)
            .u("kh", kh)
            .u("hk", hk)
            .u("ksz", ksz)
            .u("row0", run.first_row)
            .u("n", run.rows)
            .b("reset", run.reset)
            .u("hist_off", hist_off)
            .f("qscale", qscale);
        plans.push(plan(
            "delta_conv",
            conv_consts,
            [
                dim(2 * kh + value_blocks, "DeltaNet conv groups")?,
                dim(div_up(run.rows, CONV_CHUNK), "DeltaNet run length")?,
                1,
            ],
            [128, 1, 1],
            Slots::new()
                .read("qkv")
                .read("conv")
                .run_state("state", i)
                .write("mixed"),
        )?);

        let rec_consts = Consts::new()
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
        plans.push(plan(
            rec_name,
            rec_consts,
            [dim(vh, "DeltaNet value heads")?, 1, 1],
            [256, 1, 1],
            Slots::new()
                .read("mixed")
                .read("z")
                .read("beta")
                .read("alpha")
                .read("dt_bias")
                .read("decay")
                .read("norm")
                .run_state("state", i)
                .write("out"),
        )?);
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

/// Where the convolution output `mixed` lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixedBuffer {
    /// A buffer of the caller (slot `mixed`).
    Caller,
    /// Scratch of the operation: slots with the same `id` share one buffer of `bytes` bytes.
    Scratch { id: u8, bytes: usize },
}

fn mixed_slot(slots: Slots, mixed: MixedBuffer, access: Access) -> Slots {
    match (mixed, access) {
        (MixedBuffer::Caller, Access::Read) => slots.read("mixed"),
        (MixedBuffer::Caller, Access::Write) => slots.write("mixed"),
        (MixedBuffer::Caller, Access::ReadWrite) => slots.rw("mixed"),
        (MixedBuffer::Scratch { id, bytes }, access) => slots.scratch("mixed", id, bytes, access),
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
    mixed: MixedBuffer,
) -> Result<OpPlan, BackendError> {
    if run.rows == 0 || channels == 0 {
        return Err(invalid("mamba conv of an empty run"));
    }
    if !(2..=8).contains(&ksz) {
        return Err(BackendError::UnsupportedOperation(format!(
            "Mamba conv kernel {ksz} (2..8)"
        )));
    }
    let consts = Consts::new()
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
    plan(
        "mamba_conv",
        consts,
        [
            dim(div_up(channels, 64), "Mamba channels")?,
            dim(div_up(run.rows, CONV_CHUNK), "Mamba run length")?,
            1,
        ],
        [64, 1, 1],
        mixed_slot(slots, mixed, Access::Write),
    )
}

fn scan2_kernel(ns: usize) -> &'static str {
    match ns {
        16 => "mamba_scan2_ns16",
        32 => "mamba_scan2_ns32",
        64 => "mamba_scan2_ns64",
        128 => "mamba_scan2_ns128",
        _ => "mamba_scan2_gen",
    }
}

fn scan1_kernel(ns: usize) -> &'static str {
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
    mixed: MixedBuffer,
) -> Result<OpPlan, BackendError> {
    shape.check()?;
    if run.rows == 0 {
        return Err(invalid("mamba scan of an empty run"));
    }
    let head_dim = shape.head_dim();
    let consts = Consts::new()
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
    let slots = mixed_slot(Slots::new(), mixed, Access::Read)
        .read("projected")
        .read("dt_bias")
        .read("decay")
        .read("skip")
        .run_state("state", run_index)
        .write("out");
    plan(
        scan2_kernel(shape.state),
        consts,
        [
            dim(shape.heads, "Mamba heads")?,
            dim(div_up(head_dim, 64), "Mamba head width")?,
            1,
        ],
        [64, 1, 1],
        slots,
    )
}

/// `mamba_scan1_*` for one run: selective scan of a Mamba-1 layer (see the kernel for the operands). Slots: `mixed`, `dt`, `split`, `projected`, `decay`,
/// `skip`, `state`, `out` (`rows x inner`, final: already gated).
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
    let consts = Consts::new()
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
        scan1_kernel(shape.state),
        consts,
        [dim(div_up(shape.inner, 64), "Mamba channels")?, 1, 1],
        [64, 1, 1],
        slots,
    )
}

/// `mamba_gate_norm` for all `rows` of the batch: in place on `y`. Slots: `y` (read-write), `projected`, `norm`.
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
    let consts = Consts::new()
        .u("inner", shape.inner)
        .u("norm_groups", shape.norm_groups)
        .u("rows", rows)
        .u("proj_stride", proj_stride)
        .u("proj_gate_off", proj_gate_off)
        .f("eps", shape.eps);
    plan(
        "mamba_gate_norm",
        consts,
        [
            dim(shape.norm_groups, "Mamba norm groups")?,
            dim(rows, "Mamba rows")?,
            1,
        ],
        [256, 1, 1],
        Slots::new().rw("y").read("projected").read("norm"),
    )
}

/// `mamba_bc_norm` (Mamba-1 `bc_rms_eps`): the three parts of every row of the `x_proj` output normalised in place; none when the model has no such norm.
/// Slot: `split` (read-write, `rows x (rank + 2 state)` floats).
pub fn plan_mamba1_bc_norm(
    shape: &Mamba1Shape,
    rows: usize,
) -> Result<Option<OpPlan>, BackendError> {
    shape.check()?;
    let Some(eps) = shape.bc_rms_eps else {
        return Ok(None);
    };
    if rows == 0 {
        return Err(invalid("mamba bc norm of an empty batch"));
    }
    let consts = Consts::new()
        .u("rows", rows)
        .u("rank", shape.rank)
        .u("state", shape.state)
        .u("split_stride", shape.split_width())
        .f("eps", eps);
    plan(
        "mamba_bc_norm",
        consts,
        [3, dim(rows, "Mamba rows")?, 1],
        [128, 1, 1],
        Slots::new().rw("split"),
    )
    .map(Some)
}

/// The complete Mamba-2 mixer after the input projection, for the runs of a batch: per run `mamba_conv` + `mamba_scan2_*`, then `mamba_gate_norm` for all
/// rows. The caller applies the input projection before (`projected`: `rows x proj_width`) and the output projection after (on `out`, `rows x inner`).
/// The conv output `mixed` is scratch id 1 of `rows x conv_dim` floats. The state of a layer in a run's state buffer starts at byte `state_offset`
/// (history, then recurrent state).
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
    let mixed = MixedBuffer::Scratch {
        id: 1,
        bytes: rows * shape.conv_dim() * 4,
    };
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
            mixed,
        )?);
        plans.push(plan_mamba2_scan(
            i,
            run,
            shape,
            width,
            shape.inner + shape.conv_dim(),
            rec_off,
            mixed,
        )?);
    }
    plans.push(plan_mamba_gate_norm(shape, rows, width, 0)?);
    Ok(plans)
}

#[cfg(test)]
mod tests {
    use super::super::moe_plan::testkit::check_plan;
    use super::super::SlotSource;
    use super::*;

    fn word(p: &OpPlan, i: usize) -> u32 {
        u32::from_le_bytes(p.constants[4 * i..4 * i + 4].try_into().unwrap())
    }

    fn qwen_shape() -> DeltaShape {
        DeltaShape {
            k_heads: 16,
            v_heads: 32,
            k_dim: 128,
            v_dim: 128,
            conv_kernel: 4,
            tiled_heads: false,
            beta_scale: 1.0,
            eps: 1e-6,
        }
    }

    #[test]
    fn gate_plans_match_their_headers() {
        let p = plan_split_q_gate(3, 16, 256, true).unwrap();
        check_plan(&p, &[]);
        assert_eq!(
            (word(&p, 0), word(&p, 1), word(&p, 2), word(&p, 3)),
            (3 * 16 * 256, 16, 256, 1)
        );
        assert_eq!(p.threadgroups, [48, 1, 1]);
        let p = plan_split_q_gate(1, 2, 8, false).unwrap();
        assert_eq!((word(&p, 3), p.threadgroups), (0, [1, 1, 1]));
        let p = plan_sigmoid_gate_mul(100_000).unwrap();
        check_plan(&p, &[]);
        assert_eq!(p.slots[0].access, Access::ReadWrite);
        assert_eq!(p.threadgroups, [391, 1, 1]);
        assert!(plan_split_q_gate(0, 1, 1, false).is_err());
        assert!(plan_sigmoid_gate_mul(0).is_err());
        assert!(attention_gate_supported());
    }

    #[test]
    fn delta_support_rules() {
        assert!(delta_supported(&qwen_shape()));
        for hk in [32, 64, 96, 128] {
            assert!(
                delta_supported(&DeltaShape {
                    k_dim: hk,
                    ..qwen_shape()
                }),
                "hk {hk}"
            );
        }
        assert!(!delta_supported(&DeltaShape {
            k_dim: 80,
            ..qwen_shape()
        }));
        assert!(!delta_supported(&DeltaShape {
            v_dim: 129,
            ..qwen_shape()
        }));
        assert!(!delta_supported(&DeltaShape {
            v_dim: 0,
            ..qwen_shape()
        }));
        assert!(!delta_supported(&DeltaShape {
            conv_kernel: 1,
            ..qwen_shape()
        }));
        assert!(!delta_supported(&DeltaShape {
            conv_kernel: 9,
            ..qwen_shape()
        }));
        assert!(!delta_supported(&DeltaShape {
            v_heads: 24,
            ..qwen_shape()
        }));
        assert!(!delta_supported(&DeltaShape {
            k_heads: 0,
            ..qwen_shape()
        }));
        assert!(delta_supported(&DeltaShape {
            tiled_heads: true,
            ..qwen_shape()
        }));
    }

    #[test]
    fn delta_plans_follow_the_headers_and_the_state_layout() {
        let shape = qwen_shape();
        let runs = [
            RunSpec {
                first_row: 0,
                rows: 40,
                reset: true,
            },
            RunSpec {
                first_row: 40,
                rows: 0,
                reset: false,
            },
            RunSpec {
                first_row: 40,
                rows: 1,
                reset: false,
            },
        ];
        let state_offset = 3 * shape.layer_state_bytes();
        let plans = plan_delta_layer(&shape, &runs, state_offset).unwrap();
        assert_eq!(plans.len(), 4, "the empty run is skipped");
        let conv_dim = shape.conv_dim();
        assert_eq!(conv_dim, 2 * 16 * 128 + 32 * 128);
        let (conv, rec) = (&plans[0], &plans[1]);
        assert_eq!(
            (&*conv.kernel_name, &*rec.kernel_name),
            ("delta_conv", "delta_rec_hk128")
        );
        // delta_conv: 2 * 16 head groups + 4096 / 128 value groups, 3 chunks of 16 tokens
        assert_eq!(conv.threadgroups, [2 * 16 + 32, 3, 1]);
        assert_eq!(word(conv, 0) as usize, conv_dim);
        assert_eq!(
            (
                word(conv, 1),
                word(conv, 2),
                word(conv, 3),
                word(conv, 4),
                word(conv, 5),
                word(conv, 6)
            ),
            (16, 128, 4, 0, 40, 1)
        );
        assert_eq!(
            word(conv, 7) as usize,
            (state_offset + shape.rec_bytes()) / 4,
            "history follows the recurrent matrices"
        );
        assert_eq!(f32::from_bits(word(conv, 8)), 128f32.powf(-0.5));
        // delta_rec
        assert_eq!(rec.threadgroups, [32, 1, 1]);
        assert_eq!(word(rec, 8) as usize, state_offset / 4);
        assert_eq!((word(rec, 4), word(rec, 7)), (0, 1));
        // the second run uses its own state buffer and rows 40..41
        assert_eq!(plans[2].slots[2].source, SlotSource::RunState(2));
        assert_eq!(plans[3].slots[7].source, SlotSource::RunState(2));
        assert_eq!(
            (word(&plans[2], 4), word(&plans[2], 5), word(&plans[2], 6)),
            (40, 1, 0)
        );
        for p in &plans {
            // the conv slots read the activations and write mixed; the recurrence reads mixed
            check_plan(p, &[]);
        }
        assert_eq!(conv.slots[3].access, Access::Write);
        assert_eq!(rec.slots[0].access, Access::Read);
        assert_eq!(rec.slots[0].name, "mixed");
        let tiled = plan_delta_layer(
            &DeltaShape {
                tiled_heads: true,
                ..shape
            },
            &runs[..1],
            0,
        )
        .unwrap();
        assert_eq!(word(&tiled[1], 4), 1);
        assert!(
            plan_delta_layer(&shape, &runs, 2).is_err(),
            "unaligned state offset"
        );
        assert!(plan_delta_layer(&DeltaShape { k_dim: 80, ..shape }, &runs, 0).is_err());
        assert!(plan_delta_layer(&shape, &[], 0).unwrap().is_empty());
    }

    #[test]
    fn delta_conv_grid_covers_every_channel() {
        for (kh, vh, hk, hv) in [
            (16usize, 32usize, 128usize, 128usize),
            (4, 8, 64, 100),
            (1, 1, 32, 1),
            (2, 6, 96, 17),
        ] {
            let shape = DeltaShape {
                k_heads: kh,
                v_heads: vh,
                k_dim: hk,
                v_dim: hv,
                ..qwen_shape()
            };
            let plans = plan_delta_layer(
                &shape,
                &[RunSpec {
                    first_row: 0,
                    rows: 1,
                    reset: true,
                }],
                0,
            )
            .unwrap();
            let groups = plans[0].threadgroups[0] as usize;
            // groups 0 .. 2 kh: one head each (hk <= 128 channels); the others: 128 value channels each
            let value_groups = groups - 2 * kh;
            assert!(value_groups * 128 >= vh * hv, "{kh} {vh} {hk} {hv}");
            assert!(
                (value_groups - 1) * 128 < vh * hv,
                "no empty group at the end"
            );
        }
    }

    fn mamba2() -> Mamba2Shape {
        Mamba2Shape {
            inner: 4096,
            state: 128,
            heads: 64,
            groups: 8,
            conv_kernel: 4,
            norm_groups: 8,
            step_min: 0.0,
            eps: 1e-5,
        }
    }

    #[test]
    fn mamba2_mixer_plans() {
        let shape = mamba2();
        assert_eq!(shape.conv_dim(), 4096 + 2 * 8 * 128);
        assert_eq!(shape.head_dim(), 64);
        assert_eq!(shape.proj_width(), 4096 + shape.conv_dim() + 64);
        let runs = [
            RunSpec {
                first_row: 0,
                rows: 5,
                reset: false,
            },
            RunSpec {
                first_row: 5,
                rows: 3,
                reset: true,
            },
        ];
        let plans = plan_mamba2_mixer(&shape, &runs, 8, 1024).unwrap();
        assert_eq!(plans.len(), 5);
        let names: Vec<&str> = plans.iter().map(|p| &*p.kernel_name).collect();
        assert_eq!(
            names,
            [
                "mamba_conv",
                "mamba_scan2_ns128",
                "mamba_conv",
                "mamba_scan2_ns128",
                "mamba_gate_norm"
            ]
        );
        for p in &plans {
            check_plan(p, &[]);
        }
        let conv = &plans[0];
        assert_eq!(
            conv.threadgroups,
            [(shape.conv_dim() as u32).div_ceil(64), 1, 1]
        );
        assert_eq!(
            word(conv, 3) as usize,
            4096,
            "x starts after the gate in the projected row"
        );
        assert_eq!(
            word(conv, 7),
            1024 / 4,
            "history at the layer's state offset"
        );
        let scan = &plans[1];
        assert_eq!(scan.threadgroups, [64, 1, 1]);
        let (hist, _) = shape.state_floats();
        assert_eq!(
            word(scan, 11) as usize,
            1024 / 4 + hist,
            "recurrent state follows the history"
        );
        assert_eq!(
            word(scan, 7) as usize,
            4096 + shape.conv_dim(),
            "dt at the end of the projected row"
        );
        assert_eq!(f32::from_bits(word(scan, 12)), 0.0);
        let SlotSource::Scratch { id, bytes } = conv.slots[4].source.clone() else {
            panic!("mixed is scratch")
        };
        assert_eq!((id, bytes), (1, 8 * shape.conv_dim() * 4));
        assert_eq!(scan.slots[0].source, conv.slots[4].source);
        assert_eq!(scan.slots[0].access, Access::Read);
        let norm = &plans[4];
        assert_eq!(norm.threadgroups, [8, 8, 1]);
        assert_eq!(word(norm, 4), 0, "gate at the start of the projected row");
        assert_eq!(shape.layer_state_bytes() % 256, 0);
    }

    #[test]
    fn mamba_scan_variant_selection() {
        let mut shape = mamba2();
        for (state, expected) in [
            (16, "mamba_scan2_ns16"),
            (32, "mamba_scan2_ns32"),
            (64, "mamba_scan2_ns64"),
            (128, "mamba_scan2_ns128"),
            (48, "mamba_scan2_gen"),
            (256, "mamba_scan2_gen"),
        ] {
            shape.state = state;
            let p = plan_mamba2_scan(
                0,
                &RunSpec {
                    first_row: 0,
                    rows: 1,
                    reset: true,
                },
                &shape,
                1,
                0,
                0,
                MixedBuffer::Caller,
            )
            .unwrap();
            assert_eq!(p.kernel_name, expected);
            check_plan(&p, &[]);
        }
        shape.state = 257;
        assert!(plan_mamba2_scan(
            0,
            &RunSpec {
                first_row: 0,
                rows: 1,
                reset: true
            },
            &shape,
            1,
            0,
            0,
            MixedBuffer::Caller
        )
        .is_err());
        let m1 = |state| Mamba1Shape {
            inner: 2048,
            state,
            rank: 128,
            conv_kernel: 4,
            bc_rms_eps: Some(1e-5),
        };
        for (state, expected) in [
            (8, "mamba_scan1_ns8"),
            (16, "mamba_scan1_ns16"),
            (32, "mamba_scan1_ns32"),
            (24, "mamba_scan1_gen"),
            (64, "mamba_scan1_gen"),
        ] {
            let p = plan_mamba1_scan(
                0,
                &RunSpec {
                    first_row: 2,
                    rows: 4,
                    reset: false,
                },
                &m1(state),
                4096,
                2048,
                0,
            )
            .unwrap();
            assert_eq!(p.kernel_name, expected);
            assert_eq!(p.threadgroups, [32, 1, 1]);
            check_plan(&p, &[]);
        }
    }

    #[test]
    fn mamba1_pieces() {
        let shape = Mamba1Shape {
            inner: 2048,
            state: 16,
            rank: 128,
            conv_kernel: 4,
            bc_rms_eps: Some(1e-5),
        };
        assert_eq!(shape.split_width(), 128 + 32);
        let bc = plan_mamba1_bc_norm(&shape, 7)
            .unwrap()
            .expect("norm requested");
        check_plan(&bc, &[]);
        assert_eq!(bc.threadgroups, [3, 7, 1]);
        assert_eq!(word(&bc, 3) as usize, shape.split_width());
        assert!(plan_mamba1_bc_norm(
            &Mamba1Shape {
                bc_rms_eps: None,
                ..shape
            },
            7
        )
        .unwrap()
        .is_none());
        let run = RunSpec {
            first_row: 0,
            rows: 9,
            reset: true,
        };
        let conv =
            plan_mamba_conv(1, &run, 2048, 4, 4096, 0, 17, false, MixedBuffer::Caller).unwrap();
        check_plan(&conv, &[]);
        assert_eq!(conv.slots[2].source, SlotSource::Dummy, "no bias");
        assert_eq!(conv.slots[3].source, SlotSource::RunState(1));
        assert_eq!(conv.slots[4].source, SlotSource::Buffer);
        assert_eq!(conv.threadgroups, [32, 1, 1]);
        assert!(plan_mamba_conv(0, &run, 2048, 9, 4096, 0, 0, false, MixedBuffer::Caller).is_err());
        assert!(plan_mamba_conv(
            0,
            &RunSpec { rows: 0, ..run },
            2048,
            4,
            4096,
            0,
            0,
            false,
            MixedBuffer::Caller
        )
        .is_err());
    }
}
