//! Dispatch planning of the non-matmul operations of the vulkan backend. Owner: package 3 (dispatch planning of attention, norms, rope, activations, KV write, MoE, DeltaNet, Mamba).
//!
//! Every `plan_*` function turns the arguments of one [`Stream`](crate::backend::Stream) operation into [`OpPlan`]s: the name of the SPIR-V shader
//! (`shaders::spirv(name)`), the push-constant bytes, the workgroup counts and the list of storage-buffer bindings. Nothing here touches Vulkan: the
//! package that owns the stream (package 4) resolves the binding names to buffers, uploads the small host arrays, binds, pushes and dispatches, and
//! inserts a barrier between two dependent plans. The shader side of every plan (bindings, push-constant layout, grid rule) is documented in
//! `shaders/vulkan/INTERFACE_OPS.md` and in the `// FLLM-SHADER:` header of every `.comp` file; the unit tests compare the plans with those headers.
//!
//! Level of verification: **L0** (compiles, unit-tested on the CPU; no shader has ever run on a device).

pub mod attention;
pub mod moe;
pub mod recurrent;

#[cfg(test)]
mod mirror;
#[cfg(test)]
mod tests;

use crate::backend::BackendError;
use std::sync::Arc;

/// Largest push-constant block any shader of this package uses (Vulkan guarantees 128 bytes).
pub const MAX_PUSH_BYTES: usize = 128;
/// `maxComputeWorkGroupCount` is at least 65535 per dimension; plans never exceed it.
pub const MAX_GROUPS_PER_DIM: u32 = 65535;

/// How a shader uses a binding (for barrier planning and for `readonly` views).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

/// Where the buffer bound to a slot comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum SlotSource {
    /// A [`DeviceBuffer`](crate::backend::DeviceBuffer) of the caller; `BindingSlot::name` tells which argument of the operation (see INTERFACE_OPS.md).
    Buffer,
    /// A small host array (little endian), uploaded for this dispatch (positions, slots, block tables, lengths, indices, biases of the attention).
    Host(Arc<Vec<u8>>),
    /// Uninitialised scratch storage of at least `bytes`. Slots with the same `id` of the plans of ONE operation share one buffer (its size is the
    /// maximum over the plans); the plans of an operation run in the order of the returned vector with a barrier in between.
    Scratch { id: u8, bytes: usize },
    /// Any valid storage buffer of at least 16 bytes: an optional input that is absent (bias, weight, sinks) which the shader does not read.
    Dummy,
    /// The state buffer of run `n` of a `delta_layer` / Mamba call (`DeltaRun::state`); the layer's state starts at the offset in the push constants.
    RunState(usize),
}

/// One storage-buffer binding of a dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct BindingSlot {
    /// Binding number in descriptor set 0 (the slots of a plan are numbered 0, 1, 2, ... in order).
    pub binding: u32,
    pub name: &'static str,
    pub access: Access,
    pub source: SlotSource,
}

/// One compute dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct OpPlan {
    /// Name for `shaders::spirv(name)` (an output name of `variants_ops.list` or the stem of a `.comp` file).
    pub shader_name: &'static str,
    /// Push-constant bytes (little-endian 32-bit words, at most [`MAX_PUSH_BYTES`]).
    pub push_constants: Vec<u8>,
    /// Names of the push-constant fields in order (documentation and test aid; they equal the members of the shader's `push_constant` block).
    pub push_names: Vec<&'static str>,
    /// Workgroup counts for `vkCmdDispatch`; every component is <= [`MAX_GROUPS_PER_DIM`]; a zero component means nothing to do.
    pub groups: [u32; 3],
    pub bindings: Vec<BindingSlot>,
}

impl OpPlan {
    /// Number of storage buffers the pipeline needs.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// The slot called `name`.
    pub fn slot(&self, name: &str) -> Option<&BindingSlot> {
        self.bindings.iter().find(|b| b.name == name)
    }
}

/// Builder of push-constant bytes; remembers the field names. `finish` fails when a value does not fit 32 bits or the block exceeds 128 bytes.
#[derive(Default)]
pub(crate) struct Push {
    bytes: Vec<u8>,
    names: Vec<&'static str>,
    error: Option<String>,
}

impl Push {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// An unsigned 32-bit field.
    pub(crate) fn u(mut self, name: &'static str, value: usize) -> Self {
        match u32::try_from(value) {
            Ok(v) => self.bytes.extend_from_slice(&v.to_le_bytes()),
            Err(_) => {
                self.error.get_or_insert_with(|| {
                    format!("push constant {name} = {value} does not fit 32 bits")
                });
                self.bytes.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        self.names.push(name);
        self
    }

    /// A boolean as 0 / 1.
    pub(crate) fn b(self, name: &'static str, value: bool) -> Self {
        self.u(name, usize::from(value))
    }

    /// A 32-bit float field.
    pub(crate) fn f(mut self, name: &'static str, value: f32) -> Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self.names.push(name);
        self
    }

    pub(crate) fn finish(self) -> Result<(Vec<u8>, Vec<&'static str>), BackendError> {
        if let Some(e) = self.error {
            return Err(BackendError::InvalidShape(e));
        }
        if self.bytes.len() > MAX_PUSH_BYTES {
            return Err(BackendError::InvalidShape(format!(
                "push constants of {} bytes exceed {MAX_PUSH_BYTES}",
                self.bytes.len()
            )));
        }
        Ok((self.bytes, self.names))
    }
}

/// Binding helpers: slot `n` in order.
pub(crate) struct Slots(Vec<BindingSlot>);

impl Slots {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }
    fn add(mut self, name: &'static str, access: Access, source: SlotSource) -> Self {
        let binding = self.0.len() as u32;
        self.0.push(BindingSlot {
            binding,
            name,
            access,
            source,
        });
        self
    }
    pub(crate) fn read(self, name: &'static str) -> Self {
        self.add(name, Access::Read, SlotSource::Buffer)
    }
    pub(crate) fn write(self, name: &'static str) -> Self {
        self.add(name, Access::Write, SlotSource::Buffer)
    }
    pub(crate) fn rw(self, name: &'static str) -> Self {
        self.add(name, Access::ReadWrite, SlotSource::Buffer)
    }
    /// A buffer that is a [`SlotSource::Buffer`] when `present`, a dummy otherwise.
    pub(crate) fn read_opt(self, name: &'static str, present: bool) -> Self {
        let source = if present {
            SlotSource::Buffer
        } else {
            SlotSource::Dummy
        };
        self.add(name, Access::Read, source)
    }
    pub(crate) fn host(self, name: &'static str, data: Arc<Vec<u8>>) -> Self {
        self.add(name, Access::Read, SlotSource::Host(data))
    }
    pub(crate) fn scratch(self, name: &'static str, id: u8, bytes: usize, access: Access) -> Self {
        self.add(
            name,
            access,
            SlotSource::Scratch {
                id,
                bytes: bytes.max(16),
            },
        )
    }
    pub(crate) fn dummy(self, name: &'static str, access: Access) -> Self {
        self.add(name, access, SlotSource::Dummy)
    }
    pub(crate) fn run_state(self, name: &'static str, run: usize) -> Self {
        self.add(name, Access::ReadWrite, SlotSource::RunState(run))
    }
    pub(crate) fn done(self) -> Vec<BindingSlot> {
        self.0
    }
}

/// Assemble a plan.
pub(crate) fn plan(
    shader_name: &'static str,
    push: Push,
    groups: [u32; 3],
    bindings: Slots,
) -> Result<OpPlan, BackendError> {
    let (push_constants, push_names) = push.finish()?;
    Ok(OpPlan {
        shader_name,
        push_constants,
        push_names,
        groups,
        bindings: bindings.done(),
    })
}

pub(crate) fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

/// `ceil(a / b)`.
pub(crate) fn div_up(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// One-dimensional grid for a grid-stride shader of `wg` invocations per workgroup: `ceil(items / wg)` workgroups, at most 65535 (the shader loops).
pub fn groups_1d(items: usize, wg: usize) -> [u32; 3] {
    let n = div_up(items, wg).min(MAX_GROUPS_PER_DIM as usize);
    [n as u32, 1, 1]
}

/// Grid for one workgroup per row (`group_index() = x + y * numGroupsX` in the shaders): `[rows, 1, 1]`, or `[65535, ceil(rows / 65535), 1]` above 65535 rows.
pub fn groups_rows(rows: usize) -> [u32; 3] {
    let max = MAX_GROUPS_PER_DIM as usize;
    if rows <= max {
        [rows as u32, 1, 1]
    } else {
        [max as u32, div_up(rows, max) as u32, 1]
    }
}

/// `values` as little-endian u32 words (positions, slots, token ids).
pub fn pack_u32(values: &[usize]) -> Result<Vec<u8>, BackendError> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        let w = u32::try_from(v).map_err(|_| invalid(format!("value {v} does not fit 32 bits")))?;
        out.extend_from_slice(&w.to_le_bytes());
    }
    Ok(out)
}

/// `values` as little-endian i32 words (lengths, block tables, row indices).
pub fn pack_i32(values: &[i64]) -> Result<Vec<u8>, BackendError> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        let w = i32::try_from(v).map_err(|_| invalid(format!("value {v} does not fit 32 bits")))?;
        out.extend_from_slice(&w.to_le_bytes());
    }
    Ok(out)
}

/// `values` as little-endian f32 words.
pub fn pack_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Norms
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// Row geometry of the norm shaders: row `r` of the input starts at float `x_off + r * x_stride`, row `r` of the output at `o_off + r * o_stride`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowLayout {
    pub x_stride: usize,
    pub x_off: usize,
    pub o_stride: usize,
    pub o_off: usize,
}

impl RowLayout {
    /// Dense rows of `cols` floats.
    pub fn dense(cols: usize) -> Self {
        Self {
            x_stride: cols,
            x_off: 0,
            o_stride: cols,
            o_off: 0,
        }
    }
}

/// `Stream::rms_norm` (`shader norm_rms`): `rows` rows of `cols` floats; `has_weight = false` multiplies by 1 (the `weight` slot is then a dummy).
pub fn plan_rms_norm(
    rows: usize,
    cols: usize,
    eps: f32,
    has_weight: bool,
) -> Result<OpPlan, BackendError> {
    plan_rms_norm_strided(rows, cols, eps, has_weight, RowLayout::dense(cols))
}

/// RMSNorm over slices of wider rows (for example the dt / B / C parts of a Mamba-1 `x_proj` row).
pub fn plan_rms_norm_strided(
    rows: usize,
    cols: usize,
    eps: f32,
    has_weight: bool,
    layout: RowLayout,
) -> Result<OpPlan, BackendError> {
    if rows == 0 || cols == 0 {
        return Err(invalid("rms_norm of an empty tensor"));
    }
    let push = Push::new()
        .u("cols", cols)
        .u("rows", rows)
        .u("x_stride", layout.x_stride)
        .u("x_off", layout.x_off)
        .u("o_stride", layout.o_stride)
        .u("o_off", layout.o_off)
        .f("eps", eps)
        .u("flags", usize::from(has_weight));
    plan(
        "norm_rms",
        push,
        groups_rows(rows),
        Slots::new()
            .read("x")
            .read_opt("weight", has_weight)
            .write("out"),
    )
}

/// `Stream::rms_norm_grouped` (`shader norm_rms_grouped`): `rows` rows of `cols` floats divided into `groups` groups.
pub fn plan_rms_norm_grouped(
    rows: usize,
    cols: usize,
    groups: usize,
    eps: f32,
    has_weight: bool,
) -> Result<OpPlan, BackendError> {
    if rows == 0 || cols == 0 {
        return Err(invalid("rms_norm_grouped of an empty tensor"));
    }
    if groups == 0 || !cols.is_multiple_of(groups) {
        return Err(invalid("rms_norm_grouped requires groups > 0 and cols divisible by groups"));
    }
    let group_size = cols / groups;
    let total_groups = rows * groups;
    let push = Push::new()
        .u("group_size", group_size)
        .u("groups", groups)
        .u("total_groups", total_groups)
        .u("cols_total", cols)
        .f("eps", eps)
        .u("flags", usize::from(has_weight));
    plan(
        "norm_rms_grouped",
        push,
        groups_rows(total_groups),
        Slots::new()
            .read("x")
            .read_opt("weight", has_weight)
            .write("out"),
    )
}

/// `Stream::layer_norm` (`norm_layer`): `rows` rows of `cols` floats with weight and optional bias.
pub fn plan_layer_norm(
    rows: usize,
    cols: usize,
    eps: f32,
    has_bias: bool,
) -> Result<OpPlan, BackendError> {
    if rows == 0 || cols == 0 {
        return Err(invalid("layer_norm of an empty tensor"));
    }
    let l = RowLayout::dense(cols);
    let push = Push::new()
        .u("cols", cols)
        .u("rows", rows)
        .u("x_stride", l.x_stride)
        .u("x_off", l.x_off)
        .u("o_stride", l.o_stride)
        .u("o_off", l.o_off)
        .f("eps", eps)
        .u("flags", usize::from(has_bias));
    plan(
        "norm_layer",
        push,
        groups_rows(rows),
        Slots::new()
            .read("x")
            .read("weight")
            .read_opt("bias", has_bias)
            .write("out"),
    )
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// `Stream::rope` (`rope_apply`) for `positions.len()` rows; q is `rows x heads x dim`, k is `rows x kv_heads x dim`, cos / sin the tables of `RopeParams`
/// (`positions x rope_dim / 2`, bound as `cos` and `sin`). The positions are uploaded with the dispatch (slot `positions`).
pub fn plan_rope(
    positions: &[usize],
    heads: usize,
    kv_heads: usize,
    dim: usize,
    rope_dim: usize,
    neox: bool,
    table_rows: usize,
) -> Result<OpPlan, BackendError> {
    let rows = positions.len();
    let half = rope_dim / 2;
    if rows == 0 || heads + kv_heads == 0 {
        return Err(invalid("rope of an empty batch"));
    }
    if !rope_dim.is_multiple_of(2) || half == 0 || rope_dim > dim {
        return Err(invalid(format!(
            "rope width {rope_dim} invalid for head width {dim}"
        )));
    }
    if let Some(&p) = positions.iter().find(|&&p| p >= table_rows) {
        return Err(invalid(format!(
            "rope position {p} >= table length {table_rows}"
        )));
    }
    let items = rows * (heads + kv_heads) * half;
    let push = Push::new()
        .u("rows", rows)
        .u("heads", heads)
        .u("kv_heads", kv_heads)
        .u("dim", dim)
        .u("half_dim", half)
        .b("neox", neox);
    let slots = Slots::new()
        .rw("q")
        .rw("k")
        .host("positions", Arc::new(pack_u32(positions)?))
        .read("cos")
        .read("sin");
    plan("rope_apply", push, groups_1d(items, 256), slots)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Activations, residual, gating, argmax
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// The gated and plain activations of `act_glu.comp`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GluMode {
    /// `Stream::swiglu`: `silu(g) * u`.
    Swiglu,
    /// `Stream::geglu`: `gelu_tanh(g) * u`.
    Geglu,
    /// `Stream::swiglu_oai` (gpt-oss).
    SwigluOai { alpha: f32, limit: f32 },
    /// `silu(g)` (non-gated MLP; `up` is not read).
    Silu,
    /// `gelu_tanh(g)`.
    Gelu,
    /// `max(g, 0)^2`.
    Relu2,
    /// `max(g, 0)`.
    Relu,
    /// `gelu_erf(g)` (`Stream::activation` mode 2).
    GeluErf,
    /// `gelu_erf(g) * u`.
    GeluErfGated,
    /// `max(g, 0) * u`.
    ReluGated,
    /// `max(g, 0)^2 * u`.
    Relu2Gated,
    /// `g` (strided copy).
    Copy,
}

impl GluMode {
    /// Output name in `variants_ops.list`.
    pub fn shader_name(self) -> &'static str {
        match self {
            Self::Swiglu => "act_swiglu",
            Self::Geglu => "act_geglu",
            Self::SwigluOai { .. } => "act_swiglu_oai",
            Self::Silu => "act_silu",
            Self::Gelu => "act_gelu",
            Self::Relu2 => "act_relu2",
            Self::Relu => "act_relu",
            Self::GeluErf => "act_gelu_erf",
            Self::GeluErfGated => "act_geglu_erf",
            Self::ReluGated => "act_reglu",
            Self::Relu2Gated => "act_reglu2",
            Self::Copy => "act_copy",
        }
    }

    /// Whether the `up` operand is read.
    pub fn gated(self) -> bool {
        matches!(
            self,
            Self::Swiglu | Self::Geglu | Self::SwigluOai { .. } | Self::GeluErfGated | Self::ReluGated | Self::Relu2Gated
        )
    }
}

/// Operand geometry of `act_glu`: element `(r, c)` is `gate[gate_off + r * gate_stride + c]`, `up[up_off + r * up_stride + c]`, `out[r * out_stride + c]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GluLayout {
    pub gate_stride: usize,
    pub gate_off: usize,
    pub up_stride: usize,
    pub up_off: usize,
    pub out_stride: usize,
}

impl GluLayout {
    pub fn dense(width: usize) -> Self {
        Self {
            gate_stride: width,
            gate_off: 0,
            up_stride: width,
            up_off: 0,
            out_stride: width,
        }
    }
}

/// Activation over `count` contiguous floats (`Stream::swiglu`, `geglu`, `swiglu_oai`).
pub fn plan_glu(mode: GluMode, count: usize) -> Result<OpPlan, BackendError> {
    plan_glu_strided(mode, 1, count, GluLayout::dense(count))
}

/// Activation over `n_rows` rows of `width` floats with explicit strides (fused gate/up projections: bind the same buffer to `gate` and `up`).
pub fn plan_glu_strided(
    mode: GluMode,
    n_rows: usize,
    width: usize,
    layout: GluLayout,
) -> Result<OpPlan, BackendError> {
    if n_rows == 0 || width == 0 {
        return Err(invalid("activation of an empty tensor"));
    }
    let (alpha, limit) = match mode {
        GluMode::SwigluOai { alpha, limit } => (alpha, limit),
        _ => (0.0, 0.0),
    };
    let push = Push::new()
        .u("n_rows", n_rows)
        .u("width", width)
        .u("gate_stride", layout.gate_stride)
        .u("gate_off", layout.gate_off)
        .u("up_stride", layout.up_stride)
        .u("up_off", layout.up_off)
        .u("out_stride", layout.out_stride)
        .f("alpha", alpha)
        .f("limit", limit);
    let slots = Slots::new()
        .read("gate")
        .read_opt("up", mode.gated())
        .write("out");
    plan(
        mode.shader_name(),
        push,
        groups_1d(n_rows * width, 256),
        slots,
    )
}

/// `Stream::residual_add` (`misc_residual`): `x += residual * scale` over `count` floats.
pub fn plan_residual_add(count: usize, scale: f32) -> Result<OpPlan, BackendError> {
    if count == 0 {
        return Err(invalid("residual_add of an empty tensor"));
    }
    plan(
        "misc_residual",
        Push::new().u("count", count).f("scale", scale),
        groups_1d(count, 256),
        Slots::new().rw("x").read("residual"),
    )
}

/// `Stream::clamp` (`ewise_clamp`: `x = clamp(x, -limit, limit)`) and `Stream::softplus_gate_mul` (`ewise_softplus`: `x *= softplus(beta g) / beta`).
pub fn plan_ewise(softplus: bool, count: usize, param: f32) -> Result<OpPlan, BackendError> {
    if count == 0 {
        return Err(invalid("element-wise operation on an empty tensor"));
    }
    plan(
        if softplus { "ewise_softplus" } else { "ewise_clamp" },
        Push::new().u("count", count).f("param", param),
        groups_1d(count, 256),
        Slots::new().rw("x").read_opt("gate", softplus),
    )
}

/// `Stream::scale_inplace` (`ewise_scale`: `x *= factor`).
pub fn plan_scale(count: usize, factor: f32) -> Result<OpPlan, BackendError> {
    if count == 0 {
        return Err(invalid("scale of an empty tensor"));
    }
    plan("ewise_scale", Push::new().u("count", count).f("param", factor), groups_1d(count, 256), Slots::new().rw("x"))
}

/// `Stream::head_gate_mul` (`act_head_gate_sigmoid` / `act_head_gate_softplus`): `x[i] *= act(gate[i / dv])` over `count` floats.
pub fn plan_head_gate(count: usize, dv: usize, softplus_beta: Option<f32>) -> Result<OpPlan, BackendError> {
    if count == 0 || dv == 0 || !count.is_multiple_of(dv) {
        return Err(invalid("per-head gate: the output is not a whole number of heads"));
    }
    plan(
        if softplus_beta.is_some() { "act_head_gate_softplus" } else { "act_head_gate_sigmoid" },
        Push::new().u("count", count).u("dv", dv).f("beta", softplus_beta.unwrap_or(1.0)),
        groups_1d(count, 256),
        Slots::new().rw("x").read("gate"),
    )
}

/// Which output `mla_move` produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlaPart {
    Q,
    K,
    V,
    Latent,
}

/// `Stream::mla_assemble` / `mla_latent` (`mla_move_*`): one dispatch per part. Operands: `a`, `b`, `dst` as in the header of `mla_move.comp`.
pub fn plan_mla_move(part: MlaPart, rows: usize, heads: usize, nope: usize, rope: usize, value: usize, kv_lora: usize) -> Result<OpPlan, BackendError> {
    let dk = nope + rope;
    let (name, count) = match part {
        MlaPart::Q => ("mla_move_q", rows * heads * dk),
        MlaPart::K => ("mla_move_k", rows * heads * dk),
        MlaPart::V => ("mla_move_v", rows * heads * value),
        MlaPart::Latent => ("mla_move_latent", rows * kv_lora),
    };
    if count == 0 {
        return Err(invalid("MLA data movement of an empty tensor"));
    }
    let push = Push::new()
        .u("count", count)
        .u("rows", rows)
        .u("heads", heads)
        .u("nope", nope)
        .u("rope", rope)
        .u("value", value)
        .u("kv_lora", kv_lora);
    plan(
        name,
        push,
        groups_1d(count, 256),
        Slots::new().read("a").read_opt("b", part == MlaPart::K).write("dst"),
    )
}

/// `Stream::sigmoid_gate_mul` (`act_sigmoid_mul`): `x *= sigmoid(gate)` over `count` floats.
pub fn plan_sigmoid_gate_mul(count: usize) -> Result<OpPlan, BackendError> {
    if count == 0 {
        return Err(invalid("sigmoid_gate_mul of an empty tensor"));
    }
    plan(
        "act_sigmoid_mul",
        Push::new().u("count", count),
        groups_1d(count, 256),
        Slots::new().rw("x").read("gate"),
    )
}

/// `Stream::split_q_gate` (`misc_split_q_gate`).
pub fn plan_split_q_gate(
    rows: usize,
    heads: usize,
    dk: usize,
    per_head: bool,
) -> Result<OpPlan, BackendError> {
    let total = rows * heads * dk;
    if total == 0 {
        return Err(invalid("split_q_gate of an empty tensor"));
    }
    let push = Push::new()
        .u("total", total)
        .u("heads", heads)
        .u("dk", dk)
        .b("per_head", per_head);
    plan(
        "misc_split_q_gate",
        push,
        groups_1d(total, 256),
        Slots::new().read("projected").write("q").write("gate"),
    )
}

/// `Stream::argmax_rows` (`misc_argmax`): one int32 per row in `out`.
pub fn plan_argmax(rows: usize, vocab: usize) -> Result<OpPlan, BackendError> {
    if rows == 0 || vocab == 0 {
        return Err(invalid("argmax of an empty tensor"));
    }
    plan(
        "misc_argmax",
        Push::new().u("vocab", vocab).u("rows", rows),
        groups_rows(rows),
        Slots::new().read("logits").write("out"),
    )
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// Geometry of an embedding table for `plan_embedding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddingTable {
    pub quant: crate::backend::QuantType,
    /// Vocabulary size (rows of the table).
    pub rows: usize,
    /// Hidden size (columns).
    pub cols: usize,
    /// Word offsets of the planes 1..3 of the quantized device layout (`DeviceLayout::plane_word_offset[1..4]` of the weights package; unused for F32 / F16 / BF16).
    pub planes: [usize; 3],
}

/// Shader suffix and weights per block of the formats of `embed.comp` / `moe_linear.comp` (`None`: not supported there). F32 / F16 / BF16 have no
/// blocks (`qk` = 1) and are stored row-major without planes.
pub fn format_tag(quant: crate::backend::QuantType) -> Option<(&'static str, usize)> {
    use crate::backend::QuantType as Q;
    Some(match quant {
        Q::F32 => ("f32", 1),
        Q::F16 => ("f16", 1),
        Q::BF16 => ("bf16", 1),
        Q::Q4_0 => ("q4_0", 32),
        Q::Q4_1 => ("q4_1", 32),
        Q::Q5_0 => ("q5_0", 32),
        Q::Q5_1 => ("q5_1", 32),
        Q::Q8_0 => ("q8_0", 32),
        Q::Q2_K => ("q2_k", 256),
        Q::Q3_K => ("q3_k", 256),
        Q::Q4_K => ("q4_k", 256),
        Q::Q5_K => ("q5_k", 256),
        Q::Q6_K => ("q6_k", 256),
        Q::IQ4_NL => ("iq4_nl", 32),
        Q::Mxfp4 => ("mxfp4", 32),
        _ => return None,
    })
}

/// Names of the `embed_*` shaders (static table: plans carry `&'static str`).
fn embed_shader(tag: &str) -> Option<&'static str> {
    Some(match tag {
        "f32" => "embed_f32",
        "f16" => "embed_f16",
        "bf16" => "embed_bf16",
        "q4_0" => "embed_q4_0",
        "q4_1" => "embed_q4_1",
        "q5_0" => "embed_q5_0",
        "q5_1" => "embed_q5_1",
        "q8_0" => "embed_q8_0",
        "q2_k" => "embed_q2_k",
        "q3_k" => "embed_q3_k",
        "q4_k" => "embed_q4_k",
        "q5_k" => "embed_q5_k",
        "q6_k" => "embed_q6_k",
        "iq4_nl" => "embed_iq4_nl",
        "mxfp4" => "embed_mxfp4",
        _ => return None,
    })
}

/// `Stream::embedding_lookup` (`embed_*`): `out[t] = table[ids[t]]` for every id (checked against the vocabulary here). The ids are uploaded with the dispatch.
pub fn plan_embedding(table: &EmbeddingTable, ids: &[i64]) -> Result<OpPlan, BackendError> {
    use crate::backend::QuantType;
    let (tag, qk) = format_tag(table.quant).ok_or(BackendError::UnsupportedQuant(table.quant))?;
    let name = embed_shader(tag).ok_or(BackendError::UnsupportedQuant(table.quant))?;
    if ids.is_empty() {
        return Err(invalid("embedding lookup without ids"));
    }
    if table.cols == 0
        || table.rows == 0
        || !table.cols.is_multiple_of(qk)
        || (qk > 1 && !table.cols.is_multiple_of(8))
    {
        return Err(invalid(format!(
            "{:?} embedding table {} x {} is not a whole number of blocks",
            table.quant, table.rows, table.cols
        )));
    }
    if let Some(&id) = ids.iter().find(|&&id| id < 0 || id as usize >= table.rows) {
        return Err(invalid(format!("embedding token id {id} out of bounds")));
    }
    // word indices are 32 bit
    let raw_elements = table
        .rows
        .checked_mul(table.cols)
        .filter(|&n| n <= u32::MAX as usize);
    if raw_elements.is_none() {
        return Err(BackendError::UnsupportedOperation(
            "embedding table with more than 2^32 elements".into(),
        ));
    }
    let quantized = !matches!(
        table.quant,
        QuantType::F32 | QuantType::F16 | QuantType::BF16
    );
    let work = if quantized {
        ids.len() * (table.cols / 8)
    } else {
        ids.len() * table.cols
    };
    let nb = if quantized { table.cols / qk } else { 0 };
    let push = Push::new()
        .u("n_ids", ids.len())
        .u("cols", table.cols)
        .u("nb", nb)
        .u("off1", table.planes[0])
        .u("off2", table.planes[1])
        .u("off3", table.planes[2]);
    let id_bytes = pack_u32(&ids.iter().map(|&i| i as usize).collect::<Vec<_>>())?;
    plan(
        name,
        push,
        groups_1d(work, 256),
        Slots::new()
            .read("weights")
            .host("ids", Arc::new(id_bytes))
            .write("out"),
    )
}
