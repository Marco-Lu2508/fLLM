//! Dispatch planning of the non-matmul operations of the metal backend. Owner: package p06b (norms, RoPE, activations, residual, argmax,
//! embedding, KV-cache write, paged attention); the MoE and recurrent planners (`moe_plan`, `recurrent_plan`) belong to package p06c and reuse
//! the types of this file.
//!
//! Every `plan_*` function turns the arguments of one [`Stream`](crate::backend::Stream) operation into [`OpPlan`]s: the name of the kernel
//! (`pipelines.get(kernel_name, &[])`), the bytes of its `constant` parameter struct, the threadgroup grid and the list of buffer
//! bindings. Nothing here touches Metal: the stream (package p07) resolves the slot names to buffers, uploads the small host arrays
//! (`SlotSource::Host`), binds, dispatches and declares the accesses for the hazard tracker (`Access` of every slot). The kernel side of every plan
//! (buffer indices, parameter struct, grid rule) is documented in `metal/INTERFACE_OPS.md` and in the `FLLM-KERNEL` header comment of every
//! `.metal` file; the unit tests compare the plans with those headers and with the source text, so the plans cannot drift from the kernels.
//!
//! Conventions (shared with package p06c, whose planners fill the same [`OpPlan`]):
//! * the `constant` struct of the kernel (parameters; [`OpPlan::constants`], at most [`MAX_PARAM_BYTES`]) is bound with `setBytes` at buffer
//!   [`CONSTANTS_BUFFER_INDEX`] (0), the buffer slots follow from index 1 in the order of [`OpPlan::slots`]; the kernels declare the struct as
//!   `constant <struct> &p [[buffer(0)]]`;
//! * parameter structs consist of 4-byte scalars (`uint`, `int`, `float`), optionally led by 8-byte `ulong` fields at 8-byte aligned offsets;
//!   the struct definitions live in `metal/common_ops.metal` and [`Params`] writes the same layout (explicit `_pad` words at the end);
//! * `threadgroups` is the grid in threadgroups (`dispatchThreadgroups`), `threads_per_group` the threadgroup size; Metal has no 65535 limit;
//! * variants are separate kernel names (no function constants), threadgroup memory is declared statically in the kernels;
//! * activations are f32, one row per token; kernel names are global (`fllm_<op>_<variant>`), see `INTERFACE_OPS.md`.
//!
//! Level of verification: **L0** (compiles; unit-tested on the CPU; no kernel has ever run on a device).

pub mod arch_plan;
pub mod attention;
pub mod embedding;
pub mod moe_plan;
pub mod recurrent_plan;

#[cfg(test)]
mod mirror;
#[cfg(test)]
mod tests;

// Hardware check for the Mac test round (needs a Metal device; `cargo test --release -p fllm-engine --features metal metal_ops_kernels -- --nocapture`).
// The whole `backend::metal` module exists only for Apple targets with the feature `metal`; the `target_vendor` condition also keeps the host
// harnesses (which compile this file without the Metal core) working.
#[cfg(all(test, target_vendor = "apple"))]
mod gpu_check;

use crate::backend::BackendError;
use std::sync::Arc;

/// Largest parameter struct passed with `setBytes` (Metal limit).
pub const MAX_PARAM_BYTES: usize = 4096;

/// Buffer index of the `constant` parameter struct of every kernel; the buffer slots start at index 1.
pub const CONSTANTS_BUFFER_INDEX: u32 = 0;

/// How a kernel uses a binding (for the hazard tracker of the command recorder).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

/// Where the buffer bound to a slot comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum SlotSource {
    /// A [`DeviceBuffer`](crate::backend::DeviceBuffer) of the caller; [`BufferSlot::name`] tells which argument of the operation (see INTERFACE_OPS.md).
    Buffer,
    /// A small host array (little endian), uploaded for this dispatch (positions, slots, block tables, lengths, indices, sinks).
    Host(Arc<Vec<u8>>),
    /// Uninitialised scratch storage of at least `bytes`. Slots with the same `id` in the plans of ONE operation share one buffer (size = the maximum
    /// over the plans); the plans of an operation run in the order of the returned vector with a barrier in between where they depend on each other.
    Scratch { id: u8, bytes: usize },
    /// Any valid buffer of at least 16 bytes: an optional input that is absent (weight, bias, sinks) and which the kernel does not read.
    Dummy,
    /// The state buffer of run `n` of a recurrent-layer call (`DeltaRun::state`); the layer's state starts at the offset in the parameters.
    RunState(usize),
}

/// One buffer binding of a dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct BufferSlot {
    /// Metal buffer index (`[[buffer(index)]]`): 1, 2, 3, ... in order (index 0 is the constants struct).
    pub index: u32,
    pub name: &'static str,
    pub access: Access,
    pub source: SlotSource,
}

/// One compute dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct OpPlan {
    /// Kernel function name (an entry of the embedded MSL sources).
    pub kernel_name: String,
    /// Bytes of the `constant` parameter struct, bound with `setBytes` at [`CONSTANTS_BUFFER_INDEX`].
    pub constants: Vec<u8>,
    /// Grid in threadgroups.
    pub threadgroups: [u32; 3],
    /// Threads per threadgroup.
    pub threads_per_group: [u32; 3],
    /// Buffer bindings, in binding order (buffer index 1, 2, ...).
    pub slots: Vec<BufferSlot>,
}

impl OpPlan {
    /// Number of buffers the kernel needs (not counting the constants struct).
    pub fn buffer_count(&self) -> usize {
        self.slots.len()
    }

    /// The slot called `name`.
    pub fn slot(&self, name: &str) -> Option<&BufferSlot> {
        self.slots.iter().find(|b| b.name == name)
    }

    /// Total threads of the grid (`threadgroups * threads_per_group`), saturating.
    pub fn total_threads(&self) -> u64 {
        let g: u64 = self.threadgroups.iter().map(|&v| u64::from(v)).product();
        let t: u64 = self
            .threads_per_group
            .iter()
            .map(|&v| u64::from(v))
            .product();
        g.saturating_mul(t)
    }

    /// Whether the grid is empty (a zero component): nothing to dispatch.
    pub fn is_empty(&self) -> bool {
        self.threadgroups.contains(&0)
    }
}
/// Builder of parameter-struct bytes; remembers the field names. `finish` fails when a value does not fit 32 bits or the struct exceeds
/// [`MAX_PARAM_BYTES`]. 64-bit fields (`ulong`) must start at 8-byte aligned offsets (put them first); a struct with a 64-bit field is padded to
/// a multiple of 8 bytes with `_pad` words, as MSL does.
#[derive(Default)]
pub struct Params {
    bytes: Vec<u8>,
    names: Vec<&'static str>,
    error: Option<String>,
    wide: bool,
}

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    /// An unsigned 32-bit field.
    pub fn u(mut self, name: &'static str, value: usize) -> Self {
        match u32::try_from(value) {
            Ok(v) => self.bytes.extend_from_slice(&v.to_le_bytes()),
            Err(_) => {
                self.error.get_or_insert_with(|| {
                    format!("parameter {name} = {value} does not fit 32 bits")
                });
                self.bytes.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        self.names.push(name);
        self
    }

    /// A signed 32-bit field.
    pub fn i(mut self, name: &'static str, value: i64) -> Self {
        match i32::try_from(value) {
            Ok(v) => self.bytes.extend_from_slice(&v.to_le_bytes()),
            Err(_) => {
                self.error
                    .get_or_insert_with(|| format!("parameter {name} = {value} does not fit i32"));
                self.bytes.extend_from_slice(&0i32.to_le_bytes());
            }
        }
        self.names.push(name);
        self
    }

    /// A boolean as 0 / 1.
    pub fn b(self, name: &'static str, value: bool) -> Self {
        self.u(name, usize::from(value))
    }

    /// A 32-bit float field.
    pub fn f(mut self, name: &'static str, value: f32) -> Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self.names.push(name);
        self
    }

    /// A 64-bit unsigned field (`ulong`); the offset must be a multiple of 8.
    pub fn u64(mut self, name: &'static str, value: u64) -> Self {
        if !self.bytes.len().is_multiple_of(8) {
            self.error.get_or_insert_with(|| {
                format!(
                    "64-bit parameter {name} at the unaligned offset {}",
                    self.bytes.len()
                )
            });
        }
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self.names.push(name);
        self.wide = true;
        self
    }

    pub fn finish(mut self) -> Result<(Vec<u8>, Vec<&'static str>), BackendError> {
        if let Some(e) = self.error {
            return Err(BackendError::InvalidShape(e));
        }
        while self.wide && !self.bytes.len().is_multiple_of(8) {
            self.bytes.extend_from_slice(&0u32.to_le_bytes());
            self.names.push("_pad");
        }
        if self.bytes.len() > MAX_PARAM_BYTES {
            return Err(BackendError::InvalidShape(format!(
                "parameter struct of {} bytes exceeds {MAX_PARAM_BYTES}",
                self.bytes.len()
            )));
        }
        Ok((self.bytes, self.names))
    }
}

/// Builder of the buffer bindings: slot `n` in call order.
#[derive(Default)]
pub struct Slots(Vec<BufferSlot>);

impl Slots {
    pub fn new() -> Self {
        Self::default()
    }

    fn add(mut self, name: &'static str, access: Access, source: SlotSource) -> Self {
        let index = self.0.len() as u32 + 1;
        self.0.push(BufferSlot {
            index,
            name,
            access,
            source,
        });
        self
    }

    pub fn read(self, name: &'static str) -> Self {
        self.add(name, Access::Read, SlotSource::Buffer)
    }

    pub fn write(self, name: &'static str) -> Self {
        self.add(name, Access::Write, SlotSource::Buffer)
    }

    pub fn rw(self, name: &'static str) -> Self {
        self.add(name, Access::ReadWrite, SlotSource::Buffer)
    }

    /// A buffer that is a [`SlotSource::Buffer`] when `present`, a dummy otherwise.
    pub fn read_opt(self, name: &'static str, present: bool) -> Self {
        let source = if present {
            SlotSource::Buffer
        } else {
            SlotSource::Dummy
        };
        self.add(name, Access::Read, source)
    }

    pub fn host(self, name: &'static str, data: Arc<Vec<u8>>) -> Self {
        self.add(name, Access::Read, SlotSource::Host(data))
    }

    pub fn scratch(self, name: &'static str, id: u8, bytes: usize, access: Access) -> Self {
        self.add(
            name,
            access,
            SlotSource::Scratch {
                id,
                bytes: bytes.max(16),
            },
        )
    }

    pub fn dummy(self, name: &'static str, access: Access) -> Self {
        self.add(name, access, SlotSource::Dummy)
    }

    pub fn run_state(self, name: &'static str, run: usize) -> Self {
        self.add(name, Access::ReadWrite, SlotSource::RunState(run))
    }

    pub fn done(self) -> Vec<BufferSlot> {
        self.0
    }
}

/// Assemble a plan (constants struct at buffer 0, the slots from buffer 1 on).
pub fn plan(
    kernel_name: &'static str,
    params: Params,
    threadgroups: [u32; 3],
    threads_per_group: [u32; 3],
    slots: Slots,
) -> Result<OpPlan, BackendError> {
    let (constants, field_names) = params.finish()?;
    #[cfg(test)]
    tests::remember_fields(kernel_name, &field_names);
    #[cfg(not(test))]
    let _ = field_names;
    Ok(OpPlan {
        kernel_name: kernel_name.to_string(),
        constants,
        threadgroups,
        threads_per_group,
        slots: slots.done(),
    })
}
pub fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

/// `ceil(a / b)`.
pub fn div_up(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// `a` rounded up to a multiple of `b`.
pub fn round_up(a: usize, b: usize) -> usize {
    a.next_multiple_of(b)
}

/// A grid dimension as `u32`; plans above 2^32 - 1 groups per dimension are rejected.
pub fn grid_dim(groups: usize) -> Result<u32, BackendError> {
    u32::try_from(groups)
        .map_err(|_| invalid(format!("{groups} threadgroups in one grid dimension")))
}

/// One-dimensional grid of `ceil(items / threads)` groups of `threads` threads.
pub fn groups_1d(items: usize, threads: usize) -> Result<([u32; 3], [u32; 3]), BackendError> {
    Ok((
        [grid_dim(div_up(items, threads))?, 1, 1],
        [threads as u32, 1, 1],
    ))
}

/// Threads of a reduction threadgroup that loops over `items` elements: a multiple of 32, 32..=512 (the kernels keep one partial result per
/// simdgroup in a 32-entry array, so a group has at most 32 simdgroups on every device with SIMD width >= 16).
pub fn reduce_threads(items: usize) -> usize {
    round_up(items.max(1), 32).clamp(32, 512)
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

/// Row geometry of the norm kernels: row `r` of the input starts at float `x_off + r * x_stride`, row `r` of the output at `o_off + r * o_stride`.
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

/// `Stream::rms_norm` (`fllm_rms_norm_f32` / `fllm_rms_norm_f32x4`): `rows` rows of `cols` floats; `has_weight = false` multiplies by 1 (the
/// `weight` slot is then a dummy). `out` may be the same buffer range as `x`.
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
    let (kernel, items) = norm_variant("fllm_rms_norm_f32", "fllm_rms_norm_f32x4", cols);
    let params = norm_params(rows, cols, eps, usize::from(has_weight), layout);
    plan(
        kernel,
        params,
        [grid_dim(rows)?, 1, 1],
        [reduce_threads(items) as u32, 1, 1],
        Slots::new()
            .read("x")
            .read_opt("weight", has_weight)
            .write("out"),
    )
}

/// `Stream::layer_norm` (`fllm_layer_norm_f32` / `fllm_layer_norm_f32x4`): `rows` rows of `cols` floats with weight and optional bias; biased
/// variance like the CPU backend.
pub fn plan_layer_norm(
    rows: usize,
    cols: usize,
    eps: f32,
    has_bias: bool,
) -> Result<OpPlan, BackendError> {
    if rows == 0 || cols == 0 {
        return Err(invalid("layer_norm of an empty tensor"));
    }
    let (kernel, items) = norm_variant("fllm_layer_norm_f32", "fllm_layer_norm_f32x4", cols);
    let params = norm_params(
        rows,
        cols,
        eps,
        usize::from(has_bias),
        RowLayout::dense(cols),
    );
    plan(
        kernel,
        params,
        [grid_dim(rows)?, 1, 1],
        [reduce_threads(items) as u32, 1, 1],
        Slots::new()
            .read("x")
            .read("weight")
            .read_opt("bias", has_bias)
            .write("out"),
    )
}

/// Scalar or `packed_float4` variant of a norm kernel and the number of items one thread loops over.
fn norm_variant(scalar: &'static str, vector: &'static str, cols: usize) -> (&'static str, usize) {
    if cols.is_multiple_of(4) {
        (vector, cols / 4)
    } else {
        (scalar, cols)
    }
}

fn norm_params(rows: usize, cols: usize, eps: f32, flags: usize, layout: RowLayout) -> Params {
    Params::new()
        .u("cols", cols)
        .u("rows", rows)
        .u("x_stride", layout.x_stride)
        .u("x_off", layout.x_off)
        .u("o_stride", layout.o_stride)
        .u("o_off", layout.o_off)
        .f("eps", eps)
        .u("flags", flags)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// `Stream::rope` (`fllm_rope_f32`) for `positions.len()` rows; q is `rows x heads x dim`, k is `rows x kv_heads x dim`, `cos` / `sin` the tables
/// of `RopeParams` (`table_rows x rope_dim / 2`). The positions are uploaded with the dispatch (slot `positions`). The dimensions of a head from
/// `rope_dim` on (partial rotary) are not touched.
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
    let tx = round_up(half, 32).min(256);
    let params = Params::new()
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
    plan(
        "fllm_rope_f32",
        params,
        [
            grid_dim(div_up(half, tx))?,
            grid_dim(heads + kv_heads)?,
            grid_dim(rows)?,
        ],
        [tx as u32, 1, 1],
        slots,
    )
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Activations, residual, gating, argmax
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// The gated and plain activations of `act_glu.metal`.
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
    /// `gelu_erf(g)` (exact erf).
    GeluErf,
    /// `gelu_erf(g) * u`.
    GegluErf,
    /// `max(g, 0) * u`.
    Reglu,
    /// `max(g, 0)^2 * u`.
    Relu2Glu,
}

impl GluMode {
    /// Kernel function name.
    pub fn kernel_name(self) -> &'static str {
        match self {
            Self::Swiglu => "fllm_act_swiglu",
            Self::Geglu => "fllm_act_geglu",
            Self::SwigluOai { .. } => "fllm_act_swiglu_oai",
            Self::Silu => "fllm_act_silu",
            Self::Gelu => "fllm_act_gelu",
            Self::Relu2 => "fllm_act_relu2",
            Self::Relu => "fllm_act_relu",
            Self::GeluErf => "fllm_act_gelu_erf",
            Self::GegluErf => "fllm_act_geglu_erf",
            Self::Reglu => "fllm_act_reglu",
            Self::Relu2Glu => "fllm_act_relu2glu",
        }
    }

    /// Whether the `up` operand is read.
    pub fn gated(self) -> bool {
        matches!(
            self,
            Self::Swiglu | Self::Geglu | Self::SwigluOai { .. } | Self::GegluErf | Self::Reglu | Self::Relu2Glu
        )
    }

    /// The mode of `Stream::activation` (`gated = false`) or `Stream::activation_glu` (`gated = true`): 0 SiLU, 1 GELU (tanh), 2 GELU (erf), 3 ReLU,
    /// 4 ReLU squared.
    pub fn from_activation(mode: i32, gated: bool) -> Option<Self> {
        Some(match (mode, gated) {
            (0, false) => Self::Silu,
            (1, false) => Self::Gelu,
            (2, false) => Self::GeluErf,
            (3, false) => Self::Relu,
            (4, false) => Self::Relu2,
            (0, true) => Self::Swiglu,
            (1, true) => Self::Geglu,
            (2, true) => Self::GegluErf,
            (3, true) => Self::Reglu,
            (4, true) => Self::Relu2Glu,
            _ => return None,
        })
    }
}

/// Operand geometry of the activation kernels: element `(r, c)` is `gate[gate_off + r * gate_stride + c]`, `up[up_off + r * up_stride + c]`,
/// `out[r * out_stride + c]`.
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
/// `out` may be the same buffer range as `gate` (elementwise).
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
    let params = Params::new()
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
    let tx = round_up(width, 32).min(256);
    plan(
        mode.kernel_name(),
        params,
        [grid_dim(div_up(width, tx))?, grid_dim(n_rows)?, 1],
        [tx as u32, 1, 1],
        slots,
    )
}

/// `Stream::residual_add` (`fllm_residual_add_f32`): `x += residual * scale` over `count` floats.
pub fn plan_residual_add(count: usize, scale: f32) -> Result<OpPlan, BackendError> {
    if count == 0 {
        return Err(invalid("residual_add of an empty tensor"));
    }
    let (groups, threads) = groups_1d(count, 256)?;
    plan(
        "fllm_residual_add_f32",
        Params::new().u("count", count).f("scale", scale),
        groups,
        threads,
        Slots::new().rw("x").read("residual"),
    )
}

/// `Stream::sigmoid_gate_mul` (`fllm_sigmoid_mul_f32`): `x *= sigmoid(gate)` over `count` floats.
pub fn plan_sigmoid_gate_mul(count: usize) -> Result<OpPlan, BackendError> {
    if count == 0 {
        return Err(invalid("sigmoid_gate_mul of an empty tensor"));
    }
    let (groups, threads) = groups_1d(count, 256)?;
    plan(
        "fllm_sigmoid_mul_f32",
        Params::new().u("count", count).f("scale", 1.0),
        groups,
        threads,
        Slots::new().rw("x").read("gate"),
    )
}

/// `Stream::split_q_gate` (`fllm_split_q_gate_f32`).
pub fn plan_split_q_gate(
    rows: usize,
    heads: usize,
    dk: usize,
    per_head: bool,
) -> Result<OpPlan, BackendError> {
    if rows * heads * dk == 0 {
        return Err(invalid("split_q_gate of an empty tensor"));
    }
    let tx = round_up(dk, 32).min(256);
    plan(
        "fllm_split_q_gate_f32",
        Params::new()
            .u("rows", rows)
            .u("heads", heads)
            .u("dk", dk)
            .b("per_head", per_head),
        [grid_dim(div_up(dk, tx))?, grid_dim(heads)?, grid_dim(rows)?],
        [tx as u32, 1, 1],
        Slots::new().read("projected").write("q").write("gate"),
    )
}

/// `Stream::argmax_rows` (`fllm_argmax_f32`): one int32 per row in `out`; the lowest index wins a tie and a NaN never wins.
pub fn plan_argmax(rows: usize, vocab: usize) -> Result<OpPlan, BackendError> {
    if rows == 0 || vocab == 0 {
        return Err(invalid("argmax of an empty tensor"));
    }
    plan(
        "fllm_argmax_f32",
        Params::new().u("vocab", vocab).u("rows", rows),
        [grid_dim(rows)?, 1, 1],
        [vocab.min(512).next_power_of_two().max(32) as u32, 1, 1],
        Slots::new().read("logits").write("out"),
    )
}
