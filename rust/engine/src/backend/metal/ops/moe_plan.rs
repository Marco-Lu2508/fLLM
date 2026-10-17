//! Dispatch planning of the mixture-of-experts operations of the metal backend: routing, expert matrix-vector products, combination, row gather /
//! scatter. Owner: package 6c (kernels `moe_topk`, `moe_linear_<fmt>`, `moe_combine`, `moe_gather_rows`, `moe_scatter_add_rows` in `rust/engine/metal`).
//!
//! Every `plan_*` function turns the arguments of one [`Stream`](crate::backend::Stream) operation into [`OpPlan`]s: kernel name, the bytes of the
//! constants struct (bound with `setBytes` at buffer [`CONSTANTS_BUFFER_INDEX`]), threadgroup grid, threads per threadgroup and the buffer slots
//! (index, name, access, source). Nothing here touches Metal; the stream (package 7) resolves the slot names to buffers, binds, dispatches and relies on
//! the declared accesses for its barrier tracking. The kernel side of every plan (constants struct, buffer order, threads) is documented in the
//! `// FLLM-KERNEL:` header of each `.metal` file and in `rust/engine/metal/INTERFACE_RECURRENT.md`; the unit tests compare every plan with
//! those headers, so the plans cannot drift from the sources (the prose of the interface file can).
//!
//! Level of verification: **L0** (compiles, unit-tested on the CPU; no kernel has ever run on a device).

use super::{Access, BufferSlot, OpPlan, SlotSource};
use crate::backend::{BackendError, QuantType};
use std::sync::Arc;

/// Buffer index at which every kernel of this package expects its constants struct (`setBytes`); the buffer slots follow from index 1 on, in the order of
/// the slot list.
pub const CONSTANTS_BUFFER_INDEX: u32 = 0;
/// Most experts `moe_topk` handles per row (threadgroup memory of the logits).
pub const MAX_EXPERTS: usize = 2048;
/// Most selected experts per row.
pub const MAX_TOP_K: usize = 32;
/// Threads per threadgroup of the element-wise kernels (grid-stride loops over `threads_per_grid`).
pub const ELEMENTWISE_THREADS: u32 = 256;
/// Threadgroups an element-wise kernel is launched with at most (the kernels loop).
pub const MAX_ELEMENTWISE_GROUPS: usize = 65535;
/// Alignment of the experts inside a weight arena (bytes; `AddressTable` / `Arena` of `buffers.rs` align to 256 as well).
pub const ARENA_ALIGN: usize = 256;

// ---------------------------------------------------------------------------------------------------------------------------------------------
// shared builders (also used by `recurrent_plan`)
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// Builder of the constants struct: 4-byte little-endian scalars in declaration order, no padding (the MSL structs consist of `uint` / `float` only).
#[derive(Default)]
pub(crate) struct Consts {
    bytes: Vec<u8>,
    fields: Vec<(&'static str, char)>,
    error: Option<String>,
}

impl Consts {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// An unsigned 32-bit field.
    pub(crate) fn u(mut self, name: &'static str, value: usize) -> Self {
        match u32::try_from(value) {
            Ok(v) => self.bytes.extend_from_slice(&v.to_le_bytes()),
            Err(_) => {
                self.error.get_or_insert_with(|| {
                    format!("constant {name} = {value} does not fit 32 bits")
                });
                self.bytes.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        self.fields.push((name, 'u'));
        self
    }

    /// A boolean as 0 / 1.
    pub(crate) fn b(self, name: &'static str, value: bool) -> Self {
        self.u(name, usize::from(value))
    }

    /// A 32-bit float field.
    pub(crate) fn f(mut self, name: &'static str, value: f32) -> Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self.fields.push((name, 'f'));
        self
    }
}

/// Buffer slots in binding order: the first one gets index 1 (index 0 is the constants struct).
#[derive(Default)]
pub(crate) struct Slots(Vec<BufferSlot>);

impl Slots {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn add(mut self, name: &'static str, access: Access, source: SlotSource) -> Self {
        let index = self.0.len() as u32 + 1;
        self.0.push(make_slot(index, name, access, source));
        self
    }

    /// A buffer the kernel only reads.
    pub(crate) fn read(self, name: &'static str) -> Self {
        self.add(name, Access::Read, SlotSource::Buffer)
    }

    /// A buffer the kernel only writes.
    pub(crate) fn write(self, name: &'static str) -> Self {
        self.add(name, Access::Write, SlotSource::Buffer)
    }

    /// A buffer the kernel reads and writes.
    pub(crate) fn rw(self, name: &'static str) -> Self {
        self.add(name, Access::ReadWrite, SlotSource::Buffer)
    }

    /// A read-only buffer when `present`, a dummy (any valid buffer of at least 16 bytes) otherwise.
    pub(crate) fn read_opt(self, name: &'static str, present: bool) -> Self {
        let source = if present {
            SlotSource::Buffer
        } else {
            SlotSource::Dummy
        };
        self.add(name, Access::Read, source)
    }

    /// A small host array (little endian) that the stream uploads for this dispatch.
    pub(crate) fn host(self, name: &'static str, data: Arc<Vec<u8>>) -> Self {
        self.add(name, Access::Read, SlotSource::Host(data))
    }

    /// Uninitialised scratch storage; slots with the same `id` of the plans of ONE operation share one buffer (its size is the maximum).
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

    /// Any valid buffer of at least 16 bytes (an input that is absent and not read).
    pub(crate) fn dummy(self, name: &'static str, access: Access) -> Self {
        self.add(name, access, SlotSource::Dummy)
    }

    /// The state buffer of run `n` of a `delta_layer` / Mamba call.
    pub(crate) fn run_state(self, name: &'static str, run: usize) -> Self {
        self.add(name, Access::ReadWrite, SlotSource::RunState(run))
    }
}

/// The only place that constructs a [`BufferSlot`] (the type belongs to `ops.rs` of package 6b; adapt here if its fields differ).
fn make_slot(index: u32, name: &'static str, access: Access, source: SlotSource) -> BufferSlot {
    BufferSlot {
        index,
        name,
        access,
        source,
    }
}

/// Assemble an [`OpPlan`]; every grid component must be at least 1.
pub(crate) fn plan(
    kernel_name: &'static str,
    consts: Consts,
    threadgroups: [u32; 3],
    threads_per_group: [u32; 3],
    slots: Slots,
) -> Result<OpPlan, BackendError> {
    if let Some(e) = consts.error {
        return Err(invalid(e));
    }
    if threadgroups.contains(&0) || threads_per_group.contains(&0) {
        return Err(invalid(format!("{kernel_name}: empty dispatch grid")));
    }
    #[cfg(test)]
    testkit::remember_constants(kernel_name, &consts.fields);
    Ok(OpPlan {
        kernel_name: kernel_name.into(),
        constants: consts.bytes,
        threadgroups,
        threads_per_group,
        slots: slots.0,
    })
}

pub(crate) fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

/// `ceil(a / b)`.
pub(crate) fn div_up(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// A grid component as `u32`.
pub(crate) fn dim(value: usize, what: &str) -> Result<u32, BackendError> {
    u32::try_from(value).map_err(|_| {
        BackendError::UnsupportedOperation(format!("{what} = {value} exceeds the dispatch grid"))
    })
}

/// One-dimensional grid for a grid-stride kernel of `wg` threads per threadgroup: `ceil(items / wg)` threadgroups, at most [`MAX_ELEMENTWISE_GROUPS`].
pub fn groups_1d(items: usize, wg: usize) -> [u32; 3] {
    [
        div_up(items, wg).clamp(1, MAX_ELEMENTWISE_GROUPS) as u32,
        1,
        1,
    ]
}

/// `values` as little-endian `i32` (row indices of gather / scatter).
pub fn pack_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `values` as little-endian `f32`.
pub fn pack_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `values` as little-endian `u64` (byte offsets / GPU addresses of the expert tables).
pub fn pack_u64(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// routing
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// `Stream::moe_topk` (`moe_topk`): softmax over `experts` logits per row, the `k` best with ties to the lower index; `ids` (int32) and `weights` hold
/// `rows x k` entries. `normalize` renormalises over the selected experts, `scale` multiplies the weights. Slots: `logits`, `ids`, `weights`.
pub fn plan_moe_topk(
    rows: usize,
    experts: usize,
    k: usize,
    normalize: bool,
    scale: f32,
) -> Result<OpPlan, BackendError> {
    if rows == 0 || k == 0 || k > experts {
        return Err(invalid(format!(
            "moe_topk of {rows} rows, {experts} experts, k = {k}"
        )));
    }
    if experts > MAX_EXPERTS || k > MAX_TOP_K {
        return Err(BackendError::UnsupportedOperation(format!(
            "moe_topk supports at most {MAX_EXPERTS} experts and k = {MAX_TOP_K} (got {experts}, {k})"
        )));
    }
    let consts = Consts::new()
        .u("rows", rows)
        .u("experts", experts)
        .u("k", k)
        .b("normalize_scores", normalize)
        .f("scale", scale);
    plan(
        "moe_topk",
        consts,
        [dim(rows, "moe_topk rows")?, 1, 1],
        [128, 1, 1],
        Slots::new().read("logits").write("ids").write("weights"),
    )
}

/// `Stream::moe_route` (same kernel as `moe_topk`: normalised, scale 1). The number of rows comes from the size of the logits buffer. The slots are
/// named `logits`, `indices`, `weights` like the arguments of `moe_route` (the kernel order is logits, ids, weights).
pub fn plan_moe_route(
    rows: usize,
    num_experts: usize,
    top_k: usize,
) -> Result<OpPlan, BackendError> {
    let mut p = plan_moe_topk(rows, num_experts, top_k, true, 1.0)?;
    p.slots[1].name = "indices";
    Ok(p)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// expert matrix-vector products
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// How `moe_linear` reaches the weights of the experts of one projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertAccess {
    /// All experts live in ONE buffer (`arena`): `table[e]` = byte offset of expert `e` in it (`ulong`), `bias_table[e]` = byte offset of its bias vector
    /// in `bias_arena`. The buffer is bound as a whole, so it is limited by `maxBufferLength`; this is the form the weight upload of package 6a / 7 should
    /// produce when it carves the experts of a projection out of one allocation.
    Arena,
    /// `table[e]` / `bias_table[e]` = absolute 64-bit GPU address of the expert's weight / bias buffer (`AddressTable::build` of `buffers.rs`, the stream's
    /// `upload_pointer_table`); the stream must make the expert buffers resident (`builder.table(..)`). `arena` and `bias_arena` are dummies.
    Address,
}

/// The kernel of one weight format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeFormat {
    /// Kernel name (`moe_linear_<tag>`).
    pub kernel: &'static str,
    /// Weights per block.
    pub block_elems: usize,
    /// Bytes per block in the raw GGUF layout.
    pub block_bytes: usize,
}

/// Formats `moe_linear` decodes (raw GGUF blocks; the float formats, Q2_K / Q3_K, the other IQ formats, NVFP4, Q4Z and FP8 are not covered).
pub fn moe_format(quant: QuantType) -> Option<MoeFormat> {
    let (kernel, block_elems, block_bytes) = match quant {
        QuantType::Q4_0 => ("moe_linear_q4_0", 32, 18),
        QuantType::Q4_1 => ("moe_linear_q4_1", 32, 20),
        QuantType::Q5_0 => ("moe_linear_q5_0", 32, 22),
        QuantType::Q5_1 => ("moe_linear_q5_1", 32, 24),
        QuantType::Q8_0 => ("moe_linear_q8_0", 32, 34),
        QuantType::IQ4_NL => ("moe_linear_iq4_nl", 32, 18),
        QuantType::Mxfp4 => ("moe_linear_mxfp4", 32, 17),
        QuantType::Q4_K => ("moe_linear_q4_k", 256, 144),
        QuantType::Q5_K => ("moe_linear_q5_k", 256, 176),
        QuantType::Q6_K => ("moe_linear_q6_k", 256, 210),
        _ => return None,
    };
    Some(MoeFormat {
        kernel,
        block_elems,
        block_bytes,
    })
}

/// Does `moe_linear` take experts of this format?
pub fn moe_linear_supported(quant: QuantType) -> bool {
    moe_format(quant).is_some()
}

/// Bytes of one expert of `rows x cols` weights (blocks back to back, no padding).
pub fn expert_bytes(quant: QuantType, rows: usize, cols: usize) -> Result<usize, BackendError> {
    let f = moe_format(quant).ok_or(BackendError::UnsupportedQuant(quant))?;
    if cols == 0 || !cols.is_multiple_of(f.block_elems) {
        return Err(BackendError::UnsupportedQuant(quant));
    }
    rows.checked_mul(cols / f.block_elems)
        .and_then(|b| b.checked_mul(f.block_bytes))
        .ok_or_else(|| invalid("expert size overflows"))
}

/// Offsets (bytes) of consecutive blocks of `sizes` inside one arena, each starting at a multiple of `align`, and the total size. Used for the weight
/// arena (`align` = [`ARENA_ALIGN`]) and for the bias arena (`align` = 4).
pub fn arena_offsets(sizes: &[usize], align: usize) -> (Vec<u64>, usize) {
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut at = 0usize;
    for &s in sizes {
        at = at.next_multiple_of(align.max(1));
        offsets.push(at as u64);
        at += s;
    }
    (offsets, at.next_multiple_of(align.max(1)))
}

/// One `moe_linear` call: `out[slot] = W[ids[slot]] x[slot / x_div]` for `slots` slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeLinear {
    pub quant: QuantType,
    /// Output features and input features of every expert.
    pub rows: usize,
    pub cols: usize,
    pub slots: usize,
    pub x_div: usize,
    pub n_experts: usize,
    pub has_bias: bool,
    pub access: ExpertAccess,
}

/// `Stream::moe_linear` (`moe_linear_<fmt>`). Slots in kernel order: `table` (`ulong[n_experts]`), `bias_table` (dummy without bias), `ids` (int32,
/// `slots` entries), `x` (f32, `ceil(slots / x_div)` rows of `cols`), `out` (f32, `slots x rows`), `arena`, `bias_arena` (dummies for
/// [`ExpertAccess::Address`]; `bias_arena` is a dummy without bias).
pub fn plan_moe_linear(c: &MoeLinear) -> Result<OpPlan, BackendError> {
    let format = moe_format(c.quant).ok_or(BackendError::UnsupportedQuant(c.quant))?;
    if c.rows == 0 || c.cols == 0 || c.slots == 0 || c.x_div == 0 || c.n_experts == 0 {
        return Err(invalid("moe_linear with an empty dimension"));
    }
    if !c.cols.is_multiple_of(format.block_elems) {
        return Err(BackendError::UnsupportedQuant(c.quant));
    }
    let arena = c.access == ExpertAccess::Arena;
    let flags = usize::from(c.has_bias) | (usize::from(!arena) << 1);
    let consts = Consts::new()
        .u("rows", c.rows)
        .u("cols", c.cols)
        .u("slots", c.slots)
        .u("x_div", c.x_div)
        .u("n_experts", c.n_experts)
        .u("flags", flags)
        .u("nb", c.cols / format.block_elems);
    let slots = Slots::new()
        .read("table")
        .read_opt("bias_table", c.has_bias)
        .read("ids")
        .read("x")
        .write("out")
        .read_opt("arena", arena)
        .read_opt("bias_arena", arena && c.has_bias);
    plan(
        format.kernel,
        consts,
        [
            dim(c.rows, "moe_linear rows")?,
            dim(c.slots, "moe_linear slots")?,
            1,
        ],
        [64, 1, 1],
        slots,
    )
}

/// `Stream::moe_combine` (`moe_combine`). Slots: `out`, `y`, `weights`.
pub fn plan_moe_combine(tokens: usize, k: usize, hidden: usize) -> Result<OpPlan, BackendError> {
    if tokens == 0 || k == 0 || hidden == 0 {
        return Err(invalid("moe_combine of an empty tensor"));
    }
    let consts = Consts::new()
        .u("tokens", tokens)
        .u("k", k)
        .u("hidden", hidden);
    plan(
        "moe_combine",
        consts,
        groups_1d(tokens * hidden, ELEMENTWISE_THREADS as usize),
        [ELEMENTWISE_THREADS, 1, 1],
        Slots::new().write("out").read("y").read("weights"),
    )
}

/// `Backend::moe_expert_gemm` / `Stream` equivalent: `out[r] = sum_j weights[r * top_k + j] * (W[indices[r * top_k + j]] x[r])` for experts of one shape and
/// format: `moe_linear` (`slots = rows * top_k`, `x_div = top_k`, no bias, the `y` slot is scratch id 0 of `rows * top_k * c.rows` floats) followed by
/// `moe_combine`. The slots are renamed after the arguments of `moe_expert_gemm`: `expert_indices` (the `ids` slot) and `expert_weights`. The `slots`,
/// `x_div` and `has_bias` of `c` are ignored.
pub fn plan_moe_expert_gemm(
    c: &MoeLinear,
    rows: usize,
    top_k: usize,
) -> Result<Vec<OpPlan>, BackendError> {
    if rows == 0 || top_k == 0 {
        return Err(invalid("moe_expert_gemm of an empty batch"));
    }
    let lin = MoeLinear {
        slots: rows * top_k,
        x_div: top_k,
        has_bias: false,
        ..*c
    };
    let y_bytes = rows * top_k * c.rows * 4;
    let mut first = plan_moe_linear(&lin)?;
    first.slots[2].name = "expert_indices";
    first.slots[4] = make_slot(
        5,
        "y",
        Access::Write,
        SlotSource::Scratch {
            id: 0,
            bytes: y_bytes.max(16),
        },
    );
    let mut second = plan_moe_combine(rows, top_k, c.rows)?;
    second.slots[1] = make_slot(
        2,
        "y",
        Access::Read,
        SlotSource::Scratch {
            id: 0,
            bytes: y_bytes.max(16),
        },
    );
    second.slots[2].name = "expert_weights";
    Ok(vec![first, second])
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// row gather / scatter
// ---------------------------------------------------------------------------------------------------------------------------------------------

fn check_rows(idx: &[i32], weights: Option<&[f32]>, row_len: usize) -> Result<(), BackendError> {
    if idx.is_empty() || row_len == 0 {
        return Err(invalid("row gather / scatter of an empty tensor"));
    }
    if weights.is_some_and(|w| w.len() != idx.len()) {
        return Err(invalid("gather / scatter weights do not match the indices"));
    }
    if idx.iter().any(|&i| i < 0) {
        return Err(invalid("negative row index"));
    }
    Ok(())
}

/// Slots 3 and 4 of the gather / scatter kernels.
fn index_slots(slots: Slots, idx: &[i32], weights: Option<&[f32]>) -> Slots {
    let slots = slots.host("idx", Arc::new(pack_i32(idx)));
    match weights {
        Some(w) => slots.host("weights", Arc::new(pack_f32(w))),
        None => slots.dummy("weights", Access::Read),
    }
}

/// `Stream::gather_rows` (`moe_gather_rows`): `dst[i] = src[idx[i]] * weights[i]`. Slots: `dst`, `src`, `idx` (host), `weights` (host, dummy without).
pub fn plan_gather_rows(
    idx: &[i32],
    weights: Option<&[f32]>,
    row_len: usize,
) -> Result<OpPlan, BackendError> {
    check_rows(idx, weights, row_len)?;
    let consts = Consts::new()
        .u("n", idx.len())
        .u("row_len", row_len)
        .b("has_weights", weights.is_some());
    plan(
        "moe_gather_rows",
        consts,
        groups_1d(idx.len() * row_len, ELEMENTWISE_THREADS as usize),
        [ELEMENTWISE_THREADS, 1, 1],
        index_slots(Slots::new().write("dst").read("src"), idx, weights),
    )
}

/// `Stream::scatter_add_rows` (`moe_scatter_add_rows`): `dst[idx[i]] += src[i] * weights[i]`; the indices of one call are distinct (Stream contract).
/// Slots: `dst` (read-write), `src`, `idx` (host), `weights` (host, dummy without).
pub fn plan_scatter_add_rows(
    idx: &[i32],
    weights: Option<&[f32]>,
    row_len: usize,
) -> Result<OpPlan, BackendError> {
    check_rows(idx, weights, row_len)?;
    let consts = Consts::new()
        .u("n", idx.len())
        .u("row_len", row_len)
        .b("has_weights", weights.is_some());
    plan(
        "moe_scatter_add_rows",
        consts,
        groups_1d(idx.len() * row_len, ELEMENTWISE_THREADS as usize),
        [ELEMENTWISE_THREADS, 1, 1],
        index_slots(Slots::new().rw("dst").read("src"), idx, weights),
    )
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// test support: headers of the .metal sources
// ---------------------------------------------------------------------------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod testkit {
    //! Parser of the `// FLLM-KERNEL:` headers of the sources and the checks that compare a plan with them.
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    thread_local! {
        static CONSTANTS: RefCell<HashMap<&'static str, Vec<(&'static str, char)>>> = RefCell::new(HashMap::new());
    }

    pub(crate) fn remember_constants(kernel: &'static str, fields: &[(&'static str, char)]) {
        CONSTANTS.with(|c| {
            c.borrow_mut().insert(kernel, fields.to_vec());
        });
    }

    /// The constants (name, `u`/`f`) of the plan of `kernel` that was built last on this thread.
    pub(crate) fn last_constants(kernel: &str) -> Vec<(&'static str, char)> {
        CONSTANTS.with(|c| c.borrow().get(kernel).cloned().unwrap_or_default())
    }

    /// All sources of the package (file name, text).
    pub(crate) const SOURCES: &[(&str, &str)] = &[
        (
            "arch_ops.metal",
            include_str!("../../../../metal/arch_ops.metal"),
        ),
        (
            "moe_route.metal",
            include_str!("../../../../metal/moe_route.metal"),
        ),
        (
            "moe_linear.metal",
            include_str!("../../../../metal/moe_linear.metal"),
        ),
        (
            "moe_combine.metal",
            include_str!("../../../../metal/moe_combine.metal"),
        ),
        (
            "moe_rows.metal",
            include_str!("../../../../metal/moe_rows.metal"),
        ),
        (
            "gate_attention.metal",
            include_str!("../../../../metal/gate_attention.metal"),
        ),
        (
            "delta_conv.metal",
            include_str!("../../../../metal/delta_conv.metal"),
        ),
        (
            "delta_rec.metal",
            include_str!("../../../../metal/delta_rec.metal"),
        ),
        (
            "mamba_conv.metal",
            include_str!("../../../../metal/mamba_conv.metal"),
        ),
        (
            "mamba_scan1.metal",
            include_str!("../../../../metal/mamba_scan1.metal"),
        ),
        (
            "mamba_scan2.metal",
            include_str!("../../../../metal/mamba_scan2.metal"),
        ),
        (
            "mamba_norm.metal",
            include_str!("../../../../metal/mamba_norm.metal"),
        ),
    ];

    #[derive(Debug)]
    pub(crate) struct Header {
        pub(crate) file: &'static str,
        pub(crate) names: Vec<String>,
        pub(crate) constants: Vec<String>,
        /// (index, name, access letters)
        pub(crate) buffers: Vec<(u32, String, String)>,
        pub(crate) threads: [u32; 3],
        /// `struct ..._params` bodies of the file as (name, kind) lists, to compare the constants with the real MSL struct.
        pub(crate) structs: Vec<Vec<(String, char)>>,
        /// Kernel names the file defines (`host_name("..")` of the instantiations and `kernel void name(`).
        pub(crate) kernels: Vec<String>,
    }

    fn parse_structs(text: &str) -> Vec<Vec<(String, char)>> {
        let mut out = Vec::new();
        let mut rest = text;
        while let Some(at) = rest.find("struct ") {
            rest = &rest[at + 7..];
            let Some(open) = rest.find('{') else { break };
            let name = rest[..open].trim();
            if !name.ends_with("_params") {
                continue;
            }
            let Some(close) = rest.find('}') else { break };
            let fields = rest[open + 1..close]
                .split(';')
                .filter_map(|decl| {
                    let mut it = decl.split_whitespace();
                    let kind = it.next()?;
                    let field = it.next()?;
                    let k = match kind {
                        "uint" | "int" => 'u',
                        "float" => 'f',
                        other => panic!("unexpected field type {other} in {name}"),
                    };
                    Some((field.to_string(), k))
                })
                .collect();
            out.push(fields);
            rest = &rest[close..];
        }
        out
    }

    fn parse_kernels(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut rest = text;
        while let Some(at) = rest.find("host_name(\"") {
            rest = &rest[at + 11..];
            if let Some(end) = rest.find('"') {
                out.push(rest[..end].to_string());
            }
        }
        for line in text.lines() {
            let line = line.trim_start();
            if let Some(after) = line.strip_prefix("kernel void ") {
                if let Some(end) = after.find('(') {
                    out.push(after[..end].to_string());
                }
            }
        }
        out
    }

    pub(crate) fn headers() -> Vec<Header> {
        let mut out = Vec::new();
        for &(file, text) in SOURCES {
            let structs = parse_structs(text);
            let kernels = parse_kernels(text);
            for line in text
                .lines()
                .filter_map(|l| l.strip_prefix("// FLLM-KERNEL:"))
            {
                let parts: Vec<&str> = line.split('|').map(str::trim).collect();
                let names = parts[0].split(',').map(|s| s.trim().to_string()).collect();
                let field = |key: &str| -> &str {
                    parts
                        .iter()
                        .find_map(|p| p.strip_prefix(key))
                        .unwrap_or_else(|| panic!("{file}: header without `{key}`"))
                        .trim()
                };
                let constants = field("constants:")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                let buffers = field("buffers:")
                    .split("),")
                    .map(|b| {
                        let b = b.trim().trim_end_matches(')');
                        let (index, rest) = b.split_once(' ').expect("buffer index");
                        let (name, access) = rest.split_once('(').expect("buffer access");
                        (
                            index.parse().expect("index"),
                            name.trim().to_string(),
                            access.to_string(),
                        )
                    })
                    .collect();
                let t: Vec<u32> = field("threads:")
                    .split(',')
                    .map(|v| v.trim().parse().expect("threads"))
                    .collect();
                out.push(Header {
                    file,
                    names,
                    constants,
                    buffers,
                    threads: [t[0], t[1], t[2]],
                    structs: structs.clone(),
                    kernels: kernels.clone(),
                });
            }
        }
        out
    }

    fn matches(pattern: &str, name: &str) -> bool {
        match pattern.split_once('<') {
            Some((prefix, _)) => name.starts_with(prefix) && name.len() > prefix.len(),
            None => pattern == name,
        }
    }

    pub(crate) fn find<'h>(headers: &'h [Header], kernel: &str) -> &'h Header {
        headers
            .iter()
            .find(|h| h.names.iter().any(|p| matches(p, kernel)))
            .unwrap_or_else(|| panic!("no FLLM-KERNEL header for kernel {kernel}"))
    }

    fn letters(a: Access) -> &'static str {
        match a {
            Access::Read => "r",
            Access::Write => "w",
            Access::ReadWrite => "rw",
        }
    }

    /// Compare `plan` (the constants are those of the plan of this kernel built last on this thread) with the header of its kernel: constants (names, kinds, byte count and the MSL struct), buffer slots
    /// (index, name, access), threads per threadgroup and that the kernel exists in the source. `renamed` lists slots the plan renames on purpose
    /// (index, plan name).
    pub(crate) fn check_plan(plan: &OpPlan, renamed: &[(u32, &str)]) {
        let headers = headers();
        let name = plan.kernel_name.to_string();
        let h = find(&headers, &name);
        assert!(
            h.kernels.contains(&name),
            "{}: kernel {name} not defined",
            h.file
        );
        let consts = last_constants(&name);
        let names: Vec<&str> = consts.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            h.constants.iter().map(String::as_str).collect::<Vec<_>>(),
            "{name}: constants differ from the header"
        );
        assert_eq!(
            plan.constants.len(),
            4 * consts.len(),
            "{name}: constants byte count"
        );
        let kinds: Vec<(String, char)> = consts.iter().map(|(n, k)| (n.to_string(), *k)).collect();
        assert!(
            h.structs.contains(&kinds),
            "{name}: no MSL struct in {} has the fields {kinds:?}",
            h.file
        );
        assert_eq!(plan.slots.len(), h.buffers.len(), "{name}: slot count");
        for (slot, (index, bname, access)) in plan.slots.iter().zip(&h.buffers) {
            assert_eq!(slot.index, *index, "{name}: slot index of {bname}");
            let expected = renamed
                .iter()
                .find(|(i, _)| i == index)
                .map_or(bname.as_str(), |(_, n)| n);
            assert_eq!(slot.name, expected, "{name}: slot name at {index}");
            assert_eq!(letters(slot.access), access, "{name}: access of {bname}");
        }
        assert_eq!(
            plan.threads_per_group, h.threads,
            "{name}: threads per threadgroup"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::{check_plan, last_constants};
    use super::*;

    fn word(p: &OpPlan, i: usize) -> u32 {
        u32::from_le_bytes(p.constants[4 * i..4 * i + 4].try_into().unwrap())
    }

    #[test]
    fn topk_plan_matches_the_kernel_header() {
        let p = plan_moe_topk(7, 128, 8, true, 2.5).unwrap();
        check_plan(&p, &[]);
        assert_eq!(p.threadgroups, [7, 1, 1]);
        assert_eq!(
            (word(&p, 0), word(&p, 1), word(&p, 2), word(&p, 3)),
            (7, 128, 8, 1)
        );
        assert_eq!(f32::from_bits(word(&p, 4)), 2.5);
        let r = plan_moe_route(3, 64, 4).unwrap();
        assert_eq!(r.kernel_name, "moe_topk");
        assert_eq!(r.slots[1].name, "indices");
        assert_eq!(word(&r, 3), 1);
        assert_eq!(f32::from_bits(word(&r, 4)), 1.0);
    }

    #[test]
    fn topk_rejects_what_the_kernel_cannot_do() {
        assert!(plan_moe_topk(0, 8, 2, true, 1.0).is_err());
        assert!(plan_moe_topk(1, 8, 9, true, 1.0).is_err());
        assert!(plan_moe_topk(1, MAX_EXPERTS + 1, 2, true, 1.0).is_err());
        assert!(plan_moe_topk(1, 64, MAX_TOP_K + 1, true, 1.0).is_err());
        assert!(plan_moe_topk(1, MAX_EXPERTS, MAX_TOP_K, false, 1.0).is_ok());
    }

    #[test]
    fn every_format_has_a_kernel_and_a_consistent_block() {
        let formats = [
            (QuantType::Q4_0, "moe_linear_q4_0", 32, 18),
            (QuantType::Q4_1, "moe_linear_q4_1", 32, 20),
            (QuantType::Q5_0, "moe_linear_q5_0", 32, 22),
            (QuantType::Q5_1, "moe_linear_q5_1", 32, 24),
            (QuantType::Q8_0, "moe_linear_q8_0", 32, 34),
            (QuantType::IQ4_NL, "moe_linear_iq4_nl", 32, 18),
            (QuantType::Mxfp4, "moe_linear_mxfp4", 32, 17),
            (QuantType::Q4_K, "moe_linear_q4_k", 256, 144),
            (QuantType::Q5_K, "moe_linear_q5_k", 256, 176),
            (QuantType::Q6_K, "moe_linear_q6_k", 256, 210),
        ];
        for (quant, kernel, elems, bytes) in formats {
            let f = moe_format(quant).unwrap();
            assert_eq!(
                (f.kernel, f.block_elems, f.block_bytes),
                (kernel, elems, bytes)
            );
            assert_eq!(
                f.block_elems,
                quant.block_size(),
                "{quant:?}: block size of the backend"
            );
            // ... and the block geometry of the GGUF probe (the table the loader uses)
            let ty = fllm_gguf_probe::quant_types::by_id(quant.to_gguf_type_id())
                .expect("known GGUF type");
            assert_eq!(
                (ty.block_elements, ty.bytes_per_block),
                (elems, bytes),
                "{quant:?}: GGUF block geometry"
            );
            let mut c = MoeLinear {
                quant,
                rows: 4,
                cols: elems * 2,
                slots: 3,
                x_div: 1,
                n_experts: 8,
                has_bias: false,
                access: ExpertAccess::Arena,
            };
            let p = plan_moe_linear(&c).unwrap();
            check_plan(&p, &[]);
            assert_eq!(p.kernel_name, kernel);
            c.access = ExpertAccess::Address;
            c.has_bias = true;
            check_plan(&plan_moe_linear(&c).unwrap(), &[]);
            assert_eq!(expert_bytes(quant, 4, elems * 2).unwrap(), 4 * 2 * bytes);
        }
        for q in [
            QuantType::F32,
            QuantType::F16,
            QuantType::Q2_K,
            QuantType::Q3_K,
            QuantType::NvFp4,
            QuantType::Q4Z32,
            QuantType::Q4_K_Marlin,
        ] {
            assert!(!moe_linear_supported(q), "{q:?}");
            assert!(matches!(
                expert_bytes(q, 1, 256),
                Err(BackendError::UnsupportedQuant(_))
            ));
        }
    }

    #[test]
    fn linear_flags_and_dummies() {
        let mut c = MoeLinear {
            quant: QuantType::Q4_K,
            rows: 768,
            cols: 2048,
            slots: 8,
            x_div: 8,
            n_experts: 128,
            has_bias: false,
            access: ExpertAccess::Arena,
        };
        let p = plan_moe_linear(&c).unwrap();
        assert_eq!(word(&p, 5), 0);
        assert_eq!(word(&p, 6), 8, "nb = 2048 / 256");
        assert_eq!(p.threadgroups, [768, 8, 1]);
        assert_eq!(p.slots[1].source, SlotSource::Dummy, "no bias table");
        assert_eq!(p.slots[5].source, SlotSource::Buffer, "arena");
        assert_eq!(p.slots[6].source, SlotSource::Dummy, "no bias arena");
        c.has_bias = true;
        let p = plan_moe_linear(&c).unwrap();
        assert_eq!(word(&p, 5), 1);
        assert_eq!(p.slots[1].source, SlotSource::Buffer);
        assert_eq!(p.slots[6].source, SlotSource::Buffer);
        c.access = ExpertAccess::Address;
        let p = plan_moe_linear(&c).unwrap();
        assert_eq!(word(&p, 5), 3, "bias + address table");
        assert_eq!(p.slots[5].source, SlotSource::Dummy);
        assert_eq!(p.slots[6].source, SlotSource::Dummy);
        assert!(
            plan_moe_linear(&MoeLinear { cols: 2000, ..c }).is_err(),
            "cols is not a multiple of 256"
        );
        assert!(plan_moe_linear(&MoeLinear {
            quant: QuantType::F32,
            ..c
        })
        .is_err());
        assert!(plan_moe_linear(&MoeLinear { slots: 0, ..c }).is_err());
    }

    #[test]
    fn arena_offsets_are_aligned_and_disjoint() {
        let sizes = [100usize, 256, 1, 4096];
        let (offsets, total) = arena_offsets(&sizes, ARENA_ALIGN);
        assert_eq!(offsets, vec![0, 256, 512, 768]);
        assert_eq!(total, 768 + 4096);
        for (i, w) in offsets.windows(2).enumerate() {
            assert!(w[0] as usize + sizes[i] <= w[1] as usize);
            assert_eq!(w[1] % ARENA_ALIGN as u64, 0);
        }
        let (bias, total) = arena_offsets(&[16, 16, 16], 4);
        assert_eq!(bias, vec![0, 16, 32]);
        assert_eq!(total, 48);
        assert_eq!(
            pack_u64(&[1, 0x0102_0304_0506_0708]),
            vec![1, 0, 0, 0, 0, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1]
        );
    }

    #[test]
    fn combine_and_rows_plans() {
        let p = plan_moe_combine(4, 8, 2048).unwrap();
        check_plan(&p, &[]);
        assert_eq!(p.threadgroups, [32, 1, 1]);
        assert_eq!(
            plan_moe_combine(5000, 8, 4096).unwrap().threadgroups,
            [MAX_ELEMENTWISE_GROUPS as u32, 1, 1]
        );
        let g = plan_gather_rows(&[3, 1, 2], Some(&[0.5, 1.0, 2.0]), 16).unwrap();
        check_plan(&g, &[]);
        assert_eq!(
            g.slots[2].source,
            SlotSource::Host(Arc::new(pack_i32(&[3, 1, 2])))
        );
        assert_eq!(
            g.slots[3].source,
            SlotSource::Host(Arc::new(pack_f32(&[0.5, 1.0, 2.0])))
        );
        assert_eq!(word(&g, 2), 1);
        let s = plan_scatter_add_rows(&[0, 5], None, 8).unwrap();
        check_plan(&s, &[]);
        assert_eq!(s.slots[0].access, Access::ReadWrite);
        assert_eq!(s.slots[3].source, SlotSource::Dummy);
        assert_eq!(word(&s, 2), 0);
        assert!(plan_gather_rows(&[-1], None, 4).is_err());
        assert!(plan_gather_rows(&[1, 2], Some(&[1.0]), 4).is_err());
        assert!(plan_scatter_add_rows(&[], None, 4).is_err());
    }

    #[test]
    fn expert_gemm_is_linear_plus_combine() {
        let c = MoeLinear {
            quant: QuantType::Q6_K,
            rows: 512,
            cols: 1024,
            slots: 0,
            x_div: 0,
            n_experts: 64,
            has_bias: true,
            access: ExpertAccess::Arena,
        };
        let plans = plan_moe_expert_gemm(&c, 5, 4).unwrap();
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].kernel_name, "moe_linear_q6_k");
        assert_eq!(word(&plans[0], 2), 20, "slots = rows * top_k");
        assert_eq!(word(&plans[0], 3), 4, "x_div = top_k");
        assert_eq!(word(&plans[0], 5), 0, "no bias");
        assert_eq!(plans[0].slots[2].name, "expert_indices");
        assert_eq!(plans[0].slots[4].name, "y");
        assert_eq!(plans[1].kernel_name, "moe_combine");
        assert_eq!(plans[1].slots[1].source, plans[0].slots[4].source.clone());
        assert_eq!(plans[1].slots[1].access, Access::Read);
        assert_eq!(plans[1].slots[2].name, "expert_weights");
        let _ = last_constants("moe_combine");
    }
}
