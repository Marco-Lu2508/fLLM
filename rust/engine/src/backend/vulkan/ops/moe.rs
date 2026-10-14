//! Plans of the mixture-of-experts operations: routing, expert matrix-vector products, combination, row gather / scatter.
//! Shaders: `moe_topk`, `moe_linear_<fmt>_{arena,bda}`, `moe_combine`, `moe_gather_rows`, `moe_scatter_add_rows`.

use super::*;
use crate::backend::QuantType;

/// Most experts `moe_topk` handles per row (shared memory of the logits).
pub const MAX_EXPERTS: usize = 1024;
/// Most selected experts per row.
pub const MAX_TOP_K: usize = 32;

/// `Stream::moe_topk` (`moe_topk`): softmax over `experts` logits per row, the `k` best with ties to the lower index; `ids` (int32) and `weights` hold
/// `rows x k` entries. `normalize` renormalises over the selected experts, `scale` multiplies the weights.
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
        return Err(BackendError::UnsupportedOperation(format!("moe_topk supports at most {MAX_EXPERTS} experts and k = {MAX_TOP_K} (got {experts}, {k})")));
    }
    let push = Push::new()
        .u("rows", rows)
        .u("experts", experts)
        .u("k", k)
        .b("normalize_scores", normalize)
        .f("scale", scale);
    plan(
        "moe_topk",
        push,
        groups_rows(rows),
        Slots::new().read("logits").write("ids").write("weights"),
    )
}

/// `Stream::moe_route` (same shader as `moe_topk`: normalised, scale 1). The rows come from the size of the logits buffer.
pub fn plan_moe_route(
    rows: usize,
    num_experts: usize,
    top_k: usize,
) -> Result<OpPlan, BackendError> {
    let mut p = plan_moe_topk(rows, num_experts, top_k, true, 1.0)?;
    // moe_route's output order is (weights, indices); the shader's slots are (logits, ids, weights): the caller maps the names
    p.bindings[1].name = "indices";
    p.bindings[2].name = "weights";
    Ok(p)
}

/// How the weights of the experts of one projection are reached by `moe_linear`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertAccess {
    /// All experts live in ONE storage buffer (`weights`; `table[e]` = word offset of expert `e`, `bias_table[e]` = float offset of its bias in `biases`).
    Arena,
    /// `table[e]` / `bias_table[e]` are 64-bit device addresses (`uvec2`); needs `bufferDeviceAddress` and a buffer created with the device-address usage.
    BufferAddress,
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
    /// Word offsets of the planes 1..3 inside one expert in the planar device layout (identical for all experts).
    pub planes: [usize; 3],
}

fn moe_linear_shader(tag: &str, access: ExpertAccess) -> Option<&'static str> {
    macro_rules! names {
        ($($t:literal),*) => {
            match (tag, access) {
                $(
                    ($t, ExpertAccess::Arena) => Some(concat!("moe_linear_", $t, "_arena")),
                    ($t, ExpertAccess::BufferAddress) => Some(concat!("moe_linear_", $t, "_bda")),
                )*
                _ => None,
            }
        };
    }
    names!(
        "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "iq4_nl",
        "mxfp4"
    )
}

/// Formats `moe_linear` decodes (the quantized formats of the planar device layout; the float formats are not there).
pub fn moe_linear_supported(quant: QuantType) -> bool {
    matches!(format_tag(quant), Some((tag, qk)) if qk > 1 && moe_linear_shader(tag, ExpertAccess::Arena).is_some())
}

/// `Stream::moe_linear` (`moe_linear_<fmt>_<access>`). Slots: `table`, `bias_table` (dummy without bias), `ids` (int32, `slots` entries), `x`, `out`, then
/// `weights` / `biases` (the arena buffers; dummies for [`ExpertAccess::BufferAddress`]).
pub fn plan_moe_linear(c: &MoeLinear) -> Result<OpPlan, BackendError> {
    let (tag, qk) = format_tag(c.quant).ok_or(BackendError::UnsupportedQuant(c.quant))?;
    let name = moe_linear_shader(tag, c.access).ok_or(BackendError::UnsupportedQuant(c.quant))?;
    if c.rows == 0 || c.cols == 0 || c.slots == 0 || c.x_div == 0 || c.n_experts == 0 {
        return Err(invalid("moe_linear with an empty dimension"));
    }
    if !c.cols.is_multiple_of(qk) || !c.cols.is_multiple_of(8) {
        return Err(BackendError::UnsupportedQuant(c.quant));
    }
    if c.rows > MAX_GROUPS_PER_DIM as usize {
        return Err(BackendError::UnsupportedOperation(format!(
            "moe_linear with {} output rows (> {MAX_GROUPS_PER_DIM})",
            c.rows
        )));
    }
    let slots_y = c.slots.min(MAX_GROUPS_PER_DIM as usize);
    let slots_z = div_up(c.slots, MAX_GROUPS_PER_DIM as usize);
    let push = Push::new()
        .u("rows", c.rows)
        .u("cols", c.cols)
        .u("slots", c.slots)
        .u("x_div", c.x_div)
        .u("n_experts", c.n_experts)
        .u("flags", usize::from(c.has_bias))
        .u("nb", c.cols / qk)
        .u("off1", c.planes[0])
        .u("off2", c.planes[1])
        .u("off3", c.planes[2]);
    let arena = c.access == ExpertAccess::Arena;
    let slots = Slots::new()
        .read("table")
        .read_opt("bias_table", c.has_bias)
        .read("ids")
        .read("x")
        .write("out")
        .read_opt("weights", arena)
        .read_opt("biases", arena && c.has_bias);
    plan(
        name,
        push,
        [c.rows as u32, slots_y as u32, slots_z as u32],
        slots,
    )
}

/// `Stream::moe_combine` (`moe_combine`).
pub fn plan_moe_combine(tokens: usize, k: usize, hidden: usize) -> Result<OpPlan, BackendError> {
    if tokens == 0 || k == 0 || hidden == 0 {
        return Err(invalid("moe_combine of an empty tensor"));
    }
    let push = Push::new()
        .u("tokens", tokens)
        .u("k", k)
        .u("hidden", hidden);
    plan(
        "moe_combine",
        push,
        groups_1d(tokens * hidden, 256),
        Slots::new().write("out").read("y").read("weights"),
    )
}

/// `Stream::moe_expert_gemm`: `out[r] = sum_k weights[r * top_k + k] * (W[indices[r * top_k + k]] x[r])` for experts of one shape and format, as
/// `moe_linear` (x_div = top_k, scratch `y` of `rows * top_k * out_width` floats, scratch id 0) followed by `moe_combine`.
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
    let mut first = plan_moe_linear(&lin)?;
    let y_bytes = rows * top_k * c.rows * 4;
    first.bindings[4] = BindingSlot {
        binding: 4,
        name: "y",
        access: Access::Write,
        source: SlotSource::Scratch {
            id: 0,
            bytes: y_bytes.max(16),
        },
    };
    first.bindings[2].name = "expert_indices";
    let mut second = plan_moe_combine(rows, top_k, c.rows)?;
    second.bindings[1] = BindingSlot {
        binding: 1,
        name: "y",
        access: Access::Read,
        source: SlotSource::Scratch {
            id: 0,
            bytes: y_bytes.max(16),
        },
    };
    second.bindings[2].name = "expert_weights";
    Ok(vec![first, second])
}

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

/// `Stream::gather_rows` (`moe_gather_rows`): `dst[i] = src[idx[i]] * weights[i]`. Slots: `dst`, `src`, `idx` (host), `weights` (host, dummy without).
pub fn plan_gather_rows(
    idx: &[i32],
    weights: Option<&[f32]>,
    row_len: usize,
) -> Result<OpPlan, BackendError> {
    check_rows(idx, weights, row_len)?;
    let n = idx.len();
    let push = Push::new()
        .u("n", n)
        .u("row_len", row_len)
        .b("has_weights", weights.is_some());
    plan(
        "moe_gather_rows",
        push,
        groups_1d(n * row_len, 256),
        index_slots(idx, weights).prepend_dst_src(Access::Write),
    )
}

/// `Stream::scatter_add_rows` (`moe_scatter_add_rows`): `dst[idx[i]] += src[i] * weights[i]`. Slots: `dst`, `src`, `idx` (host), `weights` (host).
pub fn plan_scatter_add_rows(
    idx: &[i32],
    weights: Option<&[f32]>,
    row_len: usize,
) -> Result<OpPlan, BackendError> {
    check_rows(idx, weights, row_len)?;
    let n = idx.len();
    let push = Push::new()
        .u("n", n)
        .u("row_len", row_len)
        .b("has_weights", weights.is_some());
    plan(
        "moe_scatter_add_rows",
        push,
        groups_1d(n * row_len, 256),
        index_slots(idx, weights).prepend_dst_src(Access::ReadWrite),
    )
}

/// Slots 2 and 3 of the gather / scatter shaders.
struct IndexSlots {
    idx: Arc<Vec<u8>>,
    weights: Option<Arc<Vec<u8>>>,
}

fn index_slots(idx: &[i32], weights: Option<&[f32]>) -> IndexSlots {
    IndexSlots {
        idx: Arc::new(idx.iter().flat_map(|v| v.to_le_bytes()).collect()),
        weights: weights.map(|w| Arc::new(pack_f32(w))),
    }
}

impl IndexSlots {
    fn prepend_dst_src(self, dst: Access) -> Slots {
        let s = Slots::new();
        let s = match dst {
            Access::Write => s.write("dst"),
            _ => s.rw("dst"),
        };
        let s = s.read("src").host("idx", self.idx);
        match self.weights {
            Some(w) => s.host("weights", w),
            None => s.dummy("weights", Access::Read),
        }
    }
}
