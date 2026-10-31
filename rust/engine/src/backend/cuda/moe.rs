//! CUDA side of the device mixture-of-experts operations (`cuda/moe.cu`): general routing, expert-major sort, the grouped tensor-core
//! GEMM, the combination with the shared expert. The `Stream` / `Backend` methods in `cuda.rs` call these functions.
//!
//! `FLLM_CUDA_MOE=0` (also `off`, `false`, `no`) switches the operations off: the decoder then keeps the earlier behaviour (the
//! softmax-only decode path with `moe_topk` / `moe_linear`, host routing for everything else).

use super::*;

extern "C" {
    fn fllm_cuda_moe_route2(
        st: *mut RawStream,
        logits: *const f32,
        bias: *const f32,
        ids: *mut i32,
        w: *mut f32,
        rows: c_int,
        experts: c_int,
        k: c_int,
        sigmoid: c_int,
        normalize: c_int,
        scale: f32,
        eps: f32,
        groups: c_int,
        groups_used: c_int,
        top2: c_int,
    ) -> c_int;
    fn fllm_cuda_moe_align(
        st: *mut RawStream,
        ids: *const i32,
        sorted: *mut i32,
        block_expert: *mut i32,
        post: *mut i32,
        pairs: c_int,
        expert_offset: c_int,
        num_local: c_int,
        block: c_int,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn fllm_cuda_moe_grouped(
        st: *mut RawStream,
        kind: c_int,
        glu: c_int,
        wtab: *const *const c_void,
        wtab2: *const *const c_void,
        btab: *const *const c_void,
        btab2: *const *const c_void,
        x: *const f32,
        pw: *const f32,
        y: *mut f32,
        sorted: *const i32,
        block_expert: *const i32,
        post: *const i32,
        rows: c_int,
        cols: c_int,
        pairs: c_int,
        xdiv: c_int,
        block: c_int,
        max_blocks: c_int,
        glu_mode: c_int,
        alpha: f32,
        limit: f32,
    ) -> c_int;
    fn fllm_cuda_moe_combine2(
        st: *mut RawStream,
        out: *mut f32,
        y: *const f32,
        w: *const f32,
        shared: *const f32,
        gate: *const f32,
        tokens: c_int,
        k: c_int,
        hidden: c_int,
    ) -> c_int;
    fn fllm_cuda_moe_gate_scale(st: *mut RawStream, gate: *mut f32, x: *const f32, g: *const f32, tokens: c_int, hidden: c_int) -> c_int;
    fn fllm_cuda_moe_zero_fold(st: *mut RawStream, zs: *mut f32, ids: *mut i32, w: *mut f32, rows: c_int, k: c_int, routed: c_int, rewrite: c_int) -> c_int;
}

/// `FLLM_CUDA_MOE` (default on): the device MoE operations of this file.
pub(super) fn moe_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("FLLM_CUDA_MOE").ok().as_deref().map(str::trim), Some("0" | "off" | "false" | "no")))
}

/// GGUF type ids of the device layouts the grouped GEMM decodes: F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, MXFP4, Q4_K, Q5_K, Q6_K.
pub(super) fn grouped_kind(kind: u32) -> bool {
    matches!(
        kind,
        1 | 30 | 2 | 3 | 6 | 7 | 8 | 10..=14 | 16..=23 | 29 | 39 | 40 | 0x8000_0010..=0x8000_0012 | 0x8000_0021
    )
}

/// Does the grouped GEMM run `quant_type` weights with `cols` inputs in tiles of `block` pairs on a device of compute capability `cc`?
pub(super) fn grouped_supported(cc: u32, quant_type: QuantType, cols: usize, block: usize) -> bool {
    let kind = quant_type.to_gguf_type_id();
    let width = if matches!(kind, 10..=14 | 16..=19 | 21 | 22 | 29) {
        256
    } else if matches!(kind, 1 | 30 | 0x8000_0021) {
        1
    } else if kind == 40 {
        64
    } else {
        quant_type.q4z_group().unwrap_or(32)
    };
    cc >= 75 && grouped_kind(kind) && matches!(block, 16 | 64) && cols.is_multiple_of(16) && cols.is_multiple_of(width)
}

fn opt<T>(buf: Option<&DeviceBuffer>) -> Result<*const T, BackendError> {
    match buf {
        Some(b) => Ok(dev_ptr(b)?.cast()),
        None => Ok(std::ptr::null()),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn route_ex(
    raw: *mut RawStream,
    logits: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    ids: &mut DeviceBuffer,
    weights: &mut DeviceBuffer,
    rows: usize,
    experts: usize,
    k: usize,
    route: &MoeRoute,
) -> Result<(), BackendError> {
    if logits.size < rows * experts * 4 || ids.size < rows * k * 4 || weights.size < rows * k * 4 || bias.is_some_and(|b| b.size < experts * 4) {
        return Err(invalid("moe_route_ex: buffers too small"));
    }
    // SAFETY: sizes checked above; the kernel reads `rows x experts` logits and writes `rows x k` ids and weights.
    check(unsafe {
        fllm_cuda_moe_route2(
            raw,
            dev_ptr(logits)?.cast(),
            opt(bias)?,
            dev_ptr(ids)?.cast(),
            dev_ptr(weights)?.cast(),
            to_c(rows, "rows")?,
            to_c(experts, "experts")?,
            to_c(k, "k")?,
            c_int::from(route.sigmoid),
            c_int::from(route.normalize),
            route.scale,
            route.eps,
            to_c(route.groups, "groups")?,
            to_c(route.groups_used, "groups used")?,
            c_int::from(route.group_top2),
        )
    })
}

pub(super) fn align(raw: *mut RawStream, ids: &DeviceBuffer, sort: &mut MoeSort, expert_offset: usize, num_local: usize) -> Result<(), BackendError> {
    let (sorted, blocks, post) = crate::backend::moe::moe_sort_bytes(sort.pairs, num_local, sort.block);
    if ids.size < sort.pairs * 4 || sort.sorted.size < sorted || sort.block_expert.size < blocks || sort.post.size < post {
        return Err(invalid("moe_align: buffers too small"));
    }
    // SAFETY: sizes checked above.
    check(unsafe {
        fllm_cuda_moe_align(
            raw,
            dev_ptr(ids)?.cast(),
            dev_ptr(&sort.sorted)?.cast(),
            dev_ptr(&sort.block_expert)?.cast(),
            dev_ptr(&sort.post)?.cast(),
            to_c(sort.pairs, "pairs")?,
            to_c(expert_offset, "expert offset")?,
            to_c(num_local, "local experts")?,
            to_c(sort.block, "block")?,
        )
    })
}

pub(super) fn grouped(raw: *mut RawStream, cc: u32, call: MoeGrouped<'_>) -> Result<(), BackendError> {
    let MoeGrouped { out, x, table, table_up, bias, bias_up, input_weights, sort, quant_type, rows, cols, x_div, glu } = call;
    if !grouped_supported(cc, quant_type, cols, sort.block) || x_div == 0 {
        return Err(BackendError::UnsupportedQuant(quant_type));
    }
    let pairs = sort.pairs;
    if x.size < pairs.div_ceil(x_div) * cols * 4 || out.size < pairs * rows * 4 || input_weights.is_some_and(|w| w.size < pairs * 4) {
        return Err(invalid("moe_grouped_linear: buffers too small"));
    }
    if glu.is_some() != table_up.is_some() {
        return Err(invalid("moe_grouped_linear: a fused gate/up call needs both the second table and the activation"));
    }
    let g = glu.unwrap_or(MoeGlu { mode: 0, alpha: 0.0, limit: 0.0 });
    // SAFETY: sizes checked above; the tables were built by `upload_pointer_table` / `upload_bias_table` for weights of `quant_type`
    // and outlive the launch (the caller keeps the weights alive).
    check(unsafe {
        fllm_cuda_moe_grouped(
            raw,
            quant_type.to_gguf_type_id() as c_int,
            c_int::from(glu.is_some()),
            opt(Some(table))?,
            opt(table_up)?,
            opt(bias)?,
            opt(bias_up)?,
            dev_ptr(x)?.cast(),
            opt(input_weights)?,
            dev_ptr(out)?.cast(),
            dev_ptr(&sort.sorted)?.cast(),
            dev_ptr(&sort.block_expert)?.cast(),
            dev_ptr(&sort.post)?.cast(),
            to_c(rows, "rows")?,
            to_c(cols, "cols")?,
            to_c(pairs, "pairs")?,
            to_c(x_div, "x_div")?,
            to_c(sort.block, "block")?,
            to_c(sort.max_blocks, "blocks")?,
            g.mode,
            g.alpha,
            g.limit,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn combine_ex(
    raw: *mut RawStream,
    out: &mut DeviceBuffer,
    y: &DeviceBuffer,
    weights: Option<&DeviceBuffer>,
    shared: Option<&DeviceBuffer>,
    gate: Option<&DeviceBuffer>,
    tokens: usize,
    k: usize,
    hidden: usize,
) -> Result<(), BackendError> {
    if out.size < tokens * hidden * 4
        || y.size < tokens * k * hidden * 4
        || weights.is_some_and(|w| w.size < tokens * k * 4)
        || shared.is_some_and(|s| s.size < tokens * hidden * 4)
        || gate.is_some_and(|g| g.size < tokens * 4)
    {
        return Err(invalid("moe_combine_ex: buffers too small"));
    }
    // SAFETY: sizes checked above.
    check(unsafe {
        fllm_cuda_moe_combine2(
            raw,
            dev_ptr(out)?.cast(),
            dev_ptr(y)?.cast(),
            opt(weights)?,
            opt(shared)?,
            opt(gate)?,
            to_c(tokens, "tokens")?,
            to_c(k, "k")?,
            to_c(hidden, "hidden")?,
        )
    })
}

pub(super) fn gate_scale(raw: *mut RawStream, gate: &mut DeviceBuffer, x: &DeviceBuffer, g: &DeviceBuffer, tokens: usize, hidden: usize) -> Result<(), BackendError> {
    if gate.size < tokens * 4 || x.size < tokens * hidden * 4 || g.size < hidden * 4 {
        return Err(invalid("moe_gate_scale: buffers too small"));
    }
    // SAFETY: sizes checked above.
    check(unsafe { fllm_cuda_moe_gate_scale(raw, dev_ptr(gate)?.cast(), dev_ptr(x)?.cast(), dev_ptr(g)?.cast(), to_c(tokens, "tokens")?, to_c(hidden, "hidden")?) })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn zero_fold(
    raw: *mut RawStream,
    zero_scale: &mut DeviceBuffer,
    ids: &mut DeviceBuffer,
    weights: &mut DeviceBuffer,
    rows: usize,
    k: usize,
    routed: usize,
    rewrite: bool,
) -> Result<(), BackendError> {
    if zero_scale.size < rows * 4 || ids.size < rows * k * 4 || weights.size < rows * k * 4 {
        return Err(invalid("moe_zero_fold: buffers too small"));
    }
    // SAFETY: sizes checked above.
    check(unsafe {
        fllm_cuda_moe_zero_fold(
            raw,
            dev_ptr(zero_scale)?.cast(),
            dev_ptr(ids)?.cast(),
            dev_ptr(weights)?.cast(),
            to_c(rows, "rows")?,
            to_c(k, "k")?,
            to_c(routed, "routed experts")?,
            c_int::from(rewrite),
        )
    })
}

impl CudaStream {
    /// `Stream::moe_expert_gemm` on the device: sort the pairs, one grouped product over all experts, weighted combination. `false`
    /// (nothing done) when the weights or the GPU are not covered; the caller then takes the host path.
    pub(super) fn expert_gemm_device(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        experts: &[DeviceWeight],
        indices: &DeviceBuffer,
        weights: &DeviceBuffer,
        top_k: usize,
    ) -> Result<bool, BackendError> {
        let Some(first) = experts.first() else {
            return Ok(false);
        };
        let (hidden, width, quant) = (first.cols, first.rows, first.quant_type);
        if !moe_enabled() || self.cc < 75 || top_k == 0 || hidden == 0 || !x.size.is_multiple_of(hidden * 4) {
            return Ok(false);
        }
        if experts.iter().any(|e| e.cols != hidden || e.rows != width || e.quant_type != quant) {
            return Ok(false);
        }
        let tokens = x.size / (hidden * 4);
        let pairs = tokens * top_k;
        let block = crate::backend::moe::moe_block_for(pairs, experts.len());
        if pairs == 0 || !grouped_supported(self.cc, quant, hidden, block) {
            return Ok(false);
        }
        if indices.size < pairs * 4 || weights.size < pairs * 4 || out.size < tokens * width * 4 {
            return Err(invalid("moe_expert_gemm: routing or output buffers smaller than rows x top_k"));
        }
        let mut table = Vec::with_capacity(experts.len() * 8);
        for e in experts {
            table.extend_from_slice(&(dev_ptr(&e.buffer)? as usize as u64).to_le_bytes());
        }
        let mut tab = self.alloc(table.len())?;
        self.upload(&mut tab, &table)?;
        let (sorted, blocks, post) = crate::backend::moe::moe_sort_bytes(pairs, experts.len(), block);
        let mut sort = MoeSort { sorted: self.alloc(sorted)?, block_expert: self.alloc(blocks)?, post: self.alloc(post)?, pairs, block, max_blocks: blocks / 4 };
        self.moe_align(indices, &mut sort, 0, experts.len())?;
        // an index outside the expert list contributes nothing: its row of `y` must read as zero
        let mut y = self.alloc(pairs * width * 4)?;
        self.zero_buffer(&mut y)?;
        self.moe_grouped_linear(MoeGrouped {
            out: &mut y,
            x,
            table: &tab,
            table_up: None,
            bias: None,
            bias_up: None,
            input_weights: None,
            sort: &sort,
            quant_type: quant,
            rows: width,
            cols: hidden,
            x_div: top_k,
            glu: None,
        })?;
        self.moe_combine_ex(out, &y, Some(weights), None, None, tokens, top_k, width)?;
        // the temporaries are freed when this function returns
        self.synchronize()?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_kinds_and_shapes() {
        let q = QuantType::from_gguf_type_id;
        assert!(grouped_supported(80, q(12), 2048, 64));
        assert!(!grouped_supported(80, q(12), 1408, 64), "Q4_K needs multiples of 256");
        assert!(grouped_supported(80, q(8), 1408, 16));
        assert!(grouped_supported(75, q(1), 2048, 16));
        assert!(!grouped_supported(70, q(8), 2048, 16), "tensor cores start at sm_75");
        assert!(!grouped_supported(80, q(8), 2048, 32), "only the 16 and 64 pair tiles exist");
        assert!(grouped_supported(80, q(10), 2048, 16), "Q2_K is now supported in grouped GEMM");
        assert!(grouped_supported(80, q(40), 2048, 16), "NVFP4 is supported in grouped GEMM");
        assert!(grouped_supported(80, q(0x8000_0010), 2048, 16), "Q4Z is supported in grouped GEMM");
        assert!(grouped_supported(80, q(0x8000_0021), 2048, 16), "FP8B is supported in grouped GEMM");
        assert!(grouped_supported(80, q(0x8000_0021), 528, 64), "FP8B with a partial last tile");
        assert!(!grouped_supported(80, q(0x8000_0155), 2048, 16), "only the 128 x 128 tiles have a grouped kernel; other tiles use the per-expert path");
        assert!(!grouped_supported(80, q(0x8000_0020), 2048, 16), "FP8M has row scales the grouped kernel does not apply");
    }
}
