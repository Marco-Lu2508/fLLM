//! Latent-cache multi-head latent attention on the CUDA backend (`FLLM_MLA=latent`): the FFI of `cuda/mla_latent.cu` / `cuda/mla_decode.cu`, the
//! stream operations behind `Stream::mla_latent_*`, and the capability check that decides whether a model may use the latent cache.
//! The host side (cache geometry, memory accounting, the dense kv_b matrix) is `crate::mla_latent`.
use super::*;
use crate::config::MlaConfig;
use crate::kv_format::KvFormat;

extern "C" {
    fn fllm_cuda_mla_latent_supported(dev: c_int, kv_lora: c_int, rope: c_int, nope: c_int, value: c_int, kv_fmt: c_int) -> c_int;
    fn fllm_cuda_mla_kv(
        st: *mut RawStream,
        kv_a: *const f32,
        latent: *const f32,
        positions: *const i32,
        cs: *const f32,
        sn: *const f32,
        k_row: *mut f32,
        rows: c_int,
        kv_lora: c_int,
        rope: c_int,
        neox: c_int,
    ) -> c_int;
    fn fllm_cuda_mla_query(
        st: *mut RawStream,
        q_full: *const f32,
        w: *const u16,
        positions: *const i32,
        cs: *const f32,
        sn: *const f32,
        q_lat: *mut u16,
        rows: c_int,
        heads: c_int,
        nope: c_int,
        rope: c_int,
        value: c_int,
        kv_lora: c_int,
        neox: c_int,
    ) -> c_int;
    fn fllm_cuda_mla_output(
        st: *mut RawStream,
        o_lat: *const u16,
        w: *const u16,
        out: *mut f32,
        rows: c_int,
        heads: c_int,
        nope: c_int,
        value: c_int,
        kv_lora: c_int,
    ) -> c_int;
    fn fllm_cuda_mla_attention(
        st: *mut RawStream,
        q_lat: *const u16,
        pool: *const u16,
        o_lat: *mut u16,
        tables: *const i64,
        tables_len: usize,
        lengths: *const i64,
        rows: c_int,
        heads: c_int,
        kv_lora: c_int,
        rope: c_int,
        stride: c_int,
        block_size: c_int,
        kd: *const KvDesc,
        scale: f32,
        mode: c_int,
    ) -> c_int;
}

/// May a model with this MLA configuration use the latent cache on CUDA device `device` with cache format `format`? False for AMD builds, GPUs before
/// Turing, dimensions the kernels do not cover and cache formats they cannot read; the caller then keeps the expanded cache.
pub fn latent_supported(device: usize, m: &MlaConfig, format: KvFormat) -> bool {
    if !crate::mla_latent::format_supported(format) {
        return false;
    }
    let dims = [m.kv_lora, m.rope, m.nope, m.value].map(|v| c_int::try_from(v).unwrap_or(0));
    // SAFETY: plain query; the C side selects the device and declines on any failure.
    unsafe { fllm_cuda_mla_latent_supported(device as c_int, dims[0], dims[1], dims[2], dims[3], format.code()) != 0 }
}


fn check_rope(shape: &MlaShape, rope: &RopeParams, positions: &[usize]) -> Result<(), BackendError> {
    let half = shape.rope / 2;
    if rope.rope_dim != shape.rope || shape.rope == 0 || !shape.rope.is_multiple_of(2) {
        return Err(invalid("latent MLA: rotary width does not match the layer"));
    }
    let top = positions.iter().copied().max().map_or(0, |p| p + 1);
    if rope.cos.len() < top * half || rope.sin.len() < top * half {
        return Err(invalid("latent MLA: rotary tables do not cover the positions"));
    }
    Ok(())
}

pub(super) fn kv(
    s: &mut CudaStream,
    k_row: &mut DeviceBuffer,
    kv_a: &DeviceBuffer,
    latent: &DeviceBuffer,
    positions: &[usize],
    shape: &MlaShape,
    rope: &RopeParams,
) -> Result<(), BackendError> {
    let rows = positions.len();
    if rows == 0 {
        return Ok(());
    }
    check_rope(shape, rope, positions)?;
    let width = shape.kv_lora + shape.rope;
    if k_row.size < rows * width * 4 || kv_a.size < rows * width * 4 || latent.size < rows * shape.kv_lora * 4 {
        return Err(invalid("latent MLA cache row: buffers smaller than the layer needs"));
    }
    let cos = s.table(&rope.cos)?;
    let sin = s.table(&rope.sin)?;
    let pos: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    // SAFETY: sizes and table coverage checked above; the positions are staged by the C side before it returns.
    check(unsafe {
        fllm_cuda_mla_kv(
            s.raw,
            dev_ptr(kv_a)?.cast(),
            dev_ptr(latent)?.cast(),
            pos.as_ptr(),
            cos,
            sin,
            dev_ptr(k_row)?.cast(),
            to_c(rows, "rows")?,
            to_c(shape.kv_lora, "latent width")?,
            to_c(shape.rope, "rope width")?,
            c_int::from(rope.neox),
        )
    })
}

pub(super) fn query(
    s: &mut CudaStream,
    q_lat: &mut DeviceBuffer,
    q_full: &DeviceBuffer,
    w_kvb: &DeviceBuffer,
    positions: &[usize],
    shape: &MlaShape,
    rope: &RopeParams,
) -> Result<(), BackendError> {
    let rows = positions.len();
    if rows == 0 {
        return Ok(());
    }
    check_rope(shape, rope, positions)?;
    let dk = shape.nope + shape.rope;
    let width = shape.kv_lora + shape.rope;
    if q_lat.size < rows * shape.heads * width * 2
        || q_full.size < rows * shape.heads * dk * 4
        || w_kvb.size < shape.heads * (shape.nope + shape.value) * shape.kv_lora * 2
    {
        return Err(invalid("latent MLA query: buffers smaller than the layer needs"));
    }
    let cos = s.table(&rope.cos)?;
    let sin = s.table(&rope.sin)?;
    let pos: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    // SAFETY: sizes and table coverage checked above.
    check(unsafe {
        fllm_cuda_mla_query(
            s.raw,
            dev_ptr(q_full)?.cast(),
            dev_ptr(w_kvb)?.cast(),
            pos.as_ptr(),
            cos,
            sin,
            dev_ptr(q_lat)?.cast(),
            to_c(rows, "rows")?,
            to_c(shape.heads, "heads")?,
            to_c(shape.nope, "nope width")?,
            to_c(shape.rope, "rope width")?,
            to_c(shape.value, "value width")?,
            to_c(shape.kv_lora, "latent width")?,
            c_int::from(rope.neox),
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    s: &mut CudaStream,
    o_lat: &mut DeviceBuffer,
    q_lat: &DeviceBuffer,
    kv: &DeviceKvCache,
    block_tables: &[i64],
    lengths: &[i64],
    layer_idx: usize,
    params: &AttentionParams,
    shape: &MlaShape,
    reference: bool,
) -> Result<(), BackendError> {
    let rows = lengths.len();
    if rows == 0 {
        return Ok(());
    }
    params.validate_blocks(kv, block_tables, lengths, layer_idx, true)?;
    let width = shape.kv_lora + shape.rope;
    if kv.kv_heads != 1 || kv.head_dim != width || params.num_heads != shape.heads {
        return Err(invalid("latent MLA attention: the KV cache is not a latent cache of this layer"));
    }
    if q_lat.size < rows * shape.heads * width * 2 || o_lat.size < rows * shape.heads * shape.kv_lora * 2 {
        return Err(invalid("latent MLA attention: buffers smaller than rows x heads x width"));
    }
    let (key_layout, _) = kv.layouts();
    if !crate::mla_latent::format_supported(key_layout.fmt) {
        return Err(BackendError::UnsupportedOperation(format!("latent MLA cache in {}", key_layout.fmt.name())));
    }
    let kd = KvDesc::new(&key_layout);
    // SAFETY: `validate_blocks` checked tables, lengths and parameters against the cache; the pool pointer addresses the requested layer.
    let rc = unsafe {
        fllm_cuda_mla_attention(
            s.raw,
            dev_ptr(q_lat)?.cast(),
            dev_ptr(&kv.key_pool)?.add(kv.layer_key_offset(layer_idx) * 2).cast(),
            dev_ptr(o_lat)?.cast(),
            block_tables.as_ptr(),
            block_tables.len(),
            lengths.as_ptr(),
            to_c(rows, "rows")?,
            to_c(shape.heads, "heads")?,
            to_c(shape.kv_lora, "latent width")?,
            to_c(shape.rope, "rope width")?,
            to_c(params.stride, "block table stride")?,
            to_c(params.block_size, "block size")?,
            &kd,
            params.scale,
            c_int::from(reference),
        )
    };
    check(rc)
}

pub(super) fn output(
    s: &mut CudaStream,
    out: &mut DeviceBuffer,
    o_lat: &DeviceBuffer,
    w_kvb: &DeviceBuffer,
    rows: usize,
    shape: &MlaShape,
) -> Result<(), BackendError> {
    if rows == 0 {
        return Ok(());
    }
    if out.size < rows * shape.heads * shape.value * 4
        || o_lat.size < rows * shape.heads * shape.kv_lora * 2
        || w_kvb.size < shape.heads * (shape.nope + shape.value) * shape.kv_lora * 2
    {
        return Err(invalid("latent MLA output: buffers smaller than the layer needs"));
    }
    // SAFETY: sizes checked above.
    check(unsafe {
        fllm_cuda_mla_output(
            s.raw,
            dev_ptr(o_lat)?.cast(),
            dev_ptr(w_kvb)?.cast(),
            dev_ptr(out)?.cast(),
            to_c(rows, "rows")?,
            to_c(shape.heads, "heads")?,
            to_c(shape.nope, "nope width")?,
            to_c(shape.value, "value width")?,
            to_c(shape.kv_lora, "latent width")?,
        )
    })
}
