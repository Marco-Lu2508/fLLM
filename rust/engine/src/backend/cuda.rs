//! CUDA backend (Cargo feature `cuda`).
//!
//! The kernels live in `rust/engine/cuda/` and are compiled by `build.rs` with `nvcc` into a
//! static library that links the CUDA runtime statically. The binary only needs the NVIDIA driver
//! (`libcuda`) at run time, which the runtime loads lazily: without it `list_devices` reports no
//! CUDA device and fLLM continues on the CPU or another backend.
//!
//! Semantics follow the other backends (see `Backend` / `Stream` in `mod.rs`): activations are
//! f32, weights keep their GGUF block layout on the device, the KV cache uses the shared 16-bit
//! word layout. `fllm-rs selftest-gpu --device cuda` checks every operation against the CPU
//! reference.

use super::cpu::{CpuBackend, CpuStream};
use super::*;
use crate::devices::{DeviceInfo, DeviceType};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::atomic::{AtomicU64, Ordering};

#[repr(C)]
struct RawStream {
    _private: [u8; 0],
}

/// Layout of one KV pool for the kernels in `cuda/kv_quant.cu` (mirrors `FllmKvDesc`).
#[repr(C)]
#[derive(Clone, Copy)]
struct KvDesc {
    fmt: c_int,
    kv: c_int,
    group: c_int,
    dim: c_int,
    gshift: c_int,
    gps: c_int,
    hi_off: i64,
    scale_off: i64,
    min_off: i64,
}

impl KvDesc {
    fn new(layout: &crate::kv_format::KvLayout) -> Self {
        Self {
            fmt: layout.fmt.code(),
            kv: layout.kv as c_int,
            group: layout.group as c_int,
            dim: layout.dim as c_int,
            gshift: if layout.group < layout.dim { layout.group.trailing_zeros() as c_int } else { -1 },
            gps: (layout.kv / layout.group) as c_int,
            hi_off: layout.hi_off as i64,
            scale_off: layout.scale_off as i64,
            min_off: layout.min_off as i64,
        }
    }
}

extern "C" {
    fn fllm_cuda_last_error() -> *const c_char;
    fn fllm_cuda_device_count() -> c_int;
    fn fllm_cuda_device_info(
        dev: c_int,
        name: *mut c_char,
        name_len: usize,
        cc_major: *mut c_int,
        cc_minor: *mut c_int,
        free_bytes: *mut usize,
        total_bytes: *mut usize,
        sm_count: *mut c_int,
    ) -> c_int;
    fn fllm_cuda_malloc(dev: c_int, ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn fllm_cuda_free(ptr: *mut c_void) -> c_int;
    fn fllm_cuda_h2d_sync(dev: c_int, dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn fllm_cuda_memset(dev: c_int, ptr: *mut c_void, value: c_int, bytes: usize) -> c_int;
    fn fllm_cuda_stream_create(dev: c_int) -> *mut RawStream;
    fn fllm_cuda_stream_destroy(st: *mut RawStream);
    fn fllm_cuda_stream_sync(st: *mut RawStream) -> c_int;
    fn fllm_cuda_upload(st: *mut RawStream, dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn fllm_cuda_download(st: *mut RawStream, dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn fllm_cuda_copy(st: *mut RawStream, dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn fllm_cuda_linear(
        st: *mut RawStream,
        w: *const c_void,
        kind: c_int,
        x: *const f32,
        y: *mut f32,
        bias: *const f32,
        rows: c_int,
        cols: c_int,
        batch: c_int,
    ) -> c_int;
    fn fllm_cuda_linear_q4z(
        st: *mut RawStream,
        w: *const c_void,
        group: c_int,
        x: *const f32,
        y: *mut f32,
        bias: *const f32,
        rows: c_int,
        cols: c_int,
        batch: c_int,
        perm: *const c_int,
    ) -> c_int;
    fn fllm_cuda_q4z_repack(dev: c_int, awq_words: *const c_void, out: *mut c_void, k: c_int, n: c_int) -> c_int;
    fn fllm_cuda_q4z_supported(rows: c_int, cols: c_int, group: c_int) -> c_int;
    fn fllm_cuda_q4z_accumulate(on: c_int, scale: f32) -> c_int;
    fn fllm_cuda_q4z_f16out(on: c_int) -> c_int;
    fn fllm_cuda_nvfp4_accumulate(on: c_int, scale: f32) -> c_int;
    fn fllm_cuda_nvfp4_f16out(on: c_int) -> c_int;
    fn fllm_cuda_glu_f16(st: *mut RawStream, gate16: *const c_void, up16: *const c_void, out: *mut f32, count: usize, mode: c_int) -> c_int;
    fn fllm_cuda_nvfp4_cutlass_ok(x: *const f32, y: *const f32, bias: *const f32, rows: c_int, cols: c_int, batch: c_int, mode: c_int) -> c_int;
    fn fllm_cuda_rope_kv_f16(
        st: *mut RawStream,
        q16: *const u16,
        k16: *const u16,
        v16: *const u16,
        qb: *const f32,
        kb: *const f32,
        vb: *const f32,
        q_out: *mut f32,
        pool_k_layer: *mut u16,
        pool_v_layer: *mut u16,
        positions: *const i32,
        slots: *const i32,
        rows: c_int,
        cs: *const f32,
        sn: *const f32,
        heads: c_int,
        kv_heads: c_int,
        dk: c_int,
        dv: c_int,
        half_: c_int,
        neox: c_int,
    ) -> c_int;
    fn fllm_cuda_add_rms_norm(
        st: *mut RawStream,
        x: *mut f32,
        branch: *const f32,
        scale: f32,
        w: *const f32,
        out: *mut f32,
        rows: c_int,
        cols: c_int,
        eps: f32,
        quant: c_int,
        kind: c_int,
    ) -> c_int;
    fn fllm_cuda_linear_q4z_add_norm(
        st: *mut RawStream,
        w: *const c_void,
        group: c_int,
        act: *const f32,
        res: *mut f32,
        bias: *const f32,
        rows: c_int,
        cols: c_int,
        batch: c_int,
        perm: *const c_int,
        scale: f32,
        nw: *const f32,
        out: *mut f32,
        eps: f32,
        quant: c_int,
        kind: c_int,
    ) -> c_int;
    fn fllm_cuda_rope_kv_f32(
        st: *mut RawStream,
        q: *mut f32,
        k: *const f32,
        v: *const f32,
        pool_k_layer: *mut u16,
        pool_v_layer: *mut u16,
        positions: *const i32,
        slots: *const i32,
        rows: c_int,
        cs: *const f32,
        sn: *const f32,
        heads: c_int,
        kv_heads: c_int,
        dk: c_int,
        dv: c_int,
        half_: c_int,
        neox: c_int,
    ) -> c_int;
    fn fllm_cuda_set_fuse(which: c_int, on: c_int) -> c_int;
    fn fllm_cuda_gptq_repack(dev: c_int, words: *const c_void, out: *mut c_void, k: c_int, n: c_int) -> c_int;
    fn fllm_cuda_linear_fp8_marlin(
        st: *mut RawStream,
        w: *const c_void,
        x: *const f32,
        y: *mut f32,
        bias: *const f32,
        rows: c_int,
        cols: c_int,
        batch: c_int,
    ) -> c_int;
    fn fllm_cuda_fp8_marlin_repack(dev: c_int, words: *const c_void, out: *mut c_void, k: c_int, n: c_int) -> c_int;
    fn fllm_cuda_fp8_marlin_supported(rows: c_int, cols: c_int) -> c_int;
    fn fllm_cuda_argmax(st: *mut RawStream, logits: *const f32, rows: c_int, vocab: c_int, host_out: *mut i32) -> c_int;
    fn fllm_cuda_step_begin(st: *mut RawStream) -> c_int;
    fn fllm_cuda_step_end(st: *mut RawStream) -> c_int;
    fn fllm_cuda_get_opt(id: c_int) -> c_int;
    fn fllm_cuda_set_opt(id: c_int, value: c_int) -> c_int;
    fn fllm_cuda_graph_supported(st: *mut RawStream) -> c_int;
    fn fllm_cuda_graph_begin(st: *mut RawStream, id: c_int, pos: usize, slot: usize, table: usize, len: usize, max_len: c_int) -> c_int;
    fn fllm_cuda_graph_end(st: *mut RawStream, id: c_int, abort: c_int) -> c_int;
    fn fllm_cuda_graph_set(st: *mut RawStream, id: c_int, kind: c_int, data: *const c_void, bytes: usize) -> c_int;
    fn fllm_cuda_graph_launch(st: *mut RawStream, id: c_int) -> c_int;
    fn fllm_cuda_linear_nvfp4(
        st: *mut RawStream,
        w: *const c_void,
        x: *const f32,
        y: *mut f32,
        bias: *const f32,
        gscale: f32,
        rows: c_int,
        cols: c_int,
        batch: c_int,
    ) -> c_int;
    fn fllm_cuda_embedding(
        st: *mut RawStream,
        w: *const c_void,
        kind: c_int,
        ids: *const i64,
        n: c_int,
        out: *mut f32,
        cols: c_int,
    ) -> c_int;
    fn fllm_cuda_rms_norm(
        st: *mut RawStream,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        rows: c_int,
        cols: c_int,
        eps: f32,
    ) -> c_int;
    fn fllm_cuda_layer_norm(
        st: *mut RawStream,
        x: *const f32,
        w: *const f32,
        b: *const f32,
        out: *mut f32,
        rows: c_int,
        cols: c_int,
        eps: f32,
    ) -> c_int;
    fn fllm_cuda_gather_rows(st: *mut RawStream, dst: *mut f32, src: *const f32, idx: *const i32, w: *const f32, n: c_int, row_len: c_int) -> c_int;
    fn fllm_cuda_scatter_add_rows(st: *mut RawStream, dst: *mut f32, src: *const f32, idx: *const i32, w: *const f32, n: c_int, row_len: c_int) -> c_int;
    fn fllm_cuda_zero(st: *mut RawStream, ptr: *mut c_void, bytes: usize) -> c_int;
    fn fllm_cuda_host_register(dev: c_int, ptr: *mut c_void, bytes: usize) -> c_int;
    fn fllm_cuda_host_unregister(ptr: *mut c_void) -> c_int;
    fn fllm_cuda_moe_route(st: *mut RawStream, logits: *const f32, ids: *mut i32, w: *mut f32, rows: c_int, experts: c_int, k: c_int, normalize: c_int, scale: f32) -> c_int;
    fn fllm_cuda_moe_combine(st: *mut RawStream, out: *mut f32, y: *const f32, w: *const f32, tokens: c_int, k: c_int, hidden: c_int) -> c_int;
    fn fllm_cuda_swiglu_oai(st: *mut RawStream, gate: *const f32, up: *const f32, out: *mut f32, count: usize, alpha: f32, limit: f32) -> c_int;
    fn fllm_cuda_moe_linear(st: *mut RawStream, tab: *const *const c_void, btab: *const *const c_void, kind: c_int, ids: *const i32, x: *const f32, y: *mut f32, rows: c_int, cols: c_int, slots: c_int, xdiv: c_int) -> c_int;
    fn fllm_cuda_set_exact(on: c_int) -> c_int;
    fn fllm_cuda_repack(st: *mut RawStream, dst: *mut c_void, src: *const c_void, kind: c_int, rows: usize, cols: usize) -> c_int;
    fn fllm_cuda_glu(
        st: *mut RawStream,
        gate: *const f32,
        up: *const f32,
        out: *mut f32,
        count: usize,
        mode: c_int,
    ) -> c_int;
    fn fllm_cuda_residual(st: *mut RawStream, x: *mut f32, res: *const f32, count: usize, scale: f32) -> c_int;
    fn fllm_cuda_split_q_gate(
        st: *mut RawStream,
        qg: *const f32,
        q: *mut f32,
        gate: *mut f32,
        rows: c_int,
        heads: c_int,
        dk: c_int,
        per_head: c_int,
    ) -> c_int;
    fn fllm_cuda_sigmoid_mul(st: *mut RawStream, x: *mut f32, g: *const f32, n: usize) -> c_int;
    fn fllm_cuda_mla_latent(st: *mut RawStream, kv_a: *const f32, latent: *mut f32, rows: c_int, kv_lora: c_int, width: c_int) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn fllm_cuda_mla_assemble(
        st: *mut RawStream,
        q_full: *const f32,
        kv_a: *const f32,
        kv: *const f32,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        rows: c_int,
        heads: c_int,
        nope: c_int,
        rope: c_int,
        value: c_int,
        kv_lora: c_int,
    ) -> c_int;
    fn fllm_cuda_mamba2_supported(inner: c_int, state: c_int, heads: c_int, groups: c_int, ksz: c_int, norm_groups: c_int) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn fllm_cuda_mamba2_layer(
        st: *mut RawStream,
        runs: *const c_void,
        nruns: c_int,
        rows: c_int,
        inner: c_int,
        state: c_int,
        heads: c_int,
        groups: c_int,
        ksz: c_int,
        norm_groups: c_int,
        step_min: f32,
        eps: f32,
        state_off: usize,
        rec_bytes: usize,
        proj: *const f32,
        convw: *const f32,
        convb: *const f32,
        dt_bias: *const f32,
        decay: *const f32,
        skip: *const f32,
        norm: *const f32,
        mixed: *mut f32,
        out: *mut f32,
    ) -> c_int;
    fn fllm_cuda_shortconv_supported(hidden: c_int, ksz: c_int) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn fllm_cuda_shortconv_layer(
        st: *mut RawStream,
        runs: *const c_void,
        nruns: c_int,
        hidden: c_int,
        ksz: c_int,
        state_off: usize,
        proj: *const f32,
        convw: *const f32,
        convb: *const f32,
        out: *mut f32,
    ) -> c_int;
    fn fllm_cuda_mamba1_supported(inner: c_int, state: c_int, rank: c_int, ksz: c_int) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn fllm_cuda_mamba1_conv(
        st: *mut RawStream,
        runs: *const c_void,
        nruns: c_int,
        inner: c_int,
        ksz: c_int,
        state_off: usize,
        rec_bytes: usize,
        proj: *const f32,
        convw: *const f32,
        convb: *const f32,
        mixed: *mut f32,
    ) -> c_int;
    fn fllm_cuda_mamba1_prep(st: *mut RawStream, split: *mut f32, dt_in: *mut f32, rows: c_int, rank: c_int, state: c_int, eps: f32) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn fllm_cuda_mamba1_scan(
        st: *mut RawStream,
        runs: *const c_void,
        nruns: c_int,
        inner: c_int,
        state: c_int,
        rank: c_int,
        state_off: usize,
        split: *const f32,
        dt: *const f32,
        mixed: *const f32,
        proj: *const f32,
        decay: *const f32,
        skip: *const f32,
        out: *mut f32,
    ) -> c_int;
    fn fllm_cuda_act(st: *mut RawStream, x: *const f32, out: *mut f32, n: usize, mode: c_int) -> c_int;
    fn fllm_cuda_act_glu(st: *mut RawStream, gate: *const f32, up: *const f32, out: *mut f32, n: usize, mode: c_int) -> c_int;
    fn fllm_cuda_softplus_mul(st: *mut RawStream, x: *mut f32, g: *const f32, n: usize, beta: f32) -> c_int;
    fn fllm_cuda_clamp(st: *mut RawStream, x: *mut f32, n: usize, limit: f32) -> c_int;
    fn fllm_cuda_delta_supported(kh: c_int, vh: c_int, hk: c_int, hv: c_int, ksz: c_int) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn fllm_cuda_delta_layer(
        st: *mut RawStream,
        runs: *const c_void,
        nruns: c_int,
        max_n: c_int,
        kh: c_int,
        vh: c_int,
        hk: c_int,
        hv: c_int,
        ksz: c_int,
        tiled: c_int,
        beta_scale: f32,
        eps: f32,
        state_off: usize,
        rec_bytes: usize,
        qkv: *const f32,
        z: *const f32,
        beta: *const f32,
        alpha: *const f32,
        convw: *const f32,
        dt_bias: *const f32,
        decay: *const f32,
        norm: *const f32,
        mixed: *mut f32,
        out: *mut f32,
    ) -> c_int;
    fn fllm_cuda_rope(
        st: *mut RawStream,
        q: *mut f32,
        k: *mut f32,
        positions: *const i32,
        rows: c_int,
        cs: *const f32,
        sn: *const f32,
        heads: c_int,
        kv_heads: c_int,
        dim: c_int,
        half_: c_int,
        neox: c_int,
    ) -> c_int;
    fn fllm_cuda_kv_write(
        st: *mut RawStream,
        src: *const f32,
        pool_layer: *mut u16,
        slots: *const i32,
        rows: c_int,
        kv: c_int,
        kind: c_int,
        group: c_int,
        scale_base_words: c_int,
    ) -> c_int;
    fn fllm_cuda_kvq_write(
        st: *mut RawStream,
        src: *const f32,
        pool_layer: *mut u16,
        slots: *const i32,
        rows: c_int,
        desc: *const KvDesc,
    ) -> c_int;
    fn fllm_cuda_kvq_attention(
        st: *mut RawStream,
        q: *const f32,
        kpool_layer: *const u16,
        vpool_layer: *const u16,
        out: *mut f32,
        tables_host: *const i64,
        tables_len: usize,
        lengths_host: *const i64,
        rows: c_int,
        sinks: *const f32,
        alibi: *const f32,
        heads: c_int,
        kv_heads: c_int,
        dk: c_int,
        dv: c_int,
        stride: c_int,
        block_size: c_int,
        kd: *const KvDesc,
        vd: *const KvDesc,
        scale: f32,
        softcap: f32,
        window: c_int,
        chunk: c_int,
    ) -> c_int;
    fn fllm_cuda_attention(
        st: *mut RawStream,
        q: *const f32,
        kpool_layer: *const u16,
        vpool_layer: *const u16,
        out: *mut f32,
        tables_host: *const i64,
        tables_len: usize,
        lengths_host: *const i64,
        rows: c_int,
        sinks: *const f32,
        alibi: *const f32,
        heads: c_int,
        kv_heads: c_int,
        dk: c_int,
        dv: c_int,
        stride: c_int,
        block_size: c_int,
        kind: c_int,
        kgroup: c_int,
        vgroup: c_int,
        kscale_base: c_int,
        vscale_base: c_int,
        scale: f32,
        softcap: f32,
        window: c_int,
        chunk: c_int,
    ) -> c_int;
}

/// Weight formats the kernels decode natively; every other type is converted on upload. The 256-value formats Q2_K, Q3_K and
/// IQ1_S .. IQ3_S (ids 10, 11, 16-19, 21, 22, 29) use the generic decoders of `iq_decode.cuh`.
const NATIVE_KINDS: [u32; 20] = [0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 29, 30];
/// Widest attention head the kernel holds in shared memory.
const MAX_ATTENTION_DIM: usize = 512;

fn last_error() -> String {
    // SAFETY: the library returns a NUL-terminated buffer owned by thread-local storage.
    unsafe { CStr::from_ptr(fllm_cuda_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn check(code: c_int) -> Result<(), BackendError> {
    if code == 0 {
        Ok(())
    } else {
        Err(BackendError::DriverError(last_error()))
    }
}

fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

/// Answer of a fused kernel entry point: 0 = launched, 3 = declined (nothing was launched), anything else is an error.
fn fused_result(code: c_int) -> Result<bool, BackendError> {
    match code {
        0 => Ok(true),
        3 => Ok(false),
        other => check(other).map(|()| true),
    }
}

/// What the GEMM of `weight` reads: `(quant, kind)` for the fused norm kernels. 1 = FP16 activations of the Marlin 4-bit GEMM (no
/// activation permutation), 2 = NVFP4 blocks, 3 = int8 blocks of the K-quant GEMVs (`kind` is the GGUF type), 0 = nothing to prepare.
fn next_activation_format(weight: Option<&DeviceWeight>) -> (c_int, c_int) {
    let Some(w) = weight else {
        return (0, 0);
    };
    let kind = w.quant_type.to_gguf_type_id();
    if kind == 40 {
        (2, 0)
    } else if w.quant_type.q4z_group().is_some() {
        if w.perm.is_some() {
            (0, 0)
        } else {
            (1, 0)
        }
    } else if matches!(kind, 12..=14) {
        (3, kind as c_int)
    } else {
        (0, 0)
    }
}

/// The fused decode kernels (docs/gpu-backends.md, "Fewer kernels per decode token").
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fusion {
    /// residual add + RMSNorm (+ activation conversion of the next GEMM), `FLLM_FUSE_ADD_NORM`
    AddNorm = 0,
    /// RMSNorm + activation quantization of the next GEMM, `FLLM_FUSE_NORM_QUANT`
    NormQuant = 1,
    /// rotary embedding + KV cache write, `FLLM_FUSE_ROPE_KV`
    RopeKv = 2,
    /// gated activation + int8 quantization, `FLLM_FUSE_GLU_Q8`
    GluQ8 = 3,
}

/// Force a fusion on or off (self test); `None` returns to the environment.
pub fn set_fusion(which: Fusion, on: Option<bool>) {
    // SAFETY: plain setter of a process-wide flag.
    unsafe { fllm_cuda_set_fuse(which as c_int, on.map_or(-1, c_int::from)) };
}

fn to_c(n: usize, what: &str) -> Result<c_int, BackendError> {
    c_int::try_from(n).map_err(|_| invalid(format!("{what} ({n}) exceeds the 32-bit kernel limit")))
}

/// Device allocation owned by an [`Arc`] shared by every window into it.
struct CudaAlloc {
    ptr: *mut c_void,
}
// SAFETY: a device pointer is just an address; the CUDA runtime is thread safe.
unsafe impl Send for CudaAlloc {}
unsafe impl Sync for CudaAlloc {}
impl Drop for CudaAlloc {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `fllm_cuda_malloc` and is freed exactly once.
        unsafe { fllm_cuda_free(self.ptr) };
    }
}

fn dev_ptr(buf: &DeviceBuffer) -> Result<*mut u8, BackendError> {
    let alloc = buf
        .handle
        .downcast_ref::<CudaAlloc>()
        .ok_or_else(|| BackendError::BufferError("buffer does not belong to the CUDA backend".into()))?;
    // SAFETY: `offset` stays inside the allocation (checked by `DeviceBuffer::slice`).
    Ok(unsafe { (alloc.ptr as *mut u8).add(buf.offset) })
}

/// CUDA devices as inventory entries, numbered from `first_index`.
pub fn list_devices(first_index: usize) -> Vec<DeviceInfo> {
    // SAFETY: plain queries; a missing driver makes the count negative.
    let count = unsafe { fllm_cuda_device_count() };
    (0..count.max(0))
        .filter_map(|dev| {
            let mut name = [0 as c_char; 256];
            let (mut major, mut minor, mut sms) = (0, 0, 0);
            let (mut free, mut total) = (0usize, 0usize);
            // SAFETY: all pointers are valid for the sizes passed.
            let code = unsafe {
                fllm_cuda_device_info(dev, name.as_mut_ptr(), name.len(), &mut major, &mut minor, &mut free, &mut total, &mut sms)
            };
            if code != 0 {
                return None;
            }
            // SAFETY: `fllm_cuda_device_info` wrote a NUL-terminated name.
            let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
            Some(DeviceInfo {
                index: first_index + dev as usize,
                name,
                description: if major == 0 { format!("CUDA/ROCm, {sms} CUs") } else { format!("CUDA, sm_{major}{minor}, {sms} SMs") },
                dev_type: DeviceType::Gpu,
                free_memory_bytes: free as u64,
                total_memory_bytes: total as u64,
            })
        })
        .collect()
}

/// Weight formats the CUDA kernels read as they are stored in the GGUF file.
pub fn native_formats() -> Vec<String> {
    NATIVE_KINDS
        .iter()
        .map(|&k| format!("{:?}", QuantType::from_gguf_type_id(k)))
        .collect()
}

pub struct CudaBackend {
    device: i32,
    name: String,
    /// Blackwell (sm_12x): the FP4 tensor cores read NVFP4 weights directly.
    fp4: bool,
    /// Compute capability as major * 10 + minor (0 when unknown).
    cc: u32,
    /// AMD GPU (HIP build of the kernels): no compute capability, but the Q4Z kernels of hip/hip_q4z.cu.
    rocm: bool,
    capabilities: BackendCapabilities,
    next_id: AtomicU64,
    perm_cache: std::sync::Mutex<std::collections::HashMap<Vec<u8>, DeviceBuffer>>,
}

impl CudaBackend {
    pub fn new(device: usize) -> Result<Self, BackendError> {
        // SAFETY: plain query.
        let count = unsafe { fllm_cuda_device_count() };
        if count < 0 {
            return Err(BackendError::DeviceNotFound(format!("no CUDA driver: {}", last_error())));
        }
        if device >= count as usize {
            return Err(BackendError::DeviceNotFound(format!(
                "CUDA device {device} requested, {count} present"
            )));
        }
        let info = list_devices(0).into_iter().nth(device).ok_or_else(|| {
            BackendError::DeviceNotFound(format!("CUDA device {device} cannot be queried"))
        })?;
        let cc = info
            .description
            .split("sm_")
            .nth(1)
            .and_then(|t| t.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|t| t.parse::<u32>().ok())
            .unwrap_or(0);
        let mut supported_quants: Vec<QuantType> = NATIVE_KINDS.iter().map(|&k| QuantType::from_gguf_type_id(k)).collect();
        // AMD GPUs (the HIP build of the kernels lists them as "CUDA/ROCm"): 4-bit weights run through hip/hip_q4z.cu
        let rocm = info.description.starts_with("CUDA/ROCm");
        if cc >= 75 || rocm {
            // 4-bit AWQ / GPTQ weights run through the Marlin GEMM of vLLM
            supported_quants.extend([QuantType::Q4Z32, QuantType::Q4Z64, QuantType::Q4Z128]);
        }
        let capabilities = BackendCapabilities {
            name: info.name.clone(),
            supported_quants,
            supports_rms_norm: true,
            supports_layer_norm: true,
            supports_rope: true,
            supports_swiglu: true,
            supports_geglu: true,
            supports_paged_attention: true,
            supports_kv_cache_write: true,
            supports_embedding_lookup: true,
            supports_device_sampling: false, // host fallback
            supports_residual_add: true,
            supports_moe: false, // host fallback
            max_batch_size: 4096,
            max_context: 1 << 20,
        };
        Ok(Self {
            device: device as i32,
            fp4: info.description.starts_with("CUDA, sm_12"),
            cc,
            rocm,
            name: info.name,
            capabilities,
            next_id: AtomicU64::new(1),
            perm_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    fn alloc(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-pointer.
        check(unsafe { fllm_cuda_malloc(self.device, &mut ptr, size) }).map_err(|e| match e {
            BackendError::DriverError(msg) => {
                BackendError::OutOfMemory { requested: size, available: 0 }.also_log(&msg)
            }
            other => other,
        })?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceBuffer::new(id, size, self.device as usize, Arc::new(CudaAlloc { ptr })))
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_q4z(
        &self,
        name: &str,
        raw_bytes: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        group: usize,
        bias: Option<&[f32]>,
    ) -> Result<DeviceWeight, BackendError> {
        let groups = cols / group;
        let q4z_bytes = rows * groups * (group / 2 + 3);
        let (blocks_bytes, perm_buf) = if quant_type.is_q4zp() {
            let perm_bytes = cols * 4;
            if raw_bytes.len() != q4z_bytes + perm_bytes {
                return Err(invalid(format!(
                    "{name}: {} bytes are not {rows} x {cols} {quant_type:?} (expected {})",
                    raw_bytes.len(),
                    q4z_bytes + perm_bytes
                )));
            }
            let (b, p) = raw_bytes.split_at(q4z_bytes);
            let mut cache = self.perm_cache.lock().unwrap();
            let p_buf = if let Some(existing) = cache.get(p) {
                existing.clone()
            } else {
                let buf = self.alloc(perm_bytes)?;
                self.upload_sync(&buf, p)?;
                cache.insert(p.to_vec(), buf.clone());
                buf
            };
            (b, Some(p_buf))
        } else {
            if raw_bytes.len() != q4z_bytes {
                return Err(invalid(format!("{name}: {} bytes are not {rows} x {cols} {quant_type:?}", raw_bytes.len())));
            }
            (raw_bytes, None)
        };
        if bias.is_some_and(|b| b.len() != rows) {
            return Err(invalid(format!("{name}: bias does not have {rows} values")));
        }
        let (awq, scales, zeros) = super::q4z::marlin_arrays(blocks_bytes, rows, cols, group);
        let align = |v: usize| v.div_ceil(256) * 256;
        let qbytes = rows * cols / 2;
        let scale_at = align(qbytes);
        let zero_at = scale_at + align(groups * rows * 2);
        let buffer = self.alloc(zero_at + groups * rows / 2)?;
        {
            let staging = self.alloc(qbytes)?;
            // SAFETY: little-endian host; the u32 words are reinterpreted as their bytes.
            let awq_bytes = unsafe { std::slice::from_raw_parts(awq.as_ptr().cast::<u8>(), awq.len() * 4) };
            self.upload_sync(&staging, awq_bytes)?;
            // SAFETY: both buffers are live device allocations of the sizes the kernel reads and writes.
            check(unsafe {
                fllm_cuda_q4z_repack(self.device, dev_ptr(&staging)?.cast(), dev_ptr(&buffer)?.cast(), to_c(cols, "cols")?, to_c(rows, "rows")?)
            })?;
        }
        let scale_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload_sync(&buffer.slice(scale_at, scale_bytes.len())?, &scale_bytes)?;
        // SAFETY: little-endian host; the u32 words are reinterpreted as their bytes.
        let zero_bytes = unsafe { std::slice::from_raw_parts(zeros.as_ptr().cast::<u8>(), zeros.len() * 4) };
        self.upload_sync(&buffer.slice(zero_at, zero_bytes.len())?, zero_bytes)?;
        let bias = match bias {
            Some(b) => {
                let buf = self.alloc(b.len() * 4)?;
                let raw: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
                self.upload_sync(&buf, &raw)?;
                Some(buf)
            }
            None => None,
        };
        Ok(DeviceWeight {
            name: name.to_string(),
            buffer,
            quant_type,
            rows,
            cols,
            bias,
            scale: 1.0,
            perm: perm_buf,
        })
    }

    /// NVFP4 weights on GPUs without FP4 tensor cores: vLLM's Marlin W4A16 path (see `upload_q4z`): Marlin tiles, the scales in the
    /// S0E5M3 form, then one float, the per-tensor factor times 2^7.
    fn upload_nvfp4_marlin(
        &self,
        name: &str,
        raw_bytes: &[u8],
        rows: usize,
        cols: usize,
        scale: f32,
        bias: Option<&[f32]>,
    ) -> Result<DeviceWeight, BackendError> {
        if bias.is_some_and(|b| b.len() != rows) {
            return Err(invalid(format!("{name}: bias does not have {rows} values")));
        }
        log_once(
            "cuda-nvfp4-marlin",
            &format!("NVFP4 weights run through the Marlin W4A16 kernel on this GPU (first: {name})"),
        );
        let (words, scales) = super::q4z::nvfp4_marlin_arrays(raw_bytes, rows, cols);
        let align = |v: usize| v.div_ceil(256) * 256;
        let qbytes = rows * cols / 2;
        let scale_at = align(qbytes);
        let global_at = scale_at + align(cols / 16 * rows);
        let buffer = self.alloc(global_at + 16)?;
        {
            let staging = self.alloc(qbytes)?;
            // SAFETY: little-endian host; the u32 words are reinterpreted as their bytes.
            let word_bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4) };
            self.upload_sync(&staging, word_bytes)?;
            // SAFETY: both buffers are live device allocations of the sizes the kernel reads and writes.
            check(unsafe {
                fllm_cuda_gptq_repack(self.device, dev_ptr(&staging)?.cast(), dev_ptr(&buffer)?.cast(), to_c(cols, "cols")?, to_c(rows, "rows")?)
            })?;
        }
        self.upload_sync(&buffer.slice(scale_at, scales.len())?, &scales)?;
        let mut global = [0u8; 16];
        global[..4].copy_from_slice(&(scale * 128.0).to_le_bytes());
        self.upload_sync(&buffer.slice(global_at, 16)?, &global)?;
        let bias = match bias {
            Some(b) => {
                let buf = self.alloc(b.len() * 4)?;
                let raw: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
                self.upload_sync(&buf, &raw)?;
                Some(buf)
            }
            None => None,
        };
        Ok(DeviceWeight { name: name.to_string(), buffer, quant_type: QuantType::Nvfp4Marlin, rows, cols, bias, scale: 1.0, perm: None })
    }

    fn upload_fp8_marlin(
        &self,
        name: &str,
        raw_bytes: &[u8],
        rows: usize,
        cols: usize,
        bias: Option<&[f32]>,
    ) -> Result<DeviceWeight, BackendError> {
        let qbytes = rows * cols;
        let expected_bytes = qbytes + rows * 2;
        if raw_bytes.len() != expected_bytes {
            return Err(invalid(format!(
                "{name}: {} bytes are not {rows} x {cols} FP8M (expected {expected_bytes})",
                raw_bytes.len()
            )));
        }
        if bias.is_some_and(|b| b.len() != rows) {
            return Err(invalid(format!("{name}: bias does not have {rows} values")));
        }
        let (weight_bytes, scale_raw) = raw_bytes.split_at(qbytes);
        let words = super::q4z::fp8_marlin_words(weight_bytes, rows, cols);
        let scales = super::q4z::fp8_marlin_scales(scale_raw, rows);
        let align = |v: usize| v.div_ceil(256) * 256;
        let scale_at = align(qbytes);
        let buffer = self.alloc(scale_at + rows * 2)?;
        {
            let staging = self.alloc(qbytes)?;
            let words_bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4) };
            self.upload_sync(&staging, words_bytes)?;
            check(unsafe {
                fllm_cuda_fp8_marlin_repack(self.device, dev_ptr(&staging)?.cast(), dev_ptr(&buffer)?.cast(), to_c(cols, "cols")?, to_c(rows, "rows")?)
            })?;
        }
        let scale_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload_sync(&buffer.slice(scale_at, scale_bytes.len())?, &scale_bytes)?;
        let bias_buf = if let Some(b) = bias {
            let buf = self.alloc(rows * 4)?;
            let bytes = unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u8>(), rows * 4) };
            self.upload_sync(&buf, bytes)?;
            Some(buf)
        } else {
            None
        };
        Ok(DeviceWeight {
            name: name.to_string(),
            quant_type: QuantType::Fp8Marlin,
            rows,
            cols,
            buffer,
            bias: bias_buf,
            scale: 1.0,
            perm: None,
        })
    }

    fn upload_sync(&self, dst: &DeviceBuffer, src: &[u8]) -> Result<(), BackendError> {
        // SAFETY: `dst` holds at least `src.len()` bytes (callers size it from `src`).
        check(unsafe { fllm_cuda_h2d_sync(self.device, dev_ptr(dst)? as *mut c_void, src.as_ptr().cast(), src.len()) })
    }
}

trait AlsoLog {
    fn also_log(self, msg: &str) -> Self;
}
impl AlsoLog for BackendError {
    fn also_log(self, msg: &str) -> Self {
        log_once("cuda-oom", &format!("CUDA allocation failed: {msg}"));
        self
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_id(&self) -> usize {
        self.device as usize
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn create_stream(&self) -> Result<Box<dyn Stream>, BackendError> {
        // SAFETY: plain constructor.
        let raw = unsafe { fllm_cuda_stream_create(self.device) };
        if raw.is_null() {
            return Err(BackendError::DriverError(last_error()));
        }
        Ok(Box::new(CudaStream {
            raw,
            device: self.device,
            tables: HashMap::new(),
            cpu: CpuBackend::new(),
            cpu_stream: CpuStream,
            next_id: Arc::new(AtomicU64::new(1 << 40)),
            staging: None,
            staging_raw: None,
            staging_bias: None,
            pinned: std::collections::BTreeMap::new(),
            pin_failed: std::env::var_os("FLLM_NO_HOST_PIN").is_some(),
            step_open: false,
            blocks_checked: None,
        }))
    }

    fn allocate_buffer(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        self.alloc(size)
    }

    fn free_buffer(&self, _buffer: DeviceBuffer) -> Result<(), BackendError> {
        Ok(())
    }

    fn upload_weight(
        &self,
        name: &str,
        raw_bytes: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        bias: Option<&[f32]>,
    ) -> Result<DeviceWeight, BackendError> {
        let kind = quant_type.to_gguf_type_id();
        if quant_type == QuantType::Fp8Marlin {
            let ok = unsafe { fllm_cuda_fp8_marlin_supported(to_c(rows, "rows")?, to_c(cols, "cols")?) } != 0;
            if self.cc >= 75 && ok {
                return self.upload_fp8_marlin(name, raw_bytes, rows, cols, bias);
            }
            log_once(
                "cuda-fp8m-fallback",
                &format!("FP8M weights are expanded to F32 on this GPU (first: {name}); Marlin FP8 needs sm_75 and 128-aligned shapes"),
            );
        }
        if let Some(group) = quant_type.q4z_group() {
            // SAFETY: plain shape check.
            let ok = unsafe { fllm_cuda_q4z_supported(to_c(rows, "rows")?, to_c(cols, "cols")?, group as c_int) } != 0;
            if (self.cc >= 75 || self.rocm) && ok {
                return self.upload_q4z(name, raw_bytes, quant_type, rows, cols, group, bias);
            }
            log_once(
                "cuda-q4z-fallback",
                &format!("4-bit {quant_type:?} weights are expanded to F32 on this GPU (first: {name}); the Marlin GEMM needs sm_75 and 128-aligned shapes"),
            );
        }
        let converted: Vec<u8>;
        let (bytes, quant_type): (&[u8], QuantType) = if (kind == 20 && cols.is_multiple_of(32)) || (kind == 23 && cols.is_multiple_of(256)) {
            // IQ4_NL / IQ4_XS run natively as 32-value blocks with an fp16 scale and a 16-entry grid (IQ4_NL layout)
            let blocks = rows * cols / 32;
            let nl = if kind == 20 {
                raw_bytes.to_vec()
            } else {
                iq4_xs_to_nl(raw_bytes, rows * cols / 256)
            };
            if nl.len() != blocks * 18 {
                return Err(invalid(format!("{name}: {} bytes are not {rows} x {cols} {quant_type:?}", raw_bytes.len())));
            }
            converted = legacy_device_layout(&nl, rows, cols, 2);
            (&converted, QuantType::IQ4_NL)
        } else if kind == 39 && cols.is_multiple_of(32) {
            // MXFP4 decodes through the 16-entry grid of the IQ4_NL kernels: blocks become `[f16 scale][16 quant bytes]`
            if raw_bytes.len() != rows * (cols / 32) * 17 {
                return Err(invalid(format!("{name}: {} bytes are not {rows} x {cols} MXFP4", raw_bytes.len())));
            }
            converted = legacy_device_layout(&mxfp4_to_nl(raw_bytes), rows, cols, 2);
            (&converted, QuantType::Mxfp4)
        } else if NATIVE_KINDS.contains(&kind) || (kind == 40 && self.fp4) {
            let (block, width) = (quant_type.block_size(), quant_type.block_bytes());
            if !cols.is_multiple_of(block) || raw_bytes.len() != rows * (cols / block) * width {
                return Err(invalid(format!(
                    "{name}: {} bytes are not {rows} x {cols} {quant_type:?}",
                    raw_bytes.len()
                )));
            }
            if kind == 14 {
                // Q6_K super-blocks (210 bytes) are padded to 224 so every field is 16-byte aligned
                // for the asynchronous copies of the GEMM kernels.
                let blocks = raw_bytes.len() / 210;
                let mut padded = vec![0u8; blocks * 224];
                for b in 0..blocks {
                    padded[b * 224..b * 224 + 210].copy_from_slice(&raw_bytes[b * 210..(b + 1) * 210]);
                }
                converted = padded;
                (&converted, quant_type)
            } else if matches!(kind, 2 | 3 | 6 | 7) && cols.is_multiple_of(32) {
                converted = legacy_device_layout(raw_bytes, rows, cols, kind);
                (&converted, quant_type)
            } else if kind == 8 && cols.is_multiple_of(32) {
                converted = q8_0_device_layout(raw_bytes, rows, cols);
                (&converted, quant_type)
            } else if kind == 12 && cols.is_multiple_of(64) && rows.is_multiple_of(256) && self.cc >= 80 && is_marlin_q4k_enabled() {
                let marlin_bytes = q4_k_marlin_device_layout(raw_bytes, rows, cols);
                let mut dual = Vec::with_capacity(raw_bytes.len() + marlin_bytes.len());
                dual.extend_from_slice(raw_bytes);
                dual.extend_from_slice(&marlin_bytes);
                converted = dual;
                (&converted, quant_type)
            } else if kind == 40 && cols.is_multiple_of(256) && rows.is_multiple_of(16) {
                converted = nvfp4_device_layout(raw_bytes, rows, cols);
                (&converted, quant_type)
            } else {
                (raw_bytes, quant_type)
            }
        } else if cols.is_multiple_of(256) {
            // Formats without a CUDA decoder (Q2_K, Q3_K, the IQ family ...) are re-quantized once to Q5_K: 5.5 bits per
            // weight, an order of magnitude finer than the 2-4 bit grids they came from, and served by the fast kernels.
            use rayon::prelude::*;
            let values = crate::compute::dequantize(raw_bytes, kind, rows * cols)
                .map_err(|e| BackendError::UnsupportedOperation(format!("{name}: {e}")))?;
            let row_bytes = cols / 256 * 176;
            let mut out = vec![0u8; rows * row_bytes];
            out.par_chunks_mut(row_bytes).zip(values.par_chunks(cols)).for_each(|(dst, src)| {
                let mut row = Vec::with_capacity(row_bytes);
                fllm_checkpoint::quantize_row(fllm_checkpoint::QType::Q5K, src, &mut row);
                dst.copy_from_slice(&row);
            });
            log_once(
                &format!("cuda-requant-{kind}"),
                &format!(
                    "{quant_type:?} weights have no CUDA decoder; re-quantized to Q5_K on upload (first: {name}, {} MiB)",
                    out.len() >> 20
                ),
            );
            converted = out;
            (&converted, QuantType::Q5_K)
        } else {
            // One-time conversion; costs 4 bytes per weight instead of the quantized size.
            let values = crate::compute::dequantize(raw_bytes, kind, rows * cols)
                .map_err(|e| BackendError::UnsupportedOperation(format!("{name}: {e}")))?;
            log_once(
                &format!("cuda-dequant-{kind}"),
                &format!(
                    "{quant_type:?} weights have no CUDA decoder; converted to F32 on upload \
                     (first: {name}, {} MiB)",
                    (values.len() * 4) >> 20
                ),
            );
            converted = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            (&converted, QuantType::F32)
        };
        let buffer = self.alloc(bytes.len())?;
        self.upload_sync(&buffer, bytes)?;
        let bias = match bias {
            Some(b) if b.len() != rows => {
                return Err(invalid(format!("{name}: bias has {} values for {rows} rows", b.len())))
            }
            Some(b) => {
                let buf = self.alloc(b.len() * 4)?;
                let raw: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
                self.upload_sync(&buf, &raw)?;
                Some(buf)
            }
            None => None,
        };
        Ok(DeviceWeight {
            name: name.to_string(),
            buffer,
            quant_type,
            rows,
            cols,
            bias,
            scale: 1.0,
            perm: None,
        })
    }

    fn upload_weight_scaled(
        &self,
        name: &str,
        raw_bytes: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        bias: Option<&[f32]>,
        scale: f32,
    ) -> Result<DeviceWeight, BackendError> {
        if scale == 1.0 && quant_type != QuantType::NvFp4 {
            return self.upload_weight(name, raw_bytes, quant_type, rows, cols, bias);
        }
        if quant_type == QuantType::NvFp4 && self.cc >= 75 && (!self.fp4 || std::env::var_os("FLLM_NVFP4_MARLIN").is_some()) {
            // SAFETY: plain shape check.
            let ok = unsafe { fllm_cuda_q4z_supported(to_c(rows, "rows")?, to_c(cols, "cols")?, 16) } != 0;
            if ok && raw_bytes.len() == rows * (cols / 64) * 36 {
                return self.upload_nvfp4_marlin(name, raw_bytes, rows, cols, scale, bias);
            }
        }
        if quant_type == QuantType::NvFp4 && self.fp4 && cols.is_multiple_of(256) && rows.is_multiple_of(16) && raw_bytes.len() == rows * (cols / 64) * 36 {
            let mut weight = self.upload_weight(name, raw_bytes, quant_type, rows, cols, bias)?;
            weight.scale = scale;
            return Ok(weight);
        }
        // no FP4 tensor cores here: fold the scale into half-precision weights
        let values = crate::compute::dequantize(raw_bytes, quant_type.to_gguf_type_id(), rows * cols)
            .map_err(|e| BackendError::UnsupportedOperation(format!("{name}: {e}")))?;
        log_once(
            "cuda-nvfp4-fallback",
            &format!("NVFP4 weights are expanded to F16 on this GPU (first: {name}); FP4 tensor cores need sm_120"),
        );
        let bytes: Vec<u8> = values.iter().flat_map(|v| f32_to_f16_bits(v * scale).to_le_bytes()).collect();
        self.upload_weight(name, &bytes, QuantType::F16, rows, cols, bias)
    }

    fn upload_bias_table(&self, weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
        let mut table = Vec::with_capacity(weights.len() * 8);
        for w in weights {
            let bias = w.bias.as_ref().ok_or_else(|| invalid("upload_bias_table: weight without a bias"))?;
            table.extend_from_slice(&(dev_ptr(bias)? as usize as u64).to_le_bytes());
        }
        let buf = self.alloc(table.len().max(8))?;
        self.upload_sync(&buf, &table)?;
        Ok(buf)
    }

    fn upload_pointer_table(&self, weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
        let mut table = Vec::with_capacity(weights.len() * 8);
        for w in weights {
            table.extend_from_slice(&(dev_ptr(&w.buffer)? as usize as u64).to_le_bytes());
        }
        let buf = self.alloc(table.len().max(8))?;
        self.upload_sync(&buf, &table)?;
        Ok(buf)
    }

    #[allow(clippy::too_many_arguments)]
    fn allocate_kv_cache(
        &self,
        layers: usize,
        pool_blocks: usize,
        block_size: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        kv_kind: i32,
    ) -> Result<DeviceKvCache, BackendError> {
        let (words_per_layer_k, words_per_layer_v) =
            DeviceKvCache::geometry(layers, pool_blocks, block_size, kv_heads, head_dim, value_dim, kv_kind)?;
        let key_pool = self.alloc((layers * words_per_layer_k * 2).div_ceil(4) * 4)?;
        let value_pool = self.alloc((layers * words_per_layer_v * 2).div_ceil(4) * 4)?;
        for pool in [&key_pool, &value_pool] {
            // SAFETY: the pool is a live allocation of `pool.size` bytes.
            check(unsafe { fllm_cuda_memset(self.device, dev_ptr(pool)? as *mut c_void, 0, pool.size) })?;
        }
        Ok(DeviceKvCache {
            key_pool,
            value_pool,
            kv_kind,
            layers,
            pool_blocks,
            block_size,
            kv_heads,
            head_dim,
            value_dim,
            words_per_layer_k,
            words_per_layer_v,
        })
    }
}

pub struct CudaStream {
    raw: *mut RawStream,
    device: i32,
    /// Device copies of constant host tables (rope cos/sin, sinks, ALiBi slopes), by `Arc` address.
    tables: HashMap<usize, (Arc<Vec<f32>>, DeviceBuffer)>,
    cpu: CpuBackend,
    cpu_stream: CpuStream,
    next_id: Arc<AtomicU64>,
    /// Device buffers that weights streamed from host memory pass through (`host_linear`): the weight in its
    /// device layout, and the raw GGUF bytes for formats that are converted on the device.
    staging: Option<DeviceBuffer>,
    staging_raw: Option<DeviceBuffer>,
    staging_bias: Option<DeviceBuffer>,
    /// Page ranges (start -> end) of host weight memory that is page-locked for fast copies.
    pinned: std::collections::BTreeMap<usize, usize>,
    pin_failed: bool,
    /// A forward pass is running (`begin_step`) and the library caches the host data of its attention calls.
    step_open: bool,
    /// Block tables (address, length) and lengths (address, length) whose blocks were walked by `attention_paged` in this step.
    blocks_checked: Option<[usize; 4]>,
}

// SAFETY: the raw stream is only used through `&mut self`.
unsafe impl Send for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        for &start in self.pinned.keys() {
            // SAFETY: only ranges registered by `pin` are released.
            unsafe { fllm_cuda_host_unregister(start as *mut c_void) };
        }
        // SAFETY: `raw` is owned by this struct.
        unsafe { fllm_cuda_stream_destroy(self.raw) };
    }
}

impl CudaStream {
    /// Page-lock the pages of `raw` that are not locked yet (read-only file mappings qualify). Failure disables pinning:
    /// copies from pageable memory work too, at about half the speed.
    fn pin(&mut self, raw: &[u8]) {
        use std::ops::Bound::Excluded;
        if self.pin_failed || raw.is_empty() {
            return;
        }
        const PAGE: usize = 4096;
        let (start, end) = (raw.as_ptr() as usize, raw.as_ptr() as usize + raw.len());
        // Model tensors (megabytes of a read-only file mapping) are rounded outwards so that one range covers the whole tensor. A
        // small buffer (a heap vector) shares its first and last page with other allocations: only the pages it fills completely
        // may be locked read-only, otherwise a later device-to-host copy into a neighbour fails with "invalid argument".
        let (a, b) = if raw.len() >= 1 << 20 {
            (start & !(PAGE - 1), (end + PAGE - 1) & !(PAGE - 1))
        } else {
            ((start + PAGE - 1) & !(PAGE - 1), end & !(PAGE - 1))
        };
        if a >= b {
            return;
        }
        let mut cur = a;
        if let Some((_, &end)) = self.pinned.range(..=a).next_back() {
            cur = cur.max(end);
        }
        let mut pieces = Vec::new();
        for (&s, &e) in self.pinned.range((Excluded(a), Excluded(b))) {
            if s > cur {
                pieces.push((cur, s));
            }
            cur = cur.max(e);
        }
        if cur < b {
            pieces.push((cur, b));
        }
        for (s, e) in pieces {
            // SAFETY: the range lies inside the caller's live mapping (page-rounded outwards, never unmapped while the
            // model is loaded).
            if unsafe { fllm_cuda_host_register(self.device, s as *mut c_void, e - s) } != 0 {
                self.pin_failed = true;
                return;
            }
            self.pinned.insert(s, e);
        }
    }

    fn alloc(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointer.
        check(unsafe { fllm_cuda_malloc(self.device, &mut ptr, size) })?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceBuffer::new(id, size, self.device as usize, Arc::new(CudaAlloc { ptr })))
    }

    /// Device copy of a constant table, uploaded once per `Arc`.
    fn table(&mut self, values: &Arc<Vec<f32>>) -> Result<*const f32, BackendError> {
        let key = Arc::as_ptr(values) as usize;
        if let Some((_, buf)) = self.tables.get(&key) {
            return Ok(dev_ptr(buf)? as *const f32);
        }
        let buf = self.alloc(values.len().max(1) * 4)?;
        // SAFETY: `buf` holds `values.len() * 4` bytes.
        check(unsafe {
            fllm_cuda_upload(self.raw, dev_ptr(&buf)?.cast(), values.as_ptr().cast(), values.len() * 4)
        })?;
        // pageable host memory is staged before the call returns; the Arc keeps the table alive.
        let ptr = dev_ptr(&buf)? as *const f32;
        self.tables.insert(key, (Arc::clone(values), buf));
        Ok(ptr)
    }

    fn copy_to_host(&mut self, buf: &DeviceBuffer) -> Result<DeviceBuffer, BackendError> {
        let mut bytes = vec![0u8; buf.size];
        self.download(&mut bytes, buf)?;
        let mut host = self.cpu.allocate_buffer(buf.size)?;
        self.cpu_stream.upload(&mut host, &bytes)?;
        Ok(host)
    }

    fn weight_to_host(&mut self, w: &DeviceWeight) -> Result<DeviceWeight, BackendError> {
        Ok(DeviceWeight {
            buffer: self.copy_to_host(&w.buffer)?,
            bias: w.bias.as_ref().map(|b| self.copy_to_host(b)).transpose()?,
            perm: w.perm.as_ref().map(|p| self.copy_to_host(p)).transpose()?,
            scale: 1.0,
            ..w.clone()
        })
    }

    fn copy_from_host(&mut self, host: &DeviceBuffer, dst: &mut DeviceBuffer) -> Result<(), BackendError> {
        let mut bytes = vec![0u8; dst.size.min(host.size)];
        self.cpu_stream.download(&mut bytes, host)?;
        self.upload(dst, &bytes)
    }

    fn glu(&mut self, out: &mut DeviceBuffer, gate: &DeviceBuffer, up: &DeviceBuffer, mode: c_int) -> Result<(), BackendError> {
        if gate.size != up.size || !gate.size.is_multiple_of(4) || out.size < gate.size {
            return Err(invalid("gated activation operands differ in size"));
        }
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_glu(self.raw, dev_ptr(gate)?.cast(), dev_ptr(up)?.cast(), dev_ptr(out)?.cast(), gate.size / 4, mode)
        })
    }
}

impl Stream for CudaStream {
    fn begin_step(&mut self) {
        // SAFETY: plain call on a live stream.
        let ok = unsafe { fllm_cuda_step_begin(self.raw) } == 0;
        self.step_open = ok && prefill_switch(PrefillSwitch::StepCache) != 0;
        self.blocks_checked = None;
    }

    fn end_step(&mut self) {
        // SAFETY: plain call on a live stream.
        unsafe { fllm_cuda_step_end(self.raw) };
        self.step_open = false;
        self.blocks_checked = None;
    }

    fn synchronize(&mut self) -> Result<(), BackendError> {
        // SAFETY: live stream.
        check(unsafe { fllm_cuda_stream_sync(self.raw) })
    }

    fn upload(&mut self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<(), BackendError> {
        if src.len() > dst.size {
            return Err(BackendError::BufferError(format!(
                "upload of {} bytes into a {} byte buffer",
                src.len(),
                dst.size
            )));
        }
        // SAFETY: bounds checked above.
        check(unsafe { fllm_cuda_upload(self.raw, dev_ptr(dst)?.cast(), src.as_ptr().cast(), src.len()) })
    }

    fn download(&mut self, dst: &mut [u8], src: &DeviceBuffer) -> Result<(), BackendError> {
        if dst.len() > src.size {
            return Err(BackendError::BufferError(format!(
                "download of {} bytes from a {} byte buffer",
                dst.len(),
                src.size
            )));
        }
        // SAFETY: bounds checked above; the call returns after the copy completed.
        check(unsafe { fllm_cuda_download(self.raw, dst.as_mut_ptr().cast(), dev_ptr(src)?.cast(), dst.len()) })
    }

    fn copy_buffer(&mut self, dst: &mut DeviceBuffer, src: &DeviceBuffer) -> Result<(), BackendError> {
        let bytes = dst.size.min(src.size);
        // SAFETY: both buffers hold at least `bytes`.
        check(unsafe { fllm_cuda_copy(self.raw, dev_ptr(dst)?.cast(), dev_ptr(src)?.cast(), bytes) })
    }

    fn embedding_lookup(&mut self, out: &mut DeviceBuffer, ids: &[i64], weight: &DeviceWeight) -> Result<(), BackendError> {
        let kind = weight.quant_type.to_gguf_type_id();
        if !NATIVE_KINDS.contains(&kind) && kind != 20 && kind != 39 {
            return Err(BackendError::UnsupportedQuant(weight.quant_type));
        }
        if let Some(&id) = ids.iter().find(|&&id| id < 0 || id as usize >= weight.rows) {
            return Err(invalid(format!("embedding token id {id} out of bounds")));
        }
        if ids.is_empty() {
            return Ok(());
        }
        if out.size < ids.len() * weight.cols * 4 {
            return Err(invalid("embedding output smaller than ids x hidden"));
        }
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_embedding(
                self.raw,
                dev_ptr(&weight.buffer)?.cast(),
                kind as c_int,
                ids.as_ptr(),
                to_c(ids.len(), "token count")?,
                dev_ptr(out)?.cast(),
                to_c(weight.cols, "hidden size")?,
            )
        })
    }

    fn rms_norm(&mut self, out: &mut DeviceBuffer, x: &DeviceBuffer, weight: &DeviceBuffer, eps: f32) -> Result<(), BackendError> {
        let cols = weight.size / 4;
        if cols == 0 || !x.size.is_multiple_of(cols * 4) || out.size < x.size {
            return Err(invalid("rms_norm input is not rows of the weight width"));
        }
        let rows = x.size / (cols * 4);
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_rms_norm(self.raw, dev_ptr(x)?.cast(), dev_ptr(weight)?.cast(), dev_ptr(out)?.cast(), to_c(rows, "rows")?, to_c(cols, "width")?, eps)
        })
    }

    fn layer_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        eps: f32,
    ) -> Result<(), BackendError> {
        let cols = weight.size / 4;
        if cols == 0 || !x.size.is_multiple_of(cols * 4) || out.size < x.size {
            return Err(invalid("layer_norm input is not rows of the weight width"));
        }
        if bias.is_some_and(|b| b.size < cols * 4) {
            return Err(invalid("layer_norm bias width differs from weight"));
        }
        let rows = x.size / (cols * 4);
        let bias_ptr = match bias {
            Some(b) => dev_ptr(b)? as *const f32,
            None => std::ptr::null(),
        };
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_layer_norm(self.raw, dev_ptr(x)?.cast(), dev_ptr(weight)?.cast(), bias_ptr, dev_ptr(out)?.cast(), to_c(rows, "rows")?, to_c(cols, "width")?, eps)
        })
    }

    fn rope(&mut self, q: &mut DeviceBuffer, k: &mut DeviceBuffer, positions: &[usize], params: &RopeParams) -> Result<(), BackendError> {
        params.validate(positions, q.size, k.size)?;
        if positions.is_empty() {
            return Ok(());
        }
        let cos = self.table(&params.cos)?;
        let sin = self.table(&params.sin)?;
        let pos: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        // SAFETY: `validate` checked table coverage and buffer sizes.
        check(unsafe {
            fllm_cuda_rope(
                self.raw,
                dev_ptr(q)?.cast(),
                dev_ptr(k)?.cast(),
                pos.as_ptr(),
                to_c(positions.len(), "rows")?,
                cos,
                sin,
                to_c(params.heads, "heads")?,
                to_c(params.kv_heads, "kv heads")?,
                to_c(params.dim, "head width")?,
                to_c(params.rope_dim / 2, "rope width")?,
                c_int::from(params.neox),
            )
        })
    }

    fn moe_topk(
        &mut self,
        logits: &DeviceBuffer,
        ids: &mut DeviceBuffer,
        weights: &mut DeviceBuffer,
        rows: usize,
        experts: usize,
        k: usize,
        normalize: bool,
        scale: f32,
    ) -> Result<(), BackendError> {
        if logits.size < rows * experts * 4 || ids.size < rows * k * 4 || weights.size < rows * k * 4 {
            return Err(invalid("moe_topk: buffers too small"));
        }
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_moe_route(
                self.raw,
                dev_ptr(logits)?.cast(),
                dev_ptr(ids)?.cast(),
                dev_ptr(weights)?.cast(),
                to_c(rows, "rows")?,
                to_c(experts, "experts")?,
                to_c(k, "k")?,
                c_int::from(normalize),
                scale,
            )
        })
    }

    fn swiglu_oai(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        alpha: f32,
        limit: f32,
    ) -> Result<(), BackendError> {
        let count = gate.size / 4;
        if up.size < count * 4 || out.size < count * 4 {
            return Err(invalid("swiglu_oai: buffers too small"));
        }
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_swiglu_oai(self.raw, dev_ptr(gate)?.cast(), dev_ptr(up)?.cast(), dev_ptr(out)?.cast(), count, alpha, limit)
        })
    }

    fn moe_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        table: &DeviceBuffer,
        bias_table: Option<&DeviceBuffer>,
        ids: &DeviceBuffer,
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        slots: usize,
        x_div: usize,
    ) -> Result<(), BackendError> {
        let kind = quant_type.to_gguf_type_id();
        let k_quant = matches!(kind, 12..=14);
        if !(k_quant || matches!(kind, 2 | 3 | 6 | 7 | 8 | 20 | 39)) || !cols.is_multiple_of(if k_quant { 256 } else { 32 }) || x_div == 0 {
            return Err(BackendError::UnsupportedQuant(quant_type));
        }
        let inputs = slots.div_ceil(x_div);
        if x.size < inputs * cols * 4 || out.size < slots * rows * 4 || ids.size < slots * 4 {
            return Err(invalid("moe_linear: buffers too small"));
        }
        // SAFETY: sizes checked above; the table was built by `upload_pointer_table` for weights of `quant_type`.
        check(unsafe {
            fllm_cuda_moe_linear(
                self.raw,
                dev_ptr(table)?.cast(),
                match bias_table {
                    Some(b) => dev_ptr(b)?.cast(),
                    None => std::ptr::null(),
                },
                kind as c_int,
                dev_ptr(ids)?.cast(),
                dev_ptr(x)?.cast(),
                dev_ptr(out)?.cast(),
                to_c(rows, "rows")?,
                to_c(cols, "cols")?,
                to_c(slots, "slots")?,
                to_c(x_div, "x_div")?,
            )
        })
    }

    fn moe_combine(
        &mut self,
        out: &mut DeviceBuffer,
        y: &DeviceBuffer,
        weights: &DeviceBuffer,
        tokens: usize,
        k: usize,
        hidden: usize,
    ) -> Result<(), BackendError> {
        if out.size < tokens * hidden * 4 || y.size < tokens * k * hidden * 4 || weights.size < tokens * k * 4 {
            return Err(invalid("moe_combine: buffers too small"));
        }
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_moe_combine(
                self.raw,
                dev_ptr(out)?.cast(),
                dev_ptr(y)?.cast(),
                dev_ptr(weights)?.cast(),
                to_c(tokens, "tokens")?,
                to_c(k, "k")?,
                to_c(hidden, "hidden")?,
            )
        })
    }

    fn gather_rows(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
        idx: &[i32],
        weights: Option<&[f32]>,
        row_len: usize,
    ) -> Result<(), BackendError> {
        let n = idx.len();
        if n == 0 {
            return Ok(());
        }
        if weights.is_some_and(|w| w.len() != n) || dst.size < n * row_len * 4 {
            return Err(invalid("gather_rows: operand sizes do not match"));
        }
        let max = idx.iter().copied().max().unwrap_or(0);
        if max < 0 || idx.iter().any(|&i| i < 0) || (max as usize + 1) * row_len * 4 > src.size {
            return Err(invalid("gather_rows: row index out of range"));
        }
        // SAFETY: bounds checked above; the index arrays are staged by the call.
        check(unsafe {
            fllm_cuda_gather_rows(
                self.raw,
                dev_ptr(dst)?.cast(),
                dev_ptr(src)?.cast(),
                idx.as_ptr(),
                weights.map_or(std::ptr::null(), |w| w.as_ptr()),
                to_c(n, "rows")?,
                to_c(row_len, "row length")?,
            )
        })
    }

    fn scatter_add_rows(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
        idx: &[i32],
        weights: Option<&[f32]>,
        row_len: usize,
    ) -> Result<(), BackendError> {
        let n = idx.len();
        if n == 0 {
            return Ok(());
        }
        if weights.is_some_and(|w| w.len() != n) || src.size < n * row_len * 4 {
            return Err(invalid("scatter_add_rows: operand sizes do not match"));
        }
        let max = idx.iter().copied().max().unwrap_or(0);
        if max < 0 || idx.iter().any(|&i| i < 0) || (max as usize + 1) * row_len * 4 > dst.size {
            return Err(invalid("scatter_add_rows: row index out of range"));
        }
        // SAFETY: bounds checked above; the index arrays are staged by the call.
        check(unsafe {
            fllm_cuda_scatter_add_rows(
                self.raw,
                dev_ptr(dst)?.cast(),
                dev_ptr(src)?.cast(),
                idx.as_ptr(),
                weights.map_or(std::ptr::null(), |w| w.as_ptr()),
                to_c(n, "rows")?,
                to_c(row_len, "row length")?,
            )
        })
    }

    fn zero_buffer(&mut self, buf: &mut DeviceBuffer) -> Result<(), BackendError> {
        // SAFETY: the buffer owns `size` bytes at its pointer.
        check(unsafe { fllm_cuda_zero(self.raw, dev_ptr(buf)?.cast(), buf.size) })
    }

    fn supports_host_linear(&self, quant_type: QuantType) -> bool {
        matches!(quant_type.to_gguf_type_id(), 8 | 12 | 13 | 14 | 39)
    }

    fn host_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        raw: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        bias: Option<&[f32]>,
    ) -> Result<(), BackendError> {
        let kind = quant_type.to_gguf_type_id();
        if !self.supports_host_linear(quant_type) || !cols.is_multiple_of(if matches!(kind, 8 | 39) { 32 } else { 256 }) {
            return Err(BackendError::UnsupportedQuant(quant_type));
        }
        let (block, width) = (quant_type.block_size(), quant_type.block_bytes());
        if raw.len() != rows * (cols / block) * width {
            return Err(invalid(format!("host weight of {} bytes is not {rows} x {cols} {quant_type:?}", raw.len())));
        }
        self.pin(raw);
        let converted = matches!(kind, 14 | 8 | 39);
        let device_bytes = match kind {
            14 => raw.len() / 210 * 224,
            8 => rows * padded_blocks(cols) * 34,
            39 => rows * padded_blocks(cols) * 18,
            _ => raw.len(),
        };
        let need = |buf: &Option<DeviceBuffer>, bytes: usize| buf.as_ref().is_none_or(|b| b.size < bytes);
        if need(&self.staging, device_bytes) {
            // the freed buffer may still be read by queued kernels: cudaFree waits for them
            self.staging = None;
            self.staging = Some(self.alloc((device_bytes + device_bytes / 4 + 255) & !255)?);
        }
        let staging = self.staging.as_ref().ok_or_else(|| invalid("staging buffer missing"))?.slice(0, device_bytes)?;
        if converted {
            if need(&self.staging_raw, raw.len()) {
                self.staging_raw = None;
                self.staging_raw = Some(self.alloc((raw.len() + raw.len() / 4 + 255) & !255)?);
            }
            let tmp = self.staging_raw.as_ref().ok_or_else(|| invalid("staging buffer missing"))?;
            // SAFETY: `tmp` holds at least `raw.len()` bytes; pageable memory is staged before the call returns.
            check(unsafe { fllm_cuda_upload(self.raw, dev_ptr(tmp)?.cast(), raw.as_ptr().cast(), raw.len()) })?;
            // SAFETY: both buffers cover the converted weight.
            check(unsafe {
                fllm_cuda_repack(self.raw, dev_ptr(&staging)?.cast(), dev_ptr(tmp)?.cast(), kind as c_int, rows, cols)
            })?;
        } else {
            // SAFETY: `staging` holds `raw.len()` bytes.
            check(unsafe { fllm_cuda_upload(self.raw, dev_ptr(&staging)?.cast(), raw.as_ptr().cast(), raw.len()) })?;
        }
        let weight = DeviceWeight {
            name: String::from("host-streamed"),
            buffer: staging,
            quant_type,
            rows,
            cols,
            bias: None,
            scale: 1.0,
            perm: None,
        };
        let bias_buf = match bias {
            Some(b) if b.len() == rows => {
                if need(&self.staging_bias, rows * 4) {
                    self.staging_bias = None;
                    self.staging_bias = Some(self.alloc((rows * 4 + 255) & !255)?);
                }
                let buf = self.staging_bias.as_ref().ok_or_else(|| invalid("staging buffer missing"))?.slice(0, rows * 4)?;
                // SAFETY: `buf` holds `rows` floats; pageable memory is staged before the call returns.
                check(unsafe { fllm_cuda_upload(self.raw, dev_ptr(&buf)?.cast(), b.as_ptr().cast(), rows * 4) })?;
                Some(buf)
            }
            Some(_) => return Err(invalid("host_linear: bias length does not match the rows")),
            None => None,
        };
        self.quantized_linear(out, x, &weight, bias_buf.as_ref())
    }

    fn quantized_linear(&mut self, out: &mut DeviceBuffer, x: &DeviceBuffer, weight: &DeviceWeight, bias: Option<&DeviceBuffer>) -> Result<(), BackendError> {
        let kind = weight.quant_type.to_gguf_type_id();
        if weight.quant_type == QuantType::Fp8Marlin {
            let (rows, cols) = (weight.rows, weight.cols);
            if rows == 0 || cols == 0 || !x.size.is_multiple_of(cols * 4) {
                return Err(invalid("linear input dimensions do not match"));
            }
            let batch = x.size / (cols * 4);
            if batch == 0 {
                return Ok(());
            }
            if out.size < batch * rows * 4 {
                return Err(invalid("linear output smaller than batch x rows"));
            }
            if bias.is_some_and(|b| b.size < rows * 4) {
                return Err(invalid("linear bias shorter than the output rows"));
            }
            let bias_ptr = match bias {
                Some(b) => dev_ptr(b)? as *const f32,
                None => std::ptr::null(),
            };
            return check(unsafe {
                fllm_cuda_linear_fp8_marlin(
                    self.raw,
                    dev_ptr(&weight.buffer)?.cast(),
                    dev_ptr(x)?.cast(),
                    dev_ptr(out)?.cast(),
                    bias_ptr,
                    to_c(rows, "rows")?,
                    to_c(cols, "cols")?,
                    to_c(batch, "batch")?,
                )
            });
        }
        if let Some(group) = weight.quant_type.q4z_group() {
            let (rows, cols) = (weight.rows, weight.cols);
            if rows == 0 || cols == 0 || !x.size.is_multiple_of(cols * 4) {
                return Err(invalid("linear input dimensions do not match"));
            }
            let batch = x.size / (cols * 4);
            if batch == 0 {
                return Ok(());
            }
            if out.size < batch * rows * 4 || bias.is_some_and(|b| b.size < rows * 4) {
                return Err(invalid("linear output or bias smaller than batch x rows"));
            }
            let bias_ptr = match bias {
                Some(b) => dev_ptr(b)? as *const f32,
                None => std::ptr::null(),
            };
            let perm_ptr = match &weight.perm {
                Some(p) => dev_ptr(p)? as *const c_int,
                None => std::ptr::null(),
            };
            // SAFETY: sizes checked above; the weight was uploaded by `upload_q4z`.
            return check(unsafe {
                fllm_cuda_linear_q4z(
                    self.raw,
                    dev_ptr(&weight.buffer)?.cast(),
                    group as c_int,
                    dev_ptr(x)?.cast(),
                    dev_ptr(out)?.cast(),
                    bias_ptr,
                    to_c(rows, "rows")?,
                    to_c(cols, "cols")?,
                    to_c(batch, "batch")?,
                    perm_ptr,
                )
            });
        }
        if !NATIVE_KINDS.contains(&kind) && kind != 40 && kind != 20 && kind != 39 && kind != 41 {
            return Err(BackendError::UnsupportedQuant(weight.quant_type));
        }
        let (rows, cols) = (weight.rows, weight.cols);
        if rows == 0 || cols == 0 || !x.size.is_multiple_of(cols * 4) {
            return Err(invalid("linear input dimensions do not match"));
        }
        let batch = x.size / (cols * 4);
        if batch == 0 {
            return Ok(());
        }
        if out.size < batch * rows * 4 {
            return Err(invalid("linear output smaller than batch x rows"));
        }
        if bias.is_some_and(|b| b.size < rows * 4) {
            return Err(invalid("linear bias shorter than the output rows"));
        }
        let bias_ptr = match bias {
            Some(b) => dev_ptr(b)? as *const f32,
            None => std::ptr::null(),
        };
        if kind == 40 {
            // SAFETY: sizes checked above; the weight was uploaded by `upload_weight_scaled`.
            return check(unsafe {
                fllm_cuda_linear_nvfp4(
                    self.raw,
                    dev_ptr(&weight.buffer)?.cast(),
                    dev_ptr(x)?.cast(),
                    dev_ptr(out)?.cast(),
                    bias_ptr,
                    weight.scale,
                    to_c(rows, "rows")?,
                    to_c(cols, "cols")?,
                    to_c(batch, "batch")?,
                )
            });
        }
        let (weight_ptr, effective_kind) = if kind == 12 && batch >= 16 && !is_exact_math() {
            let raw_bytes_len = rows * (cols / 256) * 144;
            let marlin_bytes_len = (cols * rows / 2) + (cols * rows / 8);
            if weight.buffer.size >= raw_bytes_len + marlin_bytes_len {
                let ptr = unsafe { (dev_ptr(&weight.buffer)? as *const u8).add(raw_bytes_len) };
                (ptr as *const std::ffi::c_void, 41 /* KIND_Q4_K_MARLIN */)
            } else {
                (dev_ptr(&weight.buffer)? as *const std::ffi::c_void, kind as c_int)
            }
        } else {
            (dev_ptr(&weight.buffer)? as *const std::ffi::c_void, kind as c_int)
        };
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_linear(
                self.raw,
                weight_ptr,
                effective_kind,
                dev_ptr(x)?.cast(),
                dev_ptr(out)?.cast(),
                bias_ptr,
                to_c(rows, "rows")?,
                to_c(cols, "cols")?,
                to_c(batch, "batch")?,
            )
        })
    }

    fn linear_f16_out(&mut self, out16: &mut DeviceBuffer, x: &DeviceBuffer, weight: &DeviceWeight) -> Result<bool, BackendError> {
        if weight.quant_type.to_gguf_type_id() == 40 {
            // NVFP4 prompts: the CUTLASS epilogue writes halves (the decode kernels cannot, they decline)
            if super::env_flag_once!("FLLM_NO_F16_NVFP4") {
                return Ok(false);
            }
            let (rows, cols) = (weight.rows, weight.cols);
            if rows == 0 || cols == 0 || !x.size.is_multiple_of(cols * 4) {
                return Ok(false);
            }
            let batch = x.size / (cols * 4);
            if batch == 0 || out16.size < batch * rows * 2 {
                return Ok(false);
            }
            // SAFETY: pointers of live device buffers and sizes checked above; the call only inspects its arguments.
            let ok = unsafe {
                fllm_cuda_nvfp4_cutlass_ok(dev_ptr(x)?.cast(), dev_ptr(out16)?.cast(), std::ptr::null(), to_c(rows, "rows")?, to_c(cols, "cols")?, to_c(batch, "batch")?, 2)
            };
            if ok == 0 {
                return Ok(false);
            }
            // SAFETY: plain setter of a thread-local flag in the CUDA library.
            unsafe { fllm_cuda_nvfp4_f16out(1) };
            let result = self.quantized_linear(out16, x, weight, None);
            // SAFETY: as above.
            unsafe { fllm_cuda_nvfp4_f16out(0) };
            return result.map(|()| true);
        }
        if weight.quant_type.q4z_group().is_none() || weight.perm.is_some() || super::env_flag_once!("FLLM_NO_FUSED_QKV") {
            return Ok(false);
        }
        // SAFETY: plain setter of a thread-local flag in the CUDA library.
        unsafe { fllm_cuda_q4z_f16out(1) };
        let result = self.quantized_linear(out16, x, weight, None);
        // SAFETY: as above.
        unsafe { fllm_cuda_q4z_f16out(0) };
        result.map(|()| true)
    }

    fn glu_f16(&mut self, out: &mut DeviceBuffer, gate16: &DeviceBuffer, up16: &DeviceBuffer, geglu: bool) -> Result<bool, BackendError> {
        if gate16.size != up16.size || !gate16.size.is_multiple_of(4) || out.size < gate16.size {
            return Err(invalid("gated activation operands differ in size"));
        }
        // SAFETY: sizes checked above; gate16 and up16 hold count halves each in buffers sized for FP32.
        check(unsafe {
            fllm_cuda_glu_f16(self.raw, dev_ptr(gate16)?.cast(), dev_ptr(up16)?.cast(), dev_ptr(out)?.cast(), gate16.size / 4, c_int::from(geglu))
        })
        .map(|()| true)
    }

    #[allow(clippy::too_many_arguments)]
    fn rope_kv_write_f16(
        &mut self,
        q_out: &mut DeviceBuffer,
        q16: &DeviceBuffer,
        k16: &DeviceBuffer,
        v16: &DeviceBuffer,
        biases: [Option<&DeviceBuffer>; 3],
        kv: &DeviceKvCache,
        layer_idx: usize,
        positions: &[usize],
        slots: &[usize],
        params: &RopeParams,
    ) -> Result<bool, BackendError> {
        if kv.kv_kind != 1 || params.rope_dim == 0 || params.rope_dim > params.dim || positions.len() != slots.len() {
            return Ok(false);
        }
        let rows = positions.len();
        if rows == 0 {
            return Ok(true);
        }
        let (dk, dv) = (kv.head_dim, kv.value_dim);
        if q_out.size < rows * params.heads * dk * 4
            || q16.size < rows * params.heads * dk * 2
            || k16.size < rows * kv.kv_heads * dk * 2
            || v16.size < rows * kv.kv_heads * dv * 2
            || dk != params.dim
        {
            return Err(invalid("fused rope and cache write: buffers smaller than rows x heads x dim"));
        }
        validate_kv_write(kv, layer_idx, slots, rows * kv.kv_heads * dk * 4, rows * kv.kv_heads * dv * 4)?;
        let cos = self.table(&params.cos)?;
        let sin = self.table(&params.sin)?;
        let pos: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let slot_ids: Vec<i32> = slots.iter().map(|&s| s as i32).collect();
        let bias_ptr = |b: Option<&DeviceBuffer>| -> Result<*const f32, BackendError> {
            Ok(match b {
                Some(b) => dev_ptr(b)? as *const f32,
                None => std::ptr::null(),
            })
        };
        // SAFETY: sizes and slot ranges validated above; positions and slots are staged by the library before the launch.
        check(unsafe {
            fllm_cuda_rope_kv_f16(
                self.raw,
                dev_ptr(q16)?.cast(),
                dev_ptr(k16)?.cast(),
                dev_ptr(v16)?.cast(),
                bias_ptr(biases[0])?,
                bias_ptr(biases[1])?,
                bias_ptr(biases[2])?,
                dev_ptr(q_out)?.cast(),
                dev_ptr(&kv.key_pool)?.add(layer_idx * kv.words_per_layer_k * 2).cast(),
                dev_ptr(&kv.value_pool)?.add(layer_idx * kv.words_per_layer_v * 2).cast(),
                pos.as_ptr(),
                slot_ids.as_ptr(),
                to_c(rows, "rows")?,
                cos,
                sin,
                to_c(params.heads, "heads")?,
                to_c(kv.kv_heads, "kv heads")?,
                to_c(dk, "head width")?,
                to_c(dv, "value width")?,
                to_c(params.rope_dim / 2, "rope width")?,
                c_int::from(params.neox),
            )
        })?;
        Ok(true)
    }

    fn quantized_linear_add(
        &mut self,
        residual: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
        scale: f32,
    ) -> Result<bool, BackendError> {
        if weight.quant_type.to_gguf_type_id() == 40 {
            // NVFP4 prompts: the CUTLASS epilogue adds scale * (product) to the residual stream (the decode kernels cannot, they decline)
            if bias.is_some() || scale == 0.0 || super::env_flag_once!("FLLM_NO_FUSED_ADD") {
                return Ok(false);
            }
            let (rows, cols) = (weight.rows, weight.cols);
            if rows == 0 || cols == 0 || !x.size.is_multiple_of(cols * 4) {
                return Ok(false);
            }
            let batch = x.size / (cols * 4);
            if batch == 0 || residual.size < batch * rows * 4 {
                return Ok(false);
            }
            // SAFETY: pointers of live device buffers and sizes checked above; the call only inspects its arguments.
            let ok = unsafe {
                fllm_cuda_nvfp4_cutlass_ok(dev_ptr(x)?.cast(), dev_ptr(residual)?.cast(), std::ptr::null(), to_c(rows, "rows")?, to_c(cols, "cols")?, to_c(batch, "batch")?, 1)
            };
            if ok == 0 {
                return Ok(false);
            }
            // SAFETY: plain setter of a thread-local flag in the CUDA library.
            unsafe { fllm_cuda_nvfp4_accumulate(1, scale) };
            let result = self.quantized_linear(residual, x, weight, None);
            // SAFETY: as above.
            unsafe { fllm_cuda_nvfp4_accumulate(0, 0.0) };
            return result.map(|()| true);
        }
        if weight.quant_type.q4z_group().is_none() || super::env_flag_once!("FLLM_NO_FUSED_ADD") {
            return Ok(false);
        }
        // SAFETY: plain setter of a thread-local flag in the CUDA library.
        unsafe { fllm_cuda_q4z_accumulate(1, scale) };
        let result = self.quantized_linear(residual, x, weight, bias);
        // SAFETY: as above.
        unsafe { fllm_cuda_q4z_accumulate(0, 0.0) };
        result.map(|()| true)
    }

    fn add_rms_norm(
        &mut self,
        x: &mut DeviceBuffer,
        branch: &DeviceBuffer,
        scale: f32,
        weight: &DeviceBuffer,
        eps: f32,
        out: &mut DeviceBuffer,
        next: Option<&DeviceWeight>,
    ) -> Result<bool, BackendError> {
        let cols = weight.size / 4;
        if cols == 0 || !x.size.is_multiple_of(cols * 4) || branch.size < x.size || out.size < x.size {
            return Ok(false);
        }
        let rows = x.size / (cols * 4);
        let (quant, kind) = next_activation_format(next);
        // SAFETY: sizes checked above; the entry point validates alignment and shape and declines (returns 3) without launching.
        fused_result(unsafe {
            fllm_cuda_add_rms_norm(
                self.raw,
                dev_ptr(x)?.cast(),
                dev_ptr(branch)?.cast(),
                scale,
                dev_ptr(weight)?.cast(),
                dev_ptr(out)?.cast(),
                to_c(rows, "rows")?,
                to_c(cols, "width")?,
                eps,
                quant,
                kind,
            )
        })
    }

    fn quantized_linear_add_norm(
        &mut self,
        residual: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
        scale: f32,
        norm: &DeviceBuffer,
        eps: f32,
        out: &mut DeviceBuffer,
        next: Option<&DeviceWeight>,
    ) -> Result<bool, BackendError> {
        let Some(group) = weight.quant_type.q4z_group() else {
            return Ok(false);
        };
        if scale == 0.0 || std::env::var_os("FLLM_NO_FUSED_ADD").is_some() {
            return Ok(false);
        }
        let (rows, cols) = (weight.rows, weight.cols);
        if rows == 0 || cols == 0 || !x.size.is_multiple_of(cols * 4) || norm.size != rows * 4 {
            return Ok(false);
        }
        let batch = x.size / (cols * 4);
        if batch == 0 || residual.size < batch * rows * 4 || out.size < batch * rows * 4 || bias.is_some_and(|b| b.size < rows * 4) {
            return Ok(false);
        }
        let bias_ptr = match bias {
            Some(b) => dev_ptr(b)? as *const f32,
            None => std::ptr::null(),
        };
        let perm_ptr = match &weight.perm {
            Some(p) => dev_ptr(p)? as *const c_int,
            None => std::ptr::null(),
        };
        let (quant, kind) = next_activation_format(next);
        // SAFETY: sizes checked above; the weight was uploaded by `upload_q4z`; the entry point declines (returns 3) before launching anything
        // when the alignment or the width does not suit the fused kernel.
        fused_result(unsafe {
            fllm_cuda_linear_q4z_add_norm(
                self.raw,
                dev_ptr(&weight.buffer)?.cast(),
                group as c_int,
                dev_ptr(x)?.cast(),
                dev_ptr(residual)?.cast(),
                bias_ptr,
                to_c(rows, "rows")?,
                to_c(cols, "cols")?,
                to_c(batch, "batch")?,
                perm_ptr,
                scale,
                dev_ptr(norm)?.cast(),
                dev_ptr(out)?.cast(),
                eps,
                quant,
                kind,
            )
        })
    }

    fn rope_kv_write(
        &mut self,
        q: &mut DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        kv: &DeviceKvCache,
        layer_idx: usize,
        positions: &[usize],
        slots: &[usize],
        params: &RopeParams,
    ) -> Result<bool, BackendError> {
        if kv.kv_kind != 1 || params.rope_dim == 0 || params.rope_dim > params.dim || positions.len() != slots.len() {
            return Ok(false);
        }
        let rows = positions.len();
        if rows == 0 {
            return Ok(true);
        }
        let (dk, dv) = (kv.head_dim, kv.value_dim);
        if dk != params.dim || kv.kv_heads != params.kv_heads {
            return Ok(false);
        }
        params.validate(positions, q.size, k.size)?;
        validate_kv_write(kv, layer_idx, slots, k.size, v.size)?;
        let cos = self.table(&params.cos)?;
        let sin = self.table(&params.sin)?;
        let pos: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let slot_ids: Vec<i32> = slots.iter().map(|&s| s as i32).collect();
        // SAFETY: sizes, table coverage and slot ranges validated above; positions and slots are staged by the library before the launch.
        fused_result(unsafe {
            fllm_cuda_rope_kv_f32(
                self.raw,
                dev_ptr(q)?.cast(),
                dev_ptr(k)?.cast(),
                dev_ptr(v)?.cast(),
                dev_ptr(&kv.key_pool)?.add(layer_idx * kv.words_per_layer_k * 2).cast(),
                dev_ptr(&kv.value_pool)?.add(layer_idx * kv.words_per_layer_v * 2).cast(),
                pos.as_ptr(),
                slot_ids.as_ptr(),
                to_c(rows, "rows")?,
                cos,
                sin,
                to_c(params.heads, "heads")?,
                to_c(kv.kv_heads, "kv heads")?,
                to_c(dk, "head width")?,
                to_c(dv, "value width")?,
                to_c(params.rope_dim / 2, "rope width")?,
                c_int::from(params.neox),
            )
        })
    }

    fn graph_supported(&self) -> bool {
        // SAFETY: plain query on a live stream.
        unsafe { fllm_cuda_graph_supported(self.raw) != 0 }
    }

    fn graph_begin(&mut self, id: usize, sizes: [usize; 4], max_len: usize) -> Result<(), BackendError> {
        // SAFETY: plain call on a live stream.
        check(unsafe { fllm_cuda_graph_begin(self.raw, to_c(id, "graph id")?, sizes[0], sizes[1], sizes[2], sizes[3], to_c(max_len, "context")?) })
    }

    fn graph_end(&mut self, id: usize, abort: bool) -> Result<(), BackendError> {
        // SAFETY: plain call on a live stream.
        check(unsafe { fllm_cuda_graph_end(self.raw, to_c(id, "graph id")?, c_int::from(abort)) })
    }

    fn graph_set(&mut self, id: usize, array: usize, data: &[u8]) -> Result<(), BackendError> {
        // SAFETY: `data` is valid for its length; the library copies it before returning.
        check(unsafe { fllm_cuda_graph_set(self.raw, to_c(id, "graph id")?, to_c(array, "array")?, data.as_ptr().cast(), data.len()) })
    }

    fn graph_launch(&mut self, id: usize) -> Result<bool, BackendError> {
        // SAFETY: plain call on a live stream.
        match unsafe { fllm_cuda_graph_launch(self.raw, to_c(id, "graph id")?) } {
            0 => Ok(true),
            2 => Ok(false),
            code => check(code).map(|()| true),
        }
    }

    fn argmax_rows(&mut self, logits: &DeviceBuffer, rows: usize, vocab: usize) -> Result<Vec<i32>, BackendError> {
        if rows == 0 || vocab == 0 || logits.size < rows * vocab * 4 {
            return Err(invalid("argmax input smaller than rows x vocab"));
        }
        let mut ids = vec![0i32; rows];
        // SAFETY: sizes checked above; `ids` has room for `rows` ints and the call returns after the copy completed.
        check(unsafe { fllm_cuda_argmax(self.raw, dev_ptr(logits)?.cast(), to_c(rows, "rows")?, to_c(vocab, "vocab")?, ids.as_mut_ptr()) })?;
        Ok(ids)
    }

    fn swiglu(&mut self, out: &mut DeviceBuffer, gate: &DeviceBuffer, up: &DeviceBuffer) -> Result<(), BackendError> {
        self.glu(out, gate, up, 0)
    }

    fn geglu(&mut self, out: &mut DeviceBuffer, gate: &DeviceBuffer, up: &DeviceBuffer) -> Result<(), BackendError> {
        self.glu(out, gate, up, 1)
    }

    fn residual_add(&mut self, x: &mut DeviceBuffer, residual: &DeviceBuffer, scale: f32) -> Result<(), BackendError> {
        if !x.size.is_multiple_of(4) || residual.size < x.size {
            return Err(invalid("residual shorter than its target"));
        }
        // SAFETY: sizes checked above.
        check(unsafe { fllm_cuda_residual(self.raw, dev_ptr(x)?.cast(), dev_ptr(residual)?.cast(), x.size / 4, scale) })
    }

    #[allow(clippy::too_many_arguments)]
    fn split_q_gate(
        &mut self,
        q: &mut DeviceBuffer,
        gate: &mut DeviceBuffer,
        projected: &DeviceBuffer,
        rows: usize,
        heads: usize,
        dk: usize,
        per_head: bool,
    ) -> Result<(), BackendError> {
        let width = rows * heads * dk * 4;
        if q.size < width || gate.size < width || projected.size < 2 * width {
            return Err(invalid("query/gate split: buffers smaller than rows x heads x head width"));
        }
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_split_q_gate(
                self.raw,
                dev_ptr(projected)?.cast(),
                dev_ptr(q)?.cast(),
                dev_ptr(gate)?.cast(),
                to_c(rows, "rows")?,
                to_c(heads, "heads")?,
                to_c(dk, "head width")?,
                c_int::from(per_head),
            )
        })
    }

    fn sigmoid_gate_mul(&mut self, x: &mut DeviceBuffer, gate: &DeviceBuffer) -> Result<(), BackendError> {
        if !x.size.is_multiple_of(4) || gate.size < x.size {
            return Err(invalid("attention gate shorter than its target"));
        }
        // SAFETY: sizes checked above.
        check(unsafe { fllm_cuda_sigmoid_mul(self.raw, dev_ptr(x)?.cast(), dev_ptr(gate)?.cast(), x.size / 4) })
    }

    fn supports_attention_gate(&self) -> bool {
        true
    }

    fn supports_arch_ops(&self) -> bool {
        true
    }

    fn mla_latent(&mut self, latent: &mut DeviceBuffer, kv_a: &DeviceBuffer, rows: usize, kv_lora: usize, width: usize) -> Result<(), BackendError> {
        if latent.size < rows * kv_lora * 4 || kv_a.size < rows * width * 4 || kv_lora > width {
            return Err(invalid("MLA latent copy: buffers smaller than rows x width"));
        }
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_mla_latent(self.raw, dev_ptr(kv_a)?.cast(), dev_ptr(latent)?.cast(), to_c(rows, "rows")?, to_c(kv_lora, "latent width")?, to_c(width, "width")?)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_assemble(
        &mut self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        v: &mut DeviceBuffer,
        q_full: &DeviceBuffer,
        kv_a: &DeviceBuffer,
        kv: &DeviceBuffer,
        rows: usize,
        s: &MlaShape,
    ) -> Result<(), BackendError> {
        let dk = s.nope + s.rope;
        if q.size < rows * s.heads * dk * 4
            || k.size < rows * s.heads * dk * 4
            || v.size < rows * s.heads * s.value * 4
            || q_full.size < rows * s.heads * dk * 4
            || kv_a.size < rows * (s.kv_lora + s.rope) * 4
            || kv.size < rows * s.heads * (s.nope + s.value) * 4
        {
            return Err(invalid("MLA assembly: buffers smaller than the layer needs"));
        }
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_mla_assemble(
                self.raw,
                dev_ptr(q_full)?.cast(),
                dev_ptr(kv_a)?.cast(),
                dev_ptr(kv)?.cast(),
                dev_ptr(q)?.cast(),
                dev_ptr(k)?.cast(),
                dev_ptr(v)?.cast(),
                to_c(rows, "rows")?,
                to_c(s.heads, "heads")?,
                to_c(s.nope, "nope width")?,
                to_c(s.rope, "rope width")?,
                to_c(s.value, "value width")?,
                to_c(s.kv_lora, "latent width")?,
            )
        })
    }

    fn supports_mamba2(&self, s: &MambaShape) -> bool {
        let dims = [s.inner, s.state, s.heads, s.groups, s.conv_kernel, s.norm_groups].map(|n| c_int::try_from(n).unwrap_or(0));
        // SAFETY: plain arithmetic on the arguments.
        unsafe { fllm_cuda_mamba2_supported(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]) != 0 }
    }

    #[allow(clippy::too_many_arguments)]
    fn mamba2_layer(
        &mut self,
        s: &MambaShape,
        w: &MambaWeights<'_>,
        runs: &[DeltaRun<'_>],
        state_offset: usize,
        projected: &DeviceBuffer,
        mixed: &mut DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        if !self.supports_mamba2(s) {
            return Err(BackendError::UnsupportedOperation("mamba2_layer for this shape".into()));
        }
        let rows: usize = runs.iter().map(|r| r.rows).sum();
        if rows == 0 {
            return Ok(());
        }
        let conv_dim = s.conv_dim();
        let width = s.inner + conv_dim + s.heads;
        let state_end = state_offset + s.layer_state_bytes();
        if projected.size < rows * width * 4
            || mixed.size < rows * conv_dim * 4
            || out.size < rows * s.inner * 4
            || w.conv.size < conv_dim * s.conv_kernel * 4
            || w.conv_bias.is_some_and(|b| b.size < conv_dim * 4)
            || w.dt_bias.size < s.heads * 4
            || w.decay.size < s.heads * 4
            || w.skip.size < s.heads * 4
            || w.norm.size < s.inner * 4
            || runs.iter().any(|r| r.state.size < state_end || r.first_row + r.rows > rows)
        {
            return Err(invalid("Mamba-2 buffers smaller than the layer needs"));
        }
        // mirrors SsmRun in mamba2.cu
        #[repr(C)]
        struct RunC {
            state: u64,
            row0: i32,
            n: i32,
            reset: i32,
            pad: i32,
        }
        let mut list = Vec::with_capacity(runs.len());
        for r in runs {
            list.push(RunC {
                state: dev_ptr(r.state)? as u64,
                row0: to_c(r.first_row, "run start")?,
                n: to_c(r.rows, "run length")?,
                reset: i32::from(r.reset),
                pad: 0,
            });
        }
        let bias_ptr = match w.conv_bias {
            Some(b) => dev_ptr(b)? as *const f32,
            None => std::ptr::null(),
        };
        // SAFETY: sizes checked above; `list` outlives the call (the C side copies it before returning).
        check(unsafe {
            fllm_cuda_mamba2_layer(
                self.raw,
                list.as_ptr().cast(),
                to_c(list.len(), "runs")?,
                to_c(rows, "rows")?,
                to_c(s.inner, "inner width")?,
                to_c(s.state, "state size")?,
                to_c(s.heads, "heads")?,
                to_c(s.groups, "groups")?,
                to_c(s.conv_kernel, "convolution kernel")?,
                to_c(s.norm_groups, "norm groups")?,
                s.step_min,
                s.eps,
                state_offset,
                s.rec_bytes(),
                dev_ptr(projected)?.cast(),
                dev_ptr(w.conv)?.cast(),
                bias_ptr,
                dev_ptr(w.dt_bias)?.cast(),
                dev_ptr(w.decay)?.cast(),
                dev_ptr(w.skip)?.cast(),
                dev_ptr(w.norm)?.cast(),
                dev_ptr(mixed)?.cast(),
                dev_ptr(out)?.cast(),
            )
        })
    }

    fn supports_shortconv(&self, hidden: usize, kernel: usize) -> bool {
        // SAFETY: plain arithmetic on the arguments.
        unsafe {
            fllm_cuda_shortconv_supported(c_int::try_from(hidden).unwrap_or(0), c_int::try_from(kernel).unwrap_or(0)) != 0
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn shortconv_layer(
        &mut self,
        hidden: usize,
        kernel: usize,
        conv: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        runs: &[DeltaRun<'_>],
        state_offset: usize,
        projected: &DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        if !self.supports_shortconv(hidden, kernel) {
            return Err(BackendError::UnsupportedOperation("shortconv_layer for this shape".into()));
        }
        let rows: usize = runs.iter().map(|r| r.rows).sum();
        if rows == 0 {
            return Ok(());
        }
        let state_end = state_offset + hidden * (kernel - 1) * 4;
        if projected.size < rows * 3 * hidden * 4
            || out.size < rows * hidden * 4
            || conv.size < hidden * kernel * 4
            || bias.is_some_and(|b| b.size < hidden * 4)
            || runs.iter().any(|r| r.state.size < state_end || r.first_row + r.rows > rows)
        {
            return Err(invalid("ShortConv buffers smaller than the layer needs"));
        }
        // mirrors ConvRun in shortconv.cu
        #[repr(C)]
        struct RunC {
            state: u64,
            row0: i32,
            n: i32,
            reset: i32,
            pad: i32,
        }
        let mut list = Vec::with_capacity(runs.len());
        for r in runs {
            list.push(RunC {
                state: dev_ptr(r.state)? as u64,
                row0: to_c(r.first_row, "run start")?,
                n: to_c(r.rows, "run length")?,
                reset: i32::from(r.reset),
                pad: 0,
            });
        }
        let bias_ptr = match bias {
            Some(b) => dev_ptr(b)? as *const f32,
            None => std::ptr::null(),
        };
        // SAFETY: sizes checked above; `list` outlives the call (the C side copies it before returning).
        check(unsafe {
            fllm_cuda_shortconv_layer(
                self.raw,
                list.as_ptr().cast(),
                to_c(list.len(), "runs")?,
                to_c(hidden, "width")?,
                to_c(kernel, "convolution kernel")?,
                state_offset,
                dev_ptr(projected)?.cast(),
                dev_ptr(conv)?.cast(),
                bias_ptr,
                dev_ptr(out)?.cast(),
            )
        })
    }

    fn supports_mamba1(&self, s: &Mamba1Shape) -> bool {
        let dims = [s.inner, s.state, s.rank, s.conv_kernel].map(|n| c_int::try_from(n).unwrap_or(0));
        // SAFETY: plain arithmetic on the arguments.
        unsafe { fllm_cuda_mamba1_supported(dims[0], dims[1], dims[2], dims[3]) != 0 }
    }

    #[allow(clippy::too_many_arguments)]
    fn mamba1_conv(
        &mut self,
        s: &Mamba1Shape,
        conv: &DeviceBuffer,
        conv_bias: Option<&DeviceBuffer>,
        runs: &[DeltaRun<'_>],
        state_offset: usize,
        proj: &DeviceBuffer,
        mixed: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        if !self.supports_mamba1(s) {
            return Err(BackendError::UnsupportedOperation("mamba1_conv for this shape".into()));
        }
        let rows: usize = runs.iter().map(|r| r.rows).sum();
        if rows == 0 {
            return Ok(());
        }
        let state_end = state_offset + s.layer_state_bytes();
        if proj.size < rows * 2 * s.inner * 4
            || mixed.size < rows * s.inner * 4
            || conv.size < s.inner * s.conv_kernel * 4
            || conv_bias.is_some_and(|b| b.size < s.inner * 4)
            || runs.iter().any(|r| r.state.size < state_end || r.first_row + r.rows > rows)
        {
            return Err(invalid("Mamba-1 buffers smaller than the layer needs"));
        }
        let list = ssm_runs(runs)?;
        let bias_ptr = match conv_bias {
            Some(b) => dev_ptr(b)? as *const f32,
            None => std::ptr::null(),
        };
        // SAFETY: sizes checked above; `list` outlives the call (the C side copies it before returning).
        check(unsafe {
            fllm_cuda_mamba1_conv(
                self.raw,
                list.as_ptr().cast(),
                to_c(list.len(), "runs")?,
                to_c(s.inner, "inner width")?,
                to_c(s.conv_kernel, "convolution kernel")?,
                state_offset,
                s.rec_bytes(),
                dev_ptr(proj)?.cast(),
                dev_ptr(conv)?.cast(),
                bias_ptr,
                dev_ptr(mixed)?.cast(),
            )
        })
    }

    fn mamba1_prep(&mut self, s: &Mamba1Shape, split: &mut DeviceBuffer, dt_in: &mut DeviceBuffer, rows: usize) -> Result<(), BackendError> {
        if split.size < rows * (s.rank + 2 * s.state) * 4 || dt_in.size < rows * s.rank * 4 {
            return Err(invalid("Mamba-1 projection buffers smaller than the layer needs"));
        }
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_mamba1_prep(
                self.raw,
                dev_ptr(split)?.cast(),
                dev_ptr(dt_in)?.cast(),
                to_c(rows, "rows")?,
                to_c(s.rank, "time step rank")?,
                to_c(s.state, "state size")?,
                s.bc_rms_eps.unwrap_or(0.0),
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn mamba1_scan(
        &mut self,
        s: &Mamba1Shape,
        decay: &DeviceBuffer,
        skip: &DeviceBuffer,
        runs: &[DeltaRun<'_>],
        state_offset: usize,
        split: &DeviceBuffer,
        dt: &DeviceBuffer,
        mixed: &DeviceBuffer,
        proj: &DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        let rows: usize = runs.iter().map(|r| r.rows).sum();
        if rows == 0 {
            return Ok(());
        }
        let state_end = state_offset + s.layer_state_bytes();
        if split.size < rows * (s.rank + 2 * s.state) * 4
            || dt.size < rows * s.inner * 4
            || mixed.size < rows * s.inner * 4
            || proj.size < rows * 2 * s.inner * 4
            || out.size < rows * s.inner * 4
            || decay.size < s.inner * s.state * 4
            || skip.size < s.inner * 4
            || runs.iter().any(|r| r.state.size < state_end || r.first_row + r.rows > rows)
        {
            return Err(invalid("Mamba-1 buffers smaller than the layer needs"));
        }
        let list = ssm_runs(runs)?;
        // SAFETY: sizes checked above; `list` outlives the call (the C side copies it before returning).
        check(unsafe {
            fllm_cuda_mamba1_scan(
                self.raw,
                list.as_ptr().cast(),
                to_c(list.len(), "runs")?,
                to_c(s.inner, "inner width")?,
                to_c(s.state, "state size")?,
                to_c(s.rank, "time step rank")?,
                state_offset,
                dev_ptr(split)?.cast(),
                dev_ptr(dt)?.cast(),
                dev_ptr(mixed)?.cast(),
                dev_ptr(proj)?.cast(),
                dev_ptr(decay)?.cast(),
                dev_ptr(skip)?.cast(),
                dev_ptr(out)?.cast(),
            )
        })
    }

    fn activation(&mut self, out: &mut DeviceBuffer, x: &DeviceBuffer, mode: i32) -> Result<(), BackendError> {
        if !x.size.is_multiple_of(4) || out.size < x.size {
            return Err(invalid("activation output smaller than its input"));
        }
        // SAFETY: sizes checked above.
        check(unsafe { fllm_cuda_act(self.raw, dev_ptr(x)?.cast(), dev_ptr(out)?.cast(), x.size / 4, mode) })
    }

    fn activation_glu(&mut self, out: &mut DeviceBuffer, gate: &DeviceBuffer, up: &DeviceBuffer, mode: i32) -> Result<(), BackendError> {
        if !gate.size.is_multiple_of(4) || up.size < gate.size || out.size < gate.size {
            return Err(invalid("gated activation buffers differ in size"));
        }
        // SAFETY: sizes checked above.
        check(unsafe {
            fllm_cuda_act_glu(self.raw, dev_ptr(gate)?.cast(), dev_ptr(up)?.cast(), dev_ptr(out)?.cast(), gate.size / 4, mode)
        })
    }

    fn softplus_gate_mul(&mut self, x: &mut DeviceBuffer, gate: &DeviceBuffer, beta: f32) -> Result<(), BackendError> {
        if !x.size.is_multiple_of(4) || gate.size < x.size || beta <= 0.0 {
            return Err(invalid("attention gate shorter than its target"));
        }
        // SAFETY: sizes checked above.
        check(unsafe { fllm_cuda_softplus_mul(self.raw, dev_ptr(x)?.cast(), dev_ptr(gate)?.cast(), x.size / 4, beta) })
    }

    fn clamp(&mut self, x: &mut DeviceBuffer, limit: f32) -> Result<(), BackendError> {
        // SAFETY: the buffer is a whole number of floats.
        check(unsafe { fllm_cuda_clamp(self.raw, dev_ptr(x)?.cast(), x.size / 4, limit) })
    }

    fn supports_delta(&self, s: &DeltaShape) -> bool {
        let dims = [s.k_heads, s.v_heads, s.k_dim, s.v_dim, s.conv_kernel].map(|n| c_int::try_from(n).unwrap_or(0));
        // SAFETY: plain arithmetic on the arguments.
        unsafe { fllm_cuda_delta_supported(dims[0], dims[1], dims[2], dims[3], dims[4]) != 0 }
    }

    #[allow(clippy::too_many_arguments)]
    fn delta_layer(
        &mut self,
        s: &DeltaShape,
        w: &DeltaWeights<'_>,
        runs: &[DeltaRun<'_>],
        state_offset: usize,
        qkv: &DeviceBuffer,
        z: &DeviceBuffer,
        beta: &DeviceBuffer,
        alpha: &DeviceBuffer,
        mixed: &mut DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        if !self.supports_delta(s) {
            return Err(BackendError::UnsupportedOperation("delta_layer for this shape".into()));
        }
        let rows: usize = runs.iter().map(|r| r.rows).sum();
        if rows == 0 {
            return Ok(());
        }
        let (conv_dim, value_dim) = (s.conv_dim(), s.value_dim());
        let state_end = state_offset + s.layer_state_bytes();
        if qkv.size < rows * conv_dim * 4
            || mixed.size < rows * conv_dim * 4
            || z.size < rows * value_dim * 4
            || out.size < rows * value_dim * 4
            || beta.size < rows * s.v_heads * 4
            || alpha.size < rows * s.v_heads * 4
            || w.conv.size < conv_dim * s.conv_kernel * 4
            || w.dt_bias.size < s.v_heads * 4
            || w.decay.size < s.v_heads * 4
            || w.norm.size < s.v_dim * 4
            || runs.iter().any(|r| r.state.size < state_end || r.first_row + r.rows > rows)
        {
            return Err(invalid("DeltaNet buffers smaller than the layer needs"));
        }
        // mirrors DeltaRun in deltanet.cu
        #[repr(C)]
        struct RunC {
            state: u64,
            row0: i32,
            n: i32,
            reset: i32,
            pad: i32,
        }
        let mut list = Vec::with_capacity(runs.len());
        for r in runs {
            list.push(RunC {
                state: dev_ptr(r.state)? as u64,
                row0: to_c(r.first_row, "run start")?,
                n: to_c(r.rows, "run length")?,
                reset: i32::from(r.reset),
                pad: 0,
            });
        }
        let max_n = runs.iter().map(|r| r.rows).max().unwrap_or(0);
        // SAFETY: sizes checked above; `list` outlives the call (the C side copies it before returning).
        check(unsafe {
            fllm_cuda_delta_layer(
                self.raw,
                list.as_ptr().cast(),
                to_c(list.len(), "runs")?,
                to_c(max_n, "run length")?,
                to_c(s.k_heads, "key heads")?,
                to_c(s.v_heads, "value heads")?,
                to_c(s.k_dim, "key width")?,
                to_c(s.v_dim, "value width")?,
                to_c(s.conv_kernel, "convolution kernel")?,
                c_int::from(s.tiled_heads),
                s.beta_scale,
                s.eps,
                state_offset,
                s.rec_bytes(),
                dev_ptr(qkv)?.cast(),
                dev_ptr(z)?.cast(),
                dev_ptr(beta)?.cast(),
                dev_ptr(alpha)?.cast(),
                dev_ptr(w.conv)?.cast(),
                dev_ptr(w.dt_bias)?.cast(),
                dev_ptr(w.decay)?.cast(),
                dev_ptr(w.norm)?.cast(),
                dev_ptr(mixed)?.cast(),
                dev_ptr(out)?.cast(),
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_paged(
        &mut self,
        out: &mut DeviceBuffer,
        q: &DeviceBuffer,
        kv: &DeviceKvCache,
        block_tables: &[i64],
        lengths: &[i64],
        layer_idx: usize,
        params: &AttentionParams,
    ) -> Result<(), BackendError> {
        // the blocks of a step are the same for all its layers: they are walked for the first attention call only
        let key = [block_tables.as_ptr() as usize, block_tables.len(), lengths.as_ptr() as usize, lengths.len()];
        let walked = self.step_open && self.blocks_checked == Some(key);
        params.validate_blocks(kv, block_tables, lengths, layer_idx, !walked)?;
        if self.step_open {
            self.blocks_checked = Some(key);
        }
        let (dk, dv, heads) = (params.head_dim, params.value_dim, params.num_heads);
        let rows = lengths.len();
        if rows == 0 {
            return Ok(());
        }
        if q.size < rows * heads * dk * 4 || out.size < rows * heads * dv * 4 {
            return Err(invalid("attention q/out smaller than rows x heads x dim"));
        }
        if dk > MAX_ATTENTION_DIM || dv > MAX_ATTENTION_DIM {
            warn_fallback_reason(
                "attention_paged",
                &format!("head width {dk}/{dv} exceeds the CUDA kernel's {MAX_ATTENTION_DIM}; the whole KV pool is copied every call"),
            );
            let cq = self.copy_to_host(q)?;
            let host_kv = DeviceKvCache {
                key_pool: self.copy_to_host(&kv.key_pool)?,
                value_pool: self.copy_to_host(&kv.value_pool)?,
                ..kv.clone()
            };
            let mut co = self.cpu.allocate_buffer(out.size)?;
            self.cpu_stream.attention_paged(&mut co, &cq, &host_kv, block_tables, lengths, layer_idx, params)?;
            return self.copy_from_host(&co, out);
        }
        let slots = kv.slots();
        let (kk, kvv) = (kv.kv_heads * dk, kv.kv_heads * dv);
        let sinks = match &params.sinks {
            Some(s) => self.table(s)?,
            None => std::ptr::null(),
        };
        let alibi = match &params.alibi_slopes {
            Some(s) => self.table(s)?,
            None => std::ptr::null(),
        };
        if !matches!(kv.kv_kind, 1 | 2) {
            let (key_layout, value_layout) = kv.layouts();
            let (kd, vd) = (KvDesc::new(&key_layout), KvDesc::new(&value_layout));
            // SAFETY: as below; the descriptors live across the call and describe the pools of this cache.
            return check(unsafe {
                fllm_cuda_kvq_attention(
                    self.raw,
                    dev_ptr(q)?.cast(),
                    dev_ptr(&kv.key_pool)?.add(layer_idx * kv.words_per_layer_k * 2).cast(),
                    dev_ptr(&kv.value_pool)?.add(layer_idx * kv.words_per_layer_v * 2).cast(),
                    dev_ptr(out)?.cast(),
                    block_tables.as_ptr(),
                    block_tables.len(),
                    lengths.as_ptr(),
                    to_c(rows, "rows")?,
                    sinks,
                    alibi,
                    to_c(heads, "heads")?,
                    to_c(params.num_kv_heads, "kv heads")?,
                    to_c(dk, "key width")?,
                    to_c(dv, "value width")?,
                    to_c(params.stride, "block table stride")?,
                    to_c(params.block_size, "block size")?,
                    &kd,
                    &vd,
                    params.scale,
                    params.softcap.unwrap_or(0.0),
                    params.window.map_or(0, |w| w.min(c_int::MAX as usize) as c_int),
                    params.chunk.map_or(0, |c| c.min(c_int::MAX as usize) as c_int),
                )
            });
        }
        // SAFETY: `validate` checked the block tables, lengths and parameters against the cache;
        // pool pointers address the requested layer inside the allocations.
        check(unsafe {
            fllm_cuda_attention(
                self.raw,
                dev_ptr(q)?.cast(),
                dev_ptr(&kv.key_pool)?.add(layer_idx * kv.words_per_layer_k * 2).cast(),
                dev_ptr(&kv.value_pool)?.add(layer_idx * kv.words_per_layer_v * 2).cast(),
                dev_ptr(out)?.cast(),
                block_tables.as_ptr(),
                block_tables.len(),
                lengths.as_ptr(),
                to_c(rows, "rows")?,
                sinks,
                alibi,
                to_c(heads, "heads")?,
                to_c(params.num_kv_heads, "kv heads")?,
                to_c(dk, "key width")?,
                to_c(dv, "value width")?,
                to_c(params.stride, "block table stride")?,
                to_c(params.block_size, "block size")?,
                kv.kv_kind,
                to_c(kv_quant_group(dk), "key group")?,
                to_c(kv_quant_group(dv), "value group")?,
                to_c((slots * kk).div_ceil(2), "key scale base")?,
                to_c((slots * kvv).div_ceil(2), "value scale base")?,
                params.scale,
                params.softcap.unwrap_or(0.0),
                params.window.map_or(0, |w| w.min(c_int::MAX as usize) as c_int),
                params.chunk.map_or(0, |c| c.min(c_int::MAX as usize) as c_int),
            )
        })
    }

    fn kv_cache_write(&mut self, kv: &DeviceKvCache, layer_idx: usize, slots: &[usize], k: &DeviceBuffer, v: &DeviceBuffer) -> Result<(), BackendError> {
        validate_kv_write(kv, layer_idx, slots, k.size, v.size)?;
        if slots.is_empty() {
            return Ok(());
        }
        let ids: Vec<i32> = slots.iter().map(|&s| s as i32).collect();
        let total = kv.slots();
        let rows = to_c(slots.len(), "rows")?;
        let (key_layout, value_layout) = kv.layouts();
        for (src, pool, words, width, layout) in [
            (k, &kv.key_pool, kv.words_per_layer_k, kv.head_dim, key_layout),
            (v, &kv.value_pool, kv.words_per_layer_v, kv.value_dim, value_layout),
        ] {
            let kvw = kv.kv_heads * width;
            if !layout.fmt.is_legacy() {
                let desc = KvDesc::new(&layout);
                // SAFETY: `validate_kv_write` checked sources and slot ranges; the descriptor matches the pool.
                check(unsafe {
                    fllm_cuda_kvq_write(
                        self.raw,
                        dev_ptr(src)?.cast(),
                        dev_ptr(pool)?.add(layer_idx * words * 2).cast(),
                        ids.as_ptr(),
                        rows,
                        &desc,
                    )
                })?;
                continue;
            }
            // SAFETY: `validate_kv_write` checked sources and slot ranges.
            check(unsafe {
                fllm_cuda_kv_write(
                    self.raw,
                    dev_ptr(src)?.cast(),
                    dev_ptr(pool)?.add(layer_idx * words * 2).cast(),
                    ids.as_ptr(),
                    rows,
                    to_c(kvw, "kv width")?,
                    layout.fmt.code(),
                    to_c(kv_quant_group(width), "kv group")?,
                    to_c((total * kvw).div_ceil(2), "scale base")?,
                )
            })?;
        }
        Ok(())
    }

    fn moe_route(&mut self, weights_out: &mut DeviceBuffer, indices_out: &mut DeviceBuffer, router_logits: &DeviceBuffer, num_experts: usize, top_k: usize) -> Result<(), BackendError> {
        warn_fallback_reason("moe_route", "no CUDA router kernel yet");
        let cl = self.copy_to_host(router_logits)?;
        let mut cw = self.cpu.allocate_buffer(weights_out.size)?;
        let mut ci = self.cpu.allocate_buffer(indices_out.size)?;
        self.cpu_stream.moe_route(&mut cw, &mut ci, &cl, num_experts, top_k)?;
        self.copy_from_host(&cw, weights_out)?;
        self.copy_from_host(&ci, indices_out)
    }

    fn moe_expert_gemm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        expert_weights: &[DeviceWeight],
        expert_indices: &DeviceBuffer,
        expert_weights_buf: &DeviceBuffer,
        top_k: usize,
    ) -> Result<(), BackendError> {
        warn_fallback_reason("moe_expert_gemm", "no CUDA expert kernel yet; expert weights are copied every call");
        let cx = self.copy_to_host(x)?;
        let cidx = self.copy_to_host(expert_indices)?;
        let cwts = self.copy_to_host(expert_weights_buf)?;
        let experts = expert_weights.iter().map(|w| self.weight_to_host(w)).collect::<Result<Vec<_>, _>>()?;
        let mut co = self.cpu.allocate_buffer(out.size)?;
        self.cpu_stream.moe_expert_gemm(&mut co, &cx, &experts, &cidx, &cwts, top_k)?;
        self.copy_from_host(&co, out)
    }

    fn sampling_logits(&mut self, token_ids_out: &mut [u32], logits: &DeviceBuffer, params: &SamplingParams) -> Result<(), BackendError> {
        warn_fallback_reason("sampling_logits", "no CUDA sampling kernel yet");
        let cl = self.copy_to_host(logits)?;
        self.cpu_stream.sampling_logits(token_ids_out, &cl, params)
    }
}

/// IEEE half bits of `v` (round to nearest even), used to expand weights on GPUs without FP4 support.
fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    let man = x & 0x007f_ffff;
    if ((x >> 23) & 0xff) == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (man | 0x0080_0000) >> (1 - exp);
        return sign | ((m + 0x0fff + ((m >> 13) & 1)) >> 13) as u16;
    }
    let half = ((exp as u32) << 10) | (man >> 13);
    let rounded = half + (((man & 0x1fff) + 0x0fff + ((man >> 13) & 1)) >> 13);
    sign | rounded as u16
}

/// GGUF NVFP4 rows are `[4 scales + 32 nibble bytes]` per 64 values (inside a 16-value sub-block byte j holds element j
/// in the low and element j + 8 in the high nibble). The device layout is the one of CUTLASS block-scaled GEMMs, so the
/// same weights serve the small-batch kernel in `nvfp4.cu` and the CUTLASS GEMM for prompts:
/// `[rows x cols / 2 bytes: element 2i in the low nibble of byte i]`, padded to 256 bytes, then the UE4M3 scales in the
/// swizzled atom layout (`nvfp4_scale_index`), `ceil(rows / 128) * cols / 64 * 512` bytes. `rows` and `cols` must be
/// multiples of 16 and 256.
fn nvfp4_device_layout(raw: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let blocks = cols / 64;
    let row_bytes = blocks * 36;
    let nibble_bytes = rows * cols / 2;
    let scale_offset = nibble_bytes.div_ceil(256) * 256;
    let mut out = vec![0u8; scale_offset + rows.div_ceil(128) * blocks * 512];
    let (nibbles, scales) = out.split_at_mut(scale_offset);
    nibbles[..nibble_bytes].par_chunks_mut(cols / 2).enumerate().for_each(|(r, dst)| {
        for b in 0..blocks {
            let block = &raw[r * row_bytes + b * 36..][..36];
            for sub in 0..4 {
                for j in 0..8 {
                    let byte = block[4 + sub * 8 + j];
                    for (e, nib) in [(sub * 16 + j, byte & 15), (sub * 16 + j + 8, byte >> 4)] {
                        dst[b * 32 + e / 2] |= nib << (4 * (e % 2));
                    }
                }
            }
        }
    });
    for r in 0..rows {
        for b in 0..blocks {
            for i in 0..4 {
                scales[nvfp4_scale_index(r, b * 4 + i, blocks)] = raw[r * row_bytes + b * 36 + i];
            }
        }
    }
    out
}

/// Offset of the scale of 16-value block `kb` of row `n` in the swizzled layout (CUTLASS `Sm1xxBlkScaledConfig`): atoms of
/// 128 rows x 4 blocks (512 bytes), atom columns consecutive along K, atom rows `blocks_per_row / 4` atoms apart; inside an
/// atom row `n` sits at `(n % 32) * 16 + (n / 32 % 4) * 4`, the four blocks are adjacent.
fn nvfp4_scale_index(n: usize, kb: usize, blocks_per_row: usize) -> usize {
    (n / 128) * blocks_per_row * 512 + (kb / 4) * 512 + (n % 32) * 16 + (n / 32 % 4) * 4 + kb % 4
}

static TOTAL_REPACK_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn total_repack_ms() -> f64 {
    TOTAL_REPACK_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0
}

fn q4_k_marlin_device_layout(raw: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let t_repack_start = std::time::Instant::now();

    let blocks_per_row = cols / 256;
    let k_tiles = cols / 16;
    let n_tiles = rows / 64;

    let w_bytes = cols * rows / 2;
    let s_bytes = cols * rows / 16;
    let m_bytes = cols * rows / 16;
    let total_bytes = w_bytes + s_bytes + m_bytes;

    let mut out = vec![0u8; total_bytes];

    // 1 & 2. Extract and directly write transposed scales & mins using sequential row access
    let (_, sm_out) = out.split_at_mut(w_bytes);
    let (scales_out, mins_out) = sm_out.split_at_mut(s_bytes);

    let row_bytes = blocks_per_row * 144;
    let s_ptr = scales_out.as_mut_ptr() as usize;
    let m_ptr = mins_out.as_mut_ptr() as usize;

    raw.par_chunks_exact(row_bytes)
        .enumerate()
        .for_each(|(r, row_raw)| {
            let c64 = r / 64;
            let rem = r % 64;
            let i = rem / 8;
            let j = rem % 8;
            let base_in_kg = c64 * 64 + j * 8 + i;
            let s_dst = s_ptr as *mut u16;
            let m_dst = m_ptr as *mut u16;

            for b in 0..blocks_per_row {
                let blk_offset = b * 144;
                let d_bits = u16::from_le_bytes([row_raw[blk_offset], row_raw[blk_offset + 1]]);
                let dmin_bits = u16::from_le_bytes([row_raw[blk_offset + 2], row_raw[blk_offset + 3]]);
                let d = half::f16::from_bits(d_bits).to_f32();
                let dmin = half::f16::from_bits(dmin_bits).to_f32();
                let scales = &row_raw[blk_offset + 4..blk_offset + 16];

                for g in 0..8 {
                    let (sc, mn) = if g < 4 {
                        (scales[g] & 63, scales[g + 4] & 63)
                    } else {
                        (
                            (scales[g + 4] & 15) | ((scales[g - 4] >> 6) << 4),
                            (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4),
                        )
                    };
                    let s_val = f32_to_f16_bits(d * sc as f32);
                    let m_val = f32_to_f16_bits(-(dmin * mn as f32));

                    let kg = b * 8 + g;
                    let dst_idx = kg * rows + base_in_kg;
                    unsafe {
                        *s_dst.add(dst_idx) = s_val;
                        *m_dst.add(dst_idx) = m_val;
                    }
                }
            }
        });

    // 3. Repack weights into 16x64 tiles with precomputed block offsets
    let w_u32 = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u32, k_tiles * n_tiles * 128)
    };

    const TC_OFFSETS: [usize; 4] = [0, 1, 8, 9];
    const PACK_IDX: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

    w_u32.par_chunks_exact_mut(n_tiles * 128)
        .enumerate()
        .for_each(|(kt, k_slice)| {
            let cur_k = kt * 16;

            let mut precomp = [(0usize, false); 16];
            for tc_idx in 0..4 {
                let tc_row = tc_idx * 2;
                for i in 0..4 {
                    let r = cur_k + tc_row + TC_OFFSETS[i];
                    let blk = r / 256;
                    let elem = r % 256;
                    let group = elem / 32;
                    let j = elem % 32;
                    let offset = blk * 144 + 16 + (group / 2) * 32 + j;
                    let is_hi = !group.is_multiple_of(2);
                    precomp[tc_idx * 4 + i] = (offset, is_hi);
                }
            }

            for nt in 0..n_tiles {
                let tile = &mut k_slice[nt * 128..(nt + 1) * 128];
                let cur_n_base = nt * 64;

                for warp_id in 0..4 {
                    let warp_n = cur_n_base + warp_id * 16;
                    for th_id in 0..32 {
                        let tc_col = th_id / 4;
                        let tc_idx = th_id % 4;
                        let cur_n = warp_n + tc_col;

                        let row_off_0 = cur_n * row_bytes;
                        let row_off_8 = (cur_n + 8) * row_bytes;

                        let p_base = tc_idx * 4;
                        let mut vals = [0u8; 8];
                        for i in 0..4 {
                            let (off, is_hi) = precomp[p_base + i];
                            let b0 = raw[row_off_0 + off];
                            let b8 = raw[row_off_8 + off];
                            vals[i] = if is_hi { b0 >> 4 } else { b0 & 0x0f };
                            vals[4 + i] = if is_hi { b8 >> 4 } else { b8 & 0x0f };
                        }

                        let mut res = 0u32;
                        for i in 0..8 {
                            res |= ((vals[PACK_IDX[i]] & 0x0f) as u32) << (4 * i);
                        }

                        tile[th_id * 4 + warp_id] = res;
                    }
                }
            }
        });

    TOTAL_REPACK_US.fetch_add(t_repack_start.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    out
}

/// GGUF Q8_0 blocks are `[f16 scale, 32 x i8]` (34 bytes, only 2-byte aligned). For widths that are a multiple
/// of 256 the device stores each row as `[scales: blocks * 2][quants: blocks * 32]` (same size) so every
/// quant vector starts on a 16-byte boundary.
fn q8_0_device_layout(raw: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let blocks = cols / 32;
    let padded = padded_blocks(cols);
    let (row_bytes, out_row) = (blocks * 34, padded * 34);
    let mut out = vec![0u8; rows * out_row];
    for r in 0..rows {
        let src = &raw[r * row_bytes..(r + 1) * row_bytes];
        let dst = &mut out[r * out_row..(r + 1) * out_row];
        for b in 0..blocks {
            dst[b * 2..b * 2 + 2].copy_from_slice(&src[b * 34..b * 34 + 2]);
            dst[padded * 2 + b * 32..padded * 2 + (b + 1) * 32].copy_from_slice(&src[b * 34 + 2..(b + 1) * 34]);
        }
    }
    out
}

/// 32-value blocks per row in the device layouts: the width rounded up to a multiple of 256 (zero blocks), so every
/// array of a row starts on a 16-byte boundary (`pad256` in `common.cuh`).
fn padded_blocks(cols: usize) -> usize {
    cols.div_ceil(256) * 8
}

/// Q4_0 / Q4_1 / Q5_0 / Q5_1 blocks (`d [m] [qh] qs`, 18-24 bytes) are stored per row as separate arrays
/// `[d][m][qh][qs]` (same size) so every array starts on a 16-byte boundary; see `qgemm_leg.cu`.
fn legacy_device_layout(raw: &[u8], rows: usize, cols: usize, kind: u32) -> Vec<u8> {
    let affine = matches!(kind, 3 | 7);
    let five = matches!(kind, 6 | 7);
    let blocks = cols / 32;
    let padded = padded_blocks(cols);
    let block_bytes = 2 + if affine { 2 } else { 0 } + if five { 4 } else { 0 } + 16;
    let (row_bytes, out_row) = (blocks * block_bytes, padded * block_bytes);
    let m_off = padded * 2;
    let qh_off = padded * 2 * if affine { 2 } else { 1 };
    let qs_off = qh_off + if five { padded * 4 } else { 0 };
    let mut out = vec![0u8; rows * out_row];
    for r in 0..rows {
        let src = &raw[r * row_bytes..(r + 1) * row_bytes];
        let dst = &mut out[r * out_row..(r + 1) * out_row];
        for b in 0..blocks {
            let blk = &src[b * block_bytes..(b + 1) * block_bytes];
            dst[b * 2..b * 2 + 2].copy_from_slice(&blk[..2]);
            let mut at = 2;
            if affine {
                dst[m_off + b * 2..m_off + b * 2 + 2].copy_from_slice(&blk[at..at + 2]);
                at += 2;
            }
            if five {
                dst[qh_off + b * 4..qh_off + b * 4 + 4].copy_from_slice(&blk[at..at + 4]);
                at += 4;
            }
            dst[qs_off + b * 16..qs_off + b * 16 + 16].copy_from_slice(&blk[at..at + 16]);
        }
    }
    out
}

/// IQ4_XS super-blocks (`d: f16, scales_h: u16, scales_l: [u8; 4], qs: [u8; 128]`, 256 values) rewritten as eight
/// IQ4_NL blocks (`d: f16, qs: [u8; 16]`) each: the 6-bit sub-block scales are folded into the fp16 block scale.
fn iq4_xs_to_nl(raw: &[u8], superblocks: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(superblocks * 8 * 18);
    for sb in raw.as_chunks::<136>().0.iter().take(superblocks) {
        let d = crate::quant_iq::f16_to_f32(u16::from_le_bytes([sb[0], sb[1]]));
        let scales_h = u16::from_le_bytes([sb[2], sb[3]]);
        for ib in 0..8 {
            let lo = (sb[4 + ib / 2] >> (4 * (ib % 2))) & 0xf;
            let hi = ((scales_h >> (2 * ib)) & 3) as u8;
            let ls = (lo | (hi << 4)) as i32 - 32;
            out.extend_from_slice(&f32_to_f16_bits(d * ls as f32).to_le_bytes());
            out.extend_from_slice(&sb[8 + 16 * ib..8 + 16 * ib + 16]);
        }
    }
    out
}

static EXACT_MATH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static MARLIN_Q4K_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Switch between the exact FP32 kernels and the fast ones (tensor cores, integer dot products) at run time.
pub fn set_exact_math(on: bool) {
    EXACT_MATH.store(on, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: plain setter of a process-wide flag.
    unsafe { fllm_cuda_set_exact(c_int::from(on)) };
}

pub fn is_exact_math() -> bool {
    // FLLM_CUDA_EXACT is also read by the CUDA library itself: both sides must agree, or a dual-layout Q4_K weight is sent to a
    // kernel that the library refuses in exact mode
    EXACT_MATH.load(std::sync::atomic::Ordering::Relaxed) || std::env::var_os("FLLM_CUDA_EXACT").is_some()
}

/// The switches of the prefill host-work reductions and kernel fusions (docs/performance-notes.md). The environment variables give the start
/// values; `set_prefill_switch` overrides them at run time (selftest-gpu compares the variants on one device).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillSwitch {
    /// `FLLM_PREFILL_STEP_CACHE`
    StepCache = 0,
    /// `FLLM_FUSE_PREFILL_ADD_NORM`
    FuseAddNorm = 1,
    /// `FLLM_FUSE_PREFILL_NORM_F16`
    FuseNormF16 = 2,
    /// `FLLM_FUSE_PREFILL_ATTN_F16`
    FuseAttnF16 = 3,
    /// `FLLM_MARLIN_PREFILL_CFG` (0: vllm, 1: fllm)
    MarlinCfg = 4,
}

pub fn prefill_switch(which: PrefillSwitch) -> i32 {
    // SAFETY: plain getter of a process-wide flag.
    unsafe { fllm_cuda_get_opt(which as c_int) }
}

pub fn set_prefill_switch(which: PrefillSwitch, value: i32) {
    // SAFETY: plain setter of a process-wide flag.
    unsafe { fllm_cuda_set_opt(which as c_int, value) };
}

pub fn set_marlin_q4k_enabled(enabled: bool) {
    MARLIN_Q4K_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

pub fn is_marlin_q4k_enabled() -> bool {
    if std::env::var_os("FLLM_CUDA_NO_MARLIN_Q4K").is_some() || std::env::var_os("FLLM_CUDA_NO_MARLIN").is_some() {
        return false;
    }
    MARLIN_Q4K_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// MXFP4 blocks `[e8m0 scale][16 bytes: element j low, element j + 16 high nibble]` as IQ4_NL-style blocks
/// `[f16 d][16 bytes]` with d = 2^(e - 128) (the kernels' grid holds twice the E2M1 values).
fn mxfp4_to_nl(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() / 17 * 18);
    for block in raw.as_chunks::<17>().0 {
        let d = 2f32.powi(i32::from(block[0]) - 128);
        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&block[1..]);
    }
    out
}

/// Mirrors SsmRun in mamba2.cu / mamba1.cu.
#[repr(C)]
struct SsmRunC {
    state: u64,
    row0: i32,
    n: i32,
    reset: i32,
    pad: i32,
}

fn ssm_runs(runs: &[DeltaRun<'_>]) -> Result<Vec<SsmRunC>, BackendError> {
    runs.iter()
        .map(|r| {
            Ok(SsmRunC {
                state: dev_ptr(r.state)? as u64,
                row0: to_c(r.first_row, "run start")?,
                n: to_c(r.rows, "run length")?,
                reset: i32::from(r.reset),
                pad: 0,
            })
        })
        .collect()
}

#[cfg(test)]
mod nvfp4_layout_tests {
    use super::*;

    /// Nibble of element `k` of row `r` in the GGUF block layout (inside a 16-value sub-block byte j holds element j in the
    /// low and element j + 8 in the high nibble).
    fn gguf_element(raw: &[u8], row_bytes: usize, r: usize, k: usize) -> u8 {
        let block = &raw[r * row_bytes + (k / 64) * 36..][..36];
        let (sub, i) = ((k % 64) / 16, k % 16);
        let byte = block[4 + sub * 8 + i % 8];
        if i < 8 { byte & 15 } else { byte >> 4 }
    }

    #[test]
    fn nvfp4_device_layout_is_natural_order_with_swizzled_scales() {
        let (rows, cols) = (160, 512);
        let blocks = cols / 64;
        let row_bytes = blocks * 36;
        let raw: Vec<u8> = (0..rows * row_bytes).map(|i: usize| (i.wrapping_mul(2_654_435_761) >> 7) as u8).collect();
        let out = nvfp4_device_layout(&raw, rows, cols);
        let scale_offset = (rows * cols / 2).div_ceil(256) * 256;
        assert_eq!(out.len(), scale_offset + rows.div_ceil(128) * blocks * 512);
        for r in 0..rows {
            for k in 0..cols {
                let nibble = (out[r * cols / 2 + k / 2] >> (4 * (k % 2))) & 15;
                assert_eq!(nibble, gguf_element(&raw, row_bytes, r, k), "row {r} element {k}");
            }
        }
        let mut seen = std::collections::HashSet::new();
        for r in 0..rows {
            for kb in 0..cols / 16 {
                let at = nvfp4_scale_index(r, kb, blocks);
                assert!(seen.insert(at), "scale of row {r} block {kb} collides");
                assert_eq!(out[scale_offset + at], raw[r * row_bytes + (kb / 4) * 36 + kb % 4], "scale of row {r} block {kb}");
            }
        }
    }
}
