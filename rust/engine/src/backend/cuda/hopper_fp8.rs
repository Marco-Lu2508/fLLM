//! FP8 and int8 W8A8 GEMMs (cuda/fp8_gemm.cu, cuda/int8_gemm.cu): the upload layout of FP8M weights for the W8A8 kernel, the linear op that
//! runs it, and a direct path to the kernels for the self test (block-wise FP8 and plain int8 weights have no weight format in the engine yet).
//!
//! An FP8M weight (E4M3 bytes plus one f16 scale per row) is uploaded in one of two layouts. `FLLM_FP8_GEMM=0`, devices below sm_89 and shapes
//! whose width is not a multiple of 128 keep the Marlin layout (W8A16, `upload_fp8_marlin`). Otherwise (`auto` and `1` on sm_89 and newer) the
//! weight stays as the raw FP8M tensor in `DeviceWeight::buffer` (row-major E4M3, then the f16 scales: the kernel reads the first part) and its row
//! scales as FP32 sit in `DeviceWeight::perm` (a Marlin FP8 weight has no
//! permutation, so `perm.is_some()` on an `Fp8Marlin` weight means "W8A8 layout"). The activations are quantized to E4M3 in the kernel call.
use super::*;
use half::f16;

extern "C" {
    fn fllm_cuda_fp8_gemm_enabled(dev: c_int, rows: c_int, cols: c_int) -> c_int;
    fn fllm_cuda_fp8_gemm_set_mode(mode: c_int) -> c_int;
    fn fllm_cuda_int8_gemm_set_mode(mode: c_int) -> c_int;
    fn fllm_cuda_linear_fp8_w8a8(
        st: *mut RawStream,
        w: *const c_void,
        wscale: *const f32,
        wblock: c_int,
        x: *const f32,
        y: *mut f32,
        bias: *const f32,
        rows: c_int,
        cols: c_int,
        batch: c_int,
        amode: c_int,
        static_scale: f32,
    ) -> c_int;
    fn fllm_cuda_linear_int8_plain(
        st: *mut RawStream,
        w: *const c_void,
        wscale: *const f32,
        x: *const f32,
        y: *mut f32,
        bias: *const f32,
        rows: c_int,
        cols: c_int,
        batch: c_int,
    ) -> c_int;
}

/// Overrides `FLLM_FP8_GEMM` for the weights uploaded afterwards: -1 environment, 0 Marlin W8A16, 1 W8A8 with `mma.sync`, 2 CUTLASS (Hopper).
pub fn set_fp8_gemm_mode(mode: i32) {
    // SAFETY: plain setter of a process-wide flag.
    unsafe { fllm_cuda_fp8_gemm_set_mode(mode) };
}

/// Overrides `FLLM_INT8_GEMM` (Q8_0 prompts on the int8 tensor cores): -1 environment, 0 off, 1 on.
pub fn set_int8_gemm_mode(mode: i32) {
    // SAFETY: plain setter of a process-wide flag.
    unsafe { fllm_cuda_int8_gemm_set_mode(mode) };
}

/// How the activations of an FP8 W8A8 product are scaled: (mode, static factor). `FLLM_FP8_ACT_STATIC=<factor>` selects one static per-tensor
/// scale for every layer (the dequantization factor, activation = e4m3 * factor); the default is one dynamic scale per token. The FP8M
/// tensors carry no `input_scale` of a static-activation checkpoint, so a static factor is an experiment, not a recipe.
fn activation_scale() -> (c_int, f32) {
    static CELL: std::sync::OnceLock<(c_int, f32)> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| match std::env::var("FLLM_FP8_ACT_STATIC").ok().and_then(|v| v.trim().parse::<f32>().ok()) {
        Some(v) if v > 0.0 && v.is_finite() => (1, v),
        _ => (0, 0.0),
    })
}

/// The W8A8 upload of an FP8M weight, or `None` when the Marlin layout is to be used (switch off, device, shape).
pub(super) fn try_upload_w8a8(
    backend: &CudaBackend,
    name: &str,
    raw_bytes: &[u8],
    rows: usize,
    cols: usize,
    bias: Option<&[f32]>,
) -> Result<Option<DeviceWeight>, BackendError> {
    if backend.rocm || backend.cc < 89 {
        return Ok(None);
    }
    // SAFETY: plain query of the device and the switch.
    if unsafe { fllm_cuda_fp8_gemm_enabled(backend.device, to_c(rows, "rows")?, to_c(cols, "cols")?) } == 0 {
        return Ok(None);
    }
    let qbytes = rows * cols;
    if raw_bytes.len() != qbytes + rows * 2 {
        return Err(invalid(format!("{name}: {} bytes are not {rows} x {cols} FP8M (expected {})", raw_bytes.len(), qbytes + rows * 2)));
    }
    if bias.is_some_and(|b| b.len() != rows) {
        return Err(invalid(format!("{name}: bias does not have {rows} values")));
    }
    let scale_raw = &raw_bytes[qbytes..];
    // the buffer is the raw FP8M tensor (bytes, then the f16 row scales): the host fallbacks (weight_to_host, the CPU backend) read it as such
    let buffer = backend.alloc(raw_bytes.len())?;
    backend.upload_sync(&buffer, raw_bytes)?;
    let scales: Vec<u8> = scale_raw.as_chunks::<2>().0.iter().flat_map(|c| f16::from_le_bytes(*c).to_f32().to_le_bytes()).collect();
    let scale_buffer = backend.alloc(rows * 4)?;
    backend.upload_sync(&scale_buffer, &scales)?;
    let bias_buf = if let Some(b) = bias {
        let buf = backend.alloc(rows * 4)?;
        let bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        backend.upload_sync(&buf, &bytes)?;
        Some(buf)
    } else {
        None
    };
    log_once("cuda-fp8-w8a8", &format!("FP8M weights run as W8A8 on the FP8 tensor cores (FLLM_FP8_GEMM=0 keeps the Marlin W8A16 kernel; first: {name})"));
    Ok(Some(DeviceWeight {
        name: name.to_string(),
        quant_type: QuantType::Fp8Marlin,
        rows,
        cols,
        buffer,
        bias: bias_buf,
        scale: 1.0,
        perm: Some(scale_buffer),
    }))
}

/// The upload of a block-wise FP8 weight (QuantType::Fp8Block, 128x128 blocks).
pub(super) fn try_upload_fp8_block(
    backend: &CudaBackend,
    name: &str,
    raw_bytes: &[u8],
    rows: usize,
    cols: usize,
    bias: Option<&[f32]>,
) -> Result<Option<DeviceWeight>, BackendError> {
    if backend.rocm || backend.cc < 89 {
        return Ok(None);
    }
    let qbytes = rows * cols;
    let num_scales = rows.div_ceil(128) * cols.div_ceil(128);
    let expected = qbytes + num_scales * 2;
    if raw_bytes.len() != expected {
        return Err(invalid(format!("{name}: {} bytes are not {rows} x {cols} blockwise FP8 (expected {expected})", raw_bytes.len())));
    }
    if bias.is_some_and(|b| b.len() != rows) {
        return Err(invalid(format!("{name}: bias does not have {rows} values")));
    }
    let scale_raw = &raw_bytes[qbytes..];
    let buffer = backend.alloc(qbytes)?;
    backend.upload_sync(&buffer, &raw_bytes[..qbytes])?;
    let scales: Vec<u8> = scale_raw.as_chunks::<2>().0.iter().flat_map(|c| f16::from_le_bytes(*c).to_f32().to_le_bytes()).collect();
    let scale_buffer = backend.alloc(num_scales * 4)?;
    backend.upload_sync(&scale_buffer, &scales)?;
    let bias_buf = if let Some(b) = bias {
        let buf = backend.alloc(rows * 4)?;
        let bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        backend.upload_sync(&buf, &bytes)?;
        Some(buf)
    } else {
        None
    };
    log_once("cuda-fp8-block", &format!("Block-wise FP8 weights (128x128) run as W8A8 on the FP8 tensor cores; first: {name}"));
    Ok(Some(DeviceWeight {
        name: name.to_string(),
        quant_type: QuantType::Fp8Block,
        rows,
        cols,
        buffer,
        bias: bias_buf,
        scale: 1.0,
        perm: Some(scale_buffer),
    }))
}

/// `out = x * W^T (+ bias)` for an FP8M weight in the W8A8 layout (`weight.perm` holds the row scales).
pub(super) fn linear_w8a8(
    stream: &mut CudaStream,
    out: &mut DeviceBuffer,
    x: &DeviceBuffer,
    weight: &DeviceWeight,
    bias: Option<&DeviceBuffer>,
) -> Result<(), BackendError> {
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
    let scales = weight.perm.as_ref().ok_or_else(|| invalid("FP8 W8A8 weight without scales"))?;
    let bias_ptr = match bias {
        Some(b) => dev_ptr(b)? as *const f32,
        None => std::ptr::null(),
    };
    let (amode_env, static_scale) = activation_scale();
    let (wblock, amode) = if weight.quant_type == QuantType::Fp8Block { (1, 2) } else { (0, amode_env) };
    // SAFETY: sizes checked above; the weight was uploaded by `try_upload_w8a8` or `try_upload_fp8_block`.
    check(unsafe {
        fllm_cuda_linear_fp8_w8a8(
            stream.raw,
            dev_ptr(&weight.buffer)?.cast(),
            dev_ptr(scales)?.cast(),
            wblock,
            dev_ptr(x)?.cast(),
            dev_ptr(out)?.cast(),
            bias_ptr,
            to_c(rows, "rows")?,
            to_c(cols, "cols")?,
            to_c(batch, "batch")?,
            amode,
            static_scale,
        )
    })
}

/// What `direct_gemm` runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direct {
    /// FP8 weights with one scale per row, one dynamic activation scale per token
    Fp8Row,
    /// FP8 weights with one scale per 128 x 128 block (`weight_scale_inv` of DeepSeek style checkpoints), one activation scale per token and 128 values
    Fp8Block,
    /// FP8 weights with one scale per row, one static activation scale (`static_scale`)
    Fp8Static,
    /// int8 weights with one scale per row, one dynamic activation scale per token
    Int8Row,
}

/// Runs one of the W8A8 kernels on device `device` with host data: `weights` are the row-major bytes (E4M3 or int8), `wscale` the FP32 scales
/// (per row, or `ceil(rows / 128) * (cols / 128)` block scales). Returns the FP32 result and the milliseconds of the best of `repeats` launches.
#[allow(clippy::too_many_arguments)]
pub fn direct_gemm(
    device: usize,
    kind: Direct,
    weights: &[u8],
    wscale: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    (rows, cols, batch): (usize, usize, usize),
    static_scale: f32,
    repeats: usize,
) -> Result<(Vec<f32>, f64), BackendError> {
    if weights.len() != rows * cols || x.len() != batch * cols || bias.is_some_and(|b| b.len() != rows) {
        return Err(invalid("direct_gemm: operand sizes do not match the shape"));
    }
    struct Buf(*mut c_void);
    impl Drop for Buf {
        fn drop(&mut self) {
            // SAFETY: allocated by fllm_cuda_malloc below.
            unsafe { fllm_cuda_free(self.0) };
        }
    }
    let dev = device as c_int;
    let alloc = |bytes: usize| -> Result<Buf, BackendError> {
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: `p` is a valid out-pointer.
        check(unsafe { fllm_cuda_malloc(dev, &mut p, bytes.max(16)) })?;
        Ok(Buf(p))
    };
    let put = |buf: &Buf, bytes: &[u8]| -> Result<(), BackendError> {
        // SAFETY: `buf` was allocated with at least this many bytes.
        check(unsafe { fllm_cuda_h2d_sync(dev, buf.0, bytes.as_ptr().cast(), bytes.len()) })
    };
    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
    let (wb, sb, xb, yb) = (alloc(weights.len())?, alloc(wscale.len() * 4)?, alloc(x.len() * 4)?, alloc(batch * rows * 4)?);
    put(&wb, weights)?;
    put(&sb, &f32_bytes(wscale))?;
    put(&xb, &f32_bytes(x))?;
    let bias_buf = match bias {
        Some(b) => {
            let buf = alloc(b.len() * 4)?;
            put(&buf, &f32_bytes(b))?;
            Some(buf)
        }
        None => None,
    };
    // SAFETY: plain constructor; destroyed below on every path through the guard.
    let raw = unsafe { fllm_cuda_stream_create(dev) };
    if raw.is_null() {
        return Err(BackendError::DriverError(last_error()));
    }
    struct Guard(*mut RawStream);
    impl Drop for Guard {
        fn drop(&mut self) {
            // SAFETY: created above.
            unsafe { fllm_cuda_stream_destroy(self.0) };
        }
    }
    let _guard = Guard(raw);
    let bias_ptr = bias_buf.as_ref().map_or(std::ptr::null(), |b| b.0 as *const f32);
    let (r, c, b) = (to_c(rows, "rows")?, to_c(cols, "cols")?, to_c(batch, "batch")?);
    let launch = || -> Result<(), BackendError> {
        // SAFETY: all buffers hold the sizes checked above.
        check(unsafe {
            match kind {
                Direct::Fp8Row => fllm_cuda_linear_fp8_w8a8(raw, wb.0, sb.0.cast(), 0, xb.0.cast(), yb.0.cast(), bias_ptr, r, c, b, 0, 0.0),
                Direct::Fp8Block => fllm_cuda_linear_fp8_w8a8(raw, wb.0, sb.0.cast(), 1, xb.0.cast(), yb.0.cast(), bias_ptr, r, c, b, 2, 0.0),
                Direct::Fp8Static => fllm_cuda_linear_fp8_w8a8(raw, wb.0, sb.0.cast(), 0, xb.0.cast(), yb.0.cast(), bias_ptr, r, c, b, 1, static_scale),
                Direct::Int8Row => fllm_cuda_linear_int8_plain(raw, wb.0, sb.0.cast(), xb.0.cast(), yb.0.cast(), bias_ptr, r, c, b),
            }
        })
    };
    // the first launch also does the one-time setup (scratch buffers, tile tuning)
    launch()?;
    check(unsafe { fllm_cuda_stream_sync(raw) })?;
    let mut best = f64::INFINITY;
    for _ in 0..repeats.max(1) {
        let start = std::time::Instant::now();
        launch()?;
        check(unsafe { fllm_cuda_stream_sync(raw) })?;
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
    }
    let mut out = vec![0u8; batch * rows * 4];
    // SAFETY: `out` holds the result bytes.
    check(unsafe { fllm_cuda_download(raw, out.as_mut_ptr().cast(), yb.0, out.len()) })?;
    check(unsafe { fllm_cuda_stream_sync(raw) })?;
    Ok((out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(), best))
}
