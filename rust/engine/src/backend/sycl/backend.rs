//! Backend / stream implementation of the sycl backend (Intel GPUs through SYCL / oneAPI). Owner: package 9.
//!
//! The kernels live in `rust/engine/sycl/` (DPC++), `build.rs` compiles them with `icpx -fsycl` into `libfllmsycl`. Weights are packed
//! once on the host (`fllm_sycl_pack_weight`) into layouts the kernels read with aligned 8/16-byte loads and uploaded with
//! `sycl::malloc_device` USM allocations; activations are f32; the KV cache uses the shared 16-bit word layout. Operations are
//! enqueued on an in-order queue; `synchronize`, `download` and `argmax_rows` wait. `fllm-rs selftest-gpu --device sycl` checks every
//! operation against the CPU backend.
//!
//! Level of verification: L0 (compiles; never run on an Intel GPU).

use crate::backend::cpu::{CpuBackend, CpuStream};
use crate::backend::*;
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::sync::atomic::{AtomicU64, Ordering};

use super::device_info;
use super::ffi::{self, RawStream};

/// GGUF types the kernels decode in the packed layouts (23 = IQ4_XS is converted to IQ4_NL while packing).
const NATIVE_KINDS: [u32; 16] = [0, 1, 30, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20, 23, 39];
/// Widest attention head the kernel keeps in registers (16 lanes x 32 values).
const MAX_ATTENTION_DIM: usize = 512;

fn check(code: c_int) -> Result<(), BackendError> {
    if code == 0 {
        Ok(())
    } else {
        Err(BackendError::DriverError(ffi::last_error()))
    }
}

fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

fn to_c(n: usize, what: &str) -> Result<c_int, BackendError> {
    c_int::try_from(n).map_err(|_| invalid(format!("{what} ({n}) exceeds the 32-bit kernel limit")))
}

/// Device allocation owned by an [`Arc`] shared by every window into it.
struct SyclAlloc {
    ptr: *mut c_void,
    device: c_int,
}
// SAFETY: a USM device pointer is just an address; the SYCL runtime is thread safe.
unsafe impl Send for SyclAlloc {}
unsafe impl Sync for SyclAlloc {}
impl Drop for SyclAlloc {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `fllm_sycl_malloc` and is freed exactly once (after the live streams finished their work).
        unsafe { ffi::fllm_sycl_free(self.device, self.ptr) };
    }
}

fn dev_ptr(buf: &DeviceBuffer) -> Result<*mut u8, BackendError> {
    let alloc = buf
        .handle
        .downcast_ref::<SyclAlloc>()
        .ok_or_else(|| BackendError::BufferError("buffer does not belong to the SYCL backend".into()))?;
    // SAFETY: `offset` stays inside the allocation (checked by `DeviceBuffer::slice`).
    Ok(unsafe { (alloc.ptr as *mut u8).add(buf.offset) })
}

fn opt_ptr(buf: Option<&DeviceBuffer>) -> Result<*const f32, BackendError> {
    match buf {
        Some(b) => Ok(dev_ptr(b)? as *const f32),
        None => Ok(std::ptr::null()),
    }
}

/// Formats the kernels read as they are packed from the GGUF file (`fllm-rs devices` / selftest header).
pub fn native_formats() -> Vec<String> {
    NATIVE_KINDS
        .iter()
        .map(|&k| format!("{:?}", QuantType::from_gguf_type_id(k)))
        .collect()
}

/// Allow (`false`, default) or forbid (`true`) the fp16 matrix-engine kernels for large batches; `FLLM_SYCL_EXACT=1` does the same.
/// With `true` every product is computed in FP32 (what `selftest-gpu` holds the kernels to).
pub fn set_exact_math(on: bool) {
    // SAFETY: plain setter of a process-wide flag.
    unsafe { ffi::fllm_sycl_set_exact(c_int::from(on)) };
}

pub fn is_exact_math() -> bool {
    // SAFETY: plain getter.
    unsafe { ffi::fllm_sycl_is_exact() != 0 }
}

/// Create the backend for SYCL device `index`.
pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
    SyclBackend::new(index)
        .map(|b| Arc::new(b) as Arc<dyn Backend>)
        .map_err(|e| e.to_string())
}

pub struct SyclBackend {
    device: c_int,
    name: String,
    capabilities: BackendCapabilities,
    next_id: AtomicU64,
}

impl SyclBackend {
    pub fn new(device: usize) -> Result<Self, BackendError> {
        if !ffi::NATIVE {
            return Err(BackendError::DeviceNotFound(ffi::last_error()));
        }
        // SAFETY: plain query.
        let count = unsafe { ffi::fllm_sycl_device_count() };
        if count < 0 {
            return Err(BackendError::DeviceNotFound(format!("no SYCL device: {}", ffi::last_error())));
        }
        if device >= count as usize {
            return Err(BackendError::DeviceNotFound(format!(
                "SYCL device {device} requested, {count} present"
            )));
        }
        let props = device_info::props(device)
            .ok_or_else(|| BackendError::DeviceNotFound(format!("SYCL device {device} cannot be queried: {}", ffi::last_error())))?;
        let name = props.name_text();
        let capabilities = BackendCapabilities {
            name: name.clone(),
            supported_quants: NATIVE_KINDS.iter().map(|&k| QuantType::from_gguf_type_id(k)).collect(),
            supports_rms_norm: true,
            supports_grouped_rms_norm: false,
            supports_layer_norm: true,
            supports_rope: true,
            supports_swiglu: true,
            supports_geglu: true,
            supports_paged_attention: true,
            supports_kv_cache_write: true,
            supports_embedding_lookup: true,
            supports_device_sampling: false, // host fallback
            sample_on_device: false,
            supports_residual_add: true,
            supports_moe: false, // routing and expert projections run through moe_topk / moe_linear, not moe_expert_gemm
            max_batch_size: 4096,
            max_context: 1 << 20,
        };
        Ok(Self {
            device: device as c_int,
            name,
            capabilities,
            next_id: AtomicU64::new(1),
        })
    }

    fn alloc(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-pointer.
        check(unsafe { ffi::fllm_sycl_malloc(self.device, &mut ptr, size) }).map_err(|e| match e {
            BackendError::DriverError(msg) => {
                log_once("sycl-oom", &format!("SYCL allocation failed: {msg}"));
                BackendError::OutOfMemory {
                    requested: size,
                    available: 0,
                }
            }
            other => other,
        })?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceBuffer::new(
            id,
            size,
            self.device as usize,
            Arc::new(SyclAlloc { ptr, device: self.device }),
        ))
    }

    fn upload_sync(&self, dst: &DeviceBuffer, src: &[u8]) -> Result<(), BackendError> {
        // SAFETY: `dst` holds at least `src.len()` bytes (callers size it from `src`).
        check(unsafe { ffi::fllm_sycl_h2d_sync(self.device, dev_ptr(dst)?.cast(), src.as_ptr().cast(), src.len()) })
    }

    /// The device layout of `rows` x `cols` values of GGUF type `kind` (see `rust/engine/sycl/weights.cpp`).
    fn pack(&self, kind: u32, rows: usize, cols: usize, raw: &[u8]) -> Result<Vec<u8>, BackendError> {
        // SAFETY: plain size computation.
        let bytes = unsafe { ffi::fllm_sycl_weight_bytes(kind as c_int, rows, cols) };
        if bytes == 0 {
            return Err(BackendError::UnsupportedQuant(QuantType::from_gguf_type_id(kind)));
        }
        let mut out = vec![0u8; bytes];
        // SAFETY: `out` holds `bytes` bytes; the library validates `raw`'s length against rows x cols.
        check(unsafe { ffi::fllm_sycl_pack_weight(kind as c_int, rows, cols, raw.as_ptr().cast(), raw.len(), out.as_mut_ptr().cast()) })?;
        Ok(out)
    }

    /// A weight the kernels cannot read in its own format, converted on the host: formats whose values have few mantissa bits
    /// (FP8, NVFP4, the 4-bit AWQ / GPTQ groups) become f16 (exact for practical purposes); every other format of 256-wide rows is
    /// re-quantized once to Q5_K (5.5 bits per weight, finer than the 2-4 bit grids they came from); the rest becomes f32.
    fn convert(
        &self,
        name: &str,
        raw: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<u8>, QuantType), BackendError> {
        use rayon::prelude::*;
        let kind = quant_type.to_gguf_type_id();
        let values =
            crate::compute::dequantize(raw, kind, rows * cols).map_err(|e| BackendError::UnsupportedOperation(format!("{name}: {e}")))?;
        let few_bits = quant_type.q4z_group().is_some()
            || matches!(
                quant_type,
                QuantType::NvFp4 | QuantType::Fp8E4m3 | QuantType::Fp8E5m2 | QuantType::Fp8Marlin
            )
            || fllm_gguf_probe::fp8m_type(kind)
            || fllm_gguf_probe::fp8b_type(kind);
        if few_bits {
            log_once(
                &format!("sycl-f16-{kind}"),
                &format!(
                    "{quant_type:?} weights have no SYCL decoder; expanded to f16 on upload (first: {name}, {} MiB)",
                    (values.len() * 2) >> 20
                ),
            );
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                .collect();
            return Ok((self.pack(1, rows, cols, &bytes)?, QuantType::F16));
        }
        if cols.is_multiple_of(256) {
            let row_bytes = cols / 256 * 176;
            let mut out = vec![0u8; rows * row_bytes];
            out.par_chunks_mut(row_bytes).zip(values.par_chunks(cols)).for_each(|(dst, src)| {
                let mut row = Vec::with_capacity(row_bytes);
                fllm_checkpoint::quantize_row(fllm_checkpoint::QType::Q5K, src, &mut row);
                dst.copy_from_slice(&row);
            });
            log_once(
                &format!("sycl-requant-{kind}"),
                &format!(
                    "{quant_type:?} weights have no SYCL decoder; re-quantized to Q5_K on upload (first: {name}, {} MiB)",
                    out.len() >> 20
                ),
            );
            return Ok((self.pack(13, rows, cols, &out)?, QuantType::Q5_K));
        }
        log_once(
            &format!("sycl-dequant-{kind}"),
            &format!(
                "{quant_type:?} weights have no SYCL decoder; converted to f32 on upload (first: {name}, {} MiB)",
                (values.len() * 4) >> 20
            ),
        );
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok((self.pack(0, rows, cols, &bytes)?, QuantType::F32))
    }
}

impl Backend for SyclBackend {
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
        let raw = unsafe { ffi::fllm_sycl_stream_create(self.device) };
        if raw.is_null() {
            return Err(BackendError::DriverError(ffi::last_error()));
        }
        Ok(Box::new(SyclStream {
            raw,
            device: self.device,
            tables: HashMap::new(),
            cpu: CpuBackend::new(),
            cpu_stream: CpuStream,
            next_id: Arc::new(AtomicU64::new(1 << 40)),
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
        if let Some(b) = bias {
            if b.len() != rows {
                return Err(invalid(format!("{name}: bias has {} values for {rows} rows", b.len())));
            }
        }
        let kind = quant_type.to_gguf_type_id();
        // SAFETY: plain size computations.
        let (device_bytes, expected_raw) = unsafe {
            (
                ffi::fllm_sycl_weight_bytes(kind as c_int, rows, cols),
                ffi::fllm_sycl_raw_bytes(kind as c_int, rows, cols),
            )
        };
        let (packed, stored) = if NATIVE_KINDS.contains(&kind) && device_bytes != 0 {
            if raw_bytes.len() != expected_raw {
                return Err(invalid(format!(
                    "{name}: {} bytes are not {rows} x {cols} {quant_type:?}",
                    raw_bytes.len()
                )));
            }
            // IQ4_XS super-blocks are repacked as IQ4_NL blocks
            (
                self.pack(kind, rows, cols, raw_bytes)?,
                if kind == 23 { QuantType::IQ4_NL } else { quant_type },
            )
        } else {
            self.convert(name, raw_bytes, quant_type, rows, cols)?
        };
        let buffer = self.alloc(packed.len())?;
        self.upload_sync(&buffer, &packed)?;
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
            quant_type: stored,
            rows,
            cols,
            bias,
            scale: 1.0,
            perm: None,
        })
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
            check(unsafe { ffi::fllm_sycl_memset(self.device, dev_ptr(pool)?.cast(), 0, pool.size) })?;
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
            layout: None,
            layer_geoms: None,
        })
    }
}

pub struct SyclStream {
    raw: *mut RawStream,
    device: c_int,
    /// Device copies of constant host tables (rope cos/sin, sinks, ALiBi slopes), by `Arc` address.
    tables: HashMap<usize, (Arc<Vec<f32>>, DeviceBuffer)>,
    cpu: CpuBackend,
    cpu_stream: CpuStream,
    next_id: Arc<AtomicU64>,
}

// SAFETY: the raw stream is only used through `&mut self`.
unsafe impl Send for SyclStream {}

impl Drop for SyclStream {
    fn drop(&mut self) {
        // SAFETY: `raw` is owned by this struct; the library waits for the queue.
        unsafe { ffi::fllm_sycl_stream_destroy(self.raw) };
    }
}

impl SyclStream {
    fn alloc(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointer.
        check(unsafe { ffi::fllm_sycl_malloc(self.device, &mut ptr, size) })?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceBuffer::new(
            id,
            size,
            self.device as usize,
            Arc::new(SyclAlloc { ptr, device: self.device }),
        ))
    }

    /// Device copy of a constant table, uploaded once per `Arc`.
    fn table(&mut self, values: &Arc<Vec<f32>>) -> Result<*const f32, BackendError> {
        let key = Arc::as_ptr(values) as usize;
        if let Some((_, buf)) = self.tables.get(&key) {
            return Ok(dev_ptr(buf)? as *const f32);
        }
        let buf = self.alloc(values.len().max(1) * 4)?;
        // SAFETY: `buf` holds `values.len() * 4` bytes; large uploads wait for the copy, small ones are staged by the library.
        check(unsafe { ffi::fllm_sycl_upload(self.raw, dev_ptr(&buf)?.cast(), values.as_ptr().cast(), values.len() * 4) })?;
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
            ffi::fllm_sycl_glu(
                self.raw,
                dev_ptr(gate)?.cast(),
                dev_ptr(up)?.cast(),
                dev_ptr(out)?.cast(),
                gate.size / 4,
                mode,
            )
        })
    }
}

impl Stream for SyclStream {
    fn synchronize(&mut self) -> Result<(), BackendError> {
        // SAFETY: live stream.
        check(unsafe { ffi::fllm_sycl_stream_sync(self.raw) })
    }

    fn upload(&mut self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<(), BackendError> {
        if src.len() > dst.size {
            return Err(BackendError::BufferError(format!(
                "upload of {} bytes into a {} byte buffer",
                src.len(),
                dst.size
            )));
        }
        // SAFETY: bounds checked above; the library copies pageable memory before it returns (or waits).
        check(unsafe { ffi::fllm_sycl_upload(self.raw, dev_ptr(dst)?.cast(), src.as_ptr().cast(), src.len()) })
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
        check(unsafe { ffi::fllm_sycl_download(self.raw, dst.as_mut_ptr().cast(), dev_ptr(src)?.cast(), dst.len()) })
    }

    fn copy_buffer(&mut self, dst: &mut DeviceBuffer, src: &DeviceBuffer) -> Result<(), BackendError> {
        let bytes = dst.size.min(src.size);
        // SAFETY: both buffers hold at least `bytes`.
        check(unsafe { ffi::fllm_sycl_copy(self.raw, dev_ptr(dst)?.cast(), dev_ptr(src)?.cast(), bytes) })
    }

    fn embedding_lookup(&mut self, out: &mut DeviceBuffer, ids: &[i64], weight: &DeviceWeight) -> Result<(), BackendError> {
        let kind = weight.quant_type.to_gguf_type_id();
        if !NATIVE_KINDS.contains(&kind) {
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
        // SAFETY: sizes checked above; the ids are staged by the call.
        check(unsafe {
            ffi::fllm_sycl_embedding(
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
            ffi::fllm_sycl_rms_norm(
                self.raw,
                dev_ptr(x)?.cast(),
                dev_ptr(weight)?.cast(),
                dev_ptr(out)?.cast(),
                to_c(rows, "rows")?,
                to_c(cols, "width")?,
                eps,
            )
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
        // SAFETY: sizes checked above.
        check(unsafe {
            ffi::fllm_sycl_layer_norm(
                self.raw,
                dev_ptr(x)?.cast(),
                dev_ptr(weight)?.cast(),
                opt_ptr(bias)?,
                dev_ptr(out)?.cast(),
                to_c(rows, "rows")?,
                to_c(cols, "width")?,
                eps,
            )
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
        // SAFETY: `validate` checked table coverage and buffer sizes; the positions are staged by the call.
        check(unsafe {
            ffi::fllm_sycl_rope(
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
            ffi::fllm_sycl_moe_route(
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
            ffi::fllm_sycl_swiglu_oai(
                self.raw,
                dev_ptr(gate)?.cast(),
                dev_ptr(up)?.cast(),
                dev_ptr(out)?.cast(),
                count,
                alpha,
                limit,
            )
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
        // SAFETY: plain size computation (0 = unsupported kind or width).
        let layout_ok = unsafe { ffi::fllm_sycl_weight_bytes(kind as c_int, 1, cols) } != 0;
        if !NATIVE_KINDS.contains(&kind) || kind == 23 || !layout_ok || x_div == 0 {
            return Err(BackendError::UnsupportedQuant(quant_type));
        }
        let inputs = slots.div_ceil(x_div);
        if x.size < inputs * cols * 4 || out.size < slots * rows * 4 || ids.size < slots * 4 {
            return Err(invalid("moe_linear: buffers too small"));
        }
        // SAFETY: sizes checked above; the table was built by `upload_pointer_table` for weights of `quant_type`.
        check(unsafe {
            ffi::fllm_sycl_moe_linear(
                self.raw,
                dev_ptr(table)?.cast(),
                opt_ptr(bias_table)?.cast(),
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
            ffi::fllm_sycl_moe_combine(
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
            ffi::fllm_sycl_gather_rows(
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
            ffi::fllm_sycl_scatter_add_rows(
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
        check(unsafe { ffi::fllm_sycl_zero(self.raw, dev_ptr(buf)?.cast(), buf.size) })
    }

    fn quantized_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
    ) -> Result<(), BackendError> {
        let kind = weight.quant_type.to_gguf_type_id();
        if !NATIVE_KINDS.contains(&kind) {
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
        // SAFETY: sizes checked above; the weight was packed by `upload_weight`.
        check(unsafe {
            ffi::fllm_sycl_linear(
                self.raw,
                dev_ptr(&weight.buffer)?.cast(),
                kind as c_int,
                to_c(rows, "rows")?,
                to_c(cols, "cols")?,
                dev_ptr(x)?.cast(),
                dev_ptr(out)?.cast(),
                opt_ptr(bias)?,
                to_c(batch, "batch")?,
            )
        })
    }

    fn argmax_rows(&mut self, logits: &DeviceBuffer, rows: usize, vocab: usize) -> Result<Vec<i32>, BackendError> {
        if rows == 0 || vocab == 0 || logits.size < rows * vocab * 4 {
            return Err(invalid("argmax input smaller than rows x vocab"));
        }
        let mut ids = vec![0i32; rows];
        // SAFETY: sizes checked above; `ids` has room for `rows` ints and the call returns after the copy completed.
        check(unsafe {
            ffi::fllm_sycl_argmax(
                self.raw,
                dev_ptr(logits)?.cast(),
                to_c(rows, "rows")?,
                to_c(vocab, "vocab")?,
                ids.as_mut_ptr(),
            )
        })?;
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
        check(unsafe { ffi::fllm_sycl_residual(self.raw, dev_ptr(x)?.cast(), dev_ptr(residual)?.cast(), x.size / 4, scale) })
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
            ffi::fllm_sycl_split_q_gate(
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
        check(unsafe { ffi::fllm_sycl_sigmoid_mul(self.raw, dev_ptr(x)?.cast(), dev_ptr(gate)?.cast(), x.size / 4) })
    }

    fn supports_attention_gate(&self) -> bool {
        true
    }

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
        params.validate(kv, block_tables, lengths, layer_idx)?;
        check_kv_kind(kv)?;
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
                &format!("head width {dk}/{dv} exceeds the SYCL kernel's {MAX_ATTENTION_DIM}; the whole KV pool is copied every call"),
            );
            let cq = self.copy_to_host(q)?;
            let host_kv = DeviceKvCache {
                key_pool: self.copy_to_host(&kv.key_pool)?,
                value_pool: self.copy_to_host(&kv.value_pool)?,
                ..kv.clone()
            };
            let mut co = self.cpu.allocate_buffer(out.size)?;
            self.cpu_stream
                .attention_paged(&mut co, &cq, &host_kv, block_tables, lengths, layer_idx, params)?;
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
        // SAFETY: `validate` checked the block tables, lengths and parameters against the cache; the pool pointers address the
        // requested layer inside the allocations; the host arrays are staged by the call.
        check(unsafe {
            ffi::fllm_sycl_attention(
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

    fn kv_cache_write(
        &mut self,
        kv: &DeviceKvCache,
        layer_idx: usize,
        slots: &[usize],
        k: &DeviceBuffer,
        v: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        validate_kv_write(kv, layer_idx, slots, k.size, v.size)?;
        check_kv_kind(kv)?;
        if slots.is_empty() {
            return Ok(());
        }
        let ids: Vec<i32> = slots.iter().map(|&s| s as i32).collect();
        let total = kv.slots();
        let rows = to_c(slots.len(), "rows")?;
        for (src, pool, words, width) in [
            (k, &kv.key_pool, kv.words_per_layer_k, kv.head_dim),
            (v, &kv.value_pool, kv.words_per_layer_v, kv.value_dim),
        ] {
            let kvw = kv.kv_heads * width;
            // SAFETY: `validate_kv_write` checked sources and slot ranges; the slots are staged by the call.
            check(unsafe {
                ffi::fllm_sycl_kv_write(
                    self.raw,
                    dev_ptr(src)?.cast(),
                    dev_ptr(pool)?.add(layer_idx * words * 2).cast(),
                    ids.as_ptr(),
                    rows,
                    to_c(kvw, "kv width")?,
                    kv.kv_kind,
                    to_c(kv_quant_group(width), "kv group")?,
                    to_c((total * kvw).div_ceil(2), "scale base")?,
                )
            })?;
        }
        Ok(())
    }

    fn moe_route(
        &mut self,
        weights_out: &mut DeviceBuffer,
        indices_out: &mut DeviceBuffer,
        router_logits: &DeviceBuffer,
        num_experts: usize,
        top_k: usize,
    ) -> Result<(), BackendError> {
        warn_fallback_reason("moe_route", "routing goes through moe_topk on the SYCL backend");
        let cl = self.copy_to_host(router_logits)?;
        let mut cw = self.cpu.allocate_buffer(weights_out.size)?;
        let mut ci = self.cpu.allocate_buffer(indices_out.size)?;
        self.cpu_stream.moe_route(&mut cw, &mut ci, &cl, num_experts, top_k)?;
        self.copy_from_host(&cw, weights_out)?;
        self.copy_from_host(&ci, indices_out)
    }

    fn moe_expert_gemm(
        &mut self,
        _out: &mut DeviceBuffer,
        _x: &DeviceBuffer,
        _expert_weights: &[DeviceWeight],
        _expert_indices: &DeviceBuffer,
        _expert_weights_buf: &DeviceBuffer,
        _top_k: usize,
    ) -> Result<(), BackendError> {
        // The CPU backend would read the packed device layouts as GGUF blocks: the decoder uses moe_topk / moe_linear / moe_combine.
        Err(BackendError::UnsupportedOperation("moe_expert_gemm (use moe_linear)".into()))
    }

    fn sampling_logits(&mut self, token_ids_out: &mut [u32], logits: &DeviceBuffer, params: &SamplingParams) -> Result<(), BackendError> {
        warn_fallback_reason("sampling_logits", "no SYCL sampling kernel yet");
        let cl = self.copy_to_host(logits)?;
        self.cpu_stream.sampling_logits(token_ids_out, &cl, params)
    }
}

/// The kernels read and write the f16 (1) and q8_0 (2) caches only; the other kinds (bf16, fp8, int8, q4, nvfp4 ...) would be
/// misread as f16, so they are reported as unsupported (the caller falls back or skips).
fn check_kv_kind(kv: &DeviceKvCache) -> Result<(), BackendError> {
    if matches!(kv.kv_kind, 1 | 2) {
        Ok(())
    } else {
        Err(BackendError::UnsupportedOperation(format!("KV cache kind {} (expected 1 = f16 or 2 = q8_0)", kv.kv_kind)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random values in [-1, 1].
    fn noise(n: usize, mut state: u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn quantize(kind: fllm_checkpoint::QType, rows: usize, cols: usize, values: &[f32]) -> Vec<u8> {
        let mut raw = Vec::new();
        for r in 0..rows {
            let mut row = Vec::new();
            fllm_checkpoint::quantize_row(kind, &values[r * cols..(r + 1) * cols], &mut row);
            raw.extend_from_slice(&row);
        }
        raw
    }

    fn linear(backend: &dyn Backend, raw: &[u8], quant: QuantType, rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        let weight = backend.upload_weight("test", raw, quant, rows, cols, None).expect("upload weight");
        let mut stream = backend.create_stream().expect("stream");
        let mut xb = backend.allocate_buffer(x.len() * 4).expect("x");
        stream
            .upload(&mut xb, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
            .expect("upload x");
        let batch = x.len() / cols;
        let mut out = backend.allocate_buffer(batch * rows * 4).expect("out");
        stream.quantized_linear(&mut out, &xb, &weight, None).expect("linear");
        stream.synchronize().expect("synchronize");
        let mut bytes = vec![0u8; batch * rows * 4];
        stream.download(&mut bytes, &out).expect("download");
        bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    }

    /// Needs an Intel GPU (skipped without one): every batch size the dispatcher distinguishes (sub-group kernels up to 8, tiled FP32
    /// 9..15, matrix engine from 16) against the CPU backend.
    #[test]
    fn linear_matches_cpu_for_every_batch_path() {
        let Ok(device) = SyclBackend::new(0) else {
            eprintln!("no SYCL device: test skipped");
            return;
        };
        let cpu = crate::backend::cpu::CpuBackend::new();
        let (rows, cols) = (70usize, 512usize);
        let weights = noise(rows * cols, 7);
        for (kind, quant) in [
            (fllm_checkpoint::QType::Q4K, QuantType::Q4_K),
            (fllm_checkpoint::QType::Q8_0, QuantType::Q8_0),
            (fllm_checkpoint::QType::Q6K, QuantType::Q6_K),
            (fllm_checkpoint::QType::Q4_0, QuantType::Q4_0),
            (fllm_checkpoint::QType::Q5K, QuantType::Q5_K),
        ] {
            let raw = quantize(kind, rows, cols, &weights);
            for batch in [1usize, 2, 5, 8, 9, 16, 33, 100] {
                let x = noise(batch * cols, 11 + batch as u64);
                let got = linear(&device, &raw, quant, rows, cols, &x);
                let want = linear(&cpu, &raw, quant, rows, cols, &x);
                let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
                // the fp16 matrix-engine path (batch >= 16 on XMX devices) is good to about 1e-3 of the output scale
                let tol = if batch >= 16 && !is_exact_math() { 1e-2 } else { 1e-4 } * scale;
                let worst = got.iter().zip(&want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(worst <= tol, "{quant:?} batch {batch}: worst error {worst} > {tol}");
            }
        }
    }

    #[test]
    fn native_formats_cover_the_packed_kinds() {
        let names = native_formats();
        assert_eq!(names.len(), NATIVE_KINDS.len());
        for expected in ["F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q6_K", "IQ4_NL", "Mxfp4"] {
            assert!(names.iter().any(|n| n == expected), "missing {expected}");
        }
    }

    #[test]
    fn stub_build_reports_a_missing_runtime() {
        if ffi::NATIVE {
            return;
        }
        assert!(SyclBackend::new(0).is_err());
        assert!(device_info::list(1).is_empty());
        assert!(ffi::last_error().contains("no SYCL runtime"));
    }
}
