//! Reference CPU implementation of the backend traits.
//!
//! Serves as the fallback for unimplemented device operations and as the baseline reference
//! for `fllm-rs selftest-gpu`. Every operation copies its inputs out of the buffers first, so
//! an output may alias an input and no two buffer locks are ever held at once.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

mod moe;

static CPU_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

/// Internal CPU buffer backing store. The bytes live in `u64` words so every 2-, 4- and
/// 8-byte aligned window of the buffer can be viewed as `u16`/`f32` without misaligned reads.
pub struct CpuBufferInner {
    words: Mutex<Vec<u64>>,
    len: usize,
}

impl CpuBufferInner {
    fn new(len: usize) -> Self {
        Self {
            words: Mutex::new(vec![0u64; len.div_ceil(8)]),
            len,
        }
    }
}

fn inner(buf: &DeviceBuffer) -> Result<&CpuBufferInner, BackendError> {
    buf.handle
        .downcast_ref::<CpuBufferInner>()
        .ok_or_else(|| BackendError::BufferError("invalid CPU buffer handle".into()))
}

fn lock(inner: &CpuBufferInner) -> Result<MutexGuard<'_, Vec<u64>>, BackendError> {
    inner
        .words
        .lock()
        .map_err(|_| BackendError::BufferError("CPU buffer lock poisoned".into()))
}

fn bytes_of(words: &[u64], len: usize) -> &[u8] {
    debug_assert!(len <= words.len() * 8);
    // SAFETY: `words` owns at least `len` initialised bytes; u8 has no alignment requirement
    // and every bit pattern is valid. The view borrows `words` immutably.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), len) }
}

fn bytes_of_mut(words: &mut [u64], len: usize) -> &mut [u8] {
    debug_assert!(len <= words.len() * 8);
    // SAFETY: as in `bytes_of`, with the unique borrow of `words` carried over to the view.
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), len) }
}

fn window(buf: &DeviceBuffer, len: usize) -> Result<std::ops::Range<usize>, BackendError> {
    let end = buf
        .offset
        .checked_add(buf.size)
        .filter(|&end| end <= len)
        .ok_or_else(|| BackendError::BufferError("buffer slice out of range".into()))?;
    Ok(buf.offset..end)
}

fn with_buffer_slice<R>(buf: &DeviceBuffer, f: impl FnOnce(&[u8]) -> R) -> Result<R, BackendError> {
    let inner = inner(buf)?;
    let words = lock(inner)?;
    let range = window(buf, inner.len)?;
    Ok(f(&bytes_of(&words, inner.len)[range]))
}

fn with_buffer_slice_mut<R>(
    buf: &DeviceBuffer,
    f: impl FnOnce(&mut [u8]) -> R,
) -> Result<R, BackendError> {
    let inner = inner(buf)?;
    let mut words = lock(inner)?;
    let range = window(buf, inner.len)?;
    Ok(f(&mut bytes_of_mut(&mut words, inner.len)[range]))
}

/// Run `f` on the buffer viewed as `T` values (`f32`, `u16`, `u32`). The window must start
/// and end on a `T` boundary; the allocation itself is 8-byte aligned.
fn with_typed_mut<T: Copy, R>(
    buf: &DeviceBuffer,
    f: impl FnOnce(&mut [T]) -> R,
) -> Result<R, BackendError> {
    let width = std::mem::size_of::<T>();
    if !buf.offset.is_multiple_of(width) || !buf.size.is_multiple_of(width) {
        return Err(BackendError::BufferError(format!(
            "buffer window {}+{} is not aligned to {width} bytes",
            buf.offset, buf.size
        )));
    }
    with_buffer_slice_mut(buf, |bytes| {
        // SAFETY: the allocation is u64-aligned and the window starts at a multiple of
        // `size_of::<T>()`, so the pointer is aligned for `T` (alignment <= 8); the length is
        // a whole number of `T`. `T` is only instantiated with plain number types, for which
        // every bit pattern is valid. The view keeps the unique borrow of `bytes`.
        let values = unsafe {
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<T>(), bytes.len() / width)
        };
        f(values)
    })
}

fn read_f32(buf: &DeviceBuffer) -> Result<Vec<f32>, BackendError> {
    with_typed_mut::<f32, _>(buf, |s| s.to_vec())
}

fn read_bytes(buf: &DeviceBuffer) -> Result<Vec<u8>, BackendError> {
    with_buffer_slice(buf, <[u8]>::to_vec)
}

fn read_f32_range(buf: &DeviceBuffer, offset_bytes: usize, count: usize) -> Result<Vec<f32>, BackendError> {
    with_typed_mut::<f32, _>(buf, |dst| {
        let start = offset_bytes / 4;
        if dst.len() < start + count {
            return Err(BackendError::InvalidShape("read_f32_range out of bounds".into()));
        }
        Ok(dst[start..start + count].to_vec())
    })?
}

fn write_f32_range(buf: &DeviceBuffer, offset_bytes: usize, src: &[f32]) -> Result<(), BackendError> {
    with_typed_mut::<f32, _>(buf, |dst| {
        let start = offset_bytes / 4;
        if dst.len() < start + src.len() {
            return Err(BackendError::InvalidShape("write_f32_range out of bounds".into()));
        }
        dst[start..start + src.len()].copy_from_slice(src);
        Ok(())
    })?
}

/// Copy `src` to the front of `buf`, which must be large enough.
fn write_f32(buf: &DeviceBuffer, src: &[f32], op: &str) -> Result<(), BackendError> {
    with_typed_mut::<f32, _>(buf, |dst| {
        if dst.len() < src.len() {
            return Err(BackendError::InvalidShape(format!(
                "{op}: output holds {} values, needs {}",
                dst.len(),
                src.len()
            )));
        }
        dst[..src.len()].copy_from_slice(src);
        Ok(())
    })?
}

fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

/// Host CPU backend device.
pub struct CpuBackend {
    capabilities: BackendCapabilities,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        let all_quants = vec![
            QuantType::F32,
            QuantType::F16,
            QuantType::BF16,
            QuantType::Q4_0,
            QuantType::Q4_1,
            QuantType::Q5_0,
            QuantType::Q5_1,
            QuantType::Q8_0,
            QuantType::Q8_1,
            QuantType::Q2_K,
            QuantType::Q3_K,
            QuantType::Q4_K,
            QuantType::Q5_K,
            QuantType::Q6_K,
            QuantType::Q8_K,
            QuantType::IQ1_S,
            QuantType::IQ2_XXS,
            QuantType::IQ2_XS,
            QuantType::IQ2_S,
            QuantType::IQ3_XXS,
            QuantType::IQ3_S,
            QuantType::IQ4_NL,
            QuantType::IQ4_XS,
            QuantType::Fp8E4m3,
            QuantType::Fp8E5m2,
            QuantType::NvFp4,
            QuantType::Q4Z32,
            QuantType::Q4Z64,
            QuantType::Q4Z128,
            QuantType::Nf4B64,
            QuantType::Nf4B128,
            QuantType::Fp4B64,
            QuantType::Fp4B128,
            QuantType::Int8Row,
        ];
        Self {
            capabilities: BackendCapabilities {
                name: "CPU".to_string(),
                supported_quants: all_quants,
                supports_rms_norm: true,
                supports_grouped_rms_norm: true,
                supports_layer_norm: true,
                supports_rope: true,
                supports_swiglu: true,
                supports_geglu: true,
                supports_paged_attention: true,
                supports_kv_cache_write: true,
                supports_embedding_lookup: true,
                supports_device_sampling: true,
                sample_on_device: true,
                supports_residual_add: true,
                supports_moe: true,
                max_batch_size: 1024,
                max_context: 131072,
            },
        }
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "CPU"
    }

    fn device_id(&self) -> usize {
        0
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn create_stream(&self) -> Result<Box<dyn Stream>, BackendError> {
        Ok(Box::new(CpuStream))
    }

    fn allocate_buffer(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        let id = CPU_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
        Ok(DeviceBuffer::new(
            id,
            size,
            0,
            Arc::new(CpuBufferInner::new(size)),
        ))
    }

    fn free_buffer(&self, _buffer: DeviceBuffer) -> Result<(), BackendError> {
        // Drop cleans up memory automatically
        Ok(())
    }

    fn upload_pointer_table(&self, weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
        let buffers: Vec<&DeviceBuffer> = weights.iter().map(|w| &w.buffer).collect();
        moe::buffer_table(self, &buffers)
    }

    fn upload_bias_table(&self, weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
        let buffers: Vec<&DeviceBuffer> = weights
            .iter()
            .map(|w| w.bias.as_ref().ok_or_else(|| invalid("upload_bias_table: weight without a bias")))
            .collect::<Result<_, _>>()?;
        moe::buffer_table(self, &buffers)
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
        let buffer = self.allocate_buffer(raw_bytes.len())?;
        with_buffer_slice_mut(&buffer, |dst| dst.copy_from_slice(raw_bytes))?;

        let bias_buf = if let Some(b) = bias {
            if b.len() != rows {
                return Err(invalid(format!(
                    "{name}: bias has {} values for {rows} rows",
                    b.len()
                )));
            }
            let buf = self.allocate_buffer(b.len() * 4)?;
            write_f32(&buf, b, "upload_weight")?;
            Some(buf)
        } else {
            None
        };

        Ok(DeviceWeight {
            name: name.to_string(),
            buffer,
            quant_type,
            rows,
            cols,
            bias: bias_buf,
            scale: 1.0,
            perm: None,
        })
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
        let (words_per_layer_k, words_per_layer_v) = DeviceKvCache::geometry(
            layers,
            pool_blocks,
            block_size,
            kv_heads,
            head_dim,
            value_dim,
            kv_kind,
        )?;
        let key_pool = self.allocate_buffer(layers * words_per_layer_k * 2)?;
        let value_pool = self.allocate_buffer(layers * words_per_layer_v * 2)?;

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

    fn allocate_kv_cache_layered(
        &self,
        layers: usize,
        pool_blocks: usize,
        block_size: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        kv_kind: i32,
        layout: Option<crate::kv_geometry::KvPoolLayout>,
        layer_geoms: Option<Vec<crate::config::LayerGeom>>,
    ) -> Result<DeviceKvCache, BackendError> {
        let (words_per_layer_k, words_per_layer_v) = DeviceKvCache::geometry(
            layers,
            pool_blocks,
            block_size,
            kv_heads,
            head_dim,
            value_dim,
            kv_kind,
        )?;
        let key_words = layout.as_ref().map_or(layers * words_per_layer_k, |l| l.key_total);
        let value_words = layout.as_ref().map_or(layers * words_per_layer_v, |l| l.value_total);
        let key_pool = self.allocate_buffer(key_words * 2)?;
        let value_pool = self.allocate_buffer(value_words * 2)?;

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
            layout,
            layer_geoms,
        })
    }
}

/// One layer of a KV pool, addressed like `rust_decoder.rs` (see [`DeviceKvCache`]).
struct LayerCache<'a> {
    words: &'a [u16],
    layout: crate::kv_format::KvLayout,
}

impl<'a> LayerCache<'a> {
    fn new(words: &'a [u16], kv: &DeviceKvCache, key: bool) -> Self {
        let (k, v) = kv.layouts();
        Self { words, layout: if key { k } else { v } }
    }

    /// Value `c` (`head * dim + d`) of `slot`.
    fn value(&self, slot: usize, c: usize) -> f32 {
        self.layout.value(self.words, slot, c)
    }
}

/// Scalar paged attention covering every option of [`AttentionParams`]; the reference for
/// configurations the tiled kernel does not take (any head width, scale, soft cap, window,
/// chunk, sinks, ALiBi, key width != value width). Mirrors `attention_reference` in the decoder.
fn attention_scalar(
    q: &[f32],
    key: &LayerCache<'_>,
    value: &LayerCache<'_>,
    block_tables: &[i64],
    lengths: &[i64],
    p: &AttentionParams,
) -> Vec<f32> {
    let (dk, dv, heads) = (p.head_dim, p.value_dim, p.num_heads);
    let group = heads / p.num_kv_heads;
    let mut out = vec![0.0f32; lengths.len() * heads * dv];
    let mut scores = Vec::new();
    for (r, &n) in lengths.iter().enumerate() {
        let n = n as usize;
        let start = match (p.chunk, p.window) {
            (Some(c), _) => (n - 1) / c * c,
            (None, Some(w)) => n.saturating_sub(w),
            (None, None) => 0,
        };
        let slot_of = |pos: usize| {
            block_tables[r * p.stride + pos / p.block_size] as usize * p.block_size
                + pos % p.block_size
        };
        for h in 0..heads {
            let kh = h / group;
            let query = &q[(r * heads + h) * dk..(r * heads + h + 1) * dk];
            scores.clear();
            for pos in start..n {
                let slot = slot_of(pos);
                let mut s = 0.0f32;
                for (d, &qv) in query.iter().enumerate() {
                    s += qv * key.value(slot, kh * dk + d);
                }
                s *= p.scale;
                if let Some(slopes) = &p.alibi_slopes {
                    s += slopes[h] * (pos as f32 - (n - 1) as f32);
                }
                if let Some(cap) = p.softcap {
                    s = cap * (s / cap).tanh();
                }
                scores.push(s);
            }
            let sink = p.sinks.as_ref().map(|s| s[h]);
            let max = scores
                .iter()
                .copied()
                .fold(sink.unwrap_or(f32::NEG_INFINITY), f32::max);
            let mut sum = sink.map_or(0.0, |s| (s - max).exp());
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }
            let head_out = &mut out[(r * heads + h) * dv..(r * heads + h + 1) * dv];
            for (i, &w) in scores.iter().enumerate() {
                let slot = slot_of(start + i);
                let w = w / sum;
                for (d, o) in head_out.iter_mut().enumerate() {
                    *o += w * value.value(slot, kh * dv + d);
                }
            }
        }
    }
    out
}

/// Execution stream for host CPU.
pub struct CpuStream;

impl Stream for CpuStream {
    fn synchronize(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    fn upload(&mut self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<(), BackendError> {
        if src.len() > dst.size {
            return Err(BackendError::BufferError(format!(
                "upload overflow: src {} > dst {}",
                src.len(),
                dst.size
            )));
        }
        with_buffer_slice_mut(dst, |d| d[..src.len()].copy_from_slice(src))
    }

    fn download(&mut self, dst: &mut [u8], src: &DeviceBuffer) -> Result<(), BackendError> {
        if dst.len() > src.size {
            return Err(BackendError::BufferError(format!(
                "download overflow: dst {} > src {}",
                dst.len(),
                src.size
            )));
        }
        with_buffer_slice(src, |s| dst.copy_from_slice(&s[..dst.len()]))
    }

    fn copy_buffer(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        let copy_len = dst.size.min(src.size);
        let src_data = with_buffer_slice(src, |s| s[..copy_len].to_vec())?;
        with_buffer_slice_mut(dst, |d| d[..copy_len].copy_from_slice(&src_data))
    }

    fn embedding_lookup(
        &mut self,
        out: &mut DeviceBuffer,
        ids: &[i64],
        weight: &DeviceWeight,
    ) -> Result<(), BackendError> {
        let hidden = weight.cols;
        if weight.rows == 0 || !weight.buffer.size.is_multiple_of(weight.rows) {
            return Err(invalid("embedding table size is not rows x row bytes"));
        }
        let row_bytes = weight.buffer.size / weight.rows;
        let kind_id = weight.quant_type.to_gguf_type_id();
        let weight_bytes = read_bytes(&weight.buffer)?;

        let mut values = Vec::with_capacity(ids.len() * hidden);
        for &id in ids {
            if id < 0 || id as usize >= weight.rows {
                return Err(invalid(format!("embedding token id {id} out of bounds")));
            }
            let row = &weight_bytes[id as usize * row_bytes..(id as usize + 1) * row_bytes];
            values.extend(
                crate::compute::dequantize(row, kind_id, hidden)
                    .map_err(BackendError::DriverError)?,
            );
        }
        write_f32(out, &values, "embedding_lookup")
    }

    fn rms_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        eps: f32,
    ) -> Result<(), BackendError> {
        let w = read_f32(weight)?;
        let width = w.len();
        let mut values = read_f32(x)?;
        if width == 0 || !values.len().is_multiple_of(width) {
            return Err(invalid("rms_norm input is not rows of the weight width"));
        }
        for row in values.chunks_exact_mut(width) {
            let inv = (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + eps)
                .sqrt()
                .recip();
            for (val, &weight_val) in row.iter_mut().zip(&w) {
                *val *= inv * weight_val;
            }
        }
        write_f32(out, &values, "rms_norm")
    }

    fn rms_norm_grouped(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        eps: f32,
        groups: usize,
    ) -> Result<(), BackendError> {
        if groups == 0 {
            return Err(invalid("rms_norm_grouped requires groups > 0"));
        }
        let w = read_f32(weight)?;
        let width = w.len();
        if width == 0 || !width.is_multiple_of(groups) {
            return Err(invalid("rms_norm_grouped weight width is not divisible by groups"));
        }
        let mut values = read_f32(x)?;
        if !values.len().is_multiple_of(width) {
            return Err(invalid("rms_norm_grouped input is not rows of the weight width"));
        }
        let size = width / groups;
        for row in values.chunks_exact_mut(width) {
            for (chunk_x, chunk_w) in row.chunks_exact_mut(size).zip(w.chunks_exact(size)) {
                let inv = (chunk_x.iter().map(|v| v * v).sum::<f32>() / size as f32 + eps)
                    .sqrt()
                    .recip();
                for (val, &weight_val) in chunk_x.iter_mut().zip(chunk_w) {
                    *val *= inv * weight_val;
                }
            }
        }
        write_f32(out, &values, "rms_norm_grouped")
    }

    fn layer_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        eps: f32,
    ) -> Result<(), BackendError> {
        let w = read_f32(weight)?;
        let width = w.len();
        let b = bias.map(read_f32).transpose()?;
        if b.as_ref().is_some_and(|b| b.len() != width) {
            return Err(invalid("layer_norm bias width differs from weight"));
        }
        let mut values = read_f32(x)?;
        if width == 0 || !values.len().is_multiple_of(width) {
            return Err(invalid("layer_norm input is not rows of the weight width"));
        }
        for row in values.chunks_exact_mut(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inv = (var + eps).sqrt().recip();
            for (i, (val, &weight_val)) in row.iter_mut().zip(&w).enumerate() {
                let bias_val = b.as_ref().map_or(0.0, |bv| bv[i]);
                *val = (*val - mean) * inv * weight_val + bias_val;
            }
        }
        write_f32(out, &values, "layer_norm")
    }

    fn rope(
        &mut self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        positions: &[usize],
        params: &RopeParams,
    ) -> Result<(), BackendError> {
        params.validate(positions, q.size, k.size)?;
        let half = params.rope_dim / 2;
        let rotate = |values: &mut [f32], heads: usize| {
            for (r, &pos) in positions.iter().enumerate() {
                let cos = &params.cos[pos * half..(pos + 1) * half];
                let sin = &params.sin[pos * half..(pos + 1) * half];
                for h in 0..heads {
                    let head = &mut values[(r * heads + h) * params.dim..][..params.rope_dim];
                    for d in 0..half {
                        let (a_idx, b_idx) = if params.neox {
                            (d, d + half)
                        } else {
                            (2 * d, 2 * d + 1)
                        };
                        let (a, b) = (head[a_idx], head[b_idx]);
                        head[a_idx] = a * cos[d] - b * sin[d];
                        head[b_idx] = b * cos[d] + a * sin[d];
                    }
                }
            }
        };
        with_typed_mut::<f32, _>(q, |values| rotate(values, params.heads))?;
        with_typed_mut::<f32, _>(k, |values| rotate(values, params.kv_heads))?;
        Ok(())
    }

    fn quantized_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
    ) -> Result<(), BackendError> {
        // block-scaled FP8 weights are a GPU (Marlin) layout; this backend has no kernel for them
        if weight.quant_type == QuantType::Fp8Block {
            return Err(BackendError::UnsupportedQuant(QuantType::Fp8Block));
        }
        let x_data = read_f32(x)?;
        let b_data = bias.map(read_f32).transpose()?;
        let (rows, cols) = (weight.rows, weight.cols);
        if rows == 0 || cols == 0 || !x_data.len().is_multiple_of(cols) {
            return Err(invalid("linear input dimensions do not match"));
        }
        if b_data.as_ref().is_some_and(|b| b.len() < rows) {
            return Err(invalid("linear bias shorter than the output rows"));
        }
        if weight.quant_type == QuantType::Fp8Marlin {
            let w_bytes = read_bytes(&weight.buffer)?;
            let batch = x_data.len() / cols;
            if w_bytes.len() != rows * cols + rows * 2 {
                return Err(invalid(format!("FP8M weight size mismatch: {} vs {}", w_bytes.len(), rows * cols + rows * 2)));
            }
            let (weight_bytes, scale_bytes) = w_bytes.split_at(rows * cols);
            let mut out_data = vec![0.0f32; batch * rows];
            for r in 0..rows {
                let s_bytes = [scale_bytes[r * 2], scale_bytes[r * 2 + 1]];
                let scale = half::f16::from_le_bytes(s_bytes).to_f32();
                let r_w = &weight_bytes[r * cols..(r + 1) * cols];
                let w_row: Vec<f32> = r_w.iter().map(|&b| {
                    let sign = if b & 0x80 == 0 { 1.0 } else { -1.0 };
                    let exp = (b >> 3) & 0x0f;
                    let mant = b & 0x07;
                    let f = if exp == 0 {
                        sign * (mant as f32) * 2f32.powi(-9)
                    } else if exp == 0x0f && mant == 0x07 {
                        f32::NAN
                    } else {
                        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
                    };
                    f * scale
                }).collect();
                for b in 0..batch {
                    let mut sum = 0.0f32;
                    let x_row = &x_data[b * cols..(b + 1) * cols];
                    for (&xv, &wv) in x_row.iter().zip(&w_row) {
                        sum += xv * wv;
                    }
                    if let Some(bias_vec) = &b_data {
                        sum += bias_vec[r];
                    }
                    out_data[b * rows + r] = sum;
                }
            }
            return write_f32(out, &out_data, "quantized_linear");
        }
        let is_q4zp = weight.quant_type.is_q4zp();
        let perm_bytes = if is_q4zp { cols * 4 } else { 0 };
        if weight.buffer.size < perm_bytes || !(weight.buffer.size - perm_bytes).is_multiple_of(rows) {
            return Err(invalid("linear weight size is not rows x row bytes (+ perm)"));
        }
        let batch = x_data.len() / cols;
        let row_bytes = (weight.buffer.size - perm_bytes) / rows;
        let kind = match weight.quant_type {
            QuantType::Q4ZP32 => 0x8000_0010,
            QuantType::Q4ZP64 => 0x8000_0011,
            QuantType::Q4ZP128 => 0x8000_0012,
            _ => weight.quant_type.to_gguf_type_id(),
        };
        let w_bytes = read_bytes(&weight.buffer)?;
        if weight.quant_type.is_bnb() {
            // NF4 / FP4 / int8 blocks: the packed-weight kernel (f32 activations), the dequantizing loop below when the shape does not fit
            let bias = b_data.as_deref().map(|b| &b[..rows]);
            if let Some(values) = crate::cpu_bnb::linear(&w_bytes, kind, &x_data, rows, cols, bias) {
                return write_f32(out, &values, "quantized_linear");
            }
        }
        let perm: Option<Vec<usize>> = if is_q4zp {
            let p_bytes = &w_bytes[rows * row_bytes..];
            Some((0..cols).map(|c| u32::from_le_bytes(p_bytes[c * 4..(c + 1) * 4].try_into().unwrap()) as usize).collect())
        } else {
            None
        };

        // One weight row at a time keeps the dequantized copy at cols values.
        let mut out_data = vec![0.0f32; batch * rows];
        for r in 0..rows {
            let w_row = crate::compute::dequantize(
                &w_bytes[r * row_bytes..(r + 1) * row_bytes],
                kind,
                cols,
            )
            .map_err(BackendError::DriverError)?;
            for b in 0..batch {
                let mut sum = 0.0f32;
                if let Some(p) = &perm {
                    for (c, &wv) in w_row.iter().enumerate() {
                        sum += x_data[b * cols + p[c]] * wv;
                    }
                } else {
                    for (xv, wv) in x_data[b * cols..(b + 1) * cols].iter().zip(&w_row) {
                        sum += xv * wv;
                    }
                }
                if let Some(bias_vec) = &b_data {
                    sum += bias_vec[r];
                }
                out_data[b * rows + r] = sum;
            }
        }
        write_f32(out, &out_data, "quantized_linear")
    }

    fn swiglu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        let g = read_f32(gate)?;
        let u = read_f32(up)?;
        if g.len() != u.len() {
            return Err(invalid("swiglu gate and up dimensions mismatch"));
        }
        let values: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        write_f32(out, &values, "swiglu")
    }

    fn geglu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        let g = read_f32(gate)?;
        let u = read_f32(up)?;
        if g.len() != u.len() {
            return Err(invalid("geglu gate and up dimensions mismatch"));
        }
        // Tanh approximation, identical to the decoder's `gelu_tanh`.
        let values: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| {
                0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044_715 * g * g * g)).tanh()) * u
            })
            .collect();
        write_f32(out, &values, "geglu")
    }

    fn supports_kda(&self, s: &crate::backend::KdaShape) -> bool {
        s.heads > 0 && s.head_dim > 0 && s.conv_kernel >= 2
    }

    fn kda_layer(
        &mut self,
        s: &crate::backend::KdaShape,
        w: &crate::backend::KdaWeights<'_>,
        runs: &[crate::backend::DeltaRun<'_>],
        state_offset: usize,
        qkv: &DeviceBuffer,
        forget: &DeviceBuffer,
        beta: &DeviceBuffer,
        gate: &DeviceBuffer,
        _mixed: &mut DeviceBuffer,
        out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        let rows: usize = runs.iter().map(|r| r.rows).sum();
        if rows == 0 {
            return Ok(());
        }
        let width = s.heads * s.head_dim;
        let conv_dim = s.conv_dim();
        let conv_f32 = read_f32(w.conv)?;
        let dt_bias_f32 = read_f32(w.dt_bias)?;
        let rate_f32 = read_f32(w.rate)?;
        let norm_f32 = read_f32(w.norm)?;
        let qkv_f32 = read_f32(qkv)?;
        let forget_f32 = read_f32(forget)?;
        let beta_f32 = read_f32(beta)?;
        let gate_f32 = read_f32(gate)?;
        let layer_params = crate::cpu_kda::Layer {
            heads: s.heads,
            k_dim: s.head_dim,
            v_dim: s.head_dim,
            conv_kernel: s.conv_kernel,
            eps: s.eps,
            lower_bound: s.lower_bound,
            conv: &conv_f32,
            dt_bias: &dt_bias_f32,
            rate: &rate_f32,
            norm: &norm_f32,
        };
        let mut out_f32 = vec![0.0f32; rows * width];
        let rec_len = s.heads * s.head_dim * s.head_dim;
        let conv_len = conv_dim * (s.conv_kernel.saturating_sub(1));
        for run in runs {
            if run.rows == 0 {
                continue;
            }
            let start = run.first_row;
            let end = start + run.rows;
            let mut state_floats = read_f32_range(run.state, state_offset, rec_len + conv_len)?;
            if run.reset {
                state_floats.fill(0.0);
            }
            let (rec_state, conv_history) = state_floats.split_at_mut(rec_len);
            crate::cpu_kda::run(
                &layer_params,
                conv_history,
                rec_state,
                (
                    &qkv_f32[start * conv_dim..end * conv_dim],
                    &forget_f32[start * width..end * width],
                    &beta_f32[start * s.heads..end * s.heads],
                    &gate_f32[start * width..end * width],
                ),
                &mut out_f32[start * width..end * width],
                run.rows,
            );
            write_f32_range(run.state, state_offset, &state_floats)?;
        }
        write_f32(out, &out_f32, "kda_layer out")?;
        Ok(())
    }

    fn residual_add(
        &mut self,
        x: &mut DeviceBuffer,
        residual: &DeviceBuffer,
        scale: f32,
    ) -> Result<(), BackendError> {
        let res = read_f32(residual)?;
        with_typed_mut::<f32, _>(x, |xv| {
            if res.len() < xv.len() {
                return Err(invalid("residual shorter than its target"));
            }
            for (x_val, &r_val) in xv.iter_mut().zip(&res) {
                *x_val += r_val * scale;
            }
            Ok(())
        })?
    }

    #[allow(clippy::too_many_arguments)]
        #[allow(clippy::too_many_arguments)]
    fn attn_encoder(
        &mut self,
        out: &mut DeviceBuffer,
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        cu_seqlens: &[i32],
        batch_size: usize,
        total_tokens: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
        window: Option<usize>,
        global_tokens: Option<&[i32]>,
    ) -> Result<(), BackendError> {
        let q_f32 = read_f32(q)?;
        let k_f32 = read_f32(k)?;
        let v_f32 = read_f32(v)?;
        let mut out_f32 = vec![0.0f32; total_tokens * heads * head_dim];

        let gqa_ratio = heads / kv_heads;
        for b in 0..batch_size {
            let start = cu_seqlens[b] as usize;
            let end = cu_seqlens[b + 1] as usize;
            let seq_len = end - start;
            for i in 0..seq_len {
                let token_idx = start + i;
                let i_global = global_tokens.is_some_and(|g| g.contains(&(token_idx as i32)));
                for h in 0..heads {
                    let kh = h / gqa_ratio;
                    let q_vec = &q_f32[token_idx * heads * head_dim + h * head_dim..token_idx * heads * head_dim + (h + 1) * head_dim];
                    let mut scores = Vec::with_capacity(seq_len);
                    let mut max_score = -1e38f32;
                    for j in 0..seq_len {
                        let key_idx = start + j;
                        let mut visible = true;
                        if let Some(win) = window {
                            if win > 0 && win < seq_len {
                                let j_global = global_tokens.is_some_and(|g| g.contains(&(key_idx as i32)));
                                if !i_global && !j_global {
                                    let dist = (i as isize - j as isize).unsigned_abs();
                                    if dist > (win / 2) {
                                        visible = false;
                                    }
                                }
                            }
                        }
                        if visible {
                            let k_vec = &k_f32[key_idx * kv_heads * head_dim + kh * head_dim..key_idx * kv_heads * head_dim + (kh + 1) * head_dim];
                            let dot: f32 = q_vec.iter().zip(k_vec).map(|(&a, &b)| a * b).sum();
                            let s = dot * scale;
                            scores.push((j, s));
                            if s > max_score {
                                max_score = s;
                            }
                        }
                    }
                    let mut sum_exp = 0.0f32;
                    let mut exps = Vec::with_capacity(scores.len());
                    for &(j, s) in &scores {
                        let e = (s - max_score).exp();
                        exps.push((j, e));
                        sum_exp += e;
                    }
                    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                    let out_slice = &mut out_f32[token_idx * heads * head_dim + h * head_dim..token_idx * heads * head_dim + (h + 1) * head_dim];
                    for &(j, e) in &exps {
                        let weight = e * inv_sum;
                        let key_idx = start + j;
                        let v_vec = &v_f32[key_idx * kv_heads * head_dim + kh * head_dim..key_idx * kv_heads * head_dim + (kh + 1) * head_dim];
                        for d in 0..head_dim {
                            out_slice[d] += weight * v_vec[d];
                        }
                    }
                }
            }
        }
        write_f32(out, &out_f32, "attn_encoder")
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
        let (dk, dv, heads) = (params.head_dim, params.value_dim, params.num_heads);
        let rows = lengths.len();
        let q_data = read_f32(q)?;
        if q_data.len() < rows * heads * dk {
            return Err(invalid(
                "attention query smaller than rows x heads x head_dim",
            ));
        }
        let q_data = &q_data[..rows * heads * dk];
        let key_words = with_typed_mut::<u16, _>(&kv.key_pool, |s| s.to_vec())?;
        let value_words = with_typed_mut::<u16, _>(&kv.value_pool, |s| s.to_vec())?;
        let (k_off, v_off) = (kv.layer_key_offset(layer_idx), kv.layer_value_offset(layer_idx));
        let (words_k, words_v) = (kv.layer_key_words(layer_idx), kv.layer_value_words(layer_idx));
        if key_words.len() < k_off + words_k || value_words.len() < v_off + words_v {
            return Err(invalid("KV pools smaller than their layer geometry"));
        }
        let key_layer = &key_words[k_off..k_off + words_k];
        let value_layer = &value_words[v_off..v_off + words_v];

        // The tiled kernel computes exactly softmax(q k / sqrt(dk)) v; it declines (None)
        // head widths it has no tiles for, which then take the scalar path.
        let plain = params.window.is_none()
            && params.chunk.is_none()
            && params.softcap.is_none()
            && params.sinks.is_none()
            && params.alibi_slopes.is_none()
            && params.scale == 1.0 / (dk as f32).sqrt()
            && dk == dv
            && matches!(kv.kv_kind, 1 | 2);
        let tiled = if plain {
            let key_cache = crate::cpu_attention::Cache {
                data: key_layer,
                slots: kv.slots(),
                kv_heads: params.num_kv_heads,
                dim: dk,
                q8: kv.kv_kind == 2,
            };
            let value_cache = crate::cpu_attention::Cache {
                data: value_layer,
                ..key_cache
            };
            crate::cpu_attention::attention(
                q_data,
                key_cache,
                value_cache,
                block_tables,
                lengths,
                heads,
                params.block_size,
                params.stride,
            )
        } else {
            None
        };
        let attn_out = match tiled {
            Some(out) => out,
            None => attention_scalar(
                q_data,
                &LayerCache::new(key_layer, kv, true),
                &LayerCache::new(value_layer, kv, false),
                block_tables,
                lengths,
                params,
            ),
        };
        write_f32(out, &attn_out, "attention_paged")
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
        let k_data = read_f32(k)?;
        let v_data = read_f32(v)?;
        let kh = kv.layer_kv_heads(layer_idx);
        let kd = kv.layer_head_dim(layer_idx);
        let vd = kv.layer_value_dim(layer_idx);
        let (fmt_k, fmt_v) = kv.formats();
        let key_layout = crate::kv_format::KvLayout::new(fmt_k, kv.slots(), kh, kd);
        let value_layout = crate::kv_format::KvLayout::new(fmt_v, kv.slots(), kh, vd);
        for (pool, data, dim, off, words, layout) in [
            (&kv.key_pool, &k_data, kd, kv.layer_key_offset(layer_idx), kv.layer_key_words(layer_idx), key_layout),
            (&kv.value_pool, &v_data, vd, kv.layer_value_offset(layer_idx), kv.layer_value_words(layer_idx), value_layout),
        ] {
            let width = kh * dim;
            with_typed_mut::<u16, _>(pool, |all| {
                let layer = all
                    .get_mut(off..off + words)
                    .ok_or_else(|| invalid("KV pool smaller than its layer geometry"))?;
                for (r, &slot) in slots.iter().enumerate() {
                    layout.encode_slot(layer, slot, &data[r * width..(r + 1) * width]);
                }
                Ok::<(), BackendError>(())
            })??;
        }
        Ok(())
    }

    fn moe_route_ex(
        &mut self,
        logits: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        ids: &mut DeviceBuffer,
        weights: &mut DeviceBuffer,
        rows: usize,
        experts: usize,
        k: usize,
        route: &MoeRoute,
    ) -> Result<(), BackendError> {
        moe::route_ex(logits, bias, ids, weights, rows, experts, k, route)
    }

    fn moe_align(&mut self, ids: &DeviceBuffer, sort: &mut MoeSort, expert_offset: usize, num_local: usize) -> Result<(), BackendError> {
        moe::align(ids, sort, expert_offset, num_local)
    }

    fn moe_grouped_linear(&mut self, call: MoeGrouped<'_>) -> Result<(), BackendError> {
        moe::grouped(call)
    }

    fn moe_combine_ex(
        &mut self,
        out: &mut DeviceBuffer,
        y: &DeviceBuffer,
        weights: Option<&DeviceBuffer>,
        shared: Option<&DeviceBuffer>,
        gate: Option<&DeviceBuffer>,
        tokens: usize,
        k: usize,
        hidden: usize,
    ) -> Result<(), BackendError> {
        moe::combine_ex(out, y, weights, shared, gate, tokens, k, hidden)
    }

    fn moe_gate_scale(&mut self, gate: &mut DeviceBuffer, x: &DeviceBuffer, g: &DeviceBuffer, tokens: usize, hidden: usize) -> Result<(), BackendError> {
        moe::gate_scale(gate, x, g, tokens, hidden)
    }

    fn moe_route(
        &mut self,
        weights_out: &mut DeviceBuffer,
        indices_out: &mut DeviceBuffer,
        router_logits: &DeviceBuffer,
        num_experts: usize,
        top_k: usize,
    ) -> Result<(), BackendError> {
        let logits = read_f32(router_logits)?;
        if num_experts == 0 || top_k == 0 || top_k > num_experts {
            return Err(invalid("moe_route expert counts invalid"));
        }
        let rows = logits.len() / num_experts;

        let mut chosen_indices = Vec::with_capacity(rows * top_k);
        let mut chosen_weights = Vec::with_capacity(rows * top_k);

        for r in 0..rows {
            let row = &logits[r * num_experts..(r + 1) * num_experts];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();

            let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut top_sum = 0.0f32;
            for (idx, w) in indexed.iter().take(top_k) {
                chosen_indices.push(*idx as u32);
                top_sum += *w;
            }
            for (_, w) in indexed.iter().take(top_k) {
                chosen_weights.push(*w / top_sum.max(1e-8));
            }
        }

        write_f32(weights_out, &chosen_weights, "moe_route")?;
        with_typed_mut::<u32, _>(indices_out, |dst| {
            if dst.len() < chosen_indices.len() {
                return Err(invalid("moe_route index output too small"));
            }
            dst[..chosen_indices.len()].copy_from_slice(&chosen_indices);
            Ok(())
        })?
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
        let x_data = read_f32(x)?;
        let first = expert_weights
            .first()
            .ok_or_else(|| invalid("moe_expert_gemm without experts"))?;
        let hidden = first.cols;
        if hidden == 0 || !x_data.len().is_multiple_of(hidden) {
            return Err(invalid(
                "moe_expert_gemm input is not rows of the expert width",
            ));
        }
        let rows = x_data.len() / hidden;
        let out_width = first.rows;
        if expert_weights
            .iter()
            .any(|e| e.cols != hidden || e.rows != out_width)
        {
            return Err(invalid("moe experts differ in shape"));
        }
        let indices = with_typed_mut::<u32, _>(expert_indices, |s| s.to_vec())?;
        let weights = read_f32(expert_weights_buf)?;
        if indices.len() < rows * top_k || weights.len() < rows * top_k {
            return Err(invalid("moe routing buffers smaller than rows x top_k"));
        }

        let mut out_data = vec![0.0f32; rows * out_width];
        for r in 0..rows {
            let x_row = &x_data[r * hidden..(r + 1) * hidden];
            for k in 0..top_k {
                let exp_idx = indices[r * top_k + k] as usize;
                let exp_weight = weights[r * top_k + k];
                let Some(exp) = expert_weights.get(exp_idx) else {
                    continue;
                };
                let dequantized = crate::compute::dequantize(
                    &read_bytes(&exp.buffer)?,
                    exp.quant_type.to_gguf_type_id(),
                    exp.rows * exp.cols,
                )
                .map_err(BackendError::DriverError)?;
                for row_i in 0..exp.rows {
                    let w_slice = &dequantized[row_i * hidden..(row_i + 1) * hidden];
                    let dot: f32 = x_row.iter().zip(w_slice).map(|(a, b)| a * b).sum();
                    out_data[r * out_width + row_i] += dot * exp_weight;
                }
            }
        }
        write_f32(out, &out_data, "moe_expert_gemm")
    }

    fn sample_rows(&mut self, logits: &DeviceBuffer, vocab: usize, batch: &SampleBatch) -> Result<SampleOutput, BackendError> {
        batch.validate(vocab)?;
        let data = read_f32(logits)?;
        let rows = batch.rows.len();
        if vocab == 0 || data.len() < rows * vocab {
            return Err(invalid("sample_rows: the logits buffer is smaller than rows x vocab"));
        }
        let stride = batch.top_stride;
        let mut out = SampleOutput {
            ids: vec![0; rows],
            logprobs: vec![0.0; rows],
            top_ids: vec![-1; rows * stride],
            top_logprobs: vec![f32::NEG_INFINITY; rows * stride],
        };
        for (r, p) in batch.rows.iter().enumerate() {
            let pens = &batch.pen[p.pen_off as usize..(p.pen_off + p.pen_len) as usize];
            let bias = &batch.bias[p.bias_off as usize..(p.bias_off + p.bias_len) as usize];
            let mask = (p.mask_row >= 0).then(|| &batch.mask[p.mask_row as usize * batch.mask_words..(p.mask_row as usize + 1) * batch.mask_words]);
            let sample = crate::sampling::device_sample_reference(&data[r * vocab..(r + 1) * vocab], p, pens, bias, mask, stride);
            out.ids[r] = sample.id;
            out.logprobs[r] = sample.logprob;
            for (j, (id, lp)) in sample.top.iter().enumerate() {
                out.top_ids[r * stride + j] = *id;
                out.top_logprobs[r * stride + j] = *lp;
            }
        }
        Ok(out)
    }

    fn sampling_logits(
        &mut self,
        token_ids_out: &mut [u32],
        logits: &DeviceBuffer,
        params: &SamplingParams,
    ) -> Result<(), BackendError> {
        let logits_data = read_f32(logits)?;
        let (vocab, rows) = (params.vocab_size, params.rows);
        if vocab == 0 || logits_data.len() < rows * vocab || token_ids_out.len() < rows {
            return Err(invalid("sampling buffers smaller than rows x vocab"));
        }

        for r in 0..rows {
            let row = &logits_data[r * vocab..(r + 1) * vocab];
            if params.temperature <= 0.0 {
                // Argmax
                let mut best_idx = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for (idx, &v) in row.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_idx = idx;
                    }
                }
                token_ids_out[r] = best_idx as u32;
            } else {
                // Temperature + Top-P/Top-K
                let mut indexed: Vec<(usize, f32)> = row
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i, v / params.temperature))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if params.top_k > 0 && params.top_k < indexed.len() {
                    indexed.truncate(params.top_k);
                }
                let max_l = indexed[0].1;
                let mut sum = 0.0f32;
                for (_, l) in indexed.iter_mut() {
                    *l = (*l - max_l).exp();
                    sum += *l;
                }
                for (_, l) in indexed.iter_mut() {
                    *l /= sum;
                }
                if params.top_p < 1.0 {
                    let mut cum = 0.0f32;
                    let mut cut = indexed.len();
                    for (i, (_, p)) in indexed.iter().enumerate() {
                        cum += *p;
                        if cum >= params.top_p {
                            cut = i + 1;
                            break;
                        }
                    }
                    indexed.truncate(cut);
                }
                token_ids_out[r] = indexed[0].0 as u32;
            }
        }
        Ok(())
    }
}
