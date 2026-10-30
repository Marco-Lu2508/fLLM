//! Backend implementation of the native Vulkan backend (ash). Owner: package 4 (integration).
//!
//! `VulkanBackend` owns one device (`VkContext`): it allocates device memory, uploads weights in the plane layouts of `weights.rs`, allocates the
//! paged KV cache and hands out [`VulkanStream`](super::stream::VulkanStream)s, which record the operations of a decode step into one command
//! buffer. `--device vulkan[:N]` counts the usable *native* Vulkan devices (`VkContext::new(N)`); the translation of the global inventory index
//! (where wgpu devices also appear) is done by `devices::select_device`.
//!
//! Level of verification: L0 (compiled and unit-tested without a device; nothing here has ever run on a Vulkan driver).

pub(crate) mod convert;
pub(crate) mod experts;
pub(crate) mod mem;
pub(crate) mod shared;
pub(crate) mod upload;

#[cfg(test)]
mod hw_tests;
#[cfg(test)]
mod tests;

pub use shared::{is_exact_math, set_exact_math};

use super::stream::VulkanStream;
use crate::backend::vulkan::instance::{VkContext, VkError};
use crate::backend::vulkan::weights::{extended_formats, supported_formats};
use crate::backend::{log_once, Backend, BackendCapabilities, BackendError, DeviceBuffer, DeviceKvCache, DeviceWeight, QuantType, Stream};
use mem::{view_of, VkMem};
use shared::Shared;
use std::sync::Arc;

/// Quantization formats whose planar layouts and shaders exist, plus the plain float formats.
pub fn native_quants() -> Vec<QuantType> {
    let mut quants = vec![QuantType::F32, QuantType::F16, QuantType::BF16];
    quants.extend_from_slice(supported_formats());
    // IQ4_XS blocks are converted to IQ4_NL exactly on upload
    quants.push(QuantType::IQ4_XS);
    // bitsandbytes NF4 / FP4 / int8, block-wise FP8 and MLX affine types (shaders/vulkan/_q_fmt_{nf4,int8r,fp8b,mlx}.glsl); per-row scaled FP8 (FP8M)
    // is block-wise FP8 with one tile per row (converted on upload, `backend/upload.rs`)
    if shared::ext_formats_disabled() {
        return quants;
    }
    quants.extend(extended_formats());
    quants.push(QuantType::Fp8Marlin);
    // ExLlamaV2 matrices (shaders/vulkan/_q_fmt_exl2.glsl) and ExLlamaV3 matrices (exl3_{pre,mm,post}.comp)
    quants.push(QuantType::Exl2);
    quants.push(QuantType::Exl3);
    quants
}

pub struct VulkanBackend {
    shared: Arc<Shared>,
    name: String,
    capabilities: BackendCapabilities,
}

impl VulkanBackend {
    /// Open device `index` among the usable native Vulkan devices.
    pub fn new(index: usize) -> Result<Self, BackendError> {
        let ctx = VkContext::new(index).map_err(|error| match error {
            VkError::Loader(_) | VkError::NoDevice(_) => BackendError::DeviceNotFound(format!(
                "native Vulkan device {index} is not available: {error} (the wgpu backend, --features rust-gpu, is the fallback)"
            )),
            other => BackendError::from(other),
        })?;
        let name = ctx.caps().device_name.clone();
        let shared = Arc::new(Shared::new(ctx, index)?);
        let capabilities = BackendCapabilities {
            name: name.clone(),
            supported_quants: native_quants(),
            supports_rms_norm: true,
            supports_grouped_rms_norm: true,
            supports_layer_norm: true,
            supports_rope: true,
            supports_swiglu: true,
            supports_geglu: true,
            supports_paged_attention: true,
            supports_kv_cache_write: true,
            supports_embedding_lookup: true,
            // greedy sampling runs on the device (argmax_rows); sampling_logits goes through the host
            supports_device_sampling: false,
            sample_on_device: false,
            supports_residual_add: true,
            // moe_topk / moe_linear / moe_combine run on the device; moe_route / moe_expert_gemm use the host path
            supports_moe: false,
            max_batch_size: 4096,
            max_context: 1 << 20,
        };
        Ok(Self {
            shared,
            name,
            capabilities,
        })
    }

    /// The device context (for diagnostics and tests).
    pub fn context(&self) -> &Arc<VkContext> {
        &self.shared.ctx
    }
}

/// Create the backend for native Vulkan device `index` (`--device vulkan:N`). On failure the message says why; `Decoder::set_device` then
/// falls back to the wgpu backend when it is compiled in.
pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
    VulkanBackend::new(index)
        .map(|b| Arc::new(b) as Arc<dyn Backend>)
        .map_err(|e| e.to_string())
}

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_id(&self) -> usize {
        self.shared.device_id
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn create_stream(&self) -> Result<Box<dyn Stream>, BackendError> {
        Ok(Box::new(VulkanStream::new(Arc::clone(&self.shared))?))
    }

    fn allocate_buffer(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        self.shared.new_device_buffer(size)
    }

    fn free_buffer(&self, buffer: DeviceBuffer) -> Result<(), BackendError> {
        // the allocation goes back to the allocator when the last clone of the handle is dropped; the recorders keep what they still use
        drop(buffer);
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
        upload::upload_weight(&self.shared, name, raw_bytes, quant_type, rows, cols, bias, 1.0)
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
        // every shader applies the per-tensor scale (push constant), so the quantized bytes stay as they are
        upload::upload_weight(&self.shared, name, raw_bytes, quant_type, rows, cols, bias, scale)
    }

    fn alias_weights_into_table(&self, table: &DeviceBuffer, weights: &mut [&mut DeviceWeight]) -> Result<(), BackendError> {
        experts::alias_weights(&self.shared, table, weights)
    }

    fn upload_pointer_table(&self, weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
        experts::pointer_table(&self.shared, weights).inspect_err(|e| {
            log_once(
                "vulkan-expert-arena",
                &format!("the experts of a projection cannot be gathered into an arena ({e}); that layer takes the slower path"),
            )
        })
    }

    fn upload_bias_table(&self, weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
        experts::bias_table(&self.shared, weights).inspect_err(|e| {
            log_once(
                "vulkan-expert-bias-arena",
                &format!("the expert biases cannot be gathered into an arena ({e}); that layer takes the slower path"),
            )
        })
    }

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
        let key_pool = self.allocate_pool(layers, words_per_layer_k)?;
        let value_pool = self.allocate_pool(layers, words_per_layer_v)?;
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
        // one geometry for every layer: the plain pools
        let Some(layout) = layout.filter(|_| layer_geoms.is_some()) else {
            return self.allocate_kv_cache(layers, pool_blocks, block_size, kv_heads, head_dim, value_dim, kv_kind);
        };
        let (words_per_layer_k, words_per_layer_v) =
            DeviceKvCache::geometry(layers, pool_blocks, block_size, kv_heads, head_dim, value_dim, kv_kind)?;
        let key_pool = self.allocate_layered_pool(&layout.key_words)?;
        let value_pool = self.allocate_layered_pool(&layout.value_words)?;
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
            layout: Some(layout),
            layer_geoms,
        })
    }
}

impl VulkanBackend {
    /// A KV pool whose layers have their own sizes (`words[l]` 16-bit words; 0 for a layer that reads an earlier layer's cache), zero-filled.
    /// The layers lie back to back; a pool larger than `maxBufferSize` is split into allocations of whole layers.
    fn allocate_layered_pool(&self, words: &[usize]) -> Result<DeviceBuffer, BackendError> {
        let sh = &self.shared;
        let limit = sh.caps().max_buffer_size.min(sh.caps().max_memory_allocation_size).max(1);
        // (chunk, byte offset) per layer: back to back as in `KvPoolLayout`, so a pool of one allocation has the layout's offsets
        // (the KV block copies address it by them); `pool_window` binds an unaligned start through an aligned view and a skew
        let mut starts = Vec::with_capacity(words.len());
        let mut sizes: Vec<u64> = vec![0];
        for &w in words {
            let bytes = w as u64 * 2;
            if bytes.next_multiple_of(4) > limit {
                return Err(BackendError::OutOfMemory { requested: bytes as usize, available: limit as usize });
            }
            let last = sizes.len() - 1;
            if sizes[last] + bytes.next_multiple_of(4) > limit {
                sizes.push(0);
            }
            let last = sizes.len() - 1;
            starts.push((last, sizes[last]));
            sizes[last] += bytes;
        }
        let total: u64 = sizes.iter().map(|s| s.max(&4).next_multiple_of(16)).sum();
        let mut chunks = Vec::new();
        for size in &sizes {
            let size = size.max(&4).next_multiple_of(16);
            let buffer = sh.ctx.allocator().alloc_device(size, crate::backend::vulkan::memory::BufferUsage::COMPUTE)?;
            sh.zero_sync(&buffer, 0, size)?;
            chunks.push(buffer);
        }
        let head = chunks[0].clone();
        // one allocation: an ordinary buffer (the KV block copies address it as a whole), more: split like `allocate_pool`
        let chunks = if chunks.len() == 1 { Vec::new() } else { chunks };
        let mem = VkMem { buffer: head, chunks, layers_per_chunk: 0, layer_starts: starts, experts: None };
        Ok(DeviceBuffer::new(sh.next_id(), total as usize, sh.device_id, Arc::new(mem)))
    }

    /// One KV pool (`layers` layers of `words_per_layer` 16-bit words), zero-filled. A pool larger than `maxBufferSize` is split into one
    /// allocation per group of layers (the stream binds one layer window at a time, see `mem::pool_window`).
    fn allocate_pool(&self, layers: usize, words_per_layer: usize) -> Result<DeviceBuffer, BackendError> {
        let sh = &self.shared;
        let layer_bytes = (words_per_layer * 2).next_multiple_of(4);
        let total = (layers * words_per_layer * 2).next_multiple_of(4);
        let limit = sh.caps().max_buffer_size.min(sh.caps().max_memory_allocation_size).max(1) as usize;
        if total <= limit {
            let buffer = sh.new_device_buffer(total)?;
            sh.zero_sync(&view_of(&buffer)?, 0, total as u64)?;
            return Ok(buffer);
        }
        // consecutive layers start `words_per_layer * 2` bytes apart (not `layer_bytes`): a chunk holds whole layers
        let per_chunk = (limit / (words_per_layer * 2).max(1)).max(1);
        if (words_per_layer * 2).next_multiple_of(4).max(layer_bytes) > limit {
            return Err(BackendError::OutOfMemory {
                requested: layer_bytes,
                available: limit,
            });
        }
        let mut chunks = Vec::new();
        let mut first = 0;
        while first < layers {
            let n = per_chunk.min(layers - first);
            let bytes = (n * words_per_layer * 2).next_multiple_of(4);
            let buffer = sh.ctx.allocator().alloc_device(
                bytes.next_multiple_of(16) as u64,
                crate::backend::vulkan::memory::BufferUsage::COMPUTE,
            )?;
            sh.zero_sync(&buffer, 0, bytes.next_multiple_of(16) as u64)?;
            chunks.push(buffer);
            first += n;
        }
        let head = chunks[0].clone();
        let mem = VkMem {
            buffer: head,
            chunks,
            layers_per_chunk: per_chunk,
            layer_starts: Vec::new(),
            experts: None,
        };
        Ok(DeviceBuffer::new(sh.next_id(), total, sh.device_id, Arc::new(mem)))
    }
}
