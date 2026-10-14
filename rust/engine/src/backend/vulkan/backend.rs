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
use crate::backend::vulkan::weights::supported_formats;
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
            supports_layer_norm: true,
            supports_rope: true,
            supports_swiglu: true,
            supports_geglu: true,
            supports_paged_attention: true,
            supports_kv_cache_write: true,
            supports_embedding_lookup: true,
            // greedy sampling runs on the device (argmax_rows); sampling_logits goes through the host
            supports_device_sampling: false,
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
        })
    }
}

impl VulkanBackend {
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
            experts: None,
        };
        Ok(DeviceBuffer::new(sh.next_id(), total, sh.device_id, Arc::new(mem)))
    }
}
