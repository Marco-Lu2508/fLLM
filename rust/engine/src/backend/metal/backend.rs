//! Native Metal backend (objc2-metal, Apple platforms only): the `Backend`. Owner: package 7 (integration).
//!
//! Unified memory: every buffer is a shared `MTLBuffer`, weights are uploaded with the layouts of `weights.rs` (most of them the bytes of the
//! GGUF tensor, wrapped without a copy when the loader registered its memory map, see [`MetalBackend::register_host_mapping`]), the KV cache is two
//! shared pools addressed through [`DeviceKvCache`]. The streams (`stream.rs`) plan every operation with the planners of the shader packages and
//! record it into one command buffer per step.
//!
//! * `caps`: capabilities; `shared`: state shared with the streams; `upload`: weight upload; `experts`: pointer tables of MoE layers;
//!   `convert`: pure host helpers; `kernels`: inventory of the planned kernels and its check against the MSL sources.
//!
//! Level of verification: L0 (compiles for `aarch64-apple-darwin`; nothing has run on a Mac).

mod caps;
pub mod convert;
mod experts;
pub mod kernels;
pub(crate) mod shared;
mod upload;

use super::buffers::{BufferRef, BufferUsage};
use super::commands::MetalContext;
use super::device_info::config::parse_bool;
use super::stream::MetalStream;
use crate::backend::{
    log_once, Backend, BackendCapabilities, BackendError, DeviceBuffer, DeviceKvCache,
    DeviceWeight, QuantType, Stream,
};
use shared::Shared;
use std::any::Any;
use std::sync::Arc;

pub struct MetalBackend {
    shared: Arc<Shared>,
    name: String,
    capabilities: BackendCapabilities,
}

impl MetalBackend {
    /// Opens Metal device `index` (`--device metal:N`, see `device_info::open` for the numbering). Compiles three probe kernels, so a shader source
    /// the Metal compiler rejects makes this fail (the caller falls back to wgpu) instead of failing during a generation.
    ///
    /// `FLLM_METAL_PREWARM=1` compiles all shader libraries up front (and fails on the first compiler error).
    pub fn new(index: usize) -> Result<Self, BackendError> {
        let context = MetalContext::new(index)?;
        let info = context.device.info().clone();
        let shared = Arc::new(Shared::new(context, info.ordinal)?);
        let prewarm = std::env::var("FLLM_METAL_PREWARM")
            .ok()
            .and_then(|v| parse_bool(&v))
            .unwrap_or(false);
        if prewarm {
            shared.ctx.pipelines.prewarm_libraries()?;
        }
        log_once("metal-device", &format!("Metal device: {}", info.summary()));
        Ok(Self {
            capabilities: caps::capabilities(&info),
            name: caps::BACKEND_NAME.to_owned(),
            shared,
        })
    }

    /// The context of the Metal core (device, memory manager, pipeline cache), for tools and tests.
    pub fn context(&self) -> &MetalContext {
        &self.shared.ctx
    }

    /// Lets the backend wrap `[ptr, ptr + len)` (the memory-mapped GGUF file) without a copy: weights uploaded afterwards from slices inside that
    /// range are bound in place (`weights::pack_weight` keeps most formats as they are in the file). `max_item_len` is the largest tensor in bytes.
    ///
    /// # Safety
    ///
    /// The memory must stay mapped and unchanged as long as `owner` lives, and `owner` must be what keeps it mapped (see
    /// `MetalContext::register_host_mapping`). The decoder does not call this yet; without it every weight is copied once.
    pub unsafe fn register_host_mapping(
        &self,
        owner: Arc<dyn Any + Send + Sync>,
        ptr: *const u8,
        len: usize,
        max_item_len: usize,
    ) -> Result<(), BackendError> {
        // SAFETY: forwarded to the caller.
        unsafe {
            self.shared
                .ctx
                .register_host_mapping(owner, ptr, len, max_item_len)
        }
        .map_err(BackendError::from)
    }
}

/// Bytes of a KV pool of `layers` layers of `words` 16-bit words, rounded up to 4.
fn pool_bytes(layers: usize, words: usize) -> Result<usize, BackendError> {
    layers
        .checked_mul(words)
        .and_then(|w| w.checked_mul(2))
        .map(|b| b.div_ceil(4) * 4)
        .ok_or_else(|| BackendError::InvalidShape("KV pool size overflows".into()))
}

impl Backend for MetalBackend {
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
        Ok(Box::new(MetalStream::new(Arc::clone(&self.shared))))
    }

    fn allocate_buffer(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        let buffer = self
            .shared
            .ctx
            .memory
            .alloc_zeroed(size, BufferUsage::Activations)?;
        Ok(self.shared.wrap(BufferRef::whole(buffer)))
    }

    fn free_buffer(&self, _buffer: DeviceBuffer) -> Result<(), BackendError> {
        // the Metal buffer is released with the last reference (streams that still use it hold their own)
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
        self.upload_weight_impl(name, raw_bytes, quant_type, rows, cols, bias, 1.0)
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
        // the kernels apply `DeviceWeight::scale` themselves (`q_params::scale`), so the quantized bytes stay as they are
        self.upload_weight_impl(name, raw_bytes, quant_type, rows, cols, bias, scale)
    }

    fn upload_pointer_table(&self, weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
        self.expert_table(weights, false)
    }

    fn upload_bias_table(&self, weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
        self.expert_table(weights, true)
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
        let (words_per_layer_k, words_per_layer_v) = DeviceKvCache::geometry(
            layers, pool_blocks, block_size, kv_heads, head_dim, value_dim, kv_kind,
        )?;
        let memory = &self.shared.ctx.memory;
        let key = memory.alloc_zeroed(pool_bytes(layers, words_per_layer_k)?, BufferUsage::KvCache)?;
        let value = memory.alloc_zeroed(pool_bytes(layers, words_per_layer_v)?, BufferUsage::KvCache)?;
        self.shared.mark_resident(&key);
        self.shared.mark_resident(&value);
        Ok(DeviceKvCache {
            key_pool: self.shared.wrap(BufferRef::whole(key)),
            value_pool: self.shared.wrap(BufferRef::whole(value)),
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

/// `--device metal[:N]`: the native backend (`Decoder::set_device` falls back to wgpu when this fails).
pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
    MetalBackend::new(index)
        .map(|backend| Arc::new(backend) as Arc<dyn Backend>)
        .map_err(|e| e.to_string())
}
