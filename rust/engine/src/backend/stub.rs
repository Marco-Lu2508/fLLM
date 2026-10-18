//! Placeholder backend used by the scaffolds of the native backends (Vulkan, Metal, HIP, SYCL, NPU, TPU).
//!
//! Every operation reports `UnsupportedOperation`. A backend package replaces its scaffold by a real
//! implementation of [`Backend`] / [`Stream`]; until then `--device <name>` fails with a clear message.
//! Nothing here touches hardware.

use super::{
    AttentionParams, Backend, BackendCapabilities, BackendError, DeviceBuffer, DeviceKvCache, DeviceWeight, QuantType, RopeParams,
    SamplingParams, Stream,
};

/// `Err(UnsupportedOperation(what))`.
pub fn unsupported<T>(what: &str) -> Result<T, BackendError> {
    Err(BackendError::UnsupportedOperation(what.to_owned()))
}

/// Capabilities of a backend that supports nothing yet.
pub fn empty_capabilities(name: &str) -> BackendCapabilities {
    BackendCapabilities {
        name: name.to_owned(),
        supported_quants: Vec::<QuantType>::new(),
        supports_rms_norm: false,
        supports_grouped_rms_norm: false,
        supports_layer_norm: false,
        supports_rope: false,
        supports_swiglu: false,
        supports_geglu: false,
        supports_paged_attention: false,
        supports_kv_cache_write: false,
        supports_embedding_lookup: false,
        supports_device_sampling: false,
        sample_on_device: false,
        supports_residual_add: false,
        supports_moe: false,
        max_batch_size: 0,
        max_context: 0,
    }
}

/// A stream whose every operation is unsupported.
pub struct StubStream {
    backend: String,
}

impl StubStream {
    pub fn new(backend: &str) -> Self {
        Self { backend: backend.to_owned() }
    }
    fn no<T>(&self, op: &str) -> Result<T, BackendError> {
        unsupported(&format!("{}: {op} (scaffold, not implemented)", self.backend))
    }
}

impl Stream for StubStream {
    fn synchronize(&mut self) -> Result<(), BackendError> {
        self.no("synchronize")
    }
    fn upload(&mut self, _dst: &mut DeviceBuffer, _src: &[u8]) -> Result<(), BackendError> {
        self.no("upload")
    }
    fn download(&mut self, _dst: &mut [u8], _src: &DeviceBuffer) -> Result<(), BackendError> {
        self.no("download")
    }
    fn copy_buffer(&mut self, _dst: &mut DeviceBuffer, _src: &DeviceBuffer) -> Result<(), BackendError> {
        self.no("copy_buffer")
    }
    fn embedding_lookup(&mut self, _out: &mut DeviceBuffer, _ids: &[i64], _weight: &DeviceWeight) -> Result<(), BackendError> {
        self.no("embedding_lookup")
    }
    fn rms_norm(&mut self, _out: &mut DeviceBuffer, _x: &DeviceBuffer, _weight: &DeviceBuffer, _eps: f32) -> Result<(), BackendError> {
        self.no("rms_norm")
    }
    fn layer_norm(
        &mut self,
        _out: &mut DeviceBuffer,
        _x: &DeviceBuffer,
        _weight: &DeviceBuffer,
        _bias: Option<&DeviceBuffer>,
        _eps: f32,
    ) -> Result<(), BackendError> {
        self.no("layer_norm")
    }
    fn rope(&mut self, _q: &mut DeviceBuffer, _k: &mut DeviceBuffer, _positions: &[usize], _params: &RopeParams) -> Result<(), BackendError> {
        self.no("rope")
    }
    fn quantized_linear(
        &mut self,
        _out: &mut DeviceBuffer,
        _x: &DeviceBuffer,
        _weight: &DeviceWeight,
        _bias: Option<&DeviceBuffer>,
    ) -> Result<(), BackendError> {
        self.no("quantized_linear")
    }
    fn swiglu(&mut self, _out: &mut DeviceBuffer, _gate: &DeviceBuffer, _up: &DeviceBuffer) -> Result<(), BackendError> {
        self.no("swiglu")
    }
    fn geglu(&mut self, _out: &mut DeviceBuffer, _gate: &DeviceBuffer, _up: &DeviceBuffer) -> Result<(), BackendError> {
        self.no("geglu")
    }
    fn residual_add(&mut self, _x: &mut DeviceBuffer, _residual: &DeviceBuffer, _scale: f32) -> Result<(), BackendError> {
        self.no("residual_add")
    }
    #[allow(clippy::too_many_arguments)]
    fn attention_paged(
        &mut self,
        _out: &mut DeviceBuffer,
        _q: &DeviceBuffer,
        _kv: &DeviceKvCache,
        _block_tables: &[i64],
        _lengths: &[i64],
        _layer_idx: usize,
        _params: &AttentionParams,
    ) -> Result<(), BackendError> {
        self.no("attention_paged")
    }
    fn kv_cache_write(
        &mut self,
        _kv: &DeviceKvCache,
        _layer_idx: usize,
        _slots: &[usize],
        _k: &DeviceBuffer,
        _v: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        self.no("kv_cache_write")
    }
    fn moe_route(
        &mut self,
        _weights_out: &mut DeviceBuffer,
        _indices_out: &mut DeviceBuffer,
        _router_logits: &DeviceBuffer,
        _num_experts: usize,
        _top_k: usize,
    ) -> Result<(), BackendError> {
        self.no("moe_route")
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
        self.no("moe_expert_gemm")
    }
    fn sampling_logits(&mut self, _token_ids_out: &mut [u32], _logits: &DeviceBuffer, _params: &SamplingParams) -> Result<(), BackendError> {
        self.no("sampling_logits")
    }
}

/// A backend that cannot be created: `new` reports that the scaffold is not implemented. It exists so that the module layout, the
/// feature flags and `--device <name>` are in place before a package fills the backend in.
pub struct StubBackend {
    name: String,
    device_id: usize,
    caps: BackendCapabilities,
}

impl StubBackend {
    /// Always an error: the named backend is a scaffold.
    pub fn new(name: &str, _device_id: usize) -> Result<Self, BackendError> {
        Err(BackendError::DeviceNotFound(format!("the {name} backend is a scaffold in this build (not implemented yet)")))
    }
    /// For tests of the scaffolding itself.
    pub fn placeholder(name: &str, device_id: usize) -> Self {
        Self { name: name.to_owned(), device_id, caps: empty_capabilities(name) }
    }
}

impl Backend for StubBackend {
    fn name(&self) -> &str {
        &self.name
    }
    fn device_id(&self) -> usize {
        self.device_id
    }
    fn capabilities(&self) -> &BackendCapabilities {
        &self.caps
    }
    fn create_stream(&self) -> Result<Box<dyn Stream>, BackendError> {
        Ok(Box::new(StubStream::new(&self.name)))
    }
    fn allocate_buffer(&self, _size: usize) -> Result<DeviceBuffer, BackendError> {
        unsupported(&format!("{}: allocate_buffer (scaffold)", self.name))
    }
    fn free_buffer(&self, _buffer: DeviceBuffer) -> Result<(), BackendError> {
        unsupported(&format!("{}: free_buffer (scaffold)", self.name))
    }
    fn upload_weight(
        &self,
        _name: &str,
        _raw_bytes: &[u8],
        _quant_type: QuantType,
        _rows: usize,
        _cols: usize,
        _bias: Option<&[f32]>,
    ) -> Result<DeviceWeight, BackendError> {
        unsupported(&format!("{}: upload_weight (scaffold)", self.name))
    }
    #[allow(clippy::too_many_arguments)]
    fn allocate_kv_cache(
        &self,
        _layers: usize,
        _pool_blocks: usize,
        _block_size: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _value_dim: usize,
        _kv_kind: i32,
    ) -> Result<DeviceKvCache, BackendError> {
        unsupported(&format!("{}: allocate_kv_cache (scaffold)", self.name))
    }
}
