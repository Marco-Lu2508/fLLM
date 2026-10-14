//! Backend of the `hip` feature (AMD ROCm). Owner: package 8.
//!
//! There is no separate Rust implementation: the HIP build of the kernel library (`cuda/*.cu` plus `hip/*.cu`, compiled by hipcc when
//! `FLLM_HIP=1`) exports the very C API the CUDA backend calls, so [`CudaBackend`] drives AMD GPUs unchanged. [`HipBackend`] wraps it to
//! give it the name `hip` and to log what the architecture offers (wave size, matrix cores). Every operation is forwarded.
//!
//! Level of verification: L0 (compiles; never run on hardware).

use super::device_info::{arch_of_device, ArchFamily};
use crate::backend::cuda::CudaBackend;
use crate::backend::{
    Backend, BackendCapabilities, BackendError, DeviceBuffer, DeviceKvCache, DeviceWeight,
    QuantType, Stream,
};
use std::sync::Arc;

/// The CUDA backend on a HIP device, under the name `hip`.
pub struct HipBackend {
    inner: CudaBackend,
    capabilities: BackendCapabilities,
    /// Architecture of the device (`gfx1100`), when the runtime reported one.
    arch: Option<String>,
}

impl HipBackend {
    /// Backend for HIP device `index` (`--device hip:N`; the ordinal of the HIP runtime).
    pub fn new(index: usize) -> Result<Self, BackendError> {
        let inner = CudaBackend::new(index)?;
        let arch = arch_of_device(index);
        let capabilities = inner.capabilities().clone();
        Ok(Self {
            inner,
            capabilities,
            arch,
        })
    }

    /// Architecture of the device, e.g. `gfx1100`.
    pub fn arch(&self) -> Option<&str> {
        self.arch.as_deref()
    }

    /// Family of the device (`Unknown` when no architecture is known).
    pub fn family(&self) -> ArchFamily {
        self.arch
            .as_deref()
            .map_or(ArchFamily::Unknown, ArchFamily::from_arch)
    }
}

/// Create the backend for HIP device `index`.
pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
    let backend = HipBackend::new(index).map_err(|e| format!("hip:{index}: {e}"))?;
    match backend.arch() {
        Some(arch) => eprintln!(
            "[fllm-rs] hip: device {index} is {arch} ({})",
            backend.family().summary()
        ),
        None => eprintln!("[fllm-rs] hip: device {index} (architecture unknown)"),
    }
    Ok(Arc::new(backend))
}

impl Backend for HipBackend {
    fn name(&self) -> &str {
        "hip"
    }

    fn device_id(&self) -> usize {
        self.inner.device_id()
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn create_stream(&self) -> Result<Box<dyn Stream>, BackendError> {
        self.inner.create_stream()
    }

    fn allocate_buffer(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        self.inner.allocate_buffer(size)
    }

    fn free_buffer(&self, buffer: DeviceBuffer) -> Result<(), BackendError> {
        self.inner.free_buffer(buffer)
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
        self.inner
            .upload_weight(name, raw_bytes, quant_type, rows, cols, bias)
    }

    #[allow(clippy::too_many_arguments)]
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
        self.inner
            .upload_weight_scaled(name, raw_bytes, quant_type, rows, cols, bias, scale)
    }

    fn upload_pointer_table(
        &self,
        weights: &[&DeviceWeight],
    ) -> Result<DeviceBuffer, BackendError> {
        self.inner.upload_pointer_table(weights)
    }

    fn upload_bias_table(&self, weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
        self.inner.upload_bias_table(weights)
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
        self.inner.allocate_kv_cache(
            layers,
            pool_blocks,
            block_size,
            kv_heads,
            head_dim,
            value_dim,
            kv_kind,
        )
    }
}
