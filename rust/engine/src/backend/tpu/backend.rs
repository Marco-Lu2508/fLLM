//! Backend / stream implementation of the tpu backend (Google TPU through PJRT / StableHLO). Owner: package 12.
//!
//! The backend drives one TPU core through the PJRT C API of the `libtpu` plugin (see `pjrt.rs`). It has no hand-written kernels:
//! every operation of the `Stream` trait is a small StableHLO program (see `stablehlo.rs`) that the TPU compiler turns into code;
//! programs are compiled once per shape and cached. Weights are decoded on the host at upload and kept in bfloat16 (or f32, or int8
//! with group scales); activations are f32; the KV cache is f16 in one array per pool.
//!
//! Settings (environment):
//! - `FLLM_TPU_LIBRARY`: path of the PJRT plugin (`libtpu.so`); found automatically in the `libtpu` Python package.
//! - `FLLM_TPU_EXACT=1`: f32 weights and full-precision products (what `selftest-gpu` and CPU parity runs want).
//! - `FLLM_TPU_WEIGHTS=bf16|f32|int8`, `FLLM_TPU_NUMERICS=exact|fast`: the two parts separately.
//! - `FLLM_TPU_DEBUG=1`: log program compilations.
//!
//! Not implemented (reported as `UnsupportedOperation`, the decoder falls back): the q8 KV cache, DeltaNet layers, expert GEMMs
//! through pointer tables, streaming of host weights, CUDA-graph and half-precision fusion hooks.
//!
//! Level of verification: L0 (compiles; never run on a TPU).

pub mod mem;
pub mod ops;
pub mod plan;
pub mod stream;
pub mod weights;

#[cfg(test)]
mod tests;

use crate::backend::tpu::pjrt;
use crate::backend::tpu::stablehlo::DType;
use crate::backend::{
    Backend, BackendCapabilities, BackendError, DeviceBuffer, DeviceKvCache, DeviceWeight,
    QuantType, Stream,
};
use mem::{Cell, Settings, Shared};
use std::sync::Arc;

/// Formats the host decodes at upload (everything `compute::dequantize` reads) except Q6_K: advertising Q6_K would make the decoder
/// re-quantise a large output head to Q6_K, which the TPU only decodes again.
fn supported_quants() -> Vec<QuantType> {
    vec![
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
        QuantType::Q8_K,
        QuantType::IQ1_S,
        QuantType::IQ1_M,
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
        QuantType::Mxfp4,
        QuantType::Q4Z32,
        QuantType::Q4Z64,
        QuantType::Q4Z128,
        QuantType::Q4ZP32,
        QuantType::Q4ZP64,
        QuantType::Q4ZP128,
        QuantType::Fp8Marlin,
    ]
}

/// A TPU core driven through PJRT.
pub struct TpuBackend {
    shared: Arc<Shared>,
    name: String,
    caps: BackendCapabilities,
}

impl TpuBackend {
    /// The backend for TPU `index` of this process (`--device tpu:index`).
    pub fn new(index: usize) -> Result<Self, BackendError> {
        let runtime = pjrt::runtime().map_err(BackendError::DeviceNotFound)?;
        let device = runtime
            .client
            .devices()
            .get(index)
            .cloned()
            .ok_or_else(|| {
                BackendError::DeviceNotFound(format!(
                    "no TPU {index}: the plugin exposes {} device(s)",
                    runtime.client.devices().len()
                ))
            })?;
        let settings = Settings::from_env();
        let name = format!("TPU {} ({})", device.kind, runtime.client.platform_name());
        crate::backend::log_once(
            "tpu-settings",
            &format!(
                "tpu: {name}: weights {:?}, {:?} arithmetic (FLLM_TPU_EXACT=1 for f32); not on the device: q8 KV cache, DeltaNet layers, pointer-table expert GEMMs (the decoder uses its host or generic path); every new program shape is compiled once by the TPU compiler",
                settings.weights, settings.numerics
            ),
        );
        Ok(Self::with_shared(
            Arc::new(Shared::new(runtime, device, index, settings)),
            name,
        ))
    }

    /// A backend without a device: operations build their StableHLO programs and record them instead of running them (see
    /// `Shared::recorded`). For the unit tests, which check every program an operation emits.
    pub fn dry_run(settings: Settings) -> Self {
        Self::with_shared(Arc::new(Shared::dry(settings)), "TPU (dry run)".to_owned())
    }

    fn with_shared(shared: Arc<Shared>, name: String) -> Self {
        let caps = BackendCapabilities {
            name: name.clone(),
            supported_quants: supported_quants(),
            supports_rms_norm: true,
            supports_layer_norm: true,
            supports_rope: true,
            supports_swiglu: true,
            supports_geglu: true,
            supports_paged_attention: true,
            supports_kv_cache_write: true,
            supports_embedding_lookup: true,
            supports_device_sampling: true,
            sample_on_device: false,
            supports_residual_add: true,
            supports_moe: false,
            max_batch_size: 1024,
            max_context: 131_072,
        };
        Self { shared, name, caps }
    }
    pub fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }
}

impl Backend for TpuBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_id(&self) -> usize {
        self.shared.index
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.caps
    }

    fn create_stream(&self) -> Result<Box<dyn Stream>, BackendError> {
        Ok(Box::new(stream::TpuStream::new(Arc::clone(&self.shared))))
    }

    fn allocate_buffer(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        let cell = Cell::flat(size);
        Ok(DeviceBuffer::new(
            self.shared.next_id(),
            size,
            self.shared.index,
            cell,
        ))
    }

    fn free_buffer(&self, _buffer: DeviceBuffer) -> Result<(), BackendError> {
        // the device array is released when the last handle to it is dropped
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
        weights::upload(&self.shared, name, raw_bytes, quant_type, rows, cols, bias)
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
        if kv_kind == 2 {
            return Err(BackendError::UnsupportedOperation(
                "tpu: the q8_0 KV cache (use the f16 cache: --kv-cache-type f16)".into(),
            ));
        }
        let (words_per_layer_k, words_per_layer_v) = DeviceKvCache::geometry(
            layers,
            pool_blocks,
            block_size,
            kv_heads,
            head_dim,
            value_dim,
            kv_kind,
        )?;
        let slots = pool_blocks * block_size;
        let rows = layers
            .checked_mul(slots)
            .filter(|&r| r <= i32::MAX as usize)
            .ok_or_else(|| {
                BackendError::InvalidShape(
                    "the KV cache has more rows than the 32-bit indices of the TPU backend address"
                        .into(),
                )
            })?;
        let mut pools = Vec::new();
        for width in [kv_heads * head_dim, kv_heads * value_dim] {
            let cell = Cell::shaped(&[rows as i64, width as i64], DType::F16);
            self.shared.materialize(&cell, DType::F16)?;
            let size = rows * width * 2;
            pools.push(DeviceBuffer::new(
                self.shared.next_id(),
                size,
                self.shared.index,
                cell,
            ));
        }
        let value_pool = pools
            .pop()
            .ok_or_else(|| BackendError::DriverError("no value pool".into()))?;
        let key_pool = pools
            .pop()
            .ok_or_else(|| BackendError::DriverError("no key pool".into()))?;
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

/// Create the backend for TPU `index` (`--device tpu:index`).
pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
    TpuBackend::new(index)
        .map(|b| Arc::new(b) as Arc<dyn Backend>)
        .map_err(|e| e.to_string())
}
