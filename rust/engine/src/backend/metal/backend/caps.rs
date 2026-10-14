//! What the Metal backend advertises through `Backend::capabilities`.

use crate::backend::metal::device_info::MetalDeviceInfo;
use crate::backend::metal::weights::supported_formats;
use crate::backend::BackendCapabilities;

/// Name of the backend (`Backend::name`, messages of the decoder).
pub const BACKEND_NAME: &str = "metal";

/// Capabilities of the Metal backend.
///
/// `supported_quants` lists the formats the quantized-linear kernels read natively (`weights::supported_formats`); every other format is
/// expanded to F16 / F32 at upload (see `upload.rs`) and then runs as a float weight. Only operations a kernel exists for are claimed:
/// `supports_device_sampling` stays false because `sampling_logits` is a host fallback except for greedy decoding (`argmax_rows`).
pub fn capabilities(info: &MetalDeviceInfo) -> BackendCapabilities {
    BackendCapabilities {
        name: format!("Metal ({})", info.name),
        supported_quants: supported_formats(),
        supports_rms_norm: true,
        supports_layer_norm: true,
        supports_rope: true,
        supports_swiglu: true,
        supports_geglu: true,
        supports_paged_attention: true,
        supports_kv_cache_write: true,
        supports_embedding_lookup: true,
        supports_device_sampling: false,
        supports_residual_add: true,
        supports_moe: true,
        max_batch_size: 4096,
        max_context: 262_144,
    }
}
