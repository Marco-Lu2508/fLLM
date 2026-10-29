//! What the Metal backend advertises through `Backend::capabilities`.

use crate::backend::metal::device_info::MetalDeviceInfo;
use crate::backend::metal::weights::supported_formats;
use crate::backend::BackendCapabilities;

/// Name of the backend (`Backend::name`, messages of the decoder).
pub const BACKEND_NAME: &str = "metal";

/// Whether the environment variable `name` is set to something but `0`.
fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| v != "0")
}

/// Whether the kernels of `quant` are switched on. The kernels of the newer families have not run on every Apple GPU yet, so each family has an
/// escape hatch: `FLLM_METAL_NO_MLX=1` (MLX affine), `FLLM_METAL_NO_FP8B=1` (block-wise FP8), `FLLM_METAL_NO_BNB=1` (bitsandbytes NF4 / FP4 /
/// INT8) leave the family out of the capability list and out of the native upload: its weights are expanded to f16 when they are uploaded.
pub(super) fn native_enabled(quant: crate::backend::QuantType) -> bool {
    use crate::backend::QuantType as Q;
    let off = if quant.mlx_affine().is_some() {
        "FLLM_METAL_NO_MLX"
    } else if quant == Q::Fp8Block || matches!(quant, Q::Other(id) if fllm_gguf_probe::fp8b_type(id)) {
        "FLLM_METAL_NO_FP8B"
    } else if quant.is_bnb() {
        "FLLM_METAL_NO_BNB"
    } else {
        return true;
    };
    !env_flag(off)
}

/// The formats of `supported_formats` whose family is not switched off (see [`native_enabled`]).
fn advertised_formats() -> Vec<crate::backend::QuantType> {
    supported_formats().into_iter().filter(|&q| native_enabled(q)).collect()
}

/// Capabilities of the Metal backend.
///
/// `supported_quants` lists the formats the quantized-linear kernels read natively (`weights::supported_formats`); every other format is
/// expanded to F16 / F32 at upload (see `upload.rs`) and then runs as a float weight. Only operations a kernel exists for are claimed:
/// `supports_device_sampling` stays false because `sampling_logits` is a host fallback except for greedy decoding (`argmax_rows`).
pub fn capabilities(info: &MetalDeviceInfo) -> BackendCapabilities {
    BackendCapabilities {
        name: format!("Metal ({})", info.name),
        supported_quants: advertised_formats(),
        supports_rms_norm: true,
        supports_grouped_rms_norm: false,
        supports_layer_norm: true,
        supports_rope: true,
        supports_swiglu: true,
        supports_geglu: true,
        supports_paged_attention: true,
        supports_kv_cache_write: true,
        supports_embedding_lookup: true,
        supports_device_sampling: false,
        sample_on_device: false,
        supports_residual_add: true,
        supports_moe: true,
        max_batch_size: 4096,
        max_context: 262_144,
    }
}
