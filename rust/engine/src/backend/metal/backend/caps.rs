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
/// INT8), `FLLM_METAL_NO_EXL=1` (EXL2) leave the family out of the capability list and out of the native upload: its weights are expanded to f16 when they are uploaded.
pub(super) fn native_enabled(quant: crate::backend::QuantType) -> bool {
    family_switch(quant).is_none_or(|name| !env_flag(name))
}

/// The environment variable that switches the family of `quant` off (`None`: the format has no escape hatch).
fn family_switch(quant: crate::backend::QuantType) -> Option<&'static str> {
    use crate::backend::QuantType as Q;
    if quant.mlx_affine().is_some() {
        Some("FLLM_METAL_NO_MLX")
    } else if quant == Q::Fp8Block || matches!(quant, Q::Other(id) if fllm_gguf_probe::fp8b_type(id)) {
        Some("FLLM_METAL_NO_FP8B")
    } else if quant.is_bnb() {
        Some("FLLM_METAL_NO_BNB")
    } else if quant == Q::Exl2 {
        Some("FLLM_METAL_NO_EXL")
    } else {
        None
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::QuantType as Q;

    #[test]
    fn every_new_family_has_an_escape_hatch_and_the_others_have_none() {
        assert_eq!(family_switch(Q::Fp8Block), Some("FLLM_METAL_NO_FP8B"));
        assert_eq!(family_switch(Q::from_gguf_type_id(fllm_gguf_probe::fp8b_type_id(16, 32).unwrap())), Some("FLLM_METAL_NO_FP8B"));
        for q in [Q::Nf4B64, Q::Nf4B128, Q::Fp4B64, Q::Fp4B128, Q::Int8Row] {
            assert_eq!(family_switch(q), Some("FLLM_METAL_NO_BNB"));
        }
        assert_eq!(family_switch(Q::Exl2), Some("FLLM_METAL_NO_EXL"));
        assert_eq!(family_switch(Q::Exl3), None);
        assert_eq!(family_switch(Q::mlx_affine_common()[0]), Some("FLLM_METAL_NO_MLX"));
        for q in [Q::Q4_K, Q::F16, Q::Fp8Marlin, Q::Mxfp4] {
            assert_eq!(family_switch(q), None);
        }
    }
}
