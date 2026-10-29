//! Weight layouts of the vulkan backend. Owner: package 2 (device layouts and repacking of quantized weights, dispatch planning of quantized_linear).
//!
//! Two halves, both plain CPU code that is tested without a GPU:
//! - [`layout`]: [`DeviceLayout`] / [`pack_weight`] turn the GGUF bytes of a weight (`Q4_0` .. `Q6_K`, `IQ4_NL`, `MXFP4`, `NVFP4`, `Q4Z*`, `Q4ZP*`) into the
//!   plane layout the shaders in `shaders/vulkan/_q_fmt_*.glsl` read (struct of arrays, 16-byte aligned planes, some scales decoded on the host).
//! - [`plan`]: [`plan_quantized_linear`] chooses the shader variant (f32 / integer dot / cooperative matrix) for a device ([`VkCapsLike`]) and
//!   produces the specialization constants, push constants, workgroup counts, bindings and the activation prepass of one `quantized_linear`.
//!
//! Package 4 (integration) uploads `PackedWeight::bytes` (and `perm`) when `Backend::upload_weight` is called, and records the prepass and the
//! main dispatch of a [`LinearPlan`] in `Stream::quantized_linear`. The binding and push-constant contract is `shaders/vulkan/INTERFACE_QUANT.md`.
//! Level of verification: L0 (compiled, CPU unit tests of the layouts; the shaders were never run on a GPU).

pub mod exl3;
pub mod layout;
pub mod plan;

pub use layout::{
    exl2_device_type, exl2_device_type_of, exl3_device_type, exl3_device_type_of, extended_formats, is_exl3, format_info, pack_rows_of_blob, pack_weight, pack_weight_slices, rows_per_slice, supported_formats, DeviceLayout,
    uses_perm, FormatInfo, Geometry, PackedWeight, WeightSlice,
};
pub use plan::{
    exact_math_from_env, mmq_shared_bytes, mmq_tile, plan_quantized_linear, MmqKnobs, plan_quantized_linear_with, split_groups,
    BindingRole, BindingSlot, LinearPlan, PlanKind, PlanOptions, PrepassBinding, PrepassKind,
    PrepassPlan, PrepassRole, SpecConstant, Variant, VkCapsLike, MATVEC_MAX_BATCH, PUSH_BYTES,
};

#[cfg(test)]
mod mirror;
#[cfg(test)]
mod tests;
