//! Weight layouts of the metal backend and the dispatch planner of `quantized_linear` (package p06a).
//!
//! Everything in here is pure CPU code (no Metal objects), so it can be unit tested on any host.
//!
//! * [`format`]: which [`QuantType`]s the Metal kernels know, their block geometry and kernel tags.
//! * [`layout`]: [`pack_weight`] turns the bytes of a GGUF tensor into the device image the kernels read. Most formats are used
//!   **as they are in the file** (the kernels read the GGUF block layout directly, so a memory-mapped model needs no copy and no
//!   second set of weights); only MXFP4 and the Q4Z (AWQ / GPTQ) formats are re-sorted into two planes because their blocks
//!   (17, 19, 35, 67 bytes) cannot be read with aligned loads.
//! * [`plan`]: [`plan_quantized_linear`] chooses the kernel (matrix-vector for batch 1..8, `simdgroup_matrix` matrix-matrix above),
//!   its function constants, the grid and the parameter block.
//!
//! The kernels are in `rust/engine/metal/` (`common.metal`, `common_q_*.metal`, `q_*.metal`); the contract between both sides is
//! `rust/engine/metal/INTERFACE_QUANT.md`. Verification level: L0 (unit tests on the CPU, kernels never run on a GPU).

mod format;
mod layout;
mod plan;
#[cfg(target_vendor = "apple")]
pub mod gpu_check;
pub mod sample;
#[cfg(test)]
mod tests;

pub use format::{format_info, supported_formats, Family, FormatInfo};
pub use layout::{
    device_layout, expected_raw_len, pack_weight, DeviceLayout, PackedWeight, Planes, WeightError,
};
pub use plan::{
    all_kernel_names, plan_quantized_linear, BufferSlots, LinearOp, LinearPlan, MetalCapsLike,
    PermuteParams, PlanConstant, PlanError, PrepassPlan, QParams, FC_NB, FLAG_BIAS, FLAG_RESIDUAL,
};
