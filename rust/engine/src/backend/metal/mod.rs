//! Native Metal backend (objc2-metal, Apple platforms only).
//!
//! Layout: `device_info` (devices, configuration), `buffers` (unified-memory buffers, no-copy GGUF windows), `pipelines` (MSL
//! libraries and function constants), `commands` (command recorder, hazard tracking), `ops` / `weights` (dispatch plans of the
//! kernels in `rust/engine/metal/*.metal`), `stream` / `backend` (the `Stream` and `Backend` of the engine).
//! Level of verification: the kernels pass their GPU self-checks on the Apple-silicon CI runner; the whole-model tests in
//! `whole_model_tests.rs` run there too (see docs/changes/metal.md for what was and was not observed).

pub mod backend;
pub mod buffers;
pub mod commands;
pub mod device_info;
pub mod ops;
pub mod pipelines;
pub mod shaders;
pub mod stream;
pub mod weights;

use super::Backend;
use crate::devices::DeviceInfo;
use std::sync::Arc;

/// Devices this backend can drive, numbered from `first_index` in the global inventory (`fllm-rs devices`).
pub fn list_devices(first_index: usize) -> Vec<DeviceInfo> {
    device_info::list(first_index)
}

/// Create the backend for device `index` of this kind (`--device metal:N`).
pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
    backend::create(index)
}

#[cfg(test)]
mod device_arch_tests;

#[cfg(test)]
mod whole_model_tests;

