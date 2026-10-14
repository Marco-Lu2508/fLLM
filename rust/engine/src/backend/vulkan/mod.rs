//! Native Vulkan backend (ash).
//!
//! Scaffold: the module layout is fixed so that the packages can fill in their files without touching this one. Until a
//! package implements `backend::create`, `--device vulkan` reports that the backend is not implemented.
//! Level of verification: L0 (compiles; never run on hardware).

pub mod backend;
pub mod commands;
pub mod device_info;
pub mod instance;
pub mod memory;
pub mod ops;
pub mod pipeline;
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

/// Create the backend for device `index` of this kind (`--device vulkan:N`).
pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
    backend::create(index)
}