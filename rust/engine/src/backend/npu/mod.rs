//! NPU backends (Intel OpenVINO, AMD XDNA, Qualcomm QNN, Apple Neural Engine) and their setup in the app.
//!
//! * [`probe`]: NPUs the operating system lists, without any vendor runtime;
//! * [`setup`] / [`zip`]: download, check and unpack of the pinned OpenVINO runtime (`fllm-rs npu-setup`, `/api/studio/npu/setup`);
//! * [`status`]: the state shown by the app (hardware, runtime, setup, last fallback to the CPU);
//! * the providers ([`openvino`] hybrid backend; [`xdna`], [`hexagon`], [`ane`] probes) behind their cargo features.
//!
//! This module is always compiled; the OpenVINO backend is in the default features and loads the OpenVINO C runtime at run time.
//! Level of verification: L0 (compiles; never run on an NPU).

pub mod backend;
pub mod device_info;
pub mod flm;
#[cfg(feature = "npu-openvino")]
pub mod host;
pub mod openvino;
pub mod probe;
pub mod setup;
pub mod status;
pub mod xdna;
pub mod hexagon;
pub mod ane;
pub mod zip;

use super::Backend;
use crate::devices::DeviceInfo;
use std::sync::Arc;

/// Devices this backend can drive, numbered from `first_index` in the global inventory (`fllm-rs devices`).
pub fn list_devices(first_index: usize) -> Vec<DeviceInfo> {
    device_info::list(first_index)
}

/// Create the backend for device `index` of this kind (`--device npu:N`).
pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
    backend::create(index)
}