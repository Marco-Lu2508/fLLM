//! Device inventory of the npu backend. Owner: package 10.
//!
//! The NPUs of every provider compiled into this build, numbered from `first_index` in the global inventory in the order the
//! providers are tried by `backend::create`: Intel NPUs (OpenVINO), then AMD XDNA, Qualcomm Hexagon, Apple Neural Engine (the last
//! three are package 11's `list` functions, each behind its own cargo feature).

use crate::devices::DeviceInfo;

/// The devices found on this machine.
pub fn list(first_index: usize) -> Vec<DeviceInfo> {
    #[allow(unused_mut)]
    let mut found: Vec<DeviceInfo> = Vec::new();
    let _ = first_index;
    #[cfg(feature = "npu-openvino")]
    found.extend(super::openvino::devices::list(first_index + found.len()));
    #[cfg(feature = "npu-xdna")]
    found.extend(super::xdna::list(first_index + found.len()));
    #[cfg(feature = "npu-qnn")]
    found.extend(super::hexagon::list(first_index + found.len()));
    #[cfg(feature = "npu-ane")]
    found.extend(super::ane::list(first_index + found.len()));
    found
}
