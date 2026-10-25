//! NPU devices as OpenVINO enumerates them.
//!
//! `ov_core_get_available_devices` lists the plugins' devices (`CPU`, `GPU.0`, `NPU`). The NPU plugin reports one device named `NPU`
//! ("NPU plugin does not currently support multiple devices", NPU README). The properties queried for the inventory are the ones the
//! plugin documents: `FULL_DEVICE_NAME`, `DEVICE_ARCHITECTURE` (the platform, e.g. `3720` for Meteor Lake / Arrow Lake-U, `4000` for
//! Lunar Lake), `NPU_DRIVER_VERSION`, `NPU_DEVICE_TOTAL_MEM_SIZE` and `NPU_DEVICE_ALLOC_MEM_SIZE`.
//!
//! `FLLM_NPU_DEVICE` replaces the device name (default `NPU`); it exists to let the plumbing be exercised against another OpenVINO
//! device (for example `CPU`) when no NPU is present.
//!
//! Verification level: L0.

use super::session::Runtime;
use crate::devices::{DeviceInfo, DeviceType};
use std::sync::OnceLock;

/// One device found through OpenVINO.
#[derive(Clone, Debug)]
pub struct OvDevice {
    /// Name to pass to `compile_model` (`NPU`).
    pub name: String,
    pub full_name: String,
    pub architecture: String,
    pub driver: String,
    pub total_memory: u64,
    pub allocated_memory: u64,
}

/// Device name prefix looked for in the enumeration.
pub fn device_prefix() -> String {
    std::env::var("FLLM_NPU_DEVICE")
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "NPU".to_owned())
}

fn matches_prefix(device: &str, prefix: &str) -> bool {
    device == prefix
        || device
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('.'))
}

/// The matching devices, queried once per process. Empty when OpenVINO or its NPU plugin is missing; the reason is logged once.
pub fn enumerate() -> &'static [OvDevice] {
    static DEVICES: OnceLock<Vec<OvDevice>> = OnceLock::new();
    if let Some(found) = DEVICES.get() {
        return found;
    }
    match query() {
        Ok(devices) => DEVICES.get_or_init(|| devices),
        Err(error) => {
            // not remembered: the runtime may be installed (or the driver loaded) while the process runs
            crate::backend::log_once(
                "openvino-devices",
                &format!("no OpenVINO NPU listed: {error}"),
            );
            &[]
        }
    }
}

fn query() -> Result<Vec<OvDevice>, String> {
    let runtime = Runtime::shared()?;
    let prefix = device_prefix();
    let names = runtime.available_devices()?;
    let property = |device: &str, key: &str| runtime.property(device, key).unwrap_or_default();
    let number = |device: &str, key: &str| property(device, key).trim().parse::<u64>().unwrap_or(0);
    Ok(names
        .iter()
        .filter(|name| matches_prefix(name, &prefix))
        .map(|name| OvDevice {
            name: name.clone(),
            full_name: property(name, "FULL_DEVICE_NAME"),
            architecture: property(name, "DEVICE_ARCHITECTURE"),
            driver: property(name, "NPU_DRIVER_VERSION"),
            total_memory: number(name, "NPU_DEVICE_TOTAL_MEM_SIZE"),
            allocated_memory: number(name, "NPU_DEVICE_ALLOC_MEM_SIZE"),
        })
        .collect())
}

/// Inventory entries numbered from `first_index`.
pub fn list(first_index: usize) -> Vec<DeviceInfo> {
    enumerate()
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let mut description = format!("OpenVINO {}", d.name);
            if !d.architecture.is_empty() {
                description.push_str(&format!(", platform {}", d.architecture));
            }
            if !d.driver.is_empty() {
                description.push_str(&format!(", driver {}", d.driver));
            }
            description.push_str(", NPU matmul offload (prefill)");
            DeviceInfo {
                index: first_index + i,
                name: if d.full_name.is_empty() {
                    d.name.clone()
                } else {
                    d.full_name.clone()
                },
                description,
                dev_type: DeviceType::Accelerator,
                free_memory_bytes: d.total_memory.saturating_sub(d.allocated_memory),
                total_memory_bytes: d.total_memory,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_matching_accepts_numbered_devices_only() {
        assert!(matches_prefix("NPU", "NPU"));
        assert!(matches_prefix("NPU.1", "NPU"));
        assert!(!matches_prefix("NPUX", "NPU"));
        assert!(!matches_prefix("GPU.0", "NPU"));
    }
}
