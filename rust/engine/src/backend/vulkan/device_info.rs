//! Device inventory of the vulkan backend. Owner: package 1 (Vulkan core).
//!
//! `list` is what `fllm-rs devices` prints: every usable Vulkan device without creating a logical device. The probe (instance creation,
//! driver loading) runs once per process and is cached; memory figures are those of the first call. Index `i` of the returned entries
//! (minus `first_index`) is the `device_index` that [`VkContext::new`](super::instance::VkContext::new) expects.
//!
//! Level of verification: L0 (compiles; never run on a driver).

use super::instance::caps::{ensure_usable, probe};
use super::instance::{VkCaps, VkDeviceType, VkError, VkInstance, VkResult};
use super::memory::heap_budget;
use crate::devices::{DeviceInfo, DeviceType};
use std::sync::OnceLock;

/// A device with everything the probe learned about it.
#[derive(Debug, Clone)]
pub struct VkDeviceReport {
    /// Index among the usable devices, which is the `device_index` of `VkContext::new` (`None` when the device is not usable).
    pub device_index: Option<usize>,
    pub caps: VkCaps,
    /// `None`: the layer can use the device; `Some(reason)`: it cannot (for example Vulkan 1.1 only, no timeline semaphores).
    pub blocker: Option<String>,
    /// Bytes of device-local memory that look free (driver budget minus usage when available, else the heap size).
    pub free_memory_bytes: u64,
}

/// Probe every admitted device, usable or not. Uncached; creates and destroys a Vulkan instance.
pub fn probe_devices() -> VkResult<Vec<VkDeviceReport>> {
    let vki = VkInstance::new()?;
    let mut reports = Vec::new();
    let mut usable = 0;
    for candidate in vki.candidates()? {
        let probed = probe(&vki, candidate.handle)?;
        let blocker = ensure_usable(&probed).err().map(|e| match e {
            VkError::Unsupported(reason) => reason,
            other => other.to_string(),
        });
        let total = probed.caps.vram_bytes;
        let free = if probed.caps.memory_budget {
            let heaps = heap_budget(&vki.instance, candidate.handle);
            let free: u64 = heaps
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    probed
                        .memory
                        .memory_heaps
                        .get(*i)
                        .is_some_and(|h| h.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL))
                })
                .map(|(_, (budget, usage))| budget.saturating_sub(*usage))
                .sum();
            if free > 0 {
                free
            } else {
                total
            }
        } else {
            total
        };
        let device_index = blocker.is_none().then(|| {
            usable += 1;
            usable - 1
        });
        reports.push(VkDeviceReport {
            device_index,
            caps: probed.caps,
            blocker,
            free_memory_bytes: free,
        });
    }
    Ok(reports)
}

fn cached() -> &'static Vec<VkDeviceReport> {
    static CACHE: OnceLock<Vec<VkDeviceReport>> = OnceLock::new();
    CACHE.get_or_init(|| match probe_devices() {
        Ok(reports) => reports,
        Err(error) => {
            // no loader / no driver is the normal case on machines without a GPU: stay quiet unless asked
            if std::env::var_os("FLLM_VULKAN_VERBOSE").is_some() {
                eprintln!("[fllm-vulkan] device probe failed: {error}");
            }
            Vec::new()
        }
    })
}

/// The device type shown by `fllm-rs devices` (software renderers, which only appear with `FLLM_ALLOW_SOFTWARE_GPU=1`, count as GPUs).
fn inventory_type(device_type: VkDeviceType) -> DeviceType {
    match device_type {
        VkDeviceType::Integrated => DeviceType::IntegratedGpu,
        VkDeviceType::Discrete | VkDeviceType::Virtual | VkDeviceType::Cpu | VkDeviceType::Other => DeviceType::Gpu,
    }
}

/// Usable Vulkan devices, numbered from `first_index` in the global inventory (empty without a Vulkan loader or driver).
pub fn list(first_index: usize) -> Vec<DeviceInfo> {
    cached()
        .iter()
        .filter_map(|r| r.device_index.map(|i| (i, r)))
        .map(|(offset, report)| DeviceInfo {
            index: first_index + offset,
            name: report.caps.device_name.clone(),
            description: format!("native, {}", report.caps.describe()),
            dev_type: inventory_type(report.caps.device_type),
            free_memory_bytes: report.free_memory_bytes,
            total_memory_bytes: report.caps.vram_bytes,
        })
        .collect()
}

/// The cached report of every device (usable or not), for diagnostics such as `selftest-gpu`.
pub fn reports() -> &'static [VkDeviceReport] {
    cached()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_types() {
        assert_eq!(inventory_type(VkDeviceType::Integrated), DeviceType::IntegratedGpu);
        assert_eq!(inventory_type(VkDeviceType::Discrete), DeviceType::Gpu);
        assert_eq!(inventory_type(VkDeviceType::Virtual), DeviceType::Gpu);
        assert_eq!(inventory_type(VkDeviceType::Cpu), DeviceType::Gpu);
    }
}
