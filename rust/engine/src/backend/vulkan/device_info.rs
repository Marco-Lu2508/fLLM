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
    /// Bytes of memory the device can use: the device-local heaps, and on an integrated GPU the host-visible heaps as well.
    pub total_memory_bytes: u64,
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
        // An integrated GPU draws its memory from the system RAM: next to the device-local heap (on an AMD APU the part the driver calls its VRAM
        // plus a share of the GTT) there is a second, host-visible heap that the GPU reads at the same speed. Weights and caches fit into the sum,
        // and the allocator already places buffers there when the device-local heap is full. The second heap of a discrete card is system memory
        // behind the PCIe bus, which does not count.
        let (shared_total, shared_free) = if probed.caps.device_type == VkDeviceType::Integrated {
            let heaps = &probed.memory.memory_heaps[..probed.memory.memory_heap_count as usize];
            let budgets = if probed.caps.memory_budget { heap_budget(&vki.instance, candidate.handle) } else { Vec::new() };
            let (mut sum_total, mut sum_free) = (0u64, 0u64);
            for (i, h) in heaps.iter().enumerate() {
                if h.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL) {
                    continue;
                }
                sum_total += h.size;
                sum_free += budgets.get(i).map_or(h.size, |(budget, usage)| budget.saturating_sub(*usage));
            }
            (sum_total, sum_free)
        } else {
            (0, 0)
        };
        let (total, free) = (total + shared_total, free + shared_free);
        let device_index = blocker.is_none().then(|| {
            usable += 1;
            usable - 1
        });
        reports.push(VkDeviceReport {
            device_index,
            caps: probed.caps,
            blocker,
            free_memory_bytes: free,
            total_memory_bytes: total,
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
            total_memory_bytes: report.total_memory_bytes,
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
