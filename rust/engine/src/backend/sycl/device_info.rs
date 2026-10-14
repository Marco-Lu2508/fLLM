//! Device inventory of the sycl backend. Owner: package 9.
//!
//! The devices are the GPUs the SYCL runtime reports (Level Zero first; the OpenCL duplicates of the same GPUs are hidden unless
//! `FLLM_SYCL_ALL_BACKENDS=1`). `ONEAPI_DEVICE_SELECTOR` narrows the list as for any SYCL program. Without the kernel library in the
//! binary (no `icpx` at build time) the list is empty.

use super::ffi::{self, DeviceProps};
use crate::devices::{DeviceInfo, DeviceType};

/// One line for `fllm-rs devices`: backend, architecture, execution units, matrix engine, driver.
pub fn describe(p: &DeviceProps) -> String {
    let arch = p.arch_text();
    let mut text = format!("SYCL, {}", if p.level_zero != 0 { "Level Zero" } else { "OpenCL" });
    if !arch.is_empty() {
        text.push_str(&format!(", {arch}"));
    }
    let units = if p.eu_count > 0 { p.eu_count } else { p.compute_units };
    if units > 0 {
        text.push_str(&format!(", {units} EUs"));
    }
    match (p.xmx_n16 != 0, p.xmx_n8 != 0) {
        (true, _) => text.push_str(", XMX 8x16x16"),
        (_, true) => text.push_str(", XMX 8x8x16"),
        _ => text.push_str(", no XMX"),
    }
    let driver = p.driver_text();
    if !driver.is_empty() {
        text.push_str(&format!(", driver {driver}"));
    }
    text
}

/// Properties of SYCL device `dev` (0-based within this backend).
pub fn props(dev: usize) -> Option<DeviceProps> {
    let mut p = DeviceProps::zeroed();
    // SAFETY: `p` is a valid out-pointer for the whole struct.
    let code = unsafe { ffi::fllm_sycl_device_info(dev as std::ffi::c_int, &mut p) };
    (code == 0).then_some(p)
}

/// The devices found on this machine, numbered from `first_index` in the global inventory.
pub fn list(first_index: usize) -> Vec<DeviceInfo> {
    if !ffi::NATIVE {
        return Vec::new();
    }
    // SAFETY: plain query; -1 means no usable device or no runtime.
    let count = unsafe { ffi::fllm_sycl_device_count() };
    (0..count.max(0) as usize)
        .filter_map(|dev| {
            let p = props(dev)?;
            let total = p.total_memory;
            Some(DeviceInfo {
                index: first_index + dev,
                name: p.name_text(),
                description: describe(&p),
                dev_type: if p.is_integrated != 0 {
                    DeviceType::IntegratedGpu
                } else {
                    DeviceType::Gpu
                },
                // integrated GPUs share system memory: the driver's number is all there is
                free_memory_bytes: if p.free_memory > 0 { p.free_memory } else { total },
                total_memory_bytes: total,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props(name: &str) -> DeviceProps {
        let mut p = DeviceProps::zeroed();
        for (slot, b) in p.name.iter_mut().zip(name.bytes()) {
            *slot = b as std::ffi::c_char;
        }
        p
    }

    #[test]
    fn description_names_the_matrix_engine_and_units() {
        let mut p = props("Intel(R) Arc(TM) A770 Graphics");
        p.level_zero = 1;
        p.eu_count = 512;
        p.xmx_n8 = 1;
        let text = describe(&p);
        assert!(
            text.contains("Level Zero") && text.contains("512 EUs") && text.contains("XMX 8x8x16"),
            "{text}"
        );
        assert_eq!(p.name_text(), "Intel(R) Arc(TM) A770 Graphics");
        p.xmx_n8 = 0;
        assert!(describe(&p).contains("no XMX"));
    }
}
