//! Device inventory of the tpu backend (Google TPU through PJRT). Owner: package 12.
//!
//! Listing devices loads the PJRT plugin and opens the client (the plugin owns the TPU for the whole process afterwards, which is what
//! `--device tpu` needs anyway). A machine without the plugin lists nothing; a plugin that fails to start is reported once on stderr.

use super::pjrt;
use crate::devices::{DeviceInfo, DeviceType};
use std::sync::Once;

/// Memory of one core by device kind, for plugins that do not report it (`PJRT_Device_MemoryStats`): HBM per core in bytes.
pub fn default_memory_bytes(kind: &str) -> u64 {
    const GIB: u64 = 1 << 30;
    let kind = kind.to_ascii_lowercase();
    let has = |needle: &str| kind.contains(needle);
    if has("v6") || has("trillium") {
        32 * GIB
    } else if has("v5p") || (has("v5") && !has("lite") && !has("v5e")) {
        95 * GIB
    } else if has("v5") {
        16 * GIB
    } else if has("v4") {
        32 * GIB
    } else if has("v3") {
        16 * GIB
    } else if has("v2") {
        8 * GIB
    } else {
        0
    }
}

/// The TPU cores found on this machine (empty when there is no plugin or it does not start).
pub fn list(first_index: usize) -> Vec<DeviceInfo> {
    static REPORT: Once = Once::new();
    let runtime = match pjrt::runtime() {
        Ok(runtime) => runtime,
        Err(message) => {
            if !pjrt::is_missing_plugin(&message) {
                REPORT.call_once(|| eprintln!("[fllm-rs] TPU backend: {message}"));
            }
            return Vec::new();
        }
    };
    let client = &runtime.client;
    client
        .devices()
        .iter()
        .enumerate()
        .map(|(i, device)| {
            let stats = client.memory_stats(device);
            let total = stats
                .and_then(|s| s.bytes_limit)
                .filter(|&l| l > 0)
                .map_or_else(|| default_memory_bytes(&device.kind), |l| l as u64);
            let used = stats.map_or(0, |s| s.bytes_in_use.max(0) as u64);
            DeviceInfo {
                index: first_index + i,
                name: format!("{} (core {})", device.kind, device.id),
                description: format!(
                    "tpu pjrt {} {}, PJRT API {}.{}",
                    client.platform_name(),
                    client.platform_version().lines().next().unwrap_or(""),
                    runtime.api.version().0,
                    runtime.api.version().1
                ),
                dev_type: DeviceType::Accelerator,
                free_memory_bytes: total.saturating_sub(used),
                total_memory_bytes: total,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_by_kind() {
        const GIB: u64 = 1 << 30;
        assert_eq!(default_memory_bytes("TPU v2"), 8 * GIB);
        assert_eq!(default_memory_bytes("TPU v3"), 16 * GIB);
        assert_eq!(default_memory_bytes("TPU v4"), 32 * GIB);
        assert_eq!(default_memory_bytes("TPU v5 lite"), 16 * GIB);
        assert_eq!(default_memory_bytes("TPU v5e"), 16 * GIB);
        assert_eq!(default_memory_bytes("TPU v5p"), 95 * GIB);
        assert_eq!(default_memory_bytes("TPU v6 lite"), 32 * GIB);
        assert_eq!(default_memory_bytes("something else"), 0);
    }

    #[test]
    fn no_plugin_lists_nothing() {
        // this machine has no TPU plugin; with one installed the list is the plugin's, so only the "no panic" part is checked
        let _ = list(1);
    }
}
