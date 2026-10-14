//! Device inventory and architecture knowledge of the hip backend. Owner: package 8.
//!
//! The kernels of the hip backend are the CUDA sources built with hipcc, behind the same C API. Their runtime
//! (`libamdhip64`) is reached through the `cuda` module, whose `list_devices` already reports every AMD GPU as
//! "CUDA/ROCm" with the name `<marketing name> (<gcnArchName>)`. This module adds
//!
//! * [`gcn_arch`] / [`ArchFamily`]: what the architecture string (`gfx1100`, `gfx90a:sramecc+:xnack-` ...) tells about wave size and
//!   matrix cores (used for logging and for the capability decisions of the backend), and
//! * [`list`]: the AMD inventory entries of this backend, without the ones the CUDA inventory shows already.
//!
//! Level of verification: L0 (compiles; never run on hardware).

use crate::devices::{DeviceInfo, DeviceType};

/// Instruction-set family of an AMD GPU, as far as the kernels care.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchFamily {
    /// gfx900 / gfx906 (Vega): wave64, no matrix cores.
    Vega,
    /// gfx908 (MI100): wave64, MFMA.
    Cdna1,
    /// gfx90a (MI200): wave64, MFMA, packed FP32.
    Cdna2,
    /// gfx940 / gfx941 / gfx942 (MI300): wave64, MFMA, FP8 (FNUZ).
    Cdna3,
    /// gfx950 (MI350): wave64, MFMA, FP8 / FP4.
    Cdna4,
    /// gfx101x (RDNA1): wave32, no matrix cores.
    Rdna1,
    /// gfx103x (RDNA2): wave32, dot products, no matrix cores.
    Rdna2,
    /// gfx110x (RDNA3): wave32, WMMA.
    Rdna3,
    /// gfx115x (RDNA3.5, Strix Point / Halo): wave32, WMMA.
    Rdna35,
    /// gfx120x (RDNA4): wave32, WMMA (gfx12 encoding), FP8.
    Rdna4,
    /// Anything else (including future architectures).
    Unknown,
}

impl ArchFamily {
    /// Family of an architecture name such as `gfx1100`.
    pub fn from_arch(arch: &str) -> Self {
        match arch {
            "gfx900" | "gfx902" | "gfx904" | "gfx906" | "gfx909" | "gfx90c" => Self::Vega,
            "gfx908" => Self::Cdna1,
            "gfx90a" => Self::Cdna2,
            "gfx940" | "gfx941" | "gfx942" => Self::Cdna3,
            "gfx950" => Self::Cdna4,
            a if a.starts_with("gfx101") => Self::Rdna1,
            a if a.starts_with("gfx103") => Self::Rdna2,
            a if a.starts_with("gfx110") => Self::Rdna3,
            a if a.starts_with("gfx115") => Self::Rdna35,
            a if a.starts_with("gfx120") => Self::Rdna4,
            _ => Self::Unknown,
        }
    }

    /// Threads per wavefront the compiler targets (hipcc defaults: wave64 on GCN / CDNA, wave32 on RDNA).
    pub fn wave_size(self) -> u32 {
        match self {
            Self::Vega | Self::Cdna1 | Self::Cdna2 | Self::Cdna3 | Self::Cdna4 | Self::Unknown => {
                64
            }
            Self::Rdna1 | Self::Rdna2 | Self::Rdna3 | Self::Rdna35 | Self::Rdna4 => 32,
        }
    }

    /// Matrix instruction set: `Some("MFMA")` (CDNA) or `Some("WMMA")` (RDNA3 and newer).
    pub fn matrix_cores(self) -> Option<&'static str> {
        match self {
            Self::Cdna1 | Self::Cdna2 | Self::Cdna3 | Self::Cdna4 => Some("MFMA"),
            Self::Rdna3 | Self::Rdna35 | Self::Rdna4 => Some("WMMA"),
            _ => None,
        }
    }

    /// Short human-readable summary, e.g. `wave32, WMMA`.
    pub fn summary(self) -> String {
        format!(
            "wave{}, {}",
            self.wave_size(),
            self.matrix_cores().unwrap_or("no matrix cores")
        )
    }
}

/// The architecture (`gfx1100`) in a device name of the form `<name> (<gcnArchName>)`; the feature suffix some runtimes append
/// (`gfx90a:sramecc+:xnack-`) is dropped. A bare architecture string is accepted too.
pub fn gcn_arch(device_name: &str) -> Option<String> {
    let start = device_name.rfind("gfx")?;
    let tail = &device_name[start..];
    let arch: String = tail
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric())
        .collect();
    // gfx + at least three hex digits (gfx803 ... gfx1201, gfx90a, gfx9-4-generic is not a device name)
    (arch.len() >= 6 && arch[3..].chars().all(|c| c.is_ascii_hexdigit())).then_some(arch)
}

/// Is this inventory entry an AMD GPU of the HIP runtime (as listed by the cuda module)?
fn is_rocm_entry(device: &DeviceInfo) -> bool {
    device.description.starts_with("CUDA/ROCm")
}

/// Two entries describe the same physical device when name and memory size agree (the HIP ordinal is not part of the entry).
fn same_device(a: &DeviceInfo, b: &DeviceInfo) -> bool {
    a.name == b.name && a.total_memory_bytes == b.total_memory_bytes
}

/// Entries of `candidates` that `known` does not show yet; the position in the result decides the global index.
fn without_known(candidates: Vec<DeviceInfo>, known: &[DeviceInfo]) -> Vec<DeviceInfo> {
    candidates
        .into_iter()
        .filter(|c| !known.iter().any(|k| same_device(k, c)))
        .collect()
}

/// Entry for an AMD GPU with the architecture facts added to the description.
fn describe(device: &DeviceInfo, index: usize) -> DeviceInfo {
    let detail = gcn_arch(&device.name)
        .map(|arch| format!("ROCm {arch}, {}", ArchFamily::from_arch(&arch).summary()))
        .unwrap_or_else(|| "ROCm".to_owned());
    DeviceInfo {
        index,
        name: device.name.clone(),
        description: format!(
            "{detail}, {}",
            device.description.trim_start_matches("CUDA/ROCm, ")
        ),
        dev_type: DeviceType::Gpu,
        free_memory_bytes: device.free_memory_bytes,
        total_memory_bytes: device.total_memory_bytes,
    }
}

/// The AMD devices of this backend, numbered from `first_index` in the global inventory.
///
/// A build with the `hip` feature links the HIP build of the kernel library behind the `cuda` module, so
/// `backend::cuda::list_devices` (called by `devices::list_devices` before the native backends) reports every AMD GPU already. To keep
/// `fllm-rs devices` free of doubles, only entries missing from that list are returned here (none, as long as both come from the
/// same runtime). `--device hip:N` and `--device cuda:N` address the same hardware with the same ordinal N.
pub fn list(first_index: usize) -> Vec<DeviceInfo> {
    let known = crate::backend::cuda::list_devices(0);
    let rocm: Vec<DeviceInfo> = known
        .iter()
        .filter(|d| is_rocm_entry(d))
        .enumerate()
        .map(|(n, d)| describe(d, first_index + n))
        .collect();
    // `known` entries are the very devices of `rocm`: this keeps the function right if the two inventories ever diverge
    let fresh = without_known(rocm, &known);
    fresh
        .into_iter()
        .enumerate()
        .map(|(n, mut d)| {
            d.index = first_index + n;
            d
        })
        .collect()
}

/// Architecture of HIP device `ordinal` (`gfx1100`), from the CUDA inventory.
pub fn arch_of_device(ordinal: usize) -> Option<String> {
    crate::backend::cuda::list_devices(0)
        .into_iter()
        .filter(is_rocm_entry)
        .nth(ordinal)
        .and_then(|d| gcn_arch(&d.name))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, total: u64) -> DeviceInfo {
        DeviceInfo {
            index: 1,
            name: name.to_owned(),
            description: "CUDA/ROCm, 96 CUs".to_owned(),
            dev_type: DeviceType::Gpu,
            free_memory_bytes: total / 2,
            total_memory_bytes: total,
        }
    }

    #[test]
    fn arch_is_read_from_the_device_name() {
        assert_eq!(
            gcn_arch("AMD Radeon RX 7900 XTX (gfx1100)").as_deref(),
            Some("gfx1100")
        );
        assert_eq!(
            gcn_arch("AMD Instinct MI210 (gfx90a:sramecc+:xnack-)").as_deref(),
            Some("gfx90a")
        );
        assert_eq!(gcn_arch("gfx942").as_deref(), Some("gfx942"));
        assert_eq!(gcn_arch("NVIDIA GeForce RTX 4090"), None);
        assert_eq!(gcn_arch("gfx"), None);
    }

    #[test]
    fn families_follow_the_isa() {
        assert_eq!(ArchFamily::from_arch("gfx1100"), ArchFamily::Rdna3);
        assert_eq!(ArchFamily::from_arch("gfx1151"), ArchFamily::Rdna35);
        assert_eq!(ArchFamily::from_arch("gfx1201"), ArchFamily::Rdna4);
        assert_eq!(ArchFamily::from_arch("gfx1030"), ArchFamily::Rdna2);
        assert_eq!(ArchFamily::from_arch("gfx90a"), ArchFamily::Cdna2);
        assert_eq!(ArchFamily::from_arch("gfx942"), ArchFamily::Cdna3);
        assert_eq!(ArchFamily::from_arch("gfx906"), ArchFamily::Vega);
        assert_eq!(ArchFamily::from_arch("gfx9999"), ArchFamily::Unknown);
        assert_eq!(ArchFamily::Rdna3.wave_size(), 32);
        assert_eq!(ArchFamily::Cdna3.wave_size(), 64);
        assert_eq!(ArchFamily::Cdna1.matrix_cores(), Some("MFMA"));
        assert_eq!(ArchFamily::Rdna4.matrix_cores(), Some("WMMA"));
        assert_eq!(ArchFamily::Rdna2.matrix_cores(), None);
    }

    #[test]
    fn known_devices_are_not_listed_twice() {
        let known = vec![entry("AMD Radeon RX 7900 XTX (gfx1100)", 24 << 30)];
        let candidates = vec![
            describe(&known[0], 5),
            describe(&entry("AMD Radeon RX 7600 (gfx1102)", 8 << 30), 6),
        ];
        let fresh = without_known(candidates, &known);
        assert_eq!(fresh.len(), 1);
        assert!(fresh[0].name.contains("gfx1102"));
        assert!(describe(&known[0], 5)
            .description
            .starts_with("ROCm gfx1100, wave32, WMMA"));
    }
}
