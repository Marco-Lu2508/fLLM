//! Detection of the AMD XDNA NPU through its PCI identity (package 11).
//!
//! AMD documents the identities in "Ryzen AI Software, Application Development": vendor 0x1022 with device 0x1502 is Phoenix/Hawk Point
//! (XDNA 1, "PHX/HPT"), device 0x17F0 is Strix/Krackan (XDNA 2, "STX/KRK"; revision 0x00, 0x10 and 0x11 are STX, 0x20 is KRK). Linux
//! exposes the identities under `/sys/bus/pci/devices/*/{vendor,device,revision}`; on other systems the scan reports nothing and the
//! caller falls back to "the library loads and lists the VitisAI provider". The classification is pure and unit-tested.

/// NPU generation of the AMD Ryzen AI parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Generation {
    /// Phoenix / Hawk Point (AIE-ML, "XDNA", 10/16 TOPS).
    Xdna1,
    /// Strix Point / Strix Halo / Krackan Point (AIE-ML v2, "XDNA 2", 50+ TOPS).
    Xdna2,
}

/// An NPU found on the PCI bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PciNpu {
    pub generation: Generation,
    /// Marketing name of the APU family ("Strix Point", ...).
    pub apu: &'static str,
    pub vendor: u16,
    pub device: u16,
    pub revision: u8,
    /// PCI address ("0000:c5:00.1") or empty if not known.
    pub address: String,
}

pub const AMD_VENDOR: u16 = 0x1022;
pub const DEVICE_PHOENIX: u16 = 0x1502;
pub const DEVICE_STRIX: u16 = 0x17f0;

/// `Some` if the PCI identity is a known XDNA NPU.
pub fn classify(vendor: u16, device: u16, revision: u8) -> Option<(Generation, &'static str)> {
    if vendor != AMD_VENDOR {
        return None;
    }
    match (device, revision) {
        (DEVICE_PHOENIX, _) => Some((Generation::Xdna1, "Phoenix / Hawk Point")),
        (DEVICE_STRIX, 0x20) => Some((Generation::Xdna2, "Krackan Point")),
        (DEVICE_STRIX, 0x00 | 0x10 | 0x11) => Some((Generation::Xdna2, "Strix Point / Strix Halo")),
        (DEVICE_STRIX, _) => Some((Generation::Xdna2, "Strix-class (unlisted revision)")),
        _ => None,
    }
}

/// Parses the sysfs form "0x1022\n" (also plain hex without prefix).
pub fn parse_sysfs_hex(text: &str) -> Option<u32> {
    let t = text.trim();
    let t = t
        .strip_prefix("0x")
        .or_else(|| t.strip_prefix("0X"))
        .unwrap_or(t);
    u32::from_str_radix(t, 16).ok()
}

/// All XDNA NPUs on the PCI bus (Linux only; empty elsewhere or when sysfs is not readable).
pub fn scan() -> Vec<PciNpu> {
    #[cfg(target_os = "linux")]
    {
        let mut found = Vec::new();
        let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") else {
            return found;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let read = |name: &str| {
                std::fs::read_to_string(path.join(name))
                    .ok()
                    .and_then(|s| parse_sysfs_hex(&s))
            };
            let (Some(vendor), Some(device)) = (read("vendor"), read("device")) else {
                continue;
            };
            let revision = read("revision").unwrap_or(0);
            if let Some((generation, apu)) = classify(vendor as u16, device as u16, revision as u8)
            {
                found.push(PciNpu {
                    generation,
                    apu,
                    vendor: vendor as u16,
                    device: device as u16,
                    revision: revision as u8,
                    address: entry.file_name().to_string_lossy().into_owned(),
                });
            }
        }
        found.sort_by(|a, b| a.address.cmp(&b.address));
        found
    }
    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_the_documented_identities() {
        assert_eq!(classify(0x1022, 0x1502, 0x00).unwrap().0, Generation::Xdna1);
        for rev in [0x00, 0x10, 0x11] {
            assert_eq!(
                classify(0x1022, 0x17f0, rev),
                Some((Generation::Xdna2, "Strix Point / Strix Halo"))
            );
        }
        assert_eq!(
            classify(0x1022, 0x17f0, 0x20),
            Some((Generation::Xdna2, "Krackan Point"))
        );
        assert_eq!(classify(0x1022, 0x17f0, 0x77).unwrap().0, Generation::Xdna2);
    }

    #[test]
    fn rejects_other_vendors_and_devices() {
        assert!(classify(0x8086, 0x17f0, 0).is_none());
        assert!(classify(0x1022, 0x1234, 0).is_none());
    }

    #[test]
    fn parses_sysfs_values() {
        assert_eq!(parse_sysfs_hex("0x1022\n"), Some(0x1022));
        assert_eq!(parse_sysfs_hex("17f0"), Some(0x17f0));
        assert_eq!(parse_sysfs_hex("zz"), None);
        assert_eq!(parse_sysfs_hex(""), None);
    }
}
