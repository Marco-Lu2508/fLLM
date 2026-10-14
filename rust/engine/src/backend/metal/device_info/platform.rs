//! Pure helpers of the device inventory: Apple chip recognition from the Metal device name, GPU vendor
//! guess and the Metal Shading Language version that an OS release supports.
//!
//! Nothing here talks to the system; every function is a pattern match on strings and numbers, so it is
//! checkable on any host. The tables are informational (heuristics for kernel selection and log lines);
//! capability decisions use the `supportsFamily` queries of the device.

/// Product line of an Apple SoC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChipLine {
    /// Mac chips (`Apple M1` … `Apple M5`).
    M,
    /// iPhone / iPad chips (`Apple A14` …), seen on iOS and in simulators.
    A,
}

/// Size class of a chip within its generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChipTier {
    Base,
    Pro,
    Max,
    Ultra,
}

/// An Apple SoC recognised by name, e.g. `Apple M2 Pro` = `{ M, 2, Pro }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AppleChip {
    pub line: ChipLine,
    pub generation: u8,
    pub tier: ChipTier,
}

impl AppleChip {
    /// Marketing label without the vendor prefix (`M2 Pro`, `A17 Pro`).
    pub fn label(&self) -> String {
        let line = match self.line {
            ChipLine::M => 'M',
            ChipLine::A => 'A',
        };
        let tier = match self.tier {
            ChipTier::Base => "",
            ChipTier::Pro => " Pro",
            ChipTier::Max => " Max",
            ChipTier::Ultra => " Ultra",
        };
        format!("{line}{}{tier}", self.generation)
    }

    /// Lowest nominal unified-memory bandwidth in GB/s over the configurations Apple sold under this name
    /// (so `M3 Max` = 300, the 30-core GPU variant). `None` where the number is not known to this table.
    /// Informational only (throughput expectations in logs), never used for correctness.
    pub fn nominal_bandwidth_gbs(&self) -> Option<f32> {
        if self.line != ChipLine::M {
            return None;
        }
        let value = match (self.generation, self.tier) {
            (1, ChipTier::Base) => 68.25,
            (1, ChipTier::Pro) => 200.0,
            (1, ChipTier::Max) => 400.0,
            (1, ChipTier::Ultra) => 800.0,
            (2, ChipTier::Base) => 100.0,
            (2, ChipTier::Pro) => 200.0,
            (2, ChipTier::Max) => 400.0,
            (2, ChipTier::Ultra) => 800.0,
            (3, ChipTier::Base) => 100.0,
            (3, ChipTier::Pro) => 150.0,
            (3, ChipTier::Max) => 300.0,
            (3, ChipTier::Ultra) => 819.0,
            (4, ChipTier::Base) => 120.0,
            (4, ChipTier::Pro) => 273.0,
            (4, ChipTier::Max) => 410.0,
            (5, ChipTier::Base) => 153.0,
            _ => return None,
        };
        Some(value)
    }

    /// The `MTLGPUFamilyAppleN` generation this chip belongs to (M1 = Apple7, M2 = Apple8, M3 and M4 = Apple9,
    /// M5 = Apple10). Only a hint for logs: the backend asks the device with `supportsFamily`.
    pub fn expected_apple_family(&self) -> Option<u8> {
        match (self.line, self.generation) {
            (ChipLine::M, 1) => Some(7),
            (ChipLine::M, 2) => Some(8),
            (ChipLine::M, 3 | 4) => Some(9),
            (ChipLine::M, 5) => Some(10),
            (ChipLine::A, 14) => Some(7),
            (ChipLine::A, 15 | 16) => Some(8),
            (ChipLine::A, 17 | 18) => Some(9),
            (ChipLine::A, 19) => Some(10),
            _ => None,
        }
    }
}

/// Recognise an Apple SoC from the Metal device name (`MTLDevice.name`): `Apple M1`, `Apple M2 Pro`,
/// `Apple M3 Max`, `Apple M3 Ultra`, `Apple M4`, `Apple A17 Pro` … Case-insensitive; a trailing
/// `GPU` word and other decoration are ignored. Names of other vendors give `None`.
pub fn parse_apple_chip(name: &str) -> Option<AppleChip> {
    let lower = name.to_ascii_lowercase();
    let mut words = lower.split_whitespace();
    // the name has to start the chip description with "apple"
    loop {
        match words.next() {
            Some("apple") => break,
            Some(_) => continue,
            None => return None,
        }
    }
    let chip = words.next()?;
    let mut chars = chip.chars();
    let line = match chars.next()? {
        'm' => ChipLine::M,
        'a' => ChipLine::A,
        _ => return None,
    };
    let rest = chars.as_str();
    let digits_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits_end == 0 {
        return None;
    }
    let generation: u8 = rest[..digits_end].parse().ok()?;
    let suffix = &rest[digits_end..];
    // A12X / A12Z style names count as the larger part of their generation
    let mut tier = match suffix {
        "" => ChipTier::Base,
        "x" | "z" => ChipTier::Pro,
        _ => return None,
    };
    for word in words {
        match word {
            "pro" => tier = ChipTier::Pro,
            "max" => tier = ChipTier::Max,
            "ultra" => tier = ChipTier::Ultra,
            _ => {}
        }
    }
    Some(AppleChip {
        line,
        generation,
        tier,
    })
}

/// GPU vendor guess from the device name (Intel Macs have Intel and AMD GPUs next to Apple's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Apple,
    Amd,
    Intel,
    Nvidia,
    Other,
}

impl GpuVendor {
    pub fn as_str(self) -> &'static str {
        match self {
            GpuVendor::Apple => "Apple",
            GpuVendor::Amd => "AMD",
            GpuVendor::Intel => "Intel",
            GpuVendor::Nvidia => "NVIDIA",
            GpuVendor::Other => "unknown",
        }
    }
}

/// Vendor from the Metal device name.
pub fn guess_vendor(name: &str) -> GpuVendor {
    let lower = name.to_ascii_lowercase();
    if lower.contains("apple") {
        GpuVendor::Apple
    } else if lower.contains("amd") || lower.contains("radeon") || lower.contains("ati ") {
        GpuVendor::Amd
    } else if lower.contains("intel") {
        GpuVendor::Intel
    } else if lower.contains("nvidia") || lower.contains("geforce") {
        GpuVendor::Nvidia
    } else {
        GpuVendor::Other
    }
}

/// Operating system family, for the language-version table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OsFamily {
    MacOs,
    Ios,
}

/// The newest Metal Shading Language version `(major, minor)` that the OS release can compile, capped at 3.2
/// (MSL 4.0 is opt-in through `FLLM_METAL_MSL=4.0`). macOS: 11 = 2.3, 12 = 2.4, 13 = 3.0, 14 = 3.1, 15 and
/// later = 3.2; iOS: 14 = 2.3, 15 = 2.4, 16 = 3.0, 17 = 3.1, 18 and later = 3.2. Older systems give 2.3.
pub fn msl_version_for_os(os: OsFamily, major: i64) -> (u16, u16) {
    let level = match os {
        OsFamily::MacOs => major - 11,
        OsFamily::Ios => major - 14,
    };
    match level {
        i64::MIN..=0 => (2, 3),
        1 => (2, 4),
        2 => (3, 0),
        3 => (3, 1),
        _ => (3, 2),
    }
}

/// Parse a language version like `3.1` or `4` into `(major, minor)`.
pub fn parse_msl_version(text: &str) -> Option<(u16, u16)> {
    let text = text.trim();
    let (major, minor) = match text.split_once('.') {
        Some((major, minor)) => (major, minor),
        None => (text, "0"),
    };
    let major: u16 = major.parse().ok()?;
    let minor: u16 = minor.parse().ok()?;
    if !(1..=9).contains(&major) {
        return None;
    }
    Some((major, minor))
}

/// Value of `MTLLanguageVersion` for `(major, minor)`: `major << 16 | minor`.
pub fn msl_version_code(version: (u16, u16)) -> usize {
    ((version.0 as usize) << 16) | version.1 as usize
}

/// Map the index of a device as given on the command line (`--device metal:N`, or the global inventory index that
/// `--device auto` hands over) to the position among the Metal devices.
///
/// `first_index_hint` is the global index of the first Metal device as announced by the last inventory call. An index
/// inside `[first, first + count)` is global; below that it is taken as the position itself. (`metal:1` on a machine
/// whose first Metal device has global index 1 therefore means the first Metal device. Macs with several GPUs are
/// the rare case this ambiguity costs.)
pub fn resolve_device_index(
    requested: usize,
    count: usize,
    first_index_hint: Option<usize>,
) -> Option<usize> {
    if let Some(first) = first_index_hint {
        if requested >= first && requested - first < count {
            return Some(requested - first);
        }
    }
    (requested < count).then_some(requested)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_index_resolution() {
        // one Metal device with global index 1 (CPU is 0)
        assert_eq!(resolve_device_index(0, 1, Some(1)), Some(0));
        assert_eq!(resolve_device_index(1, 1, Some(1)), Some(0));
        assert_eq!(resolve_device_index(2, 1, Some(1)), None);
        // without a hint the index is the position
        assert_eq!(resolve_device_index(1, 2, None), Some(1));
        assert_eq!(resolve_device_index(2, 2, None), None);
        // two devices, first at global 1: global 2 is the second device
        assert_eq!(resolve_device_index(2, 2, Some(1)), Some(1));
    }

    #[test]
    fn parses_mac_chips() {
        let chip = parse_apple_chip("Apple M1").unwrap();
        assert_eq!(
            (chip.line, chip.generation, chip.tier),
            (ChipLine::M, 1, ChipTier::Base)
        );
        assert_eq!(
            parse_apple_chip("Apple M2 Pro").unwrap().tier,
            ChipTier::Pro
        );
        assert_eq!(
            parse_apple_chip("Apple M3 Max").unwrap().tier,
            ChipTier::Max
        );
        assert_eq!(
            parse_apple_chip("Apple M3 Ultra").unwrap().tier,
            ChipTier::Ultra
        );
        assert_eq!(parse_apple_chip("apple m4 pro").unwrap().generation, 4);
        assert_eq!(parse_apple_chip("Apple M5").unwrap().generation, 5);
        assert_eq!(
            parse_apple_chip("Apple M2 Max GPU").unwrap().label(),
            "M2 Max"
        );
    }

    #[test]
    fn parses_phone_chips() {
        let chip = parse_apple_chip("Apple A17 Pro").unwrap();
        assert_eq!(
            (chip.line, chip.generation, chip.tier),
            (ChipLine::A, 17, ChipTier::Pro)
        );
        assert_eq!(chip.expected_apple_family(), Some(9));
        assert_eq!(
            parse_apple_chip("Apple A12X Bionic").unwrap().tier,
            ChipTier::Pro
        );
        assert_eq!(chip.nominal_bandwidth_gbs(), None);
    }

    #[test]
    fn rejects_foreign_names() {
        assert_eq!(parse_apple_chip("AMD Radeon Pro 5500M"), None);
        assert_eq!(parse_apple_chip("Intel(R) UHD Graphics 630"), None);
        assert_eq!(parse_apple_chip("Apple Paravirtual device"), None);
        assert_eq!(parse_apple_chip("Apple M"), None);
        assert_eq!(parse_apple_chip(""), None);
    }

    #[test]
    fn bandwidth_and_family_tables() {
        assert_eq!(
            parse_apple_chip("Apple M1 Max")
                .unwrap()
                .nominal_bandwidth_gbs(),
            Some(400.0)
        );
        assert_eq!(
            parse_apple_chip("Apple M4 Pro")
                .unwrap()
                .nominal_bandwidth_gbs(),
            Some(273.0)
        );
        assert_eq!(
            parse_apple_chip("Apple M5 Max")
                .unwrap()
                .nominal_bandwidth_gbs(),
            None
        );
        assert_eq!(
            parse_apple_chip("Apple M1")
                .unwrap()
                .expected_apple_family(),
            Some(7)
        );
        assert_eq!(
            parse_apple_chip("Apple M4")
                .unwrap()
                .expected_apple_family(),
            Some(9)
        );
        assert_eq!(
            parse_apple_chip("Apple M5")
                .unwrap()
                .expected_apple_family(),
            Some(10)
        );
    }

    #[test]
    fn vendor_guess() {
        assert_eq!(guess_vendor("Apple M2"), GpuVendor::Apple);
        assert_eq!(guess_vendor("AMD Radeon Pro 5500M"), GpuVendor::Amd);
        assert_eq!(
            guess_vendor("Intel(R) Iris(TM) Plus Graphics"),
            GpuVendor::Intel
        );
        assert_eq!(guess_vendor("Mystery GPU"), GpuVendor::Other);
    }

    #[test]
    fn language_versions() {
        assert_eq!(msl_version_for_os(OsFamily::MacOs, 10), (2, 3));
        assert_eq!(msl_version_for_os(OsFamily::MacOs, 12), (2, 4));
        assert_eq!(msl_version_for_os(OsFamily::MacOs, 13), (3, 0));
        assert_eq!(msl_version_for_os(OsFamily::MacOs, 14), (3, 1));
        assert_eq!(msl_version_for_os(OsFamily::MacOs, 15), (3, 2));
        assert_eq!(msl_version_for_os(OsFamily::MacOs, 26), (3, 2));
        assert_eq!(msl_version_for_os(OsFamily::Ios, 17), (3, 1));
        assert_eq!(parse_msl_version("3.1"), Some((3, 1)));
        assert_eq!(parse_msl_version("4"), Some((4, 0)));
        assert_eq!(parse_msl_version("x"), None);
        assert_eq!(msl_version_code((3, 1)), (3 << 16) + 1);
    }
}
