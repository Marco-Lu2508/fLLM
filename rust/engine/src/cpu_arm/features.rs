//! Runtime detection of the AArch64 extensions the ARM kernels use.
//!
//! Everything reports `false` off AArch64, so callers and tests can use it on any host.
//! Detection is the standard library's (`is_aarch64_feature_detected!`: `getauxval` on Linux and
//! Android, `sysctl` on macOS, `IsProcessorFeaturePresent` on Windows), cached in a `OnceLock`.
//! The SVE vector length is read with `rdvl`, only where SVE can be present (Linux, Android,
//! FreeBSD).
use std::sync::OnceLock;

/// What the CPU reports. A kernel tier is chosen from this and from the `FLLM_CPU_ISA` cap in
/// `cpu_kernels`, never from this alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmFeatures {
    /// Advanced SIMD (baseline of AArch64).
    pub neon: bool,
    /// FEAT_DotProd: `sdot` / `udot`.
    pub dotprod: bool,
    /// FEAT_I8MM: `smmla` / `ummla` / `usmmla` / `usdot`.
    pub i8mm: bool,
    /// FEAT_BF16: `bfdot`, `bfmmla`, `bfcvtn`.
    pub bf16: bool,
    /// FEAT_FP16: half-precision arithmetic.
    pub fp16: bool,
    /// FEAT_FHM: `fmlal` / `fmlal2` (half to single widening multiply-add).
    pub fhm: bool,
    /// FEAT_SVE.
    pub sve: bool,
    /// FEAT_SVE2.
    pub sve2: bool,
    /// SVE vector length in bytes (16 to 256); 0 without SVE or when it cannot be read.
    pub sve_bytes: usize,
}

impl ArmFeatures {
    /// Names of the detected extensions in a fixed order; SVE carries its vector length.
    pub fn names(&self) -> Vec<String> {
        let mut found = Vec::new();
        for (present, name) in [
            (self.neon, "neon"),
            (self.dotprod, "dotprod"),
            (self.i8mm, "i8mm"),
            (self.bf16, "bf16"),
            (self.fp16, "fp16"),
            (self.fhm, "fhm"),
        ] {
            if present {
                found.push(name.to_owned());
            }
        }
        if self.sve {
            found.push(if self.sve_bytes > 0 {
                format!("sve({}-bit)", self.sve_bytes * 8)
            } else {
                "sve".to_owned()
            });
        }
        if self.sve2 {
            found.push("sve2".to_owned());
        }
        found
    }

    /// Whether the SVE kernels (one 32-byte Q8_0 block per vector) can use this CPU: SVE with a
    /// vector of at least 256 bits.
    pub fn sve_wide(&self) -> bool {
        self.sve && self.sve_bytes >= 32
    }
}

/// The features of this CPU (detected once).
pub fn detect() -> ArmFeatures {
    static CACHE: OnceLock<ArmFeatures> = OnceLock::new();
    *CACHE.get_or_init(detect_uncached)
}

#[cfg(target_arch = "aarch64")]
fn detect_uncached() -> ArmFeatures {
    let sve = std::arch::is_aarch64_feature_detected!("sve");
    ArmFeatures {
        neon: std::arch::is_aarch64_feature_detected!("neon"),
        dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
        i8mm: std::arch::is_aarch64_feature_detected!("i8mm"),
        bf16: std::arch::is_aarch64_feature_detected!("bf16"),
        fp16: std::arch::is_aarch64_feature_detected!("fp16"),
        fhm: std::arch::is_aarch64_feature_detected!("fhm"),
        sve,
        sve2: std::arch::is_aarch64_feature_detected!("sve2"),
        sve_bytes: if sve { sve_vector_bytes() } else { 0 },
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn detect_uncached() -> ArmFeatures {
    ArmFeatures::default()
}

/// Bytes in one SVE vector (`rdvl #1`); 0 where SVE cannot exist.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android", target_os = "freebsd")
))]
fn sve_vector_bytes() -> usize {
    let bytes: u64;
    // SAFETY: only reached after `is_aarch64_feature_detected!("sve")`; `rdvl` reads the current
    // vector length into a general register and touches neither memory nor flags.
    unsafe {
        std::arch::asm!(
            ".arch_extension sve",
            "rdvl {bytes}, #1",
            bytes = out(reg) bytes,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    bytes as usize
}

#[cfg(all(
    target_arch = "aarch64",
    not(any(target_os = "linux", target_os = "android", target_os = "freebsd"))
))]
fn sve_vector_bytes() -> usize {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_flags() {
        let all = ArmFeatures {
            neon: true,
            dotprod: true,
            i8mm: true,
            bf16: true,
            fp16: true,
            fhm: true,
            sve: true,
            sve2: true,
            sve_bytes: 32,
        };
        assert_eq!(
            all.names(),
            [
                "neon",
                "dotprod",
                "i8mm",
                "bf16",
                "fp16",
                "fhm",
                "sve(256-bit)",
                "sve2"
            ]
        );
        assert!(all.sve_wide());
        let narrow = ArmFeatures {
            sve_bytes: 16,
            ..all
        };
        assert!(!narrow.sve_wide());
        assert!(ArmFeatures::default().names().is_empty());
    }

    #[test]
    fn detection_is_stable_and_consistent() {
        let a = detect();
        assert_eq!(a, detect());
        // Every extension below implies the baseline, and a vector length implies SVE.
        if a.i8mm || a.dotprod || a.bf16 {
            assert!(a.neon);
        }
        assert!(a.sve || a.sve_bytes == 0);
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(a, ArmFeatures::default());
        #[cfg(target_arch = "aarch64")]
        if a.sve && a.sve_bytes > 0 {
            assert!(a.sve_bytes.is_power_of_two() || a.sve_bytes.is_multiple_of(16));
        }
    }
}
