//! CPU identification for the x86 extension kernels: vendor, family and model from CPUID, the
//! feature bits `is_x86_feature_detected!` cannot report (AMX is still unstable in Rust), the OS
//! support for AMX tile data, and the dispatch policy derived from them.
//!
//! The classification is a pure function of the raw CPUID words (`classify`), so the tables can
//! be checked on any machine; only `features()` / `policy()` read the real CPU.
use std::arch::x86_64::{__cpuid, __cpuid_count};
use std::sync::OnceLock;

/// Vendor string of CPUID leaf 0 (ebx, edx, ecx order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Vendor {
    Intel,
    Amd,
    Other,
}

/// Micro-architecture classes the dispatch policy distinguishes. Only classes that matter for the
/// choice of kernel width or for AMX are separated; everything else is `Other`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Microarch {
    /// Skylake-SP / Skylake-X: first AVX-512, no VNNI, strong frequency licences.
    IntelSkylakeServer,
    /// Cascade Lake and Cooper Lake (AVX-512 VNNI; Cooper Lake adds BF16).
    IntelCascadeLake,
    /// Ice Lake server and Ice Lake-D.
    IntelIceLakeServer,
    /// Ice Lake / Tiger Lake / Rocket Lake / Cannon Lake client: one 512-bit FMA unit.
    IntelClient512,
    /// Sapphire Rapids, Emerald Rapids, Granite Rapids: AVX-512 with two units, AMX.
    IntelAmxServer,
    /// Any other Intel CPU.
    IntelOther,
    /// AMD Zen 4: 512-bit operations are executed as two 256-bit halves.
    Zen4,
    /// AMD Zen 5 and later: full-width 512-bit data paths.
    Zen5Plus,
    /// AMD without AVX-512 (Zen 3 and older) or unknown AMD family.
    AmdOther,
    Unknown,
}

/// Raw CPUID leaf 0 vendor words and leaf 1 EAX, enough for `classify`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawId {
    /// ebx, edx, ecx of leaf 0.
    pub vendor: [u32; 3],
    /// eax of leaf 1 (family, model, stepping).
    pub signature: u32,
}

impl RawId {
    pub(crate) fn vendor(&self) -> Vendor {
        const INTEL: [u32; 3] = [0x756e_6547, 0x4965_6e69, 0x6c65_746e]; // "Genu" "ineI" "ntel"
        const AMD: [u32; 3] = [0x6874_7541, 0x6974_6e65, 0x444d_4163]; // "Auth" "enti" "cAMD"
        if self.vendor == INTEL {
            Vendor::Intel
        } else if self.vendor == AMD {
            Vendor::Amd
        } else {
            Vendor::Other
        }
    }

    /// Display family: base family, plus the extended family when the base is 0xF.
    pub(crate) fn family(&self) -> u32 {
        let base = (self.signature >> 8) & 0xf;
        if base == 0xf {
            base + ((self.signature >> 20) & 0xff)
        } else {
            base
        }
    }

    /// Display model: base model, with the extended model in the high nibble for families 6 and
    /// 0xF and above.
    pub(crate) fn model(&self) -> u32 {
        let base = (self.signature >> 4) & 0xf;
        let extended = (self.signature >> 16) & 0xf;
        if matches!((self.signature >> 8) & 0xf, 6 | 0xf) {
            (extended << 4) | base
        } else {
            base
        }
    }

    pub(crate) fn stepping(&self) -> u32 {
        self.signature & 0xf
    }
}

/// Classify a CPU. `avx512` tells whether the CPU reports AVX-512F: AMD family 0x19 covers Zen 3
/// (no AVX-512) and Zen 4, which the feature bit separates.
pub(crate) fn classify(id: RawId, avx512: bool) -> Microarch {
    match id.vendor() {
        Vendor::Intel => {
            // Diamond Rapids is family 19 (0x13) model 1; everything else is family 6.
            if id.family() == 19 && id.model() == 0x01 {
                return Microarch::IntelAmxServer;
            }
            if id.family() != 6 {
                return Microarch::IntelOther;
            }
            match id.model() {
                // SKYLAKE_X: stepping 4 Skylake-SP, 5..=7 Cascade Lake, 11 Cooper Lake.
                0x55 => match id.stepping() {
                    0..=4 => Microarch::IntelSkylakeServer,
                    _ => Microarch::IntelCascadeLake,
                },
                // ICELAKE_X, ICELAKE_D
                0x6a | 0x6c => Microarch::IntelIceLakeServer,
                // ICELAKE, ICELAKE_L, TIGERLAKE_L, TIGERLAKE, ROCKETLAKE, CANNONLAKE_L
                0x7d | 0x7e | 0x8c | 0x8d | 0xa7 | 0x66 => Microarch::IntelClient512,
                // SAPPHIRERAPIDS_X, EMERALDRAPIDS_X, GRANITERAPIDS_X, GRANITERAPIDS_D
                0x8f | 0xcf | 0xad | 0xae => Microarch::IntelAmxServer,
                _ => Microarch::IntelOther,
            }
        }
        Vendor::Amd => match id.family() {
            0x19 if avx512 => Microarch::Zen4,
            f if f >= 0x1a && avx512 => Microarch::Zen5Plus,
            _ => Microarch::AmdOther,
        },
        Vendor::Other => Microarch::Unknown,
    }
}

/// Everything the x86 extension kernels need to know about the CPU.
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub(crate) struct Features {
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avx_vnni: bool,
    pub avx_vnni_int8: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512vnni: bool,
    pub avx512bf16: bool,
    /// Detected for the report only: no kernel needs `vpermb` (the 16-entry codebooks fit
    /// `vpshufb`).
    pub avx512vbmi: bool,
    /// CPU supports the AMX instructions (CPUID 7.0 EDX bits 24, 25, 22).
    pub amx_tile: bool,
    pub amx_int8: bool,
    pub amx_bf16: bool,
    /// The OS enabled the tile configuration and tile data state in XCR0.
    pub os_tile_state: bool,
}

/// CPUID leaf 7 subleaf 0, EDX bits of the AMX features.
const AMX_BF16_BIT: u32 = 1 << 22;
const AMX_TILE_BIT: u32 = 1 << 24;
const AMX_INT8_BIT: u32 = 1 << 25;

/// XCR0 bits 17 (XTILECFG) and 18 (XTILEDATA).
const XCR0_TILE_BITS: u64 = (1 << 17) | (1 << 18);

/// Decode the AMX feature bits of CPUID 7.0 EDX: (tile, int8, bf16).
pub(crate) fn amx_bits(edx7: u32) -> (bool, bool, bool) {
    (
        edx7 & AMX_TILE_BIT != 0,
        edx7 & AMX_INT8_BIT != 0,
        edx7 & AMX_BF16_BIT != 0,
    )
}

#[target_feature(enable = "xsave")]
unsafe fn read_xcr0() -> u64 {
    std::arch::x86_64::_xgetbv(0)
}

// __cpuid is a safe function on recent compilers.
#[allow(unused_unsafe)]
fn read_features() -> Features {
    let mut f = Features {
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
        f16c: std::arch::is_x86_feature_detected!("f16c"),
        avx_vnni: std::arch::is_x86_feature_detected!("avxvnni"),
        avx_vnni_int8: std::arch::is_x86_feature_detected!("avxvnniint8"),
        avx512f: std::arch::is_x86_feature_detected!("avx512f"),
        avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
        avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
        avx512vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
        avx512bf16: std::arch::is_x86_feature_detected!("avx512bf16"),
        avx512vbmi: std::arch::is_x86_feature_detected!("avx512vbmi"),
        ..Features::default()
    };
    // SAFETY: CPUID is available on every x86-64 CPU.
    let max_leaf = unsafe { __cpuid(0) }.eax;
    if max_leaf >= 7 {
        // SAFETY: leaf 7 is supported (`max_leaf`).
        let leaf7 = unsafe { __cpuid_count(7, 0) };
        (f.amx_tile, f.amx_int8, f.amx_bf16) = amx_bits(leaf7.edx);
        // SAFETY: leaf 1 always exists.
        let osxsave = unsafe { __cpuid(1) }.ecx & (1 << 27) != 0;
        if osxsave && std::arch::is_x86_feature_detected!("xsave") {
            // SAFETY: OSXSAVE is set and XSAVE reported, so XGETBV is allowed.
            f.os_tile_state = unsafe { read_xcr0() } & XCR0_TILE_BITS == XCR0_TILE_BITS;
        }
    }
    f
}

/// The features of this CPU, read once.
pub(crate) fn features() -> &'static Features {
    static FEATURES: OnceLock<Features> = OnceLock::new();
    FEATURES.get_or_init(read_features)
}

#[allow(unused_unsafe)]
fn read_raw_id() -> RawId {
    // SAFETY: CPUID leaves 0 and 1 exist on every x86-64 CPU.
    let (l0, l1) = unsafe { (__cpuid(0), __cpuid(1)) };
    RawId {
        vendor: [l0.ebx, l0.edx, l0.ecx],
        signature: l1.eax,
    }
}

/// Micro-architecture class of this CPU, read once.
pub(crate) fn microarch() -> Microarch {
    static CLASS: OnceLock<Microarch> = OnceLock::new();
    *CLASS.get_or_init(|| classify(read_raw_id(), features().avx512f))
}

/// Kernel selection policy of a micro-architecture class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Policy {
    /// 512-bit integer dot kernels (`vpdpbusd`/`vpmaddubsw` on zmm) are preferred to 256-bit ones.
    pub wide_int: bool,
    /// 512-bit f32/f16 FMA kernels are preferred to the AVX2 ones.
    pub wide_float: bool,
    /// `vdpbf16ps` BF16 kernels are allowed (the tier ladder still needs `avx512bf16`).
    pub bf16_dot: bool,
    /// Smallest prompt (tokens) for which the AMX kernels replace the vector kernels.
    pub amx_min_batch: usize,
}

/// Policy of a class, before environment overrides.
///
/// Reasoning (hypotheses, not measurements; see docs/cpu-avx512-amx.md):
/// - Zen 4 and the Intel client parts issue a 512-bit operation at the rate of two 256-bit
///   ones, so floating-point kernels gain nothing from the wider vectors; the integer dot
///   kernels still save uops, so they stay wide. `vdpbf16ps` doubles the multiply-adds per
///   instruction and stays on.
/// - Skylake-SP and Cascade Lake lower their clock under heavy 512-bit code: floating-point
///   kernels stay on 256-bit; the integer kernels follow the existing AVX-512 tiers.
/// - Zen 5, Ice Lake-SP and the AMX servers have full-width units.
pub(crate) fn policy_for(micro: Microarch) -> Policy {
    let base = Policy {
        wide_int: true,
        wide_float: true,
        bf16_dot: true,
        amx_min_batch: 8,
    };
    match micro {
        Microarch::Zen4 | Microarch::IntelClient512 => Policy {
            wide_float: false,
            ..base
        },
        Microarch::IntelSkylakeServer | Microarch::IntelCascadeLake => Policy {
            wide_float: false,
            ..base
        },
        _ => base,
    }
}

/// `true`/`false` from an environment variable (`0`, `off`, `false`, `no` and the empty string
/// are false); `None` when unset.
fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|v| {
        !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "off" | "false" | "no"
        )
    })
}

/// The policy of this CPU after the `FLLM_X86_WIDE_INT`, `FLLM_X86_WIDE_FLOAT`,
/// `FLLM_X86_BF16` and `FLLM_AMX_MIN_BATCH` overrides, read once.
pub(crate) fn policy() -> &'static Policy {
    static POLICY: OnceLock<Policy> = OnceLock::new();
    POLICY.get_or_init(|| {
        let mut p = policy_for(microarch());
        if let Some(v) = env_flag("FLLM_X86_WIDE_INT") {
            p.wide_int = v;
        }
        if let Some(v) = env_flag("FLLM_X86_WIDE_FLOAT") {
            p.wide_float = v;
        }
        if let Some(v) = env_flag("FLLM_X86_BF16") {
            p.bf16_dot = v;
        }
        if let Some(n) = std::env::var("FLLM_AMX_MIN_BATCH")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            p.amx_min_batch = n.max(1);
        }
        p
    })
}

/// Ask Linux for permission to use the AMX tile data: `arch_prctl(ARCH_REQ_XCOMP_PERM,
/// XFEATURE_XTILEDATA)`. Without it the first tile instruction raises SIGILL. Returns whether
/// the permission is granted. The grant is per process and one-time; the result is cached.
#[cfg(target_os = "linux")]
fn request_tile_permission() -> bool {
    const ARCH_REQ_XCOMP_PERM: u64 = 0x1023;
    const XFEATURE_XTILEDATA: u64 = 18;
    const SYS_ARCH_PRCTL: i64 = 158;
    let result: i64;
    // SAFETY: a plain system call with two integer arguments; `syscall` clobbers rcx and r11.
    unsafe {
        std::arch::asm!(
            "syscall",
            inlateout("rax") SYS_ARCH_PRCTL => result,
            in("rdi") ARCH_REQ_XCOMP_PERM,
            in("rsi") XFEATURE_XTILEDATA,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    result == 0
}

#[cfg(not(target_os = "linux"))]
fn request_tile_permission() -> bool {
    false
}

/// Whether AMX tiles can be used by this process: the CPU has AMX-TILE and AMX-INT8, the OS
/// enabled the tile state, and (Linux) the permission request succeeded. The request is made
/// once, on the first call.
pub(crate) fn amx_ready() -> bool {
    static READY: OnceLock<bool> = OnceLock::new();
    *READY.get_or_init(|| {
        let f = features();
        f.amx_tile
            && f.amx_int8
            && f.os_tile_state
            && f.avx512f
            && f.avx512bw
            && request_tile_permission()
    })
}

/// A one-line description for `selftest-cpu` and logs.
pub(crate) fn describe() -> String {
    let f = features();
    let p = policy();
    let mut flags = Vec::new();
    for (name, on) in [
        ("avx512bf16", f.avx512bf16),
        ("avx512vbmi", f.avx512vbmi),
        ("avxvnniint8", f.avx_vnni_int8),
        ("amx-tile", f.amx_tile),
        ("amx-int8", f.amx_int8),
        ("amx-bf16", f.amx_bf16),
        ("os-tile-state", f.os_tile_state),
    ] {
        if on {
            flags.push(name);
        }
    }
    format!(
        "{:?}; extra features: {}; policy: wide-int {}, wide-float {}, bf16-dot {}, amx-min-batch {}",
        microarch(),
        if flags.is_empty() {
            "none".to_string()
        } else {
            flags.join(" ")
        },
        p.wide_int,
        p.wide_float,
        p.bf16_dot,
        p.amx_min_batch
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTEL: [u32; 3] = [0x756e_6547, 0x4965_6e69, 0x6c65_746e];
    const AMD: [u32; 3] = [0x6874_7541, 0x6974_6e65, 0x444d_4163];

    /// Build the leaf-1 EAX word of a display family/model/stepping.
    fn signature(family: u32, model: u32, stepping: u32) -> u32 {
        let (base_family, ext_family) = if family >= 0xf {
            (0xf, family - 0xf)
        } else {
            (family, 0)
        };
        let (base_model, ext_model) = (model & 0xf, model >> 4);
        stepping | (base_model << 4) | (base_family << 8) | (ext_model << 16) | (ext_family << 20)
    }

    fn id(vendor: [u32; 3], family: u32, model: u32, stepping: u32) -> RawId {
        RawId {
            vendor,
            signature: signature(family, model, stepping),
        }
    }

    #[test]
    fn family_and_model_decode_the_extended_fields() {
        let zen4 = id(AMD, 0x19, 0x61, 2);
        assert_eq!(
            (zen4.family(), zen4.model(), zen4.stepping()),
            (0x19, 0x61, 2)
        );
        let spr = id(INTEL, 6, 0x8f, 8);
        assert_eq!((spr.family(), spr.model(), spr.stepping()), (6, 0x8f, 8));
        let zen5 = id(AMD, 0x1a, 0x44, 0);
        assert_eq!((zen5.family(), zen5.model()), (0x1a, 0x44));
    }

    #[test]
    fn vendors_are_recognised_from_the_leaf_zero_words() {
        assert_eq!(id(INTEL, 6, 0x8f, 0).vendor(), Vendor::Intel);
        assert_eq!(id(AMD, 0x19, 0x10, 0).vendor(), Vendor::Amd);
        assert_eq!(id([0, 0, 0], 6, 0x8f, 0).vendor(), Vendor::Other);
    }

    #[test]
    fn intel_models_map_to_the_documented_classes() {
        use Microarch::*;
        let cases = [
            (0x55, 4, IntelSkylakeServer),
            (0x55, 7, IntelCascadeLake),
            (0x55, 11, IntelCascadeLake),
            (0x6a, 6, IntelIceLakeServer),
            (0x7e, 5, IntelClient512),
            (0x8c, 1, IntelClient512),
            (0xa7, 1, IntelClient512),
            (0x8f, 8, IntelAmxServer),
            (0xcf, 2, IntelAmxServer),
            (0xad, 1, IntelAmxServer),
            (0xb7, 1, IntelOther), // Raptor Lake: no AVX-512
        ];
        for (model, stepping, want) in cases {
            assert_eq!(
                classify(id(INTEL, 6, model, stepping), true),
                want,
                "{model:#x}"
            );
        }
        // A non-family-6 Intel CPU is never classified by its family-6 model number.
        assert_eq!(classify(id(INTEL, 15, 0x55, 4), true), IntelOther);
        // Diamond Rapids: family 19, model 1.
        assert_eq!(classify(id(INTEL, 19, 0x01, 0), true), IntelAmxServer);
    }

    #[test]
    fn amd_zen_generations_follow_family_and_the_avx512_bit() {
        use Microarch::*;
        assert_eq!(classify(id(AMD, 0x19, 0x61, 2), true), Zen4); // Raphael
        assert_eq!(classify(id(AMD, 0x19, 0x11, 1), true), Zen4); // Genoa
        assert_eq!(classify(id(AMD, 0x19, 0x21, 0), false), AmdOther); // Zen 3 (Vermeer)
        assert_eq!(classify(id(AMD, 0x1a, 0x44, 0), true), Zen5Plus); // Granite Ridge
        assert_eq!(classify(id(AMD, 0x1a, 0x02, 0), true), Zen5Plus); // Turin
        assert_eq!(classify(id(AMD, 0x17, 0x71, 0), false), AmdOther); // Zen 2
    }

    #[test]
    fn amx_bits_come_from_the_leaf_seven_edx_word() {
        assert_eq!(amx_bits(0), (false, false, false));
        assert_eq!(amx_bits(AMX_TILE_BIT | AMX_INT8_BIT), (true, true, false));
        assert_eq!(
            amx_bits(AMX_TILE_BIT | AMX_INT8_BIT | AMX_BF16_BIT),
            (true, true, true)
        );
        // Unrelated bits do not matter.
        assert_eq!(amx_bits(0x03c0_0410), (true, true, true));
        assert_eq!(amx_bits(0x0000_0410), (false, false, false));
    }

    #[test]
    fn policy_keeps_float_narrow_where_wide_vectors_do_not_help() {
        use Microarch::*;
        assert!(!policy_for(Zen4).wide_float);
        assert!(policy_for(Zen4).wide_int);
        assert!(policy_for(Zen4).bf16_dot);
        assert!(!policy_for(IntelSkylakeServer).wide_float);
        assert!(!policy_for(IntelClient512).wide_float);
        for m in [Zen5Plus, IntelAmxServer, IntelIceLakeServer, Unknown] {
            let p = policy_for(m);
            assert!(p.wide_int && p.wide_float && p.bf16_dot, "{m:?}");
        }
    }

    #[test]
    fn the_real_cpu_reads_without_panicking() {
        let _ = features();
        let _ = policy();
        assert!(!describe().is_empty());
        // Never claims AMX without the CPU feature bits.
        if !features().amx_tile {
            assert!(!amx_ready());
        }
    }
}
