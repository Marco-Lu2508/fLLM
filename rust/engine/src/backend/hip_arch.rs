//! Which AMD GPUs the HIP build serves, and what happens with a chip whose code is not in the binary.
//!
//! The binary carries one code object per target of `FLLM_HIP_ARCHS` (`build.rs`; by default every gfx target the compiler knows plus the
//! generic targets `gfx9-generic`, `gfx10-1-generic`, `gfx10-3-generic`, `gfx11-generic`, ...). At run time the HIP runtime hands a GPU the
//! exact code object, else the generic one of its family. This module holds
//!
//! * the table `hip/arch_table.csv` (the same file `hip/hip_arch.h` mirrors and `scripts/hip_arch_check.sh` checks against the compiler):
//!   wave size, matrix-core kind, integer dot product, the kernel set fLLM runs, and the generic target of every gfx name; a name that is
//!   not in the table gets the conservative class (plain kernels, the wave size of its generation);
//! * the decision what to do before the runtime starts when a GPU has neither an exact nor a generic code object in this binary: load the code
//!   of the nearest chip through `HSA_OVERRIDE_GFX_VERSION` (Linux; the ROCm user-mode stack reads it once when it starts), or say that it
//!   cannot work and what to rebuild with;
//! * `FLLM_HIP_ARCH_OVERRIDE=gfxNNNN`: treat every AMD GPU as this chip (kernel selection on the host, and the matching
//!   `HSA_OVERRIDE_GFX_VERSION` for the runtime);
//! * the list of GPUs that are present but failed the device self-check (`hip/hip_device.cu`), for `fllm-rs devices` and the Studio.
//!
//! Level of verification: the table logic is unit tested on the CPU; the run-time part has only run on a Radeon 840M (gfx1153) so far.

#![allow(dead_code)]

use std::sync::{Mutex, OnceLock};

const TABLE_CSV: &str = include_str!("../../hip/arch_table.csv");
const DEFAULT_ARCHS: &str = include_str!("../../hip/default_archs.txt");

/// The kernel set fLLM launches on a device (`FllmHipArch` of `hip/fllm_hip.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelSet {
    /// Plain kernels (FP32 / FP16 arithmetic, DPP or integer dot product where the chip has it); no matrix cores.
    Plain,
    /// MFMA 16x16x16 (CDNA).
    Cdna,
    /// WMMA with 16-element fragments (gfx11 except gfx117x).
    Rdna3,
    /// WMMA with 8-element fragments (gfx1200 / gfx1201).
    Rdna4,
}

impl KernelSet {
    pub fn from_code(code: i32) -> Self {
        match code {
            1 => Self::Cdna,
            2 => Self::Rdna3,
            3 => Self::Rdna4,
            _ => Self::Plain,
        }
    }

    /// What the user reads: `MFMA`, `WMMA`, `none`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Plain => "none",
            Self::Cdna => "MFMA",
            Self::Rdna3 | Self::Rdna4 => "WMMA",
        }
    }
}

/// 8-bit integer dot product of a chip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dot4 {
    /// A byte loop.
    None,
    /// `v_dot4_i32_i8`
    Sdot4,
    /// `v_dot4_i32_iu8`
    Sudot4,
}

impl Dot4 {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "byte loop",
            Self::Sdot4 => "dp4a (v_dot4_i32_i8)",
            Self::Sudot4 => "dp4a (v_dot4_i32_iu8)",
        }
    }
}

/// One row of `hip/arch_table.csv`, or the conservative class of a name that is not in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchInfo {
    /// `gfx1100` (without the `:sramecc+:xnack-` suffix).
    pub gfx: String,
    pub known: bool,
    pub family: String,
    /// Threads per wavefront (0: unknown).
    pub wave: u32,
    /// The matrix unit of the hardware: `none`, `mfma`, `wmma` (16-element fragments), `wmma12` (8-element), `other`.
    pub matrix: String,
    /// The kernel set fLLM runs on it.
    pub kernels: KernelSet,
    pub dot4: Dot4,
    /// Packed FP16 dot product (`v_dot2`) that the skinny kernels use.
    pub dot2: bool,
    /// The generic target whose code object runs on it (`gfx10-3-generic`), if any.
    pub generic: Option<String>,
}

fn parse_table() -> Vec<ArchInfo> {
    let mut rows = Vec::new();
    for line in TABLE_CSV.lines().skip(1).map(str::trim).filter(|l| !l.is_empty()) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() != 8 {
            continue;
        }
        rows.push(ArchInfo {
            gfx: c[0].to_owned(),
            known: true,
            family: c[1].to_owned(),
            wave: c[2].parse().unwrap_or(0),
            matrix: c[3].to_owned(),
            kernels: match c[4] {
                "cdna" => KernelSet::Cdna,
                "rdna3" => KernelSet::Rdna3,
                "rdna4" => KernelSet::Rdna4,
                _ => KernelSet::Plain,
            },
            dot4: match c[5] {
                "sdot4" => Dot4::Sdot4,
                "sudot4" => Dot4::Sudot4,
                _ => Dot4::None,
            },
            dot2: c[6] == "1",
            generic: (c[7] != "-").then(|| c[7].to_owned()),
        });
    }
    rows
}

/// All rows of the table, in file order.
pub fn table() -> &'static [ArchInfo] {
    static TABLE: OnceLock<Vec<ArchInfo>> = OnceLock::new();
    TABLE.get_or_init(parse_table)
}

/// `gfx90a:sramecc+:xnack-` -> `gfx90a`
pub fn base_name(gcn: &str) -> &str {
    gcn.split(':').next().unwrap_or(gcn).trim()
}

/// The generic target a name that is not in the table would belong to (the family rule of LLVM's generic processors), if any.
fn generic_by_family(name: &str) -> Option<&'static str> {
    let rest = name.strip_prefix("gfx")?;
    let g = match rest.len() {
        3 if rest.starts_with('9') && !(rest.starts_with("90") && matches!(rest.as_bytes()[2], b'8' | b'a')) && !rest.starts_with("94") && !rest.starts_with("95") => "gfx9-generic",
        3 if rest.starts_with("94") || rest.starts_with("95") => "gfx9-4-generic",
        4 if rest.starts_with("101") => "gfx10-1-generic",
        4 if rest.starts_with("103") => "gfx10-3-generic",
        4 if rest.starts_with("110") || rest.starts_with("115") => "gfx11-generic",
        4 if rest.starts_with("117") => "gfx11-7-generic",
        4 if rest.starts_with("120") => "gfx12-generic",
        4 if rest.starts_with("125") => "gfx12-5-generic",
        _ => return None,
    };
    Some(g)
}

/// What is known about the chip `gcn` (a gcnArchName). A name that is not in the table gets the conservative class: plain kernels, no dot
/// products, the wave size of its generation (64 for gfx9, 32 for gfx10 and later), and the generic target its family rule gives.
pub fn classify(gcn: &str) -> ArchInfo {
    let name = base_name(gcn);
    if let Some(row) = table().iter().find(|r| r.gfx == name) {
        return row.clone();
    }
    ArchInfo {
        gfx: name.to_owned(),
        known: false,
        family: "unknown".to_owned(),
        wave: if name.starts_with("gfx9") { 64 } else if name.starts_with("gfx1") { 32 } else { 0 },
        matrix: "unknown".to_owned(),
        kernels: KernelSet::Plain,
        dot4: Dot4::None,
        dot2: false,
        generic: generic_by_family(name).map(str::to_owned),
    }
}

/// The offload architectures a default build compiles (`windows`: the Windows list). The same text feeds `build.rs`.
pub fn default_archs(windows: bool) -> Vec<String> {
    let key = if windows { "windows=" } else { "linux=" };
    DEFAULT_ARCHS
        .lines()
        .find_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.split(';').map(str::trim).filter(|a| !a.is_empty()).map(str::to_owned).collect())
        .unwrap_or_default()
}

/// The targets this binary has code for (`FLLM_HIP_BUILT_ARCHS`, recorded by `build.rs`); empty when the build did not record them.
pub fn built_archs() -> Vec<String> {
    option_env!("FLLM_HIP_BUILT_ARCHS").map(|l| l.split(';').map(str::trim).filter(|a| !a.is_empty()).map(str::to_owned).collect()).unwrap_or_default()
}

/// Which code object of the binary serves a chip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Coverage {
    /// The binary has code for exactly this gfx target.
    Exact,
    /// The HIP runtime loads the generic code object of the family (the string is its name).
    Generic(String),
    /// Neither: the runtime finds no code object for the chip.
    Missing,
}

/// `built` empty means the build recorded nothing: assume the best (`Exact`).
pub fn coverage(arch: &str, built: &[String]) -> Coverage {
    let name = base_name(arch);
    if built.is_empty() || built.iter().any(|b| b == name) {
        return Coverage::Exact;
    }
    match classify(name).generic {
        Some(g) if built.contains(&g) => Coverage::Generic(g),
        _ => Coverage::Missing,
    }
}

/// `gfx1030` -> `10.3.0`, `gfx90a` -> `9.0.10`, `gfx1151` -> `11.5.1`: the value of `HSA_OVERRIDE_GFX_VERSION` that makes the ROCm stack
/// treat a GPU as that chip.
pub fn hsa_version(arch: &str) -> Option<String> {
    let rest = base_name(arch).strip_prefix("gfx")?;
    if !rest.is_ascii() || !(3..=4).contains(&rest.len()) {
        return None;
    }
    let (major, tail) = rest.split_at(rest.len() - 2);
    let minor: u32 = tail[..1].parse().ok()?;
    let step = u32::from_str_radix(&tail[1..], 16).ok()?;
    let major: u32 = major.parse().ok()?;
    Some(format!("{major}.{minor}.{step}"))
}

/// The targets whose code can stand in for `arch` when the binary has neither the exact nor the generic code object, nearest first, limited to
/// the ones in `built`. Only chips of the same instruction set generation are offered: code with matrix-core or dot-product instructions of
/// another chip would fail at the first kernel (the device self-check catches that and reports the GPU as not usable).
pub fn override_candidates(arch: &str, built: &[String]) -> Vec<String> {
    let name = base_name(arch);
    let chain: &[&str] = match name {
        "gfx906" => &["gfx900"],
        "gfx908" | "gfx90a" => &[],
        "gfx940" | "gfx941" | "gfx942" => &["gfx942"],
        "gfx950" => &[],
        "gfx1011" | "gfx1012" => &["gfx1010"],
        _ => match name.get(..6) {
            Some("gfx101") => &["gfx1010"],
            Some("gfx103") => &["gfx1030"],
            Some("gfx110") => &["gfx1100", "gfx1101", "gfx1102", "gfx1103"],
            Some("gfx115") => &["gfx1151", "gfx1150", "gfx1152", "gfx1153"],
            Some("gfx117") => &["gfx1170", "gfx1171", "gfx1172"],
            Some("gfx120") => &["gfx1201", "gfx1200"],
            Some("gfx125") => &["gfx1250", "gfx1251"],
            _ if name.starts_with("gfx9") && name.len() == 6 => &["gfx900"],
            _ => &[],
        },
    };
    chain.iter().filter(|c| **c != name && built.iter().any(|b| b == **c)).map(|c| (*c).to_owned()).collect()
}

/// `gfx_target_version` of a KFD topology node (`100302`) as a gfx name (`gfx1032`); `None` for the CPU nodes (0).
pub fn arch_from_kfd_version(version: u32) -> Option<String> {
    if version == 0 {
        return None;
    }
    let (major, minor, step) = (version / 10000, (version / 100) % 100, version % 100);
    if minor > 9 || step > 15 {
        return None;
    }
    Some(format!("gfx{major}{minor}{step:x}"))
}

/// The `gfx_target_version` line of a KFD node `properties` file.
pub fn kfd_version_of(properties: &str) -> Option<u32> {
    properties.lines().find_map(|l| l.strip_prefix("gfx_target_version")).and_then(|v| v.trim().parse().ok())
}

/// The gfx names of the AMD GPUs the kernel driver (KFD) lists, before the HIP runtime has been started. Empty on other platforms and when
/// the driver is not loaded.
pub fn kfd_gpu_archs() -> Vec<String> {
    #[cfg(target_os = "linux")]
    {
        let Ok(dir) = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes") else { return Vec::new() };
        let mut nodes: Vec<_> = dir.flatten().map(|e| e.path()).collect();
        nodes.sort();
        return nodes
            .iter()
            .filter_map(|n| std::fs::read_to_string(n.join("properties")).ok())
            .filter_map(|p| kfd_version_of(&p))
            .filter_map(arch_from_kfd_version)
            .collect();
    }
    #[allow(unreachable_code)]
    Vec::new()
}

/// What to do before the HIP runtime starts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimePlan {
    /// `HSA_OVERRIDE_GFX_VERSION` to set, and the log line that says why.
    pub set_hsa: Option<(String, String)>,
    /// Lines for the log (one per GPU that has no exact code object, and every problem).
    pub notes: Vec<String>,
}

/// The decision of the start-up step. `present`: gfx names of the AMD GPUs (empty: unknown); `built`: the binary's targets;
/// `user_hsa`: `HSA_OVERRIDE_GFX_VERSION` is set already (then it is the user's choice and is left alone); `arch_override`:
/// `FLLM_HIP_ARCH_OVERRIDE`; `hsa_applies`: the platform's runtime reads `HSA_OVERRIDE_GFX_VERSION` (Linux).
pub fn plan_runtime(present: &[String], built: &[String], user_hsa: bool, arch_override: Option<&str>, hsa_applies: bool) -> RuntimePlan {
    let mut plan = RuntimePlan::default();
    if let Some(ov) = arch_override.map(base_name).filter(|o| !o.is_empty()) {
        let Some(version) = hsa_version(ov) else {
            plan.notes.push(format!("FLLM_HIP_ARCH_OVERRIDE={ov} is not a gfx name (like gfx1030); ignored"));
            return plan;
        };
        let known = classify(ov).known;
        plan.notes.push(format!(
            "FLLM_HIP_ARCH_OVERRIDE={ov}: every AMD GPU is treated as {ov}{}",
            if known { String::new() } else { " (not in fLLM's table: plain kernels)".to_owned() }
        ));
        let already = !present.is_empty() && present.iter().all(|p| p == ov);
        if hsa_applies && !user_hsa && !already {
            plan.set_hsa = Some((version.clone(), format!("HSA_OVERRIDE_GFX_VERSION={version} so that the runtime loads the code of {ov}")));
        } else if !hsa_applies && !already {
            plan.notes.push(format!("this platform's HIP runtime does not read HSA_OVERRIDE_GFX_VERSION: {ov} only changes fLLM's kernel selection here"));
        }
        return plan;
    }
    for arch in present {
        match coverage(arch, built) {
            Coverage::Exact => {}
            Coverage::Generic(g) => plan.notes.push(format!("{arch}: no exact code object in this binary, the HIP runtime uses the generic {g} one")),
            Coverage::Missing => {
                let candidates = override_candidates(arch, built);
                match candidates.first() {
                    _ if user_hsa => plan.notes.push(format!("{arch}: not in this binary's GPU list; HSA_OVERRIDE_GFX_VERSION is set and decides which code is loaded")),
                    Some(near) if hsa_applies => {
                        let version = hsa_version(near).unwrap_or_default();
                        if plan.set_hsa.is_none() {
                            plan.set_hsa = Some((version.clone(), format!("{arch} has no code object in this binary: HSA_OVERRIDE_GFX_VERSION={version} loads the code of {near}; the device self-check decides whether it is used")));
                        } else {
                            plan.notes.push(format!("{arch}: no code object in this binary and another GPU already needs HSA_OVERRIDE_GFX_VERSION; it may not be usable"));
                        }
                    }
                    _ => plan.notes.push(format!("{arch}: no code object in this binary (built for: {}); rebuild with FLLM_HIP_ARCHS={arch}", built.join(";"))),
                }
            }
        }
    }
    plan
}

/// Start-up step, once per process and before the first HIP call: applies [`plan_runtime`] (sets `HSA_OVERRIDE_GFX_VERSION`, logs).
pub fn prepare_runtime() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let arch_override = std::env::var("FLLM_HIP_ARCH_OVERRIDE").ok().filter(|v| !v.trim().is_empty());
        let user_hsa = std::env::var_os("HSA_OVERRIDE_GFX_VERSION").is_some_and(|v| !v.is_empty());
        let plan = plan_runtime(&kfd_gpu_archs(), &built_archs(), user_hsa, arch_override.as_deref(), crate::backend::hip_runtime::hsa_override_applies());
        for note in &plan.notes {
            eprintln!("[fllm-rs] hip: {note}");
        }
        if let Some((version, why)) = plan.set_hsa {
            // (the ROCm stack reads the variable when it starts; if it has started already this has no effect, and the device check reports the GPU)
            eprintln!("[fllm-rs] hip: {why} (set inside the process; if the GPU is then not usable, export the variable before starting fllm-rs)");
            std::env::set_var("HSA_OVERRIDE_GFX_VERSION", version);
        }
    });
}

/// A GPU that is present but cannot run this binary's code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unusable {
    /// HIP ordinal.
    pub ordinal: usize,
    pub name: String,
    pub arch: String,
    pub reason: String,
}

static UNUSABLE: Mutex<Vec<Unusable>> = Mutex::new(Vec::new());

/// What to tell the user to try for a chip whose code does not run.
pub fn remedy(arch: &str, built: &[String]) -> String {
    let near = override_candidates(arch, built);
    let mut text = format!("rebuild with FLLM_HIP_ARCHS={arch}");
    if let Some(n) = near.first() {
        if let Some(v) = hsa_version(n) {
            text.push_str(&format!(", or try HSA_OVERRIDE_GFX_VERSION={v} (the code of {n}; Linux)"));
        }
    }
    text.push_str(", or FLLM_HIP_ARCH_OVERRIDE=gfxNNNN; see docs/hip.md");
    text
}

/// Records a GPU that failed the self-check (once per ordinal), logs it and tells the placement ledger.
pub fn note_unusable(ordinal: usize, name: &str, arch: &str, reason: &str) {
    let mut list = UNUSABLE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if list.iter().any(|u| u.ordinal == ordinal) {
        return;
    }
    list.push(Unusable { ordinal, name: name.to_owned(), arch: arch.to_owned(), reason: reason.to_owned() });
    drop(list);
    let text = format!("AMD GPU {name} is present but not usable: {reason}. It is left out; the model runs on the next device. To try: {}", remedy(arch, &built_archs()));
    crate::placement::note_notice(&text);
}

/// The GPUs found by the last [`crate::backend::cuda::list_devices`] that cannot run this binary.
pub fn unusable() -> Vec<Unusable> {
    UNUSABLE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
}

/// The reason a HIP ordinal is not usable, if it was found unusable.
pub fn unusable_reason(ordinal: usize) -> Option<String> {
    unusable().into_iter().find(|u| u.ordinal == ordinal).map(|u| u.reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(list: &[&str]) -> Vec<String> {
        list.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn every_row_has_a_class_and_the_table_is_consistent() {
        let rows = table();
        assert!(rows.len() >= 35, "{} rows", rows.len());
        for r in rows {
            assert!(r.known && r.gfx.starts_with("gfx"), "{r:?}");
            assert!(r.wave == 32 || r.wave == 64, "{}: wave {}", r.gfx, r.wave);
            // wave64 is GCN / CDNA (gfx9), wave32 is everything from gfx10 on
            assert_eq!(r.wave == 64, r.gfx.starts_with("gfx9"), "{}", r.gfx);
            // matrix kernels exist only where the hardware has the matching unit
            match r.kernels {
                KernelSet::Cdna => assert_eq!(r.matrix, "mfma", "{}", r.gfx),
                KernelSet::Rdna3 => assert_eq!(r.matrix, "wmma", "{}", r.gfx),
                KernelSet::Rdna4 => assert_eq!(r.matrix, "wmma12", "{}", r.gfx),
                KernelSet::Plain => {}
            }
            // dot instructions: sudot4 is the gfx11 / gfx12 form, sdot4 the gfx9 / gfx10 one
            if r.dot4 == Dot4::Sudot4 {
                assert!(r.gfx.starts_with("gfx11") || r.gfx.starts_with("gfx12"), "{}", r.gfx);
            }
            if r.dot4 == Dot4::Sdot4 {
                assert!(r.gfx.starts_with("gfx9") || r.gfx.starts_with("gfx10"), "{}", r.gfx);
            }
            // a generic target is the family of its members
            if let Some(g) = &r.generic {
                assert!(g.ends_with("-generic"), "{}", r.gfx);
                assert_eq!(generic_by_family(&r.gfx), Some(g.as_str()), "{}", r.gfx);
            }
        }
        // no duplicate names
        let mut names: Vec<&str> = rows.iter().map(|r| r.gfx.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), rows.len());
    }

    #[test]
    fn the_families_the_kernels_were_written_for() {
        let k = |n: &str| classify(n);
        // matrix cores
        assert_eq!(k("gfx908").kernels, KernelSet::Cdna);
        assert_eq!(k("gfx90a:sramecc+:xnack-").kernels, KernelSet::Cdna);
        assert_eq!(k("gfx942").kernels, KernelSet::Cdna);
        assert_eq!(k("gfx1100").kernels, KernelSet::Rdna3);
        assert_eq!(k("gfx1153").kernels, KernelSet::Rdna3);
        assert_eq!(k("gfx1201").kernels, KernelSet::Rdna4);
        // no matrix cores we use: Vega, RDNA1, RDNA2, gfx117x (WMMA exists, never run), gfx125x
        for n in ["gfx900", "gfx90c", "gfx906", "gfx1010", "gfx1012", "gfx1030", "gfx1036", "gfx1170", "gfx1250"] {
            assert_eq!(k(n).kernels, KernelSet::Plain, "{n}");
        }
        // wave size
        assert_eq!((k("gfx906").wave, k("gfx90a").wave, k("gfx1030").wave, k("gfx1100").wave), (64, 64, 32, 32));
        // dot products: gfx900 / gfx1010 have none, gfx906 / gfx1011 / gfx103x have v_dot4_i32_i8, gfx11 / gfx12 the iu8 form
        assert_eq!(k("gfx900").dot4, Dot4::None);
        assert_eq!(k("gfx906").dot4, Dot4::Sdot4);
        assert_eq!(k("gfx1010").dot4, Dot4::None);
        assert_eq!(k("gfx1011").dot4, Dot4::Sdot4);
        assert_eq!(k("gfx1031").dot4, Dot4::Sdot4);
        assert_eq!(k("gfx1100").dot4, Dot4::Sudot4);
        assert_eq!(k("gfx1201").dot4, Dot4::Sudot4);
        // v_dot2 for the skinny kernels: gfx90a and newer CDNA, gfx11 / gfx12, not gfx908 / gfx906 / gfx125x
        assert!(k("gfx90a").dot2 && k("gfx942").dot2 && k("gfx1103").dot2 && k("gfx1200").dot2);
        assert!(!k("gfx908").dot2 && !k("gfx906").dot2 && !k("gfx1030").dot2 && !k("gfx1250").dot2);
    }

    #[test]
    fn unknown_names_get_the_conservative_class() {
        for (name, wave) in [("gfx9999", 64), ("gfx1999", 32), ("gfx1105", 32), ("gfx13", 32), ("Radeon", 0)] {
            let i = classify(name);
            assert!(!i.known, "{name}");
            assert_eq!((i.kernels, i.dot4, i.dot2, i.wave), (KernelSet::Plain, Dot4::None, false, wave), "{name}");
        }
        // the generic target of a future chip comes from its family
        assert_eq!(classify("gfx1105").generic.as_deref(), Some("gfx11-generic"));
        assert_eq!(classify("gfx1037").generic.as_deref(), Some("gfx10-3-generic"));
        assert_eq!(classify("gfx91c").generic.as_deref(), Some("gfx9-generic"));
        assert_eq!(classify("gfx9999").generic, None);
    }

    #[test]
    fn coverage_prefers_the_exact_then_the_generic_code_object() {
        let built = s(&["gfx1030", "gfx10-3-generic", "gfx1100", "gfx11-generic"]);
        assert_eq!(coverage("gfx1030", &built), Coverage::Exact);
        assert_eq!(coverage("gfx1031:xnack-", &built), Coverage::Generic("gfx10-3-generic".into()));
        assert_eq!(coverage("gfx1153", &built), Coverage::Generic("gfx11-generic".into()));
        assert_eq!(coverage("gfx1201", &built), Coverage::Missing);
        assert_eq!(coverage("gfx1010", &built), Coverage::Missing);
        // a build that recorded nothing is not second-guessed
        assert_eq!(coverage("gfx1201", &[]), Coverage::Exact);
    }

    #[test]
    fn default_lists_cover_every_family_and_use_known_names() {
        let linux = default_archs(false);
        let windows = default_archs(true);
        assert!(linux.len() > windows.len() && windows.len() >= 20);
        for list in [&linux, &windows] {
            // no duplicates
            let mut sorted = list.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), list.len());
        }
        // every row of the table is served by a specific or a generic target of the Linux list
        for r in table() {
            // (gfx940 / gfx941 are early CDNA3 steppings that no generic target covers; the 9.4.2 stand-in is the way for them)
            if matches!(r.gfx.as_str(), "gfx940" | "gfx941") {
                continue;
            }
            assert!(coverage(&r.gfx, &linux) != Coverage::Missing, "{} has no code object in the Linux default list", r.gfx);
        }
        // the generic targets of every family
        for g in ["gfx9-generic", "gfx10-1-generic", "gfx10-3-generic", "gfx11-generic", "gfx11-7-generic", "gfx12-generic"] {
            assert!(linux.iter().any(|a| a == g), "{g}");
        }
        // Windows: no Instinct targets (no Windows runtime for them), but the consumer ones including the generic ones
        assert!(!windows.iter().any(|a| a == "gfx90a" || a == "gfx942" || a == "gfx950" || a == "gfx908"));
        for r in ["gfx900", "gfx1030", "gfx1100", "gfx1151", "gfx1153", "gfx1201"] {
            assert!(coverage(r, &windows) != Coverage::Missing, "{r}");
        }
        // the old explicit lists are still inside
        for a in ["gfx906", "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1200", "gfx1201"] {
            assert_eq!(coverage(a, &linux), Coverage::Exact, "{a}");
        }
    }

    #[test]
    fn hsa_versions() {
        let v = |a: &str| hsa_version(a);
        assert_eq!(v("gfx1030").as_deref(), Some("10.3.0"));
        assert_eq!(v("gfx1032").as_deref(), Some("10.3.2"));
        assert_eq!(v("gfx1151").as_deref(), Some("11.5.1"));
        assert_eq!(v("gfx1153").as_deref(), Some("11.5.3"));
        assert_eq!(v("gfx1201").as_deref(), Some("12.0.1"));
        assert_eq!(v("gfx900").as_deref(), Some("9.0.0"));
        assert_eq!(v("gfx906").as_deref(), Some("9.0.6"));
        assert_eq!(v("gfx90a").as_deref(), Some("9.0.10"));
        assert_eq!(v("gfx90c:xnack-").as_deref(), Some("9.0.12"));
        assert_eq!(v("gfx942").as_deref(), Some("9.4.2"));
        assert_eq!(v("gfx950").as_deref(), Some("9.5.0"));
        assert_eq!(v("gfx"), None);
        assert_eq!(v("navi"), None);
        assert_eq!(v("gfx12345"), None);
    }

    #[test]
    fn kfd_versions_give_gfx_names() {
        assert_eq!(arch_from_kfd_version(100302).as_deref(), Some("gfx1032"));
        assert_eq!(arch_from_kfd_version(110503).as_deref(), Some("gfx1153"));
        assert_eq!(arch_from_kfd_version(90010).as_deref(), Some("gfx90a"));
        assert_eq!(arch_from_kfd_version(90012).as_deref(), Some("gfx90c"));
        assert_eq!(arch_from_kfd_version(90402).as_deref(), Some("gfx942"));
        assert_eq!(arch_from_kfd_version(90006).as_deref(), Some("gfx906"));
        assert_eq!(arch_from_kfd_version(0), None);
        assert_eq!(kfd_version_of("cpu_cores_count 0\ngfx_target_version 110503\nvendor_id 4098\n"), Some(110503));
        assert_eq!(kfd_version_of("cpu_cores_count 8\n"), None);
        // the names round-trip through the versions
        for r in table() {
            let v = hsa_version(&r.gfx).unwrap();
            let p: Vec<u32> = v.split('.').map(|x| x.parse().unwrap()).collect();
            assert_eq!(arch_from_kfd_version(p[0] * 10000 + p[1] * 100 + p[2]).as_deref(), Some(r.gfx.as_str()), "{}", r.gfx);
        }
    }

    #[test]
    fn nearest_code_for_a_chip_that_has_none() {
        let b = s(&["gfx900", "gfx1010", "gfx1030", "gfx1100", "gfx1102", "gfx1151"]);
        assert_eq!(override_candidates("gfx1031", &b), ["gfx1030"]);
        assert_eq!(override_candidates("gfx1036", &b), ["gfx1030"]);
        assert_eq!(override_candidates("gfx1103", &b), ["gfx1100", "gfx1102"]);
        assert_eq!(override_candidates("gfx1153", &b), ["gfx1151"]);
        assert_eq!(override_candidates("gfx90c", &b), ["gfx900"]);
        assert_eq!(override_candidates("gfx1013", &b), ["gfx1010"]);
        // never the code of a chip with other matrix / dot instructions
        assert!(override_candidates("gfx1201", &b).is_empty());
        assert!(override_candidates("gfx1170", &b).is_empty());
        assert!(override_candidates("gfx90a", &b).is_empty());
        assert!(override_candidates("gfx1250", &b).is_empty());
        assert!(override_candidates("gfx1030", &b).is_empty());
    }

    #[test]
    fn plan_runtime_decides_before_the_runtime_starts() {
        let full = default_archs(false);
        // everything covered: nothing to do
        let p = plan_runtime(&s(&["gfx1031", "gfx1153"]), &full, false, None, true);
        assert_eq!(p.set_hsa, None);
        // a build that is restricted to one chip: the nearest code is loaded through the override
        let only_1030 = s(&["gfx1030"]);
        let p = plan_runtime(&s(&["gfx1031"]), &only_1030, false, None, true);
        assert_eq!(p.set_hsa.as_ref().map(|x| x.0.as_str()), Some("10.3.0"));
        assert!(p.set_hsa.unwrap().1.contains("gfx1030"));
        // the user's own HSA_OVERRIDE_GFX_VERSION is never replaced
        let p = plan_runtime(&s(&["gfx1031"]), &only_1030, true, None, true);
        assert_eq!(p.set_hsa, None);
        assert!(p.notes[0].contains("HSA_OVERRIDE_GFX_VERSION is set"));
        // Windows: the override does nothing there, the note says what to rebuild
        let p = plan_runtime(&s(&["gfx1031"]), &only_1030, false, None, false);
        assert_eq!(p.set_hsa, None);
        assert!(p.notes[0].contains("FLLM_HIP_ARCHS=gfx1031"));
        // a chip with no stand-in
        let p = plan_runtime(&s(&["gfx1201"]), &only_1030, false, None, true);
        assert_eq!(p.set_hsa, None);
        assert!(p.notes[0].contains("rebuild with FLLM_HIP_ARCHS=gfx1201"));
        // FLLM_HIP_ARCH_OVERRIDE sets the matching version, unless the GPU already is that chip
        let p = plan_runtime(&s(&["gfx1032"]), &full, false, Some("gfx1030"), true);
        assert_eq!(p.set_hsa.as_ref().map(|x| x.0.as_str()), Some("10.3.0"));
        let p = plan_runtime(&s(&["gfx1030"]), &full, false, Some("gfx1030"), true);
        assert_eq!(p.set_hsa, None);
        let p = plan_runtime(&[], &full, false, Some("gfx1100"), true);
        assert_eq!(p.set_hsa.as_ref().map(|x| x.0.as_str()), Some("11.0.0"));
        let p = plan_runtime(&s(&["gfx1032"]), &full, false, Some("banana"), true);
        assert_eq!(p.set_hsa, None);
        assert!(p.notes[0].contains("not a gfx name"));
        let p = plan_runtime(&s(&["gfx1032"]), &full, true, Some("gfx1030"), true);
        assert_eq!(p.set_hsa, None);
    }

    #[test]
    fn unusable_gpus_are_recorded_once() {
        note_unusable(7, "AMD Radeon Test", "gfx1031", "no code object runs");
        note_unusable(7, "AMD Radeon Test", "gfx1031", "again");
        let list: Vec<_> = unusable().into_iter().filter(|u| u.ordinal == 7).collect();
        assert_eq!(list.len(), 1);
        assert_eq!(unusable_reason(7).as_deref(), Some("no code object runs"));
        assert!(remedy("gfx1031", &s(&["gfx1030"])).contains("HSA_OVERRIDE_GFX_VERSION=10.3.0"));
    }
}
