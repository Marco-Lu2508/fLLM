//! The engine variants the app can bundle, and the choice between them.
//!
//! A variant is one build of the server program (`fllm-rs`) with a different set of GPU backends. The installer holds the base variant
//! (CPU, Vulkan, DirectX; always there) and optional ones (CUDA, later HIP). At the start the app picks the best variant whose hardware is
//! present, using cheap checks that do not load any GPU stack (a file exists, a PCI device is listed).
//!
//! To add a variant (for example `hip`): add one entry to `VARIANTS` (before the base entry; the order is the order of preference); add the
//! field it needs to `Probe` and fill it in `Probe::detect`; list the sidecar in the config of the build that bundles it
//! (`tauri.<name>.conf.json`, see `tauri.cuda.conf.json`) and let the release workflow copy the binary to
//! `desktop/src-tauri/binaries/<sidecar>-<target triple>[.exe]`.
//! Nothing else changes: the settings page lists `VARIANTS`, the fallback goes to the base variant for every other one.

use std::path::PathBuf;

/// The name of the variant that is always installed and the fallback of all others.
pub const BASE: &str = "base";
/// The setting value that lets the app choose.
pub const AUTO: &str = "auto";

/// What the cheap hardware checks found. Plain data, so the choice can be tested without hardware.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Probe {
    /// The NVIDIA driver library (`nvcuda.dll`, `libcuda.so.1`) can be found.
    pub nvidia_driver: bool,
    /// Whether an NVIDIA display device is listed; `None` when the system cannot tell (then the driver library alone counts).
    pub nvidia_device: Option<bool>,
}

pub struct Variant {
    /// The name used in the setting and in `FLLM_DESKTOP_VARIANT`.
    pub name: &'static str,
    /// Other spellings accepted for the name (lower case).
    pub aliases: &'static [&'static str],
    /// The file name of the server program next to the app (without `.exe`).
    pub sidecar: &'static str,
    /// What the user sees.
    pub label: &'static str,
    /// Whether the hardware this variant is for is present.
    pub detect: fn(&Probe) -> bool,
}

fn detect_cuda(p: &Probe) -> bool {
    p.nvidia_driver && p.nvidia_device != Some(false)
}

fn detect_always(_: &Probe) -> bool {
    true
}

/// In order of preference; the base variant is last.
pub const VARIANTS: &[Variant] = &[
    Variant { name: "cuda", aliases: &["nvidia"], sidecar: "fllm-rs-cuda", label: "CUDA + Vulkan", detect: detect_cuda },
    // hook for the Windows HIP build: Variant { name: "hip", aliases: &["rocm", "amd"], sidecar: "fllm-rs-hip", label: "HIP + Vulkan", detect: detect_hip },
    Variant { name: BASE, aliases: &["cpu", "vulkan", "cpu+vulkan", "cpu-vulkan", "default"], sidecar: "fllm-rs", label: "CPU + Vulkan + DirectX", detect: detect_always },
];

pub fn base() -> &'static Variant {
    VARIANTS.iter().find(|v| v.name == BASE).expect("the base variant")
}

/// What the user (or the environment) asked for.
#[derive(Clone, Copy)]
pub enum Request {
    Auto,
    Named(&'static Variant),
}

/// Reads a setting value: "" and "auto" mean automatic, otherwise a name or alias of a variant (any case).
pub fn parse_request(text: &str) -> Result<Request, String> {
    let t = text.trim().to_ascii_lowercase();
    if t.is_empty() || t == AUTO {
        return Ok(Request::Auto);
    }
    VARIANTS
        .iter()
        .find(|v| v.name == t || v.aliases.contains(&t.as_str()))
        .map(Request::Named)
        .ok_or_else(|| {
            let names: Vec<&str> = VARIANTS.iter().map(|v| v.name).collect();
            format!("\"{}\" is not a variant (use auto or one of: {}).", text.trim(), names.join(", "))
        })
}

/// The environment value `FLLM_DESKTOP_VARIANT` beats the saved setting. An unusable value is ignored with a note.
pub fn request_from(env: Option<&str>, saved: &str) -> (Request, Option<String>) {
    if let Some(value) = env.filter(|v| !v.trim().is_empty()) {
        match parse_request(value) {
            Ok(r) => return (r, None),
            Err(e) => {
                let (r, _) = request_from(None, saved);
                return (r, Some(format!("FLLM_DESKTOP_VARIANT ignored: {e}")));
            }
        }
    }
    match parse_request(saved) {
        Ok(r) => (r, None),
        Err(e) => (Request::Auto, Some(format!("The saved variant was ignored: {e}"))),
    }
}

/// Picks the variant to start. A named variant is used when it is installed (even if its hardware was not found: the user insists);
/// otherwise, and for automatic, the first installed variant whose hardware is present. The base variant is the last resort.
pub fn choose(request: Request, installed: &dyn Fn(&Variant) -> bool, probe: &Probe) -> (&'static Variant, Option<String>) {
    let mut note = None;
    if let Request::Named(v) = request {
        if installed(v) {
            return (v, None);
        }
        note = Some(format!("The {} variant is not installed in this app; choosing automatically.", v.label));
    }
    let chosen = VARIANTS.iter().find(|v| installed(v) && (v.detect)(probe)).unwrap_or_else(base);
    (chosen, note)
}

/// The variant to try when `current` stops right after the start: the base variant, and nothing after that (so it happens once).
pub fn fallback(current: &Variant) -> Option<&'static Variant> {
    if current.name == BASE {
        None
    } else {
        Some(base())
    }
}

/// The path of the server program of a variant: next to the app.
pub fn sidecar_path(v: &Variant) -> PathBuf {
    let name = if cfg!(windows) { format!("{}.exe", v.sidecar) } else { v.sidecar.to_owned() };
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(&name)))
        .unwrap_or_else(|| PathBuf::from(name))
}

impl Probe {
    pub fn detect() -> Probe {
        Probe { nvidia_driver: nvidia_driver_present(), nvidia_device: nvidia_device_listed() }
    }
}

#[cfg(windows)]
fn nvidia_driver_present() -> bool {
    let root = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    root.join("System32").join("nvcuda.dll").is_file()
}

/// Windows lists no devices without more system calls than this check is worth: the driver library is installed with the driver only.
#[cfg(windows)]
fn nvidia_device_listed() -> Option<bool> {
    None
}

#[cfg(target_os = "linux")]
fn nvidia_driver_present() -> bool {
    const DIRS: &[&str] = &[
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/lib/x86_64-linux-gnu",
        "/lib64",
        "/usr/lib/x86_64-linux-gnu/nvidia/current",
        "/usr/lib/wsl/lib",
        "/usr/local/nvidia/lib64",
        "/usr/local/cuda/compat",
    ];
    if DIRS.iter().any(|d| std::path::Path::new(d).join("libcuda.so.1").exists()) {
        return true;
    }
    // the loader cache lists the library names it knows
    std::fs::read("/etc/ld.so.cache").map(|bytes| bytes.windows(12).any(|w| w == b"libcuda.so.1")).unwrap_or(false)
}

/// Looks through the PCI devices for an NVIDIA (0x10de) display or 3D controller. `None` when no PCI device is visible (containers, WSL).
#[cfg(target_os = "linux")]
fn nvidia_device_listed() -> Option<bool> {
    let entries = std::fs::read_dir("/sys/bus/pci/devices").ok()?;
    let mut seen = false;
    for entry in entries.flatten() {
        seen = true;
        let read = |name: &str| std::fs::read_to_string(entry.path().join(name)).unwrap_or_default();
        if read("vendor").trim() == "0x10de" && read("class").trim().starts_with("0x03") {
            return Some(true);
        }
    }
    if seen {
        Some(false)
    } else {
        None
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn nvidia_driver_present() -> bool {
    false
}

#[cfg(not(any(windows, target_os = "linux")))]
fn nvidia_device_listed() -> Option<bool> {
    None
}

impl PartialEq for Variant {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl std::fmt::Debug for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Variant({})", self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(_: &Variant) -> bool {
        true
    }
    fn only_base(v: &Variant) -> bool {
        v.name == BASE
    }
    const NVIDIA: Probe = Probe { nvidia_driver: true, nvidia_device: Some(true) };
    const NOTHING: Probe = Probe { nvidia_driver: false, nvidia_device: Some(false) };

    #[test]
    fn automatic_takes_cuda_when_the_hardware_is_there() {
        assert_eq!(choose(Request::Auto, &all, &NVIDIA).0.name, "cuda");
    }

    #[test]
    fn automatic_takes_base_without_hardware_or_without_the_binary() {
        assert_eq!(choose(Request::Auto, &all, &NOTHING).0.name, BASE);
        assert_eq!(choose(Request::Auto, &only_base, &NVIDIA).0.name, BASE);
    }

    #[test]
    fn driver_without_a_listed_device_is_not_enough_but_an_unknown_device_list_is() {
        let stale_driver = Probe { nvidia_driver: true, nvidia_device: Some(false) };
        assert_eq!(choose(Request::Auto, &all, &stale_driver).0.name, BASE);
        let unknown = Probe { nvidia_driver: true, nvidia_device: None };
        assert_eq!(choose(Request::Auto, &all, &unknown).0.name, "cuda");
        let no_driver = Probe { nvidia_driver: false, nvidia_device: Some(true) };
        assert_eq!(choose(Request::Auto, &all, &no_driver).0.name, BASE);
    }

    #[test]
    fn a_named_variant_wins_over_detection_when_installed() {
        let cuda = parse_request("CUDA").ok().unwrap();
        assert_eq!(choose(cuda, &all, &NOTHING), (VARIANTS.iter().find(|v| v.name == "cuda").unwrap(), None));
    }

    #[test]
    fn a_named_variant_that_is_missing_falls_back_to_automatic_with_a_note() {
        let cuda = parse_request("cuda").ok().unwrap();
        let (v, note) = choose(cuda, &only_base, &NVIDIA);
        assert_eq!(v.name, BASE);
        assert!(note.unwrap().contains("not installed"));
    }

    #[test]
    fn parsing_accepts_names_aliases_and_auto() {
        assert!(matches!(parse_request(""), Ok(Request::Auto)));
        assert!(matches!(parse_request(" Auto "), Ok(Request::Auto)));
        assert!(matches!(parse_request("cpu"), Ok(Request::Named(v)) if v.name == BASE));
        assert!(matches!(parse_request("CPU+Vulkan"), Ok(Request::Named(v)) if v.name == BASE));
        assert!(matches!(parse_request("nvidia"), Ok(Request::Named(v)) if v.name == "cuda"));
        assert!(parse_request("tpu").is_err());
    }

    #[test]
    fn the_environment_beats_the_saved_setting_and_bad_values_are_ignored() {
        assert!(matches!(request_from(Some("cuda"), "base").0, Request::Named(v) if v.name == "cuda"));
        assert!(matches!(request_from(Some("  "), "base").0, Request::Named(v) if v.name == BASE));
        assert!(matches!(request_from(None, "cuda").0, Request::Named(v) if v.name == "cuda"));
        let (r, note) = request_from(Some("bogus"), "cuda");
        assert!(matches!(r, Request::Named(v) if v.name == "cuda"));
        assert!(note.unwrap().contains("FLLM_DESKTOP_VARIANT"));
        let (r, note) = request_from(None, "bogus");
        assert!(matches!(r, Request::Auto));
        assert!(note.is_some());
    }

    #[test]
    fn fallback_goes_to_base_once() {
        let cuda = VARIANTS.iter().find(|v| v.name == "cuda").unwrap();
        assert_eq!(fallback(cuda).map(|v| v.name), Some(BASE));
        assert!(fallback(base()).is_none());
    }

    #[test]
    fn the_table_is_well_formed() {
        assert_eq!(VARIANTS.last().unwrap().name, BASE);
        assert_eq!(VARIANTS.iter().filter(|v| v.name == BASE).count(), 1);
        assert_eq!(base().sidecar, "fllm-rs");
    }
}
