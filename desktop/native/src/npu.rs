//! The pure heart of the Studio's `npu.js` (the `@pure` zone, lines 8 to 226): everything the NPU
//! card shows, and everything the device grid decides — how the same graphics chip through several
//! driver interfaces becomes one entry, what a device does for the loaded model, and what the
//! footer says about where the model runs.
//!
//! The drawing lives in `settings.rs` (the NPU card and the compute devices of `Settings >
//! Hardware`); the fixtures are `npu_fixtures.js`, which loads this zone out of the Studio's own
//! file.

use crate::tour::Device;
use serde_json::Value;

/// What the NPU card warns about, in one line (`NPU_WARNING`).
pub const NPU_WARNING: &str = "Experimental: the NPU only speeds up reading long prompts (the processor writes the answer). \
The integrated graphics are usually faster. A converted copy of the weights is kept in memory.";

/// The four paragraphs under the card (`NPU_EXPLANATION`).
pub const NPU_EXPLANATION: [&str; 4] = [
    "fLLM uses an Intel NPU as a helper: the model runs on the processor, and the large matrix multiplications of a \
long prompt (64 tokens or more) run on the NPU. Writing the answer is not sped up and stays on the processor.",
    "On a Core Ultra 5 235U the NPU read prompts 1.5 to 3 times faster than the processor alone, and about as fast as \
the integrated graphics, which need no setup. The NPU is worth it when the graphics are busy or absent, or to keep \
the processor free. The first start compiles the NPU programs (about a second each, shown as \"still compiling\"; \
the processor serves those meanwhile); later starts take them from a cache.",
    "The NPU keeps a converted 8-bit copy of the weights it computes with, so a model needs about its own size again \
in memory (a budget limits it). Only ordinary (dense) models are accelerated; mixture-of-experts and other special \
models stay on the processor.",
    "It is never chosen automatically. To use it, pick NPU under Advanced when you load a model, or start fLLM with \
--device npu (or FLLM_NPU_HOST=1). If the NPU cannot take a model, fLLM runs it on the processor and says why here.",
];

/// The makers by their vendor key (`NPU_VENDOR_NAMES`).
pub const NPU_VENDOR_NAMES: [(&str, &str); 4] = [
    ("intel", "Intel"),
    ("amd", "AMD"),
    ("qualcomm", "Qualcomm"),
    ("other", "Unknown maker"),
];

/// The summary of every state the API knows (`NPU_SUMMARIES`).
pub const NPU_SUMMARIES: [(&str, &str); 7] = [
    ("not_in_build", "This build of fLLM has no NPU support."),
    ("no_npu", "No NPU was found on this computer."),
    ("unsupported_hardware", "An NPU was detected, but this kind of NPU is not supported yet."),
    (
        "setup_unavailable",
        "An Intel NPU was found, but fLLM has no automatic setup for this system. Install the OpenVINO runtime \
yourself and fLLM will find it.",
    ),
    (
        "no_device",
        "The OpenVINO runtime is installed but does not list an NPU. The NPU driver may be missing, or this user \
may not be allowed to open the device.",
    ),
    (
        "ready",
        "Ready (experimental). Choose NPU under Advanced when you load a model; long prompts then run on it.",
    ),
    ("installing", "Installing the runtime ..."),
];

/// How a driver interface is named for a person (`DEVICE_BACKEND_NAMES`).
pub const DEVICE_BACKEND_NAMES: [(&str, &str); 6] = [
    ("cuda", "NVIDIA CUDA"),
    ("vulkan", "Vulkan"),
    ("hip", "AMD ROCm"),
    ("metal", "Metal"),
    ("dx12", "DirectX 12"),
    ("sycl", "Intel SYCL"),
];

/// Millions with one decimal (`npuMegabytes`).
pub fn npu_megabytes(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / 1e6)
}

/// The progress bar and its text for a setup run — `npuProgressModel` of `npu.js`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NpuProgress {
    pub visible: bool,
    pub percent: u64,
    pub text: String,
}

pub fn npu_progress_model(progress: Option<&Value>) -> NpuProgress {
    let Some(p) = progress.filter(|p| p.get("busy").and_then(Value::as_bool).unwrap_or(false)) else {
        return NpuProgress::default();
    };
    let total = p.get("total_bytes").and_then(Value::as_u64).unwrap_or(0);
    let done = p.get("done_bytes").and_then(Value::as_u64).unwrap_or(0);
    let percent = if total > 0 { ((done as f64 / total as f64) * 100.0).round().min(100.0) as u64 } else { 0 };
    let detail = if total > 0 {
        format!(" ({} of {} MB)", npu_megabytes(done), npu_megabytes(total))
    } else {
        String::new()
    };
    let phase = p.get("phase").and_then(Value::as_str).unwrap_or("");
    let label = match phase {
        "downloading" => "Downloading",
        "verifying" => "Checking the download",
        "unpacking" => "Unpacking",
        _ => "Working",
    };
    NpuProgress { visible: true, percent, text: format!("{label}{detail}") }
}

/// One row of the hardware list of the card (`label`, `badge`, `supported`, `problem`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NpuHardwareRow {
    pub label: String,
    pub badge: String,
    pub supported: bool,
    pub problem: Option<String>,
}

/// The button of the card (`button` of `npuCardModel`).
#[derive(Debug, Clone, PartialEq)]
pub struct NpuButton {
    pub action: &'static str,
    pub label: String,
}

/// The four formats the card lists when the API names some.
#[derive(Debug, Clone, PartialEq)]
pub struct NpuFormat {
    pub title: &'static str,
    pub items: Vec<String>,
}

/// Everything the NPU card shows, from the API state — `npuCardModel` of `npu.js`. No DOM here.
#[derive(Debug, Clone, PartialEq)]
pub struct NpuCard {
    pub summary: String,
    pub offload: String,
    pub tone: &'static str,
    pub hardware: Vec<NpuHardwareRow>,
    pub button: Option<NpuButton>,
    pub cancel: bool,
    pub remove: bool,
    pub progress: NpuProgress,
    pub error: Option<String>,
    pub formats: Vec<NpuFormat>,
    pub explanation: [&'static str; 4],
}

pub fn npu_card_model(npu: Option<&Value>) -> NpuCard {
    let explanation = NPU_EXPLANATION;
    let Some(npu) = npu else {
        return NpuCard {
            summary: String::from("Checking for an NPU ..."),
            offload: String::new(),
            tone: "muted",
            hardware: Vec::new(),
            button: None,
            cancel: false,
            remove: false,
            progress: NpuProgress::default(),
            error: None,
            formats: Vec::new(),
            explanation,
        };
    };
    let text = |obj: &Value, key: &str| obj.get(key).and_then(Value::as_str).unwrap_or("").to_string();
    let setup = npu.get("setup").cloned().unwrap_or(Value::Object(Default::default()));
    let progress = setup.get("progress").cloned().unwrap_or(Value::Object(Default::default()));
    let runtime = npu.get("runtime").cloned().unwrap_or(Value::Object(Default::default()));
    let state = text(npu, "state");

    let mut summary = NPU_SUMMARIES
        .iter()
        .find(|(key, _)| *key == state)
        .map(|(_, s)| s.to_string())
        .unwrap_or_else(|| text(npu, "state_text"));
    if state == "runtime_missing" {
        let mb = match setup.get("download_bytes").and_then(Value::as_u64) {
            Some(bytes) if bytes > 0 => format!(" (a download of about {} MB)", (bytes as f64 / 1e6).round()),
            _ => String::new(),
        };
        summary = format!(
            "An Intel NPU was found. The runtime that drives it (OpenVINO {}{mb}) is not installed yet.",
            text(&setup, "pinned_version")
        );
    }
    if state == "ready" {
        let version = text(&runtime, "version");
        if !version.is_empty() {
            summary = format!(
                "Ready (experimental), OpenVINO {version}. Choose NPU under Advanced when you load a model; long \
prompts then run on it."
            );
        }
    }
    let dev = npu.get("devices").and_then(Value::as_array).and_then(|a| a.first()).cloned();
    if state == "ready" {
        if let Some(dev) = dev.as_ref() {
            let full = text(dev, "full_name");
            if !full.is_empty() {
                let arch = text(dev, "architecture");
                summary.push_str(&format!(
                    " {full}{}.",
                    if arch.is_empty() { String::new() } else { format!(" (platform {arch})") }
                ));
            }
        }
    }
    let amd_hardware = npu
        .get("hardware")
        .and_then(Value::as_array)
        .map(|a| a.iter().any(|h| h.get("vendor").and_then(Value::as_str) == Some("amd")))
        .unwrap_or(false);
    if state == "unsupported_hardware" && amd_hardware {
        summary = String::from(
            "An AMD NPU was found. fLLM drives it with its own kernels (long prompts) or through FastFlowLM: see \
the AMD NPU card below.",
        );
    }

    let vendor_name = |vendor: &str| {
        NPU_VENDOR_NAMES
            .iter()
            .find(|(key, _)| *key == vendor)
            .map(|(_, name)| *name)
            .unwrap_or(NPU_VENDOR_NAMES[3].1)
    };
    let hardware = npu
        .get("hardware")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|h| {
                    let vendor = text(h, "vendor");
                    let supported = h.get("supported").and_then(Value::as_bool).unwrap_or(false);
                    NpuHardwareRow {
                        label: format!("{} ({})", text(h, "name"), vendor_name(&vendor)),
                        badge: if supported {
                            String::from("Intel NPU")
                        } else if vendor == "amd" {
                            String::from("AMD NPU: native kernels or FastFlowLM, see below")
                        } else {
                            String::from("detected, not supported yet")
                        },
                        supported,
                        problem: h.get("problem").and_then(Value::as_str).map(str::to_string),
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let busy = progress.get("busy").and_then(Value::as_bool).unwrap_or(false);
    let phase = progress.get("phase").and_then(Value::as_str).unwrap_or("");
    let retry = phase == "failed" || phase == "cancelled";
    let can_install = !busy && state == "runtime_missing";
    let megabytes = match setup.get("download_bytes").and_then(Value::as_u64) {
        Some(bytes) if bytes > 0 => format!(" (about {} MB)", (bytes as f64 / 1e6).round()),
        _ => String::new(),
    };
    let button = if can_install {
        Some(NpuButton {
            action: "install",
            label: if retry {
                String::from("Try again")
            } else {
                format!("Install the runtime{megabytes}")
            },
        })
    } else {
        None
    };
    let mut error = None;
    if phase == "failed" {
        if let Some(e) = progress.get("error").and_then(Value::as_str) {
            error = Some(e.to_string());
        }
    } else if state != "ready" && runtime.get("error").and_then(Value::as_str).is_some() {
        error = runtime.get("error").and_then(Value::as_str).map(str::to_string);
    } else if let Some(e) = npu.get("last_error").and_then(Value::as_str) {
        error = Some(e.to_string());
    }
    let tone = if state == "ready" {
        "good"
    } else if state == "no_device" || error.is_some() {
        "warn"
    } else {
        "muted"
    };

    let q = npu.get("quantizations").cloned().unwrap_or(Value::Object(Default::default()));
    let list = |key: &str| -> Vec<String> {
        q.get(key)
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
            .unwrap_or_default()
    };
    let titles: [(&str, &str); 4] = [
        ("eight_bit", "Computed as 8 bit with one scale per output row (the default)"),
        ("float16", "Half precision"),
        ("exact", "Exact (same values as on the processor) with FLLM_NPU_WEIGHTS=exact"),
        ("converted_int4", "Kept at 4 bit with FLLM_NPU_WEIGHTS=int4"),
    ];
    let formats = titles
        .iter()
        .map(|(key, title)| NpuFormat { title, items: list(key) })
        .filter(|f| !f.items.is_empty())
        .collect();

    NpuCard {
        summary,
        offload: npu.get("offload_line").and_then(Value::as_str).unwrap_or("").to_string(),
        tone,
        hardware,
        button,
        cancel: busy,
        remove: !busy && runtime.get("managed").and_then(Value::as_bool).unwrap_or(false),
        progress: npu_progress_model(Some(&progress)),
        error,
        formats,
        explanation,
    }
}

/// The name of a graphics device with the trademark marks and spacing removed, to compare the
/// entries of several drivers — `deviceKey` of `npu.js`.
pub fn device_key(d: &Device) -> String {
    fold_marks(&d.name).to_lowercase()
}

/// A device name without the `(R)` and `(TM)` marks and with its spacing folded — the part of
/// `deviceKey` that the label of a running device keeps (its case too).
fn fold_marks(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let chars: Vec<char> = name.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '(' && i + 1 < chars.len() && (chars[i + 1] == 'R' || chars[i + 1] == 'T') {
            let mut j = i + 1;
            let mut close = None;
            while j < chars.len() && j < i + 5 {
                if chars[j] == ')' {
                    close = Some(j);
                    break;
                }
                j += 1;
            }
            if let Some(close) = close {
                let mark: String = chars[i + 1..close].iter().collect();
                if mark == "R" || mark == "TM" {
                    i = close + 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Lower is better: the vendor's own interface first, then the native graphics API, the generic
/// (wgpu) entries last — `deviceRank` of `npu.js`.
pub fn device_rank(d: &Device) -> u8 {
    if d.api == "wgpu" {
        return 3;
    }
    match d.backend.to_lowercase().as_str() {
        "cuda" | "hip" | "metal" | "sycl" => 0,
        _ => 1,
    }
}

/// One entry of the device grid: the entry that stands for the device, plus its members and the
/// interfaces beside it — the `mergeDevices` result.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceGroup {
    pub primary: Device,
    pub members: Vec<DeviceRef>,
    pub others: Vec<(DeviceRef, String)>,
}

/// What one interface of a group is called by (`id`, `backend`, `api`).
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceRef {
    pub id: String,
    pub backend: String,
    pub api: String,
}

/// One entry per physical device: the same graphics chip is listed once per driver interface.
/// Entries with the same name and a similar memory size are one device, two identical cards stay
/// two (an entry joins a group only when the group has no entry of the same backend and interface
/// yet). The processor and NPUs are never merged. `wanted_id` becomes the entry that stands for
/// its group — `mergeDevices` of `npu.js`.
pub fn merge_devices(devices: &[Device], wanted_id: &str) -> Vec<DeviceGroup> {
    let wanted = wanted_id.to_lowercase();
    let mergeable = |d: &Device| d.backend.to_lowercase() != "cpu" && d.backend.to_lowercase() != "npu" && !device_key(d).is_empty();

    struct Group {
        mergeable: bool,
        items: Vec<Device>,
    }
    let mut groups: Vec<Group> = Vec::new();
    for d in devices.iter().filter(|d| !d.id.is_empty()) {
        let key = device_key(d);
        let mut joined = false;
        if mergeable(d) {
            for g in groups.iter_mut().filter(|g| g.mergeable) {
                let first = &g.items[0];
                if device_key(first) != key {
                    continue;
                }
                if g.items.iter().any(|o| o.backend == d.backend && o.api == d.api) {
                    continue;
                }
                let a = first.total_memory_bytes;
                let b = d.total_memory_bytes;
                let similar = a == 0 || b == 0 || (a.min(b) as f64) / (a.max(b) as f64) >= 0.5;
                if similar {
                    g.items.push(d.clone());
                    joined = true;
                    break;
                }
            }
        }
        if !joined {
            groups.push(Group { mergeable: mergeable(d), items: vec![d.clone()] });
        }
    }

    groups
        .into_iter()
        .map(|g| {
            let mut ordered = g.items.clone();
            ordered.sort_by_key(device_rank);
            let primary = ordered
                .iter()
                .find(|d| d.id.to_lowercase() == wanted)
                .cloned()
                .unwrap_or_else(|| ordered[0].clone());
            let members = g
                .items
                .iter()
                .map(|d| DeviceRef { id: d.id.clone(), backend: d.backend.clone(), api: d.api.clone() })
                .collect();
            let others = ordered
                .iter()
                .filter(|d| d.id != primary.id)
                .map(|d| {
                    (
                        DeviceRef { id: d.id.clone(), backend: d.backend.clone(), api: d.api.clone() },
                        device_backend_label(d),
                    )
                })
                .collect();
            DeviceGroup { primary, members, others }
        })
        .collect()
}

/// How a driver interface is named for a person: *Vulkan*, *Vulkan (generic)*, *NVIDIA CUDA* —
/// `deviceBackendLabel` of `npu.js`.
pub fn device_backend_label(d: &Device) -> String {
    let backend = d.backend.to_lowercase();
    if backend == "cpu" {
        return String::from("Processor");
    }
    let name = DEVICE_BACKEND_NAMES
        .iter()
        .find(|(key, _)| *key == backend)
        .map(|(_, label)| *label)
        .unwrap_or(backend.as_str());
    format!("{name}{}", if d.api == "wgpu" { String::from(" (generic)") } else { String::new() })
}

/// One entry of the device choice in the load options: `automatic`, `the processor`, every
/// graphics device that reports memory, and the NPU when it is ready. `current` is kept even when
/// it is unknown here — `deviceChoices` of `npu.js`.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceChoice {
    pub value: String,
    pub label: String,
    pub warning: Option<String>,
    pub alternative: bool,
}

pub fn device_choices(devices: &[Device], npu: Option<&Value>, current: &str) -> Vec<DeviceChoice> {
    let mut out = vec![
        DeviceChoice { value: String::from("auto"), label: String::from("Automatic (recommended)"), warning: None, alternative: false },
        DeviceChoice { value: String::from("cpu"), label: String::from("Processor only"), warning: None, alternative: false },
    ];
    let usable: Vec<Device> = devices
        .iter()
        .filter(|d| d.backend != "cpu" && d.backend != "npu" && d.available && d.total_memory_bytes > 0)
        .cloned()
        .collect();
    let groups = merge_devices(&usable, current);
    for g in &groups {
        if out.iter().any(|o| o.value == g.primary.id) {
            continue;
        }
        out.push(DeviceChoice {
            value: g.primary.id.clone(),
            label: format!("{} ({})", g.primary.name, device_backend_label(&g.primary)),
            warning: None,
            alternative: false,
        });
    }
    for g in &groups {
        for (other, label) in &g.others {
            if other.api == "wgpu" || out.iter().any(|x| x.value == other.id) {
                continue;
            }
            out.push(DeviceChoice {
                value: other.id.clone(),
                label: format!("{} ({label}, other backend)", g.primary.name),
                warning: None,
                alternative: true,
            });
        }
    }
    let npu_ready = npu.is_some_and(|n| n.get("state").and_then(Value::as_str) == Some("ready"));
    if npu_ready {
        let warning = npu
            .and_then(|n| n.get("warning").and_then(Value::as_str))
            .map(str::to_string)
            .unwrap_or_else(|| NPU_WARNING.to_string());
        out.push(DeviceChoice { value: String::from("npu"), label: String::from("NPU (experimental)"), warning: Some(warning), alternative: false });
    }
    let wanted = current.to_lowercase();
    if !wanted.is_empty() && !out.iter().any(|o| o.value == wanted) {
        out.push(DeviceChoice {
            value: wanted.clone(),
            label: format!("{wanted} (current)"),
            warning: wanted.starts_with("npu").then(|| NPU_WARNING.to_string()),
            alternative: false,
        });
    }
    out
}

/// What the footer says about where the model runs: the label beside the model, the sentence
/// under it and whether the server chose — `runningDevice` of `npu.js`. Without a hardware report
/// an automatic choice can only be shown as the processor part.
#[derive(Debug, Clone, PartialEq)]
pub struct RunningDevice {
    pub label: String,
    pub title: String,
    pub auto: bool,
}

pub fn running_device(load_device: &str, devices: &[Device], auto_choice: &str, tier: &str) -> RunningDevice {
    let want = load_device.to_lowercase();
    let cpu = RunningDevice {
        label: format!("CPU · {}", if tier.is_empty() { "simd" } else { tier }),
        title: String::from("The model runs on the processor."),
        auto: want == "auto",
    };
    if want == "cpu" {
        return cpu;
    }
    let backends: [(&str, &str); 7] = [
        ("cuda", "CUDA"),
        ("vulkan", "Vulkan"),
        ("hip", "ROCm"),
        ("metal", "Metal"),
        ("dx12", "DirectX 12"),
        ("sycl", "SYCL"),
        ("npu", "NPU"),
    ];
    let id = if want == "auto" {
        if auto_choice.is_empty() { String::from("cpu") } else { auto_choice.to_lowercase() }
    } else {
        want.clone()
    };
    if id == "cpu" {
        return cpu;
    }
    let dev = devices.iter().find(|d| d.id.to_lowercase() == id);
    let backend = dev.map(|d| d.backend.to_lowercase()).unwrap_or_else(|| id.split(':').next().unwrap_or("").to_string());
    let plain = dev.map(|d| fold_marks(&d.name)).unwrap_or_else(|| id.clone());
    let label = format!(
        "{plain} · {}",
        backends.iter().find(|(key, _)| *key == backend).map(|(_, name)| *name).unwrap_or(backend.as_str())
    );
    let title = if want == "auto" {
        format!(
            "{}: the automatic choice ({id}); layers that do not fit run on the processor.",
            dev.map(|d| d.name.clone()).unwrap_or_else(|| id.clone())
        )
    } else {
        format!("{}: chosen in the load options ({id}).", dev.map(|d| d.name.clone()).unwrap_or_else(|| id.clone()))
    };
    RunningDevice { label, title, auto: want == "auto" }
}

/// Memory use of a device for the bar on its card: whether it is known, how full it is and
/// whether that is ok, a warning or high — `deviceMemory` of `npu.js`. A device without a size has
/// no bar.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceMemory {
    pub known: bool,
    pub pct: u64,
    pub level: &'static str,
}

pub fn device_memory(d: &Device) -> DeviceMemory {
    let total = d.total_memory_bytes;
    if total == 0 {
        return DeviceMemory { known: false, pct: 0, level: "ok" };
    }
    let used = total.saturating_sub(d.free_memory_bytes);
    let pct = ((used as f64 / total as f64) * 100.0).round().clamp(0.0, 100.0) as u64;
    DeviceMemory {
        known: true,
        pct,
        level: if pct >= 90 { "high" } else if pct >= 75 { "warn" } else { "ok" },
    }
}

/// What a device does for the loaded model: `running` (the model runs on it), `auto` (no model is
/// loaded; the automatic choice would use it) or nothing — `deviceRole` of `npu.js`.
pub fn device_role(members: &[String], own_id: &str, load_device: &str, auto_choice: &str, model_loaded: bool) -> &'static str {
    let ids: Vec<String> = if members.is_empty() {
        vec![own_id.to_lowercase()]
    } else {
        members.iter().map(|m| m.to_lowercase()).collect()
    };
    let want = load_device.to_lowercase();
    let target = if want == "auto" {
        if auto_choice.is_empty() { String::from("cpu") } else { auto_choice.to_lowercase() }
    } else {
        want
    };
    if !ids.contains(&target) {
        return "";
    }
    if model_loaded {
        "running"
    } else {
        "auto"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dev(id: &str, name: &str, backend: &str, api: &str, bytes: u64) -> Device {
        Device {
            id: id.into(),
            free_memory_bytes: 0,
            name: name.into(),
            backend: backend.into(),
            api: api.into(),
            dev_type: String::new(),
            total_memory_bytes: bytes,
            available: true,
            has_id: !id.is_empty(),
        }
    }

    /// The progress bar says the phase in the Studio's words and how far it is.
    #[test]
    fn the_progress_says_the_phase_and_how_far_it_is() {
        assert_eq!(npu_progress_model(None), NpuProgress::default(), "nothing without busy");
        let not_busy = json!({"busy": false});
        assert!(!npu_progress_model(Some(&not_busy)).visible, "without busy there is no bar");
        let p = json!({"busy": true, "phase": "downloading", "done_bytes": 500_000_000, "total_bytes": 2_000_000_000});
        let model = npu_progress_model(Some(&p));
        assert!(model.visible);
        assert_eq!(model.percent, 25);
        assert_eq!(model.text, "Downloading (500.0 of 2000.0 MB)");
        let no_total = json!({"busy": true, "phase": "verifying"});
        assert_eq!(npu_progress_model(Some(&no_total)).text, "Checking the download", "no totals, no detail");
        let unknown = json!({"busy": true, "phase": "elsewhere"});
        assert_eq!(npu_progress_model(Some(&unknown)).text, "Working");
        let over = json!({"busy": true, "phase": "unpacking", "done_bytes": 9, "total_bytes": 5});
        assert_eq!(npu_progress_model(Some(&over)).percent, 100, "never more than a hundred");
    }

    /// Every state the API knows has the summary of the Studio; `ready` carries the version and
    /// the device, `runtime_missing` the download it would take.
    #[test]
    fn every_state_has_the_summary_of_the_studio() {
        assert_eq!(npu_card_model(None).summary, "Checking for an NPU ...");
        for (key, summary) in NPU_SUMMARIES {
            let npu = json!({"state": key});
            let card = npu_card_model(Some(&npu));
            if key == "ready" {
                assert!(card.summary.starts_with("Ready (experimental). Choose NPU under Advanced"),
                    "ready without a version keeps the plain text: {}", card.summary);
            } else {
                assert_eq!(card.summary, summary, "state {key}");
            }
        }

        let ready = json!({
            "state": "ready",
            "runtime": {"version": "2025.1.0", "managed": true},
            "devices": [{"full_name": "Intel AI Boost", "architecture": "Arrow Lake"}],
        });
        let card = npu_card_model(Some(&ready));
        assert_eq!(
            card.summary,
            "Ready (experimental), OpenVINO 2025.1.0. Choose NPU under Advanced when you load a model; long prompts \
then run on it. Intel AI Boost (platform Arrow Lake).",
            "the version and the device are appended"
        );
        assert_eq!(card.tone, "good");
        assert!(card.remove, "a managed runtime can be removed when nothing is busy");
        assert!(!card.cancel);

        let missing = json!({"state": "runtime_missing", "setup": {"download_bytes": 140_000_000, "pinned_version": "2025.0.1"}});
        let card = npu_card_model(Some(&missing));
        assert_eq!(
            card.summary,
            "An Intel NPU was found. The runtime that drives it (OpenVINO 2025.0.1 (a download of about 140 MB)) is \
not installed yet."
        );
        assert_eq!(card.button.as_ref().unwrap().label, "Install the runtime (about 140 MB)");
        assert_eq!(card.tone, "muted");

        // an AMD machine under an unsupported state says where to look instead
        let amd = json!({"state": "unsupported_hardware", "hardware": [{"name": "XDNA", "vendor": "amd"}]});
        let card = npu_card_model(Some(&amd));
        assert!(card.summary.starts_with("An AMD NPU was found."), "{}", card.summary);
        assert_eq!(card.tone, "muted");
    }

    /// The hardware rows carry the maker's name, the badge of what is supported and a problem.
    #[test]
    fn the_hardware_rows_carry_the_badge_of_what_is_supported() {
        let npu = json!({
            "state": "no_npu",
            "hardware": [
                {"name": "Intel AI Boost", "vendor": "intel", "supported": true},
                {"name": "Ryzen AI", "vendor": "amd", "supported": false, "problem": "needs FastFlowLM"},
                {"name": "Something", "vendor": "qualcomm", "supported": false},
            ],
        });
        let card = npu_card_model(Some(&npu));
        assert_eq!(card.hardware.len(), 3);
        assert_eq!(card.hardware[0].label, "Intel AI Boost (Intel)");
        assert_eq!(card.hardware[0].badge, "Intel NPU");
        assert_eq!(card.hardware[1].badge, "AMD NPU: native kernels or FastFlowLM, see below");
        assert_eq!(card.hardware[2].badge, "detected, not supported yet");
        assert_eq!(card.hardware[1].problem.as_deref(), Some("needs FastFlowLM"));
        assert_eq!(card.tone, "muted", "no NPU found is muted; only no_device and errors warn");
    }

    /// The four formats appear only when the API names some for them.
    #[test]
    fn the_formats_appear_only_when_there_are_any() {
        let npu = json!({"state": "ready", "quantizations": {"eight_bit": ["q8_0"], "float16": ["f16"]}});
        let card = npu_card_model(Some(&npu));
        assert_eq!(card.formats.len(), 2, "the empty two are filtered out");
        assert_eq!(card.formats[0].title, "Computed as 8 bit with one scale per output row (the default)");
        assert_eq!(card.formats[0].items, vec!["q8_0"]);
        assert_eq!(card.formats[1].title, "Half precision");
        let none = npu_card_model(Some(&json!({"state": "ready"})));
        assert!(none.formats.is_empty());
    }

    /// The device key folds trademark marks and spacing; the rank prefers the vendor's own
    /// interface and puts the generic one last.
    #[test]
    fn the_key_folds_marks_and_the_rank_puts_the_generic_last() {
        assert_eq!(
            device_key(&dev("1", "GeForce RTX™ 4060(R)", "cuda", "cuda", 0)),
            "geforce rtx™ 4060",
            "only the marks in parentheses go, a bare one stays (the JS regex)"
        );
        assert_eq!(device_key(&dev("3", "GeForce RTX (TM) 4060", "cuda", "cuda", 0)), "geforce rtx 4060");
        assert_eq!(device_key(&dev("2", "   spaced   name ", "vulkan", "vulkan", 0)), "spaced name");
        assert_eq!(device_rank(&dev("a", "x", "cuda", "cuda", 0)), 0, "the vendor's own");
        assert_eq!(device_rank(&dev("b", "x", "vulkan", "vulkan", 0)), 1, "the native API");
        assert_eq!(device_rank(&dev("c", "x", "vulkan", "wgpu", 0)), 3, "the generic last");
    }

    /// The same chip through two driver interfaces becomes one entry with the members beside it,
    /// the processor and the NPU never merge, and two identical cards stay two.
    #[test]
    fn the_same_chip_through_two_interfaces_becomes_one_entry() {
        let devices = vec![
            dev("cpu0", "Intel Core i5", "cpu", "cpu", 0),
            dev("cuda:0", "GeForce RTX 4060(R)", "cuda", "cuda", 8_000_000_000),
            dev("vulkan:1", "GeForce RTX 4060", "vulkan", "vulkan", 8_100_000_000),
            dev("vulkan:2", "GeForce RTX 4060", "vulkan", "wgpu", 8_100_000_000),
            dev("npu0", "Intel AI Boost", "npu", "npu", 0),
        ];
        let groups = merge_devices(&devices, "");
        assert_eq!(groups.len(), 3, "the processor, the NPU, and the card with its three interfaces");
        let card = groups.iter().find(|g| g.primary.id == "cuda:0").expect("the vendor's interface stands for the card");
        assert_eq!(card.members.len(), 3, "all three interfaces are members of one chip");
        assert_eq!(card.others.len(), 2, "the two others are listed beside it");
        assert_eq!(card.others[0].1, "Vulkan", "the native API comes first of the others");
        // two cards of the same model with the same interface stay apart (they are one entry each)
        let two = vec![
            dev("cuda:0", "RTX 4090", "cuda", "cuda", 24_000_000_000),
            dev("cuda:1", "RTX 4090", "cuda", "cuda", 24_000_000_000),
        ];
        assert_eq!(merge_devices(&two, "").len(), 2, "an entry joins only without the same backend and interface");
        // memory size decides whether two entries are the same chip
        let sizes = vec![
            dev("a", "Same Card", "cuda", "cuda", 10_000_000_000),
            dev("b", "Same Card", "vulkan", "vulkan", 2_000_000_000),
        ];
        assert_eq!(merge_devices(&sizes, "").len(), 2, "under half the size: another device");
    }

    /// The choice list of the load options: automatic and the processor first, then the cards,
    /// their other interfaces, the NPU when it is ready, and the current one even if unknown.
    #[test]
    fn the_choice_list_starts_with_the_two_defaults() {
        let devices = vec![
            dev("cuda:0", "RTX 4060", "cuda", "cuda", 8_000_000_000),
            dev("vulkan:1", "RTX 4060", "vulkan", "vulkan", 8_000_000_000),
            dev("cpu0", "Intel", "cpu", "cpu", 0),
            dev("ghost", "No Memory GPU", "vulkan", "vulkan", 0),
        ];
        let choices = device_choices(&devices, None, "");
        assert_eq!(choices[0].value, "auto");
        assert_eq!(choices[0].label, "Automatic (recommended)");
        assert_eq!(choices[1].value, "cpu");
        assert_eq!(choices[1].label, "Processor only");
        assert_eq!(choices[2].value, "cuda:0", "the vendor's interface is the entry of the card");
        let other = choices.iter().find(|c| c.alternative).expect("the Vulkan interface too");
        assert_eq!(other.value, "vulkan:1");
        assert!(other.label.contains("other backend"));
        assert!(!choices.iter().any(|c| c.value == "ghost"), "a device without memory is not offered");
        assert!(!choices.iter().any(|c| c.value == "cpu0"), "the processor has its own row");

        let ready = json!({"state": "ready"});
        let with_npu = device_choices(&devices, Some(&ready), "npu0");
        let npu = with_npu.iter().find(|c| c.value == "npu").expect("the NPU when it is ready");
        assert_eq!(npu.label, "NPU (experimental)");
        assert_eq!(npu.warning.as_deref(), Some(NPU_WARNING));
        let current = device_choices(&devices, None, "not-here");
        assert_eq!(current.last().unwrap().value, "not-here", "the current one is kept");
        assert_eq!(current.last().unwrap().label, "not-here (current)");
    }

    /// What the footer says about where the model runs: the processor, the automatic choice and a
    /// chosen card, each with its sentence.
    #[test]
    fn the_footer_says_where_the_model_runs() {
        let devices = vec![dev("cuda:0", "GeForce RTX 4060(R)", "cuda", "cuda", 8_000_000_000)];
        let cpu = running_device("cpu", &devices, "cuda:0", "avx2-fma");
        assert_eq!(cpu.label, "CPU · avx2-fma");
        assert_eq!(cpu.title, "The model runs on the processor.");
        assert!(!cpu.auto);

        let auto = running_device("auto", &devices, "cuda:0", "");
        assert_eq!(auto.label, "GeForce RTX 4060 · CUDA", "the marks are folded out of the label");
        assert_eq!(
            auto.title,
            "GeForce RTX 4060(R): the automatic choice (cuda:0); layers that do not fit run on the processor."
        );
        assert!(auto.auto);

        let chosen = running_device("cuda:0", &devices, "", "");
        assert_eq!(chosen.title, "GeForce RTX 4060(R): chosen in the load options (cuda:0).");
        assert!(!chosen.auto);

        // without a report the automatic choice can only be shown as the processor part
        let unknown = running_device("auto", &[], "", "");
        assert_eq!(unknown.label, "CPU · simd");
        assert!(unknown.auto);
    }

    /// The memory bar of a device: how full it is and whether that is ok, a warning or high.
    #[test]
    fn the_memory_bar_says_how_full_it_is() {
        let gpu = |free: u64| Device { free_memory_bytes: free, ..dev("cuda:0", "x", "cuda", "cuda", 10_000_000_000) };
        assert_eq!(device_memory(&gpu(9_000_000_000)), DeviceMemory { known: true, pct: 10, level: "ok" });
        assert_eq!(device_memory(&gpu(2_000_000_000)).pct, 80);
        assert_eq!(device_memory(&gpu(2_000_000_000)).level, "warn", "three quarters are a warning");
        assert_eq!(device_memory(&gpu(500_000_000)).level, "high", "nine tenths are high");
        let cpu = dev("cpu0", "x", "cpu", "cpu", 0);
        assert!(!device_memory(&cpu).known, "a device without a size has no bar");
    }

    /// What a device does for the loaded model: the running one, the one the automatic choice
    /// would use, and nothing for the rest.
    #[test]
    fn a_device_is_running_the_auto_choice_or_nothing() {
        let members = vec!["cuda:0".to_string(), "vulkan:1".to_string()];
        assert_eq!(device_role(&members, "cuda:0", "auto", "cuda:0", true), "running");
        assert_eq!(device_role(&members, "cuda:0", "auto", "cuda:0", false), "auto");
        assert_eq!(device_role(&members, "cuda:0", "cuda:0", "", true), "running");
        assert_eq!(device_role(&members, "cuda:0", "vulkan:9", "cuda:0", true), "", "another device");
        assert_eq!(device_role(&[], "cpu", "auto", "cpu", true), "running", "no members: its own id");
        assert_eq!(device_role(&[], "cpu0", "auto", "cpu", true), "", "cpu0 is not the choice 'cpu'");
    }
}
