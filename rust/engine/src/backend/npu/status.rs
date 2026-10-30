//! What the app can say about NPUs: which ones the operating system lists, whether the OpenVINO runtime is installed and sees an NPU,
//! the state of the runtime setup and the reason the last NPU load fell back to the CPU.
//!
//! Used by `GET /api/studio/npu`, by `GET /api/studio/hardware`, by `fllm-rs devices` and by the load options of the Studio.
//! Verification level: L0 (the decision logic is unit tested; nothing here has run on a machine with an NPU).

use super::probe::{self, Detected, Vendor};
use super::setup;
use serde_json::{json, Value};
use std::sync::Mutex;

/// Shown wherever the NPU can be chosen.
pub const WARNING: &str = "Experimental: slower than the processor for many models; weights are held twice; dense models only.";

/// Everything the state is decided from.
#[derive(Clone, Copy, Debug, Default)]
pub struct Facts {
    /// The binary has the OpenVINO NPU backend.
    pub compiled: bool,
    /// A runtime download is pinned for this system.
    pub setup_available: bool,
    /// The operating system lists an Intel NPU.
    pub intel_found: bool,
    /// The operating system lists an NPU of another maker (AMD, Qualcomm, unknown).
    pub other_found: bool,
    /// The OpenVINO runtime is on the disk.
    pub runtime_present: bool,
    /// The runtime loads and lists an NPU device.
    pub npu_listed: bool,
    /// A setup is running.
    pub installing: bool,
}

/// The state shown in the app: `not_in_build`, `installing`, `ready`, `no_device` (runtime there, no NPU listed), `runtime_missing`,
/// `setup_unavailable` (Intel NPU, but no download for this system), `unsupported_hardware` (only AMD / Qualcomm / unknown NPUs) or `no_npu`.
pub fn state_of(f: &Facts) -> &'static str {
    if !f.compiled {
        "not_in_build"
    } else if f.installing {
        "installing"
    } else if f.runtime_present && f.npu_listed {
        "ready"
    } else if f.runtime_present {
        "no_device"
    } else if f.intel_found && f.setup_available {
        "runtime_missing"
    } else if f.intel_found {
        "setup_unavailable"
    } else if f.other_found {
        "unsupported_hardware"
    } else {
        "no_npu"
    }
}

/// One sentence for `state`, for the command line.
pub fn state_text(state: &str) -> &'static str {
    match state {
        "not_in_build" => "this build has no Intel NPU backend",
        "installing" => "the OpenVINO runtime is being installed",
        "ready" => "ready (experimental)",
        "no_device" => "the OpenVINO runtime is installed but lists no NPU (driver or access rights?)",
        "runtime_missing" => "NPU found, runtime missing (install it in Settings, Hardware, or with `fllm-rs npu-setup`)",
        "setup_unavailable" => "NPU found, but there is no runtime download for this system (install OpenVINO yourself; see docs/backends.md)",
        "unsupported_hardware" => "detected, not supported yet",
        _ => "no NPU found",
    }
}

/// Turns the raw error of a failed NPU backend creation into a sentence for the user.
pub fn explain_failure(raw: &str) -> String {
    let lower = raw.to_ascii_lowercase();
    let cause = if (lower.contains("openvino c runtime") && lower.contains("not found")) || lower.contains("runtime missing") {
        "The OpenVINO runtime is not installed or cannot be loaded. Install it in Settings, Hardware.".to_owned()
    } else if lower.contains("no npu device") || lower.contains("no usable npu") {
        "No NPU is available: the driver is missing, the NPU is not listed by OpenVINO, or this user may not open /dev/accel.".to_owned()
    } else if lower.contains("xdna") || lower.contains("hexagon") || lower.contains("qnn") {
        "This kind of NPU is detected but not supported yet.".to_owned()
    } else {
        format!("The NPU could not take this model ({}).", raw.trim().trim_end_matches('.'))
    };
    format!("{cause} The model runs on the processor instead.")
}

static LAST_FALLBACK: Mutex<Option<String>> = Mutex::new(None);

/// Remembers (and returns) the readable reason that the NPU was not used, for the status page and the log.
pub fn record_fallback(raw: &str) -> String {
    let message = explain_failure(raw);
    *LAST_FALLBACK.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(message.clone());
    message
}

/// The reason of the last fallback in this process.
pub fn last_fallback() -> Option<String> {
    LAST_FALLBACK.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
}

/// What the OpenVINO runtime reports when it is loaded.
#[derive(Default)]
struct RuntimeView {
    present: bool,
    path: Option<String>,
    version: Option<String>,
    error: Option<String>,
    devices: Vec<Value>,
}

#[cfg(feature = "npu-openvino")]
fn runtime_view() -> RuntimeView {
    use super::openvino::{devices, ffi, session};
    let file = ffi::find_runtime_file();
    let mut view = RuntimeView { present: file.is_some(), path: file.as_ref().map(|p| p.display().to_string()), ..RuntimeView::default() };
    match session::Runtime::shared() {
        Ok(runtime) => {
            view.present = true;
            view.path = Some(runtime.api().path.display().to_string());
            view.version = runtime.api().version_string();
            view.devices = devices::enumerate()
                .iter()
                .map(|d| json!({"name": d.name, "full_name": d.full_name, "architecture": d.architecture, "driver": d.driver, "total_memory_bytes": d.total_memory}))
                .collect();
        }
        Err(error) if view.present => view.error = Some(error),
        Err(_) => {}
    }
    view
}

#[cfg(not(feature = "npu-openvino"))]
fn runtime_view() -> RuntimeView {
    RuntimeView::default()
}

fn facts_of(detected: &[Detected], view: &RuntimeView) -> Facts {
    Facts {
        compiled: cfg!(feature = "npu-openvino"),
        setup_available: setup::host_artifact().is_some(),
        intel_found: detected.iter().any(|d| d.vendor == Vendor::Intel),
        other_found: detected.iter().any(|d| d.vendor != Vendor::Intel),
        runtime_present: view.present,
        npu_listed: !view.devices.is_empty(),
        installing: setup::progress().phase.busy(),
    }
}

/// The state of the NPU support on this machine (asks the operating system and loads the runtime, so call it off the request thread).
pub fn state() -> &'static str {
    let detected = probe::detect();
    state_of(&facts_of(&detected, &runtime_view()))
}

/// Whether `--device npu` can work right now (a runtime that lists an NPU).
pub fn usable() -> bool {
    state() == "ready"
}

/// A detected NPU that the inventory of usable devices does not list, with the sentence that says why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Notice {
    pub name: String,
    pub vendor: &'static str,
    pub status: &'static str,
    pub text: String,
}

/// The NPUs that are found but cannot be used (runtime missing, not supported yet).
pub fn notices() -> Vec<Notice> {
    let detected = probe::detect();
    let view = runtime_view();
    let state = state_of(&facts_of(&detected, &view));
    detected
        .iter()
        .filter_map(|d| {
            let (status, text) = if d.vendor == Vendor::Intel {
                match state {
                    "ready" | "not_in_build" | "installing" => return None,
                    s => (s, state_text(s).to_owned()),
                }
            } else if d.vendor == Vendor::Amd && cfg!(feature = "npu-xdna-native") {
                // an AMD NPU is usable natively when the shim and XRT are there (it is then in the device list); else say what is missing
                #[cfg(feature = "npu-xdna-native")]
                match super::xdna::native::available() {
                    Ok(_) => return None,
                    Err(reason) => (
                        "native_unavailable",
                        format!("native fLLM kernels are not available ({reason}; docs/npu-xdna.md); FastFlowLM can still run models on it (docs/user/amd-npu.md)"),
                    ),
                }
                #[cfg(not(feature = "npu-xdna-native"))]
                ("unsupported_hardware", state_text("unsupported_hardware").to_owned())
            } else {
                ("unsupported_hardware", state_text("unsupported_hardware").to_owned())
            };
            Some(Notice { name: d.name.clone(), vendor: d.vendor.as_str(), status, text })
        })
        .collect()
}

/// Weight formats and how the NPU backend takes them (docs/backends.md, section 2; OPENVINO.md, "Format conversion").
pub fn quantizations() -> Value {
    json!({
        "exact": ["Q4_0", "Q8_0"],
        "converted_int4": ["Q4_1", "Q2_K", "Q3_K", "Q4_K", "IQ4_NL", "MXFP4", "NVFP4", "AWQ/GPTQ 4-bit"],
        "converted_int8": ["Q5_0", "Q5_1", "Q5_K", "Q6_K"],
        "float16": ["F16", "BF16", "F32"],
    })
}

/// `GET /api/studio/npu`.
pub fn to_json() -> Value {
    let detected = probe::detect();
    let view = runtime_view();
    let facts = facts_of(&detected, &view);
    let state = state_of(&facts);
    let marker = setup::installed();
    let version = marker.as_ref().map(|i| i.version.clone()).filter(|v| v != "unknown").or_else(|| view.version.clone());
    let artifact = setup::host_artifact();
    json!({
        "state": state,
        "state_text": state_text(state),
        "in_build": facts.compiled,
        "usable": state == "ready",
        "device": "npu",
        "warning": WARNING,
        "hardware": detected.iter().map(|d| json!({
            "vendor": d.vendor.as_str(),
            "name": d.name,
            "source": d.source,
            "problem": d.problem,
            "supported": d.vendor == Vendor::Intel,
        })).collect::<Vec<_>>(),
        "runtime": {
            "installed": view.present,
            "version": version,
            "path": view.path,
            "managed": marker.is_some(),
            "error": view.error,
        },
        "setup": {
            "available": artifact.is_some(),
            "pinned_version": setup::OPENVINO_VERSION,
            "download_bytes": artifact.map(|a| a.size),
            "source": setup::PIN_SOURCE,
            "progress": setup::progress().to_json(),
        },
        "devices": view.devices,
        "last_error": last_fallback(),
        "quantizations": quantizations(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> Facts {
        Facts { compiled: true, setup_available: true, ..Facts::default() }
    }

    #[test]
    fn the_state_follows_the_facts() {
        assert_eq!(state_of(&Facts::default()), "not_in_build");
        assert_eq!(state_of(&facts()), "no_npu");
        assert_eq!(state_of(&Facts { intel_found: true, ..facts() }), "runtime_missing");
        assert_eq!(state_of(&Facts { intel_found: true, setup_available: false, ..facts() }), "setup_unavailable");
        assert_eq!(state_of(&Facts { other_found: true, ..facts() }), "unsupported_hardware");
        assert_eq!(state_of(&Facts { intel_found: true, runtime_present: true, ..facts() }), "no_device");
        assert_eq!(state_of(&Facts { intel_found: true, runtime_present: true, npu_listed: true, ..facts() }), "ready");
        // OpenVINO listing an NPU is enough, whatever the operating system query found
        assert_eq!(state_of(&Facts { runtime_present: true, npu_listed: true, ..facts() }), "ready");
        assert_eq!(state_of(&Facts { intel_found: true, installing: true, ..facts() }), "installing");
        // an AMD NPU alone never becomes usable, even with a runtime present
        assert_eq!(state_of(&Facts { other_found: true, runtime_present: false, ..facts() }), "unsupported_hardware");
    }

    #[test]
    fn every_state_has_a_sentence() {
        for s in ["not_in_build", "installing", "ready", "no_device", "runtime_missing", "setup_unavailable", "unsupported_hardware", "no_npu"] {
            assert!(!state_text(s).is_empty());
        }
        assert!(state_text("runtime_missing").contains("runtime missing"));
        assert!(state_text("unsupported_hardware").contains("not supported yet"));
    }

    #[test]
    fn failures_are_explained_in_plain_words() {
        let missing = explain_failure("OpenVINO C runtime (libopenvino_c.so / openvino_c.dll, OpenVINO 2024.0 or newer) not found or unusable; set FLLM_OPENVINO_C_LIB");
        assert!(missing.contains("not installed") && missing.ends_with("processor instead."));
        assert!(explain_failure("no NPU device 0 (providers in this build: openvino)").contains("No NPU is available"));
        assert!(explain_failure("npu-xdna: no compute path").contains("not supported yet"));
        let other = explain_failure("compile failed: bad shape.");
        assert!(other.contains("compile failed: bad shape)") && !other.contains(".)"));
    }

    #[test]
    fn the_fallback_reason_is_kept() {
        let message = record_fallback("no NPU device 0");
        assert_eq!(last_fallback(), Some(message));
    }

    #[test]
    fn quantization_lists_match_the_documented_conversion() {
        let q = quantizations();
        assert_eq!(q["exact"].as_array().unwrap().len(), 2);
        assert!(q["converted_int4"].as_array().unwrap().iter().any(|v| v == "Q4_K"));
    }
}
