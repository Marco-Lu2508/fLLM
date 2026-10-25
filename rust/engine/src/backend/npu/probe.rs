//! Finding NPUs on this machine without any vendor runtime.
//!
//! The OpenVINO device query only works once the OpenVINO runtime is installed, but the app wants to say "an NPU was found, the runtime is
//! missing" before that. So the operating system is asked directly:
//!
//! * Linux: the accelerator class of the kernel (`/sys/class/accel/accelN`, device node `/dev/accel/accelN`); the kernel driver behind it
//!   names the vendor (`intel_vpu`, `amdxdna`);
//! * Windows: the present devices of the device-manager classes an NPU registers in (`ComputeAccelerator`, `System`), asked through
//!   PowerShell (`Get-PnpDevice`), and recognised by name (`Intel(R) AI Boost`, `AMD IPU Device`, `Qualcomm(R) Hexagon(TM) NPU`).
//!
//! Only the presence of a device is learned here, never whether a backend can use it. Verification level: L0 (the parsing and
//! classification are unit tested; the operating system calls have not been run on a machine that has an NPU).

use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vendor {
    Intel,
    Amd,
    Qualcomm,
    Other,
}

impl Vendor {
    pub fn as_str(self) -> &'static str {
        match self {
            Vendor::Intel => "intel",
            Vendor::Amd => "amd",
            Vendor::Qualcomm => "qualcomm",
            Vendor::Other => "other",
        }
    }
}

/// One NPU found in the operating system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Detected {
    pub vendor: Vendor,
    pub name: String,
    /// Where it was found (`/dev/accel/accel0`, `device manager`).
    pub source: String,
    /// A problem that keeps this user from the device (for example missing access rights), in plain words.
    pub problem: Option<String>,
}

/// Vendor behind a Linux kernel driver name.
pub fn vendor_from_driver(driver: &str) -> Option<Vendor> {
    match driver.trim() {
        "intel_vpu" | "ivpu" => Some(Vendor::Intel),
        "amdxdna" => Some(Vendor::Amd),
        _ => None,
    }
}

/// Vendor behind a PCI vendor id as sysfs prints it (`0x8086`).
pub fn vendor_from_pci_id(id: &str) -> Option<Vendor> {
    match id.trim().to_ascii_lowercase().trim_start_matches("0x") {
        "8086" => Some(Vendor::Intel),
        "1022" | "1002" => Some(Vendor::Amd),
        "17cb" => Some(Vendor::Qualcomm),
        _ => None,
    }
}

/// Vendor of a device-manager entry, `None` when the name is no NPU. Matches the names the vendors' drivers use.
pub fn vendor_from_device_name(name: &str) -> Option<Vendor> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("ai boost") || (lower.contains("intel") && lower.contains("npu")) {
        Some(Vendor::Intel)
    } else if lower.contains("ipu device") || lower.contains("xdna") || (lower.contains("amd") && lower.contains("npu")) {
        Some(Vendor::Amd)
    } else if lower.contains("hexagon") || (lower.contains("qualcomm") && lower.contains("npu")) {
        Some(Vendor::Qualcomm)
    } else if lower.contains("neural processing unit") || lower.split(|c: char| !c.is_alphanumeric()).any(|w| w == "npu") {
        Some(Vendor::Other)
    } else {
        None
    }
}

/// The device names in the output of the Windows query: one JSON string, a JSON list of strings, or plain lines.
pub fn parse_device_names(output: &str) -> Vec<String> {
    let text = output.trim();
    if text.is_empty() {
        return Vec::new();
    }
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(serde_json::Value::String(one)) => vec![one],
        Ok(serde_json::Value::Array(list)) => list.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect(),
        _ => text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned).collect(),
    }
}

/// Turns device-manager names into detections (names that are no NPU are dropped; the same name twice counts once per entry).
pub fn from_device_names(names: &[String]) -> Vec<Detected> {
    names
        .iter()
        .filter_map(|name| {
            vendor_from_device_name(name).map(|vendor| Detected {
                vendor,
                name: name.trim().to_owned(),
                source: "device manager".to_owned(),
                problem: None,
            })
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn scan() -> Vec<Detected> {
    use std::path::Path;
    let mut found = Vec::new();
    let mut nodes: Vec<String> = std::fs::read_dir("/sys/class/accel")
        .map(|entries| entries.filter_map(Result::ok).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    if nodes.is_empty() {
        // a kernel without the class directory but with the nodes
        nodes = std::fs::read_dir("/dev/accel")
            .map(|entries| entries.filter_map(Result::ok).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
            .unwrap_or_default();
    }
    nodes.retain(|n| n.starts_with("accel"));
    nodes.sort();
    for node in nodes {
        let device = Path::new("/sys/class/accel").join(&node).join("device");
        let driver = std::fs::read_link(device.join("driver"))
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_default();
        let vendor = vendor_from_driver(&driver)
            .or_else(|| std::fs::read_to_string(device.join("vendor")).ok().and_then(|v| vendor_from_pci_id(&v)))
            .unwrap_or(Vendor::Other);
        let path = format!("/dev/accel/{node}");
        let problem = match std::fs::OpenOptions::new().read(true).write(true).open(&path) {
            Ok(_) => None,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                Some(format!("no permission for {path}; add your user to the group that owns it (usually render) and log in again"))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Some(format!("{path} does not exist although the kernel lists the device")),
            Err(_) => None,
        };
        let label = match vendor {
            Vendor::Intel => "Intel NPU",
            Vendor::Amd => "AMD XDNA NPU",
            Vendor::Qualcomm => "Qualcomm NPU",
            Vendor::Other => "Accelerator",
        };
        found.push(Detected { vendor, name: format!("{label} ({node})"), source: path, problem });
    }
    found
}

#[cfg(windows)]
fn scan() -> Vec<Detected> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let script = "Get-PnpDevice -PresentOnly -Class ComputeAccelerator,System -ErrorAction SilentlyContinue | \
                  Where-Object { $_.FriendlyName -match 'NPU|AI Boost|IPU|Hexagon|Neural|XDNA' } | \
                  Select-Object -ExpandProperty FriendlyName | ConvertTo-Json -Compress";
    let output = std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output();
    match output {
        Ok(out) if out.status.success() => from_device_names(&parse_device_names(&String::from_utf8_lossy(&out.stdout))),
        _ => Vec::new(),
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
fn scan() -> Vec<Detected> {
    Vec::new()
}

/// The NPUs the operating system lists. The answer is kept for a few seconds (the Windows query starts a program).
pub fn detect() -> Vec<Detected> {
    static CACHE: Mutex<Option<(Instant, Vec<Detected>)>> = Mutex::new(None);
    let mut slot = CACHE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((at, found)) = slot.as_ref() {
        if at.elapsed() < Duration::from_secs(15) {
            return found.clone();
        }
    }
    let found = scan();
    *slot = Some((Instant::now(), found.clone()));
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_drivers_and_pci_ids_name_the_vendor() {
        assert_eq!(vendor_from_driver("intel_vpu\n"), Some(Vendor::Intel));
        assert_eq!(vendor_from_driver("amdxdna"), Some(Vendor::Amd));
        assert_eq!(vendor_from_driver("nvidia"), None);
        assert_eq!(vendor_from_pci_id("0x8086\n"), Some(Vendor::Intel));
        assert_eq!(vendor_from_pci_id("0x17CB"), Some(Vendor::Qualcomm));
        assert_eq!(vendor_from_pci_id("0x10de"), None);
    }

    #[test]
    fn device_manager_names_are_classified() {
        assert_eq!(vendor_from_device_name("Intel(R) AI Boost"), Some(Vendor::Intel));
        assert_eq!(vendor_from_device_name("AMD IPU Device"), Some(Vendor::Amd));
        assert_eq!(vendor_from_device_name("NPU Compute Accelerator Device"), Some(Vendor::Other));
        assert_eq!(vendor_from_device_name("Qualcomm(R) Hexagon(TM) NPU"), Some(Vendor::Qualcomm));
        assert_eq!(vendor_from_device_name("Intel(R) UHD Graphics"), None);
        assert_eq!(vendor_from_device_name("PCI Express Root Complex"), None);
        // a word that merely contains the letters
        assert_eq!(vendor_from_device_name("Input Output Unit"), None);
    }

    #[test]
    fn query_output_comes_as_string_list_or_lines() {
        assert_eq!(parse_device_names(""), Vec::<String>::new());
        assert_eq!(parse_device_names("\"Intel(R) AI Boost\"\r\n"), ["Intel(R) AI Boost"]);
        assert_eq!(parse_device_names("[\"A\",\"B\"]"), ["A", "B"]);
        assert_eq!(parse_device_names("A\nB\n"), ["A", "B"]);
        let found = from_device_names(&parse_device_names("[\"Intel(R) AI Boost\",\"Some Other Thing\"]"));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].vendor, Vendor::Intel);
        assert_eq!(found[0].source, "device manager");
    }
}
