//! Run-time binding of the QNN HTP backend library (package 11, feature `npu-qnn`).
//!
//! The Qualcomm AI Engine Direct SDK ships the backend as a shared library (`libQnnHtp.so` on Linux and Android, `QnnHtp.dll` on Windows on
//! Snapdragon) with exactly one exported entry point, `QnnInterface_getProviders`, which returns the versioned function table. Nothing is
//! linked; the SDK is optional at run time. Verification level: L0 (never run on a Snapdragon device).

use super::qnn::{
    self, ApiVersion, ErrorHandle, GetProvidersFn, Handle, Interface, InterfaceTable,
    HTP_DEVICE_TYPE_ON_CHIP, QNN_SUCCESS,
};
use libloading::Library;
use std::ffi::{c_char, CStr};
use std::path::PathBuf;
use std::ptr;
use std::sync::OnceLock;

/// An HTP (Hexagon Tensor Processor) found through `QnnDevice_getPlatformInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HtpDevice {
    pub device_id: u32,
    /// `QNN_SOC_MODEL_*` value of the chip.
    pub soc_model: u32,
    /// `QNN_HTP_DEVICE_ARCH_*` (68, 69, 73, 75, 79, 81) or 0 if unknown.
    pub arch: u32,
    /// Size of the vector TCM in bytes.
    pub vtcm_bytes: usize,
}

/// A loaded QNN backend library with the interface table of the chosen provider.
pub struct Qnn {
    /// Keeps the library mapped while the function pointers in `table` are in use.
    _lib: Library,
    pub path: String,
    pub provider: String,
    pub api_version: ApiVersion,
    pub table: InterfaceTable,
}

/// Library paths worth trying, most specific first.
fn library_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("FLLM_QNN_HTP_LIB") {
        out.push(PathBuf::from(p));
    }
    if let Some(root) =
        std::env::var_os("QNN_SDK_ROOT").or_else(|| std::env::var_os("QAIRT_SDK_ROOT"))
    {
        let lib = PathBuf::from(root).join("lib");
        if cfg!(windows) {
            for target in ["arm64x-windows-msvc", "aarch64-windows-msvc"] {
                out.push(lib.join(target).join("QnnHtp.dll"));
            }
        } else {
            for target in [
                "aarch64-ubuntu-gcc9.4",
                "aarch64-oe-linux-gcc11.2",
                "aarch64-oe-linux-gcc9.3",
                "aarch64-android",
            ] {
                out.push(lib.join(target).join("libQnnHtp.so"));
            }
        }
    }
    out.push(PathBuf::from(if cfg!(windows) {
        "QnnHtp.dll"
    } else {
        "libQnnHtp.so"
    }));
    out
}

fn bind(lib: Library, path: String) -> Result<Qnn, String> {
    // SAFETY: `QnnInterface_getProviders` has the signature of `GetProvidersFn` (QnnInterface.h); the provider list and the interfaces it
    // points to are static storage of the library, which stays loaded as long as `Qnn` lives. The table is copied only after the provider's
    // API version proved that all declared members exist.
    unsafe {
        let get_providers: GetProvidersFn = *lib
            .get::<GetProvidersFn>(b"QnnInterface_getProviders\0")
            .map_err(|e| format!("{path}: no QnnInterface_getProviders: {e}"))?;
        let mut list: *const *const Interface = ptr::null();
        let mut count: u32 = 0;
        let error = get_providers(&mut list, &mut count);
        if error != QNN_SUCCESS || list.is_null() {
            return Err(format!(
                "{path}: QnnInterface_getProviders failed (error code {})",
                qnn::error_code(error)
            ));
        }
        let mut seen = Vec::new();
        for i in 0..count as usize {
            let provider = *list.add(i);
            if provider.is_null() {
                continue;
            }
            let version = (*provider).api_version;
            let name = if (*provider).provider_name.is_null() {
                String::new()
            } else {
                CStr::from_ptr((*provider).provider_name)
                    .to_string_lossy()
                    .into_owned()
            };
            if version.is_supported() {
                return Ok(Qnn {
                    _lib: lib,
                    path,
                    provider: name,
                    api_version: version,
                    table: (*provider).table,
                });
            }
            seen.push(format!(
                "{name} core API {}.{}.{}",
                version.core.major, version.core.minor, version.core.patch
            ));
        }
        Err(format!(
            "{path}: no provider with core API {}.>={} (found: {})",
            qnn::API_MAJOR,
            qnn::API_MIN_MINOR,
            if seen.is_empty() {
                "none".to_owned()
            } else {
                seen.join(", ")
            }
        ))
    }
}

static QNN: OnceLock<Result<Qnn, String>> = OnceLock::new();

/// The process-wide QNN HTP library (loaded on first use). `Err` lists every candidate that failed and why.
pub fn qnn() -> Result<&'static Qnn, String> {
    QNN.get_or_init(|| {
        let mut errors = Vec::new();
        for candidate in library_candidates() {
            let shown = candidate.display().to_string();
            // SAFETY: loading a shared library runs its initializers; the QNN backends' are benign.
            match unsafe { crate::backend::npu::xdna::dl::open(&candidate) } {
                Ok(lib) => match bind(lib, shown.clone()) {
                    Ok(q) => return Ok(q),
                    Err(e) => errors.push(e),
                },
                Err(e) => errors.push(format!("{shown}: {e}")),
            }
        }
        Err(format!(
            "QNN HTP library not found ({}); set FLLM_QNN_HTP_LIB or QNN_SDK_ROOT",
            errors.join("; ")
        ))
    })
    .as_ref()
    .map_err(Clone::clone)
}

impl Qnn {
    /// `Ok` for `QNN_SUCCESS`, otherwise the error code and, if the backend provides it, its message.
    pub fn check(&self, error: ErrorHandle, what: &str) -> Result<(), String> {
        if error == QNN_SUCCESS {
            return Ok(());
        }
        let message = self.table.error_get_message.and_then(|get| {
            let mut text: *const c_char = ptr::null();
            // SAFETY: `get` is `QnnError_getMessage`; the returned string is owned by the backend and valid for the call.
            unsafe {
                if get(error, &mut text) == QNN_SUCCESS && !text.is_null() {
                    Some(CStr::from_ptr(text).to_string_lossy().into_owned())
                } else {
                    None
                }
            }
        });
        Err(match message {
            Some(m) => format!("{what}: QNN error {} ({m})", qnn::error_code(error)),
            None => format!("{what}: QNN error {}", qnn::error_code(error)),
        })
    }

    /// The HTP devices of this machine according to `QnnDevice_getPlatformInfo` (empty on a host without an HTP, for example the x86 SDK
    /// libraries on a development PC).
    pub fn devices(&self) -> Result<Vec<HtpDevice>, String> {
        let create = self.table.backend_create.ok_or("backendCreate missing")?;
        let free = self.table.backend_free.ok_or("backendFree missing")?;
        let get_info = self
            .table
            .device_get_platform_info
            .ok_or("deviceGetPlatformInfo missing")?;
        let free_info = self
            .table
            .device_free_platform_info
            .ok_or("deviceFreePlatformInfo missing")?;

        let mut backend: Handle = ptr::null_mut();
        // SAFETY: a null logger and a null configuration are allowed; the backend handle is released below on every path.
        unsafe {
            self.check(
                create(ptr::null_mut(), ptr::null(), &mut backend),
                "backendCreate",
            )?;
            let mut info: *const qnn::PlatformInfo = ptr::null();
            let result = (|| {
                self.check(
                    get_info(ptr::null_mut(), &mut info),
                    "deviceGetPlatformInfo",
                )?;
                let mut devices = Vec::new();
                if !info.is_null() {
                    let v1 = (*info).v1;
                    for i in 0..v1.num_hw_devices as usize {
                        let device = &*v1.hw_devices.add(i);
                        let extension = device.v1.device_info_extension;
                        if extension.is_null()
                            || (*extension).device_type != HTP_DEVICE_TYPE_ON_CHIP
                        {
                            continue;
                        }
                        let chip = (*extension).on_chip;
                        devices.push(HtpDevice {
                            device_id: device.v1.device_id,
                            soc_model: chip.soc_model,
                            arch: chip.arch,
                            vtcm_bytes: chip.vtcm_size,
                        });
                    }
                }
                Ok(devices)
            })();
            if !info.is_null() {
                free_info(ptr::null_mut(), info);
            }
            let _ = free(backend);
            result
        }
    }
}
