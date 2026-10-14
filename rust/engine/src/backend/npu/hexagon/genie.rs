//! Qualcomm Genie, the LLM runtime on top of QNN, bound at run time (package 11, feature `npu-qnn`).
//!
//! Genie is the only supported way to run a language model on the Hexagon NPU without writing graphs by hand: it takes a JSON dialog
//! configuration (tokenizer, sampler, and the pre-compiled QNN context binaries of the model) and answers prompts. It cannot load GGUF
//! files, so it is not a `Backend` for fLLM; the wrapper below exists to make the study concrete ("Genie as an external generation
//! provider") and to give the test round something to call. API: `Genie/GenieCommon.h` and `Genie/GenieDialog.h` of the QAIRT SDK 2.37
//! (Genie API 1.11). Verification level: L0.

use libloading::Library;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::PathBuf;
use std::ptr;
use std::sync::OnceLock;

/// `Genie_Status_t` (0 is success, negative values are errors, positive values are warnings).
type Status = i32;
/// `GenieDialogConfig_Handle_t` / `GenieDialog_Handle_t` (opaque const pointers).
type ConfigHandle = *const c_void;
type DialogHandle = *const c_void;

/// `GENIE_DIALOG_SENTENCE_COMPLETE`: the query string is a complete sentence.
const SENTENCE_COMPLETE: i32 = 0;

type VersionFn = unsafe extern "C" fn() -> u32;
type ConfigCreateFromJsonFn =
    unsafe extern "C" fn(json: *const c_char, config: *mut ConfigHandle) -> Status;
type ConfigFreeFn = unsafe extern "C" fn(config: ConfigHandle) -> Status;
type DialogCreateFn =
    unsafe extern "C" fn(config: ConfigHandle, dialog: *mut DialogHandle) -> Status;
type QueryCallback =
    unsafe extern "C" fn(response: *const c_char, sentence_code: i32, user_data: *const c_void);
type DialogQueryFn = unsafe extern "C" fn(
    dialog: DialogHandle,
    query: *const c_char,
    sentence_code: i32,
    callback: QueryCallback,
    user_data: *const c_void,
) -> Status;
type DialogFreeFn = unsafe extern "C" fn(dialog: DialogHandle) -> Status;

/// A loaded Genie library.
pub struct Genie {
    _lib: Library,
    pub path: String,
    /// `Genie_getApiMajorVersion/Minor/Patch`.
    pub api_version: (u32, u32, u32),
    config_create_from_json: ConfigCreateFromJsonFn,
    config_free: ConfigFreeFn,
    dialog_create: DialogCreateFn,
    dialog_query: DialogQueryFn,
    dialog_free: DialogFreeFn,
}

fn library_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("FLLM_GENIE_LIB") {
        out.push(PathBuf::from(p));
    }
    if let Some(root) =
        std::env::var_os("QNN_SDK_ROOT").or_else(|| std::env::var_os("QAIRT_SDK_ROOT"))
    {
        let lib = PathBuf::from(root).join("lib");
        if cfg!(windows) {
            for target in ["arm64x-windows-msvc", "aarch64-windows-msvc"] {
                out.push(lib.join(target).join("Genie.dll"));
            }
        } else {
            for target in [
                "aarch64-ubuntu-gcc9.4",
                "aarch64-oe-linux-gcc11.2",
                "aarch64-android",
            ] {
                out.push(lib.join(target).join("libGenie.so"));
            }
        }
    }
    out.push(PathBuf::from(if cfg!(windows) {
        "Genie.dll"
    } else {
        "libGenie.so"
    }));
    out
}

/// Resolves `name` (a NUL-terminated byte string) as a function pointer of type `T`.
///
/// # Safety
/// `T` must be the exact signature of the exported function.
unsafe fn symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, String> {
    lib.get::<T>(name).map(|s| *s).map_err(|e| {
        format!(
            "missing symbol {}: {e}",
            String::from_utf8_lossy(&name[..name.len().saturating_sub(1)])
        )
    })
}

fn bind(lib: Library, path: String) -> Result<Genie, String> {
    // SAFETY: the symbol names and signatures are those of GenieCommon.h / GenieDialog.h (API 1.11).
    unsafe {
        let major: VersionFn = symbol(&lib, b"Genie_getApiMajorVersion\0")?;
        let minor: VersionFn = symbol(&lib, b"Genie_getApiMinorVersion\0")?;
        let patch: VersionFn = symbol(&lib, b"Genie_getApiPatchVersion\0")?;
        let api_version = (major(), minor(), patch());
        Ok(Genie {
            config_create_from_json: symbol(&lib, b"GenieDialogConfig_createFromJson\0")?,
            config_free: symbol(&lib, b"GenieDialogConfig_free\0")?,
            dialog_create: symbol(&lib, b"GenieDialog_create\0")?,
            dialog_query: symbol(&lib, b"GenieDialog_query\0")?,
            dialog_free: symbol(&lib, b"GenieDialog_free\0")?,
            _lib: lib,
            path,
            api_version,
        })
    }
}

static GENIE: OnceLock<Result<Genie, String>> = OnceLock::new();

/// The process-wide Genie library, if present.
pub fn genie() -> Result<&'static Genie, String> {
    GENIE
        .get_or_init(|| {
            let mut errors = Vec::new();
            for candidate in library_candidates() {
                let shown = candidate.display().to_string();
                // SAFETY: loading a shared library runs its initializers.
                match unsafe { crate::backend::npu::xdna::dl::open(&candidate) } {
                    Ok(lib) => match bind(lib, shown.clone()) {
                        Ok(g) => return Ok(g),
                        Err(e) => errors.push(format!("{shown}: {e}")),
                    },
                    Err(e) => errors.push(format!("{shown}: {e}")),
                }
            }
            Err(format!(
                "Genie library not found ({}); set FLLM_GENIE_LIB or QNN_SDK_ROOT",
                errors.join("; ")
            ))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Collects the streamed answer: `user_data` is a `*mut String`.
unsafe extern "C" fn collect(
    response: *const c_char,
    _sentence_code: i32,
    user_data: *const c_void,
) {
    if response.is_null() || user_data.is_null() {
        return;
    }
    // SAFETY: `user_data` is the `&mut String` passed by `Genie::query`, alive for the whole `GenieDialog_query` call; `response` is a
    // NUL-terminated string valid for the duration of the callback.
    let out = &mut *(user_data as *mut String);
    out.push_str(&CStr::from_ptr(response).to_string_lossy());
}

impl Genie {
    /// Creates a dialog from a Genie JSON configuration (see the "Genie" chapter of the QAIRT documentation: tokenizer path, sampler,
    /// engine with the QnnHtp backend and the model's context binaries), sends one prompt and returns the complete answer.
    pub fn query(&self, config_json: &str, prompt: &str) -> Result<String, String> {
        let json = CString::new(config_json).map_err(|e| format!("config contains NUL: {e}"))?;
        let prompt = CString::new(prompt).map_err(|e| format!("prompt contains NUL: {e}"))?;
        let mut config: ConfigHandle = ptr::null();
        let mut dialog: DialogHandle = ptr::null();
        let mut answer = String::new();
        // SAFETY: handles are created and freed in this function; `answer` outlives the query call that writes to it through `collect`.
        unsafe {
            let status = (self.config_create_from_json)(json.as_ptr(), &mut config);
            if status != 0 {
                return Err(format!(
                    "GenieDialogConfig_createFromJson failed (status {status})"
                ));
            }
            let status = (self.dialog_create)(config, &mut dialog);
            let _ = (self.config_free)(config);
            if status != 0 {
                return Err(format!("GenieDialog_create failed (status {status})"));
            }
            let status = (self.dialog_query)(
                dialog,
                prompt.as_ptr(),
                SENTENCE_COMPLETE,
                collect,
                &mut answer as *mut String as *const c_void,
            );
            let _ = (self.dialog_free)(dialog);
            if status < 0 {
                return Err(format!("GenieDialog_query failed (status {status})"));
            }
        }
        Ok(answer)
    }
}
