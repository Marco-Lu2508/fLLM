//! ONNX Runtime C API, bound at run time (package 11, AMD XDNA study; feature `npu-xdna`).
//!
//! Ryzen AI Software runs models through ONNX Runtime with the VitisAI execution provider (VAI EP). The Ryzen AI build of
//! `onnxruntime` is a drop-in `OrtApi` provider, so nothing is linked: the library is opened with `libloading`, `OrtGetApiBase` is
//! resolved and the `OrtApi` function table is read slot by slot. The slot numbers below are the positions of the members in
//! `struct OrtApi` of `onnxruntime_c_api.h` (ONNX Runtime v1.22.0, ORT_API_VERSION 22); the struct is append-only, so a slot never moves,
//! and a table obtained for an older API version simply stops earlier (checked through `api_version` before a slot is read).
//! Only what the single-MatMul probe needs is bound. Verification level: L0 (never run against a real Ryzen AI installation).

use super::{onnx, testdata};
use libloading::Library;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::PathBuf;
use std::ptr;
use std::sync::OnceLock;
use std::time::Instant;

/// `OrtStatus*` (null means success).
type OrtStatus = *mut c_void;

/// Positions of the used members in `struct OrtApi` (v1.22.0 header), see the module comment.
mod slot {
    pub const GET_ERROR_MESSAGE: usize = 2;
    pub const CREATE_ENV: usize = 3;
    pub const CREATE_SESSION_FROM_ARRAY: usize = 8;
    pub const RUN: usize = 9;
    pub const CREATE_SESSION_OPTIONS: usize = 10;
    pub const GET_TENSOR_MUTABLE_DATA: usize = 51;
    pub const CREATE_CPU_MEMORY_INFO: usize = 69;
    pub const RELEASE_ENV: usize = 92;
    pub const RELEASE_STATUS: usize = 93;
    pub const RELEASE_MEMORY_INFO: usize = 94;
    pub const RELEASE_SESSION: usize = 95;
    pub const RELEASE_VALUE: usize = 96;
    pub const RELEASE_SESSION_OPTIONS: usize = 100;
    pub const GET_AVAILABLE_PROVIDERS: usize = 125;
    pub const RELEASE_AVAILABLE_PROVIDERS: usize = 126;
    pub const CREATE_TENSOR_WITH_DATA: usize = 49;
    /// Generic `SessionOptionsAppendExecutionProvider(options, name, keys, values, n)` (since ORT 1.12).
    pub const APPEND_EXECUTION_PROVIDER: usize = 216;
    /// `SessionOptionsAppendExecutionProvider_VitisAI(options, keys, values, n)` (since ORT 1.18, API version 18).
    pub const APPEND_EXECUTION_PROVIDER_VITISAI: usize = 276;
}

/// The API version in which the dedicated VitisAI append function appeared.
const API_VERSION_VITISAI: u32 = 18;
/// API versions requested from the library, newest first (the first one it knows is used).
const API_VERSIONS: std::ops::RangeInclusive<u32> = 17..=22;

/// `OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING`.
const LOGGING_LEVEL_WARNING: c_int = 2;
/// `OrtAllocatorType::OrtArenaAllocator`.
const ARENA_ALLOCATOR: c_int = 1;
/// `OrtMemType::OrtMemTypeDefault`.
const MEM_TYPE_DEFAULT: c_int = 0;
/// `ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT`.
const ELEMENT_TYPE_FLOAT: c_int = 1;

/// Name of the execution provider as ONNX Runtime lists it.
pub const VITISAI_PROVIDER: &str = "VitisAIExecutionProvider";

#[repr(C)]
struct OrtApiBase {
    get_api: unsafe extern "C" fn(version: u32) -> *const *const c_void,
    get_version_string: unsafe extern "C" fn() -> *const c_char,
}

type FnGetBase = unsafe extern "C" fn() -> *const OrtApiBase;
type FnGetErrorMessage = unsafe extern "C" fn(status: OrtStatus) -> *const c_char;
type FnRelease = unsafe extern "C" fn(object: *mut c_void);
type FnCreateEnv =
    unsafe extern "C" fn(level: c_int, log_id: *const c_char, out: *mut *mut c_void) -> OrtStatus;
type FnCreateSessionOptions = unsafe extern "C" fn(out: *mut *mut c_void) -> OrtStatus;
type FnAppendProvider = unsafe extern "C" fn(
    options: *mut c_void,
    name: *const c_char,
    keys: *const *const c_char,
    values: *const *const c_char,
    num_keys: usize,
) -> OrtStatus;
type FnAppendVitisAi = unsafe extern "C" fn(
    options: *mut c_void,
    keys: *const *const c_char,
    values: *const *const c_char,
    num_keys: usize,
) -> OrtStatus;
type FnCreateSessionFromArray = unsafe extern "C" fn(
    env: *mut c_void,
    data: *const c_void,
    len: usize,
    options: *mut c_void,
    out: *mut *mut c_void,
) -> OrtStatus;
type FnRun = unsafe extern "C" fn(
    session: *mut c_void,
    run_options: *const c_void,
    input_names: *const *const c_char,
    inputs: *const *const c_void,
    input_len: usize,
    output_names: *const *const c_char,
    output_len: usize,
    outputs: *mut *mut c_void,
) -> OrtStatus;
type FnCreateCpuMemoryInfo =
    unsafe extern "C" fn(alloc_type: c_int, mem_type: c_int, out: *mut *mut c_void) -> OrtStatus;
type FnCreateTensorWithData = unsafe extern "C" fn(
    info: *const c_void,
    data: *mut c_void,
    data_len: usize,
    shape: *const i64,
    shape_len: usize,
    element_type: c_int,
    out: *mut *mut c_void,
) -> OrtStatus;
type FnGetTensorMutableData =
    unsafe extern "C" fn(value: *mut c_void, out: *mut *mut c_void) -> OrtStatus;
type FnGetAvailableProviders =
    unsafe extern "C" fn(out: *mut *mut *mut c_char, len: *mut c_int) -> OrtStatus;
type FnReleaseAvailableProviders =
    unsafe extern "C" fn(providers: *mut *mut c_char, len: c_int) -> OrtStatus;

/// The resolved function table.
#[derive(Clone, Copy)]
struct Api {
    get_error_message: FnGetErrorMessage,
    release_status: FnRelease,
    create_env: FnCreateEnv,
    release_env: FnRelease,
    create_session_options: FnCreateSessionOptions,
    release_session_options: FnRelease,
    append_provider: FnAppendProvider,
    append_vitisai: Option<FnAppendVitisAi>,
    create_session_from_array: FnCreateSessionFromArray,
    release_session: FnRelease,
    run: FnRun,
    create_cpu_memory_info: FnCreateCpuMemoryInfo,
    release_memory_info: FnRelease,
    create_tensor_with_data: FnCreateTensorWithData,
    get_tensor_mutable_data: FnGetTensorMutableData,
    release_value: FnRelease,
    get_available_providers: FnGetAvailableProviders,
    release_available_providers: FnReleaseAvailableProviders,
}

/// Reads slot `index` of the table as a function pointer of type `T`.
///
/// # Safety
/// `table` must point to an `OrtApi` of at least `index + 1` members and `T` must be the exact function pointer type of that member.
unsafe fn slot_fn<T: Copy>(table: *const *const c_void, index: usize) -> Result<T, String> {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        std::mem::size_of::<*const c_void>()
    );
    let raw = *table.add(index);
    if raw.is_null() {
        return Err(format!("OrtApi slot {index} is null"));
    }
    Ok(std::mem::transmute_copy::<*const c_void, T>(&raw))
}

/// A loaded ONNX Runtime.
pub struct Runtime {
    /// Keeps the library mapped while the function pointers in `api` are in use.
    _lib: Library,
    api: Api,
    /// The `ORT_API_VERSION` the table was obtained for.
    pub api_version: u32,
    /// `OrtApiBase::GetVersionString` (for example "1.22.0").
    pub version: String,
    /// Path (or name) the library was loaded from.
    pub path: String,
    /// `GetAvailableProviders`.
    pub providers: Vec<String>,
}

/// Library paths worth trying, most specific first.
fn library_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("FLLM_XDNA_ORT_LIB") {
        out.push(PathBuf::from(p));
    }
    if let Some(root) = std::env::var_os("RYZEN_AI_INSTALLATION_PATH") {
        let root = PathBuf::from(root);
        if cfg!(windows) {
            out.push(root.join("deployment").join("onnxruntime.dll"));
            out.push(root.join("onnxruntime").join("bin").join("onnxruntime.dll"));
        } else {
            out.push(
                root.join("onnxruntime")
                    .join("lib")
                    .join("libonnxruntime.so"),
            );
            out.push(root.join("deployment").join("libonnxruntime.so"));
        }
    }
    out.push(PathBuf::from(if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }));
    out
}

fn bind(lib: Library, path: String) -> Result<Runtime, String> {
    // SAFETY: the symbol name and signature are those of `OrtGetApiBase` in onnxruntime_c_api.h; the table slots are read according to
    // the `slot` module and only after the API version proved them present.
    unsafe {
        let get_base: FnGetBase = *lib
            .get::<FnGetBase>(b"OrtGetApiBase\0")
            .map_err(|e| format!("{path}: no OrtGetApiBase: {e}"))?;
        let base = get_base();
        if base.is_null() {
            return Err(format!("{path}: OrtGetApiBase returned null"));
        }
        let (table, api_version) = API_VERSIONS
            .rev()
            .find_map(|v| {
                let t = ((*base).get_api)(v);
                (!t.is_null()).then_some((t, v))
            })
            .ok_or_else(|| format!("{path}: no OrtApi for versions 17..=22 (library too old)"))?;
        let version_ptr = ((*base).get_version_string)();
        let version = if version_ptr.is_null() {
            "?".to_owned()
        } else {
            CStr::from_ptr(version_ptr).to_string_lossy().into_owned()
        };

        let api = Api {
            get_error_message: slot_fn(table, slot::GET_ERROR_MESSAGE)?,
            release_status: slot_fn(table, slot::RELEASE_STATUS)?,
            create_env: slot_fn(table, slot::CREATE_ENV)?,
            release_env: slot_fn(table, slot::RELEASE_ENV)?,
            create_session_options: slot_fn(table, slot::CREATE_SESSION_OPTIONS)?,
            release_session_options: slot_fn(table, slot::RELEASE_SESSION_OPTIONS)?,
            append_provider: slot_fn(table, slot::APPEND_EXECUTION_PROVIDER)?,
            append_vitisai: if api_version >= API_VERSION_VITISAI {
                Some(slot_fn(table, slot::APPEND_EXECUTION_PROVIDER_VITISAI)?)
            } else {
                None
            },
            create_session_from_array: slot_fn(table, slot::CREATE_SESSION_FROM_ARRAY)?,
            release_session: slot_fn(table, slot::RELEASE_SESSION)?,
            run: slot_fn(table, slot::RUN)?,
            create_cpu_memory_info: slot_fn(table, slot::CREATE_CPU_MEMORY_INFO)?,
            release_memory_info: slot_fn(table, slot::RELEASE_MEMORY_INFO)?,
            create_tensor_with_data: slot_fn(table, slot::CREATE_TENSOR_WITH_DATA)?,
            get_tensor_mutable_data: slot_fn(table, slot::GET_TENSOR_MUTABLE_DATA)?,
            release_value: slot_fn(table, slot::RELEASE_VALUE)?,
            get_available_providers: slot_fn(table, slot::GET_AVAILABLE_PROVIDERS)?,
            release_available_providers: slot_fn(table, slot::RELEASE_AVAILABLE_PROVIDERS)?,
        };
        let mut rt = Runtime {
            _lib: lib,
            api,
            api_version,
            version,
            path,
            providers: Vec::new(),
        };
        rt.providers = rt.query_providers()?;
        Ok(rt)
    }
}

static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();

/// The process-wide ONNX Runtime (loaded on first use). `Err` carries every candidate that failed and why.
pub fn runtime() -> Result<&'static Runtime, String> {
    RUNTIME
        .get_or_init(|| {
            let mut errors = Vec::new();
            for candidate in library_candidates() {
                let shown = candidate.display().to_string();
                // SAFETY: loading a shared library runs its initializers; ONNX Runtime's are benign.
                match unsafe { crate::backend::npu::xdna::dl::open(&candidate) } {
                    Ok(lib) => match bind(lib, shown.clone()) {
                        Ok(rt) => return Ok(rt),
                        Err(e) => errors.push(e),
                    },
                    Err(e) => errors.push(format!("{shown}: {e}")),
                }
            }
            Err(format!(
                "ONNX Runtime not found ({}); set FLLM_XDNA_ORT_LIB or RYZEN_AI_INSTALLATION_PATH",
                errors.join("; ")
            ))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Owns one ONNX Runtime object and releases it with the matching `Release*` function.
struct Owned {
    ptr: *mut c_void,
    release: FnRelease,
}

impl Owned {
    fn new(ptr: *mut c_void, release: FnRelease) -> Self {
        Self { ptr, release }
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` was created by the runtime and is released exactly once.
            unsafe { (self.release)(self.ptr) };
        }
    }
}

/// Result of [`Runtime::probe_matmul`].
#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub ort_version: String,
    pub api_version: u32,
    pub providers: Vec<String>,
    /// Time of `CreateSessionFromArray` (includes the NPU compilation on a cold cache).
    pub session_ms: f64,
    /// Time of one `Run`.
    pub run_ms: f64,
    /// Largest absolute difference to the f32 CPU reference.
    pub max_abs_err: f32,
}

/// Provider options for the VitisAI EP taken from the environment: `FLLM_XDNA_CACHE_DIR` (`cache_dir`), `FLLM_XDNA_TARGET`
/// (`target`: X1 for Phoenix/Hawk Point, X2 for Strix/Krackan), `FLLM_XDNA_CONFIG` (`config_file`, BF16 compilation options).
fn provider_options_from_env(cache_key: &str) -> Vec<(String, String)> {
    let mut options = Vec::new();
    if let Ok(dir) = std::env::var("FLLM_XDNA_CACHE_DIR") {
        options.push(("cache_dir".to_owned(), dir));
        options.push(("cache_key".to_owned(), cache_key.to_owned()));
        options.push(("enable_cache_file_io_in_mem".to_owned(), "0".to_owned()));
    }
    if let Ok(target) = std::env::var("FLLM_XDNA_TARGET") {
        options.push(("target".to_owned(), target));
    }
    if let Ok(config) = std::env::var("FLLM_XDNA_CONFIG") {
        options.push(("config_file".to_owned(), config));
    }
    options
}

impl Runtime {
    /// Whether ONNX Runtime lists the VitisAI execution provider (the Ryzen AI build does; a stock build does not).
    pub fn has_vitisai(&self) -> bool {
        self.providers.iter().any(|p| p == VITISAI_PROVIDER)
    }

    fn check(&self, status: OrtStatus, what: &str) -> Result<(), String> {
        if status.is_null() {
            return Ok(());
        }
        // SAFETY: `status` is a live OrtStatus; the message pointer is valid until the status is released.
        unsafe {
            let msg = (self.api.get_error_message)(status);
            let text = if msg.is_null() {
                "unknown error".to_owned()
            } else {
                CStr::from_ptr(msg).to_string_lossy().into_owned()
            };
            (self.api.release_status)(status);
            Err(format!("{what}: {text}"))
        }
    }

    fn query_providers(&self) -> Result<Vec<String>, String> {
        let get: FnGetAvailableProviders = self.api.get_available_providers;
        let mut list: *mut *mut c_char = ptr::null_mut();
        let mut len: c_int = 0;
        // SAFETY: out-pointers are valid; each returned string is NUL-terminated and valid until ReleaseAvailableProviders.
        unsafe {
            self.check(get(&mut list, &mut len), "GetAvailableProviders")?;
            let mut out = Vec::new();
            for i in 0..usize::try_from(len).unwrap_or(0) {
                out.push(CStr::from_ptr(*list.add(i)).to_string_lossy().into_owned());
            }
            self.check(
                (self.api.release_available_providers)(list, len),
                "ReleaseAvailableProviders",
            )?;
            Ok(out)
        }
    }

    /// Builds a one-node `MatMul` model (`x[m,k] . w[k,n]`, float32, random data), compiles and runs it through the VitisAI EP and compares the
    /// result with a CPU reference. This is the L1 check for XDNA: it proves that the provider loads and accepts a linear layer, not
    /// that the layer ran on the NPU (ONNX Runtime partitions the graph; set `XLNX_ONNX_EP_REPORT_FILE` to see the placement).
    pub fn probe_matmul(&self, m: usize, k: usize, n: usize) -> Result<ProbeReport, String> {
        if !self.has_vitisai() {
            return Err(format!(
                "this ONNX Runtime ({}) does not list {VITISAI_PROVIDER}",
                self.path
            ));
        }
        let mut x = testdata::pseudo_random(m * k, 0x9e37_79b9_7f4a_7c15);
        let w = testdata::pseudo_random(k * n, 0xd1b5_4a32_d192_ed03);
        let expected = testdata::matmul_reference(&x, &w, m, k, n);
        let model = onnx::matmul_model_f32(m, k, n, &w)?;
        let options = provider_options_from_env(&format!("fllm_probe_{m}x{k}x{n}"));
        let api = &self.api;

        // SAFETY: every pointer handed to the runtime lives until after the call; objects are released by `Owned` in reverse order.
        unsafe {
            let mut env = ptr::null_mut();
            self.check(
                (api.create_env)(LOGGING_LEVEL_WARNING, c"fllm".as_ptr(), &mut env),
                "CreateEnv",
            )?;
            let env = Owned::new(env, api.release_env);
            let mut session_options = ptr::null_mut();
            self.check(
                (api.create_session_options)(&mut session_options),
                "CreateSessionOptions",
            )?;
            let session_options = Owned::new(session_options, api.release_session_options);

            let keys: Vec<CString> = options
                .iter()
                .map(|(k, _)| CString::new(k.as_str()).map_err(|e| e.to_string()))
                .collect::<Result<_, _>>()?;
            let values: Vec<CString> = options
                .iter()
                .map(|(_, v)| CString::new(v.as_str()).map_err(|e| e.to_string()))
                .collect::<Result<_, _>>()?;
            let key_ptrs: Vec<*const c_char> = keys.iter().map(|s| s.as_ptr()).collect();
            let value_ptrs: Vec<*const c_char> = values.iter().map(|s| s.as_ptr()).collect();
            match api.append_vitisai {
                Some(append) => self.check(
                    append(
                        session_options.ptr,
                        key_ptrs.as_ptr(),
                        value_ptrs.as_ptr(),
                        key_ptrs.len(),
                    ),
                    "AppendExecutionProvider_VitisAI",
                )?,
                None => self.check(
                    (api.append_provider)(
                        session_options.ptr,
                        c"VitisAI".as_ptr(),
                        key_ptrs.as_ptr(),
                        value_ptrs.as_ptr(),
                        key_ptrs.len(),
                    ),
                    "AppendExecutionProvider(VitisAI)",
                )?,
            }

            let started = Instant::now();
            let mut session = ptr::null_mut();
            self.check(
                (api.create_session_from_array)(
                    env.ptr,
                    model.as_ptr().cast(),
                    model.len(),
                    session_options.ptr,
                    &mut session,
                ),
                "CreateSessionFromArray",
            )?;
            let session = Owned::new(session, api.release_session);
            let session_ms = started.elapsed().as_secs_f64() * 1e3;

            let mut memory_info = ptr::null_mut();
            self.check(
                (api.create_cpu_memory_info)(ARENA_ALLOCATOR, MEM_TYPE_DEFAULT, &mut memory_info),
                "CreateCpuMemoryInfo",
            )?;
            let memory_info = Owned::new(memory_info, api.release_memory_info);
            let shape = [m as i64, k as i64];
            let mut input = ptr::null_mut();
            self.check(
                (api.create_tensor_with_data)(
                    memory_info.ptr,
                    x.as_mut_ptr().cast(),
                    x.len() * std::mem::size_of::<f32>(),
                    shape.as_ptr(),
                    shape.len(),
                    ELEMENT_TYPE_FLOAT,
                    &mut input,
                ),
                "CreateTensorWithDataAsOrtValue",
            )?;
            let input = Owned::new(input, api.release_value);

            let input_names = [c"x".as_ptr()];
            let output_names = [c"y".as_ptr()];
            let inputs = [input.ptr as *const c_void];
            let mut output: *mut c_void = ptr::null_mut();
            let started = Instant::now();
            self.check(
                (api.run)(
                    session.ptr,
                    ptr::null(),
                    input_names.as_ptr(),
                    inputs.as_ptr(),
                    1,
                    output_names.as_ptr(),
                    1,
                    &mut output,
                ),
                "Run",
            )?;
            let run_ms = started.elapsed().as_secs_f64() * 1e3;
            let output = Owned::new(output, api.release_value);
            let mut data: *mut c_void = ptr::null_mut();
            self.check(
                (api.get_tensor_mutable_data)(output.ptr, &mut data),
                "GetTensorMutableData",
            )?;
            let result = std::slice::from_raw_parts(data as *const f32, m * n);
            let max_abs_err = result
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            Ok(ProbeReport {
                ort_version: self.version.clone(),
                api_version: self.api_version,
                providers: self.providers.clone(),
                session_ms,
                run_ms,
                max_abs_err,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_numbers_follow_the_documented_order() {
        // Anchors that are documented in the OrtApi comments: CreateEnv follows GetErrorMessage directly, the ReleaseXxx block starts at
        // ReleaseEnv and keeps its internal order (Env, Status, MemoryInfo, Session, Value).
        assert_eq!(slot::CREATE_ENV, slot::GET_ERROR_MESSAGE + 1);
        assert_eq!(slot::RELEASE_STATUS, slot::RELEASE_ENV + 1);
        assert_eq!(slot::RELEASE_MEMORY_INFO, slot::RELEASE_ENV + 2);
        assert_eq!(slot::RELEASE_SESSION, slot::RELEASE_ENV + 3);
        assert_eq!(slot::RELEASE_VALUE, slot::RELEASE_ENV + 4);
        assert_eq!(
            slot::RELEASE_AVAILABLE_PROVIDERS,
            slot::GET_AVAILABLE_PROVIDERS + 1
        );
    }

    #[test]
    fn candidates_end_with_the_plain_library_name() {
        let c = library_candidates();
        assert!(!c.is_empty());
        let last = c.last().unwrap().to_string_lossy().to_string();
        assert!(last.contains("onnxruntime"));
    }
}
