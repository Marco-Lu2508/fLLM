//! Run-time binding of the OpenVINO C API (`openvino/c/openvino.h`).
//!
//! There is no link-time dependency: `libopenvino_c.so` / `openvino_c.dll` is opened with `dlopen` / `LoadLibrary` the first time an
//! OpenVINO feature is used, so a binary built with `--features npu-openvino` still starts on machines without OpenVINO (the NPU
//! backend then reports that the runtime is missing). Only the signatures of the public C header were re-declared here; no
//! OpenVINO source is copied (the header and the documentation are Apache-2.0, see THIRD_PARTY.md).
//!
//! Checked against `src/bindings/c/include/openvino/c/*.h` of OpenVINO master (2026.x) and of the tags 2023.3, 2024.0, 2025.0, 2025.4;
//! the symbols below were also looked up with `nm -D` in `libopenvino_c.so.2641` (pip wheel `openvino==2026.4.1`).
//! Verification level: L0 (compiled; the library has never been loaded by this code on an NPU machine).
//!
//! Version differences handled here:
//! * `ov_element_type_e` was renumbered in 2025.2 (`DYNAMIC` became 0, everything after it moved down by one). The numbers are never
//!   hard-coded: [`ElementTypes`] derives `F16` / `F32` at run time from the byte size of a probe tensor.
//! * `ov_core_read_model_from_memory_buffer` exists since 2024.0 (`ov_core_read_model_from_memory` before, removed in 2025.x); both
//!   and the file based `ov_core_read_model` are bound, whatever exists is used.
//! * `ov_core_compile_model` is variadic (`property_args_size` counts the key and value arguments, so 2 per property). The `_props`
//!   variants only exist in the newest headers and are not used.

use libloading::Library;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

/// `ov_status_e`; `0` is `OK`.
pub type Status = c_int;
pub const STATUS_OK: Status = 0;

macro_rules! opaque {
    ($($name:ident),*) => {$(
        #[repr(C)]
        pub struct $name {
            _private: [u8; 0],
        }
    )*};
}
opaque!(OvCore, OvModel, OvCompiledModel, OvInferRequest, OvTensor);

/// `ov_shape_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OvShape {
    pub rank: i64,
    pub dims: *mut i64,
}

/// `ov_available_devices_t`.
#[repr(C)]
pub struct OvAvailableDevices {
    pub devices: *mut *mut c_char,
    pub size: usize,
}

/// `ov_version_t`.
#[repr(C)]
pub struct OvVersion {
    pub build_number: *const c_char,
    pub description: *const c_char,
}

type FnCoreCreate = unsafe extern "C" fn(*mut *mut OvCore) -> Status;
type FnCoreFree = unsafe extern "C" fn(*mut OvCore);
type FnReadMemoryBuffer = unsafe extern "C" fn(
    *const OvCore,
    *const c_char,
    usize,
    *const OvTensor,
    *mut *mut OvModel,
) -> Status;
type FnReadMemoryOld = unsafe extern "C" fn(
    *const OvCore,
    *const c_char,
    *const OvTensor,
    *mut *mut OvModel,
) -> Status;
type FnReadFile =
    unsafe extern "C" fn(*const OvCore, *const c_char, *const c_char, *mut *mut OvModel) -> Status;
type FnCompile = unsafe extern "C" fn(
    *const OvCore,
    *const OvModel,
    *const c_char,
    usize,
    *mut *mut OvCompiledModel,
    ...
) -> Status;
type FnGetProperty =
    unsafe extern "C" fn(*const OvCore, *const c_char, *const c_char, *mut *mut c_char) -> Status;
type FnAvailableDevices = unsafe extern "C" fn(*const OvCore, *mut OvAvailableDevices) -> Status;
type FnAvailableDevicesFree = unsafe extern "C" fn(*mut OvAvailableDevices);
type FnModelFree = unsafe extern "C" fn(*mut OvModel);
type FnCompiledFree = unsafe extern "C" fn(*mut OvCompiledModel);
type FnCreateRequest =
    unsafe extern "C" fn(*const OvCompiledModel, *mut *mut OvInferRequest) -> Status;
type FnRequestFree = unsafe extern "C" fn(*mut OvInferRequest);
type FnInfer = unsafe extern "C" fn(*mut OvInferRequest) -> Status;
type FnRequestTensorByIndex =
    unsafe extern "C" fn(*const OvInferRequest, usize, *mut *mut OvTensor) -> Status;
type FnTensorCreate = unsafe extern "C" fn(c_int, OvShape, *mut *mut OvTensor) -> Status;
type FnTensorData = unsafe extern "C" fn(*const OvTensor, *mut *mut c_void) -> Status;
type FnTensorByteSize = unsafe extern "C" fn(*const OvTensor, *mut usize) -> Status;
type FnTensorFree = unsafe extern "C" fn(*mut OvTensor);
type FnShapeCreate = unsafe extern "C" fn(i64, *const i64, *mut OvShape) -> Status;
type FnShapeFree = unsafe extern "C" fn(*mut OvShape) -> Status;
type FnFree = unsafe extern "C" fn(*const c_char);
type FnLastError = unsafe extern "C" fn() -> *const c_char;
type FnErrorInfo = unsafe extern "C" fn(Status) -> *const c_char;
type FnVersion = unsafe extern "C" fn(*mut OvVersion) -> Status;
type FnVersionFree = unsafe extern "C" fn(*mut OvVersion);

/// `ov_element_type_e` values of the loaded runtime for the types this crate creates tensors of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementTypes {
    pub f16: c_int,
    pub f32: c_int,
}

/// The loaded OpenVINO C runtime. All function pointers stay valid as long as `_lib` lives; the `Api` is never dropped (it sits in a
/// process-wide `OnceLock`), because OpenVINO plugins must not be unloaded while worker threads may still run.
pub struct Api {
    _lib: Library,
    /// Path the runtime was loaded from (for messages).
    pub path: PathBuf,
    pub core_create: FnCoreCreate,
    pub core_free: FnCoreFree,
    read_memory_buffer: Option<FnReadMemoryBuffer>,
    read_memory_old: Option<FnReadMemoryOld>,
    pub read_file: FnReadFile,
    pub compile: FnCompile,
    pub get_property: FnGetProperty,
    pub available_devices: FnAvailableDevices,
    pub available_devices_free: FnAvailableDevicesFree,
    pub model_free: FnModelFree,
    pub compiled_free: FnCompiledFree,
    pub create_request: FnCreateRequest,
    pub request_free: FnRequestFree,
    pub infer: FnInfer,
    pub request_input: FnRequestTensorByIndex,
    pub request_output: FnRequestTensorByIndex,
    pub tensor_create: FnTensorCreate,
    pub tensor_data: FnTensorData,
    pub tensor_byte_size: FnTensorByteSize,
    pub tensor_free: FnTensorFree,
    pub shape_create: FnShapeCreate,
    pub shape_free: FnShapeFree,
    pub free_string: FnFree,
    last_error: Option<FnLastError>,
    error_info: Option<FnErrorInfo>,
    version: Option<FnVersion>,
    version_free: Option<FnVersionFree>,
    element_types: OnceLock<Result<ElementTypes, String>>,
}

// SAFETY: the struct holds plain function pointers and the library handle; the C API functions used here are thread safe for distinct
// objects, and the objects shared between threads are protected by the wrappers in `session.rs`.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

/// The loaded runtime. A failed load is *not* remembered: the runtime can be installed while the process runs (Settings > Hardware), and
/// the next call must then find it.
static API: std::sync::Mutex<Option<Arc<Api>>> = std::sync::Mutex::new(None);

/// Copy the function pointer of a mandatory symbol.
macro_rules! required {
    ($lib:expr, $name:literal) => {
        // SAFETY: the symbol is looked up by its C name and the type alias above is its declared signature.
        *unsafe { $lib.get(concat!($name, "\0").as_bytes()) }.map_err(|e| format!("symbol {} missing: {e}", $name))?
    };
}

/// Copy the function pointer of an optional symbol.
macro_rules! optional {
    ($lib:expr, $name:literal) => {
        // SAFETY: as for `required!`.
        unsafe { $lib.get(concat!($name, "\0").as_bytes()) }
            .ok()
            .map(|s| *s)
    };
}

impl Api {
    /// The process-wide runtime; the first call searches for the library and binds the symbols.
    pub fn get() -> Result<Arc<Api>, String> {
        let mut slot = API.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(api) = slot.as_ref() {
            return Ok(api.clone());
        }
        let api = Api::load()?;
        *slot = Some(api.clone());
        Ok(api)
    }

    fn load() -> Result<Arc<Api>, String> {
        let candidates = library_candidates();
        let mut failures = Vec::new();
        for candidate in &candidates {
            match open_library(candidate).and_then(|lib| Api::bind(lib, candidate)) {
                Ok(api) => return Ok(Arc::new(api)),
                Err(error) => failures.push(format!("{}: {error}", candidate.display())),
            }
        }
        Err(format!(
            "OpenVINO C runtime (libopenvino_c.so / openvino_c.dll, OpenVINO 2024.0 or newer) not found or unusable; set \
             FLLM_OPENVINO_C_LIB to the library or FLLM_OPENVINO_DIR / INTEL_OPENVINO_DIR to the installation. Tried: {}",
            failures.join("; ")
        ))
    }

    fn bind(lib: Library, path: &Path) -> Result<Api, String> {
        let api = Api {
            core_create: required!(lib, "ov_core_create"),
            core_free: required!(lib, "ov_core_free"),
            read_memory_buffer: optional!(lib, "ov_core_read_model_from_memory_buffer"),
            read_memory_old: optional!(lib, "ov_core_read_model_from_memory"),
            read_file: required!(lib, "ov_core_read_model"),
            compile: required!(lib, "ov_core_compile_model"),
            get_property: required!(lib, "ov_core_get_property"),
            available_devices: required!(lib, "ov_core_get_available_devices"),
            available_devices_free: required!(lib, "ov_available_devices_free"),
            model_free: required!(lib, "ov_model_free"),
            compiled_free: required!(lib, "ov_compiled_model_free"),
            create_request: required!(lib, "ov_compiled_model_create_infer_request"),
            request_free: required!(lib, "ov_infer_request_free"),
            infer: required!(lib, "ov_infer_request_infer"),
            request_input: required!(lib, "ov_infer_request_get_input_tensor_by_index"),
            request_output: required!(lib, "ov_infer_request_get_output_tensor_by_index"),
            tensor_create: required!(lib, "ov_tensor_create"),
            tensor_data: required!(lib, "ov_tensor_data"),
            tensor_byte_size: required!(lib, "ov_tensor_get_byte_size"),
            tensor_free: required!(lib, "ov_tensor_free"),
            shape_create: required!(lib, "ov_shape_create"),
            shape_free: required!(lib, "ov_shape_free"),
            free_string: required!(lib, "ov_free"),
            last_error: optional!(lib, "ov_get_last_err_msg"),
            error_info: optional!(lib, "ov_get_error_info"),
            version: optional!(lib, "ov_get_openvino_version"),
            version_free: optional!(lib, "ov_version_free"),
            path: path.to_path_buf(),
            element_types: OnceLock::new(),
            _lib: lib,
        };
        if api.read_memory_buffer.is_none() && api.read_memory_old.is_none() {
            // not fatal: the model is then handed over through temporary files
            crate::backend::log_once(
                "openvino-read-file",
                "OpenVINO C runtime has no read-from-memory call; IR files go through the temporary directory",
            );
        }
        Ok(api)
    }

    /// Text for a failed status: the runtime's last error message when it has one, else the generic description.
    pub fn error_text(&self, status: Status) -> String {
        if let Some(last) = self.last_error {
            // SAFETY: the returned pointer is either null or a NUL-terminated string owned by the runtime (must not be freed).
            let message = unsafe { last() };
            if !message.is_null() {
                // SAFETY: non-null, NUL-terminated.
                let text = unsafe { CStr::from_ptr(message) }
                    .to_string_lossy()
                    .trim()
                    .to_owned();
                if !text.is_empty() {
                    return format!("{text} (status {status})");
                }
            }
        }
        if let Some(info) = self.error_info {
            // SAFETY: static NUL-terminated string (see ov_common.h).
            let message = unsafe { info(status) };
            if !message.is_null() {
                // SAFETY: non-null, NUL-terminated.
                return format!(
                    "{} (status {status})",
                    unsafe { CStr::from_ptr(message) }.to_string_lossy()
                );
            }
        }
        format!("status {status}")
    }

    /// `Ok(())` for `OK`, else an error naming the call.
    pub fn check(&self, status: Status, call: &str) -> Result<(), String> {
        if status == STATUS_OK {
            Ok(())
        } else {
            Err(format!("{call} failed: {}", self.error_text(status)))
        }
    }

    /// Version string of the runtime (`buildNumber`), when the runtime reports one.
    pub fn version_string(&self) -> Option<String> {
        let (version, free) = (self.version?, self.version_free?);
        let mut value = OvVersion {
            build_number: std::ptr::null(),
            description: std::ptr::null(),
        };
        // SAFETY: `value` is a valid out-parameter; the strings it receives are copied before `ov_version_free`.
        let status = unsafe { version(&mut value) };
        if status != STATUS_OK {
            return None;
        }
        let text = if value.build_number.is_null() {
            None
        } else {
            // SAFETY: non-null, NUL-terminated, valid until `ov_version_free`.
            Some(
                unsafe { CStr::from_ptr(value.build_number) }
                    .to_string_lossy()
                    .into_owned(),
            )
        };
        // SAFETY: `value` was filled by `ov_get_openvino_version`.
        unsafe { free(&mut value) };
        text
    }

    /// Whether the runtime has a call to read an IR from memory (else files are used).
    pub fn reads_from_memory(&self) -> bool {
        self.read_memory_buffer.is_some() || self.read_memory_old.is_some()
    }

    /// `ov_core_read_model_from_memory_buffer` (2024.0+) or the older `ov_core_read_model_from_memory`.
    ///
    /// # Safety
    /// `core` must be a live core; `weights` is null or a live tensor holding the `.bin` bytes.
    pub unsafe fn read_from_memory(
        &self,
        core: *const OvCore,
        xml: &[u8],
        weights: *const OvTensor,
        model: *mut *mut OvModel,
    ) -> Option<Status> {
        if let Some(read) = self.read_memory_buffer {
            return Some(read(core, xml.as_ptr().cast(), xml.len(), weights, model));
        }
        if let Some(read) = self.read_memory_old {
            // the old call takes a NUL-terminated string
            let mut text = Vec::with_capacity(xml.len() + 1);
            text.extend_from_slice(xml);
            text.push(0);
            return Some(read(core, text.as_ptr().cast(), weights, model));
        }
        None
    }

    /// The `ov_element_type_e` numbers of this runtime (probed once).
    ///
    /// The enum was renumbered in 2025.2, and its documentation says it mirrors `ov::element::Type_t`, so the numbers are derived
    /// from behaviour: in both numberings the candidates `4` and `5` are `{F32, F64}` (new) or `{F16, F32}` (old); the one whose
    /// eight-element tensor has 32 bytes is `F32`, and `F16` is the number before it.
    pub fn element_types(&self) -> Result<ElementTypes, String> {
        self.element_types
            .get_or_init(|| self.probe_element_types())
            .clone()
    }

    fn probe_element_types(&self) -> Result<ElementTypes, String> {
        for candidate in [4, 5] {
            let mut dims = [8i64];
            let mut shape = OvShape {
                rank: 1,
                dims: std::ptr::null_mut(),
            };
            // SAFETY: `dims` holds `rank` values; `shape` receives an allocation that is released below.
            self.check(
                unsafe { (self.shape_create)(1, dims.as_mut_ptr(), &mut shape) },
                "ov_shape_create",
            )?;
            let mut tensor: *mut OvTensor = std::ptr::null_mut();
            // SAFETY: `shape` is a valid shape; the tensor is freed below.
            let status = unsafe { (self.tensor_create)(candidate, shape, &mut tensor) };
            let mut bytes = 0usize;
            let sized = status == STATUS_OK
                && !tensor.is_null()
                // SAFETY: `tensor` is live.
                && unsafe { (self.tensor_byte_size)(tensor, &mut bytes) } == STATUS_OK;
            if !tensor.is_null() {
                // SAFETY: `tensor` was created above and is not used afterwards.
                unsafe { (self.tensor_free)(tensor) };
            }
            // SAFETY: `shape` was created by `ov_shape_create`.
            unsafe { (self.shape_free)(&mut shape) };
            if sized && bytes == 32 {
                return Ok(ElementTypes {
                    f16: candidate - 1,
                    f32: candidate,
                });
            }
        }
        Err("cannot determine the element type numbering of this OpenVINO C runtime".into())
    }
}

/// Open a candidate. On Windows an absolute path is loaded with `LOAD_WITH_ALTERED_SEARCH_PATH`, so the dependencies
/// (`openvino.dll`) next to `openvino_c.dll` are found; the TBB directory of an OpenVINO installation is added to `PATH` first.
fn open_library(path: &Path) -> Result<Library, String> {
    #[cfg(windows)]
    {
        if path.is_absolute() {
            prepare_windows_path(path);
            use libloading::os::windows::{
                Library as WindowsLibrary, LOAD_WITH_ALTERED_SEARCH_PATH,
            };
            // SAFETY: loading a DLL runs its initialisation code; the path comes from the user's environment.
            return unsafe { WindowsLibrary::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH) }
                .map(Library::from)
                .map_err(|e| e.to_string());
        }
    }
    // SAFETY: as above.
    unsafe { Library::new(path) }.map_err(|e| e.to_string())
}

/// `<root>/runtime/bin/intel64/Release/openvino_c.dll` needs `<root>/runtime/3rdparty/tbb/bin` (tbb12.dll) on the DLL search path.
#[cfg(windows)]
fn prepare_windows_path(dll: &Path) {
    let Some(dir) = dll.parent() else { return };
    let mut extra: Vec<PathBuf> = vec![dir.to_path_buf()];
    for up in dir.ancestors().skip(1).take(4) {
        let tbb = up.join("3rdparty").join("tbb").join("bin");
        if tbb.is_dir() {
            extra.push(tbb);
        }
    }
    let mut paths: Vec<PathBuf> =
        std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).collect();
    let mut changed = false;
    for dir in extra {
        if !paths.contains(&dir) {
            paths.insert(0, dir);
            changed = true;
        }
    }
    if changed {
        if let Ok(joined) = std::env::join_paths(paths) {
            std::env::set_var("PATH", joined);
        }
    }
}

/// File names of the C runtime inside a directory, best first: the plain name, then versioned names (newest last in the sort order, so
/// reversed).
fn runtime_files_in(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
                return false;
            };
            if cfg!(windows) {
                name.eq_ignore_ascii_case("openvino_c.dll")
            } else if cfg!(target_vendor = "apple") {
                name.starts_with("libopenvino_c.") && name.ends_with(".dylib")
            } else {
                name == "libopenvino_c.so"
                    || (name.starts_with("libopenvino_c.so.") && !name.ends_with(".debug"))
            }
        })
        .collect();
    found.sort();
    found.reverse();
    // the unversioned name (a symlink to the newest) first
    found.sort_by_key(|p| {
        p.file_name().map_or(1, |n| {
            usize::from(n != "libopenvino_c.so" && n != "openvino_c.dll")
        })
    });
    found
}

/// Where an OpenVINO installation keeps the C runtime below its root.
fn installation_dirs(root: &Path) -> Vec<PathBuf> {
    vec![
        root.to_path_buf(),
        root.join("runtime").join("lib").join("intel64"),
        root.join("runtime").join("lib").join("aarch64"),
        root.join("runtime")
            .join("bin")
            .join("intel64")
            .join("Release"),
        root.join("runtime").join("bin").join("intel64"),
        root.join("lib"),
        root.join("bin"),
        root.join("libs"),
    ]
}

/// The first file of the C runtime that exists on disk (without loading it), for the status page: `None` when only the system loader's
/// own search could find it.
pub fn find_runtime_file() -> Option<PathBuf> {
    library_candidates()
        .into_iter()
        .find(|p| p.parent().is_some_and(|d| !d.as_os_str().is_empty()) && p.is_file())
}

/// Every place the C runtime is looked for, most specific first.
fn library_candidates() -> Vec<PathBuf> {
    if let Some(explicit) = std::env::var_os("FLLM_OPENVINO_C_LIB").filter(|v| !v.is_empty()) {
        return vec![PathBuf::from(explicit)];
    }
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |path: PathBuf| {
        if !out.contains(&path) {
            out.push(path);
        }
    };
    for var in [
        "FLLM_OPENVINO_DIR",
        "INTEL_OPENVINO_DIR",
        "OPENVINO_DIR",
        "OpenVINO_DIR",
    ] {
        if let Some(root) = std::env::var_os(var).filter(|v| !v.is_empty()) {
            for dir in installation_dirs(Path::new(&root)) {
                runtime_files_in(&dir).into_iter().for_each(&mut push);
            }
        }
    }
    // the system loader first (LD_LIBRARY_PATH, ldconfig, PATH on Windows)
    let default_name = if cfg!(windows) {
        "openvino_c.dll"
    } else if cfg!(target_vendor = "apple") {
        "libopenvino_c.dylib"
    } else {
        "libopenvino_c.so"
    };
    push(PathBuf::from(default_name));
    for var in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"] {
        if let Some(value) = std::env::var_os(var) {
            for dir in std::env::split_paths(&value) {
                runtime_files_in(&dir).into_iter().for_each(&mut push);
            }
        }
    }
    // pip wheels: <venv>/lib/python3.x/site-packages/openvino/libs (and Windows <venv>/Lib/site-packages/openvino/libs)
    if let Some(venv) = std::env::var_os("VIRTUAL_ENV").filter(|v| !v.is_empty()) {
        let venv = PathBuf::from(venv);
        let mut site_roots = vec![venv.join("Lib").join("site-packages")];
        if let Ok(entries) = std::fs::read_dir(venv.join("lib")) {
            site_roots.extend(
                entries
                    .filter_map(Result::ok)
                    .map(|e| e.path().join("site-packages")),
            );
        }
        for site in site_roots {
            runtime_files_in(&site.join("openvino").join("libs"))
                .into_iter()
                .for_each(&mut push);
        }
    }
    // distribution packages and the default install prefix of the archives
    let mut roots: Vec<PathBuf> = Vec::new();
    if !cfg!(windows) {
        for base in ["/opt/intel", "/usr/local", "/usr"] {
            if let Ok(entries) = std::fs::read_dir(base) {
                roots.extend(
                    entries
                        .filter_map(Result::ok)
                        .map(|e| e.path())
                        .filter(|p| {
                            p.file_name()
                                .and_then(|n| n.to_str())
                                .is_some_and(|n| n.starts_with("openvino"))
                        }),
                );
            }
        }
        for dir in [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/lib64",
            "/usr/local/lib",
            "/usr/lib",
        ] {
            runtime_files_in(Path::new(dir))
                .into_iter()
                .for_each(&mut push);
        }
    } else if let Some(program_files) = std::env::var_os("ProgramFiles") {
        let base = Path::new(&program_files).join("Intel");
        if let Ok(entries) = std::fs::read_dir(base) {
            roots.extend(
                entries
                    .filter_map(Result::ok)
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("openvino"))
                    }),
            );
        }
    }
    // the copy `fllm-rs npu-setup` installs (pip --target), and pip wheels in the user and system Python site directories
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from);
    let fllm_home = std::env::var_os("FLLM_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| home.as_ref().map(|h| h.join(".fllm")));
    if let Some(base) = fllm_home {
        runtime_files_in(&base.join("openvino").join("openvino").join("libs"))
            .into_iter()
            .for_each(&mut push);
    }
    if !cfg!(windows) {
        let mut python_roots: Vec<PathBuf> = Vec::new();
        if let Some(h) = &home {
            python_roots.push(h.join(".local").join("lib"));
        }
        python_roots.extend(["/usr/local/lib", "/usr/lib"].map(PathBuf::from));
        for base in python_roots {
            let Ok(entries) = std::fs::read_dir(&base) else { continue };
            for entry in entries.filter_map(Result::ok) {
                let name = entry.file_name();
                if name.to_str().is_some_and(|n| n.starts_with("python3")) {
                    for site in ["site-packages", "dist-packages"] {
                        runtime_files_in(&entry.path().join(site).join("openvino").join("libs"))
                            .into_iter()
                            .for_each(&mut push);
                    }
                }
            }
        }
    }
    roots.sort();
    roots.reverse();
    for root in roots {
        for dir in installation_dirs(&root) {
            runtime_files_in(&dir).into_iter().for_each(&mut push);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_library_variable_is_the_only_candidate() {
        // Not asserting on the environment of the machine: only the pure helpers.
        let dir = std::env::temp_dir().join(format!("fllm-ov-ffi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in [
            "libopenvino_c.so.2540",
            "libopenvino_c.so",
            "libopenvino_c.so.2641",
            "libopenvino_c.so.2641.debug",
            "other.so",
        ] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
        let files = runtime_files_in(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        if cfg!(all(unix, not(target_vendor = "apple"))) {
            let names: Vec<_> = files
                .iter()
                .filter_map(|p| p.file_name()?.to_str().map(str::to_owned))
                .collect();
            assert_eq!(
                names,
                [
                    "libopenvino_c.so",
                    "libopenvino_c.so.2641",
                    "libopenvino_c.so.2540"
                ]
            );
        }
    }

    #[test]
    fn installation_dirs_cover_both_layouts() {
        let dirs = installation_dirs(Path::new("/opt/intel/openvino_2026"));
        assert!(dirs.iter().any(|d| d.ends_with("runtime/lib/intel64")));
        assert!(dirs
            .iter()
            .any(|d| d.ends_with("runtime/bin/intel64/Release")));
    }
}
