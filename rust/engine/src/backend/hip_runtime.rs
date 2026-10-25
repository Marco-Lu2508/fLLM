//! Finding the AMD HIP runtime DLL at run time (Windows) and starting without it.
//!
//! The kernel library calls the HIP runtime (`amdhip64_N.dll`). On Windows the import is *delay-loaded* (`build.rs`: `/DELAYLOAD`), so
//! `fllm-rs.exe` starts on a machine without the DLL and the other backends (Vulkan, CPU) stay usable. Two delay-load hooks do the work:
//!
//! * before the DLL is loaded, [`candidate_paths`] gives the search order and the first file that loads wins;
//! * when no DLL can be loaded, the hook hands out stubs: the registration calls the compiler emits for the GPU code (they run before
//!   `main`) do nothing, `hipGetErrorString` returns a text, every other call returns `hipErrorNoDevice`. [`check`] then returns the
//!   message that tells the user what to install, and the CUDA module reports no AMD device.
//!
//! Search order (first existing file that loads): `FLLM_HIP_DLL` (a file, or a directory that holds the DLL); the directory of
//! `fllm-rs.exe` (a bundled copy); `System32` (the Adrenalin driver installs a HIP runtime there on recent versions);
//! `%HIP_PATH%\bin`; `%ROCM_PATH%\bin`; `Program Files\AMD\ROCm\<newest version>\bin`; last the Windows default DLL search (`PATH`).
//! `HSA_OVERRIDE_GFX_VERSION` belongs to the Linux ROCm stack (ROCr) and is not expected to do anything on Windows, where HIP runs on the
//! PAL driver path; see [`hsa_override_applies`].
//!
//! On other platforms none of this is active (`check` always succeeds; Linux links `libamdhip64` the ordinary way).
//! Level of verification: L0 (the pure functions are unit tested; the hooks compile in CI, never ran on a machine with an AMD GPU).

#![allow(dead_code)]

use std::path::PathBuf;

/// Name of the runtime DLL the kernel library imports (`amdhip64_7.dll`), as read from the import library at build time.
pub const DLL_NAME: &str = match option_env!("FLLM_HIP_DLL_NAME") {
    Some(name) => name,
    None => "amdhip64.dll",
};

/// Where the DLL search looks. Plain data so that the order can be tested.
#[derive(Debug, Clone, Default)]
pub struct SearchEnv {
    /// `FLLM_HIP_DLL`: the DLL itself, or the directory that contains it.
    pub explicit: Option<PathBuf>,
    /// Directory of the running executable.
    pub exe_dir: Option<PathBuf>,
    /// `%SystemRoot%\System32`.
    pub system_dir: Option<PathBuf>,
    pub hip_path: Option<PathBuf>,
    pub rocm_path: Option<PathBuf>,
    /// `Program Files\AMD\ROCm` and the names of the version directories in it (any order).
    pub rocm_root: Option<PathBuf>,
    pub rocm_versions: Vec<String>,
}

/// One place to look: what it is (for the message) and the full path of the DLL there.
pub type Candidate = (&'static str, PathBuf);

/// Version directory names (`7.1`, `6.4.2`, `5.7`) newest first; names that are not versions go last.
pub fn newest_first(mut names: Vec<String>) -> Vec<String> {
    let key = |n: &str| -> Vec<u64> { n.split('.').map(|p| p.parse::<u64>().unwrap_or(0)).collect() };
    names.sort_by(|a, b| key(b).cmp(&key(a)).then_with(|| a.cmp(b)));
    names
}

/// The DLL paths to try, in order (see the module comment).
pub fn candidate_paths(env: &SearchEnv, dll: &str) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = Vec::new();
    if let Some(explicit) = &env.explicit {
        let is_dll = explicit.extension().is_some_and(|e| e.eq_ignore_ascii_case("dll"));
        out.push(("FLLM_HIP_DLL", if is_dll { explicit.clone() } else { explicit.join(dll) }));
    }
    if let Some(dir) = &env.exe_dir {
        out.push(("next to fllm-rs.exe", dir.join(dll)));
    }
    if let Some(dir) = &env.system_dir {
        out.push(("System32 (graphics driver)", dir.join(dll)));
    }
    if let Some(dir) = &env.hip_path {
        out.push(("HIP_PATH", dir.join("bin").join(dll)));
    }
    if let Some(dir) = &env.rocm_path {
        out.push(("ROCM_PATH", dir.join("bin").join(dll)));
    }
    if let Some(root) = &env.rocm_root {
        for version in newest_first(env.rocm_versions.clone()) {
            out.push(("Program Files\\AMD\\ROCm", root.join(version).join("bin").join(dll)));
        }
    }
    out
}

/// The text shown when no runtime was found.
pub fn missing_message(dll: &str, tried: &[Candidate]) -> String {
    let mut text = format!(
        "AMD HIP runtime not found ({dll}). Install the current AMD Radeon (Adrenalin) driver, or the AMD HIP SDK for Windows \
         (https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html), or point FLLM_HIP_DLL at {dll}. \
         AMD GPUs are not used; Vulkan and the CPU still work."
    );
    if !tried.is_empty() {
        text.push_str(" Looked in: ");
        text.push_str(&tried.iter().map(|(what, path)| format!("{what} [{}]", path.display())).collect::<Vec<_>>().join("; "));
        text.push('.');
    }
    text
}

/// Does `HSA_OVERRIDE_GFX_VERSION` change which GPU code the runtime loads on this platform? It is read by the ROCm user-mode stack on
/// Linux. The Windows runtime does not use that stack, so it is not expected to apply there (not verified on hardware).
pub fn hsa_override_applies() -> bool {
    !cfg!(windows)
}

/// Can a GPU of architecture `arch` run this binary's kernels? `built` is the `;` separated list the kernels were compiled for (`None`
/// or empty: unknown, assume yes). `override_set`: `HSA_OVERRIDE_GFX_VERSION` is set (the user asked for another chip's code).
pub fn arch_is_built(arch: &str, built: Option<&str>, override_set: bool) -> bool {
    let Some(list) = built.map(str::trim).filter(|l| !l.is_empty()) else { return true };
    if override_set && hsa_override_applies() {
        return true;
    }
    list.split(';').any(|a| a.trim() == arch)
}

/// The architectures the kernels of this binary were built for (`None` when the build script did not record them).
pub fn built_archs() -> Option<&'static str> {
    option_env!("FLLM_HIP_BUILT_ARCHS")
}

#[cfg(all(windows, fllm_hip_delayload))]
fn real_search_env() -> SearchEnv {
    use std::path::Path;
    let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let rocm_root = var("ProgramFiles").map(|p| p.join("AMD").join("ROCm"));
    let rocm_versions = rocm_root
        .as_deref()
        .and_then(|root| std::fs::read_dir(root).ok())
        .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    SearchEnv {
        explicit: var("FLLM_HIP_DLL"),
        exe_dir: std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)),
        system_dir: var("SystemRoot").map(|p| p.join("System32")),
        hip_path: var("HIP_PATH"),
        rocm_path: var("ROCM_PATH"),
        rocm_root,
        rocm_versions,
    }
}

/// Is the HIP runtime usable in this process? `Ok` also on platforms where nothing is delay-loaded (then the library was needed to start).
/// Cheap after the first call.
pub fn check() -> Result<(), String> {
    #[cfg(all(windows, fllm_hip_delayload))]
    {
        return delay::check();
    }
    #[allow(unreachable_code)]
    Ok(())
}

/// Log the missing-runtime message once per process (to stderr), for the places that quietly drop AMD GPUs from the inventory.
pub fn note_missing_once(message: &str) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("[fllm-rs] hip: {message}"));
}

#[cfg(all(windows, fllm_hip_delayload))]
mod delay {
    use super::*;
    use std::ffi::{c_char, c_void, CStr};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, OnceLock};

    /// `DelayLoadProc` of `delayimp.h`.
    #[repr(C)]
    pub struct DelayLoadProc {
        by_name: i32,
        /// `LPCSTR szProcName` / `DWORD dwOrdinal`
        name_or_ordinal: *const c_char,
    }

    /// `DelayLoadInfo` of `delayimp.h` (x64 layout).
    #[repr(C)]
    pub struct DelayLoadInfo {
        cb: u32,
        pidd: *const c_void,
        ppfn: *mut c_void,
        sz_dll: *const c_char,
        dlp: DelayLoadProc,
        hmod_cur: *mut c_void,
        pfn_cur: *mut c_void,
        last_error: u32,
    }

    type Hook = Option<unsafe extern "system" fn(u32, *const DelayLoadInfo) -> *const c_void>;

    const NOTE_PRE_LOAD_LIBRARY: u32 = 1;
    const FAIL_LOAD_LIB: u32 = 3;
    const FAIL_GET_PROC: u32 = 4;

    /// The runtime could not be loaded and stubs were handed out.
    static MISSING: AtomicBool = AtomicBool::new(false);
    /// The module handle of the DLL we loaded (0: none).
    static LOADED: OnceLock<usize> = OnceLock::new();
    static TRIED: Mutex<Vec<Candidate>> = Mutex::new(Vec::new());

    extern "system" {
        fn GetModuleHandleW(name: *const u16) -> *mut c_void;
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Load the first file of the search order that exists and loads; its directory is searched for the DLLs it needs.
    fn load_runtime() -> usize {
        *LOADED.get_or_init(|| {
            let env = real_search_env();
            let candidates = candidate_paths(&env, DLL_NAME);
            for (_, path) in &candidates {
                if !path.is_file() {
                    continue;
                }
                // SAFETY: loading a library runs its initializers, as with any LoadLibrary; the DLL is the AMD HIP runtime.
                let loaded = unsafe { libloading::os::windows::Library::load_with_flags(path, libloading::os::windows::LOAD_WITH_ALTERED_SEARCH_PATH) };
                if let Ok(lib) = loaded {
                    // never unloaded: the module lives as long as the process
                    return lib.into_raw() as usize;
                }
            }
            if let Ok(mut tried) = TRIED.lock() {
                *tried = candidates;
            }
            0
        })
    }

    unsafe extern "system" fn stub_zero() -> usize {
        0
    }

    /// `hipErrorNoDevice`
    unsafe extern "system" fn stub_no_device() -> i32 {
        100
    }

    unsafe extern "system" fn stub_message() -> *const c_char {
        b"HIP runtime DLL not found\0".as_ptr().cast()
    }

    unsafe extern "system" fn hook(notify: u32, info: *const DelayLoadInfo) -> *const c_void {
        // SAFETY: the delay-load helper passes a valid `DelayLoadInfo` whose `sz_dll` is a NUL-terminated string.
        let info = unsafe { &*info };
        let dll = unsafe { CStr::from_ptr(info.sz_dll) }.to_string_lossy();
        if !dll.eq_ignore_ascii_case(DLL_NAME) {
            return std::ptr::null();
        }
        match notify {
            NOTE_PRE_LOAD_LIBRARY => load_runtime() as *const c_void,
            FAIL_LOAD_LIB => {
                MISSING.store(true, Ordering::SeqCst);
                // any valid module: GetProcAddress on it fails, which asks us for a stub per function
                unsafe { GetModuleHandleW(std::ptr::null()) as *const c_void }
            }
            FAIL_GET_PROC => {
                let name = if info.dlp.by_name != 0 { unsafe { CStr::from_ptr(info.dlp.name_or_ordinal) }.to_string_lossy().into_owned() } else { String::new() };
                if name == "hipGetErrorString" || name == "hipGetErrorName" {
                    stub_message as *const c_void
                } else if name.starts_with("__hip") {
                    stub_zero as *const c_void
                } else {
                    stub_no_device as *const c_void
                }
            }
            _ => std::ptr::null(),
        }
    }

    // The symbols delayimp.lib looks for. Defined here, they replace its defaults.
    #[allow(non_upper_case_globals)]
    #[no_mangle]
    #[used]
    pub static __pfnDliNotifyHook2: Hook = Some(hook);
    #[allow(non_upper_case_globals)]
    #[no_mangle]
    #[used]
    pub static __pfnDliFailureHook2: Hook = Some(hook);


    pub fn check() -> Result<(), String> {
        // keep the hooks linked
        std::hint::black_box((&__pfnDliNotifyHook2, &__pfnDliFailureHook2));
        if MISSING.load(Ordering::SeqCst) {
            return Err(message());
        }
        // the registration at start-up normally loaded the DLL already; if nothing did yet, do it here
        // SAFETY: plain query with a NUL-terminated wide string.
        if unsafe { GetModuleHandleW(wide(DLL_NAME).as_ptr()) }.is_null() && load_runtime() == 0 {
            // last resort: the Windows default search (PATH, application directory)
            // SAFETY: loading the runtime DLL by name.
            match unsafe { libloading::os::windows::Library::new(DLL_NAME) } {
                Ok(lib) => {
                    let _ = lib.into_raw();
                }
                Err(_) => {
                    MISSING.store(true, Ordering::SeqCst);
                    return Err(message());
                }
            }
        }
        Ok(())
    }

    fn message() -> String {
        let tried = TRIED.lock().map(|t| t.clone()).unwrap_or_default();
        missing_message(DLL_NAME, &tried)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> SearchEnv {
        SearchEnv {
            explicit: Some(PathBuf::from("D:/hip")),
            exe_dir: Some(PathBuf::from("C:/fllm")),
            system_dir: Some(PathBuf::from("C:/Windows/System32")),
            hip_path: Some(PathBuf::from("C:/Program Files/AMD/ROCm/7.1")),
            rocm_path: None,
            rocm_root: Some(PathBuf::from("C:/Program Files/AMD/ROCm")),
            rocm_versions: vec!["6.4".into(), "7.1".into(), "7.10".into(), "notes".into()],
        }
    }

    #[test]
    fn search_order() {
        let c = candidate_paths(&env(), "amdhip64_7.dll");
        let labels: Vec<&str> = c.iter().map(|(l, _)| *l).collect();
        assert_eq!(
            labels,
            ["FLLM_HIP_DLL", "next to fllm-rs.exe", "System32 (graphics driver)", "HIP_PATH", "Program Files\\AMD\\ROCm", "Program Files\\AMD\\ROCm", "Program Files\\AMD\\ROCm", "Program Files\\AMD\\ROCm"]
        );
        assert_eq!(c[0].1, PathBuf::from("D:/hip").join("amdhip64_7.dll"));
        assert_eq!(c[3].1, PathBuf::from("C:/Program Files/AMD/ROCm/7.1").join("bin").join("amdhip64_7.dll"));
        // 7.10 is newer than 7.1, and the directory that is no version goes last
        assert!(c[4].1.to_string_lossy().contains("7.10"));
        assert!(c[5].1.to_string_lossy().contains("7.1"));
        assert!(c[7].1.to_string_lossy().contains("notes"));
    }

    #[test]
    fn explicit_file_is_taken_as_it_is() {
        let e = SearchEnv { explicit: Some(PathBuf::from("D:/x/my-hip.DLL")), ..SearchEnv::default() };
        assert_eq!(candidate_paths(&e, "amdhip64_7.dll"), [("FLLM_HIP_DLL", PathBuf::from("D:/x/my-hip.DLL"))]);
        assert!(candidate_paths(&SearchEnv::default(), "a.dll").is_empty());
    }

    #[test]
    fn message_says_what_to_install() {
        let m = missing_message("amdhip64_7.dll", &candidate_paths(&env(), "amdhip64_7.dll")[..2]);
        assert!(m.contains("Adrenalin") && m.contains("HIP SDK") && m.contains("FLLM_HIP_DLL") && m.contains("Vulkan"));
        assert!(m.contains("next to fllm-rs.exe"));
    }

    #[test]
    fn built_arch_filter() {
        let built = Some("gfx1100;gfx1151");
        assert!(arch_is_built("gfx1100", built, false));
        assert!(!arch_is_built("gfx1102", built, false));
        assert!(arch_is_built("gfx1102", None, false));
        assert!(arch_is_built("gfx1102", Some(""), false));
        // the override lets another chip's code run, but only where the runtime honours it
        assert_eq!(arch_is_built("gfx1102", built, true), hsa_override_applies());
    }
}
