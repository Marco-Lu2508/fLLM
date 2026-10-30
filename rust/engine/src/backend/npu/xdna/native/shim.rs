//! Run-time binding of the XDNA shim (`rust/engine/xdna/shim/fllm_xdna_shim.cpp`), the small C++ library between fLLM and XRT.
//!
//! Nothing here is linked at build time: the shim (`libfllm_xdna_shim.so`, `fllm_xdna_shim.dll`) is opened when the backend is asked
//! for, and it in turn loads XRT (`libxrt_coreutil`). A missing shim, a missing XRT or a missing device come back as an `Err` with the
//! reason and the backend reports itself unavailable; nothing else in fLLM changes.
//!
//! Search order for the shim: `FLLM_XDNA_SHIM` (a file), the folder of the executable, the fLLM cache folder (`~/.cache/fllm/npu`,
//! `%LOCALAPPDATA%\fllm\npu`), then the system loader. `scripts/build-xdna-shim.sh` builds it on Linux.
//!
//! Verification level: L1 on one Ryzen AI 5 PRO 435 laptop under Linux (see `docs/npu-xdna.md`); Windows untested.

use libloading::Library;
use std::ffi::{c_char, c_void, CStr};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// Version of the C interface this file speaks (`fx_abi_version` of the shim).
pub const ABI_VERSION: u32 = 1;

#[cfg(windows)]
const SHIM_FILE: &str = "fllm_xdna_shim.dll";
#[cfg(not(windows))]
const SHIM_FILE: &str = "libfllm_xdna_shim.so";

type FnAbi = unsafe extern "C" fn() -> u32;
type FnLastError = unsafe extern "C" fn() -> *const c_char;
type FnDeviceCount = unsafe extern "C" fn() -> i32;
type FnOpen = unsafe extern "C" fn(u32, *const c_void, usize) -> *mut c_void;
type FnClose = unsafe extern "C" fn(*mut c_void);
type FnBufNew = unsafe extern "C" fn(*mut c_void, i32, usize, i32) -> *mut c_void;
type FnBufFree = unsafe extern "C" fn(*mut c_void);
type FnBufPtr = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type FnBufSync = unsafe extern "C" fn(*mut c_void, i32, usize) -> i32;
type FnRunStart = unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
type FnRunWait = unsafe extern "C" fn(*mut c_void, u32) -> i32;

/// The opened shim library with its entry points.
pub struct Shim {
    _lib: Library,
    pub path: PathBuf,
    last_error: FnLastError,
    device_count: FnDeviceCount,
    open: FnOpen,
    close: FnClose,
    buf_new: FnBufNew,
    buf_free: FnBufFree,
    buf_ptr: FnBufPtr,
    buf_sync: FnBufSync,
    run_start: FnRunStart,
    run_wait: FnRunWait,
}

// SAFETY: the shim keeps its error text in a thread-local and every handle is used under the engine's mutex.
unsafe impl Send for Shim {}
unsafe impl Sync for Shim {}

fn candidates() -> Vec<PathBuf> {
    let mut found = Vec::new();
    if let Some(path) = std::env::var_os("FLLM_XDNA_SHIM").filter(|v| !v.is_empty()) {
        found.push(PathBuf::from(path));
        return found;
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(PathBuf::from)) {
        found.push(dir.join(SHIM_FILE));
    }
    if let Some(dir) = cache_dir() {
        found.push(dir.join(SHIM_FILE));
    }
    found.push(PathBuf::from(SHIM_FILE));
    found
}

/// The fLLM cache folder for NPU files (`~/.cache/fllm/npu` or `%LOCALAPPDATA%\fllm\npu`), not created here.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").filter(|v| !v.is_empty()).map(|h| PathBuf::from(h).join(".cache")))
        .or_else(|| std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty()).map(PathBuf::from))?;
    Some(base.join("fllm").join("npu"))
}

/// The source of the shim, embedded so that `fllm-rs amd-npu build-shim` works without a checkout of the repository.
const SHIM_SOURCE: &str = include_str!("../../../../../xdna/shim/fllm_xdna_shim.cpp");

/// Builds the shim from the embedded source with the system C++ compiler and the XRT development files, into the fLLM cache folder, and
/// returns the path. Linux and other Unix systems; on Windows the error says what to run. Needs `libxrt-dev` (headers) and `uuid-dev`.
pub fn build_from_embedded() -> Result<PathBuf, String> {
    if cfg!(windows) {
        return Err("build the shim with MSVC against the XRT SDK: see docs/npu-xdna.md, section Windows".to_owned());
    }
    let dir = cache_dir().ok_or("no cache folder (set HOME or XDG_CACHE_HOME)")?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let source = dir.join("fllm_xdna_shim.cpp");
    std::fs::write(&source, SHIM_SOURCE).map_err(|e| format!("{}: {e}", source.display()))?;
    let include = std::env::var_os("XRT_INCLUDE")
        .map(PathBuf::from)
        .or_else(|| {
            ["/opt/xilinx/xrt/include", "/usr/include", "/usr/local/include"]
                .iter()
                .map(PathBuf::from)
                .find(|d| d.join("xrt").join("xrt_kernel.h").exists())
        })
        .ok_or("the XRT headers (xrt/xrt_kernel.h) were not found: install libxrt-dev (and uuid-dev), or set XRT_INCLUDE")?;
    // link against the development symlink when there is one, else against the versioned runtime file
    let lib_dirs: Vec<PathBuf> = std::env::var_os("XRT_LIB")
        .map(|v| vec![PathBuf::from(v)])
        .unwrap_or_else(|| ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib64", "/usr/lib", "/opt/xilinx/xrt/lib", "/usr/local/lib"].iter().map(PathBuf::from).collect());
    let mut link_args: Vec<String> = vec!["-lxrt_coreutil".to_owned()];
    if let Some(d) = lib_dirs.iter().find(|d| d.join("libxrt_coreutil.so").exists()) {
        link_args.insert(0, format!("-L{}", d.display()));
    } else if let Some(d) = lib_dirs.iter().find(|d| d.join("libxrt_coreutil.so.2").exists()) {
        link_args = vec![format!("-L{}", d.display()), "-l:libxrt_coreutil.so.2".to_owned()];
    }
    let out = dir.join(SHIM_FILE);
    let compilers: Vec<String> = std::env::var("CXX").ok().into_iter().chain(["c++".to_owned(), "g++".to_owned(), "clang++".to_owned()]).collect();
    let mut last = String::from("no C++ compiler found (tried c++, g++, clang++; set CXX)");
    for compiler in compilers {
        let result = std::process::Command::new(&compiler)
            .args(["-std=c++17", "-O2", "-fPIC", "-shared", "-fvisibility=hidden"])
            .arg(format!("-I{}", include.display()))
            .arg(&source)
            .arg("-o")
            .arg(&out)
            .args(&link_args)
            .arg("-Wl,-rpath,$ORIGIN")
            .output();
        match result {
            Ok(o) if o.status.success() => return Ok(out),
            Ok(o) => last = format!("{compiler} failed: {}", String::from_utf8_lossy(&o.stderr).lines().take(12).collect::<Vec<_>>().join("\n")),
            Err(e) => last = format!("{compiler}: {e}"),
        }
    }
    Err(last)
}

impl Shim {
    fn open_at(path: &std::path::Path) -> Result<Shim, String> {
        // SAFETY: loading a library runs its initializers; this is the shim we ship the source of (or the file the user named).
        let lib = unsafe { super::super::dl::open(path) }.map_err(|e| format!("{}: {e}", path.display()))?;
        macro_rules! sym {
            ($name:literal, $ty:ty) => {
                // SAFETY: the symbol has the signature of the declaration in fllm_xdna_shim.cpp.
                *unsafe { lib.get::<$ty>(concat!($name, "\0").as_bytes()) }
                    .map_err(|e| format!("{}: missing {}: {e}", path.display(), $name))?
            };
        }
        let abi: FnAbi = sym!("fx_abi_version", FnAbi);
        // SAFETY: plain function without arguments.
        let version = unsafe { abi() };
        if version != ABI_VERSION {
            return Err(format!("{}: interface version {version}, fLLM expects {ABI_VERSION}; rebuild the shim", path.display()));
        }
        Ok(Shim {
            path: path.to_path_buf(),
            last_error: sym!("fx_last_error", FnLastError),
            device_count: sym!("fx_device_count", FnDeviceCount),
            open: sym!("fx_open", FnOpen),
            close: sym!("fx_close", FnClose),
            buf_new: sym!("fx_buf_new", FnBufNew),
            buf_free: sym!("fx_buf_free", FnBufFree),
            buf_ptr: sym!("fx_buf_ptr", FnBufPtr),
            buf_sync: sym!("fx_buf_sync", FnBufSync),
            run_start: sym!("fx_run_start", FnRunStart),
            run_wait: sym!("fx_run_wait", FnRunWait),
            _lib: lib,
        })
    }

    /// The shim of this process, loaded once. `Err` says what is missing.
    pub fn get() -> Result<&'static Arc<Shim>, String> {
        // only a success is remembered: the shim may be built or XRT installed while the process runs
        static SHIM: OnceLock<Arc<Shim>> = OnceLock::new();
        if let Some(shim) = SHIM.get() {
            return Ok(shim);
        }
        let mut errors = Vec::new();
        for path in candidates() {
            if path.is_absolute() && !path.exists() {
                continue;
            }
            match Shim::open_at(&path) {
                Ok(shim) => return Ok(SHIM.get_or_init(|| Arc::new(shim))),
                // a bare file name that the system loader does not know just means "not installed"
                Err(e) if path.is_absolute() => errors.push(e),
                Err(_) => {}
            }
        }
        if errors.is_empty() {
            Err(format!(
                "the XDNA shim {SHIM_FILE} was not found (build it with `fllm-rs amd-npu build-shim`, or name it in FLLM_XDNA_SHIM)"
            ))
        } else {
            Err(format!("the XDNA shim could not be loaded: {}", errors.join("; ")))
        }
    }

    fn error_text(&self) -> String {
        // SAFETY: the shim returns a NUL-terminated string that lives until the next call on this thread.
        let ptr = unsafe { (self.last_error)() };
        if ptr.is_null() {
            return String::new();
        }
        // SAFETY: see above.
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }

    /// Number of NPUs XRT sees (an `Err` when XRT itself fails, for example when no driver is loaded).
    pub fn device_count(&self) -> Result<usize, String> {
        // SAFETY: plain call.
        let n = unsafe { (self.device_count)() };
        if n < 0 {
            Err(self.error_text())
        } else {
            Ok(n as usize)
        }
    }

    /// Open device `index`, load the xclbin and create the kernel context.
    pub fn open_context(self: &Arc<Self>, index: usize, xclbin: &[u8]) -> Result<Context, String> {
        // SAFETY: xclbin is a valid slice for the duration of the call; the shim copies it.
        let ptr = unsafe { (self.open)(index as u32, xclbin.as_ptr().cast(), xclbin.len()) };
        if ptr.is_null() {
            return Err(self.error_text());
        }
        Ok(Context { shim: Arc::clone(self), ptr })
    }
}

/// An open device with the GEMM kernel loaded. Not thread-safe by itself: the engine serializes the use.
pub struct Context {
    shim: Arc<Shim>,
    ptr: *mut c_void,
}

// SAFETY: the pointer is an owned heap object of the shim; the engine holds the context behind a mutex.
unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from fx_open and is closed once.
        unsafe { (self.shim.close)(self.ptr) };
    }
}

/// Which kernel argument a buffer is bound to.
#[derive(Clone, Copy)]
pub enum Arg {
    Instructions = 1,
    A = 3,
    B = 4,
    C = 5,
}

impl Context {
    pub fn alloc(&self, arg: Arg, bytes: usize) -> Result<Buf, String> {
        // SAFETY: valid context pointer.
        let ptr = unsafe { (self.shim.buf_new)(self.ptr, arg as i32, bytes, i32::from(matches!(arg, Arg::Instructions))) };
        if ptr.is_null() {
            return Err(self.shim.error_text());
        }
        // SAFETY: valid buffer pointer.
        let mapped = unsafe { (self.shim.buf_ptr)(ptr) }.cast::<u8>();
        Ok(Buf { shim: Arc::clone(&self.shim), ptr, mapped, bytes })
    }

    /// Start `a x b^T -> c` with the instruction stream `instr` (`words` 32-bit words).
    pub fn start(&self, instr: &Buf, words: usize, a: &Buf, b: &Buf, c: &Buf) -> Result<Run, String> {
        // SAFETY: all pointers are live buffers of this context.
        let run = unsafe { (self.shim.run_start)(self.ptr, instr.ptr, words as u32, a.ptr, b.ptr, c.ptr) };
        if run.is_null() {
            return Err(self.shim.error_text());
        }
        Ok(Run { shim: Arc::clone(&self.shim), ptr: run })
    }
}

/// A buffer shared between the host and the NPU (host memory, mapped).
pub struct Buf {
    shim: Arc<Shim>,
    ptr: *mut c_void,
    mapped: *mut u8,
    pub bytes: usize,
}

// SAFETY: owned shim object; access is serialized by the engine.
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Drop for Buf {
    fn drop(&mut self) {
        // SAFETY: freed once.
        unsafe { (self.shim.buf_free)(self.ptr) };
    }
}

impl Buf {
    /// The mapped memory as 16-bit words (bf16 data).
    pub fn as_u16_mut(&mut self) -> &mut [u16] {
        // SAFETY: the mapping is `bytes` long, page aligned, and exclusively borrowed here.
        unsafe { std::slice::from_raw_parts_mut(self.mapped.cast::<u16>(), self.bytes / 2) }
    }

    pub fn as_f32(&self) -> &[f32] {
        // SAFETY: the mapping is `bytes` long and page aligned.
        unsafe { std::slice::from_raw_parts(self.mapped.cast::<f32>(), self.bytes / 4) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above.
        unsafe { std::slice::from_raw_parts_mut(self.mapped, self.bytes) }
    }

    /// Make host writes visible to the NPU.
    pub fn to_device(&self) -> Result<(), String> {
        self.sync(1)
    }

    /// Make NPU writes visible to the host.
    pub fn from_device(&self) -> Result<(), String> {
        self.sync(0)
    }

    fn sync(&self, to_device: i32) -> Result<(), String> {
        // SAFETY: valid buffer pointer.
        if unsafe { (self.shim.buf_sync)(self.ptr, to_device, 0) } == 0 {
            Ok(())
        } else {
            Err(self.shim.error_text())
        }
    }
}

/// A started kernel run.
pub struct Run {
    shim: Arc<Shim>,
    ptr: *mut c_void,
}

// SAFETY: owned shim object.
unsafe impl Send for Run {}

impl Run {
    /// Wait for the run to finish (`timeout_ms` = 0: no timeout).
    pub fn wait(self, timeout_ms: u32) -> Result<(), String> {
        let ptr = self.ptr;
        let shim = Arc::clone(&self.shim);
        std::mem::forget(self);
        // SAFETY: the handle is consumed by fx_run_wait.
        if unsafe { (shim.run_wait)(ptr, timeout_ms) } == 0 {
            Ok(())
        } else {
            Err(shim.error_text())
        }
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        // a run that was never waited for still has to be finished before its handle is released
        // SAFETY: the handle is consumed by fx_run_wait.
        unsafe { (self.shim.run_wait)(self.ptr, 0) };
    }
}
