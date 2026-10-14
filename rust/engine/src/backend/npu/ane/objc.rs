//! A very small Objective-C runtime layer (package 11, feature `npu-ane`, Apple platforms only).
//!
//! The study needs about twenty Objective-C messages (NSString, NSURL, NSArray, NSNumber, MLModel, MLMultiArray, ...). The `objc2` crates
//! are not enabled by the `npu-ane` feature in `Cargo.toml` (they belong to `metal`), so this file talks to the Objective-C runtime
//! directly: `objc_getClass`, `sel_registerName` and `objc_msgSend`, exactly as `objc2` does underneath. `objc_msgSend` is called through
//! a function pointer of the exact signature of each message (never as a variadic function), which is the documented way to use it on
//! arm64 and x86-64. Foundation and Core ML are opened with `dlopen`, so nothing is linked and a missing framework is an error value.
//! Replacing this layer by `objc2` / `objc2-core-ml` is a mechanical change once the dependencies are added (see the package report).
//! Verification level: L0 (type-checked for `aarch64-apple-darwin`, never run).

use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::OnceLock;

/// `id` / `Class`.
pub type Id = *mut c_void;
/// `SEL`.
pub type Sel = *const c_void;

#[link(name = "objc")]
extern "C" {
    fn objc_getClass(name: *const c_char) -> Id;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

extern "C" {
    fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn sysctlbyname(
        name: *const c_char,
        old: *mut c_void,
        old_len: *mut usize,
        new: *mut c_void,
        new_len: usize,
    ) -> c_int;
}

/// `RTLD_NOW`.
const RTLD_NOW: c_int = 2;

/// Address of `objc_msgSend` (used by [`msg_send!`]).
pub fn msg_send_ptr() -> *const c_void {
    objc_msgSend as *const c_void
}

/// Sends a message: `msg_send!(receiver, selector, (ArgTypes...) -> ReturnType, args...)`. Must be used inside `unsafe`; the caller vouches
/// that the argument and return types are those of the Objective-C method.
macro_rules! msg_send {
    ($receiver:expr, $selector:expr, ($($arg_ty:ty),*) -> $ret:ty $(, $arg:expr)* $(,)?) => {{
        let function = std::mem::transmute::<*const std::ffi::c_void, unsafe extern "C" fn($crate::backend::npu::ane::objc::Id, $crate::backend::npu::ane::objc::Sel $(, $arg_ty)*) -> $ret>(
            $crate::backend::npu::ane::objc::msg_send_ptr(),
        );
        function($receiver, $selector $(, $arg)*)
    }};
}
pub(crate) use msg_send;

/// `sel_registerName`.
pub fn sel(name: &CStr) -> Sel {
    // SAFETY: `name` is NUL-terminated; registering a selector has no preconditions.
    unsafe { sel_registerName(name.as_ptr()) }
}

/// The class object named `name`.
pub fn class(name: &CStr) -> Result<Id, String> {
    // SAFETY: `name` is NUL-terminated.
    let id = unsafe { objc_getClass(name.as_ptr()) };
    if id.is_null() {
        Err(format!(
            "Objective-C class {} not found (framework not loaded?)",
            name.to_string_lossy()
        ))
    } else {
        Ok(id)
    }
}

/// The frameworks the probes need, opened once.
pub struct Frameworks {
    /// Handle of CoreML.framework (for `dlsym`).
    pub core_ml: *mut c_void,
}

// SAFETY: the handle is an opaque token returned by dlopen, valid for the life of the process and only used with dlsym.
unsafe impl Send for Frameworks {}
unsafe impl Sync for Frameworks {}

/// Opens Foundation and CoreML (idempotent).
pub fn frameworks() -> Result<&'static Frameworks, String> {
    static FRAMEWORKS: OnceLock<Result<Frameworks, String>> = OnceLock::new();
    FRAMEWORKS
        .get_or_init(|| {
            let open = |path: &CStr| {
                // SAFETY: `path` is NUL-terminated.
                let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
                if handle.is_null() {
                    Err(format!("cannot open {}", path.to_string_lossy()))
                } else {
                    Ok(handle)
                }
            };
            open(c"/System/Library/Frameworks/Foundation.framework/Foundation")?;
            let core_ml = open(c"/System/Library/Frameworks/CoreML.framework/CoreML")?;
            Ok(Frameworks { core_ml })
        })
        .as_ref()
        .map_err(Clone::clone)
}

impl Frameworks {
    /// Address of an exported C symbol of CoreML.framework, or null.
    pub fn symbol(&self, name: &CStr) -> *mut c_void {
        // SAFETY: `core_ml` is a live dlopen handle and `name` is NUL-terminated.
        unsafe { dlsym(self.core_ml, name.as_ptr()) }
    }
}

/// An autorelease pool; objects returned autoreleased by the framework stay valid until it is dropped.
pub struct Pool(*mut c_void);

impl Pool {
    pub fn new() -> Self {
        // SAFETY: pushing a pool has no preconditions; it is popped in `drop` on the same thread.
        Self(unsafe { objc_autoreleasePoolPush() })
    }
}

impl Default for Pool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // SAFETY: `self.0` came from `objc_autoreleasePoolPush` and is popped once.
        unsafe { objc_autoreleasePoolPop(self.0) };
    }
}

/// An owned (+1) Objective-C object, released on drop.
pub struct Obj(Id);

impl Obj {
    /// Takes ownership of `id` (the result of `alloc`/`init`, `new`, `copy`); `None` for nil.
    pub fn owned(id: Id) -> Option<Self> {
        (!id.is_null()).then_some(Self(id))
    }

    pub fn id(&self) -> Id {
        self.0
    }
}

impl Drop for Obj {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a live object owned by this wrapper; `release` takes no arguments and returns void.
        unsafe { msg_send!(self.0, sel(c"release"), () -> ()) };
    }
}

/// `[NSString stringWithUTF8String:]` (autoreleased: valid inside the current [`Pool`]).
pub fn ns_string(text: &str) -> Result<Id, String> {
    let c = std::ffi::CString::new(text).map_err(|e| format!("string contains NUL: {e}"))?;
    let class = class(c"NSString")?;
    // SAFETY: `stringWithUTF8String:` takes a NUL-terminated C string and returns an NSString*.
    let id = unsafe {
        msg_send!(class, sel(c"stringWithUTF8String:"), (*const c_char) -> Id, c.as_ptr())
    };
    if id.is_null() {
        Err("NSString creation failed".to_owned())
    } else {
        Ok(id)
    }
}

/// The UTF-8 content of an NSString (empty for nil).
///
/// # Safety
/// `ns_string` must be nil or a live `NSString*`.
pub unsafe fn to_rust_string(ns_string: Id) -> String {
    if ns_string.is_null() {
        return String::new();
    }
    // `UTF8String` returns a NUL-terminated buffer owned by the string, valid while the string lives.
    let p = msg_send!(ns_string, sel(c"UTF8String"), () -> *const c_char);
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// `[error localizedDescription]` as a Rust string (a placeholder for nil).
///
/// # Safety
/// `error` must be nil or a live `NSError*`.
pub unsafe fn error_message(error: Id) -> String {
    if error.is_null() {
        return "unknown error (no NSError)".to_owned();
    }
    // `localizedDescription` returns an NSString*.
    to_rust_string(msg_send!(error, sel(c"localizedDescription"), () -> Id))
}

/// `sysctlbyname` as a string (for `machdep.cpu.brand_string`).
pub fn sysctl_string(name: &CStr) -> Option<String> {
    let mut len: usize = 0;
    // SAFETY: first call only asks for the length; the second fills a buffer of exactly that size.
    unsafe {
        if sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        ) != 0
            || len == 0
        {
            return None;
        }
        let mut buf = vec![0u8; len];
        if sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
        buf.truncate(len);
        while buf.last() == Some(&0) {
            buf.pop();
        }
        Some(String::from_utf8_lossy(&buf).into_owned())
    }
}

/// `sysctlbyname` as a 64-bit integer (for `hw.memsize`).
pub fn sysctl_u64(name: &CStr) -> Option<u64> {
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: the buffer is a u64 and `len` says so.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0).then_some(value)
}
