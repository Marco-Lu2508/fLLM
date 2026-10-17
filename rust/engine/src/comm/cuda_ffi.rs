//! The C functions of `cuda/comm_kernels.cu`.
use std::ffi::{c_char, c_int, c_void, CStr};

extern "C" {
    pub fn fllm_comm_error_string(code: c_int) -> *const c_char;
    pub fn fllm_comm_device_count(count: *mut c_int) -> c_int;
    pub fn fllm_comm_set_device(dev: c_int) -> c_int;
    pub fn fllm_comm_malloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    pub fn fllm_comm_free(ptr: *mut c_void) -> c_int;
    pub fn fllm_comm_h2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    pub fn fllm_comm_d2h(dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    pub fn fllm_comm_stream_create(stream: *mut *mut c_void) -> c_int;
    pub fn fllm_comm_stream_destroy(stream: *mut c_void) -> c_int;
    pub fn fllm_comm_stream_sync(stream: *mut c_void) -> c_int;
    pub fn fllm_comm_can_access_peer(dev: c_int, peer: c_int, can: *mut c_int) -> c_int;
    pub fn fllm_comm_p2p_rank(dev: c_int, peer: c_int, rank: *mut c_int) -> c_int;
    pub fn fllm_comm_p2p_native_atomics(dev: c_int, peer: c_int, supported: *mut c_int) -> c_int;
    pub fn fllm_comm_enable_peer_access(dev: c_int, peer: c_int) -> c_int;
    pub fn fllm_comm_disable_peer_access(dev: c_int, peer: c_int) -> c_int;
    pub fn fllm_comm_peer_copy_async(dst: *mut c_void, dst_dev: c_int, src: *const c_void, src_dev: c_int, bytes: usize, stream: *mut c_void) -> c_int;
    pub fn fllm_comm_reduce_add_f32(dst: *mut f32, src: *const f32, n: usize, stream: *mut c_void) -> c_int;
    pub fn fllm_comm_custom_all_reduce_f32(out: *mut f32, peer_ptrs: *const *const f32, num_peers: c_int, n: usize, stream: *mut c_void) -> c_int;
}

/// `Ok` for status 0, else the CUDA error text.
pub fn check(what: &str, code: c_int) -> Result<(), String> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: fllm_comm_error_string returns a static NUL-terminated string.
    let text = unsafe { CStr::from_ptr(fllm_comm_error_string(code)) }.to_string_lossy().into_owned();
    Err(format!("{what}: CUDA error {code} ({text})"))
}

pub fn device_count() -> Result<usize, String> {
    let mut n: c_int = 0;
    // SAFETY: plain query with a valid out pointer.
    check("cudaGetDeviceCount", unsafe { fllm_comm_device_count(&mut n) })?;
    Ok(n.max(0) as usize)
}

pub fn set_device(dev: usize) -> Result<(), String> {
    // SAFETY: no pointers involved.
    check("cudaSetDevice", unsafe { fllm_comm_set_device(dev as c_int) })
}

/// Device memory owned by one rank.
pub struct DeviceMem {
    pub ptr: *mut c_void,
    pub bytes: usize,
    pub device: usize,
}

// SAFETY: a raw device address; the owner frees it on the thread that set the device.
unsafe impl Send for DeviceMem {}
unsafe impl Sync for DeviceMem {}

impl DeviceMem {
    pub fn new(device: usize, bytes: usize) -> Result<Self, String> {
        set_device(device)?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out pointer; bytes > 0 is enforced by max(1).
        check("cudaMalloc", unsafe { fllm_comm_malloc(&mut ptr, bytes.max(1)) })?;
        Ok(Self { ptr, bytes: bytes.max(1), device })
    }

    pub fn addr(&self) -> u64 {
        self.ptr as u64
    }

    pub fn upload(&self, data: &[u8]) -> Result<(), String> {
        if data.len() > self.bytes {
            return Err("upload larger than the device buffer".into());
        }
        set_device(self.device)?;
        // SAFETY: the buffer holds at least data.len() bytes.
        check("cudaMemcpy H2D", unsafe { fllm_comm_h2d(self.ptr, data.as_ptr().cast(), data.len()) })
    }

    pub fn download(&self, data: &mut [u8]) -> Result<(), String> {
        if data.len() > self.bytes {
            return Err("download larger than the device buffer".into());
        }
        set_device(self.device)?;
        // SAFETY: the buffer holds at least data.len() bytes.
        check("cudaMemcpy D2H", unsafe { fllm_comm_d2h(data.as_mut_ptr().cast(), self.ptr, data.len()) })
    }
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        let _ = set_device(self.device);
        // SAFETY: allocated by fllm_comm_malloc and freed once.
        unsafe { fllm_comm_free(self.ptr) };
    }
}

/// A non-blocking stream of one rank.
pub struct RankStream {
    pub raw: *mut c_void,
    pub device: usize,
}

// SAFETY: used by the rank's own thread; the handle is a plain pointer.
unsafe impl Send for RankStream {}
unsafe impl Sync for RankStream {}

impl RankStream {
    pub fn new(device: usize) -> Result<Self, String> {
        set_device(device)?;
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out pointer.
        check("cudaStreamCreate", unsafe { fllm_comm_stream_create(&mut raw) })?;
        Ok(Self { raw, device })
    }

    pub fn handle(&self) -> u64 {
        self.raw as u64
    }

    pub fn sync(&self) -> Result<(), String> {
        set_device(self.device)?;
        // SAFETY: the stream is alive.
        check("cudaStreamSynchronize", unsafe { fllm_comm_stream_sync(self.raw) })
    }
}

impl Drop for RankStream {
    fn drop(&mut self) {
        let _ = set_device(self.device);
        // SAFETY: created by fllm_comm_stream_create and destroyed once.
        unsafe { fllm_comm_stream_destroy(self.raw) };
    }
}


pub fn enable_peer_access(dev: usize, peer: usize) -> Result<(), String> {
    // SAFETY: integer device indices
    check("cudaDeviceEnablePeerAccess", unsafe { fllm_comm_enable_peer_access(dev as c_int, peer as c_int) })
}

#[allow(clippy::not_unsafe_ptr_arg_deref)] // thin wrapper; the callers hold valid device pointers
pub fn peer_copy_async(dst: *mut c_void, dst_dev: usize, src: *const c_void, src_dev: usize, bytes: usize, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: caller guarantees pointers and stream are valid
    check("cudaMemcpyPeerAsync", unsafe {
        fllm_comm_peer_copy_async(dst, dst_dev as c_int, src, src_dev as c_int, bytes, stream)
    })
}

#[allow(clippy::not_unsafe_ptr_arg_deref)] // thin wrapper; the callers hold valid device pointers
pub fn reduce_add_f32(dst: *mut f32, src: *const f32, n: usize, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: caller ensures dst and src hold at least n floats
    check("reduce_add_f32", unsafe { fllm_comm_reduce_add_f32(dst, src, n, stream) })
}

#[allow(clippy::not_unsafe_ptr_arg_deref)] // thin wrapper; the callers hold valid device pointers
pub fn custom_all_reduce_f32(out: *mut f32, peer_ptrs: &[*const f32], n: usize, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: peer_ptrs holds valid device pointers
    check("custom_all_reduce_f32", unsafe {
        fllm_comm_custom_all_reduce_f32(out, peer_ptrs.as_ptr(), peer_ptrs.len() as c_int, n, stream)
    })
}
