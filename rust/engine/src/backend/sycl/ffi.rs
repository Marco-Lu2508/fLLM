//! Intel GPU backend through SYCL / oneAPI: ffi. Owner: package 9 (extern "C" entry points of rust/engine/sycl/*.cpp).
//!
//! The C interface is declared once in `sycl_ffi!`. When `build.rs` found a DPC++ compiler (`icpx`) and built `libfllmsycl`, it sets the
//! cfg `fllm_sycl_native` and the macro emits an `extern "C"` block; otherwise (no oneAPI on the build machine, `cargo check`) every
//! function becomes a Rust stub that reports failure (`-1`, null, 0), so that the crate still compiles and links and `--device sycl`
//! reports "built without the SYCL runtime". Mirror of `rust/engine/sycl/fllm_sycl.h`; keep the two in step.
//! Level of verification: L0 (compiles; never run on an Intel GPU).
#![allow(unexpected_cfgs)]

use std::ffi::{c_char, c_int, c_void};

/// Opaque `FllmSyclStream`.
#[repr(C)]
pub struct RawStream {
    _private: [u8; 0],
}

/// Mirror of `FllmSyclDeviceInfo` (only ever append fields at the end, on both sides).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DeviceProps {
    pub name: [c_char; 256],
    pub driver: [c_char; 128],
    pub arch: [c_char; 64],
    pub total_memory: u64,
    pub free_memory: u64,
    pub compute_units: i32,
    pub eu_count: i32,
    pub max_wg_size: i32,
    pub local_mem_bytes: i32,
    pub is_gpu: i32,
    pub is_integrated: i32,
    pub has_fp16: i32,
    pub has_sg16: i32,
    pub xmx_n16: i32,
    pub xmx_n8: i32,
    pub level_zero: i32,
    pub vendor_id: u32,
    pub max_alloc: u64,
}

const _: () = assert!(std::mem::size_of::<DeviceProps>() == 520);

impl DeviceProps {
    pub fn zeroed() -> Self {
        // SAFETY: plain-old-data struct of integers; all-zero is a valid value.
        unsafe { std::mem::zeroed() }
    }

    fn text(field: &[c_char]) -> String {
        let bytes: Vec<u8> = field.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn name_text(&self) -> String {
        Self::text(&self.name)
    }

    pub fn driver_text(&self) -> String {
        Self::text(&self.driver)
    }

    pub fn arch_text(&self) -> String {
        Self::text(&self.arch)
    }
}

/// True when the SYCL kernels were compiled into this binary (`build.rs` found `icpx`).
pub const NATIVE: bool = cfg!(fllm_sycl_native);

/// What a stub returns for a failed call (unused when the native library is linked).
#[cfg_attr(fllm_sycl_native, allow(dead_code))]
trait Absent {
    fn absent() -> Self;
}
impl Absent for c_int {
    fn absent() -> Self {
        -1
    }
}
impl Absent for usize {
    fn absent() -> Self {
        0
    }
}
impl Absent for () {
    fn absent() {}
}
impl<T> Absent for *mut T {
    fn absent() -> Self {
        std::ptr::null_mut()
    }
}
impl<T> Absent for *const T {
    fn absent() -> Self {
        std::ptr::null()
    }
}

#[cfg_attr(fllm_sycl_native, allow(unused_macros))]
macro_rules! ret_ty {
    () => {
        ()
    };
    ($t:ty) => {
        $t
    };
}

macro_rules! sycl_ffi {
    ($( fn $name:ident($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)?; )*) => {
        #[cfg(fllm_sycl_native)]
        extern "C" {
            $( pub fn $name($($arg: $ty),*) $(-> $ret)?; )*
        }
        $(
            #[cfg(not(fllm_sycl_native))]
            #[allow(unused_variables, clippy::too_many_arguments, clippy::missing_safety_doc)]
            pub unsafe fn $name($($arg: $ty),*) $(-> $ret)? {
                <ret_ty!($($ret)?) as Absent>::absent()
            }
        )*
    };
}

sycl_ffi! {
    fn fllm_sycl_last_error() -> *const c_char;
    fn fllm_sycl_device_count() -> c_int;
    fn fllm_sycl_device_info(dev: c_int, out: *mut DeviceProps) -> c_int;
    fn fllm_sycl_malloc(dev: c_int, ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn fllm_sycl_free(dev: c_int, ptr: *mut c_void) -> c_int;
    fn fllm_sycl_h2d_sync(dev: c_int, dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn fllm_sycl_memset(dev: c_int, dst: *mut c_void, value: c_int, bytes: usize) -> c_int;
    fn fllm_sycl_set_exact(on: c_int) -> c_int;
    fn fllm_sycl_is_exact() -> c_int;
    fn fllm_sycl_stream_create(dev: c_int) -> *mut RawStream;
    fn fllm_sycl_stream_destroy(st: *mut RawStream);
    fn fllm_sycl_stream_sync(st: *mut RawStream) -> c_int;
    fn fllm_sycl_upload(st: *mut RawStream, dst: *mut c_void, src_host: *const c_void, bytes: usize) -> c_int;
    fn fllm_sycl_download(st: *mut RawStream, dst_host: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn fllm_sycl_copy(st: *mut RawStream, dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn fllm_sycl_zero(st: *mut RawStream, ptr: *mut c_void, bytes: usize) -> c_int;
    fn fllm_sycl_weight_bytes(kind: c_int, rows: usize, cols: usize) -> usize;
    fn fllm_sycl_row_bytes(kind: c_int, cols: usize) -> usize;
    fn fllm_sycl_raw_bytes(kind: c_int, rows: usize, cols: usize) -> usize;
    fn fllm_sycl_pack_weight(kind: c_int, rows: usize, cols: usize, raw: *const c_void, raw_bytes: usize, out: *mut c_void) -> c_int;
    fn fllm_sycl_linear(
        st: *mut RawStream,
        w: *const c_void,
        kind: c_int,
        rows: c_int,
        cols: c_int,
        x: *const f32,
        y: *mut f32,
        bias: *const f32,
        batch: c_int,
    ) -> c_int;
    fn fllm_sycl_moe_linear(
        st: *mut RawStream,
        tab: *const *const c_void,
        btab: *const *const f32,
        kind: c_int,
        ids: *const i32,
        x: *const f32,
        y: *mut f32,
        rows: c_int,
        cols: c_int,
        slots: c_int,
        x_div: c_int,
    ) -> c_int;
    fn fllm_sycl_embedding(st: *mut RawStream, w: *const c_void, kind: c_int, ids_host: *const i64, n: c_int, out: *mut f32, cols: c_int) -> c_int;
    fn fllm_sycl_rms_norm(st: *mut RawStream, x: *const f32, w: *const f32, out: *mut f32, rows: c_int, cols: c_int, eps: f32) -> c_int;
    fn fllm_sycl_layer_norm(
        st: *mut RawStream,
        x: *const f32,
        w: *const f32,
        b: *const f32,
        out: *mut f32,
        rows: c_int,
        cols: c_int,
        eps: f32,
    ) -> c_int;
    fn fllm_sycl_glu(st: *mut RawStream, gate: *const f32, up: *const f32, out: *mut f32, count: usize, mode: c_int) -> c_int;
    fn fllm_sycl_swiglu_oai(st: *mut RawStream, gate: *const f32, up: *const f32, out: *mut f32, count: usize, alpha: f32, limit: f32) -> c_int;
    fn fllm_sycl_residual(st: *mut RawStream, x: *mut f32, res: *const f32, count: usize, scale: f32) -> c_int;
    fn fllm_sycl_rope(
        st: *mut RawStream,
        q: *mut f32,
        k: *mut f32,
        pos_host: *const i32,
        rows: c_int,
        cos_dev: *const f32,
        sin_dev: *const f32,
        heads: c_int,
        kv_heads: c_int,
        dim: c_int,
        half_rot: c_int,
        neox: c_int,
    ) -> c_int;
    fn fllm_sycl_kv_write(
        st: *mut RawStream,
        src: *const f32,
        pool: *mut u16,
        slots_host: *const i32,
        rows: c_int,
        kv: c_int,
        kind: c_int,
        group: c_int,
        scale_base_words: c_int,
    ) -> c_int;
    fn fllm_sycl_attention(
        st: *mut RawStream,
        q: *const f32,
        kpool: *const u16,
        vpool: *const u16,
        out: *mut f32,
        tables_host: *const i64,
        table_entries: usize,
        lengths_host: *const i64,
        rows: c_int,
        sinks_dev: *const f32,
        alibi_dev: *const f32,
        heads: c_int,
        kv_heads: c_int,
        dk: c_int,
        dv: c_int,
        stride: c_int,
        block_size: c_int,
        kind: c_int,
        kgroup: c_int,
        vgroup: c_int,
        kscale_base: c_int,
        vscale_base: c_int,
        scale: f32,
        softcap: f32,
        window: c_int,
        chunk: c_int,
    ) -> c_int;
    fn fllm_sycl_argmax(st: *mut RawStream, logits: *const f32, rows: c_int, vocab: c_int, host_out: *mut i32) -> c_int;
    fn fllm_sycl_gather_rows(
        st: *mut RawStream,
        dst: *mut f32,
        src: *const f32,
        idx_host: *const i32,
        w_host: *const f32,
        n: c_int,
        row_len: c_int,
    ) -> c_int;
    fn fllm_sycl_scatter_add_rows(
        st: *mut RawStream,
        dst: *mut f32,
        src: *const f32,
        idx_host: *const i32,
        w_host: *const f32,
        n: c_int,
        row_len: c_int,
    ) -> c_int;
    fn fllm_sycl_moe_route(
        st: *mut RawStream,
        logits: *const f32,
        ids: *mut i32,
        w: *mut f32,
        rows: c_int,
        experts: c_int,
        k: c_int,
        normalize: c_int,
        scale: f32,
    ) -> c_int;
    fn fllm_sycl_moe_combine(st: *mut RawStream, out: *mut f32, y: *const f32, w: *const f32, tokens: c_int, k: c_int, hidden: c_int) -> c_int;
    fn fllm_sycl_split_q_gate(
        st: *mut RawStream,
        qg: *const f32,
        q: *mut f32,
        gate: *mut f32,
        rows: c_int,
        heads: c_int,
        dk: c_int,
        per_head: c_int,
    ) -> c_int;
    fn fllm_sycl_sigmoid_mul(st: *mut RawStream, x: *mut f32, g: *const f32, n: usize) -> c_int;
}

/// The last error text of this thread, or why the runtime is missing.
pub fn last_error() -> String {
    if !NATIVE {
        return "this build has no SYCL runtime (build.rs found no DPC++ compiler; install oneAPI and rebuild with --features sycl)"
            .to_owned();
    }
    // SAFETY: the library returns a NUL-terminated buffer owned by thread-local storage.
    unsafe { std::ffi::CStr::from_ptr(fllm_sycl_last_error()) }
        .to_string_lossy()
        .into_owned()
}
