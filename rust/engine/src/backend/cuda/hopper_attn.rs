//! Hopper and FlashInfer attention: FA3 prefill, FlashInfer variants (sliding window, soft cap, sinks, ragged batches), tensor-core decode and
//! cascade attention (cuda/fa3_prefill.cu, cuda/fi_attn.cu). The switches are read from the environment at start
//! (`FLLM_ATTN_BACKEND`, `FLLM_ATTN_FI`, `FLLM_ATTN_DECODE`, `FLLM_CASCADE_ATTN`); `selftest-gpu` flips them at run time to compare the paths on
//! one device.
use std::ffi::c_int;

extern "C" {
    fn fllm_cuda_attn_get(which: c_int) -> c_int;
    fn fllm_cuda_attn_set(which: c_int, value: c_int) -> c_int;
}

/// A run-time switch of the attention paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnSwitch {
    /// `FLLM_ATTN_FI`: 1 sends prompt runs with sliding window / soft cap / sinks and batches of several runs through FlashInfer (0 and auto: not).
    Fi = 0,
    /// `FLLM_ATTN_DECODE`: 0 = builtin (attention4.cu), 1 = flashinfer (tensor-core decode).
    Decode = 1,
    /// `FLLM_CASCADE_ATTN`: 1 = decode rows that share a prefix of cache pages read it once.
    Cascade = 2,
    /// `FLLM_ATTN_BACKEND`: 0 = auto, 1 = fa2, 2 = fa3.
    Backend = 3,
}

/// The attention path a call took (`attn_last_path`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnPath {
    /// The built-in kernels (attention3.cu / attention4.cu / attention2.cu).
    Builtin,
    /// FlashAttention-3 (sm_90a).
    Fa3,
    /// FlashInfer FA2 with the variants, several runs per launch, or decode.
    FlashInfer,
    /// Cascade attention: shared prefix and suffix states merged.
    Cascade,
}

/// What this build can run on the current device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnCaps {
    pub fa3: bool,
    pub flashinfer: bool,
}

/// Sets a switch and returns its previous value.
pub fn set_attn_switch(which: AttnSwitch, value: i32) -> i32 {
    // SAFETY: plain store into a process-wide switch.
    unsafe { fllm_cuda_attn_set(which as c_int, value) }
}

/// Forgets the path of the last call on this thread (call before the operation, read [`attn_last_path`] after it).
pub fn reset_attn_last_path() {
    // SAFETY: plain store into a thread-local record.
    unsafe { fllm_cuda_attn_set(4, 0) };
}

pub fn attn_last_path() -> AttnPath {
    // SAFETY: plain query of a thread-local record.
    match unsafe { fllm_cuda_attn_get(4) } {
        1 => AttnPath::Fa3,
        2 => AttnPath::FlashInfer,
        3 => AttnPath::Cascade,
        _ => AttnPath::Builtin,
    }
}

pub fn attn_caps() -> AttnCaps {
    // SAFETY: plain query.
    let bits = unsafe { fllm_cuda_attn_get(5) };
    AttnCaps { fa3: bits & 1 != 0, flashinfer: bits & 2 != 0 }
}
