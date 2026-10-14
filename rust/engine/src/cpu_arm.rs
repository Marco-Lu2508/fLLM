//! ARM CPU kernels (NEON, dotprod, i8mm, bf16, SVE) for the prompt GEMMs and matvecs.
//!
//! The AArch64 ladder (`CpuTier::Neon` < `NeonDotprod` < `NeonI8mm` < `NeonSve`) and the Q8_0, Q4_0/1,
//! Q5_0/1, Q4_K, Q5_K and Q6_K projections live in `cpu_kernels.rs` and `cpu_quantized.rs`
//! (they ran on hardware before this package). This module adds what was missing there:
//!
//! | Sub-module | Contents | Runs on |
//! | --- | --- | --- |
//! | `features` | runtime detection of the extensions (dotprod, i8mm, bf16, fp16, fhm, sve, sve2, SVE vector length) | all (false off AArch64) |
//! | `formats` | Q2_K, Q3_K, IQ4_NL, IQ4_XS and MXFP4: tables, scalar block decoders, the `Group` panel record | all |
//! | `emu` | scalar models of `sdot`, `smmla`, `tbl` and the 64-bit zips | all |
//! | `model` | the NEON kernels' dataflow written with `emu` (layout / scale / lane checks run on x86) | all (tests) |
//! | `quant` | NEON `sdot` / `smmla` projections of the five formats above | AArch64 little-endian |
//! | `float` | F32 / F16 / BF16 projections (`bfdot` when FEAT_BF16 is present), f16 encode | AArch64 little-endian |
//! | `delta` | gated DeltaNet recurrence step | AArch64 little-endian |
//! | `sve` | SVE `sdot` Q8_0 matvec (tier `neon-sve`, opt-in until measured) | AArch64 Linux/Android/FreeBSD |
//!
//! Dispatch is by runtime detection (`is_aarch64_feature_detected!`) through `cpu_kernels::CpuTier`;
//! `FLLM_CPU_ISA=<tier>` caps it like on x86. NEON is the architectural baseline of AArch64,
//! so nothing here sits behind a Cargo feature (`arm-simd` is accepted and has no effect).
//!
//! Verification level of everything in this module: L0 (compiled for `aarch64-unknown-linux-gnu`
//! and `aarch64-apple-darwin`; the platform-independent parts are unit-tested on x86).
//! Run `fllm-rs selftest-cpu` and `cargo test -p fllm-engine cpu_arm` on an ARM machine.
//! See docs/cpu-arm.md.
pub mod features;
pub(crate) mod formats;

#[cfg(test)]
pub(crate) mod emu;
#[cfg(test)]
mod model;

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub(crate) mod delta;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub(crate) mod float;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub(crate) mod quant;
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    any(target_os = "linux", target_os = "android", target_os = "freebsd")
))]
pub(crate) mod sve;

/// Q8_0 decode matvec on SVE (tier `neon-sve`): rows `row0..row0 + out.len()` of `w` against `nb`
/// activation records at `x`. Returns `false` (and computes nothing) in builds without an SVE
/// kernel.
///
/// # Safety
/// As [`sve::q8_gemv`]: the caller selected the SVE tier, so the CPU has SVE vectors of 256 bits
/// or more; `x` points at `nb` records of 40 bytes.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub(crate) unsafe fn q8_gemv_sve(
    w: &[u8],
    x: *const u8,
    nb: usize,
    row0: usize,
    out: &mut [f32],
) -> bool {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
    {
        sve::q8_gemv(w, x, nb, row0, out);
        true
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
    {
        let _ = (w, x, nb, row0, out);
        false
    }
}
