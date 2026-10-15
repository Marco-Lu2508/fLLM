//! x86 extensions: AVX-512 (VNNI, BF16) and AMX (INT8/BF16) kernels, AVX-VNNI-256 variants, and the
//! CPU-family policy that chooses between them. Owner: package 14.
//!
//! Everything below the facade is compiled only with the `avx512` Cargo feature on x86-64; without
//! it nothing here is reachable and the dispatch in `cpu_kernels`, `cpu_quantized`, `cpu_float`
//! and `cpu_delta` is exactly what it was. Dispatch is by runtime detection and obeys the tier cap
//! (`FLLM_CPU_ISA`): `avx512-bf16` enables the `vdpbf16ps` kernels, `amx` the tile kernels for
//! prompts; `avx512-vnni` and below leave both off.
//!
//! Modules (sub-modules live in `src/cpu_avx512/`):
//! - `ident`: CPUID family and model, features Rust cannot report yet (AMX), the policy per
//!   micro-architecture, and the Linux permission request for AMX tile data.
//! - `codebook`: IQ4_NL and MXFP4 projections on 512-bit vectors (AVX-512 VNNI and BW) and on
//!   256-bit AVX-VNNI / AVX-VNNI-INT8, none of which the AVX2 codebook kernel had.
//! - `kextra`: Q2_K, Q3_K and IQ4_XS panels on 512-bit vectors (they had 256-bit EVEX only).
//! - `float`: F32/F16/BF16 projections on 512-bit vectors and with `vdpbf16ps`.
//! - `delta`: the DeltaNet state update on 512-bit vectors.
//! - `vnni_gemm`: prompts on 16-row decoded panels with the rows in the lanes of a 512-bit
//!   `vpdpbusd` (K formats, IQ4_XS, IQ4_NL, MXFP4) for AVX-512 VNNI CPUs, with or without AMX.
//! - `amx`, `amx_decode`, `amx_gemm`: AMX-INT8 / AMX-BF16 prompt kernels (inline assembly: the
//!   tile intrinsics are unstable in Rust) over 16-row weight panels decoded on the fly.
//!
//! Test level: every kernel here is at most L0 (compiled; the AVX2-executable parts, i.e. the
//! panel decoders, layout transforms, the AVX-VNNI-256 kernels and scalar references, are
//! unit-tested, and the AMX drivers run against an emulation of the tile instructions).

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod amx;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod amx_decode;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod amx_gemm;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod codebook;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod delta;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod float;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod ident;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod kextra;
#[cfg(all(test, target_arch = "x86_64", feature = "avx512"))]
mod model;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod vnni_gemm;

/// Whether the extension kernels are compiled into this build.
pub const fn compiled() -> bool {
    cfg!(all(target_arch = "x86_64", feature = "avx512"))
}

/// One line about the CPU class, features and policy for logs and `selftest-cpu`; `None` when
/// the extension kernels are not compiled in.
pub fn describe() -> Option<String> {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    {
        Some(ident::describe())
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
    {
        None
    }
}

/// The prompt path of `cpu_quantized::linear_mixed` for `batch` tokens: AMX tiles when the `amx`
/// tier is active, else the 16-row VNNI GEMM for the formats it takes (`vnni_gemm::wanted`);
/// `None` leaves the call to the streaming and AVX2 paths. The caller has validated shapes.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) fn prompt_linear_group(
    projections: &[crate::cpu_quantized::Projection<'_>],
    kinds: &[u32],
    x: &[f32],
    cols: usize,
    batch: usize,
) -> Option<Vec<Vec<f32>>> {
    if amx_gemm::wanted(batch) {
        if let Some(out) = amx_gemm::linear_group(projections, kinds, x, cols, batch) {
            return Some(out);
        }
    }
    if vnni_gemm::wanted(batch, kinds) {
        return vnni_gemm::linear_group(projections, kinds, x, cols, batch);
    }
    None
}

/// `avx512-bf16` tier: the CPU has AVX512-BF16.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) fn tier_bf16_supported() -> bool {
    ident::features().avx512bf16
}

/// `amx` tier: AMX-TILE and AMX-INT8, tile state enabled by the OS and (Linux) permission
/// granted. The first call asks the kernel for the permission.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) fn tier_amx_supported() -> bool {
    ident::amx_ready()
}

/// 512-bit integer dot kernels allowed by the CPU-family policy (default on everywhere;
/// `FLLM_X86_WIDE_INT=0` turns them off).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) fn wide_int_allowed() -> bool {
    ident::policy().wide_int
}

/// Whether the CPU-family policy and the active tier allow the wide floating-point kernels
/// (512-bit FMA); false on CPUs where 512-bit operations run as two 256-bit halves.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) fn wide_float_allowed() -> bool {
    ident::policy().wide_float
        && crate::cpu_kernels::tier_enabled(crate::cpu_kernels::CpuTier::Avx512Bw)
}

/// The DeltaNet state update on 512-bit vectors; `false` if not applicable (the caller then runs
/// its AVX2 or scalar update).
///
/// # Safety
/// As `delta::step`: `s`, `o`, `vectors` valid for `dims`.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) unsafe fn delta_step(
    s: *mut f32,
    dims: (usize, usize),
    vectors: (&[f32], &[f32], &[f32]),
    scalars: (f32, f32),
    o: *mut f32,
) -> bool {
    wide_float_allowed()
        && delta::supports(dims.1)
        && delta::step_dispatch(s, dims, vectors, scalars, o)
}

/// The 512-bit prompt GEMMs run directly, whatever their format lists and batch thresholds say,
/// for `selftest-cpu`: the 16-row VNNI GEMM when the `avx512-vnni` tier is enabled and the AMX
/// GEMM when the `amx` tier is. One entry per path that applies, with its name; empty when no
/// path applies or `kinds` has a format without a decoder. The caller has validated shapes.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) fn direct_prompt_paths(
    projections: &[crate::cpu_quantized::Projection<'_>],
    kinds: &[u32],
    x: &[f32],
    cols: usize,
    batch: usize,
) -> Vec<(&'static str, Vec<Vec<f32>>)> {
    use crate::cpu_kernels::{tier_enabled, CpuTier};
    let mut paths = Vec::new();
    if tier_enabled(CpuTier::Avx512Vnni) && wide_int_allowed() {
        if let Some(out) = vnni_gemm::linear_group(projections, kinds, x, cols, batch) {
            paths.push(("vnni_gemm", out));
        }
    }
    if tier_enabled(CpuTier::Amx) {
        if let Some(out) = amx_gemm::linear_group(projections, kinds, x, cols, batch) {
            paths.push(("amx_gemm", out));
        }
    }
    paths
}
