//! Own CPU projection kernels and the CPU tier ladder (`CpuTier`). Dispatch is guarded at
//! runtime, never by build-host flags; `FLLM_CPU_ISA` can cap it.
//! Q8 follows the native 4-weight-row / 2-token register tile, including fp16
//! activation scales and the unsigned-weight bias in every integer accumulator.
//! Activations are packed to 8 bits; the f32 fallback in compute.rs is not, so results differ
//! between machines with and without a kernel by the 8-bit activation error.
use half::f16;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};

// Decode needs one activation row, not the duplicated prompt pair. Keeping a
// 40-byte block avoids evicting the activation vector from small CPU L1 caches.
#[repr(C)]
#[derive(Clone, Default)]
struct XSingle {
    q: [i8; 32],
    d: f32,
    init: i32,
}
// The SVE decode kernel (`cpu_arm::sve`) reads these records through a raw pointer: 32 quants,
// then the f32 scale at byte 32.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
const _: () = assert!(std::mem::size_of::<XSingle>() == 40);

#[repr(C, align(32))]
#[derive(Clone)]
pub(crate) struct XPair {
    pub(crate) q: [[i8; 32]; 2],
    pub(crate) d: [f32; 2],
    pub(crate) init: [i32; 2],
    padding: [u8; 16],
}
impl Default for XPair {
    fn default() -> Self {
        Self {
            q: [[0; 32]; 2],
            d: [0.0; 2],
            init: [0; 2],
            padding: [0; 16],
        }
    }
}
#[cfg(target_arch = "x86_64")]
/// Four weight rows of one 32-value block in the layout of the Q8_0 prompt tiles: signed
/// weights plus 128 as bytes, and one scale per row.
#[repr(C, align(32))]
#[derive(Clone)]
pub(crate) struct WPanel {
    pub(crate) q: [[u8; 32]; 4],
    pub(crate) d: [f32; 4],
    padding: [u8; 16],
}
#[cfg(target_arch = "x86_64")]
impl Default for WPanel {
    fn default() -> Self {
        Self {
            q: [[128; 32]; 4],
            d: [0.0; 4],
            padding: [0; 16],
        }
    }
}

#[cfg(target_arch = "x86_64")]
std::thread_local! {
    static Q8_W_PANEL: std::cell::RefCell<Vec<WPanel>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

/// A worker owns its panel until the broadcast joins. Taking the vector out
/// avoids a live TLS borrow and gives nested invocations separate allocations.
#[cfg(target_arch = "x86_64")]
pub(crate) struct Q8PanelScratch(Vec<WPanel>);
#[cfg(target_arch = "x86_64")]
impl Q8PanelScratch {
    pub(crate) fn new(blocks: usize) -> Self {
        let mut panel = Q8_W_PANEL.with(|slot| std::mem::take(&mut *slot.borrow_mut()));
        panel.resize(blocks, WPanel::default());
        Self(panel)
    }
}
#[cfg(target_arch = "x86_64")]
impl std::ops::Deref for Q8PanelScratch {
    type Target = [WPanel];
    fn deref(&self) -> &[WPanel] {
        &self.0
    }
}
#[cfg(target_arch = "x86_64")]
impl std::ops::DerefMut for Q8PanelScratch {
    fn deref_mut(&mut self) -> &mut [WPanel] {
        &mut self.0
    }
}
#[cfg(target_arch = "x86_64")]
impl Drop for Q8PanelScratch {
    fn drop(&mut self) {
        let panel = std::mem::take(&mut self.0);
        // Keep the larger buffer if a nested invocation already returned one.
        // During TLS teardown, simply drop this owned allocation instead.
        let _ = Q8_W_PANEL.try_with(|slot| {
            let mut retained = slot.borrow_mut();
            if panel.capacity() > retained.capacity() {
                *retained = panel;
            }
        });
    }
}

static REPACK_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turn the prompt-time weight repacking (Q4_0/Q5_0 into Q8_0 tiles, K formats into decoded
/// panels) on or off; `--no-repack` and `FLLM_NO_REPACK=1` switch it off. The packs live in
/// per-worker scratch for the duration of one projection, so this only selects which kernels
/// run; the weights themselves are never copied. Off selects the streaming kernels.
pub fn set_repack(enabled: bool) {
    REPACK_OFF.store(!enabled, std::sync::atomic::Ordering::Relaxed);
}

/// `true` unless repacking was switched off with [`set_repack`] or `FLLM_NO_REPACK`.
pub fn repack_enabled() -> bool {
    static ENV_OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env_off = *ENV_OFF.get_or_init(|| {
        std::env::var_os("FLLM_NO_REPACK").is_some_and(|v| !v.is_empty() && v != "0")
    });
    !env_off && !REPACK_OFF.load(std::sync::atomic::Ordering::Relaxed)
}

/// Kernel tiers in dispatch order. x86-64 and AArch64 each have their own ladder above
/// `Scalar`; a tier is only ever used when the CPU reports every feature it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuTier {
    Scalar,
    /// AVX2 + FMA + F16C.
    Avx2,
    /// Additionally the VEX `avxvnni` dot products.
    AvxVnni,
    /// avx512f + avx512bw + avx512vl: 512-bit `vpmaddubsw` / `vpmaddwd` kernels.
    Avx512Bw,
    /// Additionally avx512vnni: 512-bit `vpdpbusd`.
    Avx512Vnni,
    /// Additionally avx512bf16: `vdpbf16ps` kernels for BF16 weights (needs the `avx512` feature).
    Avx512Bf16,
    /// Additionally AMX-TILE/INT8 with the OS tile state and permission: tile kernels for prompts
    /// (needs the `avx512` feature).
    Amx,
    /// AArch64 Advanced SIMD: activation packing and element-wise kernels.
    Neon,
    /// FEAT_DotProd: `sdot` projection and attention kernels.
    NeonDotprod,
    /// FEAT_I8MM: `smmla` prompt tiles.
    NeonI8mm,
    /// FEAT_SVE with vectors of 256 bits or more: SVE `sdot` Q8_0 decode. Opt-in until it has
    /// run on hardware (`FLLM_ARM_SVE=1`; see `cpu_arm::sve`).
    NeonSve,
}

impl CpuTier {
    pub const ALL: [CpuTier; 11] = [
        CpuTier::Scalar,
        CpuTier::Avx2,
        CpuTier::AvxVnni,
        CpuTier::Avx512Bw,
        CpuTier::Avx512Vnni,
        CpuTier::Avx512Bf16,
        CpuTier::Amx,
        CpuTier::Neon,
        CpuTier::NeonDotprod,
        CpuTier::NeonI8mm,
        CpuTier::NeonSve,
    ];

    /// Rung on this tier's architecture ladder; 0 is scalar.
    fn level(self) -> u8 {
        match self {
            CpuTier::Scalar => 0,
            CpuTier::Avx2 | CpuTier::Neon => 1,
            CpuTier::AvxVnni | CpuTier::NeonDotprod => 2,
            CpuTier::Avx512Bw | CpuTier::NeonI8mm => 3,
            CpuTier::Avx512Vnni => 4,
            CpuTier::NeonSve => 4,
            CpuTier::Avx512Bf16 => 5,
            CpuTier::Amx => 6,
        }
    }

    /// Whether this tier belongs to the architecture the binary was built for.
    fn native(self) -> bool {
        match self {
            CpuTier::Scalar => true,
            CpuTier::Avx2
            | CpuTier::AvxVnni
            | CpuTier::Avx512Bw
            | CpuTier::Avx512Vnni
            | CpuTier::Avx512Bf16
            | CpuTier::Amx => cfg!(target_arch = "x86_64"),
            CpuTier::Neon | CpuTier::NeonDotprod | CpuTier::NeonI8mm | CpuTier::NeonSve => {
                cfg!(all(target_arch = "aarch64", target_endian = "little"))
            }
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CpuTier::Scalar => "scalar",
            CpuTier::Avx2 => "avx2",
            CpuTier::AvxVnni => "avx-vnni",
            CpuTier::Avx512Bw => "avx512bw",
            CpuTier::Avx512Vnni => "avx512-vnni",
            CpuTier::Avx512Bf16 => "avx512-bf16",
            CpuTier::Amx => "amx",
            CpuTier::Neon => "neon",
            CpuTier::NeonDotprod => "neon-dotprod",
            CpuTier::NeonI8mm => "neon-i8mm",
            CpuTier::NeonSve => "neon-sve",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tier| tier.name() == name)
    }

    /// Whether the CPU has every feature of this tier (independent of any cap).
    fn supported(self) -> bool {
        match self {
            CpuTier::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            CpuTier::Avx2 => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                    && std::arch::is_x86_feature_detected!("f16c")
            }
            #[cfg(target_arch = "x86_64")]
            CpuTier::AvxVnni => {
                CpuTier::Avx2.supported() && std::arch::is_x86_feature_detected!("avxvnni")
            }
            #[cfg(target_arch = "x86_64")]
            CpuTier::Avx512Bw => {
                CpuTier::Avx2.supported()
                    && std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vl")
            }
            #[cfg(target_arch = "x86_64")]
            CpuTier::Avx512Vnni => {
                CpuTier::Avx512Bw.supported() && std::arch::is_x86_feature_detected!("avx512vnni")
            }
            // The extension tiers exist only in builds with the `avx512` feature.
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            CpuTier::Avx512Bf16 => {
                CpuTier::Avx512Vnni.supported() && crate::cpu_avx512::tier_bf16_supported()
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            CpuTier::Amx => {
                CpuTier::Avx512Bf16.supported() && crate::cpu_avx512::tier_amx_supported()
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            CpuTier::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            CpuTier::NeonDotprod => {
                CpuTier::Neon.supported() && std::arch::is_aarch64_feature_detected!("dotprod")
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            CpuTier::NeonI8mm => {
                CpuTier::NeonDotprod.supported() && std::arch::is_aarch64_feature_detected!("i8mm")
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                any(target_os = "linux", target_os = "android", target_os = "freebsd")
            ))]
            CpuTier::NeonSve => CpuTier::NeonDotprod.supported() && crate::cpu_arm::sve::usable(),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Whether dispatch may use this tier: supported by the CPU and within the cap.
    fn enabled(self) -> bool {
        self.native() && self.level() <= tier_cap() && self.supported()
    }
}

const CAP_UNSET: u8 = u8::MAX;
const CAP_NONE: u8 = u8::MAX - 1;
static TIER_CAP: AtomicU8 = AtomicU8::new(CAP_UNSET);

/// Highest tier level dispatch may use; read once from `FLLM_CPU_ISA` (a tier name).
fn tier_cap() -> u8 {
    let cap = TIER_CAP.load(Ordering::Relaxed);
    if cap != CAP_UNSET {
        return cap;
    }
    let cap = match std::env::var("FLLM_CPU_ISA") {
        Ok(name) => match CpuTier::parse(name.trim()) {
            Some(tier) if tier.native() => tier.level(),
            _ => {
                eprintln!(
                    "[fllm-rs] ignoring FLLM_CPU_ISA={name}: no such tier on this architecture"
                );
                CAP_NONE
            }
        },
        Err(_) => CAP_NONE,
    };
    TIER_CAP.store(cap, Ordering::Relaxed);
    cap
}

/// Limit kernel dispatch to `tier` and the tiers below it on its ladder; `None` removes the
/// limit. Meant for the CPU self-test and A/B comparisons, before or between projections.
pub fn set_tier_cap(tier: Option<CpuTier>) {
    let cap = match tier {
        Some(tier) if tier.native() => tier.level(),
        Some(_) => 0,
        None => CAP_NONE,
    };
    TIER_CAP.store(cap, Ordering::Relaxed);
}

/// Tiers this CPU supports, lowest first, ignoring any cap.
pub fn supported_tiers() -> Vec<CpuTier> {
    CpuTier::ALL
        .into_iter()
        .filter(|tier| tier.native() && tier.supported())
        .collect()
}

/// Tiers this CPU supports that the `FLLM_CPU_ISA` cap allows, lowest first (all supported tiers
/// without a cap). `selftest-cpu` runs these by default.
pub fn capped_tiers() -> Vec<CpuTier> {
    let cap = tier_cap();
    supported_tiers()
        .into_iter()
        .filter(|tier| tier.level() <= cap)
        .collect()
}

/// Highest tier dispatch currently selects.
pub fn active_tier() -> CpuTier {
    supported_tiers()
        .into_iter()
        .rev()
        .find(|tier| tier.enabled())
        .unwrap_or(CpuTier::Scalar)
}

/// Whether dispatch may use `tier` now: the CPU supports it and the tier cap allows it.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) fn tier_enabled(tier: CpuTier) -> bool {
    tier.enabled()
}

/// Name of the AVX2-level x86 tier ("avx-vnni", "avx2") or "scalar". Callers outside the CPU
/// kernels read anything but "scalar" as "the x86 kernels are available", so AArch64 always
/// reports "scalar" here; see `active_tier` and `packed_simd`.
pub fn isa() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if CpuTier::Avx2.enabled() {
        if CpuTier::AvxVnni.enabled() {
            return "avx-vnni";
        }
        return "avx2";
    }
    "scalar"
}

/// Whether `q8_linear*` and `cpu_quantized::linear*` have a SIMD kernel on this CPU (and quantize
/// activations to 8 bits). Without one they return `None` and callers keep their f32 path.
pub(crate) fn packed_simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    let simd = isa() != "scalar";
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    let simd = neon_dotprod();
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little")
    )))]
    let simd = false;
    simd
}

/// AVX-512 tier of the wide kernels, chosen in front of the AVX2 tier when present.
/// `isa()` keeps naming the AVX2 tier so its dispatch (including VEX `avxvnni`) is unchanged.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Avx512 {
    /// avx512f + avx512bw + avx512vl: `vpmaddubsw` / `vpmaddwd` dots.
    Bw,
    /// Additionally avx512vnni: `vpdpbusd` on 512-bit vectors.
    Vnni,
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn avx512() -> Option<Avx512> {
    // `FLLM_X86_WIDE_INT=0` (with the `avx512` feature) keeps the 256-bit kernels.
    #[cfg(feature = "avx512")]
    if !crate::cpu_avx512::wide_int_allowed() {
        return None;
    }
    if CpuTier::Avx512Vnni.enabled() {
        Some(Avx512::Vnni)
    } else if CpuTier::Avx512Bw.enabled() {
        Some(Avx512::Bw)
    } else {
        None
    }
}

/// Baseline AArch64 SIMD, used for activation packing and f32 element-wise kernels.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub(crate) fn neon() -> bool {
    CpuTier::Neon.enabled()
}

/// NEON with `sdot` (FEAT_DotProd): the integer projection and attention kernels.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub(crate) fn neon_dotprod() -> bool {
    CpuTier::NeonDotprod.enabled()
}

/// Additionally `smmla` (FEAT_I8MM): two-row by two-token prompt tiles.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub(crate) fn neon_i8mm() -> bool {
    CpuTier::NeonI8mm.enabled()
}

/// SVE with 256-bit or wider vectors (opt-in): the Q8_0 decode matvec in `cpu_arm::sve`.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub(crate) fn neon_sve() -> bool {
    CpuTier::NeonSve.enabled()
}
pub(crate) fn encode_q8_cache_group_scalar(src: &[f32], dst: &mut [u8], scale: &mut u16) {
    assert_eq!(src.len(), dst.len());
    let amax = src.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let d = amax / 127.0f32;
    *scale = f16::from_f32(d).to_bits();
    if d == 0.0 {
        dst.fill(0);
    } else {
        for (out, &x) in dst.iter_mut().zip(src) {
            *out = (x / d).round().clamp(-127.0, 127.0) as i8 as u8;
        }
    }
}

/// Encode one existing Q8 KV-cache scale group, preserving the Rust scalar
/// format exactly. The byte slice must cover only this group's Q values.
pub(crate) fn encode_q8_cache_group(src: &[f32], dst: &mut [u8], scale: &mut u16) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if isa() != "scalar" {
        // SAFETY: `isa()` confirmed AVX2/F16C; the encoder handles any length.
        unsafe { x86::encode_q8_cache_group(src, dst, scale) };
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if neon() {
        // SAFETY: NEON was detected; the encoder handles any length.
        unsafe { neon::encode_q8_cache_group(src, dst, scale) };
        return;
    }
    encode_q8_cache_group_scalar(src, dst, scale);
}

/// Returns None on machines without AVX2/FMA; the caller keeps its portable path.
/// Packed activation rows shared by Q/K/V and gate/up projections.
#[derive(Default)]
pub struct Q8Input {
    cols: usize,
    batch: usize,
    packed: Vec<XPair>,
    single: Vec<XSingle>,
}
impl Q8Input {
    /// The activation pairs the Q8_0 prompt tiles consume (empty for a single token).
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn packed(&self) -> &[XPair] {
        &self.packed
    }

    pub fn new(x: &[f32], cols: usize) -> Self {
        let mut input = Self::default();
        input.prepare(x, cols);
        input
    }

    /// Repack into retained storage so a decoder can reuse the Q8 activation
    /// workspace between sibling projections and model layers.
    pub fn prepare(&mut self, x: &[f32], cols: usize) {
        assert!(cols > 0 && cols.is_multiple_of(32) && x.len().is_multiple_of(cols));
        let batch = x.len() / cols;
        let nb = cols / 32;
        self.cols = cols;
        self.batch = batch;
        if batch == 1 {
            self.packed.clear();
            self.single.resize(nb, XSingle::default());
            for (src, rec) in x.as_chunks::<32>().0.iter().zip(&mut self.single) {
                (rec.d, rec.init) = pack_linear_block(src, &mut rec.q);
            }
            return;
        }
        self.single.clear();
        self.packed.resize(batch.div_ceil(2) * nb, XPair::default());
        #[cfg(target_arch = "x86_64")]
        let fast = isa() != "scalar";
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        let fast = neon();
        self.packed
            .par_chunks_mut(nb)
            .enumerate()
            .for_each(|(p, dst)| {
                for (kb, rec) in dst.iter_mut().enumerate() {
                    for j in 0..2 {
                        let tok = (2 * p + j).min(batch - 1);
                        let src = &x[tok * cols + kb * 32..tok * cols + (kb + 1) * 32];
                        #[cfg(target_arch = "x86_64")]
                        if fast {
                            // SAFETY: AVX2/F16C confirmed (`fast`); `src` is one 32-value block.
                            let (d, _, sums) =
                                unsafe { x86::pack_q8_block(src, &mut rec.q[j], false) };
                            rec.d[j] = d;
                            rec.init[j] = -16 * (sums[0] + sums[1]);
                            continue;
                        }
                        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
                        if fast {
                            // SAFETY: NEON detected (`fast`); `src` is one 32-value block.
                            let (d, _, sums) =
                                unsafe { neon::pack_q8_block(src, &mut rec.q[j], false) };
                            rec.d[j] = d;
                            rec.init[j] = -16 * (sums[0] + sums[1]);
                            continue;
                        }
                        // Identical to ggml's AVX Q8_0 activation packing, including
                        // ties-to-even and scale rounding *after* quantization.
                        let amax = src.iter().map(|v| v.abs()).fold(0.0, f32::max);
                        let inv = if amax == 0.0 { 0.0 } else { 127.0 / amax };
                        rec.d[j] = f16::from_f32(amax / 127.0).to_f32();
                        for (q, &v) in rec.q[j].iter_mut().zip(src) {
                            *q = (v * inv).round_ties_even() as i8;
                        }
                        // Eight dpbusd lanes each receive -16*sum(x): total -128*sum(x).
                        rec.init[j] = -16 * rec.q[j].iter().map(|&v| v as i32).sum::<i32>();
                    }
                }
            });
    }
}
fn pack_linear_block(src: &[f32], dst: &mut [i8; 32]) -> (f32, i32) {
    #[cfg(target_arch = "x86_64")]
    if isa() != "scalar" {
        // SAFETY: AVX2/F16C confirmed; `src` is one 32-value block.
        let (d, _, sums) = unsafe { x86::pack_q8_block(src, dst, false) };
        return (d, -16 * (sums[0] + sums[1]));
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if neon() {
        // SAFETY: NEON detected; `src` is one 32-value block.
        let (d, _, sums) = unsafe { neon::pack_q8_block(src, dst, false) };
        return (d, -16 * (sums[0] + sums[1]));
    }
    let amax = src.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let inv = if amax == 0.0 { 0.0 } else { 127.0 / amax };
    for (q, &v) in dst.iter_mut().zip(src) {
        *q = (v * inv).round_ties_even() as i8;
    }
    (
        f16::from_f32(amax / 127.0).to_f32(),
        -16 * dst.iter().map(|&v| v as i32).sum::<i32>(),
    )
}
pub fn q8_linear(
    w: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    bias: Option<&[f32]>,
) -> Option<Vec<f32>> {
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if neon_dotprod() {
        return q8_linear_prepared(w, &Q8Input::new(x, cols), rows, bias);
    }
    if isa() == "scalar" {
        return None;
    }
    q8_linear_prepared(w, &Q8Input::new(x, cols), rows, bias)
}
pub fn q8_linear_prepared(
    w: &[u8],
    input: &Q8Input,
    rows: usize,
    bias: Option<&[f32]>,
) -> Option<Vec<f32>> {
    #[cfg(target_arch = "x86_64")]
    if isa() != "scalar" {
        let cols = input.cols;
        let batch = input.batch;
        let nb = cols / 32;
        let packed = &input.packed;
        assert!(rows > 0);
        assert_eq!(w.len(), rows * nb * 34);
        assert!(bias.is_none_or(|b| b.len() == rows));
        if batch == 0 {
            return Some(Vec::new());
        }
        let vnni = isa() == "avx-vnni";
        let wide = avx512();
        if batch == 1 {
            let mut output = vec![0.0; rows];
            crate::cpu_schedule::q8_decode_panels(
                &mut output,
                8,
                || (),
                |_, g, dst| {
                    if let Some(wide) = wide {
                        // SAFETY: `avx512()` confirmed the features; the scheduler checked rows and `w`.
                        unsafe {
                            match wide {
                                Avx512::Vnni => x86::gemv512_vnni(w, &input.single, g * 8, dst),
                                Avx512::Bw => x86::gemv512_bw(w, &input.single, g * 8, dst),
                            }
                        }
                    } else if vnni && dst.len() == 8 {
                        // SAFETY: AVX-VNNI confirmed; the eight panel rows lie inside the asserted weight shape.
                        unsafe { x86::gemv_vnni_zero8(w, &input.single, g * 8, dst) };
                    } else {
                        for (j, out) in dst.chunks_mut(4).enumerate() {
                            // SAFETY: `isa()` confirmed the features; the rows lie inside the asserted weight shape.
                            unsafe {
                                if vnni {
                                    if out.len() == 4 {
                                        x86::gemv_vnni_zero4(w, &input.single, g * 8 + j * 4, out);
                                    } else {
                                        x86::gemv_vnni(w, &input.single, g * 8 + j * 4, out);
                                    }
                                } else {
                                    x86::gemv_avx2(w, &input.single, g * 8 + j * 4, out);
                                }
                            }
                        }
                    }
                    if let Some(bias) = bias {
                        for (j, y) in dst.iter_mut().enumerate() {
                            *y += bias[g * 8 + j];
                        }
                    }
                },
            );
            return Some(output);
        }
        // Prompts on AMX: tile kernels over decoded 16-row panels (the `avx512` feature).
        #[cfg(feature = "avx512")]
        if crate::cpu_avx512::amx_gemm::wanted(batch) {
            let mut amx_output = vec![Vec::new()];
            let member = [Q8Projection {
                weights: w,
                rows,
                bias,
            }];
            crate::cpu_avx512::amx_gemm::q8_group(&member, packed, nb, batch, &mut amx_output);
            return amx_output.pop();
        }
        let mut output = vec![vec![0.0; batch * rows]];
        // Workers write their exclusive columns directly into token rows.
        // This removes the intermediate tiled matrix and its transpose/copy.
        crate::cpu_schedule::row_panels(
            &mut output,
            &[rows],
            batch,
            4,
            || Q8PanelScratch::new(nb),
            |panel, _, first, out| {
                let valid = (rows - first).min(4);
                // SAFETY: `isa()`/`avx512()` confirmed the features; the panel holds `nb` blocks of `valid` rows and `out` owns these columns.
                unsafe {
                    x86::pack_panel(w, first, valid, panel);
                    match wide {
                        Some(Avx512::Vnni) => x86::tile512_vnni(panel, packed, batch, out),
                        Some(Avx512::Bw) => x86::tile512_bw(panel, packed, batch, out),
                        None if vnni => x86::tile_vnni(panel, packed, batch, out),
                        None => x86::tile_avx2(panel, packed, batch, out),
                    }
                }
                for b in 0..batch {
                    for (r, y) in out.row(b).iter_mut().enumerate() {
                        *y += bias.map_or(0.0, |bias| bias[first + r]);
                    }
                }
            },
        );
        return output.pop();
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if neon_dotprod() {
        let mut output = vec![Vec::new()];
        q8_linear_group_into(
            &[Q8Projection {
                weights: w,
                rows,
                bias,
            }],
            input,
            &mut output,
        )?;
        return output.pop();
    }
    let _ = (w, input, rows, bias);
    None
}

pub struct Q8Projection<'a> {
    pub weights: &'a [u8],
    pub rows: usize,
    pub bias: Option<&'a [f32]>,
}

/// One dynamic work queue for sibling projections, without a worker rendezvous
/// between Q, K and V (or between gate and up).
pub fn q8_linear_group(projections: &[Q8Projection<'_>], input: &Q8Input) -> Option<Vec<Vec<f32>>> {
    let mut output = vec![Vec::new(); projections.len()];
    q8_linear_group_into(projections, input, &mut output)?;
    Some(output)
}

/// Evaluate sibling Q8 projections into retained output vectors.
/// `output` must have one vector per projection; capacities are preserved.
pub(crate) fn q8_linear_group_into(
    projections: &[Q8Projection<'_>],
    input: &Q8Input,
    output: &mut [Vec<f32>],
) -> Option<()> {
    #[cfg(target_arch = "x86_64")]
    if isa() != "scalar" {
        assert_eq!(output.len(), projections.len());
        let nb = input.cols / 32;
        let batch = input.batch;
        for projection in projections {
            assert!(projection.rows > 0);
            assert_eq!(projection.weights.len(), projection.rows * nb * 34);
            assert!(projection.bias.is_none_or(|b| b.len() == projection.rows));
        }
        if batch == 0 || projections.is_empty() {
            output.iter_mut().for_each(Vec::clear);
            return Some(());
        }
        let rows = projections.iter().map(|p| p.rows).collect::<Vec<_>>();
        for (out, &rows) in output.iter_mut().zip(&rows) {
            out.resize(rows * batch, 0.0);
        }
        // Prompts on AMX: tile kernels over decoded 16-row panels (the `avx512` feature).
        #[cfg(feature = "avx512")]
        if batch > 1 && crate::cpu_avx512::amx_gemm::wanted(batch) {
            crate::cpu_avx512::amx_gemm::q8_group(projections, input.packed(), nb, batch, output);
            return Some(());
        }
        let vnni = isa() == "avx-vnni";
        let wide = avx512();
        let panel_rows = if batch == 1 && (vnni || wide.is_some()) {
            8
        } else {
            4
        };
        crate::cpu_schedule::q8_decode_row_panels(
            output,
            &rows,
            batch,
            panel_rows,
            || {
                if batch == 1 {
                    None
                } else {
                    Some(Q8PanelScratch::new(nb))
                }
            },
            |panel, member, first, out| {
                let projection = &projections[member];
                let valid = (projection.rows - first).min(panel_rows);
                // SAFETY: `isa()`/`avx512()` confirmed the features; shapes were asserted above and `out` owns these columns.
                unsafe {
                    if batch == 1 && wide.is_some() {
                        let dst = out.row(0);
                        if wide == Some(Avx512::Vnni) {
                            x86::gemv512_vnni(projection.weights, &input.single, first, dst);
                        } else {
                            x86::gemv512_bw(projection.weights, &input.single, first, dst);
                        }
                    } else if batch == 1 {
                        if vnni {
                            let dst = out.row(0);
                            if dst.len() == 8 {
                                x86::gemv_vnni_zero8(projection.weights, &input.single, first, dst);
                            } else {
                                for (j, tail) in dst.chunks_mut(4).enumerate() {
                                    if tail.len() == 4 {
                                        x86::gemv_vnni_zero4(
                                            projection.weights,
                                            &input.single,
                                            first + j * 4,
                                            tail,
                                        );
                                    } else {
                                        x86::gemv_vnni(
                                            projection.weights,
                                            &input.single,
                                            first + j * 4,
                                            tail,
                                        );
                                    }
                                }
                            }
                        } else {
                            x86::gemv_avx2(projection.weights, &input.single, first, out.row(0));
                        }
                    } else {
                        let panel = panel.as_mut().expect("prompt panel workspace");
                        x86::pack_panel(projection.weights, first, valid, panel);
                        match wide {
                            Some(Avx512::Vnni) => {
                                x86::tile512_vnni(panel, &input.packed, batch, out)
                            }
                            Some(Avx512::Bw) => x86::tile512_bw(panel, &input.packed, batch, out),
                            None if vnni => x86::tile_vnni(panel, &input.packed, batch, out),
                            None => x86::tile_avx2(panel, &input.packed, batch, out),
                        }
                    }
                }
                for b in 0..batch {
                    for (r, y) in out.row(b).iter_mut().enumerate() {
                        *y += projection.bias.map_or(0.0, |bias| bias[first + r]);
                    }
                }
            },
        );
        return Some(());
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if neon_dotprod() {
        assert_eq!(output.len(), projections.len());
        let nb = input.cols / 32;
        let batch = input.batch;
        for projection in projections {
            assert!(projection.rows > 0);
            assert_eq!(projection.weights.len(), projection.rows * nb * 34);
            assert!(projection.bias.is_none_or(|b| b.len() == projection.rows));
        }
        if batch == 0 || projections.is_empty() {
            output.iter_mut().for_each(Vec::clear);
            return Some(());
        }
        let rows = projections.iter().map(|p| p.rows).collect::<Vec<_>>();
        for (out, &rows) in output.iter_mut().zip(&rows) {
            out.resize(rows * batch, 0.0);
        }
        let i8mm = neon_i8mm();
        let sve = neon_sve();
        let panel_rows = if batch == 1 { 8 } else { 4 };
        crate::cpu_schedule::q8_decode_row_panels(
            output,
            &rows,
            batch,
            panel_rows,
            || (),
            |_, member, first, out| {
                let projection = &projections[member];
                let valid = (projection.rows - first).min(panel_rows);
                // SAFETY: dotprod/i8mm were detected; the shape checks above bound every read.
                unsafe {
                    if batch == 1 {
                        // `sve` implies a CPU with SVE vectors of 256 bits or more (opt-in tier).
                        let done = sve
                            && crate::cpu_arm::q8_gemv_sve(
                                projection.weights,
                                input.single.as_ptr().cast(),
                                nb,
                                first,
                                out.row(0),
                            );
                        if !done {
                            neon::gemv(projection.weights, &input.single, first, out.row(0));
                        }
                    } else if i8mm {
                        neon::tile_i8mm(
                            projection.weights,
                            first,
                            valid,
                            &input.packed,
                            nb,
                            batch,
                            out,
                        );
                    } else {
                        neon::tile(
                            projection.weights,
                            first,
                            valid,
                            &input.packed,
                            nb,
                            batch,
                            out,
                        );
                    }
                }
                for b in 0..batch {
                    for (r, y) in out.row(b).iter_mut().enumerate() {
                        *y += projection.bias.map_or(0.0, |bias| bias[first + r]);
                    }
                }
            },
        );
        return Some(());
    }
    let _ = (projections, input, output);
    None
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
pub(crate) mod neon {
    use super::{XPair, XSingle};
    use crate::cpu_schedule::Rows;
    use half::f16;
    use std::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn half_to_float(h: u16) -> f32 {
        vgetq_lane_f32::<0>(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))))
    }

    /// Low 64-bit halves of `a` and `b`: the two rows of an `smmla` operand.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn zip_lo(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)))
    }
    /// High 64-bit halves of `a` and `b`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn zip_hi(a: int8x16_t, b: int8x16_t) -> int8x16_t {
        vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)))
    }

    /// `acc + a * b^T` for two 2x8 signed byte matrices stored row by row; the lanes are
    /// [a0.b0, a0.b1, a1.b0, a1.b1]. `vmmlaq_s32` is not stable yet, hence the instruction.
    #[inline]
    #[target_feature(enable = "neon,i8mm")]
    pub(crate) unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut acc = acc;
        std::arch::asm!(
            "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
        acc
    }

    /// Quantize 32 values with `inverse`, rounding ties to even and saturating like `as i8`.
    /// Returns the signed sums of both 16-value halves.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn quantize32(src: &[f32], inverse: f32, dst: &mut [i8; 32]) -> [i32; 2] {
        let inverse = vdupq_n_f32(inverse);
        let mut sums = [0; 2];
        for (half, sum) in sums.iter_mut().enumerate() {
            let p = src.as_ptr().add(16 * half);
            let q = |i: usize| vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p.add(4 * i)), inverse));
            let low = vcombine_s16(vqmovn_s32(q(0)), vqmovn_s32(q(1)));
            let high = vcombine_s16(vqmovn_s32(q(2)), vqmovn_s32(q(3)));
            let bytes = vcombine_s8(vqmovn_s16(low), vqmovn_s16(high));
            vst1q_s8(dst.as_mut_ptr().add(16 * half), bytes);
            *sum = i32::from(vaddlvq_s8(bytes));
        }
        sums
    }

    /// Exactly one 32-value activation block. Follows the scalar packing step for step (`vmaxnm`
    /// ignores NaN like `f32::max`, `vcvtn` rounds ties to even); `selftest-cpu` compares the
    /// resulting records with the scalar packing bit for bit.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn pack_q8_block(
        src: &[f32],
        dst: &mut [i8; 32],
        reciprocal_scale: bool,
    ) -> (f32, f32, [i32; 2]) {
        assert_eq!(src.len(), 32);
        let mut maximum = vdupq_n_f32(0.0);
        for i in 0..8 {
            maximum = vmaxnmq_f32(maximum, vabsq_f32(vld1q_f32(src.as_ptr().add(4 * i))));
        }
        let amax = vmaxnmvq_f32(maximum);
        let exact = amax / 127.0;
        let inverse = if amax == 0.0 {
            0.0
        } else if reciprocal_scale {
            exact.recip()
        } else {
            127.0 / amax
        };
        let sums = quantize32(src, inverse, dst);
        (f16::from_f32(exact).to_f32(), exact, sums)
    }

    /// Q8_K part of one 256-value block: one shared inverse, signed 16-value sums.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn quantize_i8_block(
        src: &[f32],
        inverse: f32,
        dst: &mut [i8; 32],
    ) -> [i32; 2] {
        assert_eq!(src.len(), 32);
        quantize32(src, inverse, dst)
    }

    /// Mirrors `encode_q8_cache_group_scalar` step for step (IEEE division, ties-away rounding,
    /// clamp); non-finite groups take the scalar path. `selftest-cpu` compares bytes and scale.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn encode_q8_cache_group(src: &[f32], dst: &mut [u8], scale: &mut u16) {
        assert_eq!(src.len(), dst.len());
        let end = src.len() / 4 * 4;
        let mut maximum = vdupq_n_f32(0.0);
        let mut nonfinite = vdupq_n_u32(0);
        for j in (0..end).step_by(4) {
            let abs = vabsq_f32(vld1q_f32(src.as_ptr().add(j)));
            // `!(abs <= MAX)` holds for both NaN and infinity.
            nonfinite = vorrq_u32(nonfinite, vmvnq_u32(vcleq_f32(abs, vdupq_n_f32(f32::MAX))));
            maximum = vmaxnmq_f32(maximum, abs);
        }
        let mut has_nonfinite = vmaxvq_u32(nonfinite) != 0;
        let mut amax = vmaxnmvq_f32(maximum);
        for &x in &src[end..] {
            has_nonfinite |= !x.is_finite();
            if x.abs() > amax {
                amax = x.abs();
            }
        }
        if has_nonfinite {
            super::encode_q8_cache_group_scalar(src, dst, scale);
            return;
        }
        let d = amax / 127.0f32;
        *scale = f16::from_f32(d).to_bits();
        if d == 0.0 {
            dst.fill(0);
            return;
        }
        let divisor = vdupq_n_f32(d);
        let end = src.len() / 16 * 16;
        for j in (0..end).step_by(16) {
            let p = src.as_ptr().add(j);
            // Division and ties-away rounding match `(x / d).round()`.
            let q = |i: usize| vcvtaq_s32_f32(vdivq_f32(vld1q_f32(p.add(4 * i)), divisor));
            let low = vcombine_s16(vqmovn_s32(q(0)), vqmovn_s32(q(1)));
            let high = vcombine_s16(vqmovn_s32(q(2)), vqmovn_s32(q(3)));
            let bytes = vmaxq_s8(
                vcombine_s8(vqmovn_s16(low), vqmovn_s16(high)),
                vdupq_n_s8(-127),
            );
            vst1q_s8(dst.as_mut_ptr().add(j).cast(), bytes);
        }
        for j in end..src.len() {
            dst[j] = (src[j] / d).round().clamp(-127.0, 127.0) as i8 as u8;
        }
    }

    /// Up to eight Q8_0 rows from `row0` against one decode activation row.
    ///
    /// # Safety
    /// The CPU must support NEON and dotprod; `w` holds complete rows of `x.len()` blocks
    /// through row `row0 + out.len()`.
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn gemv(w: &[u8], x: &[XSingle], row0: usize, out: &mut [f32]) {
        match out.len() {
            1 => gemv_rows::<1>(w, x, row0, out),
            2 => gemv_rows::<2>(w, x, row0, out),
            3 => gemv_rows::<3>(w, x, row0, out),
            4 => gemv_rows::<4>(w, x, row0, out),
            5 => gemv_rows::<5>(w, x, row0, out),
            6 => gemv_rows::<6>(w, x, row0, out),
            7 => gemv_rows::<7>(w, x, row0, out),
            8 => gemv_rows::<8>(w, x, row0, out),
            _ => unreachable!(),
        }
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn gemv_rows<const R: usize>(w: &[u8], x: &[XSingle], row0: usize, out: &mut [f32]) {
        let nb = x.len();
        let stride = nb * 34;
        assert!(w.len() >= (row0 + R) * stride);
        let base = w.as_ptr().add(row0 * stride);
        let zero = vdupq_n_s32(0);
        let mut acc = [vdupq_n_f32(0.0); R];
        for (kb, xr) in x.iter().enumerate() {
            let x0 = vld1q_s8(xr.q.as_ptr());
            let x1 = vld1q_s8(xr.q.as_ptr().add(16));
            for (r, acc) in acc.iter_mut().enumerate() {
                let p = base.add(r * stride + kb * 34);
                let d = half_to_float(std::ptr::read_unaligned(p.cast::<u16>()));
                let dot = vdotq_s32(
                    vdotq_s32(zero, vld1q_s8(p.add(2).cast()), x0),
                    vld1q_s8(p.add(18).cast()),
                    x1,
                );
                *acc = vfmaq_f32(*acc, vcvtq_f32_s32(dot), vdupq_n_f32(d * xr.d));
            }
        }
        for (y, acc) in out.iter_mut().zip(acc) {
            *y = vaddvq_f32(acc);
        }
    }

    /// Prompt rows `first..first + rows` (1..=4) for every token pair, with `sdot`.
    ///
    /// # Safety
    /// The CPU must support NEON and dotprod; `w` holds complete rows of `nb` blocks through
    /// row `first + rows` and `x` holds whole token pairs of `nb` blocks.
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn tile(
        w: &[u8],
        first: usize,
        rows: usize,
        x: &[XPair],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        match rows {
            1 => tile_rows::<1>(w, first, x, nb, batch, out),
            2 => tile_rows::<2>(w, first, x, nb, batch, out),
            3 => tile_rows::<3>(w, first, x, nb, batch, out),
            4 => tile_rows::<4>(w, first, x, nb, batch, out),
            _ => unreachable!(),
        }
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn tile_rows<const R: usize>(
        w: &[u8],
        first: usize,
        x: &[XPair],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        let stride = nb * 34;
        assert!(w.len() >= (first + R) * stride);
        let base = w.as_ptr().add(first * stride);
        let zero = vdupq_n_s32(0);
        for (p, pair) in x.chunks_exact(nb).enumerate() {
            let mut acc = [[vdupq_n_f32(0.0); R]; 2];
            for (kb, xr) in pair.iter().enumerate() {
                let xq = [
                    [
                        vld1q_s8(xr.q[0].as_ptr()),
                        vld1q_s8(xr.q[0].as_ptr().add(16)),
                    ],
                    [
                        vld1q_s8(xr.q[1].as_ptr()),
                        vld1q_s8(xr.q[1].as_ptr().add(16)),
                    ],
                ];
                for r in 0..R {
                    let p = base.add(r * stride + kb * 34);
                    let d = half_to_float(std::ptr::read_unaligned(p.cast::<u16>()));
                    let w0 = vld1q_s8(p.add(2).cast());
                    let w1 = vld1q_s8(p.add(18).cast());
                    for (j, acc) in acc.iter_mut().enumerate() {
                        let dot = vdotq_s32(vdotq_s32(zero, w0, xq[j][0]), w1, xq[j][1]);
                        acc[r] = vfmaq_f32(acc[r], vcvtq_f32_s32(dot), vdupq_n_f32(d * xr.d[j]));
                    }
                }
            }
            for (j, acc) in acc.iter().enumerate() {
                if 2 * p + j < batch {
                    for (y, acc) in out.row(2 * p + j).iter_mut().zip(acc) {
                        *y = vaddvq_f32(*acc);
                    }
                }
            }
        }
    }

    /// `tile` with `smmla`: rows pair up against the token pair, four instructions per block
    /// and row pair instead of eight `sdot`s.
    ///
    /// # Safety
    /// As `tile`, and the CPU must support i8mm.
    #[target_feature(enable = "neon,dotprod,i8mm")]
    pub(super) unsafe fn tile_i8mm(
        w: &[u8],
        first: usize,
        rows: usize,
        x: &[XPair],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        match rows {
            1 => tile_i8mm_rows::<1>(w, first, x, nb, batch, out),
            2 => tile_i8mm_rows::<2>(w, first, x, nb, batch, out),
            3 => tile_i8mm_rows::<3>(w, first, x, nb, batch, out),
            4 => tile_i8mm_rows::<4>(w, first, x, nb, batch, out),
            _ => unreachable!(),
        }
    }
    #[inline]
    #[target_feature(enable = "neon,dotprod,i8mm")]
    unsafe fn tile_i8mm_rows<const R: usize>(
        w: &[u8],
        first: usize,
        x: &[XPair],
        nb: usize,
        batch: usize,
        out: &mut Rows<'_>,
    ) {
        let stride = nb * 34;
        assert!(w.len() >= (first + R) * stride);
        let base = w.as_ptr().add(first * stride);
        let zero = vdupq_n_s32(0);
        let pairs = R.div_ceil(2);
        for (p, pair) in x.chunks_exact(nb).enumerate() {
            // Lanes: [row 2k token 0, row 2k token 1, row 2k+1 token 0, row 2k+1 token 1].
            let mut acc = [vdupq_n_f32(0.0); 2];
            for (kb, xr) in pair.iter().enumerate() {
                let (a0, a1) = (
                    vld1q_s8(xr.q[0].as_ptr()),
                    vld1q_s8(xr.q[0].as_ptr().add(16)),
                );
                let (b0, b1) = (
                    vld1q_s8(xr.q[1].as_ptr()),
                    vld1q_s8(xr.q[1].as_ptr().add(16)),
                );
                let xm = [
                    zip_lo(a0, b0),
                    zip_hi(a0, b0),
                    zip_lo(a1, b1),
                    zip_hi(a1, b1),
                ];
                let xd = [xr.d[0], xr.d[1], xr.d[0], xr.d[1]];
                let xd = vld1q_f32(xd.as_ptr());
                for (k, acc) in acc.iter_mut().enumerate().take(pairs) {
                    // An odd panel repeats its last row; that lane is never stored.
                    let pa = base.add(2 * k * stride + kb * 34);
                    let pb = base.add((2 * k + 1).min(R - 1) * stride + kb * 34);
                    let (wa0, wa1) = (vld1q_s8(pa.add(2).cast()), vld1q_s8(pa.add(18).cast()));
                    let (wb0, wb1) = (vld1q_s8(pb.add(2).cast()), vld1q_s8(pb.add(18).cast()));
                    let mut dot = smmla(zero, zip_lo(wa0, wb0), xm[0]);
                    dot = smmla(dot, zip_hi(wa0, wb0), xm[1]);
                    dot = smmla(dot, zip_lo(wa1, wb1), xm[2]);
                    dot = smmla(dot, zip_hi(wa1, wb1), xm[3]);
                    let da = half_to_float(std::ptr::read_unaligned(pa.cast::<u16>()));
                    let db = half_to_float(std::ptr::read_unaligned(pb.cast::<u16>()));
                    let dw = [da, da, db, db];
                    let scale = vmulq_f32(vld1q_f32(dw.as_ptr()), xd);
                    *acc = vfmaq_f32(*acc, vcvtq_f32_s32(dot), scale);
                }
            }
            let mut lanes = [[0.0_f32; 4]; 2];
            for (lanes, acc) in lanes.iter_mut().zip(acc) {
                vst1q_f32(lanes.as_mut_ptr(), acc);
            }
            for j in 0..2 {
                if 2 * p + j < batch {
                    for (r, y) in out.row(2 * p + j).iter_mut().enumerate() {
                        *y = lanes[r / 2][(r % 2) * 2 + j];
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use super::{WPanel, XPair, XSingle};
    use std::arch::x86_64::*;
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn pack_vectors(v: [__m256; 4], inverse: f32) -> __m256i {
        let inverse = _mm256_set1_ps(inverse);
        let a = _mm256_cvtps_epi32(_mm256_mul_ps(v[0], inverse));
        let b = _mm256_cvtps_epi32(_mm256_mul_ps(v[1], inverse));
        let c = _mm256_cvtps_epi32(_mm256_mul_ps(v[2], inverse));
        let d = _mm256_cvtps_epi32(_mm256_mul_ps(v[3], inverse));
        let packed = _mm256_packs_epi16(_mm256_packs_epi32(a, b), _mm256_packs_epi32(c, d));
        _mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sums16(q: __m256i) -> [i32; 2] {
        let sum = _mm256_madd_epi16(
            _mm256_maddubs_epi16(_mm256_set1_epi8(1), q),
            _mm256_set1_epi16(1),
        );
        let reduce = |v: __m128i| {
            let v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4e));
            let v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xb1));
            _mm_cvtsi128_si32(v)
        };
        [
            reduce(_mm256_castsi256_si128(sum)),
            reduce(_mm256_extracti128_si256(sum, 1)),
        ]
    }
    /// Exactly one 32-value activation block. The caller guards AVX2/F16C.
    /// The inverse expression differs between decoder linear and attention;
    /// preserve both expressions because f32 division can change a rounding tie.
    #[target_feature(enable = "avx2,f16c")]
    pub(crate) unsafe fn pack_q8_block(
        src: &[f32],
        dst: &mut [i8; 32],
        reciprocal_scale: bool,
    ) -> (f32, f32, [i32; 2]) {
        assert_eq!(src.len(), 32);
        let v = [
            _mm256_loadu_ps(src.as_ptr()),
            _mm256_loadu_ps(src.as_ptr().add(8)),
            _mm256_loadu_ps(src.as_ptr().add(16)),
            _mm256_loadu_ps(src.as_ptr().add(24)),
        ];
        let sign = _mm256_set1_ps(-0.0);
        let maximum = _mm256_max_ps(
            _mm256_max_ps(_mm256_andnot_ps(sign, v[0]), _mm256_andnot_ps(sign, v[1])),
            _mm256_max_ps(_mm256_andnot_ps(sign, v[2]), _mm256_andnot_ps(sign, v[3])),
        );
        let m = _mm_max_ps(
            _mm256_castps256_ps128(maximum),
            _mm256_extractf128_ps(maximum, 1),
        );
        let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        let amax = _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55)));
        let exact = amax / 127.0;
        let inverse = if amax == 0.0 {
            0.0
        } else if reciprocal_scale {
            exact.recip()
        } else {
            127.0 / amax
        };
        let q = pack_vectors(v, inverse);
        _mm256_storeu_si256(dst.as_mut_ptr().cast(), q);
        (half_to_float(float_to_half(exact)), exact, sums16(q))
    }
    /// Q8_K uses one scale across 256 values and retains signed 16-value sums.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn quantize_i8_block(
        src: &[f32],
        inverse: f32,
        dst: &mut [i8; 32],
    ) -> [i32; 2] {
        assert_eq!(src.len(), 32);
        let v = [
            _mm256_loadu_ps(src.as_ptr()),
            _mm256_loadu_ps(src.as_ptr().add(8)),
            _mm256_loadu_ps(src.as_ptr().add(16)),
            _mm256_loadu_ps(src.as_ptr().add(24)),
        ];
        let q = pack_vectors(v, inverse);
        _mm256_storeu_si256(dst.as_mut_ptr().cast(), q);
        sums16(q)
    }
    #[target_feature(enable = "avx2,f16c")]
    pub(super) unsafe fn pack_panel(w: &[u8], first: usize, valid: usize, out: &mut [WPanel]) {
        let nb = out.len();
        let flip = _mm256_set1_epi8(-128);
        for (kb, rec) in out.iter_mut().enumerate() {
            for r in 0..valid {
                let b = w.as_ptr().add(((first + r) * nb + kb) * 34);
                rec.d[r] = half_to_float(std::ptr::read_unaligned(b.cast()));
                let v = _mm256_xor_si256(_mm256_loadu_si256(b.add(2).cast()), flip);
                _mm256_store_si256(rec.q[r].as_mut_ptr().cast(), v);
            }
            for r in valid..4 {
                rec.d[r] = 0.0;
                _mm256_store_si256(rec.q[r].as_mut_ptr().cast(), flip);
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[cfg(test)]
    pub(crate) unsafe fn dot_avx2(init: __m256i, u: __m256i, s: __m256i) -> __m256i {
        // maddubs saturates signed i16. Split unsigned weights at bit 7 so
        // neither partial can saturate, even with 255 weights / -128 inputs.
        let lo = _mm256_and_si256(u, _mm256_set1_epi8(127));
        let hi = _mm256_and_si256(_mm256_srli_epi16(u, 7), _mm256_set1_epi8(1));
        let ones = _mm256_set1_epi16(1);
        let a = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, s), ones);
        let b = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, s), ones);
        _mm256_add_epi32(init, _mm256_add_epi32(a, _mm256_slli_epi32(b, 7)))
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn dot_centered_avx2(_init: __m256i, u: __m256i, s: __m256i) -> __m256i {
        // Activation packing bounds s to [-127,127], so even |w|=128
        // gives a pair sum <=32512 and cannot saturate maddubs. Centering
        // before the dot avoids the two-pass generic u8 AVX2 emulation.
        let w = _mm256_xor_si256(u, _mm256_set1_epi8(-128));
        _mm256_madd_epi16(
            _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(s, w)),
            _mm256_set1_epi16(1),
        )
    }
    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    pub(crate) unsafe fn dot_vnni(init: __m256i, u: __m256i, s: __m256i) -> __m256i {
        _mm256_dpbusd_avx_epi32(init, u, s)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum(v: __m256) -> f32 {
        let t = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55)))
    }
    macro_rules! tile {
        ($name:ident, $features:literal, $dot:ident) => {
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn $name(
                panel: &[WPanel],
                x: &[XPair],
                batch: usize,
                out: &mut crate::cpu_schedule::Rows<'_>,
            ) {
                let nb = panel.len();
                for (p, pair) in x.chunks_exact(nb).enumerate() {
                    let mut a = [[_mm256_setzero_ps(); 4]; 2];
                    for (xr, wr) in pair.iter().zip(panel) {
                        let x0 = _mm256_load_si256(xr.q[0].as_ptr().cast());
                        let x1 = _mm256_load_si256(xr.q[1].as_ptr().cast());
                        let i0 = _mm256_set1_epi32(xr.init[0]);
                        let i1 = _mm256_set1_epi32(xr.init[1]);
                        let dw = _mm_load_ps(wr.d.as_ptr());
                        let mut s0 = [0.0; 4];
                        let mut s1 = [0.0; 4];
                        _mm_storeu_ps(s0.as_mut_ptr(), _mm_mul_ps(dw, _mm_set1_ps(xr.d[0])));
                        _mm_storeu_ps(s1.as_mut_ptr(), _mm_mul_ps(dw, _mm_set1_ps(xr.d[1])));
                        // Keep scale products in memory: letting LLVM retain
                        // both SSE vectors spills a live AVX accumulator on
                        // every block. Volatile scalar loads force broadcasts
                        // without changing scale multiplication or FMA order.
                        // Constant row indices keep eight accumulators in registers.
                        macro_rules! row {
                            ($r:literal) => {
                                let w = _mm256_load_si256(wr.q[$r].as_ptr().cast());
                                a[0][$r] = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps($dot(i0, w, x0)),
                                    _mm256_set1_ps(std::ptr::read_volatile(&s0[$r])),
                                    a[0][$r],
                                );
                                a[1][$r] = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps($dot(i1, w, x1)),
                                    _mm256_set1_ps(std::ptr::read_volatile(&s1[$r])),
                                    a[1][$r],
                                );
                            };
                        }
                        row!(0);
                        row!(1);
                        row!(2);
                        row!(3);
                    }
                    for j in 0..2 {
                        if 2 * p + j < batch {
                            for (r, y) in out.row(2 * p + j).iter_mut().enumerate() {
                                *y = hsum(a[j][r]);
                            }
                        }
                    }
                }
            }
        };
    }
    tile!(tile_avx2, "avx2,fma,f16c", dot_centered_avx2);
    tile!(tile_vnni, "avx2,fma,f16c,avxvnni", dot_vnni);

    macro_rules! gemv {
        ($name:ident, $body:ident, $features:literal, $dot:ident) => {
            #[target_feature(enable = $features)]
            pub(super) unsafe fn $name(w: &[u8], x: &[XSingle], row0: usize, out: &mut [f32]) {
                match out.len() {
                    1 => $body::<1>(w, x, row0, out),
                    2 => $body::<2>(w, x, row0, out),
                    3 => $body::<3>(w, x, row0, out),
                    4 => $body::<4>(w, x, row0, out),
                    _ => unreachable!(),
                }
            }
            #[inline]
            #[target_feature(enable = $features)]
            unsafe fn $body<const R: usize>(w: &[u8], x: &[XSingle], row0: usize, out: &mut [f32]) {
                let nb = x.len();
                let stride = nb * 34;
                let mut a = [_mm256_setzero_ps(); 4];
                let flip = _mm256_set1_epi8(-128);
                for (kb, xr) in x.iter().enumerate() {
                    let xv = _mm256_loadu_si256(xr.q.as_ptr().cast());
                    let init = _mm256_set1_epi32(xr.init);
                    macro_rules! row {
                        ($r:literal) => {
                            if $r < R {
                                let r = $r;
                                let ptr = w.as_ptr().add((row0 + r) * stride + kb * 34);
                                let dw = half_to_float(std::ptr::read_unaligned(ptr.cast()));
                                let u =
                                    _mm256_xor_si256(_mm256_loadu_si256(ptr.add(2).cast()), flip);
                                a[r] = _mm256_fmadd_ps(
                                    _mm256_cvtepi32_ps($dot(init, u, xv)),
                                    _mm256_set1_ps(dw * xr.d),
                                    a[r],
                                );
                            }
                        };
                    }
                    row!(0);
                    row!(1);
                    row!(2);
                    row!(3);
                }
                for (r, y) in out.iter_mut().enumerate() {
                    *y = hsum(a[r]);
                }
            }
        };
    }
    gemv!(
        gemv_avx2,
        gemv_avx2_rows,
        "avx2,fma,f16c",
        dot_centered_avx2
    );
    gemv!(gemv_vnni, gemv_vnni_rows, "avx2,fma,f16c,avxvnni", dot_vnni);

    /// Convert one FP16 scale using an explicit zero source register. This
    /// keeps LLVM from selecting a live YMM accumulator as VPINSRW's source.
    #[inline(always)]
    unsafe fn half_from_ptr_zero(p: *const u8, zero: __m128i) -> f32 {
        let value: __m128;
        std::arch::asm!(
            "vpinsrw {value:x}, {zero:x}, word ptr [{p}], 0",
            "vcvtph2ps {value:x}, {value:x}",
            value = out(xmm_reg) value,
            zero = in(xmm_reg) zero,
            p = in(reg) p,
            options(pure, readonly, nostack, preserves_flags),
        );
        _mm_cvtss_f32(value)
    }

    /// Four-row VNNI decode with the original arithmetic order and an
    /// explicitly zeroed scale-conversion source to avoid false dependencies.
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub(super) unsafe fn gemv_vnni_zero4(w: &[u8], x: &[XSingle], row0: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), 4);
        let nb = x.len();
        let stride = nb * 34;
        let mut a = [_mm256_setzero_ps(); 4];
        let flip = _mm256_set1_epi8(-128);
        let zero = _mm_setzero_si128();
        for (kb, xr) in x.iter().enumerate() {
            let xv = _mm256_loadu_si256(xr.q.as_ptr().cast());
            let init = _mm256_set1_epi32(xr.init);
            macro_rules! row {
                ($r:literal) => {
                    let ptr = w.as_ptr().add((row0 + $r) * stride + kb * 34);
                    let dw = half_from_ptr_zero(ptr, zero);
                    let u = _mm256_xor_si256(_mm256_loadu_si256(ptr.add(2).cast()), flip);
                    a[$r] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(dot_vnni(init, u, xv)),
                        _mm256_set1_ps(dw * xr.d),
                        a[$r],
                    );
                };
            }
            row!(0);
            row!(1);
            row!(2);
            row!(3);
        }
        for (y, acc) in out.iter_mut().zip(a) {
            *y = hsum(acc);
        }
    }

    /// Eight-row VNNI decode with folded row addresses and explicit VEX dots.
    /// Keep each row in one asm block so LLVM cannot spill hoisted temporaries.
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub(super) unsafe fn gemv_vnni_zero8(w: &[u8], x: &[XSingle], row0: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), 8);
        let nb = x.len();
        let stride = nb * 34;
        let row1 = stride;
        let row2 = row1 * 2;
        let row3 = row2 + row1;
        let row4 = row2 * 2;
        let row5 = row4 + row1;
        let row6 = row3 * 2;
        let row7 = row6 + row1;
        let mut base = w.as_ptr().add(row0 * stride);
        let mut a = [_mm256_setzero_ps(); 8];
        let flip = _mm256_set1_epi8(-128);
        let zero = _mm_setzero_si128();
        for xr in x {
            let xv = _mm256_loadu_si256(xr.q.as_ptr().cast());
            let init = _mm256_set1_epi32(xr.init);
            let dx = _mm_set_ss(xr.d);
            macro_rules! row {
                ($r:literal, $offset:expr) => {
                    std::arch::asm!(
                        "vmovdqa {dot:y}, {init:y}",
                        "vpxor {weight:y}, {flip:y}, ymmword ptr [{base} + {offset} + 2]",
                        "{{vex}} vpdpbusd {dot:y}, {weight:y}, {xv:y}",
                        "vcvtdq2ps {dot:y}, {dot:y}",
                        "vpinsrw {scale:x}, {zero:x}, word ptr [{base} + {offset}], 0",
                        "vcvtph2ps {scale:x}, {scale:x}",
                        "vmulss {scale:x}, {scale:x}, {dx:x}",
                        "vbroadcastss {weight:y}, {scale:x}",
                        "vfmadd231ps {acc:y}, {dot:y}, {weight:y}",
                        acc = inout(ymm_reg) a[$r],
                        xv = in(ymm_reg) xv,
                        init = in(ymm_reg) init,
                        flip = in(ymm_reg) flip,
                        zero = in(xmm_reg) zero,
                        dx = in(xmm_reg) dx,
                        base = in(reg) base,
                        offset = in(reg) $offset,
                        dot = out(ymm_reg) _,
                        weight = out(ymm_reg) _,
                        scale = out(xmm_reg) _,
                        options(pure, readonly, nostack, preserves_flags),
                    );
                };
            }
            row!(0, 0usize);
            row!(1, row1);
            row!(2, row2);
            row!(3, row3);
            row!(4, row4);
            row!(5, row5);
            row!(6, row6);
            row!(7, row7);
            base = base.add(34);
        }
        for (y, acc) in out.iter_mut().zip(a) {
            *y = hsum(acc);
        }
    }

    // Wide decode: two consecutive Q8_0 blocks share one 512-bit dot and up to eight rows reuse
    // each activation pair. `$dot` takes the unsigned (signed + 128) weights of the AVX paths.
    macro_rules! gemv512 {
        ($name:ident, $body:ident, $features:literal, $dot:ident) => {
            /// `out.len()` (1..=8) rows from `row0` against one decode activation row.
            ///
            /// # Safety
            /// The CPU must support the enabled features and `w` must hold complete rows of
            /// `x.len()` Q8_0 blocks through row `row0 + out.len()`.
            #[target_feature(enable = $features)]
            pub(super) unsafe fn $name(w: &[u8], x: &[XSingle], row0: usize, out: &mut [f32]) {
                match out.len() {
                    1 => $body::<1>(w, x, row0, out),
                    2 => $body::<2>(w, x, row0, out),
                    3 => $body::<3>(w, x, row0, out),
                    4 => $body::<4>(w, x, row0, out),
                    5 => $body::<5>(w, x, row0, out),
                    6 => $body::<6>(w, x, row0, out),
                    7 => $body::<7>(w, x, row0, out),
                    8 => $body::<8>(w, x, row0, out),
                    _ => unreachable!(),
                }
            }
            #[inline]
            #[target_feature(enable = $features)]
            unsafe fn $body<const R: usize>(w: &[u8], x: &[XSingle], row0: usize, out: &mut [f32]) {
                let nb = x.len();
                let stride = nb * 34;
                assert!(w.len() >= (row0 + R) * stride);
                let base = w.as_ptr().add(row0 * stride);
                let flip = _mm512_set1_epi8(-128);
                let mut acc = [_mm512_setzero_ps(); R];
                for kb in (0..nb).step_by(2) {
                    let pair = kb + 1 < nb;
                    let xa = &x[kb];
                    let qa = _mm256_loadu_si256(xa.q.as_ptr().cast());
                    // An odd final block runs with zero upper activations, bias and scale.
                    let (xv, init, xd) = if pair {
                        let xb = &x[kb + 1];
                        (
                            join256(qa, _mm256_loadu_si256(xb.q.as_ptr().cast())),
                            _mm512_mask_blend_epi32(
                                0xff00,
                                _mm512_set1_epi32(xa.init),
                                _mm512_set1_epi32(xb.init),
                            ),
                            xb.d,
                        )
                    } else {
                        (
                            _mm512_zextsi256_si512(qa),
                            _mm512_mask_blend_epi32(
                                0xff00,
                                _mm512_set1_epi32(xa.init),
                                _mm512_setzero_si512(),
                            ),
                            0.0,
                        )
                    };
                    for (r, acc) in acc.iter_mut().enumerate() {
                        let p = base.add(r * stride + kb * 34);
                        let low = _mm256_loadu_si256(p.add(2).cast());
                        let (weights, d1) = if pair {
                            (
                                join256(low, _mm256_loadu_si256(p.add(36).cast())),
                                half_to_float(std::ptr::read_unaligned(p.add(34).cast())),
                            )
                        } else {
                            (_mm512_zextsi256_si512(low), 0.0)
                        };
                        let d0 = half_to_float(std::ptr::read_unaligned(p.cast()));
                        let dot = $dot(init, _mm512_xor_si512(weights, flip), xv);
                        *acc = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(dot),
                            halves_ps(d0 * xa.d, d1 * xd),
                            *acc,
                        );
                    }
                }
                for (y, acc) in out.iter_mut().zip(acc) {
                    *y = _mm512_reduce_add_ps(acc);
                }
            }
        };
    }
    gemv512!(
        gemv512_bw,
        gemv512_bw_rows,
        "avx2,fma,f16c,avx512f,avx512bw,avx512vl",
        dot_biased512_bw
    );
    gemv512!(
        gemv512_vnni,
        gemv512_vnni_rows,
        "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni",
        dot_biased512_vnni
    );

    // Wide prompt tile: the four panel rows pair up into two 512-bit vectors against each
    // token's activation block broadcast to both halves.
    macro_rules! tile512 {
        ($name:ident, $features:literal, $dot:ident) => {
            /// # Safety
            /// The CPU must support the enabled features; `x` holds `batch.div_ceil(2)` token
            /// pairs of `panel.len()` blocks each.
            #[target_feature(enable = $features)]
            pub(crate) unsafe fn $name(
                panel: &[WPanel],
                x: &[XPair],
                batch: usize,
                out: &mut crate::cpu_schedule::Rows<'_>,
            ) {
                let nb = panel.len();
                for (p, pair) in x.chunks_exact(nb).enumerate() {
                    let mut a = [[_mm512_setzero_ps(); 2]; 2];
                    for (xr, wr) in pair.iter().zip(panel) {
                        let weights = [
                            join256(
                                _mm256_load_si256(wr.q[0].as_ptr().cast()),
                                _mm256_load_si256(wr.q[1].as_ptr().cast()),
                            ),
                            join256(
                                _mm256_load_si256(wr.q[2].as_ptr().cast()),
                                _mm256_load_si256(wr.q[3].as_ptr().cast()),
                            ),
                        ];
                        for (j, a) in a.iter_mut().enumerate() {
                            let xv =
                                _mm512_broadcast_i64x4(_mm256_load_si256(xr.q[j].as_ptr().cast()));
                            let init = _mm512_set1_epi32(xr.init[j]);
                            for (k, a) in a.iter_mut().enumerate() {
                                let scale =
                                    halves_ps(wr.d[2 * k] * xr.d[j], wr.d[2 * k + 1] * xr.d[j]);
                                *a = _mm512_fmadd_ps(
                                    _mm512_cvtepi32_ps($dot(init, weights[k], xv)),
                                    scale,
                                    *a,
                                );
                            }
                        }
                    }
                    for (j, a) in a.iter().enumerate() {
                        if 2 * p + j < batch {
                            let (s0, s1) = hsum_halves(a[0]);
                            let (s2, s3) = hsum_halves(a[1]);
                            for (y, s) in out.row(2 * p + j).iter_mut().zip([s0, s1, s2, s3]) {
                                *y = s;
                            }
                        }
                    }
                }
            }
        };
    }
    tile512!(
        tile512_bw,
        "avx2,fma,f16c,avx512f,avx512bw,avx512vl",
        dot_biased512_bw
    );
    tile512!(
        tile512_vnni,
        "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni",
        dot_biased512_vnni
    );

    /// Unsigned (signed + 128) weights by signed activations with VNNI. `init` carries the
    /// -128 * activation-sum bias, spread over the lanes that share one scale.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub(crate) unsafe fn dot_biased512_vnni(init: __m512i, u: __m512i, s: __m512i) -> __m512i {
        _mm512_dpbusd_epi32(init, u, s)
    }
    /// Same contract on AVX-512BW: undo the weight bias and take the signed dot instead.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) unsafe fn dot_biased512_bw(_init: __m512i, u: __m512i, s: __m512i) -> __m512i {
        dot512_signed(_mm512_xor_si512(u, _mm512_set1_epi8(-128)), s)
    }
    /// Signed 8-bit dot on AVX-512BW, four products per 32-bit lane. `vpmaddubsw` needs an
    /// unsigned operand, so each weight's sign moves onto its activation. Activations are bounded
    /// to [-127, 127], so even |w| = 128 keeps every pair sum at most 32512.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) unsafe fn dot512_signed(w: __m512i, s: __m512i) -> __m512i {
        let s = _mm512_mask_sub_epi8(s, _mm512_movepi8_mask(w), _mm512_setzero_si512(), s);
        _mm512_madd_epi16(
            _mm512_maddubs_epi16(_mm512_abs_epi8(w), s),
            _mm512_set1_epi16(1),
        )
    }
    /// Two 256-bit halves as one 512-bit vector.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn join256(low: __m256i, high: __m256i) -> __m512i {
        _mm512_inserti64x4::<1>(_mm512_castsi256_si512(low), high)
    }
    /// Lanes 0..8 hold `low`, lanes 8..16 hold `high`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn halves_ps(low: f32, high: f32) -> __m512 {
        _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(low), _mm512_set1_ps(high))
    }
    /// Horizontal sums of the low and high 256-bit halves.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn hsum_halves(v: __m512) -> (f32, f32) {
        let high = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(v)));
        (hsum(_mm512_castps512_ps256(v)), hsum(high))
    }

    #[inline]
    #[target_feature(enable = "f16c")]
    pub(crate) unsafe fn half_to_float(h: u16) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)))
    }
    #[inline]
    #[target_feature(enable = "f16c")]
    pub(crate) unsafe fn float_to_half(v: f32) -> u16 {
        _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(v), 0)) as u16
    }
    #[target_feature(enable = "avx2,f16c")]
    pub(super) unsafe fn encode_q8_cache_group(src: &[f32], dst: &mut [u8], scale: &mut u16) {
        assert_eq!(src.len(), dst.len());
        let end = src.len() / 8 * 8;
        let sign = _mm256_set1_ps(-0.0);
        let mut vmax = _mm256_setzero_ps();
        let mut has_nonfinite = false;
        for j in (0..end).step_by(8) {
            let x = _mm256_loadu_ps(src.as_ptr().add(j));
            let abs = _mm256_andnot_ps(sign, x);
            let unordered = _mm256_cmp_ps(abs, abs, _CMP_UNORD_Q);
            let infinite = _mm256_cmp_ps(abs, _mm256_set1_ps(f32::MAX), _CMP_GT_OQ);
            has_nonfinite |= _mm256_movemask_ps(_mm256_or_ps(unordered, infinite)) != 0;
            let abs = _mm256_blendv_ps(abs, _mm256_setzero_ps(), unordered);
            vmax = _mm256_max_ps(vmax, abs);
        }
        let t = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        let t = _mm_max_ps(t, _mm_movehl_ps(t, t));
        let mut amax = _mm_cvtss_f32(_mm_max_ss(t, _mm_shuffle_ps(t, t, 0x55)));
        for &x in &src[end..] {
            has_nonfinite |= !x.is_finite();
            let abs = x.abs();
            if abs > amax {
                amax = abs;
            }
        }
        let d = amax / 127.0f32;
        if has_nonfinite {
            super::encode_q8_cache_group_scalar(src, dst, scale);
            return;
        }
        *scale = float_to_half(d);
        if d == 0.0 {
            dst.fill(0);
            return;
        }

        let dv = _mm256_set1_ps(d);
        let half = _mm256_set1_ps(0.5);
        let neg_half = _mm256_set1_ps(-0.5);
        let lo = _mm256_set1_epi32(-127);
        let hi = _mm256_set1_epi32(127);
        let zeros = _mm256_setzero_si256();
        let ones = _mm256_set1_epi32(1);
        for j in (0..end).step_by(8) {
            let x = _mm256_loadu_ps(src.as_ptr().add(j));
            let scaled = _mm256_div_ps(x, dv);
            // This avoids changing values just below a tie by first adding 0.5.
            let integral = _mm256_round_ps(scaled, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            let fraction = _mm256_sub_ps(scaled, integral);
            let up = _mm256_cmp_ps(fraction, half, _CMP_GE_OQ);
            let down = _mm256_cmp_ps(fraction, neg_half, _CMP_LE_OQ);
            let rounded = _mm256_cvttps_epi32(integral);
            let rounded =
                _mm256_add_epi32(rounded, _mm256_and_si256(_mm256_castps_si256(up), ones));
            let rounded =
                _mm256_sub_epi32(rounded, _mm256_and_si256(_mm256_castps_si256(down), ones));
            let rounded = _mm256_min_epi32(_mm256_max_epi32(rounded, lo), hi);

            // Pack q[0..8] in order despite AVX2's independent 128-bit lanes.
            let words = _mm256_packs_epi32(rounded, zeros);
            let first = _mm256_castsi256_si128(words);
            let second = _mm256_extracti128_si256(words, 1);
            let ordered = _mm_unpacklo_epi64(first, second);
            let bytes = _mm_packs_epi16(ordered, _mm_setzero_si128());
            _mm_storel_epi64(dst.as_mut_ptr().add(j).cast(), bytes);
        }
        for j in end..src.len() {
            dst[j] = (src[j] / d).round().clamp(-127.0, 127.0) as i8 as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    // Every AVX-512 Q8_0 kernel against the f64 reference: odd block counts (the odd final
    // block runs with a zero upper half), every row count of a decode panel at a nonzero row
    // offset, odd token counts in the prompt tile, and weights including -128.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_q8_kernels_match_the_scalar_reference_at_odd_shapes() {
        use crate::cpu_selftest::{compare, random_weights, reference_q8, Lcg};
        if !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl"))
            || isa() == "scalar"
        {
            return;
        }
        let vnni = std::arch::is_x86_feature_detected!("avx512vnni");
        let mut rng = Lcg(51);
        for blocks in [1, 2, 3, 7, 11] {
            let cols = blocks * 32;
            let weights = random_weights(8, 8, cols, &mut rng);
            let x = rng.vector(cols, 3.0);
            let input = Q8Input::new(&x, cols);
            let (want, scale) = reference_q8(&weights, 8, cols, &x);
            for rows in 1..=8 {
                let row0 = 8 - rows;
                for use_vnni in [false, true] {
                    if use_vnni && !vnni {
                        continue;
                    }
                    let mut have = vec![0.0; rows];
                    // SAFETY: AVX-512 F/BW/VL (and VNNI when used) were detected above; the
                    // weights hold eight rows of `blocks` blocks.
                    unsafe {
                        if use_vnni {
                            x86::gemv512_vnni(&weights, &input.single, row0, &mut have);
                        } else {
                            x86::gemv512_bw(&weights, &input.single, row0, &mut have);
                        }
                    }
                    let (error, ok) = compare(&have, &want[row0..], &scale[row0..]);
                    assert!(
                        ok,
                        "decode vnni={use_vnni} blocks {blocks} rows {rows}: {error:e}"
                    );
                }
            }
            let rows = 7;
            for batch in [2, 3, 5] {
                let x = rng.vector(batch * cols, 3.0);
                let input = Q8Input::new(&x, cols);
                let (want, scale) = reference_q8(&weights, rows, cols, &x);
                for use_vnni in [false, true] {
                    if use_vnni && !vnni {
                        continue;
                    }
                    let mut output = vec![vec![0.0; batch * rows]];
                    crate::cpu_schedule::row_panels(
                        &mut output,
                        &[rows],
                        batch,
                        4,
                        || Q8PanelScratch::new(blocks),
                        |panel, _, first, out| {
                            let valid = (rows - first).min(4);
                            // SAFETY: features detected above; the panel holds `blocks` records
                            // and `input.packed` whole token pairs of `blocks` blocks.
                            unsafe {
                                x86::pack_panel(&weights, first, valid, panel);
                                if use_vnni {
                                    x86::tile512_vnni(panel, &input.packed, batch, out);
                                } else {
                                    x86::tile512_bw(panel, &input.packed, batch, out);
                                }
                            }
                        },
                    );
                    let (error, ok) = compare(&output[0], &want, &scale);
                    assert!(
                        ok,
                        "prompt vnni={use_vnni} blocks {blocks} batch {batch}: {error:e}"
                    );
                }
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q8_worker_panel_reuses_capacity_and_keeps_nested_calls_exclusive() {
        use super::Q8PanelScratch;
        let mut first = Q8PanelScratch::new(7);
        first[0].d[0] = 123.0;
        let pointer = first.0.as_ptr();
        let capacity = first.0.capacity();
        drop(first);
        let smaller = Q8PanelScratch::new(3);
        assert_eq!(smaller.0.as_ptr(), pointer);
        assert_eq!(smaller.0.capacity(), capacity);
        assert_eq!(smaller[0].d[0], 123.0);
        let mut nested = Q8PanelScratch::new(3);
        assert_ne!(nested.0.as_ptr(), smaller.0.as_ptr());
        nested[0].d[0] = -321.0;
        assert_eq!(smaller[0].d[0], 123.0);
        drop(nested);
        drop(smaller);
        let restored = Q8PanelScratch::new(5);
        assert_eq!(restored.0.as_ptr(), pointer);
        assert_eq!(restored.0.capacity(), capacity);
    }
    use super::*;
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vnni_zero4_matches_vnni_for_unaligned_extreme_weights() {
        if isa() != "avx-vnni" {
            return;
        }
        let row0 = 3;
        for blocks in [1, 3, 80] {
            let rows = row0 + 4;
            let mut weights = Vec::with_capacity(1 + rows * blocks * 34);
            weights.push(0xa5); // Deliberately unalign the first Q8 block.
            for row in 0..rows {
                for block in 0..blocks {
                    let scale = match (row + block) % 4 {
                        0 => 0x3c00u16, // +1
                        1 => 0xbc00u16, // -1
                        2 => 0x0000u16, // zero
                        _ => 0x0001u16, // smallest positive subnormal
                    };
                    weights.extend_from_slice(&scale.to_le_bytes());
                    for i in 0..32 {
                        let q = match (i + row + block) % 6 {
                            0 => i8::MIN,
                            1 => i8::MIN + 1,
                            2 => i8::MAX,
                            3 => i8::MAX - 1,
                            4 => 0,
                            _ => 1,
                        };
                        weights.push(q as u8);
                    }
                }
            }
            let weights = &weights[1..];
            let input = (0..blocks)
                .map(|block| {
                    let src: [f32; 32] = std::array::from_fn(|i| match (i + block) % 4 {
                        0 => 1.0,
                        1 => -1.0,
                        2 => 0.0,
                        _ => 0.5,
                    });
                    let mut q = [0i8; 32];
                    let (d, init) = pack_linear_block(&src, &mut q);
                    XSingle { q, d, init }
                })
                .collect::<Vec<_>>();
            let mut expected = [0.0; 4];
            let mut actual = [0.0; 4];
            // SAFETY: guarded by the AVX-VNNI check above.
            unsafe {
                x86::gemv_vnni(weights, &input, row0, &mut expected);
                x86::gemv_vnni_zero4(weights, &input, row0, &mut actual);
            }
            assert_eq!(
                actual.map(f32::to_bits),
                expected.map(f32::to_bits),
                "block count {blocks}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vnni_zero8_matches_two_zero4_calls_for_unaligned_extreme_weights() {
        if isa() != "avx-vnni" {
            return;
        }
        let row0 = 3;
        for blocks in [1, 3, 80] {
            let rows = row0 + 8;
            let mut weights = Vec::with_capacity(1 + rows * blocks * 34);
            weights.push(0xa5); // Deliberately unalign the first Q8 block.
            for row in 0..rows {
                for block in 0..blocks {
                    let scale = match (row + block) % 4 {
                        0 => 0x3c00u16, // +1
                        1 => 0xbc00u16, // -1
                        2 => 0x0000u16, // zero
                        _ => 0x0001u16, // smallest positive subnormal
                    };
                    weights.extend_from_slice(&scale.to_le_bytes());
                    for i in 0..32 {
                        let q = match (i + row + block) % 6 {
                            0 => i8::MIN,
                            1 => i8::MIN + 1,
                            2 => i8::MAX,
                            3 => i8::MAX - 1,
                            4 => 0,
                            _ => 1,
                        };
                        weights.push(q as u8);
                    }
                }
            }
            let weights = &weights[1..];
            let input = (0..blocks)
                .map(|block| {
                    let src: [f32; 32] = std::array::from_fn(|i| match (i + block) % 4 {
                        0 => 1.0,
                        1 => -1.0,
                        2 => 0.0,
                        _ => 0.5,
                    });
                    let mut q = [0i8; 32];
                    let (d, init) = pack_linear_block(&src, &mut q);
                    XSingle { q, d, init }
                })
                .collect::<Vec<_>>();
            let mut expected = [0.0; 8];
            let mut actual = [0.0; 8];
            // SAFETY: guarded by the AVX-VNNI check above.
            unsafe {
                for (group, out) in expected.chunks_mut(4).enumerate() {
                    x86::gemv_vnni_zero4(weights, &input, row0 + group * 4, out);
                }
                x86::gemv_vnni_zero8(weights, &input, row0, &mut actual);
            }
            assert_eq!(
                actual.map(f32::to_bits),
                expected.map(f32::to_bits),
                "block count {blocks}"
            );
        }
    }

    fn scalar_q8_cache_group(src: &[f32]) -> (u16, Vec<u8>) {
        let amax = src.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let d = amax / 127.0f32;
        let q = src
            .iter()
            .map(|&x| {
                if d == 0.0 {
                    0
                } else {
                    (x / d).round().clamp(-127.0, 127.0) as i8 as u8
                }
            })
            .collect();
        (f16::from_f32(d).to_bits(), q)
    }
    fn next_up(x: f32) -> f32 {
        if x == 0.0 {
            f32::from_bits(1)
        } else {
            f32::from_bits(if x > 0.0 {
                x.to_bits() + 1
            } else {
                x.to_bits() - 1
            })
        }
    }
    fn next_down(x: f32) -> f32 {
        if x == 0.0 {
            f32::from_bits(0x8000_0001)
        } else {
            f32::from_bits(if x > 0.0 {
                x.to_bits() - 1
            } else {
                x.to_bits() + 1
            })
        }
    }

    #[test]
    fn q8_cache_group_encoding_matches_scalar_for_all_group_lengths() {
        for len in 1..=256 {
            let mut state = 0x5eed_1234u32 ^ len as u32;
            let src = (0..len)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    f32::from_bits(state)
                })
                .collect::<Vec<_>>();
            let (expected_scale, expected_q) = scalar_q8_cache_group(&src);
            let mut actual_scale = 0;
            let mut actual_q = vec![0; len];
            encode_q8_cache_group(&src, &mut actual_q, &mut actual_scale);
            assert_eq!(actual_scale, expected_scale, "scale, len={len}");
            assert_eq!(actual_q, expected_q, "q bytes, len={len}");
        }
    }

    #[test]
    fn q8_cache_group_rounding_matches_rust_around_signed_half_steps() {
        let mut src = vec![0.0; 64];
        src[0] = 127.0; // Makes d exactly 1.0.
        for (i, tie) in [0.5f32, 1.5, 2.5, 63.5, 126.5].into_iter().enumerate() {
            src[1 + i * 6] = next_down(tie);
            src[2 + i * 6] = tie;
            src[3 + i * 6] = next_up(tie);
            src[4 + i * 6] = next_up(-tie);
            src[5 + i * 6] = -tie;
            src[6 + i * 6] = next_down(-tie);
        }
        // Includes the known failure of adding 0.5 before truncation.
        assert_eq!(next_down(0.5), 0.499_999_97);
        let (expected_scale, expected_q) = scalar_q8_cache_group(&src);
        let mut actual_scale = 0;
        let mut actual_q = vec![0; src.len()];
        encode_q8_cache_group(&src, &mut actual_q, &mut actual_scale);
        assert_eq!(actual_scale, expected_scale);
        assert_eq!(actual_q, expected_q);
    }

    #[test]
    fn q8_cache_group_preserves_zero_subnormal_nonfinite_semantics() {
        let inputs = [
            vec![0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
            vec![f32::from_bits(1); 33],
            vec![
                f32::INFINITY,
                3.0,
                f32::NEG_INFINITY,
                f32::NAN,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            vec![f32::NAN; 17],
        ];
        for src in inputs {
            let (expected_scale, expected_q) = scalar_q8_cache_group(&src);
            let mut actual_scale = 0;
            let mut actual_q = vec![0; src.len()];
            encode_q8_cache_group(&src, &mut actual_q, &mut actual_scale);
            assert_eq!(actual_scale, expected_scale);
            assert_eq!(actual_q, expected_q);
        }
    }

    #[test]
    fn q8_cache_group_writes_only_odd_byte_window() {
        let src = [127.0, 0.5, -0.5, 1.5, -1.5, 0.0, 0.0, 0.0, 2.5];
        let (expected_scale, expected_q) = scalar_q8_cache_group(&src);
        let mut storage = [0xa5u8; 24];
        let mut scale = 0;
        encode_q8_cache_group(&src, &mut storage[7..16], &mut scale);
        assert_eq!(scale, expected_scale);
        assert_eq!(&storage[..7], &[0xa5; 7]);
        assert_eq!(&storage[7..16], expected_q.as_slice());
        assert_eq!(&storage[16..], &[0xa5; 8]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_activation_packing_preserves_rounding_and_signed_sums() {
        if isa() == "scalar" {
            return;
        }
        for amplitude in [0.0, 0.001, 1.0, 1000.0] {
            let mut input = std::array::from_fn::<_, 32, _>(|i| (i as f32 - 15.5) * amplitude);
            input[31] = 127.0 * amplitude;
            for reciprocal in [false, true] {
                let amax = input.iter().map(|v| v.abs()).fold(0.0, f32::max);
                let exact = amax / 127.0;
                let inverse = if amax == 0.0 {
                    0.0
                } else if reciprocal {
                    exact.recip()
                } else {
                    127.0 / amax
                };
                let expected = input.map(|v| (v * inverse).round_ties_even() as i8);
                let mut actual = [0; 32];
                // SAFETY: guarded by the scalar check above; one 32-value block.
                let (d, e, sums) = unsafe { x86::pack_q8_block(&input, &mut actual, reciprocal) };
                assert_eq!(actual, expected);
                assert_eq!(e, exact);
                assert_eq!(d, f16::from_f32(exact).to_f32());
                assert_eq!(
                    sums,
                    [
                        expected[..16].iter().map(|&v| v as i32).sum::<i32>(),
                        expected[16..].iter().map(|&v| v as i32).sum::<i32>()
                    ]
                );
                let mut k = [0; 32];
                // SAFETY: guarded by the scalar check above; one 32-value block.
                let signed_sums = unsafe { x86::quantize_i8_block(&input, -inverse, &mut k) };
                assert_eq!(k, expected.map(|q| -q));
                assert_eq!(signed_sums, sums.map(|s| -s));
            }
        }
    }
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn unsigned_avx2_dot_does_not_saturate() {
        if isa() == "scalar" {
            return;
        }
        // SAFETY: guarded by the scalar check above.
        unsafe {
            use std::arch::x86_64::*;
            let u = [255_u8; 32];
            let s = [-128_i8; 32];
            let dot = x86::dot_avx2(
                _mm256_set1_epi32(17),
                _mm256_loadu_si256(u.as_ptr().cast()),
                _mm256_loadu_si256(s.as_ptr().cast()),
            );
            let mut lanes = [0_i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast(), dot);
            assert_eq!(lanes, [17 - 4 * 255 * 128; 8]);
        }
    }
    #[test]
    fn q8_extremes_odd_rows_and_tokens() {
        if isa() == "scalar" {
            return;
        }
        for rows in [1, 3, 4, 5, 13] {
            for batch in [1, 2, 3, 9] {
                let cols = 96;
                let mut w = Vec::new();
                for r in 0..rows {
                    for kb in 0..3 {
                        w.extend_from_slice(&f16::from_f32(0.015 * (kb + 1) as f32).to_le_bytes());
                        w.extend((0..32).map(|i| ((i * 113 + r * 31 + kb * 17) % 256) as u8));
                    }
                }
                let x = (0..batch * cols)
                    .map(|i| ((i * 43 % 257) as f32 - 128.0) / 13.0)
                    .collect::<Vec<_>>();
                let result = q8_linear(&w, &x, rows, cols, None).unwrap();
                let short_rows = rows.min(3);
                let bias = vec![1.25; rows];
                let shared = q8_linear_group(
                    &[
                        Q8Projection {
                            weights: &w,
                            rows,
                            bias: Some(&bias),
                        },
                        Q8Projection {
                            weights: &w[..short_rows * 3 * 34],
                            rows: short_rows,
                            bias: None,
                        },
                    ],
                    &Q8Input::new(&x, cols),
                )
                .unwrap();
                assert_eq!(
                    shared[0],
                    result.iter().map(|v| v + 1.25).collect::<Vec<_>>()
                );
                assert_eq!(
                    shared[1],
                    q8_linear(&w[..short_rows * 3 * 34], &x, short_rows, cols, None).unwrap()
                );
                for b in 0..batch {
                    for r in 0..rows {
                        let mut expected = 0.0;
                        for kb in 0..3 {
                            let src = &x[b * cols + kb * 32..][..32];
                            let amax = src.iter().map(|x| x.abs()).fold(0.0, f32::max);
                            let dx = f16::from_f32(amax / 127.0).to_f32();
                            let block = &w[(r * 3 + kb) * 34..][..34];
                            let dw = f16::from_le_bytes([block[0], block[1]]).to_f32();
                            let sum = src
                                .iter()
                                .enumerate()
                                .map(|(i, v)| {
                                    (v * (127.0 / amax)).round_ties_even() as i32
                                        * block[i + 2] as i8 as i32
                                })
                                .sum::<i32>();
                            expected += sum as f32 * dx * dw;
                        }
                        assert!(
                            (result[b * rows + r] - expected).abs() < 0.001,
                            "{rows}/{batch}/{b}/{r}"
                        );
                    }
                }
            }
        }
        assert_eq!(
            q8_linear(&[0; 34], &[0.0; 32], 1, 32, Some(&[2.5])).unwrap(),
            [2.5]
        );
    }

    #[test]
    fn q8_input_and_sibling_outputs_reuse_capacity_without_changing_results() {
        if isa() == "scalar" {
            return;
        }
        let rows = [7, 1, 13];
        let mut input = Q8Input::default();
        let mut output = vec![Vec::new(); rows.len()];
        let cases = [
            (32, 0),
            (64, 1),
            (64, 1),
            (32, 3),
            (64, 16),
            (64, 3),
            (64, 3),
        ];
        let mut same_shape_storage = None;

        for (case, &(cols, batch)) in cases.iter().enumerate() {
            let nb = cols / 32;
            let mut weights = Vec::with_capacity(rows.len());
            let mut biases = Vec::with_capacity(rows.len());
            for (member, &nrows) in rows.iter().enumerate() {
                let mut member_weights = Vec::with_capacity(nrows * nb * 34);
                for row in 0..nrows {
                    for block in 0..nb {
                        let scale = 0.015625 * (1 + member + row + block) as f32;
                        member_weights.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
                        member_weights.extend((0..32).map(|i| {
                            let q =
                                ((i * 37 + row * 19 + member * 23 + block * 11) % 255) as i16 - 127;
                            q as i8 as u8
                        }));
                    }
                }
                weights.push(member_weights);
                biases.push(
                    (0..nrows)
                        .map(|r| (member as f32 - r as f32) * 0.03125)
                        .collect::<Vec<_>>(),
                );
            }
            let projections = weights
                .iter()
                .zip(&biases)
                .enumerate()
                .map(|(member, (weights, bias))| Q8Projection {
                    weights,
                    rows: rows[member],
                    bias: Some(bias),
                })
                .collect::<Vec<_>>();
            let x = (0..batch * cols)
                .map(|i| ((i * 37 + case * 29) % 211) as f32 / 37.0 - 2.0)
                .collect::<Vec<_>>();

            input.prepare(&x, cols);
            output.resize_with(projections.len(), Vec::new);
            output.iter_mut().for_each(|out| out.fill(9876.5));
            q8_linear_group_into(&projections, &input, &mut output).unwrap();
            let expected = q8_linear_group(&projections, &Q8Input::new(&x, cols)).unwrap();
            let output_bits = output
                .iter()
                .map(|out| out.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let expected_bits = expected
                .iter()
                .map(|out| out.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            assert_eq!(
                output_bits, expected_bits,
                "case {case}: cols={cols}, batch={batch}"
            );

            if case == 1 || case == 5 {
                same_shape_storage = Some((
                    if batch == 1 {
                        input.single.as_ptr() as usize
                    } else {
                        input.packed.as_ptr() as usize
                    },
                    if batch == 1 {
                        input.single.capacity()
                    } else {
                        input.packed.capacity()
                    },
                    output
                        .iter()
                        .map(|out| (out.as_ptr() as usize, out.capacity()))
                        .collect::<Vec<_>>(),
                ));
            } else if case == 2 || case == 6 {
                let (activation_ptr, activation_capacity, output_storage) =
                    same_shape_storage.take().unwrap();
                let (active_ptr, active_capacity) = if batch == 1 {
                    (input.single.as_ptr() as usize, input.single.capacity())
                } else {
                    (input.packed.as_ptr() as usize, input.packed.capacity())
                };
                assert_eq!(active_ptr, activation_ptr);
                assert_eq!(active_capacity, activation_capacity);
                assert_eq!(
                    output
                        .iter()
                        .map(|out| (out.as_ptr() as usize, out.capacity()))
                        .collect::<Vec<_>>(),
                    output_storage
                );
            }
        }
    }
}
