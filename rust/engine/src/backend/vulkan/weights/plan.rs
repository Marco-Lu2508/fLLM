//! Dispatch planning of `Stream::quantized_linear` on the Vulkan backend: which shader, specialization constants, push constants, workgroup
//! counts, bindings and which activation prepass a `rows x cols` weight needs for `batch` activation rows on a device with given
//! capabilities. Pure CPU code (no Vulkan calls); package 4 turns a [`LinearPlan`] into pipeline creation and `vkCmdDispatch`.

use super::layout::{DeviceLayout, Geometry};
use crate::backend::{BackendError, QuantType};

/// Size of the push-constant block of the quantized-linear shaders (`QPush` in `_q_common.glsl`).
pub const PUSH_BYTES: usize = 48;
/// Largest batch served by the matrix-vector shaders; bigger batches use the tiled matrix-matrix shaders.
pub const MATVEC_MAX_BATCH: u32 = 8;

/// Largest batch served by the matrix-vector shaders: [`MATVEC_MAX_BATCH`], or `FLLM_VK_MV_MAX` (1..=8) for measuring a device.
pub fn matvec_max() -> u32 {
    static MAX: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("FLLM_VK_MV_MAX").ok().and_then(|v| v.trim().parse::<u32>().ok()).filter(|v| (1..=MATVEC_MAX_BATCH).contains(v)).unwrap_or(MATVEC_MAX_BATCH)
    })
}
/// `maxComputeWorkGroupCount` that every Vulkan device offers per dimension; larger grids are split over x and y.
pub const MAX_GROUPS_PER_DIM: u64 = 65535;

/// Shared memory (bytes) the cooperative-matrix shader needs: two f16 tiles of 64 x 40 and four 16 x 16 f32 staging tiles.
const COOPMAT_SHARED: u32 = 2 * 64 * 40 * 2 + 4 * 256 * 4;
/// Shared memory of the wide cooperative-matrix shader (`mm_<fmt>_cw`): f16 tiles of 128 + 64 rows x 72 and eight 16 x 16 f32 staging tiles.
const COOPMAT_WIDE_SHARED: u32 = (128 + 64) * 72 * 2 + 8 * 256 * 4;
/// PCI vendor id of NVIDIA.
const VENDOR_NVIDIA: u32 = 0x10de;

/// Smallest batch served by the wide 128 x 64 cooperative-matrix tile (`mm_<fmt>_cw`, NVIDIA): `FLLM_VK_COOP_WIDE` = `0` / `off` never uses it, a number
/// sets the threshold (default 64); smaller batches keep the 64 x 64 tile of `mm_<fmt>_c`.
fn coop_wide_min_batch() -> Option<u32> {
    static MIN: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| match std::env::var("FLLM_VK_COOP_WIDE") {
        Ok(v) if matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "no" | "false") => None,
        Ok(v) => Some(v.trim().parse::<u32>().ok().filter(|&n| n > 0).unwrap_or(64)),
        Err(_) => Some(64),
    })
}
/// Shared memory of the tiled f32 / int8 matrix-matrix shaders.
const MM_SCALAR_SHARED: u32 = 2 * 16 * 16 * 16;
const MM_IDOT_SHARED: u32 = 64 * 8 * 4 * 2 + 64 * 4 * 4 * 2 + 64 * 4;

/// The device properties the planner looks at (a subset of what the Vulkan backend queries; package 4 fills it in).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VkCapsLike {
    /// `VK_KHR_cooperative_matrix` with the 16x16x16 shape (float16 A/B, float32 accumulator, subgroup scope), `vulkanMemoryModel` enabled.
    pub coop_matrix: bool,
    /// `shaderIntegerDotProduct` (VK_KHR_shader_integer_dot_product / Vulkan 1.3).
    pub int_dot: bool,
    /// Subgroup size the compute pipelines run with (fixed, e.g. through `VK_EXT_subgroup_size_control`); 32 if unknown.
    pub subgroup_size: u32,
    /// `shaderFloat16` (needed by the cooperative-matrix shaders).
    pub fp16: bool,
    /// `maxComputeSharedMemorySize` in bytes.
    pub max_shared: u32,
    /// Largest workgroup the pipelines may use (`maxComputeWorkGroupInvocations`, limited by `maxComputeWorkGroupSize[0]`).
    pub max_workgroup: u32,
    /// PCI vendor id (0x10de NVIDIA, 0x1002 AMD, 0x8086 Intel, 0x106b Apple, 0x13b5 ARM, 0x5143 Qualcomm); 0 = unknown.
    pub vendor: u32,
}

impl Default for VkCapsLike {
    /// The weakest device: no optional features, 16 KiB of shared memory.
    fn default() -> Self {
        Self {
            coop_matrix: false,
            int_dot: false,
            subgroup_size: 32,
            fp16: false,
            max_shared: 16384,
            max_workgroup: 256,
            vendor: 0,
        }
    }
}

/// Arithmetic flavour of a quantized-linear shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// f32 activations and arithmetic (suffix `_s`): runs everywhere and meets the exact selftest tolerance.
    Scalar,
    /// int8 activations and 4 x int8 dot products (suffix `_i`); activations are quantized by a prepass.
    IntDot,
    /// cooperative matrices with f16 tiles (suffix `_c`), matrix-matrix only.
    CoopMat,
}

impl Variant {
    pub fn suffix(self) -> char {
        match self {
            Variant::Scalar => 's',
            Variant::IntDot => 'i',
            Variant::CoopMat => 'c',
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanKind {
    /// `mv_*`: `columns` (1..=8) activation rows, `columns` is specialization constant 2.
    MatVec { columns: u32 },
    /// `mm_*`: tiled, any batch.
    MatMul,
}

/// A specialization constant of the pipeline (`layout(constant_id = id)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecConstant {
    pub id: u32,
    pub value: u32,
}

/// What a storage-buffer binding of the main shader holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingRole {
    /// The weight buffer in device layout (`PackedWeight::bytes`).
    Weights,
    /// The f32 activations (`batch x cols`, offset a multiple of 16 bytes).
    Activations,
    /// Output of the activation prepass (used instead of `Activations` when the plan has a prepass).
    PrepassOutput,
    /// f32 output, `batch x out_stride`.
    Output,
    /// f32 bias, `rows` entries; bind any small buffer when the plan has no bias (flag bit 0 is clear then).
    Bias,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingSlot {
    pub binding: u32,
    pub role: BindingRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrepassKind {
    /// `quant_act_q8[_perm]`: f32 -> int8 blocks of 32 with float scales.
    QuantizeActivations,
    /// `quant_permute`: gathers the columns of the activations (act-order GPTQ with f32 shaders).
    PermuteActivations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrepassRole {
    /// The f32 activations of the layer.
    Activations,
    /// The scratch buffer (`PrepassPlan::output_bytes`) the main shader reads.
    Output,
    /// The Q4ZP column permutation (`DeviceWeight::perm`, `PackedWeight::perm` as u32).
    Perm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrepassBinding {
    pub binding: u32,
    pub role: PrepassRole,
}

/// A small compute pass that must run before the main shader (and after the previous writer of the activations).
/// Several weights that read the same activations (q, k, v) may share one prepass result when `shader_name`, `kind`, `perm` and the
/// activation range are the same.
#[derive(Debug, Clone, PartialEq)]
pub struct PrepassPlan {
    pub kind: PrepassKind,
    pub shader_name: String,
    pub push_constants: Vec<u8>,
    pub groups: [u32; 3],
    pub bindings: Vec<PrepassBinding>,
    /// Size of the scratch buffer the prepass writes.
    pub output_bytes: usize,
    /// Whether the prepass reads the permutation table (Q4ZP).
    pub uses_perm: bool,
}

/// Everything needed to record one `quantized_linear`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearPlan {
    pub layout: DeviceLayout,
    pub variant: Variant,
    pub kind: PlanKind,
    pub shader_name: String,
    pub spec_constants: Vec<SpecConstant>,
    /// `QPush` of `_q_common.glsl`, 48 bytes little endian.
    pub push_constants: Vec<u8>,
    pub groups: [u32; 3],
    pub bindings: Vec<BindingSlot>,
    pub activation_prepass: Option<PrepassPlan>,
}

/// Optional inputs of the planner.
pub struct PlanOptions<'a> {
    /// The weight has a bias (flag bit 0 of the push constants).
    pub has_bias: bool,
    /// Per-tensor factor of the dot products (`DeviceWeight::scale`).
    pub scale: f32,
    /// First output feature of this dispatch (weights split by rows for `maxStorageBufferRange`).
    pub row_base: u32,
    /// Floats between two output rows; defaults to `rows`.
    pub out_stride: Option<u32>,
    /// Only the f32 shaders (meets the selftest's FP32 error bound); default false.
    pub exact: bool,
    /// Use this variant (an error if it is not possible).
    pub force: Option<Variant>,
    /// Tells whether a shader name is in the SPIR-V registry; default: look it up in `shaders::SHADERS`.
    pub available: Option<&'a dyn Fn(&str) -> bool>,
}

impl Default for PlanOptions<'_> {
    fn default() -> Self {
        Self {
            has_bias: false,
            scale: 1.0,
            row_base: 0,
            out_stride: None,
            exact: exact_math_from_env(),
            force: None,
            available: None,
        }
    }
}

/// `FLLM_VULKAN_EXACT=1`: plan only the f32 shaders (used by `selftest-gpu`, whose tolerance is the f32 worst-case bound). Read once per process.
pub fn exact_math_from_env() -> bool {
    static EXACT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EXACT
        .get_or_init(|| std::env::var("FLLM_VULKAN_EXACT").is_ok_and(|v| !v.is_empty() && v != "0"))
}

fn registry_has(name: &str) -> bool {
    super::super::shaders::SHADERS
        .iter()
        .any(|(n, _)| *n == name)
}

/// Word-exact image of `QPush` in `_q_common.glsl`.
struct QPush {
    rows: u32,
    cols: u32,
    batch: u32,
    flags: u32,
    scale: f32,
    nb: u32,
    off: [u32; 3],
    out_stride: u32,
    row_base: u32,
    xq_d: u32,
}

impl QPush {
    fn to_bytes(&self) -> Vec<u8> {
        let words = [
            self.rows,
            self.cols,
            self.batch,
            self.flags,
            self.scale.to_bits(),
            self.nb,
            self.off[0],
            self.off[1],
            self.off[2],
            self.out_stride,
            self.row_base,
            self.xq_d,
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Workgroup counts for `n` workgroups of a 1-D problem, split over x and y at 65535 (the shaders use `x + y * numWorkGroups.x`).
pub fn split_groups(n: u64) -> [u32; 3] {
    let n = n.max(1);
    if n <= MAX_GROUPS_PER_DIM {
        [n as u32, 1, 1]
    } else {
        [
            MAX_GROUPS_PER_DIM as u32,
            n.div_ceil(MAX_GROUPS_PER_DIM) as u32,
            1,
        ]
    }
}

/// Workgroup size and rows per workgroup of the matrix-vector shaders for a device.
fn matvec_tuning(caps: &VkCapsLike, columns: u32) -> (u32, u32) {
    let (mut wg, mut rows) = match caps.vendor {
        // AMD: two rows per group decode 12 % faster than one on a Radeon 840M (Llama-3.2-3B Q4_K_M: 27.4 -> 30.8 tok/s; work groups of 32 / 128 / 256 were slower)
        0x1002 => (64u32, 2u32),
        0x10de => (128u32, 2u32),
        _ => (64, 2),
    };
    // tuning knobs for measuring a device: FLLM_VK_MV_WG (power of two, 32..=256) and FLLM_VK_MV_ROWS (1..=8)
    static KNOBS: std::sync::OnceLock<(Option<u32>, Option<u32>)> = std::sync::OnceLock::new();
    let (knob_wg, knob_rows) = *KNOBS.get_or_init(|| {
        let read = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u32>().ok());
        (
            read("FLLM_VK_MV_WG").filter(|v| v.is_power_of_two() && (32..=256).contains(v)),
            read("FLLM_VK_MV_ROWS").filter(|v| (1..=8).contains(v)),
        )
    });
    wg = knob_wg.unwrap_or(wg);
    rows = knob_rows.unwrap_or(rows);
    let shared = |wg: u32, rows: u32| rows * columns * wg * 4;
    while shared(wg, rows) > caps.max_shared && wg > 32 {
        wg /= 2;
    }
    if shared(wg, rows) > caps.max_shared {
        rows = 1;
    }
    (wg, rows)
}

/// Switches of the prompt-path kernel `mm_q.comp` (`mmq_<fmt>_i`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmqKnobs {
    /// `FLLM_VK_MM=0` / `legacy` / `off`: never plan the `mmq_*` kernels (the `mm_*_i` kernels serve every batch above the matrix-vector range).
    pub enabled: bool,
    /// `FLLM_VK_MM_MIN_BATCH`: smallest batch served by `mmq_*` (default: the first batch above the matrix-vector range).
    pub min_batch: u32,
    /// `FLLM_VK_MM_BM` (32, 64, 128 or 256): weight rows per workgroup.
    pub bm: Option<u32>,
    /// `FLLM_VK_MM_BN` (32, 64 or 128): tokens per workgroup.
    pub bn: Option<u32>,
    /// `FLLM_VK_MM_TM` (4 or 8): weight rows per invocation.
    pub tm: Option<u32>,
    /// `FLLM_VK_MM_KS` (1 or 2): blocks of 32 values per K step (one barrier per step).
    pub ks: Option<u32>,
}

impl MmqKnobs {
    pub const DEFAULT: MmqKnobs = MmqKnobs {
        enabled: true,
        min_batch: 1,
        bm: None,
        bn: None,
        tm: None,
        ks: None,
    };

    /// Reads the `FLLM_VK_MM*` variables.
    pub fn from_env() -> Self {
        let num = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u32>().ok());
        let off = std::env::var("FLLM_VK_MM").is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "legacy" | "false"));
        MmqKnobs {
            enabled: !off,
            min_batch: num("FLLM_VK_MM_MIN_BATCH").unwrap_or(Self::DEFAULT.min_batch).max(matvec_max() + 1),
            bm: num("FLLM_VK_MM_BM").filter(|v| matches!(v, 32 | 64 | 128 | 256)),
            bn: num("FLLM_VK_MM_BN").filter(|v| matches!(v, 32 | 64 | 128)),
            tm: num("FLLM_VK_MM_TM").filter(|v| matches!(v, 4 | 8)),
            ks: num("FLLM_VK_MM_KS").filter(|v| matches!(v, 1 | 2)),
        }
    }

    fn get() -> Self {
        static KNOBS: std::sync::OnceLock<MmqKnobs> = std::sync::OnceLock::new();
        *KNOBS.get_or_init(Self::from_env)
    }
}

/// Shared memory (bytes) of `mm_q.comp` for a `bm x bn` tile and one block per K step (the worst case: two scale groups per step and the additive term).
pub fn mmq_shared_bytes(bm: u32, bn: u32) -> u32 {
    4 * (24 * bm + 22 * bn)
}

/// Tile `(BM, BN)` of the prompt-path kernel for a device and a batch with 4 x 4 outputs per invocation, `None` when the old kernels should serve
/// it (see [`mmq_tile_tm`]).
pub fn mmq_tile(caps: &VkCapsLike, batch: u32, knobs: &MmqKnobs) -> Option<(u32, u32)> {
    mmq_tile_tm(caps, batch, &MmqKnobs { tm: Some(4), ks: Some(1), ..*knobs }).map(|(bm, bn, _, _)| (bm, bn))
}

/// Tile `(BM, BN, TM, KS)` of the prompt-path kernel for a device and a batch, `None` when the old kernels should serve it (switched off, batch below
/// the threshold, or a device whose limits the tile does not fit). The workgroup has `BM * BN / (TM * 4)` invocations: every invocation owns
/// TM x 4 outputs. The default is 128 x 64 with TM 8 (256 invocations, ~18 KiB of shared memory), which suits subgroup sizes 16 to 64; the knobs
/// override it for measuring.
pub fn mmq_tile_tm(caps: &VkCapsLike, batch: u32, knobs: &MmqKnobs) -> Option<(u32, u32, u32, u32)> {
    if !knobs.enabled || batch < knobs.min_batch.max(matvec_max() + 1) {
        return None;
    }
    // 8 rows per invocation read half as many shared-memory words per dot product as 4 (measured +15-20 % prompt speed on Intel Xe-LPG)
    let tm = knobs.tm.unwrap_or(8);
    let ks = knobs.ks.unwrap_or(1);
    let mut bm = knobs.bm.unwrap_or(if tm == 8 { 128 } else { 64 });
    // AMD with a long prompt: 128 tokens per tile (the rows then drop to 64 for a workgroup of 256): every weight is read from memory half as often
    // (Radeon 840M, Llama-3.2-3B Q4_K_M, 500-token prompt, interleaved runs: 225 -> 249 tok/s)
    let mut bn = knobs.bn.unwrap_or(if caps.vendor == 0x1002 && tm == 8 && batch >= 128 { 128 } else { 64 });
    let wg = |bm: u32, bn: u32| bm * bn / (tm * 4);
    while wg(bm, bn) > caps.max_workgroup && bm > 32 {
        bm /= 2;
    }
    while wg(bm, bn) > caps.max_workgroup && bn > 64 {
        bn /= 2;
    }
    while ks * mmq_shared_bytes(bm, bn) > caps.max_shared {
        if bn > 64 {
            bn /= 2;
        } else if bm > 32 {
            bm /= 2;
        } else {
            return None;
        }
    }
    // the loaders loop, so any workgroup that fits the device and has whole rows of invocations works
    (bm >= tm && wg(bm, bn) >= 1 && wg(bm, bn) <= caps.max_workgroup).then_some((bm, bn, tm, ks))
}

/// Plan with the default options (`exact` from `FLLM_VULKAN_EXACT`, no bias, scale 1).
pub fn plan_quantized_linear(
    quant: QuantType,
    rows: usize,
    cols: usize,
    batch: usize,
    caps: &VkCapsLike,
) -> Result<LinearPlan, BackendError> {
    plan_quantized_linear_with(quant, rows, cols, batch, caps, &PlanOptions::default())
}

/// Plans `out[b][r] = scale * dot(x[b], w[r]) + bias[r]` for a `rows x cols` weight of `quant` and `batch` activation rows.
pub fn plan_quantized_linear_with(
    quant: QuantType,
    rows: usize,
    cols: usize,
    batch: usize,
    caps: &VkCapsLike,
    options: &PlanOptions<'_>,
) -> Result<LinearPlan, BackendError> {
    let layout = DeviceLayout::new(quant, rows, cols)?;
    if batch == 0 {
        return Err(BackendError::InvalidShape(
            "quantized_linear with an empty batch".into(),
        ));
    }
    let (rows32, cols32, batch32) = match (
        u32::try_from(rows),
        u32::try_from(cols),
        u32::try_from(batch),
    ) {
        (Ok(r), Ok(c), Ok(b)) => (r, c, b),
        _ => {
            return Err(BackendError::InvalidShape(format!(
                "{rows} x {cols} x {batch} exceeds 32-bit indexing"
            )))
        }
    };
    if (batch as u64) * (cols as u64) >= (u32::MAX as u64) / 2 {
        return Err(BackendError::InvalidShape(format!(
            "{batch} x {cols} activations exceed 32-bit indexing"
        )));
    }
    let tag = layout.info.tag;
    let perm = super::layout::uses_perm(quant);
    let default_available;
    let available: &dyn Fn(&str) -> bool = match options.available {
        Some(f) => f,
        None => {
            default_available = registry_has;
            &default_available
        }
    };
    let matvec = batch32 <= matvec_max();
    let prefix = if matvec { "mv" } else { "mm" };
    let prepass_name = |variant: Variant| -> String {
        match variant {
            Variant::IntDot if perm => "quant_act_q8_perm".to_string(),
            Variant::IntDot => "quant_act_q8".to_string(),
            _ if perm => "quant_permute".to_string(),
            _ => String::new(),
        }
    };
    let usable = |variant: Variant| -> bool {
        let name = format!("{prefix}_{tag}_{}", variant.suffix());
        let needs_prepass = variant == Variant::IntDot || perm;
        let shared_ok = match (variant, matvec) {
            (Variant::CoopMat, _) => caps.max_shared >= COOPMAT_SHARED,
            (Variant::IntDot, false) => caps.max_shared >= MM_IDOT_SHARED,
            (Variant::Scalar, false) => caps.max_shared >= MM_SCALAR_SHARED,
            _ => true,
        };
        let feature_ok = match variant {
            Variant::Scalar => true,
            Variant::IntDot => caps.int_dot,
            Variant::CoopMat => {
                !matvec && caps.coop_matrix && caps.fp16 && (8..=128).contains(&caps.subgroup_size)
            }
        };
        // float-chunk formats (FP8, NF4, FP4) have the f32 shaders only
        feature_ok
            && (variant == Variant::Scalar || !layout.info.float_chunk)
            && shared_ok
            && available(&name)
            && (!needs_prepass || available(&prepass_name(variant)))
    };
    let variant = match options.force {
        Some(v) => {
            if !usable(v) {
                return Err(BackendError::UnsupportedOperation(format!(
                    "quantized_linear: variant {v:?} is not usable for {quant:?} (batch {batch})"
                )));
            }
            v
        }
        None => {
            let order: &[Variant] = match (matvec, options.exact) {
                (_, true) => &[Variant::Scalar],
                (true, false) => &[Variant::IntDot, Variant::Scalar],
                (false, false) => &[Variant::CoopMat, Variant::IntDot, Variant::Scalar],
            };
            *order.iter().find(|&&v| usable(v)).ok_or_else(|| {
                BackendError::UnsupportedOperation(format!(
                    "quantized_linear: no Vulkan shader for {quant:?} (batch {batch})"
                ))
            })?
        }
    };

    // prompt path: the tiled kernel of mm_q.comp replaces mm_<fmt>_i for batches from the threshold up
    let mmq = if !matvec && variant == Variant::IntDot && options.force.is_none() {
        let name = format!("mmq_{tag}_i");
        mmq_tile_tm(caps, batch32, &MmqKnobs::get()).filter(|_| available(&name)).map(|tile| (name, tile))
    } else {
        None
    };
    // NVIDIA: the 128 x 64 cooperative-matrix tile (K step 64) for prompt-sized batches
    let coop_wide = !matvec
        && variant == Variant::CoopMat
        && caps.vendor == VENDOR_NVIDIA
        && caps.subgroup_size == 32
        && caps.max_workgroup >= 256
        && caps.max_shared >= COOPMAT_WIDE_SHARED
        && cols32 % 64 == 0
        && coop_wide_min_batch().is_some_and(|min| batch32 >= min)
        && available(&format!("mm_{tag}_cw"));
    let shader_name = match &mmq {
        Some((name, _)) => name.clone(),
        None if coop_wide => format!("mm_{tag}_cw"),
        None => format!("{prefix}_{tag}_{}", variant.suffix()),
    };
    let out_stride = options.out_stride.unwrap_or(rows32);
    let xq_d = batch32 * cols32 / 4;
    let push = QPush {
        rows: rows32,
        cols: cols32,
        batch: batch32,
        flags: u32::from(options.has_bias),
        scale: options.scale,
        nb: layout.blocks_per_row as u32,
        off: [
            layout.plane_word_offset[1],
            // the format parameter replaces the offset of the first plane that is unused (EXL2 uses planes 1 and 2: it takes the third slot)
            if layout.param != 0 && !matches!(layout.info.geometry, Geometry::Exl2 { .. }) { layout.param } else { layout.plane_word_offset[2] },
            if matches!(layout.info.geometry, Geometry::Exl2 { .. }) { layout.param } else { layout.plane_word_offset[3] },
        ],
        out_stride,
        row_base: options.row_base,
        xq_d,
    };

    let (kind, spec_constants, groups) = if matvec {
        let (wg, per_group) = matvec_tuning(caps, batch32);
        (
            PlanKind::MatVec { columns: batch32 },
            vec![
                SpecConstant { id: 0, value: wg },
                SpecConstant {
                    id: 1,
                    value: per_group,
                },
                SpecConstant {
                    id: 2,
                    value: batch32,
                },
            ],
            split_groups((rows as u64).div_ceil(per_group as u64)),
        )
    } else if let Some((_, (bm, bn, tm, ks))) = &mmq {
        (
            PlanKind::MatMul,
            vec![
                SpecConstant {
                    id: 0,
                    value: bm * bn / (tm * 4),
                },
                SpecConstant { id: 1, value: *bm },
                SpecConstant { id: 2, value: *bn },
                SpecConstant { id: 3, value: *tm },
                SpecConstant { id: 4, value: *ks },
            ],
            [
                (rows as u64).div_ceil(*bm as u64) as u32,
                (batch as u64).div_ceil(*bn as u64) as u32,
                1,
            ],
        )
    } else {
        let spec = if variant == Variant::CoopMat && !coop_wide {
            vec![SpecConstant {
                id: 0,
                value: 4 * caps.subgroup_size,
            }]
        } else {
            Vec::new()
        };
        (
            PlanKind::MatMul,
            spec,
            {
                let (tm, tn) = if coop_wide { (128, 64) } else { (64, 64) };
                [
                    (rows as u64).div_ceil(tm) as u32,
                    (batch as u64).div_ceil(tn) as u32,
                    1,
                ]
            },
        )
    };

    let activation_prepass = if variant == Variant::IntDot {
        let blocks = (batch as u64) * (cols as u64) / 32;
        let mut bindings = vec![
            PrepassBinding {
                binding: 0,
                role: PrepassRole::Activations,
            },
            PrepassBinding {
                binding: 1,
                role: PrepassRole::Output,
            },
        ];
        if perm {
            bindings.push(PrepassBinding {
                binding: 2,
                role: PrepassRole::Perm,
            });
        }
        Some(PrepassPlan {
            kind: PrepassKind::QuantizeActivations,
            shader_name: prepass_name(variant),
            push_constants: words(&[cols32, batch32, xq_d]),
            groups: split_groups(blocks.div_ceil(128)),
            bindings,
            // a multiple of 16 bytes: mm_q.comp reads the result through a uvec4 view
            output_bytes: (xq_d as usize + blocks as usize).next_multiple_of(4) * 4,
            uses_perm: perm,
        })
    } else if perm {
        let total = (batch as u64) * (cols as u64);
        Some(PrepassPlan {
            kind: PrepassKind::PermuteActivations,
            shader_name: prepass_name(variant),
            push_constants: words(&[cols32, total as u32]),
            groups: split_groups(total.div_ceil(256)),
            bindings: vec![
                PrepassBinding {
                    binding: 0,
                    role: PrepassRole::Activations,
                },
                PrepassBinding {
                    binding: 1,
                    role: PrepassRole::Output,
                },
                PrepassBinding {
                    binding: 2,
                    role: PrepassRole::Perm,
                },
            ],
            output_bytes: total as usize * 4,
            uses_perm: true,
        })
    } else {
        None
    };

    let bindings = vec![
        BindingSlot {
            binding: 0,
            role: BindingRole::Weights,
        },
        BindingSlot {
            binding: 1,
            role: if activation_prepass.is_some() {
                BindingRole::PrepassOutput
            } else {
                BindingRole::Activations
            },
        },
        BindingSlot {
            binding: 2,
            role: BindingRole::Output,
        },
        BindingSlot {
            binding: 3,
            role: BindingRole::Bias,
        },
    ];
    Ok(LinearPlan {
        layout,
        variant,
        kind,
        shader_name,
        spec_constants,
        push_constants: push.to_bytes(),
        groups,
        bindings,
        activation_prepass,
    })
}
