//! Dispatch planning of `Stream::quantized_linear` on Metal: kernel choice, function constants, grid, parameter block.
//!
//! The planner never touches Metal; the backend resolves `LinearPlan::kernel_name` with the function constants through the
//! pipeline cache, binds the buffers at the slots of [`BufferSlots`] and dispatches `threadgroups` x `threads_per_group`. The kernel
//! side of the contract is `rust/engine/metal/INTERFACE_QUANT.md`.

use super::format::{format_info, supported_formats, FormatInfo};
use super::layout::{device_layout, validate_blob_shape, DeviceLayout, Planes, WeightError};
use crate::backend::QuantType;
use std::fmt;

/// Function constant index (`q_fc_nb`): tokens per threadgroup column of the matrix-vector kernels.
pub const FC_NB: u32 = 500;

/// Bit 63 of `QParams::plane1` for MLX affine weights whose scales and biases are bfloat16 (`Q_MLX_BF16` in common.metal).
pub const MLX_BF16_FLAG: u64 = 1 << 63;

/// Simdgroups per threadgroup of the matrix-vector kernels (`Q_MV_NSG`).
const MV_NSG: u32 = 2;
/// Maximal batch of the matrix-vector kernels.
const MV_MAX_BATCH: u32 = 8;
/// Weight rows / tokens per threadgroup of the matrix-matrix kernels (`Q_BM`, `Q_BN`) and its thread count.
const MM_ROWS: u32 = 64;
const MM_TOKENS: u32 = 32;
const MM_THREADS: u32 = 128;
const MM_BK: u32 = 32;
/// Threads of the permutation prepass per threadgroup.
const PREPASS_THREADS: u32 = 256;

/// What the planner needs to know about the device (a subset of `MetalDeviceInfo` / `Pipeline`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalCapsLike {
    /// `simdgroup_matrix` (Apple7+ / Mac2): enables the matrix-matrix kernels.
    pub simdgroup_matrix: bool,
    /// Native `bfloat`; unused so far (BF16 weights are converted by bit operations), reserved for bfloat tiles.
    pub bfloat: bool,
    /// Threads per simdgroup (32 on Apple GPUs). The matrix-matrix kernels need exactly 32.
    pub simd_width: u32,
    /// `maxTotalThreadsPerThreadgroup`.
    pub max_threads: u32,
    /// Unified memory (always true on Apple silicon); informational, the kernels read the weights in place.
    pub unified_memory: bool,
    /// Accuracy first: matrix-matrix kernels use float tiles (activations are not rounded to half). Slower, about 2x more
    /// threadgroup memory. Also selected automatically for F32 weights.
    pub exact: bool,
}

impl MetalCapsLike {
    pub fn new(
        simdgroup_matrix: bool,
        bfloat: bool,
        simd_width: u32,
        max_threads: u32,
        unified_memory: bool,
    ) -> Self {
        Self {
            simdgroup_matrix,
            bfloat,
            simd_width,
            max_threads,
            unified_memory,
            exact: false,
        }
    }
}

impl Default for MetalCapsLike {
    /// A typical Apple-silicon GPU.
    fn default() -> Self {
        Self::new(true, true, 32, 1024, true)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearOp {
    /// `fllm_q_mv_*`: simd_sum reduction, batch 1..8 per dispatch column.
    MatVec,
    /// `fllm_q_mm_*`: simdgroup_matrix tiles.
    MatMul,
}

/// One function constant of the pipeline (`FunctionConstant::u32(index, value)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanConstant {
    pub index: u32,
    pub value: u32,
}

/// Buffer binding slots of the main kernel. The permutation prepass has its own slots ([`PrepassPlan::slots`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferSlots {
    pub weights: u32,
    pub x: u32,
    pub out: u32,
    pub bias: u32,
    pub params: u32,
    /// `[[threadgroup(slot)]]` of the matrix-matrix kernels.
    pub threadgroup_memory: u32,
}

const SLOTS: BufferSlots = BufferSlots {
    weights: 0,
    x: 1,
    out: 2,
    bias: 3,
    params: 4,
    threadgroup_memory: 0,
};

/// Parameter block of the kernels (`q_params` in common.metal): 48 bytes, no padding. Bind with `DispatchBuilder::bytes`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QParams {
    pub plane1: u64,
    pub rows: u32,
    pub cols: u32,
    pub batch: u32,
    pub flags: u32,
    pub scale: f32,
    pub out_stride: u32,
    pub row_base: u32,
    pub x_stride: u32,
    pub nb_blocks: u32,
    pub pad: u32,
}

/// `QParams::flags` bits.
pub const FLAG_BIAS: u32 = 1;
pub const FLAG_RESIDUAL: u32 = 2;

impl QParams {
    pub const SIZE: usize = 48;

    /// The 48 bytes in the layout of the MSL struct (little endian).
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut b = [0u8; Self::SIZE];
        b[0..8].copy_from_slice(&self.plane1.to_le_bytes());
        let words = [
            self.rows,
            self.cols,
            self.batch,
            self.flags,
            self.scale.to_bits(),
            self.out_stride,
            self.row_base,
            self.x_stride,
            self.nb_blocks,
            self.pad,
        ];
        for (i, w) in words.iter().enumerate() {
            b[8 + 4 * i..12 + 4 * i].copy_from_slice(&w.to_le_bytes());
        }
        b
    }
}

/// Parameter block of `fllm_q_permute_x` (`q_permute_params`, 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PermuteParams {
    pub batch: u32,
    pub cols: u32,
    pub x_stride: u32,
    pub pad: u32,
}

impl PermuteParams {
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut b = [0u8; 16];
        for (i, w) in [self.batch, self.cols, self.x_stride, self.pad]
            .iter()
            .enumerate()
        {
            b[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        b
    }
}

/// Column gather that runs before the main kernel of act-order GPTQ weights (Q4ZP*): `y[tok][c] = x[tok][perm[c]]`.
/// The main kernel then reads `y` (dense, `batch * cols` floats) instead of `x`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrepassPlan {
    pub kernel_name: &'static str,
    pub threadgroups: [u32; 3],
    pub threads_per_group: [u32; 3],
    /// 0 x (read), 1 y (write), 2 perm (read, `DeviceWeight::perm`), 3 parameter block.
    pub slots: [u32; 4],
    pub params: PermuteParams,
    /// Size of `y` in bytes.
    pub output_bytes: usize,
}

/// Everything needed to encode one `quantized_linear`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearPlan {
    pub quant: QuantType,
    pub op: LinearOp,
    /// `fllm_q_mv_<tag>_r4`, `_r2`, `fllm_q_mm_<tag>_h` or `_f`.
    pub kernel_name: String,
    /// Function constants of the pipeline (empty for the matrix-matrix kernels).
    pub constants: Vec<PlanConstant>,
    pub threadgroups: [u32; 3],
    pub threads_per_group: [u32; 3],
    /// `(slot, bytes)` of dynamic threadgroup memory.
    pub threadgroup_memory: Vec<(u32, u32)>,
    pub slots: BufferSlots,
    /// Parameter block with defaults: scale 1, no bias, no residual, rows start at 0, `out_stride = rows`.
    pub params: QParams,
    /// Act-order weights: run this before the main kernel and bind its output as `x`.
    pub activation_prepass: Option<PrepassPlan>,
    /// The kernel arithmetic: half tiles round weights and activations to half precision (`false` for the `_f` variant and
    /// all matrix-vector kernels).
    pub half_tiles: bool,
}

impl LinearPlan {
    /// Adds the bias (`bias[row_base + r]`); the bias buffer must be bound in any case (use any small buffer otherwise).
    pub fn with_bias(mut self) -> Self {
        self.params.flags |= FLAG_BIAS;
        self
    }

    /// `out += result` instead of `out = result` (fused residual add). The output must then be bound read-write.
    pub fn with_residual(mut self) -> Self {
        self.params.flags |= FLAG_RESIDUAL;
        self
    }

    /// Per-tensor factor on top of the format's own scales (`DeviceWeight::scale`).
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.params.scale = scale;
        self
    }

    /// Writes the results of this weight to `out[tok * out_stride + row_base + r]` (several weights into one buffer).
    pub fn with_output(mut self, row_base: u32, out_stride: u32) -> Self {
        self.params.row_base = row_base;
        self.params.out_stride = out_stride;
        self
    }

    /// Activation rows `stride` floats apart instead of `cols` (a multiple of 8; the rows start 16-byte aligned). For act-order
    /// weights only the prepass reads `x`, so the stride goes there.
    pub fn with_input_stride(mut self, stride: u32) -> Self {
        debug_assert!(stride.is_multiple_of(8) && stride >= self.params.cols);
        match &mut self.activation_prepass {
            Some(pre) => pre.params.x_stride = stride,
            None => self.params.x_stride = stride,
        }
        self
    }

    /// Whether the kernel reads the previous contents of `out`.
    pub fn reads_output(&self) -> bool {
        self.params.flags & FLAG_RESIDUAL != 0
    }

    /// `(threadgroups, threads_per_group)` as the `usize` arrays of `DispatchBuilder::threadgroups`.
    pub fn dispatch_dims(&self) -> ([usize; 3], [usize; 3]) {
        (
            self.threadgroups.map(|v| v as usize),
            self.threads_per_group.map(|v| v as usize),
        )
    }
}

impl PrepassPlan {
    /// `(threadgroups, threads_per_group)` as the `usize` arrays of `DispatchBuilder::threadgroups`.
    pub fn dispatch_dims(&self) -> ([usize; 3], [usize; 3]) {
        (
            self.threadgroups.map(|v| v as usize),
            self.threads_per_group.map(|v| v as usize),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    Weight(WeightError),
    /// `batch == 0` or a size that does not fit 32 bits.
    BadSize(String),
    /// The device cannot run the kernel (`simd_width`, `max_threads`).
    Device(String),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanError::Weight(e) => write!(f, "{e}"),
            PlanError::BadSize(s) | PlanError::Device(s) => write!(f, "metal: {s}"),
        }
    }
}

impl std::error::Error for PlanError {}

impl From<WeightError> for PlanError {
    fn from(e: WeightError) -> Self {
        PlanError::Weight(e)
    }
}

/// `q_params::pad` of a block-wise FP8 weight: log2 of the tile rows, log2 of the tile columns << 8 (0 for every other format).
fn fp8b_pad(info: &FormatInfo) -> u32 {
    if !info.tiled {
        return 0;
    }
    let (rows, cols) = fllm_gguf_probe::fp8b_shape(info.quant.to_gguf_type_id()).expect("an FP8B type id");
    rows.trailing_zeros() | (cols.trailing_zeros() << 8)
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn mm_shared_bytes(half: bool) -> u32 {
    let element = if half { 2 } else { 4 };
    (MM_ROWS + MM_TOKENS) * MM_BK * element + MM_ROWS * MM_TOKENS * 4
}

/// Plan of `out[b][r] = scale * dot(x[b], w[r]) (+ bias[r])` for `batch` activation rows and a `rows` x `cols` weight of format
/// `quant`.
///
/// * batch 1..=8: matrix-vector kernel (`r4`: 4 rows per simdgroup up to batch 4, `r2` above), one dispatch column of
///   `batch` tokens;
/// * batch > 8 and `simdgroup_matrix`, simd width 32 and >= 128 threads: matrix-matrix kernel, half tiles (float tiles with
///   `caps.exact` or for F32 weights);
/// * otherwise (no `simdgroup_matrix`): the matrix-vector kernel with `ceil(batch / 8)` dispatch columns.
pub fn plan_quantized_linear(
    quant: QuantType,
    rows: usize,
    cols: usize,
    batch: usize,
    caps: &MetalCapsLike,
) -> Result<LinearPlan, PlanError> {
    let info = format_info(quant).ok_or(WeightError::UnsupportedFormat(quant))?;
    // validates the shape (a self-describing blob has no plane geometry to ask for: its parameters sit in the header of the device image)
    let planes = if info.blob {
        validate_blob_shape(&info, rows, cols)?;
        Planes { layout: DeviceLayout::Image, plane1: 0, total_bytes: 0 }
    } else {
        device_layout(quant, rows, cols)?
    };
    let size = |what: &str, v: usize| {
        u32::try_from(v).map_err(|_| PlanError::BadSize(format!("{what} {v} does not fit 32 bits")))
    };
    if batch == 0 {
        return Err(PlanError::BadSize("batch is 0".into()));
    }
    let (rows32, cols32, batch32) = (
        size("rows", rows)?,
        size("cols", cols)?,
        size("batch", batch)?,
    );
    batch32
        .checked_mul(cols32)
        .ok_or_else(|| PlanError::BadSize("batch * cols overflows 32 bits".into()))?;
    if caps.simd_width == 0 || caps.max_threads == 0 {
        return Err(PlanError::Device("simd width or thread limit is 0".into()));
    }

    // MLX affine weights with bfloat16 scales: bit 63 of the plane-1 offset tells the decoder (`Q_MLX_BF16` in common.metal)
    let plane1 = planes.plane1 | if quant.mlx_affine().is_some_and(|m| m.bf16) { MLX_BF16_FLAG } else { 0 };
    let params = QParams {
        plane1,
        rows: rows32,
        cols: cols32,
        batch: batch32,
        flags: 0,
        scale: 1.0,
        out_stride: rows32,
        row_base: 0,
        x_stride: cols32,
        nb_blocks: (cols / info.qk) as u32,
        // block-wise FP8: log2 of the tile rows | log2 of the tile columns << 8 (`q_fmt_fp8b`)
        pad: fp8b_pad(&info),
    };
    let activation_prepass = info.permuted.then(|| PrepassPlan {
        kernel_name: "fllm_q_permute_x",
        threadgroups: [ceil_div(batch32 * cols32, PREPASS_THREADS), 1, 1],
        threads_per_group: [PREPASS_THREADS.min(caps.max_threads), 1, 1],
        slots: [0, 1, 2, 3],
        params: PermuteParams {
            batch: batch32,
            cols: cols32,
            x_stride: cols32,
            pad: 0,
        },
        output_bytes: batch * cols * 4,
    });
    // the prepass writes a dense copy, so the main kernel reads rows of `cols` floats
    let use_mm = batch32 > MV_MAX_BATCH
        && caps.simdgroup_matrix
        && caps.simd_width == 32
        && caps.max_threads >= MM_THREADS;
    let plan = if use_mm {
        let half = !(caps.exact || quant == QuantType::F32);
        LinearPlan {
            quant,
            op: LinearOp::MatMul,
            kernel_name: format!("fllm_q_mm_{}_{}", info.tag, if half { "h" } else { "f" }),
            constants: Vec::new(),
            threadgroups: [ceil_div(rows32, MM_ROWS), ceil_div(batch32, MM_TOKENS), 1],
            threads_per_group: [MM_THREADS, 1, 1],
            threadgroup_memory: vec![(SLOTS.threadgroup_memory, mm_shared_bytes(half))],
            slots: SLOTS,
            params,
            activation_prepass,
            half_tiles: half,
        }
    } else {
        let threads = MV_NSG * caps.simd_width;
        if threads > caps.max_threads {
            return Err(PlanError::Device(format!(
                "a threadgroup of {threads} threads exceeds the limit of {}",
                caps.max_threads
            )));
        }
        let nb = batch32.min(MV_MAX_BATCH);
        let (variant, nr) = if nb <= 4 { ("r4", 4) } else { ("r2", 2) };
        LinearPlan {
            quant,
            op: LinearOp::MatVec,
            kernel_name: format!("fllm_q_mv_{}_{}", info.tag, variant),
            constants: vec![PlanConstant {
                index: FC_NB,
                value: nb,
            }],
            threadgroups: [ceil_div(rows32, MV_NSG * nr), ceil_div(batch32, nb), 1],
            threads_per_group: [threads, 1, 1],
            threadgroup_memory: Vec::new(),
            slots: SLOTS,
            params,
            activation_prepass,
            half_tiles: false,
        }
    };
    Ok(plan)
}

/// Names of all kernels of the quantized linear family (for `PipelineCache::prewarm` and the consistency test).
pub fn all_kernel_names() -> Vec<String> {
    let mut names = Vec::new();
    for quant in supported_formats() {
        let Some(FormatInfo { tag, .. }) = format_info(quant) else {
            continue;
        };
        for v in ["r4", "r2"] {
            names.push(format!("fllm_q_mv_{tag}_{v}"));
        }
        for v in ["h", "f"] {
            names.push(format!("fllm_q_mm_{tag}_{v}"));
        }
    }
    names.sort();
    names.dedup();
    names.push("fllm_q_permute_x".to_owned());
    names
}
