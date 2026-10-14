//! Unified accelerator backend abstraction for fLLM.
//!
//! Designed for hardware devices with their own memory domains (CUDA, Vulkan, Metal, wgpu)
//! with non-blocking stream execution, device-resident quantized weights, paged KV caching,
//! and layer-level operator fusion.

pub mod cpu;
pub mod memory;
pub mod q4z;
pub mod selftest;
pub mod stub;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub mod metal;
#[cfg(feature = "hip")]
pub mod hip;
#[cfg(feature = "sycl")]
pub mod sycl;
#[cfg(any(feature = "npu-openvino", feature = "npu-xdna", feature = "npu-qnn", feature = "npu-ane"))]
pub mod npu;
#[cfg(feature = "tpu-pjrt")]
pub mod tpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "rust-gpu")]
pub mod wgpu_backend;

/// `std::env::var_os(name).is_some()`, looked up once per call site: the per-layer paths ask for several switches per layer and step, and every lookup
/// takes the environment lock. (The value is fixed at the first use; the switches are meant for process start.)
macro_rules! env_flag_once {
    ($name:literal) => {{
        static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FLAG.get_or_init(|| std::env::var_os($name).is_some())
    }};
}
pub(crate) use env_flag_once;

use std::any::Any;
use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, Mutex};

/// Global registry of operations that fell back to CPU, to warn only once per operation.
static FALLBACK_WARNED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

thread_local! {
    /// `(weight_scale_2, input_scale)` of the NVFP4 weight that is about to be uploaded (see `set_next_nvfp4_scales`).
    static NEXT_NVFP4_SCALES: std::cell::Cell<(f32, f32)> = const { std::cell::Cell::new((1.0, 0.0)) };
}

/// The checkpoint's static activation scale (`<name>.input_scale`, ModelOpt) of the NVFP4 weight a backend is asked to upload next on this
/// thread; 0 = the checkpoint has none. Set by the caller of `Backend::upload_weight_scaled` right before the call and cleared after it, so that
/// the trait does not grow a parameter only the CUDA backend reads.
pub fn set_next_nvfp4_input_scale(input_scale: f32) {
    NEXT_NVFP4_SCALES.with(|c| c.set((c.get().0, input_scale)));
}

/// Read by the CUDA backend inside `upload_weight_scaled`.
#[cfg(feature = "cuda")]
pub(crate) fn take_next_nvfp4_input_scale() -> f32 {
    NEXT_NVFP4_SCALES.with(|c| c.get().1)
}

/// Remembers `weight_scale_2` and `input_scale` for the device layout of the weight that `upload_weight` converts next.
#[cfg(feature = "cuda")]
pub(crate) fn set_next_nvfp4_scales(weight_scale: f32, input_scale: f32) {
    NEXT_NVFP4_SCALES.with(|c| c.set((weight_scale, input_scale)));
}

/// `(weight_scale_2, input_scale)` for the device layout, reset to `(1, 0)` (an unscaled upload) afterwards.
#[cfg(feature = "cuda")]
pub(crate) fn take_next_nvfp4_scales() -> (f32, f32) {
    NEXT_NVFP4_SCALES.with(|c| c.replace((1.0, 0.0)))
}

/// Warn the user once when an operation or quant type is unsupported by the native backend
/// and falls back to CPU execution.
pub fn warn_fallback_once(op_name: &str) {
    warn_fallback_reason(op_name, "not natively supported by the active backend");
}

/// Warn once per `op_name` that it runs on the host instead of the device, and why.
pub fn warn_fallback_reason(op_name: &str, reason: &str) {
    log_once(
        op_name,
        &format!("'{op_name}' falls back to the CPU: {reason}."),
    );
}

/// Print `message` to stderr the first time `key` is seen in this process.
pub fn log_once(key: &str, message: &str) {
    // A panic while holding the lock cannot leave the set inconsistent; keep using it.
    let mut lock = FALLBACK_WARNED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let set = lock.get_or_insert_with(HashSet::new);
    if set.insert(key.to_string()) {
        eprintln!("[fllm-backend] {message}");
    }
}

/// Errors produced by backend operations and allocations.
#[derive(Debug, Clone)]
pub enum BackendError {
    DeviceNotFound(String),
    OutOfMemory { requested: usize, available: usize },
    UnsupportedQuant(QuantType),
    UnsupportedOperation(String),
    DriverError(String),
    BufferError(String),
    InvalidShape(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound(msg) => write!(f, "Device not found: {msg}"),
            Self::OutOfMemory { requested, available } => {
                write!(
                    f,
                    "Out of memory: requested {requested} bytes, available {available} bytes"
                )
            }
            Self::UnsupportedQuant(q) => write!(f, "Unsupported quantization format: {q:?}"),
            Self::UnsupportedOperation(op) => write!(f, "Unsupported operation: {op}"),
            Self::DriverError(msg) => write!(f, "Driver error: {msg}"),
            Self::BufferError(msg) => write!(f, "Buffer error: {msg}"),
            Self::InvalidShape(msg) => write!(f, "Invalid shape: {msg}"),
        }
    }
}

impl std::error::Error for BackendError {}

impl From<String> for BackendError {
    fn from(s: String) -> Self {
        Self::DriverError(s)
    }
}

impl From<&str> for BackendError {
    fn from(s: &str) -> Self {
        Self::DriverError(s.to_string())
    }
}

/// Shape and constants of a Mamba-2 layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MambaShape {
    pub inner: usize,
    pub state: usize,
    pub heads: usize,
    pub groups: usize,
    pub conv_kernel: usize,
    pub norm_groups: usize,
    pub step_min: f32,
    pub eps: f32,
}

impl MambaShape {
    /// Channels of the convolution: `[x | B | C]`.
    pub fn conv_dim(&self) -> usize {
        self.inner + 2 * self.groups * self.state
    }

    /// Bytes of the recurrent state of one layer of one sequence (`heads x head_dim x state`, FP32).
    pub fn rec_bytes(&self) -> usize {
        self.inner * self.state * 4
    }

    /// Bytes one layer of one sequence occupies in a state buffer: the recurrent state, then the convolution history
    /// (`conv_dim x (kernel - 1)`), rounded up to 256 bytes.
    pub fn layer_state_bytes(&self) -> usize {
        (self.rec_bytes() + self.conv_dim() * (self.conv_kernel - 1) * 4).next_multiple_of(256)
    }
}

/// Shape and constants of a Mamba-1 layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mamba1Shape {
    pub inner: usize,
    pub state: usize,
    /// Rank of the time step projection.
    pub rank: usize,
    pub conv_kernel: usize,
    /// Falcon-Mamba normalises the dt / B / C parts of x_proj's output with this epsilon.
    pub bc_rms_eps: Option<f32>,
}

impl Mamba1Shape {
    /// Bytes of the recurrent state of one layer of one sequence (`inner x state`, FP32).
    pub fn rec_bytes(&self) -> usize {
        self.inner * self.state * 4
    }

    /// Bytes one layer of one sequence occupies in a state buffer: the recurrent state, then the convolution history
    /// (`inner x (kernel - 1)`), rounded up to 256 bytes.
    pub fn layer_state_bytes(&self) -> usize {
        (self.rec_bytes() + self.inner * (self.conv_kernel - 1) * 4).next_multiple_of(256)
    }
}

/// Device-resident parameters of a Mamba-2 layer (all FP32).
pub struct MambaWeights<'a> {
    /// `conv_dim x conv_kernel`
    pub conv: &'a DeviceBuffer,
    pub conv_bias: Option<&'a DeviceBuffer>,
    pub dt_bias: &'a DeviceBuffer,
    /// Negative decay rates (`-exp(A_log)`) per head.
    pub decay: &'a DeviceBuffer,
    pub skip: &'a DeviceBuffer,
    pub norm: &'a DeviceBuffer,
}

/// Dimensions of a multi-head latent attention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaShape {
    pub heads: usize,
    pub nope: usize,
    pub rope: usize,
    pub value: usize,
    pub kv_lora: usize,
}

/// Shape and constants of a Gated DeltaNet layer (Qwen3.5 / Qwen3.6 / Qwen3-Next).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeltaShape {
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
    /// Value head `j` belongs to key head `j % k_heads` (llama.cpp files) instead of `j / (v_heads / k_heads)`.
    pub tiled_heads: bool,
    pub beta_scale: f32,
    pub eps: f32,
}

impl DeltaShape {
    /// Channels of the convolution: `[q | k | v]`.
    pub fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim
    }

    pub fn value_dim(&self) -> usize {
        self.v_heads * self.v_dim
    }

    /// Bytes of the recurrent matrices of one layer of one sequence (`v_heads x k_dim x v_dim`, FP32).
    pub fn rec_bytes(&self) -> usize {
        self.v_heads * self.k_dim * self.v_dim * 4
    }

    /// Bytes one layer of one sequence occupies in a state buffer: the recurrent matrices, then the convolution history
    /// (`conv_dim x (kernel - 1)`), rounded up to 256 bytes.
    pub fn layer_state_bytes(&self) -> usize {
        (self.rec_bytes() + self.conv_dim() * (self.conv_kernel - 1) * 4).next_multiple_of(256)
    }
}

/// Consecutive rows of one sequence handed to [`Stream::delta_layer`].
pub struct DeltaRun<'a> {
    /// State buffer of the sequence (all DeltaNet layers, `layer_state_bytes` each).
    pub state: &'a DeviceBuffer,
    pub first_row: usize,
    pub rows: usize,
    /// The sequence starts with this run: whatever the state holds is ignored.
    pub reset: bool,
}

/// Device-resident parameters of a DeltaNet layer (all FP32).
pub struct DeltaWeights<'a> {
    /// `conv_dim x conv_kernel`
    pub conv: &'a DeviceBuffer,
    pub dt_bias: &'a DeviceBuffer,
    /// Negative decay rates (`-exp(A_log)`).
    pub decay: &'a DeviceBuffer,
    pub norm: &'a DeviceBuffer,
}

/// Quantization format of weights stored resident on the device.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ4_NL,
    IQ4_XS,
    Fp8E4m3,
    Fp8E5m2,
    NvFp4,
    /// OCP microscaling FP4 (GGUF type 39): 32 E2M1 values and one E8M0 scale per block.
    Mxfp4,
    /// fLLM 4-bit group format for AWQ / GPTQ weights (GGUF type 0x8000_0010 + log2(group / 32)): per group of 32, 64 or 128 values
    /// `group / 2` code bytes (value `2j` in the low nibble), an f16 scale and a zero-point byte; weight = (code - zero) * scale.
    Q4Z32,
    Q4Z64,
    Q4Z128,
    Q4ZP32,
    Q4ZP64,
    Q4ZP128,
    /// Device-only: NVFP4 weights in the Marlin W4A16 layout (CUDA, GPUs without FP4 tensor cores); never stored in a file.
    Nvfp4Marlin,
    Q4_K_Marlin,
    Fp8Marlin,
    Other(u32),
}

impl QuantType {
    pub fn from_gguf_type_id(kind: u32) -> Self {
        match kind {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            29 => Self::IQ1_M,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            30 => Self::BF16,
            0x8000_0001 => Self::Fp8E4m3,
            0x8000_0002 => Self::Fp8E5m2,
            39 => Self::Mxfp4,
            40 => Self::NvFp4,
            41 => Self::Q4_K_Marlin,
            0x8000_0010 => Self::Q4Z32,
            0x8000_0011 => Self::Q4Z64,
            0x8000_0012 => Self::Q4Z128,
            0x8000_0014 => Self::Q4ZP32,
            0x8000_0015 => Self::Q4ZP64,
            0x8000_0016 => Self::Q4ZP128,
            0x8000_0020 => Self::Fp8Marlin,
            other => Self::Other(other),
        }
    }

    pub fn to_gguf_type_id(&self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::IQ2_XXS => 16,
            Self::IQ2_XS => 17,
            Self::IQ3_XXS => 18,
            Self::IQ1_S => 19,
            Self::IQ1_M => 29,
            Self::IQ4_NL => 20,
            Self::IQ3_S => 21,
            Self::IQ2_S => 22,
            Self::IQ4_XS => 23,
            Self::BF16 => 30,
            Self::Fp8E4m3 => 0x8000_0001,
            Self::Fp8E5m2 => 0x8000_0002,
            Self::Mxfp4 => 39,
            Self::NvFp4 => 40,
            Self::Q4_K_Marlin => 41,
            Self::Q4Z32 => 0x8000_0010,
            Self::Q4Z64 => 0x8000_0011,
            Self::Q4Z128 => 0x8000_0012,
            Self::Q4ZP32 => 0x8000_0014,
            Self::Q4ZP64 => 0x8000_0015,
            Self::Q4ZP128 => 0x8000_0016,
            Self::Nvfp4Marlin => 0x8000_0013,
            Self::Fp8Marlin => 0x8000_0020,
            Self::Other(other) => *other,
        }
    }

    /// Group size of the 4-bit AWQ / GPTQ formats (`Q4Z*`).
    pub fn q4z_group(&self) -> Option<usize> {
        match self {
            Self::Q4Z32 | Self::Q4ZP32 => Some(32),
            Self::Q4Z64 | Self::Q4ZP64 => Some(64),
            Self::Q4Z128 | Self::Q4ZP128 => Some(128),
            Self::Nvfp4Marlin => Some(16),
            _ => None,
        }
    }

    pub fn is_q4zp(&self) -> bool {
        matches!(self, Self::Q4ZP32 | Self::Q4ZP64 | Self::Q4ZP128)
    }

    /// Block size: number of values represented by one quant block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q4_K_Marlin | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            Self::IQ1_S | Self::IQ1_M | Self::IQ2_XXS | Self::IQ2_XS | Self::IQ2_S | Self::IQ3_XXS | Self::IQ3_S => {
                256
            }
            Self::IQ4_NL => 32,
            Self::IQ4_XS => 256,
            Self::Fp8E4m3 | Self::Fp8E5m2 | Self::Fp8Marlin => 1,
            Self::Mxfp4 => 32,
            Self::NvFp4 => 64,
            Self::Q4Z32 | Self::Q4ZP32 => 32,
            Self::Q4Z64 | Self::Q4ZP64 => 64,
            Self::Q4Z128 | Self::Q4ZP128 => 128,
            Self::Nvfp4Marlin => 16,
            Self::Other(_) => 32,
        }
    }

    /// Bytes consumed by one quant block.
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K | Self::Q4_K_Marlin => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ2_S => 82,
            Self::IQ3_XXS => 98,
            Self::IQ1_S => 50,
            Self::IQ1_M => 56,
            Self::IQ4_NL => 18,
            Self::IQ3_S => 110,
            Self::IQ4_XS => 136,
            Self::Fp8E4m3 | Self::Fp8E5m2 | Self::Fp8Marlin => 1,
            Self::Mxfp4 => 17,
            Self::NvFp4 => 36,
            Self::Q4Z32 | Self::Q4ZP32 => 19,
            Self::Q4Z64 | Self::Q4ZP64 => 35,
            Self::Q4Z128 | Self::Q4ZP128 => 67,
            _ => 32,
        }
    }
}

/// Opaque handle to a buffer in device memory.
///
/// `offset..offset + size` is the visible window; `capacity` is the size of the whole
/// allocation, so a pooled sub-slice can be widened again when it is recycled.
#[derive(Clone)]
pub struct DeviceBuffer {
    pub id: u64,
    pub offset: usize,
    pub size: usize,
    pub capacity: usize,
    pub device_id: usize,
    pub handle: Arc<dyn Any + Send + Sync>,
}

impl DeviceBuffer {
    pub fn new(
        id: u64,
        size: usize,
        device_id: usize,
        handle: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        Self {
            id,
            offset: 0,
            size,
            capacity: size,
            device_id,
            handle,
        }
    }

    /// Create a sub-slice of this buffer without performing an allocation.
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self, BackendError> {
        if offset.checked_add(len).is_none_or(|end| end > self.size) {
            return Err(BackendError::BufferError(format!(
                "buffer sub-slice out of bounds: offset {offset} + len {len} > size {}",
                self.size
            )));
        }
        Ok(Self {
            id: self.id,
            offset: self.offset + offset,
            size: len,
            capacity: self.capacity,
            device_id: self.device_id,
            handle: Arc::clone(&self.handle),
        })
    }
}

/// Pre-allocated buffer pool preventing device allocations during decode steps.
///
/// After the first step of a given shape every `acquire` is served from the pool; the pool
/// itself never reallocates its `Vec` once it holds every buffer a step uses.
pub struct BufferPool {
    backend: Arc<dyn Backend>,
    pool: Vec<DeviceBuffer>,
    allocated_total: usize,
}

impl BufferPool {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self {
            backend,
            pool: Vec::new(),
            allocated_total: 0,
        }
    }

    /// Preallocate a set of scratch buffers of given sizes.
    pub fn preallocate(&mut self, sizes: &[usize]) -> Result<(), BackendError> {
        self.pool.reserve(sizes.len());
        for &sz in sizes {
            let buf = self.backend.allocate_buffer(sz)?;
            self.allocated_total += sz;
            self.pool.push(buf);
        }
        Ok(())
    }

    /// Acquire a buffer of at least `size` bytes from the pool: the smallest free buffer that
    /// fits, so a small request does not take the buffer a later large request needs.
    pub fn acquire(&mut self, size: usize) -> Result<DeviceBuffer, BackendError> {
        let best = self
            .pool
            .iter()
            .enumerate()
            .filter(|(_, b)| b.capacity >= size)
            .min_by_key(|(_, b)| b.capacity)
            .map(|(pos, _)| pos);
        if let Some(pos) = best {
            let buf = self.pool.swap_remove(pos);
            buf.slice(0, size)
        } else {
            let buf = self.backend.allocate_buffer(size)?;
            self.allocated_total += size;
            Ok(buf)
        }
    }

    /// Return a buffer to the pool for reuse, widened to its whole allocation.
    pub fn recycle(&mut self, buffer: DeviceBuffer) {
        self.pool.push(DeviceBuffer {
            offset: 0,
            size: buffer.capacity,
            ..buffer
        });
    }

    pub fn total_allocated_bytes(&self) -> usize {
        self.allocated_total
    }
}

/// Resident model weight stored on the device in native quantized representation.
#[derive(Clone)]
pub struct DeviceWeight {
    pub name: String,
    pub buffer: DeviceBuffer,
    pub quant_type: QuantType,
    pub rows: usize,
    pub cols: usize,
    pub bias: Option<DeviceBuffer>,
    /// Per-tensor factor applied to every output of this weight (NVFP4 checkpoints); 1.0 otherwise.
    pub scale: f32,
    pub perm: Option<DeviceBuffer>,
}

/// Device-resident Paged KV Cache block pool.
///
/// Layout, shared with the CPU decoder (`rust_decoder.rs`, `write_cache`/`read_slot`), per
/// pool and per layer `l` starting at 16-bit word `l * words_per_layer_*`, with
/// `slots = pool_blocks * block_size` and `kv = kv_heads * dim`:
/// - f16 (`kv_kind == 1`): `slots * kv` IEEE half words; value `(slot, head, d)` is word
///   `slot * kv + head * dim + d`.
/// - q8 (`kv_kind == 2`): `(slots * kv).div_ceil(2)` words holding one signed byte per value
///   (little endian, byte `slot * kv + c`), followed by `slots * (kv / group)` f16 scales,
///   scale `slot * (kv / group) + c / group`, with `group = kv_quant_group(dim)`. A value is
///   `byte as i8 * scale`.
///
/// A layer may start at an odd word, so device kernels address the pools in 16-bit units.
#[derive(Clone)]
pub struct DeviceKvCache {
    pub key_pool: DeviceBuffer,
    pub value_pool: DeviceBuffer,
    /// Packed key/value formats (`crate::kv_format::pack_kind`); 1 = f16, 2 = q8_0 for both.
    pub kv_kind: i32,
    pub layers: usize,
    pub pool_blocks: usize,
    pub block_size: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub words_per_layer_k: usize,
    pub words_per_layer_v: usize,
}

/// Values sharing one f16 scale in a q8 KV cache: 32 when `dim` is a multiple of 32,
/// otherwise the whole head.
pub fn kv_quant_group(dim: usize) -> usize {
    if dim.is_multiple_of(32) {
        32
    } else {
        dim
    }
}

/// 16-bit words of one layer of a KV pool holding `kv_heads x dim` values per slot.
pub fn kv_layer_words(kv_kind: i32, slots: usize, kv_heads: usize, dim: usize) -> usize {
    let format = crate::kv_format::KvFormat::from_code(kv_kind).unwrap_or(crate::kv_format::KvFormat::F16);
    format.layer_words(slots, kv_heads, dim)
}

impl DeviceKvCache {
    /// (key, value) formats of the pools.
    pub fn formats(&self) -> (crate::kv_format::KvFormat, crate::kv_format::KvFormat) {
        crate::kv_format::unpack_kind(self.kv_kind)
            .unwrap_or((crate::kv_format::KvFormat::F16, crate::kv_format::KvFormat::F16))
    }

    /// Region layout of the key pool (`dim = head_dim`) and the value pool (`dim = value_dim`).
    pub fn layouts(&self) -> (crate::kv_format::KvLayout, crate::kv_format::KvLayout) {
        let (k, v) = self.formats();
        (
            crate::kv_format::KvLayout::new(k, self.slots(), self.kv_heads, self.head_dim),
            crate::kv_format::KvLayout::new(v, self.slots(), self.kv_heads, self.value_dim),
        )
    }

    pub fn slots(&self) -> usize {
        self.pool_blocks * self.block_size
    }

    /// Validate the geometry and compute the per-layer sizes; pools are allocated by the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn geometry(
        layers: usize,
        pool_blocks: usize,
        block_size: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        kv_kind: i32,
    ) -> Result<(usize, usize), BackendError> {
        let Some((key_format, value_format)) = crate::kv_format::unpack_kind(kv_kind) else {
            return Err(BackendError::UnsupportedOperation(format!("unknown KV cache kind {kv_kind}")));
        };
        key_format.check_dim(head_dim).map_err(BackendError::UnsupportedOperation)?;
        value_format.check_dim(value_dim).map_err(BackendError::UnsupportedOperation)?;
        if layers == 0 || pool_blocks == 0 || block_size == 0 || kv_heads == 0 {
            return Err(BackendError::InvalidShape("empty KV cache".into()));
        }
        if head_dim == 0 || value_dim == 0 {
            return Err(BackendError::InvalidShape("zero KV head width".into()));
        }
        let slots = pool_blocks
            .checked_mul(block_size)
            .ok_or_else(|| BackendError::InvalidShape("KV slot count overflows".into()))?;
        Ok((
            key_format.layer_words(slots, kv_heads, head_dim),
            value_format.layer_words(slots, kv_heads, value_dim),
        ))
    }
}

/// RoPE configuration passed to rotary kernels.
///
/// `cos`/`sin` are `positions x rope_dim / 2` tables. They are shared (`Arc`) so a decode step
/// does not copy them; device backends upload each table once and recognise it by pointer.
#[derive(Debug, Clone)]
pub struct RopeParams {
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub rope_dim: usize,
    pub neox: bool,
    pub cos: Arc<Vec<f32>>,
    pub sin: Arc<Vec<f32>>,
}

impl RopeParams {
    /// Check table coverage and buffer sizes (`q_bytes`/`k_bytes`) for `positions`.
    pub fn validate(
        &self,
        positions: &[usize],
        q_bytes: usize,
        k_bytes: usize,
    ) -> Result<(), BackendError> {
        let half = self.rope_dim / 2;
        let shape = |msg: String| Err(BackendError::InvalidShape(msg));
        if !self.rope_dim.is_multiple_of(2) || self.rope_dim > self.dim || half == 0 {
            return shape(format!(
                "rope width {} invalid for head width {}",
                self.rope_dim, self.dim
            ));
        }
        if self.cos.len() != self.sin.len() || !self.cos.len().is_multiple_of(half) {
            return shape("rope cos/sin tables differ or are not rows of rope_dim/2".into());
        }
        let max_position = self.cos.len() / half;
        if let Some(&p) = positions.iter().find(|&&p| p >= max_position) {
            return shape(format!("rope position {p} >= table length {max_position}"));
        }
        let rows = positions.len();
        if q_bytes < rows * self.heads * self.dim * 4 || k_bytes < rows * self.kv_heads * self.dim * 4
        {
            return shape("rope q/k buffers smaller than rows x heads x dim".into());
        }
        Ok(())
    }
}

/// Check a `kv_cache_write` batch: `slots.len()` rows of `kv_heads x dim` keys and values.
pub fn validate_kv_write(
    kv: &DeviceKvCache,
    layer_idx: usize,
    slots: &[usize],
    k_bytes: usize,
    v_bytes: usize,
) -> Result<(), BackendError> {
    let shape = |msg: String| Err(BackendError::InvalidShape(msg));
    if layer_idx >= kv.layers {
        return shape(format!("layer {layer_idx} >= {} cached layers", kv.layers));
    }
    if let Some(&s) = slots.iter().find(|&&s| s >= kv.slots()) {
        return shape(format!("KV slot {s} >= {} pool slots", kv.slots()));
    }
    let rows = slots.len();
    if k_bytes < rows * kv.kv_heads * kv.head_dim * 4
        || v_bytes < rows * kv.kv_heads * kv.value_dim * 4
    {
        return shape("kv_cache_write sources smaller than rows x kv_heads x dim".into());
    }
    Ok(())
}

/// Paged Attention parameters.
///
/// Row `r` of a batch attends to positions `start..lengths[r]` of its block table, where
/// `start` is `(n - 1) / chunk * chunk` with `chunk`, otherwise `n - min(n, window)` with
/// `window`, otherwise 0. A score is `dot(q, k) * scale`, plus `alibi_slopes[h] * (p - (n-1))`,
/// then soft-capped as `cap * tanh(s / cap)`. A sink adds `exp(sinks[h])` to the softmax
/// denominator without a value.
#[derive(Debug, Clone)]
pub struct AttentionParams {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub stride: usize,
    pub block_size: usize,
    pub scale: f32,
    pub softcap: Option<f32>,
    pub window: Option<usize>,
    pub chunk: Option<usize>,
    pub sinks: Option<Arc<Vec<f32>>>,
    pub alibi_slopes: Option<Arc<Vec<f32>>>,
}

impl AttentionParams {
    /// Check the batch against the cache geometry: shapes, positive lengths and every block a
    /// row reads lies inside the pool. Devices read the cache unchecked, so this runs first.
    pub fn validate(
        &self,
        kv: &DeviceKvCache,
        block_tables: &[i64],
        lengths: &[i64],
        layer_idx: usize,
    ) -> Result<(), BackendError> {
        self.validate_blocks(kv, block_tables, lengths, layer_idx, true)
    }

    /// [`validate`](Self::validate), with the walk over every row's blocks (cost: rows x blocks per call) only when `check_blocks`. The blocks
    /// of one forward pass are the same for all its layers, so a stream that knows it has checked them once may skip them.
    pub fn validate_blocks(
        &self,
        kv: &DeviceKvCache,
        block_tables: &[i64],
        lengths: &[i64],
        layer_idx: usize,
        check_blocks: bool,
    ) -> Result<(), BackendError> {
        let shape = |msg: String| Err(BackendError::InvalidShape(msg));
        if self.num_kv_heads == 0
            || self.num_heads == 0
            || !self.num_heads.is_multiple_of(self.num_kv_heads)
        {
            return shape(format!(
                "attention heads {} not a multiple of kv heads {}",
                self.num_heads, self.num_kv_heads
            ));
        }
        if kv.kv_heads != self.num_kv_heads
            || kv.head_dim != self.head_dim
            || kv.value_dim != self.value_dim
            || kv.block_size != self.block_size
        {
            return shape("attention parameters do not match the KV cache".into());
        }
        if layer_idx >= kv.layers {
            return shape(format!("layer {layer_idx} >= {} cached layers", kv.layers));
        }
        if self.stride == 0 || block_tables.len() != lengths.len() * self.stride {
            return shape(format!(
                "block tables hold {} entries, expected {} rows x stride {}",
                block_tables.len(),
                lengths.len(),
                self.stride
            ));
        }
        for (name, values) in [("sinks", &self.sinks), ("alibi slopes", &self.alibi_slopes)] {
            if values.as_ref().is_some_and(|v| v.len() < self.num_heads) {
                return shape(format!("{name} shorter than {} heads", self.num_heads));
            }
        }
        if self.chunk == Some(0) || self.window == Some(0) {
            return shape("zero attention chunk or window".into());
        }
        if !check_blocks {
            return Ok(());
        }
        for (r, &n) in lengths.iter().enumerate() {
            if n <= 0 || (n as usize).div_ceil(self.block_size) > self.stride {
                return shape(format!("row {r}: length {n} does not fit its block table"));
            }
            let blocks = &block_tables[r * self.stride..][..(n as usize).div_ceil(self.block_size)];
            if blocks
                .iter()
                .any(|&b| b < 0 || b as usize >= kv.pool_blocks)
            {
                return shape(format!("row {r}: KV block outside the pool"));
            }
        }
        Ok(())
    }
}

/// Logit sampling parameters for device-side sampling.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub rows: usize,
    pub vocab_size: usize,
}

/// Capabilities and hardware features advertised by a backend.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub name: String,
    pub supported_quants: Vec<QuantType>,
    pub supports_rms_norm: bool,
    pub supports_layer_norm: bool,
    pub supports_rope: bool,
    pub supports_swiglu: bool,
    pub supports_geglu: bool,
    pub supports_paged_attention: bool,
    pub supports_kv_cache_write: bool,
    pub supports_embedding_lookup: bool,
    pub supports_device_sampling: bool,
    pub supports_residual_add: bool,
    pub supports_moe: bool,
    pub max_batch_size: usize,
    pub max_context: usize,
}

impl BackendCapabilities {
    pub fn supports_quant(&self, quant: QuantType) -> bool {
        self.supported_quants.contains(&quant)
    }
}

/// Asynchronous stream/command queue for enqueuing operations on a device.
pub trait Stream: Send {
    /// A forward pass of the decoder begins / ends on this stream. Between the two the host arrays of its attention calls (block tables, lengths)
    /// stay unchanged, so a backend may stage and check them once instead of once per layer. Backends without such caches ignore it.
    fn begin_step(&mut self) {}
    fn end_step(&mut self) {}

    /// Greedy sampling on the device: the index of the largest value of each of `rows` rows of `vocab` f32 logits (first index on a
    /// tie). Backends without it return `UnsupportedOperation` and the caller downloads the logits instead.
    fn argmax_rows(&mut self, _logits: &DeviceBuffer, _rows: usize, _vocab: usize) -> Result<Vec<i32>, BackendError> {
        Err(BackendError::UnsupportedOperation("argmax_rows".into()))
    }

    /// CUDA graphs of whole decode steps. `graph_begin` starts recording the operations issued on this stream into graph `id`:
    /// `sizes` are the byte sizes of the four per-step arrays (positions, cache slots, block tables, lengths) that the operations stage
    /// and that `graph_set` replaces before each `graph_launch`; `max_len` is the context length attention is sized for. Operations
    /// that synchronize or copy from pageable memory are not allowed while recording. `graph_end(.., abort)` finishes or drops the
    /// recording. `graph_launch` returns `false` when a buffer the graph points to was reallocated since (record it again).
    fn graph_supported(&self) -> bool {
        false
    }

    /// `residual += scale * (x * weight + bias)` in one pass for backends that can fold the add into the GEMM's output conversion
    /// (4-bit Marlin on CUDA). Returns `false` when it did nothing and the caller has to run the linear and the add separately.
    /// `out16` (FP16, `batch x rows` halves in a buffer sized for FP32) = `x * weight` without bias, left in half precision for a fused
    /// consumer. Returns `false` when the weight or the backend cannot do it.
    fn linear_f16_out(&mut self, _out16: &mut DeviceBuffer, _x: &DeviceBuffer, _weight: &DeviceWeight) -> Result<bool, BackendError> {
        Ok(false)
    }

    /// `out = glu(gate, up)` (SwiGLU, or GeGLU when `geglu`) of the FP16 results `gate16` and `up16` of two `linear_f16_out` calls (same
    /// geometry as `out`). Returns `false` when unsupported.
    fn glu_f16(&mut self, _out: &mut DeviceBuffer, _gate16: &DeviceBuffer, _up16: &DeviceBuffer, _geglu: bool) -> Result<bool, BackendError> {
        Ok(false)
    }

    /// Bias, rotary embedding and KV-cache write of the FP16 q/k/v of `linear_f16_out`, in one pass: `q_out` gets the rotated queries
    /// (FP32, for attention), the rotated keys and the values go into layer `layer_idx` of the FP16 cache. Returns `false` when
    /// unsupported (the caller then converts and runs `rope` / `kv_cache_write`).
    #[allow(clippy::too_many_arguments)]
    fn rope_kv_write_f16(
        &mut self,
        _q_out: &mut DeviceBuffer,
        _q16: &DeviceBuffer,
        _k16: &DeviceBuffer,
        _v16: &DeviceBuffer,
        _biases: [Option<&DeviceBuffer>; 3],
        _kv: &DeviceKvCache,
        _layer_idx: usize,
        _positions: &[usize],
        _slots: &[usize],
        _params: &RopeParams,
    ) -> Result<bool, BackendError> {
        Ok(false)
    }

    fn quantized_linear_add(
        &mut self,
        _residual: &mut DeviceBuffer,
        _x: &DeviceBuffer,
        _weight: &DeviceWeight,
        _bias: Option<&DeviceBuffer>,
        _scale: f32,
    ) -> Result<bool, BackendError> {
        Ok(false)
    }

    /// Fused residual add and RMSNorm (vLLM's `fused_add_rms_norm`): `x += scale * branch`, then `out = rms_norm(x) * weight`, in one
    /// pass over the rows of `x`. When `next` is the weight of the GEMM that reads `out` first, the activations are also left in the
    /// format that GEMM wants (FP16 for Marlin, NVFP4 or int8 blocks), so it skips its own conversion. The result is bit-identical to
    /// `residual_add` followed by `rms_norm` and the GEMM. Returns `false` (nothing done) when unsupported or switched off
    /// (`FLLM_FUSE_ADD_NORM=0`).
    #[allow(clippy::too_many_arguments)]
    fn add_rms_norm(
        &mut self,
        _x: &mut DeviceBuffer,
        _branch: &DeviceBuffer,
        _scale: f32,
        _weight: &DeviceBuffer,
        _eps: f32,
        _out: &mut DeviceBuffer,
        _next: Option<&DeviceWeight>,
    ) -> Result<bool, BackendError> {
        Ok(false)
    }

    /// `quantized_linear_add` and `add_rms_norm` in two kernels for weights whose GEMM leaves a half-precision result (4-bit Marlin): the
    /// GEMM's result, the residual update, the norm and the conversion for `next` are one kernel. `norm` must be as wide as the weight has
    /// rows. Returns `false` (nothing done) when unsupported or switched off.
    #[allow(clippy::too_many_arguments)]
    fn quantized_linear_add_norm(
        &mut self,
        _residual: &mut DeviceBuffer,
        _x: &DeviceBuffer,
        _weight: &DeviceWeight,
        _bias: Option<&DeviceBuffer>,
        _scale: f32,
        _norm: &DeviceBuffer,
        _eps: f32,
        _out: &mut DeviceBuffer,
        _next: Option<&DeviceWeight>,
    ) -> Result<bool, BackendError> {
        Ok(false)
    }

    /// Rotary embedding of `q` (in place) and the cache write of the rotated `k` and of `v`, in one kernel (vLLM runs `rotary_embedding` and
    /// `reshape_and_cache`, FlashInfer `apply_rope_pos_ids` and an append). FP32 inputs, f16 cache only; bit-identical to `rope`
    /// followed by `kv_cache_write`. Returns `false` (nothing done) when unsupported or switched off (`FLLM_FUSE_ROPE_KV=0`).
    #[allow(clippy::too_many_arguments)]
    fn rope_kv_write(
        &mut self,
        _q: &mut DeviceBuffer,
        _k: &DeviceBuffer,
        _v: &DeviceBuffer,
        _kv: &DeviceKvCache,
        _layer_idx: usize,
        _positions: &[usize],
        _slots: &[usize],
        _params: &RopeParams,
    ) -> Result<bool, BackendError> {
        Ok(false)
    }
    fn graph_begin(&mut self, _id: usize, _sizes: [usize; 4], _max_len: usize) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("graph_begin".into()))
    }
    fn graph_end(&mut self, _id: usize, _abort: bool) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("graph_end".into()))
    }
    fn graph_set(&mut self, _id: usize, _array: usize, _data: &[u8]) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("graph_set".into()))
    }
    fn graph_launch(&mut self, _id: usize) -> Result<bool, BackendError> {
        Err(BackendError::UnsupportedOperation("graph_launch".into()))
    }

    /// Wait for all operations previously enqueued on this stream to complete.
    fn synchronize(&mut self) -> Result<(), BackendError>;

    /// Asynchronously upload data from host slice to device buffer.
    fn upload(&mut self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<(), BackendError>;

    /// Download data from device buffer to host slice, synchronizing.
    fn download(&mut self, dst: &mut [u8], src: &DeviceBuffer) -> Result<(), BackendError>;

    /// Copy buffer on device.
    fn copy_buffer(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
    ) -> Result<(), BackendError>;

    /// Embedding lookup: out[i] = weight[token_ids[i]]
    fn embedding_lookup(
        &mut self,
        out: &mut DeviceBuffer,
        ids: &[i64],
        weight: &DeviceWeight,
    ) -> Result<(), BackendError>;

    /// RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
    fn rms_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        eps: f32,
    ) -> Result<(), BackendError>;

    /// LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
    fn layer_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        eps: f32,
    ) -> Result<(), BackendError>;

    /// Rotary Position Embedding on Q and K buffers in-place.
    fn rope(
        &mut self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        positions: &[usize],
        params: &RopeParams,
    ) -> Result<(), BackendError>;

    /// Quantized or float linear projection: `out[b][r] = dot(x[b], weight[r]) + bias[r]` for
    /// every batch row `b` (`x.size / (4 * cols)` rows). `bias` holds `rows` f32 values; pass
    /// `weight.bias.as_ref()` to apply the weight's own bias.
    fn quantized_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
    ) -> Result<(), BackendError>;

    /// Whether [`Stream::host_linear`] handles weights of this format.
    fn supports_host_linear(&self, _quant_type: QuantType) -> bool {
        false
    }

    /// Like [`Stream::quantized_linear`] for a weight that stays in host memory: `raw` holds the GGUF bytes of a
    /// `rows` x `cols` matrix of `quant_type`. The backend streams it to the device for this one product, which
    /// makes large batches (prompt processing) of models that do not fit in device memory fast.
    #[allow(clippy::too_many_arguments)]
    fn host_linear(
        &mut self,
        _out: &mut DeviceBuffer,
        _x: &DeviceBuffer,
        _raw: &[u8],
        quant_type: QuantType,
        _rows: usize,
        _cols: usize,
        _bias: Option<&[f32]>,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedQuant(quant_type))
    }

    /// Softmax routing of mixture-of-experts layers: for every row of `logits` (`rows` x `experts`) the `k` best experts
    /// (ties to the lower index) as `ids` (i32) with their softmax scores as `weights`, renormalised over the selected
    /// ones when `normalize` is set and multiplied by `scale`.
    #[allow(clippy::too_many_arguments)]
    fn moe_topk(
        &mut self,
        _logits: &DeviceBuffer,
        _ids: &mut DeviceBuffer,
        _weights: &mut DeviceBuffer,
        _rows: usize,
        _experts: usize,
        _k: usize,
        _normalize: bool,
        _scale: f32,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("moe_topk".into()))
    }

    /// `out[slot] = W[ids[slot]] x[slot / x_div]` for `slots` slots: `table` holds the device addresses of the experts'
    /// weights (`upload_pointer_table`), `quant_type` is the format of all of them.
    #[allow(clippy::too_many_arguments)]
    fn moe_linear(
        &mut self,
        _out: &mut DeviceBuffer,
        _x: &DeviceBuffer,
        _table: &DeviceBuffer,
        _bias_table: Option<&DeviceBuffer>,
        _ids: &DeviceBuffer,
        quant_type: QuantType,
        _rows: usize,
        _cols: usize,
        _slots: usize,
        _x_div: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedQuant(quant_type))
    }

    /// `out[t] = sum_j weights[t * k + j] * y[t * k + j]` over rows of `hidden` floats.
    fn moe_combine(
        &mut self,
        _out: &mut DeviceBuffer,
        _y: &DeviceBuffer,
        _weights: &DeviceBuffer,
        _tokens: usize,
        _k: usize,
        _hidden: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("moe_combine".into()))
    }

    /// `dst[i] = src[idx[i]] * weights[i]` for `idx.len()` rows of `row_len` floats (weights optional).
    fn gather_rows(
        &mut self,
        _dst: &mut DeviceBuffer,
        _src: &DeviceBuffer,
        _idx: &[i32],
        _weights: Option<&[f32]>,
        _row_len: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("gather_rows".into()))
    }

    /// `dst[idx[i]] += src[i] * weights[i]` (weights optional); the indices of one call are distinct.
    fn scatter_add_rows(
        &mut self,
        _dst: &mut DeviceBuffer,
        _src: &DeviceBuffer,
        _idx: &[i32],
        _weights: Option<&[f32]>,
        _row_len: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("scatter_add_rows".into()))
    }

    /// Fill the buffer with zeros.
    fn zero_buffer(&mut self, _buf: &mut DeviceBuffer) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("zero_buffer".into()))
    }

    /// gpt-oss SwiGLU: `out = min(g, limit) * sigmoid(alpha * min(g, limit)) * (clamp(u, -limit, limit) + 1)`.
    fn swiglu_oai(
        &mut self,
        _out: &mut DeviceBuffer,
        _gate: &DeviceBuffer,
        _up: &DeviceBuffer,
        _alpha: f32,
        _limit: f32,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("swiglu_oai".into()))
    }

    /// SwiGLU: out = (gate * sigmoid(gate)) * up
    fn swiglu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
    ) -> Result<(), BackendError>;

    /// GeGLU: out = (gate * gelu(gate)) * up
    fn geglu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
    ) -> Result<(), BackendError>;

    /// Gated attention: splits the `rows x heads x 2*dk` query projection into queries and gates (`per_head`: [q_h, gate_h] side by
    /// side for every head, otherwise all queries, then all gates).
    #[allow(clippy::too_many_arguments)]
    fn split_q_gate(
        &mut self,
        _q: &mut DeviceBuffer,
        _gate: &mut DeviceBuffer,
        _projected: &DeviceBuffer,
        _rows: usize,
        _heads: usize,
        _dk: usize,
        _per_head: bool,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("split_q_gate".into()))
    }

    /// In-place x = x * sigmoid(gate) (gated attention output).
    fn sigmoid_gate_mul(&mut self, _x: &mut DeviceBuffer, _gate: &DeviceBuffer) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("sigmoid_gate_mul".into()))
    }

    /// Does this stream implement `split_q_gate` and `sigmoid_gate_mul`?
    fn supports_attention_gate(&self) -> bool {
        false
    }

    /// Does this stream implement [`Stream::activation`], [`Stream::activation_glu`] and [`Stream::clamp`]?
    fn supports_arch_ops(&self) -> bool {
        false
    }

    /// `out = act(x)` element-wise. `mode`: 0 SiLU, 1 GELU (tanh), 2 GELU (erf), 3 ReLU, 4 ReLU squared.
    fn activation(&mut self, _out: &mut DeviceBuffer, _x: &DeviceBuffer, _mode: i32) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("activation".into()))
    }

    /// `out = act(gate) * up` element-wise (`mode` as in [`Stream::activation`]).
    fn activation_glu(
        &mut self,
        _out: &mut DeviceBuffer,
        _gate: &DeviceBuffer,
        _up: &DeviceBuffer,
        _mode: i32,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("activation_glu".into()))
    }

    /// In-place `x = x * softplus(beta * gate) / beta` (K2 Horizon attention output gate).
    fn softplus_gate_mul(&mut self, _x: &mut DeviceBuffer, _gate: &DeviceBuffer, _beta: f32) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("softplus_gate_mul".into()))
    }

    /// In-place `x = clamp(x, -limit, limit)`.
    fn clamp(&mut self, _x: &mut DeviceBuffer, _limit: f32) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("clamp".into()))
    }

    /// MLA data movement: `latent[r] = kv_a[r][..kv_lora]` for `rows` rows of `width` floats.
    fn mla_latent(
        &mut self,
        _latent: &mut DeviceBuffer,
        _kv_a: &DeviceBuffer,
        _rows: usize,
        _kv_lora: usize,
        _width: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("mla_latent".into()))
    }

    /// MLA data movement: builds the per-head queries (`[rotary | content]`), keys (`[shared rotary | content]`) and values of the
    /// expanded cache from the query projection `q_full`, the key-value down projection `kv_a` and the up projection `kv`.
    #[allow(clippy::too_many_arguments)]
    fn mla_assemble(
        &mut self,
        _q: &mut DeviceBuffer,
        _k: &mut DeviceBuffer,
        _v: &mut DeviceBuffer,
        _q_full: &DeviceBuffer,
        _kv_a: &DeviceBuffer,
        _kv: &DeviceBuffer,
        _rows: usize,
        _shape: &MlaShape,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("mla_assemble".into()))
    }

    /// Does this stream implement the Mamba-1 steps ([`Stream::mamba1_conv`], [`Stream::mamba1_prep`], [`Stream::mamba1_scan`])?
    fn supports_mamba1(&self, _shape: &Mamba1Shape) -> bool {
        false
    }

    /// Mamba-1 step 1: causal convolution + SiLU of the first half of `proj` (`rows x 2 inner`: `[x | gate]`) into `mixed`
    /// (`rows x inner`); the history lives `state_offset + rec_bytes` bytes into each run's state buffer.
    #[allow(clippy::too_many_arguments)]
    fn mamba1_conv(
        &mut self,
        _shape: &Mamba1Shape,
        _conv: &DeviceBuffer,
        _conv_bias: Option<&DeviceBuffer>,
        _runs: &[DeltaRun<'_>],
        _state_offset: usize,
        _proj: &DeviceBuffer,
        _mixed: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("mamba1_conv".into()))
    }

    /// Mamba-1 step 2: on the output `split` (`rows x (rank + 2 state)`) of x_proj, normalise the dt / B / C parts (RMS, `shape.bc_rms_eps`)
    /// and copy the dt part into `dt_in` (`rows x rank`), the input of dt_proj.
    fn mamba1_prep(&mut self, _shape: &Mamba1Shape, _split: &mut DeviceBuffer, _dt_in: &mut DeviceBuffer, _rows: usize) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("mamba1_prep".into()))
    }

    /// Mamba-1 step 3: the selective scan; `dt` is the output of dt_proj (`rows x inner`), `out` receives the gated scan output
    /// (`rows x inner`). The recurrent state (`inner x state` floats) starts `state_offset` bytes into each run's state buffer.
    #[allow(clippy::too_many_arguments)]
    fn mamba1_scan(
        &mut self,
        _shape: &Mamba1Shape,
        _decay: &DeviceBuffer,
        _skip: &DeviceBuffer,
        _runs: &[DeltaRun<'_>],
        _state_offset: usize,
        _split: &DeviceBuffer,
        _dt: &DeviceBuffer,
        _mixed: &DeviceBuffer,
        _proj: &DeviceBuffer,
        _out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("mamba1_scan".into()))
    }

    /// Does this stream implement [`Stream::shortconv_layer`] for this width and kernel size?
    fn supports_shortconv(&self, _hidden: usize, _kernel: usize) -> bool {
        false
    }

    /// LFM2 gated short convolution of one layer for `runs`. `projected` (`rows x 3 hidden`: `[b | c | x]`) is the input projection of
    /// the normed input; `out` (`rows x hidden`) receives `c * conv(b * x)` (the caller applies the output projection). The layer's
    /// history (`hidden x (kernel - 1)` floats) starts `state_offset` bytes into each run's state buffer and is updated in place.
    #[allow(clippy::too_many_arguments)]
    fn shortconv_layer(
        &mut self,
        _hidden: usize,
        _kernel: usize,
        _conv: &DeviceBuffer,
        _bias: Option<&DeviceBuffer>,
        _runs: &[DeltaRun<'_>],
        _state_offset: usize,
        _projected: &DeviceBuffer,
        _out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("shortconv_layer".into()))
    }

    /// Does this stream implement [`Stream::mamba2_layer`] for layers of this shape?
    fn supports_mamba2(&self, _shape: &MambaShape) -> bool {
        false
    }

    /// Mamba-2 mixer of one layer for `runs` (each: consecutive tokens of one sequence). `projected` (`rows x (inner + conv_dim + heads)`:
    /// gate, convolution input, time steps) is the input projection of the normed input; `out` (`rows x inner`) receives the gated,
    /// normalised scan output (the caller applies the output projection). `mixed` is scratch of `rows x conv_dim` floats. The layer's
    /// state starts `state_offset` bytes into each run's state buffer and is updated in place.
    #[allow(clippy::too_many_arguments)]
    fn mamba2_layer(
        &mut self,
        _shape: &MambaShape,
        _weights: &MambaWeights<'_>,
        _runs: &[DeltaRun<'_>],
        _state_offset: usize,
        _projected: &DeviceBuffer,
        _mixed: &mut DeviceBuffer,
        _out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("mamba2_layer".into()))
    }

    /// Does this stream implement [`Stream::delta_layer`] for layers of this shape?
    fn supports_delta(&self, _shape: &DeltaShape) -> bool {
        false
    }

    /// Gated DeltaNet mixer of one layer for `runs` (each: consecutive tokens of one sequence). `qkv` (`rows x conv_dim`), `z`
    /// (`rows x value_dim`), `beta` and `alpha` (`rows x v_heads`) are the projections of the normed input; `out`
    /// (`rows x value_dim`) receives the gated, normalised head outputs (the caller applies the output projection). `mixed` is
    /// scratch of `rows x conv_dim` floats. The layer's state starts `state_offset` bytes into each run's state buffer and is
    /// updated in place.
    #[allow(clippy::too_many_arguments)]
    fn delta_layer(
        &mut self,
        _shape: &DeltaShape,
        _weights: &DeltaWeights<'_>,
        _runs: &[DeltaRun<'_>],
        _state_offset: usize,
        _qkv: &DeviceBuffer,
        _z: &DeviceBuffer,
        _beta: &DeviceBuffer,
        _alpha: &DeviceBuffer,
        _mixed: &mut DeviceBuffer,
        _out: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedOperation("delta_layer".into()))
    }

    /// In-place residual addition: x = x + residual * scale
    fn residual_add(
        &mut self,
        x: &mut DeviceBuffer,
        residual: &DeviceBuffer,
        scale: f32,
    ) -> Result<(), BackendError>;

    /// Paged Attention across multiple ragged sequences in a continuous batch.
    ///
    /// Row `r` reads the keys and values of positions `..lengths[r]` through block table row
    /// `r` of `block_tables`; the caller has already stored the current position with
    /// [`Stream::kv_cache_write`]. Prefill passes one row per token with increasing lengths.
    #[allow(clippy::too_many_arguments)]
    fn attention_paged(
        &mut self,
        out: &mut DeviceBuffer,
        q: &DeviceBuffer,
        kv: &DeviceKvCache,
        block_tables: &[i64],
        lengths: &[i64],
        layer_idx: usize,
        params: &AttentionParams,
    ) -> Result<(), BackendError>;

    /// Store row `r` of `k` (`kv_heads x head_dim` f32) and `v` (`kv_heads x value_dim` f32)
    /// into cache slot `slots[r]` of layer `layer_idx`, encoded as the cache's `kv_kind`.
    /// Slots of one call must be distinct.
    fn kv_cache_write(
        &mut self,
        kv: &DeviceKvCache,
        layer_idx: usize,
        slots: &[usize],
        k: &DeviceBuffer,
        v: &DeviceBuffer,
    ) -> Result<(), BackendError>;

    /// Device-side MoE routing computation.
    fn moe_route(
        &mut self,
        weights_out: &mut DeviceBuffer,
        indices_out: &mut DeviceBuffer,
        router_logits: &DeviceBuffer,
        num_experts: usize,
        top_k: usize,
    ) -> Result<(), BackendError>;

    /// Device-side MoE expert execution.
    fn moe_expert_gemm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        expert_weights: &[DeviceWeight],
        expert_indices: &DeviceBuffer,
        expert_weights_buf: &DeviceBuffer,
        top_k: usize,
    ) -> Result<(), BackendError>;

    /// Device-side sampling: sample directly into token IDs without host logit transfer.
    fn sampling_logits(
        &mut self,
        token_ids_out: &mut [u32],
        logits: &DeviceBuffer,
        params: &SamplingParams,
    ) -> Result<(), BackendError>;
}

/// Compute device abstraction representing a physical device or adapter.
pub trait Backend: Send + Sync {
    /// Friendly name of this backend.
    fn name(&self) -> &str;

    /// Unique index of this device.
    fn device_id(&self) -> usize;

    /// Advertise capabilities and supported formats.
    fn capabilities(&self) -> &BackendCapabilities;

    /// Create an asynchronous execution stream for this device.
    fn create_stream(&self) -> Result<Box<dyn Stream>, BackendError>;

    /// Allocate raw memory on the device.
    fn allocate_buffer(&self, size: usize) -> Result<DeviceBuffer, BackendError>;

    /// Free allocated memory.
    fn free_buffer(&self, buffer: DeviceBuffer) -> Result<(), BackendError>;

    /// Upload resident weights onto device memory in native quantized format.
    fn upload_weight(
        &self,
        name: &str,
        raw_bytes: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        bias: Option<&[f32]>,
    ) -> Result<DeviceWeight, BackendError>;

    /// Like `upload_weight` for a weight whose outputs are multiplied by `scale` (the per-tensor scale
    /// of NVFP4 checkpoints). The default folds the scale into F32 weights; backends with native
    /// support keep the quantized bytes and apply the scale in the kernel.
    #[allow(clippy::too_many_arguments)]
    fn upload_weight_scaled(
        &self,
        name: &str,
        raw_bytes: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        bias: Option<&[f32]>,
        scale: f32,
    ) -> Result<DeviceWeight, BackendError> {
        if scale == 1.0 {
            return self.upload_weight(name, raw_bytes, quant_type, rows, cols, bias);
        }
        let values = crate::compute::dequantize(raw_bytes, quant_type.to_gguf_type_id(), rows * cols)
            .map_err(|e| BackendError::UnsupportedOperation(format!("{name}: {e}")))?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| (v * scale).to_le_bytes()).collect();
        self.upload_weight(name, &bytes, QuantType::F32, rows, cols, bias)
    }

    /// A device array holding the device addresses of `weights` (the experts of one mixture-of-experts projection) for
    /// kernels that pick their weights by expert index. The weights must outlive the table.
    fn upload_pointer_table(&self, _weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
        Err(BackendError::UnsupportedOperation("upload_pointer_table".into()))
    }

    /// Like [`Backend::upload_pointer_table`] for the device bias vectors of `weights` (all must have one).
    fn upload_bias_table(&self, _weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
        Err(BackendError::UnsupportedOperation("upload_bias_table".into()))
    }

    /// Allocate resident Paged KV cache blocks on device.
    #[allow(clippy::too_many_arguments)]
    fn allocate_kv_cache(
        &self,
        layers: usize,
        pool_blocks: usize,
        block_size: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        kv_kind: i32,
    ) -> Result<DeviceKvCache, BackendError>;
}

/// Names of the native backends compiled into this binary (`--device <name>`).
pub fn native_backend_names() -> Vec<&'static str> {
    let names: &[&'static str] = &[
        #[cfg(feature = "vulkan")]
        "vulkan-native",
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        "metal-native",
        #[cfg(feature = "hip")]
        "hip",
        #[cfg(feature = "sycl")]
        "sycl",
        #[cfg(any(feature = "npu-openvino", feature = "npu-xdna", feature = "npu-qnn", feature = "npu-ane"))]
        "npu",
        #[cfg(feature = "tpu-pjrt")]
        "tpu",
    ];
    names.to_vec()
}

/// Devices of the native backends, numbered from `first_index` in the global inventory.
pub fn native_list_devices(first_index: usize) -> Vec<crate::devices::DeviceInfo> {
    native_list_devices_tagged(first_index).into_iter().map(|(_, device)| device).collect()
}

/// Like [`native_list_devices`], each device with the name of the backend that lists it (`vulkan`, `metal`, `hip`, `sycl`, `npu`, `tpu`):
/// the inventory (`devices::inventory`) numbers the devices within each backend from these tags.
pub fn native_list_devices_tagged(first_index: usize) -> Vec<(&'static str, crate::devices::DeviceInfo)> {
    #[allow(unused_mut)]
    let mut devices: Vec<(&'static str, crate::devices::DeviceInfo)> = Vec::new();
    let _ = first_index;
    #[cfg(feature = "vulkan")]
    devices.extend(vulkan::list_devices(first_index + devices.len()).into_iter().map(|d| ("vulkan", d)));
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    devices.extend(metal::list_devices(first_index + devices.len()).into_iter().map(|d| ("metal", d)));
    #[cfg(feature = "hip")]
    devices.extend(hip::list_devices(first_index + devices.len()).into_iter().map(|d| ("hip", d)));
    #[cfg(feature = "sycl")]
    devices.extend(sycl::list_devices(first_index + devices.len()).into_iter().map(|d| ("sycl", d)));
    #[cfg(any(feature = "npu-openvino", feature = "npu-xdna", feature = "npu-qnn", feature = "npu-ane"))]
    devices.extend(npu::list_devices(first_index + devices.len()).into_iter().map(|d| ("npu", d)));
    #[cfg(feature = "tpu-pjrt")]
    devices.extend(tpu::list_devices(first_index + devices.len()).into_iter().map(|d| ("tpu", d)));
    devices
}

/// The one factory for native backends (used by `Decoder::set_device`, `selftest-gpu` and the benchmarks).
///
/// `Ok(None)`: `name` is not a native backend compiled into this binary (the caller tries CUDA / wgpu). `Err`: it is, but it cannot be
/// created (the caller may fall back, e.g. to wgpu for `vulkan` and `metal`).
pub fn create_native_backend(name: &str, index: usize) -> Result<Option<Arc<dyn Backend>>, String> {
    let _ = index;
    match name {
        #[cfg(feature = "vulkan")]
        "vulkan" | "vulkan-native" => vulkan::create(index).map(Some),
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        "metal" | "metal-native" => metal::create(index).map(Some),
        #[cfg(feature = "hip")]
        "hip" | "rocm" => hip::create(index).map(Some),
        #[cfg(feature = "sycl")]
        "sycl" | "xpu" => sycl::create(index).map(Some),
        #[cfg(any(feature = "npu-openvino", feature = "npu-xdna", feature = "npu-qnn", feature = "npu-ane"))]
        "npu" | "openvino" => npu::create(index).map(Some),
        #[cfg(feature = "tpu-pjrt")]
        "tpu" => tpu::create(index).map(Some),
        _ => Ok(None),
    }
}
