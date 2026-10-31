//! State shared by [`VulkanBackend`](super::VulkanBackend) and all of its streams: the device context, the pipeline registry, a blocking
//! transfer queue for uploads, the capabilities the planners see, and the switches for the fast shader variants.

use crate::backend::vulkan::commands::{Recorder, Transfer};
use crate::backend::vulkan::instance::{ComponentType, CoopScope, VkCaps, VkContext};
use crate::backend::vulkan::memory::{Buffer, BufferUsage};
use crate::backend::vulkan::pipeline::{ComputePipeline, PipelineSpec, SpecConstant as PipeConst};
use crate::backend::vulkan::shaders;
use crate::backend::vulkan::weights::{exact_math_from_env, Variant, VkCapsLike};
use crate::backend::{BackendError, DeviceBuffer};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::mem::{handle, VkMem};

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `name` is set to something other than `0` in the environment.
pub(crate) fn env_on(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

/// `FLLM_VULKAN_NO_EXT_FORMATS=1`: block-wise FP8, bitsandbytes, MLX affine, EXL2 and EXL3 weights are not read natively but expanded to F16 at upload, as
/// before the kernels existed (an escape hatch for a driver that miscompiles one of them). Read once per process.
pub fn ext_formats_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| env_on("FLLM_VULKAN_NO_EXT_FORMATS"))
}

/// `FLLM_VULKAN_VERBOSE` is set: the backend says which shader variants it plans (and why a device probe found nothing, see `device_info`).
pub fn verbose() -> bool {
    static VERBOSE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VERBOSE.get_or_init(|| env_on("FLLM_VULKAN_VERBOSE"))
}

/// 0: follow `FLLM_VULKAN_EXACT`; 1: fast variants allowed; 2: exact (f32 shaders only).
static EXACT_OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Switch between the exact f32 shaders (the `selftest-gpu` tolerance is the FP32 worst case) and the fast ones (int8 dot products,
/// cooperative matrices) at run time; overrides `FLLM_VULKAN_EXACT`. Same role as `cuda::set_exact_math`.
pub fn set_exact_math(on: bool) {
    EXACT_OVERRIDE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
}

/// Whether only the exact f32 shaders are planned.
pub fn is_exact_math() -> bool {
    match EXACT_OVERRIDE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => exact_math_from_env(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PipeKey {
    name: String,
    spec: Vec<(u32, u32)>,
    bindings: u32,
    push: u32,
}

const DISABLED_INT_DOT: u32 = 1;
const DISABLED_COOP: u32 = 2;

/// Subgroup size the cooperative-matrix shaders run with and whether the pipeline has to pin it (`None`: cooperative matrices are not usable).
fn coop_subgroup(caps: &VkCaps) -> Option<(u32, bool)> {
    // AMD RDNA runs compute at 64 invocations per subgroup by default, but its matrix instructions are 32 wide: the cooperative-matrix pipelines ask for 32
    let amd32 = caps.vendor_id == 0x1002 && caps.subgroup_size_control && caps.min_subgroup_size <= 32 && caps.max_subgroup_size >= 32 && !env_on("FLLM_VK_COOP_SG64");
    let size = if amd32 { 32 } else { caps.subgroup_size };
    if !size.is_power_of_two()
        || 4 * size > caps.max_workgroup_invocations
        || 4 * size > caps.max_workgroup_size[0]
        || !caps.subgroup_ops.basic
    {
        return None;
    }
    if caps.subgroup_size_control && size >= caps.min_subgroup_size && size <= caps.max_subgroup_size {
        return Some((size, true));
    }
    (caps.min_subgroup_size == size && caps.max_subgroup_size == size).then_some((size, false))
}

/// Subgroup size (32) the cooperative-matrix attention kernel runs with, `None` when the device cannot run it: the kernel needs 16x16x16 float16
/// cooperative matrices, subgroup basic / shuffle / vote operations, a 32-wide subgroup (pinned with `VK_EXT_subgroup_size_control` where the
/// device's default is wider, as on AMD RDNA with its 64-wide default) and 40 KiB of shared memory. NVIDIA and AMD only (not measured elsewhere).
fn attn_coop_subgroup(caps: &VkCaps) -> Option<u32> {
    let vendor_ok = matches!(caps.vendor_id, 0x10de | 0x1002);
    let ops = caps.subgroup_ops.basic && caps.subgroup_ops.shuffle && caps.subgroup_ops.vote;
    let sized = if caps.subgroup_size == 32 {
        true
    } else {
        caps.subgroup_size_control && caps.min_subgroup_size <= 32 && caps.max_subgroup_size >= 32
    };
    (vendor_ok
        && ops
        && sized
        && caps.coop_matrix_supported
        && caps.fp16
        && caps.vulkan_memory_model
        && has_coop_16x16x16(caps)
        && !env_on("FLLM_VULKAN_NO_COOP")
        && caps.max_shared_memory >= 40 * 1024
        && caps.max_workgroup_invocations.min(caps.max_workgroup_size[0]) >= 128)
        .then_some(32)
}

fn has_coop_16x16x16(caps: &VkCaps) -> bool {
    caps.coop_matrix_for(ComponentType::F16, ComponentType::F16, ComponentType::F32, ComponentType::F32)
        .iter()
        .any(|s| s.m == 16 && s.n == 16 && s.k == 16 && matches!(s.scope, CoopScope::Subgroup))
}

/// The capabilities the planner of `weights::plan` is given, and the subgroup size to pin for the cooperative-matrix pipelines.
pub(crate) fn derive_caps(caps: &VkCaps) -> (VkCapsLike, Option<u32>) {
    let coop = coop_subgroup(caps);
    let coop_ok = coop.is_some()
        && caps.coop_matrix_supported
        && caps.fp16
        && caps.vulkan_memory_model
        && has_coop_16x16x16(caps)
        && !env_on("FLLM_VULKAN_NO_COOP");
    let like = VkCapsLike {
        coop_matrix: coop_ok,
        // FLLM_VULKAN_FORCE_IDOT=1 also plans the int8 dot-product shaders where the device has the feature but does not accelerate it
        // (lavapipe): only for checking the shaders with `selftest-gpu` on a machine without a GPU, slower than the f32 shaders
        int_dot: (caps.int_dot.int8_accelerated() || (caps.int_dot.enabled && env_on("FLLM_VULKAN_FORCE_IDOT"))) && !env_on("FLLM_VULKAN_NO_IDOT"),
        subgroup_size: coop.map_or(caps.subgroup_size.max(1), |c| c.0),
        fp16: caps.fp16,
        max_shared: caps.max_shared_memory,
        max_workgroup: caps.max_workgroup_invocations.min(caps.max_workgroup_size[0]),
        vendor: caps.vendor_id,
    };
    let pin = coop.and_then(|(size, pinned)| pinned.then_some(size));
    (like, pin)
}

pub(crate) struct Shared {
    pub ctx: Arc<VkContext>,
    pub device_id: usize,
    caps_like: VkCapsLike,
    coop_pin: Option<u32>,
    pipelines: Mutex<HashMap<PipeKey, Arc<ComputePipeline>>>,
    spirv: Mutex<HashMap<String, Arc<Vec<u32>>>>,
    /// 256 zeroed bytes bound where a shader has an optional operand it does not read.
    pub dummy: Buffer,
    transfer: Mutex<Transfer>,
    next_id: AtomicU64,
    disabled: AtomicU32,
    /// 0: the prompt shader variant has not been measured, 1: it has been claimed for measuring (or measured).
    prompt_tuning: AtomicU8,
    /// The one-time measurement found the int8 dot-product matrix shaders faster than the cooperative-matrix ones on this device.
    idot_over_coop: std::sync::atomic::AtomicBool,
}

impl Shared {
    pub fn new(ctx: Arc<VkContext>, device_id: usize) -> Result<Self, BackendError> {
        let (caps_like, coop_pin) = derive_caps(ctx.caps());
        if verbose() {
            eprintln!(
                "[fllm-vulkan] {}: planner capabilities {:?}, cooperative-matrix subgroup pin {:?}, exact math {}",
                ctx.caps().device_name,
                caps_like,
                coop_pin,
                is_exact_math()
            );
        }
        let dummy = ctx.allocator().alloc_device(256, BufferUsage::COMPUTE)?;
        let mut transfer = Transfer::new(&ctx)?;
        transfer.zero(&dummy, 0, 256)?;
        Ok(Self {
            ctx,
            device_id,
            caps_like,
            coop_pin,
            pipelines: Mutex::new(HashMap::new()),
            spirv: Mutex::new(HashMap::new()),
            dummy,
            transfer: Mutex::new(transfer),
            next_id: AtomicU64::new(1),
            disabled: AtomicU32::new(0),
            prompt_tuning: AtomicU8::new(0),
            idot_over_coop: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn caps(&self) -> &VkCaps {
        self.ctx.caps()
    }

    /// The device runs the cooperative-matrix attention kernel (see [`attn_coop_subgroup`]); independent of the GEMM tuning, which may
    /// prefer the integer dot product for the projections.
    pub fn attn_coop(&self) -> bool {
        self.disabled.load(Ordering::Relaxed) & DISABLED_COOP == 0 && attn_coop_subgroup(self.ctx.caps()).is_some()
    }

    /// Planner capabilities minus the variants whose pipelines the driver refused.
    pub fn caps_like(&self) -> VkCapsLike {
        let mut like = self.caps_like;
        let disabled = self.disabled.load(Ordering::Relaxed);
        if disabled & DISABLED_INT_DOT != 0 {
            like.int_dot = false;
        }
        if disabled & DISABLED_COOP != 0 || self.idot_over_coop.load(Ordering::Relaxed) {
            like.coop_matrix = false;
        }
        like
    }

    /// True for the first caller only: this device's prompt shader variant is measured once (see `VulkanStream::tune_prompt_variant`).
    pub fn claim_prompt_tuning(&self) -> bool {
        self.prompt_tuning.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }

    /// From now on the planner prefers the int8 dot-product matrix shaders to the cooperative-matrix ones.
    pub fn prefer_int_dot_over_coop(&self) {
        self.idot_over_coop.store(true, Ordering::Relaxed);
    }

    /// A zero-filled device buffer of `size` bytes (scratch of a one-time measurement).
    pub fn new_zeroed_buffer(&self, size: u64) -> Result<Buffer, BackendError> {
        let padded = size.max(16).next_multiple_of(16);
        let buffer = self.ctx.allocator().alloc_device(padded, BufferUsage::COMPUTE)?;
        self.zero_sync(&buffer, 0, padded)?;
        Ok(buffer)
    }

    /// Never plan `variant` again (its pipeline could not be created on this driver).
    pub fn disable(&self, variant: Variant) {
        match variant {
            Variant::IntDot => self.disabled.fetch_or(DISABLED_INT_DOT, Ordering::Relaxed),
            Variant::CoopMat => self.disabled.fetch_or(DISABLED_COOP, Ordering::Relaxed),
            Variant::Scalar => 0,
        };
    }

    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Alignment of every sub-allocation and weight slice.
    pub fn align(&self) -> u64 {
        super::mem::slice_align(self.caps().min_storage_buffer_offset_alignment)
    }

    /// Wrap an allocation as a [`DeviceBuffer`] of `size` visible bytes.
    pub fn wrap(&self, buffer: Buffer, size: usize) -> DeviceBuffer {
        DeviceBuffer::new(self.next_id(), size, self.device_id, handle(VkMem::plain(buffer)))
    }

    /// A new device buffer of `size` bytes (contents undefined; the allocation is padded to a multiple of 16).
    pub fn new_device_buffer(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        let padded = size.max(16).next_multiple_of(16) as u64;
        let buffer = self.ctx.allocator().alloc_device(padded, BufferUsage::COMPUTE)?;
        Ok(self.wrap(buffer, size))
    }

    /// The shader `name` as words (kept after the first use).
    fn spirv_words(&self, name: &str) -> Result<Arc<Vec<u32>>, BackendError> {
        let mut cache = lock(&self.spirv);
        if let Some(words) = cache.get(name) {
            return Ok(Arc::clone(words));
        }
        let words = shaders::spirv(name)
            .ok_or_else(|| BackendError::UnsupportedOperation(format!("the Vulkan shader '{name}' is not part of this build")))?;
        let words = Arc::new(words);
        cache.insert(name.to_owned(), Arc::clone(&words));
        Ok(words)
    }

    /// The compute pipeline of shader `name` with the given specialization constants `(id, value)`; created once, then shared.
    pub fn pipeline(&self, name: &str, spec: &[(u32, u32)], bindings: u32, push_bytes: u32) -> Result<Arc<ComputePipeline>, BackendError> {
        let key = PipeKey {
            name: name.to_owned(),
            spec: spec.to_vec(),
            bindings,
            push: push_bytes,
        };
        let mut map = lock(&self.pipelines);
        if let Some(pipeline) = map.get(&key) {
            return Ok(Arc::clone(pipeline));
        }
        let words = self.spirv_words(name)?;
        let constants: Vec<PipeConst> = spec.iter().map(|&(id, value)| PipeConst::u32(id, value)).collect();
        let mut pipeline_spec = PipelineSpec::new(name, &words, bindings, push_bytes).with_spec_constants(&constants);
        if name.starts_with("attn_prefill_cm_") {
            // pinned even where the device's default subgroup is 32: a default of 64 (AMD) needs the pin to reach 32
            if self.ctx.caps().subgroup_size_control {
                pipeline_spec = pipeline_spec.with_subgroup_size(32);
            }
        } else if name.starts_with("mm_") && (name.ends_with("_c") || name.ends_with("_cw") || name.ends_with("_cx")) {
            if let Some(size) = self.coop_pin {
                pipeline_spec = pipeline_spec.with_subgroup_size(size);
            }
        }
        let pipeline = self.ctx.pipelines().get_or_create(&pipeline_spec)?;
        map.insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }

    /// Blocking upload of `data` to `dst` at `offset` (staging copy; for weights, tables and other setup work).
    pub fn upload_sync(&self, dst: &Buffer, offset: u64, data: &[u8]) -> Result<(), BackendError> {
        if data.is_empty() {
            return Ok(());
        }
        lock(&self.transfer).upload(dst, offset, data)?;
        Ok(())
    }

    /// Blocking fill with zeros.
    pub fn zero_sync(&self, dst: &Buffer, offset: u64, size: u64) -> Result<(), BackendError> {
        lock(&self.transfer).zero(dst, offset, size)?;
        Ok(())
    }

    /// Blocking device-to-device copies `(src, src_offset, dst, dst_offset, bytes)` in one submission.
    pub fn device_copy(&self, copies: &[(Buffer, u64, Buffer, u64, u64)]) -> Result<(), BackendError> {
        let mut recorder: Recorder = self.ctx.recorder()?;
        for (src, src_offset, dst, dst_offset, bytes) in copies {
            recorder.copy_buffer(src, *src_offset, dst, *dst_offset, *bytes)?;
        }
        recorder.submit_and_wait()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_override_wins_over_the_environment() {
        set_exact_math(true);
        assert!(is_exact_math());
        set_exact_math(false);
        assert!(!is_exact_math());
        EXACT_OVERRIDE.store(0, Ordering::Relaxed);
    }

    #[test]
    fn env_flag_parsing() {
        assert!(!env_on("FLLM_VULKAN_THIS_VARIABLE_DOES_NOT_EXIST"));
    }
}
