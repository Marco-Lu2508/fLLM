//! Native Vulkan backend (ash): compute pipelines and the pipeline cache. Owner: package 1 (Vulkan core).
//!
//! A [`ComputePipeline`] is a SPIR-V compute shader (entry point `main`) with a fixed interface: `bindings` storage buffers in descriptor
//! set 0 (binding 0..n) and `push_constant_bytes` of push constants. Specialization constants select variants (tile sizes, quant types)
//! without separate shader files. [`PipelineCache`] deduplicates pipelines by their full description, keeps a `VkPipelineCache` and stores
//! it on disk (`cache_file`), so that the second start of the engine does not recompile any shader.
//!
//! Resolve pipelines once (at backend start) and keep the `Arc<ComputePipeline>`: the lookup hashes the SPIR-V.
//!
//! Level of verification: L0 (compiles; the pure parts are unit tested, the Vulkan calls never ran).

pub mod cache_file;

use super::instance::error::{vk_call, VkError, VkResult};
use super::instance::{DeviceCore, VkCaps};
use ash::vk;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

/// SPIR-V magic number.
const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Value of a specialization constant (all are 4 bytes wide).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecValue {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
}

impl SpecValue {
    /// The 4 bytes handed to Vulkan (booleans are `VkBool32`).
    pub fn to_bits(self) -> u32 {
        match self {
            Self::U32(v) => v,
            Self::I32(v) => v as u32,
            Self::F32(v) => v.to_bits(),
            Self::Bool(v) => u32::from(v),
        }
    }
}

/// `layout(constant_id = id)` of the shader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecConstant {
    pub id: u32,
    pub value: SpecValue,
}

impl SpecConstant {
    pub fn u32(id: u32, value: u32) -> Self {
        Self {
            id,
            value: SpecValue::U32(value),
        }
    }
    pub fn i32(id: u32, value: i32) -> Self {
        Self {
            id,
            value: SpecValue::I32(value),
        }
    }
    pub fn f32(id: u32, value: f32) -> Self {
        Self {
            id,
            value: SpecValue::F32(value),
        }
    }
    pub fn bool(id: u32, value: bool) -> Self {
        Self {
            id,
            value: SpecValue::Bool(value),
        }
    }
}

/// Everything that defines a pipeline.
#[derive(Debug, Clone, Copy)]
pub struct PipelineSpec<'a> {
    /// For error messages only (not part of the cache key).
    pub name: &'a str,
    pub spirv: &'a [u32],
    /// Storage buffers in set 0, bindings `0..bindings`.
    pub bindings: u32,
    /// Bytes of push constants (a multiple of 4, at most `caps.max_push_constants`, which is at least 128).
    pub push_constant_bytes: u32,
    pub spec_constants: &'a [SpecConstant],
    /// Run with exactly this subgroup size (`VK_EXT_subgroup_size_control`); fails if the device cannot.
    pub required_subgroup_size: Option<u32>,
    /// Every subgroup is full (workgroup size is a multiple of the subgroup size); needs `caps.compute_full_subgroups`.
    pub require_full_subgroups: bool,
}

impl<'a> PipelineSpec<'a> {
    pub fn new(name: &'a str, spirv: &'a [u32], bindings: u32, push_constant_bytes: u32) -> Self {
        Self {
            name,
            spirv,
            bindings,
            push_constant_bytes,
            spec_constants: &[],
            required_subgroup_size: None,
            require_full_subgroups: false,
        }
    }

    pub fn with_spec_constants(mut self, constants: &'a [SpecConstant]) -> Self {
        self.spec_constants = constants;
        self
    }

    pub fn with_subgroup_size(mut self, size: u32) -> Self {
        self.required_subgroup_size = Some(size);
        self
    }

    pub fn with_full_subgroups(mut self) -> Self {
        self.require_full_subgroups = true;
        self
    }

    /// The identity of the pipeline (what two specs must share to be the same pipeline).
    pub fn key(&self) -> PipelineKey {
        let mut hasher = Sha256::new();
        for word in self.spirv {
            hasher.update(word.to_le_bytes());
        }
        let digest = hasher.finalize();
        let mut spirv_hash = [0u8; 16];
        spirv_hash.copy_from_slice(&digest[..16]);
        PipelineKey {
            spirv_hash,
            spirv_words: self.spirv.len(),
            bindings: self.bindings,
            push_constant_bytes: self.push_constant_bytes,
            spec: self.spec_constants.iter().map(|c| (c.id, c.value.to_bits())).collect(),
            subgroup_size: self.required_subgroup_size.unwrap_or(0),
            full_subgroups: self.require_full_subgroups,
        }
    }
}

/// Hashable identity of a [`PipelineSpec`] (the name is not part of it).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    spirv_hash: [u8; 16],
    spirv_words: usize,
    bindings: u32,
    push_constant_bytes: u32,
    spec: Vec<(u32, u32)>,
    subgroup_size: u32,
    full_subgroups: bool,
}

/// What the validation of a spec needs to know about the device.
#[derive(Debug, Clone, Copy)]
pub struct PipelineLimits {
    pub max_bindings: u32,
    pub max_push_constants: u32,
    pub subgroup_size_control: bool,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub compute_full_subgroups: bool,
}

impl PipelineLimits {
    pub fn from_caps(caps: &VkCaps) -> Self {
        Self {
            max_bindings: caps.max_per_stage_storage_buffers.min(super::commands::MAX_BINDINGS),
            max_push_constants: caps.max_push_constants,
            subgroup_size_control: caps.subgroup_size_control,
            min_subgroup_size: caps.min_subgroup_size,
            max_subgroup_size: caps.max_subgroup_size,
            compute_full_subgroups: caps.compute_full_subgroups,
        }
    }
}

/// Check a spec before any Vulkan call; the message names the shader.
pub fn validate_spec(spec: &PipelineSpec<'_>, limits: &PipelineLimits) -> VkResult<()> {
    let bad = |what: String| Err(VkError::InvalidArgument(format!("pipeline '{}': {what}", spec.name)));
    if spec.spirv.len() < 5 || spec.spirv[0] != SPIRV_MAGIC {
        return bad("not a SPIR-V module".into());
    }
    if spec.bindings > limits.max_bindings {
        return bad(format!(
            "{} storage buffers exceed the device limit of {}",
            spec.bindings, limits.max_bindings
        ));
    }
    if !spec.push_constant_bytes.is_multiple_of(4) || spec.push_constant_bytes > limits.max_push_constants {
        return bad(format!(
            "{} bytes of push constants (must be a multiple of 4, device limit {})",
            spec.push_constant_bytes, limits.max_push_constants
        ));
    }
    if let Some(size) = spec.required_subgroup_size {
        if !limits.subgroup_size_control {
            return Err(VkError::Unsupported(format!(
                "pipeline '{}': the device cannot pin the subgroup size",
                spec.name
            )));
        }
        if !size.is_power_of_two() || size < limits.min_subgroup_size || size > limits.max_subgroup_size {
            return Err(VkError::Unsupported(format!(
                "pipeline '{}': subgroup size {size} outside the device range {}..={}",
                spec.name, limits.min_subgroup_size, limits.max_subgroup_size
            )));
        }
    }
    if spec.require_full_subgroups && !limits.compute_full_subgroups {
        return Err(VkError::Unsupported(format!(
            "pipeline '{}': the device lacks computeFullSubgroups",
            spec.name
        )));
    }
    let mut ids: Vec<u32> = spec.spec_constants.iter().map(|c| c.id).collect();
    ids.sort_unstable();
    if ids.windows(2).any(|w| w[0] == w[1]) {
        return bad("duplicate specialization constant id".into());
    }
    Ok(())
}

/// A compiled compute pipeline with its layouts. Immutable; destroyed when the last `Arc` goes.
pub struct ComputePipeline {
    core: Arc<DeviceCore>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    name: String,
    bindings: u32,
    push_constant_bytes: u32,
    push_descriptors: bool,
}

impl ComputePipeline {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn bindings(&self) -> u32 {
        self.bindings
    }
    pub fn push_constant_bytes(&self) -> u32 {
        self.push_constant_bytes
    }
    /// The recorder pushes descriptors for this pipeline (`VK_KHR_push_descriptor`) instead of allocating sets.
    pub fn uses_push_descriptors(&self) -> bool {
        self.push_descriptors
    }
    pub(crate) fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }
    pub(crate) fn raw_layout(&self) -> vk::PipelineLayout {
        self.layout
    }
    pub(crate) fn raw_set_layout(&self) -> vk::DescriptorSetLayout {
        self.set_layout
    }
}

impl std::fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputePipeline")
            .field("name", &self.name)
            .field("bindings", &self.bindings)
            .field("push_constant_bytes", &self.push_constant_bytes)
            .field("push_descriptors", &self.push_descriptors)
            .finish()
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        // SAFETY: the handles were created from this device; the owner guarantees that no submitted work still uses the pipeline (see
        // `VkContext::drop`, which waits for the queue before the cache is dropped).
        unsafe {
            self.core.device.destroy_pipeline(self.pipeline, None);
            self.core.device.destroy_pipeline_layout(self.layout, None);
            self.core.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// The cache file of one device and the hash of the data it was loaded from (to skip rewriting identical data).
struct DiskCache {
    path: PathBuf,
    loaded_hash: Option<u64>,
}

/// Pipelines of one device, deduplicated, plus the `VkPipelineCache` behind them. Thread safe.
pub struct PipelineCache {
    core: Arc<DeviceCore>,
    cache: vk::PipelineCache,
    limits: PipelineLimits,
    /// Bindings up to this count use push descriptors (when the device has them).
    max_push_bindings: u32,
    pipelines: Mutex<HashMap<PipelineKey, Arc<ComputePipeline>>>,
    disk: Option<DiskCache>,
    /// Pipelines compiled in this process (not served from the in-memory map).
    compiled: AtomicU64,
}

impl PipelineCache {
    pub(crate) fn new(core: Arc<DeviceCore>, caps: &VkCaps) -> VkResult<Self> {
        let mut disk = cache_file::cache_dir().map(|dir| DiskCache {
            path: dir.join(cache_file::cache_file_name(
                caps.vendor_id,
                caps.device_id,
                &caps.pipeline_cache_uuid,
            )),
            loaded_hash: None,
        });
        let mut initial: Vec<u8> = Vec::new();
        if let Some(d) = disk.as_mut() {
            if let Ok(bytes) = std::fs::read(&d.path) {
                match cache_file::decode(&bytes) {
                    Some(payload)
                        if cache_file::vulkan_header_matches(payload, caps.vendor_id, caps.device_id, &caps.pipeline_cache_uuid) =>
                    {
                        d.loaded_hash = Some(cache_file::fnv1a64(payload));
                        initial = payload.to_vec();
                    }
                    _ => {} // damaged or from another driver: start empty, it is rewritten on the next flush
                }
            }
        }
        let create = |data: &[u8]| {
            let info = vk::PipelineCacheCreateInfo::default().initial_data(data);
            // SAFETY: `info` points to `data`, which outlives the call.
            unsafe { core.device.create_pipeline_cache(&info, None) }
        };
        let cache = match create(&initial) {
            Ok(c) => c,
            Err(_) if !initial.is_empty() => {
                if let Some(d) = disk.as_mut() {
                    d.loaded_hash = None;
                }
                create(&[]).map_err(vk_call("vkCreatePipelineCache"))?
            }
            Err(e) => return Err(VkError::from_result("vkCreatePipelineCache", e)),
        };
        let max_push_bindings = if core.push_descriptor.is_some() {
            caps.max_push_descriptors
        } else {
            0
        };
        Ok(Self {
            core,
            cache,
            limits: PipelineLimits::from_caps(caps),
            max_push_bindings,
            pipelines: Mutex::new(HashMap::new()),
            disk,
            compiled: AtomicU64::new(0),
        })
    }

    /// Path of the cache file (None when the disk cache is off).
    pub fn disk_path(&self) -> Option<&std::path::Path> {
        self.disk.as_ref().map(|d| d.path.as_path())
    }

    /// Number of distinct pipelines held.
    pub fn len(&self) -> usize {
        self.pipelines.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of pipelines compiled by this process (a first start compiles all, later starts mostly load from the cache).
    pub fn compiled_count(&self) -> u64 {
        self.compiled.load(Ordering::Relaxed)
    }

    /// The pipeline for `spec`, compiled on first use.
    pub fn get_or_create(&self, spec: &PipelineSpec<'_>) -> VkResult<Arc<ComputePipeline>> {
        validate_spec(spec, &self.limits)?;
        let key = spec.key();
        if let Some(found) = self.pipelines.lock().unwrap_or_else(PoisonError::into_inner).get(&key) {
            return Ok(Arc::clone(found));
        }
        // compile outside the lock: other threads may compile different pipelines in the meantime
        let created = Arc::new(self.compile(spec)?);
        let mut map = self.pipelines.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(Arc::clone(map.entry(key).or_insert(created)))
    }

    /// Compile many pipelines in parallel (at most 8 threads; the driver cache is internally synchronised). The results are in the
    /// order of `specs`.
    pub fn create_many(&self, specs: &[PipelineSpec<'_>]) -> Vec<VkResult<Arc<ComputePipeline>>> {
        if specs.len() <= 1 {
            return specs.iter().map(|s| self.get_or_create(s)).collect();
        }
        let threads = specs.len().min(8);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: Mutex<Vec<Option<VkResult<Arc<ComputePipeline>>>>> = Mutex::new((0..specs.len()).map(|_| None).collect());
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(spec) = specs.get(i) else { break };
                    let result = self.get_or_create(spec);
                    results.lock().unwrap_or_else(PoisonError::into_inner)[i] = Some(result);
                });
            }
        });
        results
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(VkError::InvalidArgument("pipeline worker did not run".into()))))
            .collect()
    }

    fn compile(&self, spec: &PipelineSpec<'_>) -> VkResult<ComputePipeline> {
        let device = &self.core.device;
        let use_push = self.max_push_bindings >= spec.bindings && self.core.push_descriptor.is_some();
        // SAFETY: plain object creation; every failure path destroys what was created before it. All structs referenced by the create
        // infos live on this frame until the calls return.
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..spec.bindings)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let mut set_flags = vk::DescriptorSetLayoutCreateFlags::empty();
            if use_push {
                set_flags |= vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR;
            }
            let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings).flags(set_flags);
            let set_layout = device
                .create_descriptor_set_layout(&set_info, None)
                .map_err(vk_call("vkCreateDescriptorSetLayout"))?;

            let set_layouts = [set_layout];
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(spec.push_constant_bytes)];
            let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            if spec.push_constant_bytes > 0 {
                layout_info = layout_info.push_constant_ranges(&ranges);
            }
            let layout = match device.create_pipeline_layout(&layout_info, None) {
                Ok(l) => l,
                Err(e) => {
                    device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(VkError::from_result("vkCreatePipelineLayout", e));
                }
            };

            let module_info = vk::ShaderModuleCreateInfo::default().code(spec.spirv);
            let module = match device.create_shader_module(&module_info, None) {
                Ok(m) => m,
                Err(e) => {
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(VkError::from_result("vkCreateShaderModule", e));
                }
            };

            let entries: Vec<vk::SpecializationMapEntry> = spec
                .spec_constants
                .iter()
                .enumerate()
                .map(|(i, c)| vk::SpecializationMapEntry {
                    constant_id: c.id,
                    offset: (i * 4) as u32,
                    size: 4,
                })
                .collect();
            let data: Vec<u8> = spec.spec_constants.iter().flat_map(|c| c.value.to_bits().to_le_bytes()).collect();
            let spec_info = vk::SpecializationInfo::default().map_entries(&entries).data(&data);
            let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(spec.required_subgroup_size.unwrap_or(0));
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            if !entries.is_empty() {
                stage = stage.specialization_info(&spec_info);
            }
            if spec.required_subgroup_size.is_some() {
                stage = stage.push_next(&mut required);
            }
            if spec.require_full_subgroups {
                stage = stage.flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
            }
            let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let created = device.create_compute_pipelines(self.cache, &[info], None);
            device.destroy_shader_module(module, None);
            match created {
                Ok(pipelines) => {
                    self.compiled.fetch_add(1, Ordering::Relaxed);
                    Ok(ComputePipeline {
                        core: Arc::clone(&self.core),
                        pipeline: pipelines[0],
                        layout,
                        set_layout,
                        name: spec.name.to_owned(),
                        bindings: spec.bindings,
                        push_constant_bytes: spec.push_constant_bytes,
                        push_descriptors: use_push,
                    })
                }
                Err((_, e)) => {
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_set_layout(set_layout, None);
                    Err(VkError::from_result("vkCreateComputePipelines", e))
                }
            }
        }
    }

    /// Write the driver's cache data to disk when it changed. Returns whether a file was written. Call after the start-up compilation
    /// of all pipelines; also done when the context is dropped. Failures leave the old file untouched.
    pub fn flush_to_disk(&self) -> VkResult<bool> {
        let Some(disk) = &self.disk else { return Ok(false) };
        // SAFETY: the cache handle is valid for the lifetime of `self`.
        let data = unsafe { self.core.device.get_pipeline_cache_data(self.cache) }.map_err(vk_call("vkGetPipelineCacheData"))?;
        if data.is_empty() || disk.loaded_hash == Some(cache_file::fnv1a64(&data)) {
            return Ok(false);
        }
        let io = |e: std::io::Error| VkError::Io(format!("{}: {e}", disk.path.display()));
        if let Some(parent) = disk.path.parent() {
            std::fs::create_dir_all(parent).map_err(io)?;
        }
        // write next to the target and rename: a crash never leaves a half-written cache
        let tmp = disk.path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, cache_file::encode(&data)).map_err(io)?;
        if let Err(e) = std::fs::rename(&tmp, &disk.path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(io(e));
        }
        Ok(true)
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        if self.compiled.load(Ordering::Relaxed) > 0 {
            if let Err(e) = self.flush_to_disk() {
                eprintln!("[fllm-vulkan] cannot store the pipeline cache: {e}");
            }
        }
        // SAFETY: created by this struct; pipelines created from it stay valid after it is destroyed.
        unsafe { self.core.device.destroy_pipeline_cache(self.cache, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODULE: [u32; 6] = [SPIRV_MAGIC, 0x0001_0500, 0, 1, 0, 7];

    fn limits() -> PipelineLimits {
        PipelineLimits {
            max_bindings: 16,
            max_push_constants: 128,
            subgroup_size_control: true,
            min_subgroup_size: 32,
            max_subgroup_size: 64,
            compute_full_subgroups: true,
        }
    }

    #[test]
    fn spec_value_bits() {
        assert_eq!(SpecValue::U32(7).to_bits(), 7);
        assert_eq!(SpecValue::I32(-1).to_bits(), u32::MAX);
        assert_eq!(SpecValue::F32(1.0).to_bits(), 0x3F80_0000);
        assert_eq!(SpecValue::Bool(true).to_bits(), 1);
        assert_eq!(SpecValue::Bool(false).to_bits(), 0);
    }

    #[test]
    fn key_depends_on_everything_but_the_name() {
        let base = PipelineSpec::new("a", &MODULE, 3, 16);
        assert_eq!(base.key(), PipelineSpec::new("renamed", &MODULE, 3, 16).key());
        assert_ne!(base.key(), PipelineSpec::new("a", &MODULE, 4, 16).key());
        assert_ne!(base.key(), PipelineSpec::new("a", &MODULE, 3, 20).key());
        let other_module = [SPIRV_MAGIC, 0x0001_0500, 0, 1, 0, 8];
        assert_ne!(base.key(), PipelineSpec::new("a", &other_module, 3, 16).key());
        let c1 = [SpecConstant::u32(0, 32)];
        let c2 = [SpecConstant::u32(0, 64)];
        assert_ne!(base.with_spec_constants(&c1).key(), base.with_spec_constants(&c2).key());
        assert_ne!(base.with_spec_constants(&c1).key(), base.key());
        assert_ne!(base.with_subgroup_size(32).key(), base.key());
        assert_ne!(base.with_full_subgroups().key(), base.key());
        assert_eq!(base.with_spec_constants(&c1).key(), base.with_spec_constants(&c1).key());
    }

    #[test]
    fn validation_catches_bad_specs() {
        let l = limits();
        assert!(validate_spec(&PipelineSpec::new("ok", &MODULE, 4, 32), &l).is_ok());
        assert!(validate_spec(&PipelineSpec::new("short", &MODULE[..3], 4, 32), &l).is_err());
        let not_spirv = [1u32, 2, 3, 4, 5, 6];
        assert!(validate_spec(&PipelineSpec::new("magic", &not_spirv, 4, 32), &l).is_err());
        assert!(validate_spec(&PipelineSpec::new("many", &MODULE, 17, 0), &l).is_err());
        assert!(validate_spec(&PipelineSpec::new("push", &MODULE, 1, 130), &l).is_err());
        assert!(validate_spec(&PipelineSpec::new("align", &MODULE, 1, 6), &l).is_err());
        assert!(validate_spec(&PipelineSpec::new("sg", &MODULE, 1, 0).with_subgroup_size(32), &l).is_ok());
        assert!(validate_spec(&PipelineSpec::new("sg", &MODULE, 1, 0).with_subgroup_size(16), &l).is_err());
        assert!(validate_spec(&PipelineSpec::new("sg", &MODULE, 1, 0).with_subgroup_size(48), &l).is_err());
        let mut none = l;
        none.subgroup_size_control = false;
        assert!(validate_spec(&PipelineSpec::new("sg", &MODULE, 1, 0).with_subgroup_size(32), &none).is_err());
        none = l;
        none.compute_full_subgroups = false;
        assert!(validate_spec(&PipelineSpec::new("full", &MODULE, 1, 0).with_full_subgroups(), &none).is_err());
        let dup = [SpecConstant::u32(1, 1), SpecConstant::bool(1, true)];
        assert!(validate_spec(&PipelineSpec::new("dup", &MODULE, 1, 0).with_spec_constants(&dup), &l).is_err());
    }
}
