//! Native Vulkan backend (ash): instance, device selection, logical device and the shared [`VkContext`]. Owner: package 1 (Vulkan core).
//!
//! Object model (details and threading rules in `API.md`):
//! - [`VkInstance`]: the loaded `libvulkan` plus a `VkInstance` (1.1 .. 1.3). Also used alone to list devices without creating one.
//! - [`DeviceCore`]: the logical device with its one compute queue; every other object holds an `Arc<DeviceCore>`, so the device outlives
//!   all buffers, pipelines and recorders.
//! - [`VkContext`]: `DeviceCore` + [`VkCaps`] + [`Allocator`] + [`PipelineCache`] + [`Timeline`]. Created once per device.
//!
//! Level of verification: L0 (compiles; never run on a driver).

pub mod caps;
pub mod error;
#[cfg(test)]
mod smoke;

pub use caps::{ComponentType, CoopMatrixShape, CoopScope, IntDotCaps, SubgroupOps, Vendor, VkCaps, VkDeviceType};
pub use error::{VkError, VkResult};

use super::commands::{Recorder, Timeline};
use super::memory::Allocator;
use super::pipeline::PipelineCache;
use ash::vk;
use caps::{ensure_usable, probe, Probe};
use error::vk_call;
use std::ffi::{c_char, CStr};
use std::sync::{Arc, Mutex};

/// Environment variable: admit software renderers (llvmpipe / lavapipe / SwiftShader / WARP) as devices (same switch as the wgpu path).
pub const ENV_ALLOW_SOFTWARE: &str = "FLLM_ALLOW_SOFTWARE_GPU";
/// Environment variable: `1` enables the Khronos validation layer when it is installed (debugging aid; the layer logs to stdout).
pub const ENV_VALIDATE: &str = "FLLM_VULKAN_VALIDATE";

/// `FLLM_ALLOW_SOFTWARE_GPU` is set to something other than empty / `0`.
pub fn software_allowed() -> bool {
    std::env::var_os(ENV_ALLOW_SOFTWARE).is_some_and(|v| !v.is_empty() && v != "0")
}

/// A software implementation of Vulkan (it would run the model on the CPU through a rasteriser).
pub fn is_software(device_type: VkDeviceType, name: &str) -> bool {
    if matches!(device_type, VkDeviceType::Cpu | VkDeviceType::Other) {
        return true;
    }
    let lower = name.to_ascii_lowercase();
    ["llvmpipe", "lavapipe", "swiftshader", "basic render", "software"]
        .iter()
        .any(|s| lower.contains(s))
}

/// Whether a device is offered: hardware always, software only with [`ENV_ALLOW_SOFTWARE`].
pub fn admit(device_type: VkDeviceType, name: &str, allow_software: bool) -> bool {
    allow_software || !is_software(device_type, name)
}

/// The loaded Vulkan loader and an instance.
pub struct VkInstance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    /// Version the instance was created with: the loader's version capped to 1.3, at least 1.1.
    pub api_version: u32,
    /// The validation layer is enabled.
    pub validation: bool,
}

// SAFETY: `ash::Entry` and `ash::Instance` are tables of function pointers and handles; Vulkan instance functions are thread safe except
// for the objects documented as externally synchronised, and this type exposes none of those.
unsafe impl Send for VkInstance {}
unsafe impl Sync for VkInstance {}

/// A physical device that passed the filter, with its position among the admitted devices.
#[derive(Clone)]
pub struct Candidate {
    pub handle: vk::PhysicalDevice,
    /// Index among the admitted devices (this is the `device_index` of [`VkContext::new`] and of `--device vulkan:N`).
    pub ordinal: usize,
    pub name: String,
    pub device_type: VkDeviceType,
}

impl VkInstance {
    /// Load `libvulkan` and create an instance. Fails with [`VkError::Loader`] when there is no loader or it is older than 1.1.
    pub fn new() -> VkResult<Arc<Self>> {
        // SAFETY: loading a shared library runs its initialisers; the Vulkan loader is the intended target of this call.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| VkError::Loader(e.to_string()))?;
        // SAFETY: `entry` is valid; the calls only read.
        let loader_version = unsafe { entry.try_enumerate_instance_version() }
            .map_err(vk_call("vkEnumerateInstanceVersion"))?
            .unwrap_or(vk::API_VERSION_1_0);
        if loader_version < vk::API_VERSION_1_1 {
            return Err(VkError::Loader(
                "the Vulkan loader implements only 1.0 (1.1 or newer is required)".into(),
            ));
        }
        let api_version = loader_version.min(vk::API_VERSION_1_3);
        let api_version = vk::make_api_version(0, vk::api_version_major(api_version), vk::api_version_minor(api_version), 0);

        // SAFETY: as above.
        let extensions =
            unsafe { entry.enumerate_instance_extension_properties(None) }.map_err(vk_call("vkEnumerateInstanceExtensionProperties"))?;
        let has_ext = |name: &CStr| extensions.iter().any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name));
        let mut enabled: Vec<*const c_char> = Vec::new();
        let mut flags = vk::InstanceCreateFlags::empty();
        if has_ext(ash::khr::portability_enumeration::NAME) {
            // MoltenVK and other layered implementations are only listed with this extension
            enabled.push(ash::khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            if has_ext(ash::khr::get_physical_device_properties2::NAME) {
                enabled.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
            }
        }

        let mut layers: Vec<*const c_char> = Vec::new();
        let mut validation = false;
        if std::env::var_os(ENV_VALIDATE).is_some_and(|v| v == "1") {
            // SAFETY: as above.
            let available = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
            let wanted = c"VK_LAYER_KHRONOS_validation";
            if available.iter().any(|l| l.layer_name_as_c_str().is_ok_and(|n| n == wanted)) {
                layers.push(wanted.as_ptr());
                validation = true;
            } else {
                eprintln!("[fllm-vulkan] {ENV_VALIDATE}=1 but VK_LAYER_KHRONOS_validation is not installed");
            }
        }

        let app = vk::ApplicationInfo::default()
            .application_name(c"fllm-rs")
            .application_version(1)
            .engine_name(c"fllm")
            .engine_version(1)
            .api_version(api_version);
        let info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&enabled)
            .enabled_layer_names(&layers)
            .flags(flags);
        // SAFETY: `info` and everything it points to (`app`, `enabled`, `layers`) outlive the call.
        let instance = unsafe { entry.create_instance(&info, None) }.map_err(vk_call("vkCreateInstance"))?;
        Ok(Arc::new(Self {
            entry,
            instance,
            api_version,
            validation,
        }))
    }

    /// All admitted devices in driver order (software renderers only with [`ENV_ALLOW_SOFTWARE`]).
    pub fn candidates(&self) -> VkResult<Vec<Candidate>> {
        let allow_software = software_allowed();
        // SAFETY: the instance is valid for the lifetime of `self`.
        let devices = unsafe { self.instance.enumerate_physical_devices() }.map_err(vk_call("vkEnumeratePhysicalDevices"))?;
        let mut out = Vec::new();
        for handle in devices {
            // SAFETY: `handle` comes from this instance.
            let props = unsafe { self.instance.get_physical_device_properties(handle) };
            let name = caps::c_chars_to_string(&props.device_name);
            let device_type = VkDeviceType::from_vk(props.device_type);
            if !admit(device_type, &name, allow_software) {
                continue;
            }
            out.push(Candidate {
                handle,
                ordinal: out.len(),
                name,
                device_type,
            });
        }
        Ok(out)
    }
}

impl Drop for VkInstance {
    fn drop(&mut self) {
        // SAFETY: every object created from this instance holds an `Arc<DeviceCore>` which holds an `Arc<VkInstance>`, so none is alive.
        unsafe { self.instance.destroy_instance(None) };
    }
}

/// One queue family as far as the selection is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueFamilyInfo {
    pub flags: vk::QueueFlags,
    pub count: u32,
}

/// The family used for compute and transfers: it must support compute; a family without graphics (an "async compute" family) is
/// preferred, then the one with more queues; ties go to the lowest index.
pub fn select_queue_family(families: &[QueueFamilyInfo]) -> Option<u32> {
    families
        .iter()
        .enumerate()
        .filter(|(_, f)| f.flags.contains(vk::QueueFlags::COMPUTE) && f.count > 0)
        .max_by_key(|(i, f)| (!f.flags.contains(vk::QueueFlags::GRAPHICS), f.count, std::cmp::Reverse(*i)))
        .map(|(i, _)| i as u32)
}

/// `name` is in the list of extension names.
fn has_extension(list: &[&CStr], name: &CStr) -> bool {
    list.contains(&name)
}

/// The logical device with its single queue. Everything else of the layer keeps an `Arc` of it.
pub struct DeviceCore {
    pub vki: Arc<VkInstance>,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    /// `vkQueueSubmit` needs external synchronisation of the queue.
    pub submit_lock: Mutex<()>,
    pub push_descriptor: Option<ash::khr::push_descriptor::Device>,
    pub external_memory_host: Option<ash::ext::external_memory_host::Device>,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    /// `VK_EXT_memory_budget` is enabled.
    pub memory_budget: bool,
}

// SAFETY: see `VkInstance`; the queue (the only externally synchronised object) is guarded by `submit_lock` wherever it is used.
unsafe impl Send for DeviceCore {}
unsafe impl Sync for DeviceCore {}

impl DeviceCore {
    fn create(vki: &Arc<VkInstance>, physical: vk::PhysicalDevice, probe: &Probe, queue_family: u32) -> VkResult<Arc<Self>> {
        let f = &probe.features;
        let v13 = probe.caps.api_version >= vk::API_VERSION_1_3;

        let core_features = vk::PhysicalDeviceFeatures::default()
            .shader_int16(f.shader_int16)
            .shader_int64(f.shader_int64);
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(f.storage16)
            .uniform_and_storage_buffer16_bit_access(f.storage16 && f.uniform_storage16);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .shader_float16(f.fp16)
            .shader_int8(f.int8)
            .storage_buffer8_bit_access(f.storage8)
            .uniform_and_storage_buffer8_bit_access(f.storage8 && f.uniform_storage8)
            .shader_subgroup_extended_types(f.subgroup_extended_types)
            .scalar_block_layout(f.scalar_block_layout)
            .vulkan_memory_model(f.vulkan_memory_model);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .shader_integer_dot_product(f.int_dot)
            .subgroup_size_control(f.subgroup_size_control)
            .compute_full_subgroups(f.compute_full_subgroups)
            .maintenance4(f.maintenance4);
        let mut ssc = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
            .subgroup_size_control(f.subgroup_size_control)
            .compute_full_subgroups(f.compute_full_subgroups);
        let mut dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default().shader_integer_dot_product(f.int_dot);
        let mut m4 = vk::PhysicalDeviceMaintenance4Features::default().maintenance4(f.maintenance4);
        let mut coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(f.coop_matrix);

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(core_features)
            .push_next(&mut f11)
            .push_next(&mut f12);
        if v13 {
            features2 = features2.push_next(&mut f13);
        } else {
            let has = |n: &CStr| has_extension(&probe.enable_extensions, n);
            if has(ash::ext::subgroup_size_control::NAME) {
                features2 = features2.push_next(&mut ssc);
            }
            if has(ash::khr::shader_integer_dot_product::NAME) {
                features2 = features2.push_next(&mut dot);
            }
            if has(ash::khr::maintenance4::NAME) {
                features2 = features2.push_next(&mut m4);
            }
        }
        if f.coop_matrix {
            features2 = features2.push_next(&mut coop);
        }

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let extension_ptrs: Vec<*const c_char> = probe.enable_extensions.iter().map(|n| n.as_ptr()).collect();
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extension_ptrs)
            .push_next(&mut features2);
        // SAFETY: `physical` belongs to `vki.instance`; `info` and everything it points to live until the call returns.
        let device = unsafe { vki.instance.create_device(physical, &info, None) }.map_err(vk_call("vkCreateDevice"))?;
        // SAFETY: the family index was passed in `queue_infos` with one queue.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let has_ext = |n: &CStr| has_extension(&probe.enable_extensions, n);
        let push_descriptor =
            has_ext(ash::khr::push_descriptor::NAME).then(|| ash::khr::push_descriptor::Device::new(&vki.instance, &device));
        let external_memory_host = (probe.caps.external_memory_host && has_ext(ash::ext::external_memory_host::NAME))
            .then(|| ash::ext::external_memory_host::Device::new(&vki.instance, &device));
        Ok(Arc::new(Self {
            vki: Arc::clone(vki),
            physical,
            device,
            queue,
            queue_family,
            submit_lock: Mutex::new(()),
            push_descriptor,
            external_memory_host,
            memory: probe.memory,
            memory_budget: has_ext(ash::ext::memory_budget::NAME),
        }))
    }

    /// The ash instance (for instance-level queries).
    pub fn instance(&self) -> &ash::Instance {
        &self.vki.instance
    }
}

impl Drop for DeviceCore {
    fn drop(&mut self) {
        // SAFETY: all users hold an `Arc<DeviceCore>`, so nothing else uses the device any more; idle first so no work is in flight.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}

/// Everything needed to run compute work on one Vulkan device. Create one per device with [`VkContext::new`].
pub struct VkContext {
    caps: VkCaps,
    // declaration order = drop order: pipelines and timeline go before the allocator, `core` last
    pipelines: PipelineCache,
    timeline: Arc<Timeline>,
    allocator: Arc<Allocator>,
    core: Arc<DeviceCore>,
}

impl VkContext {
    /// Open device number `device_index` of the admitted devices (`fllm-rs devices` order among the Vulkan entries).
    pub fn new(device_index: usize) -> VkResult<Arc<Self>> {
        let vki = VkInstance::new()?;
        let candidates = vki.candidates()?;
        if candidates.is_empty() {
            let hint = if software_allowed() {
                ""
            } else {
                " (software renderers are hidden; set FLLM_ALLOW_SOFTWARE_GPU=1 to use one)"
            };
            return Err(VkError::NoDevice(format!("the Vulkan loader reports no hardware GPU{hint}")));
        }
        // device_index counts the usable devices only (the same numbering as `device_info::list`)
        let mut usable = 0;
        let mut skipped = Vec::new();
        let mut chosen = None;
        for candidate in &candidates {
            let probed = probe(&vki, candidate.handle)?;
            match ensure_usable(&probed) {
                Ok(()) if usable == device_index => {
                    chosen = Some((candidate, probed));
                    break;
                }
                Ok(()) => usable += 1,
                Err(why) => skipped.push(why.to_string()),
            }
        }
        let Some((candidate, probe)) = chosen else {
            let detail = if skipped.is_empty() {
                String::new()
            } else {
                format!("; not usable: {}", skipped.join("; "))
            };
            return Err(VkError::NoDevice(format!(
                "Vulkan device {device_index} does not exist ({usable} usable){detail}"
            )));
        };

        // SAFETY: the physical device belongs to the instance.
        let families = unsafe { vki.instance.get_physical_device_queue_family_properties(candidate.handle) };
        let infos: Vec<QueueFamilyInfo> = families
            .iter()
            .map(|f| QueueFamilyInfo {
                flags: f.queue_flags,
                count: f.queue_count,
            })
            .collect();
        let family = select_queue_family(&infos)
            .ok_or_else(|| VkError::Unsupported(format!("{} has no queue family with compute support", candidate.name)))?;

        let core = DeviceCore::create(&vki, candidate.handle, &probe, family)?;
        let allocator = Allocator::new(Arc::clone(&core), &probe.caps)?;
        let pipelines = PipelineCache::new(Arc::clone(&core), &probe.caps)?;
        let timeline = Timeline::new(Arc::clone(&core))?;
        Ok(Arc::new(Self {
            caps: probe.caps,
            pipelines,
            timeline: Arc::new(timeline),
            allocator: Arc::new(allocator),
            core,
        }))
    }

    pub fn caps(&self) -> &VkCaps {
        &self.caps
    }

    pub fn allocator(&self) -> &Arc<Allocator> {
        &self.allocator
    }

    pub fn pipelines(&self) -> &PipelineCache {
        &self.pipelines
    }

    pub fn timeline(&self) -> &Arc<Timeline> {
        &self.timeline
    }

    pub fn core(&self) -> &Arc<DeviceCore> {
        &self.core
    }

    /// The ash device, for calls this layer does not wrap.
    pub fn device(&self) -> &ash::Device {
        &self.core.device
    }

    /// A new command recorder (one per stream / thread; see `API.md`).
    pub fn recorder(self: &Arc<Self>) -> VkResult<Recorder> {
        Recorder::new(self)
    }

    /// Block until the queue is idle.
    pub fn wait_idle(&self) -> VkResult<()> {
        let _guard = self.core.submit_lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: the queue is externally synchronised by `submit_lock`.
        unsafe { self.core.device.queue_wait_idle(self.core.queue) }.map_err(vk_call("vkQueueWaitIdle"))
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        // pipelines, buffers of the allocator and the timeline are destroyed after this body: nothing may be in flight
        let _ = self.wait_idle();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_renderers_are_hidden_by_default() {
        assert!(is_software(VkDeviceType::Cpu, "anything"));
        assert!(is_software(VkDeviceType::Discrete, "llvmpipe (LLVM 17.0.6, 256 bits)"));
        assert!(is_software(VkDeviceType::Other, "SwiftShader Device"));
        assert!(is_software(VkDeviceType::Integrated, "Microsoft Basic Render Driver"));
        assert!(!is_software(VkDeviceType::Integrated, "Intel(R) Iris(R) Xe Graphics (ADL GT2)"));
        assert!(!is_software(VkDeviceType::Discrete, "NVIDIA GeForce RTX 4090"));
        assert!(!is_software(VkDeviceType::Virtual, "Virtio-GPU Venus (AMD Radeon RX 7900)"));
        assert!(!admit(VkDeviceType::Cpu, "lavapipe", false));
        assert!(admit(VkDeviceType::Cpu, "lavapipe", true));
        assert!(admit(VkDeviceType::Discrete, "RTX", false));
    }

    #[test]
    fn queue_family_selection() {
        let gfx = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
        let compute = vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
        let transfer = vk::QueueFlags::TRANSFER;
        // NVIDIA-like: universal family first, async compute family second -> the compute-only family wins
        let fams = [
            QueueFamilyInfo { flags: gfx, count: 16 },
            QueueFamilyInfo { flags: transfer, count: 2 },
            QueueFamilyInfo { flags: compute, count: 8 },
        ];
        assert_eq!(select_queue_family(&fams), Some(2));
        // Intel-like: one universal family
        assert_eq!(select_queue_family(&[QueueFamilyInfo { flags: gfx, count: 1 }]), Some(0));
        // no compute at all
        assert_eq!(select_queue_family(&[QueueFamilyInfo { flags: transfer, count: 1 }]), None);
        // two compute-only families: more queues first, then the lower index
        let tie = [
            QueueFamilyInfo { flags: compute, count: 4 },
            QueueFamilyInfo { flags: compute, count: 4 },
            QueueFamilyInfo { flags: compute, count: 2 },
        ];
        assert_eq!(select_queue_family(&tie), Some(0));
        // a family without queues is never chosen
        assert_eq!(select_queue_family(&[QueueFamilyInfo { flags: compute, count: 0 }]), None);
    }
}
