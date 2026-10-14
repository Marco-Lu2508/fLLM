//! Capabilities of a Vulkan device: what the shaders of the other packages may rely on.
//!
//! [`VkCaps`] is plain data (`Clone + Debug`). It is filled by [`probe`] from a physical device without creating a logical device (so the
//! device list of `fllm-rs devices` can use it) and describes what the logical device is created with: the layer enables every
//! capability it knows about that the device supports, so "supported" and "enabled" are the same thing here.
//!
//! Level of verification: L0 (compiles; the queries have never run on a driver).

use super::error::{vk_call, VkError, VkResult};
use super::VkInstance;
use ash::vk;
use std::collections::HashSet;
use std::ffi::{c_char, CStr};

/// GPU vendor, from the PCI vendor id (or the Khronos-assigned ids of software implementations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Arm,
    Qualcomm,
    Apple,
    ImgTec,
    Broadcom,
    /// Mesa software renderers (llvmpipe / lavapipe) report this id.
    Mesa,
    Microsoft,
    Other(u32),
}

impl Vendor {
    pub fn from_id(id: u32) -> Self {
        match id {
            0x10DE => Self::Nvidia,
            0x1002 | 0x1022 => Self::Amd,
            0x8086 | 0x8087 => Self::Intel,
            0x13B5 => Self::Arm,
            0x5143 => Self::Qualcomm,
            0x106B => Self::Apple,
            0x1010 => Self::ImgTec,
            0x14E4 => Self::Broadcom,
            0x10005 => Self::Mesa,
            0x1414 => Self::Microsoft,
            other => Self::Other(other),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Arm => "ARM",
            Self::Qualcomm => "Qualcomm",
            Self::Apple => "Apple",
            Self::ImgTec => "Imagination",
            Self::Broadcom => "Broadcom",
            Self::Mesa => "Mesa",
            Self::Microsoft => "Microsoft",
            Self::Other(_) => "unknown vendor",
        }
    }
}

/// Kind of device (`VkPhysicalDeviceType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkDeviceType {
    Discrete,
    Integrated,
    Virtual,
    /// A software implementation (llvmpipe, SwiftShader, WARP).
    Cpu,
    Other,
}

impl VkDeviceType {
    pub fn from_vk(ty: vk::PhysicalDeviceType) -> Self {
        match ty {
            vk::PhysicalDeviceType::DISCRETE_GPU => Self::Discrete,
            vk::PhysicalDeviceType::INTEGRATED_GPU => Self::Integrated,
            vk::PhysicalDeviceType::VIRTUAL_GPU => Self::Virtual,
            vk::PhysicalDeviceType::CPU => Self::Cpu,
            _ => Self::Other,
        }
    }
}

/// Element type of a cooperative-matrix operand (`VkComponentTypeKHR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentType {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    /// `VK_COMPONENT_TYPE_BFLOAT16_KHR` (raw value 1000141000, VK_KHR_shader_bfloat16; not in the ash version used here).
    Bf16,
    /// A type this layer does not know (the raw `VkComponentTypeKHR`), for example the FP8 types of NVIDIA.
    Other(i32),
}

impl ComponentType {
    pub fn from_raw(raw: i32) -> Self {
        match raw {
            0 => Self::F16,
            1 => Self::F32,
            2 => Self::F64,
            3 => Self::I8,
            4 => Self::I16,
            5 => Self::I32,
            6 => Self::I64,
            7 => Self::U8,
            8 => Self::U16,
            9 => Self::U32,
            10 => Self::U64,
            1_000_141_000 => Self::Bf16,
            other => Self::Other(other),
        }
    }
}

/// Scope of a cooperative matrix operation (`VkScopeKHR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoopScope {
    Device,
    Workgroup,
    Subgroup,
    QueueFamily,
    Other(i32),
}

impl CoopScope {
    pub fn from_raw(raw: i32) -> Self {
        match raw {
            1 => Self::Device,
            2 => Self::Workgroup,
            3 => Self::Subgroup,
            5 => Self::QueueFamily,
            other => Self::Other(other),
        }
    }
}

/// One combination of `VK_KHR_cooperative_matrix`: `result = A(m x k) * B(k x n) + C(m x n)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoopMatrixShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a: ComponentType,
    pub b: ComponentType,
    pub c: ComponentType,
    pub result: ComponentType,
    pub saturating: bool,
    pub scope: CoopScope,
}

impl CoopMatrixShape {
    fn from_vk(p: &vk::CooperativeMatrixPropertiesKHR<'_>) -> Self {
        Self {
            m: p.m_size,
            n: p.n_size,
            k: p.k_size,
            a: ComponentType::from_raw(p.a_type.as_raw()),
            b: ComponentType::from_raw(p.b_type.as_raw()),
            c: ComponentType::from_raw(p.c_type.as_raw()),
            result: ComponentType::from_raw(p.result_type.as_raw()),
            saturating: p.saturating_accumulation != vk::FALSE,
            scope: CoopScope::from_raw(p.scope.as_raw()),
        }
    }

    /// `m * n * k`: the work one subgroup-level multiply-accumulate does.
    pub fn volume(&self) -> u64 {
        u64::from(self.m) * u64::from(self.n) * u64::from(self.k)
    }
}

/// Subgroup operations a compute shader may use (`VkSubgroupFeatureFlags` for the compute stage).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubgroupOps {
    pub basic: bool,
    pub vote: bool,
    pub arithmetic: bool,
    pub ballot: bool,
    pub shuffle: bool,
    pub shuffle_relative: bool,
    pub clustered: bool,
    pub quad: bool,
}

impl SubgroupOps {
    /// Operations of `flags`, or none when the compute stage is not among `stages`.
    pub fn from_flags(flags: vk::SubgroupFeatureFlags, stages: vk::ShaderStageFlags) -> Self {
        if !stages.contains(vk::ShaderStageFlags::COMPUTE) {
            return Self::default();
        }
        let has = |f| flags.contains(f);
        Self {
            basic: has(vk::SubgroupFeatureFlags::BASIC),
            vote: has(vk::SubgroupFeatureFlags::VOTE),
            arithmetic: has(vk::SubgroupFeatureFlags::ARITHMETIC),
            ballot: has(vk::SubgroupFeatureFlags::BALLOT),
            shuffle: has(vk::SubgroupFeatureFlags::SHUFFLE),
            shuffle_relative: has(vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE),
            clustered: has(vk::SubgroupFeatureFlags::CLUSTERED),
            quad: has(vk::SubgroupFeatureFlags::QUAD),
        }
    }
}

/// `VK_KHR_shader_integer_dot_product` (core in 1.3): which dot-product forms the hardware accelerates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IntDotCaps {
    /// The shader feature is enabled (SPIR-V `DotProduct` capabilities may be used).
    pub enabled: bool,
    /// `OpSDot` / `OpUDot` / `OpSUDot` on 4x8-bit values packed in a 32-bit word (`dotPacked4x8EXT` in GLSL), accelerated.
    pub packed_4x8_signed: bool,
    pub packed_4x8_unsigned: bool,
    pub packed_4x8_mixed: bool,
    /// The same on 8-bit vectors.
    pub vec_8bit_signed: bool,
    pub vec_8bit_unsigned: bool,
    pub vec_8bit_mixed: bool,
    /// Accumulating forms (`OpSDotAccSat` ...) on packed 4x8-bit data, accelerated.
    pub acc_sat_packed_4x8_signed: bool,
}

impl IntDotCaps {
    /// The packed signed 8-bit dot (what Q8_0 x Q8_0 / Q4 x Q8 kernels use) is accelerated.
    pub fn int8_accelerated(&self) -> bool {
        self.enabled && self.packed_4x8_signed
    }
}

/// Everything the layer knows about the device.
#[derive(Debug, Clone)]
pub struct VkCaps {
    // identity
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub vendor: Vendor,
    pub device_type: VkDeviceType,
    /// `VK_MAKE_API_VERSION` of the device, capped to what the instance was created with (at most 1.3).
    pub api_version: u32,
    pub driver_version: u32,
    pub driver_name: String,
    pub driver_info: String,
    pub pipeline_cache_uuid: [u8; 16],
    pub device_uuid: [u8; 16],
    pub driver_uuid: [u8; 16],
    // memory
    /// Sum of the device-local heaps (bytes).
    pub vram_bytes: u64,
    /// Device and host share one memory (integrated GPU, or no device-local type without host access): uploads can write into mapped
    /// device memory, `import_host_ptr` is cheap.
    pub unified_memory: bool,
    /// Approximate memory bandwidth in GB/s from a small table of well-known devices; `None` when unknown. Planning hint only.
    pub bandwidth_hint_gbps: Option<f32>,
    pub max_memory_allocation_size: u64,
    pub max_buffer_size: u64,
    pub max_storage_buffer_range: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub non_coherent_atom_size: u64,
    pub max_memory_allocation_count: u32,
    // compute limits
    pub max_shared_memory: u32,
    pub max_workgroup_invocations: u32,
    pub max_workgroup_size: [u32; 3],
    pub max_workgroup_count: [u32; 3],
    pub max_push_constants: u32,
    pub max_per_stage_storage_buffers: u32,
    pub max_push_descriptors: u32,
    // subgroups
    pub subgroup_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub subgroup_ops: SubgroupOps,
    /// `subgroupSizeControl` + the stage supports a required size: `PipelineSpec::required_subgroup_size` works.
    pub subgroup_size_control: bool,
    pub compute_full_subgroups: bool,
    // shader types
    /// `shaderFloat16` (arithmetic on `float16_t`).
    pub fp16: bool,
    /// `shaderInt8`.
    pub int8: bool,
    pub int16: bool,
    pub int64: bool,
    /// `storageBuffer16BitAccess`: 16-bit loads/stores from storage buffers (`float16_t` / `uint16_t` members).
    pub storage16: bool,
    /// `storageBuffer8BitAccess`.
    pub storage8: bool,
    pub subgroup_extended_types: bool,
    pub scalar_block_layout: bool,
    /// `vulkanMemoryModel` is enabled (the cooperative-matrix shaders of the quantized linear layer declare the `VulkanMemoryModel` capability).
    pub vulkan_memory_model: bool,
    // integer dot product / cooperative matrix
    pub int_dot: IntDotCaps,
    /// `VK_KHR_cooperative_matrix` is enabled; `coop_matrix` lists the supported combinations (all scopes).
    pub coop_matrix_supported: bool,
    pub coop_matrix: Vec<CoopMatrixShape>,
    // extensions the layer uses
    pub push_descriptor: bool,
    pub external_memory_host: bool,
    /// Alignment of pointer and size for `import_host_ptr` (0 when the extension is missing).
    pub min_imported_host_pointer_alignment: u64,
    pub memory_budget: bool,
    pub maintenance4: bool,
    /// Names of all extensions the device offers (not only the enabled ones).
    pub extensions: Vec<String>,
}

impl VkCaps {
    /// `major.minor.patch` of the device's Vulkan version.
    pub fn api_version_string(&self) -> String {
        format!(
            "{}.{}.{}",
            vk::api_version_major(self.api_version),
            vk::api_version_minor(self.api_version),
            vk::api_version_patch(self.api_version)
        )
    }

    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.iter().any(|e| e == name)
    }

    /// Cooperative-matrix combinations with subgroup scope whose operand types are exactly `a`, `b`, `c` and `result`.
    pub fn coop_matrix_for(&self, a: ComponentType, b: ComponentType, c: ComponentType, result: ComponentType) -> Vec<CoopMatrixShape> {
        self.coop_matrix
            .iter()
            .filter(|s| s.scope == CoopScope::Subgroup && s.a == a && s.b == b && s.c == c && s.result == result)
            .copied()
            .collect()
    }

    /// The subgroup-scope combination with the largest `m * n * k` for these types (ties: larger `k`, then larger `m`).
    pub fn best_coop_matrix(&self, a: ComponentType, b: ComponentType, c: ComponentType, result: ComponentType) -> Option<CoopMatrixShape> {
        self.coop_matrix_for(a, b, c, result)
            .into_iter()
            .max_by_key(|s| (s.volume(), s.k, s.m))
    }

    /// One-line summary for `fllm-rs devices`.
    pub fn describe(&self) -> String {
        let mut parts = vec![format!("Vulkan {}", self.api_version_string())];
        if !self.driver_name.is_empty() {
            let info = if self.driver_info.is_empty() {
                String::new()
            } else {
                format!(" {}", self.driver_info)
            };
            parts.push(format!("{}{info}", self.driver_name));
        }
        parts.push(format!("subgroup {}", self.subgroup_size));
        let mut features = Vec::new();
        if self.fp16 {
            features.push("fp16");
        }
        if self.int8 {
            features.push("int8");
        }
        if self.int_dot.int8_accelerated() {
            features.push("int-dot");
        }
        if self
            .best_coop_matrix(ComponentType::F16, ComponentType::F16, ComponentType::F32, ComponentType::F32)
            .is_some()
        {
            features.push("coop-matrix");
        }
        if self.unified_memory {
            features.push("unified-memory");
        }
        if !features.is_empty() {
            parts.push(features.join("+"));
        }
        parts.join(", ")
    }
}

/// The features the physical device supports (before anything is enabled).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FeatureSupport {
    pub shader_int16: bool,
    pub shader_int64: bool,
    pub storage16: bool,
    pub uniform_storage16: bool,
    pub storage8: bool,
    pub uniform_storage8: bool,
    pub fp16: bool,
    pub int8: bool,
    pub timeline_semaphore: bool,
    pub subgroup_extended_types: bool,
    pub scalar_block_layout: bool,
    pub vulkan_memory_model: bool,
    pub coop_matrix: bool,
    pub int_dot: bool,
    pub subgroup_size_control: bool,
    pub compute_full_subgroups: bool,
    pub maintenance4: bool,
}

/// Result of probing a physical device.
pub(crate) struct Probe {
    pub caps: VkCaps,
    pub features: FeatureSupport,
    /// Device extensions to enable (all names are in `caps.extensions`).
    pub enable_extensions: Vec<&'static CStr>,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    /// Why the layer cannot use this device (None: usable).
    pub blocker: Option<String>,
}

/// Convert a fixed-size, NUL-terminated C string field.
pub(super) fn c_chars_to_string(chars: &[c_char]) -> String {
    let bytes: Vec<u8> = chars.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn bool32(v: vk::Bool32) -> bool {
    v != vk::FALSE
}

/// Heuristic memory bandwidth of well-known devices (GB/s, rounded vendor specifications). Not measured.
pub fn bandwidth_hint(vendor: Vendor, name: &str, device_type: VkDeviceType) -> Option<f32> {
    let name = name.to_ascii_lowercase();
    let table: &[(&str, f32)] = match vendor {
        Vendor::Nvidia => &[
            ("rtx 5090", 1792.0),
            ("rtx 5080", 960.0),
            ("rtx 5070 ti", 896.0),
            ("rtx 4090", 1008.0),
            ("rtx 4080", 717.0),
            ("rtx 4070 ti", 504.0),
            ("rtx 4070", 504.0),
            ("rtx 4060 ti", 288.0),
            ("rtx 3090", 936.0),
            ("rtx 3080", 760.0),
            ("rtx 3070", 448.0),
            ("rtx 3060", 360.0),
            ("rtx 2080", 448.0),
            ("rtx 2070", 448.0),
            ("rtx 2060", 336.0),
            ("gtx 1080", 320.0),
            ("gtx 1070", 256.0),
        ],
        Vendor::Amd => &[
            ("7900 xtx", 960.0),
            ("7900 xt", 800.0),
            ("7800 xt", 624.0),
            ("7700 xt", 432.0),
            ("6900 xt", 512.0),
            ("6800 xt", 512.0),
            ("6800", 512.0),
            ("6700 xt", 384.0),
            ("mi300", 5300.0),
            ("mi250", 3277.0),
            ("mi210", 1638.0),
        ],
        Vendor::Intel => &[
            ("a770", 560.0),
            ("a750", 512.0),
            ("a580", 512.0),
            ("a380", 186.0),
            ("b580", 456.0),
            ("b570", 380.0),
        ],
        _ => &[],
    };
    if let Some((_, gbps)) = table.iter().find(|(pat, _)| name.contains(pat)) {
        return Some(*gbps);
    }
    // integrated parts share system memory: dual-channel DDR4-3200 / LPDDR5-6400 class, a guess that is only useful as an order of magnitude
    match (device_type, vendor) {
        (VkDeviceType::Integrated, Vendor::Intel | Vendor::Amd) => Some(60.0),
        (VkDeviceType::Integrated, Vendor::Apple) => None,
        _ => None,
    }
}

/// The device-local heaps of `memory`: `(total bytes, unified)`.
pub fn analyse_memory(memory: &vk::PhysicalDeviceMemoryProperties, device_type: VkDeviceType) -> (u64, bool) {
    let types = &memory.memory_types[..memory.memory_type_count as usize];
    let heaps = &memory.memory_heaps[..memory.memory_heap_count as usize];
    let vram = heaps
        .iter()
        .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size)
        .sum();
    let has_pure_device_local = types.iter().any(|t| {
        t.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            && !t.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    });
    let unified = device_type == VkDeviceType::Integrated || !has_pure_device_local;
    (vram, unified)
}

/// Query everything about `pd` without creating a logical device. Needs an instance of at least Vulkan 1.1 for the `2` query functions;
/// a device below 1.2 is reported with `blocker` set (the feature queries of 1.2 are not issued).
pub(crate) fn probe(vki: &VkInstance, pd: vk::PhysicalDevice) -> VkResult<Probe> {
    let instance = &vki.instance;
    // SAFETY: `pd` was enumerated from `instance`; all structs live on this stack frame during the calls.
    unsafe {
        let props = instance.get_physical_device_properties(pd);
        let memory = instance.get_physical_device_memory_properties(pd);
        let ext_props = instance
            .enumerate_device_extension_properties(pd)
            .map_err(vk_call("vkEnumerateDeviceExtensionProperties"))?;
        let extensions: Vec<String> = ext_props.iter().map(|e| c_chars_to_string(&e.extension_name)).collect();
        let ext_set: HashSet<&str> = extensions.iter().map(String::as_str).collect();
        let has = |name: &CStr| name.to_str().is_ok_and(|n| ext_set.contains(n));

        let device_type = VkDeviceType::from_vk(props.device_type);
        let vendor = Vendor::from_id(props.vendor_id);
        let device_name = c_chars_to_string(&props.device_name);
        let api = props.api_version.min(vki.api_version);
        let limits = &props.limits;
        let (vram_bytes, unified_memory) = analyse_memory(&memory, device_type);

        let caps = VkCaps {
            device_name: device_name.clone(),
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            vendor,
            device_type,
            api_version: api,
            driver_version: props.driver_version,
            driver_name: String::new(),
            driver_info: String::new(),
            pipeline_cache_uuid: props.pipeline_cache_uuid,
            device_uuid: [0; 16],
            driver_uuid: [0; 16],
            vram_bytes,
            unified_memory,
            bandwidth_hint_gbps: bandwidth_hint(vendor, &device_name, device_type),
            max_memory_allocation_size: 0,
            max_buffer_size: 0,
            max_storage_buffer_range: u64::from(limits.max_storage_buffer_range),
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment.max(1),
            non_coherent_atom_size: limits.non_coherent_atom_size.max(1),
            max_memory_allocation_count: limits.max_memory_allocation_count,
            max_shared_memory: limits.max_compute_shared_memory_size,
            max_workgroup_invocations: limits.max_compute_work_group_invocations,
            max_workgroup_size: limits.max_compute_work_group_size,
            max_workgroup_count: limits.max_compute_work_group_count,
            max_push_constants: limits.max_push_constants_size,
            max_per_stage_storage_buffers: limits.max_per_stage_descriptor_storage_buffers,
            max_push_descriptors: 0,
            subgroup_size: 0,
            min_subgroup_size: 0,
            max_subgroup_size: 0,
            subgroup_ops: SubgroupOps::default(),
            subgroup_size_control: false,
            compute_full_subgroups: false,
            fp16: false,
            int8: false,
            int16: false,
            int64: false,
            storage16: false,
            storage8: false,
            subgroup_extended_types: false,
            scalar_block_layout: false,
            vulkan_memory_model: false,
            int_dot: IntDotCaps::default(),
            coop_matrix_supported: false,
            coop_matrix: Vec::new(),
            push_descriptor: false,
            external_memory_host: false,
            min_imported_host_pointer_alignment: 0,
            memory_budget: false,
            maintenance4: false,
            extensions: extensions.clone(),
        };

        let mut probe = Probe {
            caps,
            features: FeatureSupport::default(),
            enable_extensions: Vec::new(),
            memory,
            blocker: None,
        };
        if api < vk::API_VERSION_1_2 {
            probe.blocker = Some(format!(
                "{device_name} offers Vulkan {}.{} only (1.2 is required for timeline semaphores and 8/16-bit storage)",
                vk::api_version_major(api),
                vk::api_version_minor(api)
            ));
            return Ok(probe);
        }
        let v13 = api >= vk::API_VERSION_1_3;
        let want_push = has(ash::khr::push_descriptor::NAME);
        let want_coop = has(ash::khr::cooperative_matrix::NAME);
        let want_host = has(ash::ext::external_memory_host::NAME);
        let want_budget = has(ash::ext::memory_budget::NAME);
        let want_dot = !v13 && has(ash::khr::shader_integer_dot_product::NAME);
        let want_ssc = !v13 && has(ash::ext::subgroup_size_control::NAME);
        let want_m4 = !v13 && has(ash::khr::maintenance4::NAME);

        // ---- properties ----
        let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
        let mut ssc_p = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut dot_p = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
        let mut m4_p = vk::PhysicalDeviceMaintenance4Properties::default();
        let mut push_p = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
        let mut coop_p = vk::PhysicalDeviceCooperativeMatrixPropertiesKHR::default();
        let mut host_p = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        {
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p11).push_next(&mut p12);
            if v13 {
                p2 = p2.push_next(&mut p13);
            } else {
                if want_ssc {
                    p2 = p2.push_next(&mut ssc_p);
                }
                if want_dot {
                    p2 = p2.push_next(&mut dot_p);
                }
                if want_m4 {
                    p2 = p2.push_next(&mut m4_p);
                }
            }
            if want_push {
                p2 = p2.push_next(&mut push_p);
            }
            if want_coop {
                p2 = p2.push_next(&mut coop_p);
            }
            if want_host {
                p2 = p2.push_next(&mut host_p);
            }
            instance.get_physical_device_properties2(pd, &mut p2);
        }

        // ---- features ----
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut ssc_f = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        let mut dot_f = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut m4_f = vk::PhysicalDeviceMaintenance4Features::default();
        let mut coop_f = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let core_features;
        {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f11).push_next(&mut f12);
            if v13 {
                f2 = f2.push_next(&mut f13);
            } else {
                if want_ssc {
                    f2 = f2.push_next(&mut ssc_f);
                }
                if want_dot {
                    f2 = f2.push_next(&mut dot_f);
                }
                if want_m4 {
                    f2 = f2.push_next(&mut m4_f);
                }
            }
            if want_coop {
                f2 = f2.push_next(&mut coop_f);
            }
            instance.get_physical_device_features2(pd, &mut f2);
            core_features = f2.features;
        }

        let features = FeatureSupport {
            shader_int16: bool32(core_features.shader_int16),
            shader_int64: bool32(core_features.shader_int64),
            storage16: bool32(f11.storage_buffer16_bit_access),
            uniform_storage16: bool32(f11.uniform_and_storage_buffer16_bit_access),
            storage8: bool32(f12.storage_buffer8_bit_access),
            uniform_storage8: bool32(f12.uniform_and_storage_buffer8_bit_access),
            fp16: bool32(f12.shader_float16),
            int8: bool32(f12.shader_int8),
            timeline_semaphore: bool32(f12.timeline_semaphore),
            subgroup_extended_types: bool32(f12.shader_subgroup_extended_types),
            scalar_block_layout: bool32(f12.scalar_block_layout),
            vulkan_memory_model: bool32(f12.vulkan_memory_model),
            coop_matrix: want_coop && bool32(coop_f.cooperative_matrix),
            int_dot: if v13 {
                bool32(f13.shader_integer_dot_product)
            } else {
                want_dot && bool32(dot_f.shader_integer_dot_product)
            },
            subgroup_size_control: if v13 {
                bool32(f13.subgroup_size_control)
            } else {
                want_ssc && bool32(ssc_f.subgroup_size_control)
            },
            compute_full_subgroups: if v13 {
                bool32(f13.compute_full_subgroups)
            } else {
                want_ssc && bool32(ssc_f.compute_full_subgroups)
            },
            maintenance4: if v13 {
                bool32(f13.maintenance4)
            } else {
                want_m4 && bool32(m4_f.maintenance4)
            },
        };
        if !features.timeline_semaphore {
            probe.blocker = Some(format!("{device_name} does not support timeline semaphores"));
        }

        // ---- fill the capabilities ----
        let caps = &mut probe.caps;
        caps.driver_name = c_chars_to_string(&p12.driver_name);
        caps.driver_info = c_chars_to_string(&p12.driver_info);
        caps.device_uuid = p11.device_uuid;
        caps.driver_uuid = p11.driver_uuid;
        caps.max_memory_allocation_size = if p11.max_memory_allocation_size != 0 {
            p11.max_memory_allocation_size
        } else {
            1 << 30
        };
        caps.max_buffer_size = if v13 && p13.max_buffer_size != 0 {
            p13.max_buffer_size
        } else if features.maintenance4 && m4_p.max_buffer_size != 0 {
            m4_p.max_buffer_size
        } else {
            caps.max_memory_allocation_size
        };
        caps.subgroup_size = p11.subgroup_size.max(1);
        caps.subgroup_ops = SubgroupOps::from_flags(p11.subgroup_supported_operations, p11.subgroup_supported_stages);
        let (min_sg, max_sg, ssc_stages) = if v13 {
            (p13.min_subgroup_size, p13.max_subgroup_size, p13.required_subgroup_size_stages)
        } else {
            (
                ssc_p.min_subgroup_size,
                ssc_p.max_subgroup_size,
                ssc_p.required_subgroup_size_stages,
            )
        };
        caps.min_subgroup_size = if min_sg == 0 { caps.subgroup_size } else { min_sg };
        caps.max_subgroup_size = if max_sg == 0 { caps.subgroup_size } else { max_sg };
        caps.subgroup_size_control = features.subgroup_size_control && ssc_stages.contains(vk::ShaderStageFlags::COMPUTE);
        caps.compute_full_subgroups = features.compute_full_subgroups;
        caps.fp16 = features.fp16;
        caps.int8 = features.int8;
        caps.int16 = features.shader_int16;
        caps.int64 = features.shader_int64;
        caps.storage16 = features.storage16;
        caps.storage8 = features.storage8;
        caps.subgroup_extended_types = features.subgroup_extended_types;
        caps.scalar_block_layout = features.scalar_block_layout;
        caps.vulkan_memory_model = features.vulkan_memory_model;
        caps.maintenance4 = features.maintenance4;

        // integer dot product: the property block moved into Vulkan13Properties
        caps.int_dot = IntDotCaps {
            enabled: features.int_dot,
            ..IntDotCaps::default()
        };
        if features.int_dot {
            let b = bool32;
            caps.int_dot = if v13 {
                IntDotCaps {
                    enabled: true,
                    packed_4x8_signed: b(p13.integer_dot_product4x8_bit_packed_signed_accelerated),
                    packed_4x8_unsigned: b(p13.integer_dot_product4x8_bit_packed_unsigned_accelerated),
                    packed_4x8_mixed: b(p13.integer_dot_product4x8_bit_packed_mixed_signedness_accelerated),
                    vec_8bit_signed: b(p13.integer_dot_product8_bit_signed_accelerated),
                    vec_8bit_unsigned: b(p13.integer_dot_product8_bit_unsigned_accelerated),
                    vec_8bit_mixed: b(p13.integer_dot_product8_bit_mixed_signedness_accelerated),
                    acc_sat_packed_4x8_signed: b(p13.integer_dot_product_accumulating_saturating4x8_bit_packed_signed_accelerated),
                }
            } else {
                IntDotCaps {
                    enabled: true,
                    packed_4x8_signed: b(dot_p.integer_dot_product4x8_bit_packed_signed_accelerated),
                    packed_4x8_unsigned: b(dot_p.integer_dot_product4x8_bit_packed_unsigned_accelerated),
                    packed_4x8_mixed: b(dot_p.integer_dot_product4x8_bit_packed_mixed_signedness_accelerated),
                    vec_8bit_signed: b(dot_p.integer_dot_product8_bit_signed_accelerated),
                    vec_8bit_unsigned: b(dot_p.integer_dot_product8_bit_unsigned_accelerated),
                    vec_8bit_mixed: b(dot_p.integer_dot_product8_bit_mixed_signedness_accelerated),
                    acc_sat_packed_4x8_signed: b(dot_p.integer_dot_product_accumulating_saturating4x8_bit_packed_signed_accelerated),
                }
            };
        }

        // cooperative matrix combinations (only the stage-compute ones matter: the property says which stages)
        if features.coop_matrix && coop_p.cooperative_matrix_supported_stages.contains(vk::ShaderStageFlags::COMPUTE) {
            let coop = ash::khr::cooperative_matrix::Instance::new(&vki.entry, instance);
            match coop.get_physical_device_cooperative_matrix_properties(pd) {
                Ok(list) => {
                    caps.coop_matrix = list.iter().map(CoopMatrixShape::from_vk).collect();
                    caps.coop_matrix_supported = !caps.coop_matrix.is_empty();
                }
                Err(_) => caps.coop_matrix_supported = false,
            }
        }

        caps.push_descriptor = want_push;
        caps.max_push_descriptors = if want_push { push_p.max_push_descriptors } else { 0 };
        caps.external_memory_host = want_host && host_p.min_imported_host_pointer_alignment != 0;
        caps.min_imported_host_pointer_alignment = if caps.external_memory_host {
            host_p.min_imported_host_pointer_alignment
        } else {
            0
        };
        caps.memory_budget = want_budget;

        // ---- extensions to enable ----
        let ext = &mut probe.enable_extensions;
        if want_push {
            ext.push(ash::khr::push_descriptor::NAME);
        }
        if features.coop_matrix {
            ext.push(ash::khr::cooperative_matrix::NAME);
        }
        if want_host {
            ext.push(ash::ext::external_memory_host::NAME);
        }
        if want_budget {
            ext.push(ash::ext::memory_budget::NAME);
        }
        if want_dot {
            ext.push(ash::khr::shader_integer_dot_product::NAME);
        }
        if want_ssc {
            ext.push(ash::ext::subgroup_size_control::NAME);
        }
        if want_m4 {
            ext.push(ash::khr::maintenance4::NAME);
        }
        // MoltenVK & co.: when the device is a portability subset, VK_KHR_portability_subset must be enabled
        if has(ash::khr::portability_subset::NAME) {
            ext.push(ash::khr::portability_subset::NAME);
        }
        Ok(probe)
    }
}

/// Fail with a readable error when the device cannot be used.
pub(crate) fn ensure_usable(probe: &Probe) -> VkResult<()> {
    match &probe.blocker {
        Some(reason) => Err(VkError::Unsupported(reason.clone())),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_caps() -> VkCaps {
        let shape = |m, n, k, a, b, c, r, scope| CoopMatrixShape {
            m,
            n,
            k,
            a,
            b,
            c,
            result: r,
            saturating: false,
            scope,
        };
        let f16 = ComponentType::F16;
        let f32t = ComponentType::F32;
        VkCaps {
            device_name: "Test GPU".into(),
            vendor_id: 0x10DE,
            device_id: 1,
            vendor: Vendor::Nvidia,
            device_type: VkDeviceType::Discrete,
            api_version: vk::API_VERSION_1_3,
            driver_version: 0,
            driver_name: "NVIDIA".into(),
            driver_info: "570.1".into(),
            pipeline_cache_uuid: [0; 16],
            device_uuid: [0; 16],
            driver_uuid: [0; 16],
            vram_bytes: 8 << 30,
            unified_memory: false,
            bandwidth_hint_gbps: None,
            max_memory_allocation_size: 1 << 30,
            max_buffer_size: 1 << 30,
            max_storage_buffer_range: 1 << 30,
            min_storage_buffer_offset_alignment: 16,
            non_coherent_atom_size: 64,
            max_memory_allocation_count: 4096,
            max_shared_memory: 49152,
            max_workgroup_invocations: 1024,
            max_workgroup_size: [1024, 1024, 64],
            max_workgroup_count: [65535; 3],
            max_push_constants: 128,
            max_per_stage_storage_buffers: 16,
            max_push_descriptors: 32,
            subgroup_size: 32,
            min_subgroup_size: 32,
            max_subgroup_size: 32,
            subgroup_ops: SubgroupOps::default(),
            subgroup_size_control: false,
            compute_full_subgroups: false,
            fp16: true,
            int8: true,
            int16: true,
            int64: true,
            storage16: true,
            storage8: true,
            subgroup_extended_types: true,
            scalar_block_layout: true,
            vulkan_memory_model: true,
            int_dot: IntDotCaps {
                enabled: true,
                packed_4x8_signed: true,
                ..IntDotCaps::default()
            },
            coop_matrix_supported: true,
            coop_matrix: vec![
                shape(16, 8, 16, f16, f16, f32t, f32t, CoopScope::Subgroup),
                shape(16, 16, 16, f16, f16, f32t, f32t, CoopScope::Subgroup),
                shape(16, 16, 16, f16, f16, f32t, f32t, CoopScope::Workgroup),
                shape(
                    16,
                    16,
                    32,
                    ComponentType::I8,
                    ComponentType::I8,
                    ComponentType::I32,
                    ComponentType::I32,
                    CoopScope::Subgroup,
                ),
            ],
            push_descriptor: true,
            external_memory_host: false,
            min_imported_host_pointer_alignment: 0,
            memory_budget: true,
            maintenance4: true,
            extensions: vec!["VK_KHR_cooperative_matrix".into()],
        }
    }

    #[test]
    fn vendor_ids() {
        assert_eq!(Vendor::from_id(0x10DE), Vendor::Nvidia);
        assert_eq!(Vendor::from_id(0x1002), Vendor::Amd);
        assert_eq!(Vendor::from_id(0x8086), Vendor::Intel);
        assert_eq!(Vendor::from_id(0x10005), Vendor::Mesa);
        assert_eq!(Vendor::from_id(0xBEEF), Vendor::Other(0xBEEF));
        assert_eq!(Vendor::Amd.name(), "AMD");
    }

    #[test]
    fn component_and_scope_mapping() {
        assert_eq!(ComponentType::from_raw(0), ComponentType::F16);
        assert_eq!(ComponentType::from_raw(3), ComponentType::I8);
        assert_eq!(ComponentType::from_raw(10), ComponentType::U64);
        assert_eq!(ComponentType::from_raw(1_000_141_000), ComponentType::Bf16);
        assert_eq!(ComponentType::from_raw(1_000_491_002), ComponentType::Other(1_000_491_002));
        assert_eq!(CoopScope::from_raw(3), CoopScope::Subgroup);
        assert_eq!(CoopScope::from_raw(2), CoopScope::Workgroup);
    }

    #[test]
    fn coop_matrix_lookup() {
        let caps = sample_caps();
        let (f16, f32t) = (ComponentType::F16, ComponentType::F32);
        // the workgroup-scope entry is ignored
        assert_eq!(caps.coop_matrix_for(f16, f16, f32t, f32t).len(), 2);
        let best = caps.best_coop_matrix(f16, f16, f32t, f32t).unwrap();
        assert_eq!((best.m, best.n, best.k), (16, 16, 16));
        let int8 = caps
            .best_coop_matrix(ComponentType::I8, ComponentType::I8, ComponentType::I32, ComponentType::I32)
            .unwrap();
        assert_eq!(int8.k, 32);
        assert!(caps
            .best_coop_matrix(ComponentType::Bf16, ComponentType::Bf16, f32t, f32t)
            .is_none());
    }

    #[test]
    fn subgroup_ops_need_the_compute_stage() {
        let flags = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC | vk::SubgroupFeatureFlags::SHUFFLE;
        let ops = SubgroupOps::from_flags(flags, vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT);
        assert!(ops.basic && ops.arithmetic && ops.shuffle && !ops.ballot && !ops.quad);
        assert_eq!(
            SubgroupOps::from_flags(flags, vk::ShaderStageFlags::FRAGMENT),
            SubgroupOps::default()
        );
    }

    #[test]
    fn describe_mentions_the_main_capabilities() {
        let text = sample_caps().describe();
        assert!(text.contains("Vulkan 1.3.0"), "{text}");
        assert!(text.contains("subgroup 32"), "{text}");
        assert!(text.contains("coop-matrix"), "{text}");
        assert!(text.contains("int-dot"), "{text}");
    }

    #[test]
    fn memory_analysis() {
        // discrete card: heap 0 = 8 GiB VRAM, heap 1 = system RAM, heap 2 = 256 MiB BAR
        let mut mem = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: 3,
            memory_type_count: 3,
            ..Default::default()
        };
        mem.memory_heaps[0] = vk::MemoryHeap {
            size: 8 << 30,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        mem.memory_heaps[1] = vk::MemoryHeap {
            size: 32 << 30,
            flags: vk::MemoryHeapFlags::empty(),
        };
        mem.memory_heaps[2] = vk::MemoryHeap {
            size: 256 << 20,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        mem.memory_types[0] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            heap_index: 0,
        };
        mem.memory_types[1] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            heap_index: 1,
        };
        mem.memory_types[2] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            heap_index: 2,
        };
        assert_eq!(analyse_memory(&mem, VkDeviceType::Discrete), ((8 << 30) + (256 << 20), false));

        // integrated GPU: one device-local heap, every type host visible
        let mut igp = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: 1,
            memory_type_count: 1,
            ..Default::default()
        };
        igp.memory_heaps[0] = vk::MemoryHeap {
            size: 16 << 30,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        igp.memory_types[0] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            heap_index: 0,
        };
        assert_eq!(analyse_memory(&igp, VkDeviceType::Integrated), (16 << 30, true));
        // a "discrete" device without any pure device-local type is unified as well
        assert!(analyse_memory(&igp, VkDeviceType::Discrete).1);
    }

    #[test]
    fn bandwidth_table() {
        assert_eq!(
            bandwidth_hint(Vendor::Nvidia, "NVIDIA GeForce RTX 4090", VkDeviceType::Discrete),
            Some(1008.0)
        );
        assert_eq!(
            bandwidth_hint(Vendor::Amd, "AMD Radeon RX 7900 XTX", VkDeviceType::Discrete),
            Some(960.0)
        );
        assert_eq!(
            bandwidth_hint(Vendor::Intel, "Intel(R) Iris(R) Xe Graphics", VkDeviceType::Integrated),
            Some(60.0)
        );
        assert_eq!(bandwidth_hint(Vendor::Nvidia, "Some Future Card", VkDeviceType::Discrete), None);
    }

    #[test]
    fn c_string_fields() {
        let mut field = [0 as c_char; 8];
        for (i, b) in b"abc".iter().enumerate() {
            field[i] = *b as c_char;
        }
        assert_eq!(c_chars_to_string(&field), "abc");
        assert_eq!(c_chars_to_string(&[0; 4]), "");
    }
}
