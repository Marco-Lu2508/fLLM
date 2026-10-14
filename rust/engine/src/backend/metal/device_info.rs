//! Device inventory of the native Metal backend (objc2-metal; compiled on Apple targets only).
//!
//! `MTLCopyAllDevices` lists the GPUs (one on Apple silicon Macs, several on Intel Macs with discrete or external
//! GPUs). Each becomes a [`MetalDevice`] that carries the raw `MTLDevice` and a [`MetalDeviceInfo`] with everything
//! the other modules need to choose code paths: memory limits, thread limits, GPU families, the Apple chip generation
//! recognised from the name, and the shader language version the running OS compiles.
//!
//! Level of verification: L0 (compiles for `aarch64-apple-darwin`; never run on a Mac).

pub mod config;
pub mod error;
pub mod platform;

pub use config::MetalConfig;
pub use error::MetalError;
pub use platform::{AppleChip, ChipLine, ChipTier, GpuVendor};

use crate::backend::BackendError;
use crate::devices::{DeviceInfo, DeviceType};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSProcessInfo;
use objc2_metal::{
    MTLArgumentBuffersTier, MTLCopyAllDevices, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLDeviceLocation, MTLGPUFamily,
};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// Where the GPU sits (`MTLDeviceLocation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceLocation {
    BuiltIn,
    Slot,
    External,
    Unspecified,
}

/// Static facts about one Metal device.
#[derive(Clone, Debug)]
pub struct MetalDeviceInfo {
    /// Position among the Metal devices (0 = the system default device).
    pub ordinal: usize,
    pub name: String,
    pub registry_id: u64,
    pub vendor: GpuVendor,
    /// The Apple SoC recognised from the name (`Apple M2 Pro`), if it is one.
    pub chip: Option<AppleChip>,
    pub location: DeviceLocation,
    /// CPU and GPU share one memory (all Apple silicon). Buffers with shared storage then cost no copy.
    pub has_unified_memory: bool,
    pub is_low_power: bool,
    pub is_headless: bool,
    pub is_removable: bool,
    /// `recommendedMaxWorkingSetSize`: the memory the GPU can use without a performance cliff.
    pub recommended_max_working_set_bytes: u64,
    /// `maxBufferLength`: largest single buffer.
    pub max_buffer_length_bytes: u64,
    /// `maxThreadgroupMemoryLength`.
    pub max_threadgroup_memory_bytes: u64,
    /// `maxThreadsPerThreadgroup` (width, height, depth).
    pub max_threads_per_threadgroup: [usize; 3],
    /// Installed RAM of the host (`NSProcessInfo.physicalMemory`).
    pub physical_memory_bytes: u64,
    /// Highest `MTLGPUFamilyAppleN` the device supports (`None` on Intel / AMD GPUs).
    pub apple_family: Option<u8>,
    /// Supports `MTLGPUFamilyMac2` (the Mac family of the current GPU generation).
    pub mac_family2: bool,
    pub metal3: bool,
    pub metal4: bool,
    /// Argument buffers tier 2 (the GPU-address tables of [`super::buffers::AddressTable`] need it).
    pub argument_buffers_tier2: bool,
    /// `simdgroup_matrix` operations: Apple7 and newer, or Mac2 (Apple's feature set tables; from memory, unverified).
    pub simdgroup_matrix: bool,
    /// SIMD-group reductions (`simd_sum`, `simd_max`, …): same families as the matrix operations.
    pub simdgroup_reduction: bool,
    /// `bfloat` in shaders: Metal 3 family GPU and a Metal Shading Language of at least 3.1.
    pub bfloat: bool,
    /// Expected SIMD width (32 Apple, 64 AMD, 16 Intel); the exact value is `Pipeline::simd_width`.
    pub simd_width_hint: usize,
    /// `(major, minor, patch)` of the operating system.
    pub os_version: (i64, i64, i64),
    /// Language version the source compiler is asked for (`FLLM_METAL_MSL` or the newest the OS supports, at most 3.2).
    pub msl_version: (u16, u16),
    /// Host page size (16 KiB on Apple silicon): alignment unit of no-copy buffers.
    pub page_size: usize,
}

impl MetalDeviceInfo {
    /// One line for logs: `Apple M2 Pro (Apple8, Metal3, unified, 21.3 GiB working set)`.
    pub fn summary(&self) -> String {
        let mut families = Vec::new();
        if let Some(n) = self.apple_family {
            families.push(format!("Apple{n}"));
        }
        if self.mac_family2 {
            families.push("Mac2".to_owned());
        }
        if self.metal4 {
            families.push("Metal4".to_owned());
        } else if self.metal3 {
            families.push("Metal3".to_owned());
        }
        let memory = if self.has_unified_memory {
            "unified memory"
        } else {
            "discrete memory"
        };
        format!(
            "{} ({}, {memory}, {:.1} GiB working set, MSL {}.{})",
            self.name,
            families.join(" "),
            self.recommended_max_working_set_bytes as f64 / (1u64 << 30) as f64,
            self.msl_version.0,
            self.msl_version.1
        )
    }

    /// Preprocessor symbols that every compiled shader unit sees, so kernels can pick code paths per device
    /// (`#if FLLM_HAS_SIMDGROUP_MATRIX`). All of them are always defined, to `0` or `1` unless stated.
    pub fn shader_defines(&self) -> Vec<(String, String)> {
        let flag = |value: bool| if value { "1" } else { "0" }.to_owned();
        vec![
            ("FLLM_METAL".to_owned(), "1".to_owned()),
            (
                "FLLM_APPLE_GPU_FAMILY".to_owned(),
                self.apple_family.unwrap_or(0).to_string(),
            ),
            (
                "FLLM_HAS_SIMDGROUP_MATRIX".to_owned(),
                flag(self.simdgroup_matrix),
            ),
            (
                "FLLM_HAS_SIMDGROUP_REDUCTION".to_owned(),
                flag(self.simdgroup_reduction),
            ),
            ("FLLM_HAS_BFLOAT".to_owned(), flag(self.bfloat)),
            (
                "FLLM_UNIFIED_MEMORY".to_owned(),
                flag(self.has_unified_memory),
            ),
            (
                "FLLM_SIMD_WIDTH_HINT".to_owned(),
                self.simd_width_hint.to_string(),
            ),
            (
                "FLLM_MAX_THREADGROUP_MEMORY".to_owned(),
                self.max_threadgroup_memory_bytes.to_string(),
            ),
        ]
    }

    /// The entry of `fllm-rs devices` for this GPU.
    pub fn to_device_info(&self, index: usize) -> DeviceInfo {
        let dev_type = if self.has_unified_memory || self.is_low_power {
            DeviceType::IntegratedGpu
        } else {
            DeviceType::Gpu
        };
        let bandwidth = self
            .chip
            .and_then(|chip| chip.nominal_bandwidth_gbs())
            .map(|gbs| format!(", ~{gbs:.0} GB/s"))
            .unwrap_or_default();
        let total = if self.has_unified_memory {
            self.physical_memory_bytes
        } else {
            self.recommended_max_working_set_bytes
        };
        DeviceInfo {
            index,
            name: self.name.clone(),
            description: format!("Metal (native), {}{bandwidth}", self.summary()),
            dev_type,
            free_memory_bytes: self.recommended_max_working_set_bytes,
            total_memory_bytes: total,
        }
    }
}

/// A Metal GPU: the raw device plus its static facts. Cheap to clone.
#[derive(Clone)]
pub struct MetalDevice {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    info: Arc<MetalDeviceInfo>,
}

impl fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalDevice({})", self.info.summary())
    }
}

impl MetalDevice {
    /// The `MTLDevice` for direct API calls.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }

    pub fn info(&self) -> &MetalDeviceInfo {
        &self.info
    }

    /// Bytes the driver currently has allocated for this device (all processes' views of it: `currentAllocatedSize`).
    pub fn current_allocated_bytes(&self) -> u64 {
        self.raw.currentAllocatedSize() as u64
    }
}

extern "C" {
    fn getpagesize() -> std::ffi::c_int;
}

/// The host page size. 16384 on Apple silicon, 4096 on Intel Macs.
pub fn host_page_size() -> usize {
    // SAFETY: `getpagesize` has no preconditions.
    let size = unsafe { getpagesize() };
    match usize::try_from(size) {
        Ok(size) if size.is_power_of_two() => size,
        _ => 16384,
    }
}

fn highest_apple_family(raw: &ProtocolObject<dyn MTLDevice>) -> Option<u8> {
    let families = [
        (10, MTLGPUFamily::Apple10),
        (9, MTLGPUFamily::Apple9),
        (8, MTLGPUFamily::Apple8),
        (7, MTLGPUFamily::Apple7),
        (6, MTLGPUFamily::Apple6),
        (5, MTLGPUFamily::Apple5),
        (4, MTLGPUFamily::Apple4),
        (3, MTLGPUFamily::Apple3),
        (2, MTLGPUFamily::Apple2),
        (1, MTLGPUFamily::Apple1),
    ];
    families
        .into_iter()
        .find(|(_, family)| raw.supportsFamily(*family))
        .map(|(n, _)| n)
}

fn gather(
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    ordinal: usize,
    config: &MetalConfig,
) -> MetalDevice {
    let name = raw.name().to_string();
    let vendor = platform::guess_vendor(&name);
    let apple_family = highest_apple_family(&raw);
    let mac_family2 = raw.supportsFamily(MTLGPUFamily::Mac2);
    let metal3 = raw.supportsFamily(MTLGPUFamily::Metal3);
    let metal4 = raw.supportsFamily(MTLGPUFamily::Metal4);
    let location = {
        let location = raw.location();
        if location == MTLDeviceLocation::BuiltIn {
            DeviceLocation::BuiltIn
        } else if location == MTLDeviceLocation::Slot {
            DeviceLocation::Slot
        } else if location == MTLDeviceLocation::External {
            DeviceLocation::External
        } else {
            DeviceLocation::Unspecified
        }
    };
    let process = NSProcessInfo::processInfo();
    let version = process.operatingSystemVersion();
    let os = if cfg!(target_os = "ios") {
        platform::OsFamily::Ios
    } else {
        platform::OsFamily::MacOs
    };
    let msl_version = config
        .msl_version
        .unwrap_or_else(|| platform::msl_version_for_os(os, version.majorVersion as i64));
    let max_threads = raw.maxThreadsPerThreadgroup();
    let simd_matrix = apple_family.is_some_and(|n| n >= 7) || mac_family2;
    let simd_width_hint = match vendor {
        GpuVendor::Amd => 64,
        GpuVendor::Intel => 16,
        _ => 32,
    };
    let info = MetalDeviceInfo {
        ordinal,
        chip: platform::parse_apple_chip(&name),
        vendor,
        registry_id: raw.registryID(),
        location,
        has_unified_memory: raw.hasUnifiedMemory(),
        is_low_power: raw.isLowPower(),
        is_headless: raw.isHeadless(),
        is_removable: raw.isRemovable(),
        recommended_max_working_set_bytes: raw.recommendedMaxWorkingSetSize(),
        max_buffer_length_bytes: raw.maxBufferLength() as u64,
        max_threadgroup_memory_bytes: raw.maxThreadgroupMemoryLength() as u64,
        max_threads_per_threadgroup: [max_threads.width, max_threads.height, max_threads.depth],
        physical_memory_bytes: process.physicalMemory(),
        apple_family,
        mac_family2,
        metal3,
        metal4,
        argument_buffers_tier2: raw.argumentBuffersSupport() == MTLArgumentBuffersTier::Tier2,
        simdgroup_matrix: simd_matrix,
        simdgroup_reduction: simd_matrix,
        bfloat: metal3 && msl_version >= (3, 1),
        simd_width_hint,
        os_version: (
            version.majorVersion as i64,
            version.minorVersion as i64,
            version.patchVersion as i64,
        ),
        msl_version,
        page_size: host_page_size(),
        name,
    };
    MetalDevice {
        raw,
        info: Arc::new(info),
    }
}

static DEVICES: OnceLock<Vec<MetalDevice>> = OnceLock::new();

/// Global index of the first Metal device in the inventory of the last [`list`] call (see
/// [`platform::resolve_device_index`]); `usize::MAX` before any call.
static FIRST_INDEX: AtomicUsize = AtomicUsize::new(usize::MAX);

fn enumerate() -> Vec<MetalDevice> {
    autoreleasepool(|_| {
        let config = MetalConfig::from_env();
        let default_id = MTLCreateSystemDefaultDevice().map(|d| d.registryID());
        let mut raw: Vec<Retained<ProtocolObject<dyn MTLDevice>>> = MTLCopyAllDevices().to_vec();
        if raw.is_empty() {
            raw.extend(MTLCreateSystemDefaultDevice());
        }
        // the system default (main display / high-power GPU) first: `--device metal` means that one
        raw.sort_by_key(|device| Some(device.registryID()) != default_id);
        raw.into_iter()
            .enumerate()
            .map(|(ordinal, device)| gather(device, ordinal, &config))
            .collect()
    })
}

/// All Metal devices of this machine, default device first. Computed once per process.
pub fn all_devices() -> &'static [MetalDevice] {
    DEVICES.get_or_init(enumerate)
}

/// The Metal device selected by `--device metal:N` (see [`platform::resolve_device_index`] for the meaning of `N`).
pub fn open(index: usize) -> Result<MetalDevice, MetalError> {
    let devices = all_devices();
    if devices.is_empty() {
        return Err(MetalError::NoDevice(
            "the system reports no Metal device".to_owned(),
        ));
    }
    let hint = match FIRST_INDEX.load(Ordering::Relaxed) {
        usize::MAX => None,
        first => Some(first),
    };
    match platform::resolve_device_index(index, devices.len(), hint) {
        Some(ordinal) => Ok(devices[ordinal].clone()),
        None => Err(MetalError::NoDevice(format!(
            "device index {index} (found {}: {})",
            devices.len(),
            devices
                .iter()
                .map(|d| d.info.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

/// The Metal devices as inventory entries, numbered from `first_index`.
pub fn list(first_index: usize) -> Vec<DeviceInfo> {
    FIRST_INDEX.store(first_index, Ordering::Relaxed);
    all_devices()
        .iter()
        .enumerate()
        .map(|(i, device)| {
            let mut entry = device.info.to_device_info(first_index + i);
            // what is left of the working set right now
            entry.free_memory_bytes = device
                .info
                .recommended_max_working_set_bytes
                .saturating_sub(device.current_allocated_bytes());
            entry
        })
        .collect()
}

impl From<MetalError> for BackendError {
    fn from(error: MetalError) -> Self {
        match error {
            MetalError::NoDevice(message) => BackendError::DeviceNotFound(message),
            MetalError::OutOfMemory { requested, limit } => BackendError::OutOfMemory {
                requested,
                available: limit,
            },
            MetalError::Unsupported(message) => BackendError::UnsupportedOperation(message),
            MetalError::Buffer(message) => BackendError::BufferError(message),
            other => BackendError::DriverError(other.to_string()),
        }
    }
}
