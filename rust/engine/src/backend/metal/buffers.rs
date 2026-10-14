//! Buffers of the native Metal backend (objc2-metal; compiled on Apple targets only).
//!
//! All buffers use shared storage: on Apple silicon the CPU and the GPU see the same memory, so uploads are
//! `memcpy`s and downloads are reads, with no staging and no blit. Three ways to get a buffer:
//!
//! - [`MemoryManager::alloc`]: a fresh buffer, counted against the memory budget;
//! - [`MappedRegion`] / [`MappedRegistry`]: host memory that already exists (the memory-mapped GGUF file) wrapped
//!   without a copy through `newBufferWithBytesNoCopy`, split into page-aligned windows (see [`plan`]);
//! - [`Arena`]: a bump allocator over one big buffer for per-step scratch data.
//!
//! [`AddressTable`] is the argument-buffer facility: a buffer of 64-bit GPU addresses that a kernel indexes to reach
//! many weights (mixture-of-experts layers) with one binding.
//!
//! Level of verification: L0 (compiles for `aarch64-apple-darwin`; never run on a Mac).

pub mod plan;

use super::device_info::{MetalConfig, MetalDevice, MetalError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLAllocation, MTLBuffer, MTLDevice, MTLResource, MTLResourceOptions};
use plan::{plan_mapping, MappingPlan, MemoryBudget};
use std::any::Any;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

/// Alignment of sub-allocations ([`Arena`], address tables): large enough for `constant` address space bindings
/// on GPUs that require 256-byte offsets.
pub const SUBALLOCATION_ALIGN: usize = 256;

/// What a buffer is for; decides the storage and hazard-tracking options.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    /// Intermediate results written and read by kernels. Hazard-tracked.
    Activations,
    /// Model weights, written once by the CPU before any kernel runs, then only read. Untracked (no per-dispatch
    /// tracking cost); the contract is that nothing writes them while kernels run.
    Weights,
    /// The paged KV cache: written and read by kernels. Hazard-tracked.
    KvCache,
    /// CPU-written inputs of a step (token ids, positions, block tables). Write-combined CPU mapping.
    Staging,
}

impl BufferUsage {
    pub fn resource_options(self) -> MTLResourceOptions {
        match self {
            BufferUsage::Activations | BufferUsage::KvCache => {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeTracked
            }
            BufferUsage::Weights => {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeUntracked
            }
            BufferUsage::Staging => {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
                    | MTLResourceOptions::HazardTrackingModeTracked
            }
        }
    }
}

/// Where the memory of a buffer comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferKind {
    /// Allocated by Metal.
    Owned(BufferUsage),
    /// Host memory wrapped without a copy; read-only for the CPU side of this backend.
    Mapped,
}

/// One `MTLBuffer`. Shared through `Arc`; the Metal buffer is released (and its bytes returned to the budget) when
/// the last reference goes.
pub struct MetalBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    id: u64,
    len: usize,
    kind: BufferKind,
    budget: Option<Arc<MemoryBudget>>,
    reserved: u64,
    /// Keeps the host memory of a [`BufferKind::Mapped`] buffer alive.
    _owner: Option<Arc<dyn Any + Send + Sync>>,
}

// SAFETY: Apple documents `MTLDevice`, `MTLBuffer` and the other resource and state objects as thread safe; only
// command buffers and encoders must not be used from several threads at once, and those are not part of this
// type. The contents of a shared buffer are plain memory; ordering between CPU and GPU accesses is the caller's
// business (synchronise before touching a buffer the GPU may still use).
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl std::fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MetalBuffer#{} {{ len: {}, kind: {:?} }}",
            self.id, self.len, self.kind
        )
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if let Some(budget) = &self.budget {
            budget.release(self.reserved);
        }
    }
}

impl MetalBuffer {
    /// The `MTLBuffer` for direct API calls.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    /// The buffer as a `MTLResource` (`useResource`, barriers).
    pub fn as_resource(&self) -> &ProtocolObject<dyn MTLResource> {
        ProtocolObject::from_ref(&*self.raw)
    }

    /// The buffer as a `MTLAllocation` (residency sets).
    pub fn as_allocation(&self) -> &ProtocolObject<dyn MTLAllocation> {
        ProtocolObject::from_ref(&*self.raw)
    }

    /// Process-unique id, the key of the hazard tracker.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Logical length in bytes (what was asked for; the Metal buffer may be larger).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn kind(&self) -> BufferKind {
        self.kind
    }

    /// CPU address of the first byte (shared storage is always CPU-accessible).
    pub fn contents_ptr(&self) -> *mut u8 {
        self.raw.contents().as_ptr().cast()
    }

    /// GPU virtual address of the first byte, for [`AddressTable`]s and `ulong` kernel parameters.
    pub fn gpu_address(&self) -> u64 {
        self.raw.gpuAddress()
    }

    /// Name shown in GPU captures and the debugger.
    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)));
    }

    fn check_range(&self, offset: usize, len: usize) -> Result<(), MetalError> {
        if offset.checked_add(len).is_some_and(|end| end <= self.len) {
            Ok(())
        } else {
            Err(MetalError::Buffer(format!(
                "range {offset}+{len} outside buffer #{} of {} bytes",
                self.id, self.len
            )))
        }
    }

    /// Copy `data` to `offset`. The GPU must not be using this range (call after `synchronize`, before the next
    /// dispatch that reads it). Mapped (read-only host) buffers refuse writes.
    pub fn write(&self, offset: usize, data: &[u8]) -> Result<(), MetalError> {
        if self.kind == BufferKind::Mapped {
            return Err(MetalError::Buffer(
                "buffer wraps read-only host memory".to_owned(),
            ));
        }
        self.check_range(offset, data.len())?;
        // SAFETY: the range is inside the buffer (checked), the source is a valid slice, and the two cannot overlap
        // because a Rust slice cannot alias memory the allocator just handed to Metal.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.contents_ptr().add(offset),
                data.len(),
            )
        };
        Ok(())
    }

    /// Copy bytes from `offset` into `out`. The GPU must have finished writing the range.
    pub fn read(&self, offset: usize, out: &mut [u8]) -> Result<(), MetalError> {
        self.check_range(offset, out.len())?;
        // SAFETY: the range is inside the buffer (checked); `out` is a valid mutable slice distinct from the buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.contents_ptr().add(offset).cast_const(),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        Ok(())
    }

    /// Set `len` bytes from `offset` to `value`.
    pub fn fill(&self, offset: usize, len: usize, value: u8) -> Result<(), MetalError> {
        if self.kind == BufferKind::Mapped {
            return Err(MetalError::Buffer(
                "buffer wraps read-only host memory".to_owned(),
            ));
        }
        self.check_range(offset, len)?;
        // SAFETY: the range is inside the buffer (checked).
        unsafe { std::ptr::write_bytes(self.contents_ptr().add(offset), value, len) };
        Ok(())
    }
}

/// A byte range of a [`MetalBuffer`]: what kernels are bound to. Cheap to clone (one `Arc`).
#[derive(Clone, Debug)]
pub struct BufferRef {
    pub buffer: Arc<MetalBuffer>,
    pub offset: usize,
    pub len: usize,
}

impl BufferRef {
    /// The whole buffer.
    pub fn whole(buffer: Arc<MetalBuffer>) -> Self {
        let len = buffer.len();
        Self {
            buffer,
            offset: 0,
            len,
        }
    }

    /// A sub-range (`offset` relative to this range).
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self, MetalError> {
        if offset.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(MetalError::Buffer(format!(
                "slice {offset}+{len} outside a range of {} bytes",
                self.len
            )));
        }
        Ok(Self {
            buffer: Arc::clone(&self.buffer),
            offset: self.offset + offset,
            len,
        })
    }

    /// GPU address of the first byte of the range.
    pub fn gpu_address(&self) -> u64 {
        self.buffer.gpu_address() + self.offset as u64
    }

    pub fn write(&self, offset: usize, data: &[u8]) -> Result<(), MetalError> {
        self.slice(offset, data.len())?;
        self.buffer.write(self.offset + offset, data)
    }

    pub fn read(&self, offset: usize, out: &mut [u8]) -> Result<(), MetalError> {
        self.slice(offset, out.len())?;
        self.buffer.read(self.offset + offset, out)
    }
}

/// Allocator with a memory budget, bound to one device.
pub struct MemoryManager {
    device: MetalDevice,
    budget: Arc<MemoryBudget>,
    max_buffer_length: usize,
    force_copy: bool,
    labels: bool,
    mapped_bytes: AtomicU64,
}

impl MemoryManager {
    /// The budget is `FLLM_METAL_MEM_LIMIT_MB` or the device's recommended working set size; wrapped host memory does
    /// not count against it (it is page cache, not an allocation).
    pub fn new(device: &MetalDevice, config: &MetalConfig) -> Self {
        let info = device.info();
        let limit =
            config
                .memory_limit_bytes
                .unwrap_or(if info.recommended_max_working_set_bytes > 0 {
                    info.recommended_max_working_set_bytes
                } else {
                    u64::MAX
                });
        Self {
            device: device.clone(),
            budget: Arc::new(MemoryBudget::new(limit)),
            max_buffer_length: usize::try_from(info.max_buffer_length_bytes).unwrap_or(usize::MAX),
            force_copy: config.force_copy_weights,
            labels: config.debug_labels,
            mapped_bytes: AtomicU64::new(0),
        }
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Largest single buffer the device allows.
    pub fn max_buffer_length(&self) -> usize {
        self.max_buffer_length
    }

    /// Bytes of host memory currently wrapped without a copy.
    pub fn mapped_bytes(&self) -> u64 {
        self.mapped_bytes.load(Ordering::Relaxed)
    }

    /// A new buffer of `len` bytes (contents unspecified).
    pub fn alloc(&self, len: usize, usage: BufferUsage) -> Result<Arc<MetalBuffer>, MetalError> {
        self.alloc_labeled(len, usage, "")
    }

    /// Like [`MemoryManager::alloc`] with a name for GPU captures (applied when the label is non-empty).
    pub fn alloc_labeled(
        &self,
        len: usize,
        usage: BufferUsage,
        label: &str,
    ) -> Result<Arc<MetalBuffer>, MetalError> {
        if len > self.max_buffer_length {
            return Err(MetalError::OutOfMemory {
                requested: len,
                limit: self.max_buffer_length,
            });
        }
        let reserved = len.max(16) as u64;
        if !self.budget.try_reserve(reserved) {
            return Err(MetalError::OutOfMemory {
                requested: len,
                limit: usize::try_from(self.budget.limit()).unwrap_or(usize::MAX),
            });
        }
        let Some(raw) = self
            .device
            .raw()
            .newBufferWithLength_options(len.max(16), usage.resource_options())
        else {
            self.budget.release(reserved);
            return Err(MetalError::OutOfMemory {
                requested: len,
                limit: usize::try_from(self.budget.available()).unwrap_or(usize::MAX),
            });
        };
        let buffer = Arc::new(MetalBuffer {
            raw,
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            len,
            kind: BufferKind::Owned(usage),
            budget: Some(Arc::clone(&self.budget)),
            reserved,
            _owner: None,
        });
        if self.labels && !label.is_empty() {
            buffer.set_label(label);
        }
        Ok(buffer)
    }

    /// A new buffer of `len` zero bytes.
    pub fn alloc_zeroed(
        &self,
        len: usize,
        usage: BufferUsage,
    ) -> Result<Arc<MetalBuffer>, MetalError> {
        let buffer = self.alloc(len, usage)?;
        buffer.fill(0, len, 0)?;
        Ok(buffer)
    }

    /// A new buffer holding a copy of `bytes`.
    pub fn from_bytes(
        &self,
        bytes: &[u8],
        usage: BufferUsage,
    ) -> Result<Arc<MetalBuffer>, MetalError> {
        let buffer = self.alloc(bytes.len(), usage)?;
        buffer.write(0, bytes)?;
        Ok(buffer)
    }

    /// The buffer range for host bytes about to be uploaded as weights: the zero-copy window when `bytes` lies
    /// inside a region registered in `registry`, else a copy into a fresh [`BufferUsage::Weights`] buffer.
    pub fn buffer_for_host_slice(
        &self,
        registry: &MappedRegistry,
        bytes: &[u8],
    ) -> Result<BufferRef, MetalError> {
        if !self.force_copy {
            if let Some(view) = registry.lookup(bytes) {
                return Ok(view);
            }
        }
        Ok(BufferRef::whole(
            self.from_bytes(bytes, BufferUsage::Weights)?,
        ))
    }
}

/// Host memory wrapped without a copy: one or more page-aligned windows over the same bytes.
pub struct MappedRegion {
    base: usize,
    len: usize,
    plan: MappingPlan,
    buffers: Vec<Arc<MetalBuffer>>,
    _owner: Arc<dyn Any + Send + Sync>,
}

impl MappedRegion {
    /// Wrap `[ptr, ptr + len)` without copying. `max_item_len` is the largest object that will be looked up later
    /// (the biggest tensor of the file): every such object ends up completely inside one window.
    ///
    /// # Safety
    ///
    /// `ptr..ptr + len` must stay mapped and unchanged for as long as `owner` (or the returned region and the buffers
    /// made from it) lives, and `owner` must be what keeps the memory mapped (e.g. the `Arc<Mmap>`). The pages that
    /// contain the range are wrapped as a whole, so the range must come from a mapping (or a page-granular
    /// allocation), not from a sub-page piece of a larger heap object that may be freed separately.
    pub unsafe fn new(
        memory: &MemoryManager,
        owner: Arc<dyn Any + Send + Sync>,
        ptr: *const u8,
        len: usize,
        max_item_len: usize,
    ) -> Result<Self, MetalError> {
        if memory.force_copy {
            return Err(MetalError::Unsupported(
                "no-copy wrapping disabled by FLLM_METAL_NO_NOCOPY".to_owned(),
            ));
        }
        let info = memory.device.info();
        let plan = plan_mapping(
            ptr as usize,
            len,
            info.page_size,
            memory.max_buffer_length,
            max_item_len,
        )
        .map_err(MetalError::Buffer)?;
        let options = BufferUsage::Weights.resource_options();
        let mut buffers = Vec::with_capacity(plan.chunks.len());
        for chunk in &plan.chunks {
            let address = plan.aligned_base + chunk.start;
            let start = NonNull::new(address as *mut c_void)
                .ok_or_else(|| MetalError::Buffer("null host mapping".to_owned()))?;
            // SAFETY: the caller guarantees the memory stays mapped; `address` is page aligned and `chunk.len` is a page
            // multiple by construction of the plan; no deallocator block, the owner keeps the memory.
            let raw = unsafe {
                memory
                    .device
                    .raw()
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        start, chunk.len, options, None,
                    )
            }
            .ok_or_else(|| {
                MetalError::Buffer(format!(
                    "newBufferWithBytesNoCopy refused {} bytes at {address:#x}",
                    chunk.len
                ))
            })?;
            buffers.push(Arc::new(MetalBuffer {
                raw,
                id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
                len: chunk.len,
                kind: BufferKind::Mapped,
                budget: None,
                reserved: 0,
                _owner: Some(Arc::clone(&owner)),
            }));
        }
        memory
            .mapped_bytes
            .fetch_add(plan.total as u64, Ordering::Relaxed);
        Ok(Self {
            base: ptr as usize,
            len,
            plan,
            buffers,
            _owner: owner,
        })
    }

    /// The window buffers (for residency sets).
    pub fn buffers(&self) -> &[Arc<MetalBuffer>] {
        &self.buffers
    }

    /// Bytes covered by the windows (the range rounded out to pages).
    pub fn covered_bytes(&self) -> usize {
        self.plan.total
    }

    /// Does `bytes` lie inside the wrapped range?
    pub fn contains(&self, bytes: &[u8]) -> bool {
        let start = bytes.as_ptr() as usize;
        start >= self.base
            && start
                .checked_add(bytes.len())
                .is_some_and(|end| end <= self.base + self.len)
    }

    /// The buffer range that holds `bytes`, or `None` if it is outside the region or longer than the planned maximum.
    pub fn view(&self, bytes: &[u8]) -> Option<BufferRef> {
        if !self.contains(bytes) {
            return None;
        }
        let (index, offset) = self.plan.locate(bytes.as_ptr() as usize, bytes.len())?;
        Some(BufferRef {
            buffer: Arc::clone(&self.buffers[index]),
            offset,
            len: bytes.len(),
        })
    }
}

/// The wrapped host regions of a backend. Weights are looked up here by address when they are uploaded.
#[derive(Default)]
pub struct MappedRegistry {
    regions: RwLock<Vec<Arc<MappedRegion>>>,
}

impl MappedRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, region: MappedRegion) -> Arc<MappedRegion> {
        let region = Arc::new(region);
        self.regions
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Arc::clone(&region));
        region
    }

    /// The zero-copy view of `bytes` if a registered region holds it.
    pub fn lookup(&self, bytes: &[u8]) -> Option<BufferRef> {
        self.regions
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .find_map(|region| region.view(bytes))
    }

    pub fn region_count(&self) -> usize {
        self.regions
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

/// A buffer of 64-bit GPU addresses. A kernel receives it as `device const ulong *table` (or a pointer-to-pointer)
/// and reads `table[i]` to reach the i-th weight, so one binding serves any number of buffers. The buffers behind the
/// addresses are not bound to the encoder, which means the recorder has to declare them resident
/// (`DispatchBuilder::use_table`) and treats the access as touching everything for barrier purposes.
pub struct AddressTable {
    table: BufferRef,
    targets: Vec<Arc<MetalBuffer>>,
    entries: usize,
}

impl AddressTable {
    /// Build the table for `entries` (entry `i` = GPU address of `entries[i]`, offset included).
    pub fn build(memory: &MemoryManager, entries: &[BufferRef]) -> Result<Self, MetalError> {
        if !memory.device.info().argument_buffers_tier2 {
            return Err(MetalError::Unsupported(
                "GPU address tables need argument buffers tier 2".to_owned(),
            ));
        }
        let mut bytes = Vec::with_capacity(entries.len() * 8);
        let mut targets: Vec<Arc<MetalBuffer>> = Vec::new();
        for entry in entries {
            bytes.extend_from_slice(&entry.gpu_address().to_le_bytes());
            if !targets.iter().any(|t| t.id() == entry.buffer.id()) {
                targets.push(Arc::clone(&entry.buffer));
            }
        }
        let buffer = memory.from_bytes(&bytes, BufferUsage::Staging)?;
        Ok(Self {
            table: BufferRef::whole(buffer),
            targets,
            entries: entries.len(),
        })
    }

    /// The buffer of addresses (bind this one).
    pub fn buffer(&self) -> &BufferRef {
        &self.table
    }

    /// The distinct buffers the addresses point into (made resident by `use_table`).
    pub fn targets(&self) -> &[Arc<MetalBuffer>] {
        &self.targets
    }

    pub fn len(&self) -> usize {
        self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries == 0
    }
}

/// Bump allocator over one buffer: `alloc` hands out aligned ranges, `reset` makes all of them available again.
/// For scratch data that lives for one step. Ranges handed out before a `reset` must not be used after it.
pub struct Arena {
    buffer: Arc<MetalBuffer>,
    cursor: usize,
    high_water: usize,
}

impl Arena {
    pub fn new(
        memory: &MemoryManager,
        capacity: usize,
        usage: BufferUsage,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            buffer: memory.alloc(capacity, usage)?,
            cursor: 0,
            high_water: 0,
        })
    }

    /// A range of `len` bytes aligned to [`SUBALLOCATION_ALIGN`].
    pub fn alloc(&mut self, len: usize) -> Result<BufferRef, MetalError> {
        let start = plan::round_up(self.cursor, SUBALLOCATION_ALIGN)
            .ok_or_else(|| MetalError::Buffer("arena offset overflows".to_owned()))?;
        let end = start
            .checked_add(len)
            .filter(|end| *end <= self.buffer.len());
        let Some(end) = end else {
            return Err(MetalError::OutOfMemory {
                requested: len,
                limit: self.buffer.len().saturating_sub(start),
            });
        };
        self.cursor = end;
        self.high_water = self.high_water.max(end);
        Ok(BufferRef {
            buffer: Arc::clone(&self.buffer),
            offset: start,
            len,
        })
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn used(&self) -> usize {
        self.cursor
    }

    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Most bytes in use at any time since creation (to size the arena).
    pub fn high_water(&self) -> usize {
        self.high_water
    }

    pub fn buffer(&self) -> &Arc<MetalBuffer> {
        &self.buffer
    }
}
