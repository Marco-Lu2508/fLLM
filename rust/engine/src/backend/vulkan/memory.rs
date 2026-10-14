//! Native Vulkan backend (ash): memory. Owner: package 1 (Vulkan core).
//!
//! A small VMA-style allocator, specialised for compute:
//! - **Blocks**: device memory comes in blocks of up to 256 MiB per `(memory type, usage)` pool. Each block has *one* `VkBuffer` over its
//!   whole range, so a suballocation is just `(vk::Buffer, offset, size)`: no `vkCreateBuffer` per tensor and no buffer-memory binding
//!   per allocation. A free list per block (best fit, coalescing) hands out 256-byte aligned ranges.
//! - **Dedicated** allocations (larger than a quarter block) get their own memory and buffer.
//! - **Staging / readback / mapped** kinds are persistently mapped host-visible memory. On unified-memory devices (integrated GPUs) the
//!   device-local kind itself is mapped, so uploads are plain `memcpy`s.
//! - **Host pointer import** (`VK_EXT_external_memory_host`) wraps pinned host memory (an mmap of the model file) without copying.
//! - Bookkeeping: [`Allocator::free_memory`] (uses `VK_EXT_memory_budget` when present), [`Allocator::stats`].
//!
//! Lifetime rule: a [`Buffer`] returns its memory to the allocator when the last clone is dropped. The GPU may still be reading it then.
//! The command [`Recorder`](super::commands::Recorder) therefore keeps a reference to every buffer it records until the submission has
//! completed on the timeline; code that uses buffers outside the recorder must do the same.
//!
//! Level of verification: L0 (compiles; the pure parts are unit tested, the Vulkan calls never ran).

pub mod freelist;
pub mod pool;
pub mod select;

pub use freelist::{align_down, align_up};
pub use select::MemoryKind;

use super::instance::error::{vk_call, VkError, VkResult};
use super::instance::{DeviceCore, VkCaps};
use ash::vk;
use pool::Pool;
use select::{pick, policy_for, MemType};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ops::BitOr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

/// Largest block of suballocated memory.
pub const BLOCK_SIZE: u64 = 256 << 20;
/// Smallest alignment of an allocation (covers `minStorageBufferOffsetAlignment`, which is at most 256 by specification).
pub const MIN_ALIGNMENT: u64 = 256;
/// Blocks are never smaller than this (except when the device forbids it).
const MIN_BLOCK_SIZE: u64 = 16 << 20;

/// What a buffer may be used for (`VkBufferUsageFlags` subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BufferUsage(u32);

impl BufferUsage {
    pub const STORAGE: Self = Self(1);
    pub const TRANSFER_SRC: Self = Self(2);
    pub const TRANSFER_DST: Self = Self(4);
    pub const UNIFORM: Self = Self(8);
    pub const INDIRECT: Self = Self(16);
    /// Everything a compute backend needs from a buffer: storage binding, copy source and target, indirect dispatch arguments.
    pub const COMPUTE: Self = Self(1 | 2 | 4 | 16);

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn to_vk(self) -> vk::BufferUsageFlags {
        let mut flags = vk::BufferUsageFlags::empty();
        if self.contains(Self::STORAGE) {
            flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if self.contains(Self::TRANSFER_SRC) {
            flags |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if self.contains(Self::TRANSFER_DST) {
            flags |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if self.contains(Self::UNIFORM) {
            flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if self.contains(Self::INDIRECT) {
            flags |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        }
        flags
    }
}

impl BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Allocator counters (a snapshot).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllocStats {
    /// Sum of the sizes of all live buffers (as requested).
    pub live_bytes: u64,
    pub live_buffers: u64,
    /// Bytes of `VkDeviceMemory` currently allocated (blocks, dedicated, imported).
    pub reserved_bytes: u64,
    /// The part of `reserved_bytes` that sits in device-local heaps.
    pub device_reserved_bytes: u64,
    /// Live `vkAllocateMemory` allocations (the driver limits this, often to 4096).
    pub allocations: u64,
    /// Live suballocation blocks.
    pub blocks: u64,
}

/// `(aligned offset, range, skew)` for binding `len` bytes at absolute buffer offset `offset` with descriptor offsets aligned to `align`:
/// the descriptor starts at `aligned offset`, covers `range = skew + len` bytes, and the data begins `skew` bytes into it.
pub fn aligned_binding(offset: u64, len: u64, align: u64) -> (u64, u64, u64) {
    let aligned = align_down(offset, align.max(1).next_power_of_two());
    let skew = offset - aligned;
    (aligned, skew + len, skew)
}

/// The mapped-memory range for flush / invalidate: `(start, size)` with `start` and the end rounded to `atom`; `size` is
/// `vk::WHOLE_SIZE` when the range reaches the end of the allocation (the only place where a non-multiple of `atom` is allowed).
pub fn flush_range(offset: u64, len: u64, atom: u64, memory_size: u64) -> (u64, u64) {
    let atom = atom.max(1).next_power_of_two();
    let start = align_down(offset, atom);
    let end = align_up(offset + len, atom);
    if end >= memory_size {
        (start, vk::WHOLE_SIZE)
    } else {
        (start, end - start)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PoolKey {
    type_index: u32,
    usage: BufferUsage,
}

#[derive(Debug, Clone, Copy)]
struct TypeChoice {
    type_index: u32,
    host_visible: bool,
    coherent: bool,
}

/// One block: device memory, the buffer over all of it and the persistent mapping (null when not host visible).
struct VkBlock {
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    buffer_size: u64,
    memory_size: u64,
    mapped: *mut u8,
    heap: u32,
}

// SAFETY: the raw pointer is the mapping of the block's memory; it is only used under the allocator lock or via `Buffer` rules.
unsafe impl Send for VkBlock {}

struct State {
    pools: HashMap<PoolKey, Pool<VkBlock>>,
    choices: HashMap<(MemoryKind, BufferUsage), TypeChoice>,
    heap_reserved: Vec<u64>,
    stats: AllocStats,
}

struct AllocShared {
    core: Arc<DeviceCore>,
    types: Vec<MemType>,
    unified: bool,
    block_size: u64,
    dedicated_threshold: u64,
    /// Alignment of every suballocation.
    align: u64,
    atom: u64,
    max_buffer_size: u64,
    max_allocation_count: u64,
    import_align: u64,
    heap_is_device_local: Vec<bool>,
    vram_total: u64,
    memory_budget: bool,
    next_id: AtomicU64,
    state: Mutex<State>,
}

/// The allocator of one device. Cheap to share (`Arc<Allocator>` lives in the [`VkContext`](super::instance::VkContext)); all methods
/// take `&self` and are thread safe.
pub struct Allocator {
    shared: Arc<AllocShared>,
}

enum Backing {
    Sub {
        key: PoolKey,
        slot: usize,
        offset: u64,
        size: u64,
    },
    Dedicated {
        memory: vk::DeviceMemory,
        heap: u32,
        memory_size: u64,
    },
    Imported {
        memory: vk::DeviceMemory,
    },
}

pub(crate) struct BufferInner {
    shared: Arc<AllocShared>,
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_size: u64,
    /// Start of the mapping of the memory (which is also the start of the buffer), null when not host visible.
    mapped_base: *mut u8,
    /// Size of the whole `VkBuffer`.
    vk_size: u64,
    /// Requested size.
    size: u64,
    coherent: bool,
    kind: MemoryKind,
    id: u64,
    /// Epoch of the last recorder that retained this buffer (see `commands`).
    pub(crate) retain_epoch: AtomicU64,
    backing: Backing,
}

// SAFETY: `mapped_base` is a mapping that stays valid until the allocation is dropped; concurrent access to the bytes is the caller's
// concern (as with any shared buffer); all other fields are plain handles used under the rules of Vulkan.
unsafe impl Send for BufferInner {}
unsafe impl Sync for BufferInner {}

impl Drop for BufferInner {
    fn drop(&mut self) {
        let shared = &self.shared;
        {
            let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.stats.live_bytes = state.stats.live_bytes.saturating_sub(self.size);
            state.stats.live_buffers = state.stats.live_buffers.saturating_sub(1);
        }
        match &self.backing {
            Backing::Sub { key, slot, offset, size } => shared.release_sub(*key, *slot, *offset, *size),
            Backing::Dedicated { memory, heap, memory_size } => {
                // SAFETY: nothing uses the buffer any more (the last reference is being dropped, and recorders hold references until the
                // GPU is done); the handles were created by this allocator.
                unsafe {
                    shared.core.device.destroy_buffer(self.handle, None);
                    shared.core.device.free_memory(*memory, None);
                }
                shared.note_freed(*heap, *memory_size);
            }
            Backing::Imported { memory } => {
                // SAFETY: as above.
                unsafe {
                    shared.core.device.destroy_buffer(self.handle, None);
                    shared.core.device.free_memory(*memory, None);
                }
                shared.note_freed(u32::MAX, self.memory_size);
            }
        }
    }
}

/// A view of device or host memory: `size` bytes at `offset` of a `VkBuffer`. Clones share the allocation (reference counted); the memory
/// is returned when the last clone is dropped. `Send + Sync`.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
    /// Absolute offset in the `VkBuffer`.
    offset: u64,
    size: u64,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("id", &self.inner.id)
            .field("kind", &self.inner.kind)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .field("mapped", &!self.inner.mapped_base.is_null())
            .finish()
    }
}

impl Buffer {
    /// Unique id of the underlying allocation (clones and slices share it).
    pub fn id(&self) -> u64 {
        self.inner.id
    }

    /// Bytes visible through this view.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Offset of this view in [`Buffer::handle`] (the `offset` of a descriptor).
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The `VkBuffer` (shared by all allocations of a block).
    pub fn handle(&self) -> vk::Buffer {
        self.inner.handle
    }

    pub fn kind(&self) -> MemoryKind {
        self.inner.kind
    }

    /// The memory is mapped (host visible): [`Buffer::mapped_ptr`], [`Buffer::write`] and [`Buffer::read`] work.
    pub fn is_host_visible(&self) -> bool {
        !self.inner.mapped_base.is_null()
    }

    pub(crate) fn inner(&self) -> &Arc<BufferInner> {
        &self.inner
    }

    /// Whether both views belong to the same allocation.
    pub fn same_allocation(&self, other: &Buffer) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// `len` bytes at `offset` of this view, sharing the allocation.
    pub fn slice(&self, offset: u64, len: u64) -> VkResult<Buffer> {
        if offset.checked_add(len).is_none_or(|end| end > self.size) {
            return Err(VkError::InvalidArgument(format!(
                "slice {offset}+{len} exceeds a buffer of {} bytes",
                self.size
            )));
        }
        Ok(Buffer {
            inner: Arc::clone(&self.inner),
            offset: self.offset + offset,
            size: len,
        })
    }

    /// A view for binding `len` bytes at `offset` of this view as a storage buffer when `offset` does not satisfy
    /// `minStorageBufferOffsetAlignment` (tensors inside an imported model file, for example). Returns the view, which starts at the
    /// aligned-down offset and may reach below the first byte of this view, and the `skew`: the data starts `skew` bytes into the view
    /// (pass it to the shader, for example as a push constant). The extra bytes in front are only readable memory of the same `VkBuffer`.
    pub fn aligned_view(&self, offset: u64, len: u64, align: u64) -> VkResult<(Buffer, u32)> {
        if offset.checked_add(len).is_none_or(|end| end > self.size) {
            return Err(VkError::InvalidArgument(format!(
                "view {offset}+{len} exceeds a buffer of {} bytes",
                self.size
            )));
        }
        let (aligned, range, skew) = aligned_binding(self.offset + offset, len, align);
        if aligned + range > self.inner.vk_size {
            return Err(VkError::InvalidArgument("aligned view exceeds the VkBuffer".into()));
        }
        Ok((
            Buffer {
                inner: Arc::clone(&self.inner),
                offset: aligned,
                size: range,
            },
            skew as u32,
        ))
    }

    /// The descriptor for binding the whole view as a storage buffer.
    pub fn descriptor_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.inner.handle,
            offset: self.offset,
            range: self.size,
        }
    }

    /// Pointer to the first byte of this view when the memory is mapped.
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        if self.inner.mapped_base.is_null() {
            None
        } else {
            // SAFETY: `offset` lies inside the mapping (views never exceed the VkBuffer, which the mapping covers).
            Some(unsafe { self.inner.mapped_base.add(self.offset as usize) })
        }
    }

    fn host_range(&self, offset: u64, len: usize) -> VkResult<*mut u8> {
        let Some(base) = self.mapped_ptr() else {
            return Err(VkError::InvalidArgument(
                "the buffer is not host visible (use a staging copy)".into(),
            ));
        };
        if offset.checked_add(len as u64).is_none_or(|end| end > self.size) {
            return Err(VkError::InvalidArgument(format!(
                "host access {offset}+{len} exceeds a buffer of {} bytes",
                self.size
            )));
        }
        // SAFETY: bounds checked above.
        Ok(unsafe { base.add(offset as usize) })
    }

    /// Copy `data` into the mapped memory at `offset` (flushing when the memory is not coherent). The caller makes sure the GPU is not
    /// using the range (timeline value reached).
    pub fn write(&self, offset: u64, data: &[u8]) -> VkResult<()> {
        let dst = self.host_range(offset, data.len())?;
        // SAFETY: `dst` is valid for `data.len()` bytes of mapped memory; source and destination cannot overlap (host slice vs. mapping).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
        self.flush(offset, data.len() as u64)
    }

    /// Copy mapped memory at `offset` into `out` (invalidating first when the memory is not coherent).
    pub fn read(&self, offset: u64, out: &mut [u8]) -> VkResult<()> {
        let src = self.host_range(offset, out.len())?;
        self.invalidate(offset, out.len() as u64)?;
        // SAFETY: as in `write`.
        unsafe { std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    /// Make host writes to `offset..offset + len` of this view visible to the device (no-op for coherent memory).
    pub fn flush(&self, offset: u64, len: u64) -> VkResult<()> {
        self.sync_range(offset, len, true)
    }

    /// Make device writes to `offset..offset + len` of this view visible to the host (no-op for coherent memory).
    pub fn invalidate(&self, offset: u64, len: u64) -> VkResult<()> {
        self.sync_range(offset, len, false)
    }

    fn sync_range(&self, offset: u64, len: u64, flush: bool) -> VkResult<()> {
        let inner = &self.inner;
        if inner.coherent || inner.mapped_base.is_null() || len == 0 {
            return Ok(());
        }
        let (start, size) = flush_range(self.offset + offset, len, inner.shared.atom, inner.memory_size);
        let range = vk::MappedMemoryRange::default().memory(inner.memory).offset(start).size(size);
        let device = &inner.shared.core.device;
        // SAFETY: the memory is mapped and belongs to this device.
        let result = unsafe {
            if flush {
                device.flush_mapped_memory_ranges(&[range])
            } else {
                device.invalidate_mapped_memory_ranges(&[range])
            }
        };
        result.map_err(vk_call(if flush {
            "vkFlushMappedMemoryRanges"
        } else {
            "vkInvalidateMappedMemoryRanges"
        }))
    }
}

/// `(budget, usage)` in bytes of every memory heap from `VK_EXT_memory_budget` (usage covers all processes).
pub fn heap_budget(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Vec<(u64, u64)> {
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    // SAFETY: `physical` belongs to `instance`; the structs live through the call. The extension must be supported by the device.
    unsafe { instance.get_physical_device_memory_properties2(physical, &mut props) };
    let count = props.memory_properties.memory_heap_count as usize;
    (0..count).map(|i| (budget.heap_budget[i], budget.heap_usage[i])).collect()
}

impl Allocator {
    pub(crate) fn new(core: Arc<DeviceCore>, caps: &VkCaps) -> VkResult<Self> {
        let mem = &core.memory;
        let types: Vec<MemType> = mem.memory_types[..mem.memory_type_count as usize]
            .iter()
            .map(|t| MemType {
                flags: t.property_flags.as_raw(),
                heap: t.heap_index,
                heap_size: mem.memory_heaps[t.heap_index as usize].size,
            })
            .collect();
        let heap_count = mem.memory_heap_count as usize;
        let heap_is_device_local: Vec<bool> = (0..heap_count)
            .map(|i| mem.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .collect();
        let align = MIN_ALIGNMENT
            .max(caps.min_storage_buffer_offset_alignment)
            .max(caps.non_coherent_atom_size)
            .next_power_of_two();
        let max_alloc = caps.max_memory_allocation_size.max(align);
        let heap_cap = if caps.vram_bytes > 0 {
            (caps.vram_bytes / 8).max(MIN_BLOCK_SIZE)
        } else {
            BLOCK_SIZE
        };
        let block_size = align_down(BLOCK_SIZE.min(max_alloc).min(heap_cap).min(caps.max_buffer_size.max(align)), align).max(align);
        Ok(Self {
            shared: Arc::new(AllocShared {
                core,
                types,
                unified: caps.unified_memory,
                block_size,
                dedicated_threshold: (block_size / 4).max(align),
                align,
                atom: caps.non_coherent_atom_size,
                max_buffer_size: caps.max_buffer_size,
                max_allocation_count: u64::from(caps.max_memory_allocation_count),
                import_align: caps.min_imported_host_pointer_alignment,
                heap_is_device_local,
                vram_total: caps.vram_bytes,
                memory_budget: caps.memory_budget,
                next_id: AtomicU64::new(1),
                state: Mutex::new(State {
                    pools: HashMap::new(),
                    choices: HashMap::new(),
                    heap_reserved: vec![0; heap_count],
                    stats: AllocStats::default(),
                }),
            }),
        })
    }

    /// GPU-side memory for weights, KV cache and activations. Mapped on unified-memory devices (see [`Allocator::is_unified`]).
    pub fn alloc_device(&self, size: u64, usage: BufferUsage) -> VkResult<Buffer> {
        self.alloc(MemoryKind::DeviceLocal, size, usage)
    }

    /// Device-local memory the CPU can write (resizable BAR / unified memory), else a staging buffer. Check [`Buffer::kind`].
    pub fn alloc_device_mapped(&self, size: u64, usage: BufferUsage) -> VkResult<Buffer> {
        self.alloc(MemoryKind::DeviceMapped, size, usage)
    }

    /// Mapped host memory for uploads (usable as copy source and, on request of the shader, as storage buffer).
    pub fn alloc_staging(&self, size: u64) -> VkResult<Buffer> {
        self.alloc(MemoryKind::Staging, size, BufferUsage::COMPUTE)
    }

    /// Mapped, cached host memory for reading results back (copy destination).
    pub fn alloc_readback(&self, size: u64) -> VkResult<Buffer> {
        self.alloc(MemoryKind::Readback, size, BufferUsage::COMPUTE)
    }

    /// Allocate `size` bytes (> 0) of `kind`. Memory is not cleared.
    pub fn alloc(&self, kind: MemoryKind, size: u64, usage: BufferUsage) -> VkResult<Buffer> {
        self.shared.allocate(kind, size, usage)
    }

    /// Wrap `len` bytes of host memory at `ptr` as a read-mostly storage buffer without copying (`VK_EXT_external_memory_host`).
    /// The buffer's view starts exactly at `ptr`; the `VkBuffer` begins at the previous multiple of
    /// [`Allocator::import_alignment`], so `Buffer::offset()` is the distance to it. Use [`Buffer::aligned_view`] to bind sub-ranges.
    ///
    /// On a discrete GPU every access crosses the PCIe bus; the import is meant for unified memory and for layers kept on the host.
    ///
    /// # Safety
    /// The whole range from `ptr` rounded down to the import alignment up to `ptr + len` rounded up to it must be mapped and stay mapped
    /// (and unmodified if the GPU reads it) until the returned buffer and every recorded use of it are gone. A read-only mapping is fine
    /// as long as no shader writes to the buffer. The buffer holds no reference to the owner of the memory.
    pub unsafe fn import_host_ptr(&self, ptr: *mut u8, len: usize) -> VkResult<Buffer> {
        self.shared.import(ptr, len)
    }

    /// `VK_EXT_external_memory_host` is available.
    pub fn can_import_host_ptr(&self) -> bool {
        self.shared.core.external_memory_host.is_some() && self.shared.import_align > 0
    }

    /// Alignment of host pointers and sizes for [`Allocator::import_host_ptr`] (0 without the extension).
    pub fn import_alignment(&self) -> u64 {
        self.shared.import_align
    }

    /// Host and device share memory: [`MemoryKind::DeviceLocal`] buffers are mapped.
    pub fn is_unified(&self) -> bool {
        self.shared.unified
    }

    /// Size of one suballocation block (<= 256 MiB).
    pub fn block_size(&self) -> u64 {
        self.shared.block_size
    }

    /// Alignment (and minimum size) of every allocation.
    pub fn alignment(&self) -> u64 {
        self.shared.align
    }

    /// Total bytes in the device-local heaps.
    pub fn total_memory(&self) -> u64 {
        self.shared.vram_total
    }

    /// Bytes still available in device-local heaps: the driver's budget minus the usage of all processes when `VK_EXT_memory_budget`
    /// is present, else the heap sizes minus what this allocator reserved. An estimate: other processes may allocate at any time.
    pub fn free_memory(&self) -> u64 {
        let shared = &self.shared;
        if shared.memory_budget {
            let heaps = heap_budget(shared.core.instance(), shared.core.physical);
            let free: u64 = heaps
                .iter()
                .enumerate()
                .filter(|(i, _)| shared.heap_is_device_local.get(*i).copied().unwrap_or(false))
                .map(|(_, (budget, usage))| budget.saturating_sub(*usage))
                .sum();
            if free > 0 {
                return free;
            }
        }
        shared.vram_total.saturating_sub(self.stats().device_reserved_bytes)
    }

    pub fn stats(&self) -> AllocStats {
        self.shared.state.lock().unwrap_or_else(PoisonError::into_inner).stats
    }
}

impl AllocShared {
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    fn device(&self) -> &ash::Device {
        &self.core.device
    }

    fn buffer_info(size: u64, usage: BufferUsage) -> vk::BufferCreateInfo<'static> {
        vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage.to_vk())
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
    }

    /// Memory type for `(kind, usage)`, cached. Creates a throw-away buffer once to learn which types the usage allows.
    fn choice(&self, state: &mut State, kind: MemoryKind, usage: BufferUsage) -> VkResult<TypeChoice> {
        if let Some(c) = state.choices.get(&(kind, usage)) {
            return Ok(*c);
        }
        let info = Self::buffer_info(self.align, usage);
        // SAFETY: plain object creation / destruction on a valid device.
        let bits = unsafe {
            let probe = self.device().create_buffer(&info, None).map_err(vk_call("vkCreateBuffer"))?;
            let bits = self.device().get_buffer_memory_requirements(probe).memory_type_bits;
            self.device().destroy_buffer(probe, None);
            bits
        };
        let mut index = pick(&self.types, bits, &policy_for(kind, self.unified));
        if index.is_none() && kind == MemoryKind::DeviceMapped {
            index = pick(&self.types, bits, &policy_for(MemoryKind::Staging, self.unified));
        }
        let Some(type_index) = index else {
            return Err(VkError::Unsupported(format!(
                "no memory type for {kind:?} buffers (type bits {bits:#x})"
            )));
        };
        let flags = self.types[type_index as usize].flags;
        let choice = TypeChoice {
            type_index,
            host_visible: flags & select::bits::HOST_VISIBLE != 0,
            coherent: flags & select::bits::HOST_COHERENT != 0,
        };
        state.choices.insert((kind, usage), choice);
        Ok(choice)
    }

    fn check_allocation_budget(&self, state: &State, requested: u64) -> VkResult<()> {
        if self.max_allocation_count > 0 && state.stats.allocations + 8 >= self.max_allocation_count {
            return Err(VkError::OutOfMemory { requested });
        }
        Ok(())
    }

    fn note_reserved(&self, state: &mut State, heap: u32, bytes: u64) {
        if let Some(h) = state.heap_reserved.get_mut(heap as usize) {
            *h += bytes;
        }
        state.stats.reserved_bytes += bytes;
        if self.heap_is_device_local.get(heap as usize).copied().unwrap_or(false) {
            state.stats.device_reserved_bytes += bytes;
        }
        state.stats.allocations += 1;
    }

    /// Bookkeeping when device memory is freed (`heap == u32::MAX`: imported memory, which is not counted in a heap).
    fn note_freed(&self, heap: u32, bytes: u64) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.stats.reserved_bytes = state.stats.reserved_bytes.saturating_sub(bytes);
        state.stats.allocations = state.stats.allocations.saturating_sub(1);
        if heap != u32::MAX {
            if let Some(h) = state.heap_reserved.get_mut(heap as usize) {
                *h = h.saturating_sub(bytes);
            }
            if self.heap_is_device_local.get(heap as usize).copied().unwrap_or(false) {
                state.stats.device_reserved_bytes = state.stats.device_reserved_bytes.saturating_sub(bytes);
            }
        }
    }

    /// Create buffer + memory (+ bind + map). `dedicated` adds the dedicated-allocation hint.
    fn make_memory(&self, size: u64, usage: BufferUsage, choice: TypeChoice, dedicated: bool) -> VkResult<VkBlock> {
        let device = self.device();
        // SAFETY: object creation on a valid device; every failure path destroys what was created before it.
        unsafe {
            let buffer = device
                .create_buffer(&Self::buffer_info(size, usage), None)
                .map_err(|e| VkError::from_result("vkCreateBuffer", e).with_requested(size))?;
            let req = device.get_buffer_memory_requirements(buffer);
            if req.memory_type_bits & (1 << choice.type_index) == 0 {
                device.destroy_buffer(buffer, None);
                return Err(VkError::Unsupported(format!(
                    "memory type {} cannot back a {size} byte buffer (type bits {:#x})",
                    choice.type_index, req.memory_type_bits
                )));
            }
            let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
            let mut alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(choice.type_index);
            if dedicated {
                alloc = alloc.push_next(&mut dedicated_info);
            }
            let memory = match device.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(e) => {
                    device.destroy_buffer(buffer, None);
                    return Err(VkError::from_result("vkAllocateMemory", e).with_requested(size));
                }
            };
            if let Err(e) = device.bind_buffer_memory(buffer, memory, 0) {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                return Err(VkError::from_result("vkBindBufferMemory", e));
            }
            let mut mapped = std::ptr::null_mut();
            if choice.host_visible {
                match device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) {
                    Ok(p) => mapped = p.cast::<u8>(),
                    Err(e) => {
                        device.destroy_buffer(buffer, None);
                        device.free_memory(memory, None);
                        return Err(VkError::from_result("vkMapMemory", e));
                    }
                }
            }
            Ok(VkBlock {
                memory,
                buffer,
                buffer_size: size,
                memory_size: req.size,
                mapped,
                heap: self.types[choice.type_index as usize].heap,
            })
        }
    }

    fn destroy_block(&self, block: &VkBlock) {
        // SAFETY: the block is not referenced by any allocation any more.
        unsafe {
            self.device().destroy_buffer(block.buffer, None);
            self.device().free_memory(block.memory, None);
        }
        self.note_freed(block.heap, block.memory_size);
    }

    fn allocate(self: &Arc<Self>, kind: MemoryKind, size: u64, usage: BufferUsage) -> VkResult<Buffer> {
        if size == 0 {
            return Err(VkError::InvalidArgument("cannot allocate a buffer of 0 bytes".into()));
        }
        if size > self.max_buffer_size {
            return Err(VkError::InvalidArgument(format!(
                "{size} bytes exceed maxBufferSize ({})",
                self.max_buffer_size
            )));
        }
        let reserved = align_up(size, self.align);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let choice = self.choice(&mut state, kind, usage)?;
        self.check_allocation_budget(&state, size)?;

        // the memory kind that actually got served (DeviceMapped may fall back to staging memory)
        let served_kind =
            if kind == MemoryKind::DeviceMapped && self.types[choice.type_index as usize].flags & select::bits::DEVICE_LOCAL == 0 {
                MemoryKind::Staging
            } else {
                kind
            };

        if reserved > self.dedicated_threshold || reserved > self.block_size {
            let block = self.make_memory(size.max(self.align), usage, choice, true)?;
            self.note_reserved(&mut state, block.heap, block.memory_size);
            state.stats.live_bytes += size;
            state.stats.live_buffers += 1;
            return Ok(Buffer {
                inner: Arc::new(BufferInner {
                    shared: Arc::clone(self),
                    handle: block.buffer,
                    memory: block.memory,
                    memory_size: block.memory_size,
                    mapped_base: block.mapped,
                    vk_size: block.buffer_size,
                    size,
                    coherent: choice.coherent,
                    kind: served_kind,
                    id: self.next_id(),
                    retain_epoch: AtomicU64::new(0),
                    backing: Backing::Dedicated {
                        memory: block.memory,
                        heap: block.heap,
                        memory_size: block.memory_size,
                    },
                }),
                offset: 0,
                size,
            });
        }

        let key = PoolKey {
            type_index: choice.type_index,
            usage,
        };
        let mut found = state.pools.entry(key).or_default().try_alloc(reserved, self.align);
        if found.is_none() {
            // new block; when the device runs short of memory try smaller ones (never below what this allocation needs)
            let mut capacity = self.block_size;
            let block = loop {
                match self.make_memory(capacity, usage, choice, false) {
                    Ok(b) => break b,
                    Err(VkError::OutOfMemory { .. }) if capacity > reserved.max(MIN_BLOCK_SIZE.min(self.block_size)) => {
                        capacity = align_up((capacity / 2).max(reserved), self.align);
                    }
                    Err(e) => return Err(e.with_requested(size)),
                }
            };
            self.note_reserved(&mut state, block.heap, block.memory_size);
            let buffer_size = block.buffer_size;
            let pool = state.pools.entry(key).or_default();
            let slot = pool.insert(block, buffer_size);
            let offset = pool.get_mut(slot).and_then(|b| b.free.alloc(reserved, self.align));
            found = offset.map(|o| (slot, o));
        }
        let Some((slot, offset)) = found else {
            return Err(VkError::OutOfMemory { requested: size });
        };
        let block = &state.pools[&key].get(slot).ok_or(VkError::OutOfMemory { requested: size })?.payload;
        let (handle, memory, memory_size, mapped, vk_size) =
            (block.buffer, block.memory, block.memory_size, block.mapped, block.buffer_size);
        state.stats.live_bytes += size;
        state.stats.live_buffers += 1;
        Ok(Buffer {
            inner: Arc::new(BufferInner {
                shared: Arc::clone(self),
                handle,
                memory,
                memory_size,
                mapped_base: mapped,
                vk_size,
                size,
                coherent: choice.coherent,
                kind: served_kind,
                id: self.next_id(),
                retain_epoch: AtomicU64::new(0),
                backing: Backing::Sub {
                    key,
                    slot,
                    offset,
                    size: reserved,
                },
            }),
            offset,
            size,
        })
    }

    fn release_sub(&self, key: PoolKey, slot: usize, offset: u64, size: u64) {
        let removed = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.pools.get_mut(&key).and_then(|pool| pool.release(slot, offset, size))
        };
        if let Some(block) = removed {
            self.destroy_block(&block);
        }
    }

    fn import(self: &Arc<Self>, ptr: *mut u8, len: usize) -> VkResult<Buffer> {
        let Some(ext) = self.core.external_memory_host.as_ref() else {
            return Err(VkError::Unsupported(
                "VK_EXT_external_memory_host is not available on this device".into(),
            ));
        };
        if ptr.is_null() || len == 0 {
            return Err(VkError::InvalidArgument("import of an empty or null host range".into()));
        }
        let align = self.import_align;
        if align == 0 || !align.is_power_of_two() {
            return Err(VkError::Unsupported("the device did not report an import alignment".into()));
        }
        let start = align_down(ptr as u64, align);
        let skew = ptr as u64 - start;
        let size = align_up(skew + len as u64, align);
        if size > self.max_buffer_size {
            return Err(VkError::InvalidArgument(format!(
                "{size} bytes exceed maxBufferSize ({}); import the file in smaller chunks",
                self.max_buffer_size
            )));
        }
        let device = self.device();
        let handle_type = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let mut host_props = vk::MemoryHostPointerPropertiesEXT::default();
        // SAFETY: the function pointer was loaded for this device; `start` is aligned as the specification requires; the caller of the
        // unsafe `import_host_ptr` guarantees that the memory is mapped.
        let result = unsafe {
            (ext.fp().get_memory_host_pointer_properties_ext)(ext.device(), handle_type, start as usize as *mut c_void, &mut host_props)
        };
        result.result().map_err(vk_call("vkGetMemoryHostPointerPropertiesEXT"))?;

        // SAFETY: creation with cleanup on every failure path; the imported memory is not owned by Vulkan (freeing it does not unmap).
        unsafe {
            let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external);
            let buffer = device.create_buffer(&info, None).map_err(vk_call("vkCreateBuffer"))?;
            let req = device.get_buffer_memory_requirements(buffer);
            let bits = req.memory_type_bits & host_props.memory_type_bits;
            let Some(type_index) = pick(&self.types, bits, &select::IMPORT) else {
                device.destroy_buffer(buffer, None);
                return Err(VkError::Unsupported("no memory type can hold the imported host pointer".into()));
            };
            let allocation_size = align_up(size.max(req.size), align);
            let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(handle_type)
                .host_pointer(start as usize as *mut c_void);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(allocation_size)
                .memory_type_index(type_index)
                .push_next(&mut import);
            let memory = match device.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(e) => {
                    device.destroy_buffer(buffer, None);
                    return Err(VkError::from_result("vkAllocateMemory(host pointer)", e).with_requested(allocation_size));
                }
            };
            if let Err(e) = device.bind_buffer_memory(buffer, memory, 0) {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                return Err(VkError::from_result("vkBindBufferMemory", e));
            }
            let flags = self.types[type_index as usize].flags;
            {
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                state.stats.reserved_bytes += allocation_size;
                state.stats.allocations += 1;
                state.stats.live_bytes += len as u64;
                state.stats.live_buffers += 1;
            }
            Ok(Buffer {
                inner: Arc::new(BufferInner {
                    shared: Arc::clone(self),
                    handle: buffer,
                    memory,
                    memory_size: allocation_size,
                    mapped_base: start as usize as *mut u8,
                    vk_size: size,
                    size: len as u64,
                    coherent: flags & select::bits::HOST_COHERENT != 0,
                    kind: MemoryKind::Staging,
                    id: self.next_id(),
                    retain_epoch: AtomicU64::new(0),
                    backing: Backing::Imported { memory },
                }),
                offset: skew,
                size: len as u64,
            })
        }
    }
}

impl Drop for AllocShared {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        let blocks: Vec<VkBlock> = state.pools.drain().flat_map(|(_, mut pool)| pool.drain()).collect();
        for block in &blocks {
            // SAFETY: the allocator is being destroyed: no buffer exists (each holds an Arc of this struct).
            unsafe {
                self.core.device.destroy_buffer(block.buffer, None);
                self.core.device.free_memory(block.memory, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_flags() {
        assert!(BufferUsage::COMPUTE.contains(BufferUsage::STORAGE));
        assert!(BufferUsage::COMPUTE.contains(BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC));
        assert!(!BufferUsage::COMPUTE.contains(BufferUsage::UNIFORM));
        let vk_flags = (BufferUsage::STORAGE | BufferUsage::UNIFORM).to_vk();
        assert!(vk_flags.contains(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER));
        assert!(!vk_flags.contains(vk::BufferUsageFlags::TRANSFER_SRC));
        assert_eq!(BufferUsage::default().to_vk(), vk::BufferUsageFlags::empty());
    }

    #[test]
    fn binding_alignment() {
        // already aligned
        assert_eq!(aligned_binding(512, 100, 256), (512, 100, 0));
        // a tensor at a 32-byte aligned file offset inside a page-aligned import
        assert_eq!(aligned_binding(4096 + 32, 1000, 256), (4096, 1032, 32));
        // alignment 1 (some drivers): never moves
        assert_eq!(aligned_binding(77, 10, 1), (77, 10, 0));
        // a non power of two request is rounded up (spec alignments are powers of two anyway)
        assert_eq!(aligned_binding(300, 10, 100), (256, 54, 44));
    }

    #[test]
    fn flush_ranges_follow_the_atom_size() {
        // inside the allocation: rounded out to the atom
        assert_eq!(flush_range(100, 50, 64, 4096), (64, 128));
        assert_eq!(flush_range(0, 64, 64, 4096), (0, 64));
        // reaching the end of the allocation: WHOLE_SIZE (the end need not be a multiple of the atom)
        assert_eq!(flush_range(4000, 90, 64, 4090), (3968, vk::WHOLE_SIZE));
        assert_eq!(flush_range(0, 4096, 64, 4096), (0, vk::WHOLE_SIZE));
    }
}
