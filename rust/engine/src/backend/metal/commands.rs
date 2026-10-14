//! Command recording of the native Metal backend (objc2-metal; compiled on Apple targets only).
//!
//! One [`Recorder`] per stream. It keeps one command buffer with one compute encoder open and appends dispatches to
//! it; the encoder is `MTLDispatchTypeConcurrent`, so independent dispatches may overlap on the GPU, and the
//! recorder inserts a `memoryBarrierWithScope(Buffers)` only where the declared buffer accesses of a dispatch
//! conflict with the ones before it (see [`hazard`]). `FLLM_METAL_SERIAL=1` switches to the serial dispatch type
//! where the driver orders every dispatch itself.
//!
//! To let the GPU start early, a command buffer is committed after `FLLM_METAL_CB_DISPATCHES` dispatches (default
//! 160) while the CPU goes on encoding the next one into a fresh command buffer. Command buffers of one queue run in
//! order, and hazard-tracked buffers are synchronised by Metal across encoders and command buffers. Waiting
//! happens only in [`Recorder::synchronize`].
//!
//! [`MetalContext`] bundles device, memory manager, pipeline cache, queue and registry for the integration layer.
//!
//! Level of verification: L0 (compiles for `aarch64-apple-darwin`; never run on a Mac).

pub mod geometry;
pub mod hazard;
pub mod residency;
pub mod smoke;

pub use hazard::Access;
pub use residency::ResidencyManager;

use super::buffers::{AddressTable, BufferRef, MappedRegion, MappedRegistry, MemoryManager};
use super::device_info::{self, MetalConfig, MetalDevice, MetalError};
use super::pipelines::{Pipeline, PipelineCache};
use geometry::Validated;
use hazard::HazardTracker;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice, MTLDispatchType,
    MTLResourceUsage, MTLSize,
};
use std::any::Any;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

/// Buffer and constant-data binding slots per kernel (Metal's limit is 31 buffer arguments).
pub const MAX_BIND_SLOTS: u32 = 31;

/// Largest blob `setBytes` accepts.
pub const MAX_INLINE_BYTES: usize = 4096;

/// Command buffers in flight at once before the recorder waits for the oldest (the queue's own limit is 64).
const MAX_IN_FLIGHT: usize = 40;

/// What the recorder needs from the configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecorderSettings {
    pub serial_dispatch: bool,
    pub dispatches_per_command_buffer: usize,
    pub labels: bool,
    pub hazard_window: usize,
}

impl RecorderSettings {
    pub fn from_config(config: &MetalConfig) -> Self {
        Self {
            serial_dispatch: config.serial_dispatch,
            dispatches_per_command_buffer: config.dispatches_per_command_buffer,
            labels: config.debug_labels,
            hazard_window: config.hazard_window,
        }
    }
}

/// Device limits the recorder validates dispatches against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    pub max_threads: [usize; 3],
    pub max_threadgroup_memory: usize,
}

impl DeviceLimits {
    pub fn of(device: &MetalDevice) -> Self {
        let info = device.info();
        Self {
            max_threads: info.max_threads_per_threadgroup,
            max_threadgroup_memory: usize::try_from(info.max_threadgroup_memory_bytes)
                .unwrap_or(usize::MAX),
        }
    }
}

/// The command queue of a device; makes recorders.
pub struct CommandQueue {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    device: MetalDevice,
    settings: RecorderSettings,
}

impl CommandQueue {
    pub fn new(device: &MetalDevice, config: &MetalConfig) -> Result<Self, MetalError> {
        let queue = autoreleasepool(|_| device.raw().newCommandQueue()).ok_or_else(|| {
            MetalError::Command("the device refused to create a command queue".to_owned())
        })?;
        if config.debug_labels {
            queue.setLabel(Some(&NSString::from_str("fllm")));
        }
        Ok(Self {
            queue,
            device: device.clone(),
            settings: RecorderSettings::from_config(config),
        })
    }

    /// The `MTLCommandQueue` for direct API calls.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    /// A new recorder (one per stream).
    pub fn recorder(&self) -> Recorder {
        Recorder::new(
            self.queue.clone(),
            self.settings,
            DeviceLimits::of(&self.device),
        )
    }
}

/// Counters of what a recorder did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RecorderStats {
    pub dispatches: u64,
    pub barriers: u64,
    pub blits: u64,
    pub command_buffers: u64,
    /// Sum of `GPUEndTime - GPUStartTime` of the finished command buffers.
    pub gpu_seconds: f64,
}

struct OpenBuffer {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    /// Dispatches and blits encoded into this command buffer.
    operations: usize,
}

/// Records dispatches into command buffers. One thread at a time (`&mut self`).
pub struct Recorder {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    settings: RecorderSettings,
    limits: DeviceLimits,
    open: Option<OpenBuffer>,
    in_flight: VecDeque<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    tracker: HazardTracker,
    last_pipeline: u64,
    stats: RecorderStats,
    explicit_barriers: u64,
}

// SAFETY: a recorder owns its command buffers and encoders exclusively. Metal forbids using one command buffer or
// encoder from two threads at the same time, not moving it between threads; every method takes `&mut self`, so the
// borrow checker rules out concurrent use. (The objects are retained references, released on whichever thread drops.)
unsafe impl Send for Recorder {}

fn command_error(what: &str) -> MetalError {
    MetalError::Command(what.to_owned())
}

/// `len` bytes from the start of `range` lie inside both the range and its buffer.
fn range_ok(range: &BufferRef, len: usize) -> bool {
    len <= range.len
        && range
            .offset
            .checked_add(len)
            .is_some_and(|end| end <= range.buffer.len())
}

fn size(dims: [usize; 3]) -> MTLSize {
    MTLSize {
        width: dims[0],
        height: dims[1],
        depth: dims[2],
    }
}

impl Recorder {
    fn new(
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        settings: RecorderSettings,
        limits: DeviceLimits,
    ) -> Self {
        Self {
            queue,
            settings,
            limits,
            open: None,
            in_flight: VecDeque::new(),
            tracker: HazardTracker::new(settings.hazard_window),
            last_pipeline: 0,
            stats: RecorderStats::default(),
            explicit_barriers: 0,
        }
    }

    /// Statistics so far (finished command buffers only for `gpu_seconds`).
    pub fn stats(&self) -> RecorderStats {
        RecorderStats {
            barriers: self.tracker.barriers() + self.explicit_barriers,
            ..self.stats
        }
    }

    /// Dispatches and blits encoded into the command buffer that is not yet committed.
    pub fn pending_operations(&self) -> usize {
        self.open.as_ref().map_or(0, |open| open.operations)
    }

    /// Nothing recorded, nothing in flight.
    pub fn is_idle(&self) -> bool {
        self.open.as_ref().is_none_or(|open| open.operations == 0) && self.in_flight.is_empty()
    }

    fn ensure_command_buffer(&mut self) -> Result<(), MetalError> {
        if self.open.is_some() {
            return Ok(());
        }
        let command_buffer = autoreleasepool(|_| {
            let command_buffer = self.queue.commandBuffer();
            if let (Some(buffer), true) = (&command_buffer, self.settings.labels) {
                buffer.setLabel(Some(&NSString::from_str("fllm step")));
            }
            command_buffer
        })
        .ok_or_else(|| command_error("the queue refused a command buffer"))?;
        self.open = Some(OpenBuffer {
            command_buffer,
            encoder: None,
            operations: 0,
        });
        Ok(())
    }

    fn ensure_encoder(&mut self) -> Result<(), MetalError> {
        self.ensure_command_buffer()?;
        let open = self
            .open
            .as_mut()
            .ok_or_else(|| command_error("no open command buffer"))?;
        if open.encoder.is_some() {
            return Ok(());
        }
        let dispatch_type = if self.settings.serial_dispatch {
            MTLDispatchType::Serial
        } else {
            MTLDispatchType::Concurrent
        };
        let encoder = autoreleasepool(|_| {
            open.command_buffer
                .computeCommandEncoderWithDispatchType(dispatch_type)
        })
        .ok_or_else(|| command_error("the command buffer refused a compute encoder"))?;
        open.encoder = Some(encoder);
        self.tracker.reset();
        self.last_pipeline = 0;
        Ok(())
    }

    fn end_encoder(&mut self) {
        if let Some(open) = self.open.as_mut() {
            if let Some(encoder) = open.encoder.take() {
                encoder.endEncoding();
            }
        }
        self.tracker.reset();
        self.last_pipeline = 0;
    }

    fn encoder(&self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>, MetalError> {
        self.open
            .as_ref()
            .and_then(|open| open.encoder.as_deref())
            .ok_or_else(|| command_error("no open compute encoder"))
    }

    /// Start a dispatch of `pipeline`; bind its arguments on the returned builder and finish with
    /// [`DispatchBuilder::threadgroups`] or [`DispatchBuilder::threads`].
    pub fn begin(&mut self, pipeline: &Pipeline) -> Result<DispatchBuilder<'_>, MetalError> {
        self.ensure_encoder()?;
        if self.last_pipeline != pipeline.id() {
            self.encoder()?.setComputePipelineState(pipeline.raw());
            self.last_pipeline = pipeline.id();
        }
        Ok(DispatchBuilder {
            recorder: self,
            max_threads: pipeline.max_threads(),
            error: None,
        })
    }

    /// Order everything encoded so far before everything encoded from now on (buffer scope). Dispatches need no explicit
    /// call: their declared accesses decide. Use it around work whose accesses were not declared.
    pub fn barrier(&mut self) -> Result<(), MetalError> {
        if self.encoder().is_ok() && !self.settings.serial_dispatch {
            self.encoder()?
                .memoryBarrierWithScope(MTLBarrierScope::Buffers);
            self.explicit_barriers += 1;
        }
        self.tracker.reset();
        Ok(())
    }

    /// Open a named group in GPU captures (no-op unless `FLLM_METAL_DEBUG=1`). Pair with [`Recorder::pop_group`].
    pub fn push_group(&mut self, label: &str) -> Result<(), MetalError> {
        if self.settings.labels {
            self.ensure_encoder()?;
            self.encoder()?.pushDebugGroup(&NSString::from_str(label));
        }
        Ok(())
    }

    pub fn pop_group(&mut self) -> Result<(), MetalError> {
        if self.settings.labels && self.encoder().is_ok() {
            self.encoder()?.popDebugGroup();
        }
        Ok(())
    }

    fn note_operation(&mut self) {
        if let Some(open) = self.open.as_mut() {
            open.operations += 1;
        }
        let limit = self.settings.dispatches_per_command_buffer;
        if limit > 0 && self.pending_operations() >= limit {
            self.commit_open();
        }
    }

    /// Copy `len` bytes between buffers on the GPU (a blit; ends the compute encoder, a new one opens on demand).
    pub fn copy_buffer(
        &mut self,
        src: &BufferRef,
        dst: &BufferRef,
        len: usize,
    ) -> Result<(), MetalError> {
        if !range_ok(src, len) || !range_ok(dst, len) {
            return Err(MetalError::Buffer(format!(
                "copy of {len} bytes from a {}-byte range to a {}-byte range",
                src.len, dst.len
            )));
        }
        if len == 0 {
            return Ok(());
        }
        self.end_encoder();
        self.ensure_command_buffer()?;
        let open = self
            .open
            .as_ref()
            .ok_or_else(|| command_error("no open command buffer"))?;
        autoreleasepool(|_| {
            let blit = open
                .command_buffer
                .blitCommandEncoder()
                .ok_or_else(|| command_error("the command buffer refused a blit encoder"))?;
            // SAFETY: both ranges were checked against the lengths of their `BufferRef`s, which lie inside the buffers.
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src.buffer.raw(),
                    src.offset,
                    dst.buffer.raw(),
                    dst.offset,
                    len,
                );
            }
            blit.endEncoding();
            Ok::<(), MetalError>(())
        })?;
        self.stats.blits += 1;
        self.note_operation();
        Ok(())
    }

    /// Set the range to zero bytes on the GPU.
    pub fn fill_zero(&mut self, dst: &BufferRef) -> Result<(), MetalError> {
        if dst.len == 0 {
            return Ok(());
        }
        if !range_ok(dst, dst.len) {
            return Err(MetalError::Buffer(
                "fill range lies outside its buffer".to_owned(),
            ));
        }
        self.end_encoder();
        self.ensure_command_buffer()?;
        let open = self
            .open
            .as_ref()
            .ok_or_else(|| command_error("no open command buffer"))?;
        autoreleasepool(|_| {
            let blit = open
                .command_buffer
                .blitCommandEncoder()
                .ok_or_else(|| command_error("the command buffer refused a blit encoder"))?;
            blit.fillBuffer_range_value(
                dst.buffer.raw(),
                objc2_foundation::NSRange {
                    location: dst.offset,
                    length: dst.len,
                },
                0,
            );
            blit.endEncoding();
            Ok::<(), MetalError>(())
        })?;
        self.stats.blits += 1;
        self.note_operation();
        Ok(())
    }

    /// End encoding and commit the open command buffer; the GPU starts on it. Does not wait.
    fn commit_open(&mut self) {
        self.end_encoder();
        if let Some(open) = self.open.take() {
            if open.operations == 0 {
                return; // nothing was encoded; an uncommitted empty buffer is simply dropped
            }
            autoreleasepool(|_| open.command_buffer.commit());
            self.in_flight.push_back(open.command_buffer);
            self.stats.command_buffers += 1;
        }
    }

    /// Check the oldest command buffers: pop the finished ones (all of them with `wait_all`, and beyond
    /// `MAX_IN_FLIGHT` waiting for the oldest). The first failure is returned, later ones are still drained.
    fn reap(&mut self, wait_all: bool) -> Result<(), MetalError> {
        let mut first_error: Option<MetalError> = None;
        autoreleasepool(|_| {
            while let Some(front) = self.in_flight.front() {
                let status = front.status();
                let finished = status == MTLCommandBufferStatus::Completed
                    || status == MTLCommandBufferStatus::Error;
                if !finished {
                    if !wait_all && self.in_flight.len() <= MAX_IN_FLIGHT {
                        break;
                    }
                    front.waitUntilCompleted();
                }
                if let Some(done) = self.in_flight.pop_front() {
                    if let Err(error) = check_finished(&done, &mut self.stats) {
                        first_error.get_or_insert(error);
                    }
                }
            }
        });
        first_error.map_or(Ok(()), Err)
    }

    /// Commit what is recorded and return without waiting. Reports a GPU error of an earlier, already finished command
    /// buffer if there was one.
    pub fn flush(&mut self) -> Result<(), MetalError> {
        self.commit_open();
        self.reap(false)
    }

    /// Commit and wait until the GPU has finished everything. After this, buffers can be read and written by the CPU.
    pub fn synchronize(&mut self) -> Result<(), MetalError> {
        self.commit_open();
        self.reap(true)
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // an encoder must be ended before it is released; the recorded work is committed, not waited for
        self.commit_open();
    }
}

fn check_finished(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    stats: &mut RecorderStats,
) -> Result<(), MetalError> {
    let status = command_buffer.status();
    if status == MTLCommandBufferStatus::Completed {
        let seconds = command_buffer.GPUEndTime() - command_buffer.GPUStartTime();
        if seconds > 0.0 {
            stats.gpu_seconds += seconds;
        }
        return Ok(());
    }
    let message = command_buffer.error().map_or_else(
        || format!("status {}", status.0),
        |error| error.localizedDescription().to_string(),
    );
    Err(MetalError::Gpu(message))
}

/// Arguments of one dispatch under construction. Created by [`Recorder::begin`]; every binding call declares at the
/// same time how the kernel uses the buffer, which is what the barrier logic works from.
pub struct DispatchBuilder<'r> {
    recorder: &'r mut Recorder,
    max_threads: usize,
    error: Option<MetalError>,
}

impl Drop for DispatchBuilder<'_> {
    fn drop(&mut self) {
        // a builder dropped without dispatching must not leave its declared accesses behind
        self.recorder.tracker.discard_pending();
    }
}

impl DispatchBuilder<'_> {
    fn fail(&mut self, error: MetalError) -> &mut Self {
        self.error.get_or_insert(error);
        self
    }

    /// Bind `buffer` at `[[buffer(index)]]`, used by the kernel as `access`.
    pub fn buffer(&mut self, index: u32, buffer: &BufferRef, access: Access) -> &mut Self {
        if index >= MAX_BIND_SLOTS {
            return self.fail(MetalError::InvalidDispatch(format!(
                "buffer index {index} is out of range (limit {MAX_BIND_SLOTS})"
            )));
        }
        if buffer
            .offset
            .checked_add(buffer.len)
            .is_none_or(|end| end > buffer.buffer.len())
        {
            return self.fail(MetalError::InvalidDispatch(format!(
                "buffer range {}+{} outside buffer #{}",
                buffer.offset,
                buffer.len,
                buffer.buffer.id()
            )));
        }
        let encoder = match self.recorder.encoder() {
            Ok(encoder) => encoder,
            Err(error) => return self.fail(error),
        };
        // SAFETY: `index` is below Metal's argument limit and the range lies inside the buffer (both checked); the buffer
        // stays alive through the `Arc` in `BufferRef` and Metal retains it for the command buffer.
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(buffer.buffer.raw()),
                buffer.offset,
                index as usize,
            )
        };
        self.recorder.tracker.declare(
            buffer.buffer.id(),
            buffer.offset,
            buffer.offset + buffer.len,
            access,
        );
        self
    }

    /// Bind a buffer the kernel only reads.
    pub fn read(&mut self, index: u32, buffer: &BufferRef) -> &mut Self {
        self.buffer(index, buffer, Access::Read)
    }

    /// Bind a buffer the kernel only writes.
    pub fn write(&mut self, index: u32, buffer: &BufferRef) -> &mut Self {
        self.buffer(index, buffer, Access::Write)
    }

    /// Bind a buffer the kernel reads and writes.
    pub fn read_write(&mut self, index: u32, buffer: &BufferRef) -> &mut Self {
        self.buffer(index, buffer, Access::ReadWrite)
    }

    /// Pass a small constant block by value (`constant Params &p [[buffer(index)]]`). Use a `#[repr(C)]` type without
    /// padding that matches the MSL struct.
    pub fn bytes<T: Copy>(&mut self, index: u32, value: &T) -> &mut Self {
        // SAFETY: `value` is a valid reference, so its `size_of::<T>()` bytes can be read; the callee only reads them.
        let raw = unsafe {
            std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
        };
        self.bytes_raw(index, raw)
    }

    /// Pass raw bytes (at most 4096) at `[[buffer(index)]]`.
    pub fn bytes_raw(&mut self, index: u32, data: &[u8]) -> &mut Self {
        if index >= MAX_BIND_SLOTS || data.len() > MAX_INLINE_BYTES || data.is_empty() {
            return self.fail(MetalError::InvalidDispatch(format!(
                "inline data of {} bytes at index {index} (limits: 1..={MAX_INLINE_BYTES} bytes, index < {MAX_BIND_SLOTS})",
                data.len()
            )));
        }
        let encoder = match self.recorder.encoder() {
            Ok(encoder) => encoder,
            Err(error) => return self.fail(error),
        };
        // SAFETY: the pointer is valid for `data.len()` bytes, which is within Metal's 4 KiB inline limit; Metal copies it.
        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::from(data).cast::<c_void>(),
                data.len(),
                index as usize,
            )
        };
        self
    }

    /// Reserve `bytes` of threadgroup memory for the `threadgroup` argument at `[[threadgroup(index)]]`.
    pub fn threadgroup_memory(&mut self, index: u32, bytes: usize) -> &mut Self {
        let limit = self.recorder.limits.max_threadgroup_memory;
        if index >= MAX_BIND_SLOTS
            || bytes == 0
            || (limit > 0 && bytes.next_multiple_of(16) > limit)
        {
            return self.fail(MetalError::InvalidDispatch(format!(
                "threadgroup memory of {bytes} bytes at index {index} (device limit {limit} bytes)"
            )));
        }
        let encoder = match self.recorder.encoder() {
            Ok(encoder) => encoder,
            Err(error) => return self.fail(error),
        };
        // SAFETY: the length is a multiple of 16 (rounded up here), the index is within range.
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(bytes.next_multiple_of(16), index as usize)
        };
        self
    }

    /// Bind the address table at `[[buffer(index)]]` and make the buffers its addresses point to resident. The access
    /// counts as touching every buffer for barrier purposes.
    pub fn table(&mut self, index: u32, table: &AddressTable, access: Access) -> &mut Self {
        self.buffer(index, table.buffer(), Access::Read);
        let encoder = match self.recorder.encoder() {
            Ok(encoder) => encoder,
            Err(error) => return self.fail(error),
        };
        let usage = if access == Access::Read {
            MTLResourceUsage::Read
        } else {
            MTLResourceUsage::Read | MTLResourceUsage::Write
        };
        for target in table.targets() {
            encoder.useResource_usage(target.as_resource(), usage);
        }
        self.recorder.tracker.declare_all(access);
        self
    }

    fn check_pending_error(&mut self) -> Result<(), MetalError> {
        match self.error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn encode(
        &mut self,
        grid: [usize; 3],
        threads: [usize; 3],
        by_threads: bool,
    ) -> Result<(), MetalError> {
        self.check_pending_error()?;
        let limits = self.recorder.limits;
        let verdict =
            geometry::validate_dispatch(grid, threads, self.max_threads, limits.max_threads)
                .map_err(MetalError::InvalidDispatch)?;
        if verdict == Validated::Empty {
            self.recorder.tracker.discard_pending();
            return Ok(());
        }
        let recorder = &mut *self.recorder;
        if recorder.settings.serial_dispatch {
            recorder.tracker.reset();
        } else if recorder.tracker.commit_dispatch() {
            recorder
                .encoder()?
                .memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        let encoder = recorder.encoder()?;
        if by_threads {
            encoder.dispatchThreads_threadsPerThreadgroup(size(grid), size(threads));
        } else {
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(grid), size(threads));
        }
        recorder.stats.dispatches += 1;
        recorder.note_operation();
        Ok(())
    }
    /// Dispatch `groups` threadgroups of `threads` threads each.
    pub fn threadgroups(
        &mut self,
        groups: [usize; 3],
        threads: [usize; 3],
    ) -> Result<(), MetalError> {
        self.encode(groups, threads, false)
    }

    /// Dispatch a grid of `total` threads cut into groups of `per_group` (the last group may be partial; the kernel must
    /// guard its indices).
    pub fn threads(&mut self, total: [usize; 3], per_group: [usize; 3]) -> Result<(), MetalError> {
        self.encode(total, per_group, true)
    }
}

/// Everything the integration layer needs for one Metal device, created together.
pub struct MetalContext {
    pub device: MetalDevice,
    pub config: MetalConfig,
    pub memory: Arc<MemoryManager>,
    pub pipelines: Arc<PipelineCache>,
    pub queue: Arc<CommandQueue>,
    pub mapped: Arc<MappedRegistry>,
    pub residency: Option<Arc<ResidencyManager>>,
}

impl MetalContext {
    /// Open device `index` (`--device metal:N`) with the settings of the environment.
    pub fn new(index: usize) -> Result<Self, MetalError> {
        Self::with_config(index, MetalConfig::from_env())
    }

    pub fn with_config(index: usize, config: MetalConfig) -> Result<Self, MetalError> {
        let device = device_info::open(index)?;
        let memory = Arc::new(MemoryManager::new(&device, &config));
        let pipelines = Arc::new(PipelineCache::new(&device, &config)?);
        let queue = Arc::new(CommandQueue::new(&device, &config)?);
        let residency = ResidencyManager::new(&device, &queue, &config).map(Arc::new);
        Ok(Self {
            device,
            config,
            memory,
            pipelines,
            queue,
            mapped: Arc::new(MappedRegistry::new()),
            residency,
        })
    }

    /// A new recorder (one per stream).
    pub fn recorder(&self) -> Recorder {
        self.queue.recorder()
    }

    /// Wrap host memory (the memory-mapped model file) without a copy, register it for weight lookups
    /// ([`MemoryManager::buffer_for_host_slice`]) and add its windows to the residency set.
    ///
    /// # Safety
    ///
    /// See [`MappedRegion::new`]: the memory must stay mapped and unchanged as long as `owner` lives.
    pub unsafe fn register_host_mapping(
        &self,
        owner: Arc<dyn Any + Send + Sync>,
        ptr: *const u8,
        len: usize,
        max_item_len: usize,
    ) -> Result<(), MetalError> {
        // SAFETY: forwarded to the caller.
        let region = unsafe { MappedRegion::new(&self.memory, owner, ptr, len, max_item_len)? };
        let region = self.mapped.register(region);
        if let Some(residency) = &self.residency {
            residency.add_many(region.buffers());
            residency.commit();
        }
        Ok(())
    }
}

// The integration layer shares the context between threads and moves recorders into streams (`Backend: Send + Sync`,
// `Stream: Send`); fail here, not in the integration, if a change breaks that.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}
    fn assertions() {
        assert_send_sync::<MetalContext>();
        assert_send_sync::<Arc<super::buffers::MetalBuffer>>();
        assert_send_sync::<BufferRef>();
        assert_send_sync::<Pipeline>();
        assert_send_sync::<AddressTable>();
        assert_send::<Recorder>();
    }
    let _ = assertions;
};
