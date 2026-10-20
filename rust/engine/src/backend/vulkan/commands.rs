//! Native Vulkan backend (ash): command recording, submission and the timeline semaphore. Owner: package 1 (Vulkan core).
//!
//! - [`Timeline`]: one timeline semaphore per device. Every submission signals the next value, so "wait for submission N" is
//!   `timeline.wait(N)`; there are no fences. Submissions happen in order on one queue.
//! - [`Recorder`]: records compute work into primary command buffers and submits it. One submit per decode step is the intended use:
//!   `begin` (implicit) ... many `dispatch` ... `submit`. The recorder owns a command pool and a descriptor-pool ring that are reset
//!   automatically once everything the recorder submitted has completed, so steady-state recording allocates nothing.
//! - Synchronisation is conservative and automatic: a global memory barrier is inserted between a dispatch / copy and the next dependent
//!   one ([`Recorder::dispatch_independent`] skips it), every command buffer starts with a barrier against everything submitted earlier, and
//!   `submit` ends with a barrier to the host so that mapped results can be read after `wait`.
//! - Lifetime: the recorder keeps a reference to every buffer and pipeline it records until the timeline passes the submission, so
//!   dropping a `Buffer` right after `submit` is safe.
//!
//! Level of verification: L0 (compiles; the pure parts are unit tested, the Vulkan calls never ran).

pub mod profile;
pub mod transfer;

use super::instance::error::{vk_call, VkError, VkResult};
use super::instance::{DeviceCore, VkContext};
use super::memory::{Buffer, BufferInner};
use super::pipeline::ComputePipeline;
use ash::vk;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub use transfer::Transfer;
use profile::{Profiler, Recorded};

/// Most storage buffers one pipeline can bind (descriptor arrays are on the stack).
pub const MAX_BINDINGS: u32 = 32;
const MAX_BINDINGS_USIZE: usize = MAX_BINDINGS as usize;
/// Longest `vkCmdUpdateBuffer`.
pub const MAX_INLINE_UPDATE: usize = 65536;

static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);

/// The device's timeline semaphore.
pub struct Timeline {
    core: Arc<DeviceCore>,
    semaphore: vk::Semaphore,
    /// Last value handed to a submission (written under the queue lock).
    submitted: AtomicU64,
    /// Highest value seen complete (a lower bound of the semaphore's counter, to avoid driver calls).
    completed: AtomicU64,
}

impl Timeline {
    pub(crate) fn new(core: Arc<DeviceCore>) -> VkResult<Self> {
        let mut ty = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut ty);
        // SAFETY: plain object creation; `info` and `ty` live through the call.
        let semaphore = unsafe { core.device.create_semaphore(&info, None) }.map_err(vk_call("vkCreateSemaphore"))?;
        Ok(Self {
            core,
            semaphore,
            submitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
        })
    }

    /// The value of the most recent submission (0: nothing submitted yet).
    pub fn last_submitted(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }

    /// The highest value the GPU has reached.
    pub fn completed(&self) -> VkResult<u64> {
        // SAFETY: the semaphore is valid for the lifetime of `self`.
        let value =
            unsafe { self.core.device.get_semaphore_counter_value(self.semaphore) }.map_err(vk_call("vkGetSemaphoreCounterValue"))?;
        self.completed.fetch_max(value, Ordering::AcqRel);
        Ok(self.completed.load(Ordering::Acquire))
    }

    /// Submission `value` has finished (cheap when it is known to be).
    pub fn is_complete(&self, value: u64) -> VkResult<bool> {
        if value <= self.completed.load(Ordering::Acquire) {
            return Ok(true);
        }
        Ok(self.completed()? >= value)
    }

    /// Wait up to `timeout` for submission `value`. `Ok(false)` on timeout.
    pub fn wait_timeout(&self, value: u64, timeout: Duration) -> VkResult<bool> {
        if self.is_complete(value)? {
            return Ok(true);
        }
        if value > self.last_submitted() {
            return Err(VkError::InvalidArgument(format!(
                "waiting for submission {value} but only {} were submitted",
                self.last_submitted()
            )));
        }
        let semaphores = [self.semaphore];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
        // SAFETY: the wait info points to local arrays that outlive the call.
        match unsafe {
            self.core
                .device
                .wait_semaphores(&info, timeout.as_nanos().min(u128::from(u64::MAX)) as u64)
        } {
            Ok(()) => {
                self.completed.fetch_max(value, Ordering::AcqRel);
                Ok(true)
            }
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(VkError::from_result("vkWaitSemaphores", e)),
        }
    }

    /// Wait for submission `value` (up to ten minutes, which only a hung GPU reaches: then it is reported as device loss).
    pub fn wait(&self, value: u64) -> VkResult<()> {
        if self.wait_timeout(value, Duration::from_secs(600))? {
            Ok(())
        } else {
            Err(VkError::DeviceLost)
        }
    }

    /// Submit command buffers; returns the timeline value that signals their completion.
    pub(crate) fn submit(&self, command_buffers: &[vk::CommandBuffer]) -> VkResult<u64> {
        let _guard = self.core.submit_lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let value = self.submitted.load(Ordering::Acquire) + 1;
        let values = [value];
        let semaphores = [self.semaphore];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let info = vk::SubmitInfo::default()
            .command_buffers(command_buffers)
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline);
        // SAFETY: the queue is externally synchronised by `submit_lock`; the command buffers are fully recorded.
        unsafe { self.core.device.queue_submit(self.core.queue, &[info], vk::Fence::null()) }.map_err(vk_call("vkQueueSubmit"))?;
        self.submitted.store(value, Ordering::Release);
        Ok(value)
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        // SAFETY: created by `new`; the device is idle or about to be (VkContext::drop waits for the queue first).
        unsafe { self.core.device.destroy_semaphore(self.semaphore, None) };
    }
}

/// Descriptor pools for devices (or pipelines) without push descriptors. Pools are reset together with the command pool.
struct DescriptorRing {
    pools: Vec<vk::DescriptorPool>,
    current: usize,
}

const RING_SETS: u32 = 1024;
const RING_DESCRIPTORS: u32 = 8192;

impl DescriptorRing {
    fn allocate(&mut self, device: &ash::Device, layout: vk::DescriptorSetLayout) -> VkResult<vk::DescriptorSet> {
        let layouts = [layout];
        loop {
            if self.current == self.pools.len() {
                let sizes = [vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(RING_DESCRIPTORS)];
                let info = vk::DescriptorPoolCreateInfo::default().max_sets(RING_SETS).pool_sizes(&sizes);
                // SAFETY: plain creation; `info` and `sizes` live through the call.
                let pool = unsafe { device.create_descriptor_pool(&info, None) }.map_err(vk_call("vkCreateDescriptorPool"))?;
                self.pools.push(pool);
            }
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pools[self.current])
                .set_layouts(&layouts);
            // SAFETY: the pool belongs to this ring, which the recorder uses from one thread.
            match unsafe { device.allocate_descriptor_sets(&info) } {
                Ok(sets) => return Ok(sets[0]),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => self.current += 1,
                Err(e) => return Err(VkError::from_result("vkAllocateDescriptorSets", e)),
            }
        }
    }

    fn reset(&mut self, device: &ash::Device) {
        for &pool in &self.pools {
            // SAFETY: no set of this pool is in use any more (the caller checked the timeline).
            let _ = unsafe { device.reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty()) };
        }
        self.current = 0;
    }

    fn destroy(&mut self, device: &ash::Device) {
        for pool in self.pools.drain(..) {
            // SAFETY: as above.
            unsafe { device.destroy_descriptor_pool(pool, None) };
        }
        self.current = 0;
    }
}

/// What a submission keeps alive until the GPU has finished it.
struct Parked {
    value: u64,
    prof: Option<Recorded>,
    _buffers: Vec<Arc<BufferInner>>,
    _pipelines: Vec<Arc<ComputePipeline>>,
}

/// Counters for checking that a step does what it should (few submits, few barriers).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecorderStats {
    pub dispatches: u64,
    pub barriers: u64,
    pub submits: u64,
    pub copies: u64,
    pub descriptor_sets: u64,
}

/// Records and submits compute work. Not `Sync`; use one recorder per stream or thread (`&mut self` methods). `Send`.
pub struct Recorder {
    ctx: Arc<VkContext>,
    pool: vk::CommandPool,
    cbs: Vec<vk::CommandBuffer>,
    /// Command buffers handed out since the last pool reset.
    next_cb: usize,
    current: Option<vk::CommandBuffer>,
    /// Timeline value of this recorder's latest submission.
    last_value: u64,
    retained_buffers: Vec<Arc<BufferInner>>,
    retained_pipelines: Vec<Arc<ComputePipeline>>,
    parked: VecDeque<Parked>,
    ring: DescriptorRing,
    epoch: u64,
    /// Writes were recorded since the last barrier.
    pending_writes: bool,
    pipeline: Option<Arc<ComputePipeline>>,
    buffers_bound: bool,
    stats: RecorderStats,
    /// GPU timing per shader (`FLLM_VK_PROFILE`); `cur_prof` holds the timestamps of the open command buffer.
    prof: Option<Profiler>,
    cur_prof: Option<Recorded>,
}

impl Recorder {
    pub fn new(ctx: &Arc<VkContext>) -> VkResult<Self> {
        let core = ctx.core();
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(core.queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        // SAFETY: plain creation on a valid device.
        let pool = unsafe { core.device.create_command_pool(&info, None) }.map_err(vk_call("vkCreateCommandPool"))?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            pool,
            cbs: Vec::new(),
            next_cb: 0,
            current: None,
            last_value: 0,
            retained_buffers: Vec::new(),
            retained_pipelines: Vec::new(),
            parked: VecDeque::new(),
            ring: DescriptorRing {
                pools: Vec::new(),
                current: 0,
            },
            epoch: NEXT_EPOCH.fetch_add(1, Ordering::Relaxed),
            pending_writes: false,
            pipeline: None,
            buffers_bound: false,
            stats: RecorderStats::default(),
            prof: Profiler::new(ctx),
            cur_prof: None,
        })
    }

    fn device(&self) -> &ash::Device {
        &self.ctx.core().device
    }

    pub fn stats(&self) -> RecorderStats {
        self.stats
    }

    /// Timeline value of the latest submission of this recorder (0: none yet).
    pub fn last_submission(&self) -> u64 {
        self.last_value
    }

    /// Commands are being recorded (`begin` was called or a recording call started one, and `submit` has not ended it).
    pub fn is_recording(&self) -> bool {
        self.current.is_some()
    }

    /// Drop the references of finished submissions; reset the pools when all of this recorder's submissions are done.
    fn reclaim(&mut self) -> VkResult<()> {
        let completed = self.ctx.timeline().completed()?;
        while self.parked.front().is_some_and(|p| p.value <= completed) {
            if let Some(p) = self.parked.pop_front() {
                if let (Some(pr), Some(rec)) = (&mut self.prof, &p.prof) {
                    pr.collect(&self.ctx.core().device, rec)?;
                }
            }
        }
        if self.current.is_none() && self.next_cb > 0 && self.last_value <= completed {
            // SAFETY: every command buffer of the pool has completed (timeline), none is recording.
            unsafe {
                self.ctx
                    .core()
                    .device
                    .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
            }
            .map_err(vk_call("vkResetCommandPool"))?;
            self.ring.reset(&self.ctx.core().device);
            self.next_cb = 0;
        }
        Ok(())
    }

    /// Start recording a command buffer (a no-op while one is open). The recording methods call this themselves.
    pub fn begin(&mut self) -> VkResult<()> {
        if self.current.is_some() {
            return Ok(());
        }
        self.reclaim()?;
        let ctx = Arc::clone(&self.ctx);
        let device = &ctx.core().device;
        let cb = if let Some(&cb) = self.cbs.get(self.next_cb) {
            cb
        } else {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: the pool belongs to this recorder.
            let cb = unsafe { device.allocate_command_buffers(&info) }.map_err(vk_call("vkAllocateCommandBuffers"))?[0];
            self.cbs.push(cb);
            cb
        };
        self.next_cb += 1;
        let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is in the initial state (fresh or after the pool reset).
        unsafe { device.begin_command_buffer(cb, &begin) }.map_err(vk_call("vkBeginCommandBuffer"))?;
        self.current = Some(cb);
        // order this buffer after everything submitted before it (the first scope of a barrier covers earlier submissions in queue order)
        Self::full_barrier(device, cb);
        self.pending_writes = false;
        self.pipeline = None;
        self.buffers_bound = false;
        self.cur_prof = None;
        if let Some(pr) = &mut self.prof {
            if self.parked.iter().filter(|p| p.prof.is_some()).count() < 3 {
                let base = pr.start(device, cb);
                self.cur_prof = Some(Recorded { base, used: 1, names: Vec::new() });
            }
        }
        self.stats.barriers += 1;
        Ok(())
    }

    fn full_barrier(device: &ash::Device, cb: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::INDIRECT_COMMAND_READ,
            );
        // SAFETY: `cb` is recording.
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    fn cb(&mut self) -> VkResult<vk::CommandBuffer> {
        self.begin()?;
        self.current
            .ok_or_else(|| VkError::InvalidArgument("no command buffer is recording".into()))
    }

    fn retain(&mut self, buffer: &Buffer) {
        let inner = buffer.inner();
        if inner.retain_epoch.swap(self.epoch, Ordering::Relaxed) != self.epoch {
            self.retained_buffers.push(Arc::clone(inner));
        }
    }

    /// Insert a barrier before the next command if writes are pending (what every dependent command does automatically).
    fn flush_pending(&mut self) -> VkResult<()> {
        let cb = self.cb()?;
        if self.pending_writes {
            Self::full_barrier(self.device(), cb);
            self.pending_writes = false;
            self.stats.barriers += 1;
        }
        Ok(())
    }

    /// Insert a global memory barrier now (compute and transfer writes -> compute and transfer reads and writes), whether or not
    /// writes are pending. Only needed to separate [`Recorder::dispatch_independent`] groups from what follows.
    pub fn memory_barrier(&mut self) -> VkResult<()> {
        let cb = self.cb()?;
        Self::full_barrier(self.device(), cb);
        self.pending_writes = false;
        self.stats.barriers += 1;
        Ok(())
    }

    /// Select the pipeline for the following dispatches. Buffers must be set again afterwards.
    pub fn bind_pipeline(&mut self, pipeline: &Arc<ComputePipeline>) -> VkResult<()> {
        let cb = self.cb()?;
        if self.pipeline.as_ref().is_some_and(|p| Arc::ptr_eq(p, pipeline)) {
            self.buffers_bound = false;
            return Ok(());
        }
        // SAFETY: `cb` is recording; the pipeline is kept alive until the submission completes.
        unsafe { self.device().cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline.raw()) };
        self.retained_pipelines.push(Arc::clone(pipeline));
        self.pipeline = Some(Arc::clone(pipeline));
        self.buffers_bound = false;
        Ok(())
    }

    /// Push constants of the bound pipeline, starting at offset 0. `data.len()` is a multiple of 4 and at most the pipeline's size.
    pub fn push_constants(&mut self, data: &[u8]) -> VkResult<()> {
        let cb = self.cb()?;
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Err(VkError::InvalidArgument("push_constants before bind_pipeline".into()));
        };
        if !data.len().is_multiple_of(4) || data.len() > pipeline.push_constant_bytes() as usize {
            return Err(VkError::InvalidArgument(format!(
                "pipeline '{}': {} bytes of push constants (declared {}, must be a multiple of 4)",
                pipeline.name(),
                data.len(),
                pipeline.push_constant_bytes()
            )));
        }
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: `cb` is recording; the range lies inside the declared push constant range.
        unsafe {
            self.ctx
                .core()
                .device
                .cmd_push_constants(cb, pipeline.raw_layout(), vk::ShaderStageFlags::COMPUTE, 0, data)
        };
        Ok(())
    }

    /// Push constants given as 32-bit words (every constant of the shaders is 4 bytes wide).
    pub fn push_words(&mut self, words: &[u32]) -> VkResult<()> {
        let mut bytes = [0u8; 256];
        if words.len() * 4 > bytes.len() {
            return Err(VkError::InvalidArgument("more than 64 push constant words".into()));
        }
        for (i, word) in words.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        self.push_constants(&bytes[..words.len() * 4])
    }

    /// Bind `buffers` as storage buffers 0..n of the bound pipeline; `n` must equal the pipeline's binding count. Every view must start at
    /// a multiple of `minStorageBufferOffsetAlignment` and fit in `maxStorageBufferRange` (use [`Buffer::aligned_view`] otherwise).
    pub fn set_buffers(&mut self, buffers: &[&Buffer]) -> VkResult<()> {
        let cb = self.cb()?;
        let Some(pipeline) = self.pipeline.clone() else {
            return Err(VkError::InvalidArgument("set_buffers before bind_pipeline".into()));
        };
        if buffers.len() != pipeline.bindings() as usize {
            return Err(VkError::InvalidArgument(format!(
                "pipeline '{}' declares {} storage buffers, {} given",
                pipeline.name(),
                pipeline.bindings(),
                buffers.len()
            )));
        }
        let caps = self.ctx.caps();
        let mut infos = [vk::DescriptorBufferInfo::default(); MAX_BINDINGS_USIZE];
        for (i, buffer) in buffers.iter().enumerate() {
            let info = buffer.descriptor_info();
            check_binding(
                info.offset,
                info.range,
                caps.min_storage_buffer_offset_alignment,
                caps.max_storage_buffer_range,
            )
            .map_err(|why| VkError::InvalidArgument(format!("pipeline '{}' binding {i}: {why}", pipeline.name())))?;
            infos[i] = info;
        }
        for buffer in buffers {
            self.retain(buffer);
        }
        let n = buffers.len();
        let mut writes: [vk::WriteDescriptorSet<'_>; MAX_BINDINGS_USIZE] = std::array::from_fn(|_| vk::WriteDescriptorSet::default());
        for i in 0..n {
            writes[i] = vk::WriteDescriptorSet::default()
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[i]));
        }
        if n > 0 {
            if pipeline.uses_push_descriptors() {
                let push = self
                    .ctx
                    .core()
                    .push_descriptor
                    .as_ref()
                    .ok_or_else(|| VkError::Unsupported("push descriptors vanished".into()))?;
                // SAFETY: `cb` is recording; the writes point to `infos`, alive for the call.
                unsafe { push.cmd_push_descriptor_set(cb, vk::PipelineBindPoint::COMPUTE, pipeline.raw_layout(), 0, &writes[..n]) };
            } else {
                let ctx = Arc::clone(&self.ctx);
                let device = &ctx.core().device;
                let set = self.ring.allocate(device, pipeline.raw_set_layout())?;
                for write in &mut writes[..n] {
                    write.dst_set = set;
                }
                // SAFETY: the set was just allocated from this recorder's ring; `cb` is recording.
                unsafe {
                    device.update_descriptor_sets(&writes[..n], &[]);
                    device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pipeline.raw_layout(), 0, &[set], &[]);
                }
                self.stats.descriptor_sets += 1;
            }
        }
        self.buffers_bound = true;
        Ok(())
    }

    /// Timestamp after the dispatch just recorded (profiling only).
    fn stamp(&mut self, cb: vk::CommandBuffer) {
        if let (Some(pr), Some(rec)) = (&self.prof, &mut self.cur_prof) {
            if rec.used < profile::SLOTS {
                pr.stamp(&self.ctx.core().device, cb, rec.base + rec.used);
                rec.used += 1;
                rec.names.push(self.pipeline.as_ref().map_or_else(String::new, |p| p.name().to_string()));
            }
        }
    }

    fn check_ready(&self) -> VkResult<()> {
        match &self.pipeline {
            None => Err(VkError::InvalidArgument("dispatch before bind_pipeline".into())),
            Some(p) if p.bindings() > 0 && !self.buffers_bound => Err(VkError::InvalidArgument(format!(
                "pipeline '{}': dispatch before set_buffers",
                p.name()
            ))),
            Some(_) => Ok(()),
        }
    }

    fn check_group_counts(&self, x: u32, y: u32, z: u32) -> VkResult<()> {
        let max = self.ctx.caps().max_workgroup_count;
        if x > max[0] || y > max[1] || z > max[2] {
            return Err(VkError::InvalidArgument(format!(
                "dispatch {x}x{y}x{z} exceeds the device limit {max:?}"
            )));
        }
        Ok(())
    }

    /// Dispatch `x * y * z` workgroups of the bound pipeline. Waits (by a barrier) for all earlier writes of this command buffer.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> VkResult<()> {
        self.dispatch_impl(x, y, z, true)
    }

    /// Like [`Recorder::dispatch`] but without the barrier before it: it must not read what a dispatch or copy since the last barrier
    /// wrote (two projections that read the same input, for example). Its own writes are covered by the next dependent command.
    pub fn dispatch_independent(&mut self, x: u32, y: u32, z: u32) -> VkResult<()> {
        self.dispatch_impl(x, y, z, false)
    }

    fn dispatch_impl(&mut self, x: u32, y: u32, z: u32, dependent: bool) -> VkResult<()> {
        let cb = self.cb()?;
        self.check_ready()?;
        self.check_group_counts(x, y, z)?;
        if x == 0 || y == 0 || z == 0 {
            return Ok(());
        }
        if dependent {
            self.flush_pending()?;
        }
        // SAFETY: `cb` is recording with a pipeline and descriptors bound.
        unsafe { self.device().cmd_dispatch(cb, x, y, z) };
        self.stamp(cb);
        self.pending_writes = true;
        self.stats.dispatches += 1;
        Ok(())
    }

    /// Dispatch with the group counts read from `args` at `offset` (three `u32`); the buffer needs [`BufferUsage::INDIRECT`](super::memory::BufferUsage).
    pub fn dispatch_indirect(&mut self, args: &Buffer, offset: u64) -> VkResult<()> {
        let cb = self.cb()?;
        self.check_ready()?;
        if !offset.is_multiple_of(4) || offset + 12 > args.size() {
            return Err(VkError::InvalidArgument(
                "indirect arguments must be 4-aligned and fit in the buffer".into(),
            ));
        }
        self.flush_pending()?;
        self.retain(args);
        // SAFETY: `cb` is recording; the buffer is retained.
        unsafe { self.device().cmd_dispatch_indirect(cb, args.handle(), args.offset() + offset) };
        self.stamp(cb);
        self.pending_writes = true;
        self.stats.dispatches += 1;
        Ok(())
    }

    /// Copy `size` bytes from `src` (at `src_offset` of the view) to `dst` (at `dst_offset`). The ranges must not overlap.
    pub fn copy_buffer(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, size: u64) -> VkResult<()> {
        if size == 0 {
            return Ok(());
        }
        if src_offset.checked_add(size).is_none_or(|e| e > src.size()) || dst_offset.checked_add(size).is_none_or(|e| e > dst.size()) {
            return Err(VkError::InvalidArgument(format!(
                "copy of {size} bytes out of range (src {}, dst {})",
                src.size(),
                dst.size()
            )));
        }
        let cb = self.cb()?;
        self.flush_pending()?;
        self.retain(src);
        self.retain(dst);
        let region = vk::BufferCopy {
            src_offset: src.offset() + src_offset,
            dst_offset: dst.offset() + dst_offset,
            size,
        };
        // SAFETY: `cb` is recording; both buffers are retained; offsets are inside the views.
        unsafe { self.device().cmd_copy_buffer(cb, src.handle(), dst.handle(), &[region]) };
        self.pending_writes = true;
        self.stats.copies += 1;
        Ok(())
    }

    /// Fill `size` bytes of `dst` at `offset` with the 32-bit `value` (offset and size multiples of 4).
    pub fn fill(&mut self, dst: &Buffer, offset: u64, size: u64, value: u32) -> VkResult<()> {
        if size == 0 {
            return Ok(());
        }
        if !offset.is_multiple_of(4) || !size.is_multiple_of(4) || offset.checked_add(size).is_none_or(|e| e > dst.size()) {
            return Err(VkError::InvalidArgument(format!(
                "fill {offset}+{size} must be 4-aligned and inside the buffer ({})",
                dst.size()
            )));
        }
        let cb = self.cb()?;
        self.flush_pending()?;
        self.retain(dst);
        // SAFETY: as in `copy_buffer`.
        unsafe { self.device().cmd_fill_buffer(cb, dst.handle(), dst.offset() + offset, size, value) };
        self.pending_writes = true;
        self.stats.copies += 1;
        Ok(())
    }

    /// Write up to 64 KiB of `data` into `dst` at `offset` from the command buffer (for small parameter blocks; the data is copied at record time).
    pub fn update_buffer(&mut self, dst: &Buffer, offset: u64, data: &[u8]) -> VkResult<()> {
        if data.is_empty() {
            return Ok(());
        }
        if data.len() > MAX_INLINE_UPDATE
            || !data.len().is_multiple_of(4)
            || !offset.is_multiple_of(4)
            || offset + data.len() as u64 > dst.size()
        {
            return Err(VkError::InvalidArgument(format!(
                "update_buffer of {} bytes at {offset}: must be 4-aligned, at most {MAX_INLINE_UPDATE} bytes and inside the buffer ({})",
                data.len(),
                dst.size()
            )));
        }
        let cb = self.cb()?;
        self.flush_pending()?;
        self.retain(dst);
        // SAFETY: as in `copy_buffer`.
        unsafe { self.device().cmd_update_buffer(cb, dst.handle(), dst.offset() + offset, data) };
        self.pending_writes = true;
        self.stats.copies += 1;
        Ok(())
    }

    /// End the command buffer and submit it; returns the timeline value that signals its completion. With nothing recorded this
    /// returns the previous value (and submits nothing).
    pub fn submit(&mut self) -> VkResult<u64> {
        let Some(cb) = self.current else {
            return Ok(self.last_value);
        };
        let ctx = Arc::clone(&self.ctx);
        let device = &ctx.core().device;
        // results become readable on the host once the timeline value is reached
        let to_host = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: `cb` is recording; ending it makes it executable.
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[to_host],
                &[],
                &[],
            );
            device.end_command_buffer(cb).map_err(vk_call("vkEndCommandBuffer"))?;
        }
        self.current = None;
        let value = self.ctx.timeline().submit(&[cb])?;
        self.last_value = value;
        self.stats.submits += 1;
        self.parked.push_back(Parked {
            value,
            prof: self.cur_prof.take(),
            _buffers: std::mem::take(&mut self.retained_buffers),
            _pipelines: std::mem::take(&mut self.retained_pipelines),
        });
        self.epoch = NEXT_EPOCH.fetch_add(1, Ordering::Relaxed);
        self.pending_writes = false;
        self.pipeline = None;
        self.buffers_bound = false;
        Ok(value)
    }

    /// Wait for the timeline to reach `value` (any submission of the device, not only this recorder's).
    pub fn wait(&self, value: u64) -> VkResult<()> {
        self.ctx.timeline().wait(value)
    }

    /// `submit` and wait for it; returns once the GPU is done.
    pub fn submit_and_wait(&mut self) -> VkResult<u64> {
        let value = self.submit()?;
        if value > 0 {
            self.wait(value)?;
        }
        Ok(value)
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if self.last_value > 0 {
            let _ = self.ctx.timeline().wait(self.last_value);
        }
        let device = &self.ctx.core().device;
        self.ring.destroy(device);
        if let Some(pr) = &mut self.prof {
            pr.destroy(device);
        }
        // SAFETY: all submissions of this recorder are complete; destroying the pool frees its command buffers (also a half-recorded one).
        unsafe { device.destroy_command_pool(self.pool, None) };
    }
}

/// Validate a storage-buffer descriptor: offset alignment and range. Pure, unit tested.
pub fn check_binding(offset: u64, range: u64, offset_alignment: u64, max_range: u64) -> Result<(), String> {
    if range == 0 {
        return Err("empty buffer view".into());
    }
    if offset_alignment > 1 && !offset.is_multiple_of(offset_alignment) {
        return Err(format!(
            "offset {offset} is not a multiple of minStorageBufferOffsetAlignment ({offset_alignment}); use Buffer::aligned_view"
        ));
    }
    if range > max_range {
        return Err(format!(
            "range {range} exceeds maxStorageBufferRange ({max_range}); bind a smaller slice"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_checks() {
        assert!(check_binding(0, 1024, 256, 1 << 30).is_ok());
        assert!(check_binding(512, 1, 256, 1 << 30).is_ok());
        assert!(check_binding(100, 1024, 256, 1 << 30).unwrap_err().contains("aligned_view"));
        assert!(check_binding(256, 0, 256, 1 << 30).is_err());
        assert!(check_binding(0, (1 << 30) + 1, 256, 1 << 30)
            .unwrap_err()
            .contains("maxStorageBufferRange"));
        // alignment 1: anything goes
        assert!(check_binding(7, 7, 1, 100).is_ok());
    }

    #[test]
    fn binding_limits_are_consistent() {
        assert_eq!(MAX_BINDINGS as usize, MAX_BINDINGS_USIZE);
        // a pool must be able to hold at least one maximal set
        const { assert!(RING_DESCRIPTORS >= MAX_BINDINGS) };
        const { assert!(RING_SETS > 0) };
    }
}
