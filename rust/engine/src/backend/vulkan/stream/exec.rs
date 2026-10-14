//! Recording machinery of [`VulkanStream`]: resolving operands to storage-buffer views (staging copies for ones that violate the device's
//! offset alignment), host arrays for the dispatches, the generic "run these plans" executor, submission and read-back.

use super::VulkanStream;
use crate::backend::vulkan::backend::mem::view_of;
use crate::backend::vulkan::memory::Buffer;
use crate::backend::vulkan::ops::{OpPlan, SlotSource};
use crate::backend::{BackendError, DeviceBuffer};
use std::sync::Arc;

/// Dispatches recorded before a submission is forced (bounds the size of one command buffer and the latency of the first result).
const MAX_DISPATCHES_PER_SUBMIT: u64 = 4096;
/// Host scratch bytes of one submission before it is forced.
const MAX_HOST_BYTES_PER_SUBMIT: u64 = 256 << 20;
/// Device scratch bytes of one submission before it is forced.
const MAX_DEVICE_BYTES_PER_SUBMIT: u64 = 1 << 30;
/// Scratch buffers an operation can ask for (`SlotSource::Scratch { id }`).
const MAX_SCRATCH_IDS: usize = 8;
/// A read-back buffer larger than this is released after use.
const READBACK_KEEP: u64 = 64 << 20;

/// An operand of an operation: the buffer a named slot of a plan refers to.
pub(super) enum Arg<'a> {
    /// A buffer of the caller (`write`: the operation modifies it).
    Dev {
        name: &'static str,
        buf: &'a DeviceBuffer,
        write: bool,
    },
    /// A view that is already a valid storage-buffer binding (a layer window of a KV pool, a slice of a weight ...).
    View { name: &'static str, buf: Buffer },
}

impl<'a> Arg<'a> {
    pub fn r(name: &'static str, buf: &'a DeviceBuffer) -> Self {
        Self::Dev { name, buf, write: false }
    }

    pub fn w(name: &'static str, buf: &'a DeviceBuffer) -> Self {
        Self::Dev { name, buf, write: true }
    }

    pub fn view(name: &'static str, buf: Buffer) -> Self {
        Self::View { name, buf }
    }
}

/// A staged copy of an unaligned operand that has to be copied back after the operation.
pub(super) struct WriteBack {
    staged: Buffer,
    dst: Buffer,
    bytes: u64,
}

/// An operand as resolved for one operation (see `resolve`).
pub(super) struct Resolved {
    /// `(allocation id, offset, size)` of the caller's `DeviceBuffer`.
    key: (u64, usize, usize),
    view: Buffer,
    /// The caller's own view when the operand had to be staged.
    original: Option<Buffer>,
    written_back: bool,
}

fn missing(plan: &OpPlan, name: &str) -> BackendError {
    BackendError::InvalidShape(format!("{}: no buffer was given for the operand '{name}'", plan.shader_name))
}

impl VulkanStream {
    /// Forget what the previous linear projection left (called by everything that is not a linear projection).
    pub(super) fn touch(&mut self) {
        self.trace = None;
    }

    /// The storage-buffer view of `db`. A view whose offset is not a multiple of `minStorageBufferOffsetAlignment` (and 16, the width of the
    /// vector loads of the shaders) cannot be bound: such an operand is copied to an aligned scratch buffer first (`Some(original view)` is
    /// returned next to the staged one). The copy is a transfer the following dispatches must wait for, so the barrier-skipping state of the
    /// projections is dropped.
    fn stage(&mut self, db: &DeviceBuffer) -> Result<(Buffer, Option<Buffer>), BackendError> {
        let view = view_of(db)?;
        let align = self.sh.caps().min_storage_buffer_offset_alignment.max(16);
        if view.offset().is_multiple_of(align) {
            return Ok((view, None));
        }
        let bytes = view.size();
        let staged = self.dev.alloc(&self.sh, bytes)?;
        self.rec.copy_buffer(&view, 0, &staged, 0, bytes)?;
        self.trace = None;
        Ok((staged, Some(view)))
    }

    /// The view of `db` for one binding; an unaligned operand is staged and, when `write` is set, copied back by [`Self::finish_writebacks`].
    pub(super) fn operand(&mut self, db: &DeviceBuffer, write: bool, wb: &mut Vec<WriteBack>) -> Result<Buffer, BackendError> {
        let (view, original) = self.stage(db)?;
        if let (true, Some(dst)) = (write, original) {
            wb.push(WriteBack {
                staged: view.clone(),
                bytes: dst.size(),
                dst,
            });
        }
        Ok(view)
    }

    /// Like [`Self::operand`] for several slots of one operation: a buffer that is named twice (a Mamba projection is both `x` and `projected`)
    /// is staged once and shared.
    fn resolve(
        &mut self,
        cache: &mut Vec<Resolved>,
        db: &DeviceBuffer,
        write: bool,
        wb: &mut Vec<WriteBack>,
    ) -> Result<Buffer, BackendError> {
        let key = (db.id, db.offset, db.size);
        if let Some(found) = cache.iter_mut().find(|r| r.key == key) {
            if let (true, false, Some(dst)) = (write, found.written_back, &found.original) {
                wb.push(WriteBack {
                    staged: found.view.clone(),
                    bytes: dst.size(),
                    dst: dst.clone(),
                });
                found.written_back = true;
            }
            return Ok(found.view.clone());
        }
        let (view, original) = self.stage(db)?;
        let mut written_back = false;
        if let (true, Some(dst)) = (write, &original) {
            wb.push(WriteBack {
                staged: view.clone(),
                bytes: dst.size(),
                dst: dst.clone(),
            });
            written_back = true;
        }
        cache.push(Resolved {
            key,
            view: view.clone(),
            original,
            written_back,
        });
        Ok(view)
    }

    /// Copy staged outputs back to where they belong.
    pub(super) fn finish_writebacks(&mut self, wb: Vec<WriteBack>) -> Result<(), BackendError> {
        if !wb.is_empty() {
            self.trace = None;
        }
        for w in wb {
            self.rec.copy_buffer(&w.staged, 0, &w.dst, 0, w.bytes)?;
        }
        Ok(())
    }

    /// A host array (positions, block tables ...) as a storage buffer that is valid until this submission has completed.
    pub(super) fn host_array(&mut self, data: &[u8]) -> Result<Buffer, BackendError> {
        let buffer = self.host.alloc(&self.sh, data.len() as u64)?;
        buffer.write(0, data)?;
        Ok(buffer)
    }

    /// Like [`Self::host_array`], but an array with the same contents as one already written in this submission (the block tables and the
    /// lengths are the same for every layer of a step) is not written again. The shaders only read these arrays.
    pub(super) fn cached_host_array(&mut self, data: &Arc<Vec<u8>>) -> Result<Buffer, BackendError> {
        if let Some((_, buffer)) = self.host_cache.iter().find(|(d, _)| Arc::ptr_eq(d, data) || **d == **data) {
            return Ok(buffer.clone());
        }
        let buffer = self.host_array(data)?;
        if self.host_cache.len() >= 8 {
            self.host_cache.remove(0);
        }
        self.host_cache.push((Arc::clone(data), buffer.clone()));
        Ok(buffer)
    }

    /// Record `plans` (in order, barriers in between unless `independent_tail`: every plan after the first one then skips the barrier before
    /// it, for plans that do not depend on each other). `args` name the buffers of the `SlotSource::Buffer` slots; `states` are the buffers of
    /// the `SlotSource::RunState` slots. A scratch slot whose name is also an argument uses that argument when it is big enough.
    pub(super) fn run_plans(
        &mut self,
        plans: &[OpPlan],
        args: &[Arg<'_>],
        states: &[&DeviceBuffer],
        independent_tail: bool,
    ) -> Result<(), BackendError> {
        self.touch();
        let mut wb: Vec<WriteBack> = Vec::new();
        let mut resolved: Vec<Resolved> = Vec::new();
        let mut named: Vec<(&'static str, Buffer)> = Vec::with_capacity(args.len());
        for arg in args {
            match arg {
                Arg::Dev { name, buf, write } => {
                    let view = self.resolve(&mut resolved, buf, *write, &mut wb)?;
                    named.push((*name, view));
                }
                Arg::View { name, buf } => named.push((name, buf.clone())),
            }
        }
        let mut state_views: Vec<Buffer> = Vec::with_capacity(states.len());
        for state in states {
            state_views.push(self.resolve(&mut resolved, state, true, &mut wb)?);
        }

        // one buffer per scratch id: the largest request of the plans
        let mut needs: [(u64, &'static str); MAX_SCRATCH_IDS] = [(0, ""); MAX_SCRATCH_IDS];
        for plan in plans {
            for slot in &plan.bindings {
                if let SlotSource::Scratch { id, bytes } = &slot.source {
                    let i = usize::from(*id);
                    if i >= MAX_SCRATCH_IDS {
                        return Err(BackendError::InvalidShape(format!(
                            "{}: scratch id {id} is out of range",
                            plan.shader_name
                        )));
                    }
                    if *bytes as u64 > needs[i].0 {
                        needs[i] = (*bytes as u64, slot.name);
                    }
                }
            }
        }
        let mut scratch: [Option<Buffer>; MAX_SCRATCH_IDS] = std::array::from_fn(|_| None);
        for (i, &(bytes, name)) in needs.iter().enumerate() {
            if bytes == 0 {
                continue;
            }
            let given = named.iter().find(|(n, b)| *n == name && b.size() >= bytes).map(|(_, b)| b.clone());
            scratch[i] = Some(match given {
                Some(buffer) => buffer,
                None => self.dev.alloc(&self.sh, bytes)?,
            });
        }

        for (index, plan) in plans.iter().enumerate() {
            let mut bufs: Vec<Buffer> = Vec::with_capacity(plan.bindings.len());
            for slot in &plan.bindings {
                let buffer = match &slot.source {
                    SlotSource::Buffer => named
                        .iter()
                        .find(|(n, _)| *n == slot.name)
                        .map(|(_, b)| b.clone())
                        .ok_or_else(|| missing(plan, slot.name))?,
                    SlotSource::Host(data) => self.cached_host_array(data)?,
                    SlotSource::Scratch { id, .. } => scratch[usize::from(*id)].clone().ok_or_else(|| missing(plan, slot.name))?,
                    SlotSource::Dummy => self.sh.dummy.clone(),
                    SlotSource::RunState(i) => state_views.get(*i).cloned().ok_or_else(|| missing(plan, slot.name))?,
                };
                bufs.push(buffer);
            }
            let pipeline = self
                .sh
                .pipeline(plan.shader_name, &[], bufs.len() as u32, plan.push_constants.len() as u32)?;
            self.rec.bind_pipeline(&pipeline)?;
            let refs: Vec<&Buffer> = bufs.iter().collect();
            self.rec.set_buffers(&refs)?;
            self.rec.push_constants(&plan.push_constants)?;
            let [x, y, z] = plan.groups;
            if independent_tail && index > 0 {
                self.rec.dispatch_independent(x, y, z)?;
            } else {
                self.rec.dispatch(x, y, z)?;
            }
        }
        self.finish_writebacks(wb)?;
        self.after_op()
    }

    /// Submit what has been recorded (if anything) and recycle the scratch frames.
    pub(super) fn flush(&mut self) -> Result<(), BackendError> {
        self.trace = None;
        if !self.rec.is_recording() {
            return Ok(());
        }
        let value = self.rec.submit()?;
        self.host_cache.clear();
        self.last_value = value;
        self.flushed_dispatches = self.rec.stats().dispatches;
        self.dev.finish_frame(&self.sh, value)?;
        self.host.finish_frame(&self.sh, value)?;
        Ok(())
    }

    /// Submit early when a command buffer or its scratch memory grows too large.
    pub(super) fn after_op(&mut self) -> Result<(), BackendError> {
        let pending = self.rec.stats().dispatches.saturating_sub(self.flushed_dispatches);
        if pending >= MAX_DISPATCHES_PER_SUBMIT
            || self.host.used() > MAX_HOST_BYTES_PER_SUBMIT
            || self.dev.used() > MAX_DEVICE_BYTES_PER_SUBMIT
        {
            self.flush()?;
        }
        Ok(())
    }

    /// Wait for everything submitted so far.
    pub(super) fn wait_idle(&mut self) -> Result<(), BackendError> {
        self.flush()?;
        if self.last_value > 0 {
            self.rec.wait(self.last_value)?;
        }
        Ok(())
    }

    /// Copy `bytes` of `src` into `out` (everything recorded so far is submitted and waited for).
    pub(super) fn read_back(&mut self, src: &Buffer, bytes: u64, out: &mut [u8]) -> Result<(), BackendError> {
        if bytes == 0 {
            return self.wait_idle();
        }
        let need = bytes.next_multiple_of(16).max(4096);
        if self.readback.as_ref().is_none_or(|b| b.size() < need) {
            // a previous read-back buffer may still be referenced by recorded commands: the recorder keeps it alive
            self.readback = Some(self.sh.ctx.allocator().alloc_readback(need.next_power_of_two())?);
        }
        let target = self
            .readback
            .clone()
            .ok_or_else(|| BackendError::BufferError("no read-back buffer".into()))?;
        self.rec.copy_buffer(src, 0, &target, 0, bytes)?;
        self.flush()?;
        self.rec.wait(self.last_value)?;
        target.read(0, &mut out[..bytes as usize])?;
        if target.size() > READBACK_KEEP {
            // a huge read-back (the logits of a long prompt) must not pin its host memory for the life of the stream
            self.readback = None;
        }
        Ok(())
    }
}
