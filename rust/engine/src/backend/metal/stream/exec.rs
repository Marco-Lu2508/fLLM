//! The core of the stream: pipelines, host arrays, scratch buffers, the runner of dispatch plans, and the transfer operations.

use super::arena::StepArena;
use super::MetalStream;
use crate::backend::cpu::{CpuBackend, CpuStream};
use crate::backend::metal::backend::shared::{buffer_ref, Shared};
use crate::backend::metal::buffers::{AddressTable, BufferRef, BufferUsage};
use crate::backend::metal::commands::Access;
use crate::backend::metal::ops::{Access as PlanAccess, OpPlan, SlotSource, CONSTANTS_BUFFER_INDEX};
use crate::backend::metal::pipelines::{FunctionConstant, Pipeline};
use crate::backend::metal::weights::PlanConstant;
use crate::backend::{Backend, BackendError, DeviceBuffer, Stream};
use std::collections::HashMap;
use std::sync::Arc;

/// Host arrays and scratch buffers recorded since the last `synchronize` beyond which the next operation waits for the GPU and starts over.
const ARENA_SOFT_LIMIT: usize = 256 << 20;

/// What a plan slot is bound to.
#[derive(Clone)]
pub(super) enum Bound {
    Buf(BufferRef),
    /// A table of GPU addresses (`upload_pointer_table`); binding it makes the buffers it points to resident.
    Table(Arc<AddressTable>),
}

pub(super) fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

/// A `DeviceBuffer` as a plan slot. The kernels read and write 32-bit words, so the window must start at a multiple of 4.
pub(super) fn dev(buffer: &DeviceBuffer) -> Result<Bound, BackendError> {
    let range = buffer_ref(buffer)?;
    if !range.offset.is_multiple_of(4) {
        return Err(BackendError::BufferError(format!(
            "buffer window starts at byte {}, which is not 4-byte aligned",
            range.offset
        )));
    }
    Ok(Bound::Buf(range))
}

/// The table behind a `DeviceBuffer` made by `upload_pointer_table` / `upload_bias_table`.
pub(super) fn table_of(buffer: &DeviceBuffer) -> Result<Arc<AddressTable>, BackendError> {
    Arc::clone(&buffer.handle)
        .downcast::<AddressTable>()
        .map_err(|_| {
            BackendError::BufferError(
                "expected an expert table made by upload_pointer_table / upload_bias_table".into(),
            )
        })
}

fn to_access(access: PlanAccess) -> Access {
    match access {
        PlanAccess::Read => Access::Read,
        PlanAccess::Write => Access::Write,
        PlanAccess::ReadWrite => Access::ReadWrite,
    }
}

impl MetalStream {
    pub(crate) fn new(shared: Arc<Shared>) -> Self {
        let rec = shared.ctx.recorder();
        Self {
            shared,
            rec,
            pipes: HashMap::new(),
            host_arena: StepArena::new(BufferUsage::Staging, 4 << 20),
            scratch_arena: StepArena::new(BufferUsage::Activations, 16 << 20),
            cpu: CpuBackend::new(),
            cpu_stream: CpuStream,
        }
    }

    /// Waits for the GPU when a lot of step data has piled up (called at the start of an operation, never in the middle of one).
    pub(super) fn ensure_headroom(&mut self) -> Result<(), BackendError> {
        if self.host_arena.footprint() + self.scratch_arena.footprint() > ARENA_SOFT_LIMIT {
            self.sync()?;
        }
        Ok(())
    }

    /// Commits, waits for the GPU and releases the per-step data.
    pub(super) fn sync(&mut self) -> Result<(), BackendError> {
        self.rec.synchronize()?;
        self.host_arena.reset();
        self.scratch_arena.reset();
        Ok(())
    }

    /// The pipeline of kernel `name` specialised with `constants` (resolved once per stream, then kept).
    pub(super) fn pipeline(
        &mut self,
        name: &str,
        constants: &[PlanConstant],
    ) -> Result<Arc<Pipeline>, BackendError> {
        if let Some(found) = self
            .pipes
            .get(name)
            .and_then(|list| list.iter().find(|(c, _)| c.as_slice() == constants))
        {
            return Ok(Arc::clone(&found.1));
        }
        let function_constants: Vec<FunctionConstant> = constants
            .iter()
            .map(|c| FunctionConstant::u32(c.index, c.value))
            .collect();
        let pipeline = self.shared.ctx.pipelines.get(name, &function_constants)?;
        self.pipes
            .entry(name.to_owned())
            .or_default()
            .push((constants.to_vec(), Arc::clone(&pipeline)));
        Ok(pipeline)
    }

    /// A copy of `data` in the staging arena (valid until the next `synchronize`).
    pub(super) fn host_array(&mut self, data: &[u8]) -> Result<BufferRef, BackendError> {
        let range = self
            .host_arena
            .alloc(&self.shared.ctx.memory, data.len().max(16))?;
        if !data.is_empty() {
            range.write(0, data)?;
        }
        Ok(range)
    }

    /// Uninitialised scratch storage (valid until the next `synchronize`).
    pub(super) fn scratch(&mut self, bytes: usize) -> Result<BufferRef, BackendError> {
        Ok(self
            .scratch_arena
            .alloc(&self.shared.ctx.memory, bytes.max(16))?)
    }

    /// Runs the plans of one operation in order. `named` binds the slots of source `Buffer` by name, `runs` the `RunState(n)` slots; host arrays are
    /// copied to the staging arena (one copy per distinct `Arc`), scratch slots with the same id share one buffer of the largest size asked for,
    /// absent optional inputs are bound to a dummy buffer. The declared accesses of the slots drive the barriers of the recorder.
    pub(super) fn run_plans(
        &mut self,
        plans: &[OpPlan],
        named: &[(&str, Bound)],
        runs: &[BufferRef],
    ) -> Result<(), BackendError> {
        self.shared.flush_residency();
        let mut scratch_sizes: Vec<(u8, usize)> = Vec::new();
        for plan in plans {
            for slot in &plan.slots {
                if let SlotSource::Scratch { id, bytes } = &slot.source {
                    match scratch_sizes.iter_mut().find(|(i, _)| i == id) {
                        Some(entry) => entry.1 = entry.1.max(*bytes),
                        None => scratch_sizes.push((*id, *bytes)),
                    }
                }
            }
        }
        let mut scratch_buffers: Vec<(u8, BufferRef)> = Vec::with_capacity(scratch_sizes.len());
        for (id, bytes) in scratch_sizes {
            let range = self.scratch(bytes)?;
            scratch_buffers.push((id, range));
        }
        let mut hosts: Vec<(usize, BufferRef)> = Vec::new();
        for plan in plans {
            if plan.is_empty() {
                continue;
            }
            let mut binds: Vec<(u32, Bound, Access)> = Vec::with_capacity(plan.slots.len());
            for slot in &plan.slots {
                let bound = match &slot.source {
                    SlotSource::Buffer => named
                        .iter()
                        .find(|(name, _)| *name == slot.name)
                        .map(|(_, bound)| bound.clone())
                        .ok_or_else(|| {
                            invalid(format!(
                                "{}: no buffer bound to the slot {}",
                                plan.kernel_name, slot.name
                            ))
                        })?,
                    SlotSource::Host(data) => {
                        let key = Arc::as_ptr(data) as usize;
                        let range = match hosts.iter().find(|(k, _)| *k == key) {
                            Some((_, range)) => range.clone(),
                            None => {
                                let range = self.host_array(data)?;
                                hosts.push((key, range.clone()));
                                range
                            }
                        };
                        Bound::Buf(range)
                    }
                    SlotSource::Scratch { id, .. } => scratch_buffers
                        .iter()
                        .find(|(i, _)| i == id)
                        .map(|(_, range)| Bound::Buf(range.clone()))
                        .ok_or_else(|| invalid("scratch slot without a buffer"))?,
                    SlotSource::Dummy if slot.access == PlanAccess::Read => {
                        Bound::Buf(self.shared.dummy.clone())
                    }
                    SlotSource::Dummy => Bound::Buf(self.scratch(256)?),
                    SlotSource::RunState(n) => runs
                        .get(*n)
                        .map(|range| Bound::Buf(range.clone()))
                        .ok_or_else(|| {
                            invalid(format!("{}: no state buffer for run {n}", plan.kernel_name))
                        })?,
                };
                binds.push((slot.index, bound, to_access(slot.access)));
            }
            let pipeline = self.pipeline(&plan.kernel_name, &[])?;
            let per_group: usize = plan.threads_per_group.iter().map(|&t| t as usize).product();
            if per_group > pipeline.max_threads() {
                return Err(BackendError::UnsupportedOperation(format!(
                    "{}: {per_group} threads per threadgroup planned, the pipeline allows {}",
                    plan.kernel_name,
                    pipeline.max_threads()
                )));
            }
            let mut dispatch = self.rec.begin(&pipeline)?;
            if !plan.constants.is_empty() {
                dispatch.bytes_raw(CONSTANTS_BUFFER_INDEX, &plan.constants);
            }
            for (index, bound, access) in &binds {
                match bound {
                    Bound::Buf(range) => dispatch.buffer(*index, range, *access),
                    Bound::Table(table) => dispatch.table(*index, table, *access),
                };
            }
            dispatch.threadgroups(
                plan.threadgroups.map(|g| g as usize),
                plan.threads_per_group.map(|t| t as usize),
            )?;
        }
        Ok(())
    }

    /// Writes `data` to the start of `dst` in stream order: directly while nothing is recorded or in flight, else through the staging arena
    /// and a blit (the CPU must not touch bytes an earlier dispatch may still read).
    pub(super) fn write_device(&mut self, dst: &BufferRef, data: &[u8]) -> Result<(), BackendError> {
        if data.is_empty() {
            return Ok(());
        }
        let dst = dst.slice(0, data.len())?;
        if self.rec.is_idle() {
            dst.write(0, data)?;
            return Ok(());
        }
        let staged = self.host_array(data)?;
        self.rec.copy_buffer(&staged, &dst, data.len())?;
        Ok(())
    }

    pub(super) fn op_upload(&mut self, dst: &DeviceBuffer, src: &[u8]) -> Result<(), BackendError> {
        if src.len() > dst.size {
            return Err(BackendError::BufferError(format!(
                "upload of {} bytes into a {} byte buffer",
                src.len(),
                dst.size
            )));
        }
        let range = buffer_ref(dst)?;
        self.write_device(&range, src)
    }

    pub(super) fn op_download(&mut self, dst: &mut [u8], src: &DeviceBuffer) -> Result<(), BackendError> {
        if dst.len() > src.size {
            return Err(BackendError::BufferError(format!(
                "download of {} bytes from a {} byte buffer",
                dst.len(),
                src.size
            )));
        }
        let range = buffer_ref(src)?;
        self.sync()?;
        range.read(0, dst)?;
        Ok(())
    }

    pub(super) fn op_copy(&mut self, dst: &DeviceBuffer, src: &DeviceBuffer) -> Result<(), BackendError> {
        let bytes = dst.size.min(src.size);
        if bytes == 0 {
            return Ok(());
        }
        let (from, to) = (buffer_ref(src)?, buffer_ref(dst)?);
        self.rec.copy_buffer(&from, &to, bytes)?;
        Ok(())
    }

    pub(super) fn op_zero(&mut self, buffer: &DeviceBuffer) -> Result<(), BackendError> {
        if buffer.size == 0 {
            return Ok(());
        }
        let range = buffer_ref(buffer)?;
        self.rec.fill_zero(&range)?;
        Ok(())
    }

    /// A copy of a device buffer in host memory, as a buffer of the CPU backend (for the host fallbacks).
    pub(super) fn copy_to_host(&mut self, buffer: &DeviceBuffer) -> Result<DeviceBuffer, BackendError> {
        let mut bytes = vec![0u8; buffer.size];
        self.op_download(&mut bytes, buffer)?;
        let mut host = self.cpu.allocate_buffer(buffer.size)?;
        self.cpu_stream.upload(&mut host, &bytes)?;
        Ok(host)
    }
}
