//! State shared by the Metal backend and all of its streams: the context of the core, device capabilities for the planners, the small
//! constant buffers, and the registry of weights that are kept in a non-default layout.

use crate::backend::metal::buffers::{BufferRef, BufferUsage, MetalBuffer};
use crate::backend::metal::commands::MetalContext;
use crate::backend::metal::device_info::config::parse_bool;
use crate::backend::metal::device_info::MetalError;
use crate::backend::metal::pipelines::FunctionConstant;
use crate::backend::metal::weights::{pack_weight, MetalCapsLike, FC_NB};
use crate::backend::{log_once, BackendError, DeviceBuffer, DeviceWeight, QuantType};
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

/// Bytes of the shared dummy buffer that absent optional inputs (bias, sinks, ...) are bound to.
const DUMMY_BYTES: usize = 256;

/// Device copies of host tables by the address of their `Arc` (the `Arc` is kept alive with the copy).
type TableMap = HashMap<usize, (Arc<Vec<f32>>, BufferRef)>;

pub struct Shared {
    pub ctx: MetalContext,
    /// Ordinal of the device (`DeviceBuffer::device_id`).
    pub device_id: usize,
    /// What the planners need to know about the device: `simdgroup_matrix`, SIMD width of the probe pipeline, thread limit, `exact` mode.
    pub caps: MetalCapsLike,
    /// 256 zero bytes for inputs a kernel does not read.
    pub dummy: BufferRef,
    /// Device copies of the constant host tables (RoPE cos / sin), by the address of their `Arc`; the `Arc` is kept so that the address stays unique.
    tables: Mutex<TableMap>,
    /// MXFP4 weights kept in the raw 17-byte GGUF block layout for `moe_linear_mxfp4` (routed experts), keyed by (buffer id, offset). The value is the
    /// planar twin built on demand when such a weight is used by the matrix kernels after all.
    raw_mxfp4: Mutex<HashMap<(u64, usize), Option<BufferRef>>>,
    /// Buffers were added to the residency set since its last commit.
    residency_dirty: AtomicBool,
}

impl Shared {
    /// Opens the shared state for `ctx` and compiles three probe kernels (one per shader family), so a source the Metal compiler rejects is
    /// reported when the backend is created (the caller then falls back to wgpu) and not in the middle of a generation.
    pub fn new(ctx: MetalContext, device_id: usize) -> Result<Self, MetalError> {
        let info = ctx.device.info().clone();
        let probe = ctx.pipelines.get_plain("fllm_residual_add_f32")?;
        let simd_width = u32::try_from(probe.simd_width()).unwrap_or(32);
        ctx.pipelines.get_plain("fllm_rms_norm_f32")?;
        ctx.pipelines
            .get("fllm_q_mv_q4_0_r4", &[FunctionConstant::u32(FC_NB, 1)])?;
        let mut caps = MetalCapsLike::new(
            info.simdgroup_matrix,
            info.bfloat,
            simd_width,
            u32::try_from(info.max_threads_per_threadgroup[0]).unwrap_or(1024),
            info.has_unified_memory,
        );
        caps.exact = std::env::var("FLLM_METAL_EXACT")
            .ok()
            .and_then(|v| parse_bool(&v))
            .unwrap_or(false);
        let dummy = BufferRef::whole(ctx.memory.alloc_zeroed(DUMMY_BYTES, BufferUsage::Weights)?);
        Ok(Self {
            ctx,
            device_id,
            caps,
            dummy,
            tables: Mutex::new(HashMap::new()),
            raw_mxfp4: Mutex::new(HashMap::new()),
            residency_dirty: AtomicBool::new(false),
        })
    }

    /// The buffer range as the `DeviceBuffer` handed to the decoder.
    pub fn wrap(&self, range: BufferRef) -> DeviceBuffer {
        let handle: Arc<dyn Any + Send + Sync> = range.buffer.clone();
        DeviceBuffer {
            id: range.buffer.id(),
            offset: range.offset,
            size: range.len,
            capacity: range.buffer.len(),
            device_id: self.device_id,
            handle,
        }
    }

    /// Adds a long-lived buffer (weights, KV cache) to the residency set; the set is committed before the next dispatch.
    pub fn mark_resident(&self, buffer: &MetalBuffer) {
        if let Some(residency) = &self.ctx.residency {
            residency.add(buffer);
            self.residency_dirty.store(true, Ordering::Relaxed);
        }
    }

    /// Commits pending residency additions (cheap when there are none).
    pub fn flush_residency(&self) {
        if self.residency_dirty.swap(false, Ordering::Relaxed) {
            if let Some(residency) = &self.ctx.residency {
                residency.commit();
            }
        }
    }

    /// Device copy of a constant host table (uploaded once, recognised by the address of the `Arc`).
    pub fn table(&self, values: &Arc<Vec<f32>>) -> Result<BufferRef, BackendError> {
        let key = Arc::as_ptr(values) as usize;
        let mut tables = self.tables.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, buffer)) = tables.get(&key) {
            return Ok(buffer.clone());
        }
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = BufferRef::whole(self.ctx.memory.from_bytes(&bytes, BufferUsage::Weights)?);
        self.mark_resident(&buffer.buffer);
        tables.insert(key, (Arc::clone(values), buffer.clone()));
        Ok(buffer)
    }

    /// Remembers that the weight in `range` is an MXFP4 tensor in the raw GGUF layout.
    pub fn register_raw_mxfp4(&self, range: &BufferRef) {
        self.raw_mxfp4
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert((range.buffer.id(), range.offset), None);
    }

    /// Is this weight an MXFP4 tensor in the raw GGUF layout (uploaded under the name of a routed expert)?
    pub fn is_raw_mxfp4(&self, weight: &DeviceWeight) -> bool {
        weight.quant_type == QuantType::Mxfp4
            && self
                .raw_mxfp4
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .contains_key(&(weight.buffer.id, weight.buffer.offset))
    }

    /// The plane-sorted copy of a raw MXFP4 weight for the matrix kernels (built on the first use, then kept). Only reached when a routed expert
    /// is used outside the MoE fast path; the raw weight stays for the table.
    pub fn planar_twin(&self, weight: &DeviceWeight) -> Result<BufferRef, BackendError> {
        let key = (weight.buffer.id, weight.buffer.offset);
        let mut map = self
            .raw_mxfp4
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(Some(twin)) = map.get(&key) {
            return Ok(twin.clone());
        }
        let raw_ref = buffer_ref(&weight.buffer)?;
        let mut raw = vec![0u8; raw_ref.len];
        raw_ref.read(0, &mut raw)?;
        let packed = pack_weight(QuantType::Mxfp4, weight.rows, weight.cols, &raw)
            .map_err(|e| BackendError::InvalidShape(format!("{}: {e}", weight.name)))?;
        let twin = BufferRef::whole(self.ctx.memory.from_bytes(&packed.bytes, BufferUsage::Weights)?);
        self.mark_resident(&twin.buffer);
        log_once(
            "metal-mxfp4-twin",
            &format!(
                "MXFP4 routed-expert weight {} is used by the matrix kernels: a plane-sorted copy is made (slow path, extra memory)",
                weight.name
            ),
        );
        map.insert(key, Some(twin.clone()));
        Ok(twin)
    }
}

/// The Metal buffer range behind a `DeviceBuffer` of this backend.
pub fn buffer_ref(buffer: &DeviceBuffer) -> Result<BufferRef, BackendError> {
    let metal = Arc::clone(&buffer.handle)
        .downcast::<MetalBuffer>()
        .map_err(|_| {
            BackendError::BufferError("the buffer does not belong to the Metal backend".into())
        })?;
    if buffer
        .offset
        .checked_add(buffer.size)
        .is_none_or(|end| end > metal.len())
    {
        return Err(BackendError::BufferError(format!(
            "buffer window {}+{} lies outside its {}-byte allocation",
            buffer.offset,
            buffer.size,
            metal.len()
        )));
    }
    Ok(BufferRef {
        buffer: metal,
        offset: buffer.offset,
        len: buffer.size,
    })
}
