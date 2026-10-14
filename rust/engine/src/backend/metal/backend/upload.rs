//! `Backend::upload_weight` of the Metal backend: device images of the weights.
//!
//! * Formats with a Metal kernel (`weights::format_info`) go through `weights::pack_weight`: 21 of them are the bytes of the GGUF tensor
//!   unchanged (borrowed; the bytes are wrapped without a copy when the loader registered its memory map, else copied once into a shared
//!   buffer), MXFP4 and the Q4Z family are re-sorted into two planes, the permutation table of act-order weights is split off.
//! * MXFP4 under the name of a routed expert (`blk.N.expK.*`) stays in the raw 17-byte block layout, which is what the `moe_linear_mxfp4`
//!   kernel reads (the pointer tables of `upload_pointer_table` point at it).
//! * Every other format (IQ1 / IQ2 / IQ3, Q8_1, Q8_K, device-only layouts, ...) is decoded on the host and stored as F16 (as F32 if a value
//!   does not fit), with the per-tensor scale folded in. That costs memory but keeps the model running; it is logged once per format.

use super::convert::{
    is_expert_name, mxfp4_raw_len, to_f16_bytes, to_f32_bytes, u32_bytes,
};
use super::MetalBackend;
use crate::backend::metal::buffers::{BufferKind, BufferRef, BufferUsage};
use crate::backend::metal::weights::{format_info, pack_weight, WeightError};
use crate::backend::{log_once, BackendError, DeviceWeight, QuantType};
use std::borrow::Cow;

fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

fn weight_error(name: &str, error: WeightError) -> BackendError {
    match error {
        WeightError::UnsupportedFormat(quant) => BackendError::UnsupportedQuant(quant),
        other => invalid(format!("{name}: {other}")),
    }
}

impl MetalBackend {
    /// A copy of `bytes` in a new shared buffer of long-lived data (weights, tables), added to the residency set.
    fn owned_buffer(&self, bytes: &[u8]) -> Result<BufferRef, BackendError> {
        let buffer = self.shared.ctx.memory.from_bytes(bytes, BufferUsage::Weights)?;
        self.shared.mark_resident(&buffer);
        Ok(BufferRef::whole(buffer))
    }

    /// The device range for host bytes that are a weight image: a window of the registered memory map if `bytes` lies in one (no copy) and its
    /// offset meets the kernels' alignment, else a copy.
    fn weight_buffer(&self, bytes: &[u8], align: usize) -> Result<BufferRef, BackendError> {
        let ctx = &self.shared.ctx;
        let view = ctx.memory.buffer_for_host_slice(&ctx.mapped, bytes)?;
        if view.offset.is_multiple_of(align) {
            if view.buffer.kind() != BufferKind::Mapped {
                self.shared.mark_resident(&view.buffer);
            }
            return Ok(view);
        }
        self.owned_buffer(bytes)
    }

    /// Bias vectors and other small float arrays.
    pub(super) fn float_buffer(&self, values: &[f32]) -> Result<BufferRef, BackendError> {
        self.owned_buffer(&to_f32_bytes(values))
    }

    /// `Backend::upload_weight_scaled` (`upload_weight` is the case `scale == 1.0`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn upload_weight_impl(
        &self,
        name: &str,
        raw: &[u8],
        quant: QuantType,
        rows: usize,
        cols: usize,
        bias: Option<&[f32]>,
        scale: f32,
    ) -> Result<DeviceWeight, BackendError> {
        if rows == 0 || cols == 0 {
            return Err(invalid(format!("{name}: empty {rows} x {cols} weight")));
        }
        if let Some(bias) = bias {
            if bias.len() != rows {
                return Err(invalid(format!(
                    "{name}: bias of {} values for {rows} rows",
                    bias.len()
                )));
            }
        }
        let mut weight = if quant == QuantType::Mxfp4 && is_expert_name(name) {
            self.upload_raw_mxfp4(name, raw, rows, cols, scale)?
        } else if format_info(quant).is_some() {
            self.upload_native(name, raw, quant, rows, cols, scale)?
        } else {
            self.upload_converted(name, raw, quant, rows, cols, scale)?
        };
        if let Some(bias) = bias {
            weight.bias = Some(self.shared.wrap(self.float_buffer(bias)?));
        }
        Ok(weight)
    }

    fn upload_native(
        &self,
        name: &str,
        raw: &[u8],
        quant: QuantType,
        rows: usize,
        cols: usize,
        scale: f32,
    ) -> Result<DeviceWeight, BackendError> {
        let packed = pack_weight(quant, rows, cols, raw).map_err(|e| weight_error(name, e))?;
        // dense rows are read as 16-byte vectors; every other format needs 4-byte alignment (INTERFACE_QUANT.md)
        let align = if matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16) {
            16
        } else {
            4
        };
        let buffer = match &packed.bytes {
            Cow::Borrowed(bytes) => self.weight_buffer(bytes, align)?,
            Cow::Owned(bytes) => self.owned_buffer(bytes)?,
        };
        let perm = match &packed.perm {
            Some(table) => Some(self.shared.wrap(self.owned_buffer(&u32_bytes(table))?)),
            None => None,
        };
        Ok(DeviceWeight {
            name: name.to_owned(),
            buffer: self.shared.wrap(buffer),
            quant_type: quant,
            rows,
            cols,
            bias: None,
            scale,
            perm,
        })
    }

    fn upload_raw_mxfp4(
        &self,
        name: &str,
        raw: &[u8],
        rows: usize,
        cols: usize,
        scale: f32,
    ) -> Result<DeviceWeight, BackendError> {
        if mxfp4_raw_len(rows, cols) != Some(raw.len()) {
            return Err(invalid(format!(
                "{name}: {} bytes are not a {rows} x {cols} MXFP4 tensor",
                raw.len()
            )));
        }
        let buffer = self.weight_buffer(raw, 4)?;
        self.shared.register_raw_mxfp4(&buffer);
        Ok(DeviceWeight {
            name: name.to_owned(),
            buffer: self.shared.wrap(buffer),
            quant_type: QuantType::Mxfp4,
            rows,
            cols,
            bias: None,
            scale,
            perm: None,
        })
    }

    /// Formats without a kernel: decode on the host, store as F16 (F32 when a value is out of range), the tensor scale folded in.
    fn upload_converted(
        &self,
        name: &str,
        raw: &[u8],
        quant: QuantType,
        rows: usize,
        cols: usize,
        scale: f32,
    ) -> Result<DeviceWeight, BackendError> {
        let mut values = crate::compute::dequantize(raw, quant.to_gguf_type_id(), rows * cols)
            .map_err(|e| {
                BackendError::UnsupportedOperation(format!(
                    "{name}: {quant:?} weights have no Metal kernel and cannot be decoded on the host either ({e})"
                ))
            })?;
        if values.len() != rows * cols {
            return Err(invalid(format!(
                "{name}: decoding {quant:?} gave {} values for {rows} x {cols}",
                values.len()
            )));
        }
        if scale != 1.0 {
            for v in &mut values {
                *v *= scale;
            }
        }
        let (bytes, target) = match to_f16_bytes(&values) {
            Some(bytes) => (bytes, QuantType::F16),
            None => (to_f32_bytes(&values), QuantType::F32),
        };
        log_once(
            &format!("metal-convert-{quant:?}"),
            &format!(
                "{quant:?} weights have no Metal kernel and are stored as {target:?} (first: {name}); this needs more memory"
            ),
        );
        self.upload_native(name, &bytes, target, rows, cols, 1.0)
    }
}
