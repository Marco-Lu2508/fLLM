//! Expert tables of mixture-of-experts layers (`Backend::upload_pointer_table` / `upload_bias_table`).
//!
//! The tables are `AddressTable`s of the Metal core: a buffer of the 64-bit GPU addresses of the experts' weight buffers, indexed by the
//! `moe_linear_*` kernels (`ExpertAccess::Address` of `ops/moe_plan.rs`). The `DeviceBuffer` returned to the decoder carries the table
//! itself as its handle, so the stream can bind it with `DispatchBuilder::table` (which also makes the expert buffers resident).

use super::shared::buffer_ref;
use super::MetalBackend;
use crate::backend::metal::buffers::AddressTable;
use crate::backend::metal::ops::moe_plan::moe_linear_supported;
use crate::backend::{BackendError, DeviceBuffer, DeviceWeight, QuantType};
use std::any::Any;
use std::sync::Arc;

impl MetalBackend {
    /// The table of the weights (`bias == false`) or of their bias vectors (`bias == true`).
    pub(super) fn expert_table(
        &self,
        weights: &[&DeviceWeight],
        bias: bool,
    ) -> Result<DeviceBuffer, BackendError> {
        let first = weights.first().ok_or_else(|| {
            BackendError::InvalidShape("an expert table needs at least one weight".into())
        })?;
        let mut entries = Vec::with_capacity(weights.len());
        for weight in weights {
            let buffer = if bias {
                weight.bias.as_ref().ok_or_else(|| {
                    BackendError::InvalidShape(format!("{}: no bias for the bias table", weight.name))
                })?
            } else {
                if weight.quant_type != first.quant_type
                    || (weight.rows, weight.cols) != (first.rows, first.cols)
                {
                    return Err(BackendError::InvalidShape(format!(
                        "{}: experts of one table must share format and shape",
                        weight.name
                    )));
                }
                if !moe_linear_supported(weight.quant_type) {
                    return Err(BackendError::UnsupportedQuant(weight.quant_type));
                }
                if weight.quant_type == QuantType::Mxfp4 && !self.shared.is_raw_mxfp4(weight) {
                    return Err(BackendError::UnsupportedOperation(format!(
                        "{}: MXFP4 expert tables need weights uploaded under the name of a routed expert (blk.N.expK.*)",
                        weight.name
                    )));
                }
                &weight.buffer
            };
            entries.push(buffer_ref(buffer)?);
        }
        let table = AddressTable::build(&self.shared.ctx.memory, &entries)?;
        let size = entries.len() * 8;
        let id = table.buffer().buffer.id();
        let handle: Arc<dyn Any + Send + Sync> = Arc::new(table);
        Ok(DeviceBuffer {
            id,
            offset: 0,
            size,
            capacity: size,
            device_id: self.shared.device_id,
            handle,
        })
    }
}
