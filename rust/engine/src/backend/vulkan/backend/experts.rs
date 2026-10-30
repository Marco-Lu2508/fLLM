//! `upload_pointer_table` / `upload_bias_table` for mixture-of-experts layers on the Vulkan backend.
//!
//! The core has no buffer device addresses, so a "table of pointers" is an **arena**: the experts of one projection are copied (on the
//! device) back to back into one storage buffer, and the table is a `uint32[n_experts]` of word offsets into it (`moe_linear_*_arena`).
//! The table buffer's handle carries the arena (`ExpertInfo`), which `Stream::moe_linear` reads back. The decoder then makes the individual
//! weights slices of the arena (`alias_weights`), so the experts do not stay on the device twice.

use super::convert::u32_bytes;
use super::mem::{expert_info, view_of, ExpertInfo, VkMem};
use super::shared::Shared;
use crate::backend::vulkan::memory::BufferUsage;
use crate::backend::vulkan::ops::moe::moe_linear_supported;
use crate::backend::vulkan::weights::DeviceLayout;
use crate::backend::{BackendError, DeviceBuffer, DeviceWeight};
use std::sync::Arc;

fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

fn table_buffer(sh: &Shared, offsets: &[u32], info: ExpertInfo) -> Result<DeviceBuffer, BackendError> {
    let bytes = u32_bytes(offsets);
    let table = sh
        .ctx
        .allocator()
        .alloc_device(bytes.len().max(16).next_multiple_of(16) as u64, BufferUsage::COMPUTE)?;
    sh.upload_sync(&table, 0, &bytes)?;
    let mem = VkMem {
        buffer: table,
        chunks: Vec::new(),
        layers_per_chunk: 0,
        layer_starts: Vec::new(),
        experts: Some(info),
    };
    Ok(DeviceBuffer::new(sh.next_id(), bytes.len(), sh.device_id, Arc::new(mem)))
}

fn check_arena_size(sh: &Shared, bytes: usize, what: &str) -> Result<(), BackendError> {
    let limit = sh.caps().max_storage_buffer_range.min(u64::from(u32::MAX));
    if bytes as u64 > limit {
        return Err(BackendError::UnsupportedOperation(format!(
            "the {what} of the experts need {bytes} bytes, more than one storage buffer binding ({limit} bytes) can hold"
        )));
    }
    Ok(())
}

/// Arena + offset table for the weights of the experts of one projection (all of one shape and format, in a single slice).
pub(crate) fn pointer_table(sh: &Shared, weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
    let first = weights.first().ok_or_else(|| invalid("upload_pointer_table without experts"))?;
    let (quant, rows, cols) = (first.quant_type, first.rows, first.cols);
    if !moe_linear_supported(quant) {
        return Err(BackendError::UnsupportedQuant(quant));
    }
    let layout = DeviceLayout::new(quant, rows, cols)?;
    for w in weights {
        if w.quant_type != quant || w.rows != rows || w.cols != cols {
            return Err(invalid("the experts of one table differ in shape or format"));
        }
        if w.scale != 1.0 {
            return Err(BackendError::UnsupportedOperation("expert weights with a tensor scale".into()));
        }
        if w.buffer.size != layout.total_bytes {
            return Err(BackendError::UnsupportedOperation("expert weight stored in several slices".into()));
        }
    }
    let each = layout.total_bytes;
    // every expert starts on a storage-buffer offset boundary, so that `alias_weights` can hand the slices back as the experts' own weights
    let stride = each.next_multiple_of(sh.align() as usize);
    let total = stride * weights.len();
    check_arena_size(sh, total, "weights")?;
    let arena = sh.ctx.allocator().alloc_device(total as u64, BufferUsage::COMPUTE)?;
    let mut copies = Vec::with_capacity(weights.len());
    for (e, w) in weights.iter().enumerate() {
        copies.push((view_of(&w.buffer)?, 0u64, arena.clone(), (e * stride) as u64, each as u64));
    }
    sh.device_copy(&copies)?;
    let offsets: Vec<u32> = (0..weights.len()).map(|e| (e * stride / 4) as u32).collect();
    let info = ExpertInfo {
        arena,
        quant,
        rows,
        cols,
        planes: [
            layout.plane_word_offset[1] as usize,
            layout.plane_word_offset[2] as usize,
            layout.plane_word_offset[3] as usize,
        ],
        count: weights.len(),
        is_bias: false,
    };
    table_buffer(sh, &offsets, info)
}

/// Makes the weights the slices of the arena behind `table` (made from them by `pointer_table`): the copies they were made from are freed
/// once nothing else holds them, so the experts exist once on the device.
pub(crate) fn alias_weights(sh: &Shared, table: &DeviceBuffer, weights: &mut [&mut DeviceWeight]) -> Result<(), BackendError> {
    let info = expert_info(table)?;
    if info.is_bias || info.count != weights.len() {
        return Err(invalid("alias_weights: the table does not belong to these weights"));
    }
    let layout = DeviceLayout::new(info.quant, info.rows, info.cols)?;
    let each = layout.total_bytes;
    let stride = each.next_multiple_of(sh.align() as usize);
    let whole = sh.wrap(info.arena.clone(), stride * weights.len());
    let slices = (0..weights.len()).map(|e| whole.slice(e * stride, each)).collect::<Result<Vec<_>, _>>()?;
    for (w, s) in weights.iter_mut().zip(slices) {
        w.buffer = s;
    }
    Ok(())
}

/// Arena + float-offset table for the biases of the experts (every weight must have one).
pub(crate) fn bias_table(sh: &Shared, weights: &[&DeviceWeight]) -> Result<DeviceBuffer, BackendError> {
    let first = weights.first().ok_or_else(|| invalid("upload_bias_table without experts"))?;
    let rows = first.rows;
    let each = rows * 4;
    let total = each * weights.len();
    check_arena_size(sh, total, "biases")?;
    let arena = sh.ctx.allocator().alloc_device(total.max(16) as u64, BufferUsage::COMPUTE)?;
    let mut copies = Vec::with_capacity(weights.len());
    for (e, w) in weights.iter().enumerate() {
        let bias = w.bias.as_ref().ok_or_else(|| invalid("upload_bias_table: weight without a bias"))?;
        if w.rows != rows || bias.size < each {
            return Err(invalid("upload_bias_table: biases of different length"));
        }
        copies.push((view_of(bias)?, 0u64, arena.clone(), (e * each) as u64, each as u64));
    }
    sh.device_copy(&copies)?;
    let offsets: Vec<u32> = (0..weights.len()).map(|e| (e * rows) as u32).collect();
    let info = ExpertInfo {
        arena,
        quant: first.quant_type,
        rows,
        cols: first.cols,
        planes: [0; 3],
        count: weights.len(),
        is_bias: true,
    };
    table_buffer(sh, &offsets, info)
}
