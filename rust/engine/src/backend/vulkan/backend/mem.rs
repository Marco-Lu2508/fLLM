//! Device-memory handles of the Vulkan backend: how a [`DeviceBuffer`] maps to a Vulkan [`Buffer`], how big weights are cut into
//! storage-buffer sized slices, and how the per-layer windows of the KV pools are bound.

use crate::backend::vulkan::memory::Buffer;
use crate::backend::vulkan::weights::{rows_per_slice, DeviceLayout};
use crate::backend::{BackendError, DeviceBuffer, QuantType};
use std::sync::Arc;

/// What `DeviceBuffer::handle` holds for every buffer of this backend.
pub(crate) struct VkMem {
    /// The allocation (the first chunk for a split KV pool).
    pub buffer: Buffer,
    /// KV pools that exceed `maxBufferSize` are split into one allocation per group of layers (empty otherwise).
    pub chunks: Vec<Buffer>,
    /// Layers per entry of `chunks`.
    pub layers_per_chunk: usize,
    /// Mixture-of-experts tables carry the arena they index.
    pub experts: Option<ExpertInfo>,
}

impl VkMem {
    pub fn plain(buffer: Buffer) -> Self {
        Self {
            buffer,
            chunks: Vec::new(),
            layers_per_chunk: 0,
            experts: None,
        }
    }
}

/// The arena behind a table of `upload_pointer_table` / `upload_bias_table` (see `experts.rs`).
#[derive(Clone)]
pub(crate) struct ExpertInfo {
    /// All experts (planar device layout, back to back) or all their biases (floats).
    pub arena: Buffer,
    pub quant: QuantType,
    pub rows: usize,
    pub cols: usize,
    /// Word offsets of the planes 1..3 inside one expert (`DeviceLayout::plane_word_offset[1..4]`).
    pub planes: [usize; 3],
    pub count: usize,
    pub is_bias: bool,
}

fn not_vulkan() -> BackendError {
    BackendError::BufferError("the buffer does not belong to the Vulkan backend".into())
}

pub(crate) fn mem_of(db: &DeviceBuffer) -> Result<&VkMem, BackendError> {
    db.handle.downcast_ref::<VkMem>().ok_or_else(not_vulkan)
}

/// The Vulkan view of the visible window of `db`.
pub(crate) fn view_of(db: &DeviceBuffer) -> Result<Buffer, BackendError> {
    let mem = mem_of(db)?;
    if !mem.chunks.is_empty() {
        return Err(BackendError::UnsupportedOperation(
            "this KV pool is split over several allocations and cannot be used as one buffer".into(),
        ));
    }
    let size = if db.size == 0 {
        16.min(mem.buffer.size().saturating_sub(db.offset as u64))
    } else {
        db.size as u64
    };
    mem.buffer.slice(db.offset as u64, size).map_err(BackendError::from)
}

/// The expert arena behind a table buffer.
pub(crate) fn expert_info(db: &DeviceBuffer) -> Result<&ExpertInfo, BackendError> {
    mem_of(db)?
        .experts
        .as_ref()
        .ok_or_else(|| BackendError::BufferError("not a table made by upload_pointer_table / upload_bias_table".into()))
}

/// Alignment of the slices of a weight inside its buffer and of the scratch allocations.
pub(crate) fn slice_align(min_storage_buffer_offset_alignment: u64) -> u64 {
    min_storage_buffer_offset_alignment.max(256)
}

/// Layout family of a weight for [`slice_table`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WeightKind {
    /// Planar device layout of `weights::layout`.
    Quant(QuantType),
    /// Row-major float matrix (F32 4 bytes, F16 / BF16 2 bytes per element).
    Float { elem_bytes: usize },
}

impl WeightKind {
    /// `None` for formats that have to be converted first.
    pub fn of(quant: QuantType) -> Option<Self> {
        match quant {
            QuantType::F32 => Some(Self::Float { elem_bytes: 4 }),
            QuantType::F16 | QuantType::BF16 => Some(Self::Float { elem_bytes: 2 }),
            q if crate::backend::vulkan::weights::format_info(q).is_some() => Some(Self::Quant(q)),
            _ => None,
        }
    }
}

/// One group of consecutive rows of a weight that is bound (and dispatched) on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SliceInfo {
    /// First row of the slice in the whole weight (the `row_base` of its dispatch).
    pub row_base: usize,
    pub rows: usize,
    /// Byte offset of the slice in the weight's buffer (a multiple of the alignment passed to [`slice_table`]).
    pub offset: usize,
    /// Bytes of the slice (the size of its binding).
    pub bytes: usize,
}

/// Cuts a `rows x cols` weight into slices of at most `max_range` bytes. Used identically by the upload and by every dispatch, so the
/// buffer layout never has to be stored. Most weights are a single slice at offset 0.
pub(crate) fn slice_table(kind: WeightKind, rows: usize, cols: usize, max_range: u64, align: u64) -> Result<Vec<SliceInfo>, BackendError> {
    let max_range = max_range.min(u64::from(u32::MAX)) as usize;
    let per = match kind {
        WeightKind::Quant(q) => rows_per_slice(q, cols, max_range)?,
        WeightKind::Float { elem_bytes } => (max_range / (cols * elem_bytes).max(1)).max(1),
    };
    let align = align.max(1) as usize;
    let mut out = Vec::new();
    let (mut first, mut cursor) = (0usize, 0usize);
    while first < rows {
        let n = per.min(rows - first);
        let bytes = match kind {
            WeightKind::Quant(q) => DeviceLayout::new(q, n, cols)?.total_bytes,
            WeightKind::Float { elem_bytes } => (n * cols * elem_bytes).next_multiple_of(4),
        };
        cursor = cursor.next_multiple_of(align);
        out.push(SliceInfo {
            row_base: first,
            rows: n,
            offset: cursor,
            bytes,
        });
        cursor += bytes;
        first += n;
    }
    Ok(out)
}

/// Bytes of the buffer that holds all `slices`.
pub(crate) fn table_bytes(slices: &[SliceInfo]) -> usize {
    slices.last().map_or(16, |s| (s.offset + s.bytes).next_multiple_of(16))
}

/// The window of layer `layer` of a KV pool (`words_per_layer` 16-bit words per layer) as a storage-buffer binding, with the layer's start
/// in 16-bit words relative to the window (the `key_base` / `value_base` of `ops::attention::KvLayout`). Binding one layer at a time keeps
/// every range far below `maxStorageBufferRange` and lets pools larger than 4 GiB work.
pub(crate) fn pool_window(pool: &DeviceBuffer, layer: usize, words_per_layer: usize, align: u64) -> Result<(Buffer, usize), BackendError> {
    let mem = mem_of(pool)?;
    let (base, local) = if mem.chunks.is_empty() {
        (view_of(pool)?, layer)
    } else {
        let chunk = layer / mem.layers_per_chunk.max(1);
        let buffer = mem
            .chunks
            .get(chunk)
            .ok_or_else(|| BackendError::InvalidShape(format!("layer {layer} is outside the split KV pool")))?;
        (buffer.clone(), layer % mem.layers_per_chunk.max(1))
    };
    let bytes = (words_per_layer * 2) as u64;
    let start = local as u64 * bytes;
    if start + bytes > base.size() {
        return Err(BackendError::InvalidShape(format!("KV layer {layer} lies outside its pool")));
    }
    let len = bytes.next_multiple_of(4).min(base.size() - start);
    let (window, skew) = base.aligned_view(start, len, align.max(1))?;
    Ok((window, skew as usize / 2))
}

/// `Arc<VkMem>` as the type-erased handle of a [`DeviceBuffer`].
pub(crate) fn handle(mem: VkMem) -> Arc<VkMem> {
    Arc::new(mem)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_slice_when_it_fits() {
        let t = slice_table(WeightKind::Float { elem_bytes: 4 }, 100, 64, 1 << 30, 256).unwrap();
        assert_eq!(
            t,
            vec![SliceInfo {
                row_base: 0,
                rows: 100,
                offset: 0,
                bytes: 100 * 64 * 4
            }]
        );
        let q = slice_table(WeightKind::Quant(QuantType::Q4_0), 128, 256, 1 << 30, 256).unwrap();
        assert_eq!(q.len(), 1);
        assert_eq!(q[0].bytes, DeviceLayout::new(QuantType::Q4_0, 128, 256).unwrap().total_bytes);
    }

    #[test]
    fn slices_cover_all_rows_and_respect_range_and_alignment() {
        for (kind, cols) in [
            (WeightKind::Quant(QuantType::Q4_K), 512usize),
            (WeightKind::Quant(QuantType::Q8_0), 96),
            (WeightKind::Quant(QuantType::Q6_K), 256),
            (WeightKind::Float { elem_bytes: 2 }, 33),
            (WeightKind::Float { elem_bytes: 4 }, 128),
        ] {
            for rows in [1usize, 63, 64, 65, 1000, 5001] {
                let range = 200_000u64;
                let t = slice_table(kind, rows, cols, range, 256).unwrap();
                assert_eq!(t.iter().map(|s| s.rows).sum::<usize>(), rows, "{kind:?} {rows}");
                let mut next_row = 0;
                let mut end = 0;
                for s in &t {
                    assert_eq!(s.row_base, next_row);
                    assert_eq!(s.offset % 256, 0);
                    assert!(s.offset >= end);
                    next_row += s.rows;
                    end = s.offset + s.bytes;
                }
                let min_rows = if matches!(kind, WeightKind::Quant(_)) { 64 } else { 1 };
                for s in &t {
                    // a slice never exceeds the range unless a single minimal group of rows already does
                    assert!(
                        s.bytes as u64 <= range || s.rows <= min_rows,
                        "{kind:?} rows {rows}: slice of {} bytes",
                        s.bytes
                    );
                }
                assert!(table_bytes(&t) >= end);
                assert_eq!(table_bytes(&t) % 16, 0);
            }
        }
    }

    #[test]
    fn weight_kinds() {
        assert_eq!(WeightKind::of(QuantType::F16), Some(WeightKind::Float { elem_bytes: 2 }));
        assert_eq!(WeightKind::of(QuantType::BF16), Some(WeightKind::Float { elem_bytes: 2 }));
        assert_eq!(WeightKind::of(QuantType::F32), Some(WeightKind::Float { elem_bytes: 4 }));
        assert_eq!(WeightKind::of(QuantType::Q4_K), Some(WeightKind::Quant(QuantType::Q4_K)));
        assert_eq!(WeightKind::of(QuantType::IQ2_XXS), None);
        assert_eq!(WeightKind::of(QuantType::Q8_K), None);
    }
}
