//! Per-step bump allocators of a stream: the small host arrays and the scratch buffers of the operations recorded since the last
//! `synchronize`. Nothing is handed out twice before the reset, so the CPU never overwrites bytes an earlier recorded dispatch still reads
//! and no scratch buffer is reused while a recorded dispatch may still use it.

use crate::backend::metal::buffers::{Arena, BufferRef, BufferUsage, MemoryManager, SUBALLOCATION_ALIGN};
use crate::backend::metal::device_info::MetalError;

/// Chunks are rounded up to this many bytes.
const CHUNK_GRANULE: usize = 1 << 16;

pub(super) struct StepArena {
    usage: BufferUsage,
    first_chunk: usize,
    chunks: Vec<Arena>,
}

impl StepArena {
    /// An arena of `usage` buffers; nothing is allocated before the first request, the first chunk has `first_chunk` bytes.
    pub(super) fn new(usage: BufferUsage, first_chunk: usize) -> Self {
        Self {
            usage,
            first_chunk,
            chunks: Vec::new(),
        }
    }

    /// `len` bytes (at least 16), aligned to 256. A new chunk (twice the size of the previous one) is allocated when the last one is full.
    pub(super) fn alloc(
        &mut self,
        memory: &MemoryManager,
        len: usize,
    ) -> Result<BufferRef, MetalError> {
        let len = len.max(16);
        if let Some(chunk) = self.chunks.last_mut() {
            if let Ok(range) = chunk.alloc(len) {
                return Ok(range);
            }
        }
        let grown = self
            .chunks
            .last()
            .map_or(self.first_chunk, |chunk| chunk.capacity().saturating_mul(2));
        let capacity = grown
            .max(len.saturating_add(SUBALLOCATION_ALIGN))
            .next_multiple_of(CHUNK_GRANULE);
        self.chunks.push(Arena::new(memory, capacity, self.usage)?);
        match self.chunks.last_mut() {
            Some(chunk) => chunk.alloc(len),
            None => Err(MetalError::Buffer("arena chunk vanished".into())),
        }
    }

    /// Bytes handed out since the last reset (an upper bound: the full cursor of every chunk).
    pub(super) fn footprint(&self) -> usize {
        self.chunks.iter().map(Arena::used).sum()
    }

    /// Makes everything available again. The newest (largest) chunk is kept, the older ones are released.
    pub(super) fn reset(&mut self) {
        let keep = self.chunks.pop();
        self.chunks.clear();
        if let Some(mut chunk) = keep {
            chunk.reset();
            self.chunks.push(chunk);
        }
    }
}
