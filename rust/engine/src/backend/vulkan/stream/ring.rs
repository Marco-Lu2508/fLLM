//! Per-stream scratch memory: bump allocation in a ring of frames.
//!
//! A frame is everything allocated between two submissions. When a frame is submitted it is stamped with the timeline value of that
//! submission and the ring moves on to the next frame; before a frame is reused its stamp is waited for. Two rings exist per stream: device
//! memory (scratch buffers such as the int8 activations of a prepass, staged copies of unaligned operands, the partial results of the split
//! attention) and host-visible memory (small per-dispatch arrays such as positions and block tables, and the source of uploads).

use crate::backend::vulkan::backend::shared::Shared;
use crate::backend::vulkan::memory::{Buffer, BufferUsage};
use crate::backend::BackendError;

/// Frames in flight.
const FRAMES: usize = 3;
/// Chunks of at least this many bytes are freed (not kept) when their frame is recycled.
const KEEP_LIMIT: u64 = 64 << 20;
/// Largest chunk made for small requests.
const MAX_GROWTH: u64 = 128 << 20;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RingKind {
    Device,
    Host,
}

struct Chunk {
    buffer: Buffer,
    used: u64,
}

#[derive(Default)]
struct Frame {
    chunks: Vec<Chunk>,
    /// First chunk that may still have room.
    active: usize,
    /// Timeline value of the submission that used the frame (0: never submitted since the last reset).
    value: u64,
}

pub(crate) struct Ring {
    kind: RingKind,
    frames: Vec<Frame>,
    current: usize,
    min_chunk: u64,
}

/// `(offset, new_used)` of an allocation of `need` bytes in a chunk of `capacity` bytes of which `used` are taken, if it fits.
pub(crate) fn bump(capacity: u64, used: u64, need: u64) -> Option<(u64, u64)> {
    let end = used.checked_add(need)?;
    (end <= capacity).then_some((used, end))
}

impl Ring {
    pub fn new(kind: RingKind) -> Self {
        Self {
            kind,
            frames: (0..FRAMES).map(|_| Frame::default()).collect(),
            current: 0,
            min_chunk: match kind {
                RingKind::Device => 8 << 20,
                RingKind::Host => 2 << 20,
            },
        }
    }

    /// `bytes` (rounded up to the alignment of the device, at least 16) of scratch memory that stays valid until the current frame has been
    /// submitted and completed. The contents are undefined.
    pub fn alloc(&mut self, sh: &Shared, bytes: u64) -> Result<Buffer, BackendError> {
        let need = bytes.max(16).next_multiple_of(sh.align());
        if let Some(view) = self.take_from_current(need)? {
            return Ok(view);
        }
        // a new chunk is needed: first let go of the big chunks of frames that have completed in the meantime
        self.reclaim(sh);
        let frame = &mut self.frames[self.current];
        let last = frame.chunks.last().map_or(0, |c| c.buffer.size());
        let size = need.max(self.min_chunk.max(last.saturating_mul(2)).min(MAX_GROWTH));
        let buffer = match self.kind {
            RingKind::Device => sh.ctx.allocator().alloc_device(size, BufferUsage::COMPUTE)?,
            RingKind::Host => sh.ctx.allocator().alloc_staging(size)?,
        };
        let view = buffer.slice(0, need)?;
        frame.chunks.push(Chunk { buffer, used: need });
        frame.active = frame.chunks.len() - 1;
        Ok(view)
    }

    /// `need` bytes from the chunks the current frame already owns.
    fn take_from_current(&mut self, need: u64) -> Result<Option<Buffer>, BackendError> {
        let frame = &mut self.frames[self.current];
        while frame.active < frame.chunks.len() {
            let chunk = &mut frame.chunks[frame.active];
            if let Some((offset, used)) = bump(chunk.buffer.size(), chunk.used, need) {
                chunk.used = used;
                return Ok(Some(chunk.buffer.slice(offset, need)?));
            }
            frame.active += 1;
        }
        Ok(None)
    }

    /// Recycle every frame (other than the current one) whose submission has completed: big chunks are freed, the others can be reused.
    fn reclaim(&mut self, sh: &Shared) {
        for (index, frame) in self.frames.iter_mut().enumerate() {
            if index == self.current || frame.value == 0 {
                continue;
            }
            if sh.ctx.timeline().is_complete(frame.value).unwrap_or(false) {
                frame.value = 0;
                frame.active = 0;
                frame.chunks.retain(|c| c.buffer.size() < KEEP_LIMIT);
                for chunk in &mut frame.chunks {
                    chunk.used = 0;
                }
            }
        }
    }

    /// Bytes handed out in the current frame.
    pub fn used(&self) -> u64 {
        self.frames[self.current].chunks.iter().map(|c| c.used).sum()
    }

    /// The commands that use the current frame were submitted as timeline value `value`: move on to the next frame (waiting for its previous
    /// use to finish) and recycle it.
    pub fn finish_frame(&mut self, sh: &Shared, value: u64) -> Result<(), BackendError> {
        self.frames[self.current].value = value;
        self.current = (self.current + 1) % self.frames.len();
        let frame = &mut self.frames[self.current];
        if frame.value > 0 {
            sh.ctx.timeline().wait(frame.value)?;
        }
        frame.value = 0;
        frame.active = 0;
        frame.chunks.retain(|c| c.buffer.size() < KEEP_LIMIT);
        for chunk in &mut frame.chunks {
            chunk.used = 0;
        }
        self.reclaim(sh);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_allocation() {
        assert_eq!(bump(1024, 0, 256), Some((0, 256)));
        assert_eq!(bump(1024, 768, 256), Some((768, 1024)));
        assert_eq!(bump(1024, 769, 256), None);
        assert_eq!(bump(1024, u64::MAX, 2), None);
    }
}
