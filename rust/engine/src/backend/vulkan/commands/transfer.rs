//! Synchronous host <-> device copies on top of the [`Recorder`]: weight upload and result readback.
//!
//! Large uploads go through two staging buffers used alternately, so the CPU fills one while the GPU copies the other. Memory the CPU can
//! map (unified-memory devices, resizable BAR) is written directly.

use super::Recorder;
use crate::backend::vulkan::instance::error::{VkError, VkResult};
use crate::backend::vulkan::instance::VkContext;
use crate::backend::vulkan::memory::Buffer;
use std::sync::Arc;

/// Bytes of one staging buffer (two exist at most). Smaller transfers allocate less.
pub const STAGING_CHUNK: u64 = 32 << 20;

struct Staging {
    buffer: Buffer,
    /// Timeline value of the last copy that reads this buffer (0: never used).
    busy: u64,
}

/// Uploads and downloads for one device. Create one per thread that needs it; the methods block until the copy has finished.
pub struct Transfer {
    ctx: Arc<VkContext>,
    recorder: Recorder,
    staging: Vec<Staging>,
}

/// Split `len` bytes into chunks of at most `chunk` bytes: `(offset, length)` pairs. Pure, unit tested.
pub fn chunks(len: u64, chunk: u64) -> Vec<(u64, u64)> {
    let chunk = chunk.max(1);
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < len {
        let n = chunk.min(len - offset);
        out.push((offset, n));
        offset += n;
    }
    out
}

impl Transfer {
    pub fn new(ctx: &Arc<VkContext>) -> VkResult<Self> {
        Ok(Self {
            ctx: Arc::clone(ctx),
            recorder: Recorder::new(ctx)?,
            staging: Vec::new(),
        })
    }

    /// The recorder used for the copies (for statistics).
    pub fn recorder(&self) -> &Recorder {
        &self.recorder
    }

    fn ensure_staging(&mut self, chunk: u64) -> VkResult<()> {
        let needed = chunk.max(self.ctx.allocator().alignment());
        if self.staging.len() == 2 && self.staging.iter().all(|s| s.buffer.size() >= needed) {
            return Ok(());
        }
        // wait for outstanding copies before the old buffers go away (the recorder would keep them alive anyway)
        for s in &self.staging {
            if s.busy > 0 {
                self.ctx.timeline().wait(s.busy)?;
            }
        }
        self.staging.clear();
        for _ in 0..2 {
            self.staging.push(Staging {
                buffer: self.ctx.allocator().alloc_staging(needed)?,
                busy: 0,
            });
        }
        Ok(())
    }

    /// Copy `data` to `dst` at `offset` (of the view). Returns when the data is in place.
    pub fn upload(&mut self, dst: &Buffer, offset: u64, data: &[u8]) -> VkResult<()> {
        let len = data.len() as u64;
        if len == 0 {
            return Ok(());
        }
        if offset.checked_add(len).is_none_or(|end| end > dst.size()) {
            return Err(VkError::InvalidArgument(format!(
                "upload of {len} bytes at {offset} exceeds a buffer of {} bytes",
                dst.size()
            )));
        }
        if dst.is_host_visible() {
            return dst.write(offset, data);
        }
        let chunk = len.min(STAGING_CHUNK);
        self.ensure_staging(chunk)?;
        for (i, (start, n)) in chunks(len, chunk).into_iter().enumerate() {
            let slot = i % 2;
            let busy = self.staging[slot].busy;
            if busy > 0 {
                self.ctx.timeline().wait(busy)?;
            }
            let part = &data[start as usize..(start + n) as usize];
            self.staging[slot].buffer.write(0, part)?;
            self.recorder.copy_buffer(&self.staging[slot].buffer, 0, dst, offset + start, n)?;
            self.staging[slot].busy = self.recorder.submit()?;
        }
        let last = self.recorder.last_submission();
        self.recorder.wait(last)
    }

    /// Fill `size` bytes of `dst` at `offset` with zeros.
    pub fn zero(&mut self, dst: &Buffer, offset: u64, size: u64) -> VkResult<()> {
        self.recorder.fill(dst, offset, size, 0)?;
        self.recorder.submit_and_wait()?;
        Ok(())
    }

    /// Copy `out.len()` bytes from `src` at `offset` (of the view) into `out`. Returns when the data has arrived.
    pub fn download(&mut self, src: &Buffer, offset: u64, out: &mut [u8]) -> VkResult<()> {
        let len = out.len() as u64;
        if len == 0 {
            return Ok(());
        }
        if offset.checked_add(len).is_none_or(|end| end > src.size()) {
            return Err(VkError::InvalidArgument(format!(
                "download of {len} bytes at {offset} exceeds a buffer of {} bytes",
                src.size()
            )));
        }
        if src.is_host_visible() {
            // the caller has waited for the producer; make sure everything submitted so far is done too
            let last = self.ctx.timeline().last_submitted();
            if last > 0 {
                self.ctx.timeline().wait(last)?;
            }
            return src.read(offset, out);
        }
        let chunk = len.min(STAGING_CHUNK);
        let readback = self.ctx.allocator().alloc_readback(chunk.max(self.ctx.allocator().alignment()))?;
        for (start, n) in chunks(len, chunk) {
            self.recorder.copy_buffer(src, offset + start, &readback, 0, n)?;
            self.recorder.submit_and_wait()?;
            readback.read(0, &mut out[start as usize..(start + n) as usize])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking() {
        assert_eq!(chunks(0, 10), vec![]);
        assert_eq!(chunks(10, 10), vec![(0, 10)]);
        assert_eq!(chunks(25, 10), vec![(0, 10), (10, 10), (20, 5)]);
        assert_eq!(chunks(3, 10), vec![(0, 3)]);
        // a zero chunk size cannot loop forever
        assert_eq!(chunks(2, 0), vec![(0, 1), (1, 1)]);
        let total: u64 = chunks(1_000_003, 4096).iter().map(|c| c.1).sum();
        assert_eq!(total, 1_000_003);
    }
}
