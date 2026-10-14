//! Residency sets (macOS 15 / iOS 18): tell the OS that the long-lived buffers (weights, KV cache) must stay
//! resident, so the first use after a pause does not fault them in page by page. Absent on older systems, in which
//! case [`ResidencyManager::new`] returns `None` and everything works without.

use super::super::buffers::MetalBuffer;
use super::super::device_info::{MetalConfig, MetalDevice};
use super::CommandQueue;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::sel;
use objc2_metal::{MTLCommandQueue, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor};
use std::sync::{Arc, Mutex};

struct SetHandle(Retained<ProtocolObject<dyn MTLResidencySet>>);

// SAFETY: every access to the set goes through the mutex of `ResidencyManager`, so it is used by one thread at a time,
// which is all Metal asks of objects that are not documented as thread safe.
unsafe impl Send for SetHandle {}

/// One residency set attached to the command queue; buffers added to it are kept resident while the queue is in use.
pub struct ResidencyManager {
    set: Mutex<SetHandle>,
}

impl ResidencyManager {
    /// `None` when disabled by `FLLM_METAL_RESIDENCY=0`, when the OS has no residency sets, or when creating the
    /// set fails.
    pub fn new(device: &MetalDevice, queue: &CommandQueue, config: &MetalConfig) -> Option<Self> {
        if !config.residency_sets {
            return None;
        }
        autoreleasepool(|_| {
            if !device
                .raw()
                .respondsToSelector(sel!(newResidencySetWithDescriptor:error:))
                || !queue.raw().respondsToSelector(sel!(addResidencySet:))
            {
                return None;
            }
            let descriptor = MTLResidencySetDescriptor::new();
            let set = device
                .raw()
                .newResidencySetWithDescriptor_error(&descriptor)
                .ok()?;
            queue.raw().addResidencySet(&set);
            Some(Self {
                set: Mutex::new(SetHandle(set)),
            })
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SetHandle> {
        self.set
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Add one buffer. Takes effect with the next [`ResidencyManager::commit`].
    pub fn add(&self, buffer: &MetalBuffer) {
        self.lock().0.addAllocation(buffer.as_allocation());
    }

    /// Add several buffers (the windows of a mapped region).
    pub fn add_many(&self, buffers: &[Arc<MetalBuffer>]) {
        let guard = self.lock();
        for buffer in buffers {
            guard.0.addAllocation(buffer.as_allocation());
        }
    }

    pub fn remove(&self, buffer: &MetalBuffer) {
        self.lock().0.removeAllocation(buffer.as_allocation());
    }

    /// Apply pending additions and removals and ask the system to make the set resident.
    pub fn commit(&self) {
        let guard = self.lock();
        guard.0.commit();
        guard.0.requestResidency();
    }

    /// Bytes currently in the set.
    pub fn allocated_bytes(&self) -> u64 {
        self.lock().0.allocatedSize()
    }
}
