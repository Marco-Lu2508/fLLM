//! Choice of the Vulkan memory type for a kind of allocation. Pure functions over plain data, unit tested with the memory layouts of
//! typical devices (discrete NVIDIA / AMD with and without resizable BAR, integrated Intel, Apple-like unified memory).

/// `VkMemoryPropertyFlags` bits (the values are fixed by the Vulkan specification).
pub mod bits {
    pub const DEVICE_LOCAL: u32 = 0x01;
    pub const HOST_VISIBLE: u32 = 0x02;
    pub const HOST_COHERENT: u32 = 0x04;
    pub const HOST_CACHED: u32 = 0x08;
    pub const LAZILY_ALLOCATED: u32 = 0x10;
    pub const PROTECTED: u32 = 0x20;
    pub const DEVICE_COHERENT_AMD: u32 = 0x40;
    pub const DEVICE_UNCACHED_AMD: u32 = 0x80;
}

/// One memory type of the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemType {
    pub flags: u32,
    pub heap: u32,
    pub heap_size: u64,
}

/// What an allocation is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    /// Weights, KV cache, activations: GPU-side memory. On unified-memory devices the type is host visible and mapped.
    DeviceLocal,
    /// Device-local memory the CPU can write directly (resizable BAR, unified memory): small, frequently CPU-written parameters. Falls
    /// back to [`MemoryKind::Staging`] when the device has no such type.
    DeviceMapped,
    /// Host memory the GPU reads over the bus: uploads. Coherent, preferably write-combined and outside the small BAR heap.
    Staging,
    /// Host memory the CPU reads back from: cached, so that reading logits is fast.
    Readback,
}

/// Selection rules of a kind: bits that must be set, bits that must not be set, and weighted preferences.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    pub required: u32,
    pub forbidden: u32,
    pub weights: &'static [(u32, i32)],
}

const NEVER: u32 = bits::PROTECTED | bits::LAZILY_ALLOCATED | bits::DEVICE_COHERENT_AMD | bits::DEVICE_UNCACHED_AMD;

const DEVICE_LOCAL_DISCRETE: Policy = Policy {
    required: bits::DEVICE_LOCAL,
    forbidden: NEVER,
    weights: &[(bits::HOST_VISIBLE, -100)],
};
const DEVICE_LOCAL_UNIFIED: Policy = Policy {
    required: bits::DEVICE_LOCAL,
    forbidden: NEVER,
    weights: &[(bits::HOST_VISIBLE, 50), (bits::HOST_COHERENT, 20), (bits::HOST_CACHED, 5)],
};
const DEVICE_MAPPED: Policy = Policy {
    required: bits::DEVICE_LOCAL | bits::HOST_VISIBLE | bits::HOST_COHERENT,
    forbidden: NEVER,
    weights: &[(bits::HOST_CACHED, -5)],
};
const STAGING: Policy = Policy {
    required: bits::HOST_VISIBLE | bits::HOST_COHERENT,
    forbidden: NEVER,
    weights: &[(bits::DEVICE_LOCAL, -50), (bits::HOST_CACHED, -5)],
};
const READBACK: Policy = Policy {
    required: bits::HOST_VISIBLE,
    forbidden: NEVER,
    weights: &[(bits::HOST_CACHED, 30), (bits::HOST_COHERENT, 10), (bits::DEVICE_LOCAL, -50)],
};
/// Memory for `VK_EXT_external_memory_host` imports: host memory, cached like the pages it wraps.
pub const IMPORT: Policy = Policy {
    required: bits::HOST_VISIBLE,
    forbidden: NEVER,
    weights: &[(bits::HOST_COHERENT, 10), (bits::HOST_CACHED, 5), (bits::DEVICE_LOCAL, 5)],
};

/// The rules for `kind`; `unified` selects host-visible device memory for [`MemoryKind::DeviceLocal`].
pub fn policy_for(kind: MemoryKind, unified: bool) -> Policy {
    match kind {
        MemoryKind::DeviceLocal if unified => DEVICE_LOCAL_UNIFIED,
        MemoryKind::DeviceLocal => DEVICE_LOCAL_DISCRETE,
        MemoryKind::DeviceMapped => DEVICE_MAPPED,
        MemoryKind::Staging => STAGING,
        MemoryKind::Readback => READBACK,
    }
}

/// Index of the best memory type allowed by `type_bits` (bit i = type i usable) under `policy`: highest score, then the largest heap,
/// then the lowest index. `None` when no type satisfies the required and forbidden bits.
pub fn pick(types: &[MemType], type_bits: u32, policy: &Policy) -> Option<u32> {
    let mut best: Option<(i32, u64, usize)> = None;
    for (i, t) in types.iter().enumerate() {
        if i >= 32 || type_bits & (1 << i) == 0 {
            continue;
        }
        if t.flags & policy.required != policy.required || t.flags & policy.forbidden != 0 {
            continue;
        }
        let score: i32 = policy.weights.iter().filter(|(bit, _)| t.flags & bit != 0).map(|(_, w)| w).sum();
        let better = match best {
            None => true,
            Some((s, h, _)) => score > s || (score == s && t.heap_size > h),
        };
        if better {
            best = Some((score, t.heap_size, i));
        }
    }
    best.map(|(_, _, i)| i as u32)
}

#[cfg(test)]
mod tests {
    use super::bits::*;
    use super::*;

    const GIB: u64 = 1 << 30;

    fn ty(flags: u32, heap: u32, heap_size: u64) -> MemType {
        MemType { flags, heap, heap_size }
    }

    /// NVIDIA-like: pure VRAM, system RAM write-combined, system RAM cached, 256 MiB BAR window.
    fn nvidia() -> Vec<MemType> {
        vec![
            ty(DEVICE_LOCAL, 0, 12 * GIB),
            ty(0, 1, 32 * GIB),
            ty(HOST_VISIBLE | HOST_COHERENT, 1, 32 * GIB),
            ty(HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, 1, 32 * GIB),
            ty(DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT, 2, 256 << 20),
        ]
    }

    /// AMD-like with resizable BAR: VRAM, a full-size host-visible VRAM type, system RAM types.
    fn amd_rebar() -> Vec<MemType> {
        vec![
            ty(DEVICE_LOCAL, 0, 16 * GIB),
            ty(HOST_VISIBLE | HOST_COHERENT, 1, 32 * GIB),
            ty(HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, 1, 32 * GIB),
            ty(DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT, 0, 16 * GIB),
        ]
    }

    /// Intel iGPU: one heap that is both device local and host visible.
    fn intel_igpu() -> Vec<MemType> {
        vec![
            ty(DEVICE_LOCAL, 0, 24 * GIB),
            ty(DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, 0, 24 * GIB),
        ]
    }

    const ALL: u32 = u32::MAX;

    #[test]
    fn discrete_cards_keep_weights_in_pure_vram() {
        assert_eq!(pick(&nvidia(), ALL, &policy_for(MemoryKind::DeviceLocal, false)), Some(0));
        assert_eq!(pick(&amd_rebar(), ALL, &policy_for(MemoryKind::DeviceLocal, false)), Some(0));
    }

    #[test]
    fn staging_avoids_the_bar_window_and_prefers_write_combined() {
        assert_eq!(pick(&nvidia(), ALL, &policy_for(MemoryKind::Staging, false)), Some(2));
        assert_eq!(pick(&amd_rebar(), ALL, &policy_for(MemoryKind::Staging, false)), Some(1));
    }

    #[test]
    fn readback_prefers_cached_system_memory() {
        assert_eq!(pick(&nvidia(), ALL, &policy_for(MemoryKind::Readback, false)), Some(3));
        assert_eq!(pick(&amd_rebar(), ALL, &policy_for(MemoryKind::Readback, false)), Some(2));
    }

    #[test]
    fn device_mapped_needs_a_bar_type() {
        assert_eq!(pick(&nvidia(), ALL, &policy_for(MemoryKind::DeviceMapped, false)), Some(4));
        assert_eq!(pick(&amd_rebar(), ALL, &policy_for(MemoryKind::DeviceMapped, false)), Some(3));
        // a device without any host-visible device-local type: the caller falls back to staging
        let plain = vec![ty(DEVICE_LOCAL, 0, GIB), ty(HOST_VISIBLE | HOST_COHERENT, 1, GIB)];
        assert_eq!(pick(&plain, ALL, &policy_for(MemoryKind::DeviceMapped, false)), None);
    }

    #[test]
    fn unified_memory_is_mapped() {
        let igpu = intel_igpu();
        assert_eq!(pick(&igpu, ALL, &policy_for(MemoryKind::DeviceLocal, true)), Some(1));
        // staging has nothing but device-local types to choose from and takes the host-visible one
        assert_eq!(pick(&igpu, ALL, &policy_for(MemoryKind::Staging, true)), Some(1));
        assert_eq!(pick(&igpu, ALL, &policy_for(MemoryKind::Readback, true)), Some(1));
    }

    #[test]
    fn type_bits_restrict_the_choice() {
        // only types 1 and 3 usable by the resource
        let bits_mask = (1 << 1) | (1 << 3);
        assert_eq!(pick(&nvidia(), bits_mask, &policy_for(MemoryKind::Readback, false)), Some(3));
        assert_eq!(pick(&nvidia(), 0, &policy_for(MemoryKind::Readback, false)), None);
        // device-local request while only host types are allowed
        assert_eq!(
            pick(&nvidia(), (1 << 1) | (1 << 2), &policy_for(MemoryKind::DeviceLocal, false)),
            None
        );
    }

    #[test]
    fn forbidden_bits_are_never_chosen() {
        let types = vec![ty(DEVICE_LOCAL | DEVICE_UNCACHED_AMD, 0, GIB), ty(DEVICE_LOCAL | PROTECTED, 0, GIB)];
        assert_eq!(pick(&types, ALL, &policy_for(MemoryKind::DeviceLocal, false)), None);
    }

    #[test]
    fn ties_go_to_the_larger_heap_then_the_lower_index() {
        let types = vec![ty(DEVICE_LOCAL, 0, GIB), ty(DEVICE_LOCAL, 1, 4 * GIB), ty(DEVICE_LOCAL, 2, 4 * GIB)];
        assert_eq!(pick(&types, ALL, &policy_for(MemoryKind::DeviceLocal, false)), Some(1));
    }

    #[test]
    fn import_type_is_host_memory() {
        assert_eq!(pick(&nvidia(), ALL, &IMPORT), Some(3));
        assert_eq!(pick(&intel_igpu(), ALL, &IMPORT), Some(1));
    }
}
