//! Free list of one memory block: best-fit allocation with alignment, coalescing on free. Pure Rust, no Vulkan, unit tested.

use std::collections::BTreeMap;

/// `value` rounded up to a multiple of `align` (a power of two).
pub fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (value + (align - 1)) & !(align - 1)
}

/// `value` rounded down to a multiple of `align` (a power of two).
pub fn align_down(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    value & !(align - 1)
}

/// The free ranges of a block of `capacity` bytes.
#[derive(Debug, Clone)]
pub struct FreeList {
    capacity: u64,
    used: u64,
    /// offset -> length of every free range; ranges never touch (they are coalesced on free).
    free: BTreeMap<u64, u64>,
}

impl FreeList {
    pub fn new(capacity: u64) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self { capacity, used: 0, free }
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Bytes handed out (including the padding that alignment cost: padding stays free, so this counts only requested sizes).
    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    /// Size of the largest free range.
    pub fn largest_free(&self) -> u64 {
        self.free.values().copied().max().unwrap_or(0)
    }

    /// Number of separate free ranges (fragmentation indicator).
    pub fn ranges(&self) -> usize {
        self.free.len()
    }

    /// Allocate `size` bytes at an offset that is a multiple of `align` (a power of two). Best fit: the range that leaves the least
    /// space over; ties go to the lowest offset. `None` when no range fits (or `size` is 0).
    pub fn alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        if size == 0 || !align.is_power_of_two() {
            return None;
        }
        let mut best: Option<(u64, u64, u64)> = None; // (leftover, range offset, aligned offset)
        for (&offset, &len) in &self.free {
            let aligned = align_up(offset, align);
            let padding = aligned - offset;
            if len < padding + size {
                continue;
            }
            let leftover = len - padding - size;
            if best.is_none_or(|(l, _, _)| leftover < l) {
                best = Some((leftover, offset, aligned));
                if leftover == 0 {
                    break;
                }
            }
        }
        let (_, range_offset, aligned) = best?;
        let len = self.free.remove(&range_offset)?;
        let padding = aligned - range_offset;
        if padding > 0 {
            self.free.insert(range_offset, padding);
        }
        let tail = len - padding - size;
        if tail > 0 {
            self.free.insert(aligned + size, tail);
        }
        self.used += size;
        Some(aligned)
    }

    /// Return `size` bytes at `offset` (as returned by [`FreeList::alloc`]). Returns false and changes nothing when the range is out of
    /// the block or overlaps free space (a double free): leaking is safer than corrupting the list.
    pub fn free(&mut self, offset: u64, size: u64) -> bool {
        let Some(end) = offset.checked_add(size) else { return false };
        if size == 0 || end > self.capacity {
            return false;
        }
        if let Some((&prev_off, &prev_len)) = self.free.range(..=offset).next_back() {
            if prev_off + prev_len > offset {
                return false;
            }
        }
        if let Some((&next_off, _)) = self.free.range(offset..).next() {
            if next_off < end {
                return false;
            }
        }
        let mut start = offset;
        let mut len = size;
        if let Some((&prev_off, &prev_len)) = self.free.range(..offset).next_back() {
            if prev_off + prev_len == offset {
                self.free.remove(&prev_off);
                start = prev_off;
                len += prev_len;
            }
        }
        if let Some(&next_len) = self.free.get(&end) {
            self.free.remove(&end);
            len += next_len;
        }
        self.free.insert(start, len);
        self.used = self.used.saturating_sub(size);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_helpers() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_down(511, 256), 256);
        assert_eq!(align_down(512, 256), 512);
    }

    #[test]
    fn sequential_allocation_and_full_block() {
        let mut list = FreeList::new(1024);
        assert_eq!(list.alloc(256, 256), Some(0));
        assert_eq!(list.alloc(256, 256), Some(256));
        assert_eq!(list.alloc(512, 256), Some(512));
        assert_eq!(list.alloc(1, 1), None);
        assert_eq!(list.used(), 1024);
        assert_eq!(list.largest_free(), 0);
    }

    #[test]
    fn alignment_padding_stays_free() {
        let mut list = FreeList::new(4096);
        assert_eq!(list.alloc(100, 1), Some(0));
        // next 256-aligned offset is 256; the 156 bytes before it remain a free range
        assert_eq!(list.alloc(64, 256), Some(256));
        assert_eq!(list.used(), 164);
        assert_eq!(list.alloc(150, 1), Some(100));
        assert!(list.free(0, 100));
        assert!(list.free(100, 150));
        assert!(list.free(256, 64));
        assert!(list.is_empty());
        assert_eq!(list.ranges(), 1);
        assert_eq!(list.largest_free(), 4096);
    }

    #[test]
    fn free_coalesces_in_every_order() {
        for order in [[0usize, 1, 2], [2, 1, 0], [1, 0, 2], [0, 2, 1], [1, 2, 0], [2, 0, 1]] {
            let mut list = FreeList::new(768);
            let offsets: Vec<u64> = (0..3).map(|_| list.alloc(256, 256).unwrap()).collect();
            for i in order {
                assert!(list.free(offsets[i], 256));
            }
            assert_eq!(list.ranges(), 1, "order {order:?}");
            assert_eq!(list.largest_free(), 768);
        }
    }

    #[test]
    fn best_fit_prefers_the_tighter_range() {
        let mut list = FreeList::new(2048);
        let a = list.alloc(512, 256).unwrap(); // 0
        let _b = list.alloc(256, 256).unwrap(); // 512
        let c = list.alloc(256, 256).unwrap(); // 768
        let _d = list.alloc(256, 256).unwrap(); // 1024
        assert!(list.free(a, 512)); // hole of 512 at 0
        assert!(list.free(c, 256)); // hole of 256 at 768
                                    // a 256 request fits both holes (and the tail); the exact fit at 768 wins over the 512 hole at 0
        assert_eq!(list.alloc(256, 256), Some(768));
        assert_eq!(list.alloc(512, 256), Some(0));
    }

    #[test]
    fn rejects_double_free_and_bad_ranges() {
        let mut list = FreeList::new(1024);
        let a = list.alloc(256, 256).unwrap();
        assert!(list.free(a, 256));
        assert!(!list.free(a, 256), "double free");
        assert!(!list.free(0, 0));
        assert!(!list.free(768, 512), "beyond the block");
        assert!(!list.free(u64::MAX, 2), "overflow");
        assert_eq!(list.alloc(0, 256), None);
        assert_eq!(list.alloc(8, 3), None, "alignment must be a power of two");
    }

    #[test]
    fn randomized_invariants() {
        // deterministic xorshift; the model is a plain occupancy bitmap
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let cap = 64 * 1024u64;
        let mut list = FreeList::new(cap);
        let mut live: Vec<(u64, u64)> = Vec::new();
        let mut occupied = vec![false; cap as usize];
        for _ in 0..4000 {
            if live.is_empty() || next() % 3 != 0 {
                let size = 1 + next() % 3000;
                let align = 1u64 << (next() % 9);
                if let Some(off) = list.alloc(size, align) {
                    assert_eq!(off % align, 0);
                    assert!(off + size <= cap);
                    for b in &mut occupied[off as usize..(off + size) as usize] {
                        assert!(!*b, "overlapping allocation");
                        *b = true;
                    }
                    live.push((off, size));
                }
            } else {
                let i = (next() % live.len() as u64) as usize;
                let (off, size) = live.swap_remove(i);
                assert!(list.free(off, size));
                for b in &mut occupied[off as usize..(off + size) as usize] {
                    *b = false;
                }
            }
            let used: u64 = live.iter().map(|&(_, s)| s).sum();
            assert_eq!(list.used(), used);
        }
        for (off, size) in live.drain(..) {
            assert!(list.free(off, size));
        }
        assert!(list.is_empty());
        assert_eq!(list.ranges(), 1);
        assert_eq!(list.largest_free(), cap);
    }
}
