//! A pool of memory blocks sharing one memory type and usage: suballocation across blocks and the policy for releasing blocks.
//! Generic over the block payload (the Vulkan handles), so the logic is unit tested without a device.

use super::freelist::FreeList;

/// One block: the payload (device memory, buffer, mapping) and its free list.
pub struct PoolBlock<T> {
    pub payload: T,
    pub free: FreeList,
}

/// Blocks of one pool. Slots are stable (a removed block leaves `None`), so `(slot, offset)` stays valid while an allocation lives.
pub struct Pool<T> {
    blocks: Vec<Option<PoolBlock<T>>>,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self { blocks: Vec::new() }
    }
}

impl<T> Pool<T> {
    /// Allocate in the first block (lowest slot) that has room; `None` when every block is too full or fragmented.
    pub fn try_alloc(&mut self, size: u64, align: u64) -> Option<(usize, u64)> {
        for (slot, block) in self.blocks.iter_mut().enumerate() {
            if let Some(block) = block {
                if let Some(offset) = block.free.alloc(size, align) {
                    return Some((slot, offset));
                }
            }
        }
        None
    }

    /// Add a block of `capacity` bytes; returns its slot (reusing the lowest empty slot).
    pub fn insert(&mut self, payload: T, capacity: u64) -> usize {
        let block = PoolBlock {
            payload,
            free: FreeList::new(capacity),
        };
        if let Some(slot) = self.blocks.iter().position(Option::is_none) {
            self.blocks[slot] = Some(block);
            slot
        } else {
            self.blocks.push(Some(block));
            self.blocks.len() - 1
        }
    }

    pub fn get(&self, slot: usize) -> Option<&PoolBlock<T>> {
        self.blocks.get(slot).and_then(Option::as_ref)
    }

    pub fn get_mut(&mut self, slot: usize) -> Option<&mut PoolBlock<T>> {
        self.blocks.get_mut(slot).and_then(Option::as_mut)
    }

    /// Number of live blocks.
    pub fn len(&self) -> usize {
        self.blocks.iter().flatten().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Give `size` bytes at `offset` of block `slot` back. When that leaves the block empty and another block exists, the block is
    /// removed and its payload returned so that the caller can release the device memory; the last block of a pool is kept.
    pub fn release(&mut self, slot: usize, offset: u64, size: u64) -> Option<T> {
        let other_blocks = self.blocks.iter().enumerate().any(|(i, b)| i != slot && b.is_some());
        let block = self.blocks.get_mut(slot)?.as_mut()?;
        if !block.free.free(offset, size) {
            return None;
        }
        if block.free.is_empty() && other_blocks {
            return self.blocks[slot].take().map(|b| b.payload);
        }
        None
    }

    /// Remove every block (pool teardown) and hand the payloads out.
    pub fn drain(&mut self) -> Vec<T> {
        self.blocks.drain(..).flatten().map(|b| b.payload).collect()
    }

    /// Free bytes summed over all blocks.
    pub fn free_bytes(&self) -> u64 {
        self.blocks.iter().flatten().map(|b| b.free.capacity() - b.free.used()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_blocks_in_order_and_grows_on_demand() {
        let mut pool: Pool<&str> = Pool::default();
        assert!(pool.try_alloc(256, 256).is_none());
        let a = pool.insert("a", 1024);
        assert_eq!(pool.try_alloc(768, 256), Some((a, 0)));
        assert_eq!(pool.try_alloc(512, 256), None);
        let b = pool.insert("b", 1024);
        assert_eq!(pool.try_alloc(512, 256), Some((b, 0)));
        assert_eq!(pool.try_alloc(256, 256), Some((a, 768)));
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn empty_blocks_are_released_unless_last() {
        let mut pool: Pool<&str> = Pool::default();
        let a = pool.insert("a", 1024);
        let b = pool.insert("b", 1024);
        let (sa, oa) = pool.try_alloc(1024, 256).unwrap();
        let (sb, ob) = pool.try_alloc(1024, 256).unwrap();
        assert_eq!((sa, sb), (a, b));
        // block b becomes empty while block a is alive -> released
        assert_eq!(pool.release(sb, ob, 1024), Some("b"));
        assert_eq!(pool.len(), 1);
        // block a is the last one: kept although empty
        assert_eq!(pool.release(sa, oa, 1024), None);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.free_bytes(), 1024);
        // the freed slot is reused
        assert_eq!(pool.insert("c", 512), b);
    }

    #[test]
    fn partial_release_keeps_the_block() {
        let mut pool: Pool<()> = Pool::default();
        let a = pool.insert((), 1024);
        let _b = pool.insert((), 1024);
        let (s1, o1) = pool.try_alloc(256, 256).unwrap();
        let (_s2, _o2) = pool.try_alloc(256, 256).unwrap();
        assert_eq!(s1, a);
        assert_eq!(pool.release(s1, o1, 256), None);
        assert_eq!(pool.len(), 2);
        // double release is ignored
        assert_eq!(pool.release(s1, o1, 256), None);
        assert_eq!(pool.free_bytes(), 2048 - 256);
    }

    #[test]
    fn drain_returns_everything() {
        let mut pool: Pool<u32> = Pool::default();
        pool.insert(1, 64);
        pool.insert(2, 64);
        let mut v = pool.drain();
        v.sort_unstable();
        assert_eq!(v, vec![1, 2]);
        assert!(pool.is_empty());
    }
}
