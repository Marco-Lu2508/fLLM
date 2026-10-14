use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocatorError {
    Value(String),
    Memory(String),
    Key(String),
    Runtime(String),
}

impl AllocatorError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Value(_) => "ValueError",
            Self::Memory(_) => "MemoryError",
            Self::Key(_) => "KeyError",
            Self::Runtime(_) => "RuntimeError",
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Value(message)
            | Self::Memory(message)
            | Self::Key(message)
            | Self::Runtime(message) => message,
        }
    }
}

impl fmt::Display for AllocatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for AllocatorError {}

#[derive(Debug, Clone)]
pub struct BlockAllocator {
    pub num_blocks: usize,
    pub refcounts: Vec<usize>,
    pub free_blocks: Vec<usize>,
}

impl BlockAllocator {
    pub fn new(num_blocks: i64) -> Result<Self, AllocatorError> {
        if num_blocks < 0 {
            return Err(AllocatorError::Value(
                "num_blocks must be nonnegative".to_owned(),
            ));
        }
        let count = num_blocks as usize;
        Ok(Self {
            num_blocks: count,
            refcounts: vec![0; count],
            free_blocks: (0..count).rev().collect(),
        })
    }

    fn check_block_id(&self, bid: i64) -> Result<usize, AllocatorError> {
        if bid < 0 || bid as u128 >= self.num_blocks as u128 {
            return Err(AllocatorError::Value(format!(
                "invalid block id {bid}; expected an integer in [0, {})",
                self.num_blocks
            )));
        }
        Ok(bid as usize)
    }

    pub fn available(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn alloc(&mut self) -> Result<usize, AllocatorError> {
        let Some(bid) = self.free_blocks.pop() else {
            return Err(AllocatorError::Memory("no free KV blocks".to_owned()));
        };
        self.refcounts[bid] = 1;
        Ok(bid)
    }

    pub fn inc_ref(&mut self, bid: i64) -> Result<(), AllocatorError> {
        let index = self.check_block_id(bid)?;
        if self.refcounts[index] == 0 {
            return Err(AllocatorError::Value(
                "cannot acquire an unowned block".to_owned(),
            ));
        }
        self.refcounts[index] += 1;
        Ok(())
    }

    pub fn dec_ref(&mut self, bid: i64) -> Result<(), AllocatorError> {
        let index = self.check_block_id(bid)?;
        if self.refcounts[index] == 0 {
            return Err(AllocatorError::Value(
                "cannot release an unowned block".to_owned(),
            ));
        }
        self.refcounts[index] -= 1;
        if self.refcounts[index] == 0 {
            self.free_blocks.push(index);
        }
        Ok(())
    }

    pub fn is_shared(&self, bid: i64) -> Result<bool, AllocatorError> {
        let index = self.check_block_id(bid)?;
        Ok(self.refcounts[index] > 1)
    }

    pub fn is_free(&self, bid: i64) -> Result<bool, AllocatorError> {
        let index = self.check_block_id(bid)?;
        Ok(self.refcounts[index] == 0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceBlocks {
    pub seq_id: i64,
    pub blocks: Vec<usize>,
    pub num_tokens: i64,
}

impl SequenceBlocks {
    fn new(seq_id: i64) -> Self {
        Self {
            seq_id,
            blocks: Vec::new(),
            num_tokens: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

#[derive(Debug)]
pub struct BlockManager {
    pub allocator: BlockAllocator,
    pub block_size: usize,
    pub tables: HashMap<i64, SequenceBlocks>,
    pub block_hashes: HashMap<usize, i64>,
}

impl BlockManager {
    pub fn new(num_blocks: i64, block_size: i64) -> Result<Self, AllocatorError> {
        // Match Python constructor order: block_size is checked before allocator capacity.
        if block_size <= 0 {
            return Err(AllocatorError::Value(
                "block_size must be positive".to_owned(),
            ));
        }
        Ok(Self {
            allocator: BlockAllocator::new(num_blocks)?,
            block_size: block_size as usize,
            tables: HashMap::new(),
            block_hashes: HashMap::new(),
        })
    }

    pub fn new_table(&mut self, seq_id: i64) -> Result<(), AllocatorError> {
        if self.tables.contains_key(&seq_id) {
            return Err(AllocatorError::Value(format!(
                "block table already exists for sequence {seq_id}"
            )));
        }
        self.tables.insert(seq_id, SequenceBlocks::new(seq_id));
        Ok(())
    }

    pub fn get_table(&self, seq_id: i64) -> Result<&SequenceBlocks, AllocatorError> {
        self.tables
            .get(&seq_id)
            .ok_or_else(|| AllocatorError::Key(format!("{seq_id}")))
    }

    pub fn get_table_mut(&mut self, seq_id: i64) -> Result<&mut SequenceBlocks, AllocatorError> {
        self.tables
            .get_mut(&seq_id)
            .ok_or_else(|| AllocatorError::Key(format!("{seq_id}")))
    }

    pub fn drop_table(&mut self, seq_id: i64) -> Result<(), AllocatorError> {
        let Some(table) = self.tables.remove(&seq_id) else {
            return Ok(());
        };
        for bid in table.blocks {
            self.block_hashes.remove(&bid);
            self.allocator.dec_ref(bid as i64)?;
        }
        Ok(())
    }

    pub fn append_block(&mut self, seq_id: i64) -> Result<usize, AllocatorError> {
        // Keep Python's operation order: allocate first, then resolve the table.
        let bid = self.allocator.alloc()?;
        self.get_table_mut(seq_id)?.blocks.push(bid);
        Ok(bid)
    }

    pub fn claim_block_for(&mut self, seq_id: i64, bid: i64) -> Result<(), AllocatorError> {
        // Keep Python's operation order, including ref acquisition before table lookup.
        self.allocator.inc_ref(bid)?;
        self.get_table_mut(seq_id)?.blocks.push(bid as usize);
        Ok(())
    }

    pub fn num_blocks_needed(
        &self,
        seq_id: i64,
        extra_tokens: i64,
    ) -> Result<usize, AllocatorError> {
        let table = self.get_table(seq_id)?;
        let total = table.num_tokens as i128 + extra_tokens as i128;
        if total <= 0 {
            return Ok(0);
        }
        let block_size = self.block_size as i128;
        let required = (total + block_size - 1) / block_size;
        Ok((required as usize).saturating_sub(table.blocks.len()))
    }

    pub fn ensure_blocks(
        &mut self,
        seq_id: i64,
        extra_tokens: i64,
    ) -> Result<Vec<usize>, AllocatorError> {
        if extra_tokens < 0 {
            return Err(AllocatorError::Value(
                "extra_tokens must be nonnegative".to_owned(),
            ));
        }
        let missing = self.num_blocks_needed(seq_id, extra_tokens)?;
        let available = self.allocator.available();
        if missing > available {
            return Err(AllocatorError::Memory(format!(
                "need {missing} KV blocks but only {available} are available"
            )));
        }
        let mut allocated = Vec::with_capacity(missing);
        for _ in 0..missing {
            allocated.push(self.append_block(seq_id)?);
        }
        Ok(allocated)
    }

    pub fn cow_last_block_if_shared<F>(
        &mut self,
        seq_id: i64,
        mut copy_block: F,
    ) -> Result<(), AllocatorError>
    where
        F: FnMut(usize, usize) -> Result<(), AllocatorError>,
    {
        let Some(old) = self.get_table(seq_id)?.blocks.last().copied() else {
            return Ok(());
        };
        if !self.allocator.is_shared(old as i64)? {
            return Ok(());
        }
        let new = self.allocator.alloc()?;
        if let Err(error) = copy_block(old, new) {
            // Roll back the speculative block and leave table/refcounts untouched.
            self.allocator.dec_ref(new as i64)?;
            return Err(error);
        }
        self.allocator.dec_ref(old as i64)?;
        self.get_table_mut(seq_id)?.blocks.pop();
        self.get_table_mut(seq_id)?.blocks.push(new);
        self.block_hashes.remove(&old);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{AllocatorError, BlockAllocator, BlockManager};

    #[test]
    fn allocation_is_lifo_and_refs_recycle_in_release_order() {
        let mut allocator = BlockAllocator::new(3).unwrap();
        assert_eq!(allocator.alloc().unwrap(), 0);
        assert_eq!(allocator.alloc().unwrap(), 1);
        allocator.inc_ref(0).unwrap();
        allocator.dec_ref(0).unwrap();
        allocator.dec_ref(0).unwrap();
        assert_eq!(allocator.alloc().unwrap(), 0);
        assert_eq!(allocator.refcounts, vec![1, 1, 0]);
    }

    #[test]
    fn ensure_blocks_preflights_capacity_without_partial_mutation() {
        let mut manager = BlockManager::new(2, 8).unwrap();
        manager.new_table(1).unwrap();
        let first = manager.append_block(1).unwrap();
        manager.get_table_mut(1).unwrap().num_tokens = 8;
        let refs = manager.allocator.refcounts.clone();
        let free = manager.allocator.free_blocks.clone();
        assert!(matches!(
            manager.ensure_blocks(1, 16),
            Err(AllocatorError::Memory(_))
        ));
        assert_eq!(manager.get_table(1).unwrap().blocks, vec![first]);
        assert_eq!(manager.allocator.refcounts, refs);
        assert_eq!(manager.allocator.free_blocks, free);
    }

    #[test]
    fn failed_cow_rolls_back_and_success_releases_old_hash() {
        let mut manager = BlockManager::new(3, 8).unwrap();
        manager.new_table(1).unwrap();
        let shared = manager.append_block(1).unwrap();
        manager.allocator.inc_ref(shared as i64).unwrap();
        manager.block_hashes.insert(shared, 99);
        let refs = manager.allocator.refcounts.clone();
        let free = manager.allocator.free_blocks.clone();
        let error = manager
            .cow_last_block_if_shared(1, |_src, _dst| {
                Err(AllocatorError::Runtime("simulated copy failure".to_owned()))
            })
            .unwrap_err();
        assert_eq!(error.message(), "simulated copy failure");
        assert_eq!(manager.get_table(1).unwrap().blocks, vec![shared]);
        assert_eq!(manager.allocator.refcounts, refs);
        assert_eq!(manager.allocator.free_blocks, free);
        assert_eq!(manager.block_hashes.get(&shared), Some(&99));

        let mut copied = None;
        manager
            .cow_last_block_if_shared(1, |src, dst| {
                copied = Some((src, dst));
                Ok(())
            })
            .unwrap();
        let (src, dst) = copied.unwrap();
        assert_eq!(src, shared);
        assert_eq!(manager.get_table(1).unwrap().blocks, vec![dst]);
        assert_eq!(manager.allocator.refcounts[shared], 1);
        assert!(!manager.block_hashes.contains_key(&shared));
    }

    #[test]
    fn constructors_and_unowned_refs_match_python_error_classes() {
        assert!(matches!(
            BlockAllocator::new(-1),
            Err(AllocatorError::Value(_))
        ));
        assert!(
            matches!(BlockManager::new(-1, 0), Err(AllocatorError::Value(message)) if message == "block_size must be positive")
        );
        assert!(BlockManager::new(1, 8).is_ok());
        let mut allocator = BlockAllocator::new(1).unwrap();
        assert!(
            matches!(allocator.inc_ref(0), Err(AllocatorError::Value(message)) if message == "cannot acquire an unowned block")
        );
        assert!(
            matches!(allocator.dec_ref(0), Err(AllocatorError::Value(message)) if message == "cannot release an unowned block")
        );
    }
}
