use crate::mem::block_allocator::AllocatorError;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixCacheError {
    Value(String),
    Memory(String),
    Key(String),
    Runtime(String),
}

impl PrefixCacheError {
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

impl fmt::Display for PrefixCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for PrefixCacheError {}

impl From<AllocatorError> for PrefixCacheError {
    fn from(error: AllocatorError) -> Self {
        match error {
            AllocatorError::Value(message) => Self::Value(message),
            AllocatorError::Memory(message) => Self::Memory(message),
            AllocatorError::Key(message) => Self::Key(message),
            AllocatorError::Runtime(message) => Self::Runtime(message),
        }
    }
}

#[derive(Debug, Clone)]
struct Node {
    tokens: Option<Vec<i64>>,
    block_id: Option<i64>,
    chain_hash: Option<u64>,
    parent: Option<usize>,
    children: Vec<(Vec<i64>, usize)>,
}

impl Node {
    fn root() -> Self {
        Self {
            tokens: None,
            block_id: None,
            chain_hash: None,
            parent: None,
            children: Vec::new(),
        }
    }

    fn page(tokens: Vec<i64>, block_id: i64, chain_hash: u64, parent: usize) -> Self {
        Self {
            tokens: Some(tokens),
            block_id: Some(block_id),
            chain_hash: Some(chain_hash),
            parent: Some(parent),
            children: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixEntry {
    pub chain_hash: u64,
    pub block_id: i64,
    pub tokens: Vec<i64>,
}

/// Block-aligned radix prefix cache matching the Python PrefixCache.
/// The booleans model whether reference-count callbacks were supplied to the
/// Python constructor. Methods receive callbacks explicitly so ownership stays
/// visible and the cache composes directly with a BlockAllocator.
#[derive(Debug)]
pub struct PrefixCache {
    max_entries: usize,
    claim_refs: bool,
    release_refs: bool,
    nodes: Vec<Option<Node>>,
    free_nodes: Vec<usize>,
    by_hash: HashMap<u64, usize>,
    blocks: HashMap<u64, PrefixEntry>,
    lru_order: Vec<u64>,
}

impl PrefixCache {
    pub fn new(
        max_entries: i64,
        claim_refs: bool,
        release_refs: bool,
    ) -> Result<Self, PrefixCacheError> {
        if max_entries < 0 {
            return Err(PrefixCacheError::Value(
                "max_entries must be >= 0".to_owned(),
            ));
        }
        if claim_refs && !release_refs && max_entries > 0 {
            return Err(PrefixCacheError::Value(
                "bounded owning cache requires release_ref".to_owned(),
            ));
        }
        Ok(Self {
            max_entries: max_entries as usize,
            claim_refs,
            release_refs,
            nodes: vec![Some(Node::root())],
            free_nodes: Vec::new(),
            by_hash: HashMap::new(),
            blocks: HashMap::new(),
            lru_order: Vec::new(),
        })
    }

    pub fn hash_chain(prev_hash: u64, tokens: &[i64]) -> u64 {
        let mut hash = prev_hash ^ 0x9E37_79B9_7F4A_7C15;
        for token in tokens {
            hash = hash.wrapping_mul(1_000_003) ^ (*token as u64);
        }
        hash ^ (hash >> 29)
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn entries_lru(&self) -> Vec<PrefixEntry> {
        self.lru_order
            .iter()
            .filter_map(|chain_hash| self.blocks.get(chain_hash).cloned())
            .collect()
    }

    pub fn tree_snapshot(&self) -> Value {
        self.snapshot_node(0)
    }

    fn snapshot_node(&self, index: usize) -> Value {
        let node = self.nodes[index].as_ref().expect("live radix node");
        let children: Vec<_> = node
            .children
            .iter()
            .map(|(_, child)| self.snapshot_node(*child))
            .collect();
        json!({
            "tokens": node.tokens,
            "block_id": node.block_id,
            "chain_hash": node.chain_hash,
            "children": children,
        })
    }

    fn allocate_node(&mut self, node: Node) -> usize {
        if let Some(index) = self.free_nodes.pop() {
            self.nodes[index] = Some(node);
            index
        } else {
            let index = self.nodes.len();
            self.nodes.push(Some(node));
            index
        }
    }

    fn find_child(&self, parent: usize, tokens: &[i64]) -> Option<usize> {
        self.nodes[parent]
            .as_ref()?
            .children
            .iter()
            .find(|(page, _)| page == tokens)
            .map(|(_, child)| *child)
    }

    fn touch(&mut self, chain_hash: u64) {
        if let Some(index) = self.lru_order.iter().position(|hash| *hash == chain_hash) {
            self.lru_order.remove(index);
            self.lru_order.push(chain_hash);
        }
    }

    fn put_block(&mut self, entry: PrefixEntry) {
        if !self.blocks.contains_key(&entry.chain_hash) {
            self.lru_order.push(entry.chain_hash);
        }
        self.blocks.insert(entry.chain_hash, entry);
    }

    fn remove_block(&mut self, chain_hash: u64) {
        self.blocks.remove(&chain_hash);
        if let Some(index) = self.lru_order.iter().position(|hash| *hash == chain_hash) {
            self.lru_order.remove(index);
        }
    }

    pub fn match_blocks(
        &mut self,
        hashes: &[u64],
        tokens_per_block: &[Vec<i64>],
        claim: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
    ) -> Result<usize, PrefixCacheError> {
        let mut node_index = 0;
        let mut matched = 0;
        for (chain_hash, tokens) in hashes.iter().zip(tokens_per_block) {
            let Some(child_index) = self.find_child(node_index, tokens) else {
                break;
            };
            let child = self.nodes[child_index].as_ref().expect("child node");
            let (Some(block_id), Some(stored_hash)) = (child.block_id, child.chain_hash) else {
                break;
            };
            if stored_hash != *chain_hash {
                break;
            }
            claim(block_id)?;
            if self.blocks.contains_key(chain_hash) {
                self.touch(*chain_hash);
            }
            node_index = child_index;
            matched += 1;
        }
        Ok(matched)
    }

    pub fn insert_path(
        &mut self,
        entries: &[PrefixEntry],
        claim_ref: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
        release_ref: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
    ) -> Result<(), PrefixCacheError> {
        let mut previous_hash = 0u64;
        for entry in entries {
            let expected = Self::hash_chain(previous_hash, &entry.tokens);
            if entry.chain_hash != expected {
                return Err(PrefixCacheError::Value(
                    "prefix hash does not match its radix path".to_owned(),
                ));
            }
            previous_hash = entry.chain_hash;
        }

        let mut node_index = 0;
        for entry in entries {
            let existing = self.find_child(node_index, &entry.tokens);
            let child_index = if let Some(index) = existing {
                index
            } else {
                let child = self.allocate_node(Node::page(
                    entry.tokens.clone(),
                    entry.block_id,
                    entry.chain_hash,
                    node_index,
                ));
                self.nodes[node_index]
                    .as_mut()
                    .expect("parent node")
                    .children
                    .push((entry.tokens.clone(), child));
                self.by_hash.insert(entry.chain_hash, child);
                if self.claim_refs {
                    claim_ref(entry.block_id)?;
                }
                self.put_block(entry.clone());
                child
            };

            let child = self.nodes[child_index].as_ref().expect("child node");
            if existing.is_some() && child.chain_hash != Some(entry.chain_hash) {
                return Err(PrefixCacheError::Runtime(
                    "radix path hash mismatch".to_owned(),
                ));
            }
            if existing.is_some() && child.block_id.is_none() {
                let child = self.nodes[child_index].as_mut().expect("child node");
                child.block_id = Some(entry.block_id);
                child.chain_hash = Some(entry.chain_hash);
                self.by_hash.insert(entry.chain_hash, child_index);
                if self.claim_refs {
                    claim_ref(entry.block_id)?;
                }
                self.put_block(entry.clone());
            } else if existing.is_some() && self.blocks.contains_key(&entry.chain_hash) {
                self.touch(entry.chain_hash);
            }
            node_index = child_index;
        }
        self.enforce_limit(release_ref)
    }

    pub fn insert(
        &mut self,
        chain_hash: u64,
        block_id: i64,
        tokens: Vec<i64>,
        claim_ref: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
        release_ref: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
    ) -> Result<(), PrefixCacheError> {
        if let Some(child_index) = self.find_child(0, &tokens) {
            let child = self.nodes[child_index].as_ref().expect("root child");
            if child.chain_hash != Some(chain_hash) {
                return Err(PrefixCacheError::Runtime(
                    "root token page already has a different prefix hash".to_owned(),
                ));
            }
            if self.blocks.contains_key(&chain_hash) {
                self.touch(chain_hash);
            }
            return Ok(());
        }
        let node = self.allocate_node(Node::page(tokens.clone(), block_id, chain_hash, 0));
        self.nodes[0]
            .as_mut()
            .expect("root")
            .children
            .push((tokens.clone(), node));
        self.by_hash.insert(chain_hash, node);
        if self.claim_refs {
            claim_ref(block_id)?;
        }
        self.put_block(PrefixEntry {
            chain_hash,
            block_id,
            tokens,
        });
        self.enforce_limit(release_ref)
    }

    fn remove_leaf(
        &mut self,
        node_index: usize,
        release_cache_ref: bool,
        release_ref: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
    ) -> Result<(), PrefixCacheError> {
        let node = self.nodes[node_index].as_ref().ok_or_else(|| {
            PrefixCacheError::Value("only radix leaves can be removed".to_owned())
        })?;
        if !node.children.is_empty() || node.parent.is_none() || node.tokens.is_none() {
            return Err(PrefixCacheError::Value(
                "only radix leaves can be removed".to_owned(),
            ));
        }
        let tokens = node.tokens.clone().expect("leaf token page");
        let chain_hash = node.chain_hash;
        let block_id = node.block_id;
        let parent_index = node.parent.expect("leaf parent");
        self.nodes[parent_index]
            .as_mut()
            .expect("parent node")
            .children
            .retain(|(page, _)| *page != tokens);
        if let Some(hash) = chain_hash {
            self.remove_block(hash);
            self.by_hash.remove(&hash);
        }
        if release_cache_ref && self.release_refs {
            if let Some(block_id) = block_id {
                release_ref(block_id)?;
            }
        }
        self.nodes[node_index] = None;
        self.free_nodes.push(node_index);
        Ok(())
    }

    fn evict_oldest_leaf(
        &mut self,
        claim_free: Option<&mut dyn FnMut(i64) -> Result<bool, PrefixCacheError>>,
        release_cache_ref: bool,
        release_ref: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
    ) -> Result<usize, PrefixCacheError> {
        let order = self.lru_order.clone();
        let mut claim_free = claim_free;
        for chain_hash in order {
            let Some(node_index) = self.by_hash.get(&chain_hash).copied() else {
                continue;
            };
            let Some(node) = self.nodes[node_index].as_ref() else {
                continue;
            };
            if !node.children.is_empty() {
                continue;
            }
            let Some(block_id) = node.block_id else {
                continue;
            };
            if let Some(can_claim) = claim_free.as_mut() {
                if !(can_claim)(block_id)? {
                    continue;
                }
            }
            self.remove_leaf(node_index, release_cache_ref, release_ref)?;
            return Ok(1);
        }
        Ok(0)
    }

    fn enforce_limit(
        &mut self,
        release_ref: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
    ) -> Result<(), PrefixCacheError> {
        while self.max_entries > 0 && self.blocks.len() > self.max_entries {
            if self.evict_oldest_leaf(None, true, release_ref)? == 0 {
                break;
            }
        }
        Ok(())
    }

    pub fn evict_oldest(
        &mut self,
        claim_free: &mut dyn FnMut(i64) -> Result<bool, PrefixCacheError>,
    ) -> Result<usize, PrefixCacheError> {
        let mut no_release = |_block_id| Ok(());
        self.evict_oldest_leaf(Some(claim_free), false, &mut no_release)
    }

    pub fn evict_for(
        &mut self,
        need: usize,
        claim_free: &mut dyn FnMut(i64) -> Result<bool, PrefixCacheError>,
    ) -> Result<usize, PrefixCacheError> {
        let mut evicted = 0;
        while evicted < need && !self.blocks.is_empty() {
            if self.evict_oldest(claim_free)? == 0 {
                break;
            }
            evicted += 1;
        }
        Ok(evicted)
    }

    pub fn clear(
        &mut self,
        release_ref: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
    ) -> Result<(), PrefixCacheError> {
        if self.claim_refs && !self.release_refs && !self.blocks.is_empty() {
            return Err(PrefixCacheError::Runtime(
                "owning cache requires release_ref to clear".to_owned(),
            ));
        }
        if self.release_refs {
            for entry in self.entries_lru() {
                release_ref(entry.block_id)?;
            }
        }
        self.nodes.clear();
        self.nodes.push(Some(Node::root()));
        self.free_nodes.clear();
        self.blocks.clear();
        self.lru_order.clear();
        self.by_hash.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefixCache, PrefixCacheError, PrefixEntry};
    use crate::mem::block_allocator::BlockAllocator;
    use std::cell::RefCell;

    fn entry(previous: u64, block_id: i64, tokens: &[i64]) -> PrefixEntry {
        PrefixEntry {
            chain_hash: PrefixCache::hash_chain(previous, tokens),
            block_id,
            tokens: tokens.to_vec(),
        }
    }

    #[test]
    fn match_follows_exact_token_pages_and_hashes() {
        let mut cache = PrefixCache::new(0, false, false).unwrap();
        let a = entry(0, 10, &[1, 2, 3, 4]);
        let b = entry(a.chain_hash, 11, &[5, 6, 7, 8]);
        let mut claim_no_ref = |_block_id| Ok(());
        let mut release_no_ref = |_block_id| Ok(());
        cache
            .insert_path(
                &[a.clone(), b.clone()],
                &mut claim_no_ref,
                &mut release_no_ref,
            )
            .unwrap();
        let mut claims = Vec::new();
        let matched = cache
            .match_blocks(
                &[a.chain_hash, b.chain_hash],
                &[a.tokens.clone(), b.tokens.clone()],
                &mut |block| {
                    claims.push(block);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(matched, 2);
        assert_eq!(claims, vec![10, 11]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn invalid_suffix_is_validated_before_tree_or_ref_mutation() {
        let allocator = RefCell::new(BlockAllocator::new(3).unwrap());
        let first = allocator.borrow_mut().alloc().unwrap();
        let second = allocator.borrow_mut().alloc().unwrap();
        let third = allocator.borrow_mut().alloc().unwrap();
        let mut cache = PrefixCache::new(0, true, true).unwrap();
        let a = entry(0, first as i64, &[1, 2, 3, 4]);
        let b = entry(a.chain_hash, second as i64, &[5, 6, 7, 8]);
        let mut claim = |block| {
            allocator
                .borrow_mut()
                .inc_ref(block)
                .map_err(PrefixCacheError::from)
        };
        let mut release = |block| {
            allocator
                .borrow_mut()
                .dec_ref(block)
                .map_err(PrefixCacheError::from)
        };
        cache
            .insert_path(std::slice::from_ref(&a), &mut claim, &mut release)
            .unwrap();
        let refs = allocator.borrow().refcounts.clone();
        let snapshot = cache.tree_snapshot();
        let bad = PrefixEntry {
            chain_hash: b.chain_hash ^ 1,
            block_id: third as i64,
            tokens: b.tokens,
        };
        assert!(
            matches!(cache.insert_path(&[a, bad], &mut claim, &mut release), Err(PrefixCacheError::Value(message)) if message == "prefix hash does not match its radix path")
        );
        assert_eq!(allocator.borrow().refcounts, refs);
        assert_eq!(cache.tree_snapshot(), snapshot);
    }

    #[test]
    fn eviction_removes_lru_leaves_before_shared_ancestors() {
        let allocator = RefCell::new(BlockAllocator::new(3).unwrap());
        let a_block = allocator.borrow_mut().alloc().unwrap();
        let b_block = allocator.borrow_mut().alloc().unwrap();
        let c_block = allocator.borrow_mut().alloc().unwrap();
        let mut cache = PrefixCache::new(0, true, true).unwrap();
        let a = entry(0, a_block as i64, &[1, 2]);
        let b = entry(a.chain_hash, b_block as i64, &[3, 4]);
        let c = entry(a.chain_hash, c_block as i64, &[5, 6]);
        let mut claim = |block| {
            allocator
                .borrow_mut()
                .inc_ref(block)
                .map_err(PrefixCacheError::from)
        };
        let mut release = |block| {
            allocator
                .borrow_mut()
                .dec_ref(block)
                .map_err(PrefixCacheError::from)
        };
        cache
            .insert_path(&[a.clone(), b], &mut claim, &mut release)
            .unwrap();
        cache
            .insert_path(&[a.clone(), c], &mut claim, &mut release)
            .unwrap();
        allocator.borrow_mut().dec_ref(a_block as i64).unwrap();
        allocator.borrow_mut().dec_ref(b_block as i64).unwrap();
        allocator.borrow_mut().dec_ref(c_block as i64).unwrap();
        let mut release_cache_only = |block: i64| {
            let cache_only = allocator.borrow().refcounts[block as usize] == 1;
            if cache_only {
                allocator.borrow_mut().dec_ref(block).unwrap();
                Ok(true)
            } else {
                Ok(false)
            }
        };
        assert_eq!(cache.evict_for(2, &mut release_cache_only).unwrap(), 2);
        assert_eq!(allocator.borrow().refcounts, vec![1, 0, 0]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.evict_for(1, &mut release_cache_only).unwrap(), 1);
        assert_eq!(allocator.borrow().available(), 3);
    }

    #[test]
    fn bounded_owning_cache_and_clear_require_release_callback() {
        assert!(matches!(
            PrefixCache::new(2, true, false),
            Err(PrefixCacheError::Value(_))
        ));
        let mut cache = PrefixCache::new(0, true, false).unwrap();
        let mut claim_no_ref = |_block_id| Ok(());
        let mut release_no_ref = |_block_id| Ok(());
        let first = entry(0, 3, &[1, 2]);
        cache
            .insert_path(&[first], &mut claim_no_ref, &mut release_no_ref)
            .unwrap();
        assert!(
            matches!(cache.clear(&mut release_no_ref), Err(PrefixCacheError::Runtime(message)) if message == "owning cache requires release_ref to clear")
        );
    }
}
