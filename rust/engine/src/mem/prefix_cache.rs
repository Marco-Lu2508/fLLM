use crate::mem::block_allocator::AllocatorError;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
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
    /// Use counter of every block (`ticks`), the blocks by that counter (`order`, oldest first) and the ones among them that have no
    /// children (`leaves`): only a leaf can be evicted, and a long chain would otherwise be searched through for its end on every eviction.
    ticks: HashMap<u64, u64>,
    order: BTreeMap<u64, u64>,
    leaves: BTreeMap<u64, u64>,
    next_tick: u64,
    /// Blocks that eviction must skip (a prefix somebody asked to keep, `Scheduler::pin_chain`).
    pinned: HashSet<u64>,
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
            ticks: HashMap::new(),
            order: BTreeMap::new(),
            leaves: BTreeMap::new(),
            next_tick: 0,
            pinned: HashSet::new(),
        })
    }

    pub fn hash_chain(prev_hash: u64, tokens: &[i64]) -> u64 {
        let mut hash = prev_hash ^ 0x9E37_79B9_7F4A_7C15;
        for token in tokens {
            hash = hash.wrapping_mul(1_000_003) ^ (*token as u64);
        }
        hash ^ (hash >> 29)
    }

    /// Whether a chain hash is in the cache already.
    pub fn contains(&self, hash: u64) -> bool {
        self.by_hash.contains_key(&hash)
    }

    /// The block a chain hash points at.
    pub fn block_of(&self, hash: u64) -> Option<i64> {
        self.by_hash
            .get(&hash)
            .and_then(|index| self.nodes[*index].as_ref())
            .and_then(|node| node.block_id)
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn entries_lru(&self) -> Vec<PrefixEntry> {
        self.order
            .values()
            .filter_map(|chain_hash| self.blocks.get(chain_hash).cloned())
            .collect()
    }

    /// The bookkeeping of the eviction order, checked from scratch: every block has exactly one use counter, `order` and `ticks` agree,
    /// and `leaves` holds exactly the blocks whose node has no children.
    #[cfg(test)]
    pub(crate) fn check_bookkeeping(&self) {
        assert_eq!(self.ticks.len(), self.blocks.len(), "a block without a use counter or the other way round");
        assert_eq!(self.order.len(), self.blocks.len());
        for (hash, tick) in &self.ticks {
            assert_eq!(self.order.get(tick), Some(hash), "order and ticks disagree");
            assert!(self.blocks.contains_key(hash));
        }
        let mut expected = 0;
        for (hash, tick) in &self.ticks {
            let leaf = self.is_leaf_hash(*hash);
            assert_eq!(self.leaves.get(tick) == Some(hash), leaf, "leaf set wrong for {hash:x}");
            expected += usize::from(leaf);
        }
        assert_eq!(self.leaves.len(), expected, "a leaf without a block");
        for hash in &self.pinned {
            assert!(self.blocks.contains_key(hash), "a pin on a block that is gone");
        }
    }

    /// Keeps a cached block out of eviction. `false` when the cache does not hold it.
    pub fn pin(&mut self, chain_hash: u64) -> bool {
        self.blocks.contains_key(&chain_hash) && {
            self.pinned.insert(chain_hash);
            true
        }
    }

    /// Lets a block be evicted again. `false` when it was not pinned.
    pub fn unpin(&mut self, chain_hash: u64) -> bool {
        self.pinned.remove(&chain_hash)
    }

    pub fn is_pinned(&self, chain_hash: u64) -> bool {
        self.pinned.contains(&chain_hash)
    }

    pub fn pinned_len(&self) -> usize {
        self.pinned.len()
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

    /// The child of `parent` that holds `tokens` *under the chain hash `hash`*. Two identities (a deployment salt, a request's
    /// `cache_salt`) can hold the same tokens at the same place of the tree; they are different blocks and must stay apart, so the tokens
    /// alone are not the key.
    fn find_child(&self, parent: usize, tokens: &[i64], hash: u64) -> Option<usize> {
        self.nodes[parent]
            .as_ref()?
            .children
            .iter()
            .find(|(page, child)| {
                page == tokens && self.nodes[*child].as_ref().is_some_and(|node| node.chain_hash == Some(hash))
            })
            .map(|(_, child)| *child)
    }

    fn touch(&mut self, chain_hash: u64) {
        let Some(old) = self.ticks.get(&chain_hash).copied() else { return };
        self.next_tick += 1;
        let tick = self.next_tick;
        self.order.remove(&old);
        self.order.insert(tick, chain_hash);
        if self.leaves.remove(&old).is_some() {
            self.leaves.insert(tick, chain_hash);
        }
        self.ticks.insert(chain_hash, tick);
    }

    /// Whether the node of a hash has no children (the node has to be in the tree already).
    fn is_leaf_hash(&self, chain_hash: u64) -> bool {
        self.by_hash
            .get(&chain_hash)
            .and_then(|index| self.nodes[*index].as_ref())
            .is_some_and(|node| node.children.is_empty())
    }

    fn put_block(&mut self, entry: PrefixEntry) {
        let hash = entry.chain_hash;
        if !self.blocks.contains_key(&hash) {
            self.next_tick += 1;
            let tick = self.next_tick;
            self.ticks.insert(hash, tick);
            self.order.insert(tick, hash);
            if self.is_leaf_hash(hash) {
                self.leaves.insert(tick, hash);
            }
        }
        self.blocks.insert(hash, entry);
    }

    fn remove_block(&mut self, chain_hash: u64) {
        self.blocks.remove(&chain_hash);
        if let Some(tick) = self.ticks.remove(&chain_hash) {
            self.order.remove(&tick);
            self.leaves.remove(&tick);
        }
        self.pinned.remove(&chain_hash);
    }

    /// A node that just got a child is no leaf any more.
    fn no_longer_a_leaf(&mut self, node_index: usize) {
        let hash = self.nodes[node_index].as_ref().and_then(|node| node.chain_hash);
        if let Some(tick) = hash.and_then(|hash| self.ticks.get(&hash)) {
            self.leaves.remove(tick);
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
            let Some(child_index) = self.find_child(node_index, tokens, *chain_hash) else {
                break;
            };
            let child = self.nodes[child_index].as_ref().expect("child node");
            let (Some(block_id), Some(_)) = (child.block_id, child.chain_hash) else {
                break;
            };
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
        self.insert_path_from(0, entries, claim_ref, release_ref)
    }

    /// `insert_path` for a chain whose first hash starts from `root` (the identity of the request: the deployment seed xor the
    /// request's salt; `0` for the default identity).
    pub fn insert_path_from(
        &mut self,
        root: u64,
        entries: &[PrefixEntry],
        claim_ref: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
        release_ref: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
    ) -> Result<(), PrefixCacheError> {
        let mut previous_hash = root;
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
            let existing = self.find_child(node_index, &entry.tokens, entry.chain_hash);
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
                self.no_longer_a_leaf(node_index);
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
        if self.find_child(0, &tokens, chain_hash).is_some() {
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
        let chain_hash = node.chain_hash;
        let block_id = node.block_id;
        let parent_index = node.parent.expect("leaf parent");
        self.nodes[parent_index]
            .as_mut()
            .expect("parent node")
            .children
            .retain(|(_, child)| *child != node_index);
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
        // the parent may have lost its last child: it can be evicted now
        if parent_index != 0 && self.nodes[parent_index].as_ref().is_some_and(|parent| parent.children.is_empty()) {
            if let Some(hash) = self.nodes[parent_index].as_ref().and_then(|parent| parent.chain_hash) {
                if let Some(tick) = self.ticks.get(&hash).copied() {
                    self.leaves.insert(tick, hash);
                }
            }
        }
        Ok(())
    }

    fn evict_oldest_leaf(
        &mut self,
        claim_free: Option<&mut dyn FnMut(i64) -> Result<bool, PrefixCacheError>>,
        release_cache_ref: bool,
        release_ref: &mut dyn FnMut(i64) -> Result<(), PrefixCacheError>,
    ) -> Result<usize, PrefixCacheError> {
        let mut claim_free = claim_free;
        let mut chosen = None;
        // the leaves, oldest first: the first one that is not pinned and that nobody uses is the one to go
        for chain_hash in self.leaves.values() {
            if self.pinned.contains(chain_hash) {
                continue;
            }
            let Some(node_index) = self.by_hash.get(chain_hash).copied() else {
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
            chosen = Some(node_index);
            break;
        }
        let Some(node_index) = chosen else { return Ok(0) };
        self.remove_leaf(node_index, release_cache_ref, release_ref)?;
        Ok(1)
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
        self.ticks.clear();
        self.order.clear();
        self.leaves.clear();
        self.pinned.clear();
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

    #[test]
    fn two_identities_with_the_same_tokens_stay_two_blocks_and_leave_one_another_alone() {
        // the same tokens at the same place of the tree under two different roots (two salts): each lookup finds its own block only
        let mut cache = PrefixCache::new(0, false, false).unwrap();
        let tokens = [1, 2, 3, 4];
        let (root_a, root_b) = (111u64, 222u64);
        let a = entry(root_a, 10, &tokens);
        let b = entry(root_b, 20, &tokens);
        assert_ne!(a.chain_hash, b.chain_hash);
        let mut claim = |_block_id| Ok(());
        let mut release = |_block_id| Ok(());
        cache.insert_path_from(root_a, std::slice::from_ref(&a), &mut claim, &mut release).unwrap();
        cache.insert_path_from(root_b, std::slice::from_ref(&b), &mut claim, &mut release).unwrap();
        assert_eq!(cache.len(), 2, "the second identity was folded into the node of the first");
        let page = vec![tokens.to_vec()];
        let mut got = Vec::new();
        assert_eq!(cache.match_blocks(&[a.chain_hash], &page, &mut |block| { got.push(block); Ok(()) }).unwrap(), 1);
        assert_eq!(cache.match_blocks(&[b.chain_hash], &page, &mut |block| { got.push(block); Ok(()) }).unwrap(), 1);
        assert_eq!(got, vec![10, 20], "each identity gets its own block");
        // a hash of a third identity finds neither
        assert_eq!(cache.match_blocks(&[PrefixCache::hash_chain(333, &tokens)], &page, &mut |_| Ok(())).unwrap(), 0);
        // evicting one of them leaves the other in place
        let mut may_free = |_block_id: i64| Ok(true);
        assert_eq!(cache.evict_oldest(&mut may_free).unwrap(), 1);
        assert_eq!(cache.len(), 1);
        let left = if cache.contains(a.chain_hash) { &a } else { &b };
        assert_eq!(cache.match_blocks(&[left.chain_hash], &page, &mut |_| Ok(())).unwrap(), 1, "the block that was not evicted is gone too");
    }

    fn chain(prev: u64, first_block: i64, pages: &[&[i64]]) -> Vec<PrefixEntry> {
        let mut out = Vec::new();
        let mut hash = prev;
        for (i, page) in pages.iter().enumerate() {
            let e = entry(hash, first_block + i as i64, page);
            hash = e.chain_hash;
            out.push(e);
        }
        out
    }

    type RefFn = Box<dyn FnMut(i64) -> Result<(), PrefixCacheError>>;

    fn no_refs() -> (RefFn, RefFn) {
        (Box::new(|_b: i64| Ok(())), Box::new(|_b: i64| Ok(())))
    }

    #[test]
    fn the_leaf_set_follows_inserts_evictions_and_reads() {
        let mut cache = PrefixCache::new(0, false, false).unwrap();
        let (mut claim, mut release) = no_refs();
        let a = chain(0, 0, &[&[1], &[2], &[3]]);
        let b = chain(a[0].chain_hash, 10, &[&[1], &[9], &[9]]); // shares block a[0]... as parent: different tokens at the root, so this is a sibling chain
        cache.insert_path(&a, &mut claim, &mut release).unwrap();
        cache.check_bookkeeping();
        assert_eq!(cache.leaves.len(), 1, "one chain has one leaf");
        let c = chain(0, 20, &[&[1], &[2], &[7]]); // shares two blocks with `a`, ends in a sibling leaf
        cache.insert_path(&c, &mut claim, &mut release).unwrap();
        cache.check_bookkeeping();
        assert_eq!(cache.leaves.len(), 2);
        let _ = b;
        // evicting leaves one by one walks the chains back to the root
        let mut free = |_b: i64| Ok(true);
        let mut seen = 0;
        while !cache.is_empty() {
            assert_eq!(cache.evict_oldest(&mut free).unwrap(), 1);
            cache.check_bookkeeping();
            seen += 1;
            assert!(seen < 20, "never ends");
        }
        assert_eq!(seen, 4, "four distinct blocks: the shared two, and one leaf per chain");
        assert!(cache.leaves.is_empty() && cache.order.is_empty());
    }

    #[test]
    fn reading_a_block_moves_it_to_the_young_end_and_it_is_evicted_last() {
        let mut cache = PrefixCache::new(0, false, false).unwrap();
        let (mut claim, mut release) = no_refs();
        let a = chain(0, 0, &[&[1]]);
        let b = chain(0, 1, &[&[2]]);
        let c = chain(0, 2, &[&[3]]);
        for chain in [&a, &b, &c] {
            cache.insert_path(chain, &mut claim, &mut release).unwrap();
        }
        // reading `a` makes it the youngest of the three
        let pages = vec![vec![1]];
        assert_eq!(cache.match_blocks(&[a[0].chain_hash], &pages, &mut |_| Ok(())).unwrap(), 1);
        cache.check_bookkeeping();
        let mut free = |_b: i64| Ok(true);
        assert_eq!(cache.evict_oldest(&mut free).unwrap(), 1);
        assert!(!cache.contains(b[0].chain_hash), "the oldest one went first");
        assert_eq!(cache.evict_oldest(&mut free).unwrap(), 1);
        assert!(!cache.contains(c[0].chain_hash));
        assert!(cache.contains(a[0].chain_hash), "the block that was read lasts longest");
    }

    #[test]
    fn a_pinned_chain_is_not_evicted_while_others_are() {
        let mut cache = PrefixCache::new(0, false, false).unwrap();
        let (mut claim, mut release) = no_refs();
        let keep = chain(0, 0, &[&[1], &[2], &[3]]);
        let other = chain(0, 10, &[&[5], &[6]]);
        cache.insert_path(&keep, &mut claim, &mut release).unwrap();
        cache.insert_path(&other, &mut claim, &mut release).unwrap();
        for entry in &keep {
            assert!(cache.pin(entry.chain_hash));
        }
        assert!(!cache.pin(0xdead), "a block the cache does not hold cannot be pinned");
        assert_eq!(cache.pinned_len(), 3);
        let mut free = |_b: i64| Ok(true);
        // the pinned chain is the oldest, and still the other chain goes
        assert_eq!(cache.evict_for(10, &mut free).unwrap(), 2);
        cache.check_bookkeeping();
        assert_eq!(cache.len(), 3);
        assert!(keep.iter().all(|e| cache.contains(e.chain_hash)));
        assert_eq!(cache.evict_for(1, &mut free).unwrap(), 0, "nothing else may go");
        // unpinned, it is evictable again
        for entry in &keep {
            assert!(cache.unpin(entry.chain_hash));
        }
        assert!(!cache.unpin(keep[0].chain_hash));
        assert_eq!(cache.evict_for(10, &mut free).unwrap(), 3);
        cache.check_bookkeeping();
        assert!(cache.is_empty());
    }

    #[test]
    fn clearing_forgets_the_pins_too() {
        // a pin that survived `clear` would sit on a hash that comes back later (the same prompt, new block) and keep it from ever being evicted
        let mut cache = PrefixCache::new(0, false, false).unwrap();
        let (mut claim, mut release) = no_refs();
        let a = chain(0, 0, &[&[1], &[2]]);
        cache.insert_path(&a, &mut claim, &mut release).unwrap();
        assert!(cache.pin(a[1].chain_hash));
        cache.clear(&mut release).unwrap();
        cache.check_bookkeeping();
        assert_eq!(cache.pinned_len(), 0, "a pin outlived the cache it was set in");
        cache.insert_path(&a, &mut claim, &mut release).unwrap();
        let mut free = |_b: i64| Ok(true);
        assert_eq!(cache.evict_for(5, &mut free).unwrap(), 2, "the old pin kept a new block from being evicted");
    }

    #[test]
    fn eviction_cost_does_not_grow_with_the_length_of_the_chain() {
        // Counted, not timed: the search for the next victim looks at leaves only. With one long chain there is one leaf, so every
        // eviction inspects one block however long the chain is (it used to walk the whole use order, 6 s for 500 evictions of 64000).
        let mut cache = PrefixCache::new(0, false, false).unwrap();
        let (mut claim, mut release) = no_refs();
        let n = 3000;
        let mut hash = 0u64;
        let mut entries = Vec::new();
        for i in 0..n {
            let e = entry(hash, i as i64, &[i as i64]);
            hash = e.chain_hash;
            entries.push(e);
        }
        cache.insert_path(&entries, &mut claim, &mut release).unwrap();
        assert_eq!(cache.leaves.len(), 1);
        let mut asked = 0usize;
        let mut free = |_b: i64| {
            asked += 1;
            Ok(true)
        };
        assert_eq!(cache.evict_for(500, &mut free).unwrap(), 500);
        assert_eq!(asked, 500, "one question per eviction: the only leaf");
        cache.check_bookkeeping();
        assert_eq!(cache.len(), n - 500);
    }
}
