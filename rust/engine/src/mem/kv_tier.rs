//! A disk tier for KV blocks: when the prefix cache drops a block, it can be written to a file and read back later instead
//! of being computed again. This is what makes a restart cheap — the prompt's KV is on disk, not gone.
//!
//! Two tiers behind one door. Host memory (`FLLM_KV_RAM_BYTES`) is the fast one: every block that is written lands there first, the
//! least recently used block goes when the budget is full, and a restart empties it. The disk is the one that survives. A read looks in
//! memory, then on disk, and a block found on disk is put into memory again. Either tier can be off; the tier as a whole is off when both are.
//!
//! Disk: off unless `FLLM_KV_DISK_DIR` names a directory. `FLLM_KV_DISK_BYTES` sets the budget (default 8 GiB); the least
//! recently used file goes first. Layout: one file per block, named after the block's chain hash, with a small header so a
//! truncated or foreign file is recognised and ignored rather than read as garbage KV.
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const MAGIC: &[u8; 8] = b"FLLMKV02";
const HEADER: usize = 8 + 8 + 8 + 8 + 4; // magic, payload length, hash, parent hash, token count

/// Where spilled blocks live and how much room they may use.
pub struct KvTier {
    dir: Option<PathBuf>,
    max_bytes: u64,
    used: AtomicU64,
    spills: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    drops: AtomicU64,
    /// Budget of the memory tier in bytes (`0`: off).
    ram_max: u64,
    ram: Mutex<Ram>,
    ram_hits: AtomicU64,
    /// Serialises writes and the budget check: two threads must not both believe there is room.
    guard: Mutex<()>,
}

impl KvTier {
    fn new(dir: Option<PathBuf>, max_bytes: u64, used: u64, ram_max: u64) -> KvTier {
        KvTier {
            dir,
            max_bytes,
            used: AtomicU64::new(used),
            spills: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            drops: AtomicU64::new(0),
            ram_max,
            ram: Mutex::new(Ram::default()),
            ram_hits: AtomicU64::new(0),
            guard: Mutex::new(()),
        }
    }

    /// Reads the environment; without `FLLM_KV_DISK_DIR` the disk is off, without `FLLM_KV_RAM_BYTES` the memory tier is off, and with
    /// neither every call is a no-op.
    pub fn from_env() -> KvTier {
        let dir = std::env::var("FLLM_KV_DISK_DIR").ok().map(|value| value.trim().to_owned()).filter(|value| !value.is_empty());
        let max_bytes = std::env::var("FLLM_KV_DISK_BYTES")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(8 << 30);
        let ram_max = std::env::var("FLLM_KV_RAM_BYTES").ok().and_then(|value| value.trim().parse::<u64>().ok()).unwrap_or(0);
        if let Some(path) = dir.as_ref() {
            let _ = fs::create_dir_all(path);
        }
        let used = dir.as_deref().map(|path| dir_size(Path::new(path))).unwrap_or(0);
        KvTier::new(dir.map(PathBuf::from), max_bytes, used, ram_max)
    }

    /// A tier in a temporary directory, for tests.
    pub fn at(dir: impl AsRef<Path>, max_bytes: u64) -> KvTier {
        let path = dir.as_ref().to_path_buf();
        let _ = fs::create_dir_all(&path);
        KvTier::new(Some(path), max_bytes, 0, 0)
    }

    /// Only the memory tier, `max_bytes` large (nothing survives a restart).
    pub fn ram_only(max_bytes: u64) -> KvTier {
        KvTier::new(None, 0, 0, max_bytes)
    }

    /// The same tier for one model identity: the blocks of the disk go into a folder of their own below the configured one, named after
    /// the identity. KV values depend on the weights (another quantization, another adapter), on the KV cache format and on the block
    /// size, and the chain hash of a block knows only the tokens: without this a server with another quantization that uses the same
    /// folder would read KV values that its own weights never produced. The memory tier needs nothing: it lives in one process.
    pub fn with_identity(mut self, identity: &str) -> KvTier {
        use sha2::{Digest, Sha256};
        if let Some(dir) = self.dir.take() {
            let name: String = Sha256::digest(identity.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect();
            let dir = dir.join(name);
            let _ = fs::create_dir_all(&dir);
            self.used = AtomicU64::new(dir_size(&dir));
            self.dir = Some(dir);
        }
        self
    }

    /// The same tier with a memory tier of `ram_max` bytes in front of it.
    pub fn with_ram(mut self, ram_max: u64) -> KvTier {
        self.ram_max = ram_max;
        self
    }

    pub fn enabled(&self) -> bool {
        self.dir.is_some() || self.ram_max > 0
    }

    fn file(&self, hash: u64) -> Option<PathBuf> {
        self.dir.as_ref().map(|dir| dir.join(format!("{hash:016x}.kv")))
    }

    /// What a stored block carries besides its KV.
    pub fn peek(&self, hash: u64) -> Option<Stored> {
        let path = self.file(hash)?;
        let mut buffer = Vec::new();
        fs::File::open(&path).and_then(|mut file| file.read_to_end(&mut buffer)).ok()?;
        parse(&buffer, hash).map(|stored| stored.0)
    }

    /// The file that keeps the pins across a restart: the hash of the last pinned block of every pinned chain, one per line. The chain
    /// back to its root is on disk as well (every block names its parent), so these hashes are enough to pin a chain again.
    fn pins_file(&self) -> Option<PathBuf> {
        self.dir.as_ref().map(|dir| dir.join("pins.txt"))
    }

    /// Replaces the saved pins. The file is written under a temporary name and renamed, like a block.
    pub fn save_pins(&self, heads: &[u64]) -> bool {
        let Some(path) = self.pins_file() else { return false };
        let text: String = heads.iter().map(|hash| format!("{hash:016x}\n")).collect();
        let temporary = path.with_extension("txt.part");
        if fs::write(&temporary, text).is_err() {
            let _ = fs::remove_file(&temporary);
            return false;
        }
        fs::rename(&temporary, &path).is_ok()
    }

    /// The saved pins (a line that is not a hash is ignored).
    pub fn load_pins(&self) -> Vec<u64> {
        let Some(path) = self.pins_file() else { return Vec::new() };
        let Ok(text) = fs::read_to_string(path) else { return Vec::new() };
        text.lines().filter_map(|line| u64::from_str_radix(line.trim(), 16).ok()).collect()
    }

    /// Every block on disk, in no particular order. A chain is put together from the `parent` hashes (`Runtime::warm_prefix_cache`).
    pub fn entries(&self) -> Vec<Stored> {
        let Some(dir) = self.dir.as_ref() else { return Vec::new() };
        let mut found: Vec<Stored> = Vec::new();
        let Ok(read) = fs::read_dir(dir) else { return Vec::new() };
        for entry in read.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("kv") {
                continue;
            }
            let Some(hash) = path.file_stem().and_then(|s| s.to_str()).and_then(|s| u64::from_str_radix(s, 16).ok()) else {
                continue;
            };
            if let Some(stored) = self.peek(hash) {
                found.push(stored);
            }
        }
        found
    }

    /// Writes one block. A block that does not fit into the budget is not written at all — half a block is worthless.
    pub fn spill(&self, hash: u64, parent: u64, tokens: &[i64], payload: &[u8]) -> bool {
        let in_memory = self.ram_put(hash, parent, tokens, payload);
        let on_disk = self.spill_disk(hash, parent, tokens, payload);
        if in_memory && !on_disk {
            // `spills` counts blocks written to either tier once; the disk path counted its own
            self.spills.fetch_add(1, Ordering::Relaxed);
        }
        in_memory || on_disk
    }

    fn spill_disk(&self, hash: u64, parent: u64, tokens: &[i64], payload: &[u8]) -> bool {
        let Some(path) = self.file(hash) else { return false };
        let size = (HEADER + tokens.len() * 8 + payload.len()) as u64;
        if size > self.max_bytes {
            self.drops.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let _guard = self.guard.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        self.make_room(size);
        let mut buffer = Vec::with_capacity(size as usize);
        buffer.extend_from_slice(MAGIC);
        buffer.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        buffer.extend_from_slice(&hash.to_le_bytes());
        buffer.extend_from_slice(&parent.to_le_bytes());
        buffer.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for token in tokens {
            buffer.extend_from_slice(&token.to_le_bytes());
        }
        buffer.extend_from_slice(payload);
        // write to a temporary name and rename: a crash must not leave a half-written block under a valid name
        let temporary = path.with_extension("kv.part");
        let written = fs::File::create(&temporary).and_then(|mut file| {
            file.write_all(&buffer)?;
            file.sync_data()
        });
        if written.is_err() {
            let _ = fs::remove_file(&temporary);
            self.drops.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if fs::rename(&temporary, &path).is_err() {
            let _ = fs::remove_file(&temporary);
            self.drops.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        self.used.fetch_add(size, Ordering::Relaxed);
        self.spills.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Reads one block back. A file that does not carry the expected header is ignored, not trusted.
    pub fn load(&self, hash: u64) -> Option<Vec<u8>> {
        self.read(hash, None)
    }

    /// Like `load`, for a block whose tokens the caller knows: a file that carries other tokens under this hash (a collision of the 64-bit
    /// chain hash, or a directory that was used by something else) is not the block that is meant. It is left alone and counts as a miss.
    pub fn load_for(&self, hash: u64, tokens: &[i64]) -> Option<Vec<u8>> {
        self.read(hash, Some(tokens))
    }

    fn read(&self, hash: u64, tokens: Option<&[i64]>) -> Option<Vec<u8>> {
        if let Some(payload) = self.ram_get(hash, tokens) {
            self.ram_hits.fetch_add(1, Ordering::Relaxed);
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(payload);
        }
        let path = self.file(hash)?;
        let mut buffer = Vec::new();
        if fs::File::open(&path).and_then(|mut file| file.read_to_end(&mut buffer)).is_err() {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let Some((stored, payload)) = parse(&buffer, hash) else {
            let _ = fs::remove_file(&path);
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        if tokens.is_some_and(|expected| expected != stored.tokens.as_slice()) {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        // touching the file is what keeps the LRU honest
        let _ = filetime_now(&path);
        self.hits.fetch_add(1, Ordering::Relaxed);
        // a block that had to come from the disk is wanted again soon: keep it in memory
        self.ram_put(hash, stored.parent, &stored.tokens, payload);
        Some(payload.to_vec())
    }

    pub fn has(&self, hash: u64) -> bool {
        self.ram_has(hash) || self.file(hash).is_some_and(|path| path.exists())
    }

    /// Drops the oldest files until `needed` bytes fit.
    fn make_room(&self, needed: u64) {
        let mut guard = 0;
        while self.used.load(Ordering::Relaxed) + needed > self.max_bytes && guard < 100_000 {
            guard += 1;
            let oldest = self.oldest_file();
            let Some((path, size)) = oldest else { break };
            if fs::remove_file(&path).is_ok() {
                self.used.fetch_sub(size.min(self.used.load(Ordering::Relaxed)), Ordering::Relaxed);
            } else {
                break;
            }
        }
    }

    fn oldest_file(&self) -> Option<(PathBuf, u64)> {
        let dir = self.dir.as_ref()?;
        let mut oldest: Option<(std::time::SystemTime, PathBuf, u64)> = None;
        for entry in fs::read_dir(dir).ok()?.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("kv") {
                continue;
            }
            let meta = entry.metadata().ok()?;
            let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            let size = meta.len();
            if oldest.as_ref().is_none_or(|(when, _, _)| modified < *when) {
                oldest = Some((modified, path, size));
            }
        }
        oldest.map(|(_, path, size)| (path, size))
    }

    /// Deletes every stored block; returns how many were removed.
    pub fn clear(&self) -> usize {
        let in_memory = self.ram_clear();
        self.clear_disk().max(in_memory)
    }

    fn clear_disk(&self) -> usize {
        let Some(dir) = self.dir.as_ref() else { return 0 };
        let Ok(read) = fs::read_dir(dir) else { return 0 };
        let mut removed = 0;
        for entry in read.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("kv") && fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
        self.used.store(0, Ordering::Relaxed);
        removed
    }

    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "enabled": self.enabled(),
            "dir": self.dir.as_ref().map(|p| p.display().to_string()),
            "max_bytes": self.max_bytes,
            "used_bytes": self.used.load(Ordering::Relaxed),
            "spills": self.spills.load(Ordering::Relaxed),
            "hits": self.hits.load(Ordering::Relaxed),
            "misses": self.misses.load(Ordering::Relaxed),
            "drops": self.drops.load(Ordering::Relaxed),
            "ram_max_bytes": self.ram_max,
            "ram_used_bytes": self.ram_used(),
            "ram_blocks": self.ram_blocks(),
            "ram_hits": self.ram_hits.load(Ordering::Relaxed),
        })
    }
}

impl KvTier {
    fn ram_lock(&self) -> std::sync::MutexGuard<'_, Ram> {
        self.ram.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Puts a block into memory, dropping the least recently used ones until it fits. A block larger than the whole budget is not kept.
    fn ram_put(&self, hash: u64, parent: u64, tokens: &[i64], payload: &[u8]) -> bool {
        let cost = ram_cost(tokens.len(), payload.len());
        if self.ram_max == 0 || cost > self.ram_max {
            return false;
        }
        let mut guard = self.ram_lock();
        let Ram { used, tick, blocks, order } = &mut *guard;
        if let Some(old) = blocks.remove(&hash) {
            order.remove(&old.tick);
            *used -= ram_cost(old.tokens.len(), old.payload.len());
        }
        while *used + cost > self.ram_max {
            let Some((_, oldest)) = order.pop_first() else { break };
            if let Some(gone) = blocks.remove(&oldest) {
                *used -= ram_cost(gone.tokens.len(), gone.payload.len());
            }
        }
        *tick += 1;
        order.insert(*tick, hash);
        *used += cost;
        blocks.insert(hash, RamBlock { parent, tokens: tokens.to_vec(), payload: payload.to_vec(), tick: *tick });
        true
    }

    /// A block from memory; reading it makes it the most recently used one. With `tokens` given, a block that carries other tokens under
    /// this hash is not the one that is meant.
    fn ram_get(&self, hash: u64, tokens: Option<&[i64]>) -> Option<Vec<u8>> {
        if self.ram_max == 0 {
            return None;
        }
        let mut guard = self.ram_lock();
        let Ram { tick, blocks, order, .. } = &mut *guard;
        let block = blocks.get_mut(&hash)?;
        if tokens.is_some_and(|expected| expected != block.tokens.as_slice()) {
            return None;
        }
        order.remove(&block.tick);
        *tick += 1;
        block.tick = *tick;
        order.insert(*tick, hash);
        Some(block.payload.clone())
    }

    fn ram_has(&self, hash: u64) -> bool {
        self.ram_max > 0 && self.ram_lock().blocks.contains_key(&hash)
    }

    fn ram_clear(&self) -> usize {
        let mut guard = self.ram_lock();
        let count = guard.blocks.len();
        *guard = Ram::default();
        count
    }

    fn ram_used(&self) -> u64 {
        self.ram_lock().used
    }

    fn ram_blocks(&self) -> usize {
        self.ram_lock().blocks.len()
    }

    /// What a block in memory knows about its place in the chain, for the tests.
    #[cfg(test)]
    fn ram_parent(&self, hash: u64) -> Option<u64> {
        self.ram_lock().blocks.get(&hash).map(|block| block.parent)
    }
}

struct RamBlock {
    /// Kept so that a block in memory carries what a file on disk carries.
    #[cfg_attr(not(test), allow(dead_code))]
    parent: u64,
    tokens: Vec<i64>,
    payload: Vec<u8>,
    tick: u64,
}

/// The blocks in memory and, by use, the order in which they go: `order` maps a counter to the hash that was used at that count.
#[derive(Default)]
struct Ram {
    used: u64,
    tick: u64,
    blocks: HashMap<u64, RamBlock>,
    order: BTreeMap<u64, u64>,
}

fn ram_cost(tokens: usize, payload: usize) -> u64 {
    (tokens * 8 + payload) as u64
}

/// A block on disk: its identity, where it belongs in the chain, and the tokens it covers.
#[derive(Debug, Clone)]
pub struct Stored {
    pub hash: u64,
    pub parent: u64,
    pub tokens: Vec<i64>,
}

/// Checks the header and returns the metadata plus the KV payload.
fn parse(buffer: &[u8], hash: u64) -> Option<(Stored, &[u8])> {
    if buffer.len() < HEADER || &buffer[..8] != MAGIC {
        return None;
    }
    let length = u64::from_le_bytes(buffer[8..16].try_into().ok()?) as usize;
    let stored_hash = u64::from_le_bytes(buffer[16..24].try_into().ok()?);
    let parent = u64::from_le_bytes(buffer[24..32].try_into().ok()?);
    let count = u32::from_le_bytes(buffer[32..36].try_into().ok()?) as usize;
    if stored_hash != hash {
        return None;
    }
    let tokens_end = HEADER.checked_add(count * 8)?;
    if buffer.len() != tokens_end.checked_add(length)? {
        return None;
    }
    let mut tokens = Vec::with_capacity(count);
    for i in 0..count {
        let start = HEADER + i * 8;
        tokens.push(i64::from_le_bytes(buffer[start..start + 8].try_into().ok()?));
    }
    Some((Stored { hash, parent, tokens }, &buffer[tokens_end..]))
}

fn dir_size(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.path().extension().and_then(|e| e.to_str()) == Some("kv"))
                .filter_map(|entry| entry.metadata().ok().map(|meta| meta.len()))
                .sum()
        })
        .unwrap_or(0)
}

/// Marks the file as just used, so the LRU sees it. `set_modified` is std since 1.75; the call is allowed to fail.
fn filetime_now(path: &Path) -> std::io::Result<()> {
    let file = fs::OpenOptions::new().write(true).open(path)?;
    file.set_modified(std::time::SystemTime::now())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fllm-kv-tier-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_block_survives_a_write_and_a_read() {
        let tier = KvTier::at(temporary("roundtrip"), 1 << 20);
        assert!(tier.spill(0xabc, 0, &[1, 2], b"kv bytes"));
        assert!(tier.has(0xabc));
        assert_eq!(tier.load(0xabc).as_deref(), Some(&b"kv bytes"[..]));
        assert_eq!(tier.stats()["spills"], 1);
        assert_eq!(tier.stats()["hits"], 1);
    }

    #[test]
    fn a_block_remembers_its_parent_and_its_tokens() {
        let tier = KvTier::at(temporary("meta"), 1 << 20);
        assert!(tier.spill(0x22, 0x11, &[7, 8, 9], b"kv"));
        let stored = tier.peek(0x22).expect("the block on disk");
        assert_eq!(stored.parent, 0x11);
        assert_eq!(stored.tokens, vec![7, 8, 9]);
        assert_eq!(stored.hash, 0x22);
        // the payload is still exactly the KV that went in
        assert_eq!(tier.load(0x22).as_deref(), Some(&b"kv"[..]));
    }

    #[test]
    fn a_missing_block_is_a_miss_and_not_a_panic() {
        let tier = KvTier::at(temporary("missing"), 1 << 20);
        assert!(!tier.has(1));
        assert!(tier.load(1).is_none());
        assert_eq!(tier.stats()["misses"], 1);
    }

    #[test]
    fn a_foreign_file_is_ignored_and_removed() {
        let tier = KvTier::at(temporary("foreign"), 1 << 20);
        let path = tier.file(7).unwrap();
        fs::write(&path, b"not a kv block at all").unwrap();
        assert!(tier.load(7).is_none(), "a file without the magic must not be trusted");
        assert!(!path.exists(), "and it is thrown away");
    }

    #[test]
    fn a_truncated_payload_is_refused() {
        let tier = KvTier::at(temporary("truncated"), 1 << 20);
        assert!(tier.spill(9, 0, &[1], b"0123456789"));
        let path = tier.file(9).unwrap();
        let mut data = fs::read(&path).unwrap();
        data.truncate(data.len() - 3);
        fs::write(&path, &data).unwrap();
        assert!(tier.load(9).is_none(), "a short block is worthless KV");
    }

    #[test]
    fn the_budget_drops_the_oldest_block() {
        let tier = KvTier::at(temporary("budget"), HEADER as u64 * 2 + 8);
        assert!(tier.spill(1, 0, &[1], b"aaaa"));
        std::thread::sleep(std::time::Duration::from_millis(12));
        assert!(tier.spill(2, 1, &[2], b"bbbb"));
        // the third one only fits if the first makes room
        assert!(tier.spill(3, 2, &[3], b"cccc"));
        assert!(!tier.has(1), "the oldest block went first");
        assert!(tier.has(3));
    }

    #[test]
    fn a_block_larger_than_the_budget_is_not_written() {
        let tier = KvTier::at(temporary("too-big"), 16);
        assert!(!tier.spill(5, 0, &[5], &[0u8; 64]));
        assert!(!tier.has(5));
        assert_eq!(tier.stats()["drops"], 1);
    }

    #[test]
    fn clearing_removes_every_block() {
        let tier = KvTier::at(temporary("clear"), 1 << 20);
        assert!(tier.spill(1, 0, &[1], b"aa"));
        assert!(tier.spill(2, 1, &[2], b"bb"));
        assert_eq!(tier.clear(), 2);
        assert!(!tier.has(1) && !tier.has(2));
        assert_eq!(tier.stats()["used_bytes"], 0);
    }

    #[test]
    fn without_a_directory_the_tier_is_off() {
        let tier = KvTier::from_env();
        if std::env::var("FLLM_KV_DISK_DIR").is_err() {
            assert!(!tier.enabled());
            assert!(!tier.spill(1, 0, &[1], b"x"));
            assert!(tier.load(1).is_none());
        }
    }

    #[test]
    fn a_block_in_memory_is_served_without_a_disk_and_remembers_its_place() {
        let tier = KvTier::ram_only(1 << 20);
        assert!(tier.enabled());
        assert!(!tier.has(0x22));
        assert!(tier.spill(0x22, 0x11, &[7, 8, 9], b"kv"));
        assert!(tier.has(0x22));
        assert_eq!(tier.load(0x22).as_deref(), Some(&b"kv"[..]));
        assert_eq!(tier.ram_parent(0x22), Some(0x11));
        assert_eq!(tier.stats()["ram_hits"], 1);
        assert_eq!(tier.stats()["spills"], 1, "a block written to memory only counts as written"); 
        assert_eq!(tier.stats()["ram_blocks"], 1);
        // nothing is on a disk that is not there
        assert!(tier.entries().is_empty());
        assert!(tier.load_for(0x22, &[7, 8, 10]).is_none(), "a block with other tokens under this hash is not the one meant");
        assert_eq!(tier.load_for(0x22, &[7, 8, 9]).as_deref(), Some(&b"kv"[..]));
    }

    #[test]
    fn the_memory_budget_drops_the_block_used_least_recently() {
        let cost = ram_cost(1, 4);
        let tier = KvTier::ram_only(cost * 2);
        assert!(tier.spill(1, 0, &[1], b"aaaa"));
        assert!(tier.spill(2, 1, &[2], b"bbbb"));
        // reading block 1 makes block 2 the oldest
        assert!(tier.load(1).is_some());
        assert!(tier.spill(3, 2, &[3], b"cccc"));
        assert!(tier.has(1) && tier.has(3), "the block that was just read stayed");
        assert!(!tier.has(2), "the least recently used block went");
        assert_eq!(tier.stats()["ram_used_bytes"], cost * 2);
        // a block larger than the whole budget is not kept, and nothing else is thrown out for it
        assert!(!tier.spill(4, 3, &[4], &[0u8; 64]));
        assert!(tier.has(1) && tier.has(3));
        // writing a block again replaces it and does not count twice
        assert!(tier.spill(1, 0, &[1], b"AAAA"));
        assert_eq!(tier.stats()["ram_used_bytes"], cost * 2);
        assert_eq!(tier.load(1).as_deref(), Some(&b"AAAA"[..]));
    }

    #[test]
    fn memory_is_read_first_and_a_block_from_the_disk_comes_back_into_memory() {
        let dir = temporary("two-tiers");
        let tier = KvTier::at(&dir, 1 << 20).with_ram(1 << 20);
        assert!(tier.spill(5, 0, &[5], b"both"));
        // gone from the disk: memory still has it
        let path = tier.file(5).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(tier.has(5));
        assert_eq!(tier.load(5).as_deref(), Some(&b"both"[..]));
        assert_eq!(tier.stats()["ram_hits"], 1);
        // gone from memory: the disk still has it, and reading it puts it back into memory
        assert!(tier.spill(6, 5, &[6], b"disk"));
        assert_eq!(tier.ram_clear(), 2);
        assert!(!tier.ram_has(6));
        assert_eq!(tier.load(6).as_deref(), Some(&b"disk"[..]));
        assert!(tier.ram_has(6), "a block read from the disk is kept in memory");
        assert_eq!(tier.ram_parent(6), Some(5), "with the place in the chain the file carried");
        assert_eq!(tier.stats()["ram_hits"], 1, "and the read from the disk is not counted as a memory hit");
    }

    #[test]
    fn clearing_empties_both_tiers() {
        let tier = KvTier::at(temporary("clear-both"), 1 << 20).with_ram(1 << 20);
        assert!(tier.spill(1, 0, &[1], b"aa"));
        assert!(tier.spill(2, 1, &[2], b"bb"));
        assert_eq!(tier.clear(), 2);
        assert!(!tier.has(1) && !tier.has(2));
        assert_eq!(tier.stats()["ram_blocks"], 0);
        assert_eq!(tier.stats()["ram_used_bytes"], 0);
        let only = KvTier::ram_only(1 << 20);
        assert!(only.spill(3, 0, &[3], b"cc"));
        assert_eq!(only.clear(), 1);
        assert!(!only.has(3));
    }

    #[test]
    fn a_tier_without_a_directory_and_without_memory_is_off() {
        let off = KvTier::new(None, 0, 0, 0);
        assert!(!off.enabled());
        assert!(!off.spill(1, 0, &[1], b"x"));
        assert!(!off.has(1));
        assert!(off.load(1).is_none());
    }

    #[test]
    fn two_identities_on_one_folder_do_not_see_each_other_and_one_identity_finds_its_own_again() {
        let dir = temporary("identity");
        let q8 = KvTier::at(&dir, 1 << 20).with_identity("model-digest|q8_0|f16");
        let q4 = KvTier::at(&dir, 1 << 20).with_identity("model-digest|q4_k_m|f16");
        assert!(q8.spill(0x10, 0, &[1, 2], b"values of the q8 weights"));
        assert!(!q4.has(0x10), "a server with other weights saw a block it did not write");
        assert!(q4.load(0x10).is_none());
        assert!(q4.entries().is_empty(), "and the restart read of that server listed it");
        assert!(q4.spill(0x10, 0, &[1, 2], b"values of the q4 weights"), "the same hash can exist once per identity");
        assert_eq!(q8.load(0x10).as_deref(), Some(&b"values of the q8 weights"[..]));
        assert_eq!(q4.load(0x10).as_deref(), Some(&b"values of the q4 weights"[..]));
        // the same identity started again finds what it wrote
        let again = KvTier::at(&dir, 1 << 20).with_identity("model-digest|q8_0|f16");
        assert!(again.has(0x10));
        assert_eq!(again.entries().len(), 1);
        // deleting the blocks of one identity leaves the other's
        assert_eq!(q4.clear(), 1);
        assert!(again.has(0x10), "clearing one identity removed the files of another");
        // the budget is counted per identity folder
        assert!(again.stats()["used_bytes"].as_u64().unwrap() > 0);
        assert_eq!(q4.stats()["used_bytes"], 0);
    }

    #[test]
    fn an_identity_does_not_change_a_tier_without_a_folder() {
        let tier = KvTier::ram_only(1 << 20).with_identity("anything");
        assert!(tier.enabled());
        assert!(tier.spill(1, 0, &[1], b"x"));
        assert_eq!(tier.load(1).as_deref(), Some(&b"x"[..]));
    }
}
