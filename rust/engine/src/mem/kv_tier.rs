//! A disk tier for KV blocks: when the prefix cache drops a block, it can be written to a file and read back later instead
//! of being computed again. This is what makes a restart cheap — the prompt's KV is on disk, not gone.
//!
//! Off unless `FLLM_KV_DISK_DIR` names a directory. `FLLM_KV_DISK_BYTES` sets the budget (default 8 GiB); the least
//! recently used file goes first. Layout: one file per block, named after the block's chain hash, with a small header so a
//! truncated or foreign file is recognised and ignored rather than read as garbage KV.
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
    /// Serialises writes and the budget check: two threads must not both believe there is room.
    guard: Mutex<()>,
}

impl KvTier {
    /// Reads the environment; without `FLLM_KV_DISK_DIR` the tier is off and every call is a no-op.
    pub fn from_env() -> KvTier {
        let dir = std::env::var("FLLM_KV_DISK_DIR").ok().map(|value| value.trim().to_owned()).filter(|value| !value.is_empty());
        let max_bytes = std::env::var("FLLM_KV_DISK_BYTES")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(8 << 30);
        if let Some(path) = dir.as_ref() {
            let _ = fs::create_dir_all(path);
        }
        let used = dir.as_deref().map(|path| dir_size(Path::new(path))).unwrap_or(0);
        KvTier {
            dir: dir.map(PathBuf::from),
            max_bytes,
            used: AtomicU64::new(used),
            spills: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            drops: AtomicU64::new(0),
            guard: Mutex::new(()),
        }
    }

    /// A tier in a temporary directory, for tests.
    pub fn at(dir: impl AsRef<Path>, max_bytes: u64) -> KvTier {
        let path = dir.as_ref().to_path_buf();
        let _ = fs::create_dir_all(&path);
        KvTier {
            dir: Some(path),
            max_bytes,
            used: AtomicU64::new(0),
            spills: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            drops: AtomicU64::new(0),
            guard: Mutex::new(()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.dir.is_some()
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

    /// Every block on disk, cheapest first (by chain depth, so a prefix can be rebuilt in order).
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
        found.sort_by_key(|stored| stored.depth);
        found
    }

    /// Writes one block. A block that does not fit into the budget is not written at all — half a block is worthless.
    pub fn spill(&self, hash: u64, parent: u64, tokens: &[i64], payload: &[u8]) -> bool {
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
        // touching the file is what keeps the LRU honest
        let _ = filetime_now(&path);
        self.hits.fetch_add(1, Ordering::Relaxed);
        let _ = stored;
        Some(payload.to_vec())
    }

    pub fn has(&self, hash: u64) -> bool {
        self.file(hash).is_some_and(|path| path.exists())
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
        })
    }
}

/// A block on disk: its identity, where it belongs in the chain, and the tokens it covers.
#[derive(Debug, Clone)]
pub struct Stored {
    pub hash: u64,
    pub parent: u64,
    pub tokens: Vec<i64>,
    /// How many blocks precede this one; 0 for a block that starts a prompt.
    pub depth: u32,
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
    Some((Stored { hash, parent, tokens, depth: 0 }, &buffer[tokens_end..]))
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
}
