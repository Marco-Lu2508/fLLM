//! Checkpoints of the recurrent state, so that a model with recurrent layers (Gated DeltaNet in Qwen3.5 and Qwen3-Next, Mamba, RWKV, ...)
//! can skip the start of a prompt it has read before.
//!
//! A model with only attention layers reuses old KV blocks (`prefix_cache`). A recurrent layer folds the whole history into one small state
//! instead, so a block cannot be pointed at: the state *at the end of the shared start* has to be copied out when the prompt was read and put
//! back when a new prompt begins with the same tokens. That is the same idea as llama.cpp's context checkpoints. A checkpoint sits on a block
//! boundary (the KV blocks of the attention layers of a hybrid model come from the prefix cache up to the same point) and is keyed by the chain
//! hash of the tokens before it, which includes the deployment seed and the `cache_salt` of the request, so two tenants never share one.
//!
//! The checkpoints live in host memory under a byte budget; the least recently used one goes first. A checkpoint of Qwen3.5-9B is about
//! 100 MiB (the 16 sequence slots of the model take 1.6 GiB), so the default budget holds a handful: one per conversation that is still open.

use std::collections::HashMap;

/// What the user chose for `FLLM_STATE_CACHE` (and Settings > Engine): `auto` (the default), `off`, or a size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateCacheMode {
    /// On, with the default budget.
    Auto,
    Off,
    /// On, with this many bytes.
    Bytes(u64),
}

/// The default budget: room for about ten checkpoints of a 9B model.
pub const DEFAULT_BUDGET: u64 = 1 << 30;

impl StateCacheMode {
    /// `unset`, `auto`, `on`: automatic. `off`, `0`, `no`, `false`: off. A number with an optional `K`, `M`, `G` (powers of 1024): a budget.
    /// Anything else is automatic, so that a typo does not switch the feature off silently.
    pub fn parse(text: &str) -> StateCacheMode {
        let t = text.trim().to_ascii_lowercase();
        match t.as_str() {
            "" | "auto" | "on" | "yes" | "true" | "1" => StateCacheMode::Auto,
            "off" | "0" | "no" | "false" => StateCacheMode::Off,
            _ => {
                let (digits, unit) = t.split_at(t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len()));
                let unit = unit.trim();
                let factor = match unit.strip_suffix("ib").or_else(|| unit.strip_suffix('b')).unwrap_or(unit) {
                    "" => 1,
                    "k" | "kib" => 1 << 10,
                    "m" | "mib" => 1 << 20,
                    "g" | "gib" => 1 << 30,
                    _ => return StateCacheMode::Auto,
                };
                match digits.parse::<u64>() {
                    Ok(n) if n > 0 => StateCacheMode::Bytes(n.saturating_mul(factor)),
                    _ => StateCacheMode::Auto,
                }
            }
        }
    }

    pub fn from_env() -> StateCacheMode {
        // FLLM_STATE_CACHE_BYTES (written by Settings > Engine) is a budget when the mode itself is not "off"
        let mode = StateCacheMode::parse(&std::env::var("FLLM_STATE_CACHE").unwrap_or_default());
        match (mode, std::env::var("FLLM_STATE_CACHE_BYTES").ok().map(|v| StateCacheMode::parse(&v))) {
            (StateCacheMode::Off, _) => StateCacheMode::Off,
            (StateCacheMode::Auto, Some(StateCacheMode::Bytes(n))) => StateCacheMode::Bytes(n),
            (m, _) => m,
        }
    }

    pub fn enabled(self) -> bool {
        self != StateCacheMode::Off
    }

    pub fn budget(self) -> u64 {
        match self {
            StateCacheMode::Bytes(n) => n,
            _ => DEFAULT_BUDGET,
        }
    }
}

struct Entry<S> {
    snapshot: S,
    tokens: usize,
    bytes: usize,
    used: u64,
}

/// The checkpoints. `S` is the snapshot type (`SequenceSnapshot` in the runtime; a small stand-in in the tests).
pub struct StateCheckpoints<S> {
    map: HashMap<u64, Entry<S>>,
    bytes: usize,
    budget: u64,
    clock: u64,
    pub hits: u64,
    pub stored: u64,
    pub evicted: u64,
}

impl<S: Clone> StateCheckpoints<S> {
    pub fn new(budget: u64) -> Self {
        StateCheckpoints { map: HashMap::new(), bytes: 0, budget, clock: 0, hits: 0, stored: 0, evicted: 0 }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn budget(&self) -> u64 {
        self.budget
    }

    pub fn contains(&self, hash: u64) -> bool {
        self.map.contains_key(&hash)
    }

    /// Stores a checkpoint and returns the hashes that had to go to stay inside the budget. The newest one always stays, even when it alone
    /// is larger than the budget (a budget smaller than one state would otherwise make the feature a no-op without saying so).
    pub fn insert(&mut self, hash: u64, tokens: usize, bytes: usize, snapshot: S) -> Vec<u64> {
        self.clock += 1;
        if let Some(old) = self.map.remove(&hash) {
            self.bytes -= old.bytes;
        }
        self.map.insert(hash, Entry { snapshot, tokens, bytes, used: self.clock });
        self.bytes += bytes;
        self.stored += 1;
        let mut dropped = Vec::new();
        while self.bytes as u64 > self.budget && self.map.len() > 1 {
            let Some(oldest) = self.map.iter().filter(|(h, _)| **h != hash).min_by_key(|(_, e)| e.used).map(|(h, _)| *h) else { break };
            if let Some(e) = self.map.remove(&oldest) {
                self.bytes -= e.bytes;
                self.evicted += 1;
                dropped.push(oldest);
            }
        }
        dropped
    }

    /// A copy of the checkpoint (the stored one stays for the next prompt) and the number of tokens it covers.
    pub fn take_copy(&mut self, hash: u64) -> Option<(S, usize)> {
        self.clock += 1;
        let clock = self.clock;
        let entry = self.map.get_mut(&hash)?;
        entry.used = clock;
        self.hits += 1;
        Some((entry.snapshot.clone(), entry.tokens))
    }

    pub fn clear(&mut self) -> usize {
        let n = self.map.len();
        self.map.clear();
        self.bytes = 0;
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mode_reads_words_and_sizes_and_a_typo_stays_on() {
        use StateCacheMode::*;
        assert_eq!(StateCacheMode::parse(""), Auto);
        assert_eq!(StateCacheMode::parse(" Auto "), Auto);
        for off in ["off", "OFF", "0", "no", "false"] {
            assert_eq!(StateCacheMode::parse(off), Off, "{off}");
        }
        assert_eq!(StateCacheMode::parse("512M"), Bytes(512 << 20));
        assert_eq!(StateCacheMode::parse("2g"), Bytes(2 << 30));
        assert_eq!(StateCacheMode::parse("64KiB"), Bytes(64 << 10));
        assert_eq!(StateCacheMode::parse("1073741824"), Bytes(1 << 30));
        assert_eq!(StateCacheMode::parse("lots"), Auto, "a typo does not switch the feature off");
        assert_eq!(StateCacheMode::parse("12x"), Auto);
        assert!(Auto.enabled() && Bytes(1).enabled() && !Off.enabled());
        assert_eq!((Auto.budget(), Bytes(7).budget()), (DEFAULT_BUDGET, 7));
    }

    #[test]
    fn the_oldest_checkpoint_goes_first_and_a_hit_makes_one_young_again() {
        let mut c = StateCheckpoints::<Vec<u8>>::new(300);
        assert!(c.insert(1, 16, 100, vec![1]).is_empty());
        assert!(c.insert(2, 16, 100, vec![2]).is_empty());
        assert!(c.insert(3, 16, 100, vec![3]).is_empty());
        assert!(c.take_copy(1).is_some(), "1 is used, so 2 is now the oldest");
        assert_eq!(c.insert(4, 16, 100, vec![4]), vec![2]);
        assert!(c.contains(1) && !c.contains(2) && c.contains(3) && c.contains(4));
        assert_eq!((c.len(), c.bytes()), (3, 300));
        assert_eq!((c.stored, c.evicted, c.hits), (4, 1, 1));
    }

    #[test]
    fn the_newest_checkpoint_stays_even_when_it_alone_is_over_the_budget() {
        let mut c = StateCheckpoints::<u8>::new(50);
        assert!(c.insert(1, 16, 400, 1).is_empty());
        assert_eq!(c.insert(2, 32, 400, 2), vec![1]);
        assert!(c.contains(2) && !c.contains(1));
        assert_eq!(c.take_copy(2), Some((2, 32)));
    }

    #[test]
    fn storing_the_same_hash_again_replaces_it_without_double_counting() {
        let mut c = StateCheckpoints::<u8>::new(1000);
        c.insert(9, 16, 100, 1);
        c.insert(9, 32, 150, 2);
        assert_eq!((c.len(), c.bytes()), (1, 150));
        assert_eq!(c.take_copy(9), Some((2, 32)));
        assert_eq!(c.clear(), 1);
        assert_eq!((c.len(), c.bytes()), (0, 0));
        assert!(c.take_copy(9).is_none());
    }
}
