//! Who may talk to the agent inside an allowed chat: a pairing store. A chat on the `--chat` list can still hold people the
//! owner does not know (a Discord channel, a Slack channel, a Matrix room). With `--pair`, a sender who is neither on the
//! `--user` list nor approved yet gets a one-time code instead of an answer; the owner approves it on the machine with
//! `fllm-rs gateway pair approve PLATFORM CODE`. Codes are 8 characters from a list without look-alikes, come from the
//! operating system's random source, expire after an hour, and at most three can wait per platform.
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const EXPIRES_SECONDS: i64 = 3600;
pub const MAX_PENDING: usize = 3;
const ALPHABET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

#[derive(Debug, Clone, PartialEq)]
pub struct Pending {
    pub code: String,
    pub platform: String,
    pub sender: String,
    pub chat: String,
    pub time: i64,
}

#[derive(Debug, Default, PartialEq)]
pub struct Store {
    pub approved: BTreeMap<String, BTreeSet<String>>,
    pub pending: Vec<Pending>,
}

pub fn store_file(base: &Path) -> PathBuf {
    base.join("gateway-pairing.json")
}

pub fn new_code() -> String {
    let mut bytes = [0u8; 8];
    // an unavailable random source must not produce a guessable code
    getrandom::fill(&mut bytes).expect("the operating system has no random source");
    bytes.iter().map(|b| ALPHABET[usize::from(*b) % ALPHABET.len()] as char).collect()
}

impl Store {
    /// A missing file is an empty store; a damaged one is an error (it must not silently drop approvals or let everybody in).
    pub fn load(file: &Path) -> Result<Store, String> {
        let text = match std::fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Store::default()),
            Err(e) => return Err(format!("cannot read {}: {e}", file.display())),
        };
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{} is damaged: {e}", file.display()))?;
        let mut store = Store::default();
        for (platform, list) in v["approved"].as_object().into_iter().flatten() {
            store.approved.insert(platform.clone(), list.as_array().into_iter().flatten().filter_map(|s| s.as_str().map(str::to_owned)).collect());
        }
        for p in v["pending"].as_array().into_iter().flatten() {
            if let (Some(code), Some(platform), Some(sender)) = (p["code"].as_str(), p["platform"].as_str(), p["sender"].as_str()) {
                store.pending.push(Pending { code: code.to_owned(), platform: platform.to_owned(), sender: sender.to_owned(), chat: p["chat"].as_str().unwrap_or_default().to_owned(), time: p["time"].as_i64().unwrap_or(0) });
            }
        }
        Ok(store)
    }

    pub fn save(&self, file: &Path) -> Result<(), String> {
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let approved: BTreeMap<&String, Vec<&String>> = self.approved.iter().map(|(k, v)| (k, v.iter().collect())).collect();
        let pending: Vec<Value> = self.pending.iter().map(|p| json!({"code": p.code, "platform": p.platform, "sender": p.sender, "chat": p.chat, "time": p.time})).collect();
        let tmp = file.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&json!({"approved": approved, "pending": pending})).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, file).map_err(|e| e.to_string())
    }

    pub fn is_approved(&self, platform: &str, sender: &str) -> bool {
        self.approved.get(platform).is_some_and(|s| s.contains(sender))
    }

    fn prune(&mut self, now: i64) {
        self.pending.retain(|p| now - p.time < EXPIRES_SECONDS);
    }

    /// The code for a sender who is not approved: the same code again while it is valid, a new one otherwise.
    pub fn request(&mut self, platform: &str, sender: &str, chat: &str, now: i64) -> Result<String, String> {
        self.prune(now);
        if let Some(p) = self.pending.iter().find(|p| p.platform == platform && p.sender == sender) {
            return Ok(p.code.clone());
        }
        if self.pending.iter().filter(|p| p.platform == platform).count() >= MAX_PENDING {
            return Err("too many pairing requests are waiting; ask the owner to approve or wait an hour".into());
        }
        let code = new_code();
        self.pending.push(Pending { code: code.clone(), platform: platform.to_owned(), sender: sender.to_owned(), chat: chat.to_owned(), time: now });
        Ok(code)
    }

    /// Approves the sender behind a code; returns the sender id.
    pub fn approve(&mut self, platform: &str, code: &str, now: i64) -> Result<String, String> {
        self.prune(now);
        let code = code.trim().to_ascii_uppercase();
        let at = self.pending.iter().position(|p| p.platform == platform && p.code == code).ok_or("no such pairing code (it may have expired)")?;
        let p = self.pending.remove(at);
        self.approved.entry(platform.to_owned()).or_default().insert(p.sender.clone());
        Ok(p.sender)
    }

    pub fn revoke(&mut self, platform: &str, sender: &str) -> bool {
        self.approved.get_mut(platform).is_some_and(|s| s.remove(sender))
    }
}

/// `fllm-rs gateway pair list|approve|revoke`.
pub fn run(args: &[String], base: &Path) -> Result<i32, String> {
    let file = store_file(base);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    let mut store = Store::load(&file)?;
    match args.first().map(String::as_str) {
        Some("list") | None => {
            store.prune(now);
            for (platform, senders) in &store.approved {
                for s in senders {
                    println!("approved  {platform}  {s}");
                }
            }
            for p in &store.pending {
                println!("waiting   {}  {}  code {}  (expires in {} min)", p.platform, p.sender, p.code, (EXPIRES_SECONDS - (now - p.time)) / 60);
            }
            if store.approved.is_empty() && store.pending.is_empty() {
                println!("nobody is approved or waiting");
            }
            Ok(0)
        }
        Some("approve") => {
            let (platform, code) = (args.get(1).ok_or("usage: fllm-rs gateway pair approve PLATFORM CODE")?, args.get(2).ok_or("usage: fllm-rs gateway pair approve PLATFORM CODE")?);
            let sender = store.approve(platform, code, now)?;
            store.save(&file)?;
            println!("approved {sender} on {platform}");
            Ok(0)
        }
        Some("revoke") => {
            let (platform, sender) = (args.get(1).ok_or("usage: fllm-rs gateway pair revoke PLATFORM SENDER")?, args.get(2).ok_or("usage: fllm-rs gateway pair revoke PLATFORM SENDER")?);
            if !store.revoke(platform, sender) {
                return Err(format!("{sender} is not approved on {platform}"));
            }
            store.save(&file)?;
            println!("revoked {sender} on {platform}");
            Ok(0)
        }
        Some(other) => Err(format!("unknown pair command '{other}' (list, approve, revoke)")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stranger_gets_one_code_that_the_owner_approves() {
        let mut s = Store::default();
        let code = s.request("discord", "u1", "c1", 1000).unwrap();
        assert_eq!(code.len(), 8);
        assert!(code.bytes().all(|b| ALPHABET.contains(&b)));
        assert_eq!(s.request("discord", "u1", "c1", 1100).unwrap(), code, "asking again shows the same code");
        assert!(!s.is_approved("discord", "u1"));
        assert!(s.approve("discord", "wrong123", 1200).is_err());
        assert!(s.approve("slack", &code, 1200).is_err(), "a code is bound to its platform");
        assert_eq!(s.approve("discord", &code.to_lowercase(), 1200).unwrap(), "u1");
        assert!(s.is_approved("discord", "u1") && !s.is_approved("slack", "u1"));
        assert!(s.approve("discord", &code, 1300).is_err(), "a code works once");
        assert!(s.revoke("discord", "u1") && !s.revoke("discord", "u1"));
    }

    #[test]
    fn codes_expire_and_waiting_requests_are_capped() {
        let mut s = Store::default();
        let first = s.request("slack", "a", "c", 0).unwrap();
        s.request("slack", "b", "c", 1).unwrap();
        s.request("slack", "c", "c", 2).unwrap();
        assert!(s.request("slack", "d", "c", 3).unwrap_err().contains("too many"));
        assert!(s.request("discord", "d", "c", 3).is_ok(), "the cap is per platform");
        assert!(s.approve("slack", &first, EXPIRES_SECONDS + 1).is_err(), "an old code is dead");
        assert!(s.request("slack", "d", "c", EXPIRES_SECONDS + 5).is_ok(), "room again after expiry");
    }

    #[test]
    fn codes_are_not_repeated() {
        let codes: BTreeSet<String> = (0..200).map(|_| new_code()).collect();
        assert_eq!(codes.len(), 200);
    }

    #[test]
    fn the_store_survives_a_restart_and_a_damaged_file_is_an_error() {
        let dir = std::env::temp_dir().join(format!("fllm-pair-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let file = store_file(&dir);
        assert_eq!(Store::load(&file).unwrap(), Store::default());
        let mut s = Store::default();
        s.request("discord", "u1", "c1", 5).unwrap();
        let code = s.pending[0].code.clone();
        s.approve("discord", &code, 6).unwrap();
        s.request("slack", "u2", "c2", 7).unwrap();
        s.save(&file).unwrap();
        assert_eq!(Store::load(&file).unwrap(), s);
        std::fs::write(&file, "{oops").unwrap();
        assert!(Store::load(&file).unwrap_err().contains("damaged"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
