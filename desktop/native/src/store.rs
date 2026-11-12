//! The chats on disk: one JSON file in the app's data folder, written after every change.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn wire(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// What the server told about one answer, kept under it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    pub tokens_per_second: f64,
    pub completion_tokens: u64,
    pub prompt_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub role: Role,
    pub content: String,
    #[serde(default)]
    pub reasoning: String,
    #[serde(default)]
    pub stats: Option<Stats>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRecord {
    pub id: u64,
    pub title: String,
    pub messages: Vec<Record>,
}

/// What the person chose for the answers. Everything is automatic until it is switched to manual: then the value is sent with every request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prefs {
    /// false: the server and the model decide (nothing is sent); true: `temperature` and `top_p` are sent.
    #[serde(default)]
    pub manual_sampling: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// false: the built-in system prompt; true: `system_prompt` (an empty text falls back to the built-in one).
    #[serde(default)]
    pub custom_system: bool,
    #[serde(default)]
    pub system_prompt: String,
    /// The longest answer in tokens; 0 lets the server decide.
    #[serde(default)]
    pub max_tokens: u32,
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.95
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs { manual_sampling: false, temperature: default_temperature(), top_p: default_top_p(), custom_system: false, system_prompt: String::new(), max_tokens: 0 }
    }
}

impl Prefs {
    /// The system prompt a request carries.
    pub fn system(&self) -> &str {
        if self.custom_system && !self.system_prompt.trim().is_empty() {
            self.system_prompt.trim()
        } else {
            BUILT_IN_SYSTEM_PROMPT
        }
    }
}

pub const BUILT_IN_SYSTEM_PROMPT: &str = "You are a helpful assistant running locally on the user's computer. Answer in the language of the user. Be concise. Use Markdown only where it helps. If you are not sure, say so. Never invent facts, quotes, links or sources.";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Store {
    #[serde(default)]
    pub prefs: Prefs,
    pub chats: Vec<ChatRecord>,
    #[serde(default)]
    pub next_id: u64,
    /// "light", "dark" or "" (follow the system).
    #[serde(default)]
    pub theme: String,
}

/// Where the chats live: `$FLLM_APP_DATA` (tests, portable use), else the user's data folder.
pub fn data_file() -> PathBuf {
    let dir = std::env::var_os("FLLM_APP_DATA").map(PathBuf::from).or_else(|| dirs::data_dir().map(|d| d.join("fllm-app"))).unwrap_or_else(|| PathBuf::from("."));
    dir.join("chats.json")
}

impl Store {
    /// A missing or unreadable file is an empty store. A damaged file is moved aside, so that nothing is overwritten that a person could still repair.
    pub fn load(path: &Path) -> Store {
        let Ok(text) = std::fs::read_to_string(path) else { return Store::default() };
        match serde_json::from_str::<Store>(&text) {
            Ok(mut s) => {
                s.next_id = s.next_id.max(s.chats.iter().map(|c| c.id + 1).max().unwrap_or(0));
                s
            }
            Err(_) => {
                let _ = std::fs::rename(path, path.with_extension("damaged.json"));
                Store::default()
            }
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        // write beside the file and rename: a crash in the middle leaves the old file whole
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?)?;
        std::fs::rename(tmp, path)
    }

    pub fn new_chat(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.chats.insert(0, ChatRecord { id, title: "New chat".into(), messages: Vec::new() });
        id
    }

    pub fn chat(&self, id: u64) -> Option<&ChatRecord> {
        self.chats.iter().find(|c| c.id == id)
    }

    pub fn chat_mut(&mut self, id: u64) -> Option<&mut ChatRecord> {
        self.chats.iter_mut().find(|c| c.id == id)
    }
}

/// The title of a chat from its first message: one line, cut at a word.
pub fn title_from(text: &str) -> String {
    let line = text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("New chat");
    if line.chars().count() <= 40 {
        return line.to_string();
    }
    let cut: String = line.chars().take(40).collect();
    let cut = cut.rsplit_once(' ').map(|(a, _)| a.to_string()).unwrap_or(cut);
    format!("{}…", cut.trim_end_matches([',', '.', ';', ':']))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_store_survives_a_round_trip_and_keeps_ids_unique() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a/b/chats.json");
        let mut s = Store::default();
        let a = s.new_chat();
        let b = s.new_chat();
        assert_ne!(a, b);
        s.chat_mut(a).unwrap().messages.push(Record { role: Role::User, content: "Hi".into(), reasoning: String::new(), stats: None });
        s.save(&path).unwrap();
        let back = Store::load(&path);
        assert_eq!(back, s);
        assert_eq!(back.chats[0].id, b, "the newest chat is first");
        let mut back = back;
        assert!(back.new_chat() > b, "ids are not reused after a restart");
    }

    #[test]
    fn a_damaged_file_is_moved_aside_not_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chats.json");
        std::fs::write(&path, "{ this is not json").unwrap();
        assert_eq!(Store::load(&path), Store::default());
        assert!(dir.path().join("chats.damaged.json").exists());
        assert!(!path.exists());
    }

    #[test]
    fn titles_are_one_line_and_cut_at_a_word() {
        assert_eq!(title_from("Hi was geht?"), "Hi was geht?");
        assert_eq!(title_from("\n\n  second line wins  \nthird"), "second line wins");
        assert_eq!(title_from(""), "New chat");
        let t = title_from("Explain in simple words how a language model writes a text word by word");
        assert!(t.ends_with('…') && t.chars().count() <= 41, "{t}");
        assert!(!t.contains("wor…"), "no word is cut in half: {t}");
    }
}
