//! The chats on disk: one JSON file in the app's data folder, written after every change.

use crate::assistants::Assistant;
use crate::rag::Passage;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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

/// A file whose text travels with a question as a whole (the Studio's `node.files`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MsgFile {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub role: Role,
    pub content: String,
    #[serde(default)]
    pub reasoning: String,
    #[serde(default)]
    pub stats: Option<Stats>,
    /// The passages of the documents that travelled with this question (the Studio's node.passages).
    #[serde(default)]
    pub passages: Option<Vec<Passage>>,
    /// The images of this question as data URLs (the Studio's `node.images`).
    #[serde(default)]
    pub images: Vec<String>,
    /// The files whose text went into this question whole (the Studio's `node.files`).
    #[serde(default)]
    pub files: Vec<MsgFile>,
}

/// Another way the conversation went: editing a message or asking for a new answer keeps the old version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variant {
    /// Where the version stands in the row of versions (the arrows under a message go through them in this order).
    pub slot: u32,
    pub messages: Vec<Record>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChatRecord {
    pub id: u64,
    pub title: String,
    /// The conversation as it is shown now.
    pub messages: Vec<Record>,
    /// The slot of the shown conversation.
    #[serde(default)]
    pub slot: u32,
    #[serde(default)]
    pub next_slot: u32,
    /// The other versions, whole conversations that share their beginning with the shown one.
    #[serde(default)]
    pub variants: Vec<Variant>,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub archived: bool,
    /// When the chat was last used (seconds since 1970); 0 for a chat from before the date was kept.
    #[serde(default)]
    pub updated: i64,
    /// The assistant of this chat (None: what the settings say, shown as Standard).
    #[serde(default)]
    pub assistant: Option<String>,
    /// The project this chat belongs to (None: it is a chat of its own).
    #[serde(default)]
    pub project: Option<String>,
    /// The documents of this chat itself: searched for every question of it (the Studio's `t.docs`).
    #[serde(default)]
    pub docs: Vec<ProjectDoc>,
}

/// Which version of a message is shown out of how many.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Versions {
    pub count: usize,
    /// 0-based.
    pub current: usize,
}

fn same(a: &Record, b: &Record) -> bool {
    a.role == b.role && a.content == b.content
}

impl ChatRecord {
    pub fn new(id: u64, title: &str, messages: Vec<Record>) -> ChatRecord {
        ChatRecord { id, title: title.into(), messages, ..Default::default() }
    }

    /// The versions of the message at `i`: the conversations that begin like the shown one and have a different message there.
    /// Two conversations with the same message at `i` count as one version (the newest of them is the one a switch shows).
    fn candidates_at(&self, i: usize) -> Vec<(u32, Option<usize>)> {
        let mut all: Vec<(u32, Option<usize>, &[Record])> = vec![(self.slot, None, &self.messages)];
        all.extend(self.variants.iter().enumerate().map(|(n, v)| (v.slot, Some(n), v.messages.as_slice())));
        all.retain(|(_, _, m)| m.len() > i && self.messages.len() > i && m[..i].iter().zip(&self.messages[..i]).all(|(a, b)| same(a, b)));
        all.sort_by_key(|(slot, _, _)| *slot);
        // one entry per distinct message at i; the newest conversation represents it
        let mut out: Vec<(u32, Option<usize>)> = Vec::new();
        let mut kinds: Vec<&Record> = Vec::new();
        for (slot, which, m) in &all {
            match kinds.iter().position(|k| same(k, &m[i])) {
                Some(k) => out[k] = (*slot, *which),
                None => {
                    kinds.push(&m[i]);
                    out.push((*slot, *which));
                }
            }
        }
        out
    }

    pub fn versions_at(&self, i: usize) -> Versions {
        let c = self.candidates_at(i);
        if c.is_empty() {
            return Versions { count: 1, current: 0 };
        }
        let mine = self.messages.get(i);
        let current = mine.and_then(|m| c.iter().position(|(_, which)| match which {
            None => true,
            Some(n) => self.variants[*n].messages.get(i).is_some_and(|x| same(x, m)),
        })).unwrap_or(0);
        Versions { count: c.len(), current }
    }

    /// Shows version `k` of the message at `i` (the conversation that has it replaces the shown one; the shown one is kept).
    pub fn switch_version(&mut self, i: usize, k: usize) -> bool {
        let c = self.candidates_at(i);
        let Some((_, Some(n))) = c.get(k).copied() else { return false };
        let v = &mut self.variants[n];
        std::mem::swap(&mut self.messages, &mut v.messages);
        std::mem::swap(&mut self.slot, &mut v.slot);
        true
    }

    /// The conversation goes another way from message `i` on: what was shown is kept as a version, the shown conversation is cut to
    /// the messages before `i` (the caller adds the new ones).
    pub fn branch_at(&mut self, i: usize) {
        if i >= self.messages.len() {
            return;
        }
        let old = Variant { slot: self.slot, messages: std::mem::take(&mut self.messages) };
        self.messages = old.messages[..i].to_vec();
        self.variants.push(old);
        self.slot = self.next_slot.max(self.variants.iter().map(|v| v.slot + 1).max().unwrap_or(0));
        self.next_slot = self.slot + 1;
    }

    /// After a new answer or edit exists, versions that are now identical to the shown one (same messages all the way) are dropped.
    pub fn forget_duplicates(&mut self) {
        let shown = self.messages.clone();
        self.variants.retain(|v| !(v.messages.len() == shown.len() && v.messages.iter().zip(&shown).all(|(a, b)| same(a, b))));
    }
}

/// One extracted document of a project knowledge (the Studio's project doc).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectDoc {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub text: String,
    /// How many pages the source had, when the server said so.
    #[serde(default)]
    pub pages: Option<u64>,
    /// The characters of `text` when the document was added.
    #[serde(default)]
    pub chars: usize,
    /// When it was added (seconds since 1970).
    #[serde(default)]
    pub added: i64,
}

/// A project: chats with shared instructions and a knowledge that is searched for every question.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Project {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub instructions: String,
    #[serde(default)]
    pub docs: Vec<ProjectDoc>,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub updated: i64,
}

impl Project {
    /// The characters of the knowledge (what `prjUsed` counts in the Studio).
    pub fn used(&self) -> usize {
        self.docs.iter().map(|d| d.text.chars().count()).sum()
    }

    /// The project as it comes from disk: a usable id, a name, the texts cut to the limits
    /// (the cleanup `prjLoad` runs on every load).
    pub fn cleaned(self) -> Option<Project> {
        if self.id.is_empty() {
            return None;
        }
        let mut name = if self.name.is_empty() { "Project".to_string() } else { self.name };
        name = name.chars().take(crate::rag::limits::NAME).collect();
        Some(Project {
            id: self.id,
            name,
            instructions: self.instructions.chars().take(crate::rag::limits::INSTRUCTIONS).collect(),
            docs: self
                .docs
                .into_iter()
                .filter(|d| !d.id.is_empty())
                .map(|d| ProjectDoc {
                    id: d.id,
                    name: if d.name.is_empty() { "document".to_string() } else { d.name },
                    text: d.text.chars().take(crate::rag::MAX_DOC_CHARS).collect(),
                    pages: d.pages,
                    chars: d.chars.min(crate::rag::MAX_DOC_CHARS),
                    added: d.added,
                })
                .collect(),
            created: if self.created > 0 { self.created } else { now_secs() },
            updated: if self.updated > 0 { self.updated } else { now_secs() },
        })
    }
}

/// Seconds since 1970 (the `Date.now()` of the Studio, in seconds like the chats keep it).
pub fn now_secs() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// An id for a new entry: `prj-…`, `pdoc-…`, `doc-…` (the `uid()` of the Studio).
pub fn uid(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("{prefix}-{:x}-{:x}", nanos, COUNTER.fetch_add(1, Ordering::Relaxed))
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
    /// The folders Code mode opened last, newest first.
    #[serde(default)]
    pub recent_folders: Vec<String>,
    /// With a custom system prompt: false replaces the automatic prompt, true is added after it.
    #[serde(default)]
    pub system_append: bool,
    /// How long the automatic system prompt is: always the full one, the short one, or by how fast the model reads.
    #[serde(default)]
    pub prompt_length: crate::prompts::Length,
    /// The sampling values besides temperature and top-p; sent when sampling is manual.
    #[serde(default)]
    pub sampling: Sampling,
    /// The level of the thinking chip (off, low, medium, high); a model that has no such level uses its own default.
    #[serde(default)]
    pub thinking_level: String,
    /// The assistant chosen for new chats (the Studio's current preset).
    #[serde(default)]
    pub assistant: String,
}

/// The more advanced sampling values. The neutral value of each is the default, and a neutral value is not sent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    #[serde(default)]
    pub top_k: u32,
    #[serde(default)]
    pub min_p: f32,
    #[serde(default = "one")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub dry_multiplier: f32,
    #[serde(default)]
    pub mirostat: u8,
    #[serde(default)]
    pub seed: Option<i64>,
    /// Words that end the answer, separated by commas.
    #[serde(default)]
    pub stop: String,
}

fn one() -> f32 {
    1.0
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling { top_k: 0, min_p: 0.0, repetition_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0, dry_multiplier: 0.0, mirostat: 0, seed: None, stop: String::new() }
    }
}

impl Sampling {
    /// The request fields for the values that are not neutral.
    pub fn fields(&self) -> serde_json::Map<String, serde_json::Value> {
        use serde_json::json;
        let mut m = serde_json::Map::new();
        if self.top_k > 0 {
            m.insert("top_k".into(), json!(self.top_k));
        }
        if self.min_p > 0.0 {
            m.insert("min_p".into(), json!(self.min_p));
        }
        if (self.repetition_penalty - 1.0).abs() > 1e-6 {
            m.insert("repetition_penalty".into(), json!(self.repetition_penalty));
        }
        if self.presence_penalty != 0.0 {
            m.insert("presence_penalty".into(), json!(self.presence_penalty));
        }
        if self.frequency_penalty != 0.0 {
            m.insert("frequency_penalty".into(), json!(self.frequency_penalty));
        }
        if self.dry_multiplier > 0.0 {
            m.insert("dry_multiplier".into(), json!(self.dry_multiplier));
        }
        if self.mirostat > 0 {
            m.insert("mirostat".into(), json!(self.mirostat));
        }
        if let Some(seed) = self.seed {
            m.insert("seed".into(), json!(seed));
        }
        let stop: Vec<&str> = self.stop.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        if !stop.is_empty() {
            m.insert("stop".into(), json!(stop));
        }
        m
    }
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.95
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs { manual_sampling: false, temperature: default_temperature(), top_p: default_top_p(), custom_system: false, system_prompt: String::new(), max_tokens: 0, recent_folders: Vec::new(), system_append: false, prompt_length: crate::prompts::Length::Full, sampling: Sampling::default(), thinking_level: String::new(), assistant: String::new() }
    }
}

/// How many folders Code mode remembers.
pub const RECENT_FOLDERS: usize = 6;

impl Prefs {
    /// Puts `folder` first in the list of recent folders (once), keeping the newest few.
    pub fn remember_folder(&mut self, folder: &str) {
        self.recent_folders.retain(|f| f != folder);
        self.recent_folders.insert(0, folder.to_string());
        self.recent_folders.truncate(RECENT_FOLDERS);
    }

    /// How your own prompt is applied to the automatic one.
    pub fn system_mode(&self) -> crate::prompts::Mode {
        if !self.custom_system {
            crate::prompts::Mode::Default
        } else if self.system_append {
            crate::prompts::Mode::Append
        } else {
            crate::prompts::Mode::Override
        }
    }
}

pub const BUILT_IN_SYSTEM_PROMPT: &str = "You are a helpful assistant running locally on the user's computer. Answer in the language of the user. Be concise. Use Markdown only where it helps. If you are not sure, say so. Never invent facts, quotes, links or sources.";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Store {
    #[serde(default)]
    pub prefs: Prefs,
    /// The saved assistants; the six built-ins are rebuilt on every start and never stored.
    #[serde(default)]
    pub assistants: Vec<Assistant>,
    /// The projects with their knowledge (the Studio keeps them in the browser storage).
    #[serde(default)]
    pub projects: Vec<Project>,
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
                // the same cleanup the Studio runs on every load: a broken entry is dropped, the rest is clamped
                s.assistants = s.assistants.drain(..).filter_map(Assistant::cleaned).collect();
                s.projects = s.projects.drain(..).filter_map(Project::cleaned).collect();
                // the documents of a chat: the same limits the Studio keeps (RAG_MAX_DOC_CHARS, RAG_MAX_CHAT_CHARS)
                for c in s.chats.iter_mut() {
                    let mut docs: Vec<ProjectDoc> = Vec::new();
                    let mut used: usize = c.docs.iter().map(|d| d.text.chars().count()).sum();
                    for d in c.docs.drain(..) {
                        if d.id.is_empty() || d.text.is_empty() {
                            continue;
                        }
                        let room = crate::rag::MAX_DOC_CHARS.saturating_sub(0).min(crate::rag::MAX_CHAT_CHARS.saturating_sub(used));
                        if room < 2000 {
                            break;
                        }
                        let mut d = d;
                        d.text = d.text.chars().take(room).collect();
                        d.chars = d.text.chars().count().min(crate::rag::MAX_DOC_CHARS);
                        used += d.chars;
                        docs.push(d);
                    }
                    c.docs = docs;
                }
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
        self.chats.insert(0, ChatRecord::new(id, "New chat", Vec::new()));
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
        s.chat_mut(a).unwrap().messages.push(Record { role: Role::User, content: "Hi".into(), reasoning: String::new(), stats: None, passages: None, images: Vec::new(), files: Vec::new() });
        s.save(&path).unwrap();
        let back = Store::load(&path);
        assert_eq!(back, s);
        assert_eq!(back.chats[0].id, b, "the newest chat is first");
        let mut back = back;
        assert!(back.new_chat() > b, "ids are not reused after a restart");
    }

    fn rec(role: Role, text: &str) -> Record {
        Record { role, content: text.into(), reasoning: String::new(), stats: None, passages: None, images: Vec::new(), files: Vec::new() }
    }

    fn chat(lines: &[(Role, &str)]) -> ChatRecord {
        ChatRecord::new(0, "t", lines.iter().map(|(r, t)| rec(*r, t)).collect())
    }

    fn texts(c: &ChatRecord) -> Vec<&str> {
        c.messages.iter().map(|m| m.content.as_str()).collect()
    }

    #[test]
    fn a_new_answer_keeps_the_old_one_as_a_version() {
        use Role::*;
        let mut c = chat(&[(User, "q"), (Assistant, "a1")]);
        assert_eq!(c.versions_at(1), Versions { count: 1, current: 0 });
        c.branch_at(1);
        assert_eq!(texts(&c), vec!["q"]);
        c.messages.push(rec(Assistant, "a2"));
        assert_eq!(c.versions_at(1), Versions { count: 2, current: 1 });
        assert_eq!(c.versions_at(0), Versions { count: 1, current: 0 }, "the question has one version");
        assert!(c.switch_version(1, 0));
        assert_eq!(texts(&c), vec!["q", "a1"]);
        assert_eq!(c.versions_at(1), Versions { count: 2, current: 0 });
        assert!(c.switch_version(1, 1));
        assert_eq!(texts(&c), vec!["q", "a2"], "going back and forth loses nothing");
        assert!(!c.switch_version(1, 1), "the shown version is not a switch");
        assert!(!c.switch_version(1, 7));
    }

    #[test]
    fn editing_a_question_branches_the_whole_rest_and_versions_nest() {
        use Role::*;
        let mut c = chat(&[(User, "q1"), (Assistant, "a1"), (User, "q2"), (Assistant, "a2")]);
        // edit the second question
        c.branch_at(2);
        c.messages.extend([rec(User, "q2 edited"), rec(Assistant, "a2 edited")]);
        assert_eq!(c.versions_at(2), Versions { count: 2, current: 1 });
        // a new answer to the edited question
        c.branch_at(3);
        c.messages.push(rec(Assistant, "a2 edited, again"));
        assert_eq!(c.versions_at(3), Versions { count: 2, current: 1 }, "the answers of the edited question");
        assert_eq!(c.versions_at(2), Versions { count: 2, current: 1 }, "the edited question is one version, whatever answer it has");
        // back to the first version of the question: its answer comes with it
        assert!(c.switch_version(2, 0));
        assert_eq!(texts(&c), vec!["q1", "a1", "q2", "a2"]);
        assert_eq!(c.versions_at(3), Versions { count: 1, current: 0 });
        // and forward again: the newest answer of the edited question is shown
        assert!(c.switch_version(2, 1));
        assert_eq!(texts(&c), vec!["q1", "a1", "q2 edited", "a2 edited, again"]);
        assert!(c.switch_version(3, 0));
        assert_eq!(texts(&c), vec!["q1", "a1", "q2 edited", "a2 edited"]);
        assert_eq!(c.variants.len(), 2);
    }

    #[test]
    fn versions_survive_the_file_and_old_files_still_load() {
        use Role::*;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chats.json");
        let mut s = Store::default();
        let id = s.new_chat();
        let c = s.chat_mut(id).unwrap();
        c.messages = vec![rec(User, "q"), rec(Assistant, "a1")];
        c.branch_at(1);
        c.messages.push(rec(Assistant, "a2"));
        s.save(&path).unwrap();
        let back = Store::load(&path);
        assert_eq!(back, s);
        assert_eq!(back.chat(id).unwrap().versions_at(1), Versions { count: 2, current: 1 });
        // a file from before versions existed
        std::fs::write(&path, r#"{"chats":[{"id":3,"title":"old","messages":[{"role":"user","content":"hi"}]}],"next_id":4,"theme":"light"}"#).unwrap();
        let old = Store::load(&path);
        assert_eq!(old.chat(3).unwrap().messages.len(), 1);
        assert_eq!(old.chat(3).unwrap().versions_at(0), Versions { count: 1, current: 0 });
    }

    #[test]
    fn a_version_that_ends_up_identical_is_not_kept_twice() {
        use Role::*;
        let mut c = chat(&[(User, "q"), (Assistant, "same")]);
        c.branch_at(1);
        c.messages.push(rec(Assistant, "same"));
        assert_eq!(c.variants.len(), 1);
        c.forget_duplicates();
        assert!(c.variants.is_empty());
        assert_eq!(c.versions_at(1), Versions { count: 1, current: 0 });
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
