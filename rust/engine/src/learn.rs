//! The store of the self-improvement loop of the coding agent: what was learned and what waits for approval.
//!
//! Nothing here talks to a model (see `learn_flow` for that) and nothing is hidden: every file lives in the Studio folder, is plain
//! text or JSON, and can be read, edited or deleted by hand.
//!
//! * `learn/skills/<key>/usage.jsonl`: one line per use of a skill (outcome, verify result, retries, thumbs) and per feedback.
//! * `learn/skills/<key>/revision.json`: the revised `SKILL.md` the agent drafted, with the reason. Never applied without approval.
//! * `learn/skills/<key>/history.jsonl` and `versions/vNNNN.md`: every text a skill had before an approved revision or a rollback.
//! * `learn/memory-proposed.json`, `learn/memory-sources.json`: memory entries waiting for approval, and where learned entries came from.
//! * `learn/profile-proposed.json`, `user-profile.md`: the transparent profile of the user and the lines waiting to join it.
//! * `learn/settings.json`: the few switches (is the profile put into the prompt).
//!
//! Memory entries live where the `memory` tool keeps them (`agent-memory.md` in the Studio folder, `.fllm/memory.md` in a project).
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// One memory entry is at most this long; the same limit as the `memory` tool.
pub const MEMORY_ENTRY_CHARS: usize = 400;
pub const MEMORY_MAX_ENTRIES: usize = 100;
pub const MEMORY_FILE_BYTES: usize = 24_000;
/// Proposals waiting for approval; further ones are dropped until some are decided.
pub const MAX_PENDING_MEMORY: usize = 40;
pub const MAX_PENDING_PROFILE: usize = 30;
/// A single reflection proposes at most this many memory entries.
pub const MAX_NEW_PER_RUN: usize = 5;
pub const PROFILE_SECTIONS: [(&str, &str); 5] = [
    ("preferences", "Preferences"),
    ("skill_level", "Skill level"),
    ("languages", "Languages"),
    ("conventions", "Conventions"),
    ("tool_habits", "Tool habits"),
];
pub const PROFILE_ITEMS_PER_SECTION: usize = 8;
pub const PROFILE_ITEM_CHARS: usize = 160;
/// The profile file may be at most this large (a hand edit included).
pub const PROFILE_FILE_CHARS: usize = 3_000;
/// At most this much of the profile goes into a system prompt.
pub const PROFILE_PROMPT_CHARS: usize = 1_200;
/// After this many uses of a skill since its last revision the agent drafts a revision.
pub const IMPROVE_AFTER_USES: usize = 5;
pub const USAGE_KEEP: usize = 200;
pub const VERSIONS_KEEP: usize = 20;
const MAX_SKILL_BYTES: usize = 64 * 1024;
/// Two memory entries whose BM25 score ratio reaches this are one and the same.
pub const SIMILAR_RATIO: f64 = 0.75;

static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

pub fn learn_dir(studio: &Path) -> PathBuf {
    studio.join("learn")
}

// ---------------------------------------------------------------------------------------------------------------
// small file helpers
// ---------------------------------------------------------------------------------------------------------------

fn read_json(path: &Path) -> Value {
    std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null)
}

fn write_text(path: &Path, text: &str) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    write_text(path, &serde_json::to_string_pretty(value).map_err(|e| e.to_string())?)
}

fn read_lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path).unwrap_or_default().lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
}

fn write_lines(path: &Path, items: &[Value]) -> Result<(), String> {
    let mut text = String::new();
    for i in items {
        text.push_str(&i.to_string());
        text.push('\n');
    }
    write_text(path, &text)
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn clip_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let cut: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", cut.trim_end())
}

// ---------------------------------------------------------------------------------------------------------------
// redaction: no secret is ever shown to the reflecting model or stored as a memory
// ---------------------------------------------------------------------------------------------------------------

fn patterns() -> &'static Vec<(regex::Regex, &'static str)> {
    static P: OnceLock<Vec<(regex::Regex, &'static str)>> = OnceLock::new();
    P.get_or_init(|| {
        let list: [(&str, &str); 11] = [
            (r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", "[redacted private key]"),
            (r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}", "[redacted key]"),
            (r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}", "[redacted token]"),
            (r"\bgithub_pat_[A-Za-z0-9_]{20,}", "[redacted token]"),
            (r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", "[redacted key]"),
            (r"\bAIza[0-9A-Za-z_\-]{30,}", "[redacted key]"),
            (r"\bxox[abprs]-[A-Za-z0-9\-]{10,}", "[redacted token]"),
            (r"\bhf_[A-Za-z0-9]{30,}", "[redacted token]"),
            (r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}", "[redacted token]"),
            (r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]{16,}", "Bearer [redacted]"),
            (r"://[^/\s:@]+:[^/\s@]+@", "://[redacted]@"),
        ];
        let mut out: Vec<(regex::Regex, &'static str)> = list.iter().filter_map(|(p, r)| regex::Regex::new(p).ok().map(|re| (re, *r))).collect();
        // `password = hunter22`, `API_KEY: "abc123"`, `--token abcdef`
        if let Ok(re) = regex::Regex::new(r#"(?i)\b([A-Za-z0-9_\-]*(?:api[_-]?key|secret|token|passwd|password|pwd|credential)[A-Za-z0-9_\-]*)\s*[:=]\s*("[^"\n]{4,}"|'[^'\n]{4,}'|[^\s"',;]{6,})"#) {
            out.push((re, "$1=[redacted]"));
        }
        if let Ok(re) = regex::Regex::new(r"(?i)(--?(?:api-?key|token|password|passwd|secret)[= ]+)[^\s]+") {
            out.push((re, "$1[redacted]"));
        }
        out
    })
}

/// `text` with keys, tokens, passwords, private keys and credentials in URLs replaced by `[redacted ...]`.
pub fn redact(text: &str) -> String {
    redact_with(text, &[])
}

/// Like [`redact`], and every literal in `secrets` (an API key given on the command line) is replaced as well.
pub fn redact_with(text: &str, secrets: &[String]) -> String {
    let mut out = text.to_owned();
    for s in secrets.iter().filter(|s| s.len() >= 6) {
        out = out.replace(s.as_str(), "[redacted]");
    }
    for (re, to) in patterns() {
        if re.is_match(&out) {
            out = re.replace_all(&out, *to).into_owned();
        }
    }
    out
}

pub fn has_secret(text: &str) -> bool {
    redact(text) != text
}

// ---------------------------------------------------------------------------------------------------------------
// words and near duplicates: normalised text plus BM25, no embeddings
// ---------------------------------------------------------------------------------------------------------------

const STOP: [&str; 34] = [
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "be", "it", "this", "that", "with", "as", "at", "by", "use", "uses", "using",
    "when", "you", "your", "we", "do", "does", "not", "from", "run", "runs",
];

/// Lower case words without short stop words; a trailing `s` is dropped so `tests` and `test` are one word.
pub fn tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && (w.chars().count() > 1 || w.chars().all(|c| c.is_ascii_digit())) && !STOP.contains(w))
        .map(|w| if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") { w[..w.len() - 1].to_owned() } else { w.to_owned() })
        .collect()
}

pub fn normalize(text: &str) -> String {
    tokens(text).join(" ")
}

fn bm25(query: &[String], doc: &[String], corpus: &[Vec<String>]) -> f64 {
    const K1: f64 = 1.2;
    const B: f64 = 0.75;
    if corpus.is_empty() || doc.is_empty() {
        return 0.0;
    }
    let n = corpus.len() as f64;
    let avg = (corpus.iter().map(Vec::len).sum::<usize>() as f64 / n).max(1.0);
    let unique: BTreeSet<&String> = query.iter().collect();
    let mut score = 0.0;
    for term in unique {
        let tf = doc.iter().filter(|w| *w == term).count() as f64;
        if tf == 0.0 {
            continue;
        }
        let df = corpus.iter().filter(|d| d.contains(term)).count() as f64;
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
        score += idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * doc.len() as f64 / avg));
    }
    score
}

/// How alike `candidate` and `existing` are, 0 to 1: the BM25 score of the candidate's words in the existing text, divided by the
/// score the candidate would get against itself. 1 for the same words.
pub fn similarity(candidate: &str, existing: &str, corpus: &[String]) -> f64 {
    let q = tokens(candidate);
    let d = tokens(existing);
    if q.is_empty() || d.is_empty() {
        return 0.0;
    }
    if q == d {
        return 1.0;
    }
    let mut docs: Vec<Vec<String>> = corpus.iter().map(|c| tokens(c)).collect();
    docs.push(q.clone());
    docs.push(d.clone());
    let own = bm25(&q, &q, &docs);
    if own <= 0.0 {
        return 0.0;
    }
    (bm25(&q, &d, &docs) / own).min(1.0)
}

/// The index and score of the entry of `existing` that `candidate` repeats, if any.
pub fn find_similar(candidate: &str, existing: &[String]) -> Option<(usize, f64)> {
    let norm = normalize(candidate);
    if norm.is_empty() {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for (i, e) in existing.iter().enumerate() {
        let s = if normalize(e) == norm { 1.0 } else { similarity(candidate, e, existing) };
        if s >= SIMILAR_RATIO && best.is_none_or(|(_, b)| s > b) {
            best = Some((i, s));
        }
    }
    best
}

// ---------------------------------------------------------------------------------------------------------------
// skills: where they are, how they were used
// ---------------------------------------------------------------------------------------------------------------

/// A skill of the user (`scope` "user") or of a project folder (`scope` "workspace", `workspace` the folder).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillRef {
    pub name: String,
    pub scope: String,
    pub workspace: String,
}

fn fnv(text: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in text.replace('\\', "/").trim_end_matches('/').bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

impl SkillRef {
    pub fn user(name: &str) -> SkillRef {
        SkillRef { name: name.to_owned(), scope: "user".into(), workspace: String::new() }
    }
    pub fn workspace(name: &str, folder: &str) -> SkillRef {
        SkillRef { name: name.to_owned(), scope: "workspace".into(), workspace: folder.to_owned() }
    }
    pub fn from_parts(name: &str, scope: &str, workspace: &str) -> Result<SkillRef, String> {
        if !crate::agent_cli::valid_session_name(name) {
            return Err(format!("'{name}' is not a skill name"));
        }
        match scope {
            "" | "user" => Ok(SkillRef::user(name)),
            "workspace" if Path::new(workspace).is_absolute() => Ok(SkillRef::workspace(name, workspace)),
            "workspace" => Err("a workspace skill needs the absolute folder of the project".into()),
            other => Err(format!("unknown scope '{other}'")),
        }
    }
    /// The name of the folder that holds the usage log and the history of this skill.
    pub fn key(&self) -> String {
        if self.scope == "workspace" { format!("ws-{:08x}-{}", fnv(&self.workspace), self.name) } else { self.name.clone() }
    }
    pub fn meta_dir(&self, studio: &Path) -> PathBuf {
        learn_dir(studio).join("skills").join(self.key())
    }
    pub fn dir(&self, studio: &Path) -> PathBuf {
        if self.scope == "workspace" { Path::new(&self.workspace).join(".fllm").join("skills").join(&self.name) } else { studio.join("skills").join(&self.name) }
    }
    pub fn file(&self, studio: &Path) -> PathBuf {
        self.dir(studio).join("SKILL.md")
    }
    pub fn read(&self, studio: &Path) -> Result<String, String> {
        std::fs::read_to_string(self.file(studio)).map_err(|_| format!("no {} skill named '{}'", self.scope, self.name))
    }
}

/// One installed skill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillInfo {
    pub skill: SkillRef,
    pub description: String,
    pub when: String,
}

/// The skills of the user and of `workspace`; a skill of the project hides a skill of the same name of the user.
pub fn list_skills(studio: &Path, workspace: Option<&Path>) -> Vec<SkillInfo> {
    let mut out: Vec<SkillInfo> = Vec::new();
    let mut add = |scope: &str, root: PathBuf| {
        for dir in crate::skills_cli::find_skills(&root) {
            let Some(name) = dir.file_name().and_then(|n| n.to_str()).map(str::to_owned) else { continue };
            if !crate::agent_cli::valid_session_name(&name) || out.iter().any(|s| s.skill.name == name) {
                continue;
            }
            let (_, description, when) = crate::skills_cli::read_meta(&std::fs::read_to_string(dir.join("SKILL.md")).unwrap_or_default());
            let skill = match (scope, workspace) {
                ("workspace", Some(w)) => SkillRef::workspace(&name, &w.display().to_string()),
                _ => SkillRef::user(&name),
            };
            out.push(SkillInfo { skill, description, when });
        }
    };
    if let Some(w) = workspace {
        add("workspace", w.join(".fllm").join("skills"));
    }
    add("user", studio.join("skills"));
    out.sort_by(|a, b| a.skill.name.cmp(&b.skill.name));
    out
}

/// The skills (by name) that the model read with the `skill` tool in `messages`.
pub fn skills_read_in(messages: &[Value]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for m in messages {
        for call in m["tool_calls"].as_array().into_iter().flatten() {
            if call["function"]["name"].as_str() != Some("skill") {
                continue;
            }
            let args = match &call["function"]["arguments"] {
                Value::String(s) => serde_json::from_str::<Value>(s).unwrap_or(Value::Null),
                other => other.clone(),
            };
            if args["action"].as_str() == Some("read") {
                if let Some(n) = args["name"].as_str().filter(|n| crate::agent_cli::valid_session_name(n)) {
                    if !out.iter().any(|o| o == n) {
                        out.push(n.to_owned());
                    }
                }
            }
        }
    }
    out
}

/// Skills whose name, description and `when` line share words with the task: `(name, score)`, best first, at most `limit`.
pub fn match_skills(task: &str, skills: &[SkillInfo], limit: usize) -> Vec<(String, f64)> {
    let q = tokens(task);
    if q.is_empty() {
        return Vec::new();
    }
    let docs: Vec<Vec<String>> = skills.iter().map(|s| tokens(&format!("{} {} {} {}", s.skill.name.replace(['-', '_'], " "), s.skill.name.replace(['-', '_'], " "), s.description, s.when))).collect();
    let mut scored: Vec<(String, f64)> = Vec::new();
    for (s, d) in skills.iter().zip(&docs) {
        let score = bm25(&q, d, &docs);
        // at least two shared words
        let shared = q.iter().collect::<BTreeSet<_>>().into_iter().filter(|w| d.contains(w)).count();
        if shared >= 2 && score > 0.0 {
            scored.push((s.skill.name.clone(), score));
        }
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    scored.truncate(limit);
    scored
}

#[derive(Clone, Debug, PartialEq)]
pub struct Usage {
    pub n: u64,
    pub t: i64,
    pub session: String,
    /// `done`, `failed` or `feedback`.
    pub outcome: String,
    pub verify: Option<bool>,
    pub retries: u32,
    /// `up` or `down` (outcome `feedback`, or a verdict given with the run).
    pub feedback: Option<String>,
    pub note: String,
    /// The task text matched the skill before the model asked for it.
    pub matched: bool,
}

impl Usage {
    pub fn new(session: &str, outcome: &str) -> Usage {
        Usage { n: 0, t: now(), session: session.to_owned(), outcome: outcome.to_owned(), verify: None, retries: 0, feedback: None, note: String::new(), matched: false }
    }
    fn to_json(&self) -> Value {
        json!({"n": self.n, "t": self.t, "session": self.session, "outcome": self.outcome, "verify": self.verify, "retries": self.retries,
               "feedback": self.feedback, "note": self.note, "matched": self.matched})
    }
    fn from_json(v: &Value) -> Option<Usage> {
        Some(Usage {
            n: v["n"].as_u64()?,
            t: v["t"].as_i64().unwrap_or(0),
            session: v["session"].as_str().unwrap_or_default().to_owned(),
            outcome: v["outcome"].as_str()?.to_owned(),
            verify: v["verify"].as_bool(),
            retries: v["retries"].as_u64().unwrap_or(0) as u32,
            feedback: v["feedback"].as_str().map(str::to_owned),
            note: v["note"].as_str().unwrap_or_default().to_owned(),
            matched: v["matched"].as_bool().unwrap_or(false),
        })
    }
}

fn usage_file(studio: &Path, s: &SkillRef) -> PathBuf {
    s.meta_dir(studio).join("usage.jsonl")
}

pub fn read_usage(studio: &Path, s: &SkillRef) -> Vec<Usage> {
    read_lines(&usage_file(studio, s)).iter().filter_map(Usage::from_json).collect()
}

/// Appends a record (`u.n` is assigned); the file keeps the newest [`USAGE_KEEP`] records. The note is redacted and short.
pub fn record_usage(studio: &Path, s: &SkillRef, mut u: Usage) -> Result<Usage, String> {
    let _g = lock();
    let mut all = read_usage(studio, s);
    u.n = all.last().map_or(1, |l| l.n + 1);
    u.note = clip_chars(&one_line(&redact(&u.note)), 300);
    all.push(u.clone());
    if all.len() > USAGE_KEEP {
        all.drain(..all.len() - USAGE_KEEP);
    }
    let lines: Vec<Value> = all.iter().map(Usage::to_json).collect();
    write_lines(&usage_file(studio, s), &lines)?;
    Ok(u)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub uses: usize,
    pub ok: usize,
    pub failed: usize,
    pub up: usize,
    pub down: usize,
    pub retries: u32,
    /// Uses, failures and thumbs down since the last revision, rollback or rejected revision.
    pub window_uses: usize,
    pub window_bad: usize,
    pub last_t: i64,
}

pub fn compute_stats(usage: &[Usage], after_n: u64) -> Stats {
    let mut s = Stats::default();
    for u in usage {
        let counts = u.outcome == "done" || u.outcome == "failed";
        let bad = u.outcome == "failed" || u.feedback.as_deref() == Some("down") || u.verify == Some(false);
        if counts {
            s.uses += 1;
            if u.outcome == "done" && u.verify != Some(false) {
                s.ok += 1;
            } else {
                s.failed += 1;
            }
            s.retries += u.retries;
        }
        match u.feedback.as_deref() {
            Some("up") => s.up += 1,
            Some("down") => s.down += 1,
            _ => {}
        }
        s.last_t = s.last_t.max(u.t);
        if u.n > after_n {
            if counts {
                s.window_uses += 1;
            }
            if bad {
                s.window_bad += 1;
            }
        }
    }
    s
}

// ---------------------------------------------------------------------------------------------------------------
// revisions of a skill: proposal, approval, history, rollback
// ---------------------------------------------------------------------------------------------------------------

fn history_file(studio: &Path, s: &SkillRef) -> PathBuf {
    s.meta_dir(studio).join("history.jsonl")
}

fn versions_dir(studio: &Path, s: &SkillRef) -> PathBuf {
    s.meta_dir(studio).join("versions")
}

fn revision_file(studio: &Path, s: &SkillRef) -> PathBuf {
    s.meta_dir(studio).join("revision.json")
}

pub fn read_history(studio: &Path, s: &SkillRef) -> Vec<Value> {
    read_lines(&history_file(studio, s))
}

pub fn stats_of(studio: &Path, s: &SkillRef) -> Stats {
    let after = read_history(studio, s).iter().filter_map(|h| h["after_n"].as_u64()).max().unwrap_or(0);
    compute_stats(&read_usage(studio, s), after)
}

/// Why a revision should be drafted now (`None`: not yet, or one is waiting already).
pub fn improve_due(studio: &Path, s: &SkillRef) -> Option<String> {
    if revision_file(studio, s).is_file() || !s.file(studio).is_file() {
        return None;
    }
    let st = stats_of(studio, s);
    if st.window_bad > 0 {
        return Some(format!("{} bad outcome(s) since the last revision", st.window_bad));
    }
    (st.window_uses >= IMPROVE_AFTER_USES).then(|| format!("{} uses since the last revision", st.window_uses))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Revision {
    pub skill: SkillRef,
    pub reason: String,
    pub trigger: String,
    pub created: i64,
    pub old: String,
    pub new: String,
}

impl Revision {
    pub fn id(&self) -> String {
        format!("skill:{}", self.skill.key())
    }
}

pub fn pending_revision(studio: &Path, s: &SkillRef) -> Option<Revision> {
    let v = read_json(&revision_file(studio, s));
    Some(Revision {
        skill: s.clone(),
        reason: v["reason"].as_str()?.to_owned(),
        trigger: v["trigger"].as_str().unwrap_or_default().to_owned(),
        created: v["created"].as_i64().unwrap_or(0),
        old: v["old"].as_str()?.to_owned(),
        new: v["new"].as_str()?.to_owned(),
    })
}

/// Every revision that waits for a decision, over all skills.
pub fn pending_revisions(studio: &Path) -> Vec<Revision> {
    let mut out = Vec::new();
    let Ok(read) = std::fs::read_dir(learn_dir(studio).join("skills")) else { return out };
    let mut dirs: Vec<_> = read.flatten().filter(|e| e.path().join("revision.json").is_file()).collect();
    dirs.sort_by_key(|e| e.file_name());
    for e in dirs {
        let v = read_json(&e.path().join("revision.json"));
        let name = v["name"].as_str().unwrap_or_default();
        let Ok(s) = SkillRef::from_parts(name, v["scope"].as_str().unwrap_or("user"), v["workspace"].as_str().unwrap_or_default()) else { continue };
        if let Some(r) = pending_revision(studio, &s) {
            out.push(r);
        }
    }
    out
}

/// Stores a revised `SKILL.md` as a proposal. The skill is not touched.
pub fn propose_revision(studio: &Path, s: &SkillRef, new_text: &str, reason: &str, trigger: &str) -> Result<(), String> {
    let _g = lock();
    let old = s.read(studio)?;
    if new_text.trim().is_empty() || new_text.len() > MAX_SKILL_BYTES {
        return Err("the revised skill is empty or larger than 64 KB".into());
    }
    if has_secret(new_text) {
        return Err("the revised skill contains something that looks like a secret; it was dropped".into());
    }
    if new_text.trim() == old.trim() {
        return Err("the revision does not change the skill".into());
    }
    let reason = clip_chars(&one_line(&redact(reason)), 400);
    write_json(
        &revision_file(studio, s),
        &json!({"name": s.name, "scope": s.scope, "workspace": s.workspace, "reason": reason, "trigger": trigger, "created": now(), "old": old, "new": new_text}),
    )
}

fn next_version(studio: &Path, s: &SkillRef) -> u32 {
    let mut max = 0;
    if let Ok(read) = std::fs::read_dir(versions_dir(studio, s)) {
        for e in read.flatten() {
            if let Some(n) = e.file_name().to_str().and_then(|f| f.strip_prefix('v')).and_then(|f| f.strip_suffix(".md")).and_then(|f| f.parse::<u32>().ok()) {
                max = max.max(n);
            }
        }
    }
    max + 1
}

fn prune_versions(studio: &Path, s: &SkillRef) {
    let Ok(read) = std::fs::read_dir(versions_dir(studio, s)) else { return };
    let mut files: Vec<PathBuf> = read.flatten().map(|e| e.path()).collect();
    files.sort();
    if files.len() > VERSIONS_KEEP {
        for old in &files[..files.len() - VERSIONS_KEEP] {
            let _ = std::fs::remove_file(old);
        }
    }
}

/// Keeps `text` as the next version and logs why.
fn snapshot(studio: &Path, s: &SkillRef, text: &str, action: &str, reason: &str) -> Result<u32, String> {
    let version = next_version(studio, s);
    write_text(&versions_dir(studio, s).join(format!("v{version:04}.md")), text)?;
    let after = read_usage(studio, s).last().map_or(0, |u| u.n);
    let mut h = read_history(studio, s);
    h.push(json!({"t": now(), "version": version, "action": action, "reason": clip_chars(&one_line(reason), 400), "after_n": after}));
    write_lines(&history_file(studio, s), &h)?;
    prune_versions(studio, s);
    Ok(version)
}

/// Applies the waiting revision. The previous text is kept as a version first. Refuses when the file was edited after the proposal
/// (unless `force`), so a hand edit is never overwritten silently. Returns the version number of the kept previous text.
pub fn approve_revision(studio: &Path, s: &SkillRef, force: bool) -> Result<u32, String> {
    let _g = lock();
    let rev = pending_revision(studio, s).ok_or_else(|| format!("no revision is waiting for '{}'", s.name))?;
    let current = s.read(studio)?;
    if current != rev.old && !force {
        return Err(format!("'{}' was edited after the revision was drafted; reject it and improve again (or use --force)", s.name));
    }
    let v = snapshot(studio, s, &current, "approve", &rev.reason)?;
    write_text(&s.file(studio), &rev.new)?;
    let _ = std::fs::remove_file(revision_file(studio, s));
    Ok(v)
}

pub fn reject_revision(studio: &Path, s: &SkillRef) -> Result<(), String> {
    let _g = lock();
    let rev = pending_revision(studio, s).ok_or_else(|| format!("no revision is waiting for '{}'", s.name))?;
    // the window of "uses since the last revision" restarts, so a rejected draft is not drafted again at once
    let after = read_usage(studio, s).last().map_or(0, |u| u.n);
    let mut h = read_history(studio, s);
    h.push(json!({"t": now(), "version": 0, "action": "reject", "reason": rev.reason, "after_n": after}));
    write_lines(&history_file(studio, s), &h)?;
    let _ = std::fs::remove_file(revision_file(studio, s));
    Ok(())
}

/// Puts an older text back. `version` None: the newest kept version. The text that is replaced becomes a version too.
pub fn rollback(studio: &Path, s: &SkillRef, version: Option<u32>) -> Result<u32, String> {
    let _g = lock();
    let target = match version {
        Some(v) => v,
        None => next_version(studio, s).saturating_sub(1),
    };
    if target == 0 {
        return Err(format!("'{}' has no earlier version", s.name));
    }
    let old = std::fs::read_to_string(versions_dir(studio, s).join(format!("v{target:04}.md"))).map_err(|_| format!("'{}' has no version {target}", s.name))?;
    let current = s.read(studio)?;
    let kept = snapshot(studio, s, &current, "rollback", &format!("rolled back to version {target}"))?;
    write_text(&s.file(studio), &old)?;
    Ok(kept)
}

pub fn read_version(studio: &Path, s: &SkillRef, version: u32) -> Option<String> {
    std::fs::read_to_string(versions_dir(studio, s).join(format!("v{version:04}.md"))).ok()
}

/// A line diff in the usual notation (`-` removed, `+` added, two lines of context); `(too large to compare)` past 1500 lines.
pub fn diff(old: &str, new: &str) -> String {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    if a.len() > 1500 || b.len() > 1500 {
        return "(too large to compare)".to_owned();
    }
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] { dp[i + 1][j + 1] + 1 } else { dp[i + 1][j].max(dp[i][j + 1]) };
        }
    }
    let mut ops: Vec<(char, &str)> = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n || j < m {
        if i < n && j < m && a[i] == b[j] {
            ops.push((' ', a[i]));
            i += 1;
            j += 1;
        } else if i < n && (j == m || dp[i + 1][j] >= dp[i][j + 1]) {
            ops.push(('-', a[i]));
            i += 1;
        } else {
            ops.push(('+', b[j]));
            j += 1;
        }
    }
    let keep: Vec<bool> = (0..ops.len()).map(|k| (k.saturating_sub(2)..=(k + 2).min(ops.len().saturating_sub(1))).any(|x| ops[x].0 != ' ')).collect();
    let mut out = String::new();
    let mut skipped = false;
    for (k, (tag, line)) in ops.iter().enumerate() {
        if keep[k] {
            out.push(*tag);
            out.push(' ');
            out.push_str(line);
            out.push('\n');
            skipped = false;
        } else if !skipped {
            out.push_str("  ...\n");
            skipped = true;
        }
    }
    if ops.iter().all(|o| o.0 == ' ') {
        return "(no difference)\n".to_owned();
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------
// memory: entries of the memory files, their sources, and the proposals
// ---------------------------------------------------------------------------------------------------------------

pub fn memory_file(studio: &Path, scope: &str, workspace: Option<&Path>) -> Result<PathBuf, String> {
    match scope {
        "global" | "user" => Ok(studio.join("agent-memory.md")),
        "workspace" | "project" => workspace.map(|w| w.join(".fllm").join("memory.md")).ok_or_else(|| "no project folder is known for a project memory".to_owned()),
        other => Err(format!("scope must be global or workspace, not '{other}'")),
    }
}

pub fn memory_entries_of(text: &str) -> Vec<String> {
    text.lines().filter_map(|l| l.trim().strip_prefix("- ").map(|e| e.trim().to_owned())).filter(|e| !e.is_empty()).collect()
}

pub fn memory_render(entries: &[String]) -> String {
    let mut out = String::from("# Memory\n\n");
    for e in entries {
        out.push_str(&format!("- {e}\n"));
    }
    out
}

pub fn read_memory(studio: &Path, scope: &str, workspace: Option<&Path>) -> Vec<String> {
    memory_file(studio, scope, workspace).ok().and_then(|f| std::fs::read_to_string(f).ok()).map(|t| memory_entries_of(&t)).unwrap_or_default()
}

fn write_memory(studio: &Path, scope: &str, workspace: Option<&Path>, entries: &[String]) -> Result<(), String> {
    let text = memory_render(entries);
    if entries.len() > MEMORY_MAX_ENTRIES || text.len() > MEMORY_FILE_BYTES {
        return Err(format!("the memory is full ({MEMORY_MAX_ENTRIES} entries, {} KB); remove entries first", MEMORY_FILE_BYTES / 1000));
    }
    write_text(&memory_file(studio, scope, workspace)?, &text)
}

fn sources_file(studio: &Path) -> PathBuf {
    learn_dir(studio).join("memory-sources.json")
}

/// Where learned entries came from: normalised text to `{session, created, origin}`.
pub fn memory_sources(studio: &Path) -> BTreeMap<String, Value> {
    read_json(&sources_file(studio)).as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default()
}

fn set_source(studio: &Path, text: &str, session: &str, origin: &str) -> Result<(), String> {
    let mut all = memory_sources(studio);
    all.insert(normalize(text), json!({"session": session, "created": now(), "origin": origin}));
    if all.len() > 1000 {
        let drop: Vec<String> = all.keys().take(all.len() - 1000).cloned().collect();
        for k in drop {
            all.remove(&k);
        }
    }
    write_json(&sources_file(studio), &Value::Object(all.into_iter().collect()))
}

fn drop_source(studio: &Path, text: &str) {
    let mut all = memory_sources(studio);
    if all.remove(&normalize(text)).is_some() {
        let _ = write_json(&sources_file(studio), &Value::Object(all.into_iter().collect()));
    }
}

fn memory_proposals_file(studio: &Path) -> PathBuf {
    learn_dir(studio).join("memory-proposed.json")
}

/// A memory entry (action `add`) or the removal of one (`remove`) that waits for approval.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryProposal {
    pub id: String,
    pub action: String,
    pub scope: String,
    pub workspace: String,
    pub text: String,
    pub kind: String,
    pub session: String,
    pub created: i64,
    pub reason: String,
}

impl MemoryProposal {
    fn to_json(&self) -> Value {
        json!({"id": self.id, "action": self.action, "scope": self.scope, "workspace": self.workspace, "text": self.text, "kind": self.kind,
               "session": self.session, "created": self.created, "reason": self.reason})
    }
    fn from_json(v: &Value) -> Option<MemoryProposal> {
        Some(MemoryProposal {
            id: v["id"].as_str()?.to_owned(),
            action: v["action"].as_str().unwrap_or("add").to_owned(),
            scope: v["scope"].as_str().unwrap_or("global").to_owned(),
            workspace: v["workspace"].as_str().unwrap_or_default().to_owned(),
            text: v["text"].as_str()?.to_owned(),
            kind: v["kind"].as_str().unwrap_or("fact").to_owned(),
            session: v["session"].as_str().unwrap_or_default().to_owned(),
            created: v["created"].as_i64().unwrap_or(0),
            reason: v["reason"].as_str().unwrap_or_default().to_owned(),
        })
    }
}

pub fn memory_proposals(studio: &Path) -> Vec<MemoryProposal> {
    read_json(&memory_proposals_file(studio)).as_array().map(|a| a.iter().filter_map(MemoryProposal::from_json).collect()).unwrap_or_default()
}

fn save_memory_proposals(studio: &Path, list: &[MemoryProposal]) -> Result<(), String> {
    write_json(&memory_proposals_file(studio), &Value::Array(list.iter().map(MemoryProposal::to_json).collect()))
}

/// A candidate memory entry as the model wrote it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryCandidate {
    pub text: String,
    pub kind: String,
    pub scope: String,
}

/// What became of the candidates of one reflection.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ingest {
    /// Entries written to a memory file (auto mode).
    pub added: Vec<String>,
    /// Entries or lines that now wait for approval.
    pub proposed: Vec<String>,
    /// `(text, reason)` of what was dropped.
    pub skipped: Vec<(String, String)>,
}

fn short_id(prefix: &str, parts: &[&str]) -> String {
    format!("{prefix}-{:08x}", fnv(&parts.join("\u{1f}")))
}

/// Cleans a model-written memory text: one line, no list dash, at most [`MEMORY_ENTRY_CHARS`]. `Err` names why it is unusable.
pub fn clean_entry(text: &str, max: usize) -> Result<String, String> {
    let t = one_line(text);
    let t = t.trim_start_matches(['-', '*', '•']).trim().trim_matches('"').trim().to_owned();
    if t.chars().count() < 12 {
        return Err("too short to be useful".into());
    }
    if t.chars().count() > max * 2 {
        return Err("too long".into());
    }
    if has_secret(&t) {
        return Err("looks like a secret".into());
    }
    Ok(clip_chars(&t, max))
}

/// Dedups the candidates against the memory files, the pending proposals and each other, then stores them: written to the memory file
/// when `auto`, else proposed. Every cap of the module is applied. `workspace` is the project folder of the run, if any.
pub fn ingest_memory(studio: &Path, workspace: Option<&Path>, session: &str, candidates: &[MemoryCandidate], auto: bool) -> Ingest {
    let _g = lock();
    let mut result = Ingest::default();
    let mut pending = memory_proposals(studio);
    let mut global = read_memory(studio, "global", None);
    let mut local = workspace.map(|w| read_memory(studio, "workspace", Some(w))).unwrap_or_default();
    let mut taken = 0;
    for c in candidates {
        if taken >= MAX_NEW_PER_RUN {
            result.skipped.push((c.text.clone(), format!("more than {MAX_NEW_PER_RUN} entries in one run")));
            continue;
        }
        let text = match clean_entry(&c.text, MEMORY_ENTRY_CHARS) {
            Ok(t) => t,
            Err(why) => {
                result.skipped.push((c.text.clone(), why));
                continue;
            }
        };
        let scope = if c.scope == "global" || workspace.is_none() { "global" } else { "workspace" };
        let mut known: Vec<String> = global.iter().chain(local.iter()).cloned().collect();
        known.extend(pending.iter().filter(|p| p.action == "add").map(|p| p.text.clone()));
        if find_similar(&text, &known).is_some() {
            result.skipped.push((text, "already remembered or already proposed".into()));
            continue;
        }
        let ws = workspace.map(|w| w.display().to_string()).unwrap_or_default();
        if auto {
            let list = if scope == "global" { &mut global } else { &mut local };
            if list.len() >= MEMORY_MAX_ENTRIES || memory_render(list).len() + text.len() > MEMORY_FILE_BYTES {
                result.skipped.push((text, "the memory is full".into()));
                continue;
            }
            list.push(text.clone());
            if write_memory(studio, scope, workspace, list).is_err() {
                list.pop();
                result.skipped.push((text, "the memory file could not be written".into()));
                continue;
            }
            let _ = set_source(studio, &text, session, "reflection");
            result.added.push(text);
        } else {
            if pending.len() >= MAX_PENDING_MEMORY {
                result.skipped.push((text, format!("{MAX_PENDING_MEMORY} proposals are waiting already; decide on them first")));
                continue;
            }
            let id = short_id("mem", &[&normalize(&text), scope, &ws]);
            pending.push(MemoryProposal {
                id,
                action: "add".into(),
                scope: scope.to_owned(),
                workspace: if scope == "workspace" { ws } else { String::new() },
                text: text.clone(),
                kind: c.kind.clone(),
                session: session.to_owned(),
                created: now(),
                reason: String::new(),
            });
            result.proposed.push(text);
        }
        taken += 1;
    }
    if !result.proposed.is_empty() {
        if let Err(e) = save_memory_proposals(studio, &pending) {
            result.skipped.push(("(proposals)".into(), e));
            result.proposed.clear();
        }
    }
    result
}

/// Proposes the removal of entries that repeat an earlier entry of the same file (the nightly consolidation).
pub fn propose_memory_cleanup(studio: &Path, workspace: Option<&Path>) -> Vec<String> {
    let _g = lock();
    let mut pending = memory_proposals(studio);
    let mut out = Vec::new();
    for scope in ["global", "workspace"] {
        if scope == "workspace" && workspace.is_none() {
            continue;
        }
        let entries = read_memory(studio, scope, workspace);
        for (i, e) in entries.iter().enumerate() {
            let Some((j, _)) = find_similar(e, &entries[..i]) else { continue };
            let ws = if scope == "workspace" { workspace.map(|w| w.display().to_string()).unwrap_or_default() } else { String::new() };
            let id = short_id("rm", &[&normalize(e), scope, &ws]);
            if pending.iter().any(|p| p.id == id) || pending.len() >= MAX_PENDING_MEMORY {
                continue;
            }
            pending.push(MemoryProposal {
                id,
                action: "remove".into(),
                scope: scope.to_owned(),
                workspace: ws,
                text: e.clone(),
                kind: "fact".into(),
                session: String::new(),
                created: now(),
                reason: format!("repeats: {}", clip_chars(&entries[j], 80)),
            });
            out.push(e.clone());
        }
    }
    if !out.is_empty() {
        let _ = save_memory_proposals(studio, &pending);
    }
    out
}

fn proposal_workspace(p: &MemoryProposal) -> Option<PathBuf> {
    (!p.workspace.is_empty()).then(|| PathBuf::from(&p.workspace))
}

/// Applies (`accept`) or drops one memory proposal.
pub fn decide_memory(studio: &Path, id: &str, accept: bool) -> Result<String, String> {
    let _g = lock();
    let mut list = memory_proposals(studio);
    let at = list.iter().position(|p| p.id == id).ok_or_else(|| format!("no memory proposal '{id}'"))?;
    let p = list.remove(at);
    let msg = if accept {
        let ws = proposal_workspace(&p);
        let mut entries = read_memory(studio, &p.scope, ws.as_deref());
        if p.action == "remove" {
            // of several entries with the same words the last one goes, so a repeated entry leaves one behind
            let norm = normalize(&p.text);
            let at = entries.iter().rposition(|e| normalize(e) == norm);
            if let Some(i) = at {
                entries.remove(i);
                write_memory(studio, &p.scope, ws.as_deref(), &entries)?;
            }
            if !entries.iter().any(|e| normalize(e) == norm) {
                drop_source(studio, &p.text);
            }
            if at.is_some() { "removed the entry".to_owned() } else { "the entry was gone already".to_owned() }
        } else {
            if find_similar(&p.text, &entries).is_none() {
                entries.push(p.text.clone());
                write_memory(studio, &p.scope, ws.as_deref(), &entries)?;
            }
            set_source(studio, &p.text, &p.session, "reflection")?;
            format!("remembered: {}", p.text)
        }
    } else {
        format!("dropped: {}", p.text)
    };
    save_memory_proposals(studio, &list)?;
    Ok(msg)
}

/// Replaces entry `index` of a memory file (`text` empty: deletes it).
pub fn edit_memory(studio: &Path, scope: &str, workspace: Option<&Path>, index: usize, text: &str) -> Result<(), String> {
    let _g = lock();
    let mut entries = read_memory(studio, scope, workspace);
    if index >= entries.len() {
        return Err(format!("there is no entry {}", index + 1));
    }
    let old = entries[index].clone();
    if text.trim().is_empty() {
        entries.remove(index);
        drop_source(studio, &old);
    } else {
        let t = clean_entry(text, MEMORY_ENTRY_CHARS)?;
        let src = memory_sources(studio).get(&normalize(&old)).cloned();
        entries[index] = t.clone();
        drop_source(studio, &old);
        if let Some(s) = src {
            let _ = set_source(studio, &t, s["session"].as_str().unwrap_or_default(), s["origin"].as_str().unwrap_or("reflection"));
        }
    }
    write_memory(studio, scope, workspace, &entries)
}

/// Adds an entry written by a person, with no approval step.
pub fn add_memory(studio: &Path, scope: &str, workspace: Option<&Path>, text: &str) -> Result<(), String> {
    let _g = lock();
    let t = clean_entry(text, MEMORY_ENTRY_CHARS)?;
    let mut entries = read_memory(studio, scope, workspace);
    if find_similar(&t, &entries).is_some() {
        return Err("that is remembered already".into());
    }
    entries.push(t);
    write_memory(studio, scope, workspace, &entries)
}

/// `fllm-rs learn forget WORDS`: removes every memory entry (both files), profile line and waiting proposal that contains `needle`
/// (case-insensitive). With `all_learned` it removes everything the loop wrote itself instead. Returns the removed texts.
pub fn forget(studio: &Path, workspace: Option<&Path>, needle: &str, all_learned: bool) -> Result<Vec<String>, String> {
    if needle.trim().is_empty() && !all_learned {
        return Err("say what to forget, or use --all-learned".into());
    }
    let _g = lock();
    let low = needle.trim().to_lowercase();
    let sources = memory_sources(studio);
    let hit = |t: &str| if all_learned { sources.contains_key(&normalize(t)) } else { t.to_lowercase().contains(&low) };
    let mut gone = Vec::new();
    for scope in ["global", "workspace"] {
        if scope == "workspace" && workspace.is_none() {
            continue;
        }
        let entries = read_memory(studio, scope, workspace);
        let keep: Vec<String> = entries.iter().filter(|e| !hit(e)).cloned().collect();
        if keep.len() != entries.len() {
            gone.extend(entries.iter().filter(|e| hit(e)).cloned());
            write_memory(studio, scope, workspace, &keep)?;
        }
    }
    for t in &gone {
        drop_source(studio, t);
    }
    let pending = memory_proposals(studio);
    let keep: Vec<MemoryProposal> = pending.iter().filter(|p| !(all_learned || hit(&p.text))).cloned().collect();
    if keep.len() != pending.len() {
        gone.extend(pending.iter().filter(|p| !keep.iter().any(|k| k.id == p.id)).map(|p| p.text.clone()));
        save_memory_proposals(studio, &keep)?;
    }
    let profile = read_profile(studio);
    if !profile.is_empty() && !all_learned {
        let lines: Vec<&str> = profile.lines().collect();
        let keep: Vec<&str> = lines.iter().copied().filter(|l| !(l.trim_start().starts_with("- ") && l.to_lowercase().contains(&low))).collect();
        if keep.len() != lines.len() {
            gone.extend(lines.iter().filter(|l| !keep.contains(l)).map(|l| l.trim_start_matches(['-', ' ']).to_owned()));
            write_text(&profile_path(studio), &(keep.join("\n") + "\n"))?;
        }
    }
    Ok(gone)
}

// ---------------------------------------------------------------------------------------------------------------
// the profile of the user
// ---------------------------------------------------------------------------------------------------------------

pub fn profile_path(studio: &Path) -> PathBuf {
    studio.join("user-profile.md")
}

pub fn read_profile(studio: &Path) -> String {
    std::fs::read_to_string(profile_path(studio)).unwrap_or_default()
}

fn settings_file(studio: &Path) -> PathBuf {
    learn_dir(studio).join("settings.json")
}

/// Whether the profile goes into the system prompt (default yes: the file is the user's own and editable).
pub fn profile_in_prompt(studio: &Path) -> bool {
    read_json(&settings_file(studio))["profile_in_prompt"].as_bool().unwrap_or(true)
}

pub fn set_profile_in_prompt(studio: &Path, on: bool) -> Result<(), String> {
    let mut v = read_json(&settings_file(studio));
    if !v.is_object() {
        v = json!({});
    }
    v["profile_in_prompt"] = json!(on);
    write_json(&settings_file(studio), &v)
}

/// The block for the system prompt, within [`PROFILE_PROMPT_CHARS`]; empty when there is no profile or it is switched off.
pub fn profile_for_prompt(studio: &Path) -> String {
    if !profile_in_prompt(studio) {
        return String::new();
    }
    let items: Vec<String> = read_profile(studio).lines().filter_map(|l| l.trim().strip_prefix("- ").map(str::to_owned)).collect();
    if items.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let mut section = String::new();
    for line in read_profile(studio).lines() {
        let t = line.trim();
        if let Some(h) = t.strip_prefix("## ") {
            section = h.trim().to_owned();
        } else if let Some(item) = t.strip_prefix("- ") {
            let line = if section.is_empty() { format!("- {item}\n") } else { format!("- {section}: {item}\n") };
            if out.chars().count() + line.chars().count() > PROFILE_PROMPT_CHARS {
                break;
            }
            out.push_str(&line);
        }
    }
    out.trim_end().to_owned()
}

fn profile_title(key: &str) -> Option<&'static str> {
    PROFILE_SECTIONS.iter().find(|(k, _)| *k == key).map(|(_, t)| *t)
}

pub fn profile_bullets_count(text: &str) -> usize {
    profile_bullets(text).len()
}

fn profile_bullets(text: &str) -> Vec<String> {
    text.lines().filter_map(|l| l.trim().strip_prefix("- ").map(|e| e.trim().to_owned())).collect()
}

/// Adds `- item` under `## Title` of a profile text (the heading is created when missing). `Err` when the section is full.
pub fn profile_insert(text: &str, title: &str, item: &str) -> Result<String, String> {
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    if lines.is_empty() {
        lines.push("# User profile".into());
    }
    let heading = format!("## {title}");
    match lines.iter().position(|l| l.trim().eq_ignore_ascii_case(&heading)) {
        Some(at) => {
            let end = lines[at + 1..].iter().position(|l| l.trim_start().starts_with('#')).map_or(lines.len(), |p| at + 1 + p);
            let count = lines[at + 1..end].iter().filter(|l| l.trim_start().starts_with("- ")).count();
            if count >= PROFILE_ITEMS_PER_SECTION {
                return Err(format!("'{title}' holds {PROFILE_ITEMS_PER_SECTION} lines already"));
            }
            let mut pos = end;
            while pos > at + 1 && lines[pos - 1].trim().is_empty() {
                pos -= 1;
            }
            lines.insert(pos, format!("- {item}"));
        }
        None => {
            if lines.last().is_some_and(|l| !l.trim().is_empty()) {
                lines.push(String::new());
            }
            lines.push(heading);
            lines.push(format!("- {item}"));
        }
    }
    Ok(lines.join("\n") + "\n")
}

fn profile_proposals_file(studio: &Path) -> PathBuf {
    learn_dir(studio).join("profile-proposed.json")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileProposal {
    pub id: String,
    pub section: String,
    pub text: String,
    pub session: String,
    pub created: i64,
}

pub fn profile_proposals(studio: &Path) -> Vec<ProfileProposal> {
    read_json(&profile_proposals_file(studio))
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| {
                    Some(ProfileProposal {
                        id: v["id"].as_str()?.to_owned(),
                        section: v["section"].as_str()?.to_owned(),
                        text: v["text"].as_str()?.to_owned(),
                        session: v["session"].as_str().unwrap_or_default().to_owned(),
                        created: v["created"].as_i64().unwrap_or(0),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn save_profile_proposals(studio: &Path, list: &[ProfileProposal]) -> Result<(), String> {
    write_json(
        &profile_proposals_file(studio),
        &Value::Array(list.iter().map(|p| json!({"id": p.id, "section": p.section, "text": p.text, "session": p.session, "created": p.created})).collect()),
    )
}

/// A line the model derived for the profile: `(section key, text)`.
pub type ProfileCandidate = (String, String);

/// Dedups and stores profile lines like [`ingest_memory`]: appended to the file when `auto`, else proposed.
pub fn ingest_profile(studio: &Path, session: &str, candidates: &[ProfileCandidate], auto: bool) -> Ingest {
    let _g = lock();
    let mut result = Ingest::default();
    let mut text = read_profile(studio);
    let mut pending = profile_proposals(studio);
    let mut taken = 0;
    for (section, raw) in candidates {
        let Some(title) = profile_title(section) else {
            result.skipped.push((raw.clone(), format!("unknown section '{section}'")));
            continue;
        };
        if taken >= MAX_NEW_PER_RUN {
            result.skipped.push((raw.clone(), "too many lines in one run".into()));
            continue;
        }
        let item = match clean_entry(raw, PROFILE_ITEM_CHARS) {
            Ok(t) => t,
            Err(why) => {
                result.skipped.push((raw.clone(), why));
                continue;
            }
        };
        let mut known = profile_bullets(&text);
        known.extend(pending.iter().map(|p| p.text.clone()));
        if find_similar(&item, &known).is_some() {
            result.skipped.push((item, "already in the profile or already proposed".into()));
            continue;
        }
        if auto {
            match profile_insert(&text, title, &item) {
                Ok(next) if next.chars().count() <= PROFILE_FILE_CHARS => {
                    text = next;
                    result.added.push(item);
                }
                Ok(_) => result.skipped.push((item, "the profile is full".into())),
                Err(e) => result.skipped.push((item, e)),
            }
        } else if pending.len() >= MAX_PENDING_PROFILE {
            result.skipped.push((item, "too many proposals are waiting".into()));
        } else {
            pending.push(ProfileProposal { id: short_id("prof", &[section, &normalize(&item)]), section: section.clone(), text: item.clone(), session: session.to_owned(), created: now() });
            result.proposed.push(item);
        }
        taken += 1;
    }
    if !result.added.is_empty() && write_text(&profile_path(studio), &text).is_err() {
        result.skipped.extend(result.added.drain(..).map(|a| (a, "the profile could not be written".into())));
    }
    if !result.proposed.is_empty() && save_profile_proposals(studio, &pending).is_err() {
        result.skipped.extend(result.proposed.drain(..).map(|a| (a, "the proposals could not be saved".into())));
    }
    result
}

pub fn decide_profile(studio: &Path, id: &str, accept: bool) -> Result<String, String> {
    let _g = lock();
    let mut list = profile_proposals(studio);
    let at = list.iter().position(|p| p.id == id).ok_or_else(|| format!("no profile proposal '{id}'"))?;
    let p = list.remove(at);
    let msg = if accept {
        let title = profile_title(&p.section).ok_or("unknown section")?;
        let text = profile_insert(&read_profile(studio), title, &p.text)?;
        if text.chars().count() > PROFILE_FILE_CHARS {
            return Err("the profile is full".into());
        }
        write_text(&profile_path(studio), &text)?;
        format!("added to the profile: {}", p.text)
    } else {
        format!("dropped: {}", p.text)
    };
    save_profile_proposals(studio, &list)?;
    Ok(msg)
}

/// Saves a profile written by hand (Settings > Personalization). Secrets are refused; the size is capped.
pub fn save_profile_text(studio: &Path, text: &str) -> Result<(), String> {
    let _g = lock();
    if text.chars().count() > PROFILE_FILE_CHARS {
        return Err(format!("the profile is limited to {PROFILE_FILE_CHARS} characters"));
    }
    if has_secret(text) {
        return Err("the profile contains something that looks like a secret; remove it first".into());
    }
    write_text(&profile_path(studio), &(text.trim_end().to_owned() + "\n"))
}

pub fn delete_profile(studio: &Path) -> Result<(), String> {
    let _g = lock();
    let _ = std::fs::remove_file(profile_path(studio));
    let _ = std::fs::remove_file(profile_proposals_file(studio));
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// overview
// ---------------------------------------------------------------------------------------------------------------

/// Everything that waits for a decision, as JSON (`fllm-rs learn status --json`, the Studio).
pub fn overview(studio: &Path) -> Value {
    let revisions: Vec<Value> = pending_revisions(studio)
        .iter()
        .map(|r| json!({"id": r.id(), "name": r.skill.name, "scope": r.skill.scope, "workspace": r.skill.workspace, "reason": r.reason, "trigger": r.trigger, "created": r.created, "diff": diff(&r.old, &r.new)}))
        .collect();
    let memory: Vec<Value> = memory_proposals(studio).iter().map(MemoryProposal::to_json).collect();
    let profile: Vec<Value> = profile_proposals(studio).iter().map(|p| json!({"id": p.id, "section": p.section, "text": p.text, "session": p.session, "created": p.created})).collect();
    json!({
        "skills": revisions,
        "memory": memory,
        "profile": profile,
        "profile_in_prompt": profile_in_prompt(studio),
        "last_run": read_json(&learn_dir(studio).join("last-run.json")),
        "pending": revisions.len() + memory.len() + profile.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fllm-learn-{tag}-{}-{}", std::process::id(), now()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn skill(studio: &Path, name: &str, body: &str) -> SkillRef {
        let dir = studio.join("skills").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), format!("---\nname: {name}\ndescription: test skill\nwhen: always\n---\n{body}\n")).unwrap();
        SkillRef::user(name)
    }

    #[test]
    fn secrets_are_redacted_in_every_common_shape() {
        let cases = [
            "key sk-abcdefghijklmnopqrstuvwx1234",
            "token ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "AKIAABCDEFGHIJKLMNOP is the id",
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
            "password = hunter2hunter2",
            "API_KEY: \"abcd1234efgh\"",
            "curl https://user:pa55word@example.com/x",
            "run with --api-key abcdefgh12345",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----",
        ];
        for c in cases {
            let r = redact(c);
            assert!(r.contains("redacted"), "not redacted: {c} -> {r}");
            assert!(has_secret(c));
            assert!(!r.contains("hunter2hunter2") && !r.contains("pa55word") && !r.contains("abcdefgh12345") && !r.contains("MIIabc"), "{r}");
        }
        assert!(!has_secret("Run cargo test --lib in rust/engine; the port is 8000"));
        assert_eq!(redact_with("the key is zzzzzz-secret-value here", &["zzzzzz-secret-value".to_owned()]), "the key is [redacted] here");
    }

    #[test]
    fn near_duplicates_are_found_without_embeddings() {
        let existing = vec!["The tests are run with cargo test --lib in the rust folder".to_owned(), "The user prefers tabs over spaces".to_owned()];
        assert_eq!(find_similar("Tests are run with cargo test --lib in rust folder", &existing).map(|f| f.0), Some(0));
        assert_eq!(find_similar("the user prefers tabs over spaces!", &existing).map(|f| f.0), Some(1));
        assert!(find_similar("The database password is rotated monthly by the ops team", &existing).is_none());
        assert!(find_similar("Deploy with docker compose up -d", &existing).is_none());
        assert_eq!(normalize("Tests, TESTS and the test!"), "test test test");
    }

    #[test]
    fn usage_log_stats_and_the_improvement_trigger() {
        let studio = temp("usage");
        let s = skill(&studio, "deploy", "1. build");
        assert_eq!(improve_due(&studio, &s), None);
        for i in 0..4 {
            record_usage(&studio, &s, Usage::new(&format!("s{i}"), "done")).unwrap();
        }
        assert_eq!(improve_due(&studio, &s), None, "four clean uses are not enough");
        record_usage(&studio, &s, Usage::new("s4", "done")).unwrap();
        assert!(improve_due(&studio, &s).unwrap().contains("5 uses"));
        let st = stats_of(&studio, &s);
        assert_eq!((st.uses, st.ok, st.failed), (5, 5, 0));

        // a failure triggers at once, a verify failure and a thumb down count as bad too
        let t = skill(&studio, "other", "x");
        record_usage(&studio, &t, Usage { verify: Some(false), ..Usage::new("a", "done") }).unwrap();
        assert!(improve_due(&studio, &t).unwrap().contains("bad"));
        let st = stats_of(&studio, &t);
        assert_eq!((st.uses, st.ok, st.failed), (1, 0, 1), "a run whose verify command failed is not a success");
        let u = skill(&studio, "third", "x");
        record_usage(&studio, &u, Usage::new("a", "done")).unwrap();
        record_usage(&studio, &u, Usage { feedback: Some("down".into()), ..Usage::new("a", "feedback") }).unwrap();
        let st = stats_of(&studio, &u);
        assert_eq!((st.uses, st.down), (1, 1));
        assert!(improve_due(&studio, &u).is_some());

        // notes are redacted and short, the file is capped
        let rec = record_usage(&studio, &s, Usage { note: format!("failed with token=abcdef123456 {}", "x".repeat(500)), ..Usage::new("s", "failed") }).unwrap();
        assert!(rec.note.contains("redacted") && rec.note.chars().count() <= 300, "{}", rec.note);
        for _ in 0..(USAGE_KEEP + 10) {
            record_usage(&studio, &s, Usage::new("s", "done")).unwrap();
        }
        let all = read_usage(&studio, &s);
        assert_eq!(all.len(), USAGE_KEEP);
        assert!(all.windows(2).all(|w| w[1].n == w[0].n + 1), "numbers keep counting up after trimming");
        let _ = std::fs::remove_dir_all(&studio);
    }

    #[test]
    fn a_revision_is_a_proposal_until_approved_and_every_text_is_kept() {
        let studio = temp("rev");
        let s = skill(&studio, "build", "1. run make");
        let original = s.read(&studio).unwrap();
        record_usage(&studio, &s, Usage::new("a", "failed")).unwrap();
        let better = original.replace("1. run make", "1. run make -j4\n2. run the tests");
        propose_revision(&studio, &s, &better, "make alone skipped the tests", "failure").unwrap();
        assert_eq!(s.read(&studio).unwrap(), original, "a proposal never touches the skill");
        assert_eq!(improve_due(&studio, &s), None, "one proposal at a time");
        let rev = pending_revision(&studio, &s).unwrap();
        assert_eq!(rev.reason, "make alone skipped the tests");
        let d = diff(&rev.old, &rev.new);
        assert!(d.contains("- 1. run make\n") && d.contains("+ 1. run make -j4\n") && d.contains("+ 2. run the tests\n"), "{d}");
        assert_eq!(pending_revisions(&studio).len(), 1);

        assert_eq!(approve_revision(&studio, &s, false).unwrap(), 1);
        assert_eq!(s.read(&studio).unwrap(), better);
        assert_eq!(read_version(&studio, &s, 1).unwrap(), original, "the previous text is kept");
        assert!(pending_revision(&studio, &s).is_none());
        // the usage window restarts: the old failure does not trigger another draft
        assert_eq!(improve_due(&studio, &s), None);

        // second revision, then rollback to the first version
        propose_revision(&studio, &s, &better.replace("-j4", "-j8"), "faster", "uses").unwrap();
        assert_eq!(approve_revision(&studio, &s, false).unwrap(), 2);
        assert!(s.read(&studio).unwrap().contains("-j8"));
        let kept = rollback(&studio, &s, Some(1)).unwrap();
        assert_eq!(kept, 3);
        assert_eq!(s.read(&studio).unwrap(), original);
        assert!(read_version(&studio, &s, 3).unwrap().contains("-j8"), "the replaced text is a version too, a rollback can be rolled back");
        let h = read_history(&studio, &s);
        assert_eq!(h.iter().map(|x| x["action"].as_str().unwrap()).collect::<Vec<_>>(), ["approve", "approve", "rollback"]);
        assert!(rollback(&studio, &s, Some(99)).is_err());
        // default rollback = newest kept version
        rollback(&studio, &s, None).unwrap();
        assert!(s.read(&studio).unwrap().contains("-j8"));
        let _ = std::fs::remove_dir_all(&studio);
    }

    #[test]
    fn a_hand_edit_is_never_overwritten_and_a_rejection_restarts_the_window() {
        let studio = temp("edit");
        let s = skill(&studio, "lint", "1. lint");
        record_usage(&studio, &s, Usage::new("a", "failed")).unwrap();
        let text = s.read(&studio).unwrap();
        propose_revision(&studio, &s, &text.replace("1. lint", "1. lint --fix"), "r", "failure").unwrap();
        let edited = text.replace("1. lint", "1. lint --strict");
        std::fs::write(s.file(&studio), &edited).unwrap();
        let err = approve_revision(&studio, &s, false).unwrap_err();
        assert!(err.contains("edited"), "{err}");
        assert_eq!(s.read(&studio).unwrap(), edited);
        reject_revision(&studio, &s).unwrap();
        assert!(pending_revision(&studio, &s).is_none());
        assert_eq!(improve_due(&studio, &s), None, "a rejected draft is not drafted again for the same failure");
        assert!(reject_revision(&studio, &s).is_err());
        // refused proposals: no change, a secret, too large
        assert!(propose_revision(&studio, &s, &edited, "same", "x").is_err());
        assert!(propose_revision(&studio, &s, &format!("{edited}\npassword = hunter2hunter2"), "s", "x").is_err());
        assert!(propose_revision(&studio, &s, &"y".repeat(MAX_SKILL_BYTES + 1), "big", "x").is_err());
        assert!(pending_revision(&studio, &s).is_none());
        let _ = std::fs::remove_dir_all(&studio);
    }

    #[test]
    fn workspace_skills_have_their_own_log_and_hide_user_skills() {
        let studio = temp("ws");
        let ws = temp("wsproj");
        skill(&studio, "same", "user version");
        let dir = ws.join(".fllm/skills/same");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), "---\nname: same\ndescription: project version\n---\nbody\n").unwrap();
        let list = list_skills(&studio, Some(&ws));
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].skill.scope, "workspace");
        assert_eq!(list[0].description, "project version");
        let w = list[0].skill.clone();
        assert_ne!(w.key(), SkillRef::user("same").key());
        record_usage(&studio, &w, Usage::new("a", "done")).unwrap();
        assert_eq!(read_usage(&studio, &SkillRef::user("same")).len(), 0);
        assert_eq!(list_skills(&studio, None)[0].skill.scope, "user");
        assert!(SkillRef::from_parts("../x", "user", "").is_err());
        assert!(SkillRef::from_parts("a", "workspace", "relative").is_err());
        let _ = std::fs::remove_dir_all(&studio);
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn skill_reads_are_found_in_a_conversation_and_tasks_match_skills() {
        let messages = vec![
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "skill", "arguments": "{\"action\":\"read\",\"name\":\"deploy-app\"}"}},
                                                                       {"id": "2", "function": {"name": "skill", "arguments": "{\"action\":\"list\"}"}},
                                                                       {"id": "3", "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}}]}),
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": "4", "function": {"name": "skill", "arguments": {"action": "read", "name": "deploy-app"}}}]}),
        ];
        assert_eq!(skills_read_in(&messages), ["deploy-app"]);
        let studio = temp("match");
        let skills = vec![
            SkillInfo { skill: SkillRef::user("deploy-app"), description: "Deploy the web app to staging".into(), when: "the user asks to deploy or release".into() },
            SkillInfo { skill: SkillRef::user("sql-query"), description: "Write a SQL query".into(), when: "a database question".into() },
        ];
        let m = match_skills("please deploy the web app to staging now", &skills, 2);
        assert_eq!(m.first().map(|x| x.0.as_str()), Some("deploy-app"));
        assert!(match_skills("rename a variable", &skills, 2).is_empty());
        let _ = std::fs::remove_dir_all(&studio);
    }

    fn cand(text: &str, scope: &str) -> MemoryCandidate {
        MemoryCandidate { text: text.to_owned(), kind: "project".into(), scope: scope.to_owned() }
    }

    #[test]
    fn memory_candidates_are_deduped_redacted_capped_and_proposed_first() {
        let studio = temp("mem");
        let ws = temp("memws");
        std::fs::write(studio.join("agent-memory.md"), "# Memory\n\n- The user prefers tabs over spaces\n").unwrap();
        let cands = vec![
            cand("The tests are run with cargo test --lib in the rust folder", "workspace"),
            cand("User prefers tabs over spaces.", "global"),
            cand("The deploy key is sk-abcdefghijklmnopqrstuvwx1234", "global"),
            cand("short", "global"),
            cand("The tests are run with cargo test --lib in rust folder", "workspace"),
            cand("The build output goes to target/release and is not committed", "workspace"),
        ];
        let r = ingest_memory(&studio, Some(&ws), "sess1", &cands, false);
        assert_eq!(r.proposed.len(), 2, "{r:?}");
        assert!(r.added.is_empty());
        let reasons: Vec<&str> = r.skipped.iter().map(|s| s.1.as_str()).collect();
        assert!(reasons.iter().any(|x| x.contains("already")) && reasons.iter().any(|x| x.contains("secret")) && reasons.iter().any(|x| x.contains("short")), "{reasons:?}");
        assert!(!std::fs::read_to_string(studio.join("agent-memory.md")).unwrap().contains("cargo test"), "a proposal writes no memory");
        // the same candidates again are all duplicates of the proposals
        let again = ingest_memory(&studio, Some(&ws), "sess2", &cands[..1], false);
        assert!(again.proposed.is_empty());

        let p = memory_proposals(&studio);
        assert_eq!(p.len(), 2);
        assert!(p[0].session == "sess1" && p[0].scope == "workspace" && p[0].workspace == ws.display().to_string());
        decide_memory(&studio, &p[0].id, true).unwrap();
        decide_memory(&studio, &p[1].id, false).unwrap();
        assert!(read_memory(&studio, "workspace", Some(&ws)).iter().any(|e| e.contains("cargo test")));
        assert!(!read_memory(&studio, "workspace", Some(&ws)).iter().any(|e| e.contains("target/release")));
        assert!(memory_proposals(&studio).is_empty());
        assert_eq!(memory_sources(&studio).values().next().unwrap()["session"], "sess1");
        assert!(decide_memory(&studio, "nope", true).is_err());

        // auto mode writes at once, still with all checks; no workspace: everything is global
        let auto = ingest_memory(&studio, None, "sess3", &[cand("Prefer pnpm over npm in every project of this user", "workspace")], true);
        assert_eq!(auto.added.len(), 1);
        assert!(read_memory(&studio, "global", None).iter().any(|e| e.contains("pnpm")));
        let _ = std::fs::remove_dir_all(&studio);
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn memory_caps_hold() {
        let studio = temp("memcap");
        let ws = temp("memcapws");
        let many: Vec<MemoryCandidate> = (0..9).map(|i| cand(&format!("Unique fact number {i} about module{i}{i}{i} and gadget{i}{i}"), "global")).collect();
        let r = ingest_memory(&studio, Some(&ws), "s", &many, true);
        assert_eq!(r.added.len(), MAX_NEW_PER_RUN, "at most {MAX_NEW_PER_RUN} per run");
        assert_eq!(r.skipped.len(), 4);
        // a full file takes nothing more
        let full: Vec<String> = (0..MEMORY_MAX_ENTRIES).map(|i| format!("Entry {i} about subsystem{i}x{i}y")).collect();
        std::fs::write(studio.join("agent-memory.md"), memory_render(&full)).unwrap();
        let r = ingest_memory(&studio, None, "s", &[cand("A brand new and different insight about zebras", "global")], true);
        assert!(r.added.is_empty() && r.skipped[0].1.contains("full"), "{r:?}");
        // the proposal queue is capped
        let _ = std::fs::remove_file(studio.join("agent-memory.md"));
        for round in 0..(MAX_PENDING_MEMORY / MAX_NEW_PER_RUN + 2) {
            let batch: Vec<MemoryCandidate> = (0..MAX_NEW_PER_RUN).map(|i| cand(&format!("Round{round} idea{i} regarding quark{round}{i}z topic{i}{round}"), "global")).collect();
            ingest_memory(&studio, None, "s", &batch, false);
        }
        assert_eq!(memory_proposals(&studio).len(), MAX_PENDING_MEMORY);
        // an entry longer than twice the limit is rejected, a long one is clipped
        assert!(clean_entry(&"w ".repeat(MEMORY_ENTRY_CHARS + 10), MEMORY_ENTRY_CHARS).is_err());
        assert!(clean_entry(&format!("{} end", "word ".repeat(90)), MEMORY_ENTRY_CHARS).unwrap().chars().count() <= MEMORY_ENTRY_CHARS);
        assert_eq!(clean_entry("- a leading dash is removed from this entry", 100).unwrap(), "a leading dash is removed from this entry");
        let _ = std::fs::remove_dir_all(&studio);
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn memory_can_be_edited_forgotten_and_cleaned_up() {
        let studio = temp("memedit");
        std::fs::write(studio.join("agent-memory.md"), "# Memory\n\n- Alpha convention for naming things\n- Beta convention for naming things\n- Gamma rule about commits\n- Gamma rule about commits!\n").unwrap();
        ingest_memory(&studio, None, "s9", &[cand("Learned insight regarding the staging cluster rollout", "global")], true);
        assert_eq!(read_memory(&studio, "global", None).len(), 5);
        edit_memory(&studio, "global", None, 2, "Gamma rule about commit messages and their length").unwrap();
        assert!(read_memory(&studio, "global", None)[2].contains("messages"));
        assert!(edit_memory(&studio, "global", None, 40, "x").is_err());
        edit_memory(&studio, "global", None, 3, "").unwrap();
        assert_eq!(read_memory(&studio, "global", None).len(), 4);
        add_memory(&studio, "global", None, "Written by hand: staging uses port 9001").unwrap();
        assert!(add_memory(&studio, "global", None, "Written by hand: staging uses port 9001").is_err());
        // the cleanup proposes a removal for the repeated entry; accepting it removes exactly that one
        std::fs::write(studio.join("agent-memory.md"), "# Memory\n\n- Alpha convention for naming things\n- Alpha convention for naming things!\n- Zulu unrelated statement here\n").unwrap();
        let dup = propose_memory_cleanup(&studio, None);
        assert_eq!(dup.len(), 1);
        assert!(propose_memory_cleanup(&studio, None).is_empty() || memory_proposals(&studio).len() == 1, "no second proposal for the same entry");
        let p = memory_proposals(&studio);
        assert_eq!(p[0].action, "remove");
        decide_memory(&studio, &p[0].id, true).unwrap();
        assert_eq!(read_memory(&studio, "global", None).len(), 2);
        // forget: by words, then everything the loop wrote
        ingest_memory(&studio, None, "s10", &[cand("Learned fact about the zebra deployment pipeline", "global")], true);
        let gone = forget(&studio, None, "zulu", false).unwrap();
        assert_eq!(gone.len(), 1);
        assert!(forget(&studio, None, "", false).is_err());
        let gone = forget(&studio, None, "", true).unwrap();
        assert!(gone.iter().any(|g| g.contains("zebra")), "{gone:?}");
        assert!(read_memory(&studio, "global", None).iter().all(|e| !e.contains("zebra")));
        assert!(read_memory(&studio, "global", None).iter().any(|e| e.contains("Alpha")), "hand-written entries stay");
        let _ = std::fs::remove_dir_all(&studio);
    }

    #[test]
    fn the_profile_is_plain_text_with_caps_dedup_and_a_prompt_block() {
        let studio = temp("prof");
        assert_eq!(profile_for_prompt(&studio), "");
        let c = |s: &str, t: &str| (s.to_owned(), t.to_owned());
        let r = ingest_profile(&studio, "s1", &[c("languages", "Writes mostly Rust and some Python"), c("preferences", "Wants short answers without filler"), c("bogus", "x is long enough"), c("languages", "password = hunter2hunter2 is used")], false);
        assert_eq!(r.proposed.len(), 2, "{r:?}");
        assert_eq!(r.skipped.len(), 2);
        assert!(!profile_path(&studio).exists(), "proposals write no profile");
        for p in profile_proposals(&studio) {
            decide_profile(&studio, &p.id, p.section == "languages").unwrap();
        }
        let text = read_profile(&studio);
        assert!(text.contains("## Languages\n- Writes mostly Rust") && !text.contains("short answers"), "{text}");
        let block = profile_for_prompt(&studio);
        assert_eq!(block, "- Languages: Writes mostly Rust and some Python");
        // the same line again is a duplicate; auto mode appends under the heading; a hand edit survives
        let again = ingest_profile(&studio, "s2", &[c("languages", "writes mostly rust and some python")], true);
        assert!(again.added.is_empty());
        save_profile_text(&studio, &format!("{}\n## Notes\n- Hand written note\n", text.trim_end())).unwrap();
        let auto = ingest_profile(&studio, "s3", &[c("languages", "Also reads Go from time to time")], true);
        assert_eq!(auto.added.len(), 1);
        let text = read_profile(&studio);
        assert!(text.contains("Hand written note") && text.find("Also reads Go").unwrap() < text.find("## Notes").unwrap(), "{text}");
        // section cap
        let mut t = String::new();
        for i in 0..PROFILE_ITEMS_PER_SECTION {
            t = profile_insert(&t, "Conventions", &format!("item {i}")).unwrap();
        }
        assert!(profile_insert(&t, "Conventions", "one too many").is_err());
        // prompt block is capped, switchable
        let long: String = (0..8).map(|i| format!("- {}\n", format!("long line {i} ").repeat(12))).collect();
        save_profile_text(&studio, &format!("## Preferences\n{long}")).unwrap();
        assert!(profile_for_prompt(&studio).chars().count() <= PROFILE_PROMPT_CHARS);
        set_profile_in_prompt(&studio, false).unwrap();
        assert_eq!(profile_for_prompt(&studio), "");
        set_profile_in_prompt(&studio, true).unwrap();
        // hand edits: secrets and size are refused; delete removes the file
        assert!(save_profile_text(&studio, "- token = abcdef123456").is_err());
        assert!(save_profile_text(&studio, &"a".repeat(PROFILE_FILE_CHARS + 1)).is_err());
        delete_profile(&studio).unwrap();
        assert_eq!(read_profile(&studio), "");
        let _ = std::fs::remove_dir_all(&studio);
    }

    #[test]
    fn the_diff_shows_changes_with_context() {
        assert_eq!(diff("a\nb\nc", "a\nb\nc"), "(no difference)\n");
        let old: String = (1..=12).map(|i| format!("line{i}\n")).collect();
        let new = old.replace("line6\n", "changed\n");
        let d = diff(&old, &new);
        assert!(d.contains("- line6\n+ changed\n") && d.contains("  ...\n") && !d.contains("line1\n"), "{d}");
    }
}
