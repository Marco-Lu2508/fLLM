//! The model-facing half of the self-improvement loop: the prompts and JSON schemas of the reflective steps, the validation of what a
//! small model answers, and the steps that run after a session (note how the skills fared, draft a skill revision, propose memory
//! entries and profile lines). A model call is a closure `body -> reply`, so the command line, the Studio server and the tests each
//! bring their own and everything here is tested without a model.
//!
//! The steps are written for 3B to 8B models: the answer is a small JSON object constrained by `response_format` (a JSON schema the
//! server turns into a grammar); a server that refuses the field is asked again without it, and an answer that still does not
//! validate is dropped. Nothing is written without approval unless the caller asked for automatic memory.
use crate::learn::{self, Ingest, MemoryCandidate, ProfileCandidate, SkillInfo, SkillRef, Usage};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// One model call: a chat-completions body in, the reply body out.
pub type Complete<'a> = &'a mut dyn FnMut(Value) -> Result<Value, String>;

/// Longest answer the reflective steps ask for.
pub const REFLECT_TOKENS: usize = 700;
pub const IMPROVE_TOKENS: usize = 1500;
const MAX_SKILL_BODY_CHARS: usize = 12_000;
const MAX_REVISIONS_PER_RUN: usize = 2;

// ---------------------------------------------------------------------------------------------------------------
// reading model output
// ---------------------------------------------------------------------------------------------------------------

fn content_text(message: &Value) -> String {
    match &message["content"] {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

/// The assistant text of a chat-completions reply without a `<think>` block.
pub fn reply_text(reply: &Value) -> String {
    let text = content_text(&reply["choices"][0]["message"]);
    match (text.find("<think>"), text.rfind("</think>")) {
        (Some(a), Some(b)) if b >= a => format!("{}{}", &text[..a], &text[b + 8..]),
        (None, Some(b)) => text[b + 8..].to_owned(),
        _ => text,
    }
}

/// The JSON object in a model answer: the whole text, a fenced block, or the outermost braces.
pub fn extract_json(text: &str) -> Option<Value> {
    let t = text.trim();
    let t = t.strip_prefix("```json").or_else(|| t.strip_prefix("```")).unwrap_or(t);
    let t = t.strip_suffix("```").unwrap_or(t).trim();
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        if v.is_object() {
            return Some(v);
        }
    }
    let (a, b) = (t.find('{')?, t.rfind('}')?);
    (b > a).then(|| serde_json::from_str::<Value>(&t[a..=b]).ok()).flatten().filter(Value::is_object)
}

/// Sends the request with its `response_format`; a server that refuses the field (or the grammar) is asked again without it and with
/// thinking switched off. Returns the answer text, which still has to be validated.
pub fn structured_call(complete: Complete, body: Value) -> Result<String, String> {
    let reply = match complete(body.clone()) {
        Ok(r) => r,
        Err(e) => {
            let low = e.to_lowercase();
            if !(low.contains("http 400") || low.contains("http 422") || low.contains("response_format") || low.contains("grammar") || low.contains("schema")) {
                return Err(e);
            }
            let mut plain = body;
            if let Some(o) = plain.as_object_mut() {
                o.remove("response_format");
                o.insert("chat_template_kwargs".into(), json!({"enable_thinking": false}));
            }
            complete(plain)?
        }
    };
    let text = reply_text(&reply);
    if text.trim().is_empty() {
        return Err("the model gave an empty answer".into());
    }
    Ok(text)
}

fn schema_format(name: &str, schema: Value) -> Value {
    json!({"type": "json_schema", "json_schema": {"name": name, "strict": true, "schema": schema}})
}

fn request(system: &str, user: &str, max_tokens: usize, name: &str, schema: Value) -> Value {
    json!({
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": false,
        "response_format": schema_format(name, schema),
    })
}

// ---------------------------------------------------------------------------------------------------------------
// the transcript the reflective steps read
// ---------------------------------------------------------------------------------------------------------------

/// A short, redacted account of a session: what was asked, which tools ran and how they ended, what was answered. Kept to
/// `max_chars` by cutting out the middle.
pub fn transcript(messages: &[Value], max_chars: usize, secrets: &[String]) -> String {
    let mut lines: Vec<String> = Vec::new();
    let flat = |t: &str, n: usize| learn::clip_chars(&t.split_whitespace().collect::<Vec<_>>().join(" "), n);
    for m in messages {
        match m["role"].as_str() {
            Some("user") => {
                let t = learn::redact_with(&content_text(m), secrets);
                if !t.trim().is_empty() && !t.starts_with("[Summary") {
                    lines.push(format!("USER: {}", flat(&t, 700)));
                }
            }
            Some("assistant") => {
                let t = learn::redact_with(&content_text(m), secrets);
                if !t.trim().is_empty() {
                    lines.push(format!("ASSISTANT: {}", flat(&t, 500)));
                }
                for call in m["tool_calls"].as_array().into_iter().flatten() {
                    let args = match &call["function"]["arguments"] {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    lines.push(format!("CALL {}({})", call["function"]["name"].as_str().unwrap_or("?"), flat(&learn::redact_with(&args, secrets), 140)));
                }
            }
            Some("tool") => lines.push(format!("RESULT: {}", flat(&learn::redact_with(&content_text(m), secrets), 160))),
            _ => {}
        }
    }
    let full = lines.join("\n");
    if full.chars().count() <= max_chars {
        return full;
    }
    let head: String = full.chars().take(max_chars / 3).collect();
    let tail: String = full.chars().rev().take(max_chars * 2 / 3).collect::<Vec<_>>().into_iter().rev().collect();
    format!("{head}\n[... the middle of the session is left out ...]\n{tail}")
}

/// Whether a session has enough in it to be worth a reflective model call.
pub fn worth_reflecting(messages: &[Value]) -> bool {
    let calls: usize = messages.iter().map(|m| m["tool_calls"].as_array().map_or(0, Vec::len)).sum();
    let text: usize = messages.iter().filter(|m| matches!(m["role"].as_str(), Some("user" | "assistant"))).map(|m| content_text(m).chars().count()).sum();
    messages.iter().any(|m| m["role"] == "user") && messages.iter().any(|m| m["role"] == "assistant") && (calls >= 2 || text >= 400)
}

// ---------------------------------------------------------------------------------------------------------------
// memory and profile: the reflective step
// ---------------------------------------------------------------------------------------------------------------

pub fn reflect_schema() -> Value {
    let kinds = ["project", "preference", "command", "mistake", "fact"];
    let sections: Vec<&str> = learn::PROFILE_SECTIONS.iter().map(|(k, _)| *k).collect();
    json!({
        "type": "object",
        "properties": {
            "memory": {"type": "array", "maxItems": learn::MAX_NEW_PER_RUN, "items": {"type": "object", "properties": {
                "text": {"type": "string", "maxLength": 300},
                "kind": {"enum": kinds},
                "scope": {"enum": ["workspace", "global"]}
            }, "required": ["text", "kind", "scope"], "additionalProperties": false}},
            "profile": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
                "section": {"enum": sections},
                "text": {"type": "string", "maxLength": learn::PROFILE_ITEM_CHARS}
            }, "required": ["section", "text"], "additionalProperties": false}}
        },
        "required": ["memory", "profile"],
        "additionalProperties": false
    })
}

const REFLECT_SYSTEM: &str = "You review a finished coding session and decide what is worth remembering for later sessions. You answer with one JSON object and nothing else.";

pub fn reflect_body(transcript: &str, known_memory: &[String], profile: &str, max_tokens: usize) -> Value {
    let mut known = String::new();
    for e in known_memory.iter().rev().take(20) {
        let line = format!("- {}\n", learn::clip_chars(e, 100));
        if known.len() + line.len() > 1500 {
            break;
        }
        known.push_str(&line);
    }
    let user = format!(
        "Session:\n{transcript}\n\nAlready remembered (do not repeat):\n{}\nProfile of the user so far:\n{}\n\n\
         Task: list what is worth keeping for FUTURE sessions in \"memory\": how to build, test and run this project (exact commands that worked), conventions of the project, \
         preferences the user stated, mistakes or traps that cost time here. Rules: at most {} entries, each one short sentence that is true and useful without this session; \
         no secrets, passwords or keys. Do NOT describe what files contain or what the code does, do not retell the task, and do not note general programming advice. \
         Good: \"Tests run with node test.js.\" Bad: \"math.js has two functions.\" \
         \"scope\" is \"workspace\" for facts about this project and \"global\" for facts about the user. \
         In \"profile\" list durable facts about the USER, taken only from what the user wrote (lines starting with USER:), never from what the agent did: \
         a stated preference, skill level, language, convention or tool habit. A user who only gave a task has shown nothing: then \"profile\" is empty. \
         When there is nothing worth keeping, use empty lists; that is the usual case.\n\
         Answer as JSON: {{\"memory\":[{{\"text\":\"...\",\"kind\":\"project|preference|command|mistake|fact\",\"scope\":\"workspace|global\"}}],\"profile\":[{{\"section\":\"preferences|skill_level|languages|conventions|tool_habits\",\"text\":\"...\"}}]}}",
        if known.is_empty() { "(nothing yet)\n" } else { &known },
        if profile.trim().is_empty() { "(empty)" } else { profile.trim() },
        learn::MAX_NEW_PER_RUN,
    );
    request(REFLECT_SYSTEM, &user, max_tokens, "reflection", reflect_schema())
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Reflection {
    pub memory: Vec<MemoryCandidate>,
    pub profile: Vec<ProfileCandidate>,
    /// Items of the answer that did not fit the schema and were left out.
    pub dropped: usize,
}

/// Validates the answer of the reflective step against the schema; items that do not fit are dropped, an answer that is not the
/// object at all is an error.
pub fn parse_reflection(text: &str) -> Result<Reflection, String> {
    let v = extract_json(text).ok_or("the model did not answer with a JSON object")?;
    let mut r = Reflection::default();
    for item in v["memory"].as_array().into_iter().flatten() {
        let Some(t) = item["text"].as_str().filter(|t| !t.trim().is_empty()) else {
            r.dropped += 1;
            continue;
        };
        let kind = item["kind"].as_str().filter(|k| ["project", "preference", "command", "mistake", "fact"].contains(k)).unwrap_or("fact");
        let scope = item["scope"].as_str().filter(|s| ["workspace", "global"].contains(s)).unwrap_or("workspace");
        r.memory.push(MemoryCandidate { text: t.to_owned(), kind: kind.to_owned(), scope: scope.to_owned() });
    }
    for item in v["profile"].as_array().into_iter().flatten() {
        match (item["section"].as_str(), item["text"].as_str()) {
            (Some(s), Some(t)) if learn::PROFILE_SECTIONS.iter().any(|(k, _)| *k == s) && !t.trim().is_empty() => r.profile.push((s.to_owned(), t.to_owned())),
            _ => r.dropped += 1,
        }
    }
    if v.get("memory").is_none() && v.get("profile").is_none() {
        return Err("the answer has neither \"memory\" nor \"profile\"".into());
    }
    Ok(r)
}

// ---------------------------------------------------------------------------------------------------------------
// skills: the improvement step
// ---------------------------------------------------------------------------------------------------------------

pub fn improve_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "change": {"enum": ["none", "revise"]},
            "reason": {"type": "string", "maxLength": 400},
            "description": {"type": "string", "maxLength": 200},
            "when": {"type": "string", "maxLength": 200},
            "steps": {"type": "string", "maxLength": 6000}
        },
        "required": ["change", "reason", "description", "when", "steps"],
        "additionalProperties": false
    })
}

const IMPROVE_SYSTEM: &str = "You maintain a library of skills: short step-by-step procedures that a coding agent follows. You answer with one JSON object and nothing else.";

pub fn improve_body(skill_text: &str, usage: &[Usage], transcript: &str, trigger: &str, max_tokens: usize) -> Value {
    let mut history = String::new();
    for u in usage.iter().rev().take(6).collect::<Vec<_>>().into_iter().rev() {
        let verdict = match (u.outcome.as_str(), u.verify, u.feedback.as_deref()) {
            (_, _, Some("down")) => "user said it did not work",
            (_, _, Some("up")) => "user said it worked",
            ("failed", _, _) => "run failed",
            (_, Some(false), _) => "checks failed afterwards",
            _ => "finished",
        };
        history.push_str(&format!("- {verdict}{}{}\n", if u.retries > 0 { format!(", {} retries", u.retries) } else { String::new() }, if u.note.is_empty() { String::new() } else { format!(": {}", u.note) }));
    }
    let user = format!(
        "The skill below was used by the agent. Reason for this review: {trigger}.\n\nHow the recent uses went:\n{}\nCurrent SKILL.md:\n{}\n\nLatest session (excerpt):\n{}\n\n\
         Task: if the skill is wrong, misses a step, or hides a trap that the uses revealed, write the improved skill. Keep what works and change as little as needed. \
         \"steps\" is the Markdown body (numbered steps with exact commands, then pitfalls), without a front matter. No secrets and no paths that exist only on one computer. \
         If the skill is fine as it is, answer {{\"change\":\"none\", ...}} with empty strings.\n\
         Answer as JSON: {{\"change\":\"none|revise\",\"reason\":\"why, one sentence\",\"description\":\"one sentence\",\"when\":\"one sentence on when to use it\",\"steps\":\"markdown\"}}",
        if history.is_empty() { "(no notes)\n" } else { &history },
        learn::clip_chars(skill_text, 6000),
        if transcript.trim().is_empty() { "(not available)" } else { transcript },
    );
    request(IMPROVE_SYSTEM, &user, max_tokens, "skill_revision", improve_schema())
}

#[derive(Debug, PartialEq, Eq)]
pub struct Improvement {
    pub reason: String,
    pub text: String,
}

fn strip_fence(text: &str) -> &str {
    let t = text.trim();
    let t = t.strip_prefix("```markdown").or_else(|| t.strip_prefix("```md")).or_else(|| t.strip_prefix("```")).unwrap_or(t);
    t.strip_suffix("```").unwrap_or(t).trim()
}

/// Builds the revised SKILL.md from the answer. `Ok(None)`: the model says the skill is fine. The name and the extra keys of the
/// front matter (`license`, `allowed-tools`) are kept from `current`; the model only writes description, `when` and the steps.
pub fn parse_improvement(text: &str, current: &str) -> Result<Option<Improvement>, String> {
    let v = extract_json(text).ok_or("the model did not answer with a JSON object")?;
    match v["change"].as_str() {
        Some("none") => return Ok(None),
        Some("revise") => {}
        _ => return Err("\"change\" is neither none nor revise".into()),
    }
    let reason = v["reason"].as_str().map(str::trim).filter(|r| !r.is_empty()).ok_or("the revision gives no reason")?;
    let mut body = strip_fence(v["steps"].as_str().unwrap_or_default()).to_owned();
    // a model that writes a whole file: take its body
    if body.starts_with("---") {
        if let Some((_, b)) = crate::skills_cli::split(&body) {
            body = b.trim().to_owned();
        }
    }
    if body.chars().count() < 20 {
        return Err("the revised steps are empty".into());
    }
    if body.chars().count() > MAX_SKILL_BODY_CHARS {
        return Err("the revised skill is too long".into());
    }
    let (name, old_description, old_when) = crate::skills_cli::read_meta(current);
    let old_body = crate::skills_cli::split(current).map_or_else(|| current.to_owned(), |(_, b)| b);
    if old_body.trim().chars().count() >= 200 && body.chars().count() * 4 < old_body.trim().chars().count() {
        return Err("the revision would drop most of the skill".into());
    }
    let pick = |new: &str, old: &str| {
        let n = new.split_whitespace().collect::<Vec<_>>().join(" ");
        if n.is_empty() { old.to_owned() } else { n }
    };
    let description = pick(v["description"].as_str().unwrap_or_default(), &old_description);
    let when = pick(v["when"].as_str().unwrap_or_default(), &old_when);
    let mut file = format!("---\nname: {name}\ndescription: {description}\n");
    if !when.is_empty() {
        file.push_str(&format!("when: {when}\n"));
    }
    if let Some((head, _)) = crate::skills_cli::split(current) {
        for key in ["license", "compatibility", "allowed-tools"] {
            let value = crate::skills_cli::head_value(&head, key);
            if !value.is_empty() {
                file.push_str(&format!("{key}: {}\n", value.split_whitespace().collect::<Vec<_>>().join(" ")));
            }
        }
    }
    file.push_str(&format!("---\n{}\n", body.trim_end()));
    if learn::has_secret(&file) {
        return Err("the revision contains something that looks like a secret".into());
    }
    if file.trim() == current.trim() {
        return Ok(None);
    }
    Ok(Some(Improvement { reason: reason.to_owned(), text: file }))
}

/// Drafts a revision of `skill` and stores it as a proposal. `Ok(true)` when a proposal was written, `Ok(false)` when the model
/// found nothing to change.
pub fn improve_skill(studio: &Path, skill: &SkillRef, trigger: &str, session_text: &str, complete: Complete) -> Result<bool, String> {
    let current = skill.read(studio)?;
    let usage = learn::read_usage(studio, skill);
    let body = improve_body(&current, &usage, session_text, trigger, IMPROVE_TOKENS);
    let answer = structured_call(complete, body)?;
    match parse_improvement(&answer, &current)? {
        Some(imp) => {
            learn::propose_revision(studio, skill, &imp.text, &imp.reason, trigger)?;
            Ok(true)
        }
        None => Ok(false),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// what runs after a session
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct RunInfo {
    pub session: String,
    pub workspace: Option<PathBuf>,
    /// `done` or `failed`.
    pub outcome: String,
    pub verify: Option<bool>,
    pub retries: u32,
    pub note: String,
    /// Skills the model read during the run.
    pub skills: Vec<String>,
    /// Skills the task text matched before the model asked for them.
    pub matched: Vec<String>,
    /// The verdict of a person, if there is one already.
    pub feedback: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Memory entries and profile lines are written at once (still deduped, redacted and capped) instead of proposed.
    pub auto_memory: bool,
    pub reflect: bool,
    pub improve: bool,
    pub transcript_chars: usize,
    /// Literal secrets (the API key of this run) to cut out of everything the model reads.
    pub secrets: Vec<String>,
}

impl Default for Options {
    fn default() -> Self {
        Options { auto_memory: false, reflect: true, improve: true, transcript_chars: 6000, secrets: Vec::new() }
    }
}

/// Writes the usage of the skills of a run. Returns the skills that were found.
pub fn record_skill_use(studio: &Path, info: &RunInfo) -> Vec<SkillRef> {
    let installed = learn::list_skills(studio, info.workspace.as_deref());
    let mut found = Vec::new();
    for name in &info.skills {
        let Some(s) = installed.iter().find(|s| &s.skill.name == name) else { continue };
        let u = Usage {
            verify: info.verify,
            retries: info.retries,
            note: info.note.clone(),
            matched: info.matched.contains(name),
            feedback: info.feedback.clone(),
            ..Usage::new(&info.session, &info.outcome)
        };
        if learn::record_usage(studio, &s.skill, u).is_ok() {
            found.push(s.skill.clone());
        }
    }
    found
}

/// The memory/profile reflection of one session: asks the model, validates, dedups, stores. `Err` means nothing was stored.
pub fn reflect(studio: &Path, workspace: Option<&Path>, session: &str, messages: &[Value], opts: &Options, complete: Complete) -> Result<(Ingest, Ingest), String> {
    if !worth_reflecting(messages) {
        return Err("the session is too short to learn from".into());
    }
    let mut known = learn::read_memory(studio, "global", None);
    if let Some(w) = workspace {
        known.extend(learn::read_memory(studio, "workspace", Some(w)));
    }
    let body = reflect_body(&transcript(messages, opts.transcript_chars, &opts.secrets), &known, &learn::clip_chars(&learn::read_profile(studio), 800), REFLECT_TOKENS);
    let answer = structured_call(complete, body)?;
    let r = parse_reflection(&answer)?;
    let memory = learn::ingest_memory(studio, workspace, session, &r.memory, opts.auto_memory);
    let profile = learn::ingest_profile(studio, session, &r.profile, opts.auto_memory);
    Ok((memory, profile))
}

fn describe(what: &str, i: &Ingest) -> Vec<String> {
    let mut out = Vec::new();
    for t in &i.added {
        out.push(format!("[learn] remembered ({what}): {}", learn::clip_chars(t, 120)));
    }
    for t in &i.proposed {
        out.push(format!("[learn] proposed ({what}): {}", learn::clip_chars(t, 120)));
    }
    out
}

/// Everything that follows a session: usage notes for the skills it used, drafts of skill revisions that are due, and the reflection
/// on memory and profile. Failures of a step are reported as a line and never stop the others.
pub fn after_run(studio: &Path, info: &RunInfo, messages: &[Value], opts: &Options, complete: Complete) -> Vec<String> {
    let mut lines = Vec::new();
    let used = record_skill_use(studio, info);
    for s in &used {
        lines.push(format!("[learn] skill '{}' used ({})", s.name, info.outcome));
    }
    if opts.improve {
        let excerpt = transcript(messages, opts.transcript_chars, &opts.secrets);
        let mut drafted = 0;
        for s in &used {
            if drafted >= MAX_REVISIONS_PER_RUN {
                break;
            }
            let Some(trigger) = learn::improve_due(studio, s) else { continue };
            match improve_skill(studio, s, &trigger, &excerpt, complete) {
                Ok(true) => {
                    drafted += 1;
                    lines.push(format!("[learn] drafted a revision of skill '{}' ({trigger}); see: fllm-rs learn review", s.name));
                }
                Ok(false) => lines.push(format!("[learn] skill '{}' reviewed ({trigger}): no change proposed", s.name)),
                Err(e) => lines.push(format!("[learn] skill '{}': no revision ({e})", s.name)),
            }
        }
    }
    if opts.reflect {
        match reflect(studio, info.workspace.as_deref(), &info.session, messages, opts, complete) {
            Ok((m, p)) => {
                lines.extend(describe("memory", &m));
                lines.extend(describe("profile", &p));
                let skipped = m.skipped.len() + p.skipped.len();
                if skipped > 0 {
                    lines.push(format!("[learn] {skipped} suggestion(s) dropped (duplicates, too short, secrets, caps)"));
                }
                if m.added.is_empty() && m.proposed.is_empty() && p.added.is_empty() && p.proposed.is_empty() {
                    lines.push("[learn] nothing new worth remembering".into());
                }
            }
            Err(e) => lines.push(format!("[learn] reflection skipped: {e}")),
        }
    }
    save_last_run(studio, &info.session, &lines);
    lines
}

pub fn save_last_run(studio: &Path, session: &str, lines: &[String]) {
    let file = learn::learn_dir(studio).join("last-run.json");
    let _ = std::fs::create_dir_all(learn::learn_dir(studio));
    let _ = std::fs::write(file, json!({"t": learn::now(), "session": session, "lines": lines}).to_string());
}

/// The nightly job: draft the skill revisions that are due and propose the removal of repeated memory entries. Never writes a
/// skill or a memory entry itself.
pub fn consolidate(studio: &Path, workspace: Option<&Path>, complete: Complete) -> Vec<String> {
    let mut lines = Vec::new();
    let mut drafted = 0;
    for info in learn::list_skills(studio, workspace) {
        if drafted >= 3 {
            break;
        }
        let Some(trigger) = learn::improve_due(studio, &info.skill) else { continue };
        match improve_skill(studio, &info.skill, &trigger, "", complete) {
            Ok(true) => {
                drafted += 1;
                lines.push(format!("drafted a revision of skill '{}' ({trigger})", info.skill.name));
            }
            Ok(false) => lines.push(format!("skill '{}' reviewed ({trigger}): no change", info.skill.name)),
            Err(e) => lines.push(format!("skill '{}': no revision ({e})", info.skill.name)),
        }
    }
    let dups = learn::propose_memory_cleanup(studio, workspace);
    for d in &dups {
        lines.push(format!("proposed to remove a repeated memory entry: {}", learn::clip_chars(d, 100)));
    }
    if lines.is_empty() {
        lines.push("nothing to consolidate: no skill is due for review and the memory has no repeated entries".into());
    }
    save_last_run(studio, "consolidate", &lines);
    lines
}

/// A thumbs up or down for the skills used in a session. Returns how many skills were noted.
pub fn record_feedback(studio: &Path, workspace: Option<&Path>, session: &str, names: &[String], value: &str) -> Result<usize, String> {
    if value != "up" && value != "down" {
        return Err("feedback is up or down".into());
    }
    let installed = learn::list_skills(studio, workspace);
    let mut n = 0;
    for name in names {
        if let Some(s) = installed.iter().find(|s| &s.skill.name == name) {
            learn::record_usage(studio, &s.skill, Usage { feedback: Some(value.to_owned()), ..Usage::new(session, "feedback") })?;
            n += 1;
        }
    }
    Ok(n)
}

/// Per skill: use counts, thumbs, pending revision, versions. For Settings > Skills and `fllm-rs skills usage`.
pub fn skill_overview(studio: &Path, workspace: Option<&Path>) -> Vec<Value> {
    learn::list_skills(studio, workspace)
        .iter()
        .map(|i| {
            let s = &i.skill;
            let st = learn::stats_of(studio, s);
            let history = learn::read_history(studio, s);
            let versions: Vec<Value> = history.iter().filter(|h| h["version"].as_u64().unwrap_or(0) > 0).cloned().collect();
            json!({
                "name": s.name, "scope": s.scope, "workspace": s.workspace, "description": i.description,
                "uses": st.uses, "ok": st.ok, "failed": st.failed, "up": st.up, "down": st.down, "retries": st.retries,
                "window_uses": st.window_uses, "window_bad": st.window_bad, "last_used": st.last_t,
                "pending": learn::pending_revision(studio, s).is_some(),
                "due": learn::improve_due(studio, s),
                "versions": versions,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------
// skills for the command line agent (the Studio has its own skill tool in the page)
// ---------------------------------------------------------------------------------------------------------------

/// The `skill` tool of the headless agent, in the catalog format of the server (with its `permission`).
pub fn skill_tool_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "skill",
            "description": "Skills are saved step-by-step procedures. action \"list\" shows their names and descriptions, action \"read\" returns one skill by name. When the task matches a skill, read it first and follow it.",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["list", "read"]},
                "name": {"type": "string", "description": "The skill to read."}
            }, "required": ["action"]}
        },
        "permission": "read"
    })
}

/// Runs the `skill` tool. A skill of the project hides a skill of the user with the same name.
pub fn skill_tool(studio: &Path, workspace: Option<&Path>, args: &Value) -> Result<String, String> {
    let skills = learn::list_skills(studio, workspace);
    match args["action"].as_str().unwrap_or("list") {
        "list" => Ok(if skills.is_empty() {
            "No skills yet.".to_owned()
        } else {
            skills.iter().map(|s| format!("{}: {}{}", s.skill.name, s.description, if s.when.is_empty() { String::new() } else { format!(" (use when: {})", s.when) })).collect::<Vec<_>>().join("\n")
        }),
        "read" => {
            let name = args["name"].as_str().ok_or("'name' is missing")?;
            let s = skills.iter().find(|s| s.skill.name == name).ok_or_else(|| format!("no skill named '{name}'"))?;
            Ok(learn::clip_chars(&s.skill.read(studio)?, 20_000))
        }
        other => Err(format!("action must be list or read, not '{other}'")),
    }
}

/// The system prompt part that names the skills: one line each, capped; `matched` are marked as likely relevant.
pub fn skills_index(skills: &[SkillInfo], matched: &[String], max_chars: usize) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let mut out = String::from("Skills are saved step-by-step procedures. When the task matches one, call the skill tool with action \"read\" and its name first, then follow it. Available skills:\n");
    let mut shown = 0;
    for s in skills.iter().take(40) {
        let line = format!("- {}: {}{}\n", s.skill.name, learn::clip_chars(&s.description, 120), if s.when.is_empty() { String::new() } else { format!(" (use when: {})", learn::clip_chars(&s.when, 100)) });
        if out.len() + line.len() > max_chars {
            break;
        }
        out.push_str(&line);
        shown += 1;
    }
    if shown < skills.len() {
        out.push_str(&format!("({} more: call the skill tool with action \"list\")\n", skills.len() - shown));
    }
    if !matched.is_empty() {
        out.push_str(&format!("These look relevant to the current task, read them first: {}.\n", matched.join(", ")));
    }
    out.trim_end().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fllm-flow-{tag}-{}-{}", std::process::id(), learn::now()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn add_skill(studio: &Path, name: &str, body: &str) -> SkillRef {
        let dir = studio.join("skills").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), format!("---\nname: {name}\ndescription: Build the project\nwhen: the user asks to build\nlicense: MIT\n---\n{body}\n")).unwrap();
        SkillRef::user(name)
    }

    fn reply(text: &str) -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": text}}]})
    }

    /// A model that answers from a script and remembers the bodies it was sent.
    struct Script {
        answers: VecDeque<Result<Value, String>>,
        seen: Vec<Value>,
    }
    impl Script {
        fn new(answers: Vec<Result<Value, String>>) -> Script {
            Script { answers: answers.into(), seen: Vec::new() }
        }
        fn call(&mut self) -> impl FnMut(Value) -> Result<Value, String> + '_ {
            move |body| {
                self.seen.push(body);
                self.answers.pop_front().unwrap_or_else(|| Err("no more answers".into()))
            }
        }
    }

    fn session() -> Vec<Value> {
        vec![
            json!({"role": "user", "content": "build the project and run the tests"}),
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "skill", "arguments": "{\"action\":\"read\",\"name\":\"build\"}"}}, {"id": "2", "function": {"name": "bash", "arguments": "{\"command\":\"cargo test\"}"}}]}),
            json!({"role": "tool", "tool_call_id": "1", "content": "1. run make"}),
            json!({"role": "tool", "tool_call_id": "2", "content": "error: make is not installed, use cargo (key sk-abcdefghijklmnopqrstuvwx1234)"}),
            json!({"role": "assistant", "content": "Built with cargo build; tests pass with cargo test."}),
        ]
    }

    #[test]
    fn model_answers_are_read_leniently_but_validated() {
        assert!(extract_json("{\"a\":1}").is_some());
        assert!(extract_json("```json\n{\"a\":1}\n```").is_some());
        assert!(extract_json("Sure! Here you go: {\"a\": {\"b\": 2}} hope it helps").is_some());
        assert!(extract_json("no json here").is_none());
        assert!(extract_json("[1,2]").is_none());
        assert_eq!(reply_text(&reply("<think>hmm</think>{\"a\":1}")), "{\"a\":1}");
        assert_eq!(reply_text(&reply("{\"a\":1}")), "{\"a\":1}");
    }

    #[test]
    fn the_reflection_schema_and_parser_agree() {
        let schema = reflect_schema();
        assert_eq!(schema["required"], json!(["memory", "profile"]));
        assert_eq!(schema["properties"]["memory"]["items"]["required"], json!(["text", "kind", "scope"]));
        assert_eq!(schema["properties"]["memory"]["maxItems"], learn::MAX_NEW_PER_RUN);
        assert!(schema["properties"]["profile"]["items"]["properties"]["section"]["enum"].as_array().unwrap().iter().any(|s| s == "tool_habits"));
        let good = r#"{"memory":[{"text":"Tests run with cargo test","kind":"command","scope":"workspace"}],"profile":[{"section":"languages","text":"Uses Rust"}]}"#;
        let r = parse_reflection(good).unwrap();
        assert_eq!((r.memory.len(), r.profile.len(), r.dropped), (1, 1, 0));
        // unusable items are dropped, odd kinds and scopes fall back, an empty answer is fine
        let messy = r#"{"memory":[{"text":"  "},{"text":"ok fact here","kind":"weird","scope":"nowhere"},{"kind":"fact"}],"profile":[{"section":"nope","text":"x"},{"section":"preferences"}]}"#;
        let r = parse_reflection(messy).unwrap();
        assert_eq!(r.memory, vec![MemoryCandidate { text: "ok fact here".into(), kind: "fact".into(), scope: "workspace".into() }]);
        assert_eq!((r.profile.len(), r.dropped), (0, 4));
        assert!(parse_reflection(r#"{"memory":[],"profile":[]}"#).unwrap().memory.is_empty());
        assert!(parse_reflection("{\"other\":1}").is_err());
        assert!(parse_reflection("I think nothing is worth keeping.").is_err());
    }

    #[test]
    fn the_transcript_is_redacted_and_cut_in_the_middle() {
        let t = transcript(&session(), 6000, &[]);
        assert!(t.contains("USER: build the project") && t.contains("CALL bash(") && t.contains("RESULT: error: make is not installed"));
        assert!(!t.contains("sk-abcdefgh") && t.contains("redacted"), "{t}");
        let custom = transcript(&[json!({"role": "user", "content": "my key is zzz-very-private-value"})], 1000, &["zzz-very-private-value".to_owned()]);
        assert!(!custom.contains("zzz-very"), "{custom}");
        let long: Vec<Value> = (0..200).map(|i| json!({"role": "user", "content": format!("message number {i} with some words")})).collect();
        let cut = transcript(&long, 800, &[]);
        assert!(cut.contains("left out") && cut.contains("message number 199") && cut.contains("message number 0"));
        assert!(cut.chars().count() < 900);
        assert!(worth_reflecting(&session()));
        assert!(!worth_reflecting(&[json!({"role": "user", "content": "hi"}), json!({"role": "assistant", "content": "hello"})]));
    }

    #[test]
    fn the_improvement_keeps_name_and_extras_and_refuses_bad_answers() {
        let current = "---\nname: build\ndescription: Build the project\nwhen: the user asks to build\nlicense: MIT\n---\n1. run make\n2. read the output\n";
        let ok = json!({"change": "revise", "reason": "make is not installed here", "description": "Build with cargo", "when": "", "steps": "1. run `cargo build`\n2. run `cargo test`\n\n## Pitfalls\n- make is missing"}).to_string();
        let imp = parse_improvement(&ok, current).unwrap().unwrap();
        assert_eq!(imp.reason, "make is not installed here");
        assert!(imp.text.starts_with("---\nname: build\ndescription: Build with cargo\nwhen: the user asks to build\nlicense: MIT\n---\n1. run `cargo build`"), "{}", imp.text);
        let none = json!({"change": "none", "reason": "", "description": "", "when": "", "steps": ""}).to_string();
        assert_eq!(parse_improvement(&none, current).unwrap(), None);
        // a model that writes a whole file inside "steps", or fences it
        let whole = json!({"change": "revise", "reason": "r", "description": "", "when": "", "steps": "```markdown\n---\nname: other\ndescription: x\n---\n1. do the new thing properly\n```"}).to_string();
        let imp = parse_improvement(&whole, current).unwrap().unwrap();
        assert!(imp.text.contains("name: build") && imp.text.contains("1. do the new thing properly") && !imp.text.contains("name: other"), "{}", imp.text);
        for (bad, why) in [
            (json!({"change": "revise", "reason": "", "description": "", "when": "", "steps": "1. a long enough step list here"}), "no reason"),
            (json!({"change": "revise", "reason": "r", "description": "", "when": "", "steps": "tiny"}), "empty steps"),
            (json!({"change": "maybe", "reason": "r", "description": "", "when": "", "steps": "1. a long enough step list here"}), "bad change"),
            (json!({"change": "revise", "reason": "r", "description": "", "when": "", "steps": "1. export TOKEN=abc and password = hunter2hunter2 here"}), "secret"),
        ] {
            assert!(parse_improvement(&bad.to_string(), current).is_err(), "{why}");
        }
        assert!(parse_improvement("sorry, I cannot", current).is_err());
        // a revision that throws away most of a long skill is refused
        let long = format!("---\nname: build\ndescription: d\n---\n{}\n", "1. a step with some detail in it\n".repeat(20));
        let short = json!({"change": "revise", "reason": "r", "description": "", "when": "", "steps": "1. only one step remains here"}).to_string();
        assert!(parse_improvement(&short, &long).unwrap_err().contains("drop most"));
        // the same text as before is no revision
        let same = json!({"change": "revise", "reason": "r", "description": "Build the project", "when": "the user asks to build", "steps": "1. run make\n2. read the output"}).to_string();
        assert_eq!(parse_improvement(&same, current).unwrap(), None);
    }

    #[test]
    fn a_server_without_structured_output_is_asked_again_without_it() {
        let mut s = Script::new(vec![Err("/v1/chat/completions: HTTP 400 Bad Request: unknown field response_format".into()), Ok(reply("{\"memory\":[],\"profile\":[]}"))]);
        let out = structured_call(&mut s.call(), reflect_body("x", &[], "", 100)).unwrap();
        assert!(out.contains("memory"));
        assert!(s.seen[0].get("response_format").is_some() && s.seen[1].get("response_format").is_none());
        assert_eq!(s.seen[1]["chat_template_kwargs"]["enable_thinking"], false);
        // another error is not retried; an empty answer is an error
        let mut s = Script::new(vec![Err("cannot reach the server".into())]);
        assert!(structured_call(&mut s.call(), reflect_body("x", &[], "", 100)).is_err());
        assert_eq!(s.seen.len(), 1);
        let mut s = Script::new(vec![Ok(reply("  "))]);
        assert!(structured_call(&mut s.call(), reflect_body("x", &[], "", 100)).is_err());
        // the request carries the caps and the schema
        let b = reflect_body("t", &["old fact".into()], "profile text", REFLECT_TOKENS);
        assert_eq!(b["max_tokens"], REFLECT_TOKENS);
        assert_eq!(b["response_format"]["type"], "json_schema");
        assert!(b["messages"][1]["content"].as_str().unwrap().contains("old fact") && b["messages"][1]["content"].as_str().unwrap().contains("profile text"));
    }

    #[test]
    fn after_a_run_the_loop_notes_the_use_drafts_a_revision_and_proposes_memory() {
        let studio = temp("after");
        let ws = temp("afterws");
        let skill = add_skill(&studio, "build", "1. run make\n2. read the output");
        let original = skill.read(&studio).unwrap();
        let reflection = json!({"memory": [{"text": "Use cargo, make is not installed on this machine", "kind": "mistake", "scope": "workspace"}, {"text": "Hunter key sk-abcdefghijklmnopqrstuvwx1234 for staging", "kind": "fact", "scope": "global"}],
                                "profile": [{"section": "languages", "text": "Works in Rust projects"}]})
        .to_string();
        let revision = json!({"change": "revise", "reason": "make is missing, cargo works", "description": "Build with cargo", "when": "", "steps": "1. run cargo build\n2. run cargo test"}).to_string();
        let info = RunInfo { session: "run-1".into(), workspace: Some(ws.clone()), outcome: "failed".into(), verify: Some(false), retries: 2, note: "make not installed".into(), skills: vec!["build".into(), "ghost".into()], matched: vec!["build".into()], feedback: None };
        // the failure makes the revision due at once; the model is called for the revision first, then for the reflection
        let mut s = Script::new(vec![Ok(reply(&revision)), Ok(reply(&reflection))]);
        let lines = after_run(&studio, &info, &session(), &Options::default(), &mut s.call());
        assert_eq!(s.seen.len(), 2);
        assert!(lines.iter().any(|l| l.contains("skill 'build' used")) && lines.iter().any(|l| l.contains("drafted a revision")), "{lines:?}");
        assert_eq!(skill.read(&studio).unwrap(), original, "the skill file is not touched");
        let rev = learn::pending_revision(&studio, &skill).unwrap();
        assert!(rev.new.contains("cargo build") && rev.reason.contains("make is missing"));
        let u = &learn::read_usage(&studio, &skill)[0];
        assert!(u.matched && u.verify == Some(false) && u.retries == 2 && u.outcome == "failed");
        // the transcript the model saw has no key; the memory has the safe entry only, as a proposal
        for body in &s.seen {
            assert!(!body.to_string().contains("sk-abcdefgh"));
        }
        let p = learn::memory_proposals(&studio);
        assert_eq!(p.len(), 1, "{lines:?}");
        assert!(p[0].text.contains("cargo") && p[0].scope == "workspace" && p[0].session == "run-1");
        assert_eq!(learn::profile_proposals(&studio).len(), 1);
        assert!(learn::read_memory(&studio, "workspace", Some(&ws)).is_empty() && learn::read_profile(&studio).is_empty());
        assert!(lines.iter().any(|l| l.contains("dropped")), "the secret entry is reported as dropped: {lines:?}");
        // the status file carries the lines
        assert!(std::fs::read_to_string(studio.join("learn/last-run.json")).unwrap().contains("run-1"));

        // with auto memory the same reflection is written at once
        let info2 = RunInfo { session: "run-2".into(), skills: vec![], ..info.clone() };
        let other = json!({"memory": [{"text": "The release build lives in target/release", "kind": "project", "scope": "workspace"}], "profile": []}).to_string();
        let mut s = Script::new(vec![Ok(reply(&other))]);
        after_run(&studio, &info2, &session(), &Options { auto_memory: true, ..Options::default() }, &mut s.call());
        assert!(learn::read_memory(&studio, "workspace", Some(&ws)).iter().any(|e| e.contains("target/release")));
        // a model that cannot answer is reported and nothing breaks
        let mut s = Script::new(vec![Ok(reply("I do not know"))]);
        let lines = after_run(&studio, &info2, &session(), &Options::default(), &mut s.call());
        assert!(lines.iter().any(|l| l.contains("reflection skipped")), "{lines:?}");
        let mut s = Script::new(vec![Err("cannot reach the server".into())]);
        let lines = after_run(&studio, &info2, &session(), &Options::default(), &mut s.call());
        assert!(lines.iter().any(|l| l.contains("reflection skipped")));
        // switches: neither step runs when off
        let mut s = Script::new(vec![]);
        after_run(&studio, &info, &session(), &Options { reflect: false, improve: false, ..Options::default() }, &mut s.call());
        assert!(s.seen.is_empty());
        let _ = std::fs::remove_dir_all(&studio);
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn consolidation_drafts_due_revisions_and_proposes_cleanups_only() {
        let studio = temp("cons");
        let skill = add_skill(&studio, "build", "1. run make\n2. read the output");
        std::fs::write(studio.join("agent-memory.md"), "# Memory\n\n- Alpha convention for naming things\n- Alpha convention for naming things!\n").unwrap();
        let mut s = Script::new(vec![]);
        let lines = consolidate(&studio, None, &mut s.call());
        assert!(s.seen.is_empty() && lines.iter().any(|l| l.contains("repeated memory entry")), "{lines:?}");
        learn::record_usage(&studio, &skill, Usage { feedback: Some("down".into()), ..Usage::new("x", "feedback") }).unwrap();
        let rev = json!({"change": "revise", "reason": "users said it failed", "description": "", "when": "", "steps": "1. run cargo build instead\n2. read the output"}).to_string();
        let mut s = Script::new(vec![Ok(reply(&rev))]);
        let lines = consolidate(&studio, None, &mut s.call());
        assert!(lines.iter().any(|l| l.contains("drafted a revision")), "{lines:?}");
        assert!(learn::pending_revision(&studio, &skill).is_some());
        assert_eq!(learn::read_memory(&studio, "global", None).len(), 2, "nothing was removed without approval");
        let _ = std::fs::remove_dir_all(&studio);
    }

    #[test]
    fn feedback_and_the_overview_count_per_skill() {
        let studio = temp("fb");
        let skill = add_skill(&studio, "build", "1. run make");
        let info = RunInfo { session: "a".into(), outcome: "done".into(), skills: vec!["build".into()], ..RunInfo::default() };
        record_skill_use(&studio, &info);
        assert_eq!(record_feedback(&studio, None, "a", &["build".into(), "nope".into()], "up").unwrap(), 1);
        assert_eq!(record_feedback(&studio, None, "b", &["build".into()], "down").unwrap(), 1);
        assert!(record_feedback(&studio, None, "b", &["build".into()], "meh").is_err());
        let o = skill_overview(&studio, None);
        assert_eq!((o[0]["uses"].as_u64(), o[0]["up"].as_u64(), o[0]["down"].as_u64()), (Some(1), Some(1), Some(1)));
        assert!(o[0]["due"].is_string(), "a thumb down makes a review due");
        learn::propose_revision(&studio, &skill, "---\nname: build\ndescription: x\n---\n1. new steps here for sure\n", "r", "t").unwrap();
        learn::approve_revision(&studio, &skill, false).unwrap();
        let o = skill_overview(&studio, None);
        assert_eq!(o[0]["versions"].as_array().unwrap().len(), 1);
        assert!(o[0]["due"].is_null());
        let _ = std::fs::remove_dir_all(&studio);
    }

    #[test]
    fn the_skill_tool_lists_reads_and_the_index_is_capped() {
        let studio = temp("tool");
        add_skill(&studio, "build", "1. run make");
        add_skill(&studio, "deploy", "1. push");
        let list = skill_tool(&studio, None, &json!({"action": "list"})).unwrap();
        assert!(list.contains("build: Build the project (use when: the user asks to build)") && list.contains("deploy"));
        assert!(skill_tool(&studio, None, &json!({"action": "read", "name": "build"})).unwrap().contains("1. run make"));
        assert!(skill_tool(&studio, None, &json!({"action": "read", "name": "nope"})).is_err());
        assert!(skill_tool(&studio, None, &json!({"action": "delete", "name": "build"})).is_err());
        assert_eq!(skill_tool_schema()["permission"], "read");
        let skills = learn::list_skills(&studio, None);
        let idx = skills_index(&skills, &["build".into()], 1800);
        assert!(idx.contains("- build:") && idx.contains("read them first: build"));
        let tiny = skills_index(&skills, &[], 250);
        assert!(tiny.contains("more: call the skill tool"), "{tiny}");
        assert_eq!(skills_index(&[], &[], 1800), "");
        let _ = std::fs::remove_dir_all(&studio);
    }
}
