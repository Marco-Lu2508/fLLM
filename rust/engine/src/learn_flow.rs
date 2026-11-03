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
pub const CREATE_TOKENS: usize = 1200;
/// A run with at least this many tool calls is complex enough to be worth a skill.
pub const CREATE_MIN_CALLS: usize = 6;
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
                if !t.trim().is_empty() && !is_injected(&t) {
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

pub fn reflect_body(transcript: &str, user_shown: &str, known_memory: &[String], profile: &str, max_tokens: usize) -> Value {
    let mut known = String::new();
    for e in known_memory.iter().rev().take(20) {
        let line = format!("- {}\n", learn::clip_chars(e, 100));
        if known.len() + line.len() > 1500 {
            break;
        }
        known.push_str(&line);
    }
    let user = format!(
        "Session:\n{transcript}\n\nMessages the user wrote (the only source for the profile):\n{user_shown}\n\nAlready remembered (do not repeat):\n{}\nProfile of the user so far:\n{}\n\n\
         Task: list what is worth keeping for FUTURE sessions in \"memory\": how to build, test and run this project (exact commands that worked), conventions of the project, \
         preferences the user stated, mistakes or traps that cost time here. Rules: at most {} entries, each one short sentence that is true and useful without this session; \
         no secrets, passwords or keys. Do NOT describe what files contain or what the code does, do not retell the task, and do not note general programming advice. \
         Good: \"Tests run with node test.js.\" Bad: \"math.js has two functions.\" \
         \"scope\" is \"workspace\" for facts about this project and \"global\" for facts about the user. \
         In \"profile\" list durable facts about the USER, taken only from the block of messages the user wrote, never from what the agent or the tools did: \
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
    /// Skills, memory entries and profile lines are written at once (checked, deduped, redacted, capped and logged with an undo).
    /// `false`: they are proposals that wait for a yes.
    pub auto: bool,
    pub reflect: bool,
    pub improve: bool,
    /// Write a skill after a complex, successful run that used none.
    pub create: bool,
    pub transcript_chars: usize,
    /// Literal secrets (the API key of this run) to cut out of everything the model reads.
    pub secrets: Vec<String>,
}

impl Default for Options {
    fn default() -> Self {
        Options { auto: true, reflect: true, improve: true, create: true, transcript_chars: 6000, secrets: Vec::new() }
    }
}

/// Writes the usage of the skills of a run and lets the watch after an automatic revision judge it. Returns the skills that were
/// found and the lines about revisions that were taken back.
pub fn record_skill_use_notes(studio: &Path, info: &RunInfo) -> (Vec<SkillRef>, Vec<String>) {
    let installed = learn::list_skills(studio, info.workspace.as_deref());
    let (mut found, mut notes) = (Vec::new(), Vec::new());
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
        if let Ok(rec) = learn::record_usage(studio, &s.skill, u) {
            found.push(s.skill.clone());
            notes.extend(learn::judge_use(studio, &s.skill, &rec));
        }
    }
    (found, notes)
}

pub fn record_skill_use(studio: &Path, info: &RunInfo) -> Vec<SkillRef> {
    record_skill_use_notes(studio, info).0
}

/// What the user wrote in a session, word for word: no tool output, no text the loop injected (review feedback, nudges).
pub fn user_text(messages: &[Value]) -> String {
    messages
        .iter()
        .filter(|m| m["role"] == "user")
        .map(content_text)
        .filter(|t| !t.trim().is_empty() && !is_injected(t))
        .collect::<Vec<_>>()
        .join("\n")
}

const INJECTED: [&str; 6] = ["[Summary", "Self-review before you finish", "You described the next step", "Your last tool call was refused", "The plan is approved", "Stop using tools now"];

fn is_injected(text: &str) -> bool {
    let t = text.trim_start();
    INJECTED.iter().any(|p| t.starts_with(p))
}

/// Whether a profile line rests on the user's own words: at least two of its words, and 40 percent of them, appear in what the user wrote.
pub fn grounded(candidate: &str, user: &str) -> bool {
    let c = learn::tokens(candidate);
    let u: std::collections::BTreeSet<String> = learn::tokens(user).into_iter().collect();
    let hit = c.iter().filter(|w| u.contains(*w)).count();
    c.len() >= 2 && hit >= 2 && hit * 10 >= c.len() * 4
}

/// The memory/profile reflection of one session: asks the model, validates, dedups, stores (or proposes, when `opts.auto` is off).
/// Profile lines are kept only when they rest on what the user wrote. `Err` means nothing was stored.
pub fn reflect(studio: &Path, workspace: Option<&Path>, session: &str, messages: &[Value], opts: &Options, complete: Complete) -> Result<(Ingest, Ingest), String> {
    if !worth_reflecting(messages) {
        return Err("the session is too short to learn from".into());
    }
    let mut known = learn::read_memory(studio, "global", None);
    if let Some(w) = workspace {
        known.extend(learn::read_memory(studio, "workspace", Some(w)));
    }
    let user = user_text(messages);
    let shown = learn::clip_chars(&learn::redact_with(&user, &opts.secrets), 1500);
    let body = reflect_body(&transcript(messages, opts.transcript_chars, &opts.secrets), &shown, &known, &learn::clip_chars(&learn::read_profile(studio), 800), REFLECT_TOKENS);
    let answer = structured_call(complete, body)?;
    let mut r = parse_reflection(&answer)?;
    let ungrounded: Vec<ProfileCandidate> = r.profile.iter().filter(|(_, t)| !grounded(t, &user)).cloned().collect();
    r.profile.retain(|(_, t)| grounded(t, &user));
    let memory = learn::ingest_memory(studio, workspace, session, &r.memory, opts.auto);
    let mut profile = learn::ingest_profile(studio, session, &r.profile, opts.auto);
    for (_, t) in ungrounded {
        profile.skipped.push((t, "not taken from the user's own words".into()));
    }
    if opts.auto {
        learn::log_ingest(studio, session, "reflection after the session", &memory.meta);
        learn::log_ingest(studio, session, "taken from the user's own messages", &profile.meta);
    }
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

/// The number of tool calls in a session.
pub fn count_calls(messages: &[Value]) -> usize {
    messages.iter().map(|m| m["tool_calls"].as_array().map_or(0, Vec::len)).sum()
}

/// A complex run that went well and used no skill is worth a skill of its own.
pub fn should_create(info: &RunInfo, used: &[SkillRef], messages: &[Value]) -> bool {
    info.outcome == "done" && info.verify != Some(false) && info.feedback.as_deref() != Some("down") && used.is_empty() && count_calls(messages) >= CREATE_MIN_CALLS
}

pub fn create_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "save": {"type": "boolean"},
            "name": {"type": "string", "maxLength": 48},
            "description": {"type": "string", "maxLength": 200},
            "when": {"type": "string", "maxLength": 200},
            "steps": {"type": "string", "maxLength": 4000},
            "pitfalls": {"type": "string", "maxLength": 1000}
        },
        "required": ["save", "name", "description", "when", "steps", "pitfalls"],
        "additionalProperties": false
    })
}

const CREATE_SYSTEM: &str = "You turn a successful coding session into a reusable skill: a short step-by-step procedure that a coding agent can follow next time. You answer with one JSON object and nothing else.";

pub fn create_body(transcript: &str, existing: &[String], max_tokens: usize) -> Value {
    let user = format!(
        "Session:\n{transcript}\n\nSkills that exist already (do not duplicate): {}\n\n\
         Task: if the way this task was solved is a repeatable procedure that would save work next time (several steps, exact commands, a trap that was hit), set \"save\" to true and write the skill: \
         \"name\" is a short slug (lowercase letters, digits, - and _), \"description\" one sentence on what it does, \"when\" one sentence on when to use it, \
         \"steps\" a numbered list with exact commands that worked, \"pitfalls\" the traps found (may be empty). No secrets, no paths that exist only on one computer. \
         If the task was a one-off, or it is covered by an existing skill, set \"save\" to false and leave the other fields empty.\n\
         Answer as JSON: {{\"save\":true|false,\"name\":\"...\",\"description\":\"...\",\"when\":\"...\",\"steps\":\"...\",\"pitfalls\":\"...\"}}",
        if existing.is_empty() { "(none)".to_owned() } else { existing.join(", ") },
    );
    request(CREATE_SYSTEM, &user, max_tokens, "skill_creation", create_schema())
}

/// Validates the answer and builds the SKILL.md. `Ok(None)`: the model found nothing worth a skill.
pub fn parse_creation(text: &str, existing: &[String]) -> Result<Option<(String, String)>, String> {
    let v = extract_json(text).ok_or("the model did not answer with a JSON object")?;
    if v["save"].as_bool() != Some(true) {
        return Ok(None);
    }
    let name = crate::skills_cli::local_name(v["name"].as_str().unwrap_or_default());
    if !crate::agent_cli::valid_session_name(&name) {
        return Err("the skill has no usable name".into());
    }
    if existing.contains(&name) {
        return Err(format!("a skill named '{name}' exists already"));
    }
    let one = |k: &str| v[k].as_str().unwrap_or_default().split_whitespace().collect::<Vec<_>>().join(" ");
    let (description, when) = (one("description"), one("when"));
    if description.is_empty() || when.is_empty() {
        return Err("the skill has no description or no 'when' line".into());
    }
    let steps = strip_fence(v["steps"].as_str().unwrap_or_default()).to_owned();
    if steps.chars().count() < 20 || steps.lines().filter(|l| !l.trim().is_empty()).count() < 2 {
        return Err("the steps are too short to be a procedure".into());
    }
    let mut file = format!("---\nname: {name}\ndescription: {description}\nwhen: {when}\n---\n{}\n", steps.trim_end());
    let pitfalls = v["pitfalls"].as_str().unwrap_or_default().trim();
    if !pitfalls.is_empty() {
        file.push_str(&format!("\n## Pitfalls\n{pitfalls}\n"));
    }
    if file.chars().count() > MAX_SKILL_BODY_CHARS {
        return Err("the skill is too long".into());
    }
    if learn::has_secret(&file) {
        return Err("the skill contains something that looks like a secret".into());
    }
    Ok(Some((name, file)))
}

/// Asks the model for a skill after a complex run and writes it (`opts.auto`) or leaves it as a proposal. `Ok(None)`: nothing worth saving.
pub fn create_skill(studio: &Path, workspace: Option<&Path>, session: &str, messages: &[Value], opts: &Options, complete: Complete) -> Result<Option<String>, String> {
    let existing: Vec<String> = learn::list_skills(studio, workspace).iter().map(|s| s.skill.name.clone()).collect();
    let body = create_body(&transcript(messages, opts.transcript_chars, &opts.secrets), &existing, CREATE_TOKENS);
    let answer = structured_call(complete, body)?;
    let Some((name, file)) = parse_creation(&answer, &existing)? else { return Ok(None) };
    if opts.auto {
        learn::create_skill_file(studio, &SkillRef::user(&name), &file, "written after a complex session that went well", session)?;
    } else {
        crate::agent_extras::save_skill(studio, &name, &file, false)?;
    }
    Ok(Some(name))
}

pub fn check_schema() -> Value {
    json!({"type": "object", "properties": {"ok": {"type": "boolean"}, "problem": {"type": "string", "maxLength": 300}}, "required": ["ok", "problem"], "additionalProperties": false})
}

pub fn check_body(old: &str, new: &str, reason: &str) -> Value {
    let user = format!(
        "Old skill:\n{}\n\nNew skill:\n{}\n\nWhy it was changed: {reason}\n\n\
         Task: answer ok=true only if the new skill still does the same job as the old one, keeps every command and step of the old one that still works, \
         addresses the stated reason, and does not contradict itself. Otherwise ok=false and say the problem in one sentence.\n\
         Answer as JSON: {{\"ok\":true|false,\"problem\":\"...\"}}",
        learn::clip_chars(old, 3500),
        learn::clip_chars(new, 3500),
    );
    request("You check a revised skill before it replaces the old one. You answer with one JSON object and nothing else.", &user, 200, "skill_check", check_schema())
}

/// `(ok, problem)`; an answer that is not the object is an error (and the revision is not applied).
pub fn parse_check(text: &str) -> Result<(bool, String), String> {
    let v = extract_json(text).ok_or("the model did not answer with a JSON object")?;
    let ok = v["ok"].as_bool().ok_or("the check has no \"ok\"")?;
    Ok((ok, v["problem"].as_str().unwrap_or_default().to_owned()))
}

#[derive(Debug, PartialEq, Eq)]
pub enum AutoOutcome {
    Applied,
    /// The skill is not the agent's; the revision was written as this sibling skill.
    Sibling(String),
    NoChange,
    Skipped(String),
}

/// The text of `new_text` as the skill `name`, with a description that says where it came from.
fn as_sibling(new_text: &str, original: &str, name: &str) -> String {
    let Some((head, body)) = crate::skills_cli::split(new_text) else { return new_text.to_owned() };
    let mut lines = Vec::new();
    for l in head {
        if l.starts_with("name:") {
            lines.push(format!("name: {name}"));
        } else if let Some(d) = l.strip_prefix("description:") {
            lines.push(format!("description: Revised copy of '{original}' by the agent; your original is unchanged. {}", d.trim()));
        } else {
            lines.push(l);
        }
    }
    format!("---\n{}\n---\n{}", lines.join("\n"), body)
}

/// Drafts a revision of a skill and applies it, after the checks: the text is valid (front matter, size, most of the old content kept,
/// no secret), and a second call of the model confirms that it still does the same job. A skill that a person wrote or edited is not
/// overwritten: the revision becomes a sibling skill. The revision is put on watch (see `learn::judge_use`).
pub fn improve_skill_auto(studio: &Path, skill: &SkillRef, trigger: &str, session_text: &str, session: &str, complete: Complete) -> Result<AutoOutcome, String> {
    let current = skill.read(studio)?;
    let usage = learn::read_usage(studio, skill);
    let body = improve_body(&current, &usage, session_text, trigger, IMPROVE_TOKENS);
    let answer = structured_call(complete, body)?;
    let Some(imp) = parse_improvement(&answer, &current)? else { return Ok(AutoOutcome::NoChange) };
    match structured_call(complete, check_body(&current, &imp.text, &imp.reason)).and_then(|t| parse_check(&t)) {
        Ok((true, _)) => {}
        Ok((false, problem)) => return Ok(AutoOutcome::Skipped(format!("the self-check rejected the revision: {}", learn::clip_chars(&problem, 120)))),
        Err(e) => return Ok(AutoOutcome::Skipped(format!("the self-check gave no usable answer ({e})"))),
    }
    match learn::apply_revision_auto(studio, skill, &imp.text, &imp.reason, trigger, session)? {
        learn::AutoApply::Applied(_) => Ok(AutoOutcome::Applied),
        learn::AutoApply::NotOwned => {
            let sib = learn::sibling_ref(skill);
            let text = as_sibling(&imp.text, &skill.name, &sib.name);
            if sib.file(studio).is_file() {
                match learn::apply_revision_auto(studio, &sib, &text, &imp.reason, trigger, session)? {
                    learn::AutoApply::Applied(_) => Ok(AutoOutcome::Sibling(sib.name)),
                    learn::AutoApply::NotOwned => Ok(AutoOutcome::Skipped(format!("'{}' was edited by hand; nothing was written", sib.name))),
                }
            } else {
                learn::create_skill_file(studio, &sib, &text, &format!("revision of '{}' (not changed because it is yours): {}", skill.name, imp.reason), session)?;
                Ok(AutoOutcome::Sibling(sib.name))
            }
        }
    }
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// Everything that follows a session: usage notes for the skills it used (and the watch over revised skills), revisions that are due,
/// a new skill after a complex run, and the reflection on memory and profile. With `opts.auto` the results are written (and logged with
/// an undo), else they wait as proposals. Failures of a step are reported as a line and never stop the others.
pub fn after_run(studio: &Path, info: &RunInfo, messages: &[Value], opts: &Options, complete: Complete) -> Vec<String> {
    let mut lines = Vec::new();
    let (used, notes) = record_skill_use_notes(studio, info);
    let mut learned: Vec<String> = Vec::new();
    for s in &used {
        lines.push(format!("[learn] skill '{}' used ({})", s.name, info.outcome));
    }
    for n in notes {
        learned.push(n.clone());
        lines.push(format!("[learn] {n}"));
    }
    let excerpt = transcript(messages, opts.transcript_chars, &opts.secrets);
    if opts.improve {
        let mut drafted = 0;
        for s in &used {
            if drafted >= MAX_REVISIONS_PER_RUN {
                break;
            }
            let Some(trigger) = learn::improve_due(studio, s) else { continue };
            if opts.auto {
                match improve_skill_auto(studio, s, &trigger, &excerpt, &info.session, complete) {
                    Ok(AutoOutcome::Applied) => {
                        drafted += 1;
                        learned.push(format!("updated skill '{}'", s.name));
                        lines.push(format!("[learn] revised skill '{}' ({trigger}); the previous text is kept, `fllm-rs learn log` can undo it", s.name));
                    }
                    Ok(AutoOutcome::Sibling(name)) => {
                        drafted += 1;
                        learned.push(format!("wrote '{name}' as a revision of your skill '{}'", s.name));
                        lines.push(format!("[learn] '{}' is yours, so the revision was written as the skill '{name}'", s.name));
                    }
                    Ok(AutoOutcome::NoChange) => lines.push(format!("[learn] skill '{}' reviewed ({trigger}): no change", s.name)),
                    Ok(AutoOutcome::Skipped(why)) => lines.push(format!("[learn] skill '{}': revision not applied ({why})", s.name)),
                    Err(e) => lines.push(format!("[learn] skill '{}': no revision ({e})", s.name)),
                }
            } else {
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
    }
    if opts.create && should_create(info, &used, messages) {
        match create_skill(studio, info.workspace.as_deref(), &info.session, messages, opts, complete) {
            Ok(Some(name)) if opts.auto => {
                learned.push(format!("created skill '{name}'"));
                lines.push(format!("[learn] wrote the skill '{name}' after a complex session (fllm-rs learn log can undo it)"));
            }
            Ok(Some(name)) => lines.push(format!("[learn] proposed the skill '{name}' (fllm-rs skills approve {name})")),
            Ok(None) => lines.push("[learn] no skill worth saving from this session".into()),
            Err(e) => lines.push(format!("[learn] no skill written ({e})")),
        }
    }
    if opts.reflect {
        match reflect(studio, info.workspace.as_deref(), &info.session, messages, opts, complete) {
            Ok((m, p)) => {
                lines.extend(describe("memory", &m));
                lines.extend(describe("profile", &p));
                if !m.added.is_empty() {
                    learned.push(format!("saved {}", plural(m.added.len(), "note", "notes")));
                }
                if !p.added.is_empty() {
                    learned.push(format!("added {} to your profile", plural(p.added.len(), "line", "lines")));
                }
                let skipped = m.skipped.len() + p.skipped.len();
                if skipped > 0 {
                    lines.push(format!("[learn] {skipped} suggestion(s) dropped (duplicates, too short, secrets, not from your words, caps)"));
                }
                if m.added.is_empty() && m.proposed.is_empty() && p.added.is_empty() && p.proposed.is_empty() {
                    lines.push("[learn] nothing new worth remembering".into());
                }
            }
            Err(e) => lines.push(format!("[learn] reflection skipped: {e}")),
        }
    }
    if !learned.is_empty() {
        lines.push(format!("[learn] Learned: {}.", learned.join(", ")));
    }
    save_last_run(studio, &info.session, &lines);
    lines
}

pub fn save_last_run(studio: &Path, session: &str, lines: &[String]) {
    let file = learn::learn_dir(studio).join("last-run.json");
    let _ = std::fs::create_dir_all(learn::learn_dir(studio));
    let summary = lines.iter().find_map(|l| l.strip_prefix("[learn] Learned: ")).unwrap_or_default();
    let _ = std::fs::write(file, json!({"t": learn::now(), "session": session, "lines": lines, "summary": summary}).to_string());
}

/// The nightly job: revise the skills that are due and remove memory entries that repeat an earlier one. With `auto` that is done and
/// logged (undo from the log); without, drafts and proposals wait for a yes.
pub fn consolidate(studio: &Path, workspace: Option<&Path>, auto: bool, complete: Complete) -> Vec<String> {
    let mut lines = Vec::new();
    let mut drafted = 0;
    for info in learn::list_skills(studio, workspace) {
        if drafted >= 3 {
            break;
        }
        let Some(trigger) = learn::improve_due(studio, &info.skill) else { continue };
        let name = &info.skill.name;
        if auto {
            match improve_skill_auto(studio, &info.skill, &trigger, "", "consolidate", complete) {
                Ok(AutoOutcome::Applied) => {
                    drafted += 1;
                    lines.push(format!("revised skill '{name}' ({trigger})"));
                }
                Ok(AutoOutcome::Sibling(sib)) => {
                    drafted += 1;
                    lines.push(format!("'{name}' is yours; wrote the revision as the skill '{sib}'"));
                }
                Ok(AutoOutcome::NoChange) => lines.push(format!("skill '{name}' reviewed ({trigger}): no change")),
                Ok(AutoOutcome::Skipped(why)) => lines.push(format!("skill '{name}': revision not applied ({why})")),
                Err(e) => lines.push(format!("skill '{name}': no revision ({e})")),
            }
        } else {
            match improve_skill(studio, &info.skill, &trigger, "", complete) {
                Ok(true) => {
                    drafted += 1;
                    lines.push(format!("drafted a revision of skill '{name}' ({trigger})"));
                }
                Ok(false) => lines.push(format!("skill '{name}' reviewed ({trigger}): no change")),
                Err(e) => lines.push(format!("skill '{name}': no revision ({e})")),
            }
        }
    }
    let dups = learn::propose_memory_cleanup_items(studio, workspace);
    for d in &dups {
        if auto {
            if learn::decide_memory(studio, &d.id, true).is_ok() {
                learn::log_change(studio, "memory_remove", json!({"scope": d.scope, "workspace": d.workspace, "text": d.text, "session": "consolidate", "reason": d.reason}), None, None);
                lines.push(format!("removed a repeated memory entry: {}", learn::clip_chars(&d.text, 100)));
            }
        } else {
            lines.push(format!("proposed to remove a repeated memory entry: {}", learn::clip_chars(&d.text, 100)));
        }
    }
    if lines.is_empty() {
        lines.push("nothing to consolidate: no skill is due for review and the memory has no repeated entries".into());
    }
    save_last_run(studio, "consolidate", &lines);
    lines
}

/// A thumbs up or down for the skills used in a session. Returns how many skills were noted.
pub fn record_feedback(studio: &Path, workspace: Option<&Path>, session: &str, names: &[String], value: &str) -> Result<usize, String> {
    record_feedback_notes(studio, workspace, session, names, value).map(|r| r.0)
}

/// Like [`record_feedback`], and the lines about revisions that a thumb down took back.
pub fn record_feedback_notes(studio: &Path, workspace: Option<&Path>, session: &str, names: &[String], value: &str) -> Result<(usize, Vec<String>), String> {
    if value != "up" && value != "down" {
        return Err("feedback is up or down".into());
    }
    let installed = learn::list_skills(studio, workspace);
    let (mut n, mut notes) = (0, Vec::new());
    for name in names {
        if let Some(s) = installed.iter().find(|s| &s.skill.name == name) {
            let rec = learn::record_usage(studio, &s.skill, Usage { feedback: Some(value.to_owned()), ..Usage::new(session, "feedback") })?;
            notes.extend(learn::judge_use(studio, &s.skill, &rec));
            n += 1;
        }
    }
    Ok((n, notes))
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
        let out = structured_call(&mut s.call(), reflect_body("x", "", &[], "", 100)).unwrap();
        assert!(out.contains("memory"));
        assert!(s.seen[0].get("response_format").is_some() && s.seen[1].get("response_format").is_none());
        assert_eq!(s.seen[1]["chat_template_kwargs"]["enable_thinking"], false);
        // another error is not retried; an empty answer is an error
        let mut s = Script::new(vec![Err("cannot reach the server".into())]);
        assert!(structured_call(&mut s.call(), reflect_body("x", "", &[], "", 100)).is_err());
        assert_eq!(s.seen.len(), 1);
        let mut s = Script::new(vec![Ok(reply("  "))]);
        assert!(structured_call(&mut s.call(), reflect_body("x", "", &[], "", 100)).is_err());
        // the request carries the caps and the schema
        let b = reflect_body("t", "u", &["old fact".into()], "profile text", REFLECT_TOKENS);
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
                                "profile": [{"section": "preferences", "text": "Wants the project built and the tests run"}]})
        .to_string();
        let revision = json!({"change": "revise", "reason": "make is missing, cargo works", "description": "Build with cargo", "when": "", "steps": "1. run cargo build\n2. run cargo test"}).to_string();
        let info = RunInfo { session: "run-1".into(), workspace: Some(ws.clone()), outcome: "failed".into(), verify: Some(false), retries: 2, note: "make not installed".into(), skills: vec!["build".into(), "ghost".into()], matched: vec!["build".into()], feedback: None };
        // the failure makes the revision due at once; the model is called for the revision first, then for the reflection
        let mut s = Script::new(vec![Ok(reply(&revision)), Ok(reply(&reflection))]);
        let ask = Options { auto: false, ..Options::default() };
        let lines = after_run(&studio, &info, &session(), &ask, &mut s.call());
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
        after_run(&studio, &info2, &session(), &Options { auto: true, ..Options::default() }, &mut s.call());
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
        let lines = consolidate(&studio, None, false, &mut s.call());
        assert!(s.seen.is_empty() && lines.iter().any(|l| l.contains("repeated memory entry")), "{lines:?}");
        learn::record_usage(&studio, &skill, Usage { feedback: Some("down".into()), ..Usage::new("x", "feedback") }).unwrap();
        let rev = json!({"change": "revise", "reason": "users said it failed", "description": "", "when": "", "steps": "1. run cargo build instead\n2. read the output"}).to_string();
        let mut s = Script::new(vec![Ok(reply(&rev))]);
        let lines = consolidate(&studio, None, false, &mut s.call());
        assert!(lines.iter().any(|l| l.contains("drafted a revision")), "{lines:?}");
        assert!(learn::pending_revision(&studio, &skill).is_some());
        assert_eq!(learn::read_memory(&studio, "global", None).len(), 2, "nothing was removed without approval");
        let _ = std::fs::remove_dir_all(&studio);
    }

    fn owned_skill(studio: &Path, name: &str, body: &str) -> SkillRef {
        let s = add_skill(studio, name, body);
        learn::mark_agent_written(studio, &s, &s.read(studio).unwrap()).unwrap();
        s
    }

    fn revision_answer(steps: &str) -> Result<Value, String> {
        Ok(reply(&json!({"change": "revise", "reason": "make is missing, cargo works", "description": "Build with cargo", "when": "", "steps": steps}).to_string()))
    }

    fn check_answer(ok: bool) -> Result<Value, String> {
        Ok(reply(&json!({"ok": ok, "problem": if ok { "" } else { "it drops the test step" }}).to_string()))
    }

    #[test]
    fn a_skill_of_the_agent_is_revised_on_its_own_watched_and_taken_back_when_the_next_use_fails() {
        let studio = temp("auto");
        let skill = owned_skill(&studio, "build", "1. run make\n2. read the output");
        let original = skill.read(&studio).unwrap();
        learn::record_usage(&studio, &skill, Usage::new("a", "failed")).unwrap();
        let mut s = Script::new(vec![revision_answer("1. run cargo build\n2. run cargo test"), check_answer(true)]);
        let out = improve_skill_auto(&studio, &skill, "a failure", "", "s1", &mut s.call()).unwrap();
        assert_eq!(out, AutoOutcome::Applied);
        assert_eq!(s.seen.len(), 2, "the draft, then the self-check");
        assert!(skill.read(&studio).unwrap().contains("cargo build"));
        assert_eq!(learn::read_version(&studio, &skill, 1).unwrap(), original, "the previous text is kept");
        assert!(learn::is_agent_owned(&studio, &skill));
        assert_eq!(learn::watch_of(&studio, &skill).unwrap().remaining, learn::WATCH_USES);
        assert_eq!(learn::improve_due(&studio, &skill), None, "no second revision while the first is on watch");
        let log = learn::read_log(&studio, 5);
        assert_eq!(log[0]["kind"], "skill_revise");
        assert!(log[0]["diff"].as_str().unwrap().contains("+ 1. run cargo build") && log[0]["session"] == "s1" && !log[0]["reason"].as_str().unwrap().is_empty());
        // a good use counts down
        let good = RunInfo { session: "s2".into(), outcome: "done".into(), verify: Some(true), skills: vec!["build".into()], ..RunInfo::default() };
        let (_, notes) = record_skill_use_notes(&studio, &good);
        assert!(notes.is_empty() && learn::watch_of(&studio, &skill).unwrap().remaining == 1);
        // a bad use puts the previous text back, says why, and logs it
        let bad = RunInfo { session: "s3".into(), outcome: "failed".into(), skills: vec!["build".into()], ..RunInfo::default() };
        let (_, notes) = record_skill_use_notes(&studio, &bad);
        assert!(notes[0].contains("did not hold up"), "{notes:?}");
        assert_eq!(skill.read(&studio).unwrap(), original);
        assert!(learn::watch_of(&studio, &skill).is_none() && learn::is_agent_owned(&studio, &skill));
        let log = learn::read_log(&studio, 5);
        assert_eq!(log[0]["kind"], "skill_rollback");
        assert!(log[0]["reason"].as_str().unwrap().contains("s3"));
        assert_eq!(learn::improve_due(&studio, &skill), None, "the failure that took the revision back does not trigger another one");
        // a thumb down also takes a revision back; two good uses confirm it
        learn::record_usage(&studio, &skill, Usage::new("a", "failed")).unwrap();
        let mut s = Script::new(vec![revision_answer("1. run cargo build\n2. run cargo test"), check_answer(true)]);
        improve_skill_auto(&studio, &skill, "a failure", "", "s4", &mut s.call()).unwrap();
        let (_, notes) = record_feedback_notes(&studio, None, "s4", &["build".into()], "down").unwrap();
        assert!(notes[0].contains("did not hold up") && skill.read(&studio).unwrap() == original);
        let _ = std::fs::remove_dir_all(&studio);
    }

    #[test]
    fn a_revision_needs_a_confirming_self_check_and_a_skill_of_a_person_gets_a_sibling() {
        let studio = temp("sib");
        let skill = add_skill(&studio, "build", "1. run make\n2. read the output");
        let original = skill.read(&studio).unwrap();
        learn::record_usage(&studio, &skill, Usage::new("a", "failed")).unwrap();
        // the check says no, or says nothing usable: nothing is written
        let mut s = Script::new(vec![revision_answer("1. run cargo build\n2. run cargo test"), check_answer(false)]);
        assert!(matches!(improve_skill_auto(&studio, &skill, "t", "", "s", &mut s.call()).unwrap(), AutoOutcome::Skipped(w) if w.contains("self-check rejected")));
        let mut s = Script::new(vec![revision_answer("1. run cargo build\n2. run cargo test"), Ok(reply("looks fine to me"))]);
        assert!(matches!(improve_skill_auto(&studio, &skill, "t", "", "s", &mut s.call()).unwrap(), AutoOutcome::Skipped(_)));
        let mut s = Script::new(vec![Ok(reply("no json at all"))]);
        assert!(improve_skill_auto(&studio, &skill, "t", "", "s", &mut s.call()).is_err(), "an invalid draft writes nothing");
        assert_eq!(skill.read(&studio).unwrap(), original);
        assert!(learn::read_log(&studio, 5).is_empty() && learn::list_skills(&studio, None).len() == 1);
        // the skill is not the agent's: the revision becomes a sibling, the original stays
        let mut s = Script::new(vec![revision_answer("1. run cargo build\n2. run cargo test"), check_answer(true)]);
        let out = improve_skill_auto(&studio, &skill, "t", "", "s", &mut s.call()).unwrap();
        assert_eq!(out, AutoOutcome::Sibling("build-agent".into()));
        assert_eq!(skill.read(&studio).unwrap(), original);
        let sib = learn::sibling_ref(&skill);
        let text = sib.read(&studio).unwrap();
        assert!(text.contains("name: build-agent") && text.contains("Revised copy of 'build'") && text.contains("cargo build"), "{text}");
        assert!(learn::is_agent_owned(&studio, &sib));
        // a later revision goes into the sibling, with its history; a sibling that a person edited is left alone
        let mut s = Script::new(vec![revision_answer("1. run cargo build --release\n2. run cargo test"), check_answer(true)]);
        assert_eq!(improve_skill_auto(&studio, &skill, "t", "", "s", &mut s.call()).unwrap(), AutoOutcome::Sibling("build-agent".into()));
        assert!(learn::read_version(&studio, &sib, 1).is_some());
        std::fs::write(sib.file(&studio), format!("{}\nmy own line\n", sib.read(&studio).unwrap())).unwrap();
        let mut s = Script::new(vec![revision_answer("1. run cargo build --locked\n2. run cargo test"), check_answer(true)]);
        assert!(matches!(improve_skill_auto(&studio, &skill, "t", "", "s", &mut s.call()).unwrap(), AutoOutcome::Skipped(w) if w.contains("edited by hand")));
        assert!(sib.read(&studio).unwrap().contains("my own line"));
        let _ = std::fs::remove_dir_all(&studio);
    }

    fn complex_session(user: &str, calls: usize) -> Vec<Value> {
        let mut m = vec![json!({"role": "user", "content": user})];
        for i in 0..calls {
            m.push(json!({"role": "assistant", "content": "", "tool_calls": [{"id": format!("c{i}"), "function": {"name": "bash", "arguments": format!("{{\"command\":\"step {i}\"}}")}}]}));
            m.push(json!({"role": "tool", "tool_call_id": format!("c{i}"), "content": "ok"}));
        }
        m.push(json!({"role": "assistant", "content": "All done."}));
        m
    }

    #[test]
    fn a_complex_successful_run_gets_a_skill_that_is_logged_and_can_be_undone() {
        let studio = temp("create");
        let nothing = reply(&json!({"memory": [], "profile": []}).to_string());
        let skill_json = json!({"save": true, "name": "Release Build", "description": "Build and test a release", "when": "the user asks for a release build", "steps": "1. run `cargo build --release`\n2. run `cargo test`", "pitfalls": "make is not installed"}).to_string();
        let info = RunInfo { session: "s1".into(), outcome: "done".into(), verify: Some(true), ..RunInfo::default() };
        let mut s = Script::new(vec![Ok(reply(&skill_json)), Ok(nothing.clone())]);
        let lines = after_run(&studio, &info, &complex_session("make a release build", 7), &Options::default(), &mut s.call());
        assert!(lines.iter().any(|l| l.contains("Learned: created skill 'release-build'")), "{lines:?}");
        let made = SkillRef::user("release-build");
        let text = made.read(&studio).unwrap();
        assert!(text.contains("when: the user asks for a release build") && text.contains("## Pitfalls\nmake is not installed") && text.contains("2. run `cargo test`"), "{text}");
        assert!(learn::is_agent_owned(&studio, &made));
        assert!(std::fs::read_to_string(studio.join("learn/last-run.json")).unwrap().contains("created skill"));
        let log = learn::read_log(&studio, 5);
        assert_eq!(log[0]["kind"], "skill_create");
        let id = log[0]["id"].as_str().unwrap().to_owned();
        assert_eq!(learn::undo(&studio, &id).unwrap(), "skill 'release-build' removed");
        assert!(!made.file(&studio).exists() && learn::undo(&studio, &id).is_err());
        // no skill: too few calls, a failed check, a skill was used, the model says no, an invalid or secret-bearing answer
        let few = complex_session("make a release build", 2);
        let mut s = Script::new(vec![Ok(nothing.clone())]);
        after_run(&studio, &info, &few, &Options::default(), &mut s.call());
        assert_eq!(s.seen.len(), 1, "only the reflection ran");
        let failed = RunInfo { verify: Some(false), outcome: "failed".into(), ..info.clone() };
        let mut s = Script::new(vec![Ok(nothing.clone())]);
        after_run(&studio, &failed, &complex_session("x", 7), &Options::default(), &mut s.call());
        assert_eq!(s.seen.len(), 1);
        let mut s = Script::new(vec![Ok(reply(&json!({"save": false, "name": "", "description": "", "when": "", "steps": "", "pitfalls": ""}).to_string())), Ok(nothing.clone())]);
        let lines = after_run(&studio, &info, &complex_session("x", 7), &Options::default(), &mut s.call());
        assert!(lines.iter().any(|l| l.contains("no skill worth saving")) && learn::list_skills(&studio, None).is_empty());
        let secret = json!({"save": true, "name": "deploy", "description": "d", "when": "w", "steps": "1. export TOKEN=abc\n2. password = hunter2hunter2", "pitfalls": ""}).to_string();
        let mut s = Script::new(vec![Ok(reply(&secret)), Ok(nothing.clone())]);
        let lines = after_run(&studio, &info, &complex_session("x", 7), &Options::default(), &mut s.call());
        assert!(lines.iter().any(|l| l.contains("no skill written")) && learn::list_skills(&studio, None).is_empty(), "{lines:?}");
        let mut s = Script::new(vec![Ok(reply("a skill would be nice")), Ok(nothing)]);
        after_run(&studio, &info, &complex_session("x", 7), &Options::default(), &mut s.call());
        assert!(learn::list_skills(&studio, None).is_empty());
        // asking first: the skill waits as a proposal
        let mut s = Script::new(vec![Ok(reply(&skill_json)), Ok(reply(&json!({"memory": [], "profile": []}).to_string()))]);
        after_run(&studio, &info, &complex_session("make a release build", 7), &Options { auto: false, ..Options::default() }, &mut s.call());
        assert!(learn::list_skills(&studio, None).is_empty() && studio.join("skills-proposed/release-build/SKILL.md").is_file());
        assert!(parse_creation(&skill_json, &["release-build".to_owned()]).is_err(), "an existing name is not taken again");
        let _ = std::fs::remove_dir_all(&studio);
    }

    #[test]
    fn notes_and_profile_lines_are_saved_on_their_own_from_the_users_words_and_each_can_be_undone() {
        let studio = temp("autonotes");
        let ws = temp("autonotesws");
        let messages = complex_session("I prefer short answers and I work in Rust. Run the tests.", 2);
        let reflection = json!({
            "memory": [{"text": "Tests run with cargo test in the rust folder", "kind": "command", "scope": "workspace"}],
            "profile": [{"section": "preferences", "text": "Prefers short answers"}, {"section": "tool_habits", "text": "Uses bash and edit tools all the time"}]
        })
        .to_string();
        let info = RunInfo { session: "s9".into(), workspace: Some(ws.clone()), outcome: "done".into(), ..RunInfo::default() };
        let mut s = Script::new(vec![Ok(reply(&reflection))]);
        let lines = after_run(&studio, &info, &messages, &Options::default(), &mut s.call());
        assert!(lines.iter().any(|l| l.contains("Learned: saved 1 note, added 1 line to your profile")), "{lines:?}");
        assert!(learn::read_memory(&studio, "workspace", Some(&ws))[0].contains("cargo test"));
        let profile = learn::read_profile(&studio);
        assert!(profile.contains(&format!("- Prefers short answers ({})", learn::today())) && !profile.contains("bash"), "{profile}");
        assert!(lines.iter().any(|l| l.contains("dropped")));
        // what the loop told the model as a "user" message is no source for the profile
        let injected = vec![json!({"role": "user", "content": "do the thing"}), json!({"role": "user", "content": "Self-review before you finish: I prefer tabs and I always use vim"}), json!({"role": "assistant", "content": "ok"})];
        assert_eq!(user_text(&injected), "do the thing");
        assert!(grounded("Prefers short answers", "keep answers short please") && !grounded("Uses vim", "do the thing") && !grounded("x", "x"));
        // the log has both, and each can be undone
        let log = learn::read_log(&studio, 10);
        assert_eq!(log.iter().map(|e| e["kind"].as_str().unwrap()).collect::<Vec<_>>(), ["profile_add", "memory_add"]);
        assert!(log.iter().all(|e| e["session"] == "s9" && !e["reason"].as_str().unwrap().is_empty()));
        learn::undo(&studio, log[0]["id"].as_str().unwrap()).unwrap();
        learn::undo(&studio, log[1]["id"].as_str().unwrap()).unwrap();
        assert!(!learn::read_profile(&studio).contains("short answers") && learn::read_memory(&studio, "workspace", Some(&ws)).is_empty());
        assert!(learn::read_log(&studio, 2).iter().all(|e| e["undone"] == true));
        let _ = std::fs::remove_dir_all(&studio);
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn the_nightly_job_applies_its_changes_and_logs_them_in_auto_mode() {
        let studio = temp("night");
        let skill = owned_skill(&studio, "build", "1. run make\n2. read the output");
        std::fs::write(studio.join("agent-memory.md"), "# Memory\n\n- Alpha convention for naming things\n- Alpha convention for naming things!\n").unwrap();
        learn::record_usage(&studio, &skill, Usage::new("x", "failed")).unwrap();
        let mut s = Script::new(vec![revision_answer("1. run cargo build\n2. read the output"), check_answer(true)]);
        let lines = consolidate(&studio, None, true, &mut s.call());
        assert!(lines.iter().any(|l| l.contains("revised skill 'build'")) && lines.iter().any(|l| l.contains("removed a repeated memory entry")), "{lines:?}");
        assert_eq!(learn::read_memory(&studio, "global", None).len(), 1);
        let log = learn::read_log(&studio, 5);
        assert!(log.iter().any(|e| e["kind"] == "memory_remove") && log.iter().any(|e| e["kind"] == "skill_revise"));
        let removed = log.iter().find(|e| e["kind"] == "memory_remove").unwrap();
        learn::undo(&studio, removed["id"].as_str().unwrap()).unwrap();
        assert_eq!(learn::read_memory(&studio, "global", None).len(), 2, "the removed entry is back");
        let _ = std::fs::remove_dir_all(&studio);
    }

    #[test]
    fn learning_can_be_switched_off_globally_per_project_and_by_environment_and_ask_mode_is_a_setting() {
        let studio = temp("switch");
        let ws = temp("switchws");
        assert!(learn::enabled(&studio, Some(&ws)) && !learn::ask_first(&studio), "on by default, applying by itself");
        learn::set_learning_enabled(&studio, false).unwrap();
        assert!(!learn::enabled(&studio, Some(&ws)) && !learn::enabled(&studio, None));
        learn::set_project_override(&ws, Some(true)).unwrap();
        assert!(learn::enabled(&studio, Some(&ws)), "a project can switch it on again");
        learn::set_project_override(&ws, None).unwrap();
        learn::set_learning_enabled(&studio, true).unwrap();
        learn::set_project_override(&ws, Some(false)).unwrap();
        assert!(!learn::enabled(&studio, Some(&ws)) && learn::enabled(&studio, None));
        learn::set_ask_first(&studio, true).unwrap();
        assert!(learn::ask_first(&studio) && learn::profile_in_prompt(&studio), "settings do not overwrite each other");
        assert!(learn::take_notice(&studio) && !learn::take_notice(&studio), "the notice is shown once");
        let _ = std::fs::remove_dir_all(&studio);
        let _ = std::fs::remove_dir_all(&ws);
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
