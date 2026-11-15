//! Row 88 of PARITY.md: *Learning from sessions* — what `learn.js` does around the coding agent.
//!
//! After a run the window hands the server what happened (`lnRunMessages`), watches for the step
//! that follows and says what it learned; two thumbs and `/good`, `/bad` tell it how the skills of
//! the last run went; the switch of the Code page and `/learn` set the global and the folder's
//! own setting; and Settings > Personalization shows the profile the agent keeps of you, the
//! memory it wrote and what waits for a yes. This file is the part that can be compared with
//! the page case by case (`tests/learn_fixtures.rs`) and the requests the window makes.

use crate::api::WireMessage;
use serde::Deserialize;
use serde_json::{json, Value};

/// `lnText(c)` of a message of this window: its text.
fn slice_units(text: &str, max: usize) -> String {
    // `String.prototype.slice(0, max)` counts UTF-16 units
    let mut used = 0;
    let mut out = String::new();
    for c in text.chars() {
        used += c.len_utf16();
        if used > max {
            break;
        }
        out.push(c);
    }
    out
}

/// `lnSkillsRead(messages)`: the names of the skills the model read (the `skill` tool, action
/// `read`), each once, in the order they were read.
pub fn skills_read(messages: &[WireMessage]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for m in messages {
        let Some(calls) = m.tool_calls.as_ref().and_then(Value::as_array) else { continue };
        for call in calls {
            if call.pointer("/function/name").and_then(Value::as_str) != Some("skill") {
                continue;
            }
            let args = match call.pointer("/function/arguments") {
                Some(Value::String(s)) => serde_json::from_str::<Value>(s).ok(),
                Some(other) => Some(other.clone()),
                None => None,
            };
            let Some(args) = args.filter(|a| a.is_object()) else { continue };
            if args.get("action").and_then(Value::as_str) != Some("read") {
                continue;
            }
            // `String(args.name || '')`: a name that is no text is the text of what it is
            let name = crate::subagents::js_text(args.get("name"), "");
            if crate::skills::valid_name(&name) && !out.contains(&name) {
                out.push(name);
            }
        }
    }
    out
}

/// `lnSkillsRead(run)` over the messages `run_messages` made: their calls have the arguments cut
/// at 800 characters, which is what the page reads them from.
pub fn skills_read_run(run: &[Value]) -> Vec<String> {
    let messages: Vec<WireMessage> = run
        .iter()
        .map(|m| {
            let mut w = WireMessage::new(m["role"].as_str().unwrap_or_default(), "");
            w.tool_calls = m.get("tool_calls").cloned();
            w
        })
        .collect();
    skills_read(&messages)
}

/// `lnRunMessages(messages, from, maxMessages, maxChars)`: the messages of one run reduced to
/// what the server needs — roles, text and calls, each part cut short, no pictures.
pub fn run_messages(messages: &[WireMessage], from: usize, max_messages: usize, max_chars: usize) -> Vec<Value> {
    let list: Vec<&WireMessage> = messages
        .iter()
        .skip(from)
        .filter(|m| matches!(m.role.as_str(), "user" | "assistant" | "tool"))
        .collect();
    let start = list.len().saturating_sub(max_messages);
    list[start..]
        .iter()
        .map(|m| {
            let mut out = json!({ "role": m.role, "content": slice_units(&m.content, max_chars) });
            if let Some(calls) = m.tool_calls.as_ref().and_then(Value::as_array).filter(|c| !c.is_empty()) {
                out["tool_calls"] = Value::Array(
                    calls
                        .iter()
                        .map(|c| {
                            let arguments = match c.pointer("/function/arguments") {
                                Some(Value::String(s)) => slice_units(s, 800),
                                Some(other) => slice_units(&other.to_string(), 800),
                                None => slice_units("{}", 800),
                            };
                            json!({ "id": c.get("id").cloned().unwrap_or(Value::Null), "function": { "name": c.pointer("/function/name").cloned().unwrap_or(Value::Null), "arguments": arguments } })
                        })
                        .collect(),
                );
            }
            out
        })
        .collect()
}

/// The defaults of the page's call: three hundred messages of three thousand characters.
pub fn run_messages_default(messages: &[WireMessage], from: usize) -> Vec<Value> {
    run_messages(messages, from, 300, 3000)
}

/// `lnRunFailed({ error, hitLimit })`.
pub fn run_failed(error: bool, hit_limit: bool) -> bool {
    error || hit_limit
}

/// What `GET /api/studio/learn` answers (`fllmLearnStatus`): the switches, what waits, whether a
/// step runs and when the last one ended.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Status {
    /// `None`: the server did not say (the page then reads it as on).
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub ask_first: bool,
    /// The open folder's own setting (`.fllm/learn.json`): `Some(false)` off, `Some(true)` on.
    #[serde(default)]
    pub project_override: Option<bool>,
    #[serde(default)]
    pub notice_pending: bool,
    #[serde(default)]
    pub notice: String,
    #[serde(default)]
    pub running: bool,
    /// How many proposals wait (`st.pending` of `/learn`).
    #[serde(default)]
    pub pending: Option<f64>,
    #[serde(default)]
    pub last_run: Option<LastRun>,
}

/// The last learning step: when it ended, what it says of itself and the lines it wrote.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct LastRun {
    #[serde(default)]
    pub t: f64,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub lines: Vec<String>,
}

/// `lnToastText(lastRun)`: the sentence for the toast after a run, or nothing.
pub fn toast_text(last: Option<&LastRun>) -> String {
    match last {
        Some(l) if !l.summary.is_empty() => format!("Learned: {}", l.summary),
        _ => String::new(),
    }
}

/// The test of `lnWatchRun`'s poll: the step is over and ended after the run began.
pub fn poll_done(status: Option<&Status>, since: f64) -> bool {
    status.is_some_and(|o| !o.running && o.last_run.as_ref().is_some_and(|l| l.t >= since))
}

/// How often and how long `lnWatchRun` asks: first after four seconds, then every five, sixty times.
pub const FIRST_POLL_MS: u64 = 4000;
pub const POLL_MS: u64 = 5000;
pub const MAX_POLLS: u32 = 60;

/// The toast of a thumb (`fllmLearnFeedback`): `notes` are what the server added.
pub fn thumb_toast(value: &str, skills: &[String], notes: &[String]) -> String {
    if value == "up" {
        format!("Noted: {} worked.", skills.join(", "))
    } else {
        let extra = if notes.is_empty() { String::new() } else { format!("; {}", notes.join(" ")) };
        format!("Noted: {} did not work. A revision is drafted on the next run{extra}.", skills.join(", "))
    }
}

/// The paragraph under the checkbox of the side (`data-learn` block of `renderCodeSide`).
pub const SIDE_TEXT: &str = "On by default. After each run, in the background, the agent saves what is worth keeping: it notes how the skills it used fared, revises a skill that failed (the old text is kept and comes back if the revision does not hold up), writes a skill after a complex task that went well, saves notes about the project and lines for your profile from your own messages. Everything is listed under What I learned in Settings > Personalization with an Undo button. Switch it off here or with /learn off; /learn project off switches it off for this folder only. /good and /bad tell it whether the skills of the last run helped.";

/// What the page says when the last run used no skill.
pub const NO_SKILL_TOAST: &str = "No skill was used in the last run.";

/// The skills of the last run that was handed to the server (`window.fllmLearnLast`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Last {
    pub session: String,
    pub workspace: String,
    pub skills: Vec<String>,
}

/// `POST /api/studio/learn/after-run`: the run, as `fllmLearnAfterRun` posts it.
pub fn after_run_body(session: &str, workspace: &str, failed: bool, skills: &[String], messages: Vec<Value>) -> Value {
    json!({
        "session": if session.is_empty() { "studio" } else { session },
        "workspace": workspace,
        "outcome": if failed { "failed" } else { "done" },
        "skills": skills,
        "messages": messages,
    })
}

/// `POST /api/studio/learn/feedback`.
pub fn feedback_body(value: &str, last: &Last) -> Value {
    json!({ "value": value, "skills": last.skills, "session": last.session, "workspace": last.workspace })
}

/// What `/learn` says (`case 'learn'`): the state of the switch, then what the server added.
pub fn learn_notice(on: bool, status: Option<&Status>) -> String {
    let head = if on {
        "on. After each run the agent saves skills, notes and profile lines on its own and logs every change (Settings > Personalization > What I learned, with Undo)."
    } else {
        "off."
    };
    let tail = match status {
        Some(st) => {
            let mut t = String::from(" ");
            if st.ask_first {
                t.push_str("It proposes and waits for your yes.");
            }
            if let Some(n) = st.pending.filter(|n| *n != 0.0 && !n.is_nan()) {
                let one = n == 1.0;
                t.push_str(&format!(" {} proposal{} wait{} for a decision.", crate::specs::index_text(n), if one { "" } else { "s" }, if one { "s" } else { "" }));
            }
            t
        }
        None => String::new(),
    };
    format!("Learning from my sessions: {head}{tail}")
}

/// The words of `/learn project …` after the request: *follows the global setting*, *on* or *off*.
pub fn project_notice(project: Option<bool>) -> String {
    format!(
        "Learning in this folder: {}.",
        match project {
            None => "follows the global setting",
            Some(true) => "on",
            Some(false) => "off",
        }
    )
}

/// The usage line of `/learn project` without a folder or a word the page knows.
pub const PROJECT_USAGE: &str = "Usage: /learn project on|off|default (for the open folder; it is kept in .fllm/learn.json there)";

/// `{ on: true, off: false, default: null }[value]`: `Some(Some(b))` for on and off, `Some(None)`
/// for default, `None` for any other word.
pub fn project_word(value: &str) -> Option<Option<bool>> {
    match value {
        "on" => Some(Some(true)),
        "off" => Some(Some(false)),
        "default" => Some(None),
        _ => None,
    }
}

/// What `/learn` does with its words (`case 'learn'`): the folder's own setting, or a look at the
/// switch (with `on`/`off` changing it first).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearnPlan {
    /// `/learn project …` with no folder or a word the page does not know.
    ProjectUsage,
    /// `/learn project on|off|default`.
    Project(Option<bool>),
    /// The state of the switch: `Some(on)` after `on`/`off`.
    Show(Option<bool>),
}

/// `const [word, value] = argText.toLowerCase().split(/\s+/)`, then the branches.
pub fn plan_learn(arg_text: &str, has_folder: bool) -> LearnPlan {
    let lower = arg_text.to_lowercase();
    let mut words = lower.split_whitespace();
    let word = words.next().unwrap_or("");
    let value = words.next().unwrap_or("");
    if word == "project" {
        return match (has_folder, project_word(value)) {
            (true, Some(project)) => LearnPlan::Project(project),
            _ => LearnPlan::ProjectUsage,
        };
    }
    match word {
        "on" => LearnPlan::Show(Some(true)),
        "off" => LearnPlan::Show(Some(false)),
        _ => LearnPlan::Show(None),
    }
}

// ---- the requests ---------------------------------------------------------------------------

async fn request(base: &str, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<Value, String> {
    crate::models::mcp_request(base, method, path, body).await
}

/// `window.fllmLearnStatus()`: `None` for any failed read.
pub async fn fetch_status(base: String, workspace: String) -> Option<Status> {
    let path = format!("/api/studio/learn?{}", crate::models::sk_query(&[("workspace", &workspace)]));
    request(&base, reqwest::Method::GET, &path, None).await.ok().and_then(|v| serde_json::from_value(v).ok())
}

/// `window.fllmLearnNotice()`: the one-sentence first-run notice (once), or nothing.
pub async fn fetch_notice(base: String, workspace: String) -> String {
    let Some(st) = fetch_status(base.clone(), workspace).await else { return String::new() };
    if !st.notice_pending {
        return String::new();
    }
    // a failed mark means the notice comes again next time
    let _ = request(&base, reqwest::Method::POST, "/api/studio/learn/settings", Some(json!({ "notice": true }))).await;
    st.notice
}

/// `window.fllmLearnSetEnabled(on)`: the global switch; the folder's own setting is cleared so
/// that the switch and the setting agree.
pub async fn set_enabled(base: String, on: bool, workspace: String) -> Result<(), String> {
    request(&base, reqwest::Method::POST, "/api/studio/learn/settings", Some(json!({ "enabled": on, "workspace": workspace, "project": Value::Null })))
        .await
        .map(|_| ())
}

/// `/learn project on|off|default`: the folder's own setting.
pub async fn set_project(base: String, workspace: String, project: Option<bool>) -> Result<(), String> {
    request(&base, reqwest::Method::POST, "/api/studio/learn/settings", Some(json!({ "workspace": workspace, "project": project })))
        .await
        .map(|_| ())
}

/// `window.fllmLearnAfterRun`: whether the server queued the step (`r.queued`).
pub async fn after_run(base: String, body: Value) -> bool {
    request(&base, reqwest::Method::POST, "/api/studio/learn/after-run", Some(body))
        .await
        .ok()
        .and_then(|v| v.get("queued").and_then(Value::as_bool))
        .unwrap_or(false)
}

/// `window.fllmLearnFeedback`: the number of skills noted and the notes of the server.
pub async fn feedback(base: String, body: Value) -> Result<(u64, Vec<String>), String> {
    let v = request(&base, reqwest::Method::POST, "/api/studio/learn/feedback", Some(body)).await?;
    let noted = v.get("noted").and_then(Value::as_u64).unwrap_or(0);
    let notes = v
        .get("notes")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_default();
    Ok((noted, notes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: &str) -> Value {
        json!({ "id": "c", "type": "function", "function": { "name": name, "arguments": args } })
    }

    #[test]
    fn the_skills_a_run_read_are_listed_once_in_order() {
        let mut m = WireMessage::new("assistant", "");
        m.tool_calls = Some(json!([
            call("skill", r#"{"action":"read","name":"deploy"}"#),
            call("skill", r#"{"action":"read","name":"deploy"}"#),
            call("skill", r#"{"action":"list"}"#),
            call("skill", r#"{"action":"read","name":"Bad Name"}"#),
            call("grep", r#"{"action":"read","name":"x"}"#),
            call("skill", r#"{"action":"read","name":"zeta"}"#),
            call("skill", "{not json"),
        ]));
        assert_eq!(skills_read(&[m]), ["deploy", "zeta"]);
    }

    #[test]
    fn a_thumb_and_a_toast_say_what_the_page_says() {
        let skills = vec!["a".to_string(), "b".to_string()];
        assert_eq!(thumb_toast("up", &skills, &[]), "Noted: a, b worked.");
        assert_eq!(thumb_toast("down", &skills, &[]), "Noted: a, b did not work. A revision is drafted on the next run.");
        assert_eq!(thumb_toast("down", &skills, &["x.".into(), "y.".into()]), "Noted: a, b did not work. A revision is drafted on the next run; x. y..");
        assert_eq!(toast_text(Some(&LastRun { summary: "a skill".into(), ..Default::default() })), "Learned: a skill");
        assert_eq!(toast_text(None), "");
    }

    #[test]
    fn the_poll_ends_when_the_step_is_over_and_newer() {
        let st = |running, t| Status { running, last_run: Some(LastRun { t, ..Default::default() }), ..Default::default() };
        assert!(poll_done(Some(&st(false, 10.0)), 10.0));
        assert!(!poll_done(Some(&st(true, 11.0)), 10.0));
        assert!(!poll_done(Some(&st(false, 9.0)), 10.0));
        assert!(!poll_done(None, 10.0));
        assert!(!poll_done(Some(&Status::default()), 10.0));
    }
}
