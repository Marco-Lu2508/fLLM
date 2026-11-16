//! Settings > Skills, scheduled tasks and skill library: the procedures the model can read, the
//! tasks it runs on a clock and the skills that come with the app (`spane-skills` of `shell.html`,
//! `skills.js` and `skilllibrary.js` of the Studio).
//!
//! The pane holds seven cards here: the list, the improvements, the form, the tasks, the task
//! form, and the two cards `skilllibrary.js` appends. The *Search of past chats* card in the
//! middle of the page is not ported — its counts and its copy come out of the browser's
//! localStorage, which this window does not have (see PARITY.md).

use crate::app::Message;
use chrono::{Datelike, Timelike};
use crate::models::card;
use crate::theme::{self, Tokens};
use iced::widget::text::Span;
use iced::widget::{button, column, container, pick_list, rich_text, row, text, text_editor, text_input, Space};
use iced::{Alignment, Element, Font, Length};

/// A skill name: lowercase letters, digits, `-` and `_`, first character a letter or digit, at
/// most 64 characters (`SK_NAME_RE` of `skills.js`).
pub const NAME_RE: &str = "^[a-z0-9][a-z0-9_-]{0,63}$";

/// Whether a name is one a skill may have (`skValidName`).
pub fn valid_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > 64 {
        return false;
    }
    let first = bytes[0];
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return false;
    }
    bytes[1..].iter().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'-' || *b == b'_')
}

/// A readable title turned into a skill name (`skSlug`): lowercase, every run of what is no name
/// a dash, no dash at the start or the end, and no longer than 64 characters.
pub fn slug(text: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in text.to_lowercase().chars() {
        // `[^a-z0-9]+` of the page is one dash — an underscore is one of those characters, too
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out.truncate(64.min(out.len()));
    while out.ends_with('-') {
        out.pop();
    }
    out
}

/// The front matter and the body of a `SKILL.md` (`skParse`): the keys of the front matter are
/// lowercased and their quotes at the ends are taken off; text without front matter is all body.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Parsed {
    pub meta: std::collections::BTreeMap<String, String>,
    pub body: String,
}

pub fn parse(text: &str) -> Parsed {
    let mut text = text;
    if let Some(rest) = text.strip_prefix('\u{feff}') {
        text = rest;
    }
    let bytes = text.as_bytes();
    // the page's `^---\r?\n`
    let open = if bytes.starts_with(b"---\r\n") { 5 } else { 4 };
    if bytes.len() < 8 || (!bytes.starts_with(b"---\n") && !bytes.starts_with(b"---\r\n")) {
        return Parsed { meta: Default::default(), body: text.to_string() };
    }
    // the front matter ends at a line that is `---` followed by nothing or by spaces and a newline
    let mut end = None;
    let mut cursor = open;
    while cursor < bytes.len() {
        let line_end = text[cursor..]
            .find('\n')
            .map(|i| cursor + i)
            .unwrap_or(text.len());
        let line = text[cursor..line_end].trim_end_matches(['\r', ' ']);
        if line == "---" {
            end = Some((cursor + (line_end - cursor) + 1).min(text.len()));
            break;
        }
        cursor = line_end + 1;
    }
    let (front, rest) = match end {
        Some(end) => (&text[open..end], &text[end..]),
        None => return Parsed { meta: Default::default(), body: text.to_string() },
    };
    let mut meta = std::collections::BTreeMap::new();
    for line in front.lines() {
        let line = line.trim_end_matches('\r');
        if line == "---" {
            continue;
        }
        let Some(i) = line.find(':') else { continue };
        if i < 1 {
            continue;
        }
        let key = line[..i].trim().to_lowercase();
        let value = line[i + 1..].trim();
        let value = value
            .strip_prefix(['"', '\''])
            .and_then(|v| v.strip_suffix(['"', '\'']))
            .unwrap_or(value);
        meta.insert(key, value.to_string());
    }
    Parsed { meta, body: rest.to_string() }
}

/// The text of a `SKILL.md` from its parts (`skCompose`): the front matter with the three fields
/// on one line each, and a body that ends in a newline.
pub fn compose(name: &str, description: &str, when: &str, body: &str) -> String {
    fn one(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }
    let mut body = body.trim_start_matches('\n').to_string();
    if !body.is_empty() && !body.ends_with('\n') {
        body.push('\n');
    }
    format!("---\nname: {}\ndescription: {}\nwhen: {}\n---\n{}", one(name), one(description), one(when), body)
}

/// The text *New from template* puts into the box (`skTemplate` of `skills.js`).
pub fn template() -> String {
    compose(
        "my-skill",
        "What this skill does, in one sentence",
        "When to use it",
        "# Title\n\n1. First step\n2. Second step\n\n## Pitfalls\n- \n",
    )
}

/// One skill as the list has it (`GET /api/studio/skills`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Skill {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub when: String,
    #[serde(default)]
    pub scope: String,
}

/// What `GET /api/studio/skills` answers.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct SkillsAnswer {
    #[serde(default)]
    pub skills: Vec<Skill>,
    #[serde(default)]
    pub user_folder: String,
}

/// The line under the list (`skFolderInfo` of `skills.js`).
pub fn folder_info(answer: &SkillsAnswer) -> String {
    format!(
        "Your skills are saved in {}. Workspace skills are saved in <folder>/.fllm/skills/<name>/SKILL.md.",
        answer.user_folder
    )
}

/// One task as the list has it (`GET /api/studio/schedule`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Task {
    #[serde(default)]
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub prompt: String,
    pub when: String,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub folder: String,
    #[serde(default)]
    pub notify: bool,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub deliver: String,
    #[serde(default)]
    pub running: bool,
    #[serde(default)]
    pub next_run: Option<i64>,
    #[serde(default)]
    pub last: Option<TaskRun>,
}

/// The last run of a task as the list shows it.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct TaskRun {
    pub time: i64,
    #[serde(default)]
    pub ok: bool,
    #[serde(default)]
    pub delivery: Option<String>,
}

/// What `GET /api/studio/schedule` answers.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct ScheduleAnswer {
    #[serde(default)]
    pub clock_running: bool,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub tasks: Vec<Task>,
}

/// The line over the tasks (`#scNote`): what the server says while its clock runs, and its own
/// sentence when it does not.
pub fn schedule_note(answer: &ScheduleAnswer) -> String {
    if answer.clock_running {
        answer.note.clone()
    } else {
        "The scheduler starts when a model is loaded. Tasks can be defined now.".into()
    }
}

/// The chips of a task and its buttons are drawn from the task itself; this is the chip of its
/// kind (`t.mode === 'agent' ? 'agent' : t.mode === 'learn' ? 'consolidate' : 'chat'`).
pub fn mode_chip(mode: &str) -> &'static str {
    match mode {
        "agent" => "agent",
        "learn" => "consolidate",
        _ => "chat",
    }
}

/// The first part of where a result is also sent (`t.deliver.split(':')[0]`), or nothing when
/// nothing is set.
pub fn deliver_chip(deliver: &str) -> String {
    if deliver.is_empty() {
        return String::new();
    }
    deliver.split(':').next().unwrap_or(deliver).to_string()
}

/// What a task's line says about its runs (`render()` of `scRefreshPane`): when it runs next, when
/// it last ran and how that went — or that it has not run yet.
pub fn task_runs_line(task: &Task) -> String {
    let mut out = String::new();
    if let Some(next) = task.next_run {
        out.push_str(&format!("Next run: {}. ", when(next)));
    }
    match &task.last {
        Some(run) => {
            out.push_str(&format!(
                "Last run: {} ({}{}).",
                when(run.time),
                if run.ok { "ok" } else { "failed" },
                match &run.delivery {
                    Some(delivery) => format!("; delivery {delivery}"),
                    None => String::new(),
                }
            ));
        }
        None => out.push_str("Has not run yet."),
    }
    out
}

/// An epoch in seconds as the page writes it (`scWhen` of `skills.js`). The Studio writes
/// `toLocaleString()` of the browser, so its shape follows the locale of that machine; this window
/// writes the shape of English (United States), which is what a browser with no other locale
/// gives: `11/14/2026, 8:45:12 PM`.
pub fn when(epoch: i64) -> String {
    let local = chrono::DateTime::from_timestamp(epoch, 0)
        .map(|dt| dt.with_timezone(&chrono::Local))
        .expect("an epoch of a task");
    let hour = local.hour() % 12;
    let hour = if hour == 0 { 12 } else { hour };
    format!(
        "{}/{}/{}, {}:{:02}:{:02} {}",
        local.month(),
        local.day(),
        local.year(),
        hour,
        local.minute(),
        local.second(),
        if local.hour() < 12 { "AM" } else { "PM" },
    )
}

/// The kinds of task the form offers (`#scMode`): the name in the request and the label.
pub const MODES: [(&str, &str); 3] = [
    ("chat", "Chat answer"),
    ("agent", "Agent run in a folder (read-only)"),
    ("learn", "Consolidate memory and skills (no prompt needed)"),
];

/// The label of a kind of task.
pub fn mode_label(mode: &str) -> Option<&'static str> {
    MODES.iter().find(|(id, _)| *id == mode).map(|(_, label)| *label)
}

/// The name in the request for a label of the kind select.
pub fn mode_id(label: &str) -> &'static str {
    MODES.iter().find(|(_, name)| *name == label).map(|(id, _)| *id).unwrap_or("chat")
}

/// The examples the schedule select offers (`#scWhenPresets`): the value of the first entry is the
/// placeholder of that select, which the page reads as *Examples* and never sends.
pub const WHEN_PRESETS: [&str; 6] = [
    "every 30 minutes",
    "hourly",
    "daily at 08:30",
    "weekdays at 09:00",
    "weekly on mon at 09:00",
    "*/15 9-17 * * mon-fri",
];

/// What the page counts as *there* (`if (s.due)`, `t.deliver ? … : ''`): everything but nothing,
/// `false`, `0` and the empty text.
pub fn truthy(value: &Option<serde_json::Value>) -> bool {
    match value {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::Number(n)) => n.as_f64().is_some_and(|n| n != 0.0),
        Some(serde_json::Value::String(s)) => !s.is_empty(),
        Some(_) => true,
    }
}

/// A number of installs the way the page writes it (`Number(k.installs).toLocaleString()`):
/// `1234` becomes `1,234`, and a value that is no number becomes what JavaScript writes for it.
pub fn installs_text(value: &Option<serde_json::Value>) -> String {
    let Some(value) = value else { return String::new() };
    let number = match value {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    };
    let Some(number) = number else { return "NaN".into() };
    let integer = number as i64;
    let digits = integer.abs().to_string();
    let mut grouped = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(c);
    }
    format!("{}{grouped}", if integer < 0 { "-" } else { "" })
}

/// One skill that comes with the app (`GET /api/studio/skills/library`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct LibrarySkill {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub when: String,
    #[serde(default)]
    pub installed: bool,
}

/// What `GET /api/studio/skills/library` answers.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct LibraryAnswer {
    #[serde(default)]
    pub skills: Vec<LibrarySkill>,
}

/// One skill of a repository (`GET /api/studio/skills/github`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct GithubSkill {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub path: String,
    pub url: String,
    #[serde(default)]
    pub installed: bool,
}

/// What `GET /api/studio/skills/github?repo=` answers.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct GithubAnswer {
    #[serde(default)]
    pub skills: Vec<GithubSkill>,
    #[serde(default)]
    pub repo: String,
    #[serde(default)]
    pub truncated: bool,
}

/// One skill of the directory (`GET /api/studio/skills/directory`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct DirectorySkill {
    pub name: String,
    pub id: String,
    #[serde(default)]
    pub installs: Option<serde_json::Value>,
    #[serde(default)]
    pub installed: bool,
}

/// What `GET /api/studio/skills/directory?…` answers.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct DirectoryAnswer {
    #[serde(default)]
    pub configured: bool,
    #[serde(default)]
    pub skills: Vec<DirectorySkill>,
}

/// What a task answered the last times (`GET /api/studio/schedule/results`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct ResultsAnswer {
    #[serde(default)]
    pub results: Vec<TaskResult>,
}

/// One of those answers: when it ran, how long it took and what came out of it.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct TaskResult {
    pub time: i64,
    #[serde(default)]
    pub ok: bool,
    #[serde(default)]
    pub seconds: f64,
    #[serde(default)]
    pub text: String,
}

/// What `GET /api/studio/schedule/check` answers.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct CheckAnswer {
    #[serde(default)]
    pub ok: bool,
    #[serde(default)]
    pub next: Vec<i64>,
    #[serde(default)]
    pub error: Option<String>,
}

/// What waits for a decision, whether one is running and what the last step wrote
/// (`GET /api/studio/learn`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct LearnOverview {
    #[serde(default)]
    pub skills: Vec<serde_json::Value>,
    #[serde(default)]
    pub memory: Vec<serde_json::Value>,
    #[serde(default)]
    pub profile: Vec<serde_json::Value>,
    #[serde(default)]
    pub running: bool,
    #[serde(default)]
    pub last_run: Option<LastRun>,
}

/// What the last learning step left behind.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct LastRun {
    #[serde(default)]
    pub lines: Vec<String>,
}

/// The line over the two lists of the card (`lnPendingText` of `learn.js` plus the sentence of a
/// step that runs now).
pub fn pending_text(overview: &LearnOverview) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (list, one, many) in [
        (&overview.skills, "skill revision", "skill revisions"),
        (&overview.memory, "memory entry", "memory entries"),
        (&overview.profile, "profile line", "profile lines"),
    ] {
        if !list.is_empty() {
            parts.push(format!("{} {}", list.len(), if list.len() == 1 { one } else { many }));
        }
    }
    let mut text = if parts.is_empty() {
        "Nothing waits for a decision.".to_string()
    } else {
        format!("Waiting for a decision: {}.", parts.join(", "))
    };
    if overview.running {
        text.push_str(" A learning step is running now.");
    }
    text
}

/// The line under it (`lnRefreshCounters`): the first four lines of the last step, and nothing
/// when there was no step.
pub fn last_run_text(overview: &LearnOverview) -> String {
    match &overview.last_run {
        Some(run) if !run.lines.is_empty() => {
            let shown: Vec<&str> = run.lines.iter().take(4).map(String::as_str).collect();
            format!("Last learning step ({}): {}", run.lines.len(), shown.join(" "))
        }
        _ => String::new(),
    }
}

/// One change of the log of automatic changes (`GET /api/studio/learn/log`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct LearnEntry {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    pub t: i64,
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub diff: Option<String>,
    #[serde(default)]
    pub undone: bool,
}

/// What `GET /api/studio/learn/log` answers.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct LearnLog {
    #[serde(default)]
    pub log: Vec<LearnEntry>,
}

/// The title of an entry of the log for the card (`lnLogTitle`), for the three kinds the feed of
/// this card holds.
pub fn log_title(entry: &LearnEntry) -> String {
    let name = entry.name.clone().unwrap_or_default();
    match entry.kind.as_str() {
        "skill_create" => format!("Wrote the skill \"{name}\""),
        "skill_revise" => format!("Revised the skill \"{name}\""),
        "skill_rollback" => format!("Put back the previous text of \"{name}\""),
        other => other.to_string(),
    }
}

/// One answer of a task as the page writes it under the task (`scShowResults`): when it ran, how
/// it went and how long it took.
pub fn result_line(answer: &TaskResult) -> String {
    // `(${r.seconds}s)` of `scShowResults`: the seconds carry their unit
    format!(
        "{} - {} ({}s)",
        when(answer.time),
        if answer.ok { "ok" } else { "failed" },
        answer.seconds
    )
}

/// One version of a skill, one draft of a revision and one use of it (the parts of a row of
/// `GET /api/studio/learn/skills` that the card shows).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Version {
    pub version: u64,
    #[serde(default)]
    pub action: String,
    pub t: i64,
}

/// A draft of the model: why it wants to change the skill and what it would write.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Revision {
    #[serde(default)]
    pub trigger: Option<String>,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub diff: String,
}

/// One recorded use of a skill.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Recent {
    pub t: i64,
    #[serde(default)]
    pub outcome: String,
    #[serde(default)]
    pub feedback: Option<String>,
    #[serde(default)]
    pub verify: Option<bool>,
    #[serde(default)]
    pub note: Option<String>,
}

/// A skill as the learning loop saw it.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct LearnSkill {
    pub name: String,
    #[serde(default)]
    pub scope: String,
    #[serde(default)]
    pub workspace: String,
    #[serde(default)]
    pub uses: u64,
    #[serde(default)]
    pub failed: u64,
    #[serde(default)]
    pub up: u64,
    #[serde(default)]
    pub down: u64,
    #[serde(default)]
    pub retries: u64,
    #[serde(default)]
    pub pending: bool,
    #[serde(default)]
    pub due: Option<serde_json::Value>,
    #[serde(default)]
    pub versions: Vec<Version>,
    #[serde(default)]
    pub revision: Option<Revision>,
    #[serde(default)]
    pub recent: Vec<Recent>,
}

/// What `GET /api/studio/learn/skills` answers.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct LearnSkills {
    #[serde(default)]
    pub skills: Vec<LearnSkill>,
}

/// Whether a row has something to show (`lnSkillRelevant`).
pub fn relevant(skill: &LearnSkill) -> bool {
    skill.uses > 0 || skill.pending || !skill.versions.is_empty() || truthy(&skill.due)
}

/// What a row says about itself (`lnStatText`): its uses, and the numbers beside them.
pub fn stat_text(skill: &LearnSkill) -> String {
    let mut parts = vec![format!("{} use{}", skill.uses, if skill.uses == 1 { "" } else { "s" })];
    if skill.failed > 0 {
        parts.push(format!("{} failed", skill.failed));
    }
    if skill.up > 0 || skill.down > 0 {
        parts.push(format!("thumbs {} up / {} down", skill.up, skill.down));
    }
    if skill.retries > 0 {
        parts.push(format!("{} retries", skill.retries));
    }
    parts.join(", ")
}

/// What the Results button of one task has read so far (the page shows its answers in a box it
/// toggles under the row).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResultsState {
    pub shown: bool,
    pub filled: Option<Result<Vec<TaskResult>, String>>,
}

/// What the dialog of the card waits for: the page asks before a rollback and before an approval
/// (`askDialog` of `learn.js`), with the values of the row that was clicked.
#[derive(Debug, Clone, PartialEq)]
pub enum LnConfirm {
    Rollback(String, String, String, u64),
    Approve(String, String, String),
}

impl LnConfirm {
    /// The title of the dialog (`askDialog({title, text, ok})`).
    pub fn title(&self) -> String {
        match self {
            LnConfirm::Rollback(name, _, _, _version) => format!("Roll back \"{name}\"?"),
            LnConfirm::Approve(name, _, _) => format!("Apply the revision of \"{name}\"?"),
        }
    }

    /// The sentence under it.
    pub fn text(&self) -> String {
        match self {
            LnConfirm::Rollback(_, _, _, version) => format!(
                "The text of version {version} is put back. The current text is kept as a new version, so this can be undone."
            ),
            LnConfirm::Approve(_, _, _) => {
                "The current text is kept as a version; you can roll back at any time.".into()
            }
        }
    }

    /// The button that says yes (`ok` of the dialog).
    pub fn ok(&self) -> &'static str {
        match self {
            LnConfirm::Rollback(..) => "Roll back",
            LnConfirm::Approve(..) => "Apply",
        }
    }

    /// What is sent when the answer is yes.
    pub fn body(&self) -> serde_json::Value {
        match self {
            LnConfirm::Rollback(name, scope, workspace, version) => serde_json::json!({
                "action": "rollback", "name": name, "scope": scope, "workspace": workspace, "version": version
            }),
            LnConfirm::Approve(name, scope, workspace) => serde_json::json!({
                "action": "approve", "name": name, "scope": scope, "workspace": workspace
            }),
        }
    }
}

/// The dialog before something is taken away (`askDialog({title, text, ok, danger})`): its
/// question, the sentence under it, and the two buttons the caller brings.
pub fn delete_window<'a>(
    t: Tokens,
    title: String,
    sentence: &'static str,
    ok: Message,
    cancel: Message,
) -> Element<'a, Message> {
    let window = container(
        column![
            text(title).size(16).font(crate::models::bold()),
            text(sentence).size(13).color(t.fg_soft),
            row![
                Space::new().width(Length::Fill),
                button(text("Cancel").size(13))
                    .on_press(cancel)
                    .padding([6, 12])
                    .style(theme::ghost_button(t)),
                button(text("Delete").size(13))
                    .on_press(ok)
                    .padding([6, 12])
                    .style(move |_, status| {
                        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                        button::Style {
                            background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })),
                            text_color: t.danger,
                            border: iced::border::rounded(8),
                            ..Default::default()
                        }
                    }),
            ]
            .spacing(8),
        ]
        .spacing(14),
    )
    .padding(16)
    .width(460)
    .style(theme::card(t, theme::R_LG, false));
    container(iced::widget::opaque(window))
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(24)
        .into()
}

/// The dialog of the card: its question, its sentence and its two buttons (`askDialog`).
pub fn confirm_window<'a>(t: Tokens, confirm: &LnConfirm) -> Element<'a, Message> {
    let title = confirm.title();
    let sentence = confirm.text();
    let ok = confirm.ok();
    let window = container(
        column![
            text(title).size(16).font(crate::models::bold()),
            text(sentence).size(13).color(t.fg_soft),
            row![
                Space::new().width(Length::Fill),
                button(text("Cancel").size(13))
                    .on_press(Message::LnConfirmCancel)
                    .padding([6, 12])
                    .style(theme::ghost_button(t)),
                button(text(ok).size(13))
                    .on_press(Message::LnConfirmOk)
                    .padding([6, 12])
                    .style(theme::primary_button(t)),
            ]
            .spacing(8),
        ]
        .spacing(14),
    )
    .padding(16)
    .width(460)
    .style(theme::card(t, theme::R_LG, false));
    container(iced::widget::opaque(window))
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(24)
        .into()
}

/// The data the seven cards need (`spane-skills` of the page; the *Search of past chats* card in
/// the middle of the Studio's pane is not ported, see the top of this file).
#[derive(Clone, Copy)]
pub struct SkillsView<'a> {
    /// Card 1: the workspace box, the list, the line under it and the sentence that replaces it.
    pub workspace: &'a str,
    pub skills: &'a [Skill],
    pub folder_info: &'a str,
    pub list_say: &'a str,
    /// Card 2: what waits for a decision, the last step, the feed and the rows with their
    /// versions and drafts.
    pub pending: &'a str,
    pub last_run: &'a str,
    pub feed: &'a [LearnEntry],
    pub revisions: &'a [LearnSkill],
    pub revisions_say: &'a str,
    /// Card 3: the form that writes one.
    pub name: &'a str,
    pub scope: &'a str,
    pub text: &'a text_editor::Content,
    /// Cards 4 and 5: the tasks, what their Results buttons read and the form over them.
    pub schedule_note: &'a str,
    pub tasks: &'a [Task],
    pub tasks_say: &'a str,
    pub results: &'a std::collections::HashMap<String, ResultsState>,
    /// Card 5: the task form (`scFill` of the page — *New task*, *Edit the task*).
    pub task_name: &'a str,
    pub task_prompt: &'a text_editor::Content,
    pub task_when: &'a str,
    pub task_preview: &'a str,
    pub task_mode: &'a str,
    pub task_folder: &'a str,
    pub task_deliver: &'a str,
    pub task_notify: bool,
    pub task_enabled: bool,
    pub task_editing: bool,
    /// Cards 6 and 7: the skills that come with the app and the finder. The sentence that
    /// replaces the library list when it could not be read sits beside the one under it — the
    /// page writes the first into the list and the second into its own line.
    pub library: &'a [LibrarySkill],
    pub library_list_say: &'a str,
    pub library_say: &'a str,
    pub import_url: &'a str,
    pub repo: &'a str,
    pub repo_say: &'a str,
    pub github: &'a [GithubSkill],
    pub directory: &'a [DirectorySkill],
    pub directory_q: &'a str,
    pub dir_view: &'a str,
    pub dir_token: &'a str,
}

/// The three views of the directory select (`#skDirView`): the name in the request and the label.
pub const VIEWS: [(&str, &str); 3] = [("all-time", "Most installed"), ("trending", "Trending"), ("hot", "Hot")];

/// The label of a view of the select.
pub fn view_label(id: &str) -> &'static str {
    VIEWS.iter().find(|(name, _)| *name == id).map(|(_, label)| *label).unwrap_or("Most installed")
}

/// A text box of the cards (`ui-input mono`).
fn input<'a>(
    t: Tokens,
    placeholder: &'a str,
    value: &'a str,
    typed: fn(String) -> Message,
) -> Element<'a, Message> {
    text_input(placeholder, value)
        .on_input(typed)
        .size(13)
        .padding([6, 10])
        .font(Font::MONOSPACE)
        .style(move |_, status| text_input::Style {
            background: iced::Background::Color(t.surface_2),
            border: iced::Border {
                radius: 8.0.into(),
                width: 1.0,
                color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator },
            },
            icon: t.muted,
            placeholder: t.muted,
            value: t.fg,
            selection: t.primary_soft,
        })
        .into()
}

pub(crate) fn editor_style<'a>(t: Tokens) -> impl Fn(&iced::Theme, text_editor::Status) -> text_editor::Style + 'a {
    move |_, status| text_editor::Style {
        background: iced::Background::Color(t.surface_2),
        border: iced::Border {
            radius: 8.0.into(),
            width: 1.0,
            color: if matches!(status, text_editor::Status::Focused { .. }) { t.fg_soft } else { t.separator },
        },
        placeholder: t.muted,
        value: t.fg,
        selection: iced::Color { a: 0.35, ..t.primary },
    }
}

/// The description of *Scheduled tasks*, with the sentence the page writes in `<strong>`.
pub const TASKS_HELP: &str = "A task runs a prompt on a schedule and keeps the last results here. Tasks run only while the app is running; a run that was missed while it was closed is not made up. An agent task may read and search one folder and look things up on the web, but cannot change anything.";

fn tasks_help_spans() -> Vec<Span<'static, String>> {
    vec![
        Span::new("A task runs a prompt on a schedule and keeps the last results here. "),
        Span::new("Tasks run only while the app is running").font(Font { weight: iced::font::Weight::Semibold, ..Font::DEFAULT }),
        Span::new("; a run that was missed while it was closed is not made up. An agent task may read and search one folder and look things up on the web, but cannot change anything."),
    ]
}

/// The line between two rows (`border-bottom` of the page).
fn divider(t: Tokens) -> Element<'static, Message> {
    container(Space::new())
        .width(Length::Fill)
        .height(Length::Fixed(1.0))
        .style(move |_| container::Style {
            background: Some(iced::Background::Color(t.separator)),
            ..Default::default()
        })
        .into()
}

/// A small button of a row (`.btn-xs` of the page).
fn small_button(t: Tokens, label: &'static str, message: Message) -> Element<'static, Message> {
    button(text(label).size(12))
        .on_press(message)
        .padding([2, 8])
        .style(theme::ghost_button(t))
        .into()
}

/// The chip the rows carry (`.quant-chip`).
fn chip(t: Tokens, label: String) -> Element<'static, Message> {
    container(text(label).size(11).color(t.fg_soft))
        .padding([1, 6])
        .style(theme::chip(t))
        .into()
}

/// The version a row of *Skills with a history* has chosen, the newest when nothing was chosen
/// (the select of the page starts on its first entry, which is the newest).
pub fn chosen_version(skill: &LearnSkill) -> u64 {
    skill.versions.last().map(|v| v.version).unwrap_or(0)
}

/// The label of that version in the select.
pub fn versions_label(skill: &LearnSkill) -> Option<String> {
    skill
        .versions
        .iter()
        .rev()
        .find(|v| v.version == chosen_version(skill))
        .map(|v| format!("v{}: {}, {}", v.version, v.action, when(v.t)))
}

/// The version a label of that select stands for (`v12: revised, 11/1/2026, 9:00:00 AM` → `12`),
/// which is what Roll back sends.
pub fn version_of(label: &str) -> Option<u64> {
    label.strip_prefix('v')?.split(':').next()?.trim().parse().ok()
}

/// A coloured line of a unified diff (`lnDiffHtml`): what was added is green, what went is red.
fn diff_line<'a>(t: Tokens, line: &'a str) -> Element<'a, Message> {
    let colour = if line.starts_with("+ ") {
        t.ok
    } else if line.starts_with("- ") {
        t.danger
    } else {
        t.muted
    };
    text(line).size(11).font(Font::MONOSPACE).color(colour).into()
}

/// Card 1 of the pane: the skills of the user and of a workspace folder.
fn skills_card<'a>(t: Tokens, v: &SkillsView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(10);
    body = body.push(
        text("A skill is a saved procedure (a Markdown file) that the model can read when a task matches it, and write itself after a task that is worth repeating. Skills of the user apply everywhere; skills of a workspace folder are kept in that folder.")
            .size(13)
            .color(t.fg_soft),
    );
    body = body.push(
        column![
            text("Workspace folder (optional)")
                .size(13)
                .color(t.fg)
                .font(Font { weight: iced::font::Weight::Semibold, ..Font::DEFAULT }),
            input(
                t,
                "Absolute path of a project folder, to see its skills",
                v.workspace,
                Message::SkWorkspace,
            ),
        ]
        .spacing(6),
    );
    if !v.list_say.is_empty() {
        body = body.push(text(v.list_say).size(13).color(t.fg_soft));
    } else if v.skills.is_empty() {
        body = body.push(
            text("No skills yet. Write one below, or let the model propose one after a long task.")
                .size(13)
                .color(t.fg_soft),
        );
    } else {
        let mut list = column![].spacing(0);
        for skill in v.skills {
            list = list.push(skill_row(t, skill));
        }
        body = body.push(list);
    }
    if !v.folder_info.is_empty() {
        body = body.push(text(v.folder_info).size(12).color(t.muted));
    }
    card(t, "Skills", body.into())
}

/// One skill of the list: its name with its scope, the buttons of its row and what it is for.
fn skill_row<'a>(t: Tokens, skill: &'a Skill) -> Element<'a, Message> {
    let mut head = row![
        text(skill.name.clone()).size(13).color(t.fg),
        chip(t, skill.scope.clone()),
        Space::new().width(Length::Fill),
    ]
    .spacing(8)
    .align_y(Alignment::Center);
    head = head.push(small_button(t, "Edit", Message::SkEdit(skill.name.clone(), skill.scope.clone())));
    head = head.push(small_button(t, "Delete", Message::SkDelete(skill.name.clone(), skill.scope.clone())));
    let mut description = skill.description.clone();
    if !skill.when.is_empty() {
        description.push_str(&format!(" Use when: {}", skill.when));
    }
    column![
        head,
        text(description).size(12).color(t.fg_soft),
        divider(t),
    ]
    .spacing(4)
    .padding([4, 0])
    .into()
}

/// Card 2 of the pane: what the agent learned and what it wants to change (`#lnSkillCard`).
fn improvements_card<'a>(t: Tokens, v: &SkillsView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(10);
    body = body.push(
        text("The coding agent learns from your Code sessions on its own (switch it off in Settings > Personalization). When it reads a skill, how it went is noted: finished or failed, whether the checks passed, retries, and your thumbs. After a failure, a thumb down, many retries or five uses it revises the skill itself: the previous text is kept, and if one of the next two uses of the revised skill goes badly the previous text is put back. After a complex task that went well it writes a new skill. A skill you wrote or edited is never overwritten: the agent writes a copy named NAME-agent next to it. Every change is listed below and under What I learned with an Undo button. The model that is loaded writes the drafts, so a small model makes small changes.")
            .size(13)
            .color(t.fg_soft),
    );
    if !v.pending.is_empty() {
        body = body.push(text(v.pending).size(12).color(t.fg_soft));
    }
    if !v.last_run.is_empty() {
        body = body.push(text(v.last_run).size(12).color(t.fg_soft));
    }
    body = body.push(
        text("What I learned about skills")
            .size(13)
            .color(t.fg)
            .font(Font { weight: iced::font::Weight::Semibold, ..Font::DEFAULT }),
    );
    if v.feed.is_empty() {
        body = body.push(
            text("Nothing was learned automatically yet.").size(12).color(t.fg_soft),
        );
    } else {
        let mut feed = column![].spacing(0);
        for entry in v.feed {
            feed = feed.push(feed_row(t, entry));
        }
        body = body.push(feed);
    }
    body = body.push(
        text("Skills with a history")
            .size(13)
            .color(t.fg)
            .font(Font { weight: iced::font::Weight::Semibold, ..Font::DEFAULT }),
    );
    if !v.revisions_say.is_empty() {
        body = body.push(text(v.revisions_say).size(12).color(t.fg_soft));
    } else if v.revisions.is_empty() {
        body = body.push(
            text("Nothing to show yet. Switch on \"Learn from this session\" in the Code page: when the agent reads a skill, how it went is noted here, and after a failure or a few uses it can draft a better version.")
                .size(12)
                .color(t.fg_soft),
        );
    } else {
        let mut list = column![].spacing(0);
        for skill in v.revisions.iter().filter(|s| relevant(s)) {
            list = list.push(revision_row(t, skill));
        }
        body = body.push(list);
    }
    body = body.push(
        row![
            button(text("Consolidate now").size(13))
                .on_press(Message::LnConsolidate)
                .padding([6, 12])
                .style(theme::secondary_button(t)),
            text("The same job can run every night: add a task of the kind \"Consolidate memory and skills\" below.")
                .size(12)
                .color(t.fg_soft),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    );
    card(t, "Skill improvements", body.into())
}

/// One entry of *What I learned*: what happened, when, and its Undo button (`lnFeedHtml`).
pub(crate) fn feed_row<'a>(t: Tokens, entry: &'a LearnEntry) -> Element<'a, Message> {
    let mut head = row![text(log_title(entry)).size(13).color(t.fg)]
        .push(Space::new().width(Length::Fill))
        .align_y(Alignment::Center);
    if entry.undone {
        head = head.push(chip(t, "undone".into()));
    } else {
        head = head.push(small_button(t, "Undo", Message::LnUndo(entry.id.clone())));
    }
    let mut description = when(entry.t);
    if let Some(session) = entry.session.as_deref().filter(|s| !s.is_empty()) {
        description.push_str(&format!(" · session {session}"));
    }
    if let Some(reason) = entry.reason.as_deref().filter(|s| !s.is_empty()) {
        description.push_str(&format!(" · {reason}"));
    }
    let mut body = column![head, text(description).size(12).color(t.fg_soft)].spacing(4);
    if let Some(diff) = entry.diff.as_deref().filter(|d| !d.is_empty()) {
        // `<details>` of the page: its changes are drawn open
        let mut lines = column![].spacing(0);
        for line in diff.lines() {
            lines = lines.push(diff_line(t, line));
        }
        body = body.push(
            column![text("Changes").size(12).color(t.fg_soft), lines].spacing(4),
        );
    }
    column![body, divider(t)].spacing(6).padding([4, 0]).into()
}

/// One row of *Skills with a history*: what it is called, how it went, the buttons that act on it
/// and the draft of the model when one waits for a yes.
fn revision_row<'a>(t: Tokens, skill: &'a LearnSkill) -> Element<'a, Message> {
    let mut head = row![
        text(skill.name.clone()).size(13).color(t.fg),
        chip(t, stat_text(skill)),
    ]
    .spacing(8)
    .align_y(Alignment::Center);
    if truthy(&skill.due) && !skill.pending {
        head = head.push(chip(t, "review due".into()));
    }
    head = head.push(Space::new().width(Length::Fill));
    if !skill.pending {
        head = head.push(small_button(
            t,
            "Improve now",
            Message::LnAction(
                "improve".into(),
                skill.name.clone(),
                skill.scope.clone(),
                skill.workspace.clone(),
                None,
            ),
        ));
    }
    if !skill.versions.is_empty() {
        // the page shows the newest first (`versions.slice().reverse()`), and Roll back takes the
        // version that is chosen in the select — the newest when nothing was chosen
        let options: Vec<String> = skill
            .versions
            .iter()
            .rev()
            .map(|v| format!("v{}: {}, {}", v.version, v.action, when(v.t)))
            .collect();
        let chosen = versions_label(skill).unwrap_or_else(|| options[0].clone());
        head = head.push(pick_list(options, Some(chosen), move |label: String| {
            Message::LnVersion(skill.name.clone(), label)
        }).padding([2, 6]));
        head = head.push(small_button(
            t,
            "Roll back",
            Message::LnAction(
                "rollback".into(),
                skill.name.clone(),
                skill.scope.clone(),
                skill.workspace.clone(),
                Some(chosen_version(skill)),
            ),
        ));
    }
    let mut body = column![head].spacing(4);
    if let Some(revision) = &skill.revision {
        let trigger = revision.trigger.clone().unwrap_or_default();
        body = body.push(
            text(format!("Draft revision ({trigger}): {}", revision.reason))
                .size(12)
                .color(t.fg_soft),
        );
        let mut lines = column![].spacing(0);
        for line in revision.diff.lines() {
            lines = lines.push(diff_line(t, line));
        }
        body = body.push(lines);
        body = body.push(
            row![
                button(text("Approve").size(12))
                    .on_press(Message::LnAction(
                        "approve".into(),
                        skill.name.clone(),
                        skill.scope.clone(),
                        skill.workspace.clone(),
                        None,
                    ))
                    .padding([3, 10])
                    .style(theme::primary_button(t)),
                small_button(
                    t,
                    "Reject",
                    Message::LnAction(
                        "reject".into(),
                        skill.name.clone(),
                        skill.scope.clone(),
                        skill.workspace.clone(),
                        None,
                    ),
                ),
                text("The skill stays as it is until you approve.").size(12).color(t.fg_soft),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        );
    }
    if !skill.recent.is_empty() {
        // `<details>` of the page: its uses are drawn open
        let mut uses = column![text("Recent uses").size(12).color(t.fg_soft)].spacing(2);
        for use_of in &skill.recent {
            let mut line = format!("{} · {}", when(use_of.t), use_of.outcome);
            if let Some(feedback) = use_of.feedback.as_deref().filter(|f| !f.is_empty()) {
                line.push_str(&format!(" · thumbs {feedback}"));
            }
            if use_of.verify == Some(false) {
                line.push_str(" · checks failed");
            }
            if let Some(note) = use_of.note.as_deref().filter(|n| !n.is_empty()) {
                line.push_str(&format!(" · {note}"));
            }
            uses = uses.push(text(line).size(12).color(t.fg_soft));
        }
        body = body.push(uses);
    }
    column![body, divider(t)].spacing(6).padding([4, 0]).into()
}

/// Card 3 of the pane: the form that writes a skill (`Write a skill`).
fn write_card<'a>(t: Tokens, v: &SkillsView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(10);
    body = body.push(
        row![
            input(t, "name (lowercase, dashes)", v.name, Message::SkName),
            pick_list(
                ["For every project", "This workspace folder"],
                Some(if v.scope == "workspace" { "This workspace folder" } else { "For every project" }),
                |label: &'static str| Message::SkScope(if label.starts_with("This") { "workspace" } else { "user" }),
            )
            .padding([6, 10]),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    );
    body = body.push(
        text_editor(v.text)
            .on_action(Message::SkText)
            .placeholder("---")
            .padding(10)
            .size(13)
            .font(Font::MONOSPACE)
            .height(Length::Fixed(220.0))
            .style(editor_style(t)),
    );
    body = body.push(
        row![
            button(text("Save skill").size(13))
                .on_press(Message::SkSave)
                .padding([6, 12])
                .style(theme::primary_button(t)),
            button(text("New from template").size(13))
                .on_press(Message::SkNew)
                .padding([6, 12])
                .style(theme::ghost_button(t)),
        ]
        .spacing(6),
    );
    card(t, "Write a skill", body.into())
}

/// Card 4 of the pane: what runs on a clock (`Scheduled tasks`).
fn tasks_card<'a>(t: Tokens, v: &SkillsView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(10);
    // one paragraph with its bold sentence in the middle (`<strong>` on the page)
    body = body.push(rich_text(tasks_help_spans()).size(13).color(t.fg_soft));
    if !v.schedule_note.is_empty() {
        body = body.push(text(v.schedule_note).size(12).color(t.fg_soft));
    }
    if !v.tasks_say.is_empty() {
        body = body.push(text(v.tasks_say).size(13).color(t.fg_soft));
    } else if v.tasks.is_empty() {
        body = body.push(text("No scheduled tasks yet.").size(13).color(t.fg_soft));
    } else {
        let mut list = column![].spacing(0);
        for task in v.tasks {
            list = list.push(task_row(t, task, v.results.get(&task.id)));
        }
        body = body.push(list);
    }
    card(t, "Scheduled tasks", body.into())
}

/// One task of the list: what it is called, when it runs, and the four buttons of its row.
fn task_row<'a>(
    t: Tokens,
    task: &'a Task,
    results: Option<&'a ResultsState>,
) -> Element<'a, Message> {
    let mut head = row![text(task.name.clone()).size(13).color(t.fg)]
        .spacing(6)
        .align_y(Alignment::Center);
    head = head.push(chip(t, task.when.clone()));
    head = head.push(chip(t, mode_chip(&task.mode).to_string()));
    if !task.deliver.is_empty() {
        head = head.push(chip(t, deliver_chip(&task.deliver)));
    }
    if !task.enabled {
        head = head.push(chip(t, "off".into()));
    }
    if task.running {
        head = head.push(text("running").size(12).color(t.primary));
    }
    head = head.push(Space::new().width(Length::Fill));
    head = head.push(small_button(t, "Run now", Message::ScAction("run".into(), task.id.clone())));
    head = head.push(small_button(t, "Results", Message::ScAction("results".into(), task.id.clone())));
    head = head.push(small_button(t, "Edit", Message::ScAction("edit".into(), task.id.clone())));
    head = head.push(small_button(t, "Delete", Message::ScAction("remove".into(), task.id.clone())));

    let mut description = task_runs_line(task);
    if task.mode == "agent" {
        description.push_str(&format!(" Folder: {}", task.folder));
    }
    let mut body = column![head, text(description).size(12).color(t.fg_soft)].spacing(4);

    // the answers of the task, in the box the Results button toggles
    if let Some(state) = results.filter(|state| state.shown) {
        match &state.filled {
            None => {}
            Some(Err(reason)) => body = body.push(text(reason.clone()).size(12).color(t.fg_soft)),
            Some(Ok(list)) if list.is_empty() => {
                body = body.push(text("No results yet.").size(12).color(t.fg_soft));
            }
            Some(Ok(list)) => {
                let mut answers = column![].spacing(6);
                for answer in list {
                    answers = answers.push(
                        column![
                            text(result_line(answer))
                            .size(12)
                            .color(t.fg),
                            text(answer.text.clone()).size(12).font(Font::MONOSPACE).color(t.fg_soft),
                        ]
                        .spacing(2),
                    );
                }
                body = body.push(answers);
            }
        }
    }
    column![body, divider(t)].spacing(6).padding([4, 0]).into()
}

/// `models::card` with a title that is built on the spot: the task form changes between *New
/// task* and *Edit the task*, so its title is owned instead of borrowed from the page.
fn titled_card<'a>(t: Tokens, title: String, body: Element<'a, Message>) -> Element<'a, Message> {
    container(column![text(title).size(13).color(t.fg_soft), body].spacing(12))
        .padding(16)
        .width(Length::Fill)
        .style(theme::card(t, theme::R_LG, false))
        .into()
}

/// One field of the task form (`app-card-item column` of the page).
fn field<'a>(t: Tokens, title: &'a str, control: Element<'a, Message>) -> Element<'a, Message> {
    column![
        text(title)
            .size(13)
            .color(t.fg)
            .font(Font { weight: iced::font::Weight::Semibold, ..Font::DEFAULT }),
        control
    ]
    .spacing(6)
    .into()
}

/// One of the boxes of the task form (`textarea`).
fn area<'a>(
    t: Tokens,
    content: &'a text_editor::Content,
    placeholder: &'a str,
    action: fn(text_editor::Action) -> Message,
    height: f32,
) -> Element<'a, Message> {
    text_editor(content)
        .on_action(action)
        .placeholder(placeholder)
        .padding(8)
        .size(13)
        .font(Font::MONOSPACE)
        .height(Length::Fixed(height))
        .style(editor_style(t))
        .into()
}

/// A switch of these cards (`ui-switch` of the page): the title and the two sides.
fn switch<'a>(
    t: Tokens,
    title: &'a str,
    on: bool,
    message: fn(bool) -> Message,
) -> Element<'a, Message> {
    row![
        text(title).size(13).color(t.fg),
        Space::new().width(Length::Fill),
        crate::settings::segmented(
            t,
            vec![("On", on, message(true)), ("Off", !on, message(false))],
        ),
    ]
    .spacing(8)
    .align_y(Alignment::Center)
    .into()
}

/// Card 5 of the pane: the form that writes a task (`New task`, *Edit the task* while one is
/// chosen).
fn task_form_card<'a>(t: Tokens, v: &SkillsView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(10);
    body = body.push(field(t, "Name", input(t, "", v.task_name, Message::ScName)));
    body = body.push(field(
        t,
        "Prompt",
        area(
            t,
            v.task_prompt,
            "Summarise the open issues in this folder's TODO.md",
            Message::ScPrompt,
            92.0,
        ),
    ));
    body = body.push(
        row![
            input(t, "daily at 08:30", v.task_when, Message::ScWhen),
            pick_list(
                WHEN_PRESETS,
                Some("Examples"),
                |preset: &'static str| Message::ScPreset(preset),
            )
            .padding([6, 10]),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    );
    if !v.task_preview.is_empty() {
        body = body.push(text(v.task_preview).size(12).color(t.fg_soft));
    }
    body = body.push(
        text("Also accepted: \"every 2 hours\", \"hourly at :15\", or a cron line (minute hour day month weekday). Local time.")
            .size(12)
            .color(t.fg_soft),
    );
    body = body.push(field(
        t,
        "Kind",
        pick_list(
            MODES.iter().map(|(_, label)| *label).collect::<Vec<_>>(),
            mode_label(v.task_mode),
            |label: &'static str| Message::ScMode(mode_id(label).to_string()),
        )
        .padding([6, 10])
        .into(),
    ));
    if v.task_mode == "agent" || v.task_mode == "learn" {
        body = body.push(field(t, "Folder", input(t, "Absolute path", v.task_folder, Message::ScFolder)));
    }
    body = body.push(field(
        t,
        "Also send the result to",
        input(
            t,
            "webhook:https://... | ntfy:https://ntfy.host/topic | telegram:CHAT_ID",
            v.task_deliver,
            Message::ScDeliver,
        ),
    ));
    body = body.push(
        text("Empty: only here. Telegram needs the bot token in FLLM_TELEGRAM_TOKEN on the server; the token is never stored in the task. The last result shows whether the delivery worked.")
            .size(12)
            .color(t.fg_soft),
    );
    body = body.push(switch(t, "Desktop notification when a result is ready", v.task_notify, Message::ScNotify));
    body = body.push(switch(t, "Enabled", v.task_enabled, Message::ScEnabled));
    body = body.push(
        row![
            button(text("Save task").size(13))
                .on_press(Message::ScSave)
                .padding([6, 12])
                .style(theme::primary_button(t)),
            button(text("Clear").size(13))
                .on_press(Message::ScClear)
                .padding([6, 12])
                .style(theme::ghost_button(t)),
        ]
        .spacing(6),
    );
    let title = if v.task_editing { "Edit the task".into() } else { "New task".into() };
    titled_card(t, title, body.into())
}

/// Card 6 of the pane: what comes with the app, and the box that imports one from a URL
/// (`skilllibrary.js` appends this card to the pane).
fn library_card<'a>(t: Tokens, v: &SkillsView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(10);
    body = body.push(
        text("One click writes them into your skills folder — as a normal file you can change afterwards.")
            .size(13)
            .color(t.fg_soft),
    );
    if !v.library_list_say.is_empty() {
        // a read that failed writes its sentence where the list was
        body = body.push(text(v.library_list_say).size(13).color(t.fg_soft));
    } else if v.library.is_empty() {
        body = body.push(text("The library is empty.").size(13).color(t.fg_soft));
    } else {
        let mut list = column![].spacing(0);
        for skill in v.library {
            list = list.push(library_row(t, skill));
        }
        body = body.push(list);
    }
    body = body.push(
        row![
            input(
                t,
                "https://raw.githubusercontent.com/you/repo/main/SKILL.md",
                v.import_url,
                Message::LibUrl,
            ),
            button(text("Import from URL").size(13))
                .on_press(Message::LibImport)
                .padding([6, 12])
                .style(theme::secondary_button(t)),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    );
    if !v.library_say.is_empty() {
        body = body.push(text(v.library_say).size(11).font(Font::MONOSPACE).color(t.fg_soft));
    }
    card(t, "Skills that come with the app", body.into())
}

/// One skill of the library: what it is for, and the button that writes it into the folder.
fn library_row<'a>(t: Tokens, skill: &'a LibrarySkill) -> Element<'a, Message> {
    let mut head = row![text(skill.name.clone()).size(12).color(t.fg)]
        .push(Space::new().width(Length::Fill))
        .align_y(Alignment::Center);
    let action: Element<'a, Message> = if skill.installed {
        // the button of the page is there and says *installed*, but does nothing (no `on_press`)
        button(text("installed").size(12))
            .padding([2, 8])
            .style(theme::ghost_button(t))
            .into()
    } else {
        button(text("Install").size(12))
            .on_press(Message::LibInstall(skill.name.clone()))
            .padding([2, 8])
            .style(theme::secondary_button(t))
            .into()
    };
    head = head.push(action);
    column![
        head,
        text(skill.description.clone()).size(12).color(t.fg_soft),
        text(format!("when: {}", skill.when)).size(11).color(t.fg_soft),
        divider(t),
    ]
    .spacing(4)
    .padding([4, 0])
    .into()
}

/// The description of *Find skills on GitHub*, with its link to the directory (`#skFindWeb`).
pub const FINDER_HELP: &str = "Skills are folders with a SKILL.md in a GitHub repository, for example anthropics/skills or vercel-labs/skills. Type the repository (or paste the command of a skills directory) to see what it offers. A skill is a text file that tells the assistant how to do something: read it before you install it. Browse the directory at skills.sh";

fn finder_spans(link_colour: Option<iced::Color>) -> Vec<Span<'static, String>> {
    let browse = Span::new("Browse the directory at skills.sh");
    vec![
        Span::new("Skills are folders with a SKILL.md in a GitHub repository, for example anthropics/skills or vercel-labs/skills. Type the repository (or paste the command of a skills directory) to see what it offers. A skill is a text file that tells the assistant how to do something: read it before you install it. "),
        Span {
            link: Some("https://www.skills.sh/".into()),
            underline: true,
            color: link_colour,
            ..browse
        },
    ]
}

/// Card 7 of the pane: the skills of a repository and the directory of skills.sh
/// (`skilllibrary.js` appends this card after the one above it).
fn finder_card<'a>(t: Tokens, v: &SkillsView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(10);
    body = body.push(rich_text(finder_spans(Some(t.primary))).size(13).color(t.fg_soft));
    body = body.push(
        row![
            input(t, "owner/repo", v.repo, Message::RepoQuery),
            button(text("Show skills").size(13))
                .on_press(Message::RepoGo)
                .padding([6, 12])
                .style(theme::secondary_button(t)),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    );
    for skill in v.github {
        // the row shows what the repository says about it, or where the file stands (`k.description
        // || k.path`)
        let detail = if skill.description.is_empty() {
            skill.path.clone()
        } else {
            skill.description.clone()
        };
        body = body.push(installable_row(
            t,
            &skill.name,
            &detail,
            skill.installed,
            Message::GithubInstall(skill.url.clone(), skill.name.clone()),
        ));
    }
    for skill in v.directory {
        let mut detail = skill.id.clone();
        let installs = installs_text(&skill.installs);
        if !installs.is_empty() {
            detail.push_str(&format!(" · {installs} installs"));
        }
        body = body.push(installable_row(
            t,
            &skill.name,
            &detail,
            skill.installed,
            Message::DirInstall(skill.id.clone()),
        ));
    }

    // `<details>` of the page: the directory is drawn open
    body = body.push(
        column![
            text("Search the skills.sh directory from here (needs a Vercel token)")
                .size(12)
                .color(t.fg_soft),
            text("skills.sh lets you search and see the most installed skills when you send a token of a Vercel project with OIDC Federation turned on (vercel env pull writes it as VERCEL_OIDC_TOKEN). The token is saved on this computer only, never shown again, and expires after about 12 hours. Without it, the GitHub search above does the same job for a repository you name.")
                .size(12)
                .color(t.fg_soft),
            row![
                input(t, "Vercel token", v.dir_token, Message::DirToken),
                button(text("Save token").size(13))
                    .on_press(Message::DirTokenSave)
                    .padding([6, 12])
                    .style(theme::secondary_button(t)),
                button(text("Remove it").size(13))
                    .on_press(Message::DirTokenRemove)
                    .padding([6, 12])
                    .style(theme::ghost_button(t)),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
            row![
                input(t, "Search skills (empty: the most installed)", v.directory_q, Message::DirQuery),
                pick_list(
                    ["Most installed", "Trending", "Hot"],
                    Some(view_label(v.dir_view)),
                    |label: &'static str| Message::DirView(
                        VIEWS
                            .iter()
                            .find(|(_, l)| *l == label)
                            .map(|(id, _)| id.to_string())
                            .unwrap_or_else(|| "all-time".into()),
                    ),
                )
                .padding([6, 10]),
                button(text("Search").size(13))
                    .on_press(Message::DirGo)
                    .padding([6, 12])
                    .style(theme::secondary_button(t)),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        ]
        .spacing(8),
    );
    if !v.repo_say.is_empty() {
        body = body.push(text(v.repo_say).size(11).font(Font::MONOSPACE).color(t.fg_soft));
    }
    card(t, "Find skills on GitHub", body.into())
}

/// One row of a list that offers an Install button — of a repository, of the directory or of the
/// library (the page writes all three the same way).
fn installable_row<'a>(
    t: Tokens,
    name: &str,
    detail: &str,
    installed: bool,
    install: Message,
) -> Element<'a, Message> {
    let name = name.to_string();
    let detail = detail.to_string();
    let mut head = row![text(name).size(12).color(t.fg)]
        .push(Space::new().width(Length::Fill))
        .align_y(Alignment::Center);
    let action: Element<'a, Message> = if installed {
        button(text("installed").size(12)).padding([2, 8]).style(theme::ghost_button(t)).into()
    } else {
        button(text("Install").size(12))
            .on_press(install)
            .padding([2, 8])
            .style(theme::secondary_button(t))
            .into()
    };
    head = head.push(action);
    column![head, text(detail).size(12).color(t.fg_soft), divider(t)]
        .spacing(4)
        .padding([4, 0])
        .into()
}

/// The seven cards of `spane-skills`, in the order the page builds them.
pub fn section<'a>(t: Tokens, v: SkillsView<'a>) -> Element<'a, Message> {
    column![
        skills_card(t, &v),
        improvements_card(t, &v),
        write_card(t, &v),
        tasks_card(t, &v),
        task_form_card(t, &v),
        library_card(t, &v),
        finder_card(t, &v),
    ]
    .spacing(14)
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule of a skill name, and a title turned into one (`skValidName`, `skSlug`).
    #[test]
    fn a_name_is_lowercase_with_dashes_and_at_most_64_characters() {
        assert!(valid_name("my-skill"));
        assert!(valid_name("skill_2"));
        assert!(valid_name("9lives"));
        assert!(!valid_name(""), "empty");
        assert!(!valid_name("-start"), "the first character has to be a letter or a digit");
        assert!(!valid_name("Upper"), "uppercase");
        assert!(!valid_name("with space"));
        assert!(!valid_name(&"a".repeat(65)), "64 at most");

        assert_eq!(slug("My Skill!"), "my-skill");
        assert_eq!(slug("  --Hello, World--  "), "hello-world");
        assert_eq!(slug("___"), "", "no name character is a dash, and a dash alone is nothing");
        assert_eq!(slug(&"x".repeat(100)), "x".repeat(64), "capped at 64");
        assert_eq!(slug(&format!("{}-", "x".repeat(70))), "x".repeat(64), "no dash at the end of a cap");
    }

    /// The front matter of a `SKILL.md` and the text that builds one back (`skParse`, `skCompose`).
    #[test]
    fn the_front_matter_round_trips_through_parse_and_compose() {
        let text = "---\nname: my-skill\ndescription: \"What it does\"\nwhen: When to use it\n---\n1. Step\n";
        let parsed = parse(text);
        assert_eq!(parsed.meta.get("name").map(String::as_str), Some("my-skill"));
        assert_eq!(parsed.meta.get("description").map(String::as_str), Some("What it does"), "the quotes at the ends are taken off");
        assert_eq!(parsed.meta.get("when").map(String::as_str), Some("When to use it"));
        assert_eq!(parsed.body, "1. Step\n", "the body starts after the second ---");

        // a file without front matter is body from the first character
        let plain = parse("# Title\n1. Step\n");
        assert!(plain.meta.is_empty());
        assert_eq!(plain.body, "# Title\n1. Step\n");

        // a byte order mark at the start is not part of the text
        assert!(parse("\u{feff}---\nname: bom\n---\nx").meta.contains_key("name"));

        // keys are lowercased, and a line without a colon is no field
        let odd = parse("---\nNAME: kept\n---\nnote");
        assert_eq!(odd.meta.get("name").map(String::as_str), Some("kept"));

        let composed = compose("my-skill", "What\n  it does", "When to use it", "1. Step");
        assert_eq!(
            composed,
            "---\nname: my-skill\ndescription: What it does\nwhen: When to use it\n---\n1. Step\n",
            "a description over two lines becomes one, and the body ends in a newline"
        );
    }

    /// *New from template* writes the file the Studio writes (`skTemplate`).
    #[test]
    fn the_template_is_the_one_of_the_page() {
        let text = template();
        assert_eq!(
            text,
            "---\nname: my-skill\ndescription: What this skill does, in one sentence\nwhen: When to use it\n---\n# Title\n\n1. First step\n2. Second step\n\n## Pitfalls\n- \n"
        );
        assert_eq!(parse(&text).meta.get("name").map(String::as_str), Some("my-skill"));
    }

    /// What waits for a decision, and what the last step wrote (`lnPendingText`,
    /// `lnRefreshCounters`).
    #[test]
    fn the_counters_say_what_waits_and_what_last_ran() {
        let overview: LearnOverview = serde_json::from_value(serde_json::json!({
            "skills": [{"name": "a"}], "memory": [], "profile": [{"id": 1}, {"id": 2}],
            "running": true,
            "last_run": {"lines": ["first", "second", "third", "fourth", "fifth"]}
        }))
        .expect("an answer of the learn overview");
        assert_eq!(
            pending_text(&overview),
            "Waiting for a decision: 1 skill revision, 2 profile lines. A learning step is running now."
        );
        assert_eq!(
            last_run_text(&overview),
            "Last learning step (5): first second third fourth",
            "four lines of the step are shown"
        );

        let nothing = LearnOverview::default();
        assert_eq!(pending_text(&nothing), "Nothing waits for a decision.");
        assert_eq!(last_run_text(&nothing), "", "a step that never ran writes nothing");
    }

    /// The title of a change of the log, and the row that has something to show
    /// (`lnLogTitle`, `lnSkillRelevant`).
    #[test]
    fn a_change_is_named_by_what_it_did() {
        let entry = LearnEntry {
            kind: "skill_revise".into(),
            name: Some("my-skill".into()),
            ..Default::default()
        };
        assert_eq!(log_title(&entry), "Revised the skill \"my-skill\"");
        let created = LearnEntry { kind: "skill_create".into(), name: Some("new".into()), ..Default::default() };
        assert_eq!(log_title(&created), "Wrote the skill \"new\"");
        let back = LearnEntry { kind: "skill_rollback".into(), name: Some("old".into()), ..Default::default() };
        assert_eq!(log_title(&back), "Put back the previous text of \"old\"");
        let other = LearnEntry { kind: "skill_other".into(), ..Default::default() };
        assert_eq!(log_title(&other), "skill_other", "an unknown kind is written as it is");

        assert!(!relevant(&LearnSkill::default()), "nothing to show yet");
        assert!(relevant(&LearnSkill { uses: 1, ..Default::default() }), "used once");
        assert!(relevant(&LearnSkill { pending: true, ..Default::default() }), "a draft waits");
        assert!(
            relevant(&LearnSkill { versions: vec![Version { version: 1, ..Default::default() }], ..Default::default() }),
            "an older text is kept"
        );
    }

    /// What a row says about itself (`lnStatText`): its uses and every number beside them.
    #[test]
    fn the_row_counts_uses_failures_thumbs_and_retries() {
        assert_eq!(stat_text(&LearnSkill { uses: 5, ..Default::default() }), "5 uses");
        assert_eq!(stat_text(&LearnSkill { uses: 1, ..Default::default() }), "1 use");
        let all = LearnSkill { uses: 5, failed: 1, up: 2, down: 0, retries: 3, ..Default::default() };
        assert_eq!(stat_text(&all), "5 uses, 1 failed, thumbs 2 up / 0 down, 3 retries");
        // a thumb with no failure is still a number worth showing
        assert_eq!(
            stat_text(&LearnSkill { down: 1, ..Default::default() }),
            "0 uses, thumbs 0 up / 1 down"
        );
    }

    /// The chips of a task and the line under it (`mode_chip`, `deliver_chip`, `task_runs_line`).
    #[test]
    fn a_task_is_told_by_its_chips_and_its_runs() {
        assert_eq!(mode_chip("agent"), "agent");
        assert_eq!(mode_chip("learn"), "consolidate", "consolidating is called that on the page");
        assert_eq!(mode_chip("chat"), "chat");
        assert_eq!(mode_chip("something else"), "chat");
        assert_eq!(deliver_chip(""), "", "nothing to send to is no chip");
        assert_eq!(deliver_chip("telegram:1234"), "telegram", "the first part before the colon");
        assert_eq!(deliver_chip("webhook:https://x/y"), "webhook");

        let task = Task { name: "morning".into(), when: "daily at 08:30".into(), ..Default::default() };
        assert_eq!(task_runs_line(&task), "Has not run yet.");
        let with_next = Task { next_run: Some(1_700_000_000), ..task.clone() };
        assert!(
            task_runs_line(&with_next).starts_with("Next run: "),
            "{}",
            task_runs_line(&with_next)
        );
        let ran = Task {
            last: Some(TaskRun { time: 1_700_000_000, ok: true, delivery: None }),
            ..with_next.clone()
        };
        assert!(
            task_runs_line(&ran).ends_with("(ok)."),
            "{}",
            task_runs_line(&ran)
        );
        let refused = Task {
            last: Some(TaskRun { time: 1_700_000_000, ok: false, delivery: Some("401".into()) }),
            ..ran
        };
        assert!(
            task_runs_line(&refused).ends_with("(failed; delivery 401)."),
            "{}",
            task_runs_line(&refused)
        );
    }

    /// The line of one answer under its task (`scShowResults`).
    #[test]
    fn an_answer_says_when_it_ran_how_it_went_and_how_long_it_took() {
        assert_eq!(
            result_line(&TaskResult {
                time: 1_699_990_000,
                ok: true,
                seconds: 1.5,
                text: String::new(),
            }),
            format!("{} - ok (1.5s)", when(1_699_990_000))
        );
        assert_eq!(
            result_line(&TaskResult {
                time: 0,
                ok: false,
                seconds: 12.0,
                text: String::new(),
            }),
            "1/1/1970, 12:00:00 AM - failed (12s)",
            "a whole number of seconds is written without a dot, as JavaScript writes it"
        );
    }

    /// A number of installs is written the way the page writes it (`toLocaleString`).
    #[test]
    fn the_installs_of_a_skill_are_grouped_by_thousands() {
        assert_eq!(installs_text(&None), "", "a value that is not there is no number");
        assert_eq!(installs_text(&Some(serde_json::json!(12))), "12");
        assert_eq!(installs_text(&Some(serde_json::json!(1234))), "1,234");
        assert_eq!(installs_text(&Some(serde_json::json!(1234567))), "1,234,567");
        assert_eq!(installs_text(&Some(serde_json::json!("1234"))), "1,234", "a number as text counts");
        assert_eq!(installs_text(&Some(serde_json::json!("many"))), "NaN", "JavaScript writes NaN");
    }

    /// What the page counts as there, and the version a label of the select stands for.
    #[test]
    fn the_truth_and_the_version_of_a_label() {
        assert!(!truthy(&None));
        assert!(!truthy(&Some(serde_json::Value::Null)));
        assert!(!truthy(&Some(serde_json::json!(false))));
        assert!(!truthy(&Some(serde_json::json!(0))));
        assert!(!truthy(&Some(serde_json::json!(""))));
        assert!(truthy(&Some(serde_json::json!("in 2 days"))));
        assert!(truthy(&Some(serde_json::json!(1))));

        let skill = LearnSkill {
            name: "a".into(),
            versions: vec![
                Version { version: 3, action: "revised".into(), t: 1_700_000_000 },
                Version { version: 4, action: "revised again".into(), t: 1_700_000_100 },
            ],
            ..Default::default()
        };
        assert_eq!(chosen_version(&skill), 4, "the newest version is the one the select starts on");
        let label = versions_label(&skill).expect("a label for it");
        assert!(label.starts_with("v4: revised again"), "{label}");
        assert_eq!(version_of(&label), Some(4));
        assert_eq!(version_of("no version"), None);
    }

    /// The two sentences that are drawn as `rich_text`, held to the page here because the window
    /// cannot look inside one.
    #[test]
    fn the_rich_sentences_of_the_card_are_the_ones_of_the_page() {
        assert_eq!(
            tasks_help_spans()
                .into_iter()
                .map(|s| s.text.to_string())
                .collect::<Vec<_>>()
                .join(""),
            TASKS_HELP
        );
        let links: Vec<String> = finder_spans(None)
            .into_iter()
            .filter_map(|s| s.link.clone())
            .collect();
        assert_eq!(links, vec!["https://www.skills.sh/".to_string()]);
        assert_eq!(
            finder_spans(None)
                .into_iter()
                .map(|s| s.text.to_string())
                .collect::<Vec<_>>()
                .join(""),
            FINDER_HELP
        );
    }

    /// The line under the list names both folders (`skFolderInfo`).
    #[test]
    fn the_line_under_the_list_names_both_folders() {
        let answer: SkillsAnswer = serde_json::from_value(serde_json::json!({
            "skills": [],
            "user_folder": "/home/m/.fllm/skills"
        }))
        .expect("an answer of the skills");
        assert_eq!(
            folder_info(&answer),
            "Your skills are saved in /home/m/.fllm/skills. Workspace skills are saved in <folder>/.fllm/skills/<name>/SKILL.md."
        );
    }
}


// ---- `/skill`: the model writes a skill from the session (`skTranscript`, `skProposalPrompt`, `skParseProposal`) ----

/// A message of the session as the transcript reads it: the role, the text and the calls it made.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionMessage {
    pub role: String,
    pub content: String,
    /// The name and the arguments of every call.
    pub calls: Vec<(String, String)>,
}

/// The session as plain text for the model (`skTranscript`): each message clipped, the middle left out when it is long.
pub fn transcript(messages: &[SessionMessage], max_chars: usize) -> String {
    use crate::sessions::{utf16_len, utf16_slice};
    let mut lines = Vec::new();
    for m in messages {
        if !matches!(m.role.as_str(), "user" | "assistant" | "tool") {
            continue;
        }
        let calls = m.calls.iter().map(|(name, args)| format!("{name}({})", utf16_slice(args, 0, 200))).collect::<Vec<_>>().join(", ");
        if m.content.trim().is_empty() && calls.is_empty() {
            continue;
        }
        let len = utf16_len(&m.content);
        let clipped = if len > 1200 { format!("{} ... {}", utf16_slice(&m.content, 0, 800), utf16_slice(&m.content, len - 300, len)) } else { m.content.clone() };
        let role = if m.role == "tool" { "TOOL RESULT".to_string() } else { m.role.to_uppercase() };
        lines.push(format!("{role}: {clipped}{}", if calls.is_empty() { String::new() } else { format!("\n[calls: {calls}]") }));
    }
    let all = lines.join("\n\n");
    let len = utf16_len(&all);
    if len <= max_chars {
        return all;
    }
    let keep = max_chars / 2;
    format!("{}\n\n[... the middle of the session is left out ...]\n\n{}", utf16_slice(&all, 0, keep), utf16_slice(&all, len - keep, len))
}

/// The instruction that asks the model to turn a finished session into a skill (`skProposalPrompt`).
pub fn proposal_prompt(transcript: &str) -> String {
    [
        "Below is a session in which a task was completed. Turn what was learned into a reusable skill: a short procedure that would let someone do the same kind of task again without the discovery work.",
        "",
        "Answer with the SKILL.md file only, in exactly this form and nothing else:",
        "---",
        "name: short-lowercase-name-with-dashes",
        "description: one sentence saying what the skill does",
        "when: one sentence saying when to use it",
        "---",
        "# Title",
        "",
        "1. Numbered steps with exact commands, file names and settings that worked.",
        "",
        "## Pitfalls",
        "- Things that failed and what to do instead.",
        "",
        "Leave out anything specific to this one run (dates, temporary paths, the names of people). If the session holds nothing reusable, answer with the single word NONE.",
        "",
        "=== SESSION ===",
        transcript,
        "=== END OF SESSION ===",
    ]
    .join("\n")
}

/// The skill the model wrote: its name and the text of its `SKILL.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    pub name: String,
    pub content: String,
}

/// `skParseProposal`: the proposal in the reply of the model, or the line that says why there is none.
pub fn parse_proposal(reply: &str) -> Result<Proposal, String> {
    let think = regex::Regex::new(r"(?s)<think>.*?</think>").expect("think");
    let cleaned = think.replace_all(reply, "");
    let mut t = cleaned.trim().to_string();
    let none = t.eq_ignore_ascii_case("none") || t.eq_ignore_ascii_case("none.");
    if t.is_empty() || none {
        return Err("The model found nothing reusable in this session.".into());
    }
    let fenced = regex::Regex::new(r"(?is)^```[a-z]*\n(.*?)\n```\s*$").expect("fence");
    if let Some(c) = fenced.captures(&t) {
        t = c[1].trim().to_string();
    }
    // a line of talk before the file
    if let Some(at) = t.find("---") {
        if at > 0 {
            t = t[at..].to_string();
        }
    }
    let parsed = parse(&t);
    let description = parsed.meta.get("description").cloned().unwrap_or_default();
    if description.is_empty() || parsed.body.trim().is_empty() {
        return Err("The model did not write a skill in the expected form.".into());
    }
    let named = parsed.meta.get("name").cloned().unwrap_or_default();
    let name = if valid_name(&named) {
        named
    } else {
        let source = if named.is_empty() { description.split(' ').take(5).collect::<Vec<_>>().join(" ") } else { named };
        slug(&source)
    };
    if !valid_name(&name) {
        return Err("The skill has no usable name.".into());
    }
    Ok(Proposal { name, content: format!("{}\n", t.trim_end()) })
}
