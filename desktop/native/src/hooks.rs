//! The **Hooks** card: shell commands at fixed points of the agent loop.
//!
//! Port of `hooksSettings` in `rust/engine/src/webui/hooks.js` (the whole file) — its eleven
//! events with their hints, the rows of the draft, the two lines of the answer of a test and the
//! hooks an installed plugin brings.

use iced::{
    Font, Length,
    alignment::Alignment,
    widget::{button, column, row, text, text_input, Space},
    Element,
};
use serde::Deserialize;

use crate::app::Message;
use crate::models::card;
use crate::theme::{self, Tokens};

/// The events of the loop in the order the page draws them: the name, the title over its rows and
/// the hint under it (`EVENTS` of `hooks.js`, verbatim).
pub const EVENTS: [(&str, &str, &str); 11] = [
    (
        "pre_tool_call",
        "Before a tool call",
        "Can block the call. Good for: a guard that refuses writes to protected paths, a check that logs every command.",
    ),
    (
        "post_tool_call",
        "After a tool call",
        "Sees the tool result in its arguments only; its output is added to the result. Good for: recording what ran.",
    ),
    (
        "on_file_edit",
        "After a file was changed",
        "Runs only when an edit succeeded. Exit 2 reports a problem without undoing the edit. Good for: a formatter, a linter, a build check.",
    ),
    (
        "on_session_start",
        "When a workspace is opened",
        "Runs when the agent switches folder. Output is shown once in the Code page. Good for: preparing an environment, printing a reminder.",
    ),
    (
        "on_session_end",
        "When the session ends",
        "Runs when the agent session is closed. Good for: cleanup, a summary of the work.",
    ),
    (
        "user_prompt_submit",
        "When you send a message",
        "Can block the message (exit 2) or add notes the agent sees with it. Good for: adding the ticket number, refusing prompts with secrets.",
    ),
    (
        "stop",
        "When the agent is about to finish",
        "Exit 2 (or a block decision) sends the reason back to the agent as the next instruction, once. Good for: \"run the tests before you finish\".",
    ),
    (
        "subagent_stop",
        "When a sub-agent finishes",
        "Runs when a sub-agent has reported back. Good for: logging, checking its result.",
    ),
    (
        "notification",
        "When the agent waits for you",
        "Runs when a permission question or a choice is waiting. Good for: a sound, a desktop notice, a message to your phone.",
    ),
    (
        "pre_compact",
        "Before the conversation is compacted",
        "Runs before older turns are summarised. Good for: saving the transcript.",
    ),
    (
        "file_changed",
        "When a watched file changes",
        "Fired by fllm-rs agent --watch. The matcher is a list of globs (src/**/*.rs, *.md); \"on\" limits it to created, modified or deleted. With a prompt, the agent starts a run with that text; {{file}} is the path.",
    ),
];

/// The two paragraphs over the list (`spane-hooks` of the page). The file names and the JSON in
/// them are in `mono` on the page; the window draws the lines the way the other cards here draw
/// theirs.
pub const DESC1: &str = "Shell commands that run when the agent reaches a point in its loop. They are the deterministic half of the loop: a prompt asks the model to run the formatter, a hook makes sure it happened. A hook gets one JSON object on stdin ({event, tool, arguments, workspace}) and runs in the workspace folder with your rights.";
pub const DESC2: &str = "Exit code 0 passes and adds the output to the tool result as notes, 2 blocks the call (stderr becomes the reason the model sees), any other code is reported as a failed hook. A hook may also print {\"decision\":\"block\",\"reason\":\"...\"} or {\"context\":\"...\"}.";

/// One hook of an event as the page reads it (`{matcher, command, prompt, on, timeout_seconds,
/// enabled}` of `collect`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Hook {
    #[serde(default)]
    pub matcher: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub on: String,
    /// Missing means the 30 seconds the page falls back on (`hook.timeout_seconds || 30`).
    #[serde(default = "thirty")]
    pub timeout_seconds: f64,
    /// The page checks the box when `enabled` is not exactly `false`.
    #[serde(default)]
    pub enabled: Option<bool>,
}

fn thirty() -> f64 {
    30.0
}

impl Hook {
    /// Whether the switch of the row starts checked (`hook.enabled === false ? '' : ' checked'`).
    pub fn enabled(&self) -> bool {
        self.enabled != Some(false)
    }

    /// What the number box holds (`hook.timeout_seconds || 30`: a timeout of 0 is written as 30).
    pub fn timeout_shown(&self) -> String {
        let value = if self.timeout_seconds == 0.0 {
            30.0
        } else {
            self.timeout_seconds
        };
        // a whole number loses its `.0`, as JavaScript writes a number
        if value.fract() == 0.0 {
            format!("{}", value as i64)
        } else {
            format!("{value}")
        }
    }
}

/// The answer of `GET /api/studio/hooks`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct HooksAnswer {
    #[serde(default)]
    pub hooks: std::collections::HashMap<String, Vec<Hook>>,
    #[serde(default)]
    pub file: String,
    #[serde(default)]
    pub from_plugins: Vec<PluginHook>,
}

/// A hook an installed plugin brings (`renderPluginHooks`): shown, not editable here.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PluginHook {
    #[serde(default)]
    pub event: String,
    #[serde(default)]
    pub matcher: Option<String>,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub plugin: String,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub timeout_seconds: Option<f64>,
}

impl PluginHook {
    /// What the row writes in the last column (`h.enabled === false ? 'off' : h.timeout_seconds +
    /// ' s'`).
    pub fn tail(&self) -> String {
        if self.enabled == Some(false) {
            "off".to_string()
        } else {
            match self.timeout_seconds {
                Some(seconds) => format!("{} s", number(seconds)),
                // `undefined + ' s'` of the page when the server left the timeout out
                None => "undefined s".to_string(),
            }
        }
    }
}

/// A whole number without a trailing `.0`, the way `JSON.stringify` writes one.
pub fn number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// One row of the draft as it stands on screen: the boxes hold what was typed, the timeout as
/// its words (parsed when the list is saved, `Number(v) || 30`).
#[derive(Debug, Clone, Default)]
pub struct HookRow {
    pub matcher: String,
    pub command: String,
    pub prompt: String,
    pub on: String,
    pub timeout: String,
    pub enabled: bool,
}

impl HookRow {
    /// An empty row of the **+ Add hook** button of the page.
    pub fn new() -> Self {
        HookRow {
            matcher: String::new(),
            command: String::new(),
            prompt: String::new(),
            on: String::new(),
            timeout: "30".to_string(),
            enabled: true,
        }
    }

    /// What a loaded hook holds in its boxes.
    pub fn from_hook(hook: &Hook) -> Self {
        HookRow {
            matcher: hook.matcher.clone(),
            command: hook.command.clone(),
            prompt: hook.prompt.clone(),
            on: hook.on.clone(),
            timeout: hook.timeout_shown(),
            enabled: hook.enabled(),
        }
    }

    /// A row whose boxes hold no command and no prompt is not part of the answer — the page
    /// leaves it out of the list it saves (`if (!entry.command && !entry.prompt) return;`).
    pub fn is_kept(&self) -> bool {
        !(self.command.trim().is_empty() && self.prompt.trim().is_empty())
    }

    /// What the page sends for this row, with the timeout the way `JSON.stringify` writes a
    /// number: whole numbers without a decimal part.
    pub fn entry(&self) -> serde_json::Value {
        let typed: f64 = self.timeout.trim().parse().unwrap_or(0.0);
        let timeout = if typed == 0.0 { 30.0 } else { typed };
        let timeout = if timeout.fract() == 0.0 {
            serde_json::Value::from(timeout as i64)
        } else {
            serde_json::Value::from(timeout)
        };
        serde_json::json!({
            "matcher": self.matcher.trim(),
            "command": self.command.trim(),
            "prompt": self.prompt.trim(),
            "on": self.on.trim(),
            "timeout_seconds": timeout,
            "enabled": self.enabled,
        })
    }
}

/// The rows of the eleven events as they come back from the server: one vector per event, in the
/// order the page draws them (`EVENTS`), with the hooks of that event below its name — the keys
/// of the answer the page does not draw are left out, the way `collect()` leaves them out.
pub fn rows_of(hooks: &std::collections::HashMap<String, Vec<Hook>>) -> Vec<Vec<HookRow>> {
    EVENTS
        .iter()
        .map(|(event, _, _)| {
            hooks
                .get(*event)
                .map(|list| list.iter().map(HookRow::from_hook).collect())
                .unwrap_or_default()
        })
        .collect()
}

/// The whole body of a save: `{"hooks": {event: [row, ...]}}` — an event without a kept row is
/// not part of it, as the draft of the page never holds one.
pub fn payload(rows: &[Vec<HookRow>]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (index, event_rows) in rows.iter().enumerate() {
        if let Some(entries) = event_payload(event_rows) {
            map.insert(EVENTS[index].0.to_string(), serde_json::Value::Array(entries));
        }
    }
    serde_json::Value::Object(map)
}

/// What one event of the draft answers with: only its kept rows, and nothing at all when there
/// are none (the page never sends an event without hooks).
pub fn event_payload(rows: &[HookRow]) -> Option<Vec<serde_json::Value>> {
    let entries: Vec<serde_json::Value> = rows.iter().filter(|row| row.is_kept()).map(HookRow::entry).collect();
    if entries.is_empty() {
        None
    } else {
        Some(entries)
    }
}

/// The sample body of the **Test** button for one event (`sampleFor` of the page).
pub fn sample_for(event: &str) -> (String, serde_json::Value) {
    match event {
        "on_file_edit" => ("edit_file".to_string(), serde_json::json!({ "path": "README.md" })),
        "file_changed" => (
            "README.md".to_string(),
            serde_json::json!({ "path": "README.md", "kind": "modified" }),
        ),
        "user_prompt_submit" => (String::new(), serde_json::json!({ "prompt": "hello" })),
        "pre_tool_call" | "post_tool_call" => (
            "bash".to_string(),
            serde_json::json!({ "command": "echo hello" }),
        ),
        _ => (String::new(), serde_json::json!({})),
    }
}

/// The one line of one run of the test (`'  ' + ... -> exit ... (` + ms + ` ms)` with the two
/// blocks behind it).
pub fn ran_line(run: &serde_json::Value) -> String {
    let command = run.get("command").and_then(serde_json::Value::as_str).unwrap_or("");
    let prompt = run.get("prompt").and_then(serde_json::Value::as_str).unwrap_or("");
    let head = if command.is_empty() {
        format!("  prompt: {prompt}")
    } else {
        format!("  {command}")
    };
    let exit = run.get("exit_code").cloned().unwrap_or(serde_json::Value::Null);
    let millis = run.get("milliseconds").cloned().unwrap_or(serde_json::Value::Null);
    let mut line = format!("{head} -> exit {exit} ({millis} ms)");
    if let Some(stdout) = run.get("stdout").and_then(serde_json::Value::as_str) {
        if !stdout.is_empty() {
            line.push_str(&format!("\n    stdout: {stdout}"));
        }
    }
    if let Some(stderr) = run.get("stderr").and_then(serde_json::Value::as_str) {
        if !stderr.is_empty() {
            line.push_str(&format!("\n    stderr: {stderr}"));
        }
    }
    line
}

/// The whole status line of a finished test (`say` of `test()`).
pub fn test_line(answer: &serde_json::Value) -> String {
    let head = if answer.get("blocked").and_then(serde_json::Value::as_bool).unwrap_or(false) {
        let reason = answer.get("reason").and_then(serde_json::Value::as_str).unwrap_or("");
        format!("BLOCKED: {reason}\n")
    } else {
        "passed\n".to_string()
    };
    let runs = answer
        .get("ran")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    if runs.is_empty() {
        return format!("{head}  (no hook matched)");
    }
    let lines: Vec<String> = runs.iter().map(ran_line).collect();
    format!("{head}{}", lines.join("\n"))
}

/// The references the card draws.
#[derive(Default)]
pub struct HooksView<'a> {
    pub rows: &'a [Vec<HookRow>],
    pub file: &'a str,
    pub status: &'a str,
    pub from_plugins: &'a [PluginHook],
}

/// The placeholder of the matcher box: `file_changed` matches the path of a file, the others the
/// names of tools.
pub fn matcher_placeholder(event: &str) -> &'static str {
    if event == "file_changed" {
        "globs, e.g. src/**/*.rs"
    } else {
        "tools (empty = all)"
    }
}

/// One hook of an event: its boxes over its buttons, on two lines — the page lays one row of
/// flex-wrap the same way at this width.
fn hook_row<'a>(t: Tokens, event_index: usize, row_index: usize, event: &str, row: &'a HookRow) -> Element<'a, Message> {
    let mut first = row![
        input(
            t,
            matcher_placeholder(event),
            &row.matcher,
            move |value| Message::HookField(event_index, row_index, HookField::Matcher, value),
        ),
        input(
            t,
            "command, e.g. cargo fmt --quiet",
            &row.command,
            move |value| Message::HookField(event_index, row_index, HookField::Command, value),
        ),
        input(
            t,
            "or a prompt for the agent",
            &row.prompt,
            move |value| Message::HookField(event_index, row_index, HookField::Prompt, value),
        ),
    ]
    .spacing(6);
    if event == "file_changed" {
        first = first.push(input(
            t,
            "created, modified, deleted",
            &row.on,
            move |value| Message::HookField(event_index, row_index, HookField::On, value),
        ));
    }
    let second = row![
        input(
            t,
            "30",
            &row.timeout,
            move |value| Message::HookField(event_index, row_index, HookField::Timeout, value),
        ),
        compact_switch(
            t,
            row.enabled,
            move |value| Message::HookEnabled(event_index, row_index, value),
        ),
        button(text("Test").size(12))
            .on_press(Message::HookTest(event_index, row_index))
            .padding([2, 8])
            .style(theme::ghost_button(t)),
        button(text("Remove").size(12))
            .on_press(Message::HookRemove(event_index, row_index))
            .padding([2, 8])
            .style(theme::ghost_button(t)),
    ]
    .spacing(6)
    .align_y(Alignment::Center);
    column![first, second]
        .spacing(6)
        .padding([4, 0])
        .into()
}

/// The title of one event with its name behind it (`{title} <span class="mono">{event}</span>`).
fn event_head<'a>(t: Tokens, title: &'a str, event: &'a str) -> Element<'a, Message> {
    row![
        text(title).size(13).color(t.fg),
        text(event)
            .size(11)
            .font(Font::MONOSPACE)
            .color(t.muted),
    ]
    .spacing(6)
    .align_y(Alignment::Center)
    .into()
}

/// The block of one event: its head, its hint, its rows and the button that adds one.
fn event_block<'a>(t: Tokens, event_index: usize, rows: &'a [HookRow]) -> Element<'a, Message> {
    let (event, title, hint) = EVENTS[event_index];
    let mut body = column![event_head(t, title, event), text(hint).size(13).color(t.fg_soft)].spacing(4);
    for (row_index, hook) in rows.iter().enumerate() {
        body = body.push(hook_row(t, event_index, row_index, event, hook));
    }
    body = body.push(
        button(text("+ Add hook").size(12))
            .on_press(Message::HookAdd(event_index))
            .padding([4, 8])
            .style(theme::ghost_button(t)),
    );
    column![body]
        .padding(iced::Padding { top: 14.0, right: 0.0, bottom: 0.0, left: 0.0 })
        .spacing(6)
        .into()
}

/// The hooks of the installed plugins (`renderPluginHooks`): shown, not editable.
fn plugin_hooks<'a>(t: Tokens, items: &'a [PluginHook]) -> Element<'a, Message> {
    let mut body = column![
        text("From plugins").size(13).color(t.fg),
        text("These run like your own hooks. Remove or change them in the plugin (Settings > Plugins), not here.")
            .size(13)
            .color(t.fg_soft),
    ]
    .spacing(4);
    for item in items {
        let matcher = item.matcher.clone().unwrap_or_else(|| "*".to_string());
        let tail = item.tail();
        body = body.push(
            row![
                text(item.event.clone())
                    .size(11)
                    .font(Font::MONOSPACE)
                    .color(t.fg),
                text(matcher)
                    .size(11)
                    .font(Font::MONOSPACE)
                    .color(t.muted),
                text(item.command.clone())
                    .size(11)
                    .font(Font::MONOSPACE)
                    .color(t.fg),
                text(item.plugin.clone()).size(11).color(t.fg_soft),
                text(tail).size(11).color(t.fg_soft),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
        );
    }
    column![body].padding(iced::Padding { top: 16.0, right: 0.0, bottom: 0.0, left: 0.0 }).into()
}

/// The card of `spane-hooks`.
pub fn section<'a>(t: Tokens, v: HooksView<'a>) -> Element<'a, Message> {
    let mut body = column![text(DESC1).size(13).color(t.fg_soft), text(DESC2).size(13).color(t.fg_soft)].spacing(10);
    for event_index in 0..EVENTS.len() {
        body = body.push(event_block(t, event_index, &v.rows[event_index]));
    }
    if !v.from_plugins.is_empty() {
        body = body.push(plugin_hooks(t, v.from_plugins));
    }
    body = body.push(
        row![
            button(text("Save").size(13))
                .on_press(Message::HooksSave)
                .padding([6, 12])
                .style(theme::primary_button(t)),
            button(text("Reload").size(13))
                .on_press(Message::HooksReload)
                .padding([6, 12])
                .style(theme::ghost_button(t)),
            Space::new().width(Length::Fill),
            text(v.file.to_string())
                .size(11)
                .font(Font::MONOSPACE)
                .color(t.fg_soft),
        ]
        .spacing(8)
        .align_y(Alignment::Center)
        .padding(iced::Padding { top: 12.0, right: 0.0, bottom: 0.0, left: 0.0 }),
    );
    if !v.status.is_empty() {
        body = body.push(text(v.status.to_string()).size(11).font(Font::MONOSPACE).color(t.fg_soft));
    }
    card(t, "Hooks", body.into())
}

/// The `ui-input` of the page.
fn input<'a>(
    t: Tokens,
    placeholder: &'a str,
    value: &'a str,
    typed: impl Fn(String) -> Message + 'a,
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
                color: if matches!(status, text_input::Status::Focused { .. }) {
                    t.fg_soft
                } else {
                    t.separator
                },
            },
            icon: t.muted,
            placeholder: t.muted,
            value: t.fg,
            selection: t.primary_soft,
        })
        .into()
}

/// The switch in the row of a hook: the page draws `ui-switch` there, this window draws its
/// On/Off sides the way the other cards of the settings draw theirs.
fn compact_switch<'a>(t: Tokens, on: bool, message: impl Fn(bool) -> Message) -> Element<'a, Message> {
    crate::settings::segmented(
        t,
        vec![("On", on, message(true)), ("Off", !on, message(false))],
    )
}

/// Which box of a hook row was typed into (`HookField` of the message).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookField {
    Matcher,
    Command,
    Prompt,
    On,
    Timeout,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The eleven events of the page with their titles and hints, in the order it draws them.
    #[test]
    fn the_events_are_the_ones_of_the_page() {
        assert_eq!(EVENTS.len(), 11);
        assert_eq!(EVENTS[0].0, "pre_tool_call");
        assert_eq!(EVENTS[0].1, "Before a tool call");
        assert_eq!(
            EVENTS[10].0,
            "file_changed",
            "the watched files are the last event"
        );
        assert!(
            EVENTS[5].2.starts_with("Can block the message (exit 2)"),
            "the hint of `user_prompt_submit` is the one of the page"
        );
    }

    /// A row without a command and without a prompt is not sent — the page drops it in
    /// `collect()`.
    #[test]
    fn a_row_that_says_nothing_is_left_out_of_the_answer() {
        let empty = HookRow::new();
        assert!(!empty.is_kept());
        assert!(event_payload(std::slice::from_ref(&empty)).is_none());
        let mut row = HookRow::new();
        row.command = "cargo fmt --quiet".to_string();
        assert!(row.is_kept());
        let entries = event_payload(std::slice::from_ref(&row)).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["timeout_seconds"], serde_json::json!(30));
        assert_eq!(entries[0]["enabled"], serde_json::json!(true));
        assert_eq!(entries[0]["on"], serde_json::json!(""));
    }

    /// The timeout as the page writes it in the box: `hook.timeout_seconds || 30`.
    #[test]
    fn the_timeout_of_the_box_is_what_the_row_holds() {
        let mut hook = Hook::default();
        assert_eq!(hook.timeout_shown(), "30", "a hook without one gets 30");
        hook.timeout_seconds = 0.0;
        assert_eq!(hook.timeout_shown(), "30", "0 means the fallback");
        hook.timeout_seconds = 45.0;
        assert_eq!(hook.timeout_shown(), "45");
        hook.timeout_seconds = 1.5;
        assert_eq!(hook.timeout_shown(), "1.5");
        assert!(hook.enabled(), "a hook that does not say otherwise is on");
        hook.enabled = Some(false);
        assert!(!hook.enabled());
    }

    /// The lines of a test answer, as `test()` writes them into the status line.
    #[test]
    fn a_test_says_how_it_ran() {
        let answer = serde_json::json!({
            "blocked": false,
            "ran": [{
                "command": "cargo fmt --quiet",
                "prompt": null,
                "exit_code": 0,
                "milliseconds": 12,
                "stdout": "",
                "stderr": "warn: something"
            }]
        });
        assert_eq!(
            test_line(&answer),
            "passed\n  cargo fmt --quiet -> exit 0 (12 ms)\n    stderr: warn: something"
        );
        let blocked = serde_json::json!({ "blocked": true, "reason": "protected path", "ran": [] });
        assert_eq!(
            test_line(&blocked),
            "BLOCKED: protected path\n  (no hook matched)",
            "no run matched, and the block says why"
        );
    }

    /// The sample body of the Test button for the five events that have one.
    #[test]
    fn the_sample_of_a_test_is_the_one_of_the_page() {
        assert_eq!(sample_for("on_file_edit").0, "edit_file");
        assert_eq!(sample_for("file_changed").1["kind"], serde_json::json!("modified"));
        assert_eq!(sample_for("user_prompt_submit").1["prompt"], serde_json::json!("hello"));
        assert_eq!(sample_for("pre_tool_call").0, "bash");
        assert_eq!(sample_for("notification").1, serde_json::json!({}));
    }
}
