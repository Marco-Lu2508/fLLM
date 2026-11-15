//! Settings > Runtime: how freely the app works — the sandbox for agent commands, computer use
//! and the turbo switch. The three cards of `runtime.js` in the `spane-runtime` pane of
//! `shell.html`, with the lines that file writes (`rtSandboxState`, `rtSandboxDetails`,
//! `rtComputerState`, `rtStatus`) and the two questions it asks before a switch goes on.

use crate::app::Message;
use crate::models::{bold, card};
use crate::theme::{self, Tokens};
use iced::widget::{button, column, container, row, text, text_input, Space};
use iced::{Alignment, Element, Font, Length};

/// What `GET /api/studio/runtime` answers (`status()` of `studio_mode.rs`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Status {
    #[serde(default)]
    pub sandbox: String,
    #[serde(default)]
    pub sandbox_from_env: bool,
    #[serde(default)]
    pub turbo: bool,
    #[serde(default)]
    pub computer: Option<Computer>,
    #[serde(default)]
    pub sandbox_status: Option<SandboxStatus>,
    /// `sandbox_available` of the runtime answer — the same value the page reads after `/sandbox`
    /// set a mode (`studio_mode.rs` of the server sends it with both answers).
    #[serde(default)]
    pub sandbox_available: bool,
}

/// `sandbox_status` of that answer: what the sandbox really does on this machine.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct SandboxStatus {
    #[serde(default)]
    pub mechanism: String,
    #[serde(default)]
    pub active: bool,
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub enforced: Vec<String>,
    #[serde(default)]
    pub not_enforced: Vec<String>,
    #[serde(default)]
    pub verification: String,
    #[serde(default)]
    pub sandbox: SandboxWhere,
}

/// `sandbox` of that answer: why the mechanism is not working here.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct SandboxWhere {
    #[serde(default)]
    pub reason: Option<String>,
}

/// `computer` of that answer (`status_json()` of `agent_computer.rs`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Computer {
    #[serde(default)]
    pub compiled: bool,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub forbidden: bool,
    #[serde(default)]
    pub available: bool,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub rate_per_minute: u64,
    #[serde(default)]
    pub max_steps: u64,
}

/// `show()` of `runtime.js`: the line under the radios, the parts joined with ` · `.
pub fn sandbox_state(d: &Status) -> String {
    let mut parts = vec![format!("sandbox: {}", d.sandbox)];
    if d.sandbox_from_env {
        parts.push("set by FLLM_SANDBOX for this server and wins over this page".into());
    }
    let st = d.sandbox_status.clone().unwrap_or_default();
    let mechanism = if st.mechanism.is_empty() {
        "the sandbox".to_string()
    } else {
        st.mechanism.clone()
    };
    if st.active {
        parts.push(format!("{mechanism} is active"));
    } else if d.sandbox == "off" {
        parts.push("sandbox is off".into());
    } else {
        // `(st.sandbox && st.sandbox.reason) || 'it does not work here'`
        let reason = st
            .sandbox
            .reason
            .clone()
            .filter(|r| !r.is_empty())
            .unwrap_or_else(|| "it does not work here".to_string());
        let tail = if st.platform == "windows" && d.sandbox == "strict" {
            " (commands are refused in this mode)"
        } else {
            " (commands run unwrapped)"
        };
        parts.push(format!("{mechanism} is not active — {reason}{tail}"));
    }
    if d.turbo {
        parts.push("turbo is on: no questions, sandbox off".into());
    }
    parts.join(" · ")
}

/// `show()` of `runtime.js`: what is and is not enforced here, and how well that is verified.
pub fn sandbox_details(st: &SandboxStatus) -> String {
    let mut lines = Vec::new();
    for item in &st.enforced {
        lines.push(format!("Enforced: {item}"));
    }
    for item in &st.not_enforced {
        lines.push(format!("Not enforced: {item}"));
    }
    if !st.verification.is_empty() {
        lines.push(format!("Verification: {}", st.verification));
    }
    lines.join("\n")
}

/// `showComputer()` of `runtime.js`: what this machine can do with the screen and the mouse.
pub fn computer_state(c: &Computer) -> String {
    if !c.compiled {
        "This build has no computer use (it was built without the computer-use feature).".into()
    } else if c.forbidden {
        "Computer use is forbidden on this server (FLLM_NO_COMPUTER_USE).".into()
    } else if c.available {
        format!(
            "Possible on this computer ({}).{}",
            c.platform,
            if c.enabled { " It is ON." } else { "" }
        )
    } else {
        format!(
            "Not possible right now: {}",
            c.reason
                .clone()
                .filter(|r| !r.is_empty())
                .unwrap_or_else(|| "unknown reason".to_string())
        )
    }
}

/// `showComputer()` of `runtime.js`: the switch is off when the build or the server says no.
pub fn computer_disabled(c: &Computer) -> bool {
    c.forbidden || !c.compiled
}

/// The two limit boxes of `runtime.js`: `Math.round` of what was typed, and only a number of at
/// least one is sent (the server clamps it to the maximum of its own: 600 and 5000). Rust's
/// `round` rounds halves away from the zero, which is what `Math.round` does for the numbers a
/// person types here.
pub fn limit(typed: &str) -> Option<u64> {
    let n: f64 = typed.trim().parse().ok()?;
    let n = n.round();
    if n >= 1.0 { Some(n as u64) } else { None }
}

/// What the settings page needs to draw the three cards. Every field is a borrow, so the view
/// is copied into each of the three cards.
#[derive(Clone, Copy)]
pub struct RuntimeView<'a> {
    /// `None` until `GET /api/studio/runtime` answered.
    pub status: Option<&'a Status>,
    /// The line of `rtStatus`: `Saved.` or what went wrong.
    pub say: &'a str,
    /// What is typed into the two limit boxes.
    pub rate: &'a str,
    pub steps: &'a str,
}

/// The question a switch asks before it goes on (`confirmDialog` of `runtime.js`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confirm {
    Computer,
    Turbo,
}

impl Confirm {
    /// The sentence of the dialog.
    pub fn question(self) -> &'static str {
        match self {
            Confirm::Computer => "Let the agent see your whole screen and use your mouse and keyboard? It can click and type in any open program as if you did it. Every action asks you first, and Esc three times (or Ctrl+Alt+Shift+Esc) stops it at once.",
            Confirm::Turbo => "Turbo removes every permission question and the sandbox. Continue?",
        }
    }

    /// The title of the dialog (the Studio's `confirmDialog` has no title of its own; the window
    /// asks with the sentence alone, like its other questions).
    pub fn title(self) -> &'static str {
        match self {
            Confirm::Computer => "Computer use",
            Confirm::Turbo => "Turbo",
        }
    }
}

/// The three cards, in the order the Studio's pane draws them.
pub fn section<'a>(t: Tokens, v: RuntimeView<'a>) -> Element<'a, Message> {
    column![sandbox_card(t, v), computer_card(t, v), turbo_card(t, v)]
        .spacing(14)
        .into()
}

/// The card with a red ring and a red title (`border:1px solid var(--destructive)` of the two
/// cards that can do damage).
fn danger_card<'a>(t: Tokens, title: &'a str, body: Element<'a, Message>) -> Element<'a, Message> {
    container(column![text(title).size(13).font(bold()).color(t.danger), body].spacing(12))
        .padding(16)
        .width(Length::Fill)
        .style(move |_| iced::widget::container::Style {
            background: Some(iced::Background::Color(t.surface)),
            text_color: Some(t.fg),
            border: iced::Border { radius: theme::R_LG.into(), width: 1.0, color: t.danger },
            shadow: iced::Shadow { color: iced::Color { a: if t.dark { 0.35 } else { 0.08 }, ..iced::Color::BLACK }, offset: iced::Vector::new(0.0, 4.0), blur_radius: 18.0 },
            ..Default::default()
        })
        .into()
}

/// The radio row of the sandbox: the three modes of `rtSandbox`, filled on the one in force and
/// dead while turbo is on or `FLLM_SANDBOX` chose for this server (the `disabled` of that file).
fn modes<'a>(t: Tokens, d: &Status, loaded: bool) -> Element<'a, Message> {
    let dead = d.turbo || d.sandbox_from_env || !loaded;
    let mut r = row![].spacing(2);
    for label in ["off", "workspace", "strict"] {
        let selected = d.sandbox == label;
        r = r.push(
            button(text(label).size(13).width(Length::Fill).align_x(Alignment::Center))
                .on_press_maybe((!dead).then_some(Message::RuntimeSandbox(label)))
                .width((label.chars().count() as f32 * 7.6 + 28.0).max(80.0))
                .padding([5, 10])
                .style(move |_, status| {
                    let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                    button::Style {
                        background: Some(iced::Background::Color(if selected {
                            t.primary_soft
                        } else if hovered && !dead {
                            t.fill_hover
                        } else {
                            iced::Color::TRANSPARENT
                        })),
                        text_color: if selected { t.primary } else if dead { t.muted } else { t.fg_soft },
                        border: iced::border::rounded(8),
                        ..Default::default()
                    }
                }),
        );
    }
    container(r)
        .padding(2)
        .style(move |_| container::Style {
            background: Some(iced::Background::Color(t.fill)),
            border: iced::border::rounded(10),
            ..Default::default()
        })
        .into()
}

/// The switch of a checkbox (`rtComputer`, `rtTurbo`): On and Off, the selected one filled.
fn switch<'a>(t: Tokens, on: bool, enabled: bool, when_on: Message, when_off: Message) -> Element<'a, Message> {
    let mut r = row![].spacing(2);
    for (label, selected, message) in [("On", on, when_on), ("Off", !on, when_off)] {
        r = r.push(
            button(text(label).size(13).width(Length::Fill).align_x(Alignment::Center))
                .on_press_maybe(enabled.then_some(message))
                .width(70.0)
                .padding([5, 10])
                .style(move |_, status| {
                    let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                    button::Style {
                        background: Some(iced::Background::Color(if selected {
                            t.primary_soft
                        } else if hovered && enabled {
                            t.fill_hover
                        } else {
                            iced::Color::TRANSPARENT
                        })),
                        text_color: if selected { t.primary } else if enabled { t.fg_soft } else { t.muted },
                        border: iced::border::rounded(8),
                        ..Default::default()
                    }
                }),
        );
    }
    container(r)
        .padding(2)
        .style(move |_| container::Style {
            background: Some(iced::Background::Color(t.fill)),
            border: iced::border::rounded(10),
            ..Default::default()
        })
        .into()
}

/// One of the two limit boxes (`rtComputerRate`, `rtComputerSteps`): typed into as text and sent
/// when Enter or the way out of the box takes it (the `change` of the HTML input).
fn limit_box<'a>(t: Tokens, value: &'a str, typed: fn(String) -> Message, sent: Message) -> Element<'a, Message> {
    text_input("", value)
        .on_input(typed)
        .on_submit(sent)
        .size(13)
        .padding([5, 8])
        .width(76.0)
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

/// The description of the first card (the `<div class="app-item-desc">` of `runtime.js`, with
/// its `<span class="mono">` words set like the rest of the sentence).
pub const SANDBOX_HELP: &str = "Approvals decide whether a command may run; the sandbox decides what it can reach once it runs. workspace: the system is read-only, the workspace writable, private /tmp, own PID namespace. strict: also no network. off: a plain shell. How it is done depends on the system: bubblewrap on Linux, a restricted low-integrity token in a job object on Windows (strict is unavailable there), Seatbelt on macOS. Windows and macOS are implemented but not yet verified on real machines, so they stay off until you choose a mode. When the mechanism does not work, the command runs unwrapped and the status below says so.";

/// The warning of the computer-use card.
pub const COMPUTER_HELP: &str = "Warning. When this is on, the coding agent can look at your whole screen (everything that is open on it, including private messages, other windows and passwords you can see) and can move the mouse, click and type on this computer as if you did it. It can do anything you can do in the programs that are open. A mistake of the model can send a message, delete a file or pay for something. Every action asks you first, a red banner shows while it is on, and you can stop it at any time with the STOP button, by pressing Esc three times, or with Ctrl+Alt+Shift+Esc anywhere. It is meant for programs that have no command line and no API; for web pages use the browser tool. It only works with a model that can see pictures and find things on them, and most small local models are bad at that. It is off until you switch it on here, and it can only be used from this computer.";

/// The description of the turbo card.
pub const TURBO_HELP: &str = "No permission questions at all: every tool call runs, the sandbox is off, and the agent does not stop to ask. Meant for a machine you trust and work you can throw away. Deny rules are ignored too, because that is what \u{201c}no restrictions\u{201d} means \u{2014} turn it off again when the job is done.";

fn sandbox_card<'a>(t: Tokens, v: RuntimeView<'a>) -> Element<'a, Message> {
    let d = v.status.cloned().unwrap_or_default();
    let state = if v.status.is_some() { sandbox_state(&d) } else { String::new() };
    let details = if v.status.is_some() {
        d.sandbox_status.as_ref().map(sandbox_details).unwrap_or_default()
    } else {
        String::new()
    };
    let body = column![
        text(SANDBOX_HELP).size(13).color(t.fg_soft),
        modes(t, &d, v.status.is_some()),
        text(state).size(11).font(Font::MONOSPACE).color(t.muted),
        text(details).size(11).font(Font::MONOSPACE).color(t.muted),
    ]
    .spacing(8);
    card(t, "Sandbox for agent commands", body.into())
}

fn computer_card<'a>(t: Tokens, v: RuntimeView<'a>) -> Element<'a, Message> {
    let c = v.status.and_then(|d| d.computer.clone()).unwrap_or_default();
    let can = !computer_disabled(&c);
    let state = if v.status.is_some() { computer_state(&c) } else { String::new() };
    let body = column![
        text(COMPUTER_HELP).size(13).color(t.fg_soft),
        row![
            text("Let the agent see the screen and use mouse and keyboard").size(13).width(Length::Fill),
            switch(t, c.enabled, can, Message::RuntimeComputer(true), Message::RuntimeComputer(false)),
        ]
        .spacing(10)
        .align_y(Alignment::Center),
        row![
            row![
                text("At most").size(12).color(t.fg_soft),
                limit_box(t, v.rate, Message::RuntimeRate, Message::RuntimeRateSave),
                text("actions per minute").size(12).color(t.fg_soft),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
            row![
                text("At most").size(12).color(t.fg_soft),
                limit_box(t, v.steps, Message::RuntimeSteps, Message::RuntimeStepsSave),
                text("actions per task").size(12).color(t.fg_soft),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        ]
        .spacing(16),
        text(state).size(11).font(Font::MONOSPACE).color(t.muted),
    ]
    .spacing(10);
    danger_card(t, "Computer use", body.into())
}

fn turbo_card<'a>(t: Tokens, v: RuntimeView<'a>) -> Element<'a, Message> {
    let turbo = v.status.is_some_and(|d| d.turbo);
    let body = column![
        text(TURBO_HELP).size(13).color(t.fg_soft),
        row![
            text("Enable turbo").size(13).width(Length::Fill),
            switch(t, turbo, v.status.is_some(), Message::RuntimeTurbo(true), Message::RuntimeTurbo(false)),
        ]
        .spacing(10)
        .align_y(Alignment::Center),
        text(v.say).size(11).font(Font::MONOSPACE).color(if v.say.starts_with("Saved") { t.muted } else { t.danger }),
    ]
    .spacing(10);
    danger_card(t, "Turbo", body.into())
}

/// The question before computer use or turbo goes on (`confirmDialog` of `runtime.js`).
pub fn confirm_window<'a>(t: Tokens, kind: Confirm) -> Element<'a, Message> {
    let window = container(
        column![
            text(kind.title()).size(16).font(bold()),
            text(kind.question()).size(13).color(t.fg_soft),
            row![
                Space::new().width(Length::Fill),
                button(text("Cancel").size(13))
                    .on_press(Message::RuntimeConfirmCancel)
                    .padding([6, 12])
                    .style(theme::ghost_button(t)),
                button(text("OK").size(13))
                    .on_press(Message::RuntimeConfirmOk)
                    .padding([6, 12])
                    .style(theme::primary_button(t)),
            ],
        ]
        .spacing(10),
    )
    .padding(16)
    .width(460)
    .style(theme::card(t, theme::R_LG, false));
    container(window).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(sandbox: &str) -> Status {
        Status {
            sandbox: sandbox.into(),
            sandbox_status: Some(SandboxStatus {
                mechanism: "bubblewrap".into(),
                active: false,
                platform: "linux".into(),
                enforced: vec!["the filesystem outside the workspace".into()],
                not_enforced: vec!["the network".into()],
                verification: "the paths of /proc/self/mountinfo".into(),
                sandbox: SandboxWhere { reason: Some("bubblewrap is not installed".into()) },
            }),
            ..Default::default()
        }
    }

    /// The line under the radios, part by part as `show()` of `runtime.js` builds it.
    #[test]
    fn the_state_line_reads_the_way_the_studio_writes_it() {
        assert_eq!(sandbox_state(&status("off")), "sandbox: off · sandbox is off");
        assert_eq!(
            sandbox_state(&status("workspace")),
            "sandbox: workspace · bubblewrap is not active — bubblewrap is not installed (commands run unwrapped)"
        );
        // `strict` on Windows is refused instead of unwrapped
        let mut w = status("strict");
        w.sandbox_status.as_mut().unwrap().platform = "windows".into();
        assert_eq!(
            sandbox_state(&w),
            "sandbox: strict · bubblewrap is not active — bubblewrap is not installed (commands are refused in this mode)"
        );
        // no reason at all: the sentence of the Studio
        let mut none = status("strict");
        none.sandbox_status.as_mut().unwrap().sandbox.reason = None;
        assert_eq!(
            sandbox_state(&none),
            "sandbox: strict · bubblewrap is not active — it does not work here (commands run unwrapped)"
        );
        // an active mechanism, an environment that chose, and turbo on top
        let mut on = status("strict");
        on.sandbox_status.as_mut().unwrap().active = true;
        on.sandbox_from_env = true;
        on.turbo = true;
        assert_eq!(
            sandbox_state(&on),
            "sandbox: strict · set by FLLM_SANDBOX for this server and wins over this page · bubblewrap is active · turbo is on: no questions, sandbox off"
        );
        // nothing of the status: the words the Studio falls back on (`mechanism` empty becomes
        // *the sandbox*, an empty mode is not `off` so the reason sentence is the one for a
        // mechanism that does not work)
        assert_eq!(
            sandbox_state(&Status::default()),
            "sandbox:  · the sandbox is not active — it does not work here (commands run unwrapped)"
        );
    }

    /// The block under the line: what is enforced, what is not, how it was verified.
    #[test]
    fn the_details_are_the_lines_of_the_status() {
        let st = status("workspace").sandbox_status.unwrap();
        assert_eq!(
            sandbox_details(&st),
            "Enforced: the filesystem outside the workspace\nNot enforced: the network\nVerification: the paths of /proc/self/mountinfo"
        );
        // nothing to say: an empty block, not a stray line
        assert_eq!(sandbox_details(&SandboxStatus::default()), "");
        // only the parts that are there, in the order of the file
        let only = SandboxStatus { enforced: vec!["a".into()], ..Default::default() };
        assert_eq!(sandbox_details(&only), "Enforced: a");
    }

    /// The line of the computer-use card: build, server, machine, reason — in that order.
    #[test]
    fn the_computer_line_says_what_this_machine_can_do() {
        let base = Computer { platform: "linux".into(), ..Default::default() };
        assert_eq!(
            computer_state(&base),
            "This build has no computer use (it was built without the computer-use feature)."
        );
        let forbidden = Computer { compiled: true, forbidden: true, ..base.clone() };
        assert_eq!(computer_state(&forbidden), "Computer use is forbidden on this server (FLLM_NO_COMPUTER_USE).");
        let ok = Computer { compiled: true, available: true, ..base.clone() };
        assert_eq!(computer_state(&ok), "Possible on this computer (linux).");
        let running = Computer { compiled: true, available: true, enabled: true, ..base.clone() };
        assert_eq!(computer_state(&running), "Possible on this computer (linux). It is ON.");
        let broken = Computer { compiled: true, reason: Some("no display".into()), ..base.clone() };
        assert_eq!(computer_state(&broken), "Not possible right now: no display");
        let quiet = Computer { compiled: true, ..base.clone() };
        assert_eq!(computer_state(&quiet), "Not possible right now: unknown reason");
        // a reason that is an empty string counts as no reason at all
        let empty = Computer { compiled: true, reason: Some(String::new()), ..base };
        assert_eq!(computer_state(&empty), "Not possible right now: unknown reason");
    }

    /// The switch is dead when the build or the server says no (`showComputer` of `runtime.js`).
    #[test]
    fn the_computer_switch_is_dead_when_the_server_says_no() {
        assert!(computer_disabled(&Computer::default()), "not compiled");
        assert!(computer_disabled(&Computer { compiled: true, forbidden: true, ..Default::default() }), "forbidden");
        assert!(!computer_disabled(&Computer { compiled: true, ..Default::default() }), "there");
    }

    /// The limit boxes: rounded like `Math.round`, and only a number of at least one is sent.
    #[test]
    fn a_limit_is_rounded_and_only_sent_when_it_is_at_least_one() {
        assert_eq!(limit("12"), Some(12));
        assert_eq!(limit("12.4"), Some(12));
        assert_eq!(limit("12.5"), Some(13), "halves go away from the zero, as in JavaScript");
        assert_eq!(limit(" 30 "), Some(30));
        assert_eq!(limit("0"), None, "zero is not sent");
        assert_eq!(limit("0.4"), None, "rounds to zero, so it is not sent");
        assert_eq!(limit("-5"), None, "negative is not sent");
        assert_eq!(limit(""), None, "an empty box sends nothing");
        assert_eq!(limit("abc"), None, "a text sends nothing");
    }
}
