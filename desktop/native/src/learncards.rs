//! Row 88, Settings > Personalization: the three cards `learn.js` fills in below the chat's own
//! personalization — *Learning from my sessions* (the two switches, what waits, what the last
//! step wrote and the feed *What I learned* with its Undo), *What the coding agent learned about
//! you* (the profile file, what waits for a yes) and *Memory of the coding agent* (its entries, what
//! waits for a yes, *Forget everything that was learned*).
//!
//! The state is `LearnPane`; its messages are `LpMsg`, wrapped as `Message::Lp`; the window
//! only passes them on (`update`) and draws what `view` makes of it.

use crate::app::Message;
use crate::models::card;
use crate::skills::{LearnEntry, feed_row};
use crate::theme::{self, Tokens};
use iced::widget::{button, column, container, row, text, text_editor, text_input, Space};
use iced::{Alignment, Element, Font, Length, Task};
use serde::Deserialize;
use serde_json::{json, Value};

/// What `GET /api/studio/learn/profile` answers.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Profile {
    #[serde(default)]
    pub text: String,
    /// `in_prompt !== false`.
    #[serde(default)]
    pub in_prompt: Option<bool>,
    #[serde(default)]
    pub max_chars: Option<u64>,
    #[serde(default)]
    pub proposals: Vec<ProfileProposal>,
}

/// A profile line that waits for a yes.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ProfileProposal {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub section: String,
    #[serde(default)]
    pub session: Option<String>,
}

/// One entry of the memory the learning loop and the memory tool keep.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct MemEntry {
    #[serde(default)]
    pub index: u64,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub source: Option<MemSource>,
}

/// Where a learned entry came from.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct MemSource {
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub created: Option<f64>,
}

/// A memory entry that waits for a yes (or a removal that does).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct MemProposal {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub action: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub scope: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

/// What `GET /api/studio/learn/memory` answers.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Memory {
    #[serde(default)]
    pub global: Vec<MemEntry>,
    #[serde(default)]
    pub workspace: Vec<MemEntry>,
    #[serde(default)]
    pub proposals: Vec<MemProposal>,
}

/// The line under an entry (`lnRefreshMemory`).
pub fn mem_source_line(scope: &str, entry: &MemEntry) -> String {
    let place = if scope == "global" { "every project" } else { "this project" };
    match &entry.source {
        Some(src) => {
            let session = src.session.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "?".into());
            let when = src.created.filter(|c| *c != 0.0).map(|c| format!(", {}", crate::skills::when(c as i64))).unwrap_or_default();
            format!("{place} · learned in session {session}{when}")
        }
        None => format!("{place} · written by hand or by the memory tool"),
    }
}

/// The line under a proposal of the memory.
pub fn mem_proposal_line(p: &MemProposal) -> String {
    let place = if p.scope == "global" { "every project" } else { "this project" };
    let mut line = format!("{place} · {}", p.kind);
    if let Some(s) = p.session.as_deref().filter(|s| !s.is_empty()) {
        line.push_str(&format!(" · from session {s}"));
    }
    if let Some(r) = p.reason.as_deref().filter(|s| !s.is_empty()) {
        line.push_str(&format!(" · {r}"));
    }
    line
}

/// The line under a profile proposal: its section with the underscore as a space.
pub fn profile_proposal_line(p: &ProfileProposal) -> String {
    let mut line = p.section.replace('_', " ");
    if let Some(s) = p.session.as_deref().filter(|s| !s.is_empty()) {
        line.push_str(&format!(" · from session {s}"));
    }
    line
}

/// The counter under the profile (`n / max`; the page counts UTF-16 units of the box).
pub fn profile_count(text: &str, max: Option<u64>) -> String {
    format!("{} / {}", text.encode_utf16().count(), max.unwrap_or(3000))
}

/// What a dialog of these cards waits for.
#[derive(Debug, Clone, PartialEq)]
pub enum Dialog {
    /// *Delete your profile?*
    DeleteProfile,
    /// *Delete this memory entry?* with the entry's text.
    DeleteMemory { scope: String, index: u64, text: String },
    /// *Edit this memory entry*: the box holds the text; an empty one deletes the entry.
    EditMemory { scope: String, index: u64, input: String },
    /// *Forget everything that was learned?*
    Forget,
}

/// A request the cards make: the route, the body and the toast that follows when it worked.
#[derive(Debug, Clone, PartialEq)]
pub struct Post {
    pub path: &'static str,
    pub body: Value,
    pub done: Option<&'static str>,
}

/// The state of the three cards.
pub struct LearnPane {
    pub status: Option<crate::learn::Status>,
    pub profile: Option<Profile>,
    pub profile_error: Option<String>,
    pub editor: text_editor::Content,
    pub memory: Option<Memory>,
    pub memory_error: Option<String>,
    pub dialog: Option<Dialog>,
}

impl Default for LearnPane {
    fn default() -> Self {
        Self { status: None, profile: None, profile_error: None, editor: text_editor::Content::new(), memory: None, memory_error: None, dialog: None }
    }
}

/// Everything these cards ask and answer.
#[derive(Debug, Clone)]
pub enum LpMsg {
    ProfileLoaded(Result<Profile, String>),
    MemoryLoaded(Result<Memory, String>),
    StatusLoaded(Option<crate::learn::Status>),
    Edit(text_editor::Action),
    Save,
    UseInPrompt(bool),
    Export,
    DeleteProfileAsk,
    ProfileProposal(String, &'static str),
    MemEditAsk(String, u64, String),
    MemEditInput(String),
    MemDeleteAsk(String, u64, String),
    MemProposal(String, &'static str),
    ForgetAsk,
    DialogCancel,
    DialogOk,
    Enabled(bool),
    AskFirst(bool),
    /// A post is over: its toast (when it has one), or the message of its failure.
    Posted(Result<Value, String>, Option<String>),
    ExportDone(Result<String, String>),
}

async fn request(base: &str, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<Value, String> {
    crate::models::mcp_request(base, method, path, body).await
}

impl LearnPane {
    /// Reads the profile, the memory and the status (`lnRefreshProfile`, `lnRefreshMemory` and the
    /// status half of `lnRefreshFeed`); the skills and the feed are the Skills card's own reads.
    pub fn refresh(&self, base: &str, workspace: &str) -> Task<Message> {
        let (b1, b2, b3) = (base.to_string(), base.to_string(), base.to_string());
        let (w2, w3) = (workspace.to_string(), workspace.to_string());
        Task::batch([
            Task::perform(
                async move { request(&b1, reqwest::Method::GET, "/api/studio/learn/profile", None).await.and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string())) },
                |r| Message::Lp(LpMsg::ProfileLoaded(r)),
            ),
            Task::perform(
                async move {
                    let path = format!("/api/studio/learn/memory?{}", crate::models::sk_query(&[("workspace", &w2)]));
                    request(&b2, reqwest::Method::GET, &path, None).await.and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
                },
                |r| Message::Lp(LpMsg::MemoryLoaded(r)),
            ),
            Task::perform(crate::learn::fetch_status(b3, w3), |st| Message::Lp(LpMsg::StatusLoaded(st))),
        ])
    }

    fn send(&self, base: &str, post: Post) -> Task<Message> {
        let base = base.to_string();
        Task::perform(async move { request(&base, reqwest::Method::POST, post.path, Some(post.body)).await }, move |r| {
            Message::Lp(LpMsg::Posted(r, post.done.map(str::to_string)))
        })
    }

    /// The request a message makes (`lnPost`, `lnRequest` of `learn.js`), or none. State that the
    /// page changes before it asks (a switch, the dialog that is answered) changes here.
    pub fn post_of(&mut self, msg: &LpMsg, workspace: &str) -> Option<Post> {
        let post = |path: &'static str, body: Value, done: Option<&'static str>| Some(Post { path, body, done });
        match msg {
            LpMsg::Save => {
                let text = self.editor.text();
                post("/api/studio/learn/profile", json!({ "action": "save", "text": text.strip_suffix('\n').unwrap_or(&text) }), Some("Profile saved."))
            }
            LpMsg::UseInPrompt(on) => {
                if let Some(p) = self.profile.as_mut() {
                    p.in_prompt = Some(*on);
                }
                post("/api/studio/learn/profile", json!({ "action": "prompt", "on": on }), None)
            }
            LpMsg::ProfileProposal(id, act) => post("/api/studio/learn/profile", json!({ "action": act, "id": id }), None),
            LpMsg::MemProposal(id, act) => post("/api/studio/learn/memory", json!({ "action": act, "id": id, "workspace": workspace }), None),
            LpMsg::AskFirst(on) => {
                if let Some(st) = self.status.as_mut() {
                    st.ask_first = *on;
                }
                let toast = if *on { "The agent now proposes and waits for your yes." } else { "The agent applies what it learns and logs it." };
                post("/api/studio/learn/settings", json!({ "ask_first": on }), Some(toast))
            }
            LpMsg::DialogOk => match self.dialog.take() {
                Some(Dialog::DeleteProfile) => post("/api/studio/learn/profile", json!({ "action": "delete" }), Some("Profile deleted.")),
                Some(Dialog::DeleteMemory { scope, index, .. }) => {
                    post("/api/studio/learn/memory", json!({ "action": "delete", "scope": scope, "index": index, "workspace": workspace }), Some("Deleted."))
                }
                Some(Dialog::EditMemory { scope, index, input }) => {
                    // an empty text deletes the entry (`text.trim() ? 'edit' : 'delete'`)
                    let action = if input.trim().is_empty() { "delete" } else { "edit" };
                    post("/api/studio/learn/memory", json!({ "action": action, "scope": scope, "index": index, "text": input, "workspace": workspace }), Some("Saved."))
                }
                Some(Dialog::Forget) => post("/api/studio/learn/memory", json!({ "action": "forget", "all_learned": true, "text": "", "workspace": workspace }), Some("Forgotten.")),
                None => None,
            },
            _ => None,
        }
    }

    /// One message. `workspace` is the folder of the Code page (`lnWorkspace()`).
    pub fn update(&mut self, msg: LpMsg, base: &str, workspace: &str, toast: &mut Option<String>) -> Task<Message> {
        if let Some(post) = self.post_of(&msg, workspace) {
            return self.send(base, post);
        }
        match msg {
            LpMsg::ProfileLoaded(Ok(p)) => {
                self.profile_error = None;
                // the box is not taken from under the cursor: only a box that holds nothing new is filled
                if self.profile.as_ref().is_none_or(|old| old.text == self.editor.text().trim_end_matches('\n')) || self.editor.text().trim().is_empty() {
                    self.editor = text_editor::Content::with_text(&p.text);
                }
                self.profile = Some(p);
                Task::none()
            }
            LpMsg::ProfileLoaded(Err(e)) => {
                self.profile_error = Some(e);
                Task::none()
            }
            LpMsg::MemoryLoaded(Ok(m)) => {
                self.memory_error = None;
                self.memory = Some(m);
                Task::none()
            }
            LpMsg::MemoryLoaded(Err(e)) => {
                self.memory_error = Some(e);
                Task::none()
            }
            LpMsg::StatusLoaded(st) => {
                if st.is_some() {
                    self.status = st;
                }
                Task::none()
            }
            LpMsg::Edit(action) => {
                self.editor.perform(action);
                Task::none()
            }
            LpMsg::Export => {
                let text = self.editor.text();
                let text = text.strip_suffix('\n').unwrap_or(&text).to_string();
                Task::perform(
                    async move {
                        let dir = crate::chatlist::export_dir();
                        let path = dir.join("user-profile.md");
                        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                        std::fs::write(&path, text).map_err(|e| e.to_string())?;
                        Ok(path.display().to_string())
                    },
                    |r| Message::Lp(LpMsg::ExportDone(r)),
                )
            }
            LpMsg::ExportDone(Ok(path)) => {
                *toast = Some(format!("Saved to {path}"));
                Task::none()
            }
            LpMsg::ExportDone(Err(e)) => {
                *toast = Some(e);
                Task::none()
            }
            LpMsg::DeleteProfileAsk => {
                self.dialog = Some(Dialog::DeleteProfile);
                Task::none()
            }
            LpMsg::MemEditAsk(scope, index, current) => {
                self.dialog = Some(Dialog::EditMemory { scope, index, input: current });
                Task::none()
            }
            LpMsg::MemEditInput(v) => {
                if let Some(Dialog::EditMemory { input, .. }) = self.dialog.as_mut() {
                    *input = v;
                }
                Task::none()
            }
            LpMsg::MemDeleteAsk(scope, index, text) => {
                self.dialog = Some(Dialog::DeleteMemory { scope, index, text });
                Task::none()
            }
            LpMsg::ForgetAsk => {
                self.dialog = Some(Dialog::Forget);
                Task::none()
            }
            LpMsg::DialogCancel => {
                self.dialog = None;
                Task::none()
            }
            LpMsg::Enabled(on) => {
                if let Some(st) = self.status.as_mut() {
                    st.enabled = Some(on);
                }
                let (b, w) = (base.to_string(), workspace.to_string());
                Task::perform(crate::learn::set_enabled(b, on, w), move |r| Message::Lp(LpMsg::Posted(r.map(|()| json!({ "enabled": on })), Some(if on { "Learning is on.".into() } else { "Learning is off. What was learned stays; nothing new is saved.".into() }))))
            }
            // asked above, by `post_of`
            LpMsg::Save | LpMsg::UseInPrompt(_) | LpMsg::ProfileProposal(..) | LpMsg::MemProposal(..) | LpMsg::AskFirst(_) | LpMsg::DialogOk => Task::none(),
            LpMsg::Posted(result, done) => {
                match (&result, done) {
                    (Ok(_), Some(text)) => *toast = Some(text),
                    (Err(e), _) => *toast = Some(e.clone()),
                    _ => {}
                }
                // `lnPost` reads everything again, whatever the answer was: the window does it,
                // the skills and the feed with the rest
                Task::none()
            }
        }
    }
}

// ---- the cards --------------------------------------------------------------------------------

const LEARN_TEXT: &str = "fLLM learns from your coding sessions by default: after a run it saves skills, notes about the project and short lines about you on its own, each with its reason, and keeps a log you can undo from. Nothing is learned from the chat page, and no secret is ever stored.";
const PROFILE_TEXT: &str = "A plain text file: preferences, skill level, languages, conventions, tool habits. The agent adds short dated lines on its own, only from what you wrote yourself (never from tool output); a line that rests on nothing you wrote is dropped. Edit it freely: up to 1,200 characters of it go into the system prompt of the coding agent, nothing else is kept about you. Secrets are refused.";
const MEMORY_TEXT: &str = "Short notes the coding agent keeps for later sessions (agent-memory.md in the Studio folder, .fllm/memory.md in a project). After a run it saves facts about the project, your preferences, commands that worked and mistakes to avoid; repeats are dropped, secrets are never stored, at most five per run. Each learned entry shows the session it came from; edit or delete any entry here, or undo it from the log above.";
const PROFILE_PLACEHOLDER: &str = "## Preferences\n- Short answers, no filler\n## Languages\n- Writes Rust and Python";

fn semibold() -> Font {
    Font { weight: iced::font::Weight::Semibold, ..Font::DEFAULT }
}

fn switch<'a>(t: Tokens, title: &'a str, on: bool, message: fn(bool) -> LpMsg) -> Element<'a, Message> {
    row![
        text(title).size(13).color(t.fg),
        Space::new().width(Length::Fill),
        crate::settings::segmented(t, vec![("On", on, Message::Lp(message(true))), ("Off", !on, Message::Lp(message(false)))]),
    ]
    .spacing(8)
    .align_y(Alignment::Center)
    .into()
}

fn small<'a>(t: Tokens, label: &'static str, message: LpMsg, primary: bool) -> Element<'a, Message> {
    let b = button(text(label).size(12)).on_press(Message::Lp(message)).padding([2, 8]);
    if primary { b.style(theme::primary_button(t)).into() } else { b.style(theme::ghost_button(t)).into() }
}

fn heading<'a>(t: Tokens, label: &'a str) -> Element<'a, Message> {
    text(label).size(13).color(t.fg).font(semibold()).into()
}

fn line<'a>(t: Tokens, label: String) -> Element<'a, Message> {
    text(label).size(12).color(t.fg_soft).into()
}

/// A row of a list: title, description under it, buttons at the end (`app-card-item`).
fn item<'a>(t: Tokens, title: String, desc: String, buttons: Vec<Element<'a, Message>>) -> Element<'a, Message> {
    let mut r = row![column![text(title).size(13).color(t.fg), text(desc).size(12).color(t.fg_soft)].spacing(2).width(Length::Fill)].spacing(6).align_y(Alignment::Center);
    for b in buttons {
        r = r.push(b);
    }
    container(r).padding([4, 0]).width(Length::Fill).into()
}

/// The three cards, one under the other.
pub fn view<'a>(t: Tokens, pane: &'a LearnPane, pending: &'a str, last_run: &'a str, feed: &'a [LearnEntry]) -> Element<'a, Message> {
    let enabled = pane.status.as_ref().map_or(true, |s| s.enabled != Some(false));
    let ask_first = pane.status.as_ref().is_some_and(|s| s.ask_first);
    let mut learn = column![line(t, LEARN_TEXT.into()), switch(t, "Learn from my sessions", enabled, LpMsg::Enabled), switch(t, "Ask me first (the agent only proposes; you decide below)", ask_first, LpMsg::AskFirst)].spacing(10);
    if let Some(note) = pane.status.as_ref().and_then(|s| s.project_override).map(|on| {
        if on { "Learning is on for the open project (its .fllm/learn.json)." } else { "Learning is off for the open project (its .fllm/learn.json)." }
    }) {
        learn = learn.push(line(t, note.into()));
    }
    if !pending.is_empty() {
        learn = learn.push(line(t, pending.to_string()));
    }
    if !last_run.is_empty() {
        learn = learn.push(line(t, last_run.to_string()));
    }
    learn = learn.push(heading(t, "What I learned"));
    if feed.is_empty() {
        learn = learn.push(line(t, "Nothing was learned automatically yet.".into()));
    } else {
        let mut f = column![].spacing(0);
        for e in feed {
            f = f.push(feed_row(t, e));
        }
        learn = learn.push(f);
    }

    // the profile
    let shown = pane.profile.as_ref();
    let mut profile = column![heading(t, "Your profile"), line(t, PROFILE_TEXT.into())].spacing(10);
    if let Some(e) = &pane.profile_error {
        profile = profile.push(line(t, e.clone()));
    }
    profile = profile.push(
        text_editor(&pane.editor)
            .on_action(|a| Message::Lp(LpMsg::Edit(a)))
            .placeholder(PROFILE_PLACEHOLDER)
            .padding(8)
            .size(13)
            .font(Font::MONOSPACE)
            .height(Length::Fixed(180.0))
            .style(crate::skills::editor_style(t)),
    );
    profile = profile.push(text(profile_count(pane.editor.text().strip_suffix('\n').unwrap_or(&pane.editor.text()), shown.and_then(|p| p.max_chars))).size(11).font(Font::MONOSPACE).color(t.muted));
    profile = profile.push(switch(t, "Use the profile in the agent's prompt", shown.is_none_or(|p| p.in_prompt != Some(false)), LpMsg::UseInPrompt));
    profile = profile.push(
        row![
            button(text("Save").size(13)).on_press(Message::Lp(LpMsg::Save)).padding([6, 12]).style(theme::primary_button(t)),
            button(text("Export").size(13)).on_press(Message::Lp(LpMsg::Export)).padding([6, 12]).style(theme::secondary_button(t)),
            button(text("Delete").size(13)).on_press(Message::Lp(LpMsg::DeleteProfileAsk)).padding([6, 12]).style(move |_, status| {
                let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                button::Style { background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })), text_color: t.danger, border: iced::border::rounded(8), ..Default::default() }
            }),
        ]
        .spacing(6),
    );
    profile = profile.push(heading(t, "Lines that wait for your yes (only when you chose to be asked first)"));
    match shown.map(|p| p.proposals.as_slice()).filter(|p| !p.is_empty()) {
        Some(list) => {
            for p in list {
                profile = profile.push(item(t, p.text.clone(), profile_proposal_line(p), vec![small(t, "Add", LpMsg::ProfileProposal(p.id.clone(), "accept"), true), small(t, "Drop", LpMsg::ProfileProposal(p.id.clone(), "reject"), false)]));
            }
        }
        None => profile = profile.push(line(t, "No profile line waits for a yes.".into())),
    }

    // the memory
    let mut memory = column![line(t, MEMORY_TEXT.into())].spacing(10);
    if let Some(e) = &pane.memory_error {
        memory = memory.push(line(t, e.clone()));
    }
    memory = memory.push(heading(t, "Waiting for your yes (only when you chose to be asked first)"));
    let data = pane.memory.as_ref();
    match data.map(|m| m.proposals.as_slice()).filter(|p| !p.is_empty()) {
        Some(list) => {
            for p in list {
                let remove = p.action == "remove";
                let title = if remove { format!("Forget: {}", p.text) } else { p.text.clone() };
                memory = memory.push(item(t, title, mem_proposal_line(p), vec![small(t, if remove { "Forget it" } else { "Remember" }, LpMsg::MemProposal(p.id.clone(), "accept"), true), small(t, "Drop", LpMsg::MemProposal(p.id.clone(), "reject"), false)]));
            }
        }
        None => memory = memory.push(line(t, "No memory entry waits for a yes.".into())),
    }
    memory = memory.push(heading(t, "Entries"));
    let mut entries = column![].spacing(0);
    let mut any = false;
    for (scope, list) in [("global", data.map(|m| m.global.as_slice())), ("workspace", data.map(|m| m.workspace.as_slice()))] {
        for e in list.unwrap_or_default() {
            any = true;
            entries = entries.push(item(
                t,
                e.text.clone(),
                mem_source_line(scope, e),
                vec![small(t, "Edit", LpMsg::MemEditAsk(scope.into(), e.index, e.text.clone()), false), small(t, "Delete", LpMsg::MemDeleteAsk(scope.into(), e.index, e.text.clone()), false)],
            ));
        }
    }
    memory = memory.push(if any { entries.into() } else { line(t, "The agent memory is empty.".into()) });
    memory = memory.push(
        row![
            Space::new().width(Length::Fill),
            button(text("Forget everything that was learned").size(13)).on_press(Message::Lp(LpMsg::ForgetAsk)).padding([6, 12]).style(move |_, status| {
                let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                button::Style { background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })), text_color: t.danger, border: iced::border::rounded(8), ..Default::default() }
            }),
        ],
    );

    column![card(t, "Learning from my sessions", learn.into()), card(t, "What the coding agent learned about you", profile.into()), card(t, "Memory of the coding agent", memory.into())]
        .spacing(12)
        .into()
}

/// The dialog that is open (`askDialog` of `learn.js`).
pub fn dialog_window<'a>(t: Tokens, d: &'a Dialog) -> Element<'a, Message> {
    let (title, sentence, ok, danger, input): (String, String, &str, bool, Option<&str>) = match d {
        Dialog::DeleteProfile => ("Delete your profile?".into(), "The profile file and the profile lines that wait for a yes are deleted. Memory entries and skills stay.".into(), "Delete", true, None),
        Dialog::DeleteMemory { text, .. } => ("Delete this memory entry?".into(), text.clone(), "Delete", true, None),
        Dialog::EditMemory { input, .. } => ("Edit this memory entry".into(), "An empty text deletes the entry.".into(), "Save", false, Some(input.as_str())),
        Dialog::Forget => (
            "Forget everything that was learned?".into(),
            "Removes every memory entry the learning loop wrote and the memory entries that wait for a yes. Entries you wrote yourself stay.".into(),
            "Forget",
            true,
            None,
        ),
    };
    let ok_button = button(text(ok).size(13)).on_press(Message::Lp(LpMsg::DialogOk)).padding([6, 12]);
    let ok_button: Element<'a, Message> = if danger {
        ok_button
            .style(move |_, status| {
                let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                button::Style { background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })), text_color: t.danger, border: iced::border::rounded(8), ..Default::default() }
            })
            .into()
    } else {
        ok_button.style(theme::primary_button(t)).into()
    };
    let mut body = column![text(title).size(16).font(crate::models::bold()), text(sentence).size(13).color(t.fg_soft)].spacing(14);
    if let Some(value) = input {
        body = body.push(text_input("", value).on_input(|v| Message::Lp(LpMsg::MemEditInput(v))).on_submit(Message::Lp(LpMsg::DialogOk)).size(13).padding([6, 10]));
    }
    body = body.push(row![Space::new().width(Length::Fill), button(text("Cancel").size(13)).on_press(Message::Lp(LpMsg::DialogCancel)).padding([6, 12]).style(theme::ghost_button(t)), ok_button].spacing(8));
    container(container(body).padding(16).width(460).style(theme::card(t, theme::R_LG, false)))
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(24)
        .into()
}
