//! Code mode: a coding agent in a folder.
//!
//! The model gets the tools of the server's agent (`/api/studio/agent/tools`: read, search, edit, run commands, ...) and answers with tool
//! calls; this file runs them through `/api/studio/agent/tool` and feeds the results back until the model answers without a call. Tools that
//! change something wait for your approval unless you chose a freer mode; every file change can be undone.

use crate::api::{self, ChatParams, StreamEvent, WireMessage};
use crate::app::Message;
use crate::md;
use crate::models::{bold, rule};
use crate::settings::segmented;
use crate::theme::{self, Tokens};
use iced::widget::text_editor::{Action, Binding, Content, KeyPress, Status};
use iced::widget::{button, checkbox, column, container, markdown, row, scrollable, text, text_editor, text_input, Space};
use iced::{keyboard, task, Alignment, Element, Font, Length, Padding, Task};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

mod session_cmds;
mod side_views;
pub use session_cmds::SessMsg;
pub use side_views::{choices_view, meter_text};

/// The most model calls one request of yours may take: a loop that never ends is stopped here.
pub const MAX_STEPS: usize = 30;

const PLAN_PROMPT: &str = "PLAN MODE: you may only read and search. You must not change files, run commands that change anything, or call tools that write. Investigate the code that matters with list_dir, glob, grep and read_file, then answer with a plan and stop. A good plan: a one-sentence goal; the files to change and what changes in each (path and function); the order of the steps; how to verify (exact commands); risks and open questions. Be specific enough that someone else could carry it out. The user will tell you when to carry the plan out.";

const CODE_PROMPT: &str = "You are a coding agent working in the user's project folder. Use the tools to look at the code before you answer or change anything: list_dir, glob and grep to find files, read_file to read them. Read a file before you edit it, make small precise edits with edit_file, and check your work (run the tests or the build with bash when there are any). Answer briefly and say what you changed. Never invent file contents or command output.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    Read,
    Edit,
    Command,
    Memory,
    Network,
    /// Operating the screen: only offered when computer use is switched on, which this window does not do.
    Computer,
    Other,
}

impl Permission {
    pub fn parse(text: &str) -> Permission {
        match text {
            "read" => Permission::Read,
            "edit" => Permission::Edit,
            "command" => Permission::Command,
            "memory" => Permission::Memory,
            "network" => Permission::Network,
            "computer" => Permission::Computer,
            _ => Permission::Other,
        }
    }
}

/// How much the agent may do on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Reading and the agent's own notes are free; edits, commands and the network ask first.
    Ask,
    /// Edits are free too.
    AutoEdit,
    /// Everything runs without asking.
    Auto,
    /// The plan agent: it may only read and answers with a plan; nothing else is offered or run.
    Plan,
}

impl Mode {
    pub fn allows(self, p: Permission) -> bool {
        match p {
            Permission::Read | Permission::Memory => true,
            Permission::Edit => matches!(self, Mode::AutoEdit | Mode::Auto),
            Permission::Command | Permission::Network | Permission::Computer | Permission::Other => self == Mode::Auto,
        }
    }
}

/// Where a tool runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolKind {
    /// On the server (`/api/studio/agent/tool`).
    Server,
    /// A tool of an MCP server, called through `/api/studio/mcp/call`.
    Mcp { server: String, tool: String },
    /// The task list of the session, kept in the window (`todo_write` replaces it, `todo_read` reads it).
    Todo { write: bool },
    /// A question to you with a few answers; your answer is the result.
    Ask,
    /// `task`: a sub-agent with a conversation of its own, started by the window (`runSubAgent`).
    Task,
    /// `create_agent`: writes the file of a custom sub-agent through `/api/studio/agents`.
    CreateAgent,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub permission: Permission,
    pub kind: ToolKind,
}

impl Tool {
    pub fn server(name: &str, permission: Permission) -> Tool {
        Tool { name: name.into(), permission, kind: ToolKind::Server }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Todo {
    pub content: String,
    pub status: TodoStatus,
}

/// What the server knows about the folder and you: the project's instruction files (AGENTS.md and the like, the nearest first), the notes the
/// agent kept in earlier sessions and the short profile of you; all of it goes into the instructions of the model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Context {
    pub files: Vec<(String, String)>,
    pub memory_workspace: Vec<String>,
    pub memory_global: Vec<String>,
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Workspace {
    pub root: String,
    pub name: String,
    pub branch: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Change {
    pub id: u64,
    pub path: String,
    pub kind: String,
    /// When the server made it (`c.time` of the page: seconds or milliseconds), 0 when it says nothing.
    pub time: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CallState {
    /// Waits for you.
    Pending,
    Running,
    Done { ok: bool, output: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub id: String,
    pub name: String,
    /// The arguments as the model wrote them (JSON text).
    pub args: String,
    pub permission: Permission,
    pub kind: ToolKind,
    pub state: CallState,
    /// The words under a card that ran without a question (`autoNote` of the page: *Approved on
    /// its own: …*).
    pub auto_note: Option<String>,
    /// The block of `item.check` on the card: the diagnostics an automatic check found after
    /// this edit, under *Automatic check*.
    pub check: Option<String>,
    /// What the model gets back when it differs from the card (`reply = …` of the automatic
    /// check: the clipped answer with the problems of the check between the two markers).
    pub sent: Option<String>,
    /// A `task` card: the conversation of its sub-agent as the card draws it (`item.children`).
    pub kid: Option<Box<KidView>>,
    /// A computer tool's card: the pictures (`item.image`, `item.shot`).
    pub cu: Option<Box<CuView>>,
    /// The user opened or closed the card (`item.open`); `None` leaves it to `defaultOpen`.
    pub open: Option<bool>,
    /// *Show all* / *Show n more lines* was pressed (`item.expanded`).
    pub expanded: bool,
}

impl Call {
    /// The state of the card as the page names it (`item.status`).
    pub fn status(&self) -> crate::toolcard::Status {
        use crate::toolcard::Status;
        match &self.state {
            CallState::Pending => Status::Pending,
            CallState::Running => Status::Running,
            CallState::Done { ok: true, .. } => Status::Ok,
            CallState::Done { ok: false, output } => {
                // the cards the page marks `denied`: refused by a rule, the mode, the user or the kind of agent
                const DENIED: [&str; 5] = ["Not available to sub-agents.", "This sub-agent is read-only.", "Blocked by a permission rule.", "You denied this.", "Stopped."];
                if DENIED.contains(&output.as_str()) || output.starts_with("Plan mode only allows reading") || output.starts_with("The user declined") {
                    Status::Denied
                } else {
                    Status::Error
                }
            }
        }
    }

    /// Whether the card is open: what the user chose, else `defaultOpen`.
    pub fn is_open(&self) -> bool {
        self.open.unwrap_or_else(|| {
            let has_output = matches!(&self.state, CallState::Done { output, .. } if !output.is_empty());
            let cu = self.cu.as_ref();
            crate::toolcard::default_open(&self.name, self.status(), has_output, cu.is_some_and(|v| v.image.is_some()), cu.is_some_and(|v| v.shot.is_some()), self.check.is_some())
        })
    }
}

/// What the card of a `task` call keeps of its sub-agent: its type, the words of the job, the
/// progress line while it works and the items of its conversation (`item.agentType`,
/// `item.progress`, `item.children`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KidView {
    pub agent_type: String,
    pub description: String,
    pub progress: String,
    pub items: Vec<KidItem>,
}

/// An item of a sub-agent's conversation: its words, or one of its calls (it asks like the
/// conversation does, and a sub-agent starts no further sub-agent, so a call here has no `kid`).
#[derive(Debug, Clone, PartialEq)]
pub enum KidItem {
    Assistant(String),
    Call(Box<Call>),
}

pub enum Item {
    User {
        /// What the view shows — the page's `display || userText`: `/init` stands in the list
        /// while the model gets `INIT_PROMPT`.
        text: String,
        /// What the model is sent with this message: the prompt when it differs from the text
        /// (the command's expansion, or the prompt with the shared output of `/status` and
        /// `/diff` in front), `None` for an ordinary message.
        sent: Option<String>,
    },
    Assistant { text: String, md: Box<markdown::Content> },
    Call(Call),
    Note(String),
    /// The self-review's message to the model (`selfReview` pushes it into `ctx.messages` and
    /// never shows it): the view skips it, the wire sends it.
    ReviewFeedback(String),
    /// The message a blocked stop hook gives the model (`ctx.messages.push` there — the notice
    /// with the same words stands in the list): the view skips it, the wire sends it.
    StopHook(String),
}

/// A notice with buttons (`pushChoices` of the page: sessions, rewind points, the models). It is
/// not kept with the session; it stands below the conversation until a button is pressed or the
/// next message is sent.
/// The skill of `/skill` in its dialog (`fllmSkillShowProposal`): the name, the scope and the text, all of them yours to change.
#[derive(Debug)]
pub struct SkillDraft {
    pub name: String,
    /// *This folder only* (else *For every project*).
    pub workspace: bool,
    pub editor: iced::widget::text_editor::Content,
    pub busy: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Choices {
    pub text: String,
    pub action: ChoiceAction,
    pub rows: Vec<(String, String)>,
}

/// What a press on one of the buttons of a `Choices` notice does (`item.action` of the page).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceAction {
    Resume,
    Rewind,
    Fork,
    Revert,
    Unrevert,
    Model,
}

/// What `/revert` took away (`code.reverted`), for *Undo the revert*.
#[derive(Debug, Clone, PartialEq)]
pub struct Reverted {
    pub items: Vec<Saved>,
    pub todos: Vec<crate::sessions::SavedTodo>,
    pub marks: Vec<crate::sessions::Mark>,
    pub compaction: Option<crate::sessions::Compaction>,
    pub restored: usize,
    pub title: String,
}

/// A screenshot a computer tool answered with (`r.image` of the page): the picture, the data URL
/// the model is sent, and its size in the pixels the coordinates refer to.
#[derive(Debug, Clone, PartialEq)]
pub struct Shot {
    pub url: String,
    pub handle: iced::widget::image::Handle,
    pub width: f64,
    pub height: f64,
}

impl Shot {
    /// `data:<mime>;base64,<data>` as the page builds it from the reply of the tool.
    pub fn from_reply(image: &Value) -> Option<Shot> {
        let data = image.get("data").and_then(Value::as_str).filter(|d| !d.is_empty())?;
        let mime = image.get("mime").and_then(Value::as_str).filter(|m| !m.is_empty()).unwrap_or("image/png");
        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data).ok()?;
        let num = |k: &str| image.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        Some(Shot { url: format!("data:{mime};base64,{data}"), handle: iced::widget::image::Handle::from_bytes(bytes), width: num("width"), height: num("height") })
    }
}

/// What a card of a computer tool shows besides its words (`item.image`, `item.shot`, `item.gone`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CuView {
    /// What the tool answered with: the screen after the action.
    pub image: Option<Shot>,
    /// The screenshot the action refers to: the card draws the marks on it.
    pub shot: Option<Shot>,
    /// The picture was not kept when the session was saved.
    pub gone: bool,
}

/// Everything the browser panel asks and answers (`browser.js`).
#[derive(Debug, Clone)]
pub enum BrMsg {
    /// The Browser button of the header (`setOpen(!br.open)`).
    Toggle,
    Hide,
    /// *Wider* / *Narrower* of the panel.
    Wide,
    /// Ctrl+L: into the address box.
    FocusAddress,
    /// One of the addresses under *Nothing is open yet* (`localhost:3000`).
    Quick(String),
    /// What `GET /api/studio/agent/browser` answered; `then_tick` when the panel was just opened.
    Status { then_tick: bool, answer: Option<Value> },
    Tick(u64),
    TickDone { epoch: u64, status: Option<Option<Value>>, frame: Option<Value> },
    /// `poke()`: look again now.
    Poke,
    Answered(crate::browser::Input, Result<Value, String>),
    UrlEdit(String),
    Submit,
    Back,
    Forward,
    Reload,
    CloseAsk,
    CloseOk,
    CloseCancel,
    Comment,
    /// Where the mouse is on the picture, and how big the picture is drawn.
    Move(iced::Point, iced::Size),
    Click,
    Double,
    Scroll(iced::mouse::ScrollDelta),
    WheelFlush,
    Key(crate::browser::KeyPress),
    Pasted(Option<String>),
    NoteEdit(iced::widget::text_editor::Action),
    NoteAdd,
    NoteCancel,
    Del(usize),
    SendComments,
    OpenSettings,
}

/// What the server says of computer use (`GET /api/studio/agent/computer`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CuStatus {
    pub enabled: bool,
    pub available: bool,
    pub steps_used: u64,
    pub max_steps: u64,
}

/// The window's side of computer use (`computer` and `code.lastShot` of the page).
#[derive(Debug, Clone, Default)]
pub struct Computer {
    pub status: Option<CuStatus>,
    /// What an action is doing right now, for the banner.
    pub busy: String,
    pub warned_vision: bool,
    pub stopping: bool,
    /// The times of the last presses of Esc (`computer.esc`).
    pub esc: Vec<f64>,
    /// Which chain of polls is the live one; an answer of an older one is ignored.
    pub epoch: u64,
    /// The last screenshot: the coordinates of the next action refer to it.
    pub last_shot: Option<Shot>,
}

/// An item of a session as it is kept on disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Saved {
    User {
        text: String,
        /// Kept with the session like the messages of the page are kept separately from its view.
        #[serde(default)]
        sent: Option<String>,
    },
    Assistant { text: String },
    Call {
        id: String,
        name: String,
        args: String,
        permission: String,
        ok: bool,
        output: String,
        #[serde(default)]
        sent: Option<String>,
        /// A `task` card keeps the conversation of its sub-agent and its type.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        children: Vec<Saved>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent: Option<String>,
        /// A computer card whose screenshot was not kept (`c.gone` of the page).
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        gone: bool,
    },
    Note { text: String },
    /// Kept with the session like the messages of the page are, so the review rounds survive it.
    ReviewFeedback { text: String },
    /// The same for the message of a blocked stop hook.
    StopHook { text: String },
}

/// A call as it is kept: one that had not finished is kept as interrupted, and a `task` keeps the
/// conversation of its sub-agent.
fn saved_call(c: &Call) -> Saved {
    let (ok, output) = match &c.state {
        CallState::Done { ok, output } => (*ok, output.clone()),
        _ => (false, "Interrupted.".to_string()),
    };
    let children: Vec<Saved> = c
        .kid
        .as_ref()
        .map(|k| {
            k.items
                .iter()
                .filter_map(|i| match i {
                    KidItem::Assistant(text) => (!text.is_empty()).then(|| Saved::Assistant { text: text.clone() }),
                    KidItem::Call(cc) => Some(saved_call(cc)),
                })
                .collect()
        })
        .unwrap_or_default();
    Saved::Call {
        id: c.id.clone(),
        name: c.name.clone(),
        args: c.args.clone(),
        permission: permission_name(c.permission).into(),
        ok,
        output,
        sent: c.sent.clone(),
        children,
        agent: c.kid.as_ref().map(|k| k.agent_type.clone()),
        // screenshots are never written to the disk: they are large, and they show the screen
        gone: c.cu.as_ref().is_some_and(|v| v.image.is_some() || v.shot.is_some() || v.gone),
    }
}

/// The call of a kept session (`kind` is the server's: the kinds of the window's own tools do
/// not matter for a call that is over).
fn restored_call(saved: Saved) -> Call {
    let Saved::Call { id, name, args, permission, ok, output, sent, children, agent, gone } = saved else {
        unreachable!("restored_call is given calls")
    };
    let description = serde_json::from_str::<Value>(&args)
        .ok()
        .and_then(|v| v.get("description").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_default();
    let kid = (name == "task" || agent.is_some()).then(|| Box::new(KidView {
        agent_type: agent.unwrap_or_default(),
        description,
        progress: String::new(),
        items: children
            .into_iter()
            .map(|c| match c {
                Saved::Assistant { text } => KidItem::Assistant(text),
                other => KidItem::Call(Box::new(restored_call(other))),
            })
            .collect(),
    }));
    Call { id, name, args, permission: Permission::parse(&permission), kind: ToolKind::Server, state: CallState::Done { ok, output }, auto_note: None, check: None, sent, kid, cu: gone.then(|| Box::new(CuView { gone: true, ..Default::default() })), open: None, expanded: false }
}

fn todo_saved(t: &Todo) -> crate::sessions::SavedTodo {
    crate::sessions::SavedTodo {
        content: t.content.clone(),
        status: match t.status {
            TodoStatus::Pending => "pending",
            TodoStatus::InProgress => "in_progress",
            TodoStatus::Completed => "completed",
        }
        .into(),
    }
}

fn todo_loaded(t: &crate::sessions::SavedTodo) -> Todo {
    Todo {
        content: t.content.clone(),
        status: match t.status.as_str() {
            "completed" => TodoStatus::Completed,
            "in_progress" => TodoStatus::InProgress,
            _ => TodoStatus::Pending,
        },
    }
}

/// The most items kept per folder: the newest ones.
pub const SAVED_ITEMS: usize = 200;

/// One run's self-review between its last answer and its `done` (`selfReview` of code.js): two
/// rounds at most, each reading the changes again, asking the review tool and the diagnostics
/// of every file this run touched, and sending the model back to work with the feedback.
#[derive(Debug, Clone, PartialEq)]
pub struct Review {
    /// The round the page is in: 1 or 2, and never more.
    pub round: u32,
    /// Where the cycle stands.
    pub stage: ReviewStage,
    /// The files this run changed, unique and at most twelve (`files` of the page).
    pub files: Vec<String>,
    /// What came back: the review text and the diagnostics that are not clean.
    pub report: crate::review::Report,
}

/// Which hook events the window knows of (`GET /api/studio/hooks`: the events of the hooks
/// themselves and the ones the plugins bring). The page asks the server on every submit, no
/// matter what it knows; this window asks only when a hook could answer — a request that is
/// not made returns exactly what an unanswered one does, and the tests, which have no server,
/// start with nothing hooked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HookGates {
    pub submit: bool,
    pub stop: bool,
    pub notification: bool,
}

/// Where a turn of the conversation stands at its end (`submitText` runs its tail in order):
/// the self-review of row 81, then the stop hook once, and then the turn is over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailPhase {
    /// The review may still start.
    Review,
    /// The stop hook may still be asked.
    Hook,
    /// The run the stop hook asked for: it ends plainly (the page checks once).
    Continue,
    /// Nothing of the tail is left.
    Done,
}

/// The three waits of one review round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewStage {
    /// `refreshChanges()` is being asked for.
    Changes,
    /// The review tool and the diagnostics are being asked — silently, as the page asks them
    /// (only the notice stands in the list).
    Tools,
    /// The feedback went out and the model is working on it again.
    Run,
}

/// What the silent tools of one review round answered (`selfReview`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReviewAnswers {
    /// The text of the review tool, when the server offers it.
    pub review: Option<String>,
    /// One entry per diagnostics call that worked: the path and its output.
    pub diagnostics: Vec<(String, String)>,
}

fn permission_name(p: Permission) -> &'static str {
    match p {
        Permission::Read => "read",
        Permission::Edit => "edit",
        Permission::Command => "command",
        Permission::Memory => "memory",
        Permission::Network => "network",
        Permission::Computer => "computer",
        Permission::Other => "other",
    }
}

pub use crate::sessions::Sessions;

#[derive(Default)]
struct Partial {
    id: String,
    name: String,
    args: String,
}

#[derive(Debug, Clone)]
pub enum CodeMsg {
    PathInput(String),
    Open,
    /// Open this folder (from the list of recent ones).
    OpenPath(String),
    /// Ask the server for its folder dialog.
    Pick,
    Picked(Result<Option<String>, String>),
    Close,
    Opened(Result<Workspace, String>),
    ToolsLoaded(Result<(Vec<Tool>, Value), String>),
    ChangesLoaded(Result<Vec<Change>, String>),
    ContextLoaded(Result<Context, String>),
    /// Your answer to a question of the agent (`ask_user`).
    Answer(String, String),
    Edit(Action),
    Send,
    Stop,
    Stream(StreamEvent),
    Approve(String),
    Deny(String),
    ToolDone(String, Result<String, String>),
    SetMode(Mode),
    Undo(Option<u64>),
    Undone(Result<String, String>),
    NewSession,
    /// Leave plan mode and tell the agent to carry out the plan it wrote.
    ExecutePlan,
    /// The *Always* button of a waiting call: what it remembers for this session, and the call
    /// runs like an *Allow* (`answerPermission('always')` of `code.js`).
    Always(String, String),
    /// The checkbox of *Review my work before it counts as done* (`code.autoReview`; the page
    /// writes `fllm.studio.codesettings.v1` when it changes).
    ToggleAutoReview(bool),
    /// The checkbox of *Check my edits automatically* (`code.autoCheck`).
    ToggleAutoCheck(bool),
    /// What the silent tools of one review round answered (`selfReview`).
    ReviewTools(ReviewAnswers),
    /// The diagnostics an automatic check asked for after an edit: the call, the reply the model
    /// had, and the answers it kept (one per path that was checked).
    CheckDone { id: String, reply: String, outputs: Vec<String> },
    /// What `user_prompt_submit` answered: the turn starts with its context in the prompt
    /// (`submitted.context` of `submitText`).
    SubmitHookDone { prompt: String, display: String, answer: HookFire },
    /// `computerBegin` was answered: the first request of the run goes out now.
    ComputerBegun { prompt: String, display: String },
    /// What the stop hook answered after the review (`fireHook('stop', …)` there).
    StopHookDone(HookFire),
    /// A `notification` hook was fired for a waiting card; nothing comes of it.
    HookFired,
    /// The custom sub-agents of the folder (`loadAgents`); `None` when the route did not answer,
    /// which leaves the list as it was.
    AgentsLoaded(Option<Vec<crate::subagents::CustomAgent>>),
    /// An event of one sub-agent's stream (the id is the `task` call that started it).
    KidStream(String, StreamEvent),
    /// The header of a card was pressed (`item.open`), or *Show all* / *Show n more lines* (`item.expanded`).
    ToggleCard(String),
    ExpandCard(String),
    /// The browser panel.
    Br(BrMsg),
    /// The sessions, the rewind and the compaction (row 79) and the blocks of the side that go with them.
    Sess(SessMsg),
    /// The browser settings were saved (`fllmBrowserChanged`): the panel reads its status again and the tools are asked for.
    BrowserChanged,
    /// What the server says of computer use (a poll of the chain `epoch`).
    ComputerStatus(u64, Option<CuStatus>),
    /// The STOP of the banner, Esc three times, Ctrl+Alt+Shift+Esc (`stopComputer(why)`).
    ComputerStop(String),
    ComputerStopped(String),
    /// A tool answered, with the picture it may have taken (`r.image`).
    ToolReply(String, Result<String, String>, Option<Shot>),
    /// *Output* of a background job: opens or closes its box, reading the output when it opens.
    JobOutput(String),
    JobOutputDone(String, Result<String, String>),
    /// *Stop* of a background job.
    JobKill(String),
    JobKillDone(String, Result<String, String>),
    /// What the server says of learning when a folder opens (`fllmLearnStatus`).
    LearnStatus(Option<crate::learn::Status>),
    /// The first-run sentence of the server, or nothing (`fllmLearnNotice`).
    LearnNotice(String),
    /// The checkbox *Learn from my sessions* of the side.
    LearnToggle(bool),
    /// The answer of the server to the switch.
    LearnSwitched(Result<(), String>),
    /// The server took the run (or did not) and the window starts to watch for its step; the
    /// number is the second the page watches from (`Date.now() / 1000 - 2`).
    LearnQueued(bool, f64),
    /// The timer of the watch: ask the server how the step goes (`tries` polls have been made).
    LearnPoll(f64, u32),
    LearnPolled(Option<crate::learn::Status>, f64, u32),
    /// *Skills helped* / *Skills did not help* and `/good`, `/bad`.
    LearnThumb(&'static str),
    LearnThumbDone(&'static str, Vec<String>, Result<(u64, Vec<String>), String>),
    /// What `/learn` was about, once the server has answered: the plan, the status and an error
    /// of the switch.
    LearnCommand(crate::learn::LearnPlan, Option<crate::learn::Status>, Option<String>),
    /// The specs of the folder as they are (`loadSpecs` of the page).
    SpecsLoaded(Vec<crate::specs::Spec>),
    /// The listing of `/spec` with no argument, ready to be written into the list.
    SpecListDone(Vec<crate::specs::Spec>),
    /// A prompt of a spec is ready: the run starts with `/spec …` as its display.
    SpecPrompt { prompt: String, display: String },
    /// An answer of the spec route that changes the list (`tick` / `untick`).
    SpecReplaced(Vec<crate::specs::Spec>),
    /// A spec route failed: the message the page would show — as a toast where the page toasts
    /// it (the checkbox of the side) and in the list where it notices it.
    SpecFailed { message: String, toast: bool },
    /// A checkbox of the side panel (`specAction` with `tick` or `untick`).
    SpecTick { name: String, index: u64, done: bool },
    /// The × of a queued message.
    QueueDrop(usize),
    /// The *Run next task* button of the side panel (`specAction` with `task_prompt`).
    SpecRunTask { name: String, index: u64 },
    /// What `/worktree` and `/batch` asked the server: what the answer does (switch the
    /// folder, notice the rows) happens here, as the page does it after its fetch.
    WorktreeDone {
        action: &'static str,
        name: Option<String>,
        batch: Option<String>,
        result: Result<Value, String>,
    },
    /// The custom commands of the folder are known (`loadCustomCommands` of `code.js`).
    CustomLoaded(Vec<crate::commands::CustomCommand>),
    /// One custom command file (or the plugin's) was read: run what it holds.
    CustomBody(Result<String, String>, String, String),
    /// The answer of `GET /api/studio/sessions/search` for `/search`, with the words.
    SearchDone(Result<Vec<crate::commands::SearchHit>, String>, String),
    /// A command the user ran itself finished (`userRun` of the page).
    UserRunDone(String, Result<String, String>, bool),
    /// `saveMemoryEntry` of the page: the notice when it saved, the toast when it did not.
    MemorySaved(Result<(), String>, String),
}

/// What the agent needs from the window for a request.
pub struct Env<'a> {
    pub base: &'a str,
    pub model: Option<&'a str>,
    pub params: ChatParams,
    /// The custom system prompt of the Settings page, added to the agent's own instructions; `None` while that is automatic.
    pub extra: Option<&'a str>,
    /// The runtime switches a command reads and writes (`/turbo`, `/sandbox`).
    pub turbo: bool,
    /// The sandbox mode as the server reports it, `None` when it says nothing (the page treats an
    /// empty one the same: `code.sandbox || 'unknown'`).
    pub sandbox: Option<&'a str>,
    pub sandbox_available: bool,
    /// The `/cost` summary of the window.
    pub usage: Option<&'a str>,
    /// Where the session of this folder is kept, so `/clear` can save before it resets.
    pub session_file: &'a std::path::Path,
    /// The context of the loaded model: `clipOutput` of the automatic check cuts its answer
    /// against it (`ctxMax()` of the page).
    pub ctx_max: usize,
    /// Which hook events the window could ask about (empty in tests, and then no request is
    /// made at all — the page gets an empty answer from a server without the route).
    pub hooks: HookGates,
    /// Whether the loaded model can look at pictures (a computer tool's screenshots go to it).
    pub vision: bool,
    /// Whether the Code page is the one shown (the browser panel does not poll behind another page).
    pub visible: bool,
    /// The names `/model` offers (`modelNames()` of the page: the models of the computer).
    pub models: &'a [String],
}

/// What a command changed and the store has to keep (`saveCodeSettings` and the rules write of
/// the page, which go into its own storage at once).
#[derive(Debug, Clone, Default)]
pub struct PendingPrefs {
    pub rules: Option<crate::commands::Rules>,
    pub personality: Option<String>,
}

pub struct CodeState {
    pub path_input: String,
    pub workspace: Option<Workspace>,
    pub tools: Vec<Tool>,
    tools_json: Value,
    pub items: Vec<Item>,
    pub editor: Content,
    pub mode: Mode,
    /// The mode to go back to when a plan is carried out.
    pub build_mode: Mode,
    /// Whether harmless commands run without a question (`code.autoApprove` of the page).
    pub auto_approve: bool,
    /// `Review my work before it counts as done` (`code.autoReview`): when a run changed files,
    /// the agent reads its own diff and the diagnostics once more before it counts as done.
    pub auto_review: bool,
    /// `Check my edits automatically` (`code.autoCheck`): after every edit the first problems
    /// of a diagnostics run go back to the model with its result.
    pub auto_check: bool,
    /// The switch as the command last set it, for the window to write into its store — the page
    /// writes its own copy at once, and this state has no store of its own.
    pub(crate) pending_auto_approve: Option<bool>,
    /// The same hand-off for `/autoreview` and its checkbox (`saveCodeSettings` of the page).
    pub(crate) pending_auto_review: Option<bool>,
    /// The same hand-off for the checkbox of `code.autoCheck`.
    pub(crate) pending_auto_check: Option<bool>,
    /// The highest change id when this run started (`turn.changeMark`): the self-review looks at
    /// the files this run changed and at nothing before them.
    pub change_mark: u64,
    /// The review between the run's end and its `done` (`selfReview` of the page), while it runs.
    pub review: Option<Review>,
    /// Whether the stream that is running belongs to the conversation itself (`submitText` of
    /// the page, where `selfReview` is called): the runs of the agents pane do not review.
    pub(crate) main_run: bool,
    /// The call whose automatic check is running: the model waits for it (`autoCheckEdit` is
    /// awaited inside `runToolCall` on the page, before the call is answered).
    pub(crate) check_hold: Option<String>,
    /// The hook events this window knows of — own ones and the plugins' — from
    /// `GET /api/studio/hooks`; empty until the list is read.
    pub hook_events: Vec<String>,
    /// Where the tail of the current turn stands (see `TailPhase`).
    pub(crate) tail: TailPhase,
    /// A notice that waits for the folder of a worktree to be open (the page awaits
    /// `setWorkspace` before it writes it).
    pub(crate) pending_notice: Option<String>,
    /// The specs of the folder, as the side panel keeps them (`code.specs`).
    pub specs: Vec<crate::specs::Spec>,
    /// The task of `/batch`, waiting for the tools of the new folder (the page awaits
    /// `setWorkspace`, which loads them, before it runs `agentRun`).
    pub(crate) pending_batch: Option<String>,
    /// The allow/deny rules of `/allow` and `/deny`, kept per folder like the page keeps them per
    /// workspace in its own storage.
    pub rules: crate::commands::Rules,
    /// How the agent talks (`/personality`): a key of `PERSONALITIES`, in the system prompt where
    /// the page puts its `personalityNote`.
    pub personality: String,
    /// The custom commands of `.fllm/commands` and of the installed plugins, loaded with the
    /// folder (`loadCustomCommands` of `code.js`).
    pub custom: Vec<crate::commands::CustomCommand>,
    /// What the window still writes into its store: the rules of this folder and the style.
    pub(crate) pending_prefs: Option<PendingPrefs>,
    /// A line for the toast of the window (`toast(...)` of the page) — `/agent` while the agent
    /// runs, and a memory that could not be saved.
    pub pending_toast: Option<String>,
    /// What `/status` and `/diff` ran: the output goes with the next message (`pending` of the
    /// page), never into the conversation itself.
    pub pending: Vec<String>,
    /// `/turbo`: the switch for the window to ask its confirm and post (the answer is noted on
    /// this page when `runtime_notice` of the window says so).
    pub(crate) pending_turbo: Option<bool>,
    /// `/sandbox <mode>`: the mode for the window to post.
    pub(crate) pending_sandbox: Option<String>,
    /// What `Always` remembered (`code.alwaysAllow` of the page): keys like `bash:npm`, `network`
    /// or `mcp:server/tool`. The page clears them with the session, and never saves them.
    pub always_allow: std::collections::HashSet<String>,
    pub running: bool,
    pub steps: usize,
    /// The last calls of this run, as signatures (`ctx.history`): the same call three times in a
    /// row is refused (`loopguard`).
    history: Vec<String>,
    pub changes: Vec<Change>,
    pub context: Context,
    pub todos: Vec<Todo>,
    pub error: Option<String>,
    acc: Vec<Partial>,
    handle: Option<task::Handle>,
    /// The custom sub-agents of the folder (`code.agents`, `loadAgents`).
    pub agents: Vec<crate::subagents::CustomAgent>,
    /// The sub-agents that work or wait for a place (at most `MAX_PARALLEL_CHILDREN` work at a time).
    kids: Vec<Kid>,
    /// *Learn from my sessions* (`code.learn`, on by default): after a run the server learns from it.
    pub learn: bool,
    /// The switch changed and the store keeps the new state (`saveCodeSettings`).
    pub(crate) pending_learn: Option<bool>,
    /// Where the run in progress began in the conversation (`turn.msgIndex`).
    learn_from: Option<usize>,
    /// The skills of the last run that was handed over (`window.fllmLearnLast`): what a thumb is about.
    pub learn_last: Option<crate::learn::Last>,
    /// What the server calls this conversation in its log (`code.sessionId`), and the title and the
    /// lock of the title the sessions list shows (`code.title`, `code.titleLocked`).
    pub session_id: String,
    pub title: String,
    pub title_locked: bool,
    /// The moments of the messages of yours (`msgIndex`, `todosBefore` and `changeMark` of the
    /// page's user items): a fork and a revert cut the conversation there.
    pub marks: Vec<crate::sessions::Mark>,
    /// What stands for the first messages after `/compact`.
    pub compaction: Option<crate::sessions::Compaction>,
    /// The summary is being written.
    pub compacting: bool,
    pub reverted: Option<Reverted>,
    /// The notices with buttons below the conversation.
    pub choices: Vec<Choices>,
    /// `/retry` waits for the revert to finish before the message goes out again.
    pub(crate) retry_pending: bool,
    /// How many times this run already summarised the conversation (two at most, `compactions`).
    compactions: u8,
    /// The context of the loaded model, as the last update read it (`ctxMax()`).
    pub ctx_max: usize,
    /// The title of the session is being asked of the model.
    titling: bool,
    /// The side blocks *Memory* and *Permissions* are open (`code.sideOpen`).
    pub mem_open: bool,
    pub perms_open: bool,
    /// The two boxes of *Memory* (one fact per line) and whether each was changed (`code.memDraft`).
    pub mem_ws: Content,
    pub mem_gl: Content,
    pub mem_dirty: (bool, bool),
    /// The two boxes of *Permissions* (`code.rulesDraft`).
    pub rules_allow: Content,
    pub rules_deny: Content,
    pub rules_dirty: bool,
    /// The model the agent was told to use with `/model` (the app switches to it).
    pub(crate) pending_model: Option<String>,
    /// The sessions of this folder, the newest first (`workspaceSessions()`): read from the file
    /// when the folder opens and after every save.
    pub sessions: Vec<crate::sessions::Record>,
    pub session_query: String,
    /// The session being renamed and the title typed so far.
    pub rename: Option<(String, String)>,
    /// The session that waits for the answer of *Delete this session?*.
    pub confirm_delete: Option<String>,
    /// The skill the model wrote from the session, waiting for *Save skill* or *Discard*.
    pub skill: Option<SkillDraft>,
    /// `/skill` asked the model and the answer has not come yet.
    pub skill_asking: bool,
    /// Messages typed while the agent was busy: each is sent when the run has ended (`code.queue`).
    pub queue: Vec<String>,
    /// The agent is told that the window draws charts, diagrams and math (*Charts, diagrams and math in answers*; the app sets it).
    pub rich_answers: bool,
    /// The first request of a run waits for the answer of `computer/begin`.
    pub beginning: bool,
    /// Whether the server answered `GET /api/studio/agent/changes` (the page: `code.changes === null`).
    pub changes_known: bool,
    /// A session the app has to write the file for (the page calls `saveCodeSession()` itself).
    pub(crate) wants_save: bool,

    /// A learning step ended: Settings > Personalization and Skills read their lists again.
    pub(crate) learn_refresh: bool,
    /// The commands the agent started in the background (`code.jobs`).
    pub jobs: Vec<crate::jobs::Job>,
    /// Computer use: what the server says, the banner, the last screenshot.
    pub computer: Computer,
    /// The browser panel (`br` of `browser.js`) and what the window adds to draw it.
    pub browser: crate::browser::Panel,
    pub br_epoch: u64,
    br_sending: bool,
    br_queue: std::collections::VecDeque<crate::browser::Input>,
    br_handle: Option<iced::widget::image::Handle>,
    br_src: String,
    br_pointer: Option<(iced::Point, iced::Size)>,
    pub br_focus: bool,
    /// The wide panel (*Wider*) instead of the narrow one.
    pub br_wide: bool,
    br_note: Content,
    br_confirm_close: bool,
    /// The comments the user left on the page, taken at the start of a task (`notes` of `agentRunNow`).
    br_notes: String,
    /// Whether the loaded model can look at pictures (a screenshot of the browser goes to it).
    vision: bool,
}

/// Where one sub-agent's loop stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KidPhase {
    /// Four others work: it starts when one is done (the page runs groups of four).
    Queued,
    /// The model is answering.
    Streaming,
    /// The calls of its last answer are being run or waiting for you.
    Calls,
}

/// One sub-agent's conversation with the model (`ctx` of `runSubAgent`): its own messages, its own
/// stream and its own step count. Its cards live in the `task` call that started it.
struct Kid {
    call_id: String,
    plan: crate::subagents::ChildPlan,
    phase: KidPhase,
    system: String,
    tools: Value,
    messages: Vec<WireMessage>,
    budget: usize,
    step: usize,
    wrap_up: bool,
    acc: Vec<Partial>,
    /// What the model has said in this step.
    text: String,
    /// The ids of the calls of this step, in the order the model made them.
    calls: Vec<String>,
    handle: Option<task::Handle>,
    /// Its last calls (`ctx.history`), for the guard against the same call again and again.
    history: Vec<String>,
}

impl Default for CodeState {
    fn default() -> Self {
        CodeState {
            path_input: String::new(),
            workspace: None,
            tools: Vec::new(),
            tools_json: Value::Null,
            items: Vec::new(),
            editor: Content::new(),
            mode: Mode::Ask,
            build_mode: Mode::Ask,
            auto_approve: true,
            pending_auto_approve: None,
            auto_review: false,
            pending_auto_review: None,
            auto_check: false,
            pending_auto_check: None,
            change_mark: 0,
            review: None,
            main_run: false,
            check_hold: None,
            hook_events: Vec::new(),
            tail: TailPhase::Done,
            pending_notice: None,
            pending_batch: None,
            specs: Vec::new(),
            rules: crate::commands::Rules::default(),
            personality: "default".into(),
            custom: Vec::new(),
            pending_prefs: None,
            skill: None,
            skill_asking: false,
            queue: Vec::new(),
            rich_answers: false,
            beginning: false,
            pending_toast: None,
            pending: Vec::new(),
            pending_turbo: None,
            pending_sandbox: None,
            always_allow: std::collections::HashSet::new(),
            running: false,
            steps: 0,
            history: Vec::new(),
            changes: Vec::new(),
            context: Context::default(),
            todos: Vec::new(),
            error: None,
            acc: Vec::new(),
            handle: None,
            agents: Vec::new(),
            kids: Vec::new(),
            learn: true,
            pending_learn: None,
            learn_from: None,
            learn_last: None,
            session_id: String::new(),
            title: String::new(),
            title_locked: false,
            marks: Vec::new(),
            compaction: None,
            compacting: false,
            reverted: None,
            choices: Vec::new(),
            retry_pending: false,
            compactions: 0,
            ctx_max: 0,
            titling: false,
            mem_open: false,
            perms_open: false,
            mem_ws: Content::new(),
            mem_gl: Content::new(),
            mem_dirty: (false, false),
            rules_allow: Content::new(),
            rules_deny: Content::new(),
            rules_dirty: false,
            pending_model: None,
            sessions: Vec::new(),
            session_query: String::new(),
            rename: None,
            confirm_delete: None,
            changes_known: false,
            wants_save: false,
            learn_refresh: false,
            jobs: Vec::new(),
            computer: Computer::default(),
            browser: crate::browser::Panel::default(),
            br_epoch: 0,
            br_sending: false,
            br_queue: std::collections::VecDeque::new(),
            br_handle: None,
            br_src: String::new(),
            br_pointer: None,
            br_focus: false,
            br_wide: false,
            br_note: Content::new(),
            br_confirm_close: false,
            br_notes: String::new(),
            vision: true,
        }
    }
}

// ---- reading the answers ---------------------------------------------------------------------------------------------------------

pub fn parse_workspace(v: &Value) -> Result<Workspace, String> {
    if let Some(e) = v.pointer("/error/message").and_then(Value::as_str).or_else(|| v.get("error").and_then(Value::as_str)) {
        return Err(e.to_string());
    }
    let root = v.get("root").and_then(Value::as_str).ok_or("the server did not name the folder")?.to_string();
    Ok(Workspace { name: v.get("name").and_then(Value::as_str).unwrap_or(&root).to_string(), branch: v.get("git_branch").and_then(Value::as_str).map(str::to_string), root })
}

/// The tools of the agent: the OpenAI-format list for the model, and name and permission of each for the window.
pub fn parse_tools(v: &Value) -> (Vec<Tool>, Value) {
    parse_tools_with(v, false)
}

/// The same with the screen and input tools when computer use is on: they only count when the
/// status route says so (`computerOn()` in `loadRegistry`).
pub fn parse_tools_with(v: &Value, computer_on: bool) -> (Vec<Tool>, Value) {
    let list: Vec<Value> = v
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|t| computer_on || t.get("permission").and_then(Value::as_str) != Some("computer"))
        .collect();
    let tools = list
        .iter()
        .filter_map(|t| Some(Tool::server(t.pointer("/function/name")?.as_str()?, Permission::parse(t.get("permission").and_then(Value::as_str).unwrap_or("")))))
        .collect();
    // the model gets the OpenAI shape only; "permission" is ours
    let for_model: Vec<Value> = list.iter().map(|t| json!({"type": "function", "function": t.get("function").cloned().unwrap_or(Value::Null)})).collect();
    (tools, Value::Array(for_model))
}

pub fn parse_changes(v: &Value) -> Vec<Change> {
    v.get("changes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|c| Change { id: c.get("id").and_then(Value::as_u64).unwrap_or(0), path: c.get("path").and_then(Value::as_str).unwrap_or_default().to_string(), kind: c.get("kind").and_then(Value::as_str).unwrap_or("edit").to_string(), time: c.get("time").and_then(Value::as_f64).unwrap_or(0.0) })
        .collect()
}

/// The tools the window itself provides: a task list and questions to you.
pub fn client_tools() -> (Vec<Tool>, Vec<Value>) {
    let def = |name: &str, description: &str, properties: Value, required: Value| json!({"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}});
    let tools = vec![
        Tool { name: "todo_write".into(), permission: Permission::Read, kind: ToolKind::Todo { write: true } },
        Tool { name: "todo_read".into(), permission: Permission::Read, kind: ToolKind::Todo { write: false } },
        Tool { name: "ask_user".into(), permission: Permission::Read, kind: ToolKind::Ask },
        Tool { name: "task".into(), permission: Permission::Read, kind: ToolKind::Task },
        Tool { name: "create_agent".into(), permission: Permission::Edit, kind: ToolKind::CreateAgent },
    ];
    let defs = vec![
        def(
            "todo_write",
            "Keep a task list for work with several steps. Send the complete list every time; exactly one task should be in_progress.",
            json!({"todos": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"]}}}),
            json!(["todos"]),
        ),
        def("todo_read", "Read the current task list (use it when you have lost track of the plan).", json!({}), json!([])),
        def(
            "ask_user",
            "Ask the user a question with a few answer choices when a decision is theirs and you cannot find the answer yourself. The answer comes back as the result.",
            json!({"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}, "description": "Two to five short choices."}}),
            json!(["question"]),
        ),
        def("task", "Start a sub-agent with a fresh context and get back only its final report. agent \"explore\" is read-only and fast: use it for broad research (\"where is X handled\", \"how does Y work\"). agent \"general\" may also edit files and run commands: use it for a self-contained job. Several task calls in one turn run in parallel. The sub-agent does not see this conversation, so the prompt must contain everything it needs: the goal, where to look, and what the report must contain.", json!({"description": {"type": "string", "description": "Three to six words naming the job."}, "prompt": {"type": "string", "description": "The complete task and what the report should contain."}, "agent": {"type": "string", "enum": ["explore", "general"], "description": "explore (default): read and search only. general: may also edit and run commands, with the usual permission questions."}}), json!(["description", "prompt"])),
        def("create_agent", "Create a reusable sub-agent as a file that the task tool can start by name afterwards (like a custom agent in Claude Code). Use it when a kind of job comes back again and again (review, test writing, migrations, research in one area). The user is asked before it is written. Give it a short lowercase name, a description of when to use it, the tools it may use (for example read_file, grep, glob, list_dir for a reviewer that must not change anything; leave the list empty for all tools) and its instructions: what it does, how it works and what its report looks like.", json!({"name": {"type": "string", "description": "Lowercase letters, digits, - and _ (for example code-reviewer)."}, "description": {"type": "string", "description": "One sentence: when to use this agent."}, "instructions": {"type": "string", "description": "The system prompt of the agent."}, "tools": {"type": "array", "items": {"type": "string"}, "description": "Tool names it may use; empty means all."}, "scope": {"type": "string", "enum": ["workspace", "user"], "description": "workspace (default): only for this folder. user: for every folder."}, "model": {"type": "string", "description": "Optional: another model for this agent."}}), json!(["name", "description", "instructions"])),
    ];
    (tools, defs)
}

/// The name a model can use for a tool of an MCP server: letters, digits, `_` and `-`, at most 64 characters.
pub fn mcp_name(server: &str, tool: &str) -> String {
    format!("mcp__{server}__{tool}").chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).take(64).collect()
}

/// The tools of the connected MCP servers (`/api/studio/mcp/tools`): each asks first, like a command.
pub fn parse_mcp_tools(v: &Value) -> (Vec<Tool>, Vec<Value>) {
    let mut tools = Vec::new();
    let mut defs = Vec::new();
    for t in v.get("tools").and_then(Value::as_array).into_iter().flatten() {
        let (Some(server), Some(tool)) = (t.get("server").and_then(Value::as_str), t.get("name").and_then(Value::as_str)) else { continue };
        let name = mcp_name(server, tool);
        let description = format!("[{server}] {}", t.get("description").and_then(Value::as_str).unwrap_or_default()).trim().to_string();
        let parameters = t.get("input_schema").filter(|s| s.is_object()).cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        tools.push(Tool { name: name.clone(), permission: Permission::Command, kind: ToolKind::Mcp { server: server.into(), tool: tool.into() } });
        defs.push(json!({"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}));
    }
    (tools, defs)
}

pub fn parse_context(v: &Value) -> Context {
    let strings = |p: &str| v.pointer(p).and_then(Value::as_array).into_iter().flatten().filter_map(|e| e.as_str().map(str::to_string)).collect::<Vec<_>>();
    Context {
        files: v.get("files").and_then(Value::as_array).into_iter().flatten().filter_map(|f| Some((f.get("file")?.as_str()?.to_string(), f.get("text")?.as_str()?.to_string()))).collect(),
        memory_workspace: strings("/memory/workspace"),
        memory_global: strings("/memory/global"),
        profile: v.get("profile").and_then(Value::as_str).unwrap_or_default().to_string(),
    }
}

fn capped(text: &str, max: usize) -> String {
    if text.chars().count() > max {
        format!("{}\n...", text.chars().take(max).collect::<String>())
    } else {
        text.to_string()
    }
}

impl Context {
    /// The part of the instructions that comes from the folder and from you, capped like the Studio caps it.
    pub fn prompt(&self) -> String {
        let mut out: Vec<String> = Vec::new();
        let (mut used, mut parts) = (0usize, Vec::new());
        for (file, text) in &self.files {
            let room = 9000usize.saturating_sub(used);
            if room < 200 {
                break;
            }
            let text = capped(text, room);
            used += text.chars().count();
            parts.push(format!("({file})\n{text}"));
        }
        if !parts.is_empty() {
            out.push(format!("Instructions of this project (follow them; the first is the most specific)\n{}", parts.join("\n\n")));
        }
        let list = |title: &str, items: &[String]| if items.is_empty() { String::new() } else { format!("{title}\n{}", items.iter().map(|e| format!("- {e}")).collect::<Vec<_>>().join("\n")) };
        let memory = [list("About this workspace:", &self.memory_workspace), list("About the user (all workspaces):", &self.memory_global)].into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n");
        if !memory.is_empty() {
            out.push(format!("Memory (facts saved in earlier sessions)\n{}", capped(&memory, 3000)));
        }
        if !self.profile.trim().is_empty() {
            out.push(format!("About the user (notes from earlier sessions and the user's own edits; adapt to them, do not mention them unless asked)\n{}", capped(self.profile.trim(), 1200)));
        }
        out.join("\n\n")
    }
}

/// The text a tool answer carries: its output, or its error.
pub fn tool_result(v: &Value) -> Result<String, String> {
    if v.get("ok").and_then(Value::as_bool) == Some(true) {
        Ok(v.get("output").and_then(Value::as_str).unwrap_or_default().to_string())
    } else {
        Err(v.get("error").and_then(Value::as_str).or_else(|| v.pointer("/error/message").and_then(Value::as_str)).unwrap_or("the tool failed").to_string())
    }
}

// ---- asking the server -----------------------------------------------------------------------------------------------------------

async fn call(base: &str, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<Value, String> {
    let mut req = reqwest::Client::new().request(method, format!("{base}{path}"));
    if let Some(b) = body {
        req = req.json(&b);
    }
    let resp = req.send().await.map_err(|e| format!("no server at {base}: {e}"))?;
    let status = resp.status();
    let data: Value = resp.json().await.unwrap_or(Value::Null);
    if status.is_success() {
        Ok(data)
    } else {
        Err(data.pointer("/error/message").and_then(Value::as_str).or_else(|| data.get("error").and_then(Value::as_str)).map(str::to_string).unwrap_or_else(|| format!("the server answered {status}")))
    }
}

/// One worktree action (`POST /api/studio/agent/worktree`, `case 'worktree'`). The page reads
/// its failures in two places — the answer of the server and the request itself — and this
/// keeps them apart as `Server` (with the message the page's `data.error.message` would give,
/// or its fallback) and `Transport` (no server at all).
pub async fn worktree_fetch(base: String, action: &str, name: Option<String>) -> Result<Value, String> {
    let mut body = json!({ "action": action });
    // the page's `{ action, name }` drops an undefined name; `null` would not
    if let Some(n) = name {
        body["name"] = json!(n);
    }
    match call(&base, reqwest::Method::POST, "/api/studio/agent/worktree", Some(body)).await {
        Ok(v) => Ok(v),
        Err(e) => Err(if e.starts_with("no server at") {
            format!("transport:{e}")
        } else if e.starts_with("the server answered") {
            // no message in the body: the fallback of the page
            "server:".to_string()
        } else {
            format!("server:{e}")
        }),
    }
}

/// The message the page shows for a failed worktree action: `data.error.message` when the body
/// has one, its own fallback when it has not, and the caught error of the request itself.
pub fn worktree_error(base: &str, e: &str) -> String {
    if let Some(rest) = e.strip_prefix("server:") {
        if rest.is_empty() {
            // no message in the body: the fallback of the page, with its full stop
            return format!("{base}.");
        }
        return rest.to_string();
    }
    format!("{base}: {}", e.strip_prefix("transport:").unwrap_or(e))
}

/// The whole `run` branch of the page's `specCommand`: find the spec in the list the window
/// has, reload once when the name is missing (`if (!sp) await loadSpecs()`), pick the task with
/// `next_index` and ask the server for its prompt. The answer is the prompt with the page's
/// display, or the message it would show.
pub async fn spec_run(
    base: String,
    current: Vec<crate::specs::Spec>,
    name: String,
    asked: Option<f64>,
) -> Result<(String, String), String> {
    use crate::specs::{decide_run, no_spec_notice, spec_act, task_prompt_body, RunDecision, ALL_DONE_NOTICE};
    let first = decide_run(&current, &name, asked);
    let decision = match first {
        RunDecision::NeedReload => match fetch_specs_maybe(&base).await {
            Some(fresh) => match decide_run(&fresh, &name, asked) {
                RunDecision::NeedReload => RunDecision::NotFound,
                other => other,
            },
            None => RunDecision::NotFound,
        },
        other => other,
    };
    match decision {
        RunDecision::Act { name, index } => {
            let v = spec_act(base, task_prompt_body(&name, index)).await?;
            let prompt = v.get("prompt").and_then(Value::as_str).unwrap_or_default().to_string();
            Ok((prompt, crate::specs::run_display(&name, index)))
        }
        RunDecision::NotFound => Err(no_spec_notice(&name)),
        RunDecision::AllDone => Err(ALL_DONE_NOTICE.to_string()),
        RunDecision::NeedReload => unreachable!("folded above"),
    }
}

/// `new_prompt` of the page: the prompt of the server, run with the words as they came.
pub async fn spec_new(base: String, name: String, description: String, display: String) -> Result<(String, String), String> {
    let v = crate::specs::spec_act(base, crate::specs::new_prompt_body(&name, &description)).await?;
    Ok((v.get("prompt").and_then(Value::as_str).unwrap_or_default().to_string(), display))
}

/// A reload that a failure turns into nothing (the page's `loadSpecs` catch: `code.specs = []`
/// would clear it — here the caller keeps what it had, which is what the page does not show
/// either way).
async fn fetch_specs_maybe(base: &str) -> Option<Vec<crate::specs::Spec>> {
    crate::specs::fetch_specs(base.to_string()).await.ok()
}

/// A GET of the agent's routes (`specAction`'s read side and the rest of the row-80 reads).
pub async fn call_get(base: &str, path: &str) -> Result<Value, String> {
    call(base, reqwest::Method::GET, path, None).await
}

/// A POST of the agent's routes with a JSON body.
pub async fn call_post(base: &str, path: &str, body: Value) -> Result<Value, String> {
    call(base, reqwest::Method::POST, path, Some(body)).await
}

pub async fn open_workspace(base: String, path: String) -> Result<Workspace, String> {
    parse_workspace(&call(&base, reqwest::Method::POST, "/api/studio/agent/workspace", Some(json!({ "path": path }))).await?)
}

/// The folder dialog of the server's computer: `None` when it was closed without a choice.
pub async fn pick_folder(base: String) -> Result<Option<String>, String> {
    let v = call(&base, reqwest::Method::POST, "/api/studio/pick-file", Some(json!({ "kind": "folder" }))).await?;
    Ok(v.get("path").and_then(Value::as_str).filter(|p| !p.is_empty()).map(str::to_string))
}

/// `GET /api/studio/agent/browser`: `None` when the server does not answer or has no such route.
pub async fn fetch_browser(base: String) -> Option<Value> {
    call(&base, reqwest::Method::GET, "/api/studio/agent/browser", None).await.ok()
}

/// One of the inputs of the panel (`post('/input', body)`): the answer's body whatever the status, an error only when nothing came.
async fn post_browser(base: String, path: &'static str, body: Value) -> Result<Value, String> {
    let resp = reqwest::Client::new().post(format!("{base}/api/studio/agent/browser{path}")).json(&body).send().await.map_err(|e| e.to_string())?;
    Ok(resp.json().await.unwrap_or_else(|_| json!({})))
}

/// One tick of the panel: the status now and then, and the frame.
async fn browser_tick(base: String, read_status: bool, since: String, enabled_hint: bool) -> (Option<Option<Value>>, Option<Value>) {
    let status = if read_status { Some(fetch_browser(base.clone()).await) } else { None };
    let enabled = match &status {
        Some(Some(v)) => crate::browser::Status::parse(v).enabled,
        Some(None) => false,
        None => enabled_hint,
    };
    let frame = if enabled { post_browser(base, "/frame", json!({ "since": since })).await.ok() } else { None };
    (status, frame)
}

/// `GET /api/studio/agent/computer`: `None` when the server does not answer or does not know it.
pub async fn fetch_computer(base: String) -> Option<CuStatus> {
    let v = call(&base, reqwest::Method::GET, "/api/studio/agent/computer", None).await.ok()?;
    let flag = |k: &str| v.get(k).is_some_and(|x| crate::jobs::truthy(Some(x)));
    let count = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
    Some(CuStatus { enabled: flag("enabled"), available: flag("available"), steps_used: count("steps_used"), max_steps: count("max_steps") })
}

/// `POST` of `computer/stop` and `computer/begin`: the answer does not matter.
async fn post_computer(base: String, what: &'static str) {
    let _ = call(&base, reqwest::Method::POST, &format!("/api/studio/agent/computer/{what}"), Some(json!({}))).await;
}

/// The next look at the status after `delay_ms` (`scheduleComputerPoll`), for the chain `epoch`.
fn computer_poll(base: String, epoch: u64, delay_ms: u64) -> Task<Message> {
    Task::perform(
        async move {
            if delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            fetch_computer(base).await
        },
        move |status| Message::Code(CodeMsg::ComputerStatus(epoch, status)),
    )
}

pub async fn fetch_tools(base: String, computer_on: bool) -> Result<(Vec<Tool>, Value), String> {
    let (mut tools, for_model) = call(&base, reqwest::Method::GET, "/api/studio/agent/tools", None).await.map(|v| parse_tools_with(&v, computer_on))?;
    let mut defs = for_model.as_array().cloned().unwrap_or_default();
    let (client, client_defs) = client_tools();
    tools.extend(client);
    defs.extend(client_defs);
    // no MCP client or no server connected: no MCP tools
    if let Ok(v) = call(&base, reqwest::Method::GET, "/api/studio/mcp/tools", None).await {
        let (mcp, mcp_defs) = parse_mcp_tools(&v);
        tools.extend(mcp);
        defs.extend(mcp_defs);
    }
    Ok((tools, Value::Array(defs)))
}

pub async fn fetch_context(base: String) -> Result<Context, String> {
    call(&base, reqwest::Method::POST, "/api/studio/agent/instructions", Some(json!({}))).await.map(|v| parse_context(&v))
}

/// A tool of an MCP server: its answer is text, or an error.
pub async fn call_mcp(base: String, server: String, tool: String, args: Value) -> Result<String, String> {
    let v = call(&base, reqwest::Method::POST, "/api/studio/mcp/call", Some(json!({"server": server, "name": tool, "arguments": args}))).await?;
    let content = match v.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    };
    if v.get("is_error").and_then(Value::as_bool) == Some(true) {
        Err(if content.is_empty() { "The tool failed.".into() } else { content })
    } else {
        Ok(content)
    }
}

pub async fn fetch_changes(base: String) -> Result<Vec<Change>, String> {
    call(&base, reqwest::Method::GET, "/api/studio/agent/changes", None).await.map(|v| parse_changes(&v))
}

/// The silent calls of one review round: the review tool first, then the diagnostics of every
/// file, one after the other — the page awaits them the same way and draws no card for them
/// (`selfReview`).
pub async fn review_tools(base: String, with_review: bool, with_diagnostics: bool, files: Vec<String>) -> ReviewAnswers {
    let mut answers = ReviewAnswers::default();
    if with_review {
        if let Ok(text) = run_tool(base.clone(), "review".into(), serde_json::json!({})).await {
            answers.review = Some(text);
        }
    }
    if with_diagnostics {
        for path in files {
            if let Ok(text) = run_tool(base.clone(), "diagnostics".into(), serde_json::json!({ "path": path.clone() })).await {
                answers.diagnostics.push((path, text));
            }
        }
    }
    answers
}

/// What one hook event answered (`fireHook` of code.js): every failure — a server without the
/// route among them — means nothing was hooked.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HookFire {
    pub blocked: bool,
    pub reason: String,
    pub context: String,
}

/// `fireHook(event, tool, args)`: ask the server's hooks for one event of the loop. The page
/// sends `{ event, tool, arguments }` and reads `blocked`, `reason` and `context` back.
pub async fn fire_hook(base: String, event: &str, tool: &str, arguments: Value) -> HookFire {
    let body = json!({ "event": event, "tool": tool, "arguments": arguments });
    match call(&base, reqwest::Method::POST, "/api/studio/agent/hook", Some(body)).await {
        Ok(v) => fire_answer(&v),
        Err(_) => HookFire::default(),
    }
}

/// What `fireHook` reads out of the answer (`blocked`, `reason`, `context`, each with the page's
/// fallbacks); anything the server does not say is the empty answer.
pub fn fire_answer(value: &Value) -> HookFire {
    HookFire {
        blocked: value.get("blocked").and_then(Value::as_bool).unwrap_or(false),
        reason: value.get("reason").and_then(Value::as_str).unwrap_or_default().to_string(),
        context: value.get("context").and_then(Value::as_str).unwrap_or_default().to_string(),
    }
}

/// `SHOT_MARK` of the page: how the message with a screenshot of the browser begins.
const BROWSER_SHOT_MARK: &str = "[Screenshot of the browser page,";

/// `dropOldImages(ctx.messages, 2)`: only the last two screenshot messages keep their pictures; an
/// older one says the picture was taken out.
fn drop_old_screenshots(wire: &mut [WireMessage]) {
    let carriers: Vec<usize> = wire.iter().enumerate().filter(|(_, m)| m.parts.is_some() && m.content.starts_with("[Screenshot from the computer tools")).map(|(i, _)| i).collect();
    let mut msgs: Vec<crate::computer::Msg> = carriers
        .iter()
        .map(|&i| crate::computer::Msg { role: wire[i].role.clone(), content: wire[i].content.clone(), images: vec![String::new()] })
        .collect();
    crate::computer::drop_old_images(&mut msgs, 2);
    for (&i, m) in carriers.iter().zip(&msgs) {
        if m.images.is_empty() {
            wire[i].content = m.content.clone();
            wire[i].parts = None;
        }
    }
}

/// A tool's answer with the picture it may carry (`r.image`: the screen after a computer action).
pub async fn run_tool_full(base: String, name: String, args: Value) -> (Result<String, String>, Option<Shot>) {
    match call(&base, reqwest::Method::POST, "/api/studio/agent/tool", Some(json!({ "name": name, "args": args }))).await {
        Ok(v) => {
            let shot = (v.get("ok").and_then(Value::as_bool) == Some(true)).then(|| v.get("image").and_then(Shot::from_reply)).flatten();
            (tool_result(&v), shot)
        }
        Err(e) => (Err(e), None),
    }
}

pub async fn run_tool(base: String, name: String, args: Value) -> Result<String, String> {
    tool_result(&call(&base, reqwest::Method::POST, "/api/studio/agent/tool", Some(json!({ "name": name, "args": args }))).await?)
}

/// `encodeURIComponent` of the page: a word in a path.
pub fn encode_query(query: &str) -> String {
    let mut out = String::new();
    let mut buf = [0u8; 4];
    for c in query.chars() {
        if c.is_ascii_alphanumeric() || "-_.!~*'()".contains(c) {
            out.push(c);
        } else {
            for b in c.encode_utf8(&mut buf).bytes() {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

/// The lines `read_file` writes in front of the text (`/^\s*\d+\t/gm` of the page).
pub fn strip_line_numbers(text: &str) -> String {
    static LINE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    LINE.get_or_init(|| regex::Regex::new(r"(?m)^\s*\d+\t").expect("a line number"))
        .replace_all(text, "")
        .into_owned()
}

/// The custom commands of an open folder (`loadCustomCommands` of `code.js`): the files under
/// `.fllm/commands` with their descriptions, then the commands an installed plugin brings — your
/// own command of the same name wins.
pub async fn load_custom_commands(base: String) -> Vec<crate::commands::CustomCommand> {
    let mut paths: Vec<String> = Vec::new();
    for pattern in [".fllm/commands/*.md", ".fllm/commands/**/*.md"] {
        let Ok(out) = run_tool(base.clone(), "glob".into(), serde_json::json!({ "pattern": pattern })).await else {
            continue;
        };
        paths.extend(
            out.split('\n')
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with("...") && *l != "no files match")
                .map(str::to_string),
        );
    }
    let mut list: Vec<crate::commands::CustomCommand> = crate::commands::command_names_from_paths(&paths)
        .into_iter()
        .map(|(name, path)| crate::commands::CustomCommand {
            name,
            description: String::new(),
            path: Some(path),
            plugin: None,
        })
        .collect();
    // the description is the front matter's, else the first line of the file (`limit: 4` reads
    // the head of it, the way the page does)
    for command in &mut list {
        let Some(path) = command.path.clone() else { continue };
        let Ok(out) = run_tool(base.clone(), "read_file".into(), serde_json::json!({ "path": path, "limit": 4 })).await
        else {
            continue;
        };
        let stripped = strip_line_numbers(&out);
        command.description = crate::commands::parse_command_file(&stripped).0;
    }
    if let Ok(v) = call(&base, reqwest::Method::GET, "/api/studio/plugins/commands", None).await {
        for row in v.get("commands").and_then(Value::as_array).into_iter().flatten() {
            let Some(name) = row.get("name").and_then(Value::as_str) else { continue };
            if list.iter().any(|c| c.name == name) {
                continue;
            }
            list.push(crate::commands::CustomCommand {
                name: name.to_string(),
                description: row.get("description").and_then(Value::as_str).unwrap_or_default().to_string(),
                path: None,
                plugin: row.get("plugin").and_then(Value::as_str).map(str::to_string),
            });
        }
    }
    list
}

/// `GET /api/studio/sessions/search` for `/search` (`case 'search'` of `code.js`).
pub async fn fetch_search(base: String, query: String) -> Result<Vec<crate::commands::SearchHit>, String> {
    let path = format!("/api/studio/sessions/search?q={}&limit=8", encode_query(&query));
    let v = call(&base, reqwest::Method::GET, &path, None).await?;
    // `data.hits || []`: anything that is no list counts as no hit
    let hits = v.get("hits").cloned().unwrap_or(Value::Null);
    Ok(serde_json::from_value(hits).unwrap_or_default())
}

pub async fn undo(base: String, body: Value) -> Result<String, String> {
    tool_result(&call(&base, reqwest::Method::POST, "/api/studio/agent/undo", Some(body)).await?)
}

pub async fn cancel(base: String) {
    let _ = call(&base, reqwest::Method::POST, "/api/studio/agent/cancel", Some(json!({}))).await;
}

// ---- the loop --------------------------------------------------------------------------------------------------------------------

/// What a tool call shows in one line: the file, the pattern or the command.
pub fn summary(name: &str, args: &Value) -> String {
    for key in ["path", "file_path", "command", "pattern", "query", "url"] {
        if let Some(s) = args.get(key).and_then(Value::as_str) {
            let first = s.lines().next().unwrap_or("");
            return if s.lines().count() > 1 || first.chars().count() > 90 { format!("{}…", first.chars().take(90).collect::<String>()) } else { first.to_string() };
        }
    }
    if args.as_object().is_none_or(|o| o.is_empty()) {
        return name.to_string();
    }
    String::new()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Line {
    Add,
    Del,
    Plain,
}

/// The lines a change would make, to show before it is allowed: removed and added lines of an edit, the lines of a new file, a command.
pub fn preview(name: &str, args: &Value) -> Vec<(Line, String)> {
    const MAX: usize = 24;
    let mut out: Vec<(Line, String)> = Vec::new();
    let push = |out: &mut Vec<(Line, String)>, kind: Line, text: &str| out.extend(text.lines().map(|l| (kind, l.to_string())));
    match name {
        "edit_file" => {
            push(&mut out, Line::Del, args.get("old_string").and_then(Value::as_str).unwrap_or_default());
            push(&mut out, Line::Add, args.get("new_string").and_then(Value::as_str).unwrap_or_default());
        }
        "multi_edit" => {
            for e in args.get("edits").and_then(Value::as_array).into_iter().flatten() {
                push(&mut out, Line::Del, e.get("old_string").and_then(Value::as_str).unwrap_or_default());
                push(&mut out, Line::Add, e.get("new_string").and_then(Value::as_str).unwrap_or_default());
            }
        }
        "write_file" => push(&mut out, Line::Add, args.get("content").and_then(Value::as_str).unwrap_or_default()),
        "patch" => {
            for l in args.get("patch").and_then(Value::as_str).unwrap_or_default().lines() {
                out.push((if l.starts_with('+') && !l.starts_with("+++") { Line::Add } else if l.starts_with('-') && !l.starts_with("---") { Line::Del } else { Line::Plain }, l.to_string()));
            }
        }
        "bash" => push(&mut out, Line::Plain, &format!("$ {}", args.get("command").and_then(Value::as_str).unwrap_or_default())),
        _ => {}
    }
    if out.len() > MAX {
        let more = out.len() - MAX;
        out.truncate(MAX);
        out.push((Line::Plain, format!("… {more} more lines")));
    }
    out
}

/// The list `todo_write` sends: `{todos: [{content, status}]}`.
pub fn parse_todos(args: &Value) -> Result<Vec<Todo>, String> {
    let list = args.get("todos").and_then(Value::as_array).ok_or("todos must be a list of {content, status}.")?;
    list.iter()
        .map(|t| {
            let content = t.get("content").and_then(Value::as_str).filter(|c| !c.trim().is_empty()).ok_or("every task needs a content.")?.to_string();
            let status = match t.get("status").and_then(Value::as_str) {
                Some("pending") => TodoStatus::Pending,
                Some("in_progress") => TodoStatus::InProgress,
                Some("completed") => TodoStatus::Completed,
                _ => return Err("status must be pending, in_progress or completed.".to_string()),
            };
            Ok(Todo { content, status })
        })
        .collect()
}

pub fn todo_text(todos: &[Todo]) -> String {
    if todos.is_empty() {
        return "The task list is empty.".into();
    }
    todos.iter().enumerate().map(|(i, t)| format!("{}. [{}] {}", i + 1, match t.status { TodoStatus::Pending => " ", TodoStatus::InProgress => "~", TodoStatus::Completed => "x" }, t.content)).collect::<Vec<_>>().join("\n")
}

impl CodeState {
    /// The session as it is kept on disk: a call that had not finished is kept as interrupted.
    pub fn saved(&self) -> Vec<Saved> {
        let mut out = self.saved_all();
        if out.len() > SAVED_ITEMS {
            out.drain(..out.len() - SAVED_ITEMS);
            // a call without the answer that announced it would confuse the model: start at a message of yours
            let first = out.iter().position(|i| matches!(i, Saved::User { .. })).unwrap_or(0);
            out.drain(..first);
        }
        out
    }

    /// Every item that is kept, uncut (a revert keeps all of it for *Undo the revert*).
    pub fn saved_all(&self) -> Vec<Saved> {
        self.items
            .iter()
            .filter_map(|i| match i {
                Item::User { text, sent } => Some(Saved::User {
                    text: text.clone(),
                    sent: sent.clone(),
                }),
                Item::Assistant { text, .. } => (!text.is_empty()).then(|| Saved::Assistant { text: text.clone() }),
                Item::Note(t) => Some(Saved::Note { text: t.clone() }),
                Item::ReviewFeedback(t) => Some(Saved::ReviewFeedback { text: t.clone() }),
                Item::StopHook(t) => Some(Saved::StopHook { text: t.clone() }),
                Item::Call(c) => Some(saved_call(c)),
            })
            .collect()
    }

    /// The index an item has in `saved()` before the cut to `SAVED_ITEMS` (an empty answer and a
    /// notice with buttons are not kept, so the later items move up).
    fn saved_index(&self, item: usize) -> usize {
        self.items[..item.min(self.items.len())]
            .iter()
            .filter(|i| !matches!(i, Item::Assistant { text, .. } if text.is_empty()))
            .count()
    }

    /// The session as a record of the list (`saveCodeSession` of the page): `None` while there is
    /// nothing to keep.
    pub fn record(&mut self) -> Option<crate::sessions::Record> {
        if self.items.is_empty() {
            return None;
        }
        if self.session_id.is_empty() {
            self.session_id = crate::sessions::new_id();
        }
        let mut items = self.saved();
        let all_len = self.items.iter().filter(|i| !matches!(i, Item::Assistant { text, .. } if text.is_empty())).count();
        // `saved()` cut the oldest items off: the marks move with them
        let dropped = all_len.saturating_sub(items.len());
        let marks: Vec<crate::sessions::Mark> = self
            .marks
            .iter()
            .filter_map(|m| {
                let at = self.saved_index(m.item);
                (at >= dropped).then(|| crate::sessions::Mark { item: at - dropped, ..m.clone() })
            })
            .collect();
        let title = if self.title.is_empty() { crate::sessions::heuristic_title(crate::sessions::first_user_text(&items).as_deref()) } else { self.title.clone() };
        let time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_millis() as f64);
        let todos = self.todos.iter().map(todo_saved).collect();
        items.shrink_to_fit();
        Some(crate::sessions::Record { id: self.session_id.clone(), title, title_locked: self.title_locked, time, items, todos, marks, compaction: self.compaction.clone().map(|mut c| {
            c.from_item = self.saved_index(c.from_item).saturating_sub(dropped);
            c
        }) })
    }

    /// The conversation and everything that belongs to it, empty (`resetSessionState`).
    pub fn reset_session(&mut self) {
        self.items.clear();
        self.todos.clear();
        self.steps = 0;
        self.history.clear();
        self.always_allow.clear();
        self.jobs.clear();
        self.session_id = String::new();
        self.title = String::new();
        self.title_locked = false;
        self.marks.clear();
        self.compaction = None;
        self.reverted = None;
        self.choices.clear();
        self.queue.clear();
    }

    /// A session of the list as the conversation (`applySession`).
    pub fn apply_record(&mut self, r: crate::sessions::Record) {
        self.reset_session();
        self.session_id = r.id;
        self.title = r.title;
        self.title_locked = r.title_locked;
        self.marks = r.marks;
        self.compaction = r.compaction;
        self.restore(r.items);
        self.todos = r.todos.iter().map(todo_loaded).collect();
    }

    /// The sessions of a folder are read (the file is the list), and the newest one is the
    /// conversation again, as the page restores it when a folder opens.
    pub fn open_sessions(&mut self, file: &Path, root: &str) {
        self.sessions = Sessions::load(file).of(root).to_vec();
        if let Some(first) = self.sessions.first().cloned() {
            self.apply_record(first);
        }
    }

    /// Takes a kept session back as the conversation.
    pub fn restore(&mut self, saved: Vec<Saved>) {
        self.steps = 0;
        self.history.clear();
        self.items = saved
            .into_iter()
            .map(|s| match s {
                Saved::User { text, sent } => Item::User { text, sent },
                Saved::Assistant { text } => {
                    let mut md = markdown::Content::new();
                    md.push_str(&text);
                    Item::Assistant { text, md: Box::new(md) }
                }
                call @ Saved::Call { .. } => Item::Call(restored_call(call)),
                Saved::Note { text } => Item::Note(text),
                Saved::ReviewFeedback { text } => Item::ReviewFeedback(text),
                Saved::StopHook { text } => Item::StopHook(text),
            })
            .collect();
    }

    fn tool(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// The calls of the last round: those after the last assistant item.
    fn last_round(&self) -> impl Iterator<Item = &Call> {
        let start = self.items.iter().rposition(|i| matches!(i, Item::Assistant { .. })).map_or(0, |i| i + 1);
        self.items[start..].iter().filter_map(|i| if let Item::Call(c) = i { Some(c) } else { None })
    }

    /// The question of the agent that waits for your answer.
    pub fn pending_ask(&self) -> Option<&Call> {
        self.items.iter().rev().find_map(|i| match i {
            Item::Call(c) if c.kind == ToolKind::Ask && c.state == CallState::Pending => Some(c),
            _ => None,
        })
    }

    /// A call of the conversation or of a sub-agent's: the cards of a `task` hold the calls of
    /// its sub-agent, and they are answered, allowed and denied like every other.
    fn call_mut(&mut self, id: &str) -> Option<&mut Call> {
        for item in self.items.iter_mut() {
            let Item::Call(c) = item else { continue };
            if c.id == id {
                return Some(c);
            }
            if let Some(kid) = c.kid.as_mut() {
                for k in kid.items.iter_mut() {
                    if let KidItem::Call(cc) = k {
                        if cc.id == id {
                            return Some(cc);
                        }
                    }
                }
            }
        }
        None
    }

    /// Every call that is on a card: the conversation's and its sub-agents'.
    fn each_call_mut(&mut self, mut f: impl FnMut(&mut Call)) {
        for item in self.items.iter_mut() {
            let Item::Call(c) = item else { continue };
            if let Some(kid) = c.kid.as_mut() {
                for k in kid.items.iter_mut() {
                    if let KidItem::Call(cc) = k {
                        f(cc);
                    }
                }
            }
            f(c);
        }
    }

    /// The ids of the calls that wait for you and that `pred` lets through (the conversation's
    /// and its sub-agents').
    fn waiting_ids(&self, pred: impl Fn(&Call) -> bool) -> Vec<String> {
        let mut out = Vec::new();
        for item in &self.items {
            let Item::Call(c) = item else { continue };
            if let Some(kid) = &c.kid {
                for k in &kid.items {
                    if let KidItem::Call(cc) = k {
                        if cc.state == CallState::Pending && pred(cc) {
                            out.push(cc.id.clone());
                        }
                    }
                }
            }
            if c.state == CallState::Pending && pred(c) {
                out.push(c.id.clone());
            }
        }
        out
    }

    /// The state of a call by its id, wherever it is.
    fn call_state(&self, id: &str) -> Option<&CallState> {
        for item in &self.items {
            let Item::Call(c) = item else { continue };
            if c.id == id {
                return Some(&c.state);
            }
            if let Some(kid) = &c.kid {
                for k in &kid.items {
                    if let KidItem::Call(cc) = k {
                        if cc.id == id {
                            return Some(&cc.state);
                        }
                    }
                }
            }
        }
        None
    }

    /// The agent's instructions (`codeSystemPrompt` of the page): the mode's prompt, the folder,
    /// the style, the project's own instructions and what the Settings page adds. A sub-agent is
    /// told the same words (`parent.system`).
    pub fn system_prompt(&self, extra: Option<&str>) -> String {
        let mut system = if self.mode == Mode::Plan { PLAN_PROMPT } else { CODE_PROMPT }.to_string();
        if let Some(w) = &self.workspace {
            system.push_str(&format!("\n\nThe project folder is {} (tool paths are relative to it).", w.root));
        }
        // the style of `/personality`, where the page puts it: after the family notes, before
        // the instructions, the memory and the profile (`codeSystemPrompt` of codeprompts.js)
        let style = crate::commands::personality_note(&self.personality);
        if !style.is_empty() {
            system.push_str("\n\n");
            system.push_str(style);
        }
        if self.rich_answers {
            system.push_str("\n\n");
            system.push_str(crate::rich::INSTRUCTIONS);
        }
        let context = self.context.prompt();
        if !context.is_empty() {
            system.push_str("\n\n");
            system.push_str(&context);
        }
        if let Some(x) = extra.filter(|x| !x.trim().is_empty()) {
            system.push_str("\n\n");
            system.push_str(x.trim());
        }
        system
    }

    /// The messages of the next request: the agent's instructions, then the session with every call answered.
    pub fn wire(&self, extra: Option<&str>) -> Vec<WireMessage> {
        self.wire_of(&self.items, extra)
    }

    /// The messages of the conversation up to `items` (the part before a message of yours, for a
    /// fork or a revert). After `/compact` a summary stands for the first ones.
    pub fn wire_of(&self, items: &[Item], extra: Option<&str>) -> Vec<WireMessage> {
        let system = self.system_prompt(extra);
        let mut wire = vec![WireMessage::new("system", system)];
        let mut i = 0;
        while i < items.len() {
            match &items[i] {
                Item::ReviewFeedback(text) | Item::StopHook(text) => {
                    // sent as it stands, and never drawn (the view skips these variants)
                    wire.push(WireMessage::new("user", text.clone()));
                }
                Item::User { text, sent } => {
                    // what the model is sent: the prompt of a command, with the output of the
                    // commands the user ran in front of it (`takePending() + notes + userText`)
                    let content = sent.clone().unwrap_or_else(|| text.clone());
                    wire.push(WireMessage::new("user", content));
                }
                Item::Note(_) => {}
                Item::Assistant { text, .. } => {
                    // the calls that follow this answer belong to it
                    let calls: Vec<&Call> = items[i + 1..].iter().take_while(|n| matches!(n, Item::Call(_))).filter_map(|n| if let Item::Call(c) = n { Some(c) } else { None }).collect();
                    let mut m = WireMessage::new("assistant", text.clone());
                    if !calls.is_empty() {
                        m.tool_calls = Some(Value::Array(calls.iter().map(|c| json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.args}})).collect()));
                    }
                    if !text.is_empty() || !calls.is_empty() {
                        wire.push(m);
                    }
                    for c in &calls {
                        let output = if let Some(sent) = &c.sent {
                            // the automatic check's reply: what the model gets, with the
                            // problems between the two markers (`reply = …` of code.js)
                            sent.clone()
                        } else { match &c.state {
                            CallState::Done { ok: true, output } => output.clone(),
                            CallState::Done { ok: false, output } => format!("Error: {output}"),
                            _ => "The call did not finish.".to_string(),
                        } };
                        let mut t = WireMessage::new("tool", output);
                        t.tool_call_id = Some(c.id.clone());
                        wire.push(t);
                    }
                    // `computerAttachImages`: the screenshots of this step go to the model as one message with pictures
                    let shots: Vec<(&Shot, String)> = calls
                        .iter()
                        .filter(|c| c.permission == Permission::Computer)
                        .filter_map(|c| {
                            let shot = c.cu.as_ref()?.image.as_ref()?;
                            let CallState::Done { ok: true, output } = &c.state else { return None };
                            Some((shot, output.clone()))
                        })
                        .collect();
                    if !shots.is_empty() {
                        let content = shots
                            .iter()
                            .map(|(s, text)| format!("[Screenshot from the computer tools: {}x{} px. {}]", crate::loopguard::js_number(s.width), crate::loopguard::js_number(s.height), crate::computer::slice_units(text, 300)))
                            .collect::<Vec<_>>()
                            .join("\n");
                        let mut parts = vec![json!({ "type": "text", "text": content })];
                        parts.extend(shots.iter().map(|(s, _)| json!({ "type": "image_url", "image_url": { "url": s.url } })));
                        wire.push(WireMessage::with_parts("user", content, parts));
                    }
                    // `attachShots`: a screenshot of the browser tool goes to a model that can see as a user
                    // message with the picture (a tool message cannot carry one); only the newest stays
                    if self.vision {
                        if let Some(shot) = calls.iter().rev().filter(|c| c.permission != Permission::Computer).find_map(|c| c.cu.as_ref()?.image.as_ref()) {
                            let text = format!("{BROWSER_SHOT_MARK} taken by browser_screenshot]");
                            let parts = vec![json!({ "type": "text", "text": text }), json!({ "type": "image_url", "image_url": { "url": shot.url } })];
                            wire.push(WireMessage::with_parts("user", text, parts));
                        }
                    }
                    i += calls.len();
                }
                Item::Call(_) => {}
            }
            i += 1;
        }
        // `/compact`: a summary stands for the first messages
        if let Some(c) = &self.compaction {
            let covered = c.covered.min(wire.len() - 1);
            let prefix: Vec<WireMessage> = c.prefix.iter().map(|(role, content)| WireMessage::new(role, content.clone())).collect();
            wire.splice(1..1 + covered, prefix);
        }
        drop_old_screenshots(&mut wire);
        // only the newest picture of the browser stays in the conversation
        let browser_shots: Vec<usize> = wire.iter().enumerate().filter(|(_, m)| m.parts.is_some() && m.content.starts_with(BROWSER_SHOT_MARK)).map(|(i, _)| i).collect();
        for &i in browser_shots.iter().rev().skip(1) {
            wire[i].content = "[an older screenshot of the browser was removed to save space]".into();
            wire[i].parts = None;
        }
        wire
    }

    /// What a command reads of the page's `code` (`dispatch`).
    fn cmd_state(&self, env: &Env) -> crate::commands::CmdState {
        crate::commands::CmdState {
            running: self.running,
            perm: Self::perm_of(&self.mode),
            last_build_perm: Self::perm_of(&self.build_mode),
            auto_approve: self.auto_approve,
            auto_review: self.auto_review,
            specs: self.specs.clone(),
            turbo: env.turbo,
            sandbox: env.sandbox.map(str::to_string),
            sandbox_available: env.sandbox_available,
            personality: self.personality.clone(),
            rules: self.rules.clone(),
            custom: self.custom.clone(),
            usage: env.usage.map(str::to_string),
            models: env.models.to_vec(),
        }
    }

    /// The word the page uses for a mode (`code.perm`): its own table has three, this window has
    /// a fourth one (*Everything*), which the commands pass through as `auto`.
    fn perm_of(mode: &Mode) -> String {
        match mode {
            Mode::Plan => "plan".into(),
            Mode::Ask => "ask".into(),
            Mode::AutoEdit => "edit".into(),
            Mode::Auto => "auto".into(),
        }
    }

    /// The mode a command's word stands for (`setPerm` of the page).
    fn mode_of(perm: &str) -> Mode {
        match perm {
            "plan" => Mode::Plan,
            "edit" => Mode::AutoEdit,
            "auto" => Mode::Auto,
            _ => Mode::Ask,
        }
    }

    /// The run of `/init` and of a custom command: the view shows the words that were typed, the
    /// model gets the prompt, and what `/status` and `/diff` shared rides in front of it
    /// (`agentRun(prompt, display)` and `takePending()` of `code.js`).
    fn run_prompt(&mut self, prompt: &str, display: &str, env: &Env) -> Task<Message> {
        if self.workspace.is_none() || env.model.is_none() || self.running {
            return Task::none();
        }
        let mut sent_full = format!("{}{prompt}", std::mem::take(&mut self.br_notes));
        if !self.pending.is_empty() {
            let shared = std::mem::take(&mut self.pending);
            sent_full = format!("{}\n\n{sent_full}", shared.join("\n\n"));
        }
        let sent = (sent_full != display).then_some(sent_full);
        self.error = None;
        self.steps = 0;
        self.compactions = 0;
        self.choices.clear();
        self.history.clear();
        // the run begins where the conversation ends now (the page's `turn.msgIndex`)
        self.learn_from = Some(self.wire(None).len().saturating_sub(1));
        // and the moment is kept: a fork or a revert goes back to before this message
        self.marks.push(crate::sessions::Mark {
            item: self.items.len(),
            changes: self.changes.iter().map(|c| c.id).max().unwrap_or(0),
            todos: self.todos.iter().map(todo_saved).collect(),
        });
        self.items.push(Item::User {
            text: display.to_string(),
            sent,
        });
        self.running = true;
        self.start_stream(env)
    }

    /// `/status` and `/diff`: a card of the command the user ran itself, its output shared with
    /// the model with the next message (`userRun` of `code.js`).
    fn user_run(&mut self, command: String, share: bool, env: &Env) -> Task<Message> {
        if self.workspace.is_none() {
            return Task::none();
        }
        let id = format!("call_{}_u{}", self.steps, self.items.len());
        let call = Call {
            kid: None,
            cu: None,
            open: None,
            expanded: false,
            id: id.clone(),
            name: "bash".into(),
            args: serde_json::json!({ "command": command }).to_string(),
            permission: Permission::Command,
            kind: ToolKind::Server,
            state: CallState::Running,
            auto_note: None,
            check: None,
            sent: None,
        };
        self.items.push(Item::Call(call));
        let base = env.base.to_string();
        Task::perform(
            async move {
                let args = serde_json::json!({ "command": command });
                let result = run_tool(base, "bash".into(), args).await;
                (id, result, share)
            },
            |(id, result, share)| Message::Code(CodeMsg::UserRunDone(id, result, share)),
        )
    }

    /// `userTool(name, args)` of the page: a card for a tool the user ran (`/check`, `/review`,
    /// `/security-review`). The page asks for no workspace and shares nothing of the output.
    fn user_tool(&mut self, name: String, args: Value, env: &Env) -> Task<Message> {
        let id = format!("call_{}_u{}", self.steps, self.items.len());
        let tool = self.tool(&name).cloned();
        let permission = tool.as_ref().map_or(Permission::Other, |t| t.permission);
        let kind = tool.as_ref().map_or(ToolKind::Server, |t| t.kind.clone());
        self.items.push(Item::Call(Call {
            id: id.clone(),
            name: name.clone(),
            args: args.to_string(),
            permission,
            kind,
            state: CallState::Running,
            auto_note: None,
            check: None,
            sent: None,
            kid: None,
            cu: None,
            open: None,
            expanded: false,
        }));
        let base = env.base.to_string();
        Task::perform(
            async move {
                let result = run_tool(base, name, args).await;
                (id, result, false)
            },
            |(id, result, share)| Message::Code(CodeMsg::UserRunDone(id, result, share)),
        )
    }

    /// `/remember`: `saveMemoryEntry` of the page, then the instructions are read again so the
    /// fact is part of the next prompt (`loadInstructions` there).
    fn remember(&mut self, scope: &'static str, entry: String, notice: String, env: &Env) -> Task<Message> {
        let base = env.base.to_string();
        Task::perform(
            async move {
                let body = serde_json::json!({ "scope": scope, "op": "append", "text": entry });
                match call(&base, reqwest::Method::POST, "/api/studio/agent/memory", Some(body)).await {
                    Ok(_) => Ok(()),
                    // the message of the server, or the page's own words
                    Err(e) => Err(if e.starts_with("the server answered") || e.starts_with("no server") {
                        "The memory could not be saved.".to_string()
                    } else {
                        e
                    }),
                }
            },
            move |r| Message::Code(CodeMsg::MemorySaved(r, notice)),
        )
    }

    /// `/search`: the stored sessions of this window (`sessions/search` of the page).
    fn search_words(&mut self, query: String, env: &Env) -> Task<Message> {
        let base = env.base.to_string();
        Task::perform(
            async move {
                match fetch_search(base, query.clone()).await {
                    Ok(hits) => Ok((hits, query)),
                    Err(e) => Err((e, query)),
                }
            },
            |r| match r {
                Ok((hits, query)) => Message::Code(CodeMsg::SearchDone(Ok(hits), query)),
                Err((e, query)) => Message::Code(CodeMsg::SearchDone(Err(e), query)),
            },
        )
    }

    /// A custom command: read its file (or ask the plugin for its body), then run what it holds
    /// (`readCustomCommand` + `expandCommand` of the page happen in the answer below).
    fn custom_command(&mut self, name: String, arg_text: String, source: String, env: &Env) -> Task<Message> {
        let Some(cmd) = self.custom.iter().find(|c| c.name == name).cloned() else {
            return Task::none();
        };
        let base = env.base.to_string();
        let work: std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send>> = match (cmd.path, cmd.plugin) {
            (_, Some(plugin)) => {
                let name = cmd.name;
                Box::pin(async move {
                    let path = format!(
                        "/api/studio/plugins/command?plugin={}&name={}",
                        encode_query(&plugin),
                        encode_query(&name)
                    );
                    let v = call(&base, reqwest::Method::GET, &path, None).await?;
                    Ok::<String, String>(v.get("body").and_then(Value::as_str).unwrap_or_default().to_string())
                })
            }
            (Some(path), None) => Box::pin(async move {
                let out = run_tool(base, "read_file".into(), serde_json::json!({ "path": path, "limit": 2000 }))
                    .await
                    .map_err(|e| {
                        if e.trim().is_empty() { "The command file could not be read.".to_string() } else { e }
                    })?;
                let stripped = strip_line_numbers(&out);
                Ok(crate::commands::parse_command_file(&stripped).1)
            }),
            (None, None) => return Task::none(),
        };
        Task::perform(work, |r| Message::Code(CodeMsg::CustomBody(r, arg_text, source)))
    }

    /// Everything a command asked for, in the order the page performs it.
    fn run_steps(&mut self, steps: Vec<crate::commands::Step>, env: &Env) -> Task<Message> {
        use crate::commands::Step;
        let mut tasks: Vec<Task<Message>> = Vec::new();
        for step in steps {
            match step {
                Step::Notice(text) => self.items.push(Item::Note(text)),
                Step::Toast(text) => self.pending_toast = Some(text),
                step @ (Step::Sessions | Step::Fork | Step::Revert | Step::Rename(_) | Step::Export | Step::Rewind | Step::Compact(_) | Step::PickModel(_) | Step::Memory | Step::Retry | Step::Skill) => {
                    tasks.push(self.session_step(step, env))
                }
                Step::NewSession => {
                    // `newCodeSession` of the page: stop, save, reset
                    if self.running {
                        tasks.push(self.stop_now(env));
                    }
                    self.persist(env.session_file);
                    self.reset_session();
                }
                Step::Undo(steps) => {
                    // the page sends `{ steps }` above one, and one is the plain call
                    let n = if steps > 1.0 { Some(steps as u64) } else { None };
                    tasks.push(self.update(CodeMsg::Undo(n), env));
                }
                // every `agentRun` of the page carries its hooks and the self-review (they sit in
                // `agentRunNow`, not in `submitText`), so a run of a command starts as a turn does
                Step::AgentRun(prompt, display) => tasks.push(self.begin_run(prompt, display, env)),
                Step::SetPerm(perm, notice) => {
                    tasks.push(self.update(CodeMsg::SetMode(Self::mode_of(perm)), env));
                    self.items.push(Item::Note(notice));
                }
                Step::OpenFolder(Some(path)) => tasks.push(self.update(CodeMsg::OpenPath(path), env)),
                Step::OpenFolder(None) => tasks.push(self.update(CodeMsg::Pick, env)),
                Step::Stop => tasks.push(self.update(CodeMsg::Stop, env)),
                Step::UserRun(command, share) => tasks.push(self.user_run(command, share, env)),
                Step::Remember { scope, entry, notice } => tasks.push(self.remember(scope, entry, notice, env)),
                Step::SaveRules { allow, deny, notice } => {
                    self.rules = crate::commands::Rules { allow, deny };
                    self.pending_prefs.get_or_insert_with(Default::default).rules = Some(self.rules.clone());
                    self.items.push(Item::Note(notice));
                }
                // the confirm and the post belong to the window (the same dialog and the same
                // route the switches of Settings > Runtime use); the flags are read in the very
                // message that ran the command, so the dialog is up at once
                Step::Turbo(next) => self.pending_turbo = Some(next),
                Step::SetSandbox(mode) => self.pending_sandbox = Some(mode),
                Step::AutoApprove(want, note) => {
                    let changed = want.is_some_and(|on| on != self.auto_approve);
                    if let Some(on) = want {
                        self.auto_approve = on;
                    }
                    self.items.push(Item::Note(note));
                    if changed {
                        self.pending_auto_approve = Some(self.auto_approve);
                    }
                }
                Step::AutoReview(want, note) => {
                    let changed = want.is_some_and(|on| on != self.auto_review);
                    if let Some(on) = want {
                        self.auto_review = on;
                    }
                    self.items.push(Item::Note(note));
                    if changed {
                        self.pending_auto_review = Some(self.auto_review);
                    }
                }
                Step::Learn(arg) => tasks.push(self.learn_command(&arg, env)),
                Step::Thumb(value) => tasks.push(self.update(CodeMsg::LearnThumb(value), env)),
                Step::UserTool { name, args } => tasks.push(self.user_tool(name, args, env)),
                Step::Worktree { action, name } => {
                    let base = env.base.to_string();
                    tasks.push(Task::perform(
                        worktree_fetch(base, action, name.clone()),
                        move |result| Message::Code(CodeMsg::WorktreeDone { action, name, batch: None, result }),
                    ));
                }
                Step::SpecList => tasks.push(Task::perform(
                    crate::specs::fetch_specs(env.base.to_string()),
                    |r| Message::Code(CodeMsg::SpecListDone(r.unwrap_or_default())),
                )),
                Step::SpecRun { name, asked } => {
                    let base = env.base.to_string();
                    let current = self.specs.clone();
                    tasks.push(Task::perform(spec_run(base, current, name, asked), |r| match r {
                        Ok((prompt, display)) => Message::Code(CodeMsg::SpecPrompt { prompt, display }),
                        Err(message) => Message::Code(CodeMsg::SpecFailed { message, toast: false }),
                    }));
                }
                Step::SpecNew { name, description, display } => {
                    let base = env.base.to_string();
                    tasks.push(Task::perform(spec_new(base, name, description, display), |r| match r {
                        Ok((prompt, display)) => Message::Code(CodeMsg::SpecPrompt { prompt, display }),
                        Err(message) => Message::Code(CodeMsg::SpecFailed { message, toast: false }),
                    }));
                }
                Step::Batch { name, task } => {
                    let base = env.base.to_string();
                    tasks.push(Task::perform(
                        worktree_fetch(base, "new", Some(name.clone())),
                        move |result| Message::Code(CodeMsg::WorktreeDone { action: "new", name: Some(name), batch: Some(task), result }),
                    ));
                }
                Step::Search(query) => tasks.push(self.search_words(query, env)),
                Step::Insights => {
                    let input = vec![Self::insight_messages(&self.wire(None))];
                    let text = crate::commands::insights_notice(&crate::commands::session_insights(&input));
                    self.items.push(Item::Note(text));
                }
                Step::SetPersonality(key, note) => {
                    self.personality = key.to_string();
                    self.pending_prefs.get_or_insert_with(Default::default).personality = Some(key.to_string());
                    self.items.push(Item::Note(note));
                }
                Step::CustomCommand { name, arg_text, source } => {
                    tasks.push(self.custom_command(name, arg_text, source, env))
                }
            }
        }
        Task::batch(tasks)
    }

    /// What `sessionInsights` counts over: the wire messages of this folder's conversation.
    fn insight_messages(wire: &[WireMessage]) -> Vec<crate::commands::InsightMessage> {
        wire.iter()
            .map(|m| crate::commands::InsightMessage {
                role: m.role.clone(),
                tool_calls: m
                    .tool_calls
                    .as_ref()
                    .map(|calls| {
                        calls
                            .as_array()
                            .map(|list| {
                                list.iter()
                                    .filter_map(|c| c.pointer("/function/name").and_then(Value::as_str).map(str::to_string))
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .unwrap_or_default(),
            })
            .collect()
    }

    /// Stop what runs, as `CodeMsg::Stop` does — the command `/stop` and `newCodeSession` both.
    fn stop_now(&mut self, env: &Env) -> Task<Message> {
        self.handle = None;
        self.running = false;
        self.beginning = false;
        // what was queued is yours again (the page puts it back in the box)
        if !self.queue.is_empty() {
            let back = self.queue.join("\n\n");
            self.queue.clear();
            self.editor = Content::with_text(&back);
            self.items.push(Item::Note("Stopped. The messages you had queued are back in the input.".into()));
        }
        self.acc.clear();
        // the sub-agents stop with the conversation: their streams end, their calls and the
        // `task` calls that started them are closed
        self.kids.clear();
        self.each_call_mut(|c| {
            if matches!(c.state, CallState::Pending | CallState::Running) {
                c.state = CallState::Done { ok: false, output: "Stopped.".into() };
            }
            if let Some(kid) = c.kid.as_mut() {
                kid.progress.clear();
            }
        });
        let base = env.base.to_string();
        Task::perform(
            async move { cancel(base.clone()).await; fetch_changes(base).await },
            |r| Message::Code(CodeMsg::ChangesLoaded(r)),
        )
    }

    /// The session of this folder onto disk (the write of `app.rs` after a run or a new session,
    /// here so a command can save before it clears the conversation): the record of this
    /// session goes to the top of the list.
    pub fn persist(&mut self, file: &std::path::Path) {
        let Some(root) = self.workspace.as_ref().map(|w| w.root.clone()) else { return };
        let mut all = Sessions::load(file);
        match self.record() {
            Some(rec) => all.put(&root, rec),
            None => {
                if !self.session_id.is_empty() {
                    all.remove(&root, &self.session_id);
                }
            }
        }
        if let Err(e) = all.save(file) {
            eprintln!("fllm-app: the code session could not be saved: {e}");
        }
        self.sessions = all.of(&root).to_vec();
    }

    fn start_stream(&mut self, env: &Env) -> Task<Message> {
        // the conversation fills the context: the older part becomes a summary first (two times at most
        // in one run, `compactions < 2` of the page's loop)
        if self.compactions < 2 && env.model.is_some() && crate::sessions::needs_compaction(self.code_estimate(env.extra), env.ctx_max, crate::sessions::COMPACT_RATIO) {
            self.compactions += 1;
            if let Some(task) = self.begin_compaction(String::new(), env, true) {
                return task;
            }
        }
        self.start_stream_now(env)
    }

    fn start_stream_now(&mut self, env: &Env) -> Task<Message> {
        let Some(model) = env.model else {
            self.running = false;
            return Task::none();
        };
        if self.steps >= MAX_STEPS {
            self.items.push(Item::Note(format!("Stopped after {MAX_STEPS} steps. Send a message to go on.")));
            self.running = false;
            return Task::none();
        }
        self.steps += 1;
        self.acc.clear();
        let wire = self.wire(env.extra);
        self.items.push(Item::Assistant { text: String::new(), md: Box::new(markdown::Content::new()) });
        self.running = true;
        let events = api::chat_stream_with_tools(env.base.to_string(), model.to_string(), wire, env.params.clone(), Some(self.tools_for_model()));
        let (task, handle) = Task::stream(events).map(|ev| Message::Code(CodeMsg::Stream(ev))).abortable();
        self.handle = Some(handle.abort_on_drop());
        task
    }

    /// The tools the model is offered: in plan mode only those that read.
    pub fn tools_for_model(&self) -> Value {
        // the task tool names the custom agents of the folder (`toolListFor`)
        let listed = |all: Vec<Value>| -> Value {
            Value::Array(
                all.into_iter()
                    .map(|d| if d.pointer("/function/name").and_then(Value::as_str) == Some("task") { crate::subagents::task_with_agents(&d, &self.agents) } else { d })
                    .collect(),
            )
        };
        if !self.tools_json.is_array() {
            return self.tools_json.clone();
        }
        if self.mode != Mode::Plan {
            return listed(self.tools_json.as_array().cloned().unwrap_or_default());
        }
        let readable = |name: &str| self.tool(name).is_some_and(|t| self.mode.allows(t.permission));
        listed(self.tools_json.as_array().into_iter().flatten().filter(|t| t.pointer("/function/name").and_then(Value::as_str).is_some_and(readable)).cloned().collect())
    }

    fn run(&mut self, base: &str, id: &str) -> Task<Message> {
        let Some(c) = self.call_mut(id) else { return Task::none() };
        if c.kind == ToolKind::Ask {
            // only your answer finishes this call
            return Task::none();
        }
        let parsed = serde_json::from_str::<Value>(if c.args.trim().is_empty() { "{}" } else { &c.args });
        let args = match parsed {
            Ok(args) => args,
            Err(e) => {
                c.state = CallState::Done { ok: false, output: format!("The arguments are not valid JSON ({e}). Call the tool again with a JSON object.") };
                return Task::none();
            }
        };
        let kind = c.kind.clone();
        let permission = c.permission;
        let (base, name, id) = (base.to_string(), c.name.clone(), id.to_string());
        match kind {
            ToolKind::Todo { write } => {
                let output = if write {
                    match parse_todos(&args) {
                        Ok(list) => {
                            self.todos = list;
                            format!("The task list was updated ({} tasks).", self.todos.len())
                        }
                        Err(e) => {
                            if let Some(c) = self.call_mut(&id) {
                                c.state = CallState::Done { ok: false, output: e };
                            }
                            return Task::none();
                        }
                    }
                } else {
                    todo_text(&self.todos)
                };
                if let Some(c) = self.call_mut(&id) {
                    c.state = CallState::Done { ok: true, output };
                }
                Task::none()
            }
            ToolKind::Mcp { server, tool } => {
                if let Some(c) = self.call_mut(&id) {
                    c.state = CallState::Running;
                }
                Task::perform(call_mcp(base, server, tool, args), move |r| Message::Code(CodeMsg::ToolDone(id.clone(), r)))
            }
            // a task starts in `finish_round`, where the window's model and settings are at hand
            ToolKind::Task => Task::none(),
            ToolKind::CreateAgent => {
                if let Some(c) = self.call_mut(&id) {
                    c.state = CallState::Running;
                }
                let root = self.workspace.as_ref().map(|w| w.root.clone()).unwrap_or_default();
                Task::perform(create_agent_file(base, root, args), move |r| Message::Code(CodeMsg::ToolDone(id.clone(), r)))
            }
            ToolKind::Server | ToolKind::Ask => {
                if let Some(c) = self.call_mut(&id) {
                    c.state = CallState::Running;
                }
                if permission == Permission::Computer {
                    self.computer_activity(&id, false);
                }
                Task::perform(run_tool_full(base, name, args), move |(r, shot)| Message::Code(CodeMsg::ToolReply(id.clone(), r, shot)))
            }
        }
    }

    /// The start of a turn of the conversation (`submitText` of the page): where the run begins
    /// is marked, the tail is reset to the review, and when the window knows of a
    /// `user_prompt_submit` hook the turn waits for its answer — `context` arrives in the
    /// prompt, the words of the box stay as they were typed.
    fn begin_run(&mut self, prompt: String, display: String, env: &Env) -> Task<Message> {
        if self.workspace.is_none() || env.model.is_none() || self.running || self.beginning {
            return Task::none();
        }
        self.change_mark = self.changes.iter().map(|c| c.id).max().unwrap_or(0);
        self.main_run = true;
        self.tail = TailPhase::Review;
        // `computerBegin()`: the step count and the remembered screenshot start again, and the
        // tool list follows the setting (the window does not wait for it)
        self.computer.last_shot = None;
        self.computer.busy.clear();
        // the comments the user left on the browser page go with this message, once
        self.br_notes = self.browser.take_comments();
        if env.hooks.submit {
            Task::perform(
                fire_hook(env.base.to_string(), "user_prompt_submit", "", json!({ "prompt": prompt.clone() })),
                move |answer| Message::Code(CodeMsg::SubmitHookDone { prompt, display, answer }),
            )
        } else {
            self.run_after_begin(prompt, display, env)
        }
    }

    /// `await computerBegin()` of the page, which comes after the submit hook and before the first
    /// request: when computer use is on, the server starts counting its actions, and only when it
    /// has answered does the run go on (a server that does not answer counts anyway).
    fn run_after_begin(&mut self, prompt: String, display: String, env: &Env) -> Task<Message> {
        if !self.computer_on() {
            return self.run_prompt(&prompt, &display, env);
        }
        // nothing else starts a run in the meantime
        self.beginning = true;
        let base = env.base.to_string();
        Task::perform(post_computer(base, "begin"), move |()| Message::Code(CodeMsg::ComputerBegun { prompt: prompt.clone(), display: display.clone() }))
    }

    /// The state one call starts in, and the words under its card when it ran without a
    /// question. The chain of `finish_round` kept whole: an unknown tool fails, a question waits
    /// in every mode, the mode decides what it allows, plan mode refuses the rest — and behind
    /// that, the switch of `/autoapprove` asks the harmless table instead of the user
    /// (`if (decision === 'ask' && code.autoApprove)` of the page).
    fn starting_state(
        &self,
        tool_known: bool,
        kind: ToolKind,
        permission: Permission,
        name: &str,
        args: &str,
        turbo: bool,
    ) -> (CallState, Option<String>) {
        if !tool_known {
            return (CallState::Done {
                ok: false,
                output: format!("There is no tool called {name}."),
            }, None);
        }
        if kind == ToolKind::Ask {
            // a question is for you in every mode
            return (CallState::Pending, None);
        }
        if kind == ToolKind::Task {
            // `runToolCall` hands a task to `runSubAgent` before it asks anything; what the
            // sub-agent may then do is asked call by call
            return (CallState::Running, None);
        }
        // the chain of `runToolCall` (code.js 1955-2010): plan first, then the rules, the mode
        // and the remembered keys of `authorize`, and the harmless table gets the last word on
        // what is left as a question
        let kind_mcp = match &kind {
            ToolKind::Mcp { server, tool } => Some((server.as_str(), tool.as_str())),
            _ => None,
        };
        let mut parsed: Value = serde_json::from_str(args).unwrap_or(Value::Null);
        // the browser tools without an address are judged by the page the browser is on, so that
        // per-domain network rules cover them too
        if permission == Permission::Network && name.starts_with("browser_") {
            if let Some(o) = parsed.as_object_mut() {
                if o.get("url").is_none_or(Value::is_null) && o.get("query").is_none_or(Value::is_null) {
                    o.insert("url".into(), json!(self.browser.current_url()));
                }
            }
        }
        let outcome = crate::commands::call_decision(
            turbo,
            &self.rules,
            self.mode == Mode::Plan,
            Self::mode_allows_for(&self.mode, Self::category_of(permission, &kind)),
            Self::permission_word(permission),
            kind_mcp,
            name,
            &parsed,
            self.auto_approve,
            if self.mode == Mode::Plan { "plan" } else { "build" },
            &self.always_allow,
        );
        match outcome {
            crate::commands::Outcome::Plan => (
                CallState::Done {
                    ok: false,
                    output: "Plan mode only allows reading. Write the plan instead.".into(),
                },
                None,
            ),
            crate::commands::Outcome::Blocked => (
                CallState::Done {
                    ok: false,
                    output: "Blocked by a permission rule.".into(),
                },
                None,
            ),
            crate::commands::Outcome::Running(note) => (CallState::Running, note),
            crate::commands::Outcome::Ask => (CallState::Pending, None),
        }
    }

    /// What a mode allows for a category, as the defaults of the page's build agent spell them:
    /// read and memory always, edits from *Edit freely*, commands, network and mcp only from
    /// *Everything* — the window's fourth mode, which the page reaches through its own switch.
    pub fn mode_allows_for(mode: &Mode, category: &str) -> bool {
        match category {
            "read" | "memory" => true,
            "edit" => matches!(mode, Mode::AutoEdit | Mode::Auto),
            _ => *mode == Mode::Auto,
        }
    }

    /// The word the page starts its categories from (`permissionInfo` begins at
    /// `entry.permission`).
    fn permission_word(permission: Permission) -> &'static str {
        match permission {
            Permission::Command => "command",
            Permission::Edit => "edit",
            Permission::Network => "network",
            Permission::Memory => "memory",
            Permission::Computer => "computer",
            Permission::Read | Permission::Other => "read",
        }
    }

    /// The category of a call for the harmless table (`permissionInfo` of the page): a tool of
    /// an MCP server is `mcp`, a shell tool is `bash`, and every other permission is named as the
    /// page names it — a permission neither knows falls back to `read`.
    fn category_of(permission: Permission, kind: &ToolKind) -> &'static str {
        if matches!(kind, ToolKind::Mcp { .. }) {
            return "mcp";
        }
        match permission {
            Permission::Command => "bash",
            Permission::Edit => "edit",
            Permission::Network => "network",
            Permission::Memory => "memory",
            Permission::Computer => "computer",
            Permission::Read | Permission::Other => "read",
        }
    }

    /// The model answered: its calls become cards, and run at once or wait for approval.
    fn finish_round(&mut self, env: &Env) -> Task<Message> {
        let partial: Vec<Partial> = std::mem::take(&mut self.acc).into_iter().filter(|p| !p.name.is_empty()).collect();
        if partial.is_empty() {
            return self.turn_tail(env);
        }
        let mut tasks = Vec::new();
        let mut ids = Vec::new();
        for (n, p) in partial.into_iter().enumerate() {
            let id = if p.id.is_empty() { format!("call_{}_{n}", self.steps) } else { p.id };
            let tool = self.tool(&p.name).cloned();
            let permission = tool.as_ref().map_or(Permission::Other, |t| t.permission);
            let kind = tool.as_ref().map_or(ToolKind::Server, |t| t.kind.clone());
            let (mut state, auto_note) =
                self.starting_state(tool.is_some(), kind.clone(), permission, &p.name, &p.args, env.turbo);
            // an agent for every folder is written outside the workspace: that is always the
            // user's call, unless a rule forbids it (`create_agent` in `runToolCall`)
            if kind == ToolKind::CreateAgent && state == CallState::Running {
                let parsed: Value = serde_json::from_str(&p.args).unwrap_or(Value::Null);
                if parsed.get("scope").and_then(Value::as_str) == Some("user") {
                    state = CallState::Pending;
                }
            }
            // the same call again and again: the model is stuck, so it is told so instead of
            // being run a third time (only a call that gets this far counts, `runToolCall`)
            let mut sent = None;
            if !matches!(state, CallState::Done { .. }) && tool.is_some() {
                if let Some(args) = guard_args(&p.args) {
                    if crate::loopguard::check(&mut self.history, &p.name, &args) {
                        state = CallState::Done { ok: false, output: crate::loopguard::REPEATED_CARD.into() };
                        sent = Some(crate::loopguard::REPEATED_REPLY.to_string());
                    }
                }
            }
            // computer use: the card shows what the model saw last with a marker where the action goes
            let mut cu = None;
            if permission == Permission::Computer && !matches!(state, CallState::Done { .. }) {
                match self.computer_prepare(&p.name) {
                    Err(early) => state = CallState::Done { ok: false, output: early.into() },
                    Ok(shot) => cu = shot.map(|shot| Box::new(CuView { shot: Some(shot), ..Default::default() })),
                }
            }
            let run_now = state == CallState::Running && kind != ToolKind::Task;
            let start_task = state == CallState::Running && kind == ToolKind::Task;
            let waiting = state == CallState::Pending;
            let tool_name = p.name.clone();
            let activity = (permission == Permission::Computer && matches!(state, CallState::Pending | CallState::Running)).then_some(waiting);
            self.items.push(Item::Call(Call { id: id.clone(), name: p.name, args: p.args, permission, kind, state, auto_note, check: None, sent, kid: None, cu, open: None, expanded: false }));
            if let Some(waiting) = activity {
                self.computer_activity(&id, waiting);
            }
            if start_task {
                tasks.push(self.start_task(&id, env));
            }
            if run_now {
                ids.push(id);
            }
            if waiting && env.hooks.notification {
                // `waitFor` of the page fires this the moment a card starts waiting — no await,
                // nothing comes of it (`fireHook('notification', …)` there)
                tasks.push(Task::perform(
                    fire_hook(env.base.to_string(), "notification", "", json!({ "kind": "tool", "name": tool_name })),
                    |_| Message::Code(CodeMsg::HookFired),
                ));
            }
        }
        for id in ids {
            tasks.push(self.run(env.base, &id));
        }
        tasks.push(self.continue_if_done(env));
        Task::batch(tasks)
    }

    /// When every call of the last round has an answer, the model gets them.
    fn continue_if_done(&mut self, env: &Env) -> Task<Message> {
        let kids = self.kids_continue(env);
        Task::batch([self.continue_parent_if_done(env), kids])
    }

    /// The conversation's own next request, once every call of its last round is answered.
    fn continue_parent_if_done(&mut self, env: &Env) -> Task<Message> {
        if self.check_hold.is_some() {
            // the answer of one call is still being checked (`await autoCheckEdit` there)
            return Task::none();
        }
        if !self.running || self.last_round().any(|c| matches!(c.state, CallState::Pending | CallState::Running)) {
            return Task::none();
        }
        self.start_stream(env)
    }

    /// The end of a round that is really over (the `finally` of the page's `agentRunNow`): the
    /// specs are read again, because the agent may have written them (`loadSpecs()` there).
    fn end_round(&mut self, env: &Env) -> Task<Message> {
        self.running = false;
        self.handle = None;
        let learn = self.learn_after_run(env, false);
        let specs = Task::perform(
            crate::specs::fetch_specs(env.base.to_string()),
            |r| Message::Code(CodeMsg::SpecsLoaded(r.unwrap_or_default())),
        );
        // the next queued message goes out now that this run has returned (the runs are chained)
        if !self.queue.is_empty() {
            let next = self.queue.remove(0);
            self.editor = Content::with_text(&next);
            let go = self.update(CodeMsg::Send, env);
            return Task::batch([specs, learn, go]);
        }
        Task::batch([specs, learn])
    }

    /// The tail of a turn in the page's order (`submitText`): the self-review of row 81 first —
    /// it may still start or advance here — then the stop hook once, then the turn is over.
    fn turn_tail(&mut self, env: &Env) -> Task<Message> {
        if let Some(rv) = self.review.as_ref() {
            return match rv.stage {
                ReviewStage::Run if rv.round < 2 => {
                    if let Some(rv) = self.review.as_mut() {
                        rv.round += 1;
                    }
                    // every round reads the changes again (`await refreshChanges()`)
                    self.begin_review(env)
                }
                _ => {
                    // the review is over (or an answer arrived out of order): the stop hook of
                    // the page follows it
                    self.review = None;
                    self.tail = TailPhase::Hook;
                    self.tail_hook_or_end(env)
                }
            };
        }
        if matches!(self.tail, TailPhase::Review) {
            if self.main_run && self.auto_review && self.mode != Mode::Plan {
                self.review = Some(Review {
                    round: 1,
                    stage: ReviewStage::Changes,
                    files: Vec::new(),
                    report: crate::review::Report::default(),
                });
                return self.begin_review(env);
            }
            self.tail = TailPhase::Hook;
        }
        self.tail_hook_or_end(env)
    }

    /// The stop hook is asked once per turn, only for a conversation run that still goes
    /// (`if (code.running)` there), and only when the window knows of one; a blocked answer
    /// sends the model back to work once, and that run ends plainly.
    fn tail_hook_or_end(&mut self, env: &Env) -> Task<Message> {
        if matches!(self.tail, TailPhase::Hook) && self.main_run && env.hooks.stop && self.running {
            // whatever comes next, the page checks this only once
            self.tail = TailPhase::Continue;
            return Task::perform(
                fire_hook(env.base.to_string(), "stop", "", json!({})),
                |answer| Message::Code(CodeMsg::StopHookDone(answer)),
            );
        }
        if matches!(self.tail, TailPhase::Continue) {
            // the run the hook asked for (or its refusal) ended: the turn is over
            self.tail = TailPhase::Done;
        }
        self.end_round(env)
    }

    /// `refreshChanges()` before a review round: the files are read from `self.changes` when the
    /// answer lands, so a refresh that failed leaves the page's behaviour of working with the
    /// changes it already has.
    fn begin_review(&mut self, env: &Env) -> Task<Message> {
        if let Some(rv) = self.review.as_mut() {
            rv.stage = ReviewStage::Changes;
        }
        Task::perform(fetch_changes(env.base.to_string()), |r| Message::Code(CodeMsg::ChangesLoaded(r)))
    }

    /// The changes are in: the files this run changed decide whether it is reviewed at all
    /// (`files` of `selfReview`: unique paths of the changes after `mark`, at most twelve).
    fn review_files_ready(&mut self, env: &Env) -> Task<Message> {
        let mark = self.change_mark;
        let mut files: Vec<String> = Vec::new();
        for c in &self.changes {
            if c.id > mark && !files.contains(&c.path) {
                files.push(c.path.clone());
            }
            if files.len() == 12 {
                break;
            }
        }
        if files.is_empty() {
            // nothing this run changed: the review does not even start (`if (!files.length) return;`)
            // — the stop hook still follows, as it does after `selfReview` returns
            self.review = None;
            self.tail = TailPhase::Hook;
            return self.tail_hook_or_end(env);
        }
        if let Some(rv) = self.review.as_mut() {
            rv.files = files;
        }
        self.ask_review_tools(env)
    }

    /// The review tool and the diagnostics are asked — silently, as the page asks them (only the
    /// notice that follows the answers stands in the list).
    fn ask_review_tools(&mut self, env: &Env) -> Task<Message> {
        if self.review.is_none() {
            return Task::none();
        }
        if let Some(rv) = self.review.as_mut() {
            rv.stage = ReviewStage::Tools;
        }
        let with_review = self.server_has("review");
        let with_diagnostics = self.server_has("diagnostics");
        if !with_review && !with_diagnostics {
            // nothing to ask — the page sends its feedback with an empty report anyway
            return Task::done(Message::Code(CodeMsg::ReviewTools(ReviewAnswers::default())));
        }
        let files = self.review.as_ref().map(|rv| rv.files.clone()).unwrap_or_default();
        Task::perform(
            review_tools(env.base.to_string(), with_review, with_diagnostics, files),
            |answers| Message::Code(CodeMsg::ReviewTools(answers)),
        )
    }

    /// Whether the server's tool list carries a tool (`serverHas` of the page).
    fn server_has(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name == name)
    }

    /// `/batch` runs its task once the tools of the new folder are read (`agentRun(task,
    /// '/batch')` after `await setWorkspace`): a run like every `agentRun` of the page, with
    /// its submit hook, its self-review and its stop hook.
    fn batch_when_ready(&mut self, env: &Env) -> Task<Message> {
        let Some(task) = self.pending_batch.take() else {
            return Task::none();
        };
        self.begin_run(task, "/batch".to_string(), env)
    }

    /// The files the automatic check would ask diagnostics for after one finished edit — the
    /// gates of `autoCheckEdit` (code.js 1941): the switch, a server with the tool, no `deny`
    /// of `modeFor` (plan mode only — `ctx.readOnly` is a sub-agent the window does not have),
    /// and the rule of `bash: diagnostics <path>` for every file.
    fn check_targets(&self, name: &str, args: &str) -> Vec<String> {
        if !self.auto_check || self.mode == Mode::Plan || !self.server_has("diagnostics") {
            return Vec::new();
        }
        if !crate::review::EDIT_TOOLS.contains(&name) {
            return Vec::new();
        }
        let value: Value = serde_json::from_str(args).unwrap_or(Value::Null);
        let parsed = self.rules.parsed();
        let agent = if self.mode == Mode::Plan { "plan" } else { "build" };
        crate::review::auto_check_targets(name, &value, &|path| {
            crate::commands::evaluate_rules(&parsed, "bash", &format!("diagnostics {path}"), agent) == Some("deny")
        })
    }

    pub fn update(&mut self, msg: CodeMsg, env: &Env) -> Task<Message> {
        self.vision = env.vision;
        self.ctx_max = env.ctx_max;
        match msg {
            CodeMsg::PathInput(s) => {
                self.path_input = s;
                Task::none()
            }
            CodeMsg::Open => {
                self.error = None;
                Task::perform(open_workspace(env.base.to_string(), self.path_input.trim().to_string()), |r| Message::Code(CodeMsg::Opened(r)))
            }
            CodeMsg::OpenPath(p) => {
                self.path_input = p;
                self.update(CodeMsg::Open, env)
            }
            CodeMsg::Pick => {
                self.error = None;
                Task::perform(pick_folder(env.base.to_string()), |r| Message::Code(CodeMsg::Picked(r)))
            }
            CodeMsg::Picked(Ok(Some(p))) => self.update(CodeMsg::OpenPath(p), env),
            CodeMsg::Picked(Ok(None)) => Task::none(),
            CodeMsg::Picked(Err(e)) => {
                self.error = Some(e);
                Task::none()
            }
            CodeMsg::Close => {
                self.workspace = None;
                self.computer.epoch += 1;
                self.computer.last_shot = None;
                self.computer.busy.clear();
                self.context = Context::default();
                self.todos.clear();
                self.items.clear();
                self.changes.clear();
                self.running = false;
                self.handle = None;
                Task::none()
            }
            CodeMsg::AgentsLoaded(Some(agents)) => {
                self.agents = agents;
                Task::none()
            }
            CodeMsg::AgentsLoaded(None) => Task::none(),
            CodeMsg::ToggleCard(id) => {
                if let Some(c) = self.call_mut(&id) {
                    c.open = Some(!c.is_open());
                }
                Task::none()
            }
            CodeMsg::ExpandCard(id) => {
                if let Some(c) = self.call_mut(&id) {
                    c.expanded = true;
                }
                Task::none()
            }
            CodeMsg::Br(m) => self.br_update(m, env),
            CodeMsg::Sess(m) => self.sess_update(m, env),
            CodeMsg::BrowserChanged => {
                let tools = Task::perform(fetch_tools(env.base.to_string(), self.computer_on()), |r| Message::Code(CodeMsg::ToolsLoaded(r)));
                Task::batch([self.br_status_task(env, false), tools])
            }
            CodeMsg::ComputerStatus(epoch, status) => {
                if epoch != self.computer.epoch || self.workspace.is_none() {
                    return Task::none();
                }
                let was = self.computer_on();
                self.computer.status = status;
                let now = self.computer_on();
                let mut tasks = Vec::new();
                if was && !now && !self.computer.stopping {
                    // switched off behind the window's back: the panic chord on the keyboard, or the setting in another window
                    if self.running {
                        tasks.push(self.stop_now(env));
                    }
                    self.items.push(Item::Note("Computer use was stopped and switched off. Switch it on again in Settings > Runtime if you want to continue.".into()));
                }
                if self.tools.iter().any(|t| t.name == "computer_screenshot") != now && !self.tools.is_empty() {
                    tasks.push(Task::perform(fetch_tools(env.base.to_string(), now), |r| Message::Code(CodeMsg::ToolsLoaded(r))));
                }
                if !now {
                    self.computer.busy.clear();
                }
                self.computer.stopping = false;
                tasks.push(computer_poll(env.base.to_string(), epoch, if now { 1500 } else { 15000 }));
                Task::batch(tasks)
            }
            CodeMsg::ComputerStop(why) => {
                self.computer.stopping = true;
                let base = env.base.to_string();
                Task::perform(post_computer(base, "stop"), move |()| Message::Code(CodeMsg::ComputerStopped(why.clone())))
            }
            CodeMsg::ComputerStopped(why) => {
                let mut tasks = Vec::new();
                if self.running {
                    tasks.push(self.stop_now(env));
                }
                self.computer.last_shot = None;
                self.computer.busy.clear();
                self.items.push(Item::Note(format!("{why} Computer use is switched off; the agent cannot see or touch the screen any more. Switch it on again in Settings > Runtime.")));
                self.pending_toast = Some("Computer use stopped and switched off".into());
                // the status is read again at once; `stopping` ends with that answer
                tasks.push(computer_poll(env.base.to_string(), self.computer.epoch, 0));
                Task::batch(tasks)
            }
            CodeMsg::ToolReply(id, result, shot) => {
                if let Some(shot) = shot {
                    let computer = self.call_mut(&id).is_some_and(|c| c.permission == Permission::Computer);
                    if let Some(c) = self.call_mut(&id) {
                        c.cu.get_or_insert_with(Default::default).image = Some(shot.clone());
                    }
                    if computer && result.is_ok() {
                        self.computer.last_shot = Some(shot);
                        if !self.computer.warned_vision && !env.vision && !env.model.is_some_and(|m| m.starts_with(crate::providers::REMOTE_PREFIX)) {
                            self.computer.warned_vision = true;
                            self.items.push(Item::Note("The loaded model cannot look at pictures, so it will not see the screenshots. Choose a vision model (the model list marks them with \"vision\"), or use a provider model that can see images.".into()));
                        }
                    }
                }
                self.computer.busy.clear();
                // `fllmBrowserPoke`: a browser tool just did something, look at the page now
                let poke = if self.call_mut(&id).is_some_and(|c| c.name.starts_with("browser_")) {
                    Task::done(Message::Code(CodeMsg::Br(BrMsg::Poke)))
                } else {
                    Task::none()
                };
                let done = self.update(CodeMsg::ToolDone(id, result), env);
                Task::batch([done, poke])
            }
            CodeMsg::JobOutput(id) => match crate::jobs::toggle_output(&mut self.jobs, &id) {
                Some(true) => {
                    let args = json!({ self.job_arg_name(): id });
                    let base = env.base.to_string();
                    Task::perform(run_tool(base, "bash_output".into(), args), move |r| Message::Code(CodeMsg::JobOutputDone(id.clone(), r)))
                }
                _ => Task::none(),
            },
            CodeMsg::JobOutputDone(id, answer) => {
                crate::jobs::output_arrived(&mut self.jobs, &id, answer);
                Task::none()
            }
            CodeMsg::JobKill(id) => {
                let args = json!({ self.job_arg_name(): id });
                let base = env.base.to_string();
                Task::perform(run_tool(base, "bash_kill".into(), args), move |r| Message::Code(CodeMsg::JobKillDone(id.clone(), r)))
            }
            CodeMsg::JobKillDone(id, answer) => {
                if let Some(toast) = crate::jobs::kill_answered(&mut self.jobs, &id, answer) {
                    self.pending_toast = Some(toast);
                }
                Task::none()
            }
            CodeMsg::LearnStatus(None) => Task::none(),
            CodeMsg::LearnStatus(Some(st)) => {
                // the switch of the server is the truth (`fllm-rs learn off`, a folder's `.fllm/learn.json`)
                if st.enabled != Some(self.learn) {
                    self.learn = st.enabled != Some(false);
                    self.pending_learn = Some(self.learn);
                }
                if st.enabled == Some(false) {
                    return Task::none();
                }
                let root = self.workspace.as_ref().map(|w| w.root.clone()).unwrap_or_default();
                Task::perform(crate::learn::fetch_notice(env.base.to_string(), root), |n| Message::Code(CodeMsg::LearnNotice(n)))
            }
            CodeMsg::LearnNotice(notice) => {
                if !notice.is_empty() {
                    self.items.push(Item::Note(notice));
                }
                Task::none()
            }
            CodeMsg::LearnToggle(on) => {
                self.learn = on;
                self.pending_learn = Some(on);
                let root = self.workspace.as_ref().map(|w| w.root.clone()).unwrap_or_default();
                Task::perform(crate::learn::set_enabled(env.base.to_string(), on, root), |r| Message::Code(CodeMsg::LearnSwitched(r)))
            }
            CodeMsg::LearnSwitched(Err(e)) => {
                self.pending_toast = Some(e);
                Task::none()
            }
            CodeMsg::LearnSwitched(Ok(())) => Task::none(),
            CodeMsg::LearnQueued(false, _) => Task::none(),
            CodeMsg::LearnQueued(true, since) => Self::learn_poll_in(crate::learn::FIRST_POLL_MS, since, 0),
            CodeMsg::LearnPoll(since, tries) => {
                let root = self.workspace.as_ref().map(|w| w.root.clone()).unwrap_or_default();
                Task::perform(crate::learn::fetch_status(env.base.to_string(), root), move |st| Message::Code(CodeMsg::LearnPolled(st, since, tries + 1)))
            }
            CodeMsg::LearnPolled(st, since, tries) => {
                if crate::learn::poll_done(st.as_ref(), since) {
                    let toast = crate::learn::toast_text(st.as_ref().and_then(|s| s.last_run.as_ref()));
                    if !toast.is_empty() {
                        self.pending_toast = Some(toast);
                    }
                    self.learn_refresh = true;
                    return Task::none();
                }
                if tries < crate::learn::MAX_POLLS {
                    return Self::learn_poll_in(crate::learn::POLL_MS, since, tries);
                }
                Task::none()
            }
            CodeMsg::LearnThumb(value) => {
                let Some(last) = self.learn_last.clone().filter(|l| !l.skills.is_empty()) else {
                    self.pending_toast = Some(crate::learn::NO_SKILL_TOAST.to_string());
                    return Task::none();
                };
                let skills = last.skills.clone();
                let body = crate::learn::feedback_body(value, &last);
                Task::perform(crate::learn::feedback(env.base.to_string(), body), move |r| Message::Code(CodeMsg::LearnThumbDone(value, skills.clone(), r)))
            }
            CodeMsg::LearnThumbDone(value, skills, Ok((_, notes))) => {
                self.pending_toast = Some(crate::learn::thumb_toast(value, &skills, &notes));
                self.learn_refresh = true;
                Task::none()
            }
            CodeMsg::LearnThumbDone(_, _, Err(e)) => {
                self.pending_toast = Some(e);
                Task::none()
            }
            CodeMsg::LearnCommand(plan, status, error) => {
                use crate::learn::LearnPlan;
                if let Some(e) = error {
                    self.pending_toast = Some(e);
                }
                match plan {
                    LearnPlan::ProjectUsage => {}
                    LearnPlan::Project(project) => {
                        if let Some(st) = &status {
                            self.learn = st.enabled == Some(true);
                            self.pending_learn = Some(self.learn);
                        }
                        self.items.push(Item::Note(crate::learn::project_notice(project)));
                    }
                    LearnPlan::Show(_) => self.items.push(Item::Note(crate::learn::learn_notice(self.learn, status.as_ref()))),
                }
                Task::none()
            }
            CodeMsg::KidStream(id, ev) => self.kid_event(&id, ev, env),
            CodeMsg::SpecsLoaded(specs) => {
                self.specs = specs;
                Task::none()
            }
            CodeMsg::SpecListDone(specs) => {
                self.specs = specs.clone();
                self.items.push(Item::Note(crate::specs::listing_notice(&specs)));
                Task::none()
            }
            // `agentRun(data.prompt, '/spec …')`: a run like every other, and the specs are
            // read again when it ends (the `finally` of `agentRunNow` calls `loadSpecs`)
            CodeMsg::SpecPrompt { prompt, display } => self.begin_run(prompt, display, env),
            CodeMsg::SpecReplaced(specs) => {
                self.specs = specs;
                Task::none()
            }
            CodeMsg::SpecFailed { message, toast } => {
                if toast {
                    // the checkbox of the side toasts and reads the list again (`catch` of 3035)
                    self.pending_toast = Some(message);
                    Task::perform(crate::specs::fetch_specs(env.base.to_string()), |r| Message::Code(CodeMsg::SpecsLoaded(r.unwrap_or_default())))
                } else {
                    self.items.push(Item::Note(message));
                    Task::none()
                }
            }
            CodeMsg::SpecTick { name, index, done } => {
                let action = crate::specs::tick_action(done);
                Task::perform(
                    crate::specs::spec_act(env.base.to_string(), crate::specs::tick_body(action, &name, index)),
                    |r| match r {
                        Ok(v) => Message::Code(CodeMsg::SpecReplaced(crate::specs::parse_specs(&v))),
                        Err(e) => Message::Code(CodeMsg::SpecFailed { message: e, toast: true }),
                    },
                )
            }
            CodeMsg::SpecRunTask { name, index } => {
                let base = env.base.to_string();
                let index = index as f64;
                Task::perform(crate::specs::spec_act(base, crate::specs::task_prompt_body(&name, index)), move |r| {
                    match r {
                        Ok(v) => Message::Code(CodeMsg::SpecPrompt {
                            prompt: v.get("prompt").and_then(Value::as_str).unwrap_or_default().to_string(),
                            display: crate::specs::run_display(&name, index),
                        }),
                        Err(e) => Message::Code(CodeMsg::SpecFailed { message: e, toast: false }),
                    }
                })
            }
            CodeMsg::WorktreeDone { action, name, batch, result } => match result {
                Ok(data) => match action {
                    "new" => {
                        let path = data.get("path").and_then(Value::as_str).unwrap_or_default().to_string();
                        let branch = data.get("branch").and_then(Value::as_str).unwrap_or_default().to_string();
                        let name = name.unwrap_or_default();
                        let notice = match &batch {
                            Some(_) => format!("Batch {name}: working on branch {branch} in {path}."),
                            None => format!(
                                "Worktree {branch} at {path}. The agent works there now; /worktree leave brings you back, /worktree remove {name} deletes it."
                            ),
                        };
                        // the page awaits `setWorkspace` before it writes this, so the notice
                        // waits for the folder to be open
                        self.pending_notice = Some(notice);
                        self.pending_batch = batch;
                        self.update(CodeMsg::OpenPath(path), env)
                    }
                    "leave" => {
                        let path = data.get("path").and_then(Value::as_str).unwrap_or_default().to_string();
                        let branch_part = match data.get("branch").and_then(Value::as_str) {
                            Some(b) if !b.is_empty() => format!(", branch {b}"),
                            _ => String::new(),
                        };
                        self.pending_notice = Some(format!("Back in the main worktree ({path}{branch_part})."));
                        self.update(CodeMsg::OpenPath(path), env)
                    }
                    "remove" => {
                        let removed = data.get("removed").and_then(Value::as_str).unwrap_or_default();
                        self.items.push(Item::Note(format!(
                            "Removed the worktree {removed}. Its branch stays until you delete it."
                        )));
                        Task::none()
                    }
                    _ => {
                        // the rows of the list, exactly as the page joins them
                        let rows: Vec<String> = data
                            .get("worktrees")
                            .and_then(Value::as_array)
                            .map(|list| {
                                list.iter()
                                    .map(|w| {
                                        format!(
                                            "{} {}  {}  {}",
                                            if w.get("current").and_then(Value::as_bool).unwrap_or(false) { "*" } else { " " },
                                            w.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()).unwrap_or("(main)"),
                                            w.get("branch").and_then(Value::as_str).unwrap_or(""),
                                            w.get("path").and_then(Value::as_str).unwrap_or("")
                                        )
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        let text = if rows.is_empty() {
                            "No worktrees. /worktree new <name> creates one.".to_string()
                        } else {
                            rows.join("\n")
                        };
                        self.items.push(Item::Note(text));
                        Task::none()
                    }
                },
                Err(e) => {
                    let text = if batch.is_some() {
                        // the page reads no message of the server here: only its own two texts
                        if e.starts_with("transport:") {
                            format!("The batch could not start: {}", e.trim_start_matches("transport:"))
                        } else {
                            "The worktree could not be created.".to_string()
                        }
                    } else {
                        worktree_error("The worktree action failed", &e)
                    };
                    self.items.push(Item::Note(text));
                    Task::none()
                }
            },
            CodeMsg::Opened(Ok(w)) => {
                let w_root = w.root.clone();
                self.path_input = w.root.clone();
                self.workspace = Some(w);
                self.error = None;
                // the notice of `/worktree new` and `/worktree leave` waits for this moment
                // (the page awaits `setWorkspace` before it writes it)
                if let Some(note) = self.pending_notice.take() {
                    self.items.push(Item::Note(note));
                }
                let base = env.base.to_string();
                Task::batch([
                    Task::perform(fetch_tools(base.clone(), self.computer_on()), |r| Message::Code(CodeMsg::ToolsLoaded(r))),
                    self.computer_start_poll(&base),
                    self.br_status_task(env, false),
                    Task::perform(fetch_changes(base.clone()), |r| Message::Code(CodeMsg::ChangesLoaded(r))),
                    Task::perform(fetch_context(base.clone()), |r| Message::Code(CodeMsg::ContextLoaded(r))),
                    // the custom commands of the folder come with it (`loadCustomCommands`
                    // runs with the other loads when the page opens a workspace)
                    Task::perform(load_custom_commands(base.clone()), |r| Message::Code(CodeMsg::CustomLoaded(r))),
                    // and the specs of `.fllm/specs` (`loadSpecs` runs with them at 814)
                    Task::perform(crate::specs::fetch_specs(base.clone()), |r| Message::Code(CodeMsg::SpecsLoaded(r.unwrap_or_default()))),
                    // what the server says of learning in this folder (`fllmLearnStatus`)
                    Task::perform(crate::learn::fetch_status(base.clone(), w_root.clone()), |st| Message::Code(CodeMsg::LearnStatus(st))),
                    // and the custom sub-agents of the folder (`loadAgents`)
                    Task::perform(fetch_agents(base, w_root), |r| Message::Code(CodeMsg::AgentsLoaded(r.ok()))),
                ])
            }
            CodeMsg::ContextLoaded(Ok(c)) => {
                self.context = c;
                Task::none()
            }
            CodeMsg::ContextLoaded(Err(_)) => Task::none(),
            CodeMsg::Answer(id, answer) => {
                if let Some(c) = self.call_mut(&id).filter(|c| c.kind == ToolKind::Ask && c.state == CallState::Pending) {
                    c.state = CallState::Done { ok: true, output: answer };
                }
                self.continue_if_done(env)
            }
            CodeMsg::Opened(Err(e)) => {
                // the worktree exists even when its folder could not be opened: its notice is
                // still true (the page's `setWorkspace` failure has its own text)
                if let Some(note) = self.pending_notice.take() {
                    self.items.push(Item::Note(note));
                }
                self.error = Some(e);
                self.end_round(env)
            }
            CodeMsg::ToolsLoaded(Ok((tools, for_model))) => {
                self.tools = tools;
                self.tools_json = for_model;
                self.batch_when_ready(env)
            }
            CodeMsg::ToolsLoaded(Err(e)) => {
                self.error = Some(e);
                // the page runs its batch even when a load of the folder had a problem
                self.batch_when_ready(env)
            }
            CodeMsg::ChangesLoaded(Ok(c)) => {
                self.changes = c;
                if self.review.as_ref().is_some_and(|rv| rv.stage == ReviewStage::Changes) {
                    return self.review_files_ready(env);
                }
                Task::none()
            }
            CodeMsg::ChangesLoaded(Err(_)) => {
                if self.review.as_ref().is_some_and(|rv| rv.stage == ReviewStage::Changes) {
                    // the page works with the changes it has; a refresh that fails is swallowed
                    return self.review_files_ready(env);
                }
                Task::none()
            }
            CodeMsg::ReviewTools(answers) => {
                let Some(rv) = self.review.as_ref() else {
                    return Task::none();
                };
                let round = rv.round;
                let files_changed = rv.files.len();
                let report = crate::review::Report {
                    review: answers.review.unwrap_or_default(),
                    problems: answers
                        .diagnostics
                        .into_iter()
                        .filter(|(_, output)| !crate::review::is_clean_diagnostic(Some(output)))
                        .map(|(path, output)| crate::review::Problem { path, output })
                        .collect(),
                };
                let feedback = crate::review::review_feedback(&report, round);
                let Some(feedback) = feedback else {
                    // the second round with nothing a machine can state: the review is over —
                    // and the stop hook follows it (`selfReview` returned)
                    self.review = None;
                    self.tail = TailPhase::Hook;
                    return self.tail_hook_or_end(env);
                };
                let notice = if round == 1 {
                    crate::review::reviewing_notice(files_changed)
                } else {
                    crate::review::another_round_notice(report.problems.len())
                };
                if let Some(rv) = self.review.as_mut() {
                    rv.report = report;
                    rv.stage = ReviewStage::Run;
                }
                // the notice stands in the list, the feedback goes to the model unseen
                self.items.push(Item::Note(notice));
                self.items.push(Item::ReviewFeedback(feedback));
                self.start_stream(env)
            }
            CodeMsg::Edit(action) => {
                self.editor.perform(action);
                Task::none()
            }
            CodeMsg::Send => {
                let typed = self.editor.text().trim().to_string();
                // a question of the agent is waiting: what you write is its answer (the page
                // answers its questions from the card and queues everything else in the box;
                // this window answers in the box, so the words go to the question and the
                // commands of the table wait with them)
                if let Some(id) = self.pending_ask().map(|c| c.id.clone()) {
                    if typed.is_empty() {
                        return Task::none();
                    }
                    self.editor = Content::new();
                    return self.update(CodeMsg::Answer(id, typed), env);
                }
                // while the agent works: `/stop` stops it, and an empty line does too (the page's
                // `submitCode`); everything else waits in the queue and is sent when the run is over
                if self.running || self.beginning {
                    let stop = typed.is_empty() || typed.eq_ignore_ascii_case("/stop");
                    self.editor = Content::new();
                    if !stop {
                        self.queue.push(typed);
                        return Task::none();
                    }
                    return self.update(CodeMsg::Stop, env);
                }
                // every command of the page is answered here — the switch of `/autoapprove`
                // included, which is its case of `runCodeCommand` — and without asking for a
                // workspace or a model
                if let Some(steps) = crate::commands::dispatch(&typed, &self.cmd_state(env)) {
                    self.editor = Content::new();
                    return self.run_steps(steps, env);
                }
                let prompt = self.editor.text().trim().to_string();
                if prompt.is_empty() || self.workspace.is_none() || env.model.is_none() {
                    return Task::none();
                }
                self.editor = Content::new();
                self.begin_run(prompt.clone(), prompt, env)
            }
            CodeMsg::Stop => self.stop_now(env),
            CodeMsg::QueueDrop(i) => {
                if i < self.queue.len() {
                    self.queue.remove(i);
                }
                Task::none()
            }
            CodeMsg::Stream(ev) => {
                if !self.running {
                    return Task::none();
                }
                match ev {
                    StreamEvent::Delta(s) => {
                        if let Some(Item::Assistant { text, md }) = self.items.last_mut() {
                            text.push_str(&s);
                            md.push_str(&s);
                        }
                        Task::none()
                    }
                    StreamEvent::ToolCall { index, id, name, arguments } => {
                        while self.acc.len() <= index {
                            self.acc.push(Partial::default());
                        }
                        let p = &mut self.acc[index];
                        if let Some(id) = id {
                            p.id = id;
                        }
                        if let Some(n) = name {
                            p.name = n;
                        }
                        p.args.push_str(&arguments);
                        Task::none()
                    }
                    StreamEvent::Done => self.finish_round(env),
                    StreamEvent::Error(e) => {
                        self.items.push(Item::Note(e));
                        self.running = false;
                        self.handle = None;
                        // a run that ended in an error is still learned from, as a failed one
                        self.learn_after_run(env, true)
                    }
                    StreamEvent::Reasoning(_) | StreamEvent::Logprobs(_) | StreamEvent::Usage { .. } | StreamEvent::Timings { .. } => Task::none(),
                }
            }
            CodeMsg::Approve(id) => {
                if !self.call_mut(&id).is_some_and(|c| c.state == CallState::Pending) {
                    return Task::none();
                }
                let run = self.run(env.base, &id);
                Task::batch([run, self.continue_if_done(env)])
            }
            CodeMsg::Always(id, key) => {
                if !self.call_mut(&id).is_some_and(|c| c.state == CallState::Pending) {
                    return Task::none();
                }
                if key == "edit" {
                    // the page sets its perm straight (not through `setPerm`): the mode changes
                    // and `lastBuildPerm` stays as it was
                    self.mode = Mode::AutoEdit;
                } else {
                    self.always_allow.insert(key);
                }
                let run = self.run(env.base, &id);
                Task::batch([run, self.continue_if_done(env)])
            }
            CodeMsg::ToggleAutoReview(on) => {
                self.auto_review = on;
                self.pending_auto_review = Some(on);
                Task::none()
            }
            CodeMsg::ToggleAutoCheck(on) => {
                self.auto_check = on;
                self.pending_auto_check = Some(on);
                Task::none()
            }
            CodeMsg::Deny(id) => {
                if let Some(c) = self.call_mut(&id).filter(|c| c.state == CallState::Pending) {
                    let output = if c.kind == ToolKind::Ask { "The user did not answer. Decide yourself or go on without it." } else { "The user declined this call. Do not repeat it; ask what to do instead or take another way." };
                    c.state = CallState::Done { ok: false, output: output.into() };
                    self.computer.busy.clear();
                }
                self.continue_if_done(env)
            }
            CodeMsg::ToolDone(id, result) => {
                let (changed, name, args, reply, ok) = {
                    let Some(c) = self.call_mut(&id) else { return Task::none() };
                    if !matches!(c.state, CallState::Running) {
                        return Task::none();
                    }
                    let changes_files = matches!(c.permission, Permission::Edit | Permission::Command);
                    let reply = match &result {
                        Ok(output) => output.clone(),
                        Err(output) => output.clone(),
                    };
                    let carried = (c.name.clone(), c.args.clone(), result.is_ok());
                    c.state = match result {
                        Ok(output) => CallState::Done { ok: true, output },
                        Err(output) => CallState::Done { ok: false, output },
                    };
                    (changes_files, carried.0, carried.1, reply, carried.2)
                };
                let mut tasks = Vec::new();
                if changed {
                    tasks.push(Task::perform(fetch_changes(env.base.to_string()), |r| Message::Code(CodeMsg::ChangesLoaded(r))));
                }
                if ok {
                    // `trackJob(call.name, args, r.output)`: a command started in the background is kept
                    let value: Value = serde_json::from_str(if args.trim().is_empty() { "{}" } else { &args }).unwrap_or(Value::Null);
                    crate::jobs::track_job(&mut self.jobs, &name, &value, &reply);
                }
                if ok && name == "create_agent" {
                    // `await loadAgents()` of `createAgentFile`: the new agent is offered at once
                    tasks.push(self.load_agents(env));
                }
                // `autoCheckEdit`: after a successful edit the check runs first, and the model
                // gets its answer with the problems between the two markers — the card is done
                // while it runs, as on the page, and the round waits on `check_hold`
                let targets = if ok { self.check_targets(&name, &args) } else { Vec::new() };
                if targets.is_empty() {
                    tasks.push(self.continue_if_done(env));
                    return Task::batch(tasks);
                }
                self.check_hold = Some(id.clone());
                let base = env.base.to_string();
                let held = id.clone();
                tasks.push(Task::perform(
                    async move {
                        let mut problems = Vec::new();
                        for path in targets {
                            // one diagnostics run per path, answered one after the other; an
                            // error is thrown away, as the page's `.catch(() => null)` does
                            if let Ok(text) = run_tool(base.clone(), "diagnostics".into(), crate::review::auto_check_args(&path)).await {
                                if crate::review::auto_check_keep(&text) {
                                    problems.push(text);
                                }
                            }
                        }
                        CodeMsg::CheckDone { id: held, reply, outputs: problems }
                    },
                    Message::Code,
                ));
                Task::batch(tasks)
            }
            CodeMsg::SubmitHookDone { prompt, display, answer } => {
                // `submitted.context` of `submitText`: what the hooks have to say rides along
                // with this message, in its own block after the words of the box
                let prompt = if answer.context.is_empty() {
                    prompt
                } else {
                    format!("{prompt}\n\n[hooks]\n{}", answer.context)
                };
                self.run_after_begin(prompt, display, env)
            }
            CodeMsg::ComputerBegun { prompt, display } => {
                // a stop in the meantime has taken the run back
                if !std::mem::take(&mut self.beginning) {
                    return Task::none();
                }
                self.run_prompt(&prompt, &display, env)
            }
            CodeMsg::StopHookDone(answer) => {
                if answer.blocked && self.running {
                    // the notice stands in the list, the words for the model do not (the page
                    // pushes them into `ctx.messages`), and the model runs once more — the tail
                    // stands on `Continue`, so that run ends plainly
                    self.items.push(Item::Note(format!("A stop hook asks the agent to continue: {}", answer.reason)));
                    self.items
                        .push(Item::StopHook(format!(
                            "A stop hook asks you to continue before you finish: {}",
                            answer.reason
                        )));
                    return self.start_stream(env);
                }
                self.tail = TailPhase::Done;
                self.end_round(env)
            }
            CodeMsg::HookFired => Task::none(),
            CodeMsg::CheckDone { id, reply, outputs } => {
                self.check_hold = None;
                if !outputs.is_empty() {
                    let problems = outputs.join("\n");
                    if let Some(c) = self.call_mut(&id) {
                        // the card carries the block of `item.check`, the model gets the reply
                        // with the findings between the markers (`item.check` and `reply = …`)
                        c.check = Some(problems.clone());
                        c.sent = Some(crate::review::auto_check_reply(&reply, &problems, env.ctx_max));
                    }
                }
                self.continue_if_done(env)
            }
            CodeMsg::SetMode(m) => {
                if m != Mode::Plan {
                    self.build_mode = m;
                } else if self.mode != Mode::Plan {
                    self.build_mode = self.mode;
                }
                self.mode = m;
                if m == Mode::Plan {
                    // a plan changes nothing: what waited for approval is refused (a
                    // sub-agent's calls as well)
                    self.each_call_mut(|c| {
                        if c.state == CallState::Pending && c.kind != ToolKind::Ask {
                            c.state = CallState::Done { ok: false, output: "Plan mode only allows reading. Write the plan instead.".into() };
                        }
                    });
                }
                // calls that wait now may be allowed
                let ids: Vec<String> = self.waiting_ids(|c| c.kind != ToolKind::Ask && m.allows(c.permission));
                let mut tasks = Vec::new();
                for id in ids {
                    tasks.push(self.run(env.base, &id));
                }
                tasks.push(self.continue_if_done(env));
                Task::batch(tasks)
            }
            CodeMsg::Undo(to) => {
                let body = match to {
                    Some(id) => json!({ "to": id }),
                    None => json!({ "steps": 1 }),
                };
                Task::perform(undo(env.base.to_string(), body), |r| Message::Code(CodeMsg::Undone(r)))
            }
            CodeMsg::Undone(result) => {
                match result {
                    // `Undone: {output}` of `undoChange`
                    Ok(text) => self.items.push(Item::Note(format!("Undone: {text}"))),
                    Err(e) => self.error = Some(e),
                }
                Task::perform(fetch_changes(env.base.to_string()), |r| Message::Code(CodeMsg::ChangesLoaded(r)))
            }
            CodeMsg::ExecutePlan => {
                if self.running || self.mode != Mode::Plan {
                    return Task::none();
                }
                self.mode = self.build_mode;
                self.editor = Content::with_text("Carry out the plan above.");
                self.update(CodeMsg::Send, env)
            }
            CodeMsg::CustomLoaded(list) => {
                self.custom = list;
                Task::none()
            }
            CodeMsg::CustomBody(result, arg_text, source) => match result {
                Ok(body) => {
                    let prompt = crate::commands::expand_command(&body, &arg_text);
                    self.begin_run(prompt, source, env)
                }
                Err(e) => {
                    self.items.push(Item::Note(e));
                    Task::none()
                }
            },
            CodeMsg::SearchDone(result, query) => {
                let text = match result {
                    Ok(hits) => crate::commands::search_hits_text(&hits, &query),
                    Err(e) => format!("The search failed: {e}"),
                };
                self.items.push(Item::Note(text));
                Task::none()
            }
            CodeMsg::UserRunDone(id, result, share) => {
                let mut shared = None;
                if let Some(Item::Call(c)) = self.items.iter_mut().find(|i| matches!(i, Item::Call(c) if c.id == id)) {
                    let output = match &result {
                        Ok(o) => o.clone(),
                        Err(e) => e.clone(),
                    };
                    c.state = CallState::Done { ok: result.is_ok(), output: output.clone() };
                    if share {
                        let command = serde_json::from_str::<Value>(&c.args)
                            .ok()
                            .and_then(|v| v.get("command").and_then(Value::as_str).map(str::to_string))
                            .unwrap_or_default();
                        shared = Some(crate::commands::share_text(&command, &output));
                    }
                }
                // the output goes with the next message, never into the conversation
                if let Some(text) = shared {
                    self.pending.push(text);
                }
                Task::none()
            }
            CodeMsg::MemorySaved(result, notice) => match result {
                Ok(()) => {
                    self.items.push(Item::Note(notice));
                    // the instructions are read again, so the fact is part of the next prompt
                    let base = env.base.to_string();
                    Task::perform(fetch_context(base), |r| Message::Code(CodeMsg::ContextLoaded(r)))
                }
                Err(e) => {
                    self.pending_toast = Some(e);
                    Task::none()
                }
            },
            CodeMsg::NewSession => {
                if self.running {
                    return Task::none();
                }
                self.reset_session();
                Task::none()
            }
        }
    }
}

// ---- sub-agents ------------------------------------------------------------------------------------------------------------------

/// The arguments as `parseArgs` reads them for the guard: an empty text is `{}`, a text that is
/// no JSON is no call to count (the call is refused before the guard anyway).
fn guard_args(text: &str) -> Option<Value> {
    if text.trim().is_empty() {
        return Some(json!({}));
    }
    serde_json::from_str(text).ok()
}

/// `GET /api/studio/agents?workspace=…`: the custom agents of the folder and of the user.
pub async fn fetch_agents(base: String, root: String) -> Result<Vec<crate::subagents::CustomAgent>, String> {
    let query = if root.is_empty() { String::new() } else { format!("?workspace={}", encode_query(&root)) };
    call_get(&base, &format!("/api/studio/agents{query}")).await.map(|v| crate::subagents::parse_agents(&v))
}

/// `createAgentFile` of the page: the file of a custom agent through the agents route. The answer
/// is the line the model reads, or the message of the route (`Error 400` without one).
pub async fn create_agent_file(base: String, root: String, args: Value) -> Result<String, String> {
    let scope = if args.get("scope").and_then(Value::as_str) == Some("user") { "user" } else { "workspace" };
    // `args.x || ''`: what the model wrote goes on as it is, unless it is empty or zero
    let raw = |key: &str| match args.get(key) {
        Some(Value::String(s)) if s.is_empty() => json!(""),
        Some(Value::Number(n)) if n.as_f64() == Some(0.0) => json!(""),
        Some(Value::Null | Value::Bool(false)) | None => json!(""),
        Some(v) => v.clone(),
    };
    let name = match raw("name") {
        Value::String(s) => s.trim().to_string(),
        other => other.to_string().trim().to_string(),
    };
    let body = json!({
        "name": name,
        "description": raw("description"),
        "tools": args.get("tools").filter(|t| t.is_array()).cloned().unwrap_or_else(|| json!([])),
        "model": raw("model"),
        "body": raw("instructions"),
        "scope": scope,
        "workspace": if root.is_empty() { Value::Null } else { json!(root) },
    });
    match call_post(&base, "/api/studio/agents", body).await {
        Ok(_) => Ok(format!(
            "The agent \"{name}\" was created ({}). Start it with the task tool: agent \"{name}\".",
            if scope == "user" { "for every folder" } else { "for this folder" }
        )),
        Err(e) => Err(match e.strip_prefix("the server answered ") {
            Some(status) => format!("Error {}", status.split_whitespace().next().unwrap_or_default()),
            None => e,
        }),
    }
}

impl CodeState {
    /// The timer of the watch for a learning step (`window.setTimeout(poll, …)` of `lnWatchRun`).
    fn learn_poll_in(ms: u64, since: f64, tries: u32) -> Task<Message> {
        Task::perform(async move { tokio::time::sleep(std::time::Duration::from_millis(ms)).await }, move |_| Message::Code(CodeMsg::LearnPoll(since, tries)))
    }

    /// `/learn` (`case 'learn'`): the plan of the words, the request and the status behind it.
    fn learn_command(&mut self, arg: &str, env: &Env) -> Task<Message> {
        use crate::learn::LearnPlan;
        let plan = crate::learn::plan_learn(arg, self.workspace.is_some());
        let (base, root) = (env.base.to_string(), self.workspace.as_ref().map(|w| w.root.clone()).unwrap_or_default());
        match plan {
            LearnPlan::ProjectUsage => {
                self.items.push(Item::Note(crate::learn::PROJECT_USAGE.to_string()));
                Task::none()
            }
            LearnPlan::Project(project) => Task::perform(
                async move {
                    // the page does not look at the answer of this request
                    let _ = crate::learn::set_project(base.clone(), root.clone(), project).await;
                    crate::learn::fetch_status(base, root).await
                },
                move |st| Message::Code(CodeMsg::LearnCommand(plan, st, None)),
            ),
            LearnPlan::Show(set) => {
                if let Some(on) = set {
                    self.learn = on;
                    self.pending_learn = Some(on);
                }
                Task::perform(
                    async move {
                        let mut error = None;
                        if let Some(on) = set {
                            error = crate::learn::set_enabled(base.clone(), on, root.clone()).await.err();
                        }
                        (crate::learn::fetch_status(base, root).await, error)
                    },
                    move |(st, error)| Message::Code(CodeMsg::LearnCommand(plan, st, error)),
                )
            }
        }
    }

    /// The answer is delivered: the server learns from the run in the background
    /// (`fllmLearnAfterRun`), unless learning is off or the run said nothing.
    fn learn_after_run(&mut self, env: &Env, failed: bool) -> Task<Message> {
        let Some(from) = self.learn_from.take() else { return Task::none() };
        if !self.learn {
            return Task::none();
        }
        let wire = self.wire(None);
        let messages = crate::learn::run_messages_default(&wire[1..], from);
        if messages.is_empty() {
            return Task::none();
        }
        // the skills are read from the reduced messages, as the page reads them (`lnSkillsRead(run)`)
        let skills = crate::learn::skills_read_run(&messages);
        if self.session_id.is_empty() {
            self.session_id = crate::sessions::new_id();
        }
        let workspace = self.workspace.as_ref().map(|w| w.root.clone()).unwrap_or_default();
        self.learn_last = Some(crate::learn::Last { session: self.session_id.clone(), workspace: workspace.clone(), skills: skills.clone() });
        let since = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64()) - 2.0;
        let body = crate::learn::after_run_body(&self.session_id, &workspace, failed, &skills, messages);
        Task::perform(crate::learn::after_run(env.base.to_string(), body), move |queued| Message::Code(CodeMsg::LearnQueued(queued, since)))
    }

    /// The pause of `schedule(ms)`: the tick of this chain comes after it.
    fn br_schedule(&mut self, ms: u64) -> Task<Message> {
        self.br_epoch += 1;
        let epoch = self.br_epoch;
        Task::perform(
            async move {
                if ms > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                }
            },
            move |()| Message::Code(CodeMsg::Br(BrMsg::Tick(epoch))),
        )
    }

    /// The next input goes to the server when none is on its way (the page chains them so that fast typing arrives in order).
    fn br_pump(&mut self, env: &Env) -> Task<Message> {
        if self.br_sending {
            return Task::none();
        }
        let Some(input) = self.br_queue.pop_front() else { return Task::none() };
        self.br_sending = true;
        let base = env.base.to_string();
        let body = input.body();
        Task::perform(post_browser(base, "/input", body), move |r| Message::Code(CodeMsg::Br(BrMsg::Answered(input.clone(), r))))
    }

    fn br_send(&mut self, input: crate::browser::Input, env: &Env) -> Task<Message> {
        self.br_queue.push_back(input);
        self.br_pump(env)
    }

    fn br_status_task(&self, env: &Env, then_tick: bool) -> Task<Message> {
        Task::perform(fetch_browser(env.base.to_string()), move |answer| Message::Code(CodeMsg::Br(BrMsg::Status { then_tick, answer })))
    }

    /// The picture of a frame as the stage draws it.
    fn br_take_picture(&mut self) {
        match &self.browser.image {
            None => {
                self.br_handle = None;
                self.br_src.clear();
            }
            Some(src) if *src != self.br_src => {
                self.br_src = src.clone();
                let data = src.split_once(',').map_or(src.as_str(), |(_, d)| d);
                self.br_handle = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data).ok().map(iced::widget::image::Handle::from_bytes);
            }
            Some(_) => {}
        }
    }

    /// The stage as the window draws it: where it is and how big (the picture starts at its corner).
    fn br_rect(&self) -> Option<(crate::browser::Rect, (f64, f64))> {
        let (p, size) = self.br_pointer?;
        let rect = crate::browser::Rect { left: 0.0, top: 0.0, width: f64::from(size.width), height: f64::from(size.height) };
        Some((rect, (f64::from(p.x), f64::from(p.y))))
    }

    fn br_update(&mut self, m: BrMsg, env: &Env) -> Task<Message> {
        match m {
            BrMsg::Toggle => {
                self.browser.open = !self.browser.open;
                if self.browser.open {
                    self.br_status_task(env, true)
                } else {
                    // the timer ends
                    self.br_epoch += 1;
                    Task::none()
                }
            }
            BrMsg::Hide => {
                self.browser.open = false;
                self.br_epoch += 1;
                Task::none()
            }
            BrMsg::Wide => {
                self.br_wide = !self.br_wide;
                Task::none()
            }
            BrMsg::FocusAddress => {
                if !self.browser.open {
                    return Task::none();
                }
                self.br_focus = false;
                iced::widget::operation::focus(iced::widget::Id::new("br-address"))
            }
            BrMsg::Quick(address) => {
                self.browser.url_input = address;
                self.browser.url_focused = false;
                match self.browser.submit() {
                    Some(i) => self.br_send(i, env),
                    None => Task::none(),
                }
            }
            BrMsg::Status { then_tick, answer } => {
                self.browser.apply_status(answer.as_ref());
                if then_tick && self.browser.open { self.br_schedule(0) } else { Task::none() }
            }
            BrMsg::Tick(epoch) => {
                if epoch != self.br_epoch || !self.browser.open {
                    return Task::none();
                }
                if !env.visible {
                    return self.br_schedule(1500);
                }
                let read = self.browser.tick_reads_status();
                let since = self.browser.key.clone();
                let hint = self.browser.tick_asks_frame();
                Task::perform(browser_tick(env.base.to_string(), read, since, hint), move |(status, frame)| Message::Code(CodeMsg::Br(BrMsg::TickDone { epoch, status, frame })))
            }
            BrMsg::TickDone { epoch, status, frame } => {
                if epoch != self.br_epoch || !self.browser.open {
                    return Task::none();
                }
                if let Some(answer) = status {
                    self.browser.apply_status(answer.as_ref());
                }
                if let Some(data) = frame {
                    self.browser.apply_frame(&data);
                    self.br_take_picture();
                }
                let pause = self.browser.tick_pause_ms();
                self.br_schedule(pause)
            }
            BrMsg::Poke => {
                if self.browser.open { self.br_schedule(0) } else { self.br_status_task(env, false) }
            }
            BrMsg::Answered(input, answer) => {
                self.br_sending = false;
                let was_locate = matches!(input, crate::browser::Input::Locate { .. });
                if let Some(toast) = self.browser.input_answered(&input, answer) {
                    self.pending_toast = Some(toast);
                }
                if was_locate && self.browser.draft.is_some() {
                    self.br_note = Content::new();
                }
                let mut tasks = vec![self.br_pump(env)];
                match input {
                    crate::browser::Input::Close => {
                        self.browser.closed();
                        self.br_take_picture();
                        tasks.push(self.br_status_task(env, false));
                    }
                    other => {
                        // `setTimeout(poke, 120)`, and a new address is followed by `loadStatus()`
                        tasks.push(Task::perform(async { tokio::time::sleep(std::time::Duration::from_millis(120)).await }, |()| Message::Code(CodeMsg::Br(BrMsg::Poke))));
                        if matches!(other, crate::browser::Input::Navigate(_)) {
                            tasks.push(self.br_status_task(env, false));
                        }
                    }
                }
                Task::batch(tasks)
            }
            BrMsg::UrlEdit(v) => {
                self.browser.url_input = v;
                self.browser.url_focused = true;
                self.br_focus = false;
                Task::none()
            }
            BrMsg::Submit => {
                self.browser.url_focused = false;
                match self.browser.submit() {
                    Some(i) => self.br_send(i, env),
                    None => Task::none(),
                }
            }
            BrMsg::Back => self.br_send(crate::browser::Input::Back, env),
            BrMsg::Forward => self.br_send(crate::browser::Input::Forward, env),
            BrMsg::Reload => self.br_send(crate::browser::Input::Reload, env),
            BrMsg::CloseAsk => {
                self.br_confirm_close = true;
                Task::none()
            }
            BrMsg::CloseCancel => {
                self.br_confirm_close = false;
                Task::none()
            }
            BrMsg::CloseOk => {
                self.br_confirm_close = false;
                self.br_send(crate::browser::Input::Close, env)
            }
            BrMsg::Comment => {
                self.browser.toggle_commenting();
                if !self.browser.commenting {
                    self.br_note = Content::new();
                }
                Task::none()
            }
            BrMsg::Move(point, size) => {
                self.br_pointer = Some((point, size));
                Task::none()
            }
            BrMsg::Click => {
                let Some((rect, at)) = self.br_rect() else { return Task::none() };
                match self.browser.click(at, rect, false) {
                    Some(input) => {
                        // `stage.focus()`: the keys go to the page now
                        self.br_focus = true;
                        self.browser.url_focused = false;
                        self.br_send(input, env)
                    }
                    None => Task::none(),
                }
            }
            BrMsg::Double => {
                let Some((rect, at)) = self.br_rect() else { return Task::none() };
                match self.browser.dblclick(at, rect) {
                    Some(input) => self.br_send(input, env),
                    None => Task::none(),
                }
            }
            BrMsg::Scroll(delta) => {
                let Some((rect, at)) = self.br_rect() else { return Task::none() };
                // the window's wheel turns the other way round: up is positive there, down on the web
                let (d, unit) = match delta {
                    iced::mouse::ScrollDelta::Lines { x, y } => ((-f64::from(x), -f64::from(y)), crate::browser::WheelUnit::Lines),
                    iced::mouse::ScrollDelta::Pixels { x, y } => ((-f64::from(x), -f64::from(y)), crate::browser::WheelUnit::Pixels),
                };
                if self.browser.wheel(at, rect, d, unit) {
                    Task::perform(async { tokio::time::sleep(std::time::Duration::from_millis(50)).await }, |()| Message::Code(CodeMsg::Br(BrMsg::WheelFlush)))
                } else {
                    Task::none()
                }
            }
            BrMsg::WheelFlush => match self.browser.flush_wheel() {
                Some(input) => self.br_send(input, env),
                None => Task::none(),
            },
            BrMsg::Key(press) => {
                if !self.br_focus || !self.browser.open || !env.visible {
                    return Task::none();
                }
                // the paste shortcut is handled by the paste event, which has the text
                if (press.ctrl || press.meta) && !press.alt && press.key.to_lowercase() == "v" && self.browser.running {
                    return iced::clipboard::read().map(|text| Message::Code(CodeMsg::Br(BrMsg::Pasted(text))));
                }
                match self.browser.key(&press) {
                    Some(input) => self.br_send(input, env),
                    None => Task::none(),
                }
            }
            BrMsg::Pasted(text) => match text.and_then(|t| self.browser.paste(&t)) {
                Some(input) => self.br_send(input, env),
                None => Task::none(),
            },
            BrMsg::NoteEdit(action) => {
                self.br_focus = false;
                self.br_note.perform(action);
                Task::none()
            }
            BrMsg::NoteAdd => {
                let note = self.br_note.text();
                if self.browser.add_comment(&note) {
                    self.br_note = Content::new();
                }
                Task::none()
            }
            BrMsg::NoteCancel => {
                self.browser.cancel_note();
                self.br_note = Content::new();
                Task::none()
            }
            BrMsg::Del(i) => {
                self.browser.delete_comment(i);
                Task::none()
            }
            BrMsg::SendComments => {
                // the box gets the sentence when it is empty, and the message goes
                if self.editor.text().trim().is_empty() {
                    let sentence = self.browser.send_text("");
                    self.editor = Content::with_text(&sentence);
                }
                self.update(CodeMsg::Send, env)
            }
            BrMsg::OpenSettings => Task::done(Message::Go(crate::app::View::Settings)),
        }
    }

    /// `computerOn()`.
    pub fn computer_on(&self) -> bool {
        self.computer.status.as_ref().is_some_and(|s| s.enabled && s.available)
    }

    /// The chain of polls starts again (a folder was opened): the older one ends by itself.
    fn computer_start_poll(&mut self, base: &str) -> Task<Message> {
        self.computer.epoch += 1;
        computer_poll(base.to_string(), self.computer.epoch, 0)
    }

    /// One press of Esc (`escapeStreak`): `true` when the third one came within the time.
    pub fn computer_esc(&mut self, now_ms: f64) -> bool {
        if !self.computer_on() {
            return false;
        }
        let (times, fire) = crate::computer::escape_streak_default(&self.computer.esc, now_ms);
        self.computer.esc = times;
        fire
    }

    /// `computerActivity(item, waiting)`: what the banner says an action is doing.
    fn computer_activity(&mut self, id: &str, waiting: bool) {
        let Some(c) = self.call_mut(id) else { return };
        let args: Value = serde_json::from_str(&c.args).unwrap_or(Value::Null);
        let (verb, target) = crate::computer::tool_label(&c.name, &args).unwrap_or_else(|| (c.name.clone(), String::new()));
        self.computer.busy = format!("{} {verb} {target}", if waiting { "Waiting for your answer:" } else { "Operating your computer:" });
    }

    /// `computerPrepare(item)` at the start of a computer card: the screenshot the action refers
    /// to is remembered; an action before the first screenshot is an error.
    fn computer_prepare(&self, name: &str) -> Result<Option<Shot>, &'static str> {
        if crate::computer::is_look(name) {
            return Ok(None);
        }
        match &self.computer.last_shot {
            Some(shot) => Ok(Some(shot.clone())),
            None => Err("Take a screenshot first (computer_screenshot): coordinates refer to the last screenshot you took."),
        }
    }

    /// `jobArgName()`: what the server calls the id in `bash_output`.
    pub fn job_arg_name(&self) -> String {
        let props: Vec<String> = self
            .tools_json
            .as_array()
            .into_iter()
            .flatten()
            .find(|d| d.pointer("/function/name").and_then(Value::as_str) == Some("bash_output"))
            .and_then(|d| d.pointer("/function/parameters/properties"))
            .and_then(Value::as_object)
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        crate::jobs::job_arg_name(&props)
    }

    fn kid_index(&self, call_id: &str) -> Option<usize> {
        self.kids.iter().position(|k| k.call_id == call_id)
    }

    /// What a sub-agent is sent: its instructions and its messages so far (`None` once it is
    /// over). The window never shows it; the tests read the conversation through it.
    pub fn kid_wire(&self, call_id: &str) -> Option<Vec<WireMessage>> {
        let kid = &self.kids[self.kid_index(call_id)?];
        let mut wire = vec![WireMessage::new("system", kid.system.clone())];
        wire.extend(kid.messages.iter().cloned());
        Some(wire)
    }

    /// The tools a sub-agent is offered (`None` once it is over).
    pub fn kid_tools(&self, call_id: &str) -> Option<Value> {
        Some(self.kids[self.kid_index(call_id)?].tools.clone())
    }

    /// How many sub-agents work, and how many wait for a place.
    pub fn kids_working_and_waiting(&self) -> (usize, usize) {
        let waiting = self.kids.iter().filter(|k| k.phase == KidPhase::Queued).count();
        (self.kids.len() - waiting, waiting)
    }

    /// `loadAgents()`; the Settings page asks for it after it saved or deleted an agent
    /// (`window.fllmReloadAgents`).
    pub fn reload_agents(&self, base: &str) -> Task<Message> {
        let root = self.workspace.as_ref().map(|w| w.root.clone()).unwrap_or_default();
        Task::perform(fetch_agents(base.to_string(), root), |r| Message::Code(CodeMsg::AgentsLoaded(r.ok())))
    }

    fn load_agents(&self, env: &Env) -> Task<Message> {
        self.reload_agents(env.base)
    }

    /// A `task` call has been made: `runSubAgent` decides what it is, and a sub-agent that may
    /// start does so as soon as one of the four places is free.
    fn start_task(&mut self, id: &str, env: &Env) -> Task<Message> {
        let args: Value = {
            let Some(c) = self.call_mut(id) else { return Task::none() };
            match serde_json::from_str(if c.args.trim().is_empty() { "{}" } else { &c.args }) {
                Ok(v) => v,
                Err(_) => {
                    c.state = CallState::Done { ok: false, output: "The arguments are not valid JSON.".into() };
                    c.sent = Some("Error: the arguments were not valid JSON. Call the tool again with a valid JSON object.".into());
                    return Task::none();
                }
            }
        };
        let plan = match crate::subagents::plan_child(&args, &self.agents, self.mode == Mode::Plan, 0) {
            Ok(plan) => plan,
            Err(refusal) => {
                if let Some(c) = self.call_mut(id) {
                    c.state = CallState::Done { ok: false, output: refusal.output };
                    c.sent = Some(refusal.reply);
                }
                return Task::none();
            }
        };
        let mut tools = self.tools_for_model();
        if let (Some(allowed), Some(list)) = (&plan.allowed, tools.as_array_mut()) {
            list.retain(|t| t.pointer("/function/name").and_then(Value::as_str).is_some_and(|n| allowed.iter().any(|a| a == n)));
        }
        let system = format!("{}{}", self.system_prompt(env.extra), plan.system_suffix);
        if let Some(c) = self.call_mut(id) {
            c.kid = Some(Box::new(KidView { agent_type: plan.agent_type.clone(), description: plan.description.clone(), progress: "starting".into(), items: Vec::new() }));
        }
        self.kids.push(Kid {
            call_id: id.to_string(),
            phase: KidPhase::Queued,
            system,
            tools,
            messages: vec![WireMessage::new("user", plan.first_message.clone())],
            budget: crate::subagents::child_budget(env.ctx_max),
            step: 0,
            wrap_up: false,
            acc: Vec::new(),
            text: String::new(),
            calls: Vec::new(),
            handle: None,
            history: Vec::new(),
            plan,
        });
        self.kid_pump(env)
    }

    /// Starts the sub-agents that wait, as far as the places go (`MAX_PARALLEL_CHILDREN`).
    fn kid_pump(&mut self, env: &Env) -> Task<Message> {
        let mut tasks = Vec::new();
        loop {
            let working = self.kids.iter().filter(|k| k.phase != KidPhase::Queued).count();
            if working >= crate::subagents::MAX_PARALLEL_CHILDREN {
                break;
            }
            let Some(next) = self.kids.iter().find(|k| k.phase == KidPhase::Queued).map(|k| k.call_id.clone()) else { break };
            tasks.push(self.kid_step(&next, env));
        }
        Task::batch(tasks)
    }

    /// One step of a sub-agent's loop: the wrap-up check, the progress line, and the request.
    fn kid_step(&mut self, call_id: &str, env: &Env) -> Task<Message> {
        let Some(i) = self.kid_index(call_id) else { return Task::none() };
        let Some(model) = self.kids[i].plan.model.clone().or_else(|| env.model.map(str::to_string)) else {
            return self.kid_finish(call_id, Err(crate::subagents::ChildFailure::Error("No model is loaded.".into())), env);
        };
        let kid = &mut self.kids[i];
        let fixed = crate::subagents::estimate_tokens(&kid.system) + crate::subagents::estimate_tokens(&kid.tools.to_string());
        if !kid.wrap_up && crate::subagents::needs_wrap_up(kid.step, kid.plan.max_steps, fixed, crate::subagents::estimate_messages(&kid.messages), kid.budget) {
            // out of steps or room: the report is asked for and further calls are ignored
            kid.wrap_up = true;
            kid.messages.push(WireMessage::new("user", crate::subagents::WRAP_UP_MESSAGE));
        }
        let progress = format!("step {}{}", kid.step + 1, if kid.wrap_up { ", writing report" } else { "" });
        kid.acc.clear();
        kid.text.clear();
        kid.phase = KidPhase::Streaming;
        let mut wire = vec![WireMessage::new("system", kid.system.clone())];
        wire.extend(kid.messages.iter().cloned());
        let (tools, id) = (kid.tools.clone(), kid.call_id.clone());
        let events = api::chat_stream_with_tools(env.base.to_string(), model, wire, env.params.clone(), Some(tools));
        let (task, handle) = Task::stream(events).map(move |ev| Message::Code(CodeMsg::KidStream(id.clone(), ev))).abortable();
        self.kids[i].handle = Some(handle.abort_on_drop());
        if let Some(view) = self.call_mut(call_id).and_then(|c| c.kid.as_mut()) {
            view.progress = progress;
            view.items.push(KidItem::Assistant(String::new()));
        }
        task
    }

    /// What a sub-agent's stream says.
    fn kid_event(&mut self, call_id: &str, ev: StreamEvent, env: &Env) -> Task<Message> {
        let Some(i) = self.kid_index(call_id) else { return Task::none() };
        if !self.running || self.kids[i].phase != KidPhase::Streaming {
            return Task::none();
        }
        match ev {
            StreamEvent::Delta(s) => {
                self.kids[i].text.push_str(&s);
                if let Some(KidItem::Assistant(text)) = self.call_mut(call_id).and_then(|c| c.kid.as_mut()).and_then(|v| v.items.last_mut()) {
                    text.push_str(&s);
                }
                Task::none()
            }
            StreamEvent::ToolCall { index, id, name, arguments } => {
                let acc = &mut self.kids[i].acc;
                while acc.len() <= index {
                    acc.push(Partial::default());
                }
                let p = &mut acc[index];
                if let Some(id) = id {
                    p.id = id;
                }
                if let Some(n) = name {
                    p.name = n;
                }
                p.args.push_str(&arguments);
                Task::none()
            }
            StreamEvent::Done => self.kid_round(call_id, env),
            StreamEvent::Error(e) => self.kid_finish(call_id, Err(crate::subagents::ChildFailure::Error(e)), env),
            StreamEvent::Reasoning(_) | StreamEvent::Logprobs(_) | StreamEvent::Usage { .. } | StreamEvent::Timings { .. } => Task::none(),
        }
    }

    /// The model has answered one step: no calls end the loop with this answer as the report,
    /// calls become cards of the task, asked about and run like the conversation's own.
    fn kid_round(&mut self, call_id: &str, env: &Env) -> Task<Message> {
        let Some(i) = self.kid_index(call_id) else { return Task::none() };
        let kid = &mut self.kids[i];
        let partial: Vec<Partial> = std::mem::take(&mut kid.acc).into_iter().filter(|p| !p.name.is_empty()).collect();
        // out of steps or room the calls are ignored (`wrapUp ? [] : result.calls`)
        let partial = if kid.wrap_up { Vec::new() } else { partial };
        let content = kid.text.clone();
        if !content.is_empty() || !partial.is_empty() {
            let mut m = WireMessage::new("assistant", content.clone());
            if !partial.is_empty() {
                m.tool_calls = Some(Value::Array(
                    partial
                        .iter()
                        .enumerate()
                        .map(|(n, p)| json!({"id": format!("{}~{}.{n}", kid.call_id, kid.step), "type": "function", "function": {"name": p.name, "arguments": if p.args.is_empty() { "{}" } else { p.args.as_str() }}}))
                        .collect(),
                ));
            }
            kid.messages.push(m);
        }
        if partial.is_empty() {
            return self.kid_finish(call_id, Ok(content.trim().to_string()), env);
        }
        let step = kid.step;
        let read_only = kid.plan.read_only;
        if content.is_empty() {
            // nothing to show of this step but its calls (`ctx.view.splice` of the page)
            if let Some(view) = self.call_mut(call_id).and_then(|c| c.kid.as_mut()) {
                if matches!(view.items.last(), Some(KidItem::Assistant(t)) if t.is_empty()) {
                    view.items.pop();
                }
            }
        }
        let mut ids = Vec::new();
        let mut run_ids = Vec::new();
        let mut tasks = Vec::new();
        for (n, p) in partial.into_iter().enumerate() {
            let id = format!("{call_id}~{step}.{n}");
            let tool = self.tool(&p.name).cloned();
            let permission = tool.as_ref().map_or(Permission::Other, |t| t.permission);
            let kind = tool.as_ref().map_or(ToolKind::Server, |t| t.kind.clone());
            // what a sub-agent is refused before anything is asked (`runToolCall`, depth one)
            let client = matches!(kind, ToolKind::Todo { .. } | ToolKind::Ask | ToolKind::Task | ToolKind::CreateAgent);
            let denial = tool.as_ref().and_then(|_| {
                crate::subagents::child_denial(&p.name, client, permission == Permission::Computer, permission == Permission::Read && kind == ToolKind::Server, read_only)
            });
            let (state, auto_note, sent) = match denial {
                Some((card, reply)) => (CallState::Done { ok: false, output: card.into() }, None, Some(reply)),
                None => {
                    let (mut state, note) = self.starting_state(tool.is_some(), kind.clone(), permission, &p.name, &p.args, env.turbo);
                    let mut sent = None;
                    if !matches!(state, CallState::Done { .. }) && tool.is_some() {
                        if let (Some(args), Some(i)) = (guard_args(&p.args), self.kid_index(call_id)) {
                            if crate::loopguard::check(&mut self.kids[i].history, &p.name, &args) {
                                state = CallState::Done { ok: false, output: crate::loopguard::REPEATED_CARD.into() };
                                sent = Some(crate::loopguard::REPEATED_REPLY.to_string());
                            }
                        }
                    }
                    (state, note, sent)
                }
            };
            let run_now = state == CallState::Running;
            let waiting = state == CallState::Pending;
            let tool_name = p.name.clone();
            let call = Call { id: id.clone(), name: p.name, args: p.args, permission, kind, state, auto_note, check: None, sent, kid: None, cu: None, open: None, expanded: false };
            if let Some(view) = self.call_mut(call_id).and_then(|c| c.kid.as_mut()) {
                view.items.push(KidItem::Call(Box::new(call)));
            }
            ids.push(id.clone());
            if run_now {
                run_ids.push(id);
            }
            if waiting && env.hooks.notification {
                tasks.push(Task::perform(
                    fire_hook(env.base.to_string(), "notification", "", json!({ "kind": "tool", "name": tool_name })),
                    |_| Message::Code(CodeMsg::HookFired),
                ));
            }
        }
        if let Some(i) = self.kid_index(call_id) {
            self.kids[i].calls = ids;
            self.kids[i].phase = KidPhase::Calls;
        }
        for id in run_ids {
            tasks.push(self.run(env.base, &id));
        }
        tasks.push(self.continue_if_done(env));
        Task::batch(tasks)
    }

    /// The sub-agents whose calls are all answered go on: the answers are told to the model in
    /// the order of the calls and the next step starts.
    fn kids_continue(&mut self, env: &Env) -> Task<Message> {
        if self.check_hold.is_some() || !self.running {
            return Task::none();
        }
        let ready: Vec<String> = self
            .kids
            .iter()
            .filter(|k| k.phase == KidPhase::Calls && k.calls.iter().all(|id| matches!(self.call_state(id), Some(CallState::Done { .. }))))
            .map(|k| k.call_id.clone())
            .collect();
        let mut tasks = Vec::new();
        for id in ready {
            let Some(i) = self.kid_index(&id) else { continue };
            let ids = self.kids[i].calls.clone();
            let mut replies = Vec::new();
            for call_id in &ids {
                let reply = self.call_mut(call_id).map(|c| match (&c.sent, &c.state) {
                    (Some(sent), _) => sent.clone(),
                    (None, CallState::Done { ok: true, output }) => output.clone(),
                    (None, CallState::Done { ok: false, output }) => format!("Error: {output}"),
                    _ => "The call did not finish.".to_string(),
                });
                replies.push((call_id.clone(), reply.unwrap_or_default()));
            }
            let kid = &mut self.kids[i];
            for (call_id, content) in replies {
                let mut m = WireMessage::new("tool", content);
                m.tool_call_id = Some(call_id);
                kid.messages.push(m);
            }
            kid.calls.clear();
            kid.step += 1;
            if kid.step >= kid.plan.max_steps {
                let last = kid.text.trim().to_string();
                tasks.push(self.kid_finish(&id, Ok(last), env));
            } else {
                tasks.push(self.kid_step(&id, env));
            }
        }
        Task::batch(tasks)
    }

    /// The sub-agent is done: its card gets the report and the model gets the reply
    /// (`runSubAgent`'s end), and a place is free for the next one.
    fn kid_finish(&mut self, call_id: &str, result: Result<String, crate::subagents::ChildFailure>, env: &Env) -> Task<Message> {
        let Some(i) = self.kid_index(call_id) else { return Task::none() };
        let kid = self.kids.remove(i);
        let (status, output, reply) = crate::subagents::child_outcome(result, !self.running, &kid.plan.description, env.ctx_max);
        if let Some(c) = self.call_mut(call_id) {
            c.state = CallState::Done { ok: status == "ok", output };
            c.sent = Some(reply);
            if let Some(view) = c.kid.as_mut() {
                view.progress.clear();
                if matches!(view.items.last(), Some(KidItem::Assistant(t)) if t.is_empty()) {
                    view.items.pop();
                }
            }
        }
        let pump = self.kid_pump(env);
        Task::batch([pump, self.continue_if_done(env)])
    }
}

// ---- the page --------------------------------------------------------------------------------------------------------------------

fn mono_box<'a>(t: Tokens, body: Element<'a, Message>) -> Element<'a, Message> {
    container(body).padding([8, 12]).width(Length::Fill).style(theme::inset(t)).into()
}

fn call_card<'a>(t: Tokens, c: &'a Call) -> Element<'a, Message> {
    let args: Value = serde_json::from_str(&c.args).unwrap_or(Value::Null);
    let status = c.status();
    let open = c.is_open();
    let (glyph, glyph_colour) = match status {
        crate::toolcard::Status::Pending => ("?", t.primary),
        crate::toolcard::Status::Running => ("…", t.muted),
        crate::toolcard::Status::Ok => ("✓", t.ok),
        crate::toolcard::Status::Error => ("✗", t.danger),
        crate::toolcard::Status::Denied => ("⊘", t.muted),
    };
    let state_label: Element<'a, Message> = match &c.state {
        CallState::Pending => text("Waiting for you").size(12).color(t.primary).into(),
        CallState::Running => text("Running…").size(12).color(t.muted).into(),
        CallState::Done { ok: true, .. } => text("Done").size(12).color(t.ok).into(),
        CallState::Done { ok: false, .. } => text("Failed").size(12).color(t.danger).into(),
    };
    // every card is named as the page names it (`toolLabel`): `Read file`, `Check code`, `Sub-agent (explore)` …
    let mcp = match &c.kind {
        ToolKind::Mcp { server, tool } => Some((server.as_str(), tool.as_str())),
        _ => None,
    };
    let (verb, target) = crate::toolcard::label(&c.name, &args, mcp);
    let progress = c.kid.as_ref().filter(|_| c.state == CallState::Running).map(|k| k.progress.clone()).unwrap_or_default();
    let header = row![
        text(glyph).size(13).color(glyph_colour).width(14),
        text(verb).size(13).font(Font { weight: iced::font::Weight::Semibold, ..Font::MONOSPACE }),
        text(target).size(13).font(Font::MONOSPACE).color(t.fg_soft).width(Length::Fill).wrapping(text::Wrapping::None),
        text(progress).size(12).color(t.muted),
        state_label,
    ]
    .spacing(10)
    .align_y(Alignment::Center);
    // the summary of the page's `<details>`: a press opens or closes the card
    let header = button(header).on_press(Message::Code(CodeMsg::ToggleCard(c.id.clone()))).padding(0).width(Length::Fill).style(|_, _| button::Style::default());
    let mut body = column![header].spacing(8);
    if let Some(reason) = &c.auto_note {
        // `tool-auto` of the page: its own line over the body of the card
        body = body.push(
            container(text(format!("Approved on its own: {reason}")).size(11).color(t.muted))
                .padding(iced::Padding { top: 2.0, right: 0.0, bottom: 0.0, left: 22.0 })
                .width(Length::Fill),
        );
    }

    if open && c.name == "task" {
        // the job as it was given (the first 1500 characters), the conversation of the sub-agent
        // under it, and its report when it is done
        let prompt: String = args.get("prompt").and_then(Value::as_str).unwrap_or_default().chars().take(1500).collect();
        body = body.push(mono_box(t, text(prompt).size(12).font(Font::MONOSPACE).color(t.fg_soft).into()));
        let mut kids = column![].spacing(8);
        let items = c.kid.as_ref().map(|k| k.items.as_slice()).unwrap_or_default();
        if items.is_empty() {
            kids = kids.push(text("Starting ...").size(12).color(t.muted));
        }
        for item in items {
            kids = kids.push(match item {
                // the answer that is on its way (the parent's own list draws the same dots)
                KidItem::Assistant(words) if words.is_empty() => {
                    if c.state == CallState::Running { text("…").size(13).color(t.muted).into() } else { Element::from(Space::new().height(0)) }
                }
                KidItem::Assistant(words) => text(words.as_str()).size(13).color(t.fg_soft).into(),
                KidItem::Call(inner) => call_card(t, inner),
            });
        }
        body = body.push(container(kids).padding(Padding { top: 0.0, right: 0.0, bottom: 0.0, left: 14.0 }).width(Length::Fill));
        if let CallState::Done { output, ok } = &c.state {
            if !output.is_empty() {
                body = body.push(
                    column![
                        text("Report").size(11).font(Font { weight: iced::font::Weight::Semibold, ..Font::MONOSPACE }).color(t.muted),
                        mono_box(t, text(output.as_str()).size(13).color(if *ok { t.fg } else { t.danger }).into()),
                    ]
                    .spacing(4),
                );
            }
        }
    }

    if open {
        // a command stands in its card as the page draws it (`tool-cmd`)
        if c.name == "bash" {
            body = body.push(mono_box(t, text(format!("$ {}", crate::computer::text_or_empty(args.get("command").filter(|v| crate::jobs::truthy(Some(v)))))).size(12).font(Font::MONOSPACE).color(t.fg).into()));
        }
        // an edit shows what it removes and adds, each line on a tinted band, sixty lines at first
        let rows = crate::toolcard::diff_rows(&c.name, &args);
        if !rows.is_empty() {
            let shown_n = if c.expanded { rows.len() } else { rows.len().min(crate::toolcard::DIFF_LINES) };
            let more = rows.len() - shown_n;
            let mut lines = column![].spacing(0);
            for (kind, l) in rows.into_iter().take(shown_n) {
                let (mark, color, band) = match kind {
                    crate::toolcard::Kind::Add => ("+ ", t.ok, Some(iced::Color { a: 0.14, ..t.ok })),
                    crate::toolcard::Kind::Del => ("− ", t.danger, Some(iced::Color { a: 0.14, ..t.danger })),
                    crate::toolcard::Kind::More => ("  ", t.muted, None),
                    crate::toolcard::Kind::Ctx => ("  ", t.fg_soft, None),
                };
                lines = lines.push(
                    container(text(format!("{mark}{}", l.replace('\t', "    "))).size(12).font(Font::MONOSPACE).color(color))
                        .width(Length::Fill)
                        .padding([1, 6])
                        .style(move |_| container::Style { background: band.map(iced::Background::Color), ..Default::default() }),
                );
            }
            if more > 0 {
                lines = lines.push(button(text(format!("Show {more} more lines")).size(11)).on_press(Message::Code(CodeMsg::ExpandCard(c.id.clone()))).padding([2, 8]).style(theme::ghost_button(t)));
            }
            body = body.push(container(lines).padding([6, 6]).width(Length::Fill).style(theme::inset(t)));
        }
        if let (CallState::Done { output, ok }, true) = (&c.state, c.name != "task") {
            if !output.is_empty() {
                let (shown, more) = crate::toolcard::output_view(output, c.expanded);
                // a tab is not in the monospace font: four spaces show it
                let shown = shown.replace('\t', "    ");
                let mut out = column![text(shown).size(12).font(Font::MONOSPACE).color(if *ok { t.fg_soft } else { t.danger })].spacing(4);
                if let Some(label) = more {
                    out = out.push(button(text(label).size(11)).on_press(Message::Code(CodeMsg::ExpandCard(c.id.clone()))).padding([2, 8]).style(theme::ghost_button(t)));
                }
                body = body.push(mono_box(t, out.into()));
            }
        }
    }
    if let (true, Some(problems)) = (open, &c.check) {
        // `tool-check` of the page: a block of its own — *Automatic check* over the findings
        body = body.push(
            column![
                text("Automatic check")
                    .size(11)
                    .font(Font { weight: iced::font::Weight::Semibold, ..Font::MONOSPACE })
                    .color(t.muted),
                mono_box(t, text(problems.clone()).size(12).font(Font::MONOSPACE).color(t.fg_soft).into()),
            ]
            .spacing(4),
        );
    }
    if open && (c.permission == Permission::Computer || c.cu.as_ref().is_some_and(|v| v.image.is_some())) {
        if let Some(picture) = computer_body(t, c, &args) {
            body = body.push(picture);
        }
    }
    if c.state == CallState::Pending && c.kind == ToolKind::Ask {
        let question = args.get("question").and_then(Value::as_str).unwrap_or("The agent has a question.").to_string();
        body = body.push(text(question).size(14));
        let mut options = row![].spacing(8);
        for o in args.get("options").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
            options = options.push(button(text(o.to_string()).size(13)).on_press(Message::Code(CodeMsg::Answer(c.id.clone(), o.to_string()))).padding([5, 14]).style(theme::primary_button(t)));
        }
        body = body.push(options.push(button(text("Skip").size(13)).on_press(Message::Code(CodeMsg::Deny(c.id.clone()))).padding([5, 14]).style(theme::secondary_button(t))));
        body = body.push(text("Or type your answer below and press Enter.").size(12).color(t.muted));
    } else if c.state == CallState::Pending {
        let args_value: Value = serde_json::from_str(&c.args).unwrap_or(Value::Null);
        let kind_mcp = match &c.kind {
            ToolKind::Mcp { server, tool } => Some((server.as_str(), tool.as_str())),
            _ => None,
        };
        // the middle button of the page's `permissionHtml`: it shows up only for what `Always`
        // could remember, and for a command the part behind `bash:` stands in its label
        if c.permission == Permission::Computer {
            body = body.push(text(crate::computer::computer_question(&c.name, &args_value)).size(14).font(Font { weight: iced::font::Weight::Bold, ..Font::default() }));
        }
        let always = crate::commands::always_key(CodeState::permission_word(c.permission), kind_mcp, &c.name, &args_value);
        let mut buttons = row![button(text("Allow").size(13))
            .on_press(Message::Code(CodeMsg::Approve(c.id.clone())))
            .padding([5, 16])
            .style(theme::primary_button(t))];
        if let Some(key) = always {
            // the page slices five characters off after testing the prefix; `strip_prefix`
            // reads the same
            let label = match key.strip_prefix("bash:") {
                Some(head) => format!("Always ({head})"),
                None => "Always".to_string(),
            };
            buttons = buttons.push(button(text(label).size(13))
                .on_press(Message::Code(CodeMsg::Always(c.id.clone(), key)))
                .padding([5, 16])
                .style(theme::ghost_button(t)));
        }
        body = body.push(buttons
            .push(button(text("Deny").size(13)).on_press(Message::Code(CodeMsg::Deny(c.id.clone()))).padding([5, 16]).style(theme::secondary_button(t)))
            .spacing(8));
    }
    container(body).padding(12).width(Length::Fill).style(theme::card(t, theme::R_MD, c.state == CallState::Pending)).into()
}

fn timeline<'a>(s: &'a CodeState, t: Tokens, theme: &iced::Theme, tint: bool) -> Element<'a, Message> {
    let mut col = column![].spacing(12).width(Length::Fill);
    if s.items.is_empty() {
        let mut intro = column![text("Tell the agent what to do in this folder: explain the code, find a bug, add a test, rename something.").size(14).color(t.muted)].spacing(6);
        if !s.context.files.is_empty() {
            intro = intro.push(text(format!("Project instructions read: {}", s.context.files.iter().map(|f| f.0.as_str()).collect::<Vec<_>>().join(", "))).size(12).color(t.muted));
        }
        let notes = s.context.memory_workspace.len() + s.context.memory_global.len();
        if notes > 0 {
            intro = intro.push(text(format!("{notes} saved memory note{} are part of the instructions.", if notes == 1 { "" } else { "s" })).size(12).color(t.muted));
        }
        col = col.push(container(intro).padding([24, 4]));
    }
    for (i, item) in s.items.iter().enumerate() {
        let element: Element<'a, Message> = match item {
            // the feedback of the self-review and the words of a stop hook are for the model
            // only — they draw nothing
            Item::ReviewFeedback(_) | Item::StopHook(_) => Space::new().into(),
            Item::User { text: text_, .. } => {
                let bubble = row![Space::new().width(Length::Fill), container(text(text_.as_str()).size(15)).padding([9, 15]).max_width(560).style(theme::bubble(t, tint))];
                if !s.running && s.turn_allowed(i) {
                    // `Fork` and `Revert` under one of your messages (`turn-actions` of the page)
                    let acts = row![
                        Space::new().width(Length::Fill),
                        button(text("Fork").size(12)).on_press(Message::Code(CodeMsg::Sess(SessMsg::TurnFork(i)))).padding([2, 8]).style(theme::ghost_button(t)),
                        button(text("Revert").size(12)).on_press(Message::Code(CodeMsg::Sess(SessMsg::TurnRevert(i)))).padding([2, 8]).style(theme::ghost_button(t)),
                    ]
                    .spacing(4);
                    column![bubble, acts].spacing(2).into()
                } else {
                    bubble.into()
                }
            }
            Item::Assistant { text: body, md: content } => {
                if body.is_empty() {
                    if s.running && i + 1 == s.items.len() { text("…").size(15).color(t.muted).into() } else { Space::new().height(0).into() }
                } else {
                    md::view(content, t, theme)
                }
            }
            Item::Call(c) => call_card(t, c),
            Item::Note(n) => text(n.as_str()).size(13).color(t.muted).into(),
        };
        col = col.push(element);
    }
    for notice in side_views::choices_view(s, t) {
        col = col.push(notice);
    }
    // the same column as the header and the composer
    let column_ = container(col).padding(Padding { top: 8.0, right: 24.0, bottom: 16.0, left: 24.0 }).max_width(868).width(Length::Fill);
    scrollable(container(column_).center_x(Length::Fill).width(Length::Fill)).anchor_bottom().height(Length::Fill).into()
}

pub fn page<'a>(
    s: &'a CodeState,
    t: Tokens,
    theme: &iced::Theme,
    can_send: bool,
    recent: &'a [String],
    tint: bool,
    turbo: bool,
) -> Element<'a, Message> {
    let Some(w) = &s.workspace else {
        let input = text_input("/path/to/your/project", &s.path_input)
            .on_input(|v| Message::Code(CodeMsg::PathInput(v)))
            .on_submit(Message::Code(CodeMsg::Open))
            .size(14)
            .padding([8, 12])
            .style(move |_, status| text_input::Style {
                background: iced::Background::Color(t.surface_2),
                border: iced::Border { radius: 10.0.into(), width: 1.0, color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator } },
                icon: t.muted,
                placeholder: t.muted,
                value: t.fg,
                selection: t.primary_soft,
            });
        let mut card = column![
            text("Code").size(26).font(bold()),
            text("The agent reads, searches and changes the files of one folder, and can run commands there. Edits and commands ask you first, unless you let it.").size(14).color(t.fg_soft),
            row![
                input,
                button(text("Open").size(14)).on_press(Message::Code(CodeMsg::Open)).padding([8, 18]).style(theme::primary_button(t)),
                button(text("Choose…").size(14)).on_press(Message::Code(CodeMsg::Pick)).padding([8, 18]).style(theme::secondary_button(t)),
            ]
            .spacing(10)
            .align_y(Alignment::Center),
        ]
        .spacing(12)
        .max_width(560);
        if !recent.is_empty() {
            card = card.push(text("Recent").size(12).color(t.muted));
            for f in recent {
                let name = std::path::Path::new(f).file_name().map_or(f.clone(), |n| n.to_string_lossy().into_owned());
                card = card.push(
                    button(row![text(name).size(14).font(bold()), text(f.as_str()).size(12).font(Font::MONOSPACE).color(t.muted).wrapping(text::Wrapping::None)].spacing(10).align_y(Alignment::Center))
                        .on_press(Message::Code(CodeMsg::OpenPath(f.clone())))
                        .width(Length::Fill)
                        .padding([7, 12])
                        .style(theme::secondary_button(t)),
                );
            }
        }
        if let Some(e) = &s.error {
            card = card.push(text(e.as_str()).size(13).color(t.danger));
        }
        return container(card).width(Length::Fill).height(Length::Fill).center_x(Length::Fill).center_y(Length::Fill).padding(24).into();
    };

    let mode = segmented(t, vec![("Plan", s.mode == Mode::Plan, Message::Code(CodeMsg::SetMode(Mode::Plan))), ("Ask", s.mode == Mode::Ask, Message::Code(CodeMsg::SetMode(Mode::Ask))), ("Edit freely", s.mode == Mode::AutoEdit, Message::Code(CodeMsg::SetMode(Mode::AutoEdit))), ("Everything", s.mode == Mode::Auto, Message::Code(CodeMsg::SetMode(Mode::Auto)))]);
    let branch = w.branch.clone().unwrap_or_default();
    let header = row![
        container(column![row![text(w.name.as_str()).size(15).font(bold()), text(branch).size(12).color(t.muted).font(Font::MONOSPACE), text(side_views::meter_text(s).unwrap_or_default()).size(12).color(t.muted).wrapping(text::Wrapping::None)].spacing(8).align_y(Alignment::Center), text(w.root.as_str()).size(12).color(t.muted).font(Font::MONOSPACE).wrapping(text::Wrapping::None)].spacing(2)).clip(true).width(Length::Fill),
        mode,
        button(text(if s.browser.running { "Browser ●" } else { "Browser" }).size(13).wrapping(text::Wrapping::None))
            .on_press(Message::Code(CodeMsg::Br(BrMsg::Toggle)))
            .padding([5, 14])
            .style(pick_style(t, s.browser.open, false)),
        button(text("New session").size(13).wrapping(text::Wrapping::None)).on_press_maybe((!s.running).then_some(Message::Code(CodeMsg::NewSession))).padding([5, 14]).style(theme::secondary_button(t)),
        button(text("Other folder").size(13).wrapping(text::Wrapping::None)).on_press_maybe((!s.running).then_some(Message::Code(CodeMsg::Close))).padding([5, 14]).style(theme::secondary_button(t)),
    ]
    .spacing(10)
    .align_y(Alignment::Center)
    // with the Browser panel open the column is narrow: what does not fit goes to the next line
    .wrap()
    .vertical_spacing(6);
    let header = container(container(header).max_width(820).width(Length::Fill)).center_x(Length::Fill).padding([10, 24]);

    let mut bar = column![].spacing(6);
    if !s.todos.is_empty() {
        let done = s.todos.iter().filter(|t| t.status == TodoStatus::Completed).count();
        let mut list = column![text(format!("Tasks · {done} of {} done", s.todos.len())).size(12).color(t.muted)].spacing(3);
        for todo in &s.todos {
            let (mark, color) = match todo.status {
                TodoStatus::Pending => ("[ ]", t.fg_soft),
                TodoStatus::InProgress => ("[~]", t.primary),
                TodoStatus::Completed => ("[x]", t.muted),
            };
            list = list.push(row![text(mark).size(12).font(Font::MONOSPACE).color(color), text(todo.content.as_str()).size(13).color(color)].spacing(8));
        }
        bar = bar.push(container(list).padding([8, 12]).width(Length::Fill).style(theme::inset(t)));
    }
    if let Some(e) = &s.error {
        bar = bar.push(text(e.as_str()).size(13).color(t.danger));
    }
    if s.mode == Mode::Plan && !s.running && matches!(s.items.last(), Some(Item::Assistant { text, .. }) if !text.trim().is_empty()) {
        bar = bar.push(
            row![
                text("The plan is ready. Nothing was changed.").size(13).color(t.fg_soft).width(Length::Fill),
                button(text("Carry out the plan").size(12)).on_press_maybe(can_send.then_some(Message::Code(CodeMsg::ExecutePlan))).padding([4, 14]).style(theme::primary_button(t)),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
        );
    }
    if !s.changes.is_empty() {
        let files: std::collections::BTreeSet<&str> = s.changes.iter().map(|c| c.path.as_str()).collect();
        let first = s.changes.first().map(|c| c.id);
        bar = bar.push(
            row![
                text(format!("{} change{} in {} file{}", s.changes.len(), if s.changes.len() == 1 { "" } else { "s" }, files.len(), if files.len() == 1 { "" } else { "s" })).size(13).color(t.fg_soft).width(Length::Fill),
                button(text("Undo last").size(12)).on_press_maybe((!s.running).then_some(Message::Code(CodeMsg::Undo(None)))).padding([3, 12]).style(theme::secondary_button(t)),
                button(text("Undo all").size(12)).on_press_maybe((!s.running).then_some(Message::Code(CodeMsg::Undo(first)))).padding([3, 12]).style(theme::secondary_button(t)),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
        );
    }

    let editor = text_editor(&s.editor)
        .placeholder(if s.pending_ask().is_some() { "Type your answer" } else if can_send { "Tell the agent what to do" } else { "Load a model first" })
        .on_action(|a| Message::Code(CodeMsg::Edit(a)))
        .key_binding(|kp: KeyPress| match kp.key.as_ref() {
            keyboard::Key::Named(keyboard::key::Named::Enter) if !kp.modifiers.shift() && matches!(kp.status, Status::Focused { .. }) => Some(Binding::Custom(Message::Code(CodeMsg::Send))),
            _ => Binding::from_key_press(kp),
        })
        .padding(Padding { top: 14.0, right: 16.0, bottom: 6.0, left: 16.0 })
        .size(15)
        .height(Length::Shrink)
        .min_height(44)
        .max_height(160)
        .style(move |_, _| text_editor::Style { background: iced::Background::Color(iced::Color::TRANSPARENT), border: iced::Border { width: 0.0, ..Default::default() }, placeholder: t.muted, value: t.fg, selection: iced::Color { a: 0.35, ..t.primary } });
    let asking = s.pending_ask().is_some();
    let action = if asking {
        button(text("Answer").size(13)).on_press_maybe((!s.editor.text().trim().is_empty()).then_some(Message::Code(CodeMsg::Send))).padding([6, 16]).style(theme::primary_button(t))
    } else if s.running {
        button(text("Stop").size(13)).on_press(Message::Code(CodeMsg::Stop)).padding([6, 16]).style(theme::primary_button(t))
    } else {
        button(text("Send").size(13)).on_press_maybe((can_send && !s.editor.text().trim().is_empty()).then_some(Message::Code(CodeMsg::Send))).padding([6, 16]).style(theme::primary_button(t))
    };
    // the messages that wait for the run to end, each with its ×
    let mut waiting = row![].spacing(6);
    for (i, q) in s.queue.iter().enumerate() {
        let line: String = q.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(60).collect();
        waiting = waiting.push(
            container(row![text(line).size(12).color(t.fg_soft), button(text("×").size(12)).on_press(Message::Code(CodeMsg::QueueDrop(i))).padding([0, 4]).style(theme::ghost_button(t))].spacing(4).align_y(Alignment::Center))
                .padding([3, 10])
                .style(theme::chip(t)),
        );
    }
    let queue_row: Element<'_, Message> = if s.queue.is_empty() { Space::new().height(0).into() } else { container(waiting).padding(Padding { top: 8.0, right: 16.0, bottom: 0.0, left: 16.0 }).into() };
    let hint = if s.running { "The agent is working. Enter queues your message, Esc stops it." } else { "Enter sends, Shift+Enter adds a new line" };
    let composer = container(column![queue_row, editor, row![text(hint).size(12).color(t.muted), Space::new().width(Length::Fill), action].align_y(Alignment::Center).padding(Padding { top: 0.0, right: 10.0, bottom: 8.0, left: 16.0 })])
        .style(theme::card(t, theme::R_XL, false))
        .max_width(820);
    let composer = container(composer).width(Length::Fill).center_x(Length::Fill).padding(Padding { top: 8.0, right: 24.0, bottom: 20.0, left: 24.0 });
    let bar = container(container(bar).max_width(820).width(Length::Fill)).center_x(Length::Fill).padding(Padding { top: 0.0, right: 24.0, bottom: 0.0, left: 24.0 });

    let mut main = column![].width(Length::Fill).height(Length::Fill);
    if let Some(banner) = computer_banner(s, t, turbo) {
        main = main.push(banner);
    }
    let main = main.push(header).push(rule(t)).push(timeline(s, t, theme, tint)).push(bar).push(composer);
    let mut layout = row![main].align_y(Alignment::Start).width(Length::Fill).height(Length::Fill);
    if s.browser.open {
        layout = layout.push(browser_panel(s, t));
    }
    layout.push(side_settings(s, t)).into()
}

/// The primary style when `primary`, else the ghost one (`ghost`) or the secondary one.
fn pick_style(t: Tokens, primary: bool, ghost: bool) -> impl Fn(&iced::Theme, button::Status) -> button::Style {
    move |th, st| match (primary, ghost) {
        (true, _) => theme::primary_button(t)(th, st),
        (false, true) => theme::ghost_button(t)(th, st),
        (false, false) => theme::secondary_button(t)(th, st),
    }
}

/// The width of the browser panel: the narrow one next to the conversation, or the wide one for a page that needs room.
fn browser_width(s: &CodeState) -> f32 {
    if s.br_wide {
        660.0
    } else {
        400.0
    }
}

/// An icon of the panel in one color.
fn br_svg<'a>(bytes: &'static [u8], size: f32, color: iced::Color) -> Element<'a, Message> {
    iced::widget::svg(crate::app::icon(bytes)).width(size).height(size).style(move |_, _| iced::widget::svg::Style { color: Some(color) }).into()
}

/// A round icon button with a tip under it; gray and silent while there is nothing it can do.
fn br_icon_button<'a>(t: Tokens, bytes: &'static [u8], tip: &'static str, msg: Option<BrMsg>, active: bool) -> Element<'a, Message> {
    let color = if msg.is_none() {
        iced::Color { a: 0.45, ..t.muted }
    } else if active {
        t.primary
    } else {
        t.fg_soft
    };
    let face = container(br_svg(bytes, 18.0, color)).center_x(Length::Fill).center_y(Length::Fill);
    let b = button(face).width(34).height(34).padding(0).on_press_maybe(msg.map(|m| Message::Code(CodeMsg::Br(m)))).style(move |th, st| {
        let mut s = theme::ghost_button(t)(th, st);
        s.border.radius = 10.0.into();
        if active && !matches!(st, button::Status::Hovered | button::Status::Pressed) {
            s.background = Some(iced::Background::Color(t.primary_soft));
        }
        s
    });
    iced::widget::tooltip(b, text(tip).size(11), iced::widget::tooltip::Position::Bottom).gap(4).into()
}

/// A pill with an icon and a word (*Comment*, *Close browser*).
fn br_pill<'a>(t: Tokens, bytes: &'static [u8], label: &'a str, msg: Option<BrMsg>, active: bool, danger: bool) -> Element<'a, Message> {
    let color = if msg.is_none() {
        iced::Color { a: 0.45, ..t.muted }
    } else if active {
        t.primary
    } else if danger {
        t.danger
    } else {
        t.fg_soft
    };
    button(row![br_svg(bytes, 13.0, color), text(label).size(12)].spacing(6).align_y(Alignment::Center))
        .padding([5, 12])
        .on_press_maybe(msg.map(|m| Message::Code(CodeMsg::Br(m))))
        .style(move |th, st| {
            let mut s = theme::secondary_button(t)(th, st);
            s.text_color = color;
            if active {
                s.background = Some(iced::Background::Color(t.primary_soft));
                s.border = iced::Border { width: 1.0, color: t.primary, radius: 999.0.into() };
            }
            s
        })
        .into()
}

/// What the agent is doing in the browser right now (a browser tool that runs or waits for you), in the words of its card.
fn br_agent_action(s: &CodeState) -> Option<(bool, String)> {
    s.items.iter().rev().find_map(|it| match it {
        Item::Call(c) if c.name.starts_with("browser_") && matches!(c.state, CallState::Running | CallState::Pending) => {
            let args: Value = serde_json::from_str(&c.args).unwrap_or(Value::Null);
            let (what, more) = crate::browser::tool_label(&c.name, &args)?;
            let line = if more.is_empty() { what } else { format!("{what} {}", crate::computer::slice_units(&more, 48)) };
            Some((matches!(c.state, CallState::Pending), line))
        }
        _ => None,
    })
}

/// The host of an address for the caption under the picture.
fn br_host(url: &str) -> String {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    rest.split(['/', '?', '#']).next().unwrap_or("").trim_start_matches("www.").to_string()
}

/// *Browser* of the Code page (`codeBrowser`): the live page, the address bar, the comments.
fn browser_panel<'a>(s: &'a CodeState, t: Tokens) -> Element<'a, Message> {
    let b = &s.browser;
    let v = b.view();
    let bold = Font { weight: iced::font::Weight::Bold, ..Font::default() };
    let semibold = Font { weight: iced::font::Weight::Semibold, ..Font::default() };
    let panel_width = browser_width(s);
    let running = v.back_enabled;
    let agent = br_agent_action(s);

    // the head: what it is, whether it runs, and the two things you do with the panel itself
    let logo = container(br_svg(ICON_BR_GLOBE, 16.0, t.primary)).padding(7).style(move |_| container::Style {
        background: Some(iced::Background::Color(t.primary_soft)),
        border: iced::border::rounded(10),
        ..Default::default()
    });
    let dot_color = if v.live_dot { t.ok } else { iced::Color { a: 0.6, ..t.muted } };
    let dot = container(Space::new()).width(7).height(7).style(move |_| container::Style { background: Some(iced::Background::Color(dot_color)), border: iced::border::rounded(999), ..Default::default() });
    let title = column![text("Browser").size(14).font(bold), row![dot, text(v.state.clone()).size(11).color(t.muted)].spacing(5).align_y(Alignment::Center)].spacing(1);
    let head = row![
        logo,
        title,
        Space::new().width(Length::Fill),
        br_icon_button(t, ICON_BR_EXPAND_OR_SHRINK[s.br_wide as usize], if s.br_wide { "Narrower" } else { "Wider" }, Some(BrMsg::Wide), false),
        br_icon_button(t, ICON_BR_X, "Hide the panel", Some(BrMsg::Hide), false),
    ]
    .spacing(8)
    .align_y(Alignment::Center);

    // navigation and the address in one pill
    let secure = v.url_input.starts_with("https://");
    let lead = br_svg(if secure { ICON_BR_LOCK } else { ICON_BR_GLOBE }, 14.0, if secure { t.ok } else { t.muted });
    let input = {
        let input = text_input("Search or enter an address, for example localhost:3000", &v.url_input)
            .id("br-address")
            .size(13)
            .padding([6, 4])
            .on_submit(Message::Code(CodeMsg::Br(BrMsg::Submit)))
            .style(move |th, st| {
                let mut s = iced::widget::text_input::default(th, st);
                s.background = iced::Background::Color(iced::Color::TRANSPARENT);
                s.border = iced::Border { width: 0.0, color: iced::Color::TRANSPARENT, radius: 0.0.into() };
                s.value = t.fg;
                s.placeholder = t.muted;
                s.selection = t.primary_soft;
                s
            });
        if v.url_enabled { input.on_input(|x| Message::Code(CodeMsg::Br(BrMsg::UrlEdit(x)))) } else { input }
    };
    let go = button(container(br_svg(ICON_BR_ARROW, 14.0, t.on_primary)).center_x(Length::Fill).center_y(Length::Fill))
        .width(26)
        .height(26)
        .padding(0)
        .on_press_maybe(v.url_enabled.then_some(Message::Code(CodeMsg::Br(BrMsg::Submit))))
        .style(move |th, st| {
            let mut s = theme::primary_button(t)(th, st);
            s.border.radius = 999.0.into();
            s
        });
    let focused = b.url_focused;
    let pill = container(row![lead, input, go].spacing(6).align_y(Alignment::Center))
        .padding(iced::Padding { top: 2.0, right: 3.0, bottom: 2.0, left: 10.0 })
        .width(Length::Fill)
        .style(move |_| container::Style {
            background: Some(iced::Background::Color(t.fill)),
            border: iced::Border { width: 1.0, color: if focused { t.primary } else { iced::Color::TRANSPARENT }, radius: 999.0.into() },
            ..Default::default()
        });
    let bar = row![
        br_icon_button(t, ICON_BR_BACK, "Back", running.then_some(BrMsg::Back), false),
        br_icon_button(t, ICON_BR_FORWARD, "Forward", running.then_some(BrMsg::Forward), false),
        br_icon_button(t, ICON_BR_RELOAD, "Reload", running.then_some(BrMsg::Reload), false),
        pill,
    ]
    .spacing(2)
    .align_y(Alignment::Center);

    // what the agent is doing
    let activity: Option<Element<'a, Message>> = agent.as_ref().map(|(waits, line)| {
        let colour = if *waits { t.danger } else { t.primary };
        let words = if *waits { format!("Waiting for you: {line}") } else { format!("The agent: {line}") };
        container(row![br_svg(ICON_BR_BOT, 14.0, colour), text(words).size(12).color(t.fg).wrapping(text::Wrapping::None)].spacing(8).align_y(Alignment::Center))
            .padding([6, 12])
            .width(Length::Fill)
            .style(move |_| container::Style {
                background: Some(iced::Background::Color(t.primary_soft)),
                border: iced::Border { width: 1.0, color: iced::Color { a: 0.5, ..colour }, radius: 999.0.into() },
                ..Default::default()
            })
            .into()
    });

    // the picture of the page, in a card
    let width = panel_width - 28.0;
    let busy = agent.is_some();
    let stage: Element<'a, Message> = match (&s.br_handle, v.show_image) {
        (Some(handle), true) => {
            let inner = width - 4.0;
            let height = inner * (b.view.1 / b.view.0) as f32;
            let size = iced::Size::new(inner, height);
            let picture = iced::widget::image(handle.clone()).width(inner).height(height);
            let layered: Element<'a, Message> = match b.pin(f64::from(inner)) {
                Some([x, y, w, h]) => {
                    let frame = container(Space::new()).width(w as f32).height(h as f32).style(move |_| container::Style {
                        background: Some(iced::Background::Color(iced::Color { a: 0.14, ..t.primary })),
                        border: iced::Border { width: 2.0, color: t.primary, radius: 3.0.into() },
                        ..Default::default()
                    });
                    iced::widget::stack![picture, container(frame).padding(iced::Padding { top: y as f32, right: 0.0, bottom: 0.0, left: x as f32 })].into()
                }
                None => picture.into(),
            };
            let layered: Element<'a, Message> = if v.commenting {
                let chip = container(text("Comment mode: click an element").size(11).color(t.on_primary))
                    .padding([3, 10])
                    .style(move |_| container::Style { background: Some(iced::Background::Color(t.danger)), border: iced::border::rounded(999), ..Default::default() });
                iced::widget::stack![layered, container(chip).padding(10).width(Length::Fill).height(Length::Fill).align_x(Alignment::Center).align_y(iced::alignment::Vertical::Bottom)].into()
            } else {
                layered
            };
            let area = iced::widget::mouse_area(layered)
                .on_move(move |p| Message::Code(CodeMsg::Br(BrMsg::Move(p, size))))
                .on_press(Message::Code(CodeMsg::Br(BrMsg::Click)))
                .on_double_click(Message::Code(CodeMsg::Br(BrMsg::Double)))
                .on_scroll(|d| Message::Code(CodeMsg::Br(BrMsg::Scroll(d))))
                .interaction(if v.commenting { iced::mouse::Interaction::Crosshair } else { iced::mouse::Interaction::Pointer });
            let ring = if s.br_focus {
                t.primary
            } else if v.commenting {
                t.danger
            } else if busy {
                iced::Color { a: 0.7, ..t.primary }
            } else {
                t.separator
            };
            let ring_width = if s.br_focus || v.commenting || busy { 2.0 } else { 1.0 };
            let mut card = column![container(area).style(move |_| container::Style { border: iced::Border { radius: 8.0.into(), ..Default::default() }, ..Default::default() }).clip(true)];
            let host = br_host(&b.url);
            if !b.title.is_empty() || !host.is_empty() {
                let mut caption = row![].spacing(8).align_y(Alignment::Center);
                caption = caption.push(text(crate::computer::slice_units(&b.title, 46)).size(12).font(semibold).wrapping(text::Wrapping::None));
                if !host.is_empty() {
                    caption = caption.push(text(host).size(11).color(t.muted).wrapping(text::Wrapping::None));
                }
                caption = caption.push(Space::new().width(Length::Fill));
                if s.br_focus {
                    caption = caption.push(text("you are typing in the page").size(11).color(t.primary));
                }
                card = card.push(container(caption).padding([7, 10]).width(Length::Fill));
            }
            container(card.spacing(0))
                .width(Length::Fill)
                .padding(1)
                .clip(true)
                .style(move |_| container::Style {
                    background: Some(iced::Background::Color(t.bg)),
                    border: iced::Border { width: ring_width, color: ring, radius: 10.0.into() },
                    ..Default::default()
                })
                .into()
        }
        _ => {
            let (icon, headline) = if !running { (ICON_BR_GLOBE, "") } else { (ICON_BR_POINTER, "") };
            let _ = headline;
            let mut empty = column![br_svg(icon, 34.0, iced::Color { a: 0.55, ..t.muted }), text(v.empty.text.clone()).size(13).color(t.muted).align_x(iced::alignment::Horizontal::Center)]
                .spacing(12)
                .align_x(Alignment::Center);
            if let Some((label, primary)) = v.empty.button {
                empty = empty.push(button(text(label).size(12)).on_press(Message::Code(CodeMsg::Br(BrMsg::OpenSettings))).padding([5, 14]).style(pick_style(t, primary, false)));
            }
            if running && b.url.is_empty() && v.url_enabled {
                let mut chips = row![].spacing(6);
                for quick in ["localhost:3000", "localhost:5173", "localhost:8080"] {
                    chips = chips.push(button(text(quick).size(11)).on_press(Message::Code(CodeMsg::Br(BrMsg::Quick(quick.to_string())))).padding([3, 10]).style(theme::secondary_button(t)));
                }
                empty = empty.push(chips);
            }
            container(empty)
                .padding(24)
                .width(Length::Fill)
                .center_x(Length::Fill)
                .style(move |_| container::Style {
                    background: Some(iced::Background::Color(t.bg)),
                    border: iced::Border { width: 1.0, color: t.separator, radius: 10.0.into() },
                    ..Default::default()
                })
                .into()
        }
    };

    let mut body = column![head, bar].spacing(12);
    if let Some(a) = activity {
        body = body.push(a);
    }
    body = body.push(stage);
    body = body.push(
        row![
            br_pill(t, ICON_BR_COMMENT, "Comment", running.then_some(BrMsg::Comment), v.commenting, false),
            br_pill(t, ICON_BR_POWER, "Close browser", running.then_some(BrMsg::CloseAsk), false, true),
            Space::new().width(Length::Fill),
            text(if v.show_image { format!("page {}×{}", b.view.0 as i64, b.view.1 as i64) } else { String::new() }).size(11).color(t.muted),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    );
    if s.br_confirm_close {
        body = body.push(
            container(
                column![
                    text("Close the browser? Pages that are open are lost; logins stay in its profile.").size(12),
                    row![
                        button(text("Close").size(12)).on_press(Message::Code(CodeMsg::Br(BrMsg::CloseOk))).padding([4, 12]).style(theme::primary_button(t)),
                        button(text("Cancel").size(12)).on_press(Message::Code(CodeMsg::Br(BrMsg::CloseCancel))).padding([4, 12]).style(theme::ghost_button(t)),
                    ]
                    .spacing(6),
                ]
                .spacing(8),
            )
            .padding(12)
            .width(Length::Fill)
            .style(move |_| container::Style { background: Some(iced::Background::Color(t.fill)), border: iced::border::rounded(10), ..Default::default() }),
        );
    }
    if b.draft.is_some() {
        let note = text_editor(&s.br_note)
            .placeholder("What should be changed or checked here?")
            .on_action(|a| Message::Code(CodeMsg::Br(BrMsg::NoteEdit(a))))
            .padding(8)
            .size(12)
            .height(Length::Fixed(76.0))
            .style(crate::skills::editor_style(t));
        body = body.push(
            container(
                column![
                    text(b.note_parts().map(|(h, _)| h).unwrap_or_default()).size(12).font(Font::MONOSPACE),
                    text(b.note_parts().map(|(_, x)| x).unwrap_or_default()).size(11).color(t.muted),
                    note,
                    row![
                        Space::new().width(Length::Fill),
                        button(text("Cancel").size(12)).on_press(Message::Code(CodeMsg::Br(BrMsg::NoteCancel))).padding([4, 12]).style(theme::ghost_button(t)),
                        button(text("Add comment").size(12)).on_press(Message::Code(CodeMsg::Br(BrMsg::NoteAdd))).padding([4, 12]).style(theme::primary_button(t)),
                    ]
                    .spacing(6),
                ]
                .spacing(6),
            )
            .padding(12)
            .width(Length::Fill)
            .style(move |_| container::Style { border: iced::Border { width: 1.0, color: t.danger, radius: 10.0.into() }, ..Default::default() }),
        );
    }
    if !b.comments.is_empty() {
        let mut list = column![text(format!("Comments for the agent ({})", b.comments.len())).size(12).font(bold)].spacing(6);
        for (i, cm) in b.comments.iter().enumerate() {
            let name = if cm.name.is_empty() { String::new() } else { format!(" \"{}\"", crate::computer::slice_units(&cm.name, 40)) };
            list = list.push(
                container(
                    row![
                        column![text(format!("{}{}", cm.role, name)).size(11).font(Font::MONOSPACE).color(t.muted), text(cm.note.clone()).size(12)].spacing(2).width(Length::Fill),
                        br_icon_button(t, ICON_BR_X, "Remove", Some(BrMsg::Del(i)), false),
                    ]
                    .spacing(6)
                    .align_y(Alignment::Center),
                )
                .padding([6, 10])
                .width(Length::Fill)
                .style(move |_| container::Style { background: Some(iced::Background::Color(t.fill)), border: iced::border::rounded(10), ..Default::default() }),
            );
        }
        list = list.push(
            row![
                button(text("Send to the agent").size(12)).on_press(Message::Code(CodeMsg::Br(BrMsg::SendComments))).padding([4, 12]).style(theme::primary_button(t)),
                text("They are also added to your next message.").size(11).color(t.muted),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
        );
        body = body.push(list);
    }
    if !v.foot.is_empty() {
        body = body.push(text(v.foot.clone()).size(11).color(t.muted));
    }
    container(scrollable(body.padding(14).spacing(12)))
        .width(Length::Fixed(panel_width))
        .height(Length::Fill)
        .style(move |_| container::Style { background: Some(iced::Background::Color(t.surface)), border: iced::Border { width: 1.0, color: t.separator, ..Default::default() }, ..Default::default() })
        .into()
}

const ICON_BR_GLOBE: &[u8] = include_bytes!("../icons/globe.svg");
const ICON_BR_X: &[u8] = include_bytes!("../icons/x.svg");
const ICON_BR_LOCK: &[u8] = include_bytes!("../icons/lock.svg");
const ICON_BR_ARROW: &[u8] = include_bytes!("../icons/arrow-right.svg");
const ICON_BR_BACK: &[u8] = include_bytes!("../icons/chevron-left.svg");
const ICON_BR_FORWARD: &[u8] = include_bytes!("../icons/chevron-right.svg");
const ICON_BR_RELOAD: &[u8] = include_bytes!("../icons/reload.svg");
const ICON_BR_COMMENT: &[u8] = include_bytes!("../icons/comment.svg");
const ICON_BR_POWER: &[u8] = include_bytes!("../icons/power.svg");
const ICON_BR_BOT: &[u8] = include_bytes!("../icons/bot.svg");
const ICON_BR_POINTER: &[u8] = include_bytes!("../icons/pointer.svg");
const ICON_BR_EXPAND_OR_SHRINK: [&[u8]; 2] = [include_bytes!("../icons/expand.svg"), include_bytes!("../icons/shrink.svg")];

/// The banner on top while computer use is on (`renderComputerBanner`): what the agent is doing
/// or that it is allowed to, how many actions are used, and the STOP button.
fn computer_banner<'a>(s: &'a CodeState, t: Tokens, turbo: bool) -> Option<Element<'a, Message>> {
    if !s.computer_on() {
        return None;
    }
    let bold = Font { weight: iced::font::Weight::Bold, ..Font::default() };
    let st = s.computer.status.as_ref()?;
    let line: Element<'a, Message> = if s.computer.busy.is_empty() {
        let mut r = row![text("Computer use is ON.").size(13).font(bold), text(" The agent can see your screen and use your mouse and keyboard").size(13)].align_y(Alignment::Center);
        r = if turbo { r.push(text(" WITHOUT ASKING (turbo)").size(13).font(bold)) } else { r.push(text(", and asks you before every action").size(13)) };
        r.push(text(".").size(13)).into()
    } else {
        text(s.computer.busy.as_str()).size(13).font(bold).into()
    };
    let used = if st.steps_used > 0 { format!(" {} of {} actions used in this task.", st.steps_used, st.max_steps) } else { String::new() };
    let keys = text(format!("Emergency stop: press Esc three times, or Ctrl+Alt+Shift+Esc anywhere.{used}")).size(11);
    let colour = if turbo { t.danger } else { t.primary };
    let dot = container(Space::new()).width(9).height(9).style(move |_| container::Style { background: Some(iced::Background::Color(colour)), border: iced::border::rounded(999), ..Default::default() });
    let stop = button(text("STOP").size(13).font(bold)).on_press(Message::Code(CodeMsg::ComputerStop("You pressed STOP.".into()))).padding([5, 16]).style(theme::primary_button(t));
    Some(
        container(row![dot, column![line, keys].spacing(2).width(Length::Fill), stop].spacing(10).align_y(Alignment::Center))
            .padding([8, 24])
            .width(Length::Fill)
            .style(move |_| container::Style { background: Some(iced::Background::Color(t.primary_soft)), border: iced::Border { width: 1.0, color: colour, radius: 0.0.into() }, ..Default::default() })
            .into(),
    )
}

/// What a computer card shows under its words (`computerBodyHtml`): the screen after the action,
/// or the screen the action refers to with its marks drawn on it.
fn computer_body<'a>(t: Tokens, c: &'a Call, args: &Value) -> Option<Element<'a, Message>> {
    let cu = c.cu.as_ref()?;
    let Some(shot) = cu.image.as_ref().or(cu.shot.as_ref()) else {
        return cu.gone.then(|| text("(the screenshot is not kept after the session was saved)").size(12).color(t.muted).into());
    };
    let picture = iced::widget::image(shot.handle.clone()).width(Length::Fill);
    let marks = if cu.image.is_some() { String::new() } else { crate::computer::marks_svg(shot.width, shot.height, &crate::computer::computer_marks(&c.name, args)) };
    Some(if marks.is_empty() {
        picture.into()
    } else {
        iced::widget::stack![picture, iced::widget::svg(iced::widget::svg::Handle::from_memory(marks.into_bytes())).width(Length::Fill)].into()
    })
}

/// *Background jobs* of the side (`jobsHtml`): a dot, the command, *Output*, *Stop* while it runs and
/// its state; the output under it while the box is open.
fn jobs_block<'a>(s: &'a CodeState, t: Tokens) -> Element<'a, Message> {
    let semibold = Font { weight: iced::font::Weight::Semibold, ..Font::default() };
    let mut head = row![text("Background jobs").size(14).font(semibold)].spacing(8).align_y(Alignment::Center);
    if !s.jobs.is_empty() {
        head = head.push(text(s.jobs.len().to_string()).size(12).color(t.muted));
    }
    let mut block = column![head].spacing(8);
    if s.jobs.is_empty() {
        block = block.push(text(crate::jobs::EMPTY).size(12).color(t.muted));
    }
    for j in &s.jobs {
        let running = j.status == "running";
        let dot_colour = if running { t.ok } else { t.muted };
        let dot = container(Space::new()).width(7).height(7).style(move |_| container::Style {
            background: Some(iced::Background::Color(dot_colour)),
            border: iced::border::rounded(999),
            ..Default::default()
        });
        let mut actions = row![button(text("Output").size(12)).on_press(Message::Code(CodeMsg::JobOutput(j.id.clone()))).padding([2, 8]).style(theme::ghost_button(t))].spacing(4).align_y(Alignment::Center);
        if running {
            actions = actions.push(button(text("Stop").size(12)).on_press(Message::Code(CodeMsg::JobKill(j.id.clone()))).padding([2, 8]).style(theme::ghost_button(t)));
        }
        actions = actions.push(text(j.status.clone()).size(12).color(t.muted));
        let mut entry = column![
            row![dot, text(j.command.clone()).size(12).font(Font::MONOSPACE).wrapping(text::Wrapping::None)].spacing(6).align_y(Alignment::Center),
            actions,
        ]
        .spacing(4);
        if j.open {
            entry = entry.push(
                container(text(crate::jobs::shown_output(j).to_string()).size(11).font(Font::MONOSPACE))
                    .padding([6, 8])
                    .width(Length::Fill)
                    .style(move |_| container::Style { border: iced::Border { width: 1.0, color: t.separator, radius: 6.0.into() }, ..Default::default() }),
            );
        }
        block = block.push(entry);
    }
    block.into()
}

/// The side of the Code page — the blocks the rows build so far: *Specs* (row 80, the
/// `Specs` block of `renderCodeSide` with `specsHtml`) and *Settings* (row 81, the two
/// switches of `code.autoCheck` and `code.autoReview`), with their words of the page. The
/// others it sits among (tasks, changed files, jobs, sessions, memory, permissions and the
/// style) come with their own rows.
fn side_settings<'a>(s: &'a CodeState, t: Tokens) -> Element<'a, Message> {
    let semibold = Font { weight: iced::font::Weight::Semibold, ..Font::default() };
    let mut specs = column![].spacing(8);
    let mut head = row![text("Specs").size(14).font(semibold)].spacing(8).align_y(Alignment::Center);
    if !s.specs.is_empty() {
        head = head.push(text(s.specs.len().to_string()).size(12).color(t.muted));
    }
    specs = specs.push(head);
    if s.specs.is_empty() {
        specs = specs.push(text(crate::specs::PITCH).size(12).color(t.muted));
    }
    for sp in &s.specs {
        let mut files = row![].spacing(6);
        for (on, label, file) in [(sp.requirements, "Req", "requirements.md"), (sp.design, "Design", "design.md"), (sp.tasks_file, "Tasks", "tasks.md")] {
            let title = crate::specs::chip_title(file, on);
            let chip = container(text(label).size(11).color(if on { t.fg } else { t.muted })).padding([2, 8]).style(move |_| container::Style {
                background: on.then_some(iced::Background::Color(t.fill)),
                border: iced::Border { width: 1.0, color: t.separator, radius: 999.0.into() },
                ..Default::default()
            });
            files = files.push(iced::widget::tooltip(chip, text(title).size(11), iced::widget::tooltip::Position::Bottom));
        }
        let mut boxed = column![
            row![text(sp.name.as_str()).size(13).font(semibold), Space::new().width(Length::Fill), text(format!("{}/{}", sp.done, sp.total)).size(12).color(t.muted)]
                .align_y(Alignment::Center),
            files,
        ]
        .spacing(6);
        for task in &sp.tasks {
            let (name, index) = (sp.name.clone(), task.index);
            let tick = checkbox(task.done)
                .label(task.text.as_str())
                .size(16)
                .text_size(12)
                .on_toggle(move |done| Message::Code(CodeMsg::SpecTick { name: name.clone(), index, done }));
            boxed = boxed.push(container(tick).padding(Padding { top: 0.0, right: 0.0, bottom: 0.0, left: crate::specs::task_indent(task.depth) }));
        }
        if let Some(next) = sp.tasks.iter().find(|task| !task.done) {
            let (name, index) = (sp.name.clone(), next.index);
            let run = button(text("Run next task").size(12))
                .on_press_maybe((!s.running).then_some(Message::Code(CodeMsg::SpecRunTask { name, index })))
                .padding([4, 10])
                .style(theme::secondary_button(t));
            boxed = boxed.push(iced::widget::tooltip(run, text("The agent does this task and ticks it").size(11), iced::widget::tooltip::Position::Bottom));
        } else if sp.total > 0 {
            boxed = boxed.push(text("Every task is done.").size(12).color(t.muted));
        }
        specs = specs.push(
            container(boxed).padding(10).width(Length::Fill).style(move |_| container::Style {
                border: iced::Border { width: 1.0, color: t.separator, radius: 8.0.into() },
                ..Default::default()
            }),
        );
    }
    let check_label = "Check my edits automatically".to_string();
    let review_label = "Review my work before it counts as done".to_string();
    let block = column![
        text("Settings").size(14).font(Font { weight: iced::font::Weight::Semibold, ..Font::default() }),
        checkbox(s.auto_check)
            .label(check_label)
            .size(16)
            .text_size(13)
            .on_toggle(|value| Message::Code(CodeMsg::ToggleAutoCheck(value))),
        text("After every edit the agent runs a quick syntax or build check (python, node, cargo check, tsc, go vet, JSON) and gets the first errors with the result. Slower, but it catches mistakes at once.")
            .size(12)
            .color(t.muted),
        checkbox(s.auto_review)
            .label(review_label)
            .size(16)
            .text_size(13)
            .on_toggle(|value| Message::Code(CodeMsg::ToggleAutoReview(value))),
        text("When a run changed files, the agent reads its own diff once more together with the findings of a fixed rule set and the diagnostics, and fixes what is wrong (two rounds at most). It costs one extra step per run. Switch it on with /autoreview on.")
            .size(12)
            .color(t.muted),
        checkbox(s.learn)
            .label("Learn from my sessions")
            .size(16)
            .text_size(13)
            .on_toggle(|value| Message::Code(CodeMsg::LearnToggle(value))),
        text(crate::learn::SIDE_TEXT).size(12).color(t.muted),
        side_views::style_row(s, t),
        row![
            button(text("Skills helped").size(12)).on_press(Message::Code(CodeMsg::LearnThumb("up"))).padding([4, 8]).style(theme::ghost_button(t)),
            button(text("Skills did not help").size(12)).on_press(Message::Code(CodeMsg::LearnThumb("down"))).padding([4, 8]).style(theme::ghost_button(t)),
        ]
        .spacing(6),
    ]
    .spacing(10);
    let side = column![side_views::tasks_block(s, t), side_views::changes_block(s, t), specs, jobs_block(s, t), side_views::sessions_block(s, t), side_views::memory_block(s, t), side_views::permissions_block(s, t), block].spacing(22);
    container(scrollable(container(side).padding(Padding { top: 0.0, right: 12.0, bottom: 0.0, left: 0.0 })))
        .width(Length::Fixed(272.0))
        .padding([18.0, 16.0])
        .style(move |_| container::Style {
            background: Some(iced::Background::Color(t.surface)),
            border: iced::Border { width: 1.0, color: t.separator, ..Default::default() },
            ..Default::default()
        })
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `fireHook` makes of the server's answer, with the page's fallbacks.
    #[test]
    fn the_hook_answer_reads_what_the_page_reads() {
        let full = fire_answer(&json!({ "blocked": true, "reason": "run the tests", "context": "[x]" }));
        assert_eq!(
            full,
            HookFire { blocked: true, reason: "run the tests".into(), context: "[x]".into() }
        );
        // every field the page falls back on when the server says nothing of it
        assert_eq!(fire_answer(&json!({})), HookFire::default());
        assert_eq!(fire_answer(&json!({ "blocked": 0, "reason": 5 })), HookFire::default());
    }

    /// The chain of `starting_state`: what waits, what runs, and the words under a card the
    /// harmless table let through without a question — the switch of `/autoapprove` decides it.
    #[test]
    fn the_switch_decides_what_runs_without_a_question() {
        let mut code = state(Mode::Ask);
        let server = ToolKind::Server;

        // an unknown tool fails whatever the switch says
        let (start, note) = code.starting_state(false, server.clone(), Permission::Command, "nope", "{}", false);
        assert!(matches!(start, CallState::Done { ok: false, .. }));
        assert!(note.is_none());

        // a question waits in every case
        code.auto_approve = true;
        let (start, note) =
            code.starting_state(true, ToolKind::Ask, Permission::Read, "ask_user", "{\"question\":\"?\"}", false);
        assert!(matches!(start, CallState::Pending));
        assert!(note.is_none());

        // the switch off: a harmless command waits like every other one
        code.auto_approve = false;
        let (start, note) = code.starting_state(true, server.clone(), Permission::Command, "bash", "{\"command\":\"ls\"}", false);
        assert!(matches!(start, CallState::Pending));
        assert!(note.is_none());

        // the switch on: the table lets it through and its reason stands under the card
        code.auto_approve = true;
        let (start, note) = code.starting_state(true, server.clone(), Permission::Command, "bash", "{\"command\":\"ls -la\"}", false);
        assert!(matches!(start, CallState::Running));
        assert_eq!(note.as_deref(), Some("reads, builds or tests only"));

        // … and keeps a command that changes things waiting
        let (start, note) = code.starting_state(true, server.clone(), Permission::Command, "bash", "{\"command\":\"git push\"}", false);
        assert!(matches!(start, CallState::Pending));
        assert!(note.is_none());

        // an edit is never approved by the table, whatever it holds
        let (start, _) = code.starting_state(true, server.clone(), Permission::Edit, "edit_file", "{\"path\":\"a\"}", false);
        assert!(matches!(start, CallState::Pending));

        // plan mode refuses before the switch is asked
        let mut plan = state(Mode::Plan);
        plan.auto_approve = true;
        let (start, _) = plan.starting_state(true, server.clone(), Permission::Edit, "edit_file", "{\"path\":\"a\"}", false);
        assert!(matches!(start, CallState::Done { ok: false, .. }));

        // a mode that allows something runs it without the table, and no words stand under it
        let (start, note) = code.starting_state(true, server, Permission::Read, "read_file", "{\"path\":\"a\"}", false);
        assert!(matches!(start, CallState::Running));
        assert!(note.is_none());
    }

    /// The category of a call and the words of a shell command, as `permissionInfo` reads them.
    #[test]
    fn the_category_and_the_subject_are_the_ones_of_the_page() {
        assert_eq!(CodeState::category_of(Permission::Command, &ToolKind::Server), "bash");
        assert_eq!(CodeState::category_of(Permission::Command, &ToolKind::Mcp { server: "s".into(), tool: "t".into() }), "mcp");
        assert_eq!(CodeState::category_of(Permission::Edit, &ToolKind::Server), "edit");
        assert_eq!(CodeState::category_of(Permission::Other, &ToolKind::Server), "read");
        let subject = |permission: &str, name: &str, args: serde_json::Value| {
            crate::commands::permission_info(permission, None, name, &args).1
        };
        assert_eq!(subject("command", "bash", serde_json::json!({ "command": "ls -la" })), ["ls -la"]);
        assert_eq!(
            subject("command", "bash", serde_json::json!({ "pattern": "todo", "path": "." })),
            ["bash todo ."],
            "without a command the words of the arguments follow the name, in the order they were written (the page walks its insertion order)"
        );
        assert_eq!(
            subject("edit", "edit_file", serde_json::json!({ "path": "a" })),
            ["a"],
            "an edit is judged by its path, as permissionInfo reads it"
        );
        assert_eq!(
            subject(
                "edit",
                "multi_edit",
                serde_json::json!({ "patch": "--- a/x.rs\n+++ b/x.rs\n@@ -1 +1 @@\n-1\n+2\n" })
            ),
            ["x.rs"],
            "and a patch brings every file it names"
        );
    }

    fn tools() -> Vec<Tool> {
        let mut list: Vec<Tool> = ["read_file:read", "grep:read", "edit_file:edit", "write_file:edit", "bash:command", "web_fetch:network"]
            .iter()
            .map(|s| {
                let (n, p) = s.split_once(':').unwrap();
                Tool::server(n, Permission::parse(p))
            })
            .collect();
        list.extend(client_tools().0);
        list.extend(parse_mcp_tools(&json!({"tools": [{"server": "github", "name": "list_issues", "description": "d"}]})).0);
        list
    }

    fn state(mode: Mode) -> CodeState {
        CodeState {
            workspace: Some(Workspace { root: "/p".into(), name: "p".into(), branch: Some("main".into()) }),
            tools: tools(),
            tools_json: json!([{"type": "function", "function": {"name": "read_file"}}]),
            mode,
            ..Default::default()
        }
    }

    fn env() -> Env<'static> {
        Env {
            base: "http://127.0.0.1:9",
            model: Some("m"),
            params: ChatParams::default(),
            extra: None,
            turbo: false,
            sandbox: None,
            sandbox_available: true,
            usage: None,
            // nothing saves a session in the tests: the write fails and only says so
            session_file: std::path::Path::new(""),
            ctx_max: 8192,
            hooks: HookGates::default(),
            vision: true,
            visible: true,
            models: &[],
        }
    }

    fn say(s: &mut CodeState, text: &str) {
        s.editor = Content::with_text(text);
        let _ = s.update(CodeMsg::Send, &env());
    }

    /// One answer of the model: some text and calls `(id, name, arguments)`, then the end of the stream.
    fn answer(s: &mut CodeState, text: &str, calls: &[(&str, &str, &str)]) {
        if !text.is_empty() {
            let _ = s.update(CodeMsg::Stream(StreamEvent::Delta(text.into())), &env());
        }
        for (i, (id, name, args)) in calls.iter().enumerate() {
            // the id and the name come first, the arguments in pieces
            let _ = s.update(CodeMsg::Stream(StreamEvent::ToolCall { index: i, id: Some((*id).into()), name: Some((*name).into()), arguments: String::new() }), &env());
            for chunk in args.as_bytes().chunks(5) {
                let _ = s.update(CodeMsg::Stream(StreamEvent::ToolCall { index: i, id: None, name: None, arguments: String::from_utf8_lossy(chunk).into() }), &env());
            }
        }
        let _ = s.update(CodeMsg::Stream(StreamEvent::Done), &env());
    }

    fn call<'a>(s: &'a CodeState, id: &str) -> &'a Call {
        s.items.iter().find_map(|i| if let Item::Call(c) = i { (c.id == id).then_some(c) } else { None }).unwrap()
    }

    fn assistants(s: &CodeState) -> usize {
        s.items.iter().filter(|i| matches!(i, Item::Assistant { .. })).count()
    }

    #[test]
    fn what_may_run_alone_depends_on_the_mode() {
        use Permission::*;
        let table = [(Mode::Ask, [true, false, false, false, true]), (Mode::AutoEdit, [true, true, false, false, true]), (Mode::Auto, [true, true, true, true, true])];
        for (mode, want) in table {
            assert_eq!([Read, Edit, Command, Network, Memory].map(|p| mode.allows(p)), want, "{mode:?}");
        }
        assert!(!Mode::AutoEdit.allows(Other) && Mode::Auto.allows(Other), "an unknown kind of tool always asks, except when everything is allowed");
    }

    #[test]
    fn the_tool_list_is_read_for_the_window_and_for_the_model() {
        let (list, for_model) = parse_tools(&json!({"tools": [
            {"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {}}, "permission": "read"},
            {"type": "function", "function": {"name": "bash"}, "permission": "command"},
            {"type": "function", "function": {"name": "odd"}, "permission": "something new"},
            {"nonsense": true}
        ]}));
        assert_eq!(list.iter().map(|t| (t.name.as_str(), t.permission)).collect::<Vec<_>>(), vec![("read_file", Permission::Read), ("bash", Permission::Command), ("odd", Permission::Other)]);
        assert!(for_model[0].get("permission").is_none(), "the model gets the OpenAI shape only");
        assert_eq!(for_model[0]["function"]["name"], "read_file");
        assert_eq!(parse_tools(&json!({})).0.len(), 0);
    }

    #[test]
    fn answers_of_the_server_become_results_and_errors() {
        assert_eq!(tool_result(&json!({"ok": true, "output": "hello"})), Ok("hello".into()));
        assert_eq!(tool_result(&json!({"ok": false, "error": "nope.txt does not exist."})), Err("nope.txt does not exist.".into()));
        assert!(tool_result(&json!({})).is_err());
        assert_eq!(parse_workspace(&json!({"root": "/p", "name": "p", "git_branch": "main"})).unwrap().branch.as_deref(), Some("main"));
        assert_eq!(parse_workspace(&json!({"error": {"message": "not a folder"}})), Err("not a folder".into()));
        assert_eq!(parse_changes(&json!({"changes": [{"id": 3, "path": "a.rs", "kind": "edit"}]})), vec![Change { id: 3, path: "a.rs".into(), kind: "edit".into(), time: 0.0 }]);
    }

    #[test]
    fn a_change_is_shown_before_it_is_allowed() {
        let p = preview("edit_file", &json!({"path": "a", "old_string": "let x = 1;\nlet y = 2;", "new_string": "let x = 10;"}));
        assert_eq!(p, vec![(Line::Del, "let x = 1;".into()), (Line::Del, "let y = 2;".into()), (Line::Add, "let x = 10;".into())]);
        assert_eq!(preview("bash", &json!({"command": "cargo test"})), vec![(Line::Plain, "$ cargo test".into())]);
        assert_eq!(preview("write_file", &json!({"content": "a\nb"})).len(), 2);
        let long = preview("write_file", &json!({"content": "x\n".repeat(100)}));
        assert_eq!(long.len(), 25);
        assert_eq!(long.last().unwrap().1, "… 76 more lines");
        assert!(preview("read_file", &json!({"path": "a"})).is_empty());
        let patch = preview("patch", &json!({"patch": "--- a\n+++ b\n@@\n-old\n+new"}));
        assert_eq!(patch.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![Line::Plain, Line::Plain, Line::Plain, Line::Del, Line::Add]);
        assert_eq!(summary("read_file", &json!({"path": "src/main.rs"})), "src/main.rs");
        assert_eq!(summary("bash", &json!({"command": "ls\nrm x"})), "ls…");
        assert_eq!(summary("list_dir", &json!({})), "list_dir");
    }

    #[test]
    fn a_read_runs_alone_and_the_model_gets_its_result_for_the_next_step() {
        let mut s = state(Mode::Ask);
        say(&mut s, "What is in a.txt?");
        assert!(s.running && assistants(&s) == 1 && s.steps == 1);
        answer(&mut s, "I will look.", &[("c1", "read_file", r#"{"path":"a.txt"}"#)]);
        assert_eq!(call(&s, "c1").state, CallState::Running, "reading needs no approval");
        assert_eq!(call(&s, "c1").args, r#"{"path":"a.txt"}"#, "the pieces of the arguments are joined");
        assert_eq!(assistants(&s), 1, "the model waits for the result");
        let _ = s.update(CodeMsg::ToolDone("c1".into(), Ok("1\thello".into())), &env());
        assert_eq!(call(&s, "c1").state, CallState::Done { ok: true, output: "1\thello".into() });
        assert_eq!((assistants(&s), s.steps), (2, 2), "the next request is on its way");
        // what that request carries: instructions, the question, the answer with its call, the result
        let wire = s.wire(None);
        let roles: Vec<&str> = wire.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool"]);
        assert!(wire[0].content.contains("/p"), "the folder is named");
        assert_eq!(wire[2].tool_calls.as_ref().unwrap()[0]["function"]["name"], "read_file");
        assert_eq!(wire[2].tool_calls.as_ref().unwrap()[0]["function"]["arguments"], r#"{"path":"a.txt"}"#);
        assert_eq!((wire[3].tool_call_id.as_deref(), wire[3].content.as_str()), (Some("c1"), "1\thello"));
        // the final answer ends the turn
        answer(&mut s, "It says hello.", &[]);
        assert!(!s.running);
    }

    #[test]
    fn an_edit_waits_for_approval_and_a_denial_goes_back_to_the_model() {
        let mut s = state(Mode::Ask);
        say(&mut s, "Fix it");
        answer(&mut s, "", &[("e1", "edit_file", r#"{"path":"a.rs","old_string":"1","new_string":"2"}"#), ("r1", "read_file", r#"{"path":"b.rs"}"#)]);
        assert_eq!(call(&s, "e1").state, CallState::Pending);
        assert_eq!(call(&s, "r1").state, CallState::Running, "a read in the same round does not wait");
        let _ = s.update(CodeMsg::ToolDone("r1".into(), Ok("x".into())), &env());
        assert_eq!(assistants(&s), 1, "the model waits while one call of the round is open");
        let _ = s.update(CodeMsg::Deny("e1".into()), &env());
        assert!(matches!(&call(&s, "e1").state, CallState::Done { ok: false, output } if output.contains("declined")));
        assert_eq!(assistants(&s), 2, "now the round is complete: the model hears about it");
        let tool_msgs: Vec<String> = s.wire(None).into_iter().filter(|m| m.role == "tool").map(|m| m.content).collect();
        assert_eq!(tool_msgs.len(), 2);
        assert!(tool_msgs[0].starts_with("Error: The user declined"), "{tool_msgs:?}");
    }

    #[test]
    fn allowing_runs_the_call_and_a_freer_mode_releases_waiting_calls() {
        let mut s = state(Mode::Ask);
        // this test is about the modes, so the switch of `/autoapprove` stands off: with it on
        // (its default) the harmless `ls` runs before any mode is touched
        s.auto_approve = false;
        say(&mut s, "x");
        answer(&mut s, "", &[("w1", "write_file", r#"{"path":"n.txt","content":"hi"}"#), ("b1", "bash", r#"{"command":"ls"}"#)]);
        assert_eq!((call(&s, "w1").state.clone(), call(&s, "b1").state.clone()), (CallState::Pending, CallState::Pending));
        let _ = s.update(CodeMsg::Approve("w1".into()), &env());
        assert_eq!(call(&s, "w1").state, CallState::Running);
        let _ = s.update(CodeMsg::Approve("w1".into()), &env());
        assert_eq!(call(&s, "w1").state, CallState::Running, "allowing twice does not run it twice");
        let _ = s.update(CodeMsg::SetMode(Mode::AutoEdit), &env());
        assert_eq!(call(&s, "b1").state, CallState::Pending, "a command still asks when only edits are free");
        let _ = s.update(CodeMsg::SetMode(Mode::Auto), &env());
        assert_eq!(call(&s, "b1").state, CallState::Running);
    }

    #[test]
    fn a_wrong_call_is_answered_to_the_model_without_asking_anybody() {
        let mut s = state(Mode::Auto);
        say(&mut s, "x");
        answer(&mut s, "", &[("u1", "no_such_tool", "{}"), ("j1", "read_file", "{not json")]);
        assert!(matches!(&call(&s, "u1").state, CallState::Done { ok: false, output } if output.contains("no tool called")));
        assert!(matches!(&call(&s, "j1").state, CallState::Done { ok: false, output } if output.contains("not valid JSON")));
        assert_eq!(assistants(&s), 2, "the model can try again");
    }

    #[test]
    fn stop_ends_the_turn_and_closes_open_calls() {
        let mut s = state(Mode::Ask);
        say(&mut s, "x");
        answer(&mut s, "", &[("w1", "write_file", r#"{"path":"n","content":""}"#), ("r1", "read_file", r#"{"path":"a"}"#)]);
        let _ = s.update(CodeMsg::Stop, &env());
        assert!(!s.running);
        assert!(matches!(call(&s, "w1").state, CallState::Done { ok: false, .. }) && matches!(call(&s, "r1").state, CallState::Done { ok: false, .. }));
        // events and results that arrive late change nothing
        let n = s.items.len();
        let _ = s.update(CodeMsg::Stream(StreamEvent::Delta("late".into())), &env());
        let _ = s.update(CodeMsg::ToolDone("r1".into(), Ok("late".into())), &env());
        assert_eq!(s.items.len(), n);
        assert!(matches!(&call(&s, "r1").state, CallState::Done { output, .. } if output == "Stopped."));
    }

    #[test]
    fn a_loop_that_never_ends_is_stopped() {
        let mut s = state(Mode::Auto);
        say(&mut s, "x");
        for i in 0..MAX_STEPS + 3 {
            if !s.running {
                break;
            }
            let id = format!("c{i}");
            answer(&mut s, "", &[(&id, "read_file", r#"{"path":"a"}"#)]);
            let _ = s.update(CodeMsg::ToolDone(id, Ok("same".into())), &env());
        }
        assert!(!s.running && s.steps == MAX_STEPS);
        assert!(matches!(s.items.last(), Some(Item::Note(n)) if n.contains("Stopped after")));
    }

    #[test]
    fn a_kept_session_is_the_same_conversation_again() {
        let mut s = state(Mode::Ask);
        say(&mut s, "x");
        answer(&mut s, "Looking.", &[("c1", "read_file", r#"{"path":"a"}"#), ("c2", "edit_file", r#"{"path":"a","old_string":"1","new_string":"2"}"#)]);
        let _ = s.update(CodeMsg::ToolDone("c1".into(), Ok("1".into())), &env());
        let wire_before = s.wire(None);
        let saved = s.saved();
        assert!(saved.iter().any(|i| matches!(i, Saved::Call { id, ok: false, output, .. } if id == "c2" && output == "Interrupted.")), "a waiting call is kept as interrupted: {saved:?}");
        let json = serde_json::to_string(&saved).unwrap();
        let mut back = state(Mode::Ask);
        back.restore(serde_json::from_str(&json).unwrap());
        assert_eq!(back.saved(), saved);
        // the model sees the same conversation, with the open call answered as an error
        let wire = back.wire(None);
        assert_eq!(wire.len(), wire_before.len());
        assert_eq!(wire.last().unwrap().content, "Error: Interrupted.");
        assert!(matches!(back.items.get(1), Some(Item::Assistant { text, .. }) if text == "Looking."));
        // a long session keeps the newest items, starting at a message of the user
        let mut long = state(Mode::Ask);
        for i in 0..150 {
            long.items.push(Item::User { text: format!("q{i}"), sent: None });
            long.items.push(Item::Note(format!("n{i}")));
        }
        let kept = long.saved();
        assert!(kept.len() <= SAVED_ITEMS && matches!(kept.first(), Some(Saved::User { .. })));
        assert!(matches!(kept.last(), Some(Saved::Note { text }) if text == "n149"));
    }

    #[test]
    fn plan_mode_offers_only_reading_refuses_the_rest_and_the_plan_can_be_carried_out() {
        let mut s = state(Mode::Ask);
        s.tools_json = json!([
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "grep"}},
            {"type": "function", "function": {"name": "edit_file"}},
            {"type": "function", "function": {"name": "bash"}}
        ]);
        assert_eq!(s.tools_for_model().as_array().unwrap().len(), 4);
        let _ = s.update(CodeMsg::SetMode(Mode::Plan), &env());
        let offered = s.tools_for_model();
        let names: Vec<&str> = offered.as_array().unwrap().iter().filter_map(|t| t.pointer("/function/name").and_then(Value::as_str)).collect();
        assert_eq!(names, vec!["read_file", "grep"], "the model is not even offered a tool that writes");
        assert_eq!(s.build_mode, Mode::Ask);
        say(&mut s, "How would you rename the module?");
        assert!(s.wire(None)[0].content.starts_with("PLAN MODE"));
        // a model that calls a writing tool anyway is told no, and nobody is asked
        answer(&mut s, "", &[("e1", "edit_file", r#"{"path":"a","old_string":"1","new_string":"2"}"#), ("r1", "read_file", r#"{"path":"a"}"#)]);
        assert!(matches!(&call(&s, "e1").state, CallState::Done { ok: false, output } if output.contains("Plan mode only allows reading")));
        assert_eq!(call(&s, "r1").state, CallState::Running);
        let _ = s.update(CodeMsg::ToolDone("r1".into(), Ok("x".into())), &env());
        answer(&mut s, "1. Rename the file.\n2. Fix the imports.", &[]);
        assert!(!s.running);
        // carrying it out leaves plan mode for the mode before, and the agent is asked to do it
        let _ = s.update(CodeMsg::ExecutePlan, &env());
        assert_eq!(s.mode, Mode::Ask);
        assert!(s.running);
        assert!(matches!(s.items.iter().rev().find(|i| matches!(i, Item::User { .. })), Some(Item::User { text, .. }) if text == "Carry out the plan above."));
        assert!(s.wire(None)[0].content.starts_with("You are a coding agent"));
        // outside plan mode the button does nothing
        let before = s.items.len();
        let _ = s.update(CodeMsg::ExecutePlan, &env());
        assert_eq!(s.items.len(), before);
    }

    #[test]
    fn switching_to_plan_mode_refuses_what_waited_for_approval() {
        let mut s = state(Mode::Ask);
        say(&mut s, "x");
        answer(&mut s, "", &[("w1", "write_file", r#"{"path":"n","content":"hi"}"#)]);
        assert_eq!(call(&s, "w1").state, CallState::Pending);
        let _ = s.update(CodeMsg::SetMode(Mode::Plan), &env());
        assert!(matches!(&call(&s, "w1").state, CallState::Done { ok: false, .. }));
        assert_eq!(s.build_mode, Mode::Ask, "the mode before is remembered");
        let _ = s.update(CodeMsg::SetMode(Mode::AutoEdit), &env());
        let _ = s.update(CodeMsg::SetMode(Mode::Plan), &env());
        assert_eq!(s.build_mode, Mode::AutoEdit);
    }

    #[test]
    fn the_screen_tools_are_not_offered_and_mcp_tools_get_safe_names() {
        let (list, for_model) = parse_tools(&json!({"tools": [
            {"type": "function", "function": {"name": "read_file"}, "permission": "read"},
            {"type": "function", "function": {"name": "computer_click"}, "permission": "computer"}
        ]}));
        assert_eq!(list.len(), 1);
        assert_eq!(for_model.as_array().unwrap().len(), 1, "the model does not hear of computer use");
        assert_eq!(mcp_name("my.server", "do thing"), "mcp__my_server__do_thing");
        assert_eq!(mcp_name(&"s".repeat(40), &"t".repeat(40)).len(), 64);
        let (tools, defs) = parse_mcp_tools(&json!({"tools": [
            {"server": "fs", "name": "read", "description": "Reads", "input_schema": {"type": "object", "properties": {"p": {"type": "string"}}}},
            {"server": "fs", "name": "list"},
            {"nonsense": 1}
        ]}));
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].kind, ToolKind::Mcp { server: "fs".into(), tool: "read".into() });
        assert_eq!(tools[0].permission, Permission::Command, "an MCP tool asks first, like a command");
        assert_eq!(defs[0]["function"]["description"], "[fs] Reads");
        assert_eq!(defs[1]["function"]["parameters"], json!({"type": "object", "properties": {}}), "a tool without a schema takes no arguments");
    }

    #[test]
    fn the_instructions_of_the_folder_the_notes_and_the_profile_reach_the_model() {
        let ctx = parse_context(&json!({
            "files": [{"file": "AGENTS.md", "text": "Always run cargo test."}, {"file": "../CLAUDE.md", "text": "Be brief."}],
            "memory": {"workspace": ["uses tabs"], "global": ["prefers German"]},
            "profile": "  likes short answers  "
        }));
        assert_eq!(ctx.files.len(), 2);
        let mut s = state(Mode::Ask);
        s.context = ctx.clone();
        let system = s.wire(Some("Custom line.")).remove(0).content;
        assert!(system.contains("Instructions of this project (follow them; the first is the most specific)\n(AGENTS.md)\nAlways run cargo test.\n\n(../CLAUDE.md)\nBe brief."), "{system}");
        assert!(system.contains("Memory (facts saved in earlier sessions)\nAbout this workspace:\n- uses tabs\nAbout the user (all workspaces):\n- prefers German"));
        assert!(system.contains("likes short answers"));
        assert!(system.find("Instructions of this project").unwrap() < system.find("Custom line.").unwrap(), "your own prompt comes last");
        assert_eq!(Context::default().prompt(), "");
        // a very long file is cut, and the next file only gets what is left
        let long = Context { files: vec![("A.md".into(), "x".repeat(20_000)), ("B.md".into(), "y".repeat(500))], ..Default::default() };
        let p = long.prompt();
        assert!(p.chars().count() < 9_400 && p.contains("..."), "{}", p.chars().count());
        assert!(!p.contains("B.md"), "nothing is left for the second file");
    }

    #[test]
    fn the_task_list_is_kept_by_the_window_and_the_model_can_read_it_back() {
        let mut s = state(Mode::Ask);
        say(&mut s, "x");
        let todos = r#"{"todos":[{"content":"read the code","status":"completed"},{"content":"fix it","status":"in_progress"},{"content":"test it","status":"pending"}]}"#;
        answer(&mut s, "", &[("t1", "todo_write", todos)]);
        assert!(matches!(call(&s, "t1").state, CallState::Done { ok: true, .. }), "no approval, no server");
        assert_eq!(s.todos.len(), 3);
        assert_eq!(s.todos[1], Todo { content: "fix it".into(), status: TodoStatus::InProgress });
        assert_eq!(assistants(&s), 2, "the model goes on at once");
        answer(&mut s, "", &[("t2", "todo_read", "")]);
        assert!(matches!(&call(&s, "t2").state, CallState::Done { ok: true, output } if output == "1. [x] read the code\n2. [~] fix it\n3. [ ] test it"));
        answer(&mut s, "", &[("t3", "todo_write", r#"{"todos":[{"content":"a","status":"nearly"}]}"#)]);
        assert!(matches!(&call(&s, "t3").state, CallState::Done { ok: false, output } if output.contains("status must be")));
        assert_eq!(s.todos.len(), 3, "a wrong list changes nothing");
        // a new session starts without the list
        let _ = s.update(CodeMsg::Stop, &env());
        let _ = s.update(CodeMsg::NewSession, &env());
        assert!(s.todos.is_empty());
    }

    #[test]
    fn a_question_of_the_agent_waits_for_you_in_every_mode_and_your_words_are_the_answer() {
        for mode in [Mode::Ask, Mode::Auto, Mode::Plan] {
            let mut s = state(mode);
            say(&mut s, "x");
            answer(&mut s, "", &[("q1", "ask_user", r#"{"question":"Which file?","options":["a.rs","b.rs"]}"#)]);
            assert_eq!(call(&s, "q1").state, CallState::Pending, "{mode:?}");
            assert_eq!(assistants(&s), 1, "the model waits for the answer");
            assert_eq!(s.pending_ask().map(|c| c.id.as_str()), Some("q1"));
            // allowing is not answering
            let _ = s.update(CodeMsg::Approve("q1".into()), &env());
            assert_eq!(call(&s, "q1").state, CallState::Pending);
            let _ = s.update(CodeMsg::SetMode(Mode::Auto), &env());
            assert_eq!(call(&s, "q1").state, CallState::Pending, "a freer mode does not answer for you");
            let _ = s.update(CodeMsg::SetMode(mode), &env());
            // typing in the box answers
            s.editor = Content::with_text("a.rs please");
            let _ = s.update(CodeMsg::Send, &env());
            assert_eq!(call(&s, "q1").state, CallState::Done { ok: true, output: "a.rs please".into() });
            assert_eq!(assistants(&s), 2);
            assert!(s.pending_ask().is_none());
            assert!(!s.items.iter().any(|i| matches!(i, Item::User { text, .. } if text == "a.rs please")), "it is not a new request");
        }
        // an option, and skipping
        let mut s = state(Mode::Ask);
        say(&mut s, "x");
        answer(&mut s, "", &[("q1", "ask_user", r#"{"question":"Q?","options":["yes","no"]}"#)]);
        let _ = s.update(CodeMsg::Answer("q1".into(), "no".into()), &env());
        assert_eq!(call(&s, "q1").state, CallState::Done { ok: true, output: "no".into() });
        let _ = s.update(CodeMsg::Answer("q1".into(), "yes".into()), &env());
        assert_eq!(call(&s, "q1").state, CallState::Done { ok: true, output: "no".into() }, "answered once");
        answer(&mut s, "", &[("q2", "ask_user", r#"{"question":"Q2?"}"#)]);
        let _ = s.update(CodeMsg::Deny("q2".into()), &env());
        assert!(matches!(&call(&s, "q2").state, CallState::Done { ok: false, output } if output.contains("did not answer")));
        // an empty box does not answer
        answer(&mut s, "", &[("q3", "ask_user", r#"{"question":"Q3?"}"#)]);
        s.editor = Content::with_text("   ");
        let _ = s.update(CodeMsg::Send, &env());
        assert_eq!(call(&s, "q3").state, CallState::Pending);
    }

    #[test]
    fn an_mcp_tool_asks_like_a_command() {
        let mut s = state(Mode::Ask);
        say(&mut s, "x");
        answer(&mut s, "", &[("m1", "mcp__github__list_issues", r#"{"repo":"a/b"}"#)]);
        assert_eq!(call(&s, "m1").state, CallState::Pending);
        assert_eq!(call(&s, "m1").kind, ToolKind::Mcp { server: "github".into(), tool: "list_issues".into() });
        let _ = s.update(CodeMsg::Approve("m1".into()), &env());
        assert_eq!(call(&s, "m1").state, CallState::Running);
        let _ = s.update(CodeMsg::ToolDone("m1".into(), Err("The server is not connected.".into())), &env());
        assert!(matches!(&call(&s, "m1").state, CallState::Done { ok: false, output } if output.contains("not connected")));
        assert_eq!(assistants(&s), 2, "the model hears about it");
    }

    #[test]
    fn nothing_is_sent_without_a_folder_a_model_or_text() {
        let mut s = state(Mode::Ask);
        let _ = s.update(CodeMsg::Send, &env());
        assert!(s.items.is_empty() && !s.running, "no text");
        s.editor = Content::with_text("hi");
        let _ = s.update(CodeMsg::Send, &Env { model: None, ..env() });
        assert!(s.items.is_empty(), "no model");
        s.workspace = None;
        let _ = s.update(CodeMsg::Send, &env());
        assert!(s.items.is_empty(), "no folder");
    }

    #[test]
    fn an_error_of_the_stream_is_shown_and_ends_the_turn() {
        let mut s = state(Mode::Ask);
        say(&mut s, "x");
        let _ = s.update(CodeMsg::Stream(StreamEvent::Error("context length exceeded".into())), &env());
        assert!(!s.running);
        assert!(matches!(s.items.last(), Some(Item::Note(n)) if n == "context length exceeded"));
        // a new request starts fresh
        say(&mut s, "again");
        assert!(s.running && s.steps == 1);
    }
}
