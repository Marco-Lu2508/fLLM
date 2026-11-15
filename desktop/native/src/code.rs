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
use std::collections::BTreeMap;
use std::path::Path;

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
    Call { id: String, name: String, args: String, permission: String, ok: bool, output: String, #[serde(default)] sent: Option<String> },
    Note { text: String },
    /// Kept with the session like the messages of the page are, so the review rounds survive it.
    ReviewFeedback { text: String },
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

/// Sessions by folder, in one file next to the chats.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Sessions(pub BTreeMap<String, Vec<Saved>>);

impl Sessions {
    pub fn load(path: &Path) -> Sessions {
        let Ok(text) = std::fs::read_to_string(path) else { return Sessions::default() };
        match serde_json::from_str(&text) {
            Ok(s) => s,
            Err(_) => {
                let _ = std::fs::rename(path, path.with_extension("damaged.json"));
                Sessions::default()
            }
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(self).map_err(std::io::Error::other)?)?;
        std::fs::rename(tmp, path)
    }
}

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
    pub(crate) pending_toast: Option<String>,
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
    pub changes: Vec<Change>,
    pub context: Context,
    pub todos: Vec<Todo>,
    pub error: Option<String>,
    acc: Vec<Partial>,
    handle: Option<task::Handle>,
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
            rules: crate::commands::Rules::default(),
            personality: "default".into(),
            custom: Vec::new(),
            pending_prefs: None,
            pending_toast: None,
            pending: Vec::new(),
            pending_turbo: None,
            pending_sandbox: None,
            always_allow: std::collections::HashSet::new(),
            running: false,
            steps: 0,
            changes: Vec::new(),
            context: Context::default(),
            todos: Vec::new(),
            error: None,
            acc: Vec::new(),
            handle: None,
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
    // the screen and input tools are for computer use, which is not switched on here: the model is not offered them
    let list: Vec<Value> = v.get("tools").and_then(Value::as_array).cloned().unwrap_or_default().into_iter().filter(|t| t.get("permission").and_then(Value::as_str) != Some("computer")).collect();
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
        .map(|c| Change { id: c.get("id").and_then(Value::as_u64).unwrap_or(0), path: c.get("path").and_then(Value::as_str).unwrap_or_default().to_string(), kind: c.get("kind").and_then(Value::as_str).unwrap_or("edit").to_string() })
        .collect()
}

/// The tools the window itself provides: a task list and questions to you.
pub fn client_tools() -> (Vec<Tool>, Vec<Value>) {
    let def = |name: &str, description: &str, properties: Value, required: Value| json!({"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}});
    let tools = vec![
        Tool { name: "todo_write".into(), permission: Permission::Read, kind: ToolKind::Todo { write: true } },
        Tool { name: "todo_read".into(), permission: Permission::Read, kind: ToolKind::Todo { write: false } },
        Tool { name: "ask_user".into(), permission: Permission::Read, kind: ToolKind::Ask },
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

pub async fn open_workspace(base: String, path: String) -> Result<Workspace, String> {
    parse_workspace(&call(&base, reqwest::Method::POST, "/api/studio/agent/workspace", Some(json!({ "path": path }))).await?)
}

/// The folder dialog of the server's computer: `None` when it was closed without a choice.
pub async fn pick_folder(base: String) -> Result<Option<String>, String> {
    let v = call(&base, reqwest::Method::POST, "/api/studio/pick-file", Some(json!({ "kind": "folder" }))).await?;
    Ok(v.get("path").and_then(Value::as_str).filter(|p| !p.is_empty()).map(str::to_string))
}

pub async fn fetch_tools(base: String) -> Result<(Vec<Tool>, Value), String> {
    let (mut tools, for_model) = call(&base, reqwest::Method::GET, "/api/studio/agent/tools", None).await.map(|v| parse_tools(&v))?;
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
        let mut out: Vec<Saved> = self
            .items
            .iter()
            .filter_map(|i| match i {
                Item::User { text, sent } => Some(Saved::User {
                    text: text.clone(),
                    sent: sent.clone(),
                }),
                Item::Assistant { text, .. } => (!text.is_empty()).then(|| Saved::Assistant { text: text.clone() }),
                Item::Note(t) => Some(Saved::Note { text: t.clone() }),
                Item::ReviewFeedback(t) => Some(Saved::ReviewFeedback { text: t.clone() }),
                Item::Call(c) => {
                    let (ok, output) = match &c.state {
                        CallState::Done { ok, output } => (*ok, output.clone()),
                        _ => (false, "Interrupted.".to_string()),
                    };
                    Some(Saved::Call { id: c.id.clone(), name: c.name.clone(), args: c.args.clone(), permission: permission_name(c.permission).into(), ok, output, sent: c.sent.clone() })
                }
            })
            .collect();
        if out.len() > SAVED_ITEMS {
            out.drain(..out.len() - SAVED_ITEMS);
            // a call without the answer that announced it would confuse the model: start at a message of yours
            let first = out.iter().position(|i| matches!(i, Saved::User { .. })).unwrap_or(0);
            out.drain(..first);
        }
        out
    }

    /// Takes a kept session back as the conversation.
    pub fn restore(&mut self, saved: Vec<Saved>) {
        self.steps = 0;
        self.items = saved
            .into_iter()
            .map(|s| match s {
                Saved::User { text, sent } => Item::User { text, sent },
                Saved::Assistant { text } => {
                    let mut md = markdown::Content::new();
                    md.push_str(&text);
                    Item::Assistant { text, md: Box::new(md) }
                }
                Saved::Call { id, name, args, permission, ok, output, sent } => Item::Call(Call { id, name, args, permission: Permission::parse(&permission), kind: ToolKind::Server, state: CallState::Done { ok, output }, auto_note: None, check: None, sent }),
                Saved::Note { text } => Item::Note(text),
                Saved::ReviewFeedback { text } => Item::ReviewFeedback(text),
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

    fn call_mut(&mut self, id: &str) -> Option<&mut Call> {
        self.items.iter_mut().find_map(|i| match i {
            Item::Call(c) if c.id == id => Some(c),
            _ => None,
        })
    }

    /// The messages of the next request: the agent's instructions, then the session with every call answered.
    pub fn wire(&self, extra: Option<&str>) -> Vec<WireMessage> {
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
        let context = self.context.prompt();
        if !context.is_empty() {
            system.push_str("\n\n");
            system.push_str(&context);
        }
        if let Some(x) = extra.filter(|x| !x.trim().is_empty()) {
            system.push_str("\n\n");
            system.push_str(x.trim());
        }
        let mut wire = vec![WireMessage::new("system", system)];
        let mut i = 0;
        while i < self.items.len() {
            match &self.items[i] {
                Item::ReviewFeedback(text) => {
                    // sent as it stands, and never drawn (the view skips this variant)
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
                    let calls: Vec<&Call> = self.items[i + 1..].iter().take_while(|n| matches!(n, Item::Call(_))).filter_map(|n| if let Item::Call(c) = n { Some(c) } else { None }).collect();
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
                    i += calls.len();
                }
                Item::Call(_) => {}
            }
            i += 1;
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
            turbo: env.turbo,
            sandbox: env.sandbox.map(str::to_string),
            sandbox_available: env.sandbox_available,
            personality: self.personality.clone(),
            rules: self.rules.clone(),
            custom: self.custom.clone(),
            usage: env.usage.map(str::to_string),
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
        let mut sent_full = prompt.to_string();
        if !self.pending.is_empty() {
            let shared = std::mem::take(&mut self.pending);
            sent_full = format!("{}\n\n{sent_full}", shared.join("\n\n"));
        }
        let sent = (sent_full != display).then_some(sent_full);
        self.error = None;
        self.steps = 0;
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
                Step::NewSession => {
                    // `newCodeSession` of the page: stop, save, reset
                    if self.running {
                        tasks.push(self.stop_now(env));
                    }
                    self.persist(env.session_file);
                    self.items.clear();
                    self.todos.clear();
                    self.steps = 0;
                    self.always_allow.clear();
                }
                Step::Undo(steps) => {
                    // the page sends `{ steps }` above one, and one is the plain call
                    let n = if steps > 1.0 { Some(steps as u64) } else { None };
                    tasks.push(self.update(CodeMsg::Undo(n), env));
                }
                Step::AgentRun(prompt, display) => {
                    // a run of the agents pane ends without a self-review (the page calls
                    // `selfReview` from `submitText` only)
                    self.main_run = false;
                    tasks.push(self.run_prompt(&prompt, &display, env))
                }
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
                Step::UserTool { name, args } => tasks.push(self.user_tool(name, args, env)),
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
        self.acc.clear();
        for i in self.items.iter_mut() {
            if let Item::Call(c) = i {
                if matches!(c.state, CallState::Pending | CallState::Running) {
                    c.state = CallState::Done { ok: false, output: "Stopped.".into() };
                }
            }
        }
        let base = env.base.to_string();
        Task::perform(
            async move { cancel(base.clone()).await; fetch_changes(base).await },
            |r| Message::Code(CodeMsg::ChangesLoaded(r)),
        )
    }

    /// The session of this folder onto disk (the write of `app.rs` after a run or a new session,
    /// here so a command can save before it clears the conversation).
    pub fn persist(&self, file: &std::path::Path) {
        let Some(w) = &self.workspace else { return };
        let mut all = Sessions::load(file);
        let saved = self.saved();
        if saved.is_empty() {
            all.0.remove(&w.root);
        } else {
            all.0.insert(w.root.clone(), saved);
        }
        if let Err(e) = all.save(file) {
            eprintln!("fllm-app: the code session could not be saved: {e}");
        }
    }

    fn start_stream(&mut self, env: &Env) -> Task<Message> {
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
        if self.mode != Mode::Plan {
            return self.tools_json.clone();
        }
        let readable = |name: &str| self.tool(name).is_some_and(|t| self.mode.allows(t.permission));
        Value::Array(self.tools_json.as_array().into_iter().flatten().filter(|t| t.pointer("/function/name").and_then(Value::as_str).is_some_and(readable)).cloned().collect())
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
            ToolKind::Server | ToolKind::Ask => {
                if let Some(c) = self.call_mut(&id) {
                    c.state = CallState::Running;
                }
                Task::perform(run_tool(base, name, args), move |r| Message::Code(CodeMsg::ToolDone(id.clone(), r)))
            }
        }
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
        // the chain of `runToolCall` (code.js 1955-2010): plan first, then the rules, the mode
        // and the remembered keys of `authorize`, and the harmless table gets the last word on
        // what is left as a question
        let kind_mcp = match &kind {
            ToolKind::Mcp { server, tool } => Some((server.as_str(), tool.as_str())),
            _ => None,
        };
        let parsed: Value = serde_json::from_str(args).unwrap_or(Value::Null);
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
            // between the last answer and the `done` sits the self-review (`selfReview` of the
            // page, called right after its `runLoop`): the run stays open while it works
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
                        // two rounds are done, or an answer arrived out of order: the run ends
                        self.review = None;
                        self.end_round()
                    }
                };
            }
            if self.main_run && self.auto_review && self.mode != Mode::Plan {
                self.review = Some(Review {
                    round: 1,
                    stage: ReviewStage::Changes,
                    files: Vec::new(),
                    report: crate::review::Report::default(),
                });
                return self.begin_review(env);
            }
            return self.end_round();
        }
        let mut tasks = Vec::new();
        let mut ids = Vec::new();
        for (n, p) in partial.into_iter().enumerate() {
            let id = if p.id.is_empty() { format!("call_{}_{n}", self.steps) } else { p.id };
            let tool = self.tool(&p.name).cloned();
            let permission = tool.as_ref().map_or(Permission::Other, |t| t.permission);
            let kind = tool.as_ref().map_or(ToolKind::Server, |t| t.kind.clone());
            let (state, auto_note) =
                self.starting_state(tool.is_some(), kind.clone(), permission, &p.name, &p.args, env.turbo);
            let run_now = state == CallState::Running;
            self.items.push(Item::Call(Call { id: id.clone(), name: p.name, args: p.args, permission, kind, state, auto_note, check: None, sent: None }));
            if run_now {
                ids.push(id);
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
        if self.check_hold.is_some() {
            // the answer of one call is still being checked (`await autoCheckEdit` there)
            return Task::none();
        }
        if !self.running || self.last_round().any(|c| matches!(c.state, CallState::Pending | CallState::Running)) {
            return Task::none();
        }
        self.start_stream(env)
    }

    /// The end of a round that is really over (the `finally` of the page's `submitText`).
    fn end_round(&mut self) -> Task<Message> {
        self.running = false;
        self.handle = None;
        Task::none()
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
            self.review = None;
            return self.end_round();
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
                self.context = Context::default();
                self.todos.clear();
                self.items.clear();
                self.changes.clear();
                self.running = false;
                self.handle = None;
                Task::none()
            }
            CodeMsg::Opened(Ok(w)) => {
                self.path_input = w.root.clone();
                self.workspace = Some(w);
                self.error = None;
                let base = env.base.to_string();
                Task::batch([
                    Task::perform(fetch_tools(base.clone()), |r| Message::Code(CodeMsg::ToolsLoaded(r))),
                    Task::perform(fetch_changes(base.clone()), |r| Message::Code(CodeMsg::ChangesLoaded(r))),
                    Task::perform(fetch_context(base.clone()), |r| Message::Code(CodeMsg::ContextLoaded(r))),
                    // the custom commands of the folder come with it (`loadCustomCommands`
                    // runs with the other loads when the page opens a workspace)
                    Task::perform(load_custom_commands(base), |r| Message::Code(CodeMsg::CustomLoaded(r))),
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
                self.error = Some(e);
                Task::none()
            }
            CodeMsg::ToolsLoaded(Ok((tools, for_model))) => {
                self.tools = tools;
                self.tools_json = for_model;
                Task::none()
            }
            CodeMsg::ToolsLoaded(Err(e)) => {
                self.error = Some(e);
                Task::none()
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
                    // the second round with nothing a machine can state: the review is over
                    self.review = None;
                    return self.end_round();
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
                // `submitCode`); everything else waits in the box, where the page queues it —
                // this window has no queue
                if self.running {
                    let stop = typed.is_empty() || typed.eq_ignore_ascii_case("/stop");
                    if !stop {
                        return Task::none();
                    }
                    self.editor = Content::new();
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
                // `turn.changeMark`: the self-review only looks at what this run changes
                self.change_mark = self.changes.iter().map(|c| c.id).max().unwrap_or(0);
                self.main_run = true;
                self.run_prompt(&prompt, &prompt, env)
            }
            CodeMsg::Stop => self.stop_now(env),
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
                        Task::none()
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
                    // a plan changes nothing: what waited for approval is refused
                    for i in self.items.iter_mut() {
                        if let Item::Call(c) = i {
                            if c.state == CallState::Pending && c.kind != ToolKind::Ask {
                                c.state = CallState::Done { ok: false, output: "Plan mode only allows reading. Write the plan instead.".into() };
                            }
                        }
                    }
                }
                // calls that wait now may be allowed
                let ids: Vec<String> = self.items.iter().filter_map(|i| if let Item::Call(c) = i { (c.state == CallState::Pending && c.kind != ToolKind::Ask && m.allows(c.permission)).then(|| c.id.clone()) } else { None }).collect();
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
                    self.change_mark = self.changes.iter().map(|c| c.id).max().unwrap_or(0);
                    self.main_run = true;
                    self.run_prompt(&prompt, &source, env)
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
                self.items.clear();
                self.todos.clear();
                self.steps = 0;
                self.always_allow.clear();
                Task::none()
            }
        }
    }
}

// ---- the page --------------------------------------------------------------------------------------------------------------------

fn mono_box<'a>(t: Tokens, body: Element<'a, Message>) -> Element<'a, Message> {
    container(body).padding([8, 12]).width(Length::Fill).style(theme::inset(t)).into()
}

fn call_card<'a>(t: Tokens, c: &'a Call) -> Element<'a, Message> {
    let args: Value = serde_json::from_str(&c.args).unwrap_or(Value::Null);
    let state_label: Element<'a, Message> = match &c.state {
        CallState::Pending => text("Waiting for you").size(12).color(t.primary).into(),
        CallState::Running => text("Running…").size(12).color(t.muted).into(),
        CallState::Done { ok: true, .. } => text("Done").size(12).color(t.ok).into(),
        CallState::Done { ok: false, .. } => text("Failed").size(12).color(t.danger).into(),
    };
    let header = row![
        text(c.name.as_str()).size(13).font(Font { weight: iced::font::Weight::Semibold, ..Font::MONOSPACE }),
        text(summary(&c.name, &args)).size(13).font(Font::MONOSPACE).color(t.fg_soft).width(Length::Fill).wrapping(text::Wrapping::None),
        state_label,
    ]
    .spacing(10)
    .align_y(Alignment::Center);
    let mut body = column![header].spacing(8);
    if let Some(reason) = &c.auto_note {
        // `tool-auto` of the page: its own line over the body of the card
        body = body.push(
            container(text(format!("Approved on its own: {reason}")).size(11).color(t.muted))
                .padding(iced::Padding { top: 2.0, right: 0.0, bottom: 0.0, left: 22.0 })
                .width(Length::Fill),
        );
    }

    // a command is already in the header; an edit shows what it removes and adds, each line on a tinted band
    let lines = if c.name == "bash" { Vec::new() } else { preview(&c.name, &args) };
    if !lines.is_empty() {
        let shown = column(lines.into_iter().map(|(kind, l)| {
            let (mark, color, band) = match kind {
                Line::Add => ("+ ", t.ok, Some(iced::Color { a: 0.14, ..t.ok })),
                Line::Del => ("- ", t.danger, Some(iced::Color { a: 0.14, ..t.danger })),
                Line::Plain => ("  ", t.fg_soft, None),
            };
            container(text(format!("{mark}{}", l.replace('\t', "    "))).size(12).font(Font::MONOSPACE).color(color))
                .width(Length::Fill)
                .padding([1, 6])
                .style(move |_| container::Style { background: band.map(iced::Background::Color), ..Default::default() })
                .into()
        }));
        body = body.push(container(shown).padding([6, 6]).width(Length::Fill).style(theme::inset(t)));
    }
    if let CallState::Done { output, ok } = &c.state {
        let all: Vec<&str> = output.lines().collect();
        // a tab is not in the monospace font: four spaces show it
        let mut shown: String = all.iter().take(14).map(|l| l.replace('\t', "    ")).collect::<Vec<_>>().join("\n");
        if all.len() > 14 {
            shown.push_str(&format!("\n… {} more lines", all.len() - 14));
        }
        if !shown.trim().is_empty() {
            body = body.push(mono_box(t, text(shown).size(12).font(Font::MONOSPACE).color(if *ok { t.fg_soft } else { t.danger }).into()));
        }
    }
    if let Some(problems) = &c.check {
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
            // the feedback of the self-review is for the model only — it draws nothing
            Item::ReviewFeedback(_) => Space::new().into(),
            Item::User { text: text_, .. } => row![Space::new().width(Length::Fill), container(text(text_.as_str()).size(15)).padding([9, 15]).max_width(560).style(theme::bubble(t, tint))].into(),
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
        container(column![row![text(w.name.as_str()).size(15).font(bold()), text(branch).size(12).color(t.muted).font(Font::MONOSPACE)].spacing(8).align_y(Alignment::Center), text(w.root.as_str()).size(12).color(t.muted).font(Font::MONOSPACE).wrapping(text::Wrapping::None)].spacing(2)).clip(true).width(Length::Fill),
        mode,
        button(text("New session").size(13)).on_press_maybe((!s.running).then_some(Message::Code(CodeMsg::NewSession))).padding([5, 14]).style(theme::secondary_button(t)),
        button(text("Other folder").size(13)).on_press_maybe((!s.running).then_some(Message::Code(CodeMsg::Close))).padding([5, 14]).style(theme::secondary_button(t)),
    ]
    .spacing(10)
    .align_y(Alignment::Center);
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
    let composer = container(column![editor, row![text("Enter sends, Shift+Enter adds a new line").size(12).color(t.muted), Space::new().width(Length::Fill), action].align_y(Alignment::Center).padding(Padding { top: 0.0, right: 10.0, bottom: 8.0, left: 16.0 })])
        .style(theme::card(t, theme::R_XL, false))
        .max_width(820);
    let composer = container(composer).width(Length::Fill).center_x(Length::Fill).padding(Padding { top: 8.0, right: 24.0, bottom: 20.0, left: 24.0 });
    let bar = container(container(bar).max_width(820).width(Length::Fill)).center_x(Length::Fill).padding(Padding { top: 0.0, right: 24.0, bottom: 0.0, left: 24.0 });

    let main = column![header, rule(t), timeline(s, t, theme, tint), bar, composer].width(Length::Fill).height(Length::Fill);
    row![main, side_settings(s, t)].align_y(Alignment::Start).width(Length::Fill).height(Length::Fill).into()
}

/// The side of the Code page — the part row 81 builds: the *Settings* block of `renderCodeSide`
/// with the two switches of `code.autoCheck` and `code.autoReview`, with their words of the
/// page. The other blocks it sits among (sessions, memory, permissions, the style and what the
/// agent learns) are still open and come with their own rows.
fn side_settings<'a>(s: &'a CodeState, t: Tokens) -> Element<'a, Message> {
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
    ]
    .spacing(10);
    container(block)
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
            ["bash . todo"],
            "without a command the words of the arguments follow the name — in the order the JSON \
             parser of the window keeps them, where the page walks its own insertion order (the \
             difference shows only in this case, which a call with a `command` never reaches)"
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
        assert_eq!(parse_changes(&json!({"changes": [{"id": 3, "path": "a.rs", "kind": "edit"}]})), vec![Change { id: 3, path: "a.rs".into(), kind: "edit".into() }]);
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
