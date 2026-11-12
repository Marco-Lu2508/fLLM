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
use iced::widget::{button, column, container, markdown, row, scrollable, text, text_editor, text_input, Space};
use iced::{keyboard, task, Alignment, Element, Font, Length, Padding, Task};
use serde_json::{json, Value};

/// The most model calls one request of yours may take: a loop that never ends is stopped here.
pub const MAX_STEPS: usize = 30;

const CODE_PROMPT: &str = "You are a coding agent working in the user's project folder. Use the tools to look at the code before you answer or change anything: list_dir, glob and grep to find files, read_file to read them. Read a file before you edit it, make small precise edits with edit_file, and check your work (run the tests or the build with bash when there are any). Answer briefly and say what you changed. Never invent file contents or command output.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    Read,
    Edit,
    Command,
    Memory,
    Network,
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
}

impl Mode {
    pub fn allows(self, p: Permission) -> bool {
        match p {
            Permission::Read | Permission::Memory => true,
            Permission::Edit => self != Mode::Ask,
            Permission::Command | Permission::Network | Permission::Other => self == Mode::Auto,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub permission: Permission,
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
    pub state: CallState,
}

pub enum Item {
    User(String),
    Assistant { text: String, md: Box<markdown::Content> },
    Call(Call),
    Note(String),
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
    Close,
    Opened(Result<Workspace, String>),
    ToolsLoaded(Result<(Vec<Tool>, Value), String>),
    ChangesLoaded(Result<Vec<Change>, String>),
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
}

/// What the agent needs from the window for a request.
pub struct Env<'a> {
    pub base: &'a str,
    pub model: Option<&'a str>,
    pub params: ChatParams,
    /// The custom system prompt of the Settings page, added to the agent's own instructions; `None` while that is automatic.
    pub extra: Option<&'a str>,
}

pub struct CodeState {
    pub path_input: String,
    pub workspace: Option<Workspace>,
    pub tools: Vec<Tool>,
    tools_json: Value,
    pub items: Vec<Item>,
    pub editor: Content,
    pub mode: Mode,
    pub running: bool,
    pub steps: usize,
    pub changes: Vec<Change>,
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
            running: false,
            steps: 0,
            changes: Vec::new(),
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
    let list = v.get("tools").and_then(Value::as_array).cloned().unwrap_or_default();
    let tools = list
        .iter()
        .filter_map(|t| Some(Tool { name: t.pointer("/function/name")?.as_str()?.to_string(), permission: Permission::parse(t.get("permission").and_then(Value::as_str).unwrap_or("")) }))
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

pub async fn fetch_tools(base: String) -> Result<(Vec<Tool>, Value), String> {
    call(&base, reqwest::Method::GET, "/api/studio/agent/tools", None).await.map(|v| parse_tools(&v))
}

pub async fn fetch_changes(base: String) -> Result<Vec<Change>, String> {
    call(&base, reqwest::Method::GET, "/api/studio/agent/changes", None).await.map(|v| parse_changes(&v))
}

pub async fn run_tool(base: String, name: String, args: Value) -> Result<String, String> {
    tool_result(&call(&base, reqwest::Method::POST, "/api/studio/agent/tool", Some(json!({ "name": name, "args": args }))).await?)
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

impl CodeState {
    fn tool(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// The calls of the last round: those after the last assistant item.
    fn last_round(&self) -> impl Iterator<Item = &Call> {
        let start = self.items.iter().rposition(|i| matches!(i, Item::Assistant { .. })).map_or(0, |i| i + 1);
        self.items[start..].iter().filter_map(|i| if let Item::Call(c) = i { Some(c) } else { None })
    }

    fn call_mut(&mut self, id: &str) -> Option<&mut Call> {
        self.items.iter_mut().find_map(|i| match i {
            Item::Call(c) if c.id == id => Some(c),
            _ => None,
        })
    }

    /// The messages of the next request: the agent's instructions, then the session with every call answered.
    pub fn wire(&self, extra: Option<&str>) -> Vec<WireMessage> {
        let mut system = CODE_PROMPT.to_string();
        if let Some(w) = &self.workspace {
            system.push_str(&format!("\n\nThe project folder is {} (tool paths are relative to it).", w.root));
        }
        if let Some(x) = extra.filter(|x| !x.trim().is_empty()) {
            system.push_str("\n\n");
            system.push_str(x.trim());
        }
        let mut wire = vec![WireMessage::new("system", system)];
        let mut i = 0;
        while i < self.items.len() {
            match &self.items[i] {
                Item::User(t) => wire.push(WireMessage::new("user", t.clone())),
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
                        let output = match &c.state {
                            CallState::Done { ok: true, output } => output.clone(),
                            CallState::Done { ok: false, output } => format!("Error: {output}"),
                            _ => "The call did not finish.".to_string(),
                        };
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
        let events = api::chat_stream_with_tools(env.base.to_string(), model.to_string(), wire, env.params, Some(self.tools_json.clone()));
        let (task, handle) = Task::stream(events).map(|ev| Message::Code(CodeMsg::Stream(ev))).abortable();
        self.handle = Some(handle.abort_on_drop());
        task
    }

    fn run(&mut self, base: &str, id: &str) -> Task<Message> {
        let Some(c) = self.call_mut(id) else { return Task::none() };
        let parsed = serde_json::from_str::<Value>(if c.args.trim().is_empty() { "{}" } else { &c.args });
        match parsed {
            Ok(args) => {
                c.state = CallState::Running;
                let (base, name, id) = (base.to_string(), c.name.clone(), id.to_string());
                Task::perform(run_tool(base, name, args), move |r| Message::Code(CodeMsg::ToolDone(id.clone(), r)))
            }
            Err(e) => {
                c.state = CallState::Done { ok: false, output: format!("The arguments are not valid JSON ({e}). Call the tool again with a JSON object.") };
                Task::none()
            }
        }
    }

    /// The model answered: its calls become cards, and run at once or wait for approval.
    fn finish_round(&mut self, env: &Env) -> Task<Message> {
        let partial: Vec<Partial> = std::mem::take(&mut self.acc).into_iter().filter(|p| !p.name.is_empty()).collect();
        if partial.is_empty() {
            self.running = false;
            self.handle = None;
            return Task::none();
        }
        let mut tasks = Vec::new();
        let mut ids = Vec::new();
        for (n, p) in partial.into_iter().enumerate() {
            let id = if p.id.is_empty() { format!("call_{}_{n}", self.steps) } else { p.id };
            let tool = self.tool(&p.name).cloned();
            let permission = tool.as_ref().map_or(Permission::Other, |t| t.permission);
            let state = if tool.is_none() { CallState::Done { ok: false, output: format!("There is no tool called {}.", p.name) } } else if self.mode.allows(permission) { CallState::Running } else { CallState::Pending };
            let run_now = state == CallState::Running;
            self.items.push(Item::Call(Call { id: id.clone(), name: p.name, args: p.args, permission, state }));
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
        if !self.running || self.last_round().any(|c| matches!(c.state, CallState::Pending | CallState::Running)) {
            return Task::none();
        }
        self.start_stream(env)
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
            CodeMsg::Close => {
                self.workspace = None;
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
                    Task::perform(fetch_changes(base), |r| Message::Code(CodeMsg::ChangesLoaded(r))),
                ])
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
                Task::none()
            }
            CodeMsg::ChangesLoaded(Err(_)) => Task::none(),
            CodeMsg::Edit(action) => {
                self.editor.perform(action);
                Task::none()
            }
            CodeMsg::Send => {
                let prompt = self.editor.text().trim().to_string();
                if prompt.is_empty() || self.running || self.workspace.is_none() || env.model.is_none() {
                    return Task::none();
                }
                self.editor = Content::new();
                self.error = None;
                self.steps = 0;
                self.items.push(Item::User(prompt));
                self.running = true;
                self.start_stream(env)
            }
            CodeMsg::Stop => {
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
                    async move {
                        cancel(base.clone()).await;
                        fetch_changes(base).await
                    },
                    |r| Message::Code(CodeMsg::ChangesLoaded(r)),
                )
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
                        Task::none()
                    }
                    StreamEvent::Reasoning(_) | StreamEvent::Usage { .. } | StreamEvent::Timings { .. } => Task::none(),
                }
            }
            CodeMsg::Approve(id) => {
                if !self.call_mut(&id).is_some_and(|c| c.state == CallState::Pending) {
                    return Task::none();
                }
                let run = self.run(env.base, &id);
                Task::batch([run, self.continue_if_done(env)])
            }
            CodeMsg::Deny(id) => {
                if let Some(c) = self.call_mut(&id).filter(|c| c.state == CallState::Pending) {
                    c.state = CallState::Done { ok: false, output: "The user declined this call. Do not repeat it; ask what to do instead or take another way.".into() };
                }
                self.continue_if_done(env)
            }
            CodeMsg::ToolDone(id, result) => {
                let changed = {
                    let Some(c) = self.call_mut(&id) else { return Task::none() };
                    if !matches!(c.state, CallState::Running) {
                        return Task::none();
                    }
                    let changes_files = matches!(c.permission, Permission::Edit | Permission::Command);
                    c.state = match result {
                        Ok(output) => CallState::Done { ok: true, output },
                        Err(output) => CallState::Done { ok: false, output },
                    };
                    changes_files
                };
                let next = self.continue_if_done(env);
                if changed {
                    Task::batch([next, Task::perform(fetch_changes(env.base.to_string()), |r| Message::Code(CodeMsg::ChangesLoaded(r)))])
                } else {
                    next
                }
            }
            CodeMsg::SetMode(m) => {
                self.mode = m;
                // calls that wait now may be allowed
                let ids: Vec<String> = self.items.iter().filter_map(|i| if let Item::Call(c) = i { (c.state == CallState::Pending && m.allows(c.permission)).then(|| c.id.clone()) } else { None }).collect();
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
                    Ok(text) => self.items.push(Item::Note(text)),
                    Err(e) => self.error = Some(e),
                }
                Task::perform(fetch_changes(env.base.to_string()), |r| Message::Code(CodeMsg::ChangesLoaded(r)))
            }
            CodeMsg::NewSession => {
                if self.running {
                    return Task::none();
                }
                self.items.clear();
                self.steps = 0;
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
    if c.state == CallState::Pending {
        body = body.push(row![
            button(text("Allow").size(13)).on_press(Message::Code(CodeMsg::Approve(c.id.clone()))).padding([5, 16]).style(theme::primary_button(t)),
            button(text("Deny").size(13)).on_press(Message::Code(CodeMsg::Deny(c.id.clone()))).padding([5, 16]).style(theme::secondary_button(t)),
        ]
        .spacing(8));
    }
    container(body).padding(12).width(Length::Fill).style(theme::card(t, theme::R_MD, c.state == CallState::Pending)).into()
}

fn timeline<'a>(s: &'a CodeState, t: Tokens, theme: &iced::Theme) -> Element<'a, Message> {
    let mut col = column![].spacing(12).width(Length::Fill);
    if s.items.is_empty() {
        col = col.push(container(text("Tell the agent what to do in this folder: explain the code, find a bug, add a test, rename something.").size(14).color(t.muted)).padding([24, 4]));
    }
    for (i, item) in s.items.iter().enumerate() {
        let element: Element<'a, Message> = match item {
            Item::User(text_) => row![Space::new().width(Length::Fill), container(text(text_.as_str()).size(15)).padding([9, 15]).max_width(560).style(theme::bubble(t))].into(),
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

pub fn page<'a>(s: &'a CodeState, t: Tokens, theme: &iced::Theme, can_send: bool) -> Element<'a, Message> {
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
            row![input, button(text("Open").size(14)).on_press(Message::Code(CodeMsg::Open)).padding([8, 18]).style(theme::primary_button(t))].spacing(10).align_y(Alignment::Center),
        ]
        .spacing(12)
        .max_width(560);
        if let Some(e) = &s.error {
            card = card.push(text(e.as_str()).size(13).color(t.danger));
        }
        return container(card).width(Length::Fill).height(Length::Fill).center_x(Length::Fill).center_y(Length::Fill).padding(24).into();
    };

    let mode = segmented(t, vec![("Ask", s.mode == Mode::Ask, Message::Code(CodeMsg::SetMode(Mode::Ask))), ("Edit freely", s.mode == Mode::AutoEdit, Message::Code(CodeMsg::SetMode(Mode::AutoEdit))), ("Everything", s.mode == Mode::Auto, Message::Code(CodeMsg::SetMode(Mode::Auto)))]);
    let branch = w.branch.as_deref().map(|b| format!(" · {b}")).unwrap_or_default();
    let header = row![
        column![text(w.name.as_str()).size(15).font(bold()), text(format!("{}{}", w.root, branch)).size(12).color(t.muted).font(Font::MONOSPACE)].spacing(2).width(Length::Fill),
        mode,
        button(text("New session").size(13)).on_press_maybe((!s.running).then_some(Message::Code(CodeMsg::NewSession))).padding([5, 14]).style(theme::secondary_button(t)),
        button(text("Other folder").size(13)).on_press_maybe((!s.running).then_some(Message::Code(CodeMsg::Close))).padding([5, 14]).style(theme::secondary_button(t)),
    ]
    .spacing(10)
    .align_y(Alignment::Center);
    let header = container(container(header).max_width(820).width(Length::Fill)).center_x(Length::Fill).padding([10, 24]);

    let mut bar = column![].spacing(6);
    if let Some(e) = &s.error {
        bar = bar.push(text(e.as_str()).size(13).color(t.danger));
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
        .placeholder(if can_send { "Tell the agent what to do" } else { "Load a model first" })
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
    let action = if s.running {
        button(text("Stop").size(13)).on_press(Message::Code(CodeMsg::Stop)).padding([6, 16]).style(theme::primary_button(t))
    } else {
        button(text("Send").size(13)).on_press_maybe((can_send && !s.editor.text().trim().is_empty()).then_some(Message::Code(CodeMsg::Send))).padding([6, 16]).style(theme::primary_button(t))
    };
    let composer = container(column![editor, row![text("Enter sends, Shift+Enter adds a new line").size(12).color(t.muted), Space::new().width(Length::Fill), action].align_y(Alignment::Center).padding(Padding { top: 0.0, right: 10.0, bottom: 8.0, left: 16.0 })])
        .style(theme::card(t, theme::R_XL, false))
        .max_width(820);
    let composer = container(composer).width(Length::Fill).center_x(Length::Fill).padding(Padding { top: 8.0, right: 24.0, bottom: 20.0, left: 24.0 });
    let bar = container(container(bar).max_width(820).width(Length::Fill)).center_x(Length::Fill).padding(Padding { top: 0.0, right: 24.0, bottom: 0.0, left: 24.0 });

    column![header, rule(t), timeline(s, t, theme), bar, composer].width(Length::Fill).height(Length::Fill).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Tool> {
        ["read_file:read", "grep:read", "edit_file:edit", "write_file:edit", "bash:command", "web_fetch:network"]
            .iter()
            .map(|s| {
                let (n, p) = s.split_once(':').unwrap();
                Tool { name: n.into(), permission: Permission::parse(p) }
            })
            .collect()
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
        Env { base: "http://127.0.0.1:9", model: Some("m"), params: ChatParams::default(), extra: None }
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
