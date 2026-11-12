//! The window: state, messages, update and view.

use crate::api::{self, ModelInfo, StreamEvent, WireMessage};
use crate::store::{self, Record, Role, Stats, Store};
use crate::chatlist::{self, ChatList, ListMsg};
use crate::md;
use crate::prompts;
use crate::code::{self, CodeState};
use crate::engine_settings::{self, EngineState};
use crate::models::{self, ModelsState};
use crate::settings;
use crate::theme::{self, Tokens};
use iced::widget::text_editor::{Action, Binding, Content, KeyPress, Status};
use iced::widget::{button, column, container, markdown, mouse_area, row, scrollable, stack, svg, text, text_editor, Space};
use iced::{keyboard, task, time, Alignment, Element, Font, Length, Padding, Subscription, Task, Theme};
use std::path::PathBuf;
use std::time::Duration;

const STARTERS: [&str; 4] = [
    "Explain in simple words how a language model writes a text word by word.",
    "Write a fast Rust function for the dot product of two f32 slices.",
    "Help me plan my week: ask me what I need to get done.",
    "Summarize a topic I name in five bullet points.",
];

pub(crate) fn icon(bytes: &'static [u8]) -> svg::Handle {
    svg::Handle::from_memory(bytes)
}

const ICON_PLUS: &[u8] = include_bytes!("../icons/plus.svg");
const ICON_SEND: &[u8] = include_bytes!("../icons/arrow-up.svg");
const ICON_STOP: &[u8] = include_bytes!("../icons/stop.svg");
const ICON_CHAT: &[u8] = include_bytes!("../icons/chat.svg");
const ICON_CUBE: &[u8] = include_bytes!("../icons/cube.svg");
const ICON_SETTINGS: &[u8] = include_bytes!("../icons/settings.svg");
/// How many of the newest messages a chat draws before "Show earlier messages" asks for more.
pub const CHAT_WINDOW: usize = 60;

pub(crate) const ICON_MORE: &[u8] = include_bytes!("../icons/more.svg");
pub(crate) const ICON_PIN: &[u8] = include_bytes!("../icons/pin.svg");
const ICON_CODE: &[u8] = include_bytes!("../icons/code.svg");
const ICON_CHEVRON: &[u8] = include_bytes!("../icons/chevron-down.svg");
const ICON_CHECK: &[u8] = include_bytes!("../icons/check.svg");

/// A message with its rendered Markdown.
pub struct Msg {
    pub role: Role,
    pub content: String,
    pub reasoning: String,
    pub stats: Option<Stats>,
    pub md: markdown::Content,
}

impl Msg {
    fn from_record(r: &Record) -> Msg {
        Msg { role: r.role, content: r.content.clone(), reasoning: r.reasoning.clone(), stats: r.stats.clone(), md: markdown::Content::parse(&r.content) }
    }

    fn record(&self) -> Record {
        Record { role: self.role, content: self.content.clone(), reasoning: self.reasoning.clone(), stats: self.stats.clone() }
    }
}

#[derive(Debug, Clone)]
pub enum Server {
    Connecting,
    Ready(ModelInfo),
    /// The server runs but no model is loaded: the Models page is where to go.
    NoModel,
    Down(String),
}

/// The page on the right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Chat,
    Models,
    Settings,
    Code,
}

/// The reply that is being written right now.
struct Live {
    chat: u64,
    _handle: task::Handle,
}

pub struct App {
    base: String,
    data_file: PathBuf,
    store: Store,
    system_dark: bool,
    server: Server,
    current: Option<u64>,
    /// The messages of the open chat.
    messages: Vec<Msg>,
    editor: Content,
    live: Option<Live>,
    /// Tokens used by the open chat (prompt + answer of the last reply), for the context meter.
    used_tokens: u64,
    /// Whether `used_tokens` was counted by the server (else it is an estimate and the meter shows a tilde).
    used_exact: bool,
    error: Option<String>,
    /// The chat row under the pointer (shows its delete button).
    hover: Option<u64>,
    page: View,
    models: ModelsState,
    /// The text box of the custom system prompt, and the text of the "longest answer" box.
    system_editor: Content,
    max_tokens_text: String,
    engine: EngineState,
    /// The list of models under the model name in the top bar.
    picker_open: bool,
    code: CodeState,
    /// How many of the newest messages the conversation draws: laying out hundreds of messages on every frame of a stream would stall the window.
    window: usize,
    list: ChatList,
    /// A line for the user that stays until it is clicked or the next message is sent ("Saved to ...").
    toast: Option<String>,
    /// The clock the model was told in this chat (held for 30 minutes, so that the start of the prompt stays the same and can be cached).
    clock: Option<(u64, i64)>,
    sampling_text: settings::SamplingTexts,
    /// The message of yours that the box is changing (the answer to it is written again as a new version).
    editing: Option<usize>,
    /// The server this app started itself (stopped when the app ends), and why it did not, when it did not.
    sidecar: Option<crate::sidecar::Sidecar>,
    sidecar_note: Option<String>,
    /// A load, unload or restart is under way: the server is away for a moment and the window says so instead of "no server".
    switching: bool,
}

#[derive(Debug, Clone)]
pub enum Message {
    Connected(Result<ModelInfo, String>),
    Retry,
    Tick,
    NewChat,
    OpenChat(u64),
    DeleteChat(u64),
    Edit(Action),
    Send,
    Stop,
    Starter(usize),
    Stream(u64, StreamEvent),
    Copy(usize),
    CopyText(String),
    Hover(Option<u64>),
    ToggleTheme,
    LinkClicked(markdown::Uri),
    Go(View),
    /// Once a second while the Models page is open or a download runs.
    Poll,
    OverviewLoaded(Result<models::Overview, String>),
    DownloadsLoaded(Result<Vec<models::Download>, String>),
    Filter(String),
    PickQuant(String, String),
    LoadModel(String),
    UnloadModel,
    Pull(String),
    DownloadAction(u64, &'static str),
    /// A load, unload, download or download action is done (or failed with this message).
    ModelAction(Result<String, String>),
    SamplingManual(bool),
    Temperature(f32),
    TopP(f32),
    SystemCustom(bool),
    SystemEdit(Action),
    MaxTokensText(String),
    /// Write the answer settings to the disk (after a slider is let go, a box is submitted, a choice is made).
    SavePrefs,
    ResetPrefs,
    EngineLoaded(Result<(engine_settings::EngineData, engine_settings::ServerData), String>),
    /// A switch or a choice changed: written at once.
    EngineSet(String, serde_json::Value),
    EngineDraft(String, String),
    /// Enter in a number or text box.
    EngineSubmit(String),
    ServerSet(String, serde_json::Value),
    /// An engine or server setting was written (or refused with this message).
    EngineSaved(Result<(), String>),
    EngineReset,
    RestartServer,
    RestartDone(Result<(), String>),
    TogglePicker,
    ClosePicker,
    /// Draw the next older messages of a long chat too.
    ShowEarlier,
    List(ListMsg),
    DismissToast,
    /// The thinking chip: the next level of the model.
    CycleThinking,
    PromptLength(crate::prompts::Length),
    SystemAppend(bool),
    SamplingField(settings::Field, String),
    /// Write the answer at this position again; the old one stays as a version.
    Regenerate(usize),
    /// Change this message of yours: it goes into the box.
    EditMessage(usize),
    CancelEdit,
    /// Show another version of the message at this position.
    SwitchVersion(usize, usize),
    Code(code::CodeMsg),
}

impl App {
    pub fn new() -> (App, Task<Message>) {
        let data = store::data_file();
        let (base, sidecar, note) = crate::sidecar::choose(&data.with_file_name("server.log"));
        let (mut app, task) = App::with(base, data);
        app.sidecar = sidecar;
        app.sidecar_note = note;
        // a server the app just started needs a moment: look every second until it answers
        app.switching = app.sidecar.is_some();
        (app, task)
    }

    /// Whether this app started the server it talks to.
    pub fn owns_server(&self) -> bool {
        self.sidecar.is_some()
    }

    pub fn with(base: String, data_file: PathBuf) -> (App, Task<Message>) {
        let store = Store::load(&data_file);
        let system_dark = std::env::var("FLLM_THEME").map(|v| v == "dark").unwrap_or(false);
        let mut app = App {
            base: base.clone(),
            data_file,
            store,
            system_dark,
            server: Server::Connecting,
            current: None,
            messages: Vec::new(),
            editor: Content::new(),
            live: None,
            used_tokens: 0,
            used_exact: false,
            error: None,
            hover: None,
            page: View::Chat,
            models: ModelsState::default(),
            system_editor: Content::new(),
            max_tokens_text: String::new(),
            engine: EngineState::default(),
            picker_open: false,
            code: CodeState::default(),
            window: CHAT_WINDOW,
            list: ChatList::default(),
            toast: None,
            clock: None,
            sampling_text: settings::SamplingTexts::default(),
            editing: None,
            switching: false,
            sidecar: None,
            sidecar_note: None,
        };
        app.system_editor = Content::with_text(&app.store.prefs.system_prompt);
        app.sampling_text = settings::SamplingTexts::from(&app.store.prefs.sampling);
        app.max_tokens_text = if app.store.prefs.max_tokens > 0 { app.store.prefs.max_tokens.to_string() } else { String::new() };
        if let Some(first) = app.store.chats.first().map(|c| c.id) {
            app.open(first);
        }
        (app, Task::perform(async move { api::fetch_model(&base).await }, Message::Connected))
    }

    /// What the server connection looks like now (for tests and for the status line).
    pub fn server_state(&self) -> &Server {
        &self.server
    }

    /// The id of the open chat, `None` for a chat that has not started.
    pub fn open_chat(&self) -> Option<u64> {
        self.current
    }

    /// The text of the last message of the open chat.
    pub fn last_text(&self) -> Option<&str> {
        self.messages.last().map(|m| m.content.as_str())
    }

    /// Whether a reply is being written.
    /// The text of the message at `i` of the open chat.
    pub fn message_text(&self, i: usize) -> Option<&str> {
        self.messages.get(i).map(|m| m.content.as_str())
    }

    /// What is in the message box.
    pub fn editor_text(&self) -> String {
        self.editor.text()
    }

    pub fn is_streaming(&self) -> bool {
        self.live.is_some()
    }

    /// How many messages the open chat shows.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    fn dark(&self) -> bool {
        match self.store.theme.as_str() {
            "dark" => true,
            "light" => false,
            _ => self.system_dark,
        }
    }

    pub fn tokens(&self) -> Tokens {
        Tokens::new(self.dark())
    }

    pub fn theme(&self) -> Theme {
        self.tokens().iced_theme()
    }

    fn save(&self) {
        if let Err(e) = self.store.save(&self.data_file) {
            eprintln!("fllm-app: the chats could not be saved: {e}");
        }
    }

    fn open(&mut self, id: u64) {
        if let Some(chat) = self.store.chat(id) {
            self.messages = chat.messages.iter().map(Msg::from_record).collect();
            self.window = CHAT_WINDOW;
            self.editing = None;
            self.current = Some(id);
            self.used_tokens = estimate_tokens(&self.messages);
            self.used_exact = false;
            self.error = None;
        }
    }

    /// Writes the open chat back to the store.
    fn sync_current(&mut self) {
        if let Some(id) = self.current {
            let records: Vec<Record> = self.messages.iter().map(Msg::record).collect();
            if let Some(chat) = self.store.chat_mut(id) {
                chat.messages = records;
                chat.updated = chrono::Local::now().timestamp();
            }
        }
    }

    fn can_send(&self) -> bool {
        matches!(self.server, Server::Ready(_)) && self.live.is_none() && !self.editor.text().trim().is_empty()
    }

    fn send(&mut self, prompt: String) -> Task<Message> {
        let Server::Ready(model) = self.server.clone() else { return Task::none() };
        if self.live.is_some() || prompt.trim().is_empty() {
            return Task::none();
        }
        let id = match self.current {
            Some(id) => id,
            None => {
                let id = self.store.new_chat();
                self.current = Some(id);
                id
            }
        };
        if let Some(i) = self.editing.take() {
            // the changed message starts a new version of everything after it
            if i < self.messages.len() {
                self.sync_current();
                if let Some(chat) = self.store.chat_mut(id) {
                    chat.branch_at(i);
                }
                self.messages.truncate(i);
            }
        }
        let first = self.messages.is_empty();
        self.messages.push(Msg { role: Role::User, md: markdown::Content::parse(&prompt), content: prompt.clone(), reasoning: String::new(), stats: None });
        self.messages.push(Msg { role: Role::Assistant, content: String::new(), reasoning: String::new(), stats: None, md: markdown::Content::new() });
        if first {
            if let Some(chat) = self.store.chat_mut(id) {
                chat.title = store::title_from(&prompt);
            }
        }
        self.sync_current();
        self.save();
        self.error = None;
        self.editor = Content::new();

        self.stream(id, model.id)
    }

    /// Starts the answer to the messages as they are (the last one is the empty answer).
    fn stream(&mut self, id: u64, model: String) -> Task<Message> {
        self.hold_clock(id);
        let wire = self.wire_messages();
        let events = api::chat_stream(self.base.clone(), model, wire, self.chat_params());
        let (task, handle) = Task::stream(events).map(move |ev| Message::Stream(id, ev)).abortable();
        self.live = Some(Live { chat: id, _handle: handle.abort_on_drop() });
        task
    }

    /// Takes the shown conversation of a chat from the store again (after another version was chosen).
    fn reload_messages(&mut self, id: u64) {
        if let Some(chat) = self.store.chat(id) {
            self.messages = chat.messages.iter().map(Msg::from_record).collect();
            self.used_tokens = estimate_tokens(&self.messages);
            self.used_exact = false;
        }
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Connected(Ok(model)) => {
                self.server = Server::Ready(model);
                self.switching = false;
                Task::none()
            }
            Message::Connected(Err(e)) if e == models::NO_MODEL => {
                self.server = Server::NoModel;
                self.switching = false;
                Task::none()
            }
            Message::Code(msg) => {
                if let code::CodeMsg::Opened(Ok(w)) = &msg {
                    self.store.prefs.remember_folder(&w.root);
                    self.save();
                }
                let was_running = self.code.running;
                let ends_session = matches!(msg, code::CodeMsg::NewSession);
                let opened = if let code::CodeMsg::Opened(Ok(w)) = &msg { Some(w.root.clone()) } else { None };
                let model = if let Server::Ready(m) = &self.server { Some(m.id.clone()) } else { None };
                let extra = (self.store.prefs.custom_system && !self.store.prefs.system_prompt.trim().is_empty()).then(|| self.store.prefs.system_prompt.clone());
                let base = self.base.clone();
                let env = code::Env { base: &base, model: model.as_deref(), params: self.chat_params(), extra: extra.as_deref() };
                let task = self.code.update(msg, &env);
                // the conversation of a folder is kept: it comes back when the folder is opened again
                let file = self.data_file.with_file_name("code-sessions.json");
                if let Some(root) = opened {
                    if let Some(saved) = code::Sessions::load(&file).0.remove(&root) {
                        self.code.restore(saved);
                    }
                } else if ends_session || (was_running && !self.code.running) {
                    if let Some(w) = &self.code.workspace {
                        let mut all = code::Sessions::load(&file);
                        let saved = self.code.saved();
                        if saved.is_empty() {
                            all.0.remove(&w.root);
                        } else {
                            all.0.insert(w.root.clone(), saved);
                        }
                        if let Err(e) = all.save(&file) {
                            eprintln!("fllm-app: the code session could not be saved: {e}");
                        }
                    }
                }
                task
            }
            Message::Regenerate(i) => {
                if self.live.is_some() || !self.messages.get(i).is_some_and(|m| m.role == Role::Assistant) {
                    return Task::none();
                }
                let (Some(id), Server::Ready(model)) = (self.current, self.server.clone()) else { return Task::none() };
                self.sync_current();
                if let Some(chat) = self.store.chat_mut(id) {
                    chat.branch_at(i);
                }
                self.messages.truncate(i);
                self.messages.push(Msg { role: Role::Assistant, content: String::new(), reasoning: String::new(), stats: None, md: markdown::Content::new() });
                self.sync_current();
                self.save();
                self.error = None;
                self.editing = None;
                self.stream(id, model.id)
            }
            Message::EditMessage(i) => {
                if self.live.is_some() {
                    return Task::none();
                }
                if let Some(m) = self.messages.get(i).filter(|m| m.role == Role::User) {
                    self.editor = Content::with_text(&m.content);
                    self.editing = Some(i);
                }
                Task::none()
            }
            Message::CancelEdit => {
                if self.editing.take().is_some() {
                    self.editor = Content::new();
                }
                Task::none()
            }
            Message::SwitchVersion(i, k) => {
                if self.live.is_some() {
                    return Task::none();
                }
                let Some(id) = self.current else { return Task::none() };
                self.sync_current();
                if self.store.chat_mut(id).is_some_and(|c| c.switch_version(i, k)) {
                    self.reload_messages(id);
                    self.save();
                }
                Task::none()
            }
            Message::CycleThinking => {
                let m = self.model_info();
                self.store.prefs.thinking_level = m.thinking.next(&m.thinking.level_for(&self.store.prefs.thinking_level));
                self.save();
                Task::none()
            }
            Message::PromptLength(l) => {
                self.store.prefs.prompt_length = l;
                self.save();
                Task::none()
            }
            Message::SystemAppend(on) => {
                self.store.prefs.system_append = on;
                self.save();
                Task::none()
            }
            Message::SamplingField(field, typed) => {
                self.sampling_text.apply(field, typed, &mut self.store.prefs.sampling);
                Task::none()
            }
            Message::ShowEarlier => {
                self.window += CHAT_WINDOW;
                Task::none()
            }
            Message::TogglePicker => {
                self.picker_open = !self.picker_open;
                if self.picker_open && !self.models.fetched {
                    return Task::perform(models::fetch_overview(self.base.clone()), Message::OverviewLoaded);
                }
                Task::none()
            }
            Message::ClosePicker => {
                self.picker_open = false;
                Task::none()
            }
            Message::Connected(Err(e)) => {
                self.server = Server::Down(e);
                Task::none()
            }
            Message::Retry => {
                self.server = Server::Connecting;
                let base = self.base.clone();
                Task::perform(async move { api::fetch_model(&base).await }, Message::Connected)
            }
            Message::Tick => {
                // the server that the app started may have ended on its own (a port taken, a binary that refuses its arguments)
                if let Some(s) = self.sidecar.as_mut() {
                    if !s.running() {
                        self.sidecar = None;
                        self.switching = false;
                        self.sidecar_note = Some(format!("The server that the app started has ended. Its output is in {}.", self.data_file.with_file_name("server.log").display()));
                    }
                }
                if matches!(self.server, Server::Down(_)) {
                    return self.update(Message::Retry);
                }
                Task::none()
            }
            Message::NewChat => {
                if self.live.is_some() {
                    return Task::none();
                }
                self.current = None;
                self.messages.clear();
                self.window = CHAT_WINDOW;
                self.editing = None;
                self.used_tokens = 0;
                self.error = None;
                Task::none()
            }
            Message::OpenChat(id) => {
                if self.live.is_none() {
                    self.open(id);
                }
                Task::none()
            }
            Message::DeleteChat(id) => self.update(Message::List(ListMsg::Delete(id))),
            Message::DismissToast => {
                self.toast = None;
                Task::none()
            }
            Message::List(msg) => {
                let live = self.live.as_ref().map(|l| l.chat);
                let now = chrono::Local::now().timestamp();
                match self.list.update(msg, &mut self.store, live, now) {
                    chatlist::Effect::Nothing => {}
                    chatlist::Effect::Save => self.save(),
                    chatlist::Effect::Deleted(ids) => {
                        if self.current.is_some_and(|c| ids.contains(&c)) {
                            self.current = None;
                            self.messages.clear();
                        }
                        self.save();
                    }
                    chatlist::Effect::Toast(t) => self.toast = Some(t),
                }
                Task::none()
            }
            Message::Edit(action) => {
                self.editor.perform(action);
                Task::none()
            }
            Message::Send => {
                if !self.can_send() {
                    return Task::none();
                }
                let prompt = self.editor.text().trim().to_string();
                self.send(prompt)
            }
            Message::Starter(i) => {
                if matches!(self.server, Server::Ready(_)) {
                    return self.send(STARTERS[i].to_string());
                }
                Task::none()
            }
            Message::Stop => {
                self.finish();
                Task::none()
            }
            Message::Stream(chat, ev) => {
                if self.live.as_ref().map(|l| l.chat) != Some(chat) {
                    return Task::none();
                }
                self.on_event(ev);
                Task::none()
            }
            Message::Copy(i) => match self.messages.get(i) {
                Some(m) => iced::clipboard::write(m.content.clone()),
                None => Task::none(),
            },
            Message::CopyText(s) => iced::clipboard::write(s),
            Message::Hover(id) => {
                self.hover = id;
                Task::none()
            }
            Message::ToggleTheme => {
                self.store.theme = if self.dark() { "light" } else { "dark" }.into();
                self.save();
                Task::none()
            }
            Message::LinkClicked(uri) => {
                open_link(&uri);
                Task::none()
            }
            Message::Go(page) => {
                self.page = page;
                self.picker_open = false;
                if page == View::Models {
                    return self.refresh_models();
                }
                if page == View::Settings && !matches!(self.server, Server::Down(_)) {
                    return Task::perform(engine_settings::fetch(self.base.clone()), Message::EngineLoaded);
                }
                Task::none()
            }
            Message::EngineLoaded(Ok((engine, server))) => {
                self.engine.engine = engine;
                self.engine.server = server;
                self.engine.loaded = true;
                self.engine.error = None;
                self.engine.drafts.clear();
                Task::none()
            }
            Message::EngineLoaded(Err(e)) => {
                self.engine.error = Some(e);
                Task::none()
            }
            Message::EngineSet(key, value) => {
                if self.engine.locked(&key) {
                    return Task::none();
                }
                self.engine.engine.values.insert(key.clone(), value.clone());
                self.engine.restart_needed = true;
                Task::perform(engine_settings::save_engine(self.base.clone(), serde_json::json!({ key: value })), Message::EngineSaved)
            }
            Message::EngineDraft(key, text) => {
                self.engine.drafts.insert(key, text);
                Task::none()
            }
            Message::EngineSubmit(key) => {
                let Some(text) = self.engine.drafts.remove(&key) else { return Task::none() };
                if key == "idle_unload_minutes" {
                    return match text.trim().parse::<u64>() {
                        Ok(m) if m <= 10_080 => {
                            self.engine.server.idle_unload_minutes = m;
                            Task::perform(engine_settings::save_server(self.base.clone(), serde_json::json!({ key: m })), Message::EngineSaved)
                        }
                        _ => {
                            self.engine.error = Some("The idle time is a whole number of minutes from 0 (never) to 10080.".into());
                            Task::none()
                        }
                    };
                }
                let Some(field) = engine_settings::field(&key) else { return Task::none() };
                match engine_settings::value_from_text(field, &text) {
                    Ok(value) => {
                        self.engine.error = None;
                        self.engine.engine.values.insert(key.clone(), value.clone());
                        self.engine.restart_needed = true;
                        Task::perform(engine_settings::save_engine(self.base.clone(), serde_json::json!({ key: value })), Message::EngineSaved)
                    }
                    Err(e) => {
                        self.engine.drafts.insert(key, text);
                        self.engine.error = Some(e);
                        Task::none()
                    }
                }
            }
            Message::ServerSet(key, value) => {
                if key == "load_on_start" {
                    self.engine.server.load_on_start = value.as_bool().unwrap_or(false);
                }
                Task::perform(engine_settings::save_server(self.base.clone(), serde_json::json!({ key: value })), Message::EngineSaved)
            }
            Message::EngineSaved(Ok(())) => {
                self.engine.error = None;
                Task::none()
            }
            Message::EngineSaved(Err(e)) => {
                self.engine.error = Some(e);
                // what the page shows must be what the server has
                Task::perform(engine_settings::fetch(self.base.clone()), Message::EngineLoaded)
            }
            Message::EngineReset => {
                let body = self.engine.reset_body();
                if let Some(map) = body.as_object() {
                    for (k, v) in map {
                        self.engine.engine.values.insert(k.clone(), v.clone());
                    }
                }
                self.engine.drafts.clear();
                self.engine.restart_needed = true;
                Task::perform(engine_settings::save_engine(self.base.clone(), body), Message::EngineSaved)
            }
            Message::RestartServer => {
                self.engine.restarting = true;
                self.switching = true;
                Task::perform(engine_settings::restart(self.base.clone()), Message::RestartDone)
            }
            Message::RestartDone(result) => {
                self.engine.restarting = false;
                match result {
                    Ok(()) => {
                        // the server is gone for a moment: the window waits for it like for any server that is not there yet
                        self.engine.restart_needed = false;
                        self.engine.loaded = false;
                        self.server = Server::Connecting;
                        let base = self.base.clone();
                        Task::perform(async move { api::fetch_model(&base).await }, Message::Connected)
                    }
                    Err(e) => {
                        self.engine.error = Some(e);
                        Task::none()
                    }
                }
            }
            Message::Poll => {
                let base = self.base.clone();
                let mut tasks = vec![Task::perform(models::fetch_downloads(base.clone()), Message::DownloadsLoaded)];
                // the server may have switched its model since (a load or unload takes a moment, the Studio restarts its engine for it)
                if self.page == View::Models && self.live.is_none() {
                    tasks.push(Task::perform(async move { api::fetch_model(&base).await }, Message::Connected));
                }
                if self.page == View::Models && !self.models.any_active() {
                    tasks.push(Task::perform(models::fetch_overview(self.base.clone()), Message::OverviewLoaded));
                }
                Task::batch(tasks)
            }
            Message::OverviewLoaded(Ok(overview)) => {
                self.models.overview = overview;
                self.models.fetched = true;
                self.models.error = None;
                Task::none()
            }
            Message::OverviewLoaded(Err(e)) => {
                self.models.fetched = true;
                self.models.error = Some(e);
                Task::none()
            }
            Message::DownloadsLoaded(Ok(list)) => {
                // a download that finished: the file is in the list of this computer now
                let finished = self.models.downloads.iter().any(|old| old.active() && list.iter().any(|d| d.id == old.id && d.state == "done"));
                self.models.downloads = list;
                if finished {
                    return Task::perform(models::fetch_overview(self.base.clone()), Message::OverviewLoaded);
                }
                Task::none()
            }
            Message::DownloadsLoaded(Err(_)) => Task::none(),
            Message::Filter(text) => {
                self.models.filter = text;
                Task::none()
            }
            Message::PickQuant(id, quant) => {
                self.models.quant.insert(id, quant);
                Task::none()
            }
            Message::LoadModel(reference) => {
                if self.models.busy.is_some() || self.live.is_some() {
                    return Task::none();
                }
                self.models.busy = Some(reference.clone());
                self.models.error = None;
                self.picker_open = false;
                self.switching = true;
                Task::perform(models::load(self.base.clone(), reference), Message::ModelAction)
            }
            Message::UnloadModel => {
                if self.models.busy.is_some() || self.live.is_some() {
                    return Task::none();
                }
                self.models.busy = Some(String::new());
                self.models.error = None;
                self.picker_open = false;
                self.switching = true;
                Task::perform(models::unload(self.base.clone()), Message::ModelAction)
            }
            Message::Pull(id) => {
                let Some(entry) = self.models.overview.catalog.iter().find(|e| e.id == id) else { return Task::none() };
                let quant = self.models.chosen_quant(entry);
                self.models.error = None;
                Task::perform(models::pull(self.base.clone(), id, quant), Message::ModelAction)
            }
            Message::DownloadAction(id, action) => Task::perform(models::download_action(self.base.clone(), id, action), Message::ModelAction),
            Message::ModelAction(Ok(_)) => {
                self.models.busy = None;
                let base = self.base.clone();
                Task::batch([
                    Task::perform(models::fetch_overview(base.clone()), Message::OverviewLoaded),
                    Task::perform(models::fetch_downloads(base.clone()), Message::DownloadsLoaded),
                    Task::perform(async move { api::fetch_model(&base).await }, Message::Connected),
                ])
            }
            Message::ModelAction(Err(e)) => {
                self.models.busy = None;
                self.switching = false;
                self.models.error = Some(e);
                Task::none()
            }
            Message::SamplingManual(on) => {
                self.store.prefs.manual_sampling = on;
                self.save();
                Task::none()
            }
            Message::Temperature(v) => {
                self.store.prefs.temperature = v;
                Task::none()
            }
            Message::TopP(v) => {
                self.store.prefs.top_p = v;
                Task::none()
            }
            Message::SystemCustom(on) => {
                self.store.prefs.custom_system = on;
                self.save();
                Task::none()
            }
            Message::SystemEdit(action) => {
                let edit = action.is_edit();
                self.system_editor.perform(action);
                if edit {
                    self.store.prefs.system_prompt = self.system_editor.text().trim_end_matches('\n').to_string();
                    self.save();
                }
                Task::none()
            }
            Message::MaxTokensText(text) => {
                let digits: String = text.chars().filter(char::is_ascii_digit).take(6).collect();
                self.store.prefs.max_tokens = digits.parse().unwrap_or(0);
                self.max_tokens_text = digits;
                Task::none()
            }
            Message::SavePrefs => {
                self.save();
                Task::none()
            }
            Message::ResetPrefs => {
                self.store.prefs = store::Prefs::default();
                self.sampling_text = settings::SamplingTexts::default();
                self.system_editor = Content::new();
                self.max_tokens_text.clear();
                self.save();
                Task::none()
            }
        }
    }

    fn refresh_models(&self) -> Task<Message> {
        let base = self.base.clone();
        Task::batch([Task::perform(models::fetch_overview(base.clone()), Message::OverviewLoaded), Task::perform(models::fetch_downloads(base), Message::DownloadsLoaded)])
    }

    fn on_event(&mut self, ev: StreamEvent) {
        let Some(last) = self.messages.last_mut() else { return };
        match ev {
            StreamEvent::Delta(s) => {
                last.content.push_str(&s);
                last.md.push_str(&s);
            }
            StreamEvent::Reasoning(s) => last.reasoning.push_str(&s),
            StreamEvent::ToolCall { .. } => {}
            StreamEvent::Usage { prompt, completion } => {
                self.used_tokens = prompt + completion;
                self.used_exact = true;
                last.stats.get_or_insert_with(Stats::default).completion_tokens = completion;
            }
            StreamEvent::Timings { tokens_per_second, prompt_per_second, .. } => {
                let s = last.stats.get_or_insert_with(Stats::default);
                s.tokens_per_second = tokens_per_second;
                s.prompt_per_second = prompt_per_second;
            }
            StreamEvent::Done => self.finish(),
            StreamEvent::Error(e) => {
                self.error = Some(e);
                // an answer that never started is not kept
                if self.messages.last().is_some_and(|m| m.role == Role::Assistant && m.content.is_empty() && m.reasoning.is_empty()) {
                    self.messages.pop();
                }
                self.finish();
            }
        }
    }

    /// The reply is over (finished, stopped or failed): save it and free the composer.
    fn finish(&mut self) {
        if self.live.take().is_some() {
            self.sync_current();
            if let Some(chat) = self.current.and_then(|id| self.store.chat_mut(id)) {
                chat.forget_duplicates();
            }
            self.save();
        }
    }

    pub fn subscription(&self) -> Subscription<Message> {
        // a server that was told to switch its model is back within seconds: look more often then
        let every = if self.switching { 1 } else { 4 };
        let retry = if matches!(self.server, Server::Down(_)) { time::every(Duration::from_secs(every)).map(|_| Message::Tick) } else { Subscription::none() };
        let poll = if (self.page == View::Models && !matches!(self.server, Server::Down(_))) || self.models.any_active() {
            time::every(Duration::from_secs(1)).map(|_| Message::Poll)
        } else {
            Subscription::none()
        };
        Subscription::batch([retry, poll])
    }

    // ---- view ------------------------------------------------------------------------------------------------------------

    pub fn view(&self) -> Element<'_, Message> {
        let t = self.tokens();
        let mut main = column![self.main(t)];
        if let Some(line) = &self.toast {
            main = main.push(
                container(
                    button(row![text(line.as_str()).size(13).color(t.fg_soft).width(Length::Fill), text("Dismiss").size(12).color(t.muted)].spacing(12).align_y(Alignment::Center))
                        .on_press(Message::DismissToast)
                        .padding([8, 14])
                        .width(Length::Fill)
                        .style(theme::secondary_button(t)),
                )
                .padding([6, 24]),
            );
        }
        let body = row![self.sidebar(t), main.width(Length::Fill)];
        container(body).width(Length::Fill).height(Length::Fill).style(theme::page(t)).into()
    }

    fn sidebar(&self, t: Tokens) -> Element<'_, Message> {
        let brand = row![
            container(Space::new().width(10).height(10)).style(move |_| container::Style {
                background: Some(iced::Background::Color(t.primary)),
                border: iced::Border { radius: 5.0.into(), ..Default::default() },
                ..Default::default()
            }),
            text("fLLM").size(15).font(bold()),
        ]
        .spacing(8)
        .align_y(Alignment::Center);

        let new_chat = button(row![svg(icon(ICON_PLUS)).width(16).height(16).style(accent_svg(t)), text("New chat").size(14)].spacing(10).align_y(Alignment::Center))
            .on_press_maybe(self.live.is_none().then_some(Message::NewChat))
            .padding([7, 10])
            .width(Length::Fill)
            .style(theme::row_button(t, false));

        let chats = self.list.view(t, &self.store, self.current, self.hover, self.live.is_none(), chrono::Local::now().timestamp());

        let footer = container(
            row![
                container(Space::new().width(7).height(7)).style(move |_| container::Style {
                    background: Some(iced::Background::Color(if matches!(self.server, Server::Ready(_)) { t.ok } else { t.muted })),
                    border: iced::border::rounded(4),
                    ..Default::default()
                }),
                Space::new().width(8),
                text(self.status_line()).size(12).color(t.muted).width(Length::Fill),
                button(text(if self.dark() { "Switch to light" } else { "Switch to dark" }).size(12))
                    .on_press(Message::ToggleTheme)
                    .padding([3, 8])
                    .style(theme::ghost_button(t)),
            ]
            .align_y(Alignment::Center),
        )
        .padding(Padding { top: 0.0, right: 0.0, bottom: 0.0, left: 10.0 });

        let nav = column![
            nav_item(t, ICON_CHAT, "Chat", self.page == View::Chat, Some(Message::Go(View::Chat))),
            nav_item(t, ICON_CODE, "Code", self.page == View::Code, Some(Message::Go(View::Code))),
            nav_item(t, ICON_CUBE, "Models", self.page == View::Models, Some(Message::Go(View::Models))),
            nav_item(t, ICON_SETTINGS, "Settings", self.page == View::Settings, Some(Message::Go(View::Settings))),
        ]
        .spacing(2);

        container(
            column![
                container(brand).padding(Padding { top: 0.0, right: 10.0, bottom: 0.0, left: 10.0 }).height(52).align_y(Alignment::Center),
                new_chat,
                nav,
                container(text("Chats").size(12).font(bold()).color(t.muted)).padding(Padding { top: 16.0, right: 10.0, bottom: 4.0, left: 10.0 }),
                chats,
                footer,
            ]
            .spacing(4)
            .padding(12),
        )
        .width(248)
        .height(Length::Fill)
        .style(theme::sidebar(t))
        .into()
    }

    fn status_line(&self) -> String {
        match &self.server {
            Server::Connecting => "Connecting…".into(),
            Server::Ready(_) => "Connected".into(),
            Server::NoModel => "No model loaded".into(),
            Server::Down(_) => "No server".into(),
        }
    }


    fn main(&self, t: Tokens) -> Element<'_, Message> {
        if self.page == View::Models && !matches!(self.server, Server::Down(_)) {
            return models::page(&self.models, t, matches!(self.server, Server::NoModel));
        }
        if self.page == View::Code && !matches!(self.server, Server::Down(_)) {
            return code::page(&self.code, t, &self.theme(), matches!(self.server, Server::Ready(_)), &self.store.prefs.recent_folders);
        }
        if self.page == View::Settings {
            return settings::page(t, &self.store.prefs, &self.system_editor, &self.max_tokens_text, &self.sampling_text, engine_settings::view(t, &self.engine, !matches!(self.server, Server::Down(_))));
        }
        let content: Element<'_, Message> = match &self.server {
            Server::Down(_) | Server::Connecting if self.switching => self.switching_card(t),
            Server::Down(reason) => self.offline(t, reason),
            Server::NoModel => self.no_model(t),
            _ if self.messages.is_empty() => self.welcome(t),
            _ => self.conversation(t),
        };
        let page: Element<'_, Message> = column![self.topbar(t), content, self.composer(t)].width(Length::Fill).height(Length::Fill).into();
        if self.picker_open {
            return stack![page, mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::ClosePicker), self.picker(t)].into();
        }
        page
    }

    fn switching_card(&self, t: Tokens) -> Element<'_, Message> {
        let what = match self.models.busy.as_deref() {
            Some("") => "Unloading the model…".to_string(),
            Some(r) => format!("Loading {}…", r.rsplit(['/', '\\']).next().unwrap_or(r)),
            None => "The server is starting…".to_string(),
        };
        let card = column![text(what).size(22).font(bold()), text("The server restarts its engine for this. It takes a few seconds, larger models longer.").size(14).color(t.fg_soft)].spacing(10).max_width(520);
        container(card).width(Length::Fill).height(Length::Fill).center_x(Length::Fill).center_y(Length::Fill).padding(24).into()
    }

    /// The list of models that opens under the name in the top bar: the ones on this computer, the loaded one marked.
    fn picker(&self, t: Tokens) -> Element<'_, Message> {
        let loaded = match &self.server {
            Server::Ready(m) => Some(m.id.as_str()),
            _ => None,
        };
        let mut list = column![].spacing(2);
        if self.models.overview.installed.is_empty() {
            list = list.push(container(text(if self.models.fetched { "No models downloaded yet." } else { "Looking…" }).size(13).color(t.muted)).padding([8, 10]));
        }
        for m in &self.models.overview.installed {
            let is_loaded = loaded.is_some_and(|l| m.name.starts_with(l) || l.starts_with(&m.name)) || self.models.overview.loaded.as_deref() == Some(m.name.as_str());
            let mark: Element<'_, Message> = if is_loaded { svg(icon(ICON_CHECK)).width(14).height(14).style(accent_svg(t)).into() } else { Space::new().width(14).height(14).into() };
            list = list.push(
                button(row![mark, text(ellipsize(&m.name, 34)).size(13).font(Font::MONOSPACE).width(Length::Fill), text(models::size(m.bytes)).size(12).color(t.muted)].spacing(10).align_y(Alignment::Center))
                    .on_press(if is_loaded { Message::ClosePicker } else { Message::LoadModel(m.reference.clone()) })
                    .padding([7, 10])
                    .width(Length::Fill)
                    .style(theme::row_button(t, false)),
            );
        }
        let manage = button(text("Manage models…").size(13).color(t.fg_soft)).on_press(Message::Go(View::Models)).padding([7, 10]).width(Length::Fill).style(theme::row_button(t, false));
        let panel = container(column![scrollable(list).height(Length::Shrink), container(Space::new().height(1)).width(Length::Fill).style(move |_| container::Style { background: Some(iced::Background::Color(t.separator)), ..Default::default() }), manage].spacing(6))
            .padding(6)
            .width(380)
            .max_height(360)
            .style(theme::card(t, theme::R_LG, false));
        // under the model name: the top bar is 52 high and the column is centered like the chat
        container(container(panel).max_width(780).width(Length::Fill)).center_x(Length::Fill).padding(Padding { top: 46.0, right: 24.0, bottom: 0.0, left: 24.0 }).width(Length::Fill).height(Length::Fill).into()
    }

    fn topbar(&self, t: Tokens) -> Element<'_, Message> {
        let (name, max) = match &self.server {
            Server::Ready(m) => (m.id.clone(), m.max_context),
            Server::Connecting => ("Connecting…".into(), 0),
            Server::NoModel | Server::Down(_) => ("No model".into(), 0),
        };
        let pill = button(row![svg(icon(ICON_CUBE)).width(15).height(15).style(accent_svg(t)), text(name).size(13).font(Font::MONOSPACE), svg(icon(ICON_CHEVRON)).width(13).height(13).style(muted_svg(t))].spacing(8).align_y(Alignment::Center))
            .on_press_maybe((!matches!(self.server, Server::Down(_)) && self.live.is_none()).then_some(Message::TogglePicker))
            .padding([5, 12])
            .style(move |_, status| {
                let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                button::Style { background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })), text_color: t.fg_soft, border: iced::border::rounded(999), ..Default::default() }
            });
        let meter: Element<'_, Message> = if max > 0 {
            let ratio = (self.used_tokens as f32 / max as f32).clamp(0.0, 1.0);
            let used_part = ((ratio * 1000.0).round() as u16).max(if self.used_tokens > 0 { 8 } else { 0 });
            let free_part = 1000 - used_part.min(1000);
            let mut bar = row![].width(80).height(4);
            if used_part > 0 {
                bar = bar.push(container(Space::new()).width(Length::FillPortion(used_part)).height(4).style(move |_| container::Style {
                    background: Some(iced::Background::Color(if ratio > 0.8 { t.danger } else { t.primary })),
                    border: iced::border::rounded(2),
                    ..Default::default()
                }));
            }
            if free_part > 0 {
                bar = bar.push(Space::new().width(Length::FillPortion(free_part)).height(4));
            }
            row![
                container(bar).style(move |_| container::Style { background: Some(iced::Background::Color(t.fill_hover)), border: iced::border::rounded(2), ..Default::default() }),
                text(format!("{}{} / {}", if self.used_exact || self.used_tokens == 0 { "" } else { "~" }, group(self.used_tokens), group(max))).size(12).color(t.muted),
            ]
            .spacing(10)
            .align_y(Alignment::Center)
            .into()
        } else {
            Space::new().into()
        };
        // the thinking chip: only for a model that can think; a click steps to the next level
        let model = self.model_info();
        let chip: Element<'_, Message> = if model.thinking.supported() {
            let level = self.thinking_level();
            let (label, _help) = model.thinking.chip(&level);
            let active = level != "off";
            let can_switch = model.thinking.control != api::ThinkControl::Always && self.live.is_none();
            button(text(label).size(13))
                .on_press_maybe(can_switch.then_some(Message::CycleThinking))
                .padding([5, 12])
                .style(move |_, status| {
                    let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                    button::Style {
                        background: Some(iced::Background::Color(if active { t.primary_soft } else if hovered { t.fill_hover } else { t.fill })),
                        text_color: if active { t.primary } else { t.fg_soft },
                        border: iced::border::rounded(999),
                        ..Default::default()
                    }
                })
                .into()
        } else {
            Space::new().into()
        };
        let inner = container(row![pill, Space::new().width(10), chip, Space::new().width(Length::Fill), meter].align_y(Alignment::Center)).max_width(780).width(Length::Fill);
        container(inner)
            .center_x(Length::Fill)
            .padding([0, 24])
            .height(52)
            .align_y(Alignment::Center)
            .style(move |_| container::Style {
                border: iced::Border { color: t.separator, width: 0.0, radius: 0.0.into() },
                ..Default::default()
            })
            .into()
    }

    fn welcome(&self, t: Tokens) -> Element<'_, Message> {
        let model = match &self.server {
            Server::Ready(m) => format!("Running locally on this computer · {} tokens of context.", group(m.max_context)),
            _ => "Connecting to the server…".into(),
        };
        let name = match &self.server {
            Server::Ready(m) => m.id.clone(),
            _ => "fLLM".into(),
        };
        let mut starters = column![].spacing(10);
        for pair in STARTERS.chunks(2).enumerate() {
            let (row_i, chunk) = pair;
            let mut r = row![].spacing(10);
            for (j, s) in chunk.iter().enumerate() {
                let i = row_i * 2 + j;
                r = r.push(
                    button(text(*s).size(13).color(t.fg_soft))
                        .on_press_maybe(matches!(self.server, Server::Ready(_)).then_some(Message::Starter(i)))
                        .padding([12, 14])
                        .width(Length::FillPortion(1))
                        .style(starter_style(t)),
                );
            }
            starters = starters.push(r);
        }
        let hero = column![
            text(name).size(30).font(bold()),
            text(model).size(14).color(t.muted),
            Space::new().height(18),
            starters,
        ]
        .spacing(6)
        .align_x(Alignment::Center)
        .max_width(680);
        container(hero).width(Length::Fill).height(Length::Fill).center_x(Length::Fill).center_y(Length::Fill).padding(24).into()
    }

    fn offline<'a>(&'a self, t: Tokens, reason: &'a str) -> Element<'a, Message> {
        let card = column![
            text("No fLLM server found").size(22).font(bold()),
            text(format!("The window needs a running server at {}. Start one with `fllm-rs studio` (or `fllm-rs serve --model …`), or point the window at another with FLLM_URL.", self.base)).size(14).color(t.fg_soft),
            text(reason).size(12).color(t.muted),
            text(self.sidecar_note.clone().unwrap_or_default()).size(12).color(t.muted),
            button(text("Try again").size(14)).on_press(Message::Retry).padding([8, 18]).style(theme::primary_button(t)),
        ]
        .spacing(10)
        .align_x(Alignment::Start)
        .max_width(520);
        container(card).width(Length::Fill).height(Length::Fill).center_x(Length::Fill).center_y(Length::Fill).padding(24).into()
    }

    fn no_model(&self, t: Tokens) -> Element<'_, Message> {
        let card = column![
            text("No model is loaded").size(22).font(bold()),
            text("The server is running, but it has nothing to answer with. Pick a model you have already, or download one.").size(14).color(t.fg_soft),
            button(text("Open Models").size(14)).on_press(Message::Go(View::Models)).padding([8, 18]).style(theme::primary_button(t)),
        ]
        .spacing(10)
        .align_x(Alignment::Start)
        .max_width(520);
        container(card).width(Length::Fill).height(Length::Fill).center_x(Length::Fill).center_y(Length::Fill).padding(24).into()
    }

    /// The messages of the next request: the system prompt, then the chat without the empty answer that is about to be written.
    pub fn wire_messages(&self) -> Vec<WireMessage> {
        let mut wire = vec![WireMessage::new("system", self.system_prompt())];
        let upto = match self.messages.last() {
            Some(last) if last.role == Role::Assistant && last.content.is_empty() => self.messages.len() - 1,
            _ => self.messages.len(),
        };
        wire.extend(self.messages[..upto].iter().map(|m| WireMessage::new(m.role.wire(), m.content.clone())));
        wire
    }

    /// The sampling settings the next request carries: nothing while they are automatic.
    pub fn chat_params(&self) -> api::ChatParams {
        let p = &self.store.prefs;
        let mut extra = if p.manual_sampling { p.sampling.fields() } else { serde_json::Map::new() };
        let model = self.model_info();
        for (k, v) in model.thinking.fields(&model.thinking.level_for(&p.thinking_level)) {
            extra.insert(k, v);
        }
        api::ChatParams {
            temperature: p.manual_sampling.then_some(p.temperature),
            top_p: p.manual_sampling.then_some(p.top_p),
            max_tokens: (p.max_tokens > 0).then_some(p.max_tokens),
            extra,
        }
    }

    /// The language of the computer, as a tag (`de-DE`).
    pub fn locale() -> String {
        for var in ["LC_ALL", "LC_MESSAGES", "LANG"] {
            if let Ok(v) = std::env::var(var) {
                let tag = v.split('.').next().unwrap_or("").replace('_', "-");
                if !tag.is_empty() && tag != "C" && tag != "POSIX" {
                    return tag;
                }
            }
        }
        "en".into()
    }

    /// The model in use (a default one while none is loaded).
    pub fn model_info(&self) -> ModelInfo {
        match &self.server {
            Server::Ready(m) => m.clone(),
            _ => ModelInfo::default(),
        }
    }

    /// The level of the thinking chip for the loaded model.
    pub fn thinking_level(&self) -> String {
        self.model_info().thinking.level_for(&self.store.prefs.thinking_level)
    }

    /// How fast the model read its last prompt (tokens per second; 0 when unknown).
    fn last_prompt_speed(&self) -> f64 {
        self.messages.iter().rev().find_map(|m| m.stats.as_ref().map(|s| s.prompt_per_second).filter(|v| *v > 0.0)).unwrap_or(0.0)
    }

    /// The system prompt the next request carries: built from the facts about the model and the computer, then your text applied.
    pub fn system_prompt(&self) -> String {
        let p = &self.store.prefs;
        let model = self.model_info();
        let now_secs = chrono::Local::now().timestamp();
        let held = self.clock.filter(|(c, _)| Some(*c) == self.current).map(|(_, t)| t);
        let now = prompts::local_time(prompts::frozen_clock(held, now_secs));
        let locale = Self::locale();
        let level = model.thinking.level_for(&p.thinking_level);
        let facts = prompts::Facts {
            size: prompts::size(model.max_context, model.detail.parameters, p.prompt_length, self.last_prompt_speed()),
            family: prompts::family(&model.detail),
            now,
            locale: locale.clone(),
            local: true,
            model_name: model.detail.name.clone(),
            thinking: model.thinking.supported() && level != "off",
            tools: prompts::Tools::default(),
        };
        prompts::resolve(&prompts::build(&facts), &[(p.system_mode(), p.system_prompt.as_str())], &now, &locale)
    }

    /// The clock of the open chat is held from its first request for 30 minutes.
    fn hold_clock(&mut self, id: u64) {
        let now = chrono::Local::now().timestamp();
        let held = self.clock.filter(|(c, _)| *c == id).map(|(_, t)| t);
        self.clock = Some((id, prompts::frozen_clock(held, now)));
    }

    /// The answer settings.
    pub fn prefs(&self) -> &store::Prefs {
        &self.store.prefs
    }

    /// The page on the right now.
    pub fn page(&self) -> View {
        self.page
    }

    /// The file the chats are kept in.
    pub fn data_file(&self) -> &std::path::Path {
        &self.data_file
    }

    /// The list of chats in the sidebar.
    pub fn chat_list(&self) -> &ChatList {
        &self.list
    }

    /// The line for the user, when there is one.
    pub fn toast(&self) -> Option<&str> {
        self.toast.as_deref()
    }

    /// The state of Code mode.
    pub fn code(&self) -> &CodeState {
        &self.code
    }

    /// The engine and server settings as the page shows them.
    pub fn engine(&self) -> &EngineState {
        &self.engine
    }

    /// Whether the list of models under the model name is open.
    pub fn picker_open(&self) -> bool {
        self.picker_open
    }

    /// Whether a load, unload or restart is under way.
    pub fn switching(&self) -> bool {
        self.switching
    }

    /// The state of the Models page.
    pub fn models(&self) -> &ModelsState {
        &self.models
    }

    fn conversation(&self, t: Tokens) -> Element<'_, Message> {
        let mut col = column![].spacing(22).max_width(780).width(Length::Fill);
        let start = self.messages.len().saturating_sub(self.window);
        if start > 0 {
            let more = start.min(CHAT_WINDOW);
            col = col.push(
                container(button(text(format!("Show {more} earlier message{}", if more == 1 { "" } else { "s" })).size(13)).on_press(Message::ShowEarlier).padding([6, 16]).style(theme::secondary_button(t))).width(Length::Fill).center_x(Length::Fill),
            );
        }
        for (i, m) in self.messages.iter().enumerate().skip(start) {
            col = col.push(self.message(t, i, m));
        }
        if let Some(e) = &self.error {
            col = col.push(container(text(e).size(13).color(t.danger)).padding([8, 12]).style(theme::inset(t)));
        }
        scrollable(container(col.padding([20, 0])).width(Length::Fill).center_x(Length::Fill))
            .anchor_bottom()
            .height(Length::Fill)
            .into()
    }

    /// The arrows under a message that has more than one version: "‹ 2 / 3 ›".
    fn version_switch(&self, t: Tokens, i: usize) -> Option<Element<'_, Message>> {
        let v = self.current.and_then(|id| self.store.chat(id))?.versions_at(i);
        if v.count < 2 {
            return None;
        }
        let step = |label: &'static str, to: Option<usize>| button(text(label).size(13)).on_press_maybe(to.filter(|_| self.live.is_none()).map(|k| Message::SwitchVersion(i, k))).padding([0, 6]).style(theme::ghost_button(t));
        Some(
            row![step("‹", v.current.checked_sub(1)), text(format!("{} / {}", v.current + 1, v.count)).size(12).color(t.muted), step("›", (v.current + 1 < v.count).then_some(v.current + 1))]
                .spacing(2)
                .align_y(Alignment::Center)
                .into(),
        )
    }

    fn message<'a>(&'a self, t: Tokens, i: usize, m: &'a Msg) -> Element<'a, Message> {
        match m.role {
            Role::User => {
                let bubble = container(text(&m.content).size(15)).padding([9, 15]).max_width(560).style(theme::bubble(t));
                let mut line = row![].spacing(14).align_y(Alignment::Center);
                if let Some(v) = self.version_switch(t, i) {
                    line = line.push(v);
                }
                if self.live.is_none() {
                    line = line.push(button(text("Edit").size(12)).on_press(Message::EditMessage(i)).padding([3, 0]).style(theme::ghost_button(t)));
                }
                column![row![Space::new().width(Length::Fill), bubble], row![Space::new().width(Length::Fill), line]].spacing(4).into()
            }
            Role::Assistant => {
                let live = self.live.is_some() && i + 1 == self.messages.len();
                let mut col = column![].spacing(8);
                if !m.reasoning.is_empty() {
                    col = col.push(text(format!("Thought process · {} characters", m.reasoning.chars().count())).size(12).color(t.muted));
                }
                if !m.content.is_empty() {
                    col = col.push(md::view(&m.md, t, &self.theme()));
                } else if live {
                    col = col.push(text("…").size(15).color(t.muted));
                }
                if !live && !m.content.is_empty() {
                    let mut meta = row![button(text("Copy answer").size(12)).on_press(Message::Copy(i)).padding([3, 0]).style(theme::ghost_button(t))].spacing(14).align_y(Alignment::Center);
                    meta = meta.push(button(text("Retry").size(12)).on_press(Message::Regenerate(i)).padding([3, 0]).style(theme::ghost_button(t)));
                    if let Some(v) = self.version_switch(t, i) {
                        meta = meta.push(v);
                    }
                    if let Some(s) = &m.stats {
                        if s.tokens_per_second > 0.0 {
                            meta = meta.push(
                                text(format!("{:.1} tokens/s · {} tokens{}", s.tokens_per_second, s.completion_tokens, if s.prompt_per_second > 0.0 { format!(" · prompt {:.0} tokens/s", s.prompt_per_second) } else { String::new() }))
                                    .size(12)
                                    .color(t.muted),
                            );
                        }
                    }
                    col = col.push(meta);
                }
                col.into()
            }
        }
    }

    fn composer(&self, t: Tokens) -> Element<'_, Message> {
        let editor = text_editor(&self.editor)
            .placeholder("Ask anything or give an instruction")
            .on_action(Message::Edit)
            .key_binding(|kp: KeyPress| match kp.key.as_ref() {
                keyboard::Key::Named(keyboard::key::Named::Enter) if !kp.modifiers.shift() && matches!(kp.status, Status::Focused { .. }) => Some(Binding::Custom(Message::Send)),
                _ => Binding::from_key_press(kp),
            })
            .padding(Padding { top: 14.0, right: 16.0, bottom: 6.0, left: 16.0 })
            .size(15)
            .height(Length::Shrink)
            .min_height(44)
            .max_height(180)
            .style(move |_, _| text_editor::Style {
                background: iced::Background::Color(iced::Color::TRANSPARENT),
                border: iced::Border { width: 0.0, ..Default::default() },
                placeholder: t.muted,
                value: t.fg,
                selection: iced::Color { a: 0.35, ..t.primary },
            });
        let busy = self.live.is_some();
        let action = if busy {
            button(svg(icon(ICON_STOP)).width(14).height(14).style(move |_, _| svg::Style { color: Some(t.on_primary) }))
                .on_press(Message::Stop)
                .padding(10)
                .style(theme::primary_button(t))
        } else {
            button(svg(icon(ICON_SEND)).width(16).height(16).style(move |_, _| svg::Style { color: Some(if self.can_send() { t.on_primary } else { t.muted }) }))
                .on_press_maybe(self.can_send().then_some(Message::Send))
                .padding(9)
                .style(theme::primary_button(t))
        };
        let hint = text("Enter sends, Shift+Enter adds a new line").size(12).color(t.muted);
        let banner: Element<'_, Message> = if self.editing.is_some() {
            row![text("Changing a message: the answers after it are written again, the old ones stay as a version.").size(12).color(t.muted).width(Length::Fill), button(text("Cancel").size(12)).on_press(Message::CancelEdit).padding([2, 8]).style(theme::ghost_button(t))]
                .align_y(Alignment::Center)
                .padding(Padding { top: 8.0, right: 10.0, bottom: 0.0, left: 16.0 })
                .into()
        } else {
            Space::new().height(0).into()
        };
        let card = container(column![banner, editor, row![hint, Space::new().width(Length::Fill), action].align_y(Alignment::Center).padding(Padding { top: 0.0, right: 10.0, bottom: 10.0, left: 16.0 })])
            .style(theme::card(t, theme::R_XL, false))
            .max_width(780);
        container(card).width(Length::Fill).center_x(Length::Fill).padding(Padding { top: 8.0, right: 24.0, bottom: 20.0, left: 24.0 }).into()
    }
}

/// A rough count of the tokens of a chat (four characters to a token) until the server has counted the real ones. The meter says so with a tilde.
fn estimate_tokens(messages: &[Msg]) -> u64 {
    messages.iter().map(|m| m.content.chars().count() as u64 / 4 + 4).sum()
}

/// `text` cut to `max` characters with an ellipsis (iced has no text overflow of its own).
pub fn ellipsize(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", cut.trim_end())
}

/// The interface font: Inter, shipped inside the program, so the window looks the same on every system (and in the headless tests).
pub const UI_FONT: Font = Font::with_name("Inter");
pub const FONT_REGULAR: &[u8] = include_bytes!("../assets/fonts/Inter-Regular.otf");
pub const FONT_SEMIBOLD: &[u8] = include_bytes!("../assets/fonts/Inter-SemiBold.otf");

fn bold() -> Font {
    Font { weight: iced::font::Weight::Semibold, ..UI_FONT }
}

/// 8192 -> "8,192".
pub fn group(n: u64) -> String {
    let digits = n.to_string();
    let mut parts: Vec<&str> = digits.as_bytes().rchunks(3).map(|c| std::str::from_utf8(c).unwrap_or("")).collect();
    parts.reverse();
    parts.join(",")
}

pub(crate) fn muted_svg(t: Tokens) -> impl Fn(&Theme, svg::Status) -> svg::Style {
    move |_, _| svg::Style { color: Some(t.muted) }
}

fn accent_svg(t: Tokens) -> impl Fn(&Theme, svg::Status) -> svg::Style {
    move |_, _| svg::Style { color: Some(t.primary) }
}

fn nav_item<'a>(t: Tokens, bytes: &'static [u8], label: &'a str, selected: bool, on_press: Option<Message>) -> Element<'a, Message> {
    button(row![svg(icon(bytes)).width(16).height(16).style(muted_svg(t)), text(label).size(14)].spacing(10).align_y(Alignment::Center))
        .on_press_maybe(on_press)
        .padding([7, 10])
        .width(Length::Fill)
        .style(theme::nav_button(t, selected))
        .into()
}

fn starter_style(t: Tokens) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| button::Style {
        background: Some(iced::Background::Color(match status {
            button::Status::Hovered | button::Status::Pressed => t.fill,
            _ => t.surface,
        })),
        text_color: t.fg_soft,
        border: iced::Border { color: t.separator, width: 0.5, radius: theme::R_LG.into() },
        ..Default::default()
    }
}

/// Opens a link in the system browser.
fn open_link(uri: &str) {
    if !(uri.starts_with("http://") || uri.starts_with("https://")) {
        return;
    }
    #[cfg(target_os = "linux")]
    let cmd = "xdg-open";
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    let cmd = "explorer";
    let _ = std::process::Command::new(cmd).arg(uri).spawn();
}

pub fn run() -> iced::Result {
    iced::application(App::new, App::update, App::view)
        .title("fLLM")
        .font(FONT_REGULAR)
        .font(FONT_SEMIBOLD)
        .default_font(UI_FONT)
        .theme(App::theme)
        .subscription(App::subscription)
        .window_size((1280.0, 820.0))
        .antialiasing(true)
        .run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_grouped_in_thousands() {
        assert_eq!([0, 7, 999, 1000, 8192, 32768, 1234567].map(group), ["0", "7", "999", "1,000", "8,192", "32,768", "1,234,567"]);
    }

    #[test]
    fn long_titles_are_cut_with_a_real_ellipsis() {
        assert_eq!(ellipsize("short", 26), "short");
        let cut = ellipsize("Explain in simple words how a language model writes", 26);
        assert!(cut.ends_with('…') && cut.chars().count() <= 26, "{cut}");
        assert_eq!(ellipsize("äöüäöü", 4), "äöü…");
    }

    #[test]
    fn the_estimate_is_four_characters_to_a_token_plus_a_little_for_the_frame() {
        let m = |c: &str| Msg { role: Role::User, content: c.into(), reasoning: String::new(), stats: None, md: markdown::Content::new() };
        assert_eq!(estimate_tokens(&[m(&"a".repeat(40))]), 14);
        assert_eq!(estimate_tokens(&[]), 0);
    }
}
