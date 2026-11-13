//! The window: state, messages, update and view.

use crate::api::{self, ModelInfo, StreamEvent, WireMessage};
use crate::assistants::{self, Assistant, SysMode};
use crate::memory;
use crate::promptlib;
use crate::palette;
use crate::shortcuts;
use crate::speech;
use crate::tour;
use crate::artifacts;
use crate::research;
use crate::tools;
use crate::rag::{self, IndexedDoc};
use crate::store::{self, Record, Role, Stats, Store};
use crate::chatlist::{self, ChatList, ListMsg};
use crate::md;
use crate::prompts;
use crate::code::{self, CodeState};
use crate::engine_settings::{self, EngineState};
use crate::models::{self, ModelsState};
use crate::settings;
use crate::theme::{self, Tokens};
use iced::widget::text_editor::{Action, Binding, Content, KeyPress, Motion, Status};
use iced::widget::{button, column, container, markdown, mouse_area, row, scrollable, stack, svg, text, text_editor, text_input, Space};
use iced::{keyboard, task, time, Alignment, Element, Font, Length, Padding, Subscription, Task, Theme};
use chrono::TimeZone;
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
const ICON_LINES: &[u8] = include_bytes!("../icons/lines.svg");
const ICON_GHOST: &[u8] = include_bytes!("../icons/ghost.svg");
const ICON_GLOBE: &[u8] = include_bytes!("../icons/globe.svg");
const ICON_SEARCH: &[u8] = include_bytes!("../icons/search.svg");
const ICON_CHECK: &[u8] = include_bytes!("../icons/check.svg");
const ICON_FOLDER: &[u8] = include_bytes!("../icons/folder.svg");
const ICON_CHEVRON_RIGHT: &[u8] = include_bytes!("../icons/chevron-right.svg");
const ICON_X: &[u8] = include_bytes!("../icons/x.svg");
const ICON_FILE: &[u8] = include_bytes!("../icons/file.svg");

/// A message with its rendered Markdown.
pub struct Msg {
    pub role: Role,
    pub content: String,
    pub reasoning: String,
    pub stats: Option<Stats>,
    /// The passages of the documents that went with this question (shown under it).
    pub passages: Option<Vec<crate::rag::Passage>>,
    /// The images of this question as data URLs and the files whose text went in whole.
    pub images: Vec<String>,
    pub files: Vec<store::MsgFile>,
    /// On an assistant message: the tool calls it made, as the model sent them.
    pub tool_calls: Option<serde_json::Value>,
    /// On a tool message: the call it answers.
    pub tool_call_id: Option<String>,
    /// What the tools did under the answer (the Studio's node.tools).
    pub tool_cards: Vec<crate::memory::ToolCard>,
    /// The sources of an answer that used the web tools (`ctSourcesHtml` under it).
    pub sources: Vec<tools::Source>,
    /// The research run of this message, with its log (the Studio's `node.research`).
    pub research: Option<research::ResearchUi>,
    /// The words of this answer with the probability the model gave them (`node.logprobs`).
    pub logprobs: Vec<api::Logprob>,
    /// Whether the reasoning of this message is unfolded (the `open` of the Studio's details box;
    /// while the answer has not started it is shown open regardless).
    pub reasoning_open: bool,
    pub md: markdown::Content,
}

impl Msg {
    fn from_record(r: &Record) -> Msg {
        Msg {
            role: r.role,
            content: r.content.clone(),
            reasoning: r.reasoning.clone(),
            stats: r.stats.clone(),
            passages: r.passages.clone(),
            images: r.images.clone(),
            files: r.files.clone(),
            tool_calls: r.tool_calls.clone(),
            tool_call_id: r.tool_call_id.clone(),
            tool_cards: r.tool_cards.clone(),
            sources: r.sources.clone(),
            research: r.research.clone(),
            logprobs: r.logprobs.clone(),
            reasoning_open: false,
            md: markdown::Content::parse(&r.content),
        }
    }

    fn record(&self) -> Record {
        Record {
            role: self.role,
            content: self.content.clone(),
            reasoning: self.reasoning.clone(),
            stats: self.stats.clone(),
            passages: self.passages.clone(),
            images: self.images.clone(),
            files: self.files.clone(),
            tool_calls: self.tool_calls.clone(),
            tool_call_id: self.tool_call_id.clone(),
            tool_cards: self.tool_cards.clone(),
            sources: self.sources.clone(),
            research: self.research.clone(),
            logprobs: self.logprobs.clone(),
        }
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
    /// The page of one project (the Studio's project window).
    Project,
}

/// What a row of the command palette runs (`run` of the items of `palette.js`).
#[derive(Debug, Clone, PartialEq)]
pub enum PaletteAction {
    NewChat,
    GoChat,
    GoCode,
    GoModels,
    GoSettings,
    ToggleTheme,
    Shortcuts,
    Tour,
    Regenerate,
    ExportMarkdown(u64),
    ExportJson(u64),
    OpenChat(u64),
}

impl PaletteAction {
    /// The message that does the work of the row.
    fn message(&self) -> Message {
        match self {
            PaletteAction::NewChat => Message::NewChat,
            PaletteAction::GoChat => Message::Go(View::Chat),
            PaletteAction::GoCode => Message::Go(View::Code),
            PaletteAction::GoModels => Message::Go(View::Models),
            PaletteAction::GoSettings => Message::Go(View::Settings),
            PaletteAction::ToggleTheme => Message::ToggleTheme,
            PaletteAction::Shortcuts => Message::ShortcutsOpen,
            PaletteAction::Tour => Message::TourShow,
            PaletteAction::Regenerate => Message::Retry,
            PaletteAction::ExportMarkdown(id) => {
                Message::List(ListMsg::Export(*id, chatlist::Format::Markdown))
            }
            PaletteAction::ExportJson(id) => {
                Message::List(ListMsg::Export(*id, chatlist::Format::Json))
            }
            PaletteAction::OpenChat(id) => Message::OpenChat(*id),
        }
    }
}

/// What the copy of the memory on the server gave back (the Studio's memImportFromServer and memSyncSoon).
#[derive(Debug, Clone)]
pub enum PersonalSync {
    /// The entries of the server's copy (a first run without a local one).
    Import(Vec<serde_json::Value>),
    /// The text that is on the server now.
    Stored(String),
}

/// A file or image chosen for the next question (the Studio's `state.attachments`).
#[derive(Debug, Clone, PartialEq)]
pub enum Attachment {
    /// An image as a data URL, shrunk the way the Studio shrinks one.
    Image { name: String, data_url: String },
    /// Text that goes into the question whole because it fits the context.
    Text { name: String, text: String, size: u64 },
    /// Text that does not fit: the passages that match the question are used instead.
    Doc { name: String, text: String, size: u64, pages: Option<u64>, truncated: bool },
}

/// The search index of the project knowledge, with the signature of the documents it was built from.
struct RagCache {
    sig: String,
    index: rag::Index,
}

/// The reply that is being written right now.
struct Live {
    chat: u64,
    /// When the run began, so a long one can be announced when it ends (the `watchRun` of
    /// `voice.js` measures the same stretch).
    started: std::time::Instant,
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
    /// The schema, grammar or pattern of the output format of the sampling panel.
    spec_editor: Content,
    max_tokens_text: String,
    engine: EngineState,
    /// The list of models under the model name in the top bar.
    picker_open: bool,
    /// The assistant menu under the assistant name in the top bar, and the editor window of one assistant.
    asst_menu: bool,
    asst_edit: Option<assistants::Editor>,
    /// The model of a just-chosen assistant: applied when the list of models arrives (it is fetched on demand).
    asst_model_pending: Option<String>,
    /// The project page that is open, the projects whose chats show in the sidebar and the chats to move.
    prj_view: Option<String>,
    prj_open: std::collections::BTreeSet<String>,
    prj_pick: Option<Vec<u64>>,
    /// The project whose deletion waits for the *yes* of the dialog (the Studio's askDialog).
    prj_delete: Option<String>,
    /// The project a new chat joins (the Studio's prj.pending) and the row for a new project.
    prj_pending: Option<String>,
    prj_creating: bool,
    prj_new_name: String,
    /// The project being renamed in the sidebar: its id and the typed name.
    prj_renaming: Option<(String, String)>,
    /// The instructions box of the project page.
    prj_instr: Content,
    /// The search index of the project knowledge, rebuilt when the documents change.
    rag_cache: Option<RagCache>,
    /// The questions whose passage chip is open (the Studio's <details> under the message).
    rag_open: std::collections::BTreeSet<usize>,
    /// The images and files chosen for the next question (the Studio's `state.attachments`).
    attachments: Vec<Attachment>,
    /// The *About me* box of Settings, its text as typed.
    personal_about: Content,
    /// The row to add a memory and the memory whose text is being changed.
    mem_new: String,
    mem_edit: Option<(String, String)>,
    /// The dialog that forgets everything, and the dialog of the *Remember* button.
    mem_clear: bool,
    remember: Option<usize>,
    remember_text: String,
    /// The menu under a "/" in the box: which saved prompts are listed and which one is highlighted.
    slash: Vec<usize>,
    slash_index: usize,
    /// Whether the menu of the response styles is open over the composer.
    style_menu: bool,
    /// The style staged for the next new chat (the Studio's `state.newStyle`).
    new_style: String,
    /// The next chat is temporary (the Studio's `state.tempNext`).
    temp_next: bool,
    /// The dialog before a temporary chat with messages is gone for good.
    temp_close: bool,
    /// The welcome tour: the step that is open (None = it is closed; the Studio's `tourModal`).
    pub tour: Option<usize>,
    /// What the engine is doing right now: the status of the hero (`updateHero` of the Studio).
    pub hero: Option<models::Status>,
    /// The tokens per second of the last moments: the samples of the two-minute line.
    hero_samples: Vec<f64>,
    /// The last counters of the server's own clock, for the rate between two moments.
    hero_last: Option<(u64, f64)>,
    /// The models on this computer for the start screen (asked once, like `heroHubAsked`).
    hero_overview: Option<models::Overview>,
    hero_hub_asked: bool,
    /// The prices per million tokens (the `prices` of the Studio's usage pane).
    pub prices: Vec<crate::usage::Price>,
    /// What the last press of the usage pane said ("Prices saved." ...).
    pub usage_status: Option<String>,
    /// What the tour has to say (the line about this computer, a model's memory, the loaded model
    /// and the suggestions of the hub).
    pub tour_data: tour::TourData,
    /// The overview of the keyboard shortcuts (the `helpModal` of the Studio).
    pub shortcuts_open: bool,
    /// The shortcut that waits for its new keys (the *Change* button; a second press cancels).
    pub sc_capture: Option<String>,
    /// The last line the panel says (*Saved.*, a refused combination, *All shortcuts ... defaults*).
    pub sc_message: String,
    /// Whether the window is in front (the alerts of `voice.js` stay silent while it is: the
    /// `announce` of `voice.js` returns when the document has focus and is not hidden).
    pub window_focused: bool,
    /// When the agent of Code mode began its run, so a long one can be announced when it ends
    /// (the `watchRun('btnCodeSend')` of `voice.js`, five seconds there).
    pub code_started: Option<std::time::Instant>,
    /// The voices of this computer, for the select of the settings (empty until it was asked for
    /// or when no speech program is installed).
    pub voice_names: Vec<String>,
    /// Whether this computer can show a system notification.
    pub voice_can_notify: bool,
    /// The command palette (`searchModal` of `shell.html`): open, what was typed, which row is
    /// active and the rows in the order they are shown.
    pub palette_open: bool,
    pub palette_query: String,
    pub palette_active: i32,
    pub palette_rows: Vec<(palette::PaletteItem, PaletteAction)>,
    /// The notice under the messages: how many messages a rewind hides, until when it stands.
    rewind_notice: Option<(usize, u64)>,
    /// Whether the model may use the web tools of the server (the Studio's `state.useWeb`: a
    /// choice of the window, not kept between starts).
    use_web: bool,
    /// The room for web results of the answer that is being written (`ctBudget`, one per answer).
    answer_budget: Option<tools::Budget>,
    /// The sources of the answer that is being written, so the model can cite them as `[n]`.
    answer_sources: Vec<tools::Source>,
    /// The round of tool calls that is running: every call in its order with the result (None
    /// until it came), and which of them the server still has to answer.
    tool_hold: Vec<Option<(String, String, crate::memory::ToolCard)>>,
    tool_web_jobs: Vec<(usize, String, String, String)>,
    /// The chip in the composer: the question becomes a research project (btnToggleResearch).
    research: bool,
    /// Whether the server refused the JSON schema of the plan (the run asks plainly then).
    research_no_schema: bool,
    /// The sources of the report that is streaming: added to the answer when it is over.
    research_sources: Option<Vec<(u32, String, String, String)>>,
    /// The artifact panel: which family and version it shows (the Studio's `ART` state).
    art: Option<artifacts::Panel>,
    /// The JavaScript runs that are going or standing under their code block, by the code that runs.
    runs: std::collections::HashMap<String, artifacts::CodeRun>,
    /// The text box of a prompt library row per id (so a half-typed row keeps its place).
    plib_texts: std::collections::HashMap<String, Content>,
    /// The copy on the server: the text that is up there, whether it is off, and one request at a time.
    mem_synced_text: Option<String>,
    mem_sync_off: bool,
    mem_syncing: bool,
    /// How many rounds of tool calls the answer of this question has already done.
    tool_round: u8,
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
    /// The copy as the model wrote it (the Studio's copy-md).
    CopyMd(usize),
    /// The answer read aloud by the computer's own voice (the Studio's speak).
    Speak(usize),
    CopyText(String),
    /// The conversation ends at this message; nothing is deleted, Undo brings the rest back.
    RewindTo(usize),
    RewindUndo,
    /// The chip in the composer: the model may search the web and read pages (btnToggleWeb).
    ToggleWeb,
    /// One round of web tool calls came back from the server.
    ToolRound(ToolRoundResult),
    /// The Esc key: it ends a running answer or a research run (the title of the Studio's send
    /// button says "Stop the answer (Esc)").
    Esc,
    /// The reasoning of a message is shown or folded away (the `details` box of the Studio).
    ReasoningToggle(usize),
    /// The import of chats beside the select mode (the btnImportThreads of the Studio).
    ImportChats,
    /// The chosen file of that import: its path and its text.
    ImportChosen(Result<(String, Option<String>), String>),
    /// Open a fenced block of an answer in the artifact panel (the Open/Preview button of the
    /// block): the code of it, and the info string of its fence.
    ArtOpen(String, String),
    /// A button of the artifact panel (the `data-art` buttons of the Studio).
    ArtAction(artifacts::ArtAction),
    /// Run the JavaScript of a code block in the sandboxed runner (the Run button of the block).
    RunCode(String),
    /// Stop the run that is going under this block (the same button while it runs).
    RunStop(String),
    /// One message of the runner for this block (the `worker.onmessage` of the Studio).
    RunEvent { code: String, ev: artifacts::RunEvent },
    /// Open an address of the sources in the browser of the computer.
    OpenUrl(String),
    /// The chip in the composer: the question becomes a research project (btnToggleResearch).
    ResearchToggle,
    /// How the research run goes right now (the progress line above the answer).
    ResearchProgress(Box<research::ResearchUi>),
    /// The run prepared the report — or gave up and the model answers on its own.
    ResearchDone(Box<research::Prepared>),
    /// The log under the answer of a research run, opened or closed.
    ResearchLogToggle(usize),
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
    /// The shape the answer must hold to (the Studio's selConstraint).
    SamplingConstraint(String),
    /// The schema, grammar or pattern of the shape above (the Studio's txtConstraintSpec).
    SamplingSpec(String),
    /// The editor of that specification was written in.
    SamplingSpecEdit(Action),
    /// Whether every answer shows its token probabilities (the Studio's chkLogprobs).
    SamplingLogprobs(bool),
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
    /// The assistant menu under the name in the top bar.
    AsstMenuToggle,
    AsstMenuClose,
    /// Choose the assistant for this chat (or for new chats when none is open).
    AsstApply(String),
    AsstNew,
    /// Edit an assistant; a built-in one is edited as a copy.
    AsstEditOpen(String),
    AsstSaveCurrent,
    AsstClose,
    AsstSave,
    AsstDelete,
    AsstExport,
    AsstImport,
    /// The chosen export file: its text, or why it could not be read.
    AsstImported(Result<String, String>),
    AsstName(String),
    AsstSystem(Action),
    AsstStarters(Action),
    AsstSysText(Action),
    AsstTemp(String),
    AsstTopP(String),
    AsstTopK(String),
    AsstModel(String),
    AsstModelToggle,
    AsstSysMode(SysMode),
    AsstSysToggle,
    AsstIcon(String),
    AsstColor(String),
    /// The projects: the rows in the sidebar, the page, its documents and the picker of a move.
    PrjNew,
    PrjNewInput(String),
    /// `true` keeps the typed name.
    PrjNewEnd(bool),
    PrjRenameStart(String),
    PrjRenameInput(String),
    PrjRenameEnd(bool),
    /// Asks whether the project really goes away (the Studio's askDialog).
    PrjDelete(String),
    PrjDeleteCancel,
    PrjDeleteOk,
    PrjToggle(String),
    PrjOpen(String),
    PrjClose,
    PrjInstr(Action),
    PrjNewChat,
    PrjChat(u64),
    PrjAddFiles,
    /// The dialog of *Add files*: its path with the text when the file is text, or why it failed.
    PrjFileChosen(Result<(String, Option<String>), String>),
    /// The server read a binary document: its name, its text, its pages and whether it was cut short.
    PrjExtracted(Result<(String, String, Option<u64>, bool), String>),
    PrjDocRemove(String),
    /// The picker of a project: `Some("")` means *No project*, `None` cancels.
    PrjPick(Option<String>),
    /// Shows or hides the passages under the question at `i` (the chip of the Studio's ragChipsHtml).
    RagToggle(usize),
    /// Attaches an image or a file to the next question (the + of the composer).
    Attach,
    /// The dialog of *Attach* with its path and the text when the file is UTF-8 text.
    AttachChosen(Result<(String, Option<String>), String>),
    /// The image is read and shrunk: its name and its data URL.
    ImageReady(Result<(String, String), String>),
    /// The server read a document: its name, its text, its pages, whether it was cut short and the size of the file.
    AttachExtracted(Result<(String, String, Option<u64>, bool, u64), String>),
    /// Files dropped onto the window.
    FilesDropped(Vec<PathBuf>),
    /// Removes the pending attachment at `i`.
    AttachmentRemove(usize),
    /// Removes the document with this id from the knowledge of this chat.
    ChatDocRemove(String),
    /// The *About me* box of Settings (the Studio's memAbout).
    AboutEdit(Action),
    UseAbout(bool),
    UseMemory(bool),
    /// The row to add a memory (memNew and memAdd).
    MemNewInput(String),
    MemAddNow,
    /// A memory row: click to change it, Enter or the tick saves, the cross deletes (memList).
    MemEditStart(String),
    MemEditInput(String),
    MemEditEnd(bool),
    MemDelete(String),
    /// Forget everything (memClear), asked for first like the Studio.
    MemClear,
    MemClearOk,
    MemClearCancel,
    /// The *Remember* button under a message (prjDecorateMessages).
    RememberOpen(usize),
    RememberEdit(String),
    RememberOk,
    RememberCancel,
    /// The copy of the memory on the server: what it gave back, or that there is none.
    PersonalSynced(api::SyncAnswer<PersonalSync>),
    /// Takes the change of a memory tool card back (the Undo under an answer).
    MemoryUndo(usize, usize),
    /// The library of the settings.
    PromptAdd,
    PromptDel(String),
    /// A saved prompt's title as it is typed (clamped to the limit).
    PromptTitle(String, String),
    /// A saved prompt's text: its box and the action of the box.
    PromptText(String, Action),
    /// The menu under a "/" in the box.
    SlashPick(usize),
    SlashUp,
    SlashDown,
    SlashEnter,
    SlashClose,
    /// The menu of the response styles over the composer.
    StyleMenu,
    StylePick(String),
    /// The button of the temporary chat in the top bar (the Studio's toggleTemporary).
    TempToggle,
    TempCloseOk,
    TempCloseCancel,
    /// The welcome tour opens (the first start, or *Show the welcome tour* of Settings).
    TourShow,
    /// The status of the engine came back (the hero of the start screen).
    HeroPoll,
    /// The status arrived; the samples of the line take it.
    HeroStatus(Result<models::Status, String>),
    /// The models on this computer for the start screen.
    HeroOverview(Result<models::Overview, String>),
    /// Load the model the start screen offers first.
    HeroLoad(String),
    /// Download the model the start screen offers (the recommended one when nothing is here).
    HeroDownload { model: String, quant: String },
    /// A model file was chosen for the start screen (file and text, like the project's dialog).
    HeroChosen(Result<(String, Option<String>), String>),
    /// The start screen points to the models page.
    HeroBrowse,
    /// The start screen asks for a model file (`Choose a model file…`).
    HeroChoose,
    /// The counters of the usage pane are set back to zero (`Reset counters`).
    UsageReset,
    /// The price rows came from the server (`loadPrices` of the Studio).
    PricesLoaded(Result<Vec<crate::usage::Price>, String>),
    /// The price rows were written back (`savePrices` of the Studio).
    PricesSaved(Result<Vec<crate::usage::Price>, String>),
    /// A price row was added (`+ Add price`).
    PriceAdd,
    /// A price row was removed (`Remove`).
    PriceRemove(usize),
    /// One field of a price row was typed (`data-f` of the Studio).
    PriceField(usize, &'static str, String),
    /// What the usage pane says under its buttons (`say` of the Studio).
    UsageStatus(String),
    /// The price rows are written back (`Save prices` of the Studio).
    UsageSave,
    /// One step forward; the last step closes it (*Get started*).
    TourNext,
    /// One step back.
    TourBack,
    /// The tour closes (*Skip*, or the click beside it).
    TourSkip,
    /// *Open Models* on the third step closes the tour and goes to the Models page.
    TourOpenModels,
    /// What the tour waits for: the hub and the hardware of this machine.
    TourLoaded {
        overview: Result<models::Overview, String>,
        hardware: Result<tour::Hardware, String>,
    },
    /// The overview of the keyboard shortcuts opens (`Mod+/`, `?`, or Settings, General).
    ShortcutsOpen,
    /// The overview closes (*Close*, or the click beside it).
    ShortcutsClose,
    /// A shortcut waits for its new keys (the *Change* button; a second press cancels).
    ScChange(String),
    /// One changed shortcut goes back to its default keys (the *Reset* button).
    ScReset(String),
    /// Every change is undone (*Reset all*).
    ScResetAll,
    /// A key of the window was pressed and no text field took it (the keydown of `boot.js`).
    Key(iced::keyboard::Key, iced::keyboard::Modifiers),
    /// Walk the chats like `Alt+ArrowDown` and `Alt+ArrowUp` do (the `stepChat` of `boot.js`).
    ChatStep(i32),
    /// The command palette opens (`Mod+K`, the `search` shortcut of the Studio).
    PaletteOpen,
    /// The command palette closes (Esc, the click beside it, or a row was run).
    PaletteClose,
    /// The window came to the front or went behind it (the alerts stay silent while it is there).
    WindowFocus(bool),
    /// The language of the dictation and of the reading (the `selVoiceLang` of `voice.js`).
    VoiceLang(String),
    /// The voice of the reading (the `selVoiceName` of `voice.js`).
    VoiceVoice(String),
    /// How fast the reading speaks (the `rngVoiceRate` of `voice.js`, 0.5 to 2).
    VoiceRate(f32),
    /// How high the reading speaks (the `rngVoicePitch` of `voice.js`, 0.5 to 2).
    VoicePitch(f32),
    /// Read every finished answer aloud (the `chkVoiceAuto` of `voice.js`).
    VoiceAuto(bool),
    /// Say something when a run ends while the window is behind (the `chkNotify` of `voice.js`).
    VoiceNotify(bool),
    /// Play the short tone at that moment (the `chkNotifySound` of `voice.js`).
    VoiceSound(bool),
    /// *Read a sample* of the settings (the `btnVoiceTest` of `voice.js`).
    VoiceTest,
    /// *Play* of the settings (the `btnChimeTest` of `voice.js`).
    VoiceChimeTest,
    /// What this computer has: its voices and whether it can notify (`fillVoices` and
    /// `notifyNote` of `voice.js`, read from the speech programs here).
    VoiceSystem { voices: Vec<String>, can_notify: bool },
    /// The text of the palette changed (`input` of `searchModalInput`).
    PaletteQuery(String),
    /// The active row moved with the arrow keys (`paletteStep` of `palette.js`).
    PaletteActive(i32),
    /// The row `i` runs (`run` of `palette.js`: the window closes and the row does its work).
    PaletteRun(usize),
    Code(code::CodeMsg),
}

/// What the web tools of one round gave back, in the order of the calls (the memory calls of the
/// same round are already in the window when this arrives).
#[derive(Debug, Clone)]
pub struct ToolRoundResult {
    /// Index of the call in the round, its id, the text for the model and the card under the answer.
    pub results: Vec<(usize, String, String, crate::memory::ToolCard)>,
    pub budget_left: usize,
    pub sources: Vec<tools::Source>,
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
            spec_editor: Content::new(),
            max_tokens_text: String::new(),
            engine: EngineState::default(),
            picker_open: false,
            asst_menu: false,
            asst_edit: None,
            asst_model_pending: None,
            prj_view: None,
            prj_open: Default::default(),
            prj_pick: None,
            prj_delete: None,
            prj_pending: None,
            prj_creating: false,
            prj_new_name: String::new(),
            prj_renaming: None,
            prj_instr: Content::new(),
            rag_cache: None,
            rag_open: Default::default(),
            attachments: Vec::new(),
            personal_about: Content::new(),
            mem_new: String::new(),
            mem_edit: None,
            mem_clear: false,
            remember: None,
            remember_text: String::new(),
            slash: Vec::new(),
            slash_index: 0,
            style_menu: false,
            new_style: "default".into(),
            temp_next: false,
            temp_close: false,
            tour: None,
            hero: None,
            hero_samples: Vec::new(),
            hero_last: None,
            hero_overview: None,
            hero_hub_asked: false,
            prices: Vec::new(),
            usage_status: None,
            tour_data: tour::TourData::default(),
            shortcuts_open: false,
            sc_capture: None,
            sc_message: String::new(),
            window_focused: false,
            code_started: None,
            voice_names: Vec::new(),
            voice_can_notify: true,
            palette_open: false,
            palette_query: String::new(),
            palette_active: -1,
            palette_rows: Vec::new(),
            rewind_notice: None,
            use_web: false,
            answer_budget: None,
            answer_sources: Vec::new(),
            tool_hold: Vec::new(),
            tool_web_jobs: Vec::new(),
            research: false,
            research_no_schema: false,
            research_sources: None,
            art: None,
            runs: std::collections::HashMap::new(),
            plib_texts: std::collections::HashMap::new(),
            mem_synced_text: None,
            mem_sync_off: false,
            mem_syncing: false,
            tool_round: 0,
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
        app.spec_editor = Content::with_text(&app.store.prefs.sampling.constraint_spec);
        app.personal_about = Content::with_text(&app.store.prefs.personal.about);
        app.sampling_text = settings::SamplingTexts::from(&app.store.prefs.sampling);
        app.max_tokens_text = if app.store.prefs.max_tokens > 0 { app.store.prefs.max_tokens.to_string() } else { String::new() };
        if let Some(first) = app.store.chats.first().map(|c| c.id) {
            app.open(first);
        }
        // every saved prompt gets its box here: the settings page draws them from this map
        for p in &app.store.prefs.saved_prompts {
            app.plib_texts.insert(p.id.clone(), Content::with_text(&p.text));
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

    /// What is in the message box (the tests read it after a pick of the library).
    pub fn composer_text(&self) -> String {
        self.editor.text()
    }

    /// Whether the menu of the prompt library under a "/" is open.
    pub fn slash_open(&self) -> bool {
        !self.slash.is_empty()
    }

    /// The tools the next request would offer (the tests read them).
    pub fn offered_tools(&self) -> Option<serde_json::Value> {
        self.tool_defs()
    }

    /// Whether a temporary chat is staged for the next new chat (the tests read it).
    pub fn staged_temporary(&self) -> bool {
        self.temp_next
    }

    /// How many chats the store holds (the tests count them around a temporary one).
    pub fn chat_count(&self) -> usize {
        self.store.chats.len()
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

    /// The passages under the message at `i` (what the chip under it draws).
    pub fn message_passages(&self, i: usize) -> Option<&[crate::rag::Passage]> {
        self.messages.get(i).and_then(|m| m.passages.as_deref())
    }

    /// What is attached to the next question (the Studio's `state.attachments`).
    pub fn attachments(&self) -> &[Attachment] {
        &self.attachments
    }

    /// The tokens of the open chat as the meter counts them (an estimate until the server says more).
    pub fn used_tokens(&self) -> u64 {
        self.used_tokens
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

    /// Whether the window is in the dark theme (the tests read it beside the theme command).
    pub fn dark(&self) -> bool {
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
        // a temporary chat that is left is gone for good (the Studio's leaveThread)
        self.store.chats.retain(|c| !c.temporary || c.id == id);
        if let Some(chat) = self.store.chat(id) {
            self.messages = chat.messages.iter().map(Msg::from_record).collect();
            self.window = CHAT_WINDOW;
            self.editing = None;
            self.current = Some(id);
            // a chat that is open takes the pending project of a new one away (the Studio's prjChrome)
            self.prj_pending = None;
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

    pub fn can_send(&self) -> bool {
        matches!(self.server, Server::Ready(_)) && self.live.is_none() && (!self.editor.text().trim().is_empty() || !self.attachments.is_empty())
    }

    fn send(&mut self, prompt: String) -> Task<Message> {
        let Server::Ready(model) = self.server.clone() else { return Task::none() };
        // a question may be nothing but an attachment (the Studio's `if (!text && !state.attachments.length)`)
        if self.live.is_some() || (prompt.trim().is_empty() && self.attachments.is_empty()) {
            return Task::none();
        }
        // what a rewind hides cannot come back into the row once a new question is asked: it is
        // kept as a version of the chat (the conversation tree shows those later)
        self.commit_rewinds();
        // one answer has its own room for web results and its own sources (ctGenerate)
        self.answer_budget = Some(tools::budget(model.max_context));
        self.answer_sources.clear();
        // "remember that …" at the start of a message saves the thing (the Studio's prjCaptureMemory)
        if self.memory_capture_on() {
            if let Some(fact) = memory::extract(&prompt) {
                self.remember(fact);
            }
        }
        let id = match self.current {
            Some(id) => id,
            None => {
                // the chat remembers the assistant that was chosen for new chats and the project it was started from
                let wanted = self.active_assistant_id();
                let id = self.store.new_chat();
                if let Some(chat) = self.store.chat_mut(id) {
                    chat.assistant = wanted;
                    // the style that was staged for a new chat becomes this chat's (state.newStyle)
                    chat.style = self.new_style.clone();
                    // and the staging of the temporary chat becomes this chat's (state.tempNext)
                    chat.temporary = self.temp_next;
                }
                self.temp_next = false;
                if let Some(pid) = self.prj_pending.take().filter(|pid| self.store.projects.iter().any(|p| p.id == *pid)) {
                    if let Some(chat) = self.store.chat_mut(id) {
                        chat.project = Some(pid);
                    }
                }
                self.current = Some(id);
                id
            }
        };
        let mut carried: Option<(Vec<String>, Vec<store::MsgFile>)> = None;
        if let Some(i) = self.editing.take() {
            // the changed message starts a new version of everything after it
            if i < self.messages.len() {
                carried = Some((self.messages[i].images.clone(), self.messages[i].files.clone()));
                self.sync_current();
                if let Some(chat) = self.store.chat_mut(id) {
                    chat.branch_at(i);
                }
                self.messages.truncate(i);
            }
        }
        let first = self.messages.is_empty();
        self.tool_round = 0;
        // research mode takes the question when there is nothing attached (the Studio's
        // `state.research && text && !images.length`; a question with anything attached goes the
        // normal way, so nothing that was attached is left out of the answer)
        let research_ok = self.research && !prompt.trim().is_empty() && self.attachments.is_empty();
        // the attachments of this question: images and whole files on the message, texts that do not
        // fit as documents of this chat (the Studio's addChild and ragStoreDocs). Editing a message
        // keeps the images and files it had and leaves the pending attachments where they are.
        let mut images: Vec<String> = Vec::new();
        let mut files: Vec<store::MsgFile> = Vec::new();
        let mut seed_name: Option<String> = None;
        if let Some((old_images, old_files)) = carried {
            images = old_images;
            files = old_files;
        } else {
            let pending = std::mem::take(&mut self.attachments);
            seed_name = pending.first().map(|a| match a {
                Attachment::Image { name, .. } | Attachment::Text { name, .. } | Attachment::Doc { name, .. } => name.clone(),
            });
            for a in pending {
                match a {
                    Attachment::Image { data_url, .. } => images.push(data_url),
                    Attachment::Text { name, text, .. } => files.push(store::MsgFile { name, text }),
                    Attachment::Doc { name, text, pages, truncated, .. } => self.store_chat_doc(id, name, text, pages, truncated),
                }
            }
        }
        self.messages.push(Msg {
            role: Role::User,
            md: markdown::Content::parse(&prompt),
            content: prompt.clone(),
            reasoning: String::new(),
            stats: None,
            passages: None,
            images,
            files,
            tool_calls: None,
            tool_call_id: None,
            tool_cards: Vec::new(),
            sources: Vec::new(),
            research: None,
        logprobs: Vec::new(),
        reasoning_open: false,
        });
        self.messages.push(Msg {
            role: Role::Assistant,
            content: String::new(),
            reasoning: String::new(),
            stats: None,
            passages: None,
            images: Vec::new(),
            files: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
            tool_cards: Vec::new(),
            sources: Vec::new(),
            research: None,
            logprobs: Vec::new(),
            reasoning_open: false,
            md: markdown::Content::new(),
        });
        // the passages of the knowledge travel with this question (the project's documents first,
        // then the documents of this chat)
        let passages = self.passages_for(&prompt);
        if let Some(user) = self.messages.iter_mut().filter(|m| m.role == Role::User).last() {
            user.passages = passages;
        }
        if first {
            // the Studio names a chat after its first attachment when the question is empty
            let seed = if !prompt.trim().is_empty() { prompt.clone() } else { seed_name.unwrap_or_default() };
            if let Some(chat) = self.store.chat_mut(id) {
                chat.title = store::title_from(&seed);
            }
        }
        self.sync_current();
        self.save();
        self.error = None;
        self.editor = Content::new();
        self.research_sources = None;

        if research_ok {
            return self.research_begin(id, prompt, model);
        }
        self.stream(id, model.id)
    }

    /// Research mode: the question becomes a small research project — plan the searches, read the
    /// best pages, take notes, then a cited report (the Studio's `researchRun`). The window is
    /// repainted through the progress stream; the run itself is the task behind `live`, so Stop
    /// cancels every request of it.
    fn research_begin(&mut self, id: u64, question: String, model: ModelInfo) -> Task<Message> {
        use futures::StreamExt;
        let ui = research::ResearchUi::new(research::now_ms());
        if let Some(last) = self.messages.last_mut() {
            last.research = Some(ui);
        }
        let base = self.base.clone();
        let model_id = model.id.clone();
        let ctx = model.max_context;
        let no_schema = self.research_no_schema;
        let (tx, rx) = futures::channel::mpsc::unbounded();
        let progress = Task::stream(rx.map(|ui| Message::ResearchProgress(Box::new(ui))));
        let (work, handle) = Task::perform(
            async move { research::prepare(&base, &model_id, ctx, &question, no_schema, tx).await },
            |prepared| match prepared {
                Some(p) => Message::ResearchDone(Box::new(p)),
                None => Message::DismissToast,
            },
        )
        .abortable();
        self.live = Some(Live {
            chat: id,
            started: std::time::Instant::now(),
            _handle: handle.abort_on_drop(),
        });
        Task::batch([progress, work])
    }

    /// The report of a research run: its own two messages, with the size of the answer cut to
    /// what is left of the context (the Studio's `researchStream`).
    fn stream_research(&mut self, id: u64, system: String, user: String) -> Task<Message> {
        self.hold_clock(id);
        let Server::Ready(model) = self.server.clone() else { return Task::none() };
        let wire = vec![
            WireMessage { role: "system".into(), content: system.clone(), parts: None, tool_calls: None, tool_call_id: None },
            WireMessage { role: "user".into(), content: user.clone(), parts: None, tool_calls: None, tool_call_id: None },
        ];
        let prompt_tokens = (system.chars().count().div_ceil(4) + user.chars().count().div_ceil(4)) as f64 * 1.1;
        let max = research::research_max_tokens(model.max_context, prompt_tokens.ceil() as usize, 1800).max(200);
        let mut params = self.chat_params();
        params.max_tokens = Some(max as u32);
        let events = api::chat_stream(self.base.clone(), model.id, wire, params);
        let (task, handle) = Task::stream(events).map(move |ev| Message::Stream(id, ev)).abortable();
        self.live = Some(Live {
            chat: id,
            started: std::time::Instant::now(),
            _handle: handle.abort_on_drop(),
        });
        task
    }

    /// Starts the answer to the messages as they are (the last one is the empty answer).
    fn stream(&mut self, id: u64, model: String) -> Task<Message> {
        self.hold_clock(id);
        let wire = self.wire_messages();
        let defs = self.tool_defs();
        use futures::StreamExt;
        let events: futures::stream::BoxStream<'static, StreamEvent> = match defs {
            Some(tools_json) => api::chat_stream_with_tools(self.base.clone(), model, wire, self.chat_params(), Some(tools_json)).boxed(),
            None => api::chat_stream(self.base.clone(), model, wire, self.chat_params()).boxed(),
        };
        let (task, handle) = Task::stream(events).map(move |ev| Message::Stream(id, ev)).abortable();
        self.live = Some(Live {
            chat: id,
            started: std::time::Instant::now(),
            _handle: handle.abort_on_drop(),
        });
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
                // the first start shows the tour once, after the hardware is known (the Studio
                // waits for it too); an existing user never sees it
                if !self.store.prefs.tour_seen && self.store.chats.is_empty() {
                    let base = self.base.clone();
                    return Task::perform(
                        async move {
                            let overview = models::fetch_overview(base.clone()).await;
                            let hardware = models::fetch_hardware(base).await;
                            Message::TourLoaded { overview, hardware }
                        },
                        |m| m,
                    );
                }
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
                let was_running_before = was_running;
                let task = self.code.update(msg, &env);
                if !was_running_before && self.code.running {
                    // the run began (the `started` of watchRun in voice.js)
                    self.code_started = Some(std::time::Instant::now());
                } else if was_running_before && !self.code.running {
                    // the run ended: five seconds or more say so (watchRun of voice.js)
                    if let Some(at) = self.code_started.take() {
                        let ms = at.elapsed().as_millis() as u64;
                        if ms >= 5_000 {
                            self.announce(String::from("The agent has finished."));
                        }
                    }
                }
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
                // a new version starts from the row as it is: what a rewind hides becomes a version
                self.commit_rewinds();
                let (Some(id), Server::Ready(model)) = (self.current, self.server.clone()) else { return Task::none() };
                // a new answer has its own room for web results and its own sources
                self.answer_budget = Some(tools::budget(model.max_context));
                self.answer_sources.clear();
                self.sync_current();
                if let Some(chat) = self.store.chat_mut(id) {
                    chat.branch_at(i);
                }
                self.messages.truncate(i);
                self.messages.push(Msg {
                    role: Role::Assistant,
                    content: String::new(),
                    reasoning: String::new(),
                    stats: None,
                    passages: None,
                    images: Vec::new(),
                    files: Vec::new(),
                    tool_calls: None,
                    tool_call_id: None,
                    tool_cards: Vec::new(),
                    sources: Vec::new(),
                    research: None,
                    logprobs: Vec::new(),
                    reasoning_open: false,
                    md: markdown::Content::new(),
                });
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
            Message::SamplingConstraint(v) => {
                self.store.prefs.sampling.constraint = v;
                self.save();
                Task::none()
            }
            Message::SamplingSpec(v) => {
                self.store.prefs.sampling.constraint_spec = v.clone();
                self.spec_editor = Content::with_text(&v);
                self.save();
                Task::none()
            }
            Message::SamplingSpecEdit(action) => {
                let edit = action.is_edit();
                self.spec_editor.perform(action);
                if edit {
                    self.store.prefs.sampling.constraint_spec = self.spec_editor.text().trim_end_matches('\n').to_string();
                    self.save();
                }
                Task::none()
            }
            Message::SamplingLogprobs(v) => {
                self.store.prefs.sampling.logprobs = v;
                self.save();
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
            Message::AsstMenuToggle => {
                self.asst_menu = !self.asst_menu;
                self.picker_open = false;
                Task::none()
            }
            Message::AsstMenuClose => {
                self.asst_menu = false;
                Task::none()
            }
            Message::AsstApply(id) => self.apply_assistant(&id, false),
            Message::AsstNew => {
                self.asst_edit = Some(assistants::Editor::default());
                self.asst_menu = false;
                Task::none()
            }
            Message::AsstEditOpen(id) => {
                if let Some(a) = assistants::all(&self.store.assistants).into_iter().find(|x| x.id == id) {
                    self.asst_edit = Some(assistants::Editor::of(&a, a.builtin));
                    self.asst_menu = false;
                }
                Task::none()
            }
            Message::AsstSaveCurrent => {
                // "Save current": the instructions of the assistant in the box, the values of the sampler
                let mut editor = assistants::Editor::default();
                if let Some(a) = self.active_assistant() {
                    editor.system = Content::with_text(&a.system);
                    editor.icon = a.icon;
                    editor.color = a.color;
                }
                editor.temperature = format!("{}", self.store.prefs.temperature);
                editor.top_p = format!("{}", self.store.prefs.top_p);
                editor.top_k = self.store.prefs.sampling.top_k.to_string();
                self.asst_edit = Some(editor);
                self.asst_menu = false;
                Task::none()
            }
            Message::AsstClose => {
                self.asst_edit = None;
                Task::none()
            }
            Message::AsstSave => {
                let Some(editor) = self.asst_edit.clone() else { return Task::none() };
                let Some(draft) = editor.draft() else {
                    self.toast = Some("Give the assistant a name.".into());
                    return Task::none();
                };
                let was_new = editor.id.is_none();
                let was_active = self.active_assistant_id().as_deref() == Some(draft.id.as_str());
                match self.store.assistants.iter().position(|x| x.id == draft.id) {
                    Some(i) => self.store.assistants[i] = draft.clone(),
                    None => self.store.assistants.push(draft.clone()),
                }
                self.asst_edit = None;
                self.toast = Some(format!("Assistant \"{}\" saved", draft.name));
                self.save();
                if was_new || was_active {
                    return self.apply_assistant(&draft.id, true);
                }
                Task::none()
            }
            Message::AsstDelete => {
                let Some(id) = self.asst_edit.as_ref().and_then(|e| e.id.clone()) else { return Task::none() };
                self.store.assistants.retain(|x| x.id != id);
                if self.store.prefs.assistant == id {
                    self.store.prefs.assistant.clear();
                }
                if let Some(cid) = self.current {
                    if let Some(chat) = self.store.chat_mut(cid) {
                        if chat.assistant.as_deref() == Some(id.as_str()) {
                            chat.assistant = None;
                        }
                    }
                }
                self.asst_edit = None;
                self.save();
                Task::none()
            }
            Message::AsstExport => {
                let Some(editor) = self.asst_edit.clone() else { return Task::none() };
                let Some(a) = editor.draft() else {
                    self.toast = Some("Give the assistant a name first.".into());
                    return Task::none();
                };
                let body = assistants::export_json(&a);
                let path = chatlist::export_dir().join(format!("{}.assistant.json", chatlist::file_stem(&a.name)));
                self.toast = Some(match std::fs::write(&path, body) {
                    Ok(()) => format!("Saved to {}", path.display()),
                    Err(e) => format!("The assistant could not be saved to {}: {e}", path.display()),
                });
                Task::none()
            }
            Message::AsstImport => {
                self.asst_menu = false;
                let base = self.base.clone();
                Task::perform(
                    async move {
                        match api::pick_file_with_text(base).await {
                            Ok((_, Some(text))) => Ok(text),
                            // a file that is not UTF-8 text cannot be an exported assistant
                            Ok((path, None)) => {
                                let name = path.rsplit(['/', '\\']).next().unwrap_or(path.as_str()).to_string();
                                Err(format!("{name} is not a text file."))
                            }
                            Err(e) => Err(e),
                        }
                    },
                    Message::AsstImported,
                )
            }
            Message::AsstImported(result) => {
                let parsed = result.and_then(|text| assistants::parse_import(&text, assistants::new_id));
                match parsed {
                    Ok(a) => {
                        self.toast = Some(format!("Assistant \"{}\" imported", a.name));
                        self.store.assistants.push(a);
                        self.save();
                    }
                    // the dialog was closed: nothing was imported, and there is nothing to say
                    Err(e) if e.is_empty() => {}
                    Err(e) => self.toast = Some(e),
                }
                Task::none()
            }
            Message::AsstName(v) => {
                if let Some(e) = &mut self.asst_edit {
                    e.name = v;
                }
                Task::none()
            }
            Message::AsstSystem(action) => {
                if let Some(e) = &mut self.asst_edit {
                    e.system.perform(action);
                }
                Task::none()
            }
            Message::AsstStarters(action) => {
                if let Some(e) = &mut self.asst_edit {
                    e.starters.perform(action);
                }
                Task::none()
            }
            Message::AsstSysText(action) => {
                if let Some(e) = &mut self.asst_edit {
                    e.sys_text.perform(action);
                }
                Task::none()
            }
            Message::AsstTemp(v) => {
                if let Some(e) = &mut self.asst_edit {
                    e.temperature = v;
                }
                Task::none()
            }
            Message::AsstTopP(v) => {
                if let Some(e) = &mut self.asst_edit {
                    e.top_p = v;
                }
                Task::none()
            }
            Message::AsstTopK(v) => {
                if let Some(e) = &mut self.asst_edit {
                    e.top_k = v;
                }
                Task::none()
            }
            Message::AsstModel(v) => {
                if let Some(e) = &mut self.asst_edit {
                    e.model = v;
                    e.model_open = false;
                }
                Task::none()
            }
            Message::AsstModelToggle => {
                if let Some(e) = &mut self.asst_edit {
                    e.model_open = !e.model_open;
                    e.sys_open = false;
                }
                Task::none()
            }
            Message::AsstSysMode(mode) => {
                if let Some(e) = &mut self.asst_edit {
                    e.sys_mode = mode;
                    e.sys_open = false;
                }
                Task::none()
            }
            Message::AsstSysToggle => {
                if let Some(e) = &mut self.asst_edit {
                    e.sys_open = !e.sys_open;
                    e.model_open = false;
                }
                Task::none()
            }
            Message::AsstIcon(v) => {
                if let Some(e) = &mut self.asst_edit {
                    e.icon = v;
                }
                Task::none()
            }
            Message::AsstColor(v) => {
                if let Some(e) = &mut self.asst_edit {
                    e.color = v;
                }
                Task::none()
            }
            Message::PrjNew => {
                self.prj_creating = true;
                self.prj_new_name.clear();
                Task::none()
            }
            Message::PrjNewInput(v) => {
                self.prj_new_name = v.chars().take(120).collect();
                Task::none()
            }
            Message::PrjNewEnd(keep) => {
                self.prj_creating = false;
                let name: String = self.prj_new_name.trim().chars().take(rag::limits::NAME).collect();
                self.prj_new_name.clear();
                if !keep || name.is_empty() {
                    return Task::none();
                }
                let now = store::now_secs();
                let id = store::uid("prj");
                self.store.projects.push(store::Project {
                    id: id.clone(),
                    name: name.clone(),
                    instructions: String::new(),
                    docs: Vec::new(),
                    created: now,
                    updated: now,
                });
                self.prj_open.insert(id.clone());
                self.save();
                self.open_project(&id);
                Task::none()
            }
            Message::PrjRenameStart(id) => {
                let name = self.store.projects.iter().find(|p| p.id == id).map(|p| p.name.clone());
                if let Some(name) = name {
                    self.prj_renaming = Some((id, name));
                }
                Task::none()
            }
            Message::PrjRenameInput(v) => {
                if let Some((_, name)) = &mut self.prj_renaming {
                    *name = v.chars().take(120).collect();
                }
                Task::none()
            }
            Message::PrjRenameEnd(keep) => {
                let Some((id, typed)) = self.prj_renaming.take() else { return Task::none() };
                let name: String = typed.trim().chars().take(rag::limits::NAME).collect();
                if keep && !name.is_empty() {
                    if let Some(p) = self.store.projects.iter_mut().find(|p| p.id == id) {
                        p.name = name;
                        p.updated = store::now_secs();
                    }
                    self.save();
                }
                Task::none()
            }
            Message::PrjDelete(id) => {
                if self.store.projects.iter().any(|p| p.id == id) {
                    self.prj_delete = Some(id);
                }
                Task::none()
            }
            Message::PrjDeleteCancel => {
                self.prj_delete = None;
                Task::none()
            }
            Message::PrjDeleteOk => {
                let Some(id) = self.prj_delete.take() else { return Task::none() };
                let Some(pos) = self.store.projects.iter().position(|p| p.id == id) else { return Task::none() };
                self.store.projects.remove(pos);
                for c in self.store.chats.iter_mut().filter(|c| c.project.as_deref() == Some(id.as_str())) {
                    c.project = None;
                }
                self.prj_open.remove(&id);
                if self.prj_view.as_deref() == Some(id.as_str()) {
                    self.prj_view = None;
                    self.page = View::Chat;
                }
                if self.prj_pending.as_deref() == Some(id.as_str()) {
                    self.prj_pending = None;
                }
                self.rag_cache = None;
                self.save();
                // the Studio says nothing after the dialog; it only cleared the caches and re-rendered
                Task::none()
            }
            Message::PrjToggle(id) => {
                if !self.prj_open.remove(&id) {
                    self.prj_open.insert(id);
                }
                Task::none()
            }
            Message::PrjOpen(id) => {
                self.open_project(&id);
                Task::none()
            }
            Message::PrjClose => {
                self.page = View::Chat;
                Task::none()
            }
            Message::PrjInstr(action) => {
                let edit = action.is_edit();
                self.prj_instr.perform(action);
                if edit {
                    let text: String = self.prj_instr.text().chars().take(rag::limits::INSTRUCTIONS).collect();
                    if let Some(pid) = self.prj_view.clone() {
                        if let Some(p) = self.store.projects.iter_mut().find(|p| p.id == pid) {
                            p.instructions = text;
                            p.updated = store::now_secs();
                        }
                        self.save();
                    }
                }
                Task::none()
            }
            Message::PrjNewChat => {
                let id = self.prj_view.clone();
                let task = self.update(Message::NewChat);
                // the new chat is what you want to see (the Studio closes the page and opens it)
                self.page = View::Chat;
                if let Some(pid) = id.filter(|pid| self.store.projects.iter().any(|p| p.id == *pid)) {
                    self.prj_open.insert(pid.clone());
                    self.prj_pending = Some(pid);
                }
                task
            }
            Message::PrjChat(id) => {
                if self.live.is_none() {
                    self.page = View::Chat;
                    self.open(id);
                }
                Task::none()
            }
            Message::PrjAddFiles => {
                let base = self.base.clone();
                Task::perform(api::pick_file_with_text(base), Message::PrjFileChosen)
            }
            Message::PrjFileChosen(result) => {
                match result {
                    // the dialog was closed
                    Err(e) if e.is_empty() => {}
                    Err(e) => self.toast = Some(e),
                    Ok((path, text)) => {
                        let name = path.rsplit(['/', '\\']).next().unwrap_or(path.as_str()).to_string();
                        let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
                        if ["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "avif", "ico", "tiff"].contains(&ext.as_str()) {
                            self.toast = Some(format!("{name}: images cannot be searched. Add text, PDF or Office files."));
                        } else if let Some(text) = text {
                            self.store_doc(&name, text, None);
                        } else {
                            // the window reads the file itself and lets the server do the rest
                            match std::fs::read(&path) {
                                Err(e) => self.toast = Some(format!("The file could not be read: {e}")),
                                Ok(bytes) => {
                                    if bytes.len() as u64 > rag::MAX_TEXT_ATTACHMENT {
                                        self.toast = Some(format!("{name} is too large (more than 20.0 MB)."));
                                    } else if ["pdf", "docx", "xlsx", "epub", "html", "htm", "xhtml"].contains(&ext.as_str()) {
                                        self.toast = Some(format!("Reading {name}..."));
                                        let base = self.base.clone();
                                        return Task::perform(
                                            async move {
                                                let (text, pages, truncated) = api::extract_file(base, name.clone(), bytes).await?;
                                                Ok::<(String, String, Option<u64>, bool), String>((name, text, pages, truncated))
                                            },
                                            Message::PrjExtracted,
                                        );
                                    } else {
                                        self.toast = Some(format!("{name} is not a text file."));
                                    }
                                }
                            }
                        }
                    }
                }
                Task::none()
            }
            Message::PrjExtracted(result) => {
                match result {
                    Ok((name, text, pages, _truncated)) => self.store_doc(&name, text, pages),
                    Err(e) => self.toast = Some(e),
                }
                Task::none()
            }
            Message::PrjDocRemove(id) => {
                if let Some(pid) = self.prj_view.clone() {
                    if let Some(p) = self.store.projects.iter_mut().find(|p| p.id == pid) {
                        p.docs.retain(|d| d.id != id);
                        p.updated = store::now_secs();
                    }
                    self.save();
                }
                Task::none()
            }
            Message::PrjPick(choice) => {
                let Some(ids) = self.prj_pick.take() else { return Task::none() };
                let project = choice.filter(|s| !s.is_empty());
                let mut n = 0;
                for id in &ids {
                    if let Some(c) = self.store.chat_mut(*id) {
                        c.project = project.clone();
                        n += 1;
                    }
                }
                if n > 0 {
                    if let Some(pid) = &project {
                        self.prj_open.insert(pid.clone());
                    }
                    self.save();
                    // the line of the Studio's prjMove
                    self.toast = Some(match &project {
                        Some(pid) => {
                            let name = self.store.projects.iter().find(|p| p.id == *pid).map(|p| p.name.clone()).unwrap_or_default();
                            format!("{} chat{} moved to \"{}\"", n, if n == 1 { "" } else { "s" }, name)
                        }
                        None => format!("{} chat{} taken out of the project", n, if n == 1 { "" } else { "s" }),
                    });
                }
                Task::none()
            }
            Message::RagToggle(i) => {
                if !self.rag_open.remove(&i) {
                    self.rag_open.insert(i);
                }
                Task::none()
            }
            Message::Attach => {
                let base = self.base.clone();
                Task::perform(api::pick_file_with_text(base), Message::AttachChosen)
            }
            Message::AttachChosen(result) => match result {
                // the dialog was closed
                Err(e) if e.is_empty() => Task::none(),
                Err(e) => {
                    self.toast = Some(e);
                    Task::none()
                }
                Ok((path, text)) => self.add_picked(path, text),
            },
            Message::ImageReady(result) => {
                match result {
                    Ok((name, data_url)) => self.attachments.push(Attachment::Image { name, data_url }),
                    Err(e) => self.toast = Some(e),
                }
                Task::none()
            }
            Message::AttachExtracted(result) => {
                match result {
                    Ok((name, text, pages, truncated, size)) => self.rag_place(name, text, size, pages, truncated),
                    Err(e) => self.toast = Some(e),
                }
                Task::none()
            }
            Message::FilesDropped(paths) => {
                let tasks: Vec<Task<Message>> = paths.into_iter().map(|p| self.add_picked(p.to_string_lossy().into_owned(), None)).collect();
                Task::batch(tasks)
            }
            Message::AttachmentRemove(i) => {
                if i < self.attachments.len() {
                    self.attachments.remove(i);
                }
                Task::none()
            }
            Message::ChatDocRemove(id) => {
                if let Some(cid) = self.current {
                    if let Some(chat) = self.store.chat_mut(cid) {
                        chat.docs.retain(|d| d.id != id);
                    }
                    self.rag_cache = None;
                    self.save();
                }
                Task::none()
            }
            Message::AboutEdit(action) => {
                let edit = action.is_edit();
                self.personal_about.perform(action);
                if edit {
                    let about: String = self.personal_about.text().chars().take(memory::limits::ABOUT).collect();
                    self.store.prefs.personal.about = about;
                    self.save();
                }
                Task::none()
            }
            Message::UseAbout(v) => {
                self.store.prefs.personal.use_about = v;
                self.save();
                Task::none()
            }
            Message::UseMemory(v) => {
                self.store.prefs.personal.use_memory = v;
                self.save();
                Task::none()
            }
            Message::MemNewInput(v) => {
                self.mem_new = v.chars().take(memory::limits::MEMORY).collect();
                Task::none()
            }
            Message::MemAddNow => {
                let text = std::mem::take(&mut self.mem_new);
                if !text.trim().is_empty() {
                    self.remember(text);
                }
                Task::none()
            }
            Message::MemEditStart(id) => {
                let text = self
                    .store
                    .prefs
                    .personal
                    .memories
                    .iter()
                    .find(|m| m.id == id)
                    .map(|m| m.text.clone());
                if let Some(text) = text {
                    self.mem_edit = Some((id, text));
                }
                Task::none()
            }
            Message::MemEditInput(v) => {
                if let Some((_, text)) = &mut self.mem_edit {
                    *text = v;
                }
                Task::none()
            }
            Message::MemEditEnd(keep) => {
                let Some((id, typed)) = self.mem_edit.take() else { return Task::none() };
                if !keep {
                    return Task::none();
                }
                let text: String = typed.split_whitespace().collect::<Vec<_>>().join(" ");
                let text: String = text.chars().take(memory::limits::MEMORY).collect();
                if text.is_empty() {
                    // an empty row takes the entry out (the Studio's change handler)
                    self.mem_delete(&id);
                    return Task::none();
                }
                if memory::secret(&text) {
                    self.toast = Some("That looks like a password, key or card number. It is not saved to memory.".into());
                    return Task::none();
                }
                if let Some(m) = self.store.prefs.personal.memories.iter_mut().find(|m| m.id == id) {
                    m.text = text;
                    m.updated_at = store::now_millis();
                }
                self.save();
                Task::none()
            }
            Message::MemDelete(id) => {
                self.mem_delete(&id);
                Task::none()
            }
            Message::MemClear => {
                if !self.store.prefs.personal.memories.is_empty() {
                    self.mem_clear = true;
                }
                Task::none()
            }
            Message::MemClearCancel => {
                self.mem_clear = false;
                Task::none()
            }
            Message::MemClearOk => {
                self.mem_clear = false;
                self.store.prefs.personal.memories.clear();
                self.save();
                Task::none()
            }
            Message::RememberOpen(i) => {
                if !self.store.prefs.personal.use_memory {
                    self.toast = Some("Memory is switched off in Settings > Personalization.".into());
                    return Task::none();
                }
                let text = self.messages.get(i).map(|m| {
                    m.content
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .chars()
                        .take(300)
                        .collect::<String>()
                });
                if let Some(text) = text {
                    self.remember = Some(i);
                    self.remember_text = text;
                }
                Task::none()
            }
            Message::RememberEdit(v) => {
                self.remember_text = v;
                Task::none()
            }
            Message::RememberCancel => {
                self.remember = None;
                Task::none()
            }
            Message::RememberOk => {
                let text = std::mem::take(&mut self.remember_text);
                self.remember = None;
                if !text.trim().is_empty() {
                    self.remember(text);
                }
                Task::none()
            }
            // the prompt library of the settings
            Message::PromptAdd => {
                if self.store.prefs.saved_prompts.len() >= promptlib::limits::PROMPTS {
                    self.toast = Some(format!("The library holds at most {} prompts.", promptlib::limits::PROMPTS));
                } else {
                    let title = format!("Prompt {}", self.store.prefs.saved_prompts.len() + 1);
                    let id = store::uid("pr");
                    self.store.prefs.saved_prompts.push(promptlib::SavedPrompt {
                        id: id.clone(),
                        title,
                        text: String::new(),
                    });
                    self.plib_texts.insert(id, Content::new());
                    self.save();
                    return iced::widget::operation::focus_next();
                }
                Task::none()
            }
            Message::PromptDel(id) => {
                self.store.prefs.saved_prompts.retain(|p| p.id != id);
                self.plib_texts.remove(&id);
                self.save();
                Task::none()
            }
            Message::PromptTitle(id, value) => {
                if let Some(p) = self.store.prefs.saved_prompts.iter_mut().find(|p| p.id == id) {
                    // the Studio trims the title when you leave the field; here it is clamped as you type
                    p.title = value.chars().take(promptlib::limits::TITLE).collect();
                    self.save();
                }
                Task::none()
            }
            Message::PromptText(id, action) => {
                let known = self
                    .store
                    .prefs
                    .saved_prompts
                    .iter()
                    .find(|p| p.id == id)
                    .map(|p| p.text.clone())
                    .unwrap_or_default();
                let content = self.plib_texts.entry(id.clone()).or_insert_with(|| Content::with_text(&known));
                content.perform(action);
                let text: String = content.text().chars().take(promptlib::limits::TEXT).collect();
                if let Some(p) = self.store.prefs.saved_prompts.iter_mut().find(|p| p.id == id) {
                    p.text = text;
                    self.save();
                }
                Task::none()
            }
            Message::SlashPick(i) => {
                self.slash_pick(i);
                Task::none()
            }
            Message::SlashUp => {
                if !self.slash.is_empty() {
                    self.slash_index = (self.slash_index + self.slash.len() - 1) % self.slash.len();
                }
                Task::none()
            }
            Message::SlashDown => {
                if !self.slash.is_empty() {
                    self.slash_index = (self.slash_index + 1) % self.slash.len();
                }
                Task::none()
            }
            Message::SlashEnter => {
                self.slash_pick(self.slash_index);
                Task::none()
            }
            Message::SlashClose => {
                self.slash.clear();
                self.slash_index = 0;
                Task::none()
            }
            Message::ToggleWeb => {
                self.use_web = !self.use_web;
                if self.use_web {
                    self.toast = Some(
                        "The model may now search the web and read pages when it needs to. Questions it can answer itself are not searched."
                            .into(),
                    );
                }
                Task::none()
            }
            Message::OpenUrl(url) => {
                open_page(&url);
                Task::none()
            }
            Message::ToolRound(result) => {
                if let Some(budget) = self.answer_budget.as_mut() {
                    budget.left = result.budget_left;
                }
                self.answer_sources = result.sources;
                for (index, id, text, card) in result.results {
                    if let Some(slot) = self.tool_hold.get_mut(index) {
                        *slot = Some((id, text, card));
                    }
                }
                // a call the server never answered gets an error line, so the round can close
                for slot in self.tool_hold.iter_mut() {
                    if slot.is_none() {
                        *slot = Some((
                            String::new(),
                            "Error: the web request failed.".to_string(),
                            crate::memory::ToolCard { label: "Web".into(), status: "error".into(), change: None, undone: false, quiet: false },
                        ));
                    }
                }
                match self.live.as_ref().map(|l| l.chat) {
                    Some(chat) => self.tool_round_finalize(chat).unwrap_or_else(Task::none),
                    None => {
                        self.finish();
                        Task::none()
                    }
                }
            }
            Message::ImportChats => {
                let base = self.base.clone();
                Task::perform(api::pick_file_with_text(base), Message::ImportChosen)
            }
            Message::ImportChosen(result) => {
                let text = match result {
                    Ok((_, text)) => text,
                    Err(_) => None,
                };
                if let Some(text) = text {
                    let taken: std::collections::HashSet<u64> = self.store.chats.iter().map(|c| c.id).collect();
                    let mut next = self.store.next_id;
                    match chatlist::parse_import(&text, &mut next, &taken) {
                        Ok(chats) => {
                            if chats.is_empty() {
                                self.toast = Some("No chats were found in the file.".into());
                            } else {
                                let n = chats.len();
                                for chat in chats {
                                    self.store.chats.push(chat);
                                }
                                self.store.next_id = next;
                                self.save();
                                self.toast = Some(if n == 1 { "1 chat imported".into() } else { format!("{n} chats imported") });
                            }
                        }
                        Err(message) => self.toast = Some(message),
                    }
                }
                Task::none()
            }
            Message::ArtOpen(code, info) => {
                let families = artifacts::chat_families(&self.art_messages());
                // the Studio's openFromBlock: the family whose version holds this code, the message
                // it came from first; else a family of its own for this one block
                let mut hit: Option<(artifacts::Family, usize)> = None;
                for f in &families {
                    if let Some(j) = f.versions.iter().position(|v| v.code.trim() == code.trim()) {
                        let better = hit.as_ref().is_none();
                        if better {
                            hit = Some((f.clone(), j));
                        }
                    }
                }
                self.art = Some(match hit {
                    Some((f, j)) => artifacts::Panel::open(f, j, false),
                    None => {
                        let lang = info.split_whitespace().next().unwrap_or("").to_lowercase();
                        let fam = artifacts::Family {
                            id: "adhoc".into(),
                            kind: artifacts::artifact_kind(&lang),
                            title: artifacts::parse_info(&info),
                            lang,
                            versions: vec![artifacts::Version { n: 1, msg_id: String::new(), code }],
                            touched: 0,
                        };
                        artifacts::Panel::open(fam, 0, true)
                    }
                });
                Task::none()
            }
            Message::ArtAction(action) => {
                use artifacts::ArtAction as Act;
                match action {
                    Act::Close => self.art = None,
                    Act::Prev => {
                        if let Some(p) = self.art.as_mut() {
                            p.set_version(p.ver.saturating_sub(1));
                        }
                    }
                    Act::Next => {
                        if let Some(p) = self.art.as_mut() {
                            p.set_version(p.ver + 1);
                        }
                    }
                    Act::TabPreview => {
                        if let Some(p) = self.art.as_mut() {
                            p.set_tab(artifacts::Tab::Preview);
                        }
                    }
                    Act::TabCode => {
                        if let Some(p) = self.art.as_mut() {
                            p.set_tab(artifacts::Tab::Code);
                        }
                    }
                    Act::Reload => {
                        if let Some(p) = self.art.as_mut() {
                            p.render_body();
                        }
                    }
                    Act::OpenInBrowser => {
                        let file = self.art.as_ref().map(|p| {
                            (
                                chatlist::export_dir().join(artifacts::file_name(&p.fam.title, p.fam.kind, &p.fam.lang)),
                                p.version().code.clone(),
                            )
                        });
                        if let Some((path, text)) = file {
                            let _ = std::fs::write(&path, text);
                            return self.update(Message::OpenUrl(format!("file://{}", path.display())));
                        }
                    }
                    Act::Larger => {
                        if let Some(p) = self.art.as_mut() {
                            p.large = !p.large;
                        }
                    }
                    Act::Copy => {
                        if let Some(p) = self.art.as_ref() {
                            return iced::clipboard::write(p.version().code.clone());
                        }
                    }
                    Act::Download => {
                        let file = self.art.as_ref().map(|p| {
                            (
                                chatlist::export_dir().join(artifacts::file_name(&p.fam.title, p.fam.kind, &p.fam.lang)),
                                p.version().code.clone(),
                            )
                        });
                        if let Some((path, text)) = file {
                            // the file is small: written right here, like the exports of the chat list
                            let _ = std::fs::write(&path, text);
                            self.toast = Some("Saved to your downloads.".into());
                        }
                    }
                }
                Task::none()
            }
            Message::RunCode(code) => {
                if self.runs.contains_key(&code) {
                    // the same button stops a run that is going (`runBlock` toggles like this)
                    if let Some(run) = self.runs.get_mut(&code) {
                        run.stop("Stopped");
                    }
                    return Task::none();
                }
                use futures::StreamExt;
                let (tx, rx) = futures::channel::mpsc::unbounded();
                let key = code.clone();
                let events = rx.map(move |ev| Message::RunEvent { code: key.clone(), ev });
                self.runs.insert(code.clone(), artifacts::CodeRun::starting());
                std::thread::spawn(move || artifacts::run_javascript(&code, tx));
                Task::stream(events)
            }
            Message::RunStop(code) => {
                if let Some(run) = self.runs.get_mut(&code) {
                    run.stop("Stopped");
                }
                Task::none()
            }
            Message::RunEvent { code, ev } => {
                // a finished run keeps its output under the block until the next one starts
                if let Some(run) = self.runs.get_mut(&code) {
                    run.apply(ev);
                }
                Task::none()
            }
            Message::Esc => {
                // closeOverlays of boot.js: the windows go before anything else happens
                if self.palette_open {
                    self.palette_open = false;
                    return Task::none();
                }
                if self.shortcuts_open {
                    return self.update(Message::ShortcutsClose);
                }
                if self.tour.is_some() {
                    return self.update(Message::TourSkip);
                }
                // the large view of the panel goes first (the keydown of artifacts.js), then a
                // run that goes
                if self.art.as_ref().is_some_and(|p| p.large) {
                    if let Some(p) = self.art.as_mut() {
                        p.large = false;
                    }
                    return Task::none();
                }
                // only something that runs is stopped; everything else ignores the key
                let stopping = self.live.is_some()
                    || self.messages.last().is_some_and(|m| m.research.as_ref().is_some_and(|r| r.running));
                if stopping {
                    return self.update(Message::Stop);
                }
                Task::none()
            }
            Message::ReasoningToggle(i) => {
                if let Some(m) = self.messages.get_mut(i) {
                    m.reasoning_open = !m.reasoning_open;
                }
                Task::none()
            }
            Message::ResearchToggle => {
                self.research = !self.research;
                if self.research {
                    self.toast = Some(
                        "Research: your question is searched on the web in several steps and answered with a cited report. It takes longer than a normal answer."
                            .into(),
                    );
                }
                Task::none()
            }
            Message::ResearchLogToggle(i) => {
                if let Some(ui) = self.messages.get_mut(i).and_then(|m| m.research.as_mut()) {
                    ui.log_open = !ui.log_open;
                }
                self.sync_current();
                self.save();
                Task::none()
            }
            Message::ResearchProgress(ui) => {
                // the run repaints its progress line above the answer
                if let Some(last) = self.messages.last_mut() {
                    if last.research.is_some() {
                        last.research = Some(*ui);
                    }
                }
                Task::none()
            }
            Message::ResearchDone(prepared) => {
                let chat = self.live.as_ref().map(|l| l.chat);
                let prepared = *prepared;
                if let Some(last) = self.messages.last_mut() {
                    last.research = Some(prepared.ui.clone());
                }
                self.research_no_schema = prepared.no_schema;
                self.research_sources = if prepared.fallback { None } else { Some(prepared.sources) };
                self.error = None;
                let Server::Ready(model) = self.server.clone() else {
                    self.finish();
                    return Task::none();
                };
                match chat {
                    // the model answers on its own: a normal answer, the notice says why
                    Some(id) if prepared.fallback => self.stream(id, model.id),
                    // the report of the run is streamed into the answer
                    Some(id) => self.stream_research(id, prepared.system, prepared.user),
                    None => {
                        self.finish();
                        Task::none()
                    }
                }
            }
            Message::StyleMenu => {
                self.style_menu = !self.style_menu;
                Task::none()
            }
            Message::TempToggle => {
                let temporary_now = self.current.and_then(|id| self.store.chat(id)).is_some_and(|c| c.temporary);
                if temporary_now {
                    let has_messages = self.current.and_then(|id| self.store.chat(id)).is_some_and(|c| !c.messages.is_empty());
                    if has_messages {
                        // the Studio asks before a temporary chat with messages is gone for good
                        self.temp_close = true;
                    } else {
                        self.close_temporary();
                    }
                } else if self.temp_next && self.current.is_none() {
                    // the staging is taken back before a chat exists
                    self.temp_next = false;
                } else {
                    // start a new chat and stage the next one as temporary (startNewChat + tempNext)
                    let _ = self.update(Message::NewChat);
                    self.temp_next = true;
                }
                Task::none()
            }
            Message::TempCloseOk => {
                self.temp_close = false;
                self.close_temporary();
                Task::none()
            }
            Message::TempCloseCancel => {
                self.temp_close = false;
                Task::none()
            }
            Message::HeroPoll => {
                let base = self.base.clone();
                Task::perform(async move { models::fetch_status(base).await }, Message::HeroStatus)
            }
            Message::HeroStatus(Ok(status)) => {
                // the rate between two moments of the server's own clock (a slow poll does not
                // skew it — the clock is the server's), and the samples of the two-minute line
                let rate = crate::hero::trace_rate(
                    status.generated_tokens_total,
                    status.uptime_secs,
                    self.hero_last.map(|(t, _)| t),
                    self.hero_last.map(|(_, u)| u),
                );
                self.hero_last = Some((status.generated_tokens_total, status.uptime_secs));
                self.hero_samples.push(rate);
                if self.hero_samples.len() > 60 {
                    self.hero_samples.remove(0);
                }
                let setup = status.setup;
                self.hero = Some(status);
                // the models on this computer are asked once for the start screen (heroHubAsked)
                if setup && !self.hero_hub_asked {
                    self.hero_hub_asked = true;
                    let base = self.base.clone();
                    Task::perform(async move { models::fetch_overview(base).await }, Message::HeroOverview)
                } else {
                    Task::none()
                }
            }
            Message::HeroStatus(Err(_)) => Task::none(),
            Message::HeroOverview(Ok(overview)) => {
                self.hero_overview = Some(overview);
                Task::none()
            }
            Message::HeroOverview(Err(_)) => Task::none(),
            Message::HeroLoad(reference) => {
                Task::perform(models::load(self.base.clone(), reference), Message::ModelAction)
            }
            Message::HeroDownload { model, quant } => {
                Task::perform(models::pull(self.base.clone(), model, quant), Message::ModelAction)
            }
            Message::HeroChosen(Ok((path, _))) => {
                Task::perform(models::load(self.base.clone(), path), Message::ModelAction)
            }
            Message::HeroChosen(Err(_)) => Task::none(),
            Message::HeroBrowse => self.update(Message::Go(View::Models)),
            Message::UsageReset => {
                self.store.usage = crate::usage::UsageData::default();
                self.save();
                self.usage_status = Some(String::from("Counters reset."));
                Task::none()
            }
            Message::PricesLoaded(Ok(prices)) => {
                self.prices = prices;
                Task::none()
            }
            Message::PricesLoaded(Err(_)) => Task::none(),
            Message::PricesSaved(Ok(prices)) => {
                self.prices = prices;
                self.usage_status = Some(String::from("Prices saved."));
                Task::none()
            }
            Message::PricesSaved(Err(e)) => {
                self.usage_status = Some(e);
                Task::none()
            }
            Message::PriceAdd => {
                // the row the Studio adds: empty pattern, both prices at zero
                let mut row = crate::usage::Price::default();
                row.refresh_text();
                self.prices.push(row);
                Task::none()
            }
            Message::UsageSave => {
                // savePrices of the Studio: write the rows and show what the server kept
                self.usage_status = Some(String::from("Saving…"));
                let prices = self.prices.clone();
                Task::perform(models::save_prices(self.base.clone(), prices), Message::PricesSaved)
            }
            Message::PriceRemove(index) => {
                if index < self.prices.len() {
                    self.prices.remove(index);
                    self.usage_status = Some(String::from("Removed. Press Save prices to write it."));
                }
                Task::none()
            }
            Message::PriceField(index, field, value) => {
                if let Some(row) = self.prices.get_mut(index) {
                    match field {
                        "pattern" => row.pattern = value,
                        "input" => {
                            row.input_per_million = value.parse().unwrap_or(0.0);
                            row.input_text = value;
                        }
                        "output" => {
                            row.output_per_million = value.parse().unwrap_or(0.0);
                            row.output_text = value;
                        }
                        "note" => row.note = value,
                        _ => {}
                    }
                }
                Task::none()
            }
            Message::UsageStatus(text) => {
                self.usage_status = Some(text);
                Task::none()
            }
            Message::HeroChoose => {
                let base = self.base.clone();
                Task::perform(api::pick_file_with_text(base), Message::HeroChosen)
            }
            Message::TourShow => {
                // the Studio's openTour: the key is written, the first step opens, and the chat is
                // what it opens over (`switchSurface('chat')`)
                self.page = View::Chat;
                self.store.prefs.tour_seen = true;
                self.tour = Some(0);
                self.save();
                if self.tour_data.computer().is_none() {
                    let base = self.base.clone();
                    return Task::perform(
                        async move {
                            let overview = models::fetch_overview(base.clone()).await;
                            let hardware = models::fetch_hardware(base).await;
                            Message::TourLoaded { overview, hardware }
                        },
                        |m| m,
                    );
                }
                Task::none()
            }
            Message::TourNext => {
                // the last step says *Get started* and closes (the Next button of `voice.js`)
                match self.tour {
                    Some(step) if step + 1 < tour::STEPS.len() => self.tour = Some(step + 1),
                    _ => self.tour = None,
                }
                Task::none()
            }
            Message::TourBack => {
                if let Some(0) | Some(1) | Some(2) | Some(3) = self.tour {
                    self.tour = self.tour.map(|step| step.saturating_sub(1));
                }
                Task::none()
            }
            Message::TourSkip => {
                self.tour = None;
                Task::none()
            }
            Message::TourOpenModels => {
                self.tour = None;
                self.page = View::Models;
                Task::none()
            }
            Message::TourLoaded { overview, hardware } => {
                match hardware {
                    Ok(hw) => self.tour_data.read_hardware(&hw),
                    Err(e) => self.tour_data.read_error(&e),
                }
                if let Ok(o) = &overview {
                    self.tour_data.read_overview(o, o);
                }
                // the data was for the first start: open it now (nothing else opened the tour)
                if self.tour.is_none() && !self.store.prefs.tour_seen && self.store.chats.is_empty() {
                    self.store.prefs.tour_seen = true;
                    self.tour = Some(0);
                    self.save();
                }
                Task::none()
            }
            Message::ShortcutsOpen => {
                // openShortcuts of boot.js: the capture is empty and so is the message
                self.shortcuts_open = true;
                self.sc_capture = None;
                self.sc_message.clear();
                Task::none()
            }
            Message::ShortcutsClose => {
                self.shortcuts_open = false;
                self.sc_capture = None;
                Task::none()
            }
            Message::ScChange(id) => {
                // a second press on Change is a cancel (the click handler of boot.js)
                self.sc_capture = if self.sc_capture.as_deref() == Some(id.as_str()) {
                    None
                } else {
                    Some(id)
                };
                self.sc_message.clear();
                Task::none()
            }
            Message::ScReset(id) => {
                self.store.prefs.shortcut_overrides.remove(&id);
                self.sc_message.clear();
                self.save();
                Task::none()
            }
            Message::ScResetAll => {
                self.store.prefs.shortcut_overrides.clear();
                self.sc_message =
                    String::from("All shortcuts are back to their defaults.");
                self.save();
                Task::none()
            }
            Message::ChatStep(delta) => {
                // stepChat of boot.js: the chats of the Studio are walked by their time, the
                // newest first, from the one that is open
                let mut ids: Vec<(i64, u64)> = self
                    .store
                    .chats
                    .iter()
                    .map(|c| (c.updated, c.id))
                    .collect();
                ids.sort_by_key(|(at, _)| std::cmp::Reverse(*at));
                if !ids.is_empty() {
                    let current = self.current;
                    let i = ids
                        .iter()
                        .position(|(_, id)| Some(*id) == current)
                        .unwrap_or(0);
                    let next = (i as i32 + delta).clamp(0, ids.len() as i32 - 1) as usize;
                    let id = ids[next].1;
                    return self.update(Message::OpenChat(id));
                }
                Task::none()
            }
            Message::PaletteOpen => {
                // open of palette.js: the field starts empty and the rows come at once
                self.palette_open = true;
                self.palette_query.clear();
                self.palette_rows = self.palette_build();
                self.palette_active = if self.palette_rows.is_empty() { -1 } else { 0 };
                Task::none()
            }
            Message::PaletteClose => {
                self.palette_open = false;
                Task::none()
            }
            Message::PaletteQuery(q) => {
                // refresh of palette.js: the rows are built anew, the first one is active
                self.palette_query = q;
                self.palette_rows = self.palette_build();
                self.palette_active = if self.palette_rows.is_empty() { -1 } else { 0 };
                Task::none()
            }
            Message::PaletteActive(index) => {
                self.palette_active = index;
                Task::none()
            }
            Message::PaletteRun(i) => {
                // run of palette.js: the window closes first, then the row does its work
                let action = self.palette_rows.get(i).map(|(_, a)| a.clone());
                self.palette_open = false;
                if let Some(action) = action {
                    return self.update(action.message());
                }
                Task::none()
            }
            Message::WindowFocus(focused) => {
                self.window_focused = focused;
                Task::none()
            }
            Message::VoiceSystem { voices, can_notify } => {
                // fillVoices and notifyNote of voice.js: what the settings show about this machine
                self.voice_names = voices;
                self.voice_can_notify = can_notify;
                if self.voice_can_notify {
                    let voice = &self.store.prefs.voice;
                    if voice.notify && !voice.asked {
                        // the Studio asks while the click that sent a message still counts as a
                        // gesture; here there is no permission to ask for, the flag only says it
                        self.store.prefs.voice.asked = true;
                        self.save();
                    }
                }
                Task::none()
            }
            Message::VoiceLang(lang) => {
                self.store.prefs.voice.lang = lang;
                self.save();
                Task::none()
            }
            Message::VoiceVoice(voice) => {
                self.store.prefs.voice.voice = voice;
                self.save();
                Task::none()
            }
            Message::VoiceRate(rate) => {
                self.store.prefs.voice.rate = rate;
                self.save();
                Task::none()
            }
            Message::VoicePitch(pitch) => {
                self.store.prefs.voice.pitch = pitch;
                self.save();
                Task::none()
            }
            Message::VoiceAuto(auto) => {
                self.store.prefs.voice.auto = auto;
                self.save();
                Task::none()
            }
            Message::VoiceNotify(on) => {
                self.store.prefs.voice.notify = on;
                if on && !self.store.prefs.voice.asked {
                    // the gesture of voice.js: the first switch that asks
                    self.store.prefs.voice.asked = true;
                }
                self.save();
                Task::none()
            }
            Message::VoiceSound(on) => {
                self.store.prefs.voice.sound = on;
                self.save();
                Task::none()
            }
            Message::VoiceTest => {
                // the sample of voice.js: the marks are left out while reading
                speech::speak_with(
                    "This is how **fLLM** sounds. Code blocks are skipped, and so are `marks` and [links](https://example.com).",
                    &self.store.prefs.voice.clone(),
                );
                Task::none()
            }
            Message::VoiceChimeTest => {
                speech::chime();
                Task::none()
            }
            Message::Key(key, modifiers) => {
                let is_mac = cfg!(target_os = "macos");
                let combo = shortcuts::combo_of_key(&key, modifiers, is_mac);
                if combo.is_empty() {
                    return Task::none();
                }
                // rebinding: the next combination is the new one (the keydown of boot.js)
                if let Some(id) = self.sc_capture.clone() {
                    if combo == "Escape" {
                        self.sc_capture = None;
                        self.sc_message.clear();
                        return Task::none();
                    }
                    let combos = shortcuts::effective_combos(&self.store.prefs.shortcut_overrides);
                    match shortcuts::check_combo(&combos, &id, &combo) {
                        Ok(new) => {
                            self.store.prefs.shortcut_overrides.insert(id, new);
                            self.sc_capture = None;
                            self.sc_message = String::from("Saved.");
                            self.save();
                        }
                        Err(error) => self.sc_message = error,
                    }
                    return Task::none();
                }
                if self.palette_open {
                    // the arrow keys of the palette (Home and End only while the field is empty,
                    // the caret keeps them otherwise — the check of palette.js)
                    let walk = matches!(combo.as_str(), "ArrowDown" | "ArrowUp")
                        || matches!(combo.as_str(), "Home" | "End")
                            && self.palette_query.is_empty();
                    if walk {
                        self.palette_active = palette::palette_step(
                            self.palette_active,
                            self.palette_rows.len(),
                            &combo,
                        );
                    }
                    // nothing else runs while the window is open
                    return Task::none();
                }
                let combos = shortcuts::effective_combos(&self.store.prefs.shortcut_overrides);
                let Some(id) = shortcuts::shortcut_for(&combos, &combo) else {
                    return Task::none();
                };
                match id {
                    "newChat" => return self.update(Message::NewChat),
                    "stop" => return self.update(Message::Stop),
                    "surfaceChat" => return self.update(Message::Go(View::Chat)),
                    "surfaceCode" => return self.update(Message::Go(View::Code)),
                    "openModels" => return self.update(Message::Go(View::Models)),
                    "openSettings" => return self.update(Message::Go(View::Settings)),
                    "regenerate" => return self.update(Message::Retry),
                    "shortcuts" => return self.update(Message::ShortcutsOpen),
                    "search" => return self.update(Message::PaletteOpen),
                    "nextChat" => return self.update(Message::ChatStep(1)),
                    "prevChat" => return self.update(Message::ChatStep(-1)),
                    // search, focusComposer, toggleSidebar, codeAgent and codeSessions have no
                    // surface of their own here yet; the list shows them like the Studio does
                    _ => {}
                }
                Task::none()
            }
            Message::StylePick(style) => {
                self.style_menu = false;
                // the Studio writes the choice into the chat, or into state.newStyle without one
                if let Some(chat) = self.current.and_then(|id| self.store.chat_mut(id)) {
                    chat.style = style;
                    self.save();
                } else {
                    self.new_style = style;
                }
                Task::none()
            }
            Message::MemoryUndo(mi, ci) => {
                let change = self.messages.get(mi).and_then(|m| m.tool_cards.get(ci)).and_then(|c| c.change.clone());
                match change.and_then(|c| memory::undo(&self.store.prefs.personal.memories, &c).map(|list| (c, list))) {
                    Some((change, list)) => {
                        self.store.prefs.personal.memories = list;
                        if let Some(card) = self.messages.get_mut(mi).and_then(|m| m.tool_cards.get_mut(ci)) {
                            card.undone = true;
                            let _ = change;
                        }
                        self.save();
                        self.toast = Some("Memory change undone".into());
                    }
                    None => self.toast = Some("That memory entry has changed since; it was left as it is.".into()),
                }
                Task::none()
            }
            Message::PersonalSynced(answer) => {
                self.mem_syncing = false;
                match answer {
                    // the server answered with an error: it has no such route (the Studio switches off)
                    api::SyncAnswer::Refused => self.mem_sync_off = true,
                    // it could not be reached: the next tick tries again
                    api::SyncAnswer::Unreachable => {}
                    api::SyncAnswer::Got(PersonalSync::Stored(text)) => self.mem_synced_text = Some(text),
                    api::SyncAnswer::Got(PersonalSync::Import(list)) => {
                        let list = memory::sanitize(&list, || store::uid("mem"));
                        if !list.is_empty() {
                            self.store.prefs.personal.memories = list;
                            self.store.prefs.personal_loaded = true;
                            self.save();
                            let n = self.store.prefs.personal.memories.len();
                            self.toast =
                                Some(format!("{} remembered fact{} restored from the copy on this computer.", n, if n == 1 { "" } else { "s" }));
                            self.mem_synced_text = Some(self.personal_body().to_string());
                        }
                    }
                }
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
                // the notice of a rewind stands nine seconds (the Studio removes it after 9000 ms)
                if self.rewind_notice.is_some_and(|(_, until)| chrono::Local::now().timestamp() as u64 > until) {
                    self.rewind_notice = None;
                }
                // the copy of the memory on the server: asked for at the start, written when it differs
                // (the Studio's memImportFromServer and memSyncSoon)
                let mut sync = Task::none();
                if !self.mem_sync_off && !self.mem_syncing {
                    if !self.store.prefs.personal_loaded {
                        self.mem_syncing = true;
                        let base = self.base.clone();
                        sync = Task::perform(async move { api::get_memory(&base).await.map(PersonalSync::Import) }, Message::PersonalSynced);
                    } else {
                        let body = self.personal_body();
                        let text = body.to_string();
                        if self.mem_synced_text.as_deref() != Some(text.as_str()) {
                            self.mem_syncing = true;
                            let base = self.base.clone();
                            sync = Task::perform(
                                async move { api::put_memory(&base, body).await.map(|_| PersonalSync::Stored(text)) },
                                Message::PersonalSynced,
                            );
                        }
                    }
                }
                // the server that the app started may have ended on its own (a port taken, a binary that refuses its arguments)
                if let Some(s) = self.sidecar.as_mut() {
                    if !s.running() {
                        self.sidecar = None;
                        self.switching = false;
                        self.sidecar_note = Some(format!("The server that the app started has ended. Its output is in {}.", self.data_file.with_file_name("server.log").display()));
                    }
                }
                if matches!(self.server, Server::Down(_)) {
                    return Task::batch([sync, self.update(Message::Retry)]);
                }
                sync
            }
            Message::NewChat => {
                if self.live.is_some() {
                    return Task::none();
                }
                self.current = None;
                // a temporary chat that is left is gone for good (the Studio's leaveThread)
                self.store.chats.retain(|c| !c.temporary);
                self.prj_pending = None;
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
                    chatlist::Effect::AskProject(ids) => self.prj_pick = Some(ids),
                    chatlist::Effect::PersonalToggled(off) => {
                        self.save();
                        self.toast = Some(if off {
                            "Your instructions and memory are not used in this chat.".into()
                        } else {
                            "Your instructions and memory are used in this chat.".into()
                        });
                    }
                    chatlist::Effect::MemoryToggled(off) => {
                        self.save();
                        self.toast = Some(if off {
                            "Memory is not used or changed in this chat.".into()
                        } else {
                            "Memory is used in this chat.".into()
                        });
                    }
                }
                Task::none()
            }
            Message::Edit(action) => {
                self.editor.perform(action);
                // what a "/" in the box offers is asked again on every change (plibUpdateSlash)
                self.slash_recompute();
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
                    let list = self.starter_list();
                    if let Some(s) = list.get(i) {
                        return self.send(s.clone());
                    }
                }
                Task::none()
            }
            Message::Stop => {
                // a research run that is still planning says so under its answer (the Studio's
                // `r.stopped` in the finally of researchRun)
                if let Some(ui) = self.messages.last_mut().and_then(|m| m.research.as_mut()) {
                    if ui.running {
                        let now = research::now_ms();
                        ui.seconds = now.saturating_sub(ui.t0_ms) / 1000;
                        ui.running = false;
                        ui.stopped = true;
                        ui.event((now.saturating_sub(ui.t0_ms)) as f64 / 1000.0, "Stopped.".to_string());
                    }
                }
                self.research_sources = None;
                self.finish();
                Task::none()
            }
            Message::Stream(chat, ev) => {
                if self.live.as_ref().map(|l| l.chat) != Some(chat) {
                    return Task::none();
                }
                self.on_event(ev).unwrap_or_else(Task::none)
            }
            Message::Copy(i) => match self.messages.get(i) {
                // what you can see in the answer, without the Markdown marks (plainTextOf)
                Some(m) => iced::clipboard::write(md::plain(&m.content)),
                None => Task::none(),
            },
            Message::CopyMd(i) => match self.messages.get(i) {
                Some(m) => iced::clipboard::write(m.content.clone()),
                None => Task::none(),
            },
            Message::Speak(i) => {
                let text = self.messages.get(i).map(|m| speech::strip_for_speech(&m.content)).unwrap_or_default();
                if text.is_empty() || speech::speak_out(&text) {
                    Task::none()
                } else {
                    self.toast = Some("This computer cannot read aloud.".into());
                    Task::none()
                }
            }
            Message::RewindTo(i) => {
                if i + 1 >= self.messages.len() {
                    self.toast = Some("Nothing after this message to rewind.".into());
                } else {
                    self.messages.truncate(i + 1);
                    let now = chrono::Local::now().timestamp();
                    let mut hidden = 0;
                    if let Some(chat) = self.current.and_then(|id| self.store.chat_mut(id)) {
                        // the hidden part is kept in the chat, not thrown away (Rewind.tail)
                        let tail = chat.messages.split_off(i + 1);
                        hidden = tail.len();
                        if chat.rewinds.len() >= 20 {
                            chat.rewinds.remove(0);
                        }
                        chat.rewinds.push(store::Rewind { tail, at: now });
                    }
                    self.save();
                    // the notice with Undo instead of a toast that vanishes (rewindNotice)
                    self.rewind_notice = Some((hidden, now as u64 + 9));
                }
                Task::none()
            }
            Message::RewindUndo => {
                let mut restored: Vec<Record> = Vec::new();
                if let Some(chat) = self.current.and_then(|id| self.store.chat_mut(id)) {
                    if let Some(r) = chat.rewinds.pop() {
                        chat.messages.extend(r.tail.iter().cloned());
                        restored = r.tail;
                    }
                }
                if !restored.is_empty() {
                    self.messages.extend(restored.iter().map(Msg::from_record));
                    self.save();
                    self.rewind_notice = None;
                    self.toast = Some("Rewind undone".into());
                }
                Task::none()
            }
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
                if page == View::Settings {
                    // the settings of voice.js fill their selects with what the machine has
                    // (fillVoices and notifyNote run once when the pane opens)
                    let voices = Task::perform(
                        async {
                            // the programs answer at once (`--list-voices`, a spawned notify), so
                            // they may run where the task runs
                            Message::VoiceSystem {
                                voices: speech::list_voices(),
                                can_notify: speech::can_notify(),
                            }
                        },
                        |m| m,
                    );
                    // the prices of the usage pane come with them (loadPrices of usage.js)
                    let prices =
                        Task::perform(models::fetch_prices(self.base.clone()), Message::PricesLoaded);
                    if !matches!(self.server, Server::Down(_)) {
                        return Task::batch(vec![
                            voices,
                            prices,
                            Task::perform(
                                engine_settings::fetch(self.base.clone()),
                                Message::EngineLoaded,
                            ),
                        ]);
                    }
                    return Task::batch(vec![voices, prices]);
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
                // a model of a just-chosen assistant waits for this list
                if let Some(model) = self.asst_model_pending.take() {
                    return self.switch_model_of(&model);
                }
                Task::none()
            }
            Message::OverviewLoaded(Err(e)) => {
                self.models.fetched = true;
                self.models.error = Some(e);
                self.asst_model_pending = None;
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

    /// One event of the answer. The task of the next round comes back when the answer called a tool
    /// (the Studio's ctGenerate runs the call and asks again).
    fn on_event(&mut self, ev: StreamEvent) -> Option<Task<Message>> {
        if self.messages.is_empty() {
            return None;
        }
        match ev {
            StreamEvent::Delta(s) => {
                if let Some(last) = self.messages.last_mut() {
                    last.content.push_str(&s);
                    last.md.push_str(&s);
                }
            }
            StreamEvent::Logprobs(v) => {
                if let Some(last) = self.messages.last_mut() {
                    last.logprobs.extend(v);
                }
            }
            StreamEvent::Reasoning(s) => {
                if let Some(last) = self.messages.last_mut() {
                    last.reasoning.push_str(&s);
                }
            }
            StreamEvent::ToolCall { index, id, name, arguments } => {
                // the parts of the calls collect on the answer (the Studio's aNode.tools)
                if let Some(last) = self.messages.last_mut() {
                    let calls = last.tool_calls.get_or_insert_with(|| serde_json::json!([]));
                    let arr = calls.as_array_mut().expect("the tool calls are an array");
                    while arr.len() <= index {
                        arr.push(serde_json::json!({ "id": "", "type": "function", "function": { "name": "", "arguments": "" } }));
                    }
                    let call = &mut arr[index];
                    if let Some(id) = id {
                        call["id"] = serde_json::Value::String(id);
                    }
                    if let Some(name) = name {
                        call["function"]["name"] = serde_json::Value::String(name);
                    }
                    if !arguments.is_empty() {
                        let mut joined = call["function"]["arguments"].as_str().unwrap_or("").to_string();
                        joined.push_str(&arguments);
                        call["function"]["arguments"] = serde_json::Value::String(joined);
                    }
                }
            }
            StreamEvent::Usage { prompt, completion } => {
                self.used_tokens = prompt + completion;
                self.used_exact = true;
                // the counters of the usage pane take every answer (record of usage.js)
                let name = self
                    .hero
                    .as_ref()
                    .map(|s| s.model_name.clone())
                    .unwrap_or_default();
                crate::usage::record(&mut self.store.usage, &name, prompt, completion);
                self.save();
                if let Some(last) = self.messages.last_mut() {
                    last.stats.get_or_insert_with(Stats::default).completion_tokens = completion;
                }
            }
            StreamEvent::Timings { tokens_per_second, prompt_per_second, .. } => {
                if let Some(last) = self.messages.last_mut() {
                    let s = last.stats.get_or_insert_with(Stats::default);
                    s.tokens_per_second = tokens_per_second;
                    s.prompt_per_second = prompt_per_second;
                }
            }
            StreamEvent::Done => {
                if self.tool_calls_pending() {
                    return self.run_tool_round();
                }
                // the report of a research run ends with its sources (researchSourcesMarkdown)
                if let Some(sources) = self.research_sources.take() {
                    let text = research::sources_markdown(&sources);
                    if let Some(last) = self.messages.last_mut() {
                        last.content.push_str(&text);
                        last.md = markdown::Content::parse(&last.content);
                        if let Some(ui) = last.research.as_mut() {
                            let now = research::now_ms();
                            ui.seconds = now.saturating_sub(ui.t0_ms) / 1000;
                            ui.running = false;
                            ui.event((now.saturating_sub(ui.t0_ms)) as f64 / 1000.0, "Done.".to_string());
                        }
                    }
                }
                self.finish();
            }
            StreamEvent::Error(e) => {
                self.error = Some(e);
                // an answer that never started is not kept
                if self.messages.last().is_some_and(|m| m.role == Role::Assistant && m.content.is_empty() && m.reasoning.is_empty()) {
                    self.messages.pop();
                }
                self.finish();
            }
        }
        None
    }

    /// Whether the answer that just ended carries tool calls that still have to be answered.
    fn tool_calls_pending(&self) -> bool {
        self.messages.last().is_some_and(|m| matches!(m.role, Role::Assistant) && m.tool_calls.is_some())
    }

    /// The calls of the answer run here (the window offers the memory tool only), the model is told
    /// what happened and the answer continues (the Studio's ctGenerate with `env.memory`).
    fn run_tool_round(&mut self) -> Option<Task<Message>> {
        let chat = self.live.as_ref().map(|l| l.chat)?;
        let flags = self.tool_flags();
        if !flags.web && !flags.memory {
            // the switches went off while the answer ran: the calls are dropped, so no later
            // question ever carries them without their answers
            if let Some(last) = self.messages.last_mut() {
                last.tool_calls = None;
            }
            self.finish();
            return None;
        }
        if self.tool_round as usize >= tools::MAX_ROUNDS {
            self.finish();
            return None;
        }
        let calls = self.messages.last().and_then(|m| m.tool_calls.clone()).unwrap_or_else(|| serde_json::json!([]));
        let jobs: Vec<(String, String, String)> = calls
            .as_array()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .take(tools::MAX_CALLS)
            .map(|call| {
                let id = call.get("id").and_then(serde_json::Value::as_str).unwrap_or_default().to_string();
                let name = call.pointer("/function/name").and_then(serde_json::Value::as_str).unwrap_or_default().to_string();
                let raw_args = call.pointer("/function/arguments").and_then(serde_json::Value::as_str).unwrap_or("{}").to_string();
                (id, name, raw_args)
            })
            .collect();
        if jobs.is_empty() {
            self.finish();
            return None;
        }
        // what one answer may still take from the server for web results (ctBudget, one per answer)
        let budget = self.answer_budget.unwrap_or_else(|| tools::budget(self.model_info().max_context));
        let sources = std::mem::take(&mut self.answer_sources);
        let now = store::now_millis();
        self.tool_hold = Vec::new();
        self.tool_web_jobs = Vec::new();
        let fetcher = api::EngineFetcher { base: self.base.clone() };
        for (index, (id, name, raw_args)) in jobs.into_iter().enumerate() {
            let memory_here = name == "memory" && flags.memory;
            let web_here = name == "web_search" || name == "web_fetch";
            if !web_here {
                // the memory calls are answered in this window (they need no server); a call of a
                // tool this chat does not have gets the Studio's line about it
                let mem_list = std::sync::Arc::new(std::sync::Mutex::new(self.store.prefs.personal.memories.clone()));
                let mem_for_closure = std::sync::Arc::clone(&mem_list);
                let mut env = tools::Env {
                    budget,
                    sources: Vec::new(),
                    memory: memory_here.then(|| {
                        Box::new(move |args: &serde_json::Value| {
                            let mut guard = mem_for_closure.lock().expect("the memory of the round");
                            let outcome = memory::tool_apply(&guard, args, now, || store::uid("mem"));
                            *guard = outcome.list.clone();
                            tools::MemoryRun { text: outcome.text, change: outcome.change, list: outcome.list }
                        })
                            as Box<dyn FnMut(&serde_json::Value) -> tools::MemoryRun + Send>
                    }),
                };
                let outcome = futures::executor::block_on(tools::run(&name, &raw_args, &mut env, &fetcher));
                if memory_here {
                    let final_list = mem_list.lock().expect("the memory of the round").clone();
                    if final_list != self.store.prefs.personal.memories {
                        self.store.prefs.personal.memories = final_list;
                        self.save();
                    }
                }
                let card = crate::memory::ToolCard {
                    label: outcome.card.label,
                    status: outcome.card.status.to_string(),
                    change: outcome.card.change,
                    undone: false,
                    quiet: outcome.card.quiet,
                };
                self.tool_hold.push(Some((id, outcome.text, card)));
            } else {
                self.tool_hold.push(None);
                self.tool_web_jobs.push((index, id, name, raw_args));
            }
        }
        if self.tool_web_jobs.is_empty() {
            // everything of the round is answered here: the next round starts at once
            return self.tool_round_finalize(chat);
        }
        // the web calls go to the server, one after the other (the Studio awaits them in order)
        let jobs = std::mem::take(&mut self.tool_web_jobs);
        let base = self.base.clone();
        let task = Task::perform(
            async move {
                let fetcher = api::EngineFetcher { base };
                let mut env = tools::Env { budget, sources, memory: None };
                let mut results = Vec::new();
                for (index, id, name, raw_args) in jobs {
                    let outcome = tools::run(&name, &raw_args, &mut env, &fetcher).await;
                    let card = crate::memory::ToolCard {
                        label: outcome.card.label,
                        status: outcome.card.status.to_string(),
                        change: outcome.card.change,
                        undone: false,
                        quiet: outcome.card.quiet,
                    };
                    results.push((index, id, outcome.text, card));
                }
                ToolRoundResult { results, budget_left: env.budget.left, sources: env.sources }
            },
            Message::ToolRound,
        );
        Some(task)
    }

    /// Everything of the round has its answer: the results in the order of the calls go back to
    /// the model, the cards stand under it, and the next round starts (the Studio's loop).
    fn tool_round_finalize(&mut self, chat: u64) -> Option<Task<Message>> {
        let mut cards = Vec::new();
        let mut results: Vec<(String, String)> = Vec::new();
        for slot in std::mem::take(&mut self.tool_hold) {
            let Some((id, text, card)) = slot else { continue };
            results.push((id, text));
            cards.push(card);
        }
        if let Some(last) = self.messages.last_mut() {
            last.tool_cards.extend(cards);
        }
        for (id, text) in results {
            self.messages.push(Msg {
                role: Role::Tool,
                content: text,
                reasoning: String::new(),
                stats: None,
                passages: None,
                images: Vec::new(),
                files: Vec::new(),
                tool_calls: None,
                tool_call_id: Some(id),
                tool_cards: Vec::new(),
                sources: Vec::new(),
                research: None,
                logprobs: Vec::new(),
                reasoning_open: false,
                md: markdown::Content::new(),
            });
        }
        self.messages.push(Msg {
            role: Role::Assistant,
            content: String::new(),
            reasoning: String::new(),
            stats: None,
            passages: None,
            images: Vec::new(),
            files: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
            tool_cards: Vec::new(),
            sources: Vec::new(),
            research: None,
            logprobs: Vec::new(),
            reasoning_open: false,
            md: markdown::Content::new(),
        });
        self.tool_round += 1;
        let Server::Ready(model) = self.server.clone() else {
            self.finish();
            return None;
        };
        Some(self.stream(chat, model.id))
    }

    /// The reply is over (finished, stopped or failed): save it and free the composer.
    fn finish(&mut self) {
        if let Some(live) = self.live.take() {
            let ms = live.started.elapsed().as_millis() as u64;
            self.sync_current();
            if let Some(chat) = self.current.and_then(|id| self.store.chat_mut(id)) {
                chat.forget_duplicates();
            }
            self.save();
            // watchRun of voice.js: what it does when a run ends
            self.after_run(ms);
        }
    }

    /// What happens when a run ends: the answer may be read alone, and a long run says something
    /// when the window is behind (`watchRun` of `voice.js` — twenty seconds for an answer, five
    /// for an agent run).
    pub fn after_run(&mut self, ms: u64) {
        let finished = self
            .messages
            .last()
            .filter(|m| !matches!(m.role, Role::User) && !m.content.trim().is_empty())
            .map(|m| m.content.clone());
        let Some(content) = finished else {
            return;
        };
        let voice = self.store.prefs.voice.clone();
        if voice.auto {
            // the studio reads the finished answer aloud (strip and chunk are speech.rs's)
            let parts = speech::chunk_for_speech(&speech::strip_for_speech(&content), 0);
            for part in parts {
                speech::speak_with(&part, &voice);
            }
        }
        if ms >= 20_000 {
            self.announce(String::from("Your answer is ready."));
        }
    }

    /// The alert of `voice.js`: silent while the window is in front, the short tone when that is
    /// on, the system notification when that is on and allowed.
    pub fn announce(&mut self, body: String) {
        if self.window_focused {
            return;
        }
        let voice = self.store.prefs.voice.clone();
        if voice.sound {
            speech::chime();
        }
        if voice.notify {
            speech::notify(&body);
        }
        self.toast = Some(format!("Alert: {body}"));
    }

    pub fn subscription(&self) -> Subscription<Message> {
        // a server that was told to switch its model is back within seconds: look more often then
        let every = if self.switching { 1 } else { 4 };
        let retry = if matches!(self.server, Server::Down(_)) { time::every(Duration::from_secs(every)).map(|_| Message::Tick) } else { Subscription::none() };
        // the start screen keeps its line of tokens per second up to date (the Studio polls every 2000 ms)
        let hero = if !matches!(self.server, Server::Down(_)) {
            time::every(Duration::from_secs(2)).map(|_| Message::HeroPoll)
        } else {
            Subscription::none()
        };
        let poll = if (self.page == View::Models && !matches!(self.server, Server::Down(_))) || self.models.any_active() {
            time::every(Duration::from_secs(1)).map(|_| Message::Poll)
        } else {
            Subscription::none()
        };
        // files dragged onto the window are attached the way the dialog attaches them; Esc ends a
        // running answer (the title of the Studio's send button promises it)
        let dropped = iced::event::listen_with(|event, _status, _id| match event {
            iced::Event::Window(iced::window::Event::FileDropped(path)) => Some(Message::FilesDropped(vec![path])),
            iced::Event::Keyboard(iced::keyboard::Event::KeyPressed {
                key: keyboard::Key::Named(keyboard::key::Named::Escape),
                ..
            }) => Some(Message::Esc),
            iced::Event::Keyboard(iced::keyboard::Event::KeyPressed { key, modifiers, .. }) => {
                Some(Message::Key(key, modifiers))
            }
            iced::Event::Window(iced::window::Event::Focused) => Some(Message::WindowFocus(true)),
            iced::Event::Window(iced::window::Event::Unfocused) => {
                Some(Message::WindowFocus(false))
            }
            _ => None,
        });
        Subscription::batch([retry, poll, hero, dropped])
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
        if let Some((hidden, _)) = self.rewind_notice {
            main = main.push(
                container(
                    row![
                        text(format!("Rewound: {hidden} message(s) after this one are hidden.")).size(13).color(t.fg_soft),
                        button(text("Undo").size(12)).on_press(Message::RewindUndo).padding([2, 8]).style(theme::ghost_button(t)),
                    ]
                    .spacing(10)
                    .align_y(Alignment::Center),
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

        let projects = self.prj_sidebar(t);

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
                projects,
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
        let page: Element<'_, Message> = if self.page == View::Settings {
            // the dialogs of this page (forget everything) open over it: it joins the stack below
            settings::page(
                t,
                &self.store.prefs,
                &self.system_editor,
                &self.spec_editor,
                &self.max_tokens_text,
                &self.sampling_text,
                engine_settings::view(t, &self.engine, !matches!(self.server, Server::Down(_))),
                self.personal_view(),
                settings::VoiceView {
                    voices: &self.voice_names,
                    can_notify: self.voice_can_notify,
                },
                settings::UsageView {
                    data: &self.store.usage,
                    prices: &self.prices,
                    status: self.usage_status.as_deref(),
                },
            )
        } else if self.page == View::Project {
            // the page of a project sits where the conversation sits, so the dialogs can open over it
            self.project_page(t)
        } else {
            let content: Element<'_, Message> = match &self.server {
                Server::Down(_) | Server::Connecting if self.switching => self.switching_card(t),
                Server::Down(reason) => self.offline(t, reason),
                Server::NoModel => self.no_model(t),
                _ if self.messages.is_empty() => self.welcome(t),
                _ => self.conversation(t),
            };
            let chat = column![self.topbar(t), content, self.composer(t)].width(Length::Fill).height(Length::Fill);
            // the panel of the Studio takes its side of the chat; the large view is a layer over
            // everything (the fixed box of `.art-panel.is-large`)
            if self.art.is_some() && !self.art.as_ref().is_some_and(|p| p.large) {
                row![chat, container(self.art_panel(t)).width(Length::Fill).max_width(580.0).height(Length::Fill)]
                    .spacing(0)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .into()
            } else {
                chat.into()
            }
        };
        if !self.picker_open
            && !self.asst_menu
            && self.asst_edit.is_none()
            && self.prj_pick.is_none()
            && self.prj_delete.is_none()
            && self.remember.is_none()
            && !self.mem_clear
            && !self.temp_close
            && self.tour.is_none()
            && !self.shortcuts_open
            && !self.palette_open
            // the large view of the artifact panel is a layer of its own
            && !self.art.as_ref().is_some_and(|p| p.large)
        {
            return page;
        }
        let mut layers = stack![page];
        if self.art.as_ref().is_some_and(|p| p.large) {
            // the large view of the panel: dimmed over the chat, the panel itself in the middle
            // (the fixed inset of `.art-panel.is-large`)
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::ArtAction(artifacts::ArtAction::Larger)))
                .push(
                    container(self.art_panel(t))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .max_width(1100.0)
                        .padding(24)
                        .style(move |_| iced::widget::container::Style {
                            background: Some(iced::Background::Color(t.surface)),
                            border: iced::Border { radius: 12.0.into(), width: 1.0, color: t.separator },
                            ..Default::default()
                        }),
                );
        }
        if self.picker_open {
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::ClosePicker))
                .push(self.picker(t));
        }
        if self.asst_menu {
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::AsstMenuClose))
                .push(self.asst_menu_panel(t));
        }
        if self.asst_edit.is_some() {
            // the editor is the top layer; a click beside it only takes the click (the menu is closed then)
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::AsstMenuClose))
                .push(self.asst_editor(t));
        }
        if self.prj_pick.is_some() {
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::PrjPick(None)))
                .push(self.prj_pick_window(t));
        }
        if self.prj_delete.is_some() {
            // the dialog of the Studio's askDialog: a click beside it only takes the click
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::PrjDeleteCancel))
                .push(self.prj_delete_window(t));
        }
        if self.mem_clear {
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::MemClearCancel))
                .push(self.mem_clear_window(t));
        }
        if self.remember.is_some() {
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::RememberCancel))
                .push(self.remember_window(t));
        }
        if self.temp_close {
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::TempCloseCancel))
                .push(self.temp_close_window(t));
        }
        if self.tour.is_some() {
            // the backdrop of `.modal-backdrop`: a click beside the window closes the tour
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::TourSkip))
                .push(self.tour_window(t));
        }
        if self.shortcuts_open {
            // the backdrop of `.modal-backdrop`: a click beside the window closes the overview
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::ShortcutsClose))
                .push(self.shortcuts_window(t));
        }
        if self.palette_open {
            // the backdrop of `.modal-backdrop`: a click beside the window closes the palette
            layers = layers
                .push(mouse_area(Space::new().width(Length::Fill).height(Length::Fill)).on_press(Message::PaletteClose))
                .push(self.palette_window(t));
        }
        layers.into()
    }

    /// The artifact panel: header with the versions, the tabs and tools of the Studio, and the
    /// body — the code, the rendered Markdown, the table of a CSV, or the text of a page (the app
    /// draws no HTML: *Open in browser* shows it rendered there).
    fn art_panel(&self, t: Tokens) -> Element<'_, Message> {
        use artifacts::{ArtAction as Act, Kind, Tab};
        let Some(p) = &self.art else { return Space::new().width(0).height(0).into() };
        let kind_label = if p.fam.title.is_empty() { "" } else { p.fam.kind.label() };
        let title = if p.fam.title.is_empty() { p.fam.kind.label().to_string() } else { p.fam.title.clone() };
        let head = row![
            column![text(title).size(13.5).font(bold()), text(kind_label).size(11.5).color(t.muted)].spacing(1),
            Space::new().width(Length::Fill),
            {
                let versions: Element<'_, Message> = if p.fam.versions.len() > 1 {
                    row![
                        iced::widget::button(text("‹").size(14))
                            .on_press(Message::ArtAction(Act::Prev))
                            .padding([0, 8])
                            .style(theme::ghost_button(t)),
                        text(format!("v{} / {}", p.fam.versions[p.ver].n, p.fam.versions.len())).size(11.5).color(t.muted),
                        iced::widget::button(text("›").size(14))
                            .on_press(Message::ArtAction(Act::Next))
                            .padding([0, 8])
                            .style(theme::ghost_button(t)),
                    ]
                    .spacing(2)
                    .align_y(Alignment::Center)
                    .into()
                } else {
                    Space::new().width(0).height(0).into()
                };
                versions
            },
            iced::widget::button(text("Close").size(12)).on_press(Message::ArtAction(Act::Close)).padding([2, 10]).style(theme::ghost_button(t)),
        ]
        .spacing(8)
        .align_y(Alignment::Center)
        .padding([8, 10]);
        let mut tools = row![].spacing(2);
        if p.fam.kind != Kind::Code && p.tab == Tab::Preview && matches!(p.fam.kind, Kind::Html | Kind::Svg) {
            tools = tools.push(iced::widget::button(text("Reload").size(12)).on_press(Message::ArtAction(Act::Reload)).padding([2, 8]).style(theme::ghost_button(t)));
        }
        tools = tools
            .push(iced::widget::button(text("Copy").size(12)).on_press(Message::ArtAction(Act::Copy)).padding([2, 8]).style(theme::ghost_button(t)))
            .push(iced::widget::button(text("Download").size(12)).on_press(Message::ArtAction(Act::Download)).padding([2, 8]).style(theme::ghost_button(t)))
            .push(
                iced::widget::button(text(if p.large { "Smaller" } else { "Larger" }).size(12))
                    .on_press(Message::ArtAction(Act::Larger))
                    .padding([2, 8])
                    .style(theme::ghost_button(t)),
            );
        let mut toolbar = row![].spacing(8).align_y(Alignment::Center).padding([6, 10]);
        if p.fam.kind != Kind::Code {
            toolbar = toolbar.push(settings::segmented(
                t,
                vec![
                    ("Preview", p.tab == Tab::Preview, Message::ArtAction(Act::TabPreview)),
                    ("Code", p.tab == Tab::Code, Message::ArtAction(Act::TabCode)),
                ],
            ));
        }
        toolbar = toolbar.push(Space::new().width(Length::Fill)).push(tools);
        let body: Element<'_, Message> = match (p.tab, p.fam.kind) {
            (Tab::Code, _) | (_, Kind::Html) | (_, Kind::Svg) => {
                let mut col = column![].spacing(8);
                if p.tab == Tab::Preview && matches!(p.fam.kind, Kind::Html | Kind::Svg) {
                    // no web view in this window: the page is opened in the browser of the computer
                    col = col.push(
                        iced::widget::button(text("Open in browser").size(13))
                            .on_press(Message::ArtAction(Act::OpenInBrowser))
                            .padding([6, 12])
                            .style(theme::ghost_button(t)),
                    );
                }
                col = col.push(
                    scrollable(
                        container(text(p.version().code.clone()).size(12.5).color(t.fg_soft).font(Font { weight: iced::font::Weight::Normal, ..crate::app::UI_FONT }).width(Length::Fill))
                            .padding([10, 12]),
                    )
                    .direction(scrollable::Direction::Vertical(scrollable::Scrollbar::default().width(6).scroller_width(6))),
                );
                col.into()
            }
            (_, Kind::Markdown) | (_, Kind::Csv) => {
                let body = p.body.as_ref().expect("the rendered body of the preview");
                scrollable(container(md::view(body, t, &self.theme())).padding([4, 2])).into()
            }
            _ => Space::new().width(0).height(0).into(),
        };
        let mut col = column![head, toolbar].spacing(0);
        col = col.push(container(body).padding([8, 6]).height(Length::Fill).width(Length::Fill));
        container(col)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(move |_| iced::widget::container::Style {
                background: Some(iced::Background::Color(t.surface)),
                border: iced::Border { width: 1.0, color: t.separator, ..Default::default() },
                ..Default::default()
            })
            .into()
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

    /// The projects over the chats in the sidebar (the Studio's projectList): a row per project with
    /// its chats when it is open, the input row for a new one and the rename row where it is asked for.
    fn prj_sidebar(&self, t: Tokens) -> Element<'_, Message> {
        let head = container(
            row![
                text("Projects").size(12).font(bold()).color(t.muted),
                Space::new().width(Length::Fill),
                button(svg(icon(ICON_PLUS)).width(14).height(14).style(muted_svg(t)))
                    .on_press(Message::PrjNew)
                    .padding(3)
                    .style(theme::ghost_button(t)),
            ]
            .spacing(4)
            .align_y(Alignment::Center),
        )
        .padding(Padding { top: 16.0, right: 10.0, bottom: 4.0, left: 10.0 });
        let mut col = column![head].spacing(2);
        if self.prj_creating {
            col = col.push(
                row![
                    asst_text_input(t, "New project", &self.prj_new_name, Message::PrjNewInput, Length::Fill),
                    button(text("OK").size(12)).on_press(Message::PrjNewEnd(true)).padding([4, 8]).style(theme::secondary_button(t)),
                    button(text("×").size(14)).on_press(Message::PrjNewEnd(false)).padding([2, 6]).style(theme::ghost_button(t)),
                ]
                .spacing(4)
                .align_y(Alignment::Center),
            );
        } else if self.store.projects.is_empty() {
            col = col.push(container(text("No projects yet").size(13).color(t.muted)).padding([6, 10]));
        }
        for p in &self.store.projects {
            let open = self.prj_open.contains(&p.id);
            if self.prj_renaming.as_ref().is_some_and(|(id, _)| *id == p.id) {
                let typed = self.prj_renaming.as_ref().map(|(_, n)| n.as_str()).unwrap_or("");
                col = col.push(
                    row![
                        asst_text_input(t, "Project name", typed, Message::PrjRenameInput, Length::Fill),
                        button(text("OK").size(12)).on_press(Message::PrjRenameEnd(true)).padding([4, 8]).style(theme::secondary_button(t)),
                        button(text("×").size(14)).on_press(Message::PrjRenameEnd(false)).padding([2, 6]).style(theme::ghost_button(t)),
                    ]
                    .spacing(4)
                    .align_y(Alignment::Center),
                );
                continue;
            }
            let mut chats: Vec<&store::ChatRecord> = self
                .store
                .chats
                .iter()
                .filter(|c| c.project.as_deref() == Some(p.id.as_str()) && !c.archived)
                .collect();
            chats.sort_by_key(|c| std::cmp::Reverse(c.updated));
            let line = row![
                button(svg(icon(if open { ICON_CHEVRON } else { ICON_CHEVRON_RIGHT })).width(13).height(13).style(muted_svg(t)))
                    .on_press(Message::PrjToggle(p.id.clone()))
                    .padding(4)
                    .style(theme::ghost_button(t)),
                button(
                    row![
                        svg(icon(ICON_FOLDER)).width(14).height(14).style(accent_svg(t)),
                        text(ellipsize(&p.name, 16)).size(13),
                    ]
                    .spacing(6)
                    .align_y(Alignment::Center),
                )
                .on_press(Message::PrjOpen(p.id.clone()))
                .padding([4, 2])
                .width(Length::Fill)
                .style(theme::plain_button(t)),
                text(chats.len().to_string()).size(11).color(t.muted),
                button(svg(icon(assistants::icon_bytes("pen"))).width(12).height(12).style(muted_svg(t)))
                    .on_press(Message::PrjRenameStart(p.id.clone()))
                    .padding(4)
                    .style(theme::ghost_button(t)),
                button(svg(icon(ICON_X)).width(12).height(12).style(muted_svg(t)))
                    .on_press(Message::PrjDelete(p.id.clone()))
                    .padding(4)
                    .style(theme::ghost_button(t)),
            ]
            .spacing(2)
            .align_y(Alignment::Center);
            col = col.push(container(line).padding(Padding { top: 2.0, right: 4.0, bottom: 2.0, left: 4.0 }));
            if open {
                if chats.is_empty() {
                    col = col
                        .push(container(text("No chats yet").size(12).color(t.muted)).padding(Padding { top: 2.0, right: 10.0, bottom: 4.0, left: 28.0 }));
                }
                for c in chats {
                    let active = self.current == Some(c.id);
                    let row_i = button(text(ellipsize(&c.title, 24)).size(13).wrapping(text::Wrapping::None))
                        .on_press_maybe(self.live.is_none().then_some(Message::PrjChat(c.id)))
                        .padding([6, 10])
                        .width(Length::Fill)
                        .style(theme::row_button(t, active));
                    col = col.push(container(row_i).padding(Padding { top: 0.0, right: 4.0, bottom: 0.0, left: 22.0 }));
                }
            }
        }
        col.into()
    }

    /// The page of one project: its name, instructions, knowledge and chats (the Studio's project window).
    fn project_page(&self, t: Tokens) -> Element<'_, Message> {
        let empty = container(Space::new().width(Length::Fill).height(Length::Fill));
        let Some(id) = self.prj_view.clone() else { return empty.into() };
        let Some(p) = self.store.projects.iter().find(|p| p.id == id) else { return empty.into() };
        let head = row![
            text(p.name.clone()).size(26).font(bold()),
            Space::new().width(Length::Fill),
            button(text("New chat in project").size(13)).on_press(Message::PrjNewChat).padding([6, 12]).style(theme::primary_button(t)),
            button(text("Rename").size(13)).on_press(Message::PrjRenameStart(p.id.clone())).padding([6, 12]).style(theme::secondary_button(t)),
        ]
        .spacing(8)
        .align_y(Alignment::Center);
        let usage = text(format!(
            "{} of {} characters used. Only the extracted text is stored, on this computer.",
            group(p.used() as u64),
            group(rag::limits::PROJECT_USAGES as u64)
        ))
        .size(12)
        .color(t.muted);
        let instructions = column![
            text("Instructions for every chat of this project").size(13).color(t.fg_soft),
            asst_area(t, &self.prj_instr, Message::PrjInstr, "For example: Answer as a support engineer for your product.", 90.0),
        ]
        .spacing(4);
        let knowledge = column![
            row![
                text("Knowledge").size(13).color(t.fg_soft),
                Space::new().width(Length::Fill),
                button(text("Add files").size(12)).on_press(Message::PrjAddFiles).padding([4, 10]).style(theme::secondary_button(t)),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        ]
        .spacing(4);
        let mut docs = column![].spacing(4);
        if p.docs.is_empty() {
            docs = docs.push(text("No documents yet. Add files that every chat of this project should be able to search.").size(13).color(t.muted));
        } else {
            for d in &p.docs {
                let size = format!("{}{}", format_bytes(d.chars as u64), d.pages.map(|n| format!(", {n} pages")).unwrap_or_default());
                docs = docs.push(
                    row![
                        svg(icon(ICON_FILE)).width(14).height(14).style(muted_svg(t)),
                        text(ellipsize(&d.name, 30)).size(13).width(Length::Fill),
                        text(size).size(12).color(t.muted),
                        button(svg(icon(ICON_X)).width(12).height(12).style(muted_svg(t)))
                            .on_press(Message::PrjDocRemove(d.id.clone()))
                            .padding(4)
                            .style(theme::ghost_button(t)),
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center),
                );
            }
        }
        let note = text("Every question in a chat of this project searches these documents; the best passages are shown under your message.")
            .size(12)
            .color(t.muted);
        let mut chats: Vec<&store::ChatRecord> = self.store.chats.iter().filter(|c| c.project.as_deref() == Some(p.id.as_str())).collect();
        chats.sort_by_key(|c| std::cmp::Reverse(c.updated));
        let mut chat_rows = column![].spacing(2);
        if chats.is_empty() {
            chat_rows = chat_rows.push(text("No chats in this project yet.").size(13).color(t.muted));
        } else {
            for c in chats {
                let date = chrono::Local
                    .timestamp_opt(c.updated, 0)
                    .single()
                    .map(|d| d.format("%m/%d/%Y").to_string())
                    .unwrap_or_default();
                let label = format!("{}{date}", if c.archived { "archived · " } else { "" });
                chat_rows = chat_rows.push(
                    button(
                        row![
                            text(ellipsize(&c.title, 34)).size(13).width(Length::Fill),
                            text(label).size(11).color(t.muted),
                        ]
                        .spacing(8)
                        .align_y(Alignment::Center),
                    )
                    .on_press_maybe(self.live.is_none().then_some(Message::PrjChat(c.id)))
                    .padding([7, 10])
                    .width(Length::Fill)
                    .style(theme::row_button(t, self.current == Some(c.id))),
                );
            }
        }
        let footer = row![
            button(text("Delete project").size(13))
                .on_press(Message::PrjDelete(p.id.clone()))
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
            Space::new().width(Length::Fill),
            button(text("Close").size(13)).on_press(Message::PrjClose).padding([6, 12]).style(theme::ghost_button(t)),
        ]
        .spacing(8)
        .align_y(Alignment::Center);
        let content = column![head, usage, instructions, knowledge, docs, note, text("Chats").size(13).color(t.fg_soft), chat_rows, footer].spacing(12);
        let inner = container(content).max_width(780).width(Length::Fill).padding(Padding { top: 14.0, right: 24.0, bottom: 32.0, left: 24.0 });
        scrollable(container(inner).center_x(Length::Fill).width(Length::Fill)).height(Length::Fill).into()
    }

    /// The dialog of the *Remember* button under a message (the Studio's askDialog for prjRemember).
    fn remember_window(&self, t: Tokens) -> Element<'_, Message> {
        let window = container(
            column![
                text("Remember").size(16).font(bold()),
                text("This is added to the instructions of every chat. Edit it first if you like. Memories stay on this computer.")
                    .size(13)
                    .color(t.fg_soft),
                text_input("Remember", &self.remember_text)
                    .on_input(Message::RememberEdit)
                    .on_submit(Message::RememberOk)
                    .size(14)
                    .padding([8, 10])
                    .style(move |_, status| text_input::Style {
                        background: iced::Background::Color(t.surface_2),
                        border: iced::Border { radius: 8.0.into(), width: 1.0, color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator } },
                        icon: t.muted,
                        placeholder: t.muted,
                        value: t.fg,
                        selection: t.primary_soft,
                    }),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Cancel").size(13)).on_press(Message::RememberCancel).padding([6, 12]).style(theme::ghost_button(t)),
                    button(text("Remember").size(13)).on_press(Message::RememberOk).padding([6, 12]).style(theme::primary_button(t)),
                ],
            ]
            .spacing(10),
        )
        .padding(16)
        .width(460)
        .style(theme::card(t, theme::R_LG, false));
        container(window).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into()
    }

    /// The welcome tour of the Studio (`tourModal` of `shell.html`, the steps and buttons of
    /// `voice.js`): the dots, the title, what the step says, and Skip / Back / Next.
    fn tour_window(&self, t: Tokens) -> Element<'_, Message> {
        let step = self.tour.unwrap_or(0);
        let last = step + 1 == tour::STEPS.len();
        // the dots above the title (`tour-dots`)
        let mut dots: Vec<Element<'_, Message>> = Vec::new();
        for i in 0..tour::STEPS.len() {
            let on = i == step;
            dots.push(
                container(Space::new().width(7).height(7))
                    .style(move |_| iced::widget::container::Style {
                        background: Some(iced::Background::Color(if on { t.primary } else { t.separator })),
                        border: iced::Border {
                            radius: 4.0.into(),
                            width: 0.0,
                            color: iced::Color::TRANSPARENT,
                        },
                        ..Default::default()
                    })
                    .into(),
            );
        }
        let mut parts: Vec<Element<'_, Message>> = vec![
            row(dots).spacing(6).into(),
            text(tour::STEPS[step]).size(16).font(bold()).into(),
        ];

        // what this step says (`tourBody`)
        match self.tour_data.body(step) {
            tour::Body::Welcome => {
                for paragraph in tour::WELCOME {
                    parts.push(text(paragraph).size(13).color(t.fg_soft).into());
                }
            }
            tour::Body::Computer { text: about, hint } => {
                // the box of `.tour-box` around the line about this computer
                parts.push(
                    container(text(about).size(13).color(t.fg))
                        .padding([8, 10])
                        .width(Length::Fill)
                        .style(move |_| iced::widget::container::Style {
                            background: Some(iced::Background::Color(t.surface_2)),
                            border: iced::Border {
                                radius: 8.0.into(),
                                width: 1.0,
                                color: t.separator,
                            },
                            ..Default::default()
                        })
                        .into(),
                );
                if !hint.is_empty() {
                    parts.push(text(hint).size(13).color(t.fg_soft).into());
                }
            }
            tour::Body::Model {
                loaded,
                hint,
                suggested,
                open_models,
            } => {
                if let Some(loaded) = loaded {
                    // `A model is already loaded: <strong>name</strong> You can start ...`
                    parts.push(
                        row![
                            text("A model is already loaded: ").size(13).color(t.fg),
                            text(loaded).size(13).font(bold()),
                            text(hint).size(13).color(t.fg),
                        ]
                        .spacing(0)
                        .into(),
                    );
                } else {
                    parts.push(text(hint).size(13).color(t.fg_soft).into());
                }
                if !suggested.is_empty() {
                    parts.push(
                        container(
                            text(format!("Suggested: {}", suggested.join(", ")))
                                .size(13)
                                .color(t.fg),
                        )
                        .padding([8, 10])
                        .width(Length::Fill)
                        .style(move |_| iced::widget::container::Style {
                            background: Some(iced::Background::Color(t.surface_2)),
                            border: iced::Border {
                                radius: 8.0.into(),
                                width: 1.0,
                                color: t.separator,
                            },
                            ..Default::default()
                        })
                        .into(),
                    );
                }
                if open_models {
                    parts.push(
                        button(text("Open Models").size(13))
                            .on_press(Message::TourOpenModels)
                            .padding([6, 12])
                            .style(theme::ghost_button(t))
                            .into(),
                    );
                }
            }
            tour::Body::Tips(tips) => {
                let mut list: Vec<Element<'_, Message>> = Vec::new();
                for tip in tips {
                    list.push(
                        row![
                            text(format!("• {}", tip.lead)).size(13).font(bold()),
                            text(tip.rest).size(13).color(t.fg_soft),
                        ]
                        .spacing(4)
                        .into(),
                    );
                }
                parts.push(column(list).spacing(8).into());
            }
        }

        // the actions: Skip, Back, Next — the last step says *Get started* and hides Skip
        let mut actions: Vec<Element<'_, Message>> = vec![Space::new().width(Length::Fill).into()];
        if !last {
            actions.push(
                button(text("Skip").size(13))
                    .on_press(Message::TourSkip)
                    .padding([6, 12])
                    .style(theme::ghost_button(t))
                    .into(),
            );
        }
        if step > 0 {
            actions.push(
                button(text("Back").size(13))
                    .on_press(Message::TourBack)
                    .padding([6, 12])
                    .style(theme::ghost_button(t))
                    .into(),
            );
        }
        actions.push(
            button(text(if last { "Get started" } else { "Next" }).size(13))
                .on_press(Message::TourNext)
                .padding([6, 12])
                .style(theme::primary_button(t))
                .into(),
        );
        parts.push(row(actions).spacing(8).into());

        let window = container(column(parts).spacing(12))
            .padding(16)
            .width(460)
            .style(theme::card(t, theme::R_LG, false));
        container(window)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .into()
    }

    /// The command palette (the `searchModal` of `shell.html`): the field, the rows in their
    /// groups with the active one marked, and the foot with its help and count.
    fn palette_window(&self, t: Tokens) -> Element<'_, Message> {
        let field = text_input(
            "Type a command, or search chats and settings",
            &self.palette_query,
        )
        .on_input(Message::PaletteQuery)
        .on_submit(Message::PaletteRun(
            self.palette_active.max(0) as usize,
        ))
        .size(14)
        .padding([10, 12])
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
        });

        let mut body: Vec<Element<'_, Message>> = vec![field.into()];
        if self.palette_rows.is_empty() {
            // the empty line of palette.js: nothing found for what was typed
            body.push(
                container(
                    text(format!(
                        "Nothing found for \"{}\".",
                        self.palette_query.trim()
                    ))
                    .size(13)
                    .color(t.muted),
                )
                .padding([12, 4])
                .into(),
            );
        } else {
            let mut rows: Vec<Element<'_, Message>> = Vec::new();
            let mut last_kind: Option<palette::Kind> = None;
            for (i, (item, _)) in self.palette_rows.iter().enumerate() {
                if last_kind != Some(item.kind) {
                    last_kind = Some(item.kind);
                    rows.push(
                        container(text(item.kind.label()).size(11).color(t.muted))
                            .padding(iced::Padding {
                                top: 8.0,
                                right: 4.0,
                                bottom: 2.0,
                                left: 4.0,
                            })
                            .into(),
                    );
                }
                let active = self.palette_active == i as i32;
                let sub: Element<'_, Message> = if item.sub.is_empty() {
                    Space::new().width(0).into()
                } else {
                    text(item.sub.as_str()).size(11).color(t.muted).into()
                };
                let hint_chip: Element<'_, Message> = if item.hint.is_empty() {
                    Space::new().width(0).into()
                } else {
                        container(text(item.hint.as_str()).size(11).color(t.fg_soft))
                            .padding([1, 5])
                            .style(move |_| iced::widget::container::Style {
                                background: Some(iced::Background::Color(t.surface_2)),
                                border: iced::Border {
                                    radius: 5.0.into(),
                                    width: 1.0,
                                    color: t.separator,
                                },
                                ..Default::default()
                            })
                        .into()
                };
                let title = row![
                    text(item.title.as_str()).size(13).color(t.fg).width(Length::Fill),
                    sub,
                    hint_chip,
                ]
                .spacing(8)
                .align_y(Alignment::Center);
                rows.push(
                    button(title)
                        .on_press(Message::PaletteRun(i))
                        .width(Length::Fill)
                        .padding([7, 8])
                        .style(move |style_theme, status| {
                            let mut s = (theme::plain_button(t))(style_theme, status);
                            if active {
                                s.background = Some(iced::Background::Color(t.primary_soft));
                            }
                            s
                        })
                        .into(),
                );
            }
            body.push(column(rows).spacing(1).into());
        }
        // the foot: how to move, and how many rows there are (the count of palette.js)
        let count = self.palette_rows.len();
        let count_text = if count == 0 {
            String::from("No results")
        } else {
            format!("{count} result{}", if count == 1 { "" } else { "s" })
        };
        body.push(
            row![
                text("Up Down to move, Enter to open, Esc to close")
                    .size(11)
                    .color(t.muted)
                    .width(Length::Fill),
                text(count_text).size(11).color(t.muted),
            ]
            .spacing(8)
            .padding(iced::Padding {
                top: 6.0,
                right: 2.0,
                bottom: 0.0,
                left: 2.0,
            })
            .into(),
        );

        let window = container(column(body).spacing(8))
            .padding(12)
            .width(600)
            .style(theme::card(t, theme::R_LG, false));
        container(window)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .into()
    }

    /// The overview of the keyboard shortcuts (the `helpModal` of `shell.html` with
    /// `renderShortcutsPanel` of `boot.js`): what every key does, the keys it has now, Change /
    /// Cancel and Reset per row, the fixed rows, and the message of the panel.
    fn shortcuts_window(&self, t: Tokens) -> Element<'_, Message> {
        let combos = shortcuts::effective_combos(&self.store.prefs.shortcut_overrides);
        let mut rows: Vec<Element<'_, Message>> = Vec::new();

        // the rows of the list: what it does, the keys, and what may be done about them
        for s in shortcuts::SHORTCUTS.iter() {
            let keys = combos.get(s.id).cloned().unwrap_or_default();
            let changed = self.store.prefs.shortcut_overrides.contains_key(s.id);
            let capturing = self.sc_capture.as_deref() == Some(s.id);
            let chip: Element<'_, Message> = if capturing {
                text("Press the new keys (Esc cancels)").size(12).color(t.primary).into()
            } else {
                container(text(shortcuts::format(&keys)).size(12).color(t.fg_soft))
                    .padding([2, 6])
                    .style(move |_| iced::widget::container::Style {
                        background: Some(iced::Background::Color(t.surface_2)),
                        border: iced::Border {
                            radius: 6.0.into(),
                            width: 1.0,
                            color: t.separator,
                        },
                        ..Default::default()
                    })
                    .into()
            };
            let mut actions: Vec<Element<'_, Message>> = vec![
                button(text(if capturing { "Cancel" } else { "Change" }).size(12))
                    .on_press(Message::ScChange(s.id.to_string()))
                    .padding([3, 8])
                    .style(theme::ghost_button(t))
                    .into(),
            ];
            if changed {
                actions.push(
                    button(text("Reset").size(12))
                        .on_press(Message::ScReset(s.id.to_string()))
                        .padding([3, 8])
                        .style(theme::ghost_button(t))
                        .into(),
                );
            }
            rows.push(
                row![
                    text(s.desc).size(13).color(t.fg).width(Length::Fill),
                    chip,
                    row(actions).spacing(4),
                ]
                .spacing(10)
                .align_y(Alignment::Center)
                .padding([4, 0])
                .into(),
            );
        }
        // the fixed rows of FIXED_KEYS, without the row of `?` (the filter of the Studio)
        for (keys, desc) in shortcuts::FIXED_KEYS {
            if keys == "?" {
                continue;
            }
            rows.push(
                row![
                    text(desc).size(13).color(t.fg_soft).width(Length::Fill),
                    container(text(shortcuts::format(&shortcuts::norm_combo(keys))).size(12).color(t.fg_soft))
                        .padding([2, 6])
                        .style(move |_| iced::widget::container::Style {
                            background: Some(iced::Background::Color(t.surface_2)),
                            border: iced::Border {
                                radius: 6.0.into(),
                                width: 1.0,
                                color: t.separator,
                            },
                            ..Default::default()
                        }),
                    Space::new().width(110.0),
                ]
                .spacing(10)
                .align_y(Alignment::Center)
                .padding([4, 0])
                .into(),
            );
        }

        let mut body: Vec<Element<'_, Message>> = vec![
            text("Keyboard shortcuts").size(16).font(bold()).into(),
            text("Choose Change, then press the new keys. A combination that is already in use is refused.")
                .size(13)
                .color(t.fg_soft)
                .into(),
            column(rows).spacing(2).into(),
        ];
        if !self.sc_message.is_empty() {
            body.push(text(self.sc_message.as_str()).size(13).color(t.fg_soft).into());
        }
        // Reset all: there only when there is something to undo (the display of boot.js)
        let mut actions: Vec<Element<'_, Message>> = Vec::new();
        if !self.store.prefs.shortcut_overrides.is_empty() {
            actions.push(
                button(text("Reset all").size(13))
                    .on_press(Message::ScResetAll)
                    .padding([6, 12])
                    .style(theme::ghost_button(t))
                    .into(),
            );
        }
        actions.push(
            button(text("Close").size(13))
                .on_press(Message::ShortcutsClose)
                .padding([6, 12])
                .style(theme::primary_button(t))
                .into(),
        );
        body.push(row(actions).spacing(8).into());

        let window = container(column(body).spacing(12))
            .padding(16)
            .width(520)
            .style(theme::card(t, theme::R_LG, false));
        container(window)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .into()
    }

    /// The dialog that forgets everything (the Studio's askDialog for memClear).
    fn mem_clear_window(&self, t: Tokens) -> Element<'_, Message> {
        let window = container(
            column![
                text("Forget everything?").size(16).font(bold()),
                text("All remembered facts will be deleted.").size(13).color(t.fg_soft),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Cancel").size(13)).on_press(Message::MemClearCancel).padding([6, 12]).style(theme::ghost_button(t)),
                    button(text("Delete all").size(13)).on_press(Message::MemClearOk).padding([6, 12]).style(move |_, status| {
                        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                        button::Style {
                            background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })),
                            text_color: t.danger,
                            border: iced::border::rounded(8),
                            ..Default::default()
                        }
                    }),
                ],
            ]
            .spacing(10),
        )
        .padding(16)
        .width(400)
        .style(theme::card(t, theme::R_LG, false));
        container(window).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into()
    }

    /// The dialog that asks before a project goes away (the Studio's askDialog for `prjDelete`).
    fn prj_delete_window(&self, t: Tokens) -> Element<'_, Message> {
        let Some(id) = self.prj_delete.clone() else { return Space::new().into() };
        let Some(p) = self.store.projects.iter().find(|p| p.id == id) else { return Space::new().into() };
        let n = self.store.chats.iter().filter(|c| c.project.as_deref() == Some(id.as_str())).count();
        let text_line = format!(
            "\"{}\" and its {} document{} will be deleted. Its {} chat{} stay in your list.",
            p.name,
            p.docs.len(),
            if p.docs.len() == 1 { "" } else { "s" },
            n,
            if n == 1 { "" } else { "s" }
        );
        let window = container(
            column![
                text("Delete project?").size(16).font(bold()),
                text(text_line).size(13).color(t.fg_soft),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Cancel").size(13)).on_press(Message::PrjDeleteCancel).padding([6, 12]).style(theme::ghost_button(t)),
                    button(text("Delete").size(13)).on_press(Message::PrjDeleteOk).padding([6, 12]).style(move |_, status| {
                        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                        button::Style {
                            background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })),
                            text_color: t.danger,
                            border: iced::border::rounded(8),
                            ..Default::default()
                        }
                    }),
                ],
            ]
            .spacing(10),
        )
        .padding(16)
        .width(420)
        .style(theme::card(t, theme::R_LG, false));
        container(window).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into()
    }

    /// The dialog before a temporary chat with messages is gone for good (the Studio's
    /// askDialog in toggleTemporary).
    fn temp_close_window(&self, t: Tokens) -> Element<'_, Message> {
        let window = container(
            column![
                text("Close the temporary chat?").size(16).font(bold()),
                text("It was never saved and will be deleted now.").size(13).color(t.fg_soft),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Cancel").size(13)).on_press(Message::TempCloseCancel).padding([6, 12]).style(theme::ghost_button(t)),
                    button(text("Close it").size(13)).on_press(Message::TempCloseOk).padding([6, 12]).style(move |_, status| {
                        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                        button::Style {
                            background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })),
                            text_color: t.danger,
                            border: iced::border::rounded(8),
                            ..Default::default()
                        }
                    }),
                ],
            ]
            .spacing(10),
        )
        .padding(16)
        .width(420)
        .style(theme::card(t, theme::R_LG, false));
        container(window).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into()
    }

    /// The small window to choose the project of a move (the Studio's pickModal).
    fn prj_pick_window(&self, t: Tokens) -> Element<'_, Message> {
        let many = self.prj_pick.as_ref().is_some_and(|ids| ids.len() > 1);
        let title = if many { "Move the chats to a project" } else { "Move the chat to a project" };
        let mut list = column![button(
            text("No project").size(13)
        )
        .on_press(Message::PrjPick(Some(String::new())))
        .padding([8, 10])
        .width(Length::Fill)
        .style(theme::row_button(t, false))]
        .spacing(2);
        for p in &self.store.projects {
            list = list.push(
                button(
                    row![svg(icon(ICON_FOLDER)).width(14).height(14).style(accent_svg(t)), text(ellipsize(&p.name, 28)).size(13)]
                        .spacing(8)
                        .align_y(Alignment::Center),
                )
                .on_press(Message::PrjPick(Some(p.id.clone())))
                .padding([8, 10])
                .width(Length::Fill)
                .style(theme::row_button(t, false)),
            );
        }
        let window = container(
            column![
                text(title).size(16).font(bold()),
                list,
                row![
                    Space::new().width(Length::Fill),
                    button(text("Cancel").size(13)).on_press(Message::PrjPick(None)).padding([6, 12]).style(theme::ghost_button(t)),
                ],
            ]
            .spacing(8),
        )
        .padding(16)
        .width(380)
        .style(theme::card(t, theme::R_LG, false));
        container(window).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into()
    }

    /// The assistant menu under the name in the top bar (the Studio's presetMenu): every assistant with
    /// its instructions, its temperature and a button to edit it, then New, Import and Save current.
    fn asst_menu_panel(&self, t: Tokens) -> Element<'_, Message> {
        let cur = self.active_assistant_id();
        let mut list = column![].spacing(2);
        for a in assistants::all(&self.store.assistants) {
            let active = cur.as_deref() == Some(a.id.as_str());
            let color = assistants::color(&a.color);
            let badge: Element<'_, Message> = svg(icon(assistants::icon_bytes(&a.icon)))
                .width(16)
                .height(16)
                .style(move |_, _| svg::Style { color: Some(color) })
                .into();
            let summary = if a.system.trim().is_empty() { "No instructions".to_string() } else { a.summary() };
            let row_i = row![
                badge,
                button(
                    column![text(ellipsize(&a.name, 26)).size(13), text(ellipsize(&summary, 46)).size(11).color(t.muted)]
                        .spacing(1)
                        .align_x(Alignment::Start),
                )
                .on_press(Message::AsstApply(a.id.clone()))
                .padding([5, 8])
                .width(Length::Fill)
                .style(theme::plain_button(t)),
                container(text(format!("T {}", a.temperature)).size(11).color(t.muted))
                    .padding([2, 6])
                    .style(move |_| container::Style { background: Some(iced::Background::Color(t.fill)), border: iced::border::rounded(6), ..Default::default() }),
                button(svg(icon(assistants::icon_bytes("pen"))).width(13).height(13).style(muted_svg(t)))
                    .on_press(Message::AsstEditOpen(a.id.clone()))
                    .padding(5)
                    .style(theme::ghost_button(t)),
            ]
            .spacing(6)
            .align_y(Alignment::Center);
            list = list.push(
                container(row_i)
                    .padding([2, 4])
                    .style(move |_| container::Style {
                        background: active.then_some(iced::Background::Color(t.primary_soft)),
                        border: iced::border::rounded(8),
                        ..Default::default()
                    }),
            );
        }
        let head = row![
            text("Assistants").size(13).color(t.muted),
            Space::new().width(Length::Fill),
            button(text("New").size(12)).on_press(Message::AsstNew).padding([3, 8]).style(theme::ghost_button(t)),
            button(text("Import").size(12)).on_press(Message::AsstImport).padding([3, 8]).style(theme::ghost_button(t)),
            button(text("Save current").size(12)).on_press(Message::AsstSaveCurrent).padding([3, 8]).style(theme::ghost_button(t)),
        ]
        .spacing(2)
        .align_y(Alignment::Center);
        let rule = container(Space::new().height(1))
            .width(Length::Fill)
            .style(move |_| container::Style { background: Some(iced::Background::Color(t.separator)), ..Default::default() });
        let panel = container(column![head, rule, scrollable(list).height(Length::Shrink)].spacing(6))
            .padding(8)
            .width(360)
            .max_height(440)
            .style(theme::card(t, theme::R_LG, false));
        // under the name, aligned with the centered top bar like the model list
        container(container(panel).max_width(780).width(Length::Fill))
            .center_x(Length::Fill)
            .padding(Padding { top: 46.0, right: 24.0, bottom: 0.0, left: 24.0 })
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    /// The editor window of one assistant (the Studio's assistantModal).
    fn asst_editor(&self, t: Tokens) -> Element<'_, Message> {
        let Some(e) = &self.asst_edit else { return Space::new().into() };
        // icon and colour rows
        let mut icons = row![].spacing(6);
        for name in assistants::ICONS {
            let on = e.icon == name;
            icons = icons.push(
                button(svg(icon(assistants::icon_bytes(name))).width(16).height(16).style(move |_, _| svg::Style { color: Some(if on { t.on_primary } else { t.muted }) }))
                    .on_press(Message::AsstIcon(name.to_string()))
                    .padding(7)
                    .style(move |_, status| {
                        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                        button::Style {
                            background: Some(iced::Background::Color(if on { t.primary } else if hovered { t.fill_hover } else { t.fill })),
                            text_color: t.fg,
                            border: iced::border::rounded(8),
                            ..Default::default()
                        }
                    }),
            );
        }
        let mut colors = row![].spacing(6);
        for name in assistants::COLORS {
            let on = e.color == name;
            let c = assistants::color(name);
            colors = colors.push(
                button(container(Space::new().width(16).height(16)).style(move |_| container::Style {
                    background: Some(iced::Background::Color(c)),
                    border: iced::border::rounded(999),
                    ..Default::default()
                }))
                .on_press(Message::AsstColor(name.to_string()))
                .padding(4)
                .style(move |_, status| {
                    let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                    button::Style {
                        background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })),
                        text_color: t.fg,
                        border: iced::Border { radius: 999.0.into(), width: 2.0, color: if on { t.primary } else if hovered { t.fg_soft } else { iced::Color::TRANSPARENT } },
                        ..Default::default()
                    }
                }),
            );
        }
        let numbers = row![
            column![asst_label(t, "Temperature"), asst_text_input(t, "0.7", &e.temperature, Message::AsstTemp, Length::Fill)].spacing(4),
            column![asst_label(t, "Top-p"), asst_text_input(t, "0.95", &e.top_p, Message::AsstTopP, Length::Fill)].spacing(4),
            column![asst_label(t, "Top-k"), asst_text_input(t, "20", &e.top_k, Message::AsstTopK, Length::Fill)].spacing(4),
        ]
        .spacing(10);
        // Default model: a select of the models of this computer (the Studio's asModel options)
        let model_shown = if e.model.is_empty() { "Any (keep the current model)".to_string() } else { e.model.clone() };
        let mut model_col = column![
            asst_label(t, "Default model"),
            button(
                row![
                    text(ellipsize(&model_shown, 44)).size(13).width(Length::Fill),
                    svg(icon(ICON_CHEVRON)).width(13).height(13).style(muted_svg(t)),
                ]
                .spacing(6)
                .align_y(Alignment::Center),
            )
            .on_press(Message::AsstModelToggle)
            .padding([7, 10])
            .width(Length::Fill)
            .style(theme::row_button(t, false)),
        ]
        .spacing(4);
        if e.model_open {
            let mut opts = column![].spacing(2);
            opts = opts.push(asst_option(t, "Any (keep the current model)".into(), e.model.is_empty(), Message::AsstModel(String::new())));
            let mut names: Vec<String> = self.models.overview.installed.iter().map(|m| m.name.clone()).collect();
            if let Server::Ready(m) = &self.server {
                if !names.contains(&m.id) {
                    names.push(m.id.clone());
                }
            }
            if !e.model.is_empty() && !names.contains(&e.model) {
                names.push(e.model.clone());
            }
            for n in names {
                let selected = e.model == n;
                opts = opts.push(asst_option(t, n.clone(), selected, Message::AsstModel(n)));
            }
            model_col = model_col.push(container(opts).style(theme::card(t, theme::R_LG, false)));
        }
        // Default system prompt for this assistant: the Studio's asSysMode select
        const SYS: [(SysMode, &str); 3] = [
            (SysMode::Inherit, "Use the global setting (Settings > Personalization)"),
            (SysMode::Append, "Add to the default prompt"),
            (SysMode::Override, "Replace the default prompt"),
        ];
        let sys_shown = SYS.iter().find(|(m, _)| *m == e.sys_mode).map(|(_, l)| *l).unwrap_or("");
        let mut sys_col = column![
            asst_label(t, "Default system prompt for this assistant"),
            button(
                row![
                    text(ellipsize(sys_shown, 46)).size(13).width(Length::Fill),
                    svg(icon(ICON_CHEVRON)).width(13).height(13).style(muted_svg(t)),
                ]
                .spacing(6)
                .align_y(Alignment::Center),
            )
            .on_press(Message::AsstSysToggle)
            .padding([7, 10])
            .width(Length::Fill)
            .style(theme::row_button(t, false)),
        ]
        .spacing(4);
        if e.sys_open {
            let mut opts = column![].spacing(2);
            for (m, label) in SYS {
                let selected = e.sys_mode == m;
                opts = opts.push(asst_option(t, label.to_string(), selected, Message::AsstSysMode(m)));
            }
            sys_col = sys_col.push(container(opts).style(theme::card(t, theme::R_LG, false)));
        }
        let mut actions = row![].spacing(8);
        actions = actions.push(
            button(text("Export").size(13))
                .on_press(Message::AsstExport)
                .padding([6, 12])
                .style(theme::ghost_button(t)),
        );
        if e.id.is_some() {
            actions = actions.push(
                button(text("Delete").size(13))
                    .on_press(Message::AsstDelete)
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
            );
        }
        actions = actions.push(Space::new().width(Length::Fill));
        actions = actions.push(button(text("Cancel").size(13)).on_press(Message::AsstClose).padding([6, 12]).style(theme::ghost_button(t)));
        actions = actions.push(button(text("Save").size(13)).on_press(Message::AsstSave).padding([6, 18]).style(theme::primary_button(t)));
        let mut body = column![].spacing(10);
        body = body.push(text(e.title.clone()).size(18).font(bold()));
        body = body.push(column![asst_label(t, "Name"), asst_text_input(t, "Name", &e.name, Message::AsstName, Length::Fill)].spacing(4));
        body = body.push(column![asst_label(t, "Icon and color"), icons, colors].spacing(6));
        body = body.push(
            column![
                asst_label(t, "Instructions (system prompt)"),
                asst_area(t, &e.system, Message::AsstSystem, "You are a precise, helpful assistant...", 96.0)
            ]
            .spacing(4),
        );
        body = body.push(numbers);
        body = body.push(model_col);
        body = body.push(
            column![
                asst_label(t, "Suggestions on an empty chat (one per line, up to 4)"),
                asst_area(t, &e.starters, Message::AsstStarters, "Leave empty for generic suggestions", 70.0)
            ]
            .spacing(4),
        );
        body = body.push(sys_col);
        body = body.push(asst_area(
            t,
            &e.sys_text,
            Message::AsstSysText,
            "Text added to or replacing the default prompt. {{date}}, {{time}} and {{language}} are filled in.",
            58.0,
        ));
        body = body.push(actions.align_y(Alignment::Center));
        let window = container(scrollable(container(body).padding(20).width(Length::Fill)).height(Length::Shrink))
            .width(560)
            .max_height(840)
            .style(theme::card(t, theme::R_LG, false));
        container(window)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .into()
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
        // the assistant beside the model (the Studio's btnPresets): badge, name, chevron; the menu opens under it
        let active = self.active_assistant();
        let (asst_name, asst_icon, asst_color) = match &active {
            Some(a) => (a.name.clone(), a.icon.clone(), a.color.clone()),
            None => ("Standard".to_string(), "pen".to_string(), "patina".to_string()),
        };
        let badge_color = assistants::color(&asst_color);
        let asst_open = self.asst_menu;
        let asst_pill = button(
            row![
                svg(icon(assistants::icon_bytes(&asst_icon))).width(13).height(13).style(move |_, _| svg::Style { color: Some(badge_color) }),
                text(ellipsize(&asst_name, 20)).size(13),
                svg(icon(ICON_CHEVRON)).width(13).height(13).style(muted_svg(t)),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
        )
        .on_press_maybe(self.asst_edit.is_none().then_some(Message::AsstMenuToggle))
        .padding([5, 12])
        .style(move |_, status| {
            let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
            button::Style {
                background: Some(iced::Background::Color(if hovered || asst_open { t.fill_hover } else { t.fill })),
                text_color: t.fg_soft,
                border: iced::border::rounded(999),
                ..Default::default()
            }
        });
        // the project of this chat beside the assistant (the Studio's prjBadge): a click opens its page
        let project_badge: Element<'_, Message> = match self.active_project() {
            Some(p) => button(
                row![
                    svg(icon(ICON_FOLDER)).width(13).height(13).style(accent_svg(t)),
                    text(ellipsize(&p.name, 16)).size(13),
                ]
                .spacing(6)
                .align_y(Alignment::Center),
            )
            .on_press(Message::PrjOpen(p.id.clone()))
            .padding([5, 12])
            .style(move |_, status| {
                let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                button::Style {
                    background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })),
                    text_color: t.fg_soft,
                    border: iced::border::rounded(999),
                    ..Default::default()
                }
            })
            .into(),
            None => Space::new().into(),
        };
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
        // the button of the temporary chat (the Studio's btnTemporary beside the token counter):
        // tinted while a temporary chat is open, or while one is staged for the next new chat
        let temp_active = self.current.and_then(|id| self.store.chat(id)).map(|c| c.temporary).unwrap_or(self.temp_next);
        let temp_button = button(
            svg(icon(ICON_GHOST)).width(16).height(16).style(move |theme, status| {
                if temp_active { accent_svg(t)(theme, status) } else { muted_svg(t)(theme, status) }
            }),
        )
        .on_press(Message::TempToggle)
        .padding(6)
        .style(slash_button(t, temp_active));
        let inner = container(row![pill, Space::new().width(10), asst_pill, Space::new().width(10), project_badge, Space::new().width(10), chip, Space::new().width(Length::Fill), meter, Space::new().width(10), temp_button].align_y(Alignment::Center)).max_width(780).width(Length::Fill);
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
        // The hero of the Studio (`updateHero`): the heading of the moment and the words under
        // it — nothing loaded with what can be done about it, the name of the model with its facts
        let status = self.hero.as_ref();
        let setup = status.is_some_and(|s| s.setup);
        // the name of the model that runs is what the Studio's hero says (its `model.name`)
        let heading = if setup {
            String::from("No model loaded")
        } else {
            match status.map(|s| s.model_name.as_str()).filter(|n| !n.is_empty()) {
                Some(name) => name.to_string(),
                None => match &self.server {
                    Server::Ready(m) => m.id.clone(),
                    _ => String::from("fLLM"),
                },
            }
        };
        let (sub, plan) = if setup {
            match self.hero_overview.as_ref() {
                None => (String::from("Looking for models on this computer …"), None),
                Some(overview) => {
                    let last = status.map(|s| s.last_model.as_str()).unwrap_or("");
                    let plan = crate::hero::start_plan(overview, last);
                    let words = if plan.mode == "load" {
                        String::from("A model on this computer is ready. Nothing to download.")
                    } else {
                        String::from("Download one, or choose a model file you already have.")
                    };
                    (words, Some(plan))
                }
            }
        } else {
            let facts = status
                .map(|s| crate::hero::hero_facts(s.parameters, &s.parameter_size, &s.quantization, s.max_context))
                .unwrap_or_default();
            let words = match &self.server {
                Server::Ready(_) if !facts.is_empty() => format!("Running locally on this computer: {facts}."),
                Server::Ready(_) => String::from("Running locally on this computer."),
                _ if !facts.is_empty() => format!("Running locally on this computer: {facts}."),
                _ => String::from("Connecting to the server…"),
            };
            (words, None)
        };
        let list = self.starter_list();
        let mut starters = column![].spacing(10);
        for pair in list.chunks(2).enumerate() {
            let (row_i, chunk) = pair;
            let mut r = row![].spacing(10);
            for (j, s) in chunk.iter().enumerate() {
                let i = row_i * 2 + j;
                r = r.push(
                    button(text(ellipsize(s, 110)).size(13).color(t.fg_soft))
                        .on_press_maybe(matches!(self.server, Server::Ready(_)).then_some(Message::Starter(i)))
                        .padding([12, 14])
                        .width(Length::FillPortion(1))
                        .style(starter_style(t)),
                );
            }
            starters = starters.push(r);
        }
        // The two-minute line of tokens per second: sixty bars, and the value of the last moment
        // beside its label (the `heroTrace` and `heroTps` of the Studio)
        let mut trace = row![].spacing(2).align_y(Alignment::End);
        let busy = status.is_some_and(|s| s.slots_running > 0);
        if !self.hero_samples.is_empty() {
            let top = 10.0f64.max(self.hero_samples.iter().cloned().fold(0.0f64, f64::max));
            for sample in self.hero_samples.iter() {
                let h = ((sample / top).clamp(0.0, 1.0) * 58.0 + 2.0) as f32;
                trace = trace.push(
                    container(Space::new().width(6.0).height(h))
                        .width(6.0)
                        .height(60.0)
                        .align_y(Alignment::End)
                        .style(move |_|iced::widget::container::Style {
                            background: Some(iced::Background::Color(if busy { t.primary } else { t.separator })),
                            border: iced::Border { radius: theme::R_SM.into(), ..Default::default() },
                            ..Default::default()
                        }),
                );
            }
            let (value, label) = crate::hero::trace_text(*self.hero_samples.last().unwrap(), busy);
            trace = trace.push(
                column![
                    text(value).size(16).font(bold()),
                    text(label).size(11).color(t.muted),
                ]
                .spacing(2)
                .align_x(Alignment::Start)
                .padding([0, 10]),
            );
        }
        // What can be done about it: the buttons of `renderHeroStart` (load the model that is
        // here first, or take the recommended one when nothing is)
        let mut actions = column![].spacing(8).align_x(Alignment::Center);
        if let Some(plan) = plan {
            if plan.mode == "load" {
                if let Some(first) = plan.primary.as_ref() {
                    let meta = crate::hero::meta_line(first.size, &first.where_, first.last, first.fits);
                    actions = actions.push(
                        button(text(format!("Load {}", first.title)).size(14).font(bold()))
                            .on_press(Message::HeroLoad(first.reference.clone()))
                            .padding([10, 18])
                            .style(primary_style(t)),
                    ).push(text(meta).size(12).color(t.muted));
                }
                for other in plan.others.iter().take(4) {
                    actions = actions.push(
                        button(text(format!("Load {}", other.title)).size(13).color(t.fg_soft))
                            .on_press(Message::HeroLoad(other.reference.clone()))
                            .padding([8, 14])
                            .style(starter_style(t)),
                    );
                }
                if plan.others.len() > 4 {
                    actions = actions.push(
                        text(format!("and {} more on the Models page", plan.others.len() - 4))
                            .size(12)
                            .color(t.muted),
                    );
                }
            } else if let Some(rec) = plan.download.as_ref() {
                // name, quantization and weight in one line (the variant decides, with the
                // warning only when it does not fit into memory — renderHeroStart of the Studio)
                let name = if rec.display_name.is_empty() { rec.id.as_str() } else { rec.display_name.as_str() };
                let variant = rec
                    .variants
                    .iter()
                    .find(|v| v.quant == rec.default_quant)
                    .or_else(|| rec.variants.first());
                let bytes = variant.map_or(0, |v| v.bytes);
                let fits = variant.map_or(rec.fits, |v| v.fits);
                let meta = crate::hero::download_meta(name, &rec.default_quant, bytes, fits);
                actions = actions.push(
                    button(text("Download and start").size(14).font(bold()))
                        .on_press(Message::HeroDownload { model: rec.id.clone(), quant: rec.default_quant.clone() })
                        .padding([10, 18])
                        .style(primary_style(t)),
                ).push(
                    text(meta).size(12).color(t.muted),
                );
            }
            let mut choices = row![].spacing(8);
            choices = choices.push(
                button(text("Choose a model file…").size(13).color(t.fg_soft))
                    .on_press(Message::HeroChoose)
                    .padding([8, 14])
                    .style(starter_style(t)),
            ).push(
                button(text("Browse models").size(13).color(t.fg_soft))
                    .on_press(Message::HeroBrowse)
                    .padding([8, 14])
                    .style(starter_style(t)),
            );
            actions = actions.push(choices);
        }
        let mut hero = column![
            text(heading).size(30).font(bold()),
            text(sub).size(14).color(t.muted),
            Space::new().height(10),
            trace,
            Space::new().height(6),
            actions,
            starters,
        ]
        .spacing(6)
        .align_x(Alignment::Center)
        .max_width(680);
        // a chat of your own assistant can change its suggestions (the Studio's "Edit suggestions")
        if let Some(a) = self.active_assistant().filter(|a| !a.builtin) {
            hero = hero.push(
                button(text("Edit suggestions").size(13).color(t.fg_soft))
                    .on_press(Message::AsstEditOpen(a.id))
                    .padding([8, 14])
                    .style(starter_style(t)),
            );
        }
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
    /// Only the newest question carries its passages (the Studio sends them with the last user message only).
    pub fn wire_messages(&self) -> Vec<WireMessage> {
        let mut wire = vec![WireMessage::new("system", self.system_prompt())];
        let upto = match self.messages.last() {
            Some(last) if last.role == Role::Assistant && last.content.is_empty() => self.messages.len() - 1,
            _ => self.messages.len(),
        };
        let last_user = self.messages[..upto].iter().rposition(|m| m.role == Role::User);
        wire.extend(self.messages[..upto].iter().enumerate().map(|(i, m)| {
            // the passages only with the newest question, the whole files before the question, and
            // an empty question that has passages or an image says so (the Studio's userContent)
            let context = if Some(i) == last_user { m.passages.as_ref().map(|p| rag::prefix(p)).unwrap_or_default() } else { String::new() };
            let files_block: String = m.files.iter().map(|f| format!("File {}:\n```\n{}\n```\n\n", f.name, f.text)).collect();
            let tail = if m.content.is_empty() {
                if !context.is_empty() { "Summarize the key points." } else { "" }
            } else {
                m.content.as_str()
            };
            let text = format!("{context}{files_block}{tail}");
            let mut out = if !m.images.is_empty() {
                let mut parts = vec![serde_json::json!({ "type": "text", "text": if text.is_empty() { "Describe the image." } else { text.as_str() } })];
                for url in &m.images {
                    parts.push(serde_json::json!({ "type": "image_url", "image_url": { "url": url } }));
                }
                WireMessage::with_parts(m.role.wire(), text, parts)
            } else {
                WireMessage::new(m.role.wire(), text)
            };
            if matches!(m.role, Role::Tool) {
                out.role = "tool".into();
                out.tool_call_id = m.tool_call_id.clone();
            } else if matches!(m.role, Role::Assistant) {
                out.tool_calls = m.tool_calls.clone();
            }
            out
        }));
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

    /// The answer of the *Attach* dialog or a dropped path: images are shrunk here, documents go
    /// through the server, text is read and placed with the next question (the Studio's addFiles).
    fn add_picked(&mut self, path: String, text: Option<String>) -> Task<Message> {
        let name = path.rsplit(['/', '\\']).next().unwrap_or(path.as_str()).to_string();
        let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
        let base = self.base.clone();
        if IMAGE_EXTS.contains(&ext.as_str()) {
            return Task::perform(
                async move {
                    let bytes = std::fs::read(&path).map_err(|e| format!("The file could not be read: {e}"))?;
                    let data_url = shrink_image(&bytes).map_err(|e| format!("{name}: {e}"))?;
                    Ok::<(String, String), String>((name, data_url))
                },
                Message::ImageReady,
            );
        }
        if server_ext(&ext) {
            return Task::perform(
                async move {
                    let bytes = std::fs::read(&path).map_err(|e| format!("The file could not be read: {e}"))?;
                    if bytes.len() as u64 > RAG_MAX_UPLOAD {
                        return Err(format!("{name} is too large (more than 50 MB)."));
                    }
                    let size = bytes.len() as u64;
                    let (text, pages, truncated) = api::extract_file(base, name.clone(), bytes).await?;
                    Ok((name, text, pages, truncated, size))
                },
                Message::AttachExtracted,
            );
        }
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if size > rag::MAX_TEXT_ATTACHMENT {
            self.toast = Some(format!("{name} is too large (more than {}).", format_bytes(rag::MAX_TEXT_ATTACHMENT)));
            return Task::none();
        }
        let text = match text {
            Some(t) => t,
            None => match std::fs::read(&path) {
                Err(e) => {
                    self.toast = Some(format!("The file could not be read: {e}"));
                    return Task::none();
                }
                Ok(bytes) => match String::from_utf8(bytes) {
                    Err(_) => {
                        self.toast = Some(format!("{name} is not a text file."));
                        return Task::none();
                    }
                    Ok(t) => t,
                },
            },
        };
        if text.contains('\0') {
            self.toast = Some(format!("{name} is not a text file."));
            return Task::none();
        }
        self.rag_place(name, text, size, None, false);
        Task::none()
    }

    /// Text with the next question: whole when it fits the context, else as a document of this chat
    /// whose passages are searched (the Studio's ragPlace).
    fn rag_place(&mut self, name: String, text: String, size: u64, pages: Option<u64>, truncated: bool) {
        let whole: u64 = self
            .attachments
            .iter()
            .filter_map(|a| match a {
                Attachment::Text { text, .. } => Some(text.chars().count() as u64 / 4 + 4),
                _ => None,
            })
            .sum();
        let room = (self.model_info().max_context as f64 * rag::FIT_SHARE) as u64;
        if text.chars().count() as u64 / 4 + 4 + whole <= room {
            self.attachments.push(Attachment::Text { name, text, size });
        } else {
            self.toast = Some(format!("{name} is too long for the context: the passages that match your question are used."));
            self.attachments.push(Attachment::Doc { name, text, size, pages, truncated });
        }
    }

    /// Stores a too-long attachment in the documents of this chat, within the limits (the Studio's
    /// ragStoreDocs): the same numbers as the documents of a project, a separate storage.
    fn store_chat_doc(&mut self, id: u64, name: String, text: String, pages: Option<u64>, truncated: bool) {
        let room = {
            let Some(chat) = self.store.chat_mut(id) else { return };
            let used: usize = chat.docs.iter().map(|d| d.chars).sum();
            rag::MAX_DOC_CHARS.min(rag::MAX_CHAT_CHARS.saturating_sub(used))
        };
        if room < 2000 {
            self.toast = Some(format!(
                "{name} was not added: the documents of this chat already use the storage limit ({} characters). Remove one first.",
                group(rag::MAX_CHAT_CHARS as u64)
            ));
            return;
        }
        let mut note = None;
        let mut text = text;
        if text.chars().count() > room {
            text = text.chars().take(room).collect();
            note = Some(format!("{name} is long: only its first {} characters are kept in this chat.", group(room as u64)));
        } else if truncated {
            note = Some(format!("{name} is very long: only the first part was read."));
        }
        let chars = text.chars().count();
        if let Some(chat) = self.store.chat_mut(id) {
            chat.docs.push(store::ProjectDoc { id: store::uid("doc"), name, text, pages, chars, added: store::now_secs() });
        }
        self.rag_cache = None;
        if note.is_some() {
            self.toast = note;
        }
    }

    /// The personalization as the Settings page draws it (the line under the list is the Studio's
    /// memUsage).
    fn personal_view(&self) -> settings::PersonalView<'_> {
        let memories = &self.store.prefs.personal.memories;
        let (_lines, dropped) = memory::select(memories, memory::limits::PROMPT);
        let usage = if memories.is_empty() {
            "Nothing remembered yet.".to_string()
        } else {
            format!(
                "{} of {} entries.{}",
                memories.len(),
                memory::limits::MEMORIES,
                if dropped > 0 {
                    format!(
                        " The {} oldest do not fit into the {} characters sent to the model and are skipped.",
                        dropped,
                        group(memory::limits::PROMPT as u64)
                    )
                } else {
                    " All of them are sent to the model.".to_string()
                }
            )
        };
        settings::PersonalView {
            about: &self.personal_about,
            use_about: self.store.prefs.personal.use_about,
            use_memory: self.store.prefs.personal.use_memory,
            memories,
            usage,
            new_text: &self.mem_new,
            edit: self.mem_edit.as_ref().map(|(id, text)| (id.as_str(), text.as_str())),
            prompts: &self.store.prefs.saved_prompts,
            plib_texts: &self.plib_texts,
        }
    }

    /// What the menu under a "/" in the box offers: the saved prompts that match what is typed,
    /// the built-in commands left alone — and only while the text is a single line
    /// (the Studio's `plibUpdateSlash`).
    fn slash_recompute(&mut self) {
        let text = self.editor.text();
        self.slash = if text.starts_with('/') && !text.contains('\n') {
            promptlib::match_prompts(&self.store.prefs.saved_prompts, &text)
        } else {
            Vec::new()
        };
        // what the Studio does with its index when the list shrinks
        self.slash_index = self.slash_index.min(self.slash.len().saturating_sub(1));
    }

    /// The picked prompt becomes the text of the box, the menu closes (plibPick).
    fn slash_pick(&mut self, i: usize) {
        if let Some(&idx) = self.slash.get(i) {
            if let Some(p) = self.store.prefs.saved_prompts.get(idx) {
                let text = p.text.clone();
                self.editor = Content::with_text(&text);
                // the cursor stands behind the text: what you type next continues it (the browser
                // puts the cursor at the end when the value was set and the box focused)
                self.editor.perform(Action::Move(Motion::DocumentEnd));
            }
        }
        self.slash.clear();
        self.slash_index = 0;
    }

    /// What the rewinds of the open chat hide becomes a version of it: a new question or a new
    /// answer starts from the row as it is, and the hidden part is kept, not thrown away.
    fn commit_rewinds(&mut self) {
        let Some(id) = self.current else { return };
        let notice_gone = self.rewind_notice.take().is_some();
        let _ = notice_gone;
        if let Some(chat) = self.store.chat_mut(id) {
            let rewinds = std::mem::take(&mut chat.rewinds);
            for r in rewinds {
                if r.tail.is_empty() {
                    continue;
                }
                let slot = chat.next_slot;
                chat.next_slot = chat.next_slot.saturating_add(1);
                chat.variants.push(store::Variant { slot, messages: r.tail });
            }
        }
    }

    /// A temporary chat is gone for good: the chat leaves the list, the view is fresh (what the
    /// Studio's startNewChat does after the dialog of toggleTemporary).
    fn close_temporary(&mut self) {
        if self.live.is_some() {
            return;
        }
        if let Some(id) = self.current.take() {
            self.store.chats.retain(|c| c.id != id);
        }
        self.prj_pending = None;
        self.messages.clear();
        self.window = CHAT_WINDOW;
        self.editing = None;
        self.used_tokens = 0;
        self.error = None;
        self.save();
    }

    /// Puts a fact into the memory, with the line the Studio shows (memRemember).
    fn remember(&mut self, text: String) {
        let now = store::now_millis();
        let (list, status, _) = memory::add(&self.store.prefs.personal.memories, &text, now, || store::uid("mem"), None);
        match status {
            "empty" => {}
            "duplicate" => self.toast = Some("Already remembered.".into()),
            "secret" => self.toast = Some("That looks like a password, key or card number. It is not saved to memory.".into()),
            "full" => {
                self.toast = Some(format!(
                    "The memory is full ({} entries). Delete some in Settings > Personalization.",
                    memory::limits::MEMORIES
                ))
            }
            _ => {
                self.store.prefs.personal.memories = list;
                self.save();
                self.toast = Some("Remembered. See Settings > Personalization.".into());
            }
        }
    }

    /// Removes one entry of the memory (the cross of a row).
    fn mem_delete(&mut self, id: &str) {
        self.store.prefs.personal.memories.retain(|m| m.id != id);
        self.save();
    }

    /// Whether a "remember that …" at the start of a message saves its thing (prjCaptureMemory).
    fn memory_capture_on(&self) -> bool {
        if !self.store.prefs.personal.use_memory {
            return false;
        }
        self.current
            .and_then(|id| self.store.chat(id))
            .map(|c| !(c.no_personal || c.no_memory))
            .unwrap_or(true)
    }

    /// What goes to the server: `{version: 1, memories}` (the Studio's memDoc).
    fn personal_body(&self) -> serde_json::Value {
        serde_json::json!({ "version": 1, "memories": self.store.prefs.personal.memories })
    }

    /// Whether the model gets the `memory` tool (the Studio's ctOfferedFlags): switched on in the
    /// settings, not for a chat that says no, not for a context under 2048 tokens, and not for a
    /// model that read the prompt slower than 120 tokens a second on this computer.
    pub fn memory_tool_offered(&self) -> bool {
        if !self.store.prefs.personal.use_memory || self.model_info().max_context < 2048 {
            return false;
        }
        let chat = self.current.and_then(|id| self.store.chat(id));
        if chat.is_some_and(|c| c.no_personal || c.no_memory) {
            return false;
        }
        let speed = self.last_prompt_speed();
        !(speed > 0.0 && speed < 120.0)
    }

    /// The project of the open chat, else the one a new chat joins (the Studio's prjChrome).
    fn active_project(&self) -> Option<&store::Project> {
        let id = match self.current {
            Some(cid) => self.store.chat(cid).and_then(|c| c.project.clone()),
            None => self.prj_pending.clone(),
        }?;
        self.store.projects.iter().find(|p| p.id == id)
    }

    /// Shows the page of a project (its instructions go into the box).
    fn open_project(&mut self, id: &str) {
        let text = match self.store.projects.iter().find(|p| p.id == id) {
            Some(p) => p.instructions.clone(),
            None => return,
        };
        self.prj_view = Some(id.to_string());
        self.page = View::Project;
        self.prj_instr = Content::with_text(&text);
    }

    /// The documents searched for a question: the knowledge of the project of this chat first, then
    /// the documents of the chat itself (the Studio's prjCombinedDocs).
    fn knowledge(&self) -> Vec<(String, String, String)> {
        let mut docs: Vec<(String, String, String)> = match self.active_project() {
            Some(p) => p.docs.iter().map(|d| (d.id.clone(), d.name.clone(), d.text.clone())).collect(),
            None => Vec::new(),
        };
        if let Some(cid) = self.current {
            if let Some(chat) = self.store.chat(cid) {
                docs.extend(chat.docs.iter().map(|d| (d.id.clone(), d.name.clone(), d.text.clone())));
            }
        }
        docs
    }

    /// The passages for a question, best first; `None` when there is nothing to search. The index is
    /// kept until the documents change (the Studio's ragThreadIndex cache).
    fn passages_for(&mut self, question: &str) -> Option<Vec<rag::Passage>> {
        let docs = self.knowledge();
        if docs.is_empty() {
            self.rag_cache = None;
            return None;
        }
        let sig = docs.iter().map(|(id, _, text)| format!("{id}:{}", text.len())).collect::<Vec<_>>().join("|");
        let stale = self.rag_cache.as_ref().is_none_or(|c| c.sig != sig);
        if stale {
            let refs: Vec<IndexedDoc<'_>> = docs.iter().map(|(id, name, text)| IndexedDoc { id, name, text }).collect();
            self.rag_cache = Some(RagCache { sig, index: rag::build_index(&refs) });
        }
        let k = rag::passage_count(self.model_info().max_context);
        let index = &self.rag_cache.as_ref().expect("the index was just built").index;
        let found = rag::search(index, question, k, None);
        (!found.is_empty()).then(|| rag::arrange(found))
    }

    /// Puts an extracted text into the knowledge of the open project, within the limits (prjAddFiles).
    fn store_doc(&mut self, name: &str, text: String, pages: Option<u64>) {
        let Some(pid) = self.prj_view.clone() else { return };
        let Some(pos) = self.store.projects.iter().position(|p| p.id == pid) else { return };
        let used_project = self.store.projects[pos].used();
        let used_total: usize = self.store.projects.iter().map(|p| p.used()).sum();
        let project_name = self.store.projects[pos].name.clone();
        match rag::fit_doc(used_project, used_total, name, &text) {
            rag::Fit::Message(m) => self.toast = Some(m),
            fit => {
                let (text, note) = fit.stored().expect("only a refused fit has no text");
                let chars = text.chars().count();
                self.store.projects[pos].docs.push(store::ProjectDoc {
                    id: store::uid("pdoc"),
                    name: name.to_string(),
                    text,
                    pages,
                    chars,
                    added: store::now_secs(),
                });
                self.store.projects[pos].updated = store::now_secs();
                self.save();
                self.toast = Some(note.unwrap_or_else(|| format!("{name} added to \"{project_name}\"")));
            }
        }
    }

    /// The assistant of the open chat; without an open chat the one chosen for new chats. An id that no
    /// longer exists (a deleted assistant) counts as none, like the Studio's `asstEffectiveId`.
    pub fn active_assistant_id(&self) -> Option<String> {
        let id = match self.current {
            Some(id) => self.store.chat(id).and_then(|c| c.assistant.clone()),
            None => (!self.store.prefs.assistant.is_empty()).then(|| self.store.prefs.assistant.clone()),
        }?;
        assistants::all(&self.store.assistants).into_iter().any(|a| a.id == id).then_some(id)
    }

    /// The chosen assistant itself (the built-ins included).
    pub fn active_assistant(&self) -> Option<Assistant> {
        let id = self.active_assistant_id()?;
        assistants::all(&self.store.assistants).into_iter().find(|a| a.id == id)
    }

    /// The suggestions on the welcome screen: the four of the assistant, else the built-in ones.
    pub fn starter_list(&self) -> Vec<String> {
        match self.active_assistant() {
            Some(a) => assistants::starters(&a),
            None => STARTERS.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// Choose an assistant: it becomes the choice for new chats and for this chat when one is open, and
    /// its sampling values are sent from now on (the Studio's `asstApply` sets the sampler too).
    fn apply_assistant(&mut self, id: &str, silent: bool) -> Task<Message> {
        let Some(a) = assistants::all(&self.store.assistants).into_iter().find(|x| x.id == id) else {
            return Task::none();
        };
        self.store.prefs.assistant = id.to_string();
        self.store.prefs.temperature = a.temperature;
        self.store.prefs.top_p = a.top_p;
        self.store.prefs.sampling.top_k = a.top_k;
        self.store.prefs.manual_sampling = true;
        if let Some(cid) = self.current {
            if let Some(chat) = self.store.chat_mut(cid) {
                chat.assistant = Some(id.to_string());
            }
        }
        self.asst_menu = false;
        self.save();
        if !silent {
            self.toast = Some(format!("Assistant \"{}\" selected", a.name));
        }
        if a.model.is_empty() {
            return Task::none();
        }
        if !self.models.fetched {
            // the list of models is not here yet: its model is applied when it arrives
            self.asst_model_pending = Some(a.model.clone());
            return Task::perform(models::fetch_overview(self.base.clone()), Message::OverviewLoaded);
        }
        self.switch_model_of(&a.model)
    }

    /// The model of an assistant: loaded when it is on this computer, otherwise said in the toast (the
    /// Studio keeps the current model in that case too).
    fn switch_model_of(&mut self, model: &str) -> Task<Message> {
        let found = self
            .models
            .overview
            .installed
            .iter()
            .find(|m| m.name == model || m.reference == model)
            .map(|m| (m.reference.clone(), m.name.clone()));
        let Some((reference, name)) = found else {
            self.toast = Some(format!("The model \"{}\" is not available here; the current model stays.", model));
            return Task::none();
        };
        let loaded = match &self.server {
            Server::Ready(m) => Some(m.id.clone()),
            _ => None,
        };
        let is_loaded = loaded.as_ref().is_some_and(|l| name.starts_with(l.as_str()) || l.starts_with(name.as_str()));
        if is_loaded {
            return Task::none();
        }
        self.update(Message::LoadModel(reference))
    }

    /// The system prompt the next request carries: built from the facts about the model and the computer,
    /// then the choice about the default prompt (the assistant's `sys_mode` when it is not "inherit"),
    /// then the instructions of the assistant.
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
            tools: {
                let flags = self.tool_flags();
                prompts::Tools { web: flags.web, memory: flags.memory, chats: false }
            },
        };
        let active = self.active_assistant();
        let mut choices: Vec<(prompts::Mode, &str)> = match active.as_ref() {
            Some(a) => match a.sys_mode {
                SysMode::Override => vec![(prompts::Mode::Override, a.sys_text.as_str())],
                SysMode::Append => vec![(prompts::Mode::Append, a.sys_text.as_str())],
                SysMode::Inherit => vec![(p.system_mode(), p.system_prompt.as_str())],
            },
            None => vec![(p.system_mode(), p.system_prompt.as_str())],
        };
        // what of yours the model is told: about me, then the memory, before the project (the
        // Studio's composeSystem: defaults, personalization, project, assistant)
        let chat = self.current.and_then(|id| self.store.chat(id));
        let private = chat.is_some_and(|c| c.no_personal);
        let memory_off = chat.is_some_and(|c| c.no_memory);
        let mut personal_parts: Vec<String> = Vec::new();
        if !private {
            let personal = &p.personal;
            if personal.use_about && !personal.about.trim().is_empty() {
                let about: String = personal.about.trim().chars().take(memory::limits::ABOUT).collect();
                personal_parts.push(format!("About the user and how they want answers (in their own words):\n{about}"));
            }
            if personal.use_memory && !memory_off {
                let cap = if facts.size == prompts::Size::Compact { memory::limits::PROMPT_COMPACT } else { memory::limits::PROMPT };
                let (lines, _dropped) = memory::select(&personal.memories, cap);
                if !lines.is_empty() {
                    let mut block = String::from("Saved memory about the user (use it when it is relevant, do not bring it up otherwise):");
                    for line in &lines {
                        block.push_str(&format!("\n- {line}"));
                    }
                    personal_parts.push(block);
                }
            }
        }
        for part in &personal_parts {
            choices.push((prompts::Mode::Append, part));
        }
        // the knowledge of the project sits after the default prompt and before the assistant's own text
        // (the Studio's composeSystem: defaults, personalization, project, assistant) with its own header
        let project_part = self.active_project().filter(|p| !p.instructions.trim().is_empty()).map(|p| {
            let text: String = p.instructions.trim().chars().take(rag::limits::INSTRUCTIONS).collect();
            format!("Instructions for the project \"{}\":\n{}", p.name, text)
        });
        if let Some(project_part) = project_part.as_deref() {
            choices.push((prompts::Mode::Append, project_part));
        }
        if let Some(a) = active.as_ref().filter(|a| !a.system.trim().is_empty()) {
            choices.push((prompts::Mode::Append, a.system.as_str()));
        }
        // the response style of this chat rides at the very end (artifacts.js applies withStylePrompt
        // to the base that composeSystem appends last); without a chat yet the style staged for the
        // next one counts
        let staged = chat.as_ref().map(|c| c.style.as_str()).unwrap_or(self.new_style.as_str());
        if let Some(instruction) = prompts::style::instruction(prompts::style::style_of(staged)) {
            choices.push((prompts::Mode::Append, instruction));
        }
        prompts::resolve(&prompts::build(&facts), &choices, &now, &locale)
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
    /// The rows of the command palette as they are shown now: the fixed commands (with the keys in
    /// force), the chats of the store, ranked by the query — the `build` of `palette.js`.
    fn palette_build(&self) -> Vec<(palette::PaletteItem, PaletteAction)> {
        use palette::{Kind, PaletteItem};
        let combos = shortcuts::effective_combos(&self.store.prefs.shortcut_overrides);
        let hint = |id: &str| {
            combos
                .get(id)
                .map(|c| shortcuts::format(c))
                .unwrap_or_default()
        };
        let chat_open = self.current.is_some() && self.page == View::Chat;

        let mut items: Vec<(PaletteItem, PaletteAction)> = vec![
            (
                PaletteItem::new("new", Kind::Command, "New chat")
                    .keywords("start conversation")
                    .hint(hint("newChat")),
                PaletteAction::NewChat,
            ),
        ];
        if chat_open {
            // the three the Studio puts behind the first one when a chat is open
            items.push((
                PaletteItem::new("regen", Kind::Command, "Regenerate the last answer")
                    .keywords("again retry")
                    .hint(hint("regenerate")),
                PaletteAction::Regenerate,
            ));
            if let Some(id) = self.current {
                items.push((
                    PaletteItem::new("export-md", Kind::Command, "Export this chat as Markdown")
                        .keywords("save download"),
                    PaletteAction::ExportMarkdown(id),
                ));
                items.push((
                    PaletteItem::new("export-json", Kind::Command, "Export this chat as JSON")
                        .keywords("save download backup"),
                    PaletteAction::ExportJson(id),
                ));
            }
        }
        items.push((
            PaletteItem::new("chat", Kind::Command, "Go to Chat").hint(hint("surfaceChat")),
            PaletteAction::GoChat,
        ));
        items.push((
            PaletteItem::new("code", Kind::Command, "Go to Code")
                .keywords("coding agent folder")
                .hint(hint("surfaceCode")),
            PaletteAction::GoCode,
        ));
        items.push((
            PaletteItem::new("models", Kind::Command, "Open Models")
                .keywords("download hugging face gguf catalog load")
                .hint(hint("openModels")),
            PaletteAction::GoModels,
        ));
        items.push((
            PaletteItem::new("settings", Kind::Command, "Open Settings").hint(hint("openSettings")),
            PaletteAction::GoSettings,
        ));
        // the title says which way it would go (the ternary of palette.js)
        let theme_title = if self.dark() {
            "Switch to the light theme"
        } else {
            "Switch to the dark theme"
        };
        items.push((
            PaletteItem::new("theme", Kind::Command, theme_title)
                .keywords("appearance colors mode"),
            PaletteAction::ToggleTheme,
        ));
        items.push((
            PaletteItem::new("shortcuts", Kind::Command, "Keyboard shortcuts")
                .keywords("keys hotkeys")
                .hint(hint("shortcuts")),
            PaletteAction::Shortcuts,
        ));
        items.push((
            PaletteItem::new("tour", Kind::Command, "Show the welcome tour"),
            PaletteAction::Tour,
        ));

        // the chats of the store, newest first, with the text around a hit under the title (the
        // sidebar searches titles and messages the same way)
        let q = self.palette_query.trim().to_lowercase();
        let mut chats: Vec<&store::ChatRecord> = self.store.chats.iter().collect();
        chats.sort_by_key(|c| std::cmp::Reverse(c.updated));
        let mut chat_rows: Vec<(PaletteItem, PaletteAction)> = Vec::new();
        for c in chats {
            let in_title = q.is_empty() || c.title.to_lowercase().contains(&q);
            let text: String = c
                .messages
                .iter()
                .filter(|m| !matches!(m.role, Role::Tool))
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let hit = if !in_title && !q.is_empty() {
                palette::snippet(&text, &q)
            } else {
                String::new()
            };
            if !in_title && hit.is_empty() {
                continue;
            }
            chat_rows.push((
                PaletteItem::new(format!("chat-{}", c.id), Kind::Chat, c.title.clone()).with_sub(hit),
                PaletteAction::OpenChat(c.id),
            ));
        }

        // rank: the chats are scored on their titles only (the `sub` is emptied for the ranking
        // of palette.js), a row found only in its text is appended after the ranking
        let all: Vec<(PaletteItem, PaletteAction)> = items
            .into_iter()
            .chain(chat_rows.iter().map(|(item, action)| {
                let mut item = item.clone();
                item.sub.clear();
                (item, action.clone())
            }))
            .collect();
        let just_items: Vec<palette::PaletteItem> = all.iter().map(|(i, _)| i.clone()).collect();
        let ranked = palette::rank_palette(&self.palette_query, &just_items, 40);
        let mut rows: Vec<(PaletteItem, PaletteAction)> =
            ranked.into_iter().map(|i| all[i].clone()).collect();

        if !q.is_empty() {
            let seen: Vec<String> = rows.iter().map(|(i, _)| i.id.clone()).collect();
            for (item, action) in &chat_rows {
                if !item.sub.is_empty() && !seen.contains(&item.id) && rows.len() < 40 {
                    rows.push((item.clone(), action.clone()));
                }
            }
        }
        rows
    }

    /// The saved state of the app (the tests read what a round wrote into it).
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// The saved state of the app, for the tests that put chats into it.
    pub fn store_mut(&mut self) -> &mut Store {
        &mut self.store
    }

    /// The keys a shortcut was changed to, if it was changed (the *Reset* button of the Studio).
    pub fn shortcut_override(&self, id: &str) -> Option<&String> {
        self.store.prefs.shortcut_overrides.get(id)
    }

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

    /// The background of a word of the answer: `hsl(hue, 70%, 42%, 0.22)` of the Studio's
    /// `.lp-tok` — the warmer, the more certain the model was.
    fn logprob_color(hue: f32) -> iced::Color {
        let h = hue.rem_euclid(360.0) / 60.0;
        let (s, l, a) = (0.70f32, 0.42f32, 0.22f32);
        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let x = c * (1.0 - (h % 2.0 - 1.0).abs());
        let m = l - c / 2.0;
        let (r, g, b) = match h as i32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        iced::Color::from_rgba(r + m, g + m, b + m, a)
    }

    /// The chip of the research mode (tests read it).
    pub fn research_state(&self) -> bool {
        self.research
    }

    /// The answers of the open chat as `(id, text, streaming)` for the artifact families: the id is
    /// the place in the chat, so a version knows where it came from.
    fn art_messages(&self) -> Vec<(String, String, bool)> {
        self.messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == Role::Assistant)
            .map(|(i, m)| (i.to_string(), m.content.clone(), self.live.is_some() && i + 1 == self.messages.len()))
            .collect()
    }

    /// The runs that stand under a block of this code (the card reads its own).
    pub fn run_of(&self, code: &str) -> Option<&artifacts::CodeRun> {
        self.runs.get(code)
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
        let body: Element<'_, Message> = scrollable(container(col.padding([20, 0])).width(Length::Fill).center_x(Length::Fill))
            .anchor_bottom()
            .height(Length::Fill)
            .into();
        // the banner of the Studio over the messages of a temporary chat (also while one is only
        // staged for the next new chat)
        let temp = self.current.and_then(|id| self.store.chat(id)).map(|c| c.temporary).unwrap_or(self.temp_next);
        if !temp {
            return body;
        }
        column![
            container(
                text("Temporary chat: it is not saved and disappears when you close it or start another chat.")
                    .size(12.5)
                    .color(t.fg)
            )
            .width(Length::Fill)
            .padding([6, 14])
            .style(move |_| container::Style {
                background: Some(iced::Background::Color(iced::Color { a: 0.16, ..t.primary })),
                border: iced::border::Border { width: 1.0, color: iced::Color { a: 0.4, ..t.primary }, radius: 0.0.into() },
                ..Default::default()
            }),
            body,
        ]
        .width(Length::Fill)
        .into()
    }

    /// The chip under a question that was answered with passages (the Studio's `ragChipsHtml`): a line
    /// with the count and the names of the documents, the passages behind a click.
    fn passage_chip<'a>(&'a self, t: Tokens, i: usize, passages: &'a [rag::Passage]) -> Element<'a, Message> {
        let open = self.rag_open.contains(&i);
        let names = rag_names(passages);
        let head = button(
            row![
                svg(icon(if open { ICON_CHEVRON } else { ICON_CHEVRON_RIGHT })).width(12).height(12).style(muted_svg(t)),
                text(format!(
                    "Using {} passage{} from {names}",
                    passages.len(),
                    if passages.len() == 1 { "" } else { "s" }
                ))
                .size(12)
                .color(t.muted),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        )
        .on_press(Message::RagToggle(i))
        .padding([4, 8])
        .style(theme::ghost_button(t));
        let mut col = column![head].spacing(4);
        if open {
            for p in passages {
                // the box is as wide as the bubble above, so the text wraps inside it instead of running off
                col = col.push(
                    container(
                        column![
                            text(format!("{}, chunk {}/{}", p.doc, p.n, p.m)).size(12).font(bold()).color(t.fg_soft),
                            text(rag_snippet(&p.text)).size(12).color(t.muted).width(Length::Fill).wrapping(text::Wrapping::Word),
                        ]
                        .spacing(2)
                        .width(Length::Fill),
                    )
                    .padding([6, 10])
                    .width(Length::Fixed(560.0))
                    .style(theme::inset(t)),
                );
            }
        }
        col.into()
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
                // the attachments of the question above its text (the Studio's attachmentsHtml)
                let mut bubble_col = column![].spacing(6);
                if !m.images.is_empty() {
                    let mut thumbs = row![].spacing(6);
                    for url in &m.images {
                        if let Some(bytes) = data_url_bytes(url) {
                            thumbs = thumbs.push(
                                iced::widget::image(iced::widget::image::Handle::from_bytes(bytes))
                                    .width(140)
                                    .height(140)
                                    .content_fit(iced::ContentFit::Contain),
                            );
                        }
                    }
                    bubble_col = bubble_col.push(thumbs);
                }
                if !m.files.is_empty() {
                    let mut chips = row![].spacing(6);
                    for f in &m.files {
                        chips = chips.push(
                            container(
                                row![
                                    svg(icon(ICON_FILE)).width(12).height(12).style(muted_svg(t)),
                                    text(ellipsize(&f.name, 24)).size(12),
                                ]
                                .spacing(4)
                                .align_y(Alignment::Center),
                            )
                            .padding([3, 8])
                            .style(theme::inset(t)),
                        );
                    }
                    bubble_col = bubble_col.push(chips);
                }
                if !m.content.is_empty() {
                    bubble_col = bubble_col.push(text(&m.content).size(15));
                }
                let bubble = container(bubble_col).padding([9, 15]).max_width(560).style(theme::bubble(t));
                let mut col = column![row![Space::new().width(Length::Fill), bubble]].spacing(4);
                if let Some(passages) = m.passages.as_ref().filter(|p| !p.is_empty()) {
                    col = col.push(row![Space::new().width(Length::Fill), self.passage_chip(t, i, passages)]);
                }
                let mut line = row![].spacing(14).align_y(Alignment::Center);
                if let Some(v) = self.version_switch(t, i) {
                    line = line.push(v);
                }
                if self.live.is_none() {
                    line = line.push(button(text("Edit").size(12)).on_press(Message::EditMessage(i)).padding([3, 0]).style(theme::ghost_button(t)));
                }
                line = line.push(button(text("Remember").size(12)).on_press(Message::RememberOpen(i)).padding([3, 0]).style(theme::ghost_button(t)));
                line = line.push(button(text("Rewind").size(12)).on_press(Message::RewindTo(i)).padding([3, 0]).style(theme::ghost_button(t)));
                col.push(row![Space::new().width(Length::Fill), line]).into()
            }
            Role::Assistant => {
                let live = self.live.is_some() && i + 1 == self.messages.len();
                let mut col = column![].spacing(8);
                // the progress of a research run above the answer, its log under it
                // (the Studio's researchSlotsHtml)
                let mut log_button: Option<Element<'_, Message>> = None;
                if let Some(ui) = &m.research {
                    let now = research::now_ms();
                    let line = ui.progress(now);
                    if !line.is_empty() {
                        col = col.push(text(line).size(13).color(if ui.notice.is_some() || ui.stopped { t.danger } else { t.muted }));
                    }
                    if !ui.queries.is_empty() || !ui.pages.is_empty() || !ui.events.is_empty() || !ui.failures.is_empty() {
                        let summary = ui.summary(now);
                        log_button = Some(button(text(summary).size(12).color(t.muted)).on_press(Message::ResearchLogToggle(i)).padding([3, 0]).style(theme::ghost_button(t)).into());
                    }
                }
                if !m.reasoning.is_empty() {
                    // the box of the Studio: open while the answer has not started, a click folds it
                    // (`cot-box` with its summary of the character count)
                    let starting = live && m.content.is_empty();
                    let label = if starting { "Thinking" } else { "Thought process" };
                    let head = button(
                        text(format!("{label} ({} characters)", m.reasoning.chars().count()))
                            .size(12)
                            .color(t.muted),
                    )
                    .on_press(Message::ReasoningToggle(i))
                    .padding([2, 0])
                    .style(theme::ghost_button(t));
                    col = col.push(head);
                    if m.reasoning_open || starting {
                        col = col.push(
                            container(text(m.reasoning.clone()).size(13).color(t.fg_soft).width(Length::Fill))
                                .padding([10, 12])
                                .width(Length::Fill)
                                .style(theme::inset(t)),
                        );
                    }
                }
                if !m.content.is_empty() {
                    col = col.push(md::view_runs(&m.md, t, &self.theme(), &self.runs));
                } else if live {
                    col = col.push(text("…").size(15).color(t.muted));
                }
                for (ci, card) in m.tool_cards.iter().enumerate() {
                    let mut line = row![
                        text(card.label.clone()).size(12).color(if card.status == "error" { t.danger } else { t.muted }),
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center);
                    if card.change.is_some() && !card.undone {
                        line = line.push(button(text("Undo").size(12)).on_press(Message::MemoryUndo(i, ci)).padding([2, 6]).style(theme::ghost_button(t)));
                    }
                    col = col.push(container(line).padding([4, 10]).style(theme::inset(t)));
                }
                // the words of the answer with their probability as chips (logprobsHtml)
                if !m.logprobs.is_empty() {
                    let mut chips = row![].spacing(4);
                    for lp in &m.logprobs {
                        let p = lp.logprob.exp().clamp(0.0, 1.0);
                        let hue = (p * 130.0).round();
                        chips = chips.push(
                            container(text(lp.token.clone()).size(12).font(iced::Font::MONOSPACE))
                                .padding([2, 5])
                                .style(move |_| container::Style {
                                    background: Some(iced::Background::Color(Self::logprob_color(hue))),
                                    border: iced::border::rounded(4),
                                    ..Default::default()
                                }),
                        );
                    }
                    col = col.push(container(chips).padding([4, 0]));
                }
                if let Some(ui) = &m.research {
                    if ui.log_open {
                        // the sections of the Studio's researchLogHtml: queries, pages, problems, steps
                        if !ui.queries.is_empty() {
                            let mut list = column![text("Queries").size(12).font(bold())].spacing(2);
                            for (n, q) in ui.queries.iter().enumerate() {
                                list = list.push(text(format!("{}. {q}", n + 1)).size(12).color(t.fg_soft));
                            }
                            col = col.push(container(list).padding([6, 10]).style(theme::inset(t)));
                        }
                        if !ui.pages.is_empty() {
                            let mut list = column![text("Pages").size(12).font(bold())].spacing(4);
                            for p in &ui.pages {
                                let head = if p.n > 0 { format!("[{}] {}", p.n, p.title) } else { p.title.clone() };
                                let mut line = column![button(text(head).size(12).color(t.primary)).on_press(Message::OpenUrl(p.url.clone())).padding([0, 0]).style(theme::ghost_button(t))]
                                    .spacing(1);
                                if !p.date.is_empty() {
                                    line = line.push(text(p.date.clone()).size(11).color(t.muted));
                                }
                                if !p.ok {
                                    line = line.push(text(p.error.clone()).size(11).color(t.danger));
                                } else if p.snippet_only {
                                    line = line.push(text("search snippet only").size(11).color(t.muted));
                                } else if !p.notes.is_empty() {
                                    line = line.push(text(p.notes.clone()).size(11).color(t.fg_soft));
                                } else {
                                    line = line.push(text("(nothing relevant)").size(11).color(t.muted));
                                }
                                list = list.push(line);
                            }
                            col = col.push(container(list).padding([6, 10]).style(theme::inset(t)));
                        }
                        if !ui.failures.is_empty() {
                            let mut list = column![text("Problems").size(12).font(bold())].spacing(2);
                            for f in &ui.failures {
                                list = list.push(text(f.clone()).size(11).color(t.danger));
                            }
                            col = col.push(container(list).padding([6, 10]).style(theme::inset(t)));
                        }
                        if !ui.events.is_empty() {
                            let mut list = column![text("Steps").size(12).font(bold())].spacing(2);
                            for (s, text_line) in &ui.events {
                                list = list.push(text(format!("{s} s  {text_line}")).size(11).color(t.muted));
                            }
                            col = col.push(container(list).padding([6, 10]).style(theme::inset(t)));
                        }
                    }
                }
                if let Some(button_el) = log_button {
                    col = col.push(button_el);
                }
                if !live && !m.sources.is_empty() {
                    // the sources of the answer, numbered like the model cites them (ctSourcesHtml)
                    let mut sources_line = row![text("Sources").size(12).color(t.muted)].spacing(10).align_y(Alignment::Center);
                    for s in &m.sources {
                        let title = format!("[{}] {}", s.n, ellipsize(&s.title, 70));
                        sources_line = sources_line.push(
                            button(text(title).size(12).color(t.primary)).on_press(Message::OpenUrl(s.url.clone())).padding([0, 2]).style(theme::ghost_button(t)),
                        );
                    }
                    col = col.push(sources_line);
                }
                if !live && !m.content.is_empty() {
                    let mut meta = row![button(text("Copy answer").size(12)).on_press(Message::Copy(i)).padding([3, 0]).style(theme::ghost_button(t))].spacing(14).align_y(Alignment::Center);
                    meta = meta.push(button(text("MD").size(12)).on_press(Message::CopyMd(i)).padding([3, 0]).style(theme::ghost_button(t)));
                    meta = meta.push(button(text("Read aloud").size(12)).on_press(Message::Speak(i)).padding([3, 0]).style(theme::ghost_button(t)));
                    meta = meta.push(button(text("Retry").size(12)).on_press(Message::Regenerate(i)).padding([3, 0]).style(theme::ghost_button(t)));
                    if let Some(v) = self.version_switch(t, i) {
                        meta = meta.push(v);
                    }
                    meta = meta.push(button(text("Remember").size(12)).on_press(Message::RememberOpen(i)).padding([3, 0]).style(theme::ghost_button(t)));
                    meta = meta.push(button(text("Rewind").size(12)).on_press(Message::RewindTo(i)).padding([3, 0]).style(theme::ghost_button(t)));
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
            // a tool answer is only sent back to the model, never drawn as a line of the chat
            Role::Tool => Space::new().height(0).into(),
        }
    }

    /// The row over the input: the documents of this chat, then what is attached to the next
    /// question (the Studio's attachmentsRow with ragDocsChipsHtml).
    fn attachments_row(&self, t: Tokens) -> Element<'_, Message> {
        let mut items: Vec<Element<'_, Message>> = Vec::new();
        if let Some(cid) = self.current {
            if let Some(chat) = self.store.chat(cid) {
                for d in &chat.docs {
                    let title = format!(
                        "{}: {} characters{}. Searched for every question in this chat.",
                        d.name,
                        group(d.chars as u64),
                        d.pages.map(|n| format!(", {n} pages")).unwrap_or_default()
                    );
                    items.push(file_chip(t, &ellipsize(&d.name, 24), Some(title), Some(Message::ChatDocRemove(d.id.clone()))));
                }
            }
        }
        for (i, a) in self.attachments.iter().enumerate() {
            match a {
                Attachment::Image { data_url, .. } => {
                    let thumb: Element<'_, Message> = match data_url_bytes(data_url) {
                        Some(bytes) => iced::widget::image(iced::widget::image::Handle::from_bytes(bytes))
                            .width(34)
                            .height(34)
                            .content_fit(iced::ContentFit::Cover)
                            .into(),
                        None => Space::new().width(34).height(34).into(),
                    };
                    items.push(
                        row![
                            thumb,
                            button(svg(icon(ICON_X)).width(10).height(10).style(muted_svg(t)))
                                .on_press(Message::AttachmentRemove(i))
                                .padding(3)
                                .style(theme::ghost_button(t)),
                        ]
                        .spacing(4)
                        .align_y(Alignment::Center)
                        .into(),
                    );
                }
                Attachment::Text { name, .. } => {
                    items.push(file_chip(t, &ellipsize(name, 24), None, Some(Message::AttachmentRemove(i))));
                }
                Attachment::Doc { name, .. } => {
                    let title = format!("{name}: too long for the context, the passages that match your question are used");
                    items.push(file_chip(t, &ellipsize(name, 24), Some(title), Some(Message::AttachmentRemove(i))));
                }
            }
        }
        if items.is_empty() {
            return Space::new().height(0).into();
        }
        scrollable(row(items).spacing(6))
            .direction(scrollable::Direction::Horizontal(scrollable::Scrollbar::new()))
            .width(Length::Fill)
            .into()
    }

    /// Which tools this chat is offered (the Studio's ctOfferedFlags): a context of a few
    /// hundred tokens has no room for schemas, and a temporary or private chat has no memory.
    fn tool_flags(&self) -> tools::Flags {
        let model = self.model_info();
        let size = prompts::size(model.max_context, model.detail.parameters, self.store.prefs.prompt_length, self.last_prompt_speed());
        if size == prompts::Size::Tiny {
            return tools::Flags::default();
        }
        let chat = self.current.and_then(|id| self.store.chat(id));
        let private = chat.is_some_and(|c| c.temporary || c.no_personal);
        tools::Flags {
            web: self.use_web,
            memory: self.memory_tool_offered() && !private,
            chats: false,
        }
    }

    /// The definitions of the offered tools in the OpenAI format (None: no tools at all), for the
    /// request of the round (the Studio's ctOffered → the `tools` of ctGenerate).
    fn tool_defs(&self) -> Option<serde_json::Value> {
        let model = self.model_info();
        let size = prompts::size(model.max_context, model.detail.parameters, self.store.prefs.prompt_length, self.last_prompt_speed());
        let defs = tools::defs(self.tool_flags(), size == prompts::Size::Compact);
        if defs.as_array().is_some_and(|a| a.is_empty()) {
            None
        } else {
            Some(defs)
        }
    }

    fn composer(&self, t: Tokens) -> Element<'_, Message> {
        let editor = text_editor(&self.editor)
            .placeholder("Ask anything or give an instruction")
            .on_action(Message::Edit)
            .key_binding(|kp: KeyPress| {
                // while the menu under a "/" is open its keys belong to it (the Studio's keydown
                // handler on the box: arrows move, Enter or Tab pick, Escape closes)
                if !self.slash.is_empty() {
                    match kp.key.as_ref() {
                        keyboard::Key::Named(keyboard::key::Named::ArrowUp) => return Some(Binding::Custom(Message::SlashUp)),
                        keyboard::Key::Named(keyboard::key::Named::ArrowDown) => return Some(Binding::Custom(Message::SlashDown)),
                        keyboard::Key::Named(keyboard::key::Named::Escape) => return Some(Binding::Custom(Message::SlashClose)),
                        keyboard::Key::Named(keyboard::key::Named::Enter) if !kp.modifiers.shift() => return Some(Binding::Custom(Message::SlashEnter)),
                        keyboard::Key::Named(keyboard::key::Named::Tab) => return Some(Binding::Custom(Message::SlashEnter)),
                        _ => {}
                    }
                }
                match kp.key.as_ref() {
                    keyboard::Key::Named(keyboard::key::Named::Enter) if !kp.modifiers.shift() && matches!(kp.status, Status::Focused { .. }) => Some(Binding::Custom(Message::Send)),
                    _ => Binding::from_key_press(kp),
                }
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
        let attach = button(svg(icon(ICON_PLUS)).width(14).height(14).style(muted_svg(t)))
            .on_press_maybe((!busy).then_some(Message::Attach))
            .padding(6)
            .style(theme::ghost_button(t));
        // the chip to allow the web tools of the server (the Studio's btnToggleWeb)
        let web_chip = button(
            row![
                svg(icon(ICON_GLOBE)).width(14).height(14).style(move |theme, status| {
                    if self.use_web { accent_svg(t)(theme, status) } else { muted_svg(t)(theme, status) }
                }),
                text("Allow web access").size(13),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        )
        .on_press(Message::ToggleWeb)
        .padding([5, 10])
        .style(slash_button(t, self.use_web));
        // the chip for the research run (the Studio's btnToggleResearch between web and style)
        let research_chip = button(
            row![
                svg(icon(ICON_SEARCH)).width(14).height(14).style(move |theme, status| {
                    if self.research { accent_svg(t)(theme, status) } else { muted_svg(t)(theme, status) }
                }),
                text("Research").size(13),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        )
        .on_press(Message::ResearchToggle)
        .padding([5, 10])
        .style(slash_button(t, self.research));
        // the chip of the response style with its menu (the Studio's #btnStyle and #styleMenu in the
        // composer toolbar: "Style", or "Style: Concise" for a chat that has one)
        let staged = self
            .current
            .and_then(|id| self.store.chat(id))
            .map(|c| c.style.as_str())
            .unwrap_or(self.new_style.as_str());
        let style_id = prompts::style::style_of(staged);
        let style_chip = button(
            row![
                svg(icon(ICON_LINES)).width(14).height(14).style(move |theme, status| {
                    if style_id == "default" { muted_svg(t)(theme, status) } else { accent_svg(t)(theme, status) }
                }),
                text(if style_id == "default" { "Style".to_string() } else { format!("Style: {}", prompts::style::label_of(style_id)) })
                    .size(13),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        )
        .on_press(Message::StyleMenu)
        .padding([5, 10])
        .style(slash_button(t, style_id != "default"));
        let style_menu: Element<'_, Message> = if self.style_menu {
            let mut items = column![].spacing(1);
            for s in prompts::style::STYLES.iter() {
                items = items.push(
                    button(text(s.label).size(13).width(Length::Fill).align_x(iced::alignment::Horizontal::Left))
                        .on_press(Message::StylePick(s.id.to_string()))
                        .padding([6, 10])
                        .width(Length::Fill)
                        // the CSS marks the current one with a tint of the primary colour
                        .style(slash_button(t, s.id == style_id)),
                );
            }
            container(items)
                .width(Length::Fill)
                .padding(Padding { top: 4.0, right: 6.0, bottom: 4.0, left: 6.0 })
                .style(move |_| container::Style {
                    background: Some(iced::Background::Color(t.surface_2)),
                    border: iced::border::rounded(10).width(1.0).color(t.separator),
                    ..Default::default()
                })
                .into()
        } else {
            Space::new().height(0).into()
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
        let attachments = self.attachments_row(t);
        // the menu of the library over the box (the Studio's #slashMenu): a column of the matches,
        // the highlighted one with the tint of the CSS .active
        let slash_menu: Element<'_, Message> = if self.slash.is_empty() {
            Space::new().height(0).into()
        } else {
            let mut items = column![].spacing(2);
            for (i, &idx) in self.slash.iter().enumerate() {
                if let Some(p) = self.store.prefs.saved_prompts.get(idx) {
                    let active = i == self.slash_index;
                    items = items.push(
                        button(
                            column![
                                text(format!("/{}", p.title)).size(13).font(bold()),
                                text(promptlib::preview(&p.text)).size(11.5).color(t.muted),
                            ]
                            .spacing(2)
                            .align_x(Alignment::Start),
                        )
                        .on_press(Message::SlashPick(i))
                        .padding([6, 10])
                        .width(Length::Fill)
                        .style(slash_button(t, active)),
                    );
                }
            }
            container(items)
                .width(Length::Fill)
                .padding(Padding { top: 4.0, right: 6.0, bottom: 4.0, left: 6.0 })
                .style(move |_| container::Style {
                    background: Some(iced::Background::Color(t.surface_2)),
                    border: iced::border::rounded(10).width(1.0).color(t.separator),
                    ..Default::default()
                })
                .into()
        };
        let card = container(column![banner, attachments, slash_menu, editor, style_menu, row![attach, web_chip, research_chip, style_chip, hint, Space::new().width(Length::Fill), action].align_y(Alignment::Center).padding(Padding { top: 0.0, right: 10.0, bottom: 10.0, left: 16.0 })])
            .style(theme::card(t, theme::R_XL, false))
            .max_width(780);
        container(card).width(Length::Fill).center_x(Length::Fill).padding(Padding { top: 8.0, right: 24.0, bottom: 20.0, left: 24.0 }).into()
    }
}

/// A rough count of the tokens of a chat (four characters to a token) until the server has counted the real ones. The meter says so with a tilde.
/// The passages that travel with the newest question count too (the Studio's `ragTokens`).
fn estimate_tokens(messages: &[Msg]) -> u64 {
    let mut total: u64 = messages
        .iter()
        .map(|m| {
            m.content.chars().count() as u64 / 4
                + 4
                + m.images.len() as u64 * 256
                + m.files.iter().map(|f| f.text.chars().count() as u64 / 4).sum::<u64>()
        })
        .sum();
    if let Some(i) = messages.iter().rposition(|m| m.role == Role::User) {
        if let Some(passages) = &messages[i].passages {
            total += rag::prefix(passages).chars().count() as u64 / 4;
        }
    }
    total
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

/// Opens an address in the browser of the computer (the sources under an answer are links in
/// the Studio). Best effort: nothing happens when the computer has no browser command.
fn open_page(url: &str) {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd").args(["/C", "start", url]).spawn();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

/// One entry of the menu under a "/": the tint of the CSS `.slash-item.active, .slash-item:hover`
/// (the primary colour at 14 percent) over the rounded background.
fn slash_button(t: Tokens, active: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        button::Style {
            background: if active || hovered {
                Some(iced::Background::Color(iced::Color { a: 0.14, ..t.primary }))
            } else {
                None
            },
            text_color: t.fg,
            border: iced::border::rounded(6),
            ..Default::default()
        }
    }
}

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

/// The images the window shrinks itself (the `file.type.startsWith('image/')` of the Studio).
const IMAGE_EXTS: [&str; 10] = ["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "avif", "ico", "tiff"];

/// What a file may weigh before the attach flow sends it to the server's extract route
/// (the Studio's `RAG_MAX_UPLOAD`; text files have their own `rag::MAX_TEXT_ATTACHMENT`).
const RAG_MAX_UPLOAD: u64 = 50 * 1024 * 1024;

/// The documents only the server can read (the Studio's `RAG_SERVER_EXT`).
fn server_ext(ext: &str) -> bool {
    matches!(ext, "pdf" | "docx" | "xlsx" | "epub" | "html" | "htm" | "xhtml")
}

fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// The media type of an image from its first bytes (what the browser knows from the file type).
fn mime_of(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        "image/png"
    } else if bytes.starts_with(&[0xFF, 0xD8]) {
        "image/jpeg"
    } else if bytes.starts_with(b"GIF8") {
        "image/gif"
    } else if bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        "image/webp"
    } else {
        "application/octet-stream"
    }
}

/// The bytes behind a data URL (for the thumbnails of the window).
fn data_url_bytes(url: &str) -> Option<Vec<u8>> {
    let encoded = url.split_once("base64,")?.1;
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.decode(encoded).ok()
}

/// The image shrunk the way the Studio shrinks one: at most 1280 pixels on the long side and then
/// JPEG (quality 0.84); a small image keeps its bytes, and an image that cannot be decoded keeps
/// them too (the Studio keeps the original data URL when the browser cannot draw it).
fn shrink_image(bytes: &[u8]) -> Result<String, String> {
    let original = format!("data:{};base64,{}", mime_of(bytes), b64(bytes));
    let img = match image::load_from_memory(bytes) {
        Err(_) => return Ok(original),
        Ok(img) => img,
    };
    let long = img.width().max(img.height()) as f64;
    let scale = (1280.0 / long).min(1.0_f64);
    if scale == 1.0 && bytes.len() < 300 * 1024 {
        return Ok(original);
    }
    let tw = ((img.width() as f64 * scale).round() as u32).max(1);
    let th = ((img.height() as f64 * scale).round() as u32).max(1);
    let small = if scale < 1.0 { img.resize_exact(tw, th, image::imageops::FilterType::Lanczos3) } else { img };
    let mut out = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 214);
    encoder.encode_image(&small).map_err(|e| format!("the image could not be read: {e}"))?;
    Ok(format!("data:image/jpeg;base64,{}", b64(&out)))
}

/// A chip of the attachment row: an icon, a name (a hover tells the size and what it is good for)
/// and a cross that removes it (the file-chip of the Studio).
fn file_chip(t: Tokens, name: &str, title: Option<String>, remove: Option<Message>) -> Element<'static, Message> {
    let mut line = row![
        svg(icon(ICON_FILE)).width(12).height(12).style(muted_svg(t)),
        text(name.to_string()).size(12),
    ]
    .spacing(4)
    .align_y(Alignment::Center);
    if let Some(msg) = remove {
        line = line.push(
            button(svg(icon(ICON_X)).width(10).height(10).style(muted_svg(t))).on_press(msg).padding(3).style(theme::ghost_button(t)),
        );
    }
    let chip: Element<'static, Message> = container(line).padding([3, 8]).style(theme::inset(t)).into();
    // the Studio shows the details of a chip as the title of a hover; the window has no hover text,
    // so an attachment that is not a plain file says it in the toast when it is added instead
    let _ = title;
    chip
}



/// The names of the documents of the passages: "A and B", and with more than two "A, B and N more"
/// (the Studio's `ragNames`).
fn rag_names(passages: &[rag::Passage]) -> String {
    let mut unique: Vec<&str> = Vec::new();
    for p in passages {
        if !unique.contains(&p.doc.as_str()) {
            unique.push(&p.doc);
        }
    }
    if unique.len() > 2 {
        format!("{}, {} and {} more", unique[0], unique[1], unique.len() - 2)
    } else {
        unique.join(" and ")
    }
}

/// What a passage shows under its source: the whitespace collapsed, cut to 700 characters (ragSnippet).
fn rag_snippet(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > 700 {
        format!("{}...", flat.chars().take(700).collect::<String>())
    } else {
        flat
    }
}

/// Bytes as the Studio shows them (`formatBytes`): "0 B", "457 B", "4.2 KB", "20.0 MB".
pub fn format_bytes(b: u64) -> String {
    if b == 0 {
        return "0 B".to_string();
    }
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut i = 0usize;
    let mut v = b as f64;
    while v >= 1024.0 && i < units.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{b} B")
    } else {
        format!("{v:.1} {}", units[i])
    }
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

fn asst_label<'a>(t: Tokens, label: &'a str) -> Element<'a, Message> {
    text(label).size(13).color(t.fg_soft).into()
}

fn asst_text_input<'a>(t: Tokens, placeholder: &'a str, value: &'a str, on_input: fn(String) -> Message, width: Length) -> Element<'a, Message> {
    text_input(placeholder, value)
        .on_input(on_input)
        .size(14)
        .padding([7, 10])
        .width(width)
        .style(move |_, status| text_input::Style {
            background: iced::Background::Color(t.surface_2),
            border: iced::Border { radius: 8.0.into(), width: 1.0, color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator } },
            icon: t.muted,
            placeholder: t.muted,
            value: t.fg,
            selection: t.primary_soft,
        })
        .into()
}

fn asst_area<'a>(t: Tokens, content: &'a Content, on_action: fn(Action) -> Message, placeholder: &'a str, min: f32) -> Element<'a, Message> {
    text_editor(content)
        .placeholder(placeholder)
        .on_action(on_action)
        .padding(10)
        .size(14)
        .height(Length::Shrink)
        .min_height(min)
        .max_height(min * 2.0)
        .style(move |_, status| text_editor::Style {
            background: iced::Background::Color(t.surface_2),
            border: iced::Border { radius: 10.0.into(), width: 1.0, color: if matches!(status, Status::Focused { .. }) { t.fg_soft } else { t.separator } },
            placeholder: t.muted,
            value: t.fg,
            selection: iced::Color { a: 0.35, ..t.primary },
        })
        .into()
}

/// One option of the two selects of the assistant editor: a mark, the label, a click.
fn asst_option(t: Tokens, label: String, selected: bool, on_press: Message) -> Element<'static, Message> {
    let mark: Element<'static, Message> = if selected {
        svg(icon(ICON_CHECK)).width(14).height(14).style(accent_svg(t)).into()
    } else {
        Space::new().width(14).height(14).into()
    };
    button(row![mark, text(ellipsize(&label, 46)).size(12)].spacing(8).align_y(Alignment::Center))
        .on_press(on_press)
        .padding([5, 8])
        .width(Length::Fill)
        .style(theme::row_button(t, false))
        .into()
}

pub fn starter_style(t: Tokens) -> impl Fn(&Theme, button::Status) -> button::Style {
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

/// The main button of the start screen (`renderHeroStart` of the Studio): the accent, not a
/// plain starter button.
pub fn primary_style(t: Tokens) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| button::Style {
        background: Some(iced::Background::Color(match status {
            button::Status::Hovered | button::Status::Pressed => t.primary_soft,
            _ => t.primary,
        })),
        text_color: t.on_primary,
        border: iced::Border { color: t.primary, width: 0.5, radius: theme::R_LG.into() },
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
        let m = |c: &str| Msg {
            role: Role::User,
            content: c.into(),
            reasoning: String::new(),
            stats: None,
            passages: None,
            images: Vec::new(),
            files: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
            tool_cards: Vec::new(),
            sources: Vec::new(),
            research: None,
            logprobs: Vec::new(),
            reasoning_open: false,
            md: markdown::Content::new(),
        };
        assert_eq!(estimate_tokens(&[m(&"a".repeat(40))]), 14);
        assert_eq!(estimate_tokens(&[]), 0);
    }
}
