//! The engine options and the server settings of the Settings page: the caches, the memory the server may use, and when it unloads a model.
//!
//! The server owns them (`/api/studio/engine`, `/api/studio/settings`) and says what each one is by default and which ones its environment
//! fixes. Every change is written at once; the engine options take effect when the server starts again, which the page offers.

use crate::app::Message;
use crate::models::{card, rule};
use crate::settings::{segmented, setting};
use crate::theme::Tokens;
use iced::widget::{button, column, container, row, text, text_input, toggler, Space};
use iced::{Alignment, Element, Length};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum Kind {
    Switch,
    /// A whole number with its unit; the lowest value allowed.
    Number(&'static str, u64),
    Choice(&'static [(&'static str, &'static str)]),
    Text(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub struct Field {
    pub key: &'static str,
    /// The id of the control on the page (`engOutputCache` ...): what the fixtures of the page are matched by.
    pub id: &'static str,
    pub title: &'static str,
    pub help: &'static str,
    pub kind: Kind,
}

/// *Answers that were given before* (`engOutputCache` ... `engCacheScope`).
pub const ANSWERS: &[Field] = &[
    Field { key: "output_cache", id: "engOutputCache", title: "Remember identical answers", help: "When the same request with temperature 0 comes again, the saved answer is returned at once instead of being computed again. Requests with a temperature above 0, with images, tools or streaming never use it.", kind: Kind::Choice(&[("exact", "On"), ("off", "Off")]) },
    Field { key: "output_cache_mb", id: "engCacheMb", title: "Memory for saved answers", help: "The cache never grows beyond this size; the oldest answers go first.", kind: Kind::Number("MB", 1) },
    Field { key: "output_cache_ttl", id: "engCacheTtl", title: "Keep answers for", help: "After this time a saved answer is dropped.", kind: Kind::Number("seconds", 1) },
    Field { key: "cache_scope", id: "engCacheScope", title: "Who shares saved answers", help: "“Everyone” shares one cache. “Per key and user” keeps the answers of each API key (and the user field of the request) apart; a request that has neither is not saved. Use it when several people use the server.", kind: Kind::Choice(&[("global", "Everyone"), ("tenant", "Per key and user"), ("off", "Nobody (no saving)")]) },
];

/// *Speed and memory*.
pub const SPEED: &[Field] = &[
    Field { key: "prefix_cache", id: "engPrefixCache", title: "Share the start of prompts", help: "Requests that begin with the same text (a system prompt, a long document) reuse the work already done for it. Turn it off if people on this server must not share anything.", kind: Kind::Switch },
    Field { key: "state_cache", id: "engStateCache", title: "Remember the start of a conversation (Qwen3.5, Mamba, RWKV)", help: "Models with recurrent layers cannot point at old work like other models, so they read the whole conversation again on every message. With this on, the server keeps a copy of the model’s state at the end of each prompt and continues from it when the next prompt starts the same way. Automatic: on. Turn it off to save memory.", kind: Kind::Switch },
    Field { key: "state_cache_mb", id: "engStateMb", title: "Memory for those copies", help: "One copy of a 9B model takes about 100 MB. The oldest copy goes first when the limit is reached.", kind: Kind::Number("MB", 1) },
    Field { key: "prefix_salt", id: "engPrefixSalt", title: "Separate this installation", help: "A text that keeps the shared prompt blocks of this server apart from those of another server. Leave it empty unless you run several.", kind: Kind::Text("") },
    Field { key: "preemption", id: "engPreemption", title: "When memory runs out", help: "“Recompute” drops the work of a waiting request and does it again later (saves memory). “Move to RAM” keeps it in the main memory (faster to continue, needs RAM).", kind: Kind::Choice(&[("recompute", "Recompute"), ("swap", "Move to RAM")]) },
];

/// *Prompt blocks on disk*.
pub const DISK: &[Field] = &[
    Field { key: "kv_disk_dir", id: "engKvDir", title: "Keep shared prompt blocks on disk", help: "A folder where the work done for long shared prompts (system prompts, documents) is saved, so a restart does not have to redo it. Empty means off. Use a fast disk; the folder is never read from another program.", kind: Kind::Text("off (type a full folder path)") },
    Field { key: "kv_disk_mb", id: "engKvMb", title: "Disk space for them", help: "The oldest blocks are removed when the folder is full.", kind: Kind::Number("MB", 1) },
    Field { key: "kv_ram_mb", id: "engKvRam", title: "Keep the most recent ones in memory", help: "Blocks that were used lately are kept in the main memory as well; reading them is much faster than reading the disk. They are gone after a restart. 0 means off. Works without the folder above.", kind: Kind::Number("MB", 0) },
];

/// *Telemetry*.
pub const TELEMETRY: &[Field] = &[
    Field { key: "otel_endpoint", id: "engOtelEndpoint", title: "Send traces to", help: "The address of an OpenTelemetry collector (for example http://127.0.0.1:4318). Nothing is sent while it is empty. Only timings and sizes are sent, never the text of a conversation.", kind: Kind::Text("off") },
    Field { key: "otel_service", id: "engOtelService", title: "Name of this server", help: "How the server is called in the traces.", kind: Kind::Text("") },
];

/// *Who may administer* and *Stop taking new requests*: the two rows of *Administration* that are not options.
pub const ADMIN: (&str, &str) = ("Who may administer", "This computer always may. Another computer needs an API key with the Admin right (Settings > Server and API). A key made for chatting never opens these controls.");
pub const DRAIN: (&str, &str) = ("Stop taking new requests", "The server finishes what it is doing and answers new requests with “busy”. Used before a restart or an update, so nobody loses an answer. Takes effect at once.");

/// Every option of the pane, in the order of the page.
pub fn all_fields() -> impl Iterator<Item = &'static Field> {
    ANSWERS.iter().chain(SPEED).chain(DISK).chain(TELEMETRY)
}

/// What the server told about its engine options.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EngineData {
    pub values: Map<String, Value>,
    pub defaults: Map<String, Value>,
    /// Options the environment of the server fixes: the page cannot change them.
    pub locked: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ServerData {
    pub idle_unload_minutes: u64,
    pub load_on_start: bool,
}

#[derive(Debug, Clone, Default)]
pub struct EngineState {
    pub engine: EngineData,
    pub server: ServerData,
    pub loaded: bool,
    /// The text being typed into a number or text box, until Enter.
    pub drafts: HashMap<String, String>,
    pub error: Option<String>,
    pub restart_needed: bool,
    pub restarting: bool,
    /// The live numbers and controls of the running server.
    pub live: LiveTexts,
    /// A question that waits for OK (`confirmDialog` of the page).
    pub ask: Option<Ask>,
}

/// What the pane asks before it does it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ask {
    Reset,
    KvClear,
}

impl Ask {
    pub fn text(self) -> &'static str {
        match self {
            Ask::Reset => "Set every option on this page back to its default?",
            Ask::KvClear => "Delete every prompt block that is saved on disk?",
        }
    }
}

/// What the running server answered to the three routes of the live numbers (`None`: no answer).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Live {
    pub config: Option<Value>,
    pub kv: Option<Value>,
    pub prefix: Option<Value>,
}

/// The lines under the cards and the state of the live controls (`live()` and `liveKv()` of engine.js).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LiveTexts {
    pub cache: String,
    pub otel: String,
    pub state: String,
    pub kv: String,
    pub draining: bool,
    /// *Empty the cache now* and the drain switch work (the server has a model).
    pub cache_controls: bool,
    /// *Delete the blocks on disk* and *Forget the shared prompts in memory* work.
    pub kv_controls: bool,
}

fn n(v: &Value, key: &str) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn bytes(v: &Value, key: &str) -> String {
    crate::modelpicker::format_bytes(v.get(key).and_then(Value::as_f64).unwrap_or(0.0))
}

/// The page's own sentences for the numbers of the running server.
pub fn live_texts(l: &Live) -> LiveTexts {
    let mut out = LiveTexts::default();
    match &l.config {
        Some(data) => {
            let cache = data.get("output_cache").cloned().unwrap_or(Value::Null);
            out.cache = format!(
                "now: {} saved answers, {} · answered from the cache {} times, computed {} times, {} prompt tokens saved{}",
                n(&cache, "entries"),
                bytes(&cache, "bytes"),
                n(&cache, "hits"),
                n(&cache, "misses"),
                n(&cache, "saved_prompt_tokens"),
                match data.get("coalescing").filter(|c| !c.is_null() && c.as_bool() != Some(false)) {
                    Some(c) => format!(" · identical requests shared: {}", n(c, "followers")),
                    None => String::new(),
                }
            );
            let t = data.get("telemetry").cloned().unwrap_or(Value::Null);
            out.otel = if t.get("enabled").and_then(Value::as_bool).unwrap_or(false) { format!("sending traces · {} batches sent, {} dropped", n(&t, "sent"), n(&t, "dropped")) } else { "telemetry is off".into() };
            out.draining = data.get("draining").and_then(Value::as_bool).unwrap_or(false);
            out.cache_controls = true;
        }
        None => out.cache = "Load a model to see the live numbers of the cache.".into(),
    }
    match &l.kv {
        Some(t) => {
            out.kv = if t.get("enabled").and_then(Value::as_bool) == Some(false) {
                "The disk copy is off.".into()
            } else {
                let disk = if t.get("dir").and_then(Value::as_str).is_some_and(|d| !d.is_empty()) { format!("{} of {} on disk", bytes(t, "used_bytes"), bytes(t, "max_bytes")) } else { "no folder".into() };
                let ram = if n(t, "ram_max_bytes") > 0 {
                    format!(" · {} of {} in memory ({} blocks, {} read from there)", bytes(t, "ram_used_bytes"), bytes(t, "ram_max_bytes"), n(t, "ram_blocks"), n(t, "ram_hits"))
                } else {
                    String::new()
                };
                format!("now: {disk}{ram} · {} blocks read back, {} written, {} skipped", n(t, "hits"), n(t, "spills"), n(t, "drops"))
            };
            out.kv_controls = true;
            out.state = match l.prefix.as_ref().and_then(|p| p.get("state_checkpoints")).filter(|s| !s.is_null()) {
                None => String::new(),
                Some(st) if !st.get("enabled").and_then(Value::as_bool).unwrap_or(false) => "This model does not need copies of its state (or they are off).".into(),
                Some(st) => format!("now: {} copies, {} of {} · {} messages continued from one, {} prompt tokens not read again", n(st, "entries"), bytes(st, "bytes"), bytes(st, "budget_bytes"), n(st, "hits"), n(st, "tokens_skipped")),
            };
        }
        None => out.kv = "Load a model to see the live numbers.".into(),
    }
    out
}

/// The three routes of the live numbers (`/admin/config`, `/admin/kv-tier`, `/admin/prefix-cache`).
pub async fn fetch_live(base: String) -> Live {
    async fn ok(base: &str, path: &str) -> Option<Value> {
        let resp = reqwest::get(format!("{base}{path}")).await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        resp.json().await.ok()
    }
    let config = ok(&base, "/admin/config").await;
    let kv = ok(&base, "/admin/kv-tier").await;
    // the page asks for the copies of the state only when the disk tier answered
    let prefix = if kv.is_some() { ok(&base, "/admin/prefix-cache").await } else { None };
    Live { config, kv, prefix }
}

/// A DELETE of the admin routes; the answer's number under `key` (what the toast says).
pub async fn admin_delete(base: String, path: &'static str, key: &'static str) -> Result<u64, String> {
    let resp = reqwest::Client::new().delete(format!("{base}{path}")).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(String::new());
    }
    let v: Value = resp.json().await.unwrap_or(Value::Null);
    Ok(v.get(key).and_then(Value::as_u64).unwrap_or(0))
}

/// The drain switch: `POST /admin/drain` to stop taking requests, `DELETE` to take them again.
pub async fn admin_drain(base: String, on: bool) -> Result<bool, String> {
    let c = reqwest::Client::new();
    let url = format!("{base}/admin/drain");
    let resp = if on { c.post(url).send().await } else { c.delete(url).send().await }.map_err(|e| e.to_string())?;
    if resp.status().is_success() {
        Ok(on)
    } else {
        Err(String::new())
    }
}

impl EngineState {
    /// What a field shows now: the saved value, else the default.
    pub fn value(&self, key: &str) -> Value {
        self.engine.values.get(key).or_else(|| self.engine.defaults.get(key)).cloned().unwrap_or(Value::Null)
    }

    pub fn is_default(&self, key: &str) -> bool {
        self.engine.defaults.get(key).is_none_or(|d| *d == self.value(key))
    }

    pub fn locked(&self, key: &str) -> bool {
        self.engine.locked.iter().any(|k| k == key)
    }

    /// The text of a box: the draft while typing, else the value.
    pub fn shown(&self, key: &str) -> String {
        if let Some(d) = self.drafts.get(key) {
            return d.clone();
        }
        match self.value(key) {
            Value::String(s) => s,
            Value::Number(n) => n.to_string(),
            _ => String::new(),
        }
    }

    /// The body that puts every option back to its default (a switch to its default value, a text box to empty).
    pub fn reset_body(&self) -> Value {
        let mut body = Map::new();
        for f in all_fields() {
            if self.locked(f.key) {
                continue;
            }
            match f.kind {
                // `String(defaults[key] ?? '')` of the page
                Kind::Text(_) => {
                    body.insert(f.key.into(), json!(self.engine.defaults.get(f.key).and_then(Value::as_str).unwrap_or("")));
                }
                _ => {
                    if let Some(d) = self.engine.defaults.get(f.key) {
                        body.insert(f.key.into(), d.clone());
                    }
                }
            }
        }
        Value::Object(body)
    }
}

pub fn parse_engine(v: &Value) -> EngineData {
    let map = |k: &str| v.get(k).and_then(Value::as_object).cloned().unwrap_or_default();
    EngineData {
        values: map("values"),
        defaults: map("defaults"),
        locked: v.get("set_by_environment").and_then(Value::as_array).into_iter().flatten().filter_map(|k| k.as_str().map(str::to_string)).collect(),
    }
}

pub fn parse_server(v: &Value) -> ServerData {
    ServerData { idle_unload_minutes: v.get("idle_unload_minutes").and_then(Value::as_u64).unwrap_or(0), load_on_start: v.get("load_on_start").and_then(Value::as_bool).unwrap_or(false) }
}

/// The value a typed text becomes for a field, or the reason it cannot: numbers are whole numbers at or above the lowest allowed value.
pub fn value_from_text(field: &Field, text: &str) -> Result<Value, String> {
    match field.kind {
        Kind::Number(unit, min) => match text.trim().parse::<u64>() {
            Ok(n) if n >= min => Ok(json!(n)),
            _ => Err(format!("{} must be a whole number of {unit}, {min} or more.", field.title)),
        },
        Kind::Text(_) => Ok(json!(text.trim())),
        Kind::Switch | Kind::Choice(_) => Err("not a text field".into()),
    }
}

pub fn field(key: &str) -> Option<&'static Field> {
    all_fields().find(|f| f.key == key)
}

// ---- asking the server -----------------------------------------------------------------------------------------------------------

async fn get_json(base: &str, path: &str) -> Result<Value, String> {
    reqwest::get(format!("{base}{path}")).await.map_err(|e| format!("no server at {base}: {e}"))?.error_for_status().map_err(|e| e.to_string())?.json().await.map_err(|e| e.to_string())
}

async fn post(base: &str, path: &str, body: Value) -> Result<(), String> {
    let resp = reqwest::Client::new().post(format!("{base}{path}")).json(&body).send().await.map_err(|e| format!("no server at {base}: {e}"))?;
    let status = resp.status();
    if status.is_success() {
        return Ok(());
    }
    let data: Value = resp.json().await.unwrap_or(Value::Null);
    Err(data.pointer("/error/message").and_then(Value::as_str).or_else(|| data.get("error").and_then(Value::as_str)).map(str::to_string).unwrap_or_else(|| format!("the server answered {status}")))
}

pub async fn fetch(base: String) -> Result<(EngineData, ServerData), String> {
    let engine = parse_engine(&get_json(&base, "/api/studio/engine").await?);
    let server = parse_server(&get_json(&base, "/api/studio/settings").await?);
    Ok((engine, server))
}

pub async fn save_engine(base: String, body: Value) -> Result<(), String> {
    post(&base, "/api/studio/engine", body).await
}

pub async fn save_server(base: String, body: Value) -> Result<(), String> {
    post(&base, "/api/studio/settings", body).await
}

pub async fn restart(base: String) -> Result<(), String> {
    post(&base, "/api/studio/restart", json!({})).await
}

// ---- the cards -------------------------------------------------------------------------------------------------------------------

fn control<'a>(t: Tokens, f: &'static Field, s: &'a EngineState) -> Element<'a, Message> {
    let locked = s.locked(f.key);
    match f.kind {
        Kind::Switch => {
            let on = s.value(f.key).as_bool().unwrap_or(false);
            toggler(on)
                .on_toggle_maybe((!locked).then_some(move |v| Message::EngineSet(f.key.into(), json!(v))))
                .size(22)
                .style(move |_, status| {
                    let on = matches!(status, toggler::Status::Active { is_toggled: true } | toggler::Status::Hovered { is_toggled: true } | toggler::Status::Disabled { is_toggled: true });
                    toggler::Style {
                        background: iced::Background::Color(if on { t.primary } else { t.fill_hover }),
                        background_border_width: 0.0,
                        background_border_color: iced::Color::TRANSPARENT,
                        foreground: iced::Background::Color(iced::Color::WHITE),
                        foreground_border_width: 0.0,
                        foreground_border_color: iced::Color::TRANSPARENT,
                        text_color: None,
                        border_radius: None,
                        padding_ratio: 0.1,
                    }
                })
                .into()
        }
        Kind::Choice(options) => {
            let now = s.value(f.key);
            let now = now.as_str().unwrap_or_default();
            if locked {
                let label = options.iter().find(|(v, _)| *v == now).map_or(now.to_string(), |(_, l)| l.to_string());
                return container(text(label).size(13).color(t.muted)).padding([5, 14]).style(crate::theme::chip(t)).into();
            }
            segmented(t, options.iter().map(|(v, label)| (*label, *v == now, Message::EngineSet(f.key.into(), json!(v)))).collect())
        }
        Kind::Number(unit, _) | Kind::Text(unit) => {
            let numeric = matches!(f.kind, Kind::Number(..));
            let mut input = text_input(if numeric { "" } else { unit }, &s.shown(f.key)).size(13).padding([6, 10]).width(if numeric { 110 } else { 200 }).style(move |_, status| text_input::Style {
                background: iced::Background::Color(t.surface_2),
                border: iced::Border { radius: 8.0.into(), width: 1.0, color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator } },
                icon: t.muted,
                placeholder: t.muted,
                value: t.fg,
                selection: t.primary_soft,
            });
            if !locked {
                input = input.on_input(move |v| Message::EngineDraft(f.key.into(), v)).on_submit(Message::EngineSubmit(f.key.into()));
            }
            if numeric {
                row![input, text(unit).size(12).color(t.muted).width(60)].spacing(8).align_y(Alignment::Center).into()
            } else {
                input.into()
            }
        }
    }
}

fn group<'a>(t: Tokens, fields: &'static [Field], s: &'a EngineState) -> Element<'a, Message> {
    let mut list = column![].spacing(14);
    for (i, f) in fields.iter().enumerate() {
        if i > 0 {
            list = list.push(rule(t));
        }
        let help: Element<'a, Message> = if s.locked(f.key) { text("Fixed by the environment of the server.").size(12).color(t.muted).into() } else { Space::new().height(0).into() };
        list = list.push(column![setting(t, f.title, f.help, control(t, f, s)), help].spacing(2));
    }
    list.into()
}

/// The line of live numbers under a group (`engCacheStats` and its kin).
fn stats<'a>(t: Tokens, line: &'a str) -> Element<'a, Message> {
    if line.is_empty() {
        Space::new().height(0).into()
    } else {
        text(line).size(11).font(iced::Font::MONOSPACE).color(t.muted).into()
    }
}

fn ghost<'a>(t: Tokens, label: &'a str, msg: Option<Message>) -> Element<'a, Message> {
    button(text(label).size(13)).on_press_maybe(msg).padding([5, 12]).style(crate::theme::ghost_button(t)).into()
}

fn switch<'a>(t: Tokens, on: bool, msg: Option<fn(bool) -> Message>) -> Element<'a, Message> {
    toggler(on)
        .on_toggle_maybe(msg)
        .size(22)
        .style(move |_, status| {
            let on = matches!(status, toggler::Status::Active { is_toggled: true } | toggler::Status::Hovered { is_toggled: true } | toggler::Status::Disabled { is_toggled: true });
            toggler::Style {
                background: iced::Background::Color(if on { t.primary } else { t.fill_hover }),
                background_border_width: 0.0,
                background_border_color: iced::Color::TRANSPARENT,
                foreground: iced::Background::Color(iced::Color::WHITE),
                foreground_border_width: 0.0,
                foreground_border_color: iced::Color::TRANSPARENT,
                text_color: None,
                border_radius: None,
                padding_ratio: 0.1,
            }
        })
        .into()
}

/// The question that waits for OK (`confirmDialog`).
pub fn ask_window<'a>(t: Tokens, ask: Ask) -> Element<'a, Message> {
    let window = container(
        column![
            text(ask.text()).size(14),
            row![
                Space::new().width(Length::Fill),
                button(text("Cancel").size(13)).on_press(Message::EngineAskCancel).padding([6, 12]).style(crate::theme::ghost_button(t)),
                button(text("OK").size(13)).on_press(Message::EngineAskOk).padding([6, 14]).style(crate::theme::primary_button(t)),
            ]
            .spacing(8),
        ]
        .spacing(14),
    )
    .padding(16)
    .width(400)
    .style(crate::theme::card(t, crate::theme::R_LG, false));
    container(iced::widget::opaque(window)).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into()
}

pub fn view<'a>(t: Tokens, s: &'a EngineState, connected: bool) -> Element<'a, Message> {
    if !connected {
        return card(t, "Server", text("Settings of the server appear when it is running.").size(13).color(t.muted).into());
    }
    if !s.loaded {
        return card(t, "Server", text(s.error.as_deref().unwrap_or("Looking…")).size(13).color(if s.error.is_some() { t.danger } else { t.muted }).into());
    }
    let live = &s.live;
    let answers = column![group(t, ANSWERS, s), stats(t, &live.cache), row![ghost(t, "Empty the cache now", live.cache_controls.then_some(Message::EngineCacheClear))]].spacing(8);
    let speed = column![group(t, &SPEED[..3], s), stats(t, &live.state), rule(t), group(t, &SPEED[3..], s)].spacing(8);
    let disk = column![
        group(t, DISK, s),
        stats(t, &live.kv),
        row![ghost(t, "Delete the blocks on disk", live.kv_controls.then_some(Message::EngineAsk(Ask::KvClear))), ghost(t, "Forget the shared prompts in memory", live.kv_controls.then_some(Message::EnginePrefixClear))].spacing(8),
    ]
    .spacing(8);
    let telemetry = column![group(t, TELEMETRY, s), stats(t, &live.otel)].spacing(8);
    let admin = column![
        setting(t, ADMIN.0, ADMIN.1, text("Keys: Server and API").size(12).color(t.muted).into()),
        rule(t),
        setting(
            t,
            DRAIN.0,
            DRAIN.1,
            switch(t, live.draining, live.cache_controls.then_some(Message::EngineDrain as fn(bool) -> Message)),
        ),
    ]
    .spacing(14);

    let mut server = column![].spacing(14);
    server = server.push(setting(t, "Load the last model at start", "The server loads the model you used last when it starts.", switch(t, s.server.load_on_start, Some(|v| Message::ServerSet("load_on_start".into(), json!(v))))));
    server = server.push(rule(t));
    let idle = s.drafts.get("idle_unload_minutes").cloned().unwrap_or_else(|| s.server.idle_unload_minutes.to_string());
    server = server.push(setting(
        t,
        "Unload the model when idle",
        "Frees the memory after this many minutes without a request. 0 means never.",
        row![
            text_input("0", &idle).size(13).padding([6, 10]).width(110).on_input(|v| Message::EngineDraft("idle_unload_minutes".into(), v)).on_submit(Message::EngineSubmit("idle_unload_minutes".into())).style(move |_, status| text_input::Style {
                background: iced::Background::Color(t.surface_2),
                border: iced::Border { radius: 8.0.into(), width: 1.0, color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator } },
                icon: t.muted,
                placeholder: t.muted,
                value: t.fg,
                selection: t.primary_soft,
            }),
            text("minutes").size(12).color(t.muted).width(60)
        ]
        .spacing(8)
        .align_y(Alignment::Center)
        .into(),
    ));

    let locked: Vec<&str> = s.engine.locked.iter().map(String::as_str).collect();
    let mut footer = row![].spacing(10).align_y(Alignment::Center);
    if let Some(e) = &s.error {
        footer = footer.push(text(e.as_str()).size(13).color(t.danger).width(Length::Fill));
    } else if s.restarting {
        footer = footer.push(text("Restarting the server…").size(13).color(t.muted).width(Length::Fill));
    } else if s.restart_needed {
        footer = footer.push(text("Saved. The changes take effect when the server restarts.").size(13).color(t.fg_soft).width(Length::Fill));
        footer = footer.push(button(text("Restart the server now").size(13)).on_press(Message::RestartServer).padding([5, 14]).style(crate::theme::primary_button(t)));
    } else if !locked.is_empty() {
        footer = footer.push(text(format!("Set by the environment of the server and not changeable here: {}.", locked.join(", "))).size(12).color(t.muted).width(Length::Fill));
    } else {
        footer = footer.push(Space::new().width(Length::Fill));
    }
    footer = footer.push(button(text("Back to the defaults").size(13)).on_press(Message::EngineAsk(Ask::Reset)).padding([5, 14]).style(crate::theme::secondary_button(t)));

    column![
        card(t, "Answers that were given before", answers.into()),
        card(t, "Speed and memory", speed.into()),
        card(t, "Prompt blocks on disk", disk.into()),
        card(t, "Telemetry", telemetry.into()),
        card(t, "Administration", admin.into()),
        card(t, "Server", server.into()),
        container(footer).width(Length::Fill)
    ]
    .spacing(14)
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_engine_answer_is_read_with_its_defaults_and_locks() {
        let e = parse_engine(&json!({"values": {"state_cache": false, "kv_ram_mb": 512}, "defaults": {"state_cache": true, "kv_ram_mb": 0, "preemption": "recompute"}, "set_by_environment": ["preemption"]}));
        assert_eq!(e.locked, vec!["preemption"]);
        let s = EngineState { engine: e, ..Default::default() };
        assert_eq!(s.value("state_cache"), json!(false), "a saved value wins");
        assert_eq!(s.value("preemption"), json!("recompute"), "else the default");
        assert!(s.is_default("preemption") && !s.is_default("state_cache") && !s.is_default("kv_ram_mb"));
        assert!(s.locked("preemption") && !s.locked("kv_ram_mb"));
        assert_eq!(s.shown("kv_ram_mb"), "512");
        assert_eq!(parse_engine(&json!({})), EngineData::default());
    }

    #[test]
    fn typed_numbers_are_checked_against_the_lowest_allowed_value() {
        let mb = field("state_cache_mb").unwrap();
        assert_eq!(value_from_text(mb, " 512 ").unwrap(), json!(512));
        assert!(value_from_text(mb, "0").unwrap_err().contains("1 or more"));
        assert!(value_from_text(mb, "lots").is_err() && value_from_text(mb, "").is_err() && value_from_text(mb, "-5").is_err());
        assert_eq!(value_from_text(field("kv_ram_mb").unwrap(), "0").unwrap(), json!(0), "0 switches that one off");
        assert_eq!(value_from_text(field("kv_disk_dir").unwrap(), "  /fast/disk ").unwrap(), json!("/fast/disk"));
        assert_eq!(value_from_text(field("kv_disk_dir").unwrap(), "").unwrap(), json!(""), "an empty folder takes the option back to off");
        assert!(value_from_text(field("preemption").unwrap(), "x").is_err());
    }

    #[test]
    fn a_draft_is_shown_until_it_is_saved() {
        let mut s = EngineState { engine: parse_engine(&json!({"values": {}, "defaults": {"kv_ram_mb": 0}})), ..Default::default() };
        assert_eq!(s.shown("kv_ram_mb"), "0");
        s.drafts.insert("kv_ram_mb".into(), "20".into());
        assert_eq!(s.shown("kv_ram_mb"), "20");
    }

    #[test]
    fn resetting_sends_every_default_but_not_the_locked_ones() {
        let s = EngineState {
            engine: parse_engine(&json!({"values": {"state_cache": false}, "defaults": {"prefix_cache": true, "state_cache": true, "state_cache_mb": 1024, "kv_ram_mb": 0, "preemption": "recompute", "output_cache": "exact", "output_cache_mb": 64, "cache_scope": "global", "kv_disk_dir": ""}, "set_by_environment": ["cache_scope"]})),
            ..Default::default()
        };
        let body = s.reset_body();
        assert_eq!(body["state_cache"], json!(true));
        assert_eq!(body["state_cache_mb"], json!(1024));
        assert_eq!(body["kv_disk_dir"], json!(""));
        assert!(body.get("cache_scope").is_none(), "the environment fixes it");
    }

    #[test]
    fn every_field_has_a_key_the_server_knows() {
        // the keys of hub::ENGINE_OPTIONS in the engine; a typo here would be an option that silently never saves
        const KNOWN: &[&str] = &["output_cache", "output_cache_mb", "output_cache_ttl", "cache_scope", "prefix_cache", "prefix_salt", "preemption", "otel_endpoint", "otel_service", "kv_disk_dir", "kv_disk_mb", "kv_ram_mb", "state_cache", "state_cache_mb"];
        for f in all_fields() {
            assert!(KNOWN.contains(&f.key), "{}", f.key);
        }
        let keys: Vec<&str> = all_fields().map(|f| f.key).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), keys.len(), "no field twice");
    }
}
