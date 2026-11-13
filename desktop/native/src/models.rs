//! The Models page: what the server has loaded, what is on this computer, what can be downloaded, and the downloads that are running.
//!
//! The server owns all of it (`/api/studio/models`, `.../load`, `.../unload`, `.../pull`, `.../progress`, `.../downloads`); this file reads
//! those answers into small types, asks for the actions, and draws the page with the same tokens as the chat.

use crate::app::{Message, UI_FONT};
use crate::theme::{self, Tokens};
use iced::widget::{button, column, container, row, scrollable, text, text_input, Space};
use iced::{Alignment, Element, Font, Length, Padding};
use serde_json::{json, Value};
use std::collections::HashMap;

/// `fetch_model` answers with this text when the server runs but has no model loaded (the window shows the Models page then).
pub const NO_MODEL: &str = "no model is loaded";

#[derive(Debug, Clone, PartialEq)]
pub struct Installed {
    pub name: String,
    /// What `models/load` takes: the file for a model from a folder, the name for one the server downloaded itself.
    pub reference: String,
    pub bytes: u64,
    /// Where it came from: `downloaded`, `added`, or the source that found it (LM Studio, Ollama ...).
    pub where_: String,
    /// What the start plan shows for it: the file of a download, the name of a file the user
    /// added, the name a found model has — the `name` argument of the Studio's `add`.
    pub shown: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub quant: String,
    pub bytes: u64,
    pub fits: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatalogEntry {
    pub id: String,
    /// The name people see (`display_name` of the catalog; the id stands in without one).
    pub display_name: String,
    pub default_quant: String,
    pub parameters_b: f64,
    pub context: u64,
    pub vision: bool,
    pub fits: bool,
    pub variants: Vec<Variant>,
    /// What the hub has on disk for this entry (the start plan ranks the model used last).
    pub installed_ref: Option<String>,
}

impl CatalogEntry {
    pub fn bytes_of(&self, quant: &str) -> u64 {
        self.variants.iter().find(|v| v.quant == quant).map_or(0, |v| v.bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Overview {
    pub loaded: Option<String>,
    pub loaded_detail: String,
    pub memory: u64,
    pub installed: Vec<Installed>,
    pub catalog: Vec<CatalogEntry>,
    /// The ids the hub suggests (the welcome tour shows the first four of them as names).
    pub recommended: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Download {
    pub id: u64,
    pub model: String,
    pub quant: String,
    pub state: String,
    pub written: u64,
    pub total: u64,
    pub mb_per_s: f64,
    pub eta_seconds: u64,
    pub error: String,
}

impl Download {
    pub fn active(&self) -> bool {
        matches!(self.state.as_str(), "queued" | "running")
    }
    pub fn ratio(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.written as f64 / self.total as f64).clamp(0.0, 1.0) as f32
        }
    }
}

/// The page's state (the window owns one).
#[derive(Debug, Clone, Default)]
pub struct ModelsState {
    pub overview: Overview,
    pub downloads: Vec<Download>,
    pub filter: String,
    /// The quantization chosen per catalog entry (else its default).
    pub quant: HashMap<String, String>,
    /// A load or unload that is under way (the name shown on the button).
    pub busy: Option<String>,
    pub error: Option<String>,
    pub fetched: bool,
}

impl ModelsState {
    pub fn chosen_quant(&self, entry: &CatalogEntry) -> String {
        self.quant.get(&entry.id).cloned().unwrap_or_else(|| entry.default_quant.clone())
    }

    /// The catalog entries whose name contains every word of the filter (smallest download first would hide the good ones: keep the server's order).
    pub fn visible(&self) -> Vec<&CatalogEntry> {
        let words: Vec<String> = self.filter.split_whitespace().map(str::to_lowercase).collect();
        self.overview.catalog.iter().filter(|e| words.iter().all(|w| e.id.to_lowercase().contains(w))).collect()
    }

    pub fn any_active(&self) -> bool {
        self.downloads.iter().any(Download::active)
    }
}

// ---- reading the answers ---------------------------------------------------------------------------------------------------------

fn text_of(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
}

fn number_of(v: &Value, key: &str) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(0)
}

pub fn parse_overview(v: &Value) -> Overview {
    let loaded = v.pointer("/loaded/name").and_then(Value::as_str).map(str::to_string);
    let loaded_detail = v
        .get("loaded")
        .filter(|l| !l.is_null())
        .map(|l| {
            let parts = [text_of(l, "parameter_size"), text_of(l, "quantization")];
            parts.iter().filter(|p| !p.is_empty()).cloned().collect::<Vec<_>>().join(" · ")
        })
        .unwrap_or_default();
    let mut installed = Vec::new();
    for item in v.get("local").and_then(Value::as_array).into_iter().flatten() {
        // a file from a folder: loaded by its path, shown by its name (or the path when there is none)
        let path = text_of(item, "path");
        let name = { let n = text_of(item, "name"); if n.is_empty() { path.clone() } else { n } };
        installed.push(Installed { name, reference: path.clone(), bytes: number_of(item, "size_bytes"), where_: String::from("added"), shown: { let n = text_of(item, "name"); if n.is_empty() { path.clone() } else { n } } });
    }
    for item in v.get("cached").and_then(Value::as_array).into_iter().flatten() {
        // a download the server keeps: the reference is its name (what people see), the file is
        // beside it (the start plan takes its title from the file — startPlan of the Studio)
        let name = text_of(item, "name");
        let path = text_of(item, "path");
        // the plan shows the file of a download (its name when the hub gives it none)
        let shown = if path.is_empty() { name.clone() } else { path };
        installed.push(Installed { name, reference: text_of(item, "name"), bytes: number_of(item, "size_bytes"), where_: String::from("downloaded"), shown });
    }
    for item in v.get("found").and_then(Value::as_array).into_iter().flatten() {
        let path = text_of(item, "path");
        installed.push(Installed { name: text_of(item, "name"), reference: path, bytes: number_of(item, "size_bytes"), where_: {
            let source = text_of(item, "source");
            if source.is_empty() { String::from("found") } else { source }
        }, shown: text_of(item, "name") });
    }
    installed.retain(|i| !i.name.is_empty() && !i.reference.is_empty());
    let catalog = v
        .get("catalog")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|e| CatalogEntry {
            id: text_of(e, "id"),
            display_name: {
                let name = text_of(e, "display_name");
                if name.is_empty() { text_of(e, "id") } else { name }
            },
            default_quant: if e.get("recommended_quant").and_then(Value::as_str).is_some() { text_of(e, "recommended_quant") } else { text_of(e, "default_quant") },
            parameters_b: e.get("parameters_billion").and_then(Value::as_f64).unwrap_or(0.0),
            context: number_of(e, "context_length"),
            vision: e.get("vision").and_then(Value::as_bool).unwrap_or(false),
            fits: e.get("fits").and_then(Value::as_bool).unwrap_or(true),
            installed_ref: e.pointer("/installed/ref").and_then(Value::as_str).map(str::to_string),
            variants: e
                .get("variants")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|x| Variant { quant: text_of(x, "quant"), bytes: number_of(x, "estimated_bytes"), fits: x.get("fits_memory").and_then(Value::as_bool).unwrap_or(true) })
                .collect(),
        })
        .filter(|e| !e.id.is_empty())
        .collect();
    let recommended = v
        .get("recommended")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    Overview {
        loaded,
        loaded_detail,
        memory: number_of(v, "available_memory_bytes"),
        installed,
        catalog,
        recommended,
    }
}

pub fn parse_downloads(v: &Value) -> Vec<Download> {
    v.get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|d| Download {
            id: number_of(d, "id"),
            model: text_of(d, "model"),
            quant: text_of(d, "quant"),
            state: text_of(d, "state"),
            written: number_of(d, "written"),
            total: number_of(d, "total"),
            mb_per_s: d.get("mb_per_s").and_then(Value::as_f64).unwrap_or(0.0),
            eta_seconds: number_of(d, "eta_seconds"),
            error: text_of(d, "error"),
        })
        .collect()
}

/// "4.9 GB", "420 MB": sizes the way a person reads them (decimal units, like the file managers and the download pages).
pub fn size(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else {
        format!("{} KB", bytes / 1000)
    }
}

pub fn duration(seconds: u64) -> String {
    if seconds >= 3600 {
        format!("{} h {} min", seconds / 3600, (seconds % 3600) / 60)
    } else if seconds >= 60 {
        format!("{} min", seconds.div_ceil(60))
    } else {
        format!("{seconds} s")
    }
}

// ---- asking the server -----------------------------------------------------------------------------------------------------------

async fn get_json(base: &str, path: &str) -> Result<Value, String> {
    reqwest::get(format!("{base}{path}"))
        .await
        .map_err(|e| format!("no server at {base}: {e}"))?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())
}

/// A POST that the server answers with a JSON error body on failure: that message is what the user should read.
async fn post(base: &str, path: &str, body: Value) -> Result<Value, String> {
    let resp = reqwest::Client::new().post(format!("{base}{path}")).json(&body).send().await.map_err(|e| format!("no server at {base}: {e}"))?;
    let status = resp.status();
    let data: Value = resp.json().await.unwrap_or(Value::Null);
    if status.is_success() {
        return Ok(data);
    }
    let message = data.pointer("/error/message").and_then(Value::as_str).or_else(|| data.get("error").and_then(Value::as_str));
    Err(message.map(str::to_string).unwrap_or_else(|| format!("the server answered {status}")))
}

pub async fn fetch_overview(base: String) -> Result<Overview, String> {
    get_json(&base, "/api/studio/models").await.map(|v| parse_overview(&v))
}

/// The hardware of this machine (`GET /api/studio/hardware`) — what the welcome tour says about
/// the computer (`describeComputer` of the Studio).
pub async fn fetch_hardware(base: String) -> Result<crate::tour::Hardware, String> {
    match get_json(&base, "/api/studio/hardware").await {
        Ok(v) => Ok(crate::tour::Hardware::from_json(&v)),
        Err(e) => Err(e),
    }
}

/// What the start screen shows: what the engine is doing right now — the setup state of a server
/// without a model, the model and the counters of its own clock (the hero of `panels.js` reads
/// them from `GET /api/studio/status`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Status {
    pub setup: bool,
    /// The model that loads with the first message (`last_model` of the setup server).
    pub last_model: String,
    pub model_name: String,
    pub parameters: u64,
    pub parameter_size: String,
    pub quantization: String,
    pub max_context: u64,
    pub uptime_secs: f64,
    pub generated_tokens_total: u64,
    pub slots_running: u64,
}

pub fn parse_status(v: &Value) -> Status {
    Status {
        setup: v.get("setup").and_then(Value::as_bool).unwrap_or(false),
        last_model: v.get("last_model").and_then(Value::as_str).unwrap_or_default().to_string(),
        model_name: v.pointer("/model/name").and_then(Value::as_str).unwrap_or_default().to_string(),
        parameters: v.pointer("/model/parameters").and_then(Value::as_u64).unwrap_or(0),
        parameter_size: v.pointer("/model/parameter_size").and_then(Value::as_str).unwrap_or_default().to_string(),
        quantization: v.pointer("/model/quantization").and_then(Value::as_str).unwrap_or_default().to_string(),
        max_context: v.pointer("/model/max_context").and_then(Value::as_u64).unwrap_or(0),
        uptime_secs: v.pointer("/metrics/uptime_secs").and_then(Value::as_f64).unwrap_or(0.0),
        generated_tokens_total: v.pointer("/metrics/generated_tokens_total").and_then(Value::as_u64).unwrap_or(0),
        slots_running: v.pointer("/metrics/slots_running").and_then(Value::as_u64).unwrap_or(0),
    }
}

/// The status of the engine every two seconds, the way the Studio polls it (`refreshStudioStatus`).
pub async fn fetch_status(base: String) -> Result<Status, String> {
    get_json(&base, "/api/studio/status").await.map(|v| parse_status(&v))
}

/// The prices per million tokens the user entered (`GET /api/studio/prices`).
pub async fn fetch_prices(base: String) -> Result<Vec<crate::usage::Price>, String> {
    get_json(&base, "/api/studio/prices").await.map(|v| parse_prices(&v))
}

/// Writes the price rows back (`POST /api/studio/prices`) and returns what the server kept.
pub async fn save_prices(base: String, prices: Vec<crate::usage::Price>) -> Result<Vec<crate::usage::Price>, String> {
    post(&base, "/api/studio/prices", json!({ "prices": prices }))
        .await
        .map(|v| parse_prices(&v))
}

fn parse_prices(v: &Value) -> Vec<crate::usage::Price> {
    v.get("prices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|r| {
            let mut price = crate::usage::Price {
                pattern: text_of(r, "pattern"),
                input_per_million: r.get("input_per_million").and_then(Value::as_f64).unwrap_or(0.0),
                output_per_million: r.get("output_per_million").and_then(Value::as_f64).unwrap_or(0.0),
                note: text_of(r, "note"),
                input_text: String::new(),
                output_text: String::new(),
            };
            price.refresh_text();
            price
        })
        .filter(|p| !p.pattern.is_empty())
        .collect()
}

pub async fn fetch_downloads(base: String) -> Result<Vec<Download>, String> {
    get_json(&base, "/api/studio/models/progress").await.map(|v| parse_downloads(&v))
}

pub async fn load(base: String, reference: String) -> Result<String, String> {
    post(&base, "/api/studio/models/load", json!({ "model": reference })).await.map(|_| reference)
}

pub async fn unload(base: String) -> Result<String, String> {
    post(&base, "/api/studio/models/unload", json!({})).await.map(|_| String::new())
}

pub async fn pull(base: String, model: String, quant: String) -> Result<String, String> {
    post(&base, "/api/studio/models/pull", json!({ "model": model, "quant": quant })).await.map(|_| model)
}

pub async fn download_action(base: String, id: u64, action: &'static str) -> Result<String, String> {
    post(&base, "/api/studio/models/downloads", json!({ "id": id, "action": action })).await.map(|_| String::new())
}

// ---- the page --------------------------------------------------------------------------------------------------------------------

pub(crate) fn bold() -> Font {
    Font { weight: iced::font::Weight::Semibold, ..UI_FONT }
}

pub(crate) fn card<'a>(t: Tokens, title: &'a str, body: Element<'a, Message>) -> Element<'a, Message> {
    container(column![text(title).size(13).font(bold()).color(t.fg_soft), body].spacing(12))
        .padding(16)
        .width(Length::Fill)
        .style(theme::card(t, theme::R_LG, false))
        .into()
}

pub(crate) fn line<'a>(t: Tokens, title: String, detail: String, action: Element<'a, Message>) -> Element<'a, Message> {
    row![
        column![text(title).size(14).font(Font::MONOSPACE), text(detail).size(12).color(t.muted)].spacing(2).width(Length::Fill),
        action
    ]
    .spacing(12)
    .align_y(Alignment::Center)
    .into()
}

pub(crate) fn small_button<'a>(t: Tokens, label: &'a str, message: Option<Message>, primary: bool) -> Element<'a, Message> {
    let b = button(text(label).size(13)).on_press_maybe(message).padding([5, 14]);
    if primary {
        b.style(theme::primary_button(t)).into()
    } else {
        b.style(theme::secondary_button(t)).into()
    }
}

/// A hairline between two rows of a list.
pub(crate) fn rule<'a>(t: Tokens) -> Element<'a, Message> {
    container(Space::new()).width(Length::Fill).height(1).style(move |_| container::Style { background: Some(iced::Background::Color(t.separator)), ..Default::default() }).into()
}

fn quant_chip<'a>(t: Tokens, entry: &'a str, quant: &'a str, selected: bool, fits: bool) -> Element<'a, Message> {
    button(text(quant).size(11))
        .on_press(Message::PickQuant(entry.to_string(), quant.to_string()))
        .padding([2, 8])
        .style(move |_, status| {
            let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
            button::Style {
                background: Some(iced::Background::Color(if selected { t.primary_soft } else if hovered { t.fill_hover } else { t.fill })),
                text_color: if selected { t.primary } else if fits { t.fg_soft } else { t.muted },
                border: iced::border::rounded(999),
                ..Default::default()
            }
        })
        .into()
}

pub fn page<'a>(s: &'a ModelsState, t: Tokens, nothing_loaded: bool) -> Element<'a, Message> {
    let mut content = column![container(text("Models").size(26).font(bold())).padding(Padding { top: 14.0, right: 0.0, bottom: 4.0, left: 2.0 })].spacing(14);
    if nothing_loaded {
        content = content.push(text("No model is loaded. Pick one below to start chatting.").size(14).color(t.fg_soft));
    }
    if let Some(error) = &s.error {
        content = content.push(text(error.as_str()).size(13).color(t.danger));
    }

    // what runs now
    let loaded: Element<'_, Message> = match &s.overview.loaded {
        Some(name) => line(
            t,
            name.clone(),
            if s.overview.loaded_detail.is_empty() { "loaded".to_string() } else { s.overview.loaded_detail.clone() },
            small_button(t, if s.busy.is_some() { "Working…" } else { "Unload" }, s.busy.is_none().then_some(Message::UnloadModel), false),
        ),
        None => text("Nothing is loaded.").size(13).color(t.muted).into(),
    };
    content = content.push(card(t, "Loaded", loaded));

    // downloads
    if !s.downloads.is_empty() {
        let mut list = column![].spacing(12);
        for (i, d) in s.downloads.iter().enumerate() {
            if i > 0 {
                list = list.push(rule(t));
            }
            let label = match d.state.as_str() {
                "queued" => "Waiting".to_string(),
                "running" if d.total > 0 => format!("{:.0}% · {} of {} · {:.1} MB/s{}", d.ratio() * 100.0, size(d.written), size(d.total), d.mb_per_s, if d.eta_seconds > 0 { format!(" · about {}", duration(d.eta_seconds)) } else { String::new() }),
                "running" => "Starting…".to_string(),
                "paused" => "Paused".to_string(),
                "failed" => format!("Failed: {}", d.error),
                "done" => "Done".to_string(),
                other => other.to_string(),
            };
            let mut actions = row![].spacing(4);
            if d.active() {
                actions = actions.push(small_button(t, "Pause", Some(Message::DownloadAction(d.id, "pause")), false));
            }
            if d.state == "paused" || d.state == "failed" {
                actions = actions.push(small_button(t, if d.state == "failed" { "Try again" } else { "Resume" }, Some(Message::DownloadAction(d.id, "resume")), false));
            }
            let end = if d.state == "done" || d.state == "failed" { ("Remove", "clear") } else { ("Cancel", "cancel") };
            actions = actions.push(small_button(t, end.0, Some(Message::DownloadAction(d.id, end.1)), false));
            let title = if d.quant.is_empty() { d.model.clone() } else { format!("{} · {}", d.model, d.quant) };
            let mut item = column![line(t, title, label, actions.into())].spacing(8);
            if d.state == "running" && d.total > 0 {
                let done = ((d.ratio() * 1000.0) as u16).clamp(1, 1000);
                item = item.push(
                    container(row![
                        container(Space::new()).width(Length::FillPortion(done)).height(4).style(move |_| container::Style { background: Some(iced::Background::Color(t.primary)), border: iced::border::rounded(2), ..Default::default() }),
                        Space::new().width(Length::FillPortion(1000 - done)),
                    ])
                    .width(Length::Fill)
                    .style(move |_| container::Style { background: Some(iced::Background::Color(t.fill_hover)), border: iced::border::rounded(2), ..Default::default() }),
                );
            }
            list = list.push(item);
        }
        content = content.push(card(t, "Downloads", list.into()));
    }

    // on this computer
    let mut here = column![].spacing(10);
    if s.overview.installed.is_empty() {
        here = here.push(text(if s.fetched { "No models downloaded yet." } else { "Looking…" }).size(13).color(t.muted));
    }
    for (i, m) in s.overview.installed.iter().enumerate() {
        if i > 0 {
            here = here.push(rule(t));
        }
        let is_loaded = s.overview.loaded.as_deref() == Some(m.name.as_str());
        let action = if is_loaded {
            text("Loaded").size(12).color(t.ok).into()
        } else {
            small_button(t, if s.busy.as_deref() == Some(m.reference.as_str()) { "Loading…" } else { "Load" }, s.busy.is_none().then(|| Message::LoadModel(m.reference.clone())), true)
        };
        here = here.push(line(t, m.name.clone(), size(m.bytes), action));
    }
    content = content.push(card(t, "On this computer", here.into()));

    // the catalog
    let search = text_input("Search the catalog", &s.filter)
        .on_input(Message::Filter)
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
    let mut shop = column![search].spacing(12);
    let visible = s.visible();
    if visible.is_empty() {
        shop = shop.push(text(if s.fetched { "Nothing matches." } else { "Looking…" }).size(13).color(t.muted));
    }
    for (i, e) in visible.into_iter().take(40).enumerate() {
        if i > 0 {
            shop = shop.push(rule(t));
        }
        let quant = s.chosen_quant(e);
        let bytes = e.bytes_of(&quant);
        let downloading = s.downloads.iter().any(|d| d.active() && d.model == e.id);
        let installed = s.overview.installed.iter().any(|i| i.name.to_lowercase().starts_with(&e.id.to_lowercase()));
        let fits = e.variants.iter().find(|v| v.quant == quant).map_or(e.fits, |v| v.fits);
        let mut facts = format!("{} · {} · {} tokens of context", if e.parameters_b > 0.0 { format!("{:.1}B", e.parameters_b) } else { "?".into() }, size(bytes), crate::app::group(e.context));
        if e.vision {
            facts.push_str(" · sees images");
        }
        facts.push_str(if fits { " · fits in memory" } else { " · too big for this computer" });
        let mut chips = row![].spacing(4);
        for v in e.variants.iter().take(6) {
            chips = chips.push(quant_chip(t, &e.id, &v.quant, v.quant == quant, v.fits));
        }
        let action = small_button(
            t,
            if downloading { "Downloading…" } else if installed { "Download again" } else { "Download" },
            (!downloading).then(|| Message::Pull(e.id.clone())),
            !installed,
        );
        shop = shop.push(column![line(t, e.id.clone(), facts, action), chips].spacing(6));
    }
    content = content.push(card(t, "Catalog", shop.into()));

    let inner = container(content).max_width(780).width(Length::Fill).padding(Padding { top: 8.0, right: 24.0, bottom: 32.0, left: 24.0 });
    scrollable(container(inner).center_x(Length::Fill).width(Length::Fill)).height(Length::Fill).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Value {
        json!({
            "available_memory_bytes": 8_309_764_096u64,
            "loaded": {"name": "qwen2.5-0.5b-instruct", "parameter_size": "630M", "quantization": "Q4_0"},
            "local": [{"name": "my-model.gguf", "path": "/models/my-model.gguf", "size_bytes": 2_000_000_000u64}],
            "cached": [{"name": "qwen3.5:4b", "path": "/cache/qwen3.5-4b.gguf", "size_bytes": 2_800_000_000u64}, {"name": "", "path": "/x"}],
            "catalog": [
                {"id": "aya-expanse:8b", "recommended_quant": "Q4_K_M", "default_quant": "Q2_K", "parameters_billion": 8.0, "context_length": 32768, "vision": false, "fits": true,
                 "variants": [{"quant": "Q8_0", "estimated_bytes": 8_908_435_456u64, "fits_memory": false}, {"quant": "Q4_K_M", "estimated_bytes": 4_908_435_456u64, "fits_memory": true}]},
                {"id": "gemma3:4b", "default_quant": "Q4_K_M", "vision": true},
                {"display_name": "no id"}
            ]
        })
    }

    #[test]
    fn the_overview_reads_loaded_installed_and_catalog() {
        let o = parse_overview(&sample());
        assert_eq!(o.loaded.as_deref(), Some("qwen2.5-0.5b-instruct"));
        assert_eq!(o.loaded_detail, "630M · Q4_0");
        assert_eq!(o.memory, 8_309_764_096);
        // a model from a folder is loaded by its file; a download is named what the hub calls it and
        // keeps its file beside it for the start plan; an entry without a name is dropped
        assert_eq!(o.installed.iter().map(|i| (i.name.as_str(), i.reference.as_str())).collect::<Vec<_>>(), vec![("my-model.gguf", "/models/my-model.gguf"), ("qwen3.5:4b", "qwen3.5:4b")]);
        assert_eq!(o.installed[1].shown, "/cache/qwen3.5-4b.gguf", "a download is titled by its file (startPlan)");
        assert_eq!(o.installed[1].where_, "downloaded", "a download says where it came from");
        assert_eq!(o.installed[0].where_, "added", "a file from a folder says so too");
        assert_eq!(o.catalog.len(), 2, "an entry without an id is dropped");
        assert_eq!(o.catalog[0].default_quant, "Q4_K_M", "the recommended quantization wins over the plain default");
        assert_eq!(o.catalog[0].bytes_of("Q8_0"), 8_908_435_456);
        assert!(!o.catalog[0].variants[0].fits && o.catalog[0].variants[1].fits);
        assert!(o.catalog[1].vision && o.catalog[1].fits, "missing fields get harmless defaults");
    }

    #[test]
    fn nothing_loaded_is_not_an_error() {
        let o = parse_overview(&json!({"loaded": null, "local": [], "cached": [], "catalog": []}));
        assert_eq!((o.loaded, o.loaded_detail.as_str(), o.installed.len()), (None, "", 0));
        assert_eq!(parse_overview(&json!({})), Overview::default());
    }

    #[test]
    fn the_filter_needs_every_word_and_ignores_case() {
        let mut s = ModelsState { overview: parse_overview(&sample()), ..Default::default() };
        assert_eq!(s.visible().len(), 2);
        s.filter = "AYA 8b".into();
        assert_eq!(s.visible().iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), vec!["aya-expanse:8b"]);
        s.filter = "aya gemma".into();
        assert!(s.visible().is_empty());
    }

    #[test]
    fn a_chosen_quantization_replaces_the_default_for_that_model_only() {
        let mut s = ModelsState { overview: parse_overview(&sample()), ..Default::default() };
        let (aya, gemma) = (s.overview.catalog[0].clone(), s.overview.catalog[1].clone());
        assert_eq!(s.chosen_quant(&aya), "Q4_K_M");
        s.quant.insert("aya-expanse:8b".into(), "Q8_0".into());
        assert_eq!((s.chosen_quant(&aya).as_str(), s.chosen_quant(&gemma).as_str()), ("Q8_0", "Q4_K_M"));
    }

    #[test]
    fn downloads_report_progress_and_which_ones_are_alive() {
        let d = parse_downloads(&json!({"active": true, "downloads": [
            {"id": 3, "model": "qwen3.5:4b", "quant": "Q4_K_M", "state": "running", "written": 500, "total": 2000, "mb_per_s": 41.5, "eta_seconds": 75},
            {"id": 4, "model": "x", "state": "failed", "error": "disk full"},
            {"id": 5, "model": "y", "state": "paused", "written": 9, "total": 0}
        ]}));
        assert_eq!(d.len(), 3);
        assert!((d[0].ratio() - 0.25).abs() < 1e-6 && d[0].active());
        assert!(!d[1].active() && d[1].error == "disk full" && d[1].ratio() == 0.0);
        assert!(!d[2].active() && d[2].ratio() == 0.0, "a paused download with an unknown total shows no bar");
        let s = ModelsState { downloads: d, ..Default::default() };
        assert!(s.any_active());
        assert!(!ModelsState::default().any_active());
    }

    #[test]
    fn sizes_and_times_read_like_a_download_page() {
        assert_eq!((size(4_908_435_456).as_str(), size(428_730_208).as_str(), size(12_000).as_str()), ("4.9 GB", "429 MB", "12 KB"));
        assert_eq!((duration(45).as_str(), duration(61).as_str(), duration(3720).as_str()), ("45 s", "2 min", "1 h 2 min"));
    }
}
