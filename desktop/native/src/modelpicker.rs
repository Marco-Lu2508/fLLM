//! The dialog *Choose a model* (`modelpicker.js` of the Studio): a path, a folder or a Hugging Face
//! name is checked by the server (`models/inspect`) before anything is added or loaded, with the
//! system's file dialog (`pick-file`), the list of folders (`fs/browse`) and the buttons
//! *Add to my models* (`models/local`) and *Load* / *Download and load*.

use crate::app::Message;
use crate::models::{post, small_button};
use crate::theme::{self, Tokens};
use iced::widget::{button, column, container, row, scrollable, text, text_input, Space};
use iced::{Alignment, Element, Font, Length};
use serde_json::{json, Value};

// ---- what the page computes without a server (the @pure block, pinned in `modelpicker_fixtures`) ----

/// What a person reads about the multi-token prediction heads of a model.
pub fn mtp_summary(facts: Option<&Facts>) -> String {
    let Some(f) = facts else { return String::new() };
    if !f.speculation_blocked.is_empty() {
        return f.speculation_blocked.clone();
    }
    match f.mtp {
        Some(true) => "Built-in multi-token prediction heads were found. They can make answers faster (Load options, Speculative decoding), on the processor or a CUDA graphics card.".into(),
        Some(false) => "This model has no multi-token prediction heads (MTP), so that option cannot speed it up. Prompt lookup and a draft model work without them. Heads are part of the model files: a model converted without them cannot get them back.".into(),
        None => "Whether it has multi-token prediction heads (MTP) is known once the files are on this computer.".into(),
    }
}

/// `1234567` as `1,234,567` (`toLocaleString('en-US')`).
fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The facts of an inspection as label and value pairs (empty values are left out).
pub fn fact_rows(facts: &Facts) -> Vec<(&'static str, String)> {
    if !facts.ok {
        return Vec::new();
    }
    let gb = |n: u64| if n >= 1_073_741_824 { format!("{:.1} GB", n as f64 / 1_073_741_824.0) } else { format!("{} MB", ((n as f64 / 1_048_576.0).round() as u64).max(1)) };
    let size = if facts.size_bytes > 0 { format!("{}{}", gb(facts.size_bytes), if facts.files > 1 { format!(" in {} files", facts.files) } else { String::new() }) } else { String::new() };
    let fit = match facts.fits {
        Some(true) => "Fits into the free memory",
        Some(false) => "May not fit into the free memory",
        None => "",
    };
    let rows = [
        ("Format", facts.format.clone()),
        ("Size", size),
        ("Architecture", facts.architecture.clone()),
        ("Quantization", facts.quantization.clone()),
        ("Context length", if facts.context_length > 0 { format!("{} tokens", grouped(facts.context_length)) } else { String::new() }),
        ("Memory", fit.to_string()),
    ];
    rows.into_iter().filter(|(_, v)| !v.is_empty()).collect()
}

/// A path that was typed with quotes or around spaces.
pub fn clean_path_text(text: &str) -> String {
    text.trim().trim_matches(|c| c == '"' || c == '\'').trim().to_string()
}

/// `formatBytes` of core.js (binary units, one decimal).
pub fn format_bytes(bytes: f64) -> String {
    if bytes.is_nan() || bytes <= 0.0 {
        return "0 B".into();
    }
    let units = ["B", "KB", "MB", "GB", "TB"];
    let i = ((bytes.ln() / 1024f64.ln()).floor().max(0.0) as usize).min(units.len() - 1);
    let v = bytes / 1024f64.powi(i as i32);
    if i == 0 {
        format!("{v:.0} {}", units[i])
    } else {
        format!("{v:.1} {}", units[i])
    }
}

// ---- what the server says ----------------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Candidate {
    pub label: String,
    pub path: String,
}

/// The answer of `models/inspect`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Facts {
    pub ok: bool,
    /// `file`, `folder`, `repo` or `not_a_model`.
    pub kind: String,
    pub path: String,
    pub title: String,
    pub format: String,
    pub size_bytes: u64,
    pub files: u64,
    pub architecture: String,
    pub quantization: String,
    pub context_length: u64,
    pub fits: Option<bool>,
    pub problem: String,
    pub fix: String,
    pub notes: Vec<String>,
    pub candidates: Vec<Candidate>,
    pub mtp: Option<bool>,
    pub speculation_blocked: String,
}

pub fn parse_facts(v: &Value) -> Facts {
    let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
    let u = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
    Facts {
        ok: v.get("ok").and_then(Value::as_bool).unwrap_or(false),
        kind: s("kind"),
        path: s("path"),
        title: s("title"),
        format: s("format"),
        size_bytes: u("size_bytes"),
        files: u("files"),
        architecture: s("architecture"),
        quantization: s("quantization"),
        context_length: u("context_length"),
        fits: v.get("fits").and_then(Value::as_bool),
        problem: s("problem"),
        fix: s("fix"),
        notes: v.get("notes").and_then(Value::as_array).map(|a| a.iter().filter_map(|n| n.as_str().map(str::to_string)).collect()).unwrap_or_default(),
        candidates: v
            .get("candidates")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|c| Candidate { label: c["label"].as_str().unwrap_or_default().into(), path: c["path"].as_str().unwrap_or_default().into() }).collect())
            .unwrap_or_default(),
        mtp: v.get("mtp").and_then(Value::as_bool),
        speculation_blocked: s("speculation_blocked"),
    }
}

/// A line of the folder list.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub folder: bool,
    pub name: String,
    pub path: String,
    pub model_folder: bool,
    pub format: String,
    pub size_bytes: u64,
}

/// The answer of `fs/browse`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Listing {
    pub path: String,
    pub parent: Option<String>,
    pub roots: Vec<Candidate>,
    pub entries: Vec<Entry>,
    pub truncated: bool,
}

pub fn parse_listing(v: &Value) -> Listing {
    Listing {
        path: v["path"].as_str().unwrap_or_default().into(),
        parent: v.get("parent").and_then(Value::as_str).filter(|p| !p.is_empty()).map(str::to_string),
        roots: v.get("roots").and_then(Value::as_array).map(|a| a.iter().map(|r| Candidate { label: r["label"].as_str().unwrap_or_default().into(), path: r["path"].as_str().unwrap_or_default().into() }).collect()).unwrap_or_default(),
        entries: v
            .get("entries")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|e| Entry {
                        folder: e["kind"].as_str() == Some("folder"),
                        name: e["name"].as_str().unwrap_or_default().into(),
                        path: e["path"].as_str().unwrap_or_default().into(),
                        model_folder: e["model_folder"].as_bool().unwrap_or(false),
                        format: e["format"].as_str().unwrap_or_default().into(),
                        size_bytes: e["size_bytes"].as_u64().unwrap_or(0),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        truncated: v.get("truncated").and_then(Value::as_bool).unwrap_or(false),
    }
}

// ---- the dialog's state --------------------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Picker {
    pub open: bool,
    pub path: String,
    pub facts: Option<Facts>,
    pub busy: bool,
    pub browsing: bool,
    pub listing: Option<Listing>,
    /// What the list of folders says when it has no list (reading, or why it failed).
    pub list_note: String,
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)] // one message at a time, and it is never stored in a list
pub enum MpMsg {
    /// The dialog opens (`fllmChooseModel`); a non-empty text is checked at once.
    Open(String),
    Edit(String),
    /// A file of the list or a candidate: the box is filled and checked at once.
    CheckPath(String),
    Check,
    Checked(String, Result<Facts, String>),
    /// The system's dialog for `file` or `folder`.
    Pick(&'static str),
    Picked(Result<Option<String>, String>),
    Browse,
    Listed(Result<Listing, String>),
    Dir(String),
    UseFolder,
    Add,
    Added(Result<Option<String>, String>),
    Load,
    /// The model was added quietly before it is loaded (its reference, or why that failed).
    LoadAdded(Result<Option<String>, String>),
    /// The download of a Hugging Face name started (its id).
    Pulled(Result<u64, String>),
    Cancel,
}

impl Picker {
    /// The *Add* and *Load* buttons work once the model was found (`mpSetButtons`).
    pub fn can_use(&self) -> bool {
        self.facts.as_ref().is_some_and(|f| f.ok) && !self.busy
    }
    pub fn can_add(&self) -> bool {
        self.can_use() && self.facts.as_ref().is_some_and(|f| f.kind != "repo")
    }
    pub fn load_label(&self) -> &'static str {
        if self.facts.as_ref().is_some_and(|f| f.kind == "repo") {
            "Download and load"
        } else {
            "Load"
        }
    }
}

pub async fn inspect(base: String, path: String) -> Result<Facts, String> {
    // an error body is a finding too, with the words the page gives it
    match post(&base, "/api/studio/models/inspect", json!({ "path": path })).await {
        Ok(v) => Ok(parse_facts(&v)),
        Err(e) if e.starts_with("no server at") => Err(e),
        Err(e) => Ok(Facts { ok: false, kind: "not_a_model".into(), problem: e, fix: "Try again, or restart fLLM if this keeps happening.".into(), ..Default::default() }),
    }
}

/// The system's file or folder dialog (`browse(kind)`): `Ok(None)` when it was closed.
pub async fn pick(base: String, kind: &'static str) -> Result<Option<String>, String> {
    let v = post(&base, "/api/studio/pick-file", json!({ "kind": kind })).await.map_err(|e| if e.is_empty() { "The file dialog could not be opened.".to_string() } else { e })?;
    Ok(v.get("path").and_then(Value::as_str).filter(|p| !p.is_empty()).map(str::to_string))
}

pub async fn browse(base: String, dir: String) -> Result<Listing, String> {
    let v = crate::models::get_json_any(&base, &format!("/api/studio/fs/browse?path={}", urlencode(&dir))).await.map_err(|e| if e.is_empty() { "The folder could not be read.".to_string() } else { e })?;
    if let Some(e) = v.pointer("/error/message").and_then(Value::as_str) {
        return Err(e.to_string());
    }
    Ok(parse_listing(&v))
}

/// `encodeURIComponent`.
pub fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.!~*'()".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `models/local`: the model is added to the list of this computer; the answer is its reference.
pub async fn add_local(base: String, path: String) -> Result<Option<String>, String> {
    let v = post(&base, "/api/studio/models/local", json!({ "path": path })).await.map_err(|e| if e.is_empty() { "The model could not be added.".to_string() } else { e })?;
    Ok(v.get("added").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_str).map(str::to_string))
}

/// `models/pull` for a Hugging Face name: the id of the download, which is waited for.
pub async fn pull_id(base: String, model: String) -> Result<u64, String> {
    let v = post(&base, "/api/studio/models/pull", json!({ "model": model, "quant": "" })).await.map_err(|e| if e.is_empty() { "The download could not be started.".to_string() } else { e })?;
    Ok(v.get("id").and_then(Value::as_u64).unwrap_or(0))
}

// ---- the picture ------------------------------------------------------------------------------------------------

fn field_style<'a>(t: Tokens) -> impl Fn(&iced::Theme, text_input::Status) -> text_input::Style + 'a {
    move |_, status| text_input::Style {
        background: iced::Background::Color(t.surface_2),
        border: iced::Border { radius: 10.0.into(), width: 1.0, color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator } },
        icon: t.muted,
        placeholder: t.muted,
        value: t.fg,
        selection: t.primary_soft,
    }
}

fn entry_button<'a>(t: Tokens, marker: &'a str, name: String, extra: Element<'a, Message>, msg: Message) -> Element<'a, Message> {
    button(row![text(marker).size(13).color(t.muted).width(16), text(name).size(13).font(Font::MONOSPACE).width(Length::Fill), extra].spacing(8).align_y(Alignment::Center))
        .on_press(msg)
        .width(Length::Fill)
        .padding([4, 8])
        .style(theme::ghost_button(t))
        .into()
}

pub fn view<'a>(t: Tokens, p: &'a Picker) -> Element<'a, Message> {
    let mp = |m: MpMsg| Message::Mp(m);
    let mut body = column![
        text("Choose a model").size(16).font(Font { weight: iced::font::Weight::Bold, ..Font::default() }),
        text("A .gguf file (the first part of a split file is enough), a Hugging Face model folder with config.json and .safetensors files, or a Hugging Face name like owner/name or owner/name:Q4_K_M.").size(12).color(t.muted),
        row![
            text_input("Path, folder or owner/name", &p.path).on_input(move |s| Message::Mp(MpMsg::Edit(s))).on_submit(Message::Mp(MpMsg::Check)).font(Font::MONOSPACE).size(13).padding([8, 12]).style(field_style(t)),
            small_button(t, "Check", Some(mp(MpMsg::Check)), false),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
        row![
            small_button(t, "Choose a file…", Some(mp(MpMsg::Pick("model"))), false),
            small_button(t, "Choose a folder…", Some(mp(MpMsg::Pick("folder"))), false),
            button(text("Look through folders").size(12)).on_press(mp(MpMsg::Browse)).padding([4, 8]).style(theme::ghost_button(t)),
        ]
        .spacing(8),
    ]
    .spacing(10);

    if p.browsing {
        let mut list = column![].spacing(2);
        match &p.listing {
            None => list = list.push(text(if p.list_note.is_empty() { "Reading the folder ..." } else { p.list_note.as_str() }).size(12).color(t.muted)),
            Some(l) => {
                let mut roots = row![].spacing(6);
                for r in &l.roots {
                    roots = roots.push(button(text(r.label.as_str()).size(11)).on_press(mp(MpMsg::Dir(r.path.clone()))).padding([2, 6]).style(theme::ghost_button(t)));
                }
                list = list.push(scrollable(roots).direction(scrollable::Direction::Horizontal(scrollable::Scrollbar::new().width(4).scroller_width(4))));
                list = list.push(text(l.path.as_str()).size(12).font(Font::MONOSPACE).color(t.fg_soft));
                if let Some(parent) = &l.parent {
                    list = list.push(entry_button(t, "..", "Up one folder".into(), Space::new().into(), mp(MpMsg::Dir(parent.clone()))));
                }
                for e in &l.entries {
                    list = list.push(if e.folder {
                        let chip: Element<Message> = if e.model_folder { text("model folder").size(11).color(t.primary).into() } else { Space::new().into() };
                        entry_button(t, "/", e.name.clone(), chip, mp(MpMsg::Dir(e.path.clone())))
                    } else {
                        let info: Element<Message> = text(format!("{}, {}", e.format, format_bytes(e.size_bytes as f64))).size(11).color(t.muted).into();
                        entry_button(t, "·", e.name.clone(), info, mp(MpMsg::CheckPath(e.path.clone())))
                    });
                }
                if l.entries.is_empty() {
                    list = list.push(text("No folders or model files here.").size(12).color(t.muted));
                }
                if l.truncated {
                    list = list.push(text("Only the first entries are shown.").size(12).color(t.muted));
                }
                list = list.push(small_button(t, "Use this folder", Some(mp(MpMsg::UseFolder)), false));
            }
        }
        body = body.push(container(scrollable(list).height(Length::Fixed(220.0))).padding(6).style(move |_| container::Style {
            background: Some(iced::Background::Color(t.surface_2)),
            border: iced::Border { width: 1.0, color: t.separator, radius: 8.0.into() },
            ..Default::default()
        }));
    }

    // the result
    if p.busy {
        body = body.push(text("Looking at it ...").size(12).color(t.muted));
    } else if let Some(f) = &p.facts {
        if !f.problem.is_empty() {
            let mut problem = column![text(f.problem.as_str()).size(13).color(t.danger).font(Font { weight: iced::font::Weight::Bold, ..Font::default() })].spacing(2);
            if !f.fix.is_empty() {
                problem = problem.push(text(f.fix.as_str()).size(12).color(t.fg_soft));
            }
            body = body.push(problem);
        }
        if f.ok {
            let mut found = column![text(f.title.as_str()).size(14).font(Font { weight: iced::font::Weight::Bold, ..Font::default() })].spacing(4);
            for (k, v) in fact_rows(f) {
                found = found.push(row![text(k).size(12).color(t.muted).width(110), text(v).size(12).font(Font::MONOSPACE)].spacing(8));
            }
            found = found.push(text(mtp_summary(Some(f))).size(12).color(t.muted));
            for n in &f.notes {
                found = found.push(text(n.as_str()).size(12).color(t.muted));
            }
            body = body.push(found);
        }
        if !f.candidates.is_empty() {
            let mut cands = column![].spacing(2);
            for c in &f.candidates {
                cands = cands.push(button(text(c.label.as_str()).size(12).font(Font::MONOSPACE)).on_press(Message::Mp(MpMsg::CheckPath(c.path.clone()))).padding([4, 8]).style(theme::ghost_button(t)));
            }
            body = body.push(cands);
        }
    }

    let actions = row![
        Space::new().width(Length::Fill),
        button(text("Cancel").size(13)).on_press(mp(MpMsg::Cancel)).padding([6, 12]).style(theme::ghost_button(t)),
        small_button(t, "Add to my models", p.can_add().then(|| mp(MpMsg::Add)), false),
        small_button(t, p.load_label(), p.can_use().then(|| mp(MpMsg::Load)), true),
    ]
    .spacing(8)
    .align_y(Alignment::Center);
    body = body.push(actions);

    let window = container(body).padding(16).width(560).style(theme::card(t, theme::R_LG, false));
    container(iced::widget::opaque(window)).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into()
}
