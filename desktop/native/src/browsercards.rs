//! Row 85, Settings > Browser: the three cards `browser.js` adds to the Settings (*Browser for the
//! coding agent*, *How it runs*, *Safety*) — the switch that lets the agent use a real browser, the
//! way it runs, the browser program, and what is refused.
//!
//! `BrowserPane` is what the page keeps in its fields; `tests/browser_fixtures.rs` runs the page's own
//! script and compares the words and the requests step by step.

use crate::app::Message;
use crate::models::card;
use crate::theme::{self, Tokens};
use iced::widget::{button, checkbox, column, container, pick_list, row, text, text_input, Space};
use iced::{Alignment, Element, Font, Length, Task};
use serde_json::{json, Value};

pub const ENABLE_TITLE: &str = "Browser for the coding agent";
pub const ENABLE_TEXT: &str = "With this on, the agent on the Code page can open a web page in a real browser (Chrome, Chromium, Edge or Brave on this computer), read it, click, type, scroll and take screenshots, for example to test the app it is building on a local dev server. You see the same page live in the Browser panel of the Code page and can take over, or leave comments on elements. Every browser action asks first, like the other network tools. It is off until you turn it on.";
pub const ENABLE_LABEL: &str = "Let the agent use a browser";
pub const RUN_TITLE: &str = "How it runs";
pub const SHOW_LABEL: &str = "Show the browser window";
pub const SHOW_TEXT: &str = "Off: the browser runs without a window (headless) and you watch it in the panel. On: a real window opens on this computer, for example to log in to a site once. Needs a display (not for servers and containers).";
pub const KEEP_LABEL: &str = "Keep logins and cookies between sessions";
pub const KEEP_TEXT: &str = "The browser uses its own profile folder inside fLLM's data folder, never the profile of your own browser. Off: every start uses a fresh, throw-away profile.";
pub const LOCAL_LABEL: &str = "Local addresses";
pub const LOCAL_TEXT: &str = "Addresses on this computer or your network can hold things that must not be reached by accident, so they are refused unless allowed here. Link-local and cloud-metadata addresses (169.254.x.x) are never reachable. A page from the internet can never load local addresses, whatever you choose.";
pub const EXE_LABEL: &str = "Browser program (optional)";
pub const EXE_PLACEHOLDER: &str = "Leave empty to look for Chrome, Edge, Brave or Chromium";
pub const EXE_TEXT: &str = "A path or a program name. The environment variable FLLM_BROWSER wins over this field.";
pub const SAFETY_TITLE: &str = "Safety";
pub const SAFETY_TEXT: &str = "Only http and https pages can be opened (no files, no browser-internal pages). Downloads, camera, microphone, location and notifications are refused. The agent's browser can hold logged-in sessions, so its controls are only reachable from this computer, even when the server has an API key. If you log in to a site in the shown window, the agent can use that session: only do it for sites you are willing to let the agent act on.";
pub const CONFIRM_ENABLE: &str = "Let the agent open web pages in a real browser on this computer? Each browser action still asks first. Pages are limited to http and https; local addresses follow the setting below.";
const NOT_UP: &str = "The browser settings are available once a model is loaded (the coding agent starts with a model).";

/// The three choices of *Local addresses*: the value the server keeps and the words of the box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local(pub &'static str, pub &'static str);

impl std::fmt::Display for Local {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.1)
    }
}

pub const LOCAL_CHOICES: [Local; 3] = [
    Local("auto", "Only localhost, while a workspace is open (default)"),
    Local("off", "None"),
    Local("all", "Localhost and the local network (10.x, 192.168.x ...)"),
];

/// What reading the settings came to (`load()`).
#[derive(Debug, Clone)]
pub enum LoadResult {
    /// The server answered with the settings.
    Ok(Value),
    /// The server answered with an error: its body.
    Refused(Value),
    Failed(String),
}

/// What saving came to (`save(body)`).
#[derive(Debug, Clone)]
pub enum SaveResult {
    Ok(Value),
    Refused(Value),
    Failed(String),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BrowserPane {
    pub enabled: bool,
    pub show: bool,
    pub keep: bool,
    pub local: String,
    pub exe: String,
    /// *Browser found: …* or the reason there is none, and whether it is drawn as a problem.
    pub found: String,
    pub found_bad: bool,
    pub profile: String,
    pub message: String,
    /// The question before the browser is switched on.
    pub confirming: bool,
    /// A save went through: the panel of the Code page reads its status again and the tools are asked for.
    pub changed: bool,
}

#[derive(Debug, Clone)]
pub enum BpMsg {
    Loaded(LoadResult),
    Show(bool),
    Keep(bool),
    Local(Local),
    Exe(String),
    Save,
    Enable(bool),
    EnableOk,
    EnableCancel,
    Saved(SaveResult),
}

impl BrowserPane {
    /// `show(data)`: what the server says goes into the boxes.
    pub fn show_data(&mut self, data: &Value) {
        let s = data.get("settings").cloned().unwrap_or(Value::Null);
        let flag = |k: &str| s.get(k).is_some_and(|x| crate::jobs::truthy(Some(x)));
        let text = |o: &Value, k: &str| crate::computer::text_or_empty(o.get(k));
        self.enabled = flag("enabled");
        self.show = flag("show_window");
        self.keep = flag("persistent_profile");
        self.local = text(&s, "local_addresses");
        self.exe = text(&s, "executable");
        match data.get("browser").filter(|b| crate::jobs::truthy(Some(b))) {
            Some(b) => {
                let env = if data.get("env_browser").is_some_and(|x| crate::jobs::truthy(Some(x))) { " (from FLLM_BROWSER)" } else { "" };
                self.found = format!("Browser found: {}{env}", crate::computer::js_string(b));
                self.found_bad = false;
            }
            None => {
                self.found = format!("No browser found. {}", crate::computer::text_or_empty(data.get("browser_error").filter(|e| crate::jobs::truthy(Some(e)))));
                self.found_bad = true;
            }
        }
        self.profile = format!("Profile folder: {}", crate::computer::js_string(data.get("profile").unwrap_or(&Value::Null)));
    }

    /// `load()`.
    pub fn loaded(&mut self, result: LoadResult) {
        match result {
            LoadResult::Ok(data) => self.show_data(&data),
            LoadResult::Refused(body) => {
                self.found = body.pointer("/error/message").and_then(Value::as_str).filter(|m| !m.is_empty()).unwrap_or(NOT_UP).to_string();
                self.found_bad = false;
            }
            LoadResult::Failed(e) => {
                self.found = format!("Could not read the settings: {e}");
                self.found_bad = false;
            }
        }
    }

    /// The body of *Save*.
    pub fn save_body(&self) -> Value {
        json!({ "show_window": self.show, "persistent_profile": self.keep, "local_addresses": self.local, "executable": self.exe.trim() })
    }

    /// What saving answered: the boxes follow the server, the line says how it went.
    pub fn saved(&mut self, result: SaveResult) {
        match result {
            SaveResult::Ok(data) => {
                self.show_data(&data);
                self.message = "Saved.".into();
                self.changed = true;
            }
            SaveResult::Refused(body) => {
                self.message = body.pointer("/error/message").and_then(Value::as_str).filter(|m| !m.is_empty()).unwrap_or("Could not save it.").to_string();
            }
            SaveResult::Failed(e) => self.message = format!("Could not save it: {e}"),
        }
    }

    pub fn update(&mut self, m: BpMsg, base: &str) -> Task<Message> {
        match m {
            BpMsg::Loaded(r) => {
                self.loaded(r);
                Task::none()
            }
            BpMsg::Show(on) => {
                self.show = on;
                Task::none()
            }
            BpMsg::Keep(on) => {
                self.keep = on;
                Task::none()
            }
            BpMsg::Local(l) => {
                self.local = l.0.to_string();
                Task::none()
            }
            BpMsg::Exe(v) => {
                self.exe = v;
                Task::none()
            }
            BpMsg::Save => self.save(base, self.save_body()),
            // switching it on asks first; switching it off does not
            BpMsg::Enable(true) => {
                self.enabled = true;
                self.confirming = true;
                Task::none()
            }
            BpMsg::Enable(false) => {
                self.enabled = false;
                self.save(base, json!({ "enabled": false }))
            }
            BpMsg::EnableOk => {
                self.confirming = false;
                self.save(base, json!({ "enabled": true }))
            }
            BpMsg::EnableCancel => {
                self.confirming = false;
                self.enabled = false;
                Task::none()
            }
            BpMsg::Saved(r) => {
                self.saved(r);
                Task::none()
            }
        }
    }

    fn save(&self, base: &str, body: Value) -> Task<Message> {
        let base = base.to_string();
        Task::perform(async move { save_settings(&base, body).await }, |r| Message::Bp(BpMsg::Saved(r)))
    }

    /// `load()` as a task (the pane opens, the window starts).
    pub fn refresh(base: &str) -> Task<Message> {
        let base = base.to_string();
        Task::perform(async move { load_settings(&base).await }, |r| Message::Bp(BpMsg::Loaded(r)))
    }
}

async fn request(base: &str, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<(bool, Value), String> {
    let mut req = reqwest::Client::new().request(method, format!("{base}{path}"));
    if let Some(b) = body {
        req = req.json(&b);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    let ok = resp.status().is_success();
    Ok((ok, resp.json().await.unwrap_or(Value::Null)))
}

pub async fn load_settings(base: &str) -> LoadResult {
    match request(base, reqwest::Method::GET, "/api/studio/agent/browser", None).await {
        Ok((true, v)) => LoadResult::Ok(v),
        Ok((false, v)) => LoadResult::Refused(v),
        Err(e) => LoadResult::Failed(e),
    }
}

pub async fn save_settings(base: &str, body: Value) -> SaveResult {
    match request(base, reqwest::Method::POST, "/api/studio/agent/browser/settings", Some(body)).await {
        Ok((true, v)) => SaveResult::Ok(v),
        Ok((false, v)) => SaveResult::Refused(v),
        Err(e) => SaveResult::Failed(e),
    }
}

fn semibold() -> Font {
    Font { weight: iced::font::Weight::Semibold, ..Font::default() }
}

fn line<'a>(t: Tokens, s: &'a str) -> Element<'a, Message> {
    text(s).size(12).color(t.fg_soft).into()
}

/// The three cards.
pub fn view<'a>(t: Tokens, pane: &'a BrowserPane) -> Element<'a, Message> {
    let found: Element<'a, Message> = if pane.found.is_empty() {
        Space::new().into()
    } else {
        text(pane.found.as_str()).size(11).font(Font::MONOSPACE).color(if pane.found_bad { t.danger } else { t.fg_soft }).into()
    };
    let first = column![
        line(t, ENABLE_TEXT),
        checkbox(pane.enabled).label(ENABLE_LABEL).on_toggle(|on| Message::Bp(BpMsg::Enable(on))).size(16).text_size(13),
        found,
    ]
    .spacing(10);
    let selected = LOCAL_CHOICES.iter().find(|l| l.0 == pane.local).cloned();
    let second = column![
        checkbox(pane.show).label(SHOW_LABEL).on_toggle(|on| Message::Bp(BpMsg::Show(on))).size(16).text_size(13),
        container(line(t, SHOW_TEXT)).padding(iced::Padding { top: 0.0, right: 0.0, bottom: 4.0, left: 24.0 }),
        checkbox(pane.keep).label(KEEP_LABEL).on_toggle(|on| Message::Bp(BpMsg::Keep(on))).size(16).text_size(13),
        container(line(t, KEEP_TEXT)).padding(iced::Padding { top: 0.0, right: 0.0, bottom: 4.0, left: 24.0 }),
        text(LOCAL_LABEL).size(13).color(t.fg).font(semibold()),
        pick_list(LOCAL_CHOICES.to_vec(), selected, |l| Message::Bp(BpMsg::Local(l))).text_size(13).width(Length::Fill),
        line(t, LOCAL_TEXT),
        text(EXE_LABEL).size(13).color(t.fg).font(semibold()),
        text_input(EXE_PLACEHOLDER, &pane.exe).on_input(|v| Message::Bp(BpMsg::Exe(v))).on_submit(Message::Bp(BpMsg::Save)).size(13).padding([6, 10]).font(Font::MONOSPACE),
        line(t, EXE_TEXT),
        row![
            button(text("Save").size(13)).on_press(Message::Bp(BpMsg::Save)).padding([5, 16]).style(theme::primary_button(t)),
            text(pane.message.as_str()).size(11).font(Font::MONOSPACE).color(t.fg_soft),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    ]
    .spacing(8);
    let third = column![line(t, SAFETY_TEXT), text(pane.profile.as_str()).size(11).font(Font::MONOSPACE).color(t.fg_soft)].spacing(6);
    column![card(t, ENABLE_TITLE, first.into()), card(t, RUN_TITLE, second.into()), card(t, SAFETY_TITLE, third.into())].spacing(12).into()
}

/// The question before the browser is switched on (`confirmDialog` of the page).
pub fn dialog_window<'a>(t: Tokens) -> Element<'a, Message> {
    let body = column![
        text("Let the agent use a browser?").size(16).font(crate::models::bold()),
        text(CONFIRM_ENABLE).size(13).color(t.fg_soft),
        row![
            Space::new().width(Length::Fill),
            button(text("Cancel").size(13)).on_press(Message::Bp(BpMsg::EnableCancel)).padding([6, 12]).style(theme::ghost_button(t)),
            button(text("OK").size(13)).on_press(Message::Bp(BpMsg::EnableOk)).padding([6, 12]).style(theme::primary_button(t)),
        ]
        .spacing(8),
    ]
    .spacing(14);
    container(container(body).padding(16).width(460).style(theme::card(t, theme::R_LG, false))).center_x(Length::Fill).center_y(Length::Fill).into()
}
