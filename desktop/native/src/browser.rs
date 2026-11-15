//! Row 85 of PARITY.md: the *Browser* panel of the Code page (`browser.js`) — the live view of the
//! page the coding agent works with. The server drives a real Chromium-family browser; the panel
//! shows its page as a stream of screenshots, forwards the user's clicks, typing and scrolling, and
//! lets the user comment on elements (the comments go with the next message to the agent).
//!
//! This file is the state machine of that panel, without a window: what the page keeps in `br`, and
//! what it does on every event. `tests/browser_fixtures.rs` runs the page's own `browser.js` in a vm
//! (`tests/fixtures/browser_fixtures.js`) and compares what each step leaves behind. The window
//! (`code.rs`) holds a [`Panel`], asks it what to send and draws what it says.

use crate::computer::{js_string, slice_units};
use serde_json::{json, Value};

/// What `GET /api/studio/agent/browser` says (`br.status`).
#[derive(Debug, Clone, PartialEq)]
pub struct Status {
    pub enabled: bool,
    pub show_window: bool,
    pub persistent_profile: bool,
    pub local_addresses: String,
    pub executable: String,
    /// The browser program that was found (`status.browser`), `None` when there is none.
    pub browser: Option<String>,
    pub browser_error: String,
    pub env_browser: bool,
    pub running: bool,
    pub url: String,
    pub profile: String,
}

impl Status {
    pub fn parse(v: &Value) -> Status {
        let s = v.get("settings").cloned().unwrap_or(Value::Null);
        let flag = |o: &Value, k: &str| o.get(k).is_some_and(|x| crate::jobs::truthy(Some(x)));
        let text = |o: &Value, k: &str| o.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
        Status {
            enabled: flag(&s, "enabled"),
            show_window: flag(&s, "show_window"),
            persistent_profile: flag(&s, "persistent_profile"),
            local_addresses: text(&s, "local_addresses"),
            executable: text(&s, "executable"),
            browser: v.get("browser").and_then(Value::as_str).filter(|b| !b.is_empty()).map(str::to_string),
            browser_error: text(v, "browser_error"),
            env_browser: flag(v, "env_browser"),
            running: flag(v, "running"),
            url: text(v, "url"),
            profile: text(v, "profile"),
        }
    }
}

/// The element the user is commenting on (`br.draft`: what `locate` answered).
#[derive(Debug, Clone, PartialEq)]
pub struct Draft(pub Value);

/// A comment waiting for the next message (`br.comments[i]`).
#[derive(Debug, Clone, PartialEq)]
pub struct Comment {
    pub reference: String,
    pub role: String,
    pub name: String,
    pub tag: String,
    pub text: String,
    pub url: String,
    pub note: String,
}

/// What the panel asks the server to do (`POST /api/studio/agent/browser/input`).
#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    Navigate(String),
    Back,
    Forward,
    Reload,
    Close,
    Click { x: f64, y: f64 },
    Dblclick { x: f64, y: f64 },
    Locate { x: f64, y: f64 },
    Wheel { x: f64, y: f64, dx: f64, dy: f64 },
    Key(String),
    Text(String),
}

impl Input {
    /// The body of the request.
    pub fn body(&self) -> Value {
        match self {
            Input::Navigate(url) => json!({ "type": "navigate", "url": url }),
            Input::Back => json!({ "type": "back" }),
            Input::Forward => json!({ "type": "forward" }),
            Input::Reload => json!({ "type": "reload" }),
            Input::Close => json!({ "type": "close" }),
            Input::Click { x, y } => json!({ "type": "click", "x": x, "y": y }),
            Input::Dblclick { x, y } => json!({ "type": "dblclick", "x": x, "y": y }),
            Input::Locate { x, y } => json!({ "type": "locate", "x": x, "y": y }),
            Input::Wheel { x, y, dx, dy } => json!({ "type": "wheel", "x": x, "y": y, "dx": dx, "dy": dy }),
            Input::Key(key) => json!({ "type": "key", "key": key }),
            Input::Text(text) => json!({ "type": "text", "text": text }),
        }
    }
}

/// The place of the stage on the screen (`getBoundingClientRect` of the picture).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub left: f64,
    pub top: f64,
    pub width: f64,
    pub height: f64,
}

/// The wheel turns that wait for their 50 ms (`br.wheel`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Wheel {
    pub dx: f64,
    pub dy: f64,
    pub x: f64,
    pub y: f64,
    pub armed: bool,
}

/// The wheel of the mouse as the window reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WheelUnit {
    Pixels,
    Lines,
    Pages,
}

/// One key of the keyboard as `keydown` of the page sees it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeyPress {
    pub key: String,
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    pub meta: bool,
    pub composing: bool,
}

/// `br`: everything the panel keeps.
#[derive(Debug, Clone, PartialEq)]
pub struct Panel {
    pub open: bool,
    pub status: Option<Status>,
    pub running: bool,
    /// The key of the last picture (`br.key`): the server answers only when it changed.
    pub key: String,
    pub view: (f64, f64),
    pub url: String,
    pub title: String,
    pub commenting: bool,
    pub comments: Vec<Comment>,
    pub draft: Option<Draft>,
    /// A page that cannot be drawn: its address (`br.undrawable`), `Some("")` when it has none.
    pub undrawable: Option<String>,
    /// The picture the stage shows (`img.src`).
    pub image: Option<String>,
    /// The text of the address box, and whether it has the focus (a frame does not overwrite it then).
    pub url_input: String,
    pub url_focused: bool,
    pub ticks: u32,
    pub wheel: Wheel,
    /// A frame is on its way (`br.pending`).
    pub pending: bool,
    /// `windows`, `macos` or anything else: the help for a missing browser differs.
    pub os: String,
}

impl Default for Panel {
    fn default() -> Panel {
        Panel {
            open: false,
            status: None,
            running: false,
            key: String::new(),
            view: (1280.0, 800.0),
            url: String::new(),
            title: String::new(),
            commenting: false,
            comments: Vec::new(),
            draft: None,
            undrawable: None,
            image: None,
            url_input: String::new(),
            url_focused: false,
            ticks: 0,
            wheel: Wheel::default(),
            pending: false,
            os: std::env::consts::OS.to_string(),
        }
    }
}

const KEY_NAMES: [&str; 14] = ["Enter", "Tab", "Escape", "Backspace", "Delete", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown", "Insert"];

fn collapse(text: &str) -> String {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"\s+").expect("whitespace")).replace_all(text, " ").into_owned()
}

/// `String(v ?? '')` for the members of what `locate` answered.
fn member(v: &Value, k: &str) -> String {
    crate::computer::text_or_empty(v.get(k))
}

fn show_member(v: &Value, k: &str) -> String {
    v.get(k).map_or_else(|| "undefined".to_string(), js_string)
}

/// What the stage says when there is no picture (`emptyHtml`): the words, and the button under them.
#[derive(Debug, Clone, PartialEq)]
pub struct Empty {
    pub text: String,
    /// `Open Browser settings`, and whether it is the primary button.
    pub button: Option<(&'static str, bool)>,
}

/// The panel as it is drawn (`renderState`, `renderComments`, `renderNote`).
#[derive(Debug, Clone, PartialEq)]
pub struct PanelView {
    pub state: String,
    pub empty: Empty,
    pub show_image: bool,
    pub live_dot: bool,
    pub back_enabled: bool,
    pub url_enabled: bool,
    pub foot: String,
    pub commenting: bool,
    pub url_input: String,
}

impl Panel {
    pub fn new(os: &str) -> Panel {
        Panel { os: os.to_string(), ..Default::default() }
    }

    fn enabled(&self) -> bool {
        self.status.as_ref().is_some_and(|s| s.enabled)
    }

    // ---- what the panel says ------------------------------------------------------------------------------------

    /// `installHelp()`.
    pub fn install_help(&self) -> &'static str {
        match self.os.as_str() {
            "windows" => "Windows has Microsoft Edge built in; if it is missing, install Google Chrome.",
            "macos" => "Install Google Chrome (or Microsoft Edge, Brave, Chromium) from its website.",
            _ => "Install Chromium or Google Chrome with your package manager (for example \"apt install chromium\"), or name a browser program in Settings > Browser.",
        }
    }

    /// `emptyHtml()`.
    pub fn empty(&self) -> Empty {
        let words = |text: &str| Empty { text: text.to_string(), button: None };
        let Some(s) = &self.status else { return words("Looking for the browser ...") };
        if !s.enabled {
            return Empty {
                text: "The browser is switched off. When it is on, the agent can open pages, read them, click and type, and you see what it sees here and can take over.".into(),
                button: Some(("Open Browser settings", true)),
            };
        }
        if s.browser.is_none() {
            return Empty { text: format!("No Chromium-based browser was found on this computer. {}", self.install_help()), button: Some(("Open Browser settings", false)) };
        }
        if let (true, Some(url)) = (self.running, &self.undrawable) {
            let shown = if url.is_empty() { "true".to_string() } else { url.clone() };
            return words(&format!("The browser shows a page that cannot be drawn here, probably an error page (the site is not reachable). Address: {shown}. The address bar and the agent still work."));
        }
        words("Nothing is open yet. The agent starts the browser when it opens a page; you can also type an address above.")
    }

    pub fn state_text(&self) -> String {
        match &self.status {
            None => String::new(),
            Some(s) if !s.enabled => "off".into(),
            Some(s) if self.running => if s.show_window { "running, window shown" } else { "running" }.into(),
            Some(_) => "ready".into(),
        }
    }

    /// Whether the picture is on the stage (`showImage`).
    pub fn show_image(&self) -> bool {
        self.running && self.image.is_some()
    }

    pub fn foot(&self) -> &'static str {
        if self.commenting {
            "Comment mode: click an element on the page, then write what the agent should do about it."
        } else if self.running {
            "Click, type and scroll in the page to use it yourself. The agent sees the result."
        } else {
            ""
        }
    }

    pub fn view(&self) -> PanelView {
        PanelView {
            state: self.state_text(),
            empty: self.empty(),
            show_image: self.show_image(),
            live_dot: self.running,
            back_enabled: self.running,
            url_enabled: self.status.is_some() && self.enabled(),
            foot: self.foot().to_string(),
            commenting: self.commenting,
            url_input: self.url_input.clone(),
        }
    }

    /// The words of the comments block (`renderComments`) without their markup, in the order the page writes them.
    pub fn comments_text(&self) -> String {
        if self.comments.is_empty() {
            return String::new();
        }
        let mut out = format!("Comments for the agent ({})", self.comments.len());
        for c in &self.comments {
            out.push_str(&c.role);
            if !c.name.is_empty() {
                out.push_str(&format!(" \"{}\"", slice_units(&c.name, 40)));
            }
            out.push_str(&c.note);
        }
        out.push_str("Send to the agent");
        out.push_str("They are also added to your next message.");
        out
    }

    /// The two lines of the note under the picture (`renderNote`): the element, and its text.
    pub fn note_parts(&self) -> Option<(String, String)> {
        let Draft(d) = self.draft.as_ref()?;
        let mut head = show_member(d, "role");
        let name = member(d, "name");
        if !name.is_empty() {
            head.push_str(&format!(" \"{}\"", slice_units(&name, 60)));
        }
        head.push_str(&format!(" ref {}, {}", show_member(d, "ref"), member(d, "tag")));
        let text = member(d, "text");
        let text = if text.is_empty() { String::new() } else { slice_units(&collapse(&text), 120) };
        Some((head, text))
    }

    /// The words of the note under the picture without their markup, in the order the page writes them.
    pub fn note_text(&self) -> String {
        match self.note_parts() {
            None => String::new(),
            Some((head, text)) => format!("{head}{text}CancelAdd comment"),
        }
    }

    /// The frame around the element being commented on, in the pixels of the shown picture (`pin`).
    pub fn pin(&self, shown_width: f64) -> Option<[f64; 4]> {
        let Draft(d) = self.draft.as_ref()?;
        let r = d.get("rect").cloned().unwrap_or(Value::Null);
        let num = |k: &str| crate::computer::js_number_of(r.get(k));
        let k = shown_width / self.view.0;
        let (w, h) = (num("w"), num("h"));
        (w > 0.0 && h > 0.0 && k > 0.0).then(|| [num("x") * k, num("y") * k, w * k, h * k])
    }

    // ---- the server's answers -----------------------------------------------------------------------------------

    /// `loadStatus()`: `None` is a server that does not answer or has no such route.
    pub fn apply_status(&mut self, answer: Option<&Value>) {
        match answer {
            None => self.status = None,
            Some(v) => {
                let s = Status::parse(v);
                self.running = s.running;
                if !s.url.is_empty() {
                    self.url = s.url.clone();
                }
                self.status = Some(s);
            }
        }
    }

    /// `pollFrame()`: the answer of `/frame`.
    pub fn apply_frame(&mut self, data: &Value) {
        self.running = data.get("running").is_some_and(|v| crate::jobs::truthy(Some(v)));
        let text = |k: &str| data.get(k).and_then(Value::as_str).filter(|s| !s.is_empty());
        if !self.running {
            self.key.clear();
            self.image = None;
        } else if data.get("undrawable").is_some_and(|v| crate::jobs::truthy(Some(v))) {
            self.undrawable = Some(text("url").unwrap_or_default().to_string());
            self.key.clear();
            self.image = None;
            if let Some(url) = text("url") {
                self.url = url.to_string();
            }
            if !self.url_focused {
                self.url_input = self.url.clone();
            }
        } else if !data.get("error").is_some_and(|v| crate::jobs::truthy(Some(v))) {
            self.undrawable = None;
            if let Some(image) = text("image") {
                self.image = Some(image.to_string());
                self.key = text("key").unwrap_or_default().to_string();
            }
            let w = crate::computer::js_number_of(data.get("width"));
            if w != 0.0 && !w.is_nan() {
                let h = crate::computer::js_number_of(data.get("height"));
                self.view = (w, if h != 0.0 && !h.is_nan() { h } else { self.view.1 });
            }
            if let Some(url) = text("url") {
                self.url = url.to_string();
            }
            self.title = text("title").unwrap_or_default().to_string();
            if !self.url_focused {
                self.url_input = if self.url == "about:blank" { String::new() } else { self.url.clone() };
            }
        }
    }

    // ---- the ticks ----------------------------------------------------------------------------------------------

    /// `tick()` before its requests: whether the status is read first (`!br.status || br.ticks++ % 12 === 0`).
    pub fn tick_reads_status(&mut self) -> bool {
        let first = self.status.is_none();
        let n = self.ticks;
        self.ticks += 1;
        first || n % 12 == 0
    }

    /// Whether a frame is asked for after that (`br.status && br.status.settings.enabled`).
    pub fn tick_asks_frame(&self) -> bool {
        self.enabled()
    }

    /// The pause until the next tick (`schedule(br.running ? 650 : 2000)`).
    pub fn tick_pause_ms(&self) -> u64 {
        if self.running { 650 } else { 2000 }
    }

    // ---- the user uses the page -------------------------------------------------------------------------------

    /// `pagePoint(e)`: a point of the preview as a point of the page.
    pub fn page_point(&self, client: (f64, f64), rect: Rect) -> Option<(f64, f64)> {
        (rect.width != 0.0).then(|| ((client.0 - rect.left) / rect.width * self.view.0, (client.1 - rect.top) / rect.height * self.view.1))
    }

    fn live(&self) -> bool {
        self.running && self.image.is_some()
    }

    /// A click on the stage: a click on the page, or the element to comment on.
    pub fn click(&self, client: (f64, f64), rect: Rect, on_button: bool) -> Option<Input> {
        if on_button || !self.live() {
            return None;
        }
        let (x, y) = self.page_point(client, rect)?;
        Some(if self.commenting { Input::Locate { x, y } } else { Input::Click { x, y } })
    }

    pub fn dblclick(&self, client: (f64, f64), rect: Rect) -> Option<Input> {
        if self.commenting || !self.live() {
            return None;
        }
        self.page_point(client, rect).map(|(x, y)| Input::Dblclick { x, y })
    }

    /// A turn of the wheel: collected for 50 ms; `true` when the timer has to be started.
    pub fn wheel(&mut self, client: (f64, f64), rect: Rect, delta: (f64, f64), unit: WheelUnit) -> bool {
        if !self.live() {
            return false;
        }
        let Some((x, y)) = self.page_point(client, rect) else { return false };
        let k = match unit {
            WheelUnit::Pixels => 1.0,
            WheelUnit::Lines => 16.0,
            WheelUnit::Pages => self.view.1,
        };
        let w = &mut self.wheel;
        w.dx += delta.0 * k;
        w.dy += delta.1 * k;
        w.x = x;
        w.y = y;
        let start = !w.armed;
        w.armed = true;
        start
    }

    /// The 50 ms are over: what was collected goes to the page.
    pub fn flush_wheel(&mut self) -> Option<Input> {
        if !self.wheel.armed {
            return None;
        }
        let w = std::mem::take(&mut self.wheel);
        Some(Input::Wheel { x: w.x, y: w.y, dx: w.dx, dy: w.dy })
    }

    /// `keydown` of the stage.
    pub fn key(&self, press: &KeyPress) -> Option<Input> {
        if !self.running || press.composing || ["Control", "Shift", "Alt", "Meta", "Dead", "Process", "CapsLock"].contains(&press.key.as_str()) {
            return None;
        }
        // the paste shortcut is handled by the paste event, which has the text
        if (press.ctrl || press.meta) && !press.alt && press.key.to_lowercase() == "v" {
            return None;
        }
        let single = press.key.chars().count() == 1;
        let function = {
            let b = press.key.as_bytes();
            b.first() == Some(&b'F') && press.key[1..].parse::<u8>().is_ok_and(|n| (1..=12).contains(&n)) && b.get(1) != Some(&b'0') && !press.key[1..].starts_with('+')
        };
        if !single && !KEY_NAMES.contains(&press.key.as_str()) && !function {
            return None;
        }
        let mut mods: Vec<&str> = Vec::new();
        if press.ctrl {
            mods.push("Control");
        }
        if press.alt {
            mods.push("Alt");
        }
        if press.meta {
            mods.push("Meta");
        }
        if press.shift && (!single || !mods.is_empty()) {
            mods.push("Shift");
        }
        let name = if press.key == " " { "Space" } else { press.key.as_str() };
        let mut parts = mods.iter().map(|m| m.to_string()).collect::<Vec<_>>();
        parts.push(name.to_string());
        Some(Input::Key(parts.join("+")))
    }

    /// `paste` on the stage.
    pub fn paste(&self, text: &str) -> Option<Input> {
        (!text.is_empty() && self.running).then(|| Input::Text(text.to_string()))
    }

    /// The address bar: the text, trimmed; nothing for an empty one.
    pub fn submit(&self) -> Option<Input> {
        let text = self.url_input.trim();
        (!text.is_empty()).then(|| Input::Navigate(text.to_string()))
    }

    /// What `sendInput` makes of an answer: a toast when the server refused or did not answer; the
    /// element when it was `locate`.
    pub fn input_answered(&mut self, input: &Input, answer: Result<Value, String>) -> Option<String> {
        match answer {
            Err(e) => Some(format!("The browser did not answer: {e}")),
            Ok(data) => {
                if matches!(input, Input::Locate { .. }) && data.get("ok").is_some_and(|v| crate::jobs::truthy(Some(v))) {
                    self.draft = Some(Draft(data.clone()));
                }
                (data.get("ok") == Some(&Value::Bool(false)))
                    .then(|| data.get("error").filter(|e| crate::jobs::truthy(Some(e))).map(js_string))
                    .flatten()
            }
        }
    }

    /// `brStop` after its question: the browser is closed at once.
    pub fn closed(&mut self) {
        self.running = false;
        self.key.clear();
        self.image = None;
    }

    // ---- comments ---------------------------------------------------------------------------------------------------

    pub fn toggle_commenting(&mut self) {
        self.commenting = !self.commenting;
        if !self.commenting {
            self.draft = None;
        }
    }

    /// *Add comment*: `false` when the note is empty (the box keeps the focus).
    pub fn add_comment(&mut self, note: &str) -> bool {
        let note = note.trim();
        if note.is_empty() {
            return false;
        }
        let Some(Draft(d)) = self.draft.take() else { return false };
        let url = {
            let u = member(&d, "url");
            if u.is_empty() { self.url.clone() } else { u }
        };
        self.comments.push(Comment {
            reference: show_member(&d, "ref"),
            role: show_member(&d, "role"),
            name: member(&d, "name"),
            tag: member(&d, "tag"),
            text: slice_units(&collapse(&member(&d, "text")), 160),
            url,
            note: note.to_string(),
        });
        true
    }

    pub fn cancel_note(&mut self) {
        self.draft = None;
    }

    pub fn delete_comment(&mut self, i: usize) {
        if i < self.comments.len() {
            self.comments.remove(i);
        }
    }

    /// `fllmBrowserTakeComments()`: the comments as text for the agent; they are sent once.
    pub fn take_comments(&mut self) -> String {
        if self.comments.is_empty() {
            return String::new();
        }
        let lines: Vec<String> = self
            .comments
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let name = if c.name.is_empty() { String::new() } else { format!(" \"{}\"", c.name) };
                let tag = if c.tag.is_empty() { "element" } else { &c.tag };
                let text = if !c.text.is_empty() && c.text != c.name { format!(", text: \"{}\"", c.text) } else { String::new() };
                let url = if c.url.is_empty() { "the page" } else { &c.url };
                format!("{}. {}{} <{}> (ref {}{}) on {}: {}", i + 1, c.role, name, tag, c.reference, text, url, c.note)
            })
            .collect();
        self.comments.clear();
        format!("[The user pointed at elements of the browser page and left comments. Act on them. The refs belong to the page that was open then; take a new browser_snapshot if a ref no longer works.]\n{}\n\n", lines.join("\n"))
    }

    /// *Send to the agent*: the box gets the sentence when it is empty.
    pub fn send_text(&self, box_text: &str) -> String {
        if box_text.trim().is_empty() { "Please act on my comments on the browser page.".to_string() } else { box_text.to_string() }
    }

    /// `fllmBrowserUrl()`: the page the browser is on, for the rules of the network tools.
    pub fn current_url(&self) -> String {
        if !self.url.is_empty() {
            return self.url.clone();
        }
        self.status.as_ref().map(|s| s.url.clone()).unwrap_or_default()
    }
}

/// `toolLabel` of the page for the browser tools: the verb and what it is about.
pub fn tool_label(name: &str, args: &Value) -> Option<(String, String)> {
    let a = |k: &str| args.get(k);
    let t = |s: &str| s.to_string();
    let nullish = |k: &str| a(k).is_none_or(Value::is_null);
    let or_empty = |k: &str| crate::computer::text_or_empty(a(k));
    let truthy = |k: &str| crate::jobs::truthy(a(k));
    let r#ref = || if nullish("ref") { String::new() } else { js_string(a("ref").unwrap_or(&Value::Null)) };
    Some(match name {
        "browser_open" => (t("Open in browser"), if truthy("url") { js_string(a("url").unwrap_or(&Value::Null)) } else { String::new() }),
        "browser_snapshot" => (t("Read the browser page"), String::new()),
        "browser_click" => (t("Browser click"), format!("ref {}", r#ref())),
        "browser_type" => (t("Browser type"), format!("ref {}: {}{}", r#ref(), slice_units(&or_empty("text"), 60), if truthy("submit") { " + Enter" } else { "" })),
        "browser_select" => (t("Browser select"), format!("ref {}: {}", r#ref(), if nullish("value") { String::new() } else { js_string(a("value").unwrap_or(&Value::Null)) })),
        "browser_press" => (t("Browser key"), if truthy("key") { js_string(a("key").unwrap_or(&Value::Null)) } else { String::new() }),
        "browser_scroll" => (
            t("Browser scroll"),
            if !nullish("ref") {
                format!("to ref {}", js_string(a("ref").unwrap_or(&Value::Null)))
            } else if truthy("direction") {
                js_string(a("direction").unwrap_or(&Value::Null))
            } else {
                t("down")
            },
        ),
        "browser_back" => (t("Browser back"), String::new()),
        "browser_screenshot" => (t("Browser screenshot"), if !nullish("ref") { format!("ref {}", js_string(a("ref").unwrap_or(&Value::Null))) } else if truthy("full_page") { t("whole page") } else { String::new() }),
        "browser_tabs" => (t("Browser tabs"), if nullish("select") { String::new() } else { format!("switch to {}", js_string(a("select").unwrap_or(&Value::Null))) }),
        "browser_close" => (t("Close browser"), if truthy("all") { t("all") } else if nullish("tab") { t("tab") } else { format!("tab {}", js_string(a("tab").unwrap_or(&Value::Null))) }),
        _ => return None,
    })
}

/// A key of the window as `keydown` of the page would name it; `None` for the ones the page does not forward.
pub fn key_press(key: &iced::keyboard::Key, modified: &iced::keyboard::Key, mods: iced::keyboard::Modifiers, composing: bool) -> Option<KeyPress> {
    use iced::keyboard::{key::Named, Key};
    // with Ctrl, Alt or the logo key held the key is named by what it is, not by what it types
    let chosen = if mods.control() || mods.alt() || mods.logo() { key } else { modified };
    let name = match chosen {
        Key::Character(c) => c.to_string(),
        Key::Named(n) => match n {
            Named::Enter => "Enter".into(),
            Named::Tab => "Tab".into(),
            Named::Escape => "Escape".into(),
            Named::Backspace => "Backspace".into(),
            Named::Delete => "Delete".into(),
            Named::ArrowUp => "ArrowUp".into(),
            Named::ArrowDown => "ArrowDown".into(),
            Named::ArrowLeft => "ArrowLeft".into(),
            Named::ArrowRight => "ArrowRight".into(),
            Named::Home => "Home".into(),
            Named::End => "End".into(),
            Named::PageUp => "PageUp".into(),
            Named::PageDown => "PageDown".into(),
            Named::Insert => "Insert".into(),
            Named::Space => " ".into(),
            Named::F1 => "F1".into(),
            Named::F2 => "F2".into(),
            Named::F3 => "F3".into(),
            Named::F4 => "F4".into(),
            Named::F5 => "F5".into(),
            Named::F6 => "F6".into(),
            Named::F7 => "F7".into(),
            Named::F8 => "F8".into(),
            Named::F9 => "F9".into(),
            Named::F10 => "F10".into(),
            Named::F11 => "F11".into(),
            Named::F12 => "F12".into(),
            _ => return None,
        },
        _ => return None,
    };
    Some(KeyPress { key: name, ctrl: mods.control(), alt: mods.alt(), shift: mods.shift(), meta: mods.logo(), composing })
}
