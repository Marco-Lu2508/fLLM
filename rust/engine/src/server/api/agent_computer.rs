//! Computer use for the coding agent: with the explicit permission of the user the agent can look at the screen
//! (`computer_screenshot`) and operate the computer (mouse, keyboard) in programs that have no API.
//!
//! This is the most dangerous tool of the product, so it is built to be switched off, visible and interruptible:
//!
//! * It is **off** until the user switches it on in Settings > Runtime (`computer_use` in `runtime.json`). While it is
//!   off no `computer_*` tool is offered and every call is refused here as well.
//! * It only works through `POST /api/studio/agent/tool` from the page: the routes are local-machine-only like every
//!   agent route (`http::studio_admin_blocked`), and the scheduled tasks, which run unattended, never reach it.
//! * Every action counts against a rate limit per minute and a step cap per task (both settings).
//! * Coordinates are those of the last screenshot the model saw; no input is sent before a screenshot was taken.
//! * `POST /api/studio/agent/computer/stop` (the big STOP button, the Escape-three-times key of the page) and the global
//!   chord Ctrl+Alt+Shift+Esc (polled here) cancel the running action and switch the permission off.
//! * A short list of key chords that are dangerous system-wide is refused ([`dangerous_chord`]). It is a convenience
//!   for careless moments, not a security boundary: the permission question of the page is the boundary.
//!
//! The pure logic (coordinate mapping with DPI and several displays, key chords, the dangerous list, the rate limiter,
//! the tool schemas) lives here and is unit tested. The operating system layer is one small module per platform
//! (`agent_computer_x11`, `agent_computer_win`, `agent_computer_mac`) without extra dependencies.
use super::agent::{agent_disabled, disabled_reply};
use super::*;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

#[cfg(all(feature = "computer-use", target_os = "linux"))]
use super::agent_computer_x11 as os;
#[cfg(all(feature = "computer-use", windows))]
use super::agent_computer_win as os;
#[cfg(all(feature = "computer-use", target_os = "macos"))]
use super::agent_computer_mac as os;
#[cfg(not(all(feature = "computer-use", any(target_os = "linux", windows, target_os = "macos"))))]
mod os {
    //! No operating system layer in this build.
    use super::*;
    const WHY: &str = "computer use is not available in this build (the computer-use feature is off, or this operating system is not supported)";
    pub fn platform() -> &'static str {
        "none"
    }
    pub fn available() -> Result<(), String> {
        Err(WHY.to_owned())
    }
    pub fn displays() -> Result<Vec<DisplayInfo>, String> {
        Err(WHY.to_owned())
    }
    pub fn capture(_: &DisplayInfo, _: (u32, u32, u32, u32)) -> Result<Frame, String> {
        Err(WHY.to_owned())
    }
    pub fn cursor() -> Result<(i32, i32), String> {
        Err(WHY.to_owned())
    }
    pub fn move_to(_: i32, _: i32) -> Result<(), String> {
        Err(WHY.to_owned())
    }
    pub fn button(_: Button, _: bool) -> Result<(), String> {
        Err(WHY.to_owned())
    }
    pub fn scroll(_: i32, _: i32) -> Result<(), String> {
        Err(WHY.to_owned())
    }
    pub fn press(_: &Chord) -> Result<(), String> {
        Err(WHY.to_owned())
    }
    pub fn type_char(_: char) -> Result<(), String> {
        Err(WHY.to_owned())
    }
    pub fn panic_down() -> bool {
        false
    }
}

/// The long side of a screenshot that goes to the model, in pixels.
pub(super) const MAX_SIDE: u32 = 1568;
const MAX_WAIT_MS: u64 = 10_000;
const MAX_TYPE_CHARS: usize = 4000;
const MAX_CHORDS: usize = 10;
/// A task that has been idle this long starts again with a fresh step count.
const IDLE_RESET_MS: u64 = 30 * 60_000;

// ---- shared types ---------------------------------------------------------------------------------

/// One screen. Positions and the scale are in *input units*: the units the operating system wants for the mouse position
/// (pixels on Windows and X11, points on macOS). `width` and `height` are physical pixels, `scale` is physical pixels per
/// input unit (1 on Windows and X11, 2 on a Retina display).
#[derive(Clone, Debug, PartialEq)]
pub struct DisplayInfo {
    pub index: usize,
    pub name: String,
    pub primary: bool,
    pub origin_x: i32,
    pub origin_y: i32,
    pub width: u32,
    pub height: u32,
    pub scale: f64,
    /// The scaling the user set for text and windows (1.0 = 100 %); informative.
    pub ui_scale: f64,
}

/// A captured picture: RGBA, row after row.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    Left,
    Middle,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Named {
    Enter,
    Tab,
    Escape,
    Backspace,
    Delete,
    Insert,
    Home,
    End,
    PageUp,
    PageDown,
    Up,
    Down,
    Left,
    Right,
    Space,
    CapsLock,
    PrintScreen,
    Menu,
    /// The Windows / Command / Super key on its own.
    Meta,
    F(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Char(char),
    Named(Named),
}

/// A key with its modifiers, for example `ctrl+shift+t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chord {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    /// Windows key, Command key, Super key.
    pub meta: bool,
    pub key: Key,
}

/// Cancellation of the running action: the stop route and the panic chord raise the epoch.
#[derive(Clone, Copy)]
pub struct Cancel {
    epoch: u64,
}

static STOP_EPOCH: AtomicU64 = AtomicU64::new(0);

impl Cancel {
    pub fn now() -> Self {
        Cancel { epoch: STOP_EPOCH.load(Ordering::SeqCst) }
    }
    pub fn cancelled(&self) -> bool {
        STOP_EPOCH.load(Ordering::SeqCst) != self.epoch
    }
    /// Sleeps in short slices; false when the action was stopped meanwhile.
    pub fn sleep(&self, ms: u64) -> bool {
        let end = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < end {
            if self.cancelled() {
                return false;
            }
            std::thread::sleep(Duration::from_millis(ms.min(25)));
        }
        !self.cancelled()
    }
}

// ---- coordinates ----------------------------------------------------------------------------------

/// What the model saw: which part of which display, at which size. Coordinates the model gives refer to this picture.
#[derive(Clone, Debug, PartialEq)]
pub struct ShotMap {
    pub display: usize,
    pub origin_x: i32,
    pub origin_y: i32,
    pub scale: f64,
    /// Size of the whole display in physical pixels.
    pub display_w: u32,
    pub display_h: u32,
    /// The captured part of the display in physical pixels, relative to the display: x, y, width, height.
    pub region: (u32, u32, u32, u32),
    pub image_w: u32,
    pub image_h: u32,
}

impl ShotMap {
    /// Image coordinates -> the position the operating system wants for the mouse (input units, whole desktop).
    pub fn to_input(&self, x: f64, y: f64) -> Result<(i32, i32), String> {
        if !x.is_finite() || !y.is_finite() {
            return Err("the coordinates must be numbers".to_owned());
        }
        if x < 0.0 || y < 0.0 || x > f64::from(self.image_w) || y > f64::from(self.image_h) {
            return Err(format!(
                "the point ({}, {}) is outside the last screenshot, which is {}x{} pixels; give coordinates inside it (take a new screenshot if the screen changed)",
                round1(x),
                round1(y),
                self.image_w,
                self.image_h
            ));
        }
        let px = f64::from(self.region.0) + x * f64::from(self.region.2) / f64::from(self.image_w);
        let py = f64::from(self.region.1) + y * f64::from(self.region.3) / f64::from(self.image_h);
        let max_x = (f64::from(self.display_w) / self.scale - 1.0).max(0.0);
        let max_y = (f64::from(self.display_h) / self.scale - 1.0).max(0.0);
        let gx = f64::from(self.origin_x) + (px / self.scale).round().min(max_x);
        let gy = f64::from(self.origin_y) + (py / self.scale).round().min(max_y);
        Ok((gx as i32, gy as i32))
    }

    /// The reverse: a desktop position -> image coordinates, or `None` when it is outside the picture.
    pub fn to_image(&self, gx: i32, gy: i32) -> Option<(f64, f64)> {
        let px = f64::from(gx - self.origin_x) * self.scale - f64::from(self.region.0);
        let py = f64::from(gy - self.origin_y) * self.scale - f64::from(self.region.1);
        if px < 0.0 || py < 0.0 || px >= f64::from(self.region.2) || py >= f64::from(self.region.3) {
            return None;
        }
        Some((px * f64::from(self.image_w) / f64::from(self.region.2), py * f64::from(self.image_h) / f64::from(self.region.3)))
    }
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

/// The size a picture gets so that its long side is at most `max` (never enlarged).
pub fn fit_size(w: u32, h: u32, max: u32) -> (u32, u32) {
    let long = w.max(h);
    if long <= max || long == 0 {
        return (w.max(1), h.max(1));
    }
    let f = f64::from(max) / f64::from(long);
    (((f64::from(w) * f).round() as u32).max(1), ((f64::from(h) * f).round() as u32).max(1))
}

/// The region the model asks for -> physical pixels of the display. The model names it in the coordinates of its last
/// screenshot of the same display; without one the numbers are pixels of the display.
pub fn region_to_physical(prev: Option<&ShotMap>, display: &DisplayInfo, region: (f64, f64, f64, f64)) -> Result<(u32, u32, u32, u32), String> {
    let (x, y, w, h) = region;
    if ![x, y, w, h].iter().all(|v| v.is_finite()) || w < 1.0 || h < 1.0 || x < 0.0 || y < 0.0 {
        return Err("region needs x, y, width and height (positive numbers)".to_owned());
    }
    let (fx, fy, fw, fh) = match prev.filter(|p| p.display == display.index) {
        Some(p) => {
            let sx = f64::from(p.region.2) / f64::from(p.image_w);
            let sy = f64::from(p.region.3) / f64::from(p.image_h);
            (f64::from(p.region.0) + x * sx, f64::from(p.region.1) + y * sy, w * sx, h * sy)
        }
        None => (x, y, w, h),
    };
    if fx.round() >= f64::from(display.width) || fy.round() >= f64::from(display.height) {
        return Err(format!("the region is outside display {} ({}x{} pixels)", display.index, display.width, display.height));
    }
    let x0 = fx.round().max(0.0);
    let y0 = fy.round().max(0.0);
    let x1 = (fx + fw).round().min(f64::from(display.width));
    let y1 = (fy + fh).round().min(f64::from(display.height));
    if x1 - x0 < 1.0 || y1 - y0 < 1.0 {
        return Err(format!("the region is outside display {} ({}x{} pixels)", display.index, display.width, display.height));
    }
    Ok((x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32))
}

// ---- key chords -----------------------------------------------------------------------------------

const MODIFIER_NAMES: [&str; 12] = ["ctrl", "control", "ctl", "alt", "option", "opt", "shift", "cmd", "command", "super", "win", "windows"];

/// Sets the modifier `name` on the chord; false when `name` is not a modifier.
fn apply_modifier(name: &str, chord: &mut Chord) -> bool {
    match name {
        "ctrl" | "control" | "ctl" => chord.ctrl = true,
        "alt" | "option" | "opt" => chord.alt = true,
        "shift" => chord.shift = true,
        "cmd" | "command" | "super" | "win" | "windows" | "meta" => chord.meta = true,
        _ => return false,
    }
    true
}

fn named_key(name: &str) -> Option<Named> {
    Some(match name {
        "enter" | "return" => Named::Enter,
        "tab" => Named::Tab,
        "esc" | "escape" => Named::Escape,
        "backspace" => Named::Backspace,
        "delete" | "del" => Named::Delete,
        "insert" | "ins" => Named::Insert,
        "home" => Named::Home,
        "end" => Named::End,
        "pageup" | "pgup" | "page_up" => Named::PageUp,
        "pagedown" | "pgdn" | "page_down" => Named::PageDown,
        "up" | "arrowup" => Named::Up,
        "down" | "arrowdown" => Named::Down,
        "left" | "arrowleft" => Named::Left,
        "right" | "arrowright" => Named::Right,
        "space" | "spacebar" => Named::Space,
        "capslock" | "caps" => Named::CapsLock,
        "printscreen" | "print" | "prtsc" | "prtscr" => Named::PrintScreen,
        "menu" | "apps" => Named::Menu,
        other => {
            let n = other.strip_prefix('f')?.parse::<u8>().ok()?;
            if (1..=24).contains(&n) {
                Named::F(n)
            } else {
                return None;
            }
        }
    })
}

/// One chord such as `ctrl+s`, `Shift+Tab`, `cmd+alt+left`, `enter`, `ctrl++`, `f5`, or `win` on its own.
pub fn parse_chord(text: &str) -> Result<Chord, String> {
    let original = text.trim();
    if original.is_empty() {
        return Err("empty key".to_owned());
    }
    let mut chord = Chord { ctrl: false, alt: false, shift: false, meta: false, key: Key::Char('a') };
    let (mods, last): (&str, String) = if original == "+" {
        ("", "+".to_owned())
    } else if let Some(head) = original.strip_suffix("++") {
        (head, "+".to_owned())
    } else {
        match original.rsplit_once('+') {
            Some((head, tail)) => (head, tail.to_owned()),
            None => ("", original.to_owned()),
        }
    };
    for part in mods.split('+').map(str::trim).filter(|p| !p.is_empty()) {
        if !apply_modifier(&part.to_ascii_lowercase(), &mut chord) {
            return Err(format!("'{part}' is not a modifier (use ctrl, alt, shift or cmd/win/super) in '{original}'"));
        }
    }
    let last_l = last.trim().to_ascii_lowercase();
    if let Some(named) = named_key(&last_l) {
        chord.key = Key::Named(named);
    } else if MODIFIER_NAMES.contains(&last_l.as_str()) || last_l == "meta" {
        // a modifier key by itself: only the Windows / Command key makes sense on its own
        let mut probe = chord;
        apply_modifier(&last_l, &mut probe);
        if probe.meta && !chord.ctrl && !chord.alt && !chord.shift && !chord.meta {
            chord.key = Key::Named(Named::Meta);
        } else {
            return Err(format!("'{original}' names no key: add a key after the modifiers, for example ctrl+s"));
        }
    } else {
        let mut chars = last.trim().chars();
        match (chars.next(), chars.next()) {
            (Some(c), None) if !c.is_control() => {
                if c.is_uppercase() {
                    chord.shift = true;
                }
                chord.key = Key::Char(c.to_lowercase().next().unwrap_or(c));
            }
            _ => return Err(format!("unknown key '{}' in '{original}' (examples: enter, tab, esc, f5, left, a, ctrl+s)", last.trim())),
        }
    }
    Ok(chord)
}

/// Several chords one after the other, separated by spaces: `ctrl+a ctrl+c`.
pub fn parse_keys(text: &str) -> Result<Vec<Chord>, String> {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.is_empty() {
        return Err("keys is empty".to_owned());
    }
    if parts.len() > MAX_CHORDS {
        return Err(format!("at most {MAX_CHORDS} keys per call"));
    }
    parts.into_iter().map(parse_chord).collect()
}

/// Key chords that act on the whole computer instead of the program in front, and that the agent may not press. A
/// convenience against careless moments, not a security boundary (the agent can still click through menus; the user's
/// permission question is the boundary). `os` is `std::env::consts::OS`.
pub fn dangerous_chord(c: &Chord, os: &str) -> Option<&'static str> {
    let named = |n: Named| c.key == Key::Named(n);
    let letter = |l: char| c.key == Key::Char(l);
    let function = |n: u8| c.key == Key::Named(Named::F(n));
    if c.ctrl && c.alt && (named(Named::Delete) || named(Named::Backspace)) {
        return Some("Ctrl+Alt+Delete and Ctrl+Alt+Backspace end or lock the session");
    }
    if c.ctrl && c.shift && named(Named::Escape) {
        return Some("Ctrl+Shift+Esc opens the task manager, and Ctrl+Alt+Shift+Esc is the emergency stop of computer use");
    }
    if c.alt && function(4) {
        return Some("Alt+F4 closes windows and opens the shutdown dialog; close a window with a click on its close button instead");
    }
    if c.ctrl && c.alt && (1..=12).any(function) {
        return Some("Ctrl+Alt+F1..F12 switch to another console and leave the desktop session");
    }
    if c.meta && matches!(os, "windows" | "linux") && (letter('l') || (os == "windows" && (letter('r') || letter('x')))) {
        return Some("this Windows / Super key chord locks the screen, opens the Run dialog or the power menu");
    }
    if os == "macos" && c.meta && ((c.ctrl && letter('q')) || (c.shift && letter('q')) || (c.alt && named(Named::Escape))) {
        return Some("this Command chord locks the screen, logs out or opens Force Quit");
    }
    if c.alt && named(Named::PrintScreen) && c.ctrl {
        return Some("Ctrl+Alt+PrintScreen is the magic SysRq key combination");
    }
    None
}

// ---- rate limit and step cap ----------------------------------------------------------------------

/// Counts the actions of the last minute.
#[derive(Default)]
pub struct RateLimiter {
    stamps: VecDeque<u64>,
}

impl RateLimiter {
    pub const fn new() -> Self {
        RateLimiter { stamps: VecDeque::new() }
    }

    /// Records an action at `now_ms`; `Err(wait_ms)` when `per_minute` actions already happened in the last minute.
    pub fn check(&mut self, now_ms: u64, per_minute: u32) -> Result<(), u64> {
        while self.stamps.front().is_some_and(|t| now_ms.saturating_sub(*t) >= 60_000) {
            self.stamps.pop_front();
        }
        if self.stamps.len() >= per_minute.max(1) as usize {
            let oldest = self.stamps.front().copied().unwrap_or(now_ms);
            return Err((oldest + 60_000).saturating_sub(now_ms).max(1));
        }
        self.stamps.push_back(now_ms);
        Ok(())
    }
}

struct State {
    limiter: RateLimiter,
    steps: u32,
    last_action_ms: u64,
    last_shot: Option<ShotMap>,
}

static STATE: Mutex<State> = Mutex::new(State { limiter: RateLimiter::new(), steps: 0, last_action_ms: 0, last_shot: None });
/// One action at a time: tool calls that arrive together do not interleave their mouse movements.
static ACTION: Mutex<()> = Mutex::new(());
static WATCHER: AtomicBool = AtomicBool::new(false);

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn now_ms() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1
}

/// Rate limit and step cap for one more action. Pure apart from the state it is given.
fn admit(state: &mut State, now: u64, per_minute: u32, max_steps: u32) -> Result<(), String> {
    if state.last_action_ms != 0 && now.saturating_sub(state.last_action_ms) > IDLE_RESET_MS {
        state.steps = 0;
    }
    if state.steps >= max_steps {
        return Err(format!(
            "step cap reached: {max_steps} computer actions in this task. Stop here and tell the user what is done and what is left; the user can start a new task (the cap is a setting in Settings > Runtime)"
        ));
    }
    if let Err(wait) = state.limiter.check(now, per_minute) {
        return Err(format!(
            "rate limit: at most {per_minute} computer actions per minute. Wait about {} seconds, and plan fewer, larger steps",
            wait.div_ceil(1000)
        ));
    }
    state.steps += 1;
    state.last_action_ms = now;
    Ok(())
}

// ---- tool schemas ---------------------------------------------------------------------------------

fn point_schema(what: &str) -> Value {
    json!({"type": "object", "description": what, "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]})
}

/// The schemas of the computer tools (permission category `computer`). They are only offered while computer use is on.
pub(super) fn catalog() -> Vec<Value> {
    use super::agent_tools::tool;
    let xy = |extra: Value| {
        let mut props = json!({
            "x": {"type": "number", "description": "Horizontal position in pixels of your last screenshot (0 = left edge)."},
            "y": {"type": "number", "description": "Vertical position in pixels of your last screenshot (0 = top edge)."},
        });
        if let (Some(p), Some(e)) = (props.as_object_mut(), extra.as_object()) {
            p.extend(e.clone());
        }
        props
    };
    vec![
        tool(
            "computer_screenshot",
            "computer",
            "Look at the screen: returns a picture of a display (or a part of it). The picture is scaled so that its long side is at most 1568 pixels; all coordinates you give to the other computer tools are pixels of the LAST screenshot you took, so take a new one whenever the screen may have changed. Use this only for programs that have no API, no command line and are not a web page: for web pages prefer the browser tool.",
            json!({
                "display": {"type": "integer", "description": "Which display (see computer_displays). Default: the main display."},
                "region": {"type": "object", "description": "Only this part, in pixels of your last screenshot of that display (zoom in to read small text).", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "width": {"type": "number"}, "height": {"type": "number"}}, "required": ["x", "y", "width", "height"]},
            }),
            &[],
        ),
        tool(
            "computer_click",
            "computer",
            "Move the mouse to a point of your last screenshot and click. The user is asked before every click. After it, take a screenshot to see what happened.",
            xy(json!({
                "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Default left."},
                "double": {"type": "boolean", "description": "Double click (open a file, select a word)."},
            })),
            &["x", "y"],
        ),
        tool("computer_move", "computer", "Move the mouse pointer to a point of your last screenshot without clicking (hover menus and tooltips).", xy(json!({})), &["x", "y"]),
        tool(
            "computer_drag",
            "computer",
            "Press the mouse button at one point of your last screenshot, move to another and release (move a window, select text, drag a file, move a slider).",
            json!({
                "from": point_schema("Where to press the button."),
                "to": point_schema("Where to release it."),
                "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Default left."},
            }),
            &["from", "to"],
        ),
        tool(
            "computer_type",
            "computer",
            "Type text with the keyboard into the focused field of the program in front (click the field first). A new line in the text presses Enter, a tab character presses Tab. Characters that the keyboard layout of this computer cannot produce are reported. For shortcuts use computer_key.",
            json!({"text": {"type": "string", "description": "The text to type (at most 4000 characters)."}}),
            &["text"],
        ),
        tool(
            "computer_key",
            "computer",
            "Press a key or a key combination, for example `enter`, `tab`, `esc`, `f5`, `ctrl+s`, `ctrl+shift+t`, `alt+left`, `cmd+c` (cmd, win and super are the same key). Several presses one after the other are separated by spaces: `ctrl+a ctrl+c`. A few chords that act on the whole computer (Ctrl+Alt+Del, Alt+F4, locking the screen, the Run dialog, ...) are refused.",
            json!({"keys": {"type": "string", "description": "The key or combination, or several separated by spaces (at most 10)."}}),
            &["keys"],
        ),
        tool(
            "computer_scroll",
            "computer",
            "Scroll the wheel at a point of your last screenshot. dy > 0 scrolls down, dy < 0 up, dx > 0 right, dx < 0 left; the unit is wheel notches (about 3 lines), at most 50 per call.",
            xy(json!({
                "dx": {"type": "integer", "description": "Horizontal notches (default 0)."},
                "dy": {"type": "integer", "description": "Vertical notches (default 0)."},
            })),
            &["x", "y"],
        ),
        tool(
            "computer_wait",
            "computer",
            "Wait for the screen to settle (an application starts, a page loads). At most 10000 ms per call.",
            json!({"ms": {"type": "integer", "description": "Milliseconds (default 1000, at most 10000)."}}),
            &[],
        ),
        tool("computer_cursor", "computer", "Where the mouse pointer is now, in pixels of your last screenshot (and in screen coordinates).", json!({}), &[]),
        tool("computer_displays", "computer", "List the displays with their sizes in pixels, positions and scaling. Display 0 is the main one unless the list says otherwise.", json!({}), &[]),
    ]
}

/// Whether a tool name belongs to computer use.
pub(super) fn is_computer_tool(name: &str) -> bool {
    name.starts_with("computer_")
}

// ---- settings -------------------------------------------------------------------------------------

/// `FLLM_NO_COMPUTER_USE=1` switches computer use off for good on this server (no setting can turn it on).
fn forbidden_by_env() -> bool {
    std::env::var("FLLM_NO_COMPUTER_USE").is_ok_and(|v| !v.trim().is_empty() && v.trim() != "0")
}

/// Whether the user switched computer use on and nothing forbids it.
pub(super) fn enabled() -> bool {
    !agent_disabled() && !forbidden_by_env() && super::studio_mode::computer_use()
}

/// Whether the tools are offered to the model: switched on and possible on this machine.
pub(super) fn offered() -> bool {
    enabled() && os::available().is_ok()
}

fn gate() -> Result<(), String> {
    if agent_disabled() {
        return Err("the coding agent is switched off on this server".to_owned());
    }
    if forbidden_by_env() {
        return Err("computer use is forbidden on this server (FLLM_NO_COMPUTER_USE)".to_owned());
    }
    if !super::studio_mode::computer_use() {
        return Err("computer use is switched off. Do not try again; tell the user that it can be switched on in Settings > Runtime if this task needs it".to_owned());
    }
    os::available()
}

// ---- the tools ------------------------------------------------------------------------------------

/// A picture for the model: PNG bytes with the size.
pub struct Picture {
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// What a computer tool returns: text for the model and, for a screenshot, the picture.
pub struct Outcome {
    pub text: String,
    pub picture: Option<Picture>,
}

impl Outcome {
    fn text(text: impl Into<String>) -> Self {
        Outcome { text: text.into(), picture: None }
    }
}

fn number(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
}

/// A point from `{x, y}` or `[x, y]`.
fn point_of(v: &Value) -> Option<(f64, f64)> {
    if let Some(a) = v.as_array() {
        return Some((number(a.first()?)?, number(a.get(1)?)?));
    }
    Some((number(v.get("x")?)?, number(v.get("y")?)?))
}

fn button_of(args: &Value) -> Result<Button, String> {
    match args.get("button").and_then(Value::as_str).map(str::to_ascii_lowercase).as_deref() {
        None | Some("left") => Ok(Button::Left),
        Some("right") => Ok(Button::Right),
        Some("middle") => Ok(Button::Middle),
        Some(other) => Err(format!("button '{other}' is not left, right or middle")),
    }
}

fn current_map() -> Result<ShotMap, String> {
    state().last_shot.clone().ok_or_else(|| "take a screenshot first (computer_screenshot): coordinates refer to the last screenshot you took".to_owned())
}

fn target(args: &Value) -> Result<(i32, i32, f64, f64), String> {
    let x = args.get("x").and_then(number);
    let y = args.get("y").and_then(number);
    let (Some(x), Some(y)) = (x, y) else { return Err("x and y are needed (pixels of your last screenshot)".to_owned()) };
    let (gx, gy) = current_map()?.to_input(x, y)?;
    Ok((gx, gy, x, y))
}

fn encode_png(frame: &Frame, target_w: u32, target_h: u32) -> Result<Picture, String> {
    use image::{codecs::png::PngEncoder, imageops, ImageEncoder, RgbaImage};
    let source = RgbaImage::from_raw(frame.width, frame.height, frame.rgba.clone()).ok_or("the captured picture has the wrong size")?;
    let picture = if (target_w, target_h) == (frame.width, frame.height) { source } else { imageops::resize(&source, target_w, target_h, imageops::FilterType::Lanczos3) };
    // opaque RGB is smaller than RGBA and the screen has no transparency
    let rgb = image::DynamicImage::ImageRgba8(picture).to_rgb8();
    let mut png = Vec::new();
    PngEncoder::new_with_quality(&mut png, image::codecs::png::CompressionType::Fast, image::codecs::png::FilterType::Adaptive)
        .write_image(rgb.as_raw(), rgb.width(), rgb.height(), image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("cannot encode the screenshot: {e}"))?;
    Ok(Picture { png, width: rgb.width(), height: rgb.height() })
}

fn screenshot(args: &Value) -> Result<Outcome, String> {
    let displays = os::displays()?;
    if displays.is_empty() {
        return Err("no display found".to_owned());
    }
    let index = match args.get("display").and_then(Value::as_u64) {
        Some(i) => i as usize,
        None => displays.iter().position(|d| d.primary).unwrap_or(0),
    };
    let display = displays.get(index).ok_or_else(|| format!("there is no display {index}; there are {} (0 to {})", displays.len(), displays.len() - 1))?;
    let previous = state().last_shot.clone();
    let region = match args.get("region").filter(|r| !r.is_null()) {
        None => (0, 0, display.width, display.height),
        Some(r) => {
            let get = |k: &str| r.get(k).and_then(number);
            let (Some(x), Some(y), Some(w), Some(h)) = (get("x"), get("y"), get("width"), get("height")) else {
                return Err("region needs x, y, width and height".to_owned());
            };
            region_to_physical(previous.as_ref(), display, (x, y, w, h))?
        }
    };
    let frame = os::capture(display, region)?;
    let (iw, ih) = fit_size(frame.width, frame.height, MAX_SIDE);
    let picture = encode_png(&frame, iw, ih)?;
    let map = ShotMap {
        display: display.index,
        origin_x: display.origin_x,
        origin_y: display.origin_y,
        scale: display.scale,
        display_w: display.width,
        display_h: display.height,
        region,
        image_w: picture.width,
        image_h: picture.height,
    };
    state().last_shot = Some(map);
    let part = if region == (0, 0, display.width, display.height) { String::new() } else { format!(", the part at {},{} of size {}x{} px", region.0, region.1, region.2, region.3) };
    let text = format!(
        "Screenshot of display {} \"{}\" ({}x{} px{part}), shown as {}x{} px. Coordinates for the other computer tools are pixels of this picture; (0,0) is its top-left corner.{}",
        display.index,
        display.name,
        display.width,
        display.height,
        picture.width,
        picture.height,
        if displays.len() > 1 { format!(" There are {} displays; pass display to look at another one.", displays.len()) } else { String::new() }
    );
    Ok(Outcome { text, picture: Some(picture) })
}

fn click(args: &Value, cancel: Cancel) -> Result<Outcome, String> {
    let (gx, gy, x, y) = target(args)?;
    let button = button_of(args)?;
    let double = args.get("double").and_then(Value::as_bool).unwrap_or(false);
    os::move_to(gx, gy)?;
    if !cancel.sleep(40) {
        return Err("stopped by the user".to_owned());
    }
    for round in 0..if double { 2 } else { 1 } {
        os::button(button, true)?;
        os::button(button, false)?;
        if round == 0 && double && !cancel.sleep(70) {
            return Err("stopped by the user".to_owned());
        }
    }
    Ok(Outcome::text(format!(
        "{} at ({}, {}). The screen may need a moment to change; take a screenshot to see the result.",
        match (button, double) {
            (Button::Left, false) => "Clicked",
            (Button::Left, true) => "Double-clicked",
            (Button::Right, false) => "Right-clicked",
            (Button::Right, true) => "Right-double-clicked",
            (Button::Middle, _) => "Middle-clicked",
        },
        round1(x),
        round1(y)
    )))
}

fn drag(args: &Value, cancel: Cancel) -> Result<Outcome, String> {
    let map = current_map()?;
    let from = args.get("from").and_then(point_of).ok_or("from needs x and y")?;
    let to = args.get("to").and_then(point_of).ok_or("to needs x and y")?;
    let (fx, fy) = map.to_input(from.0, from.1)?;
    let (tx, ty) = map.to_input(to.0, to.1)?;
    let button = button_of(args)?;
    os::move_to(fx, fy)?;
    if !cancel.sleep(60) {
        return Err("stopped by the user".to_owned());
    }
    os::button(button, true)?;
    let mut result = Ok(());
    const STEPS: i32 = 14;
    for i in 1..=STEPS {
        if cancel.cancelled() {
            result = Err("stopped by the user".to_owned());
            break;
        }
        let px = fx + (tx - fx) * i / STEPS;
        let py = fy + (ty - fy) * i / STEPS;
        if let Err(e) = os::move_to(px, py) {
            result = Err(e);
            break;
        }
        std::thread::sleep(Duration::from_millis(18));
    }
    // the button is always released, also after an error or a stop
    let released = os::button(button, false);
    result?;
    released?;
    Ok(Outcome::text(format!("Dragged from ({}, {}) to ({}, {}). Take a screenshot to see the result.", round1(from.0), round1(from.1), round1(to.0), round1(to.1))))
}

fn type_text(args: &Value, cancel: Cancel) -> Result<Outcome, String> {
    let text = args.get("text").and_then(Value::as_str).ok_or("text is missing")?;
    let count = text.chars().count();
    if count == 0 {
        return Err("text is empty".to_owned());
    }
    if count > MAX_TYPE_CHARS {
        return Err(format!("at most {MAX_TYPE_CHARS} characters per call, this text has {count}"));
    }
    current_map()?;
    let mut typed = 0usize;
    let mut skipped: Vec<char> = Vec::new();
    for c in text.chars() {
        if cancel.cancelled() {
            return Err(format!("stopped by the user after {typed} of {count} characters"));
        }
        let outcome = match c {
            '\r' => continue,
            '\n' => os::press(&Chord { ctrl: false, alt: false, shift: false, meta: false, key: Key::Named(Named::Enter) }),
            '\t' => os::press(&Chord { ctrl: false, alt: false, shift: false, meta: false, key: Key::Named(Named::Tab) }),
            c => os::type_char(c),
        };
        match outcome {
            Ok(()) => typed += 1,
            Err(e) if e.starts_with("cannot type") => {
                if !skipped.contains(&c) {
                    skipped.push(c);
                }
            }
            Err(e) => return Err(format!("{e} (after {typed} characters)")),
        }
        std::thread::sleep(Duration::from_millis(4));
    }
    let mut out = format!("Typed {typed} characters.");
    if !skipped.is_empty() {
        let list: String = skipped.iter().collect();
        out.push_str(&format!(" These could not be typed with the keyboard layout of this computer and were left out: {list}"));
    }
    Ok(Outcome::text(out))
}

fn keys(args: &Value, cancel: Cancel) -> Result<Outcome, String> {
    let text = args.get("keys").and_then(Value::as_str).ok_or("keys is missing")?;
    let chords = parse_keys(text)?;
    for chord in &chords {
        if let Some(why) = dangerous_chord(chord, std::env::consts::OS) {
            return Err(format!("refused: {why}. Nothing was pressed."));
        }
    }
    current_map()?;
    for (i, chord) in chords.iter().enumerate() {
        if cancel.cancelled() {
            return Err("stopped by the user".to_owned());
        }
        os::press(chord)?;
        if i + 1 < chords.len() && !cancel.sleep(60) {
            return Err("stopped by the user".to_owned());
        }
    }
    Ok(Outcome::text(format!("Pressed {}.", text.trim())))
}

fn scroll(args: &Value) -> Result<Outcome, String> {
    let (gx, gy, x, y) = target(args)?;
    let get = |k: &str| args.get(k).and_then(number).unwrap_or(0.0).round().clamp(-50.0, 50.0) as i32;
    let (dx, dy) = (get("dx"), get("dy"));
    if dx == 0 && dy == 0 {
        return Err("dx or dy is needed (wheel notches; dy > 0 scrolls down)".to_owned());
    }
    os::move_to(gx, gy)?;
    std::thread::sleep(Duration::from_millis(40));
    os::scroll(dx, dy)?;
    Ok(Outcome::text(format!("Scrolled {dy} notches down and {dx} right at ({}, {}). Take a screenshot to see the result.", round1(x), round1(y))))
}

fn wait(args: &Value, cancel: Cancel) -> Result<Outcome, String> {
    let ms = args.get("ms").and_then(number).unwrap_or(1000.0).clamp(0.0, MAX_WAIT_MS as f64) as u64;
    if !cancel.sleep(ms) {
        return Err("stopped by the user".to_owned());
    }
    Ok(Outcome::text(format!("Waited {ms} ms.")))
}

fn cursor() -> Result<Outcome, String> {
    let (gx, gy) = os::cursor()?;
    let inside = state().last_shot.as_ref().and_then(|m| m.to_image(gx, gy));
    Ok(Outcome::text(match inside {
        Some((x, y)) => format!("The pointer is at ({}, {}) in your last screenshot (screen position {gx}, {gy}).", x.round(), y.round()),
        None => format!("The pointer is at screen position {gx}, {gy}, which is outside your last screenshot (or you have none yet)."),
    }))
}

fn displays_text() -> Result<Outcome, String> {
    let list = os::displays()?;
    let mut out = format!("{} display{}:\n", list.len(), if list.len() == 1 { "" } else { "s" });
    for d in &list {
        out.push_str(&format!(
            "{}: \"{}\"{} {}x{} px at {},{}; scale {}; text scaling {}%\n",
            d.index,
            d.name,
            if d.primary { " (main)" } else { "" },
            d.width,
            d.height,
            d.origin_x,
            d.origin_y,
            d.scale,
            (d.ui_scale * 100.0).round()
        ));
    }
    Ok(Outcome::text(out.trim_end()))
}

/// Runs one computer tool (blocking; the page asked the user before it got here).
pub(super) fn run(name: &str, args: &Value) -> Result<Outcome, String> {
    gate()?;
    ensure_watcher();
    let cancel = Cancel::now();
    let _one_at_a_time = ACTION.lock().unwrap_or_else(|e| e.into_inner());
    if cancel.cancelled() {
        return Err("stopped by the user".to_owned());
    }
    let counted = !matches!(name, "computer_displays" | "computer_cursor" | "computer_wait");
    if counted {
        admit(&mut state(), now_ms(), super::studio_mode::computer_rate(), super::studio_mode::computer_max_steps())?;
    }
    match name {
        "computer_screenshot" => screenshot(args),
        "computer_click" => click(args, cancel),
        "computer_move" => {
            let (gx, gy, x, y) = target(args)?;
            os::move_to(gx, gy)?;
            Ok(Outcome::text(format!("Moved the pointer to ({}, {}).", round1(x), round1(y))))
        }
        "computer_drag" => drag(args, cancel),
        "computer_type" => type_text(args, cancel),
        "computer_key" => keys(args, cancel),
        "computer_scroll" => scroll(args),
        "computer_wait" => wait(args, cancel),
        "computer_cursor" => cursor(),
        "computer_displays" => displays_text(),
        other => Err(format!("unknown tool '{other}'")),
    }
}

// ---- stop, status, watcher ------------------------------------------------------------------------

/// Cancels the running action and switches the permission off. Called by the STOP button and by the panic chord.
pub(super) fn stop() {
    STOP_EPOCH.fetch_add(1, Ordering::SeqCst);
    let _ = super::studio_mode::set_computer_use(false);
    let mut s = state();
    s.steps = 0;
    s.last_shot = None;
}

/// A new task starts: the step count and the remembered screenshot are cleared.
pub(super) fn begin() {
    let mut s = state();
    s.steps = 0;
    s.last_shot = None;
}

/// Polls the global panic chord (Ctrl+Alt+Shift+Esc) while computer use is on.
fn ensure_watcher() {
    if WATCHER.swap(true, Ordering::SeqCst) {
        return;
    }
    let spawned = std::thread::Builder::new().name("computer-use-panic".to_owned()).spawn(|| {
        let mut held = false;
        loop {
            if !super::studio_mode::computer_use() {
                held = false;
                std::thread::sleep(Duration::from_millis(500));
                continue;
            }
            let down = os::panic_down();
            if down && !held {
                stop();
            }
            held = down;
            std::thread::sleep(Duration::from_millis(60));
        }
    });
    if spawned.is_err() {
        WATCHER.store(false, Ordering::SeqCst);
    }
}

/// What the page needs to know.
pub(super) fn status_json() -> Value {
    let availability = os::available();
    let (steps, shot) = {
        let s = state();
        (s.steps, s.last_shot.is_some())
    };
    json!({
        "compiled": cfg!(feature = "computer-use"),
        "enabled": enabled(),
        "forbidden": forbidden_by_env(),
        "available": availability.is_ok(),
        "reason": availability.err(),
        "platform": os::platform(),
        "rate_per_minute": super::studio_mode::computer_rate(),
        "max_steps": super::studio_mode::computer_max_steps(),
        "steps_used": steps,
        "has_screenshot": shot,
        "panic_chord": "Ctrl+Alt+Shift+Esc",
        "max_side": MAX_SIDE,
    })
}

/// `GET /api/studio/agent/computer`.
pub(in crate::server) async fn route_status() -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let status = tokio::task::spawn_blocking(|| {
        if super::studio_mode::computer_use() {
            ensure_watcher();
        }
        status_json()
    })
    .await
    .unwrap_or_else(|_| json!({}));
    Json(status).into_response()
}

/// `POST /api/studio/agent/computer/stop`: cancels the running action and switches computer use off.
pub(in crate::server) async fn route_stop() -> Response {
    stop();
    Json(status_json()).into_response()
}

/// `POST /api/studio/agent/computer/begin`: a new task starts.
pub(in crate::server) async fn route_begin() -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    begin();
    Json(json!({"ok": true})).into_response()
}

/// The JSON a tool call answers with: `{ok, output, image?}`.
pub(super) fn outcome_json(result: Result<Outcome, String>) -> Value {
    use base64::Engine;
    match result {
        Ok(o) => {
            let mut v = json!({"ok": true, "output": o.text});
            if let Some(p) = o.picture {
                v["image"] = json!({"mime": "image/png", "data": base64::engine::general_purpose::STANDARD.encode(&p.png), "width": p.width, "height": p.height});
            }
            v
        }
        Err(error) => json!({"ok": false, "error": error}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn display(index: usize, x: i32, y: i32, w: u32, h: u32, scale: f64) -> DisplayInfo {
        DisplayInfo { index, name: format!("D{index}"), primary: index == 0, origin_x: x, origin_y: y, width: w, height: h, scale, ui_scale: scale }
    }

    fn map_of(d: &DisplayInfo, region: (u32, u32, u32, u32)) -> ShotMap {
        let (iw, ih) = fit_size(region.2, region.3, MAX_SIDE);
        ShotMap { display: d.index, origin_x: d.origin_x, origin_y: d.origin_y, scale: d.scale, display_w: d.width, display_h: d.height, region, image_w: iw, image_h: ih }
    }

    #[test]
    fn fit_size_shrinks_the_long_side_and_never_enlarges() {
        assert_eq!(fit_size(1920, 1080, 1568), (1568, 882));
        assert_eq!(fit_size(3840, 2160, 1568), (1568, 882));
        assert_eq!(fit_size(1000, 700, 1568), (1000, 700));
        assert_eq!(fit_size(1080, 1920, 1568), (882, 1568));
        assert_eq!(fit_size(1568, 1568, 1568), (1568, 1568));
        assert_eq!(fit_size(0, 0, 1568), (1, 1));
    }

    #[test]
    fn a_full_size_screenshot_maps_one_to_one() {
        let d = display(0, 0, 0, 1280, 720, 1.0);
        let m = map_of(&d, (0, 0, 1280, 720));
        assert_eq!((m.image_w, m.image_h), (1280, 720));
        assert_eq!(m.to_input(0.0, 0.0).unwrap(), (0, 0));
        assert_eq!(m.to_input(640.0, 360.0).unwrap(), (640, 360));
        assert_eq!(m.to_input(1280.0, 720.0).unwrap(), (1279, 719), "the far edge is clamped onto the last pixel");
    }

    #[test]
    fn a_downscaled_screenshot_maps_back_to_physical_pixels() {
        let d = display(0, 0, 0, 3840, 2160, 1.0);
        let m = map_of(&d, (0, 0, 3840, 2160));
        assert_eq!((m.image_w, m.image_h), (1568, 882));
        let (x, y) = m.to_input(784.0, 441.0).unwrap();
        assert!((x - 1920).abs() <= 1 && (y - 1080).abs() <= 1, "{x},{y}");
        let back = m.to_image(x, y).unwrap();
        assert!((back.0 - 784.0).abs() < 1.5 && (back.1 - 441.0).abs() < 1.5, "{back:?}");
    }

    #[test]
    fn a_retina_display_gives_input_units_in_points() {
        // 2880x1800 pixels, 2 pixels per point: the mouse works in points (1440x900)
        let d = display(0, 0, 0, 2880, 1800, 2.0);
        let m = map_of(&d, (0, 0, 2880, 1800));
        assert_eq!((m.image_w, m.image_h), (1568, 980));
        let (x, y) = m.to_input(784.0, 490.0).unwrap();
        assert!((x - 720).abs() <= 1 && (y - 450).abs() <= 1, "{x},{y}");
        let (x, _) = m.to_input(1568.0, 0.0).unwrap();
        assert_eq!(x, 1439, "the right edge is the last point");
    }

    #[test]
    fn a_second_monitor_with_a_negative_origin_adds_its_offset() {
        // a monitor to the left of the main one and a little higher
        let left = display(1, -1280, -200, 1280, 720, 1.0);
        let m = map_of(&left, (0, 0, 1280, 720));
        assert_eq!(m.to_input(0.0, 0.0).unwrap(), (-1280, -200));
        assert_eq!(m.to_input(100.0, 50.0).unwrap(), (-1180, -150));
        let right = display(2, 2560, 0, 1280, 1024, 1.0);
        let m = map_of(&right, (0, 0, 1280, 1024));
        assert_eq!(m.to_input(10.0, 10.0).unwrap(), (2570, 10));
    }

    #[test]
    fn a_mixed_dpi_monitor_uses_its_own_scale() {
        // a 4K monitor at 150 % on Windows: input units are physical pixels, the scale stays 1 for the mapping
        let d = display(1, 1920, 0, 3840, 2160, 1.0);
        let m = map_of(&d, (0, 0, 3840, 2160));
        let (x, y) = m.to_input(1568.0 / 2.0, 882.0 / 2.0).unwrap();
        assert!((x - (1920 + 1920)).abs() <= 1 && (y - 1080).abs() <= 1, "{x},{y}");
    }

    #[test]
    fn a_zoomed_region_maps_into_the_region_only() {
        let d = display(0, 0, 0, 1920, 1080, 1.0);
        let m = map_of(&d, (960, 540, 400, 300));
        assert_eq!((m.image_w, m.image_h), (400, 300));
        assert_eq!(m.to_input(0.0, 0.0).unwrap(), (960, 540));
        assert_eq!(m.to_input(200.0, 150.0).unwrap(), (1160, 690));
        assert!(m.to_image(100, 100).is_none(), "outside the region");
        assert_eq!(m.to_image(1160, 690).unwrap(), (200.0, 150.0));
    }

    #[test]
    fn points_outside_the_screenshot_are_refused_with_the_size() {
        let d = display(0, 0, 0, 800, 600, 1.0);
        let m = map_of(&d, (0, 0, 800, 600));
        for (x, y) in [(-1.0, 5.0), (5.0, -0.5), (801.0, 10.0), (10.0, 601.0), (f64::NAN, 1.0), (1.0, f64::INFINITY)] {
            assert!(m.to_input(x, y).is_err(), "{x},{y}");
        }
        assert!(m.to_input(800.0, 600.0).is_ok());
        let e = m.to_input(900.0, 10.0).unwrap_err();
        assert!(e.contains("800x600"), "{e}");
    }

    #[test]
    fn a_region_is_named_in_the_coordinates_of_the_last_screenshot() {
        let d = display(0, 0, 0, 3840, 2160, 1.0);
        let prev = map_of(&d, (0, 0, 3840, 2160)); // shown as 1568x882
        // the model asks for the top-left quarter of what it saw
        let r = region_to_physical(Some(&prev), &d, (0.0, 0.0, 784.0, 441.0)).unwrap();
        assert!((i64::from(r.2) - 1920).abs() <= 2 && (i64::from(r.3) - 1080).abs() <= 2, "{r:?}");
        assert_eq!((r.0, r.1), (0, 0));
        // without a screenshot the numbers are display pixels
        assert_eq!(region_to_physical(None, &d, (100.0, 200.0, 300.0, 400.0)).unwrap(), (100, 200, 300, 400));
        // a screenshot of another display does not apply
        let other = display(1, 3840, 0, 1920, 1080, 1.0);
        assert_eq!(region_to_physical(Some(&prev), &other, (10.0, 10.0, 100.0, 100.0)).unwrap(), (10, 10, 100, 100));
        // clipped to the display, refused when nothing is left
        assert_eq!(region_to_physical(None, &d, (3800.0, 2100.0, 500.0, 500.0)).unwrap(), (3800, 2100, 40, 60));
        assert!(region_to_physical(None, &d, (5000.0, 0.0, 10.0, 10.0)).is_err());
        assert!(region_to_physical(None, &d, (0.0, 0.0, 0.0, 10.0)).is_err());
        assert!(region_to_physical(None, &d, (f64::NAN, 0.0, 10.0, 10.0)).is_err());
    }

    fn chord(text: &str) -> Chord {
        parse_chord(text).unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    #[test]
    fn chords_parse_with_modifiers_names_and_aliases() {
        assert_eq!(chord("ctrl+s"), Chord { ctrl: true, alt: false, shift: false, meta: false, key: Key::Char('s') });
        assert_eq!(chord("Ctrl+Shift+T"), Chord { ctrl: true, alt: false, shift: true, meta: false, key: Key::Char('t') });
        assert_eq!(chord("cmd+alt+left"), Chord { ctrl: false, alt: true, shift: false, meta: true, key: Key::Named(Named::Left) });
        assert_eq!(chord("Enter").key, Key::Named(Named::Enter));
        assert_eq!(chord("return").key, Key::Named(Named::Enter));
        assert_eq!(chord("esc").key, Key::Named(Named::Escape));
        assert_eq!(chord("f5").key, Key::Named(Named::F(5)));
        assert_eq!(chord("alt+F4").key, Key::Named(Named::F(4)));
        assert_eq!(chord("control+option+delete").key, Key::Named(Named::Delete));
        assert_eq!(chord("win+r"), Chord { ctrl: false, alt: false, shift: false, meta: true, key: Key::Char('r') });
        assert_eq!(chord("super").key, Key::Named(Named::Meta));
        assert_eq!(chord("A"), Chord { ctrl: false, alt: false, shift: true, meta: false, key: Key::Char('a') });
        assert_eq!(chord("ctrl++"), Chord { ctrl: true, alt: false, shift: false, meta: false, key: Key::Char('+') });
        assert_eq!(chord("+").key, Key::Char('+'));
        assert_eq!(chord("ctrl+/").key, Key::Char('/'));
        assert_eq!(chord("space").key, Key::Named(Named::Space));
    }

    #[test]
    fn bad_chords_are_explained() {
        for bad in ["", "   ", "ctrl+", "ctrl+shift", "ctrl+alt", "ctrl+foo", "banana+a", "f0", "f25", "pageupp", "ab"] {
            assert!(parse_chord(bad).is_err(), "'{bad}' should not parse");
        }
        let e = parse_chord("ctrl+shift").unwrap_err();
        assert!(e.contains("names no key"), "{e}");
        assert!(parse_keys("").is_err());
        assert!(parse_keys(&"a ".repeat(11)).is_err());
        assert_eq!(parse_keys("ctrl+a ctrl+c").unwrap().len(), 2);
    }

    #[test]
    fn the_dangerous_chords_are_refused() {
        let hits = |text: &str, os: &str| dangerous_chord(&chord(text), os).is_some();
        for os in ["windows", "linux", "macos"] {
            assert!(hits("ctrl+alt+delete", os), "{os}");
            assert!(hits("ctrl+alt+backspace", os));
            assert!(hits("ctrl+shift+esc", os));
            assert!(hits("ctrl+alt+shift+esc", os));
            assert!(hits("alt+f4", os));
            assert!(hits("ctrl+alt+f1", os));
            assert!(hits("ctrl+alt+f12", os));
        }
        assert!(hits("win+l", "windows") && hits("win+r", "windows") && hits("win+x", "windows"));
        assert!(hits("super+l", "linux") && !hits("super+r", "linux"));
        assert!(hits("cmd+ctrl+q", "macos") && hits("cmd+shift+q", "macos") && hits("cmd+alt+esc", "macos"));
        // everyday chords stay possible
        for text in ["ctrl+s", "ctrl+c", "ctrl+v", "enter", "esc", "tab", "alt+tab", "alt+left", "f5", "ctrl+shift+t", "ctrl+alt+t", "win", "ctrl+f4", "f4", "ctrl+alt+left"] {
            assert!(!hits(text, "windows") && !hits(text, "linux") && !hits(text, "macos"), "{text}");
        }
        assert!(!hits("cmd+l", "macos"), "Command+L is the address bar of a browser on a Mac");
        assert!(!hits("cmd+q", "macos"), "quitting one program is an ordinary action");
    }

    #[test]
    fn the_rate_limiter_allows_n_actions_per_minute() {
        let mut r = RateLimiter::new();
        for i in 0..5 {
            assert!(r.check(1000 + i * 10, 5).is_ok());
        }
        let wait = r.check(1100, 5).unwrap_err();
        assert!((58_000..=60_000).contains(&wait), "{wait}");
        // a refused action is not counted, and after a minute the oldest ones fall out
        for t in [61_000, 61_010, 61_020, 61_030, 61_040] {
            assert!(r.check(t, 5).is_ok(), "{t}");
        }
        assert!(r.check(61_041, 5).is_err());
        // a limit of 0 still allows one action per minute instead of locking everything
        let mut z = RateLimiter::new();
        assert!(z.check(1, 0).is_ok());
        assert!(z.check(2, 0).is_err());
    }

    #[test]
    fn admission_stops_at_the_step_cap_and_after_idle_it_starts_again() {
        let mut s = State { limiter: RateLimiter::new(), steps: 0, last_action_ms: 0, last_shot: None };
        for i in 0..3 {
            admit(&mut s, 1000 + i, 100, 3).unwrap();
        }
        let e = admit(&mut s, 1010, 100, 3).unwrap_err();
        assert!(e.contains("step cap") && e.contains('3'), "{e}");
        assert_eq!(s.steps, 3);
        // the same task after a long pause is a new task
        admit(&mut s, 1010 + IDLE_RESET_MS + 5, 100, 3).unwrap();
        assert_eq!(s.steps, 1);
        // the rate limit message says how long to wait
        let mut r = State { limiter: RateLimiter::new(), steps: 0, last_action_ms: 0, last_shot: None };
        admit(&mut r, 1, 1, 100).unwrap();
        let e = admit(&mut r, 2, 1, 100).unwrap_err();
        assert!(e.contains("rate limit") && e.contains("60 seconds"), "{e}");
        assert_eq!(r.steps, 1, "a refused action is not a step");
    }

    #[test]
    fn the_schemas_are_complete_and_in_the_computer_category() {
        let tools = catalog();
        let names: Vec<&str> = tools.iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
        for expected in ["computer_screenshot", "computer_click", "computer_move", "computer_drag", "computer_type", "computer_key", "computer_scroll", "computer_wait", "computer_cursor", "computer_displays"] {
            assert!(names.contains(&expected), "{expected} missing");
        }
        for t in &tools {
            let name = t["function"]["name"].as_str().unwrap();
            assert_eq!(t["permission"], "computer", "{name}");
            assert!(is_computer_tool(name));
            assert_eq!(t["type"], "function");
            assert!(t["function"]["description"].as_str().unwrap().len() > 20);
            let props = t["function"]["parameters"]["properties"].as_object().unwrap();
            for required in t["function"]["parameters"]["required"].as_array().unwrap() {
                assert!(props.contains_key(required.as_str().unwrap()), "{name}: required argument without a property");
            }
        }
        let click = tools.iter().find(|t| t["function"]["name"] == "computer_click").unwrap();
        assert_eq!(click["function"]["parameters"]["required"], json!(["x", "y"]));
        let drag = tools.iter().find(|t| t["function"]["name"] == "computer_drag").unwrap();
        assert_eq!(drag["function"]["parameters"]["required"], json!(["from", "to"]));
        assert!(!is_computer_tool("bash") && !is_computer_tool("read_file"));
    }

    #[test]
    fn points_come_as_objects_or_pairs_and_numbers_may_be_strings() {
        assert_eq!(point_of(&json!({"x": 3, "y": 4.5})), Some((3.0, 4.5)));
        assert_eq!(point_of(&json!([10, 20])), Some((10.0, 20.0)));
        assert_eq!(point_of(&json!({"x": "12", "y": " 7 "})), Some((12.0, 7.0)));
        assert_eq!(point_of(&json!({"x": 1})), None);
        assert_eq!(point_of(&json!("nope")), None);
        assert_eq!(button_of(&json!({})).unwrap(), Button::Left);
        assert_eq!(button_of(&json!({"button": "Right"})).unwrap(), Button::Right);
        assert!(button_of(&json!({"button": "fourth"})).is_err());
    }

    #[test]
    fn a_picture_reply_carries_the_png_as_base64() {
        let png = vec![0x89, b'P', b'N', b'G'];
        let v = outcome_json(Ok(Outcome { text: "ok".into(), picture: Some(Picture { png, width: 2, height: 1 }) }));
        assert_eq!(v["ok"], true);
        assert_eq!(v["image"]["mime"], "image/png");
        assert_eq!(v["image"]["data"], "iVBORw==");
        assert_eq!(v["image"]["width"], 2);
        let e = outcome_json(Err("no".into()));
        assert_eq!(e["ok"], false);
        assert!(e.get("image").is_none());
    }

    #[test]
    fn a_captured_frame_becomes_a_png_of_the_requested_size() {
        let frame = Frame { width: 4, height: 2, rgba: vec![200; 4 * 2 * 4] };
        let p = encode_png(&frame, 2, 1).unwrap();
        assert_eq!((p.width, p.height), (2, 1));
        assert_eq!(&p.png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        assert!(encode_png(&Frame { width: 4, height: 2, rgba: vec![0; 3] }, 4, 2).is_err());
    }

    #[test]
    fn the_stop_epoch_cancels_a_running_action() {
        let c = Cancel::now();
        assert!(!c.cancelled());
        assert!(c.sleep(5));
        STOP_EPOCH.fetch_add(1, Ordering::SeqCst);
        assert!(c.cancelled());
        assert!(!c.sleep(50));
        assert!(!Cancel::now().cancelled());
    }
}
