//! Row 86 of PARITY.md: *Computer use* in the Code page (`computer.js`) — the agent looks at the
//! screen and operates the mouse and keyboard of this computer, only after the user switched it
//! on in Settings > Runtime and with a question before every action.
//!
//! This file is the part that can be compared with the page case by case
//! (`tests/computer_fixtures.rs`: the page's own `@pure` block run in a vm): the lines that name
//! an action, the questions, the marks drawn on a screenshot, the pictures kept in a
//! conversation and the panic keys. The window's state (the banner, the poll, the cards) is in
//! `code.rs`.

use serde_json::{json, Value};

/// `String(value)` of JavaScript for a JSON value (`null` is `"null"`; arrays join with commas
/// and read `null` as nothing).
pub fn js_string(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => crate::loopguard::js_number(n.as_f64().unwrap_or(f64::NAN)),
        Value::String(s) => s.clone(),
        Value::Array(items) => items.iter().map(|i| if i.is_null() { String::new() } else { js_string(i) }).collect::<Vec<_>>().join(","),
        Value::Object(_) => "[object Object]".into(),
    }
}

/// `String(x ?? '')`: nothing for a missing or `null` value.
pub fn text_or_empty(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(other) => js_string(other),
    }
}

/// `Number(value)` of JavaScript; `None` is `undefined`.
pub fn js_number_of(v: Option<&Value>) -> f64 {
    match v {
        None => f64::NAN,
        Some(Value::Null) => 0.0,
        Some(Value::Bool(b)) => f64::from(u8::from(*b)),
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
        Some(Value::String(s)) => string_number(s),
        Some(Value::Array(_)) => string_number(&js_string(v.unwrap_or(&Value::Null))),
        Some(Value::Object(_)) => f64::NAN,
    }
}

fn string_number(s: &str) -> f64 {
    // the whitespace of JavaScript's `Number("  ")`: an empty text is zero
    let t = s.trim_matches(|c: char| c.is_whitespace() || c == '\u{feff}');
    if t.is_empty() {
        0.0
    } else {
        crate::specs::js_number(t)
    }
}

/// `Math.round`: halves go up.
fn round(x: f64) -> f64 {
    (x + 0.5).floor()
}

/// `cuNum(v)`: a finite number rounded to a tenth, anything else zero.
pub fn cu_num(v: Option<&Value>) -> f64 {
    let n = js_number_of(v);
    if n.is_finite() { round(n * 10.0) / 10.0 } else { 0.0 }
}

fn num_text(f: f64) -> String {
    crate::loopguard::js_number(f)
}

/// `p.x ?? p[0]` and its twin: a member by name, else by position.
fn member<'a>(p: &'a Value, name: &str, index: usize) -> Option<&'a Value> {
    let named = match p {
        Value::Object(map) => map.get(name),
        _ => None,
    }
    .filter(|v| !v.is_null());
    named.or_else(|| match p {
        Value::Array(items) => items.get(index),
        Value::Object(map) => map.get(&index.to_string()),
        _ => None,
    })
}

/// `cuPt(p)`: `x,y` of a point, `?` for anything that is no object.
pub fn cu_pt(p: Option<&Value>) -> String {
    match p {
        Some(v @ (Value::Object(_) | Value::Array(_))) => format!("{},{}", num_text(cu_num(member(v, "x", 0))), num_text(cu_num(member(v, "y", 1)))),
        _ => "?".into(),
    }
}

/// JavaScript truthiness of a member of the arguments.
fn truthy(v: Option<&Value>) -> bool {
    crate::jobs::truthy(v)
}

/// `computerSubject(name, args)`: the one line that names an action; the permission rules
/// (`computer: computer_click*`) match against it.
pub fn computer_subject(name: &str, args: &Value) -> String {
    let a = |k: &str| args.get(k);
    let xy = |k1: &str, k2: &str| format!("{},{}", num_text(cu_num(a(k1))), num_text(cu_num(a(k2))));
    match name {
        "computer_click" => {
            let button = match a("button") {
                Some(b) if truthy(Some(b)) && b.as_str() != Some("left") => format!(" {}", js_string(b)),
                _ => String::new(),
            };
            format!("{name} {}{}{}", xy("x", "y"), button, if truthy(a("double")) { " double" } else { "" })
        }
        "computer_move" => format!("{name} {}", xy("x", "y")),
        "computer_drag" => format!("{name} {} -> {}", cu_pt(a("from")), cu_pt(a("to"))),
        "computer_type" => format!("{name} {}", serde_json::to_string(&text_or_empty(a("text"))).unwrap_or_default()),
        "computer_key" => format!("{name} {}", text_or_empty(a("keys"))),
        "computer_scroll" => format!("{name} {} dx={} dy={}", xy("x", "y"), num_text(cu_num(a("dx"))), num_text(cu_num(a("dy")))),
        "computer_wait" => {
            let ms = match a("ms") {
                None | Some(Value::Null) => 1000.0,
                other => cu_num(other),
            };
            format!("{name} {}", num_text(ms))
        }
        _ => name.to_string(),
    }
}

/// The first `n` UTF-16 units of a text (`String.prototype.slice(0, n)`).
pub fn slice_units(text: &str, n: usize) -> String {
    let mut used = 0;
    let mut out = String::new();
    for c in text.chars() {
        used += c.len_utf16();
        if used > n {
            break;
        }
        out.push(c);
    }
    out
}

/// `computerQuestionText(name, args)`: the question on the permission card.
pub fn computer_question(name: &str, args: &Value) -> String {
    let a = |k: &str| args.get(k);
    match name {
        "computer_screenshot" => "Let the agent look at your screen?".into(),
        "computer_click" => {
            let verb = if truthy(a("double")) {
                "Double-click".to_string()
            } else {
                match a("button") {
                    Some(b) if truthy(Some(b)) && b.as_str() != Some("left") => format!("{}-click", js_string(b)),
                    _ => "Click".to_string(),
                }
            };
            format!("{verb} at the marked point?")
        }
        "computer_move" => "Move the mouse pointer to the marked point?".into(),
        "computer_drag" => "Drag with the mouse along the marked line?".into(),
        "computer_type" => {
            let text = text_or_empty(a("text"));
            let shown = serde_json::to_string(&slice_units(&text, 80)).unwrap_or_default();
            format!("Type {shown}{} on the keyboard?", if text.encode_utf16().count() > 80 { " ..." } else { "" })
        }
        "computer_key" => format!("Press {}?", text_or_empty(a("keys"))),
        "computer_scroll" => "Scroll at the marked point?".into(),
        _ => "Allow this?".into(),
    }
}

/// A mark drawn over a screenshot, in the pixels of that screenshot.
#[derive(Debug, Clone, PartialEq)]
pub enum Mark {
    Cross { x: f64, y: f64, double: Option<bool> },
    Arrow { x: f64, y: f64, dx: f64, dy: f64 },
    Line { x1: f64, y1: f64, x2: f64, y2: f64 },
}

impl Mark {
    /// The shape the page's object has (`{ kind, x, y, double }`…), for the comparison.
    pub fn to_json(&self) -> Value {
        match self {
            Mark::Cross { x, y, double: Some(d) } => json!({ "kind": "cross", "x": x, "y": y, "double": d }),
            Mark::Cross { x, y, double: None } => json!({ "kind": "cross", "x": x, "y": y }),
            Mark::Arrow { x, y, dx, dy } => json!({ "kind": "arrow", "x": x, "y": y, "dx": dx, "dy": dy }),
            Mark::Line { x1, y1, x2, y2 } => json!({ "kind": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2 }),
        }
    }
}

/// `computerMarks(name, args)`: what to draw on the screenshot an action refers to.
pub fn computer_marks(name: &str, args: &Value) -> Vec<Mark> {
    let a = |k: &str| args.get(k);
    let (x, y) = (js_number_of(a("x")), js_number_of(a("y")));
    let ok = x.is_finite() && y.is_finite();
    if (name == "computer_click" || name == "computer_move") && ok {
        return vec![Mark::Cross { x, y, double: Some(truthy(a("double"))) }];
    }
    if name == "computer_scroll" && ok {
        let part = |k: &str| {
            let n = js_number_of(a(k));
            if n.is_nan() || n == 0.0 { 0.0 } else { n }
        };
        return vec![Mark::Cross { x, y, double: None }, Mark::Arrow { x, y, dx: part("dx"), dy: part("dy") }];
    }
    if name == "computer_drag" {
        let empty = json!({});
        let pick = |v: Option<&Value>| match v {
            Some(p) if truthy(Some(p)) => p.clone(),
            _ => empty.clone(),
        };
        let (f, t) = (pick(a("from")), pick(a("to")));
        let at = |p: &Value, name: &str, i: usize| js_number_of(member(p, name, i));
        let (fx, fy, tx, ty) = (at(&f, "x", 0), at(&f, "y", 1), at(&t, "x", 0), at(&t, "y", 1));
        if [fx, fy, tx, ty].iter().all(|n| n.is_finite()) {
            return vec![Mark::Line { x1: fx, y1: fy, x2: tx, y2: ty }, Mark::Cross { x: fx, y: fy, double: None }, Mark::Cross { x: tx, y: ty, double: None }];
        }
    }
    Vec::new()
}

/// `marksSvg(width, height, marks)`: the same coordinates as the screenshot, a crosshair for
/// every mark, white underneath and red on top so it shows on any background.
pub fn marks_svg(width: f64, height: f64, marks: &[Mark]) -> String {
    if marks.is_empty() || width.is_nan() || height.is_nan() || width <= 0.0 || height <= 0.0 {
        return String::new();
    }
    let big = width.max(height);
    let s = 40f64.min(14f64.max(big / 50.0));
    let w = 2f64.max(round(big / 450.0));
    let f = |n: f64| num_text(round(n * 10.0) / 10.0);
    let layer = |stroke: &str, extra: f64| -> String {
        marks
            .iter()
            .map(|m| match m {
                Mark::Cross { x, y, double } => {
                    let r = s * 0.55;
                    let ring = if double.unwrap_or(false) { format!("<circle cx=\"{}\" cy=\"{}\" r=\"{}\"/>", f(*x), f(*y), f(r * 1.7)) } else { String::new() };
                    format!(
                        "<g stroke=\"{stroke}\" stroke-width=\"{}\" fill=\"none\" stroke-linecap=\"round\"><circle cx=\"{}\" cy=\"{}\" r=\"{}\"/>{ring}<path d=\"M{} {}H{}M{} {}H{}M{} {}V{}M{} {}V{}\"/></g>",
                        num_text(w + extra),
                        f(*x),
                        f(*y),
                        f(r),
                        f(x - s * 1.1),
                        f(*y),
                        f(x - r * 0.6),
                        f(x + r * 0.6),
                        f(*y),
                        f(x + s * 1.1),
                        f(*x),
                        f(y - s * 1.1),
                        f(y - r * 0.6),
                        f(*x),
                        f(y + r * 0.6),
                        f(y + s * 1.1),
                    )
                }
                Mark::Line { x1, y1, x2, y2 } => format!(
                    "<path d=\"M{} {}L{} {}\" stroke=\"{stroke}\" stroke-width=\"{}\" fill=\"none\" stroke-linecap=\"round\" stroke-dasharray=\"{} {}\"/>",
                    f(*x1),
                    f(*y1),
                    f(*x2),
                    f(*y2),
                    num_text(w + extra),
                    f(s * 0.5),
                    f(s * 0.35)
                ),
                Mark::Arrow { x, y, dx, dy } => {
                    let len = {
                        let h = dx.hypot(*dy);
                        if h == 0.0 { 1.0 } else { h }
                    };
                    let ex = x + (dx / len) * s * 3.0;
                    let ey = y + (dy / len) * s * 3.0;
                    format!("<path d=\"M{} {}L{} {}\" stroke=\"{stroke}\" stroke-width=\"{}\" fill=\"none\" stroke-linecap=\"round\"/>", f(*x), f(*y), f(ex), f(ey), num_text(w + extra))
                }
            })
            .collect()
    };
    format!(
        "<svg class=\"cu-mark\" viewBox=\"0 0 {} {}\" preserveAspectRatio=\"none\" aria-hidden=\"true\">{}{}</svg>",
        num_text(width),
        num_text(height),
        layer("#fff", 3.0),
        layer("#e5231b", 0.0)
    )
}

/// A message of the conversation as `computer.js` handles it: its text and the pictures that
/// go with it (data URLs).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Msg {
    pub role: String,
    pub content: String,
    pub images: Vec<String>,
}

/// The note that stands where a dropped picture was.
pub const DROPPED_NOTE: &str = "[The picture was removed from the conversation to save space. Take a new screenshot if you need to see the screen.]";

/// `dropOldImages(messages, keep)`: takes the pictures out of all but the last `keep` messages
/// that carry some; returns how many were dropped.
pub fn drop_old_images(messages: &mut [Msg], keep: usize) -> usize {
    let carrying: Vec<usize> = messages.iter().enumerate().filter(|(_, m)| !m.images.is_empty()).map(|(i, _)| i).collect();
    let mut dropped = 0;
    for &i in &carrying[..carrying.len().saturating_sub(keep)] {
        dropped += messages[i].images.len();
        messages[i].images.clear();
        messages[i].content = format!("{}\n{DROPPED_NOTE}", messages[i].content);
    }
    dropped
}

/// `withoutImages(messages)`: a copy without pictures, for saving a session.
pub fn without_images(messages: &[Msg]) -> Vec<Msg> {
    messages
        .iter()
        .map(|m| {
            if m.images.is_empty() {
                return m.clone();
            }
            let what = if m.images.len() == 1 { "a screenshot was".to_string() } else { format!("{} screenshots were", m.images.len()) };
            Msg { role: m.role.clone(), content: format!("{}\n[{what} shown here and are not kept]", m.content), images: Vec::new() }
        })
        .collect()
}

/// `toWireMessages(messages)`: a message with pictures becomes text plus image parts.
pub fn to_wire(messages: &[Msg]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            if m.images.is_empty() {
                json!({ "role": m.role, "content": m.content })
            } else {
                let mut parts = vec![json!({ "type": "text", "text": m.content })];
                parts.extend(m.images.iter().map(|url| json!({ "type": "image_url", "image_url": { "url": url } })));
                json!({ "role": m.role, "content": parts })
            }
        })
        .collect()
}

/// `escapeStreak(times, now, windowMs, needed)`: Escape pressed `needed` times within `windowMs`;
/// returns the new list of presses and whether to stop.
pub fn escape_streak(times: &[f64], now: f64, window_ms: f64, needed: usize) -> (Vec<f64>, bool) {
    let mut recent: Vec<f64> = times.iter().copied().filter(|t| now - t <= window_ms).collect();
    recent.push(now);
    if recent.len() >= needed { (Vec::new(), true) } else { (recent, false) }
}

/// The page's defaults: three presses within 1.5 seconds.
pub fn escape_streak_default(times: &[f64], now: f64) -> (Vec<f64>, bool) {
    escape_streak(times, now, 1500.0, 3)
}

/// `isPanicChord(e)`: Ctrl+Alt+Shift+Esc, the same chord the server watches for.
pub fn is_panic_chord(key: &str, ctrl: bool, alt: bool, shift: bool) -> bool {
    key == "Escape" && ctrl && alt && shift
}

/// The computer tools that only look; every other one operates the machine (`COMPUTER_LOOK`).
pub fn is_look(name: &str) -> bool {
    matches!(name, "computer_screenshot" | "computer_cursor" | "computer_displays" | "computer_wait")
}

/// `toolLabel` of the page for the computer tools: the verb and what it is about, for the card's
/// header and for the banner (*Operating your computer: Click 10, 20*).
pub fn tool_label(name: &str, args: &Value) -> Option<(String, String)> {
    let a = |k: &str| args.get(k);
    let raw = |k: &str| js_or_undefined(a(k));
    let t = |s: &str| s.to_string();
    Some(match name {
        "computer_screenshot" => (
            t("Screenshot"),
            if truthy(a("region")) {
                t("a part of the screen")
            } else if a("display").is_some() {
                format!("display {}", raw("display"))
            } else {
                t("the screen")
            },
        ),
        "computer_click" => {
            let button = match a("button") {
                Some(b) if truthy(Some(b)) && b.as_str() != Some("left") => format!("{}-", js_string(b)),
                _ => String::new(),
            };
            (if truthy(a("double")) { t("Double-click") } else { format!("{button}Click") }, format!("{}, {}", raw("x"), raw("y")))
        }
        "computer_move" => (t("Move the pointer"), format!("{}, {}", raw("x"), raw("y"))),
        "computer_drag" => {
            let end = |v: Option<&Value>, k: &str| match v.and_then(|p| p.get(k)).filter(|x| !x.is_null()) {
                Some(x) => js_string(x),
                None => "?".to_string(),
            };
            (t("Drag"), format!("{}, {} to {}, {}", end(a("from"), "x"), end(a("from"), "y"), end(a("to"), "x"), end(a("to"), "y")))
        }
        "computer_type" => (t("Type"), serde_json::to_string(&slice_units(&text_or_empty(a("text")), 60)).unwrap_or_default()),
        "computer_key" => (t("Press keys"), text_or_empty(a("keys"))),
        "computer_scroll" => {
            let dx = if truthy(a("dx")) { format!("{} right ", raw("dx")) } else { String::new() };
            let dy = if truthy(a("dy")) { format!("{} down ", raw("dy")) } else { String::new() };
            (t("Scroll"), format!("{dx}{dy}at {}, {}", raw("x"), raw("y")))
        }
        "computer_wait" => (
            t("Wait"),
            format!("{} ms", match a("ms") {
                None | Some(Value::Null) => "1000".to_string(),
                Some(v) => js_string(v),
            }),
        ),
        "computer_cursor" => (t("Pointer position"), String::new()),
        "computer_displays" => (t("List displays"), String::new()),
        _ => return None,
    })
}

/// A member in a template literal: `undefined` for a missing one.
fn js_or_undefined(v: Option<&Value>) -> String {
    v.map_or_else(|| "undefined".to_string(), js_string)
}
