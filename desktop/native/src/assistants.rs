//! Assistants: a name, an icon, a colour, instructions, sampling values, an optional model and up to four
//! starter suggestions. This is the Studio's `projects.js` (assistants, import/export) and `chat.js`
//! (the built-in presets) ported to Rust; the limits and clamps are the same numbers as `PRJ_LIMITS`.

use iced::widget::text_editor::Content;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};

/// The same limits as `PRJ_LIMITS` in the Studio's projects.js.
pub const LIMIT_NAME: usize = 60;
pub const LIMIT_SYSTEM: usize = 8000;
pub const LIMIT_STARTER: usize = 140;
pub const LIMIT_MODEL: usize = 120;
pub const LIMIT_SYS_TEXT: usize = 6000;
/// The Studio keeps the first four suggestions of an assistant.
pub const LIMIT_STARTERS: usize = 4;

/// The icon names the editor offers (`ASST_ICONS`).
pub const ICONS: [&str; 8] = ["bot", "pen", "code", "book", "globe", "spark", "chat", "star"];
/// The colour names the editor offers (`ASST_COLORS`).
pub const COLORS: [&str; 12] =
    ["patina", "kupfer", "emerald", "teal", "cyan", "blue", "purple", "orange", "red", "green", "pink", "rose"];

const ICON_BOT: &[u8] = include_bytes!("../icons/bot.svg");
const ICON_PEN: &[u8] = include_bytes!("../icons/pen.svg");
const ICON_CODE: &[u8] = include_bytes!("../icons/code.svg");
const ICON_BOOK: &[u8] = include_bytes!("../icons/book.svg");
const ICON_GLOBE: &[u8] = include_bytes!("../icons/globe.svg");
const ICON_SPARK: &[u8] = include_bytes!("../icons/spark.svg");
const ICON_CHAT: &[u8] = include_bytes!("../icons/chat.svg");
const ICON_STAR: &[u8] = include_bytes!("../icons/star.svg");

/// The svg of an icon name (unknown names fall back to the bot, like the Studio falls back to its default).
pub fn icon_bytes(name: &str) -> &'static [u8] {
    match name {
        "pen" => ICON_PEN,
        "code" => ICON_CODE,
        "book" => ICON_BOOK,
        "globe" => ICON_GLOBE,
        "spark" => ICON_SPARK,
        "chat" => ICON_CHAT,
        "star" => ICON_STAR,
        _ => ICON_BOT,
    }
}

/// The badge colour of a colour name. Eight values are the Studio's `ACCENT_COLORS`; the five names the
/// Studio lists without a thumb (its swatch falls back to copper) get their own hue here.
pub fn color(name: &str) -> iced::Color {
    let hex = match name {
        "patina" => 0xe08a4b,
        "emerald" => 0x4ade80,
        "cyan" => 0x22d3ee,
        "rose" => 0xfb7185,
        "purple" => 0xc084fc,
        "blue" => 0x94a3b8,
        "green" => 0xa3e635,
        "kupfer" => 0xc2410c,
        "teal" => 0x14b8a6,
        "orange" => 0xf97316,
        "red" => 0xef4444,
        "pink" => 0xec4899,
        _ => 0xe08a4b,
    };
    iced::Color {
        r: ((hex >> 16) & 0xff) as f32 / 255.0,
        g: ((hex >> 8) & 0xff) as f32 / 255.0,
        b: (hex & 0xff) as f32 / 255.0,
        a: 1.0,
    }
}

/// How the assistant treats the default system prompt (the Studio's `sysMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SysMode {
    /// Take what the settings say.
    #[default]
    Inherit,
    /// The assistant's `sys_text` replaces the default prompt.
    Override,
    /// The assistant's `sys_text` is added to the default prompt.
    Append,
}

impl SysMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SysMode::Inherit => "inherit",
            SysMode::Override => "override",
            SysMode::Append => "append",
        }
    }
}

/// One assistant. `builtin` is never written to disk: the six built-ins are rebuilt from `BUILTIN_PRESETS`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assistant {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default = "icon_default")]
    pub icon: String,
    #[serde(default = "color_default")]
    pub color: String,
    /// The instructions appended to the composed system prompt when this assistant is chosen.
    #[serde(default)]
    pub system: String,
    #[serde(default = "temperature_default")]
    pub temperature: f32,
    #[serde(default = "top_p_default")]
    pub top_p: f32,
    #[serde(default = "top_k_default")]
    pub top_k: u32,
    /// A model to switch to when the assistant is chosen ("" keeps the current one).
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub starters: Vec<String>,
    #[serde(default)]
    pub sys_mode: SysMode,
    #[serde(default)]
    pub sys_text: String,
    #[serde(skip)]
    pub builtin: bool,
}

fn icon_default() -> String {
    "bot".into()
}
fn color_default() -> String {
    "patina".into()
}
fn temperature_default() -> f32 {
    0.7
}
fn top_p_default() -> f32 {
    0.95
}
fn top_k_default() -> u32 {
    20
}

/// A new id for an assistant (`as-…`, well under the 64 characters the Studio allows).
pub fn new_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("as-{:x}-{:x}", nanos, COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Runs runs of whitespace to single spaces and trims (the Studio's `.replace(/\s+/g, ' ').trim()`).
fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `String(raw.x ?? '')` of the Studio: strings stay, missing and null become empty, other values are printed.
fn js_string(raw: Option<&Value>) -> String {
    match raw {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// `Number(raw.x)` of the Studio for the fields that are clamped: numbers and numeric strings count,
/// null counts as 0, `""` counts as 0, everything else is `NaN` (which the clamps turn into the default).
fn js_number(raw: Option<&Value>) -> Option<f64> {
    match raw? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) if s.trim().is_empty() => Some(0.0),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        Value::Null => Some(0.0),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// The Studio's `prjClamp`: a value outside the range is clamped, a missing one keeps the default.
fn clamp(value: Option<f64>, lo: f64, hi: f64, dflt: f64) -> f32 {
    match value {
        Some(v) if v.is_finite() => v.clamp(lo, hi) as f32,
        _ => dflt as f32,
    }
}

fn slice(text: String, limit: usize) -> String {
    text.chars().take(limit).collect()
}

impl Assistant {
    /// A clean assistant from untrusted data (a file on disk, an imported export), or `None` when it has no
    /// name. Mirrors `asstSanitize` of the Studio: same limits, same clamps, same fallbacks.
    pub fn sanitize(raw: &Value, make_id: impl Fn() -> String) -> Option<Assistant> {
        let name = slice(collapse(&js_string(raw.get("name"))), LIMIT_NAME);
        if name.is_empty() {
            return None;
        }
        let id = match raw.get("id").and_then(Value::as_str) {
            Some(s) if !s.is_empty() => slice(s.to_string(), 64),
            _ => make_id(),
        };
        let icon = js_string(raw.get("icon"));
        let icon = if ICONS.contains(&icon.as_str()) { icon } else { icon_default() };
        let color = js_string(raw.get("color"));
        let color = if COLORS.contains(&color.as_str()) { color } else { color_default() };
        let starters: Vec<String> = raw
            .get("starters")
            .and_then(Value::as_array)
            .map(|list| {
                list.iter()
                    .map(|s| slice(collapse(&js_string(Some(s))), LIMIT_STARTER))
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default()
            .into_iter()
            .take(LIMIT_STARTERS)
            .collect();
        let sys_mode = match js_string(raw.get("sys_mode").or_else(|| raw.get("sysMode"))).as_str() {
            "override" => SysMode::Override,
            "append" => SysMode::Append,
            _ => SysMode::Inherit,
        };
        Some(Assistant {
            id,
            name,
            icon,
            color,
            system: slice(js_string(raw.get("system")), LIMIT_SYSTEM),
            temperature: clamp(js_number(raw.get("temperature")), 0.0, 2.0, 0.7),
            top_p: clamp(js_number(raw.get("top_p")), 0.0, 1.0, 0.95),
            top_k: clamp(js_number(raw.get("top_k")), 0.0, 1000.0, 20.0).round() as u32,
            model: slice(js_string(raw.get("model")), LIMIT_MODEL),
            starters,
            sys_mode,
            sys_text: slice(js_string(raw.get("sys_text").or_else(|| raw.get("sysText"))), LIMIT_SYS_TEXT),
            builtin: false,
        })
    }

    /// The same cleanup for an assistant that came from disk (`prjLoad` runs `asstSanitize` on every entry).
    pub fn cleaned(self) -> Option<Assistant> {
        let id = self.id.clone();
        let raw = serde_json::to_value(&self).ok()?;
        Assistant::sanitize(&raw, move || id.clone())
    }

    /// The name cut for a row title.
    pub fn summary(&self) -> String {
        slice(collapse(&self.system), 70)
    }
}

/// The built-in presets of the Studio's chat.js with their icon, colour and suggestions (`ASST_META`).
struct Preset {
    id: &'static str,
    name: &'static str,
    system: &'static str,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    icon: &'static str,
    color: &'static str,
    starters: [&'static str; 4],
}

const BUILTIN_PRESETS: [Preset; 6] = [
    Preset {
        id: "standard",
        name: "Standard",
        system: "",
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        icon: "chat",
        color: "patina",
        starters: [
            "Explain in simple words how a language model writes a text word by word.",
            "Write a fast Rust function for the dot product of two f32 slices.",
            "Help me plan my week: ask me what I need to get done.",
            "Summarize a topic I name in five bullet points.",
        ],
    },
    Preset {
        id: "precise",
        name: "Precise and brief",
        system: "Answer briefly and factually, and only with what you are sure about. Say so when you do not know something.",
        temperature: 0.2,
        top_p: 0.9,
        top_k: 20,
        icon: "book",
        color: "teal",
        starters: [
            "What is the difference between a process and a thread?",
            "Give me the three most important facts about photosynthesis.",
            "Is this claim correct, and how sure are you? I will paste it next.",
            "Define \"idempotent\" in two sentences.",
        ],
    },
    Preset {
        id: "creative",
        name: "Creative",
        system: "You are an inventive writing partner. Offer surprising ideas and vivid language.",
        temperature: 1.0,
        top_p: 0.95,
        top_k: 60,
        icon: "spark",
        color: "purple",
        starters: [
            "Write the opening paragraph of a mystery set on a night train.",
            "Give me ten names for a small coffee shop by the sea.",
            "Describe a city that only exists at night.",
            "Invent a board game and explain its rules in a few lines.",
        ],
    },
    Preset {
        id: "programmer",
        name: "Programmer",
        system: "You are an experienced software developer. Answer with working code and a short explanation. State your assumptions explicitly.",
        temperature: 0.2,
        top_p: 0.9,
        top_k: 20,
        icon: "code",
        color: "blue",
        starters: [
            "Review this function and point out bugs. I will paste it next.",
            "Write a unit test for a function that parses ISO dates.",
            "Explain the difference between a mutex and a semaphore with a short example.",
            "Show me how to read a file line by line in Rust.",
        ],
    },
    Preset {
        id: "translator",
        name: "Translator",
        system: "Translate the user's text into the language they ask in. If they name none, translate English into the language of their interface and other languages into English. Output only the translation, without comments.",
        temperature: 0.1,
        top_p: 0.9,
        top_k: 20,
        icon: "globe",
        color: "orange",
        starters: [
            "Translate to French: \"The meeting has been moved to Thursday.\"",
            "Translate a sentence I paste next into Spanish and explain any tricky word choices.",
            "Translate the text I paste next and keep its tone.",
            "Which is the more natural English phrasing: \"I am agree\" or \"I agree\"?",
        ],
    },
    Preset {
        id: "editor",
        name: "Editor",
        system: "Improve the spelling, grammar and style of the text. Then show in a short list what you changed.",
        temperature: 0.3,
        top_p: 0.9,
        top_k: 20,
        icon: "pen",
        color: "rose",
        starters: [
            "Improve the style of the paragraph I paste next.",
            "Fix the spelling and grammar of my text and list what you changed.",
            "Make this email shorter and more polite.",
            "Rewrite this text for a reader who is new to the topic.",
        ],
    },
];

/// The six built-in assistants, rebuilt on every call (they are never written to disk).
pub fn builtins() -> Vec<Assistant> {
    BUILTIN_PRESETS
        .iter()
        .map(|p| Assistant {
            id: p.id.into(),
            name: p.name.into(),
            icon: p.icon.into(),
            color: p.color.into(),
            system: p.system.into(),
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k,
            model: String::new(),
            starters: p.starters.iter().map(|s| (*s).to_string()).collect(),
            sys_mode: SysMode::Inherit,
            sys_text: String::new(),
            builtin: true,
        })
        .collect()
}

/// The built-ins first, then the saved ones (the Studio's `asstAll`).
pub fn all(saved: &[Assistant]) -> Vec<Assistant> {
    let mut out = builtins();
    out.extend(saved.iter().cloned());
    out
}

/// The four suggestions of an assistant: its own first, then the generic ones (the Studio's `asstStarters`).
pub fn starters(a: &Assistant) -> Vec<String> {
    let own: Vec<String> = a.starters.iter().filter(|s| !s.trim().is_empty()).cloned().collect();
    let generic = [
        format!("What can you help me with as \"{}\"?", a.name),
        "Ask me three questions so you understand my task, then help me with it.".to_string(),
        "Show me a short example of what you do best.".to_string(),
        "Summarize the text I paste next in five bullet points.".to_string(),
    ];
    own.into_iter().chain(generic).take(LIMIT_STARTERS).collect()
}

/// The export file of the Studio: `{"format": "fllm-assistant", "version": 1, "assistant": {…}}` without the id.
pub fn export_json(a: &Assistant) -> String {
    let mut body = serde_json::to_value(a).unwrap_or(Value::Null);
    if let Some(obj) = body.as_object_mut() {
        obj.remove("id");
    }
    let file = serde_json::json!({ "format": "fllm-assistant", "version": 1, "assistant": body });
    serde_json::to_string_pretty(&file).unwrap_or_default()
}

/// Reads an exported assistant (`asstParseImport`): the errors are the Studio's texts.
pub fn parse_import(text: &str, make_id: impl Fn() -> String) -> Result<Assistant, String> {
    let data: Value = serde_json::from_str(text).map_err(|_| "The file is not valid JSON.".to_string())?;
    let exported = data.get("format").and_then(Value::as_str) == Some("fllm-assistant");
    let Some(raw) = data.get("assistant").filter(|_| exported) else {
        return Err("This file is not an exported assistant.".to_string());
    };
    // the Studio imports with `id: undefined`, so an imported assistant always gets a new id
    let mut raw = raw.clone();
    if let Some(obj) = raw.as_object_mut() {
        obj.remove("id");
    }
    Assistant::sanitize(&raw, make_id).ok_or_else(|| "The assistant in the file has no name.".to_string())
}

/// The editor window: one assistant as the boxes show it. Kept out of `Message` so the messages stay plain.
#[derive(Clone)]
pub struct Editor {
    /// The id of the assistant being edited; `None` for a new one (or a copy of a built-in).
    pub id: Option<String>,
    /// The window title: "New assistant", "Edit assistant" or `Copy of "..."`.
    pub title: String,
    pub name: String,
    pub icon: String,
    pub color: String,
    pub system: Content,
    pub temperature: String,
    pub top_p: String,
    pub top_k: String,
    pub model: String,
    pub starters: Content,
    pub sys_mode: SysMode,
    pub sys_text: Content,
    /// Which of the two selects of the editor is open (`Default model`, `Default system prompt`).
    pub model_open: bool,
    pub sys_open: bool,
}

impl Default for Editor {
    /// The dialog of `btnAsNew` in the Studio: empty name, the default icon and colour, the default values.
    fn default() -> Self {
        Editor {
            id: None,
            title: "New assistant".into(),
            name: String::new(),
            icon: icon_default(),
            color: color_default(),
            system: Content::new(),
            temperature: "0.7".into(),
            top_p: "0.95".into(),
            top_k: "20".into(),
            model: String::new(),
            starters: Content::new(),
            sys_mode: SysMode::Inherit,
            sys_text: Content::new(),
            model_open: false,
            sys_open: false,
        }
    }
}

impl Editor {
    /// The dialog for an existing assistant. `copy` is how a built-in is edited: `asstOpenEditor` opens a
    /// copy with a new id and the name "`…` (copy)" instead of changing the built-in.
    pub fn of(a: &Assistant, copy: bool) -> Self {
        let num = |v: f32| {
            let s = format!("{v}");
            s
        };
        Editor {
            id: (!copy).then(|| a.id.clone()),
            title: if copy { format!("Copy of \"{}\"", a.name) } else { "Edit assistant".into() },
            name: if copy { format!("{} (copy)", a.name) } else { a.name.clone() },
            icon: a.icon.clone(),
            color: a.color.clone(),
            system: Content::with_text(&a.system),
            temperature: num(a.temperature),
            top_p: num(a.top_p),
            top_k: a.top_k.to_string(),
            model: a.model.clone(),
            starters: Content::with_text(&a.starters.join("\n")),
            sys_mode: a.sys_mode,
            sys_text: Content::with_text(&a.sys_text),
            model_open: false,
            sys_open: false,
        }
    }

    /// The assistant as the boxes stand, cleaned like an import (`asstEditorRead`); `None` when the name is
    /// empty. The number boxes are passed as strings, so an empty box counts as 0 exactly like the Studio.
    pub fn draft(&self) -> Option<Assistant> {
        let raw = serde_json::json!({
            "id": self.id.clone().unwrap_or_default(),
            "name": self.name,
            "icon": self.icon,
            "color": self.color,
            "system": self.system.text().trim_end_matches('\n'),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "model": self.model,
            "starters": self.starters.text().lines().map(|l| l.to_string()).collect::<Vec<_>>(),
            "sys_mode": self.sys_mode.as_str(),
            "sys_text": self.sys_text.text().trim_end_matches('\n'),
        });
        Assistant::sanitize(&raw, new_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn sanitize_clamps_every_number_the_way_the_studio_does() {
        let raw = v(r#"{"name":"A","temperature":5,"top_p":3,"top_k":1200}"#);
        let a = Assistant::sanitize(&raw, || "x".into()).unwrap();
        assert_eq!(a.temperature, 2.0);
        assert_eq!(a.top_p, 1.0);
        assert_eq!(a.top_k, 1000);
        // missing numbers keep the defaults, a broken string keeps them too
        let a = Assistant::sanitize(&v(r#"{"name":"B","temperature":"abc"}"#), || "x".into()).unwrap();
        assert_eq!(a.temperature, 0.7);
        assert_eq!(a.top_p, 0.95);
        assert_eq!(a.top_k, 20);
        // an empty box counts as 0, like Number('') in the Studio
        let a = Assistant::sanitize(&v(r#"{"name":"C","temperature":"","top_k":1.6}"#), || "x".into()).unwrap();
        assert_eq!(a.temperature, 0.0);
        assert_eq!(a.top_k, 2);
    }

    #[test]
    fn sanitize_cuts_names_texts_and_suggestions_to_the_limits() {
        let raw = v(&format!(
            r#"{{"name":"  a\n\n  b   {} ","system":"{}","model":"{}","sys_text":"{}","starters":["{}","s2","","s3","s4","s5"],"icon":"alien","color":"mauve"}}"#,
            "n".repeat(200),
            "s".repeat(9000),
            "m".repeat(200),
            "t".repeat(7000),
            "p".repeat(200)
        ));
        let a = Assistant::sanitize(&raw, || "x".into()).unwrap();
        assert_eq!(a.name.len(), LIMIT_NAME); // collapsed and cut to 60 characters
        assert!(a.name.starts_with("a b n"));
        assert_eq!(a.system.len(), LIMIT_SYSTEM);
        assert_eq!(a.model.len(), LIMIT_MODEL);
        assert_eq!(a.sys_text.len(), LIMIT_SYS_TEXT);
        assert_eq!(a.starters.len(), LIMIT_STARTERS);
        assert_eq!(a.starters[0].len(), LIMIT_STARTER);
        assert_eq!(a.icon, "bot"); // unknown icon falls back
        assert_eq!(a.color, "patina"); // unknown colour falls back
        // no name: nothing
        assert!(Assistant::sanitize(&v(r#"{"name":"   "}"#), || "x".into()).is_none());
        assert!(Assistant::sanitize(&v(r#"{"system":"x"}"#), || "x".into()).is_none());
    }

    #[test]
    fn the_six_built_ins_keep_their_ids_and_meta() {
        let b = builtins();
        assert_eq!(b.len(), 6);
        let ids: Vec<&str> = b.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, ["standard", "precise", "creative", "programmer", "translator", "editor"]);
        assert!(b.iter().all(|a| a.builtin));
        let p = b.iter().find(|a| a.id == "programmer").unwrap();
        assert_eq!(p.icon, "code");
        assert_eq!(p.color, "blue");
        assert_eq!(p.temperature, 0.2);
        assert_eq!(p.starters.len(), 4);
        assert!(b.iter().find(|a| a.id == "standard").unwrap().system.is_empty());
        // saved assistants come after the built-ins
        let saved = Assistant::sanitize(&v(r#"{"name":"Mine","id":"as-1"}"#), || "x".into()).unwrap();
        let all = all(&[saved]);
        assert_eq!(all.len(), 7);
        assert_eq!(all[0].id, "standard");
        assert_eq!(all[6].name, "Mine");
    }

    #[test]
    fn suggestions_are_the_ones_of_the_assistant_then_the_generic_four() {
        let a = Assistant::sanitize(&v(r#"{"name":"My helper","starters":["Do the thing","Write a poem"]}"#), || "x".into()).unwrap();
        let s = starters(&a);
        assert_eq!(s.len(), 4);
        assert_eq!(s[0], "Do the thing");
        assert_eq!(s[1], "Write a poem");
        assert_eq!(s[2], "What can you help me with as \"My helper\"?");
        assert!(s[3].contains("three questions"));
        // without own ones: four generic, first with the name
        let b = builtins().into_iter().find(|a| a.id == "editor").unwrap();
        let s = starters(&b);
        assert_eq!(s.len(), 4);
        assert_eq!(s[0], "Improve the style of the paragraph I paste next.");
    }

    #[test]
    fn export_and_import_roundtrip_without_the_id() {
        let mut a = Assistant::sanitize(&v(r#"{"id":"as-keep","name":"Roundtrip","icon":"book","color":"teal","system":"Be terse.","temperature":0.3,"top_p":0.8,"top_k":7,"model":"qwen","starters":["one","two"],"sys_mode":"append","sys_text":"More detail."}"#), || "x".into()).unwrap();
        a.builtin = false;
        let file = export_json(&a);
        assert!(file.contains("fllm-assistant"));
        assert!(!file.contains("as-keep"), "the id stays out of the file");
        let back = parse_import(&file, new_id).unwrap();
        assert_ne!(back.id, "as-keep", "an import always gets a new id");
        assert_eq!(back.name, a.name);
        assert_eq!(back.system, a.system);
        assert_eq!(back.temperature, a.temperature);
        assert_eq!(back.top_k, a.top_k);
        assert_eq!(back.model, a.model);
        assert_eq!(back.starters, a.starters);
        assert_eq!(back.sys_mode, SysMode::Append);
        assert_eq!(back.sys_text, a.sys_text);
        assert_eq!(back.icon, "book");
        // a built-in exports too (its copy has no id either)
        let builtin = export_json(&builtins()[1]);
        assert!(builtin.contains("Precise and brief"));
    }

    #[test]
    fn import_says_what_is_wrong_with_the_file() {
        assert_eq!(parse_import("nope", new_id).unwrap_err(), "The file is not valid JSON.");
        assert_eq!(
            parse_import(r#"{"format":"other","assistant":{"name":"x"}}"#, new_id).unwrap_err(),
            "This file is not an exported assistant."
        );
        assert_eq!(
            parse_import(r#"{"format":"fllm-assistant"}"#, new_id).unwrap_err(),
            "This file is not an exported assistant."
        );
        assert_eq!(
            parse_import(r#"{"format":"fllm-assistant","assistant":{"name":"  "}}"#, new_id).unwrap_err(),
            "The assistant in the file has no name."
        );
    }

    #[test]
    fn the_editor_reads_back_what_the_boxes_hold() {
        let e = Editor::default();
        assert!(e.draft().is_none(), "no name, no assistant");
        let e = Editor {
            name: "  My   assistant ".into(),
            temperature: "".into(),
            system: Content::with_text("Rules.\n"),
            starters: Content::with_text("one\ntwo\nthree\nfour\nfive"),
            ..Default::default()
        };
        let a = e.draft().unwrap();
        assert_eq!(a.name, "My assistant");
        assert_eq!(a.temperature, 0.0);
        assert_eq!(a.system, "Rules.", "the trailing newline of the box stays out");
        assert_eq!(a.starters.len(), LIMIT_STARTERS);
        assert!(a.id.starts_with("as-"), "a new assistant gets an id");
        // editing keeps the id
        let mut e = Editor::of(&a, false);
        e.name.push('!');
        let again = e.draft().unwrap();
        assert_eq!(again.id, a.id);
        // a copy of a built-in is a new assistant with the (copy) name
        let copy = Editor::of(&builtins()[3], true);
        assert!(copy.id.is_none());
        assert_eq!(copy.name, "Programmer (copy)");
        let made = copy.draft().unwrap();
        assert_ne!(made.id, "programmer");
        assert_eq!(made.system, builtins()[3].system);
    }

    #[test]
    fn cleaned_accepts_what_is_on_disk_and_fixes_the_rest() {
        let a = Assistant::sanitize(&v(r#"{"id":"as-1","name":"Kept","temperature":9}"#), || "x".into()).unwrap();
        let again = a.clone().cleaned().unwrap();
        assert_eq!(again.id, "as-1", "the id of a saved assistant survives");
        assert_eq!(again.name, "Kept");
        let broken = Assistant { name: "   ".into(), temperature: 42.0, ..a }.cleaned();
        assert!(broken.is_none(), "a saved entry without a name is dropped");
        let clamped = Assistant { temperature: 42.0, icon: "alien".into(), ..builtins()[0].clone() }.cleaned().unwrap();
        assert_eq!(clamped.temperature, 2.0);
        assert_eq!(clamped.icon, "bot");
        assert!(!clamped.builtin, "the flag is rebuilt, never read");
    }

    #[test]
    fn ids_are_unique_and_short() {
        let a = new_id();
        let b = new_id();
        assert_ne!(a, b);
        assert!(a.len() <= 64, "{}", a);
        assert!(a.starts_with("as-"));
    }

    #[test]
    fn colours_and_icons_have_a_value_for_every_name() {
        for name in COLORS {
            let c = color(name);
            assert_eq!(c.a, 1.0);
            assert!(c.r > 0.0 || c.g > 0.0 || c.b > 0.0);
        }
        for name in ICONS {
            assert!(!icon_bytes(name).is_empty());
        }
        assert_eq!(icon_bytes("nope"), icon_bytes("bot"));
        assert_eq!(color("nope"), color("patina"));
    }
}
