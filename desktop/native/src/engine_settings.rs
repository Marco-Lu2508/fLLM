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
    pub title: &'static str,
    pub help: &'static str,
    pub kind: Kind,
}

pub const SPEED: &[Field] = &[
    Field { key: "prefix_cache", title: "Share the start of prompts", help: "Requests that begin with the same text reuse the work already done for it. Turn it off if people on this server must not share anything.", kind: Kind::Switch },
    Field { key: "state_cache", title: "Remember the start of a conversation", help: "For Qwen3.5, Mamba and RWKV: keep a copy of the model's state, so a message does not read the whole chat again.", kind: Kind::Switch },
    Field { key: "state_cache_mb", title: "Memory for those copies", help: "One copy of a 9B model takes about 100 MB. The oldest goes first.", kind: Kind::Number("MB", 1) },
    Field { key: "kv_ram_mb", title: "Keep recent prompt blocks in memory", help: "Much faster than reading the disk. 0 means off.", kind: Kind::Number("MB", 0) },
    Field { key: "preemption", title: "When memory runs out", help: "Recompute saves memory; moving the work to RAM continues faster and needs RAM.", kind: Kind::Choice(&[("recompute", "Recompute"), ("swap", "Move to RAM")]) },
];

pub const ANSWERS: &[Field] = &[
    Field { key: "output_cache", title: "Remember identical answers", help: "The same request with temperature 0 gets the saved answer at once.", kind: Kind::Choice(&[("exact", "On"), ("off", "Off")]) },
    Field { key: "output_cache_mb", title: "Memory for saved answers", help: "The oldest answers go first.", kind: Kind::Number("MB", 1) },
    Field { key: "cache_scope", title: "Who shares saved answers", help: "Per key and user keeps the answers of each API key apart.", kind: Kind::Choice(&[("global", "Everyone"), ("tenant", "Per key"), ("off", "Nobody")]) },
    Field { key: "kv_disk_dir", title: "Keep shared prompt blocks on disk", help: "A folder (full path). Empty means off.", kind: Kind::Text("off") },
];

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
        for f in SPEED.iter().chain(ANSWERS) {
            if self.locked(f.key) {
                continue;
            }
            match f.kind {
                Kind::Text(_) => {
                    body.insert(f.key.into(), json!(""));
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
    SPEED.iter().chain(ANSWERS).find(|f| f.key == key)
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

pub fn view<'a>(t: Tokens, s: &'a EngineState, connected: bool) -> Element<'a, Message> {
    if !connected {
        return card(t, "Server", text("Settings of the server appear when it is running.").size(13).color(t.muted).into());
    }
    if !s.loaded {
        return card(t, "Server", text(s.error.as_deref().unwrap_or("Looking…")).size(13).color(if s.error.is_some() { t.danger } else { t.muted }).into());
    }
    let mut server = column![].spacing(14);
    server = server.push(setting(
        t,
        "Load the last model at start",
        "The server loads the model you used last when it starts.",
        toggler(s.server.load_on_start)
            .on_toggle(|v| Message::ServerSet("load_on_start".into(), json!(v)))
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
            .into(),
    ));
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

    let mut footer = row![].spacing(10).align_y(Alignment::Center);
    if let Some(e) = &s.error {
        footer = footer.push(text(e.as_str()).size(13).color(t.danger).width(Length::Fill));
    } else if s.restarting {
        footer = footer.push(text("Restarting the server…").size(13).color(t.muted).width(Length::Fill));
    } else if s.restart_needed {
        footer = footer.push(text("Saved. The engine options apply when the server starts again.").size(13).color(t.fg_soft).width(Length::Fill));
        footer = footer.push(button(text("Restart now").size(13)).on_press(Message::RestartServer).padding([5, 14]).style(crate::theme::primary_button(t)));
    } else {
        footer = footer.push(Space::new().width(Length::Fill));
    }
    footer = footer.push(button(text("Reset engine options").size(13)).on_press(Message::EngineReset).padding([5, 14]).style(crate::theme::secondary_button(t)));

    column![card(t, "Speed and memory", group(t, SPEED, s)), card(t, "Saved answers and prompt blocks", group(t, ANSWERS, s)), card(t, "Server", server.into()), container(footer).width(Length::Fill)].spacing(14).into()
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
        for f in SPEED.iter().chain(ANSWERS) {
            assert!(KNOWN.contains(&f.key), "{}", f.key);
        }
        let keys: Vec<&str> = SPEED.iter().chain(ANSWERS).map(|f| f.key).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), keys.len(), "no field twice");
    }
}
