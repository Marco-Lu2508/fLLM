//! Settings > Model details (`spane-provider-fllm` of the Studio): what the server knows about the loaded model, the other names
//! apps may ask it by (`/api/copy` and `/api/delete` of the Ollama interface) and the chat format the model carries.

use crate::app::Message;
use crate::models::{card, size};
use crate::theme::{self, Tokens};
use iced::widget::{button, column, container, row, scrollable, text, text_input};
use iced::{Alignment, Element, Font, Length};
use serde_json::Value;

/// What `GET /api/studio/status` says of the model (`panels.js` reads the same fields for `providerModelFacts`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Facts {
    pub name: String,
    pub arch: String,
    pub parameter_size: String,
    pub parameters: u64,
    pub size_bytes: u64,
    pub quantization: String,
    pub max_context: u64,
    pub trained_context: u64,
    pub vocab_size: u64,
    pub aliases: Vec<String>,
    pub chat_template: String,
}

impl Facts {
    pub fn from_json(v: &Value) -> Option<Facts> {
        let m = v.get("model")?;
        let s = |k: &str| m.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
        let n = |k: &str| m.get(k).and_then(Value::as_u64).unwrap_or(0);
        let name = s("name");
        if name.is_empty() {
            return None;
        }
        Some(Facts {
            name,
            arch: s("arch"),
            parameter_size: s("parameter_size"),
            parameters: n("parameters"),
            size_bytes: n("size_bytes"),
            quantization: s("quantization"),
            max_context: n("max_context"),
            trained_context: n("trained_context"),
            vocab_size: n("vocab_size"),
            aliases: m.get("aliases").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default(),
            chat_template: s("chat_template"),
        })
    }

    /// The line under *Model*: architecture, size in parameters (and their count), size of the file.
    pub fn model_line(&self) -> String {
        format!("{} · {} ({} parameters) · {}", self.arch, self.parameter_size, group(self.parameters), size(self.size_bytes))
    }

    /// The line under *Conversation length*.
    pub fn length_line(&self) -> String {
        format!("{} tokens now (the model was trained for {}) · {} known word pieces", self.max_context, self.trained_context, self.vocab_size)
    }
}

/// 1234567 as 1,234,567 (`toLocaleString` of the Studio).
fn group(n: u64) -> String {
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

#[derive(Default)]
pub struct State {
    /// `None` until the server has answered; `Some(None)`: the server has no model.
    pub facts: Option<Option<Facts>>,
    pub error: Option<String>,
    /// The text of the field *Other names*.
    pub alias: String,
    pub note: Option<String>,
    pub busy: bool,
}

#[derive(Clone, Debug)]
pub enum FactsMsg {
    Loaded(Result<Option<Facts>, String>),
    Alias(String),
    Add,
    Remove(String),
    Done(Result<String, String>),
}

pub async fn fetch(base: String) -> Result<Option<Facts>, String> {
    crate::models::get_json(&base, "/api/studio/status").await.map(|v| Facts::from_json(&v))
}

/// `POST /api/copy`: the model answers to another name too.
pub async fn add_alias(base: String, source: String, dest: String) -> Result<String, String> {
    crate::models::mcp_request(&base, reqwest::Method::POST, "/api/copy", Some(serde_json::json!({ "source": source, "destination": dest })))
        .await
        .map(|_| format!("The name {dest} was added."))
}

/// `DELETE /api/delete`: the name goes, the model stays.
pub async fn remove_alias(base: String, name: String) -> Result<String, String> {
    crate::models::mcp_request(&base, reqwest::Method::DELETE, "/api/delete", Some(serde_json::json!({ "model": name })))
        .await
        .map(|_| format!("The name {name} was removed."))
}

pub struct View<'a> {
    pub state: &'a State,
}

fn item<'a>(t: Tokens, title: &'static str, line: String, chip: Option<String>) -> Element<'a, Message> {
    let mut r = row![column![text(title).size(14), text(line).size(12).font(Font::MONOSPACE).color(t.muted)].spacing(3).width(Length::Fill)].align_y(Alignment::Center);
    if let Some(c) = chip.filter(|c| !c.is_empty()) {
        r = r.push(
            container(text(c).size(12).font(Font::MONOSPACE).color(t.fg_soft))
                .padding([3, 10])
                .style(move |_| container::Style { background: Some(iced::Background::Color(t.fill)), border: iced::border::rounded(999), ..Default::default() }),
        );
    }
    r.into()
}

/// The two cards of the page: the facts with the other names, and the chat format.
pub fn cards<'a>(t: Tokens, v: View<'a>) -> Element<'a, Message> {
    let s = v.state;
    let mut body = column![text("Facts about the loaded model, other names apps can use for it, and the chat format it carries.").size(13).color(t.muted)].spacing(12);
    let facts = match &s.facts {
        Some(Some(f)) => Some(f),
        _ => None,
    };
    match (&s.facts, &s.error) {
        (_, Some(e)) => body = body.push(text(format!("The details could not be read: {e}")).size(13).color(t.danger)),
        (None, _) => body = body.push(text("Reading …").size(13).color(t.muted)),
        (Some(None), _) => body = body.push(text("No model is loaded. Load one on the Models page.").size(13).color(t.muted)),
        _ => {}
    }
    if let Some(f) = facts {
        body = body.push(item(t, "Model", f.model_line(), Some(f.quantization.clone())));
        body = body.push(item(t, "Conversation length", f.length_line(), None));
    }
    // other names: "Apps that ask for a name like gpt-4o-mini get this model."
    let can_add = facts.is_some() && !s.busy && !s.alias.trim().is_empty();
    body = body.push(
        row![
            column![text("Other names").size(14), text("Apps that ask for a name like gpt-4o-mini get this model.").size(12).color(t.muted)].spacing(3).width(Length::Fill),
            text_input("e.g. claude-3-sonnet", &s.alias)
                .on_input(|x| Message::Facts(FactsMsg::Alias(x)))
                .on_submit_maybe(can_add.then_some(Message::Facts(FactsMsg::Add)))
                .font(Font::MONOSPACE)
                .size(13)
                .padding([6, 10])
                .width(190)
                .style(move |_, status| iced::widget::text_input::Style {
                    background: iced::Background::Color(t.surface_2),
                    border: iced::Border { radius: 8.0.into(), width: 1.0, color: if matches!(status, iced::widget::text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator } },
                    icon: t.muted,
                    placeholder: t.muted,
                    value: t.fg,
                    selection: t.primary_soft,
                }),
            button(text("Add name").size(13)).on_press_maybe(can_add.then_some(Message::Facts(FactsMsg::Add))).padding([6, 14]).style(theme::primary_button(t)),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    );
    if let Some(f) = facts {
        if !f.aliases.is_empty() {
            let mut chips = row![].spacing(6);
            for a in &f.aliases {
                chips = chips.push(
                    container(
                        row![
                            text(a.clone()).size(12).font(Font::MONOSPACE),
                            button(text("✕").size(11).color(t.danger)).on_press_maybe((!s.busy).then(|| Message::Facts(FactsMsg::Remove(a.clone())))).padding([0, 4]).style(theme::ghost_button(t)),
                        ]
                        .spacing(4)
                        .align_y(Alignment::Center),
                    )
                    .padding([3, 10])
                    .style(move |_| container::Style { background: Some(iced::Background::Color(t.fill)), border: iced::border::rounded(999), ..Default::default() }),
                );
            }
            body = body.push(scrollable(chips).direction(scrollable::Direction::Horizontal(scrollable::Scrollbar::new().width(4).scroller_width(4))));
        }
    }
    if let Some(n) = &s.note {
        body = body.push(text(n.clone()).size(12).color(t.muted));
    }
    let first = card(t, "Model details", body.into());
    let template = facts.map(|f| f.chat_template.clone()).unwrap_or_default();
    let shown = if template.is_empty() { "The model carries no chat format.".to_string() } else { template };
    let second = card(
        t,
        "Chat format of the model",
        scrollable(container(text(shown).size(12).font(Font::MONOSPACE).color(t.fg_soft)).width(Length::Fill).padding(10).style(theme::inset(t)))
            .height(Length::Shrink)
            .into(),
    );
    column![first, second].spacing(14).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_facts_are_read_the_way_the_studio_reads_them() {
        let v = json!({"model": {"name": "qwen", "arch": "qwen3", "parameter_size": "4B", "parameters": 4022468096u64, "size_bytes": 2497280256u64,
            "quantization": "Q4_K_M", "max_context": 8192, "trained_context": 40960, "vocab_size": 151936, "aliases": ["gpt-4o-mini"], "chat_template": "{{ x }}"}});
        let f = Facts::from_json(&v).unwrap();
        assert_eq!(f.model_line(), "qwen3 · 4B (4,022,468,096 parameters) · 2.5 GB");
        assert_eq!(f.length_line(), "8192 tokens now (the model was trained for 40960) · 151936 known word pieces");
        assert_eq!(f.aliases, vec!["gpt-4o-mini"]);
        assert_eq!(Facts::from_json(&json!({"model": {"name": ""}})), None, "no model");
        assert_eq!(Facts::from_json(&json!({"setup": true})), None);
        assert_eq!(group(0), "0");
        assert_eq!(group(999), "999");
        assert_eq!(group(1000), "1,000");
    }
}
