//! Model providers: models of other servers and cloud services next to the local one. The list
//! lives in `providers.json` on this computer; keys never reach the window (the server adds them
//! when it passes a request on), chats go through `/api/studio/providers/chat`.
//!
//! The functions below are the `@pure-begin..@pure-end` block of `providers.js`, with the fixtures
//! that block produced (`tests/fixtures/providers_fixtures.json`).

use crate::app::{icon, muted_svg, Message};
use crate::models::{bold, card};
use crate::theme::{self, Tokens};
use iced::widget::{button, column, container, pick_list, row, svg, text, text_input, Space};
use iced::{Alignment, Element, Font, Length};
use serde_json::{json, Value};

/// The prefix of the id under which a model of a provider is selected.
pub const REMOTE_PREFIX: &str = "remote:";

/// A selection of a provider's model: `{provider, model}`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct Remote {
    pub provider: String,
    pub model: String,
}

/// The id under which a model of a provider is selected: `remote:<provider id>/<model>`.
pub fn remote_model_id(provider: &str, model: &str) -> String {
    format!("{REMOTE_PREFIX}{provider}/{model}")
}

/// `{provider, model}` for a selection of a provider's model, or `None` for a local model.
/// Provider ids hold no slash; model names may.
pub fn parse_remote_model_id(id: &str) -> Option<Remote> {
    let rest = id.strip_prefix(REMOTE_PREFIX)?;
    let slash = rest.find('/')?;
    // a slash at either end would leave an empty provider or an empty model (`slash < 1` covers
    // the missing one too, which JavaScript wrote as `slash < 1 || slash === rest.length - 1`)
    if slash < 1 || slash + 1 == rest.len() {
        return None;
    }
    Some(Remote { provider: rest[..slash].to_string(), model: rest[slash + 1..].to_string() })
}

/// One row of the model picker. `provider` and `model` are only there for a remote row, so the
/// fixtures (and the Studio, which writes `undefined` there) leave them out for a local one.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(default)]
pub struct Entry {
    pub id: String,
    pub label: String,
    pub remote: bool,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub active: bool,
}

/// A model of a provider as the server lists it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct ProviderModel {
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub provider_name: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub local: Option<bool>,
    #[serde(default)]
    pub id: String,
}

/// The entries of the model picker: the local models first, then `Provider · model` for every
/// model of a provider (`mergeModelEntries` of `providers.js`).
pub fn merge_model_entries(local_names: &[String], remote_models: &[ProviderModel], selected: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for name in local_names {
        if name.is_empty() || seen.iter().any(|s| s == name) {
            continue;
        }
        seen.push(name.clone());
        out.push(Entry {
            id: name.clone(),
            label: name.clone(),
            remote: false,
            provider: None,
            model: None,
            active: name == selected,
        });
    }
    for m in remote_models {
        if m.provider.is_empty() || m.id.is_empty() {
            continue;
        }
        let id = remote_model_id(&m.provider, &m.id);
        if seen.contains(&id) {
            continue;
        }
        seen.push(id.clone());
        let label = format!("{} · {}", m.provider_name.as_deref().filter(|n| !n.is_empty()).unwrap_or(&m.provider), m.id);
        out.push(Entry {
            active: id == selected,
            id,
            label,
            remote: true,
            provider: Some(m.provider.clone()),
            model: Some(m.id.clone()),
        });
    }
    out
}

/// Keeps a selection that still exists; a model of a provider that was removed or switched off
/// falls back to `fallback` (`resolveSelection` of `providers.js`).
pub fn resolve_selection(selected: &str, entries: &[Entry], fallback: &str) -> String {
    if parse_remote_model_id(selected).is_none() {
        return selected.to_string();
    }
    if entries.iter().any(|e| e.id == selected) {
        selected.to_string()
    } else {
        fallback.to_string()
    }
}

/// One of the two groups of templates.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
pub struct PresetGroup {
    pub id: String,
    pub title: String,
    pub items: Vec<Preset>,
}

/// A template of the *Choose a service* row.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
pub struct Preset {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub local: bool,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub key_url: Option<String>,
}

/// The templates in two groups: cloud services (an API key) first, then servers you run yourself
/// (optional) (`providerPresetGroups` of `providers.js`).
pub fn provider_preset_groups(presets: &[Preset]) -> Vec<PresetGroup> {
    let group_of = |p: &Preset| if p.group.as_deref() == Some("network") || (p.group.is_none() && p.local) { "network" } else { "cloud" };
    Vec::from([
        PresetGroup {
            id: "cloud".into(),
            title: "Cloud APIs".into(),
            items: presets.iter().filter(|p| group_of(p) == "cloud").cloned().collect(),
        },
        PresetGroup {
            id: "network".into(),
            title: "Other servers on your network".into(),
            items: presets.iter().filter(|p| group_of(p) == "network").cloned().collect(),
        },
    ])
}

/// Where the key of a template is made: `{url, host}` or `None` (https pages only)
/// (`providerKeyLink` of `providers.js`).
pub fn provider_key_link(preset: &Preset) -> Option<(String, String)> {
    let raw = preset.key_url.clone().unwrap_or_default();
    let url = url::Url::parse(&raw).ok()?;
    if url.scheme() != "https" {
        return None;
    }
    Some((url.to_string(), url.host_str()?.to_string()))
}

/// The line shown after *Test connection*: the models that were found, as many as fit
/// (`providerTestSummary` of `providers.js`).
pub fn provider_test_summary(r: Option<&TestAnswer>) -> String {
    let Some(r) = r else {
        return "The test failed.".into();
    };
    if !r.ok {
        return r.error.clone().unwrap_or_else(|| "The test failed.".into());
    }
    let models = r.models.clone().unwrap_or_default();
    let count = r.count.unwrap_or(models.len() as u64);
    if count == 0 {
        return "Connected, but the service lists no models for this key.".into();
    }
    let shown: Vec<&str> = models.iter().take(8).map(String::as_str).collect();
    let shown = shown.join(", ");
    let more = if count > 8 { format!(" and {} more", count - 8) } else { String::new() };
    let plural = if count == 1 { "" } else { "s" };
    if shown.is_empty() {
        format!("Connected. {count} model{plural} found.")
    } else {
        format!("Connected. {count} model{plural} found: {shown}{more}.")
    }
}

/// The answer of `POST /api/studio/providers/test`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct TestAnswer {
    #[serde(default)]
    pub ok: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub count: Option<u64>,
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

/// What the label of the picker shows for a selection (`modelLabel` of `providers.js`).
pub fn model_label(selected: &str, remote_models: &[ProviderModel]) -> String {
    if selected.is_empty() {
        return "Choose a model".into();
    }
    let Some(r) = parse_remote_model_id(selected) else {
        return selected.to_string();
    };
    let known = remote_models.iter().find(|m| m.provider == r.provider && m.id == r.model);
    let name = known
        .and_then(|m| m.provider_name.clone())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| r.provider.clone());
    format!("{name} · {}", r.model)
}

/// The settings of the local engine that other servers do not understand
/// (`LOCAL_ONLY_FIELDS` of `providers.js`).
pub const LOCAL_ONLY_FIELDS: [&str; 22] = [
    "top_k",
    "min_p",
    "repetition_penalty",
    "speculative",
    "thinking_budget",
    "chat_template_kwargs",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "grammar",
    "grammar_type",
    "regex",
    "tool_call_grammar",
    "tfs_z",
    "typical_p",
    "ignore_eos",
    "cache_prompt",
    "logit_bias_tokens",
];

/// The request for a provider's model: the same body without the settings of the local engine
/// and with the model's own name (`cleanRemoteRequest` of `providers.js`).
pub fn clean_remote_request(body: &Value, model: &str) -> Value {
    let mut out = body.as_object().cloned().unwrap_or_default();
    out.insert("model".into(), json!(model));
    for k in LOCAL_ONLY_FIELDS {
        out.remove(k);
    }
    Value::Object(out)
}

/// Where a chat for `modelId` goes, and what the body becomes on the way there
/// (`chatEndpoint` of `providers.js`).
#[derive(Debug, Clone, PartialEq)]
pub struct ChatEndpoint {
    pub url: &'static str,
    pub remote: bool,
    pub provider: Option<String>,
    pub model: Option<String>,
}

impl ChatEndpoint {
    /// `transform(body)` of the Studio: the body as the url wants it.
    pub fn transform(&self, body: &Value) -> Value {
        match (&self.provider, &self.model) {
            (Some(provider), Some(model)) => json!({ "provider": provider, "request": clean_remote_request(body, model) }),
            _ => body.clone(),
        }
    }
}

/// Where a chat for `modelId` goes: `/v1/chat/completions` for a local model, the provider route
/// for a model of a provider.
pub fn chat_endpoint(model_id: &str) -> ChatEndpoint {
    match parse_remote_model_id(model_id) {
        None => ChatEndpoint { url: "/v1/chat/completions", remote: false, provider: None, model: None },
        Some(r) => ChatEndpoint {
            url: "/api/studio/providers/chat",
            remote: true,
            provider: Some(r.provider),
            model: Some(r.model),
        },
    }
}

/// The row of the list: a provider without its key (`Provider::public` of the server).
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct Provider {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub has_key: bool,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub local: bool,
    #[serde(default)]
    pub models: Vec<String>,
}

/// What the list says under the title of a provider (`provRender` of `providers.js`).
pub fn provider_line(p: &Provider) -> String {
    let mut parts = vec![p.kind.clone()];
    if p.local {
        parts.push("this computer or network".into());
    }
    parts.push(if p.has_key { "key saved" } else { "no key" }.into());
    if !p.models.is_empty() {
        parts.push(format!("{} model{}", p.models.len(), if p.models.len() == 1 { "" } else { "s" }));
    }
    parts.join(", ")
}

/// The answer of `GET /api/studio/providers`.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct ProviderList {
    #[serde(default)]
    pub providers: Vec<Provider>,
    #[serde(default)]
    pub presets: Vec<Preset>,
    #[serde(default)]
    pub config_file: String,
}

/// The answer of `GET /api/studio/providers/models`.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct ModelList {
    #[serde(default, deserialize_with = "skip_nulls")]
    pub models: Vec<ProviderModel>,
    #[serde(default)]
    pub errors: Vec<ModelError>,
}

/// One provider whose model list could not be read.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct ModelError {
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub provider_name: String,
    #[serde(default)]
    pub error: String,
}

/// The list of a provider's models as the server sends it: a `null` in it is skipped, the way
/// `mergeModelEntries` skips one (`if (!m ...) continue`) — the window must not fail to load a
/// list because of a hole in it.
fn skip_nulls<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<ProviderModel>, D::Error> {
    let raw: Vec<Option<ProviderModel>> = serde::Deserialize::deserialize(d)?;
    Ok(raw.into_iter().flatten().collect())
}

/// The form of the second card (`provFormBody` of `providers.js`): what is typed, and which
/// provider it edits (nothing = a new one).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Form {
    pub name: String,
    pub kind: String,
    pub url: String,
    pub key: String,
    pub local: bool,
    pub clear_key: bool,
    /// The id of the provider being edited, or `None` for a new one.
    pub editing: Option<String>,
    /// The template that was clicked (*Choose a service*): only it brings the link under the key
    /// box (`provShowKeyLink` runs for a click on a template and for nothing else).
    pub preset: Option<String>,
}

impl Form {
    /// The body of `POST /api/studio/providers` as `provFormBody` builds it: the key only when
    /// one was pasted, an empty one when the saved key is to be removed, and `id` only when a
    /// provider is edited.
    pub fn body(&self) -> Value {
        let mut body = json!({
            "name": self.name.trim(),
            "kind": if self.kind.is_empty() { "openai" } else { self.kind.as_str() },
            "base_url": self.url.trim(),
            "local": self.local,
        });
        if let Some(id) = &self.editing {
            body["id"] = json!(id);
        }
        let key = self.key.trim();
        if !key.is_empty() {
            body["api_key"] = json!(key);
        } else if self.editing.is_some() && self.clear_key {
            body["api_key"] = json!("");
        }
        body
    }

    /// The title of the card: *Add a provider*, or *Edit <name>*.
    pub fn title(&self) -> String {
        match &self.editing {
            Some(_) => format!("Edit {}", self.name.trim()),
            None => "Add a provider".into(),
        }
    }

    /// The placeholder of the key box (`provEdit` and the preset click of `providers.js`).
    pub fn key_placeholder(&self, has_key: bool, preset_local: bool) -> &'static str {
        if self.editing.is_some() {
            if has_key { "(saved - leave empty to keep it)" } else { "sk-..." }
        } else if preset_local {
            "(usually no key)"
        } else {
            "Paste the key here"
        }
    }
}

/// What the settings page needs to draw the two cards.
#[derive(Clone, Copy)]
pub struct ProvidersView<'a> {
    /// The answer of `GET /api/studio/providers` (empty until it came).
    pub list: &'a ProviderList,
    /// The answer of `GET /api/studio/providers/models`.
    pub models: &'a ModelList,
    pub form: &'a Form,
    /// The line under the list (`provResult`): what a test or a refresh said, and whether it was
    /// good (`true` = the primary colour, `false` = the destructive one).
    pub result: Option<(&'a str, bool)>,
    /// The line under the form (`provFileInfo`): `Saved in <file>`.
    pub file_info: &'a str,
    /// The group of *Other servers on your network* is open (`<details>` of the Studio).
    pub network_open: bool,
}

/// The two cards of `spane-providers`, between *Usage and cost* and *Dictation* where the
/// Studio's settings menu has them.
pub fn section<'a>(t: Tokens, v: ProvidersView<'a>) -> Element<'a, Message> {
    column![list_card(t, v), form_card(t, v)].spacing(14).into()
}

fn list_card<'a>(t: Tokens, v: ProvidersView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(10);
    body = body
        .push(text(LIST_HELP).size(13).color(t.fg_soft))
        .push(text(PRIVACY_HELP).size(13).color(t.fg_soft));

    let mut rows = column![].spacing(10);
    if v.list.providers.is_empty() {
        rows = rows.push(
            text("No providers yet. Choose a service below, paste its key and test the connection.")
                .size(13)
                .color(t.muted),
        );
    }
    for p in &v.list.providers {
        let mut head = row![
            text(p.name.clone()).size(14).font(bold()),
            text(provider_line(p)).size(12).color(t.muted).width(Length::Fill),
        ]
        .spacing(8)
        .align_y(Alignment::Center);
        head = head.push(
            row![
                crate::settings::segmented(
                    t,
                    vec![
                        ("On", p.enabled, Message::ProvToggle(p.id.clone(), true)),
                        ("Off", !p.enabled, Message::ProvToggle(p.id.clone(), false)),
                    ],
                ),
                button(text("Test and refresh models").size(12))
                    .on_press(Message::ProvTest(p.id.clone()))
                    .padding([3, 8])
                    .style(theme::ghost_button(t)),
                button(text("Edit").size(12))
                    .on_press(Message::ProvEdit(p.id.clone()))
                    .padding([3, 8])
                    .style(theme::ghost_button(t)),
                button(text("Remove").size(12))
                    .on_press(Message::ProvRemove(p.id.clone()))
                    .padding([3, 8])
                    .style(theme::ghost_button(t)),
            ]
            .spacing(6),
        );
        rows = rows.push(
            column![
                head,
                text(p.base_url.clone()).size(12).font(Font::MONOSPACE).color(t.muted),
            ]
            .spacing(6),
        );
    }
    body = body.push(rows);

    body = body.push(
        button(text("Refresh all model lists").size(12))
            .on_press(Message::ProvRefreshAll)
            .padding([5, 10])
            .style(theme::ghost_button(t)),
    );
    if let Some((line, ok)) = v.result {
        body = body.push(text(line).size(12).color(if ok { t.primary } else { t.danger }));
    }
    card(t, "Model providers", body.into())
}

fn form_card<'a>(t: Tokens, v: ProvidersView<'a>) -> Element<'a, Message> {
    let form = v.form;
    let mut body = column![].spacing(12);

    // the templates in two groups, the network one behind its own line (`<details>` of the Studio)
    let mut presets = column![text("Choose a service").size(13).font(bold())].spacing(6);
    let groups = provider_preset_groups(&v.list.presets);
    let cloud = groups.first().cloned().unwrap_or_default();
    let net = groups.get(1).cloned().unwrap_or_default();
    presets = presets.push(preset_row(t, &cloud.items));
    let net_line = button(
        row![
            svg(icon(if v.network_open { crate::app::ICON_CHEVRON } else { crate::app::ICON_CHEVRON_RIGHT }))
                .width(14)
                .height(14)
                .style(muted_svg(t)),
            text(net.title.clone()).size(12).color(t.fg_soft),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    )
    .on_press(Message::ProvNetOpen)
    .padding([3, 4])
    .style(theme::ghost_button(t));
    presets = presets.push(net_line);
    if v.network_open {
        presets = presets.push(text(NETWORK_HELP).size(12).color(t.muted));
        presets = presets.push(preset_row(t, &net.items));
    }
    body = body.push(presets);

    body = body
        .push(field(t, "Name", box_input(t, "My provider", &form.name, Message::ProvFormName, false)))
        .push(field(t, "Type", kind_picker(t, &form.kind)))
        .push(field(t, "Address", box_input(t, "https://api.example.com/v1", &form.url, Message::ProvFormUrl, true)))
        .push(field(t, "Key", key_box(t, form, v)));

    let local_control =
        crate::settings::segmented(t, vec![("On", form.local, Message::ProvLocal(true)), ("Off", !form.local, Message::ProvLocal(false))]);
    body = body.push(
        field(
            t,
            "This server runs on this computer or network",
            column![
                local_control,
                text("Needed for Ollama, LM Studio and other servers you run yourself. Without it, addresses on this computer or a private network are refused.")
                    .size(12)
                    .color(t.muted),
            ]
            .spacing(6)
            .into(),
        ),
    );

    body = body.push(
        row![
            button(text("Save").size(13))
                .on_press(Message::ProvSave)
                .padding([6, 14])
                .style(theme::primary_button(t)),
            button(text("Test connection").size(13))
                .on_press(Message::ProvTestForm)
                .padding([6, 14])
                .style(theme::ghost_button(t)),
            button(text("Clear").size(13))
                .on_press(Message::ProvClear)
                .padding([6, 14])
                .style(theme::ghost_button(t)),
        ]
        .spacing(6),
    );
    if !v.file_info.is_empty() {
        body = body.push(text(v.file_info.to_string()).size(12).color(t.muted));
    }
    // the title changes with the form (*Add a provider*, *Edit <name>*), so it is owned here
    // instead of borrowed from the page
    titled_card(t, form.title(), body.into())
}

/// `models::card` with a title that is built on the spot.
fn titled_card<'a>(t: Tokens, title: String, body: Element<'a, Message>) -> Element<'a, Message> {
    container(column![text(title).size(13).font(bold()).color(t.fg_soft), body].spacing(12))
        .padding(16)
        .width(Length::Fill)
        .style(theme::card(t, theme::R_LG, false))
        .into()
}

/// The row of template buttons of one group.
fn preset_row<'a>(t: Tokens, items: &[Preset]) -> Element<'a, Message> {
    let mut r = row![].spacing(6);
    for p in items {
        let id = p.id.clone();
        r = r.push(
            button(text(p.name.clone()).size(12))
                .on_press(Message::ProvPreset(id))
                .padding([4, 10])
                .style(theme::ghost_button(t)),
        );
    }
    r.wrap().into()
}

/// One labelled field of the form (`app-card-item column` of `shell.html`).
fn field<'a>(t: Tokens, title: &'a str, control: Element<'a, Message>) -> Element<'a, Message> {
    column![text(title).size(13).font(bold()).color(t.fg_soft), control].spacing(6).into()
}

/// A text box of the form.
fn box_input<'a>(t: Tokens, placeholder: &'a str, value: &'a str, typed: fn(String) -> Message, mono: bool) -> Element<'a, Message> {
    let mut b = text_input(placeholder, value).on_input(typed).size(13).padding([6, 10]);
    if mono {
        b = b.font(Font::MONOSPACE);
    }
    b.style(move |_, status| text_input::Style {
        background: iced::Background::Color(t.surface_2),
        border: iced::Border {
            radius: 8.0.into(),
            width: 1.0,
            color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator },
        },
        icon: t.muted,
        placeholder: t.muted,
        value: t.fg,
        selection: t.primary_soft,
    })
    .into()
}

/// The type of a provider (`provKind` of `shell.html`: OpenAI-compatible, Anthropic, Ollama).
fn kind_picker<'a>(t: Tokens, kind: &'a str) -> Element<'a, Message> {
    const KINDS: [(&str, &str); 3] = [("OpenAI-compatible", "openai"), ("Anthropic", "anthropic"), ("Ollama", "ollama")];
    let now = KINDS.iter().find(|(_, id)| *id == kind).map(|(label, _)| *label).unwrap_or("OpenAI-compatible");
    pick_list(
        KINDS.map(|(label, _)| label).to_vec(),
        Some(now),
        |label: &'static str| Message::ProvKind(match label {
            "Anthropic" => "anthropic",
            "Ollama" => "ollama",
            _ => "openai",
        }),
    )
    .padding([6, 10])
    .style(move |_, status| pick_list::Style {
        background: iced::Background::Color(t.surface_2),
        border: iced::Border {
            radius: 8.0.into(),
            width: 1.0,
            color: if matches!(status, pick_list::Status::Hovered) { t.fg_soft } else { t.separator },
        },
        text_color: t.fg,
        placeholder_color: t.muted,
        handle_color: t.muted,
    })
    .into()
}

/// The key box with the link under it and the row that removes a saved key.
fn key_box<'a>(t: Tokens, form: &'a Form, v: ProvidersView<'a>) -> Element<'a, Message> {
    let preset = form
        .preset
        .as_ref()
        .and_then(|id| v.list.presets.iter().find(|p| &p.id == id));
    let has_key = v.list.providers.iter().find(|p| Some(&p.id) == form.editing.as_ref()).is_some_and(|p| p.has_key);
    let mut body = column![box_input(t, form.key_placeholder(has_key, preset.is_some_and(|p| p.local)), &form.key, Message::ProvFormKey, true)].spacing(6);
    // the link under the key field: where this service hands out keys (`provShowKeyLink`)
    if let Some(link) = preset.and_then(provider_key_link) {
        body = body.push(text(format!("Get a key at {}", link.1)).size(12).color(t.primary));
    }
    if has_key {
        body = body.push(
            row![
                crate::settings::segmented(
                    t,
                    vec![
                        ("On", form.clear_key, Message::ProvClearKey(true)),
                        ("Off", !form.clear_key, Message::ProvClearKey(false)),
                    ],
                ),
                text("Remove the saved key").size(12).color(t.muted),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        );
    }
    body.into()
}

/// The description of the first card.
pub const LIST_HELP: &str = "fLLM runs models on this computer. Add a cloud service here only if you also want its models in the chat, for example a large model that does not fit your hardware. They appear in the model menu as \u{201c}Provider \u{b7} model\u{201d}.";

/// The line under the closed group of network templates (`<summary>`'s `<div>` of `shell.html`).
pub const NETWORK_HELP: &str = "Only if you already run Ollama, LM Studio, vLLM or llama.cpp on this computer or another one and want its models here. fLLM itself needs none of them.";

/// The privacy note of the first card (`<strong>Privacy:</strong>` of `shell.html` — the window
/// sets the word like the rest of the sentence, iced has no bold inside a flowing paragraph).
pub const PRIVACY_HELP: &str = "Privacy: everything you type in a chat with such a model is sent to that service. Keys are stored in a file on this computer and are never shown again; this page never receives them, fLLM adds them when it passes your request on.";

/// `GET /api/studio/providers`: the providers (without keys) and the templates.
pub async fn fetch_providers(base: String) -> Result<ProviderList, String> {
    crate::models::get_json(&base, "/api/studio/providers")
        .await
        .and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
}

/// `GET /api/studio/providers/models[?refresh=1]`: the models of the enabled providers, from the
/// cache or freshly listed (`provLoadModels` of `providers.js`).
pub async fn fetch_provider_models(base: String, refresh: bool) -> Result<ModelList, String> {
    let path = if refresh { "/api/studio/providers/models?refresh=1" } else { "/api/studio/providers/models" };
    crate::models::get_json(&base, path)
        .await
        .and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
}

/// `POST /api/studio/providers` with the body of `provFormBody`: adds or updates a provider, and
/// answers with the saved one (`{status: "saved", provider}`).
pub async fn post_provider(base: String, body: Value) -> Result<Value, String> {
    crate::models::post_fallback(&base, "/api/studio/providers", body, "Could not save it.").await
}

/// `POST /api/studio/providers/test` with `{id}` or the fields of the form: lists the models of
/// the provider (`provTest` of `providers.js` shows the summary of `providerTestSummary`).
pub async fn test_provider(base: String, body: Value) -> Result<TestAnswer, String> {
    crate::models::post_fallback(&base, "/api/studio/providers/test", body, "The test failed.")
        .await
        .and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
}

/// `DELETE /api/studio/providers` with `{id}`.
pub async fn delete_provider(base: String, id: String) -> Result<Value, String> {
    delete_json(&base, "/api/studio/providers", json!({ "id": id }), "Could not remove it.").await
}

/// A DELETE whose failures read the way the POSTs of the window do: the message of the server,
/// `fallback` when it sent none (`fail_words` of `models.rs`).
async fn delete_json(base: &str, path: &str, body: Value, fallback: &str) -> Result<Value, String> {
    let resp = match reqwest::Client::new().delete(format!("{base}{path}")).json(&body).send().await {
        Ok(r) => r,
        Err(_) => return Err(fallback.to_string()),
    };
    let status = resp.status();
    let data: Value = resp.json().await.unwrap_or(Value::Null);
    if status.is_success() {
        return Ok(data);
    }
    Err(crate::models::fail_words(&data, fallback))
}

/// The window's own line under the list: `Testing...` while a test runs, then the summary or the
/// message of the server (`provShow` of `providers.js`).
pub fn result_line(text: &str, ok: bool) -> (String, bool) {
    (text.to_string(), ok)
}

/// The question before a provider is removed (`confirmDialog` of `providers.js`).
pub fn remove_window<'a>(t: Tokens) -> Element<'a, Message> {
    let window = container(
        column![
            // the question of `confirmDialog`, alone — the Studio's dialog has no second line
            text("Remove this provider and its saved key?").size(16).font(bold()),
            row![
                Space::new().width(Length::Fill),
                button(text("Cancel").size(13))
                    .on_press(Message::ProvRemoveCancel)
                    .padding([6, 12])
                    .style(theme::ghost_button(t)),
                button(text("Remove").size(13))
                    .on_press(Message::ProvRemoveOk)
                    .padding([6, 12])
                    .style(move |_, status| {
                        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                        button::Style {
                            background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })),
                            text_color: t.danger,
                            border: iced::border::rounded(8),
                            ..Default::default()
                        }
                    }),
            ],
        ]
        .spacing(10),
    )
    .padding(16)
    .width(460)
    .style(theme::card(t, theme::R_LG, false));
    container(window).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// The fixtures the Studio's own `@pure` block produced
    /// (`tests/fixtures/providers_fixtures.json`, written by `providers_fixtures.js`).
    fn fixtures() -> Value {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/providers_fixtures.json");
        serde_json::from_str(&std::fs::read_to_string(path).expect("the fixtures of providers.js")).expect("json")
    }

    fn models_of(raw: &Value) -> Vec<ProviderModel> {
        // a `null` in the list is skipped, the way the window skips it
        let list: Vec<Option<ProviderModel>> =
            serde_json::from_value(raw.clone()).expect("the model list of the server");
        list.into_iter().flatten().collect()
    }

    /// `remoteModelId` and `parseRemoteModelId`: the id of a provider's model, and the way back.
    #[test]
    fn the_id_of_a_provider_model_round_trips() {
        let f = fixtures();
        for row in f["remote_id"].as_array().expect("remote_id") {
            let (provider, model, want) = (&row[0], &row[1], row[2].as_str().unwrap());
            assert_eq!(remote_model_id(provider.as_str().unwrap(), model.as_str().unwrap()), want, "{row}");
        }
        for row in f["parse_remote_id"].as_array().expect("parse_remote_id") {
            let raw = row[0].clone();
            // a non-string id cannot reach the function in Rust: the type is `&str`, and the
            // Studio answers null for it (`typeof id !== 'string'`)
            let got = raw.as_str().and_then(parse_remote_model_id);
            match &row[1] {
                Value::Null => assert!(got.is_none(), "{row}"),
                want => {
                    let got = got.unwrap_or_else(|| panic!("{row}"));
                    let want = serde_json::from_value::<Remote>(want.clone()).expect("a remote");
                    assert_eq!(got, want, "{row}");
                }
            }
        }
    }

    /// `mergeModelEntries`: the local names first, then one entry per model of a provider —
    /// duplicates and holes skipped, the label with the display name when the server sent one.
    #[test]
    fn the_picker_lists_the_local_models_first_and_then_the_ones_of_a_provider() {
        let f = fixtures();
        for row in f["merge_entries"].as_array().expect("merge_entries") {
            let locals: Vec<String> = row[0].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
            let models: Vec<ProviderModel> = row[1]
                .as_array()
                .map(|a| a.iter().filter_map(|v| serde_json::from_value::<Option<ProviderModel>>(v.clone()).ok().flatten()).collect())
                .unwrap_or_default();
            let selected = row[2].as_str().unwrap();
            let want: Vec<Entry> = serde_json::from_value(row[3].clone()).expect("the entries");
            assert_eq!(merge_model_entries(&locals, &models, selected), want, "{row}");
        }
    }

    /// `resolveSelection`: a selection that still exists is kept, one that is gone falls back.
    #[test]
    fn a_selection_that_is_gone_falls_back() {
        let f = fixtures();
        for row in f["resolve_selection"].as_array().expect("resolve_selection") {
            let entries = merge_model_entries(&["local-a".into()], &models_of(&fixtures()["merge_entries"][0][1]), "remote:openai/gpt-4o");
            let got = resolve_selection(row[0].as_str().unwrap(), &entries, row[1].as_str().unwrap());
            assert_eq!(got, row[2].as_str().unwrap(), "{row}");
        }
    }

    /// `providerPresetGroups`: cloud first, the network group when it says so or when it is local.
    #[test]
    fn the_templates_come_in_two_groups() {
        let f = fixtures();
        let presets: Vec<Preset> = serde_json::from_value(f["preset_input"].clone()).expect("the templates");
        let want: Vec<PresetGroup> = serde_json::from_value(f["preset_groups"].clone()).expect("the groups");
        assert_eq!(provider_preset_groups(&presets), want, "the five templates");
        // nothing at all: the two empty groups, in the same order
        let empty: Vec<PresetGroup> = serde_json::from_value(f["preset_groups_empty"].clone()).expect("the groups");
        assert_eq!(provider_preset_groups(&[]), empty, "no templates");
        // a value that is not a list is no list of templates (`Array.isArray` of the Studio)
        assert_eq!(provider_preset_groups(&[]), empty, "the fixture asks with a text instead");
    }

    /// `providerKeyLink`: https pages only, with the host that hands out the key.
    #[test]
    fn only_an_https_page_hands_out_a_key() {
        let f = fixtures();
        for row in f["key_link"].as_array().expect("key_link") {
            let preset = Preset { key_url: row[0].as_str().map(str::to_string), ..Preset::default() };
            let got = provider_key_link(&preset);
            match &row[1] {
                Value::Null => assert!(got.is_none(), "{row}"),
                want => {
                    let (url, host) = got.unwrap_or_else(|| panic!("{row}"));
                    assert_eq!(url, want["url"].as_str().unwrap(), "{row}");
                    assert_eq!(host, want["host"].as_str().unwrap(), "{row}");
                }
            }
        }
    }

    /// `providerTestSummary`: the line under *Test connection* — the answer of the server in,
    /// the line of the Studio out.
    #[test]
    fn the_line_of_a_test_says_what_it_found() {
        let f = fixtures();
        for row in f["test_summary"].as_array().expect("test_summary") {
            let answer: Option<TestAnswer> = serde_json::from_value(row[1].clone()).expect("the answer of the server");
            assert_eq!(provider_test_summary(answer.as_ref()), row[2].as_str().unwrap(), "{}", row[0]);
        }
    }

    /// `modelLabel`: what the label of the picker shows for a selection.
    #[test]
    fn the_label_of_the_picker_names_the_model() {
        let f = fixtures();
        for row in f["model_label"].as_array().expect("model_label") {
            let models: Vec<ProviderModel> = match &row[2] {
                Value::Null => Vec::new(),
                list => serde_json::from_value(list.clone()).expect("the model list"),
            };
            assert_eq!(model_label(row[1].as_str().unwrap(), &models), row[3].as_str().unwrap(), "{}", row[0]);
        }
    }

    /// `LOCAL_ONLY_FIELDS` and `cleanRemoteRequest`: the settings of the local engine a provider
    /// does not take, and the body with them out and the model's own name in.
    #[test]
    fn a_request_for_a_provider_loses_the_settings_of_the_local_engine() {
        let f = fixtures();
        let want: Vec<String> = serde_json::from_value(f["local_only_fields"].clone()).expect("the fields");
        assert_eq!(LOCAL_ONLY_FIELDS.to_vec(), want);
        let body: Value = serde_json::from_str(
            r#"{"model":"local-model","messages":[{"role":"user","content":"hi"}],"temperature":0.7,"top_k":40,"min_p":0.05,"repetition_penalty":1.1,"speculative":"draft","thinking_budget":1024,"chat_template_kwargs":{"a":1},"dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":2,"dry_penalty_last_n":-1,"mirostat":2,"mirostat_tau":0.8,"mirostat_eta":0.001,"grammar":"x","grammar_type":"y","regex":"z","tool_call_grammar":"w","tfs_z":1.0,"typical_p":0.95,"ignore_eos":true,"cache_prompt":true,"logit_bias_tokens":[1,2],"stream":true}"#,
        )
        .expect("the body of the fixture");
        let rows = f["clean_request"].as_array().expect("clean_request");
        assert_eq!(clean_remote_request(&body, rows[0][0].as_str().unwrap()), rows[0][1], "gpt-4o");
        let small: Value = serde_json::from_str(r#"{"temperature":0.2,"top_k":10,"messages":[]}"#).unwrap();
        assert_eq!(clean_remote_request(&small, "m"), rows[1][1], "keeps what is not local");
    }

    /// `chatEndpoint`: where a chat for a model id goes, and what the body becomes on the way.
    #[test]
    fn a_chat_of_a_provider_goes_through_the_server_with_the_model_its_own_name() {
        let f = fixtures();
        let sample: Value =
            serde_json::from_str(r#"{"model":"x","messages":[],"temperature":0.5,"top_k":40,"stream":true}"#).expect("the sample");
        for row in f["chat_endpoint"].as_array().expect("chat_endpoint") {
            let ep = chat_endpoint(row[0].as_str().unwrap());
            let want = &row[1];
            assert_eq!(ep.url, want["url"].as_str().unwrap(), "{row}");
            assert_eq!(ep.remote, want["remote"].as_bool().unwrap(), "{row}");
            assert_eq!(ep.provider.as_deref(), want["provider"].as_str(), "{row}");
            assert_eq!(ep.model.as_deref(), want["model"].as_str(), "{row}");
            assert_eq!(ep.transform(&sample), want["transform"], "{row}");
        }
    }
}
