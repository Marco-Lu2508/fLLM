//! Settings > MCP servers: the servers that are running, the registry that offers them and the
//! form that writes one into `mcp.json` (`spane-mcp` of `shell.html` and `mcp.js` of the Studio).
//!
//! Three cards on the page: the list with what every server does, *Find servers* (inserted before
//! the form by `mcpRegistry()`), and *Add a server* with its six fields.

use crate::app::Message;
use crate::models::{bold, card};
use crate::theme::{self, Tokens};
use iced::widget::text::Span;
use iced::widget::{
    button, column, container, pick_list, rich_text, row, scrollable, text, text_editor,
    text_input, Space,
};
use iced::{Alignment, Element, Font, Length};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

/// The answer of `GET /api/studio/mcp/servers`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ServersAnswer {
    #[serde(default)]
    pub servers: Vec<Server>,
    #[serde(default)]
    pub config_file: String,
    #[serde(default)]
    pub config_error: Option<String>,
}

/// One tool of a server (`status.tools` of the Studio's answer).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
}

/// One server as the list has it (`Entry::describe` of `server/api/mcp.rs`).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Server {
    pub name: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub headers: Vec<(String, String)>,
    /// The names of the environment variables — their values stay in `mcp.json`, they often are
    /// secrets.
    #[serde(default)]
    pub env_keys: Vec<String>,
    #[serde(default)]
    pub state: String,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub tool_count: u64,
    #[serde(default)]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub stderr: Option<String>,
}

impl Server {
    /// Whether the server is up (`state === 'ready'`).
    pub fn ready(&self) -> bool {
        self.state == "ready"
    }

    /// Whether it could not be started (`state === 'failed'`).
    pub fn failed(&self) -> bool {
        self.state == "failed"
    }
}

/// The line under a server: its address when it is a remote one, and its command with its
/// arguments joined by a space otherwise (`[s.command].concat(s.args).join(' ')`).
pub fn line_of(s: &Server) -> String {
    if !s.url.is_empty() {
        return format!("HTTP {}", s.url);
    }
    std::iter::once(s.command.as_str())
        .chain(s.args.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}

/// What follows the state of a ready server: ` - 3 tools`, and `tool` for one
/// (`s.tool_count === 1 ? '' : 's'`).
pub fn ready_tools(s: &Server) -> String {
    if !s.ready() {
        return String::new();
    }
    format!(" - {} tool{}", s.tool_count, if s.tool_count == 1 { "" } else { "s" })
}

/// The line under the form: where the servers are kept, and the problem the file has if the
/// server found one (`mcpFileInfo` of `mcp.js`).
pub fn file_info(answer: &ServersAnswer) -> String {
    match &answer.config_error {
        Some(problem) => format!("Saved in {} - problem: {}", answer.config_file, problem),
        None => format!("Saved in {}", answer.config_file),
    }
}

/// What the environment box shows when a server is edited: the names without their values, which
/// stay on disk (`s.env_keys.map((k) => `${k}=`)`).
pub fn env_lines_for_edit(s: &Server) -> String {
    s.env_keys.iter().map(|k| format!("{k}=")).collect::<Vec<_>>().join("\n")
}

/// The hint under the environment box while a server is edited, and nothing otherwise.
pub fn env_hint(editing: Option<&Server>) -> String {
    match editing {
        Some(s) if !s.env_keys.is_empty() => {
            "Saved values are not shown. A variable left without a value keeps its saved value."
                .into()
        }
        _ => String::new(),
    }
}

/// The headers as the box shows them when a server is edited (`${k}=${v}` per line).
pub fn headers_lines(s: &Server) -> String {
    s.headers.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("\n")
}

/// `NAME=value` lines to their names and values; a line without an `=` after its first character
/// is the reason the save was refused (`mcpPairs` of `mcp.js`).
pub fn header_pairs(text: &str) -> Option<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for line in text.split('\n') {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let i = t.find('=')?;
        if i < 1 {
            return None;
        }
        out.insert(t[..i].trim().to_string(), t[i + 1..].to_string());
    }
    Some(out)
}

/// The environment of the body: the same lines, where an empty value for a variable the server
/// already has means *keep what is saved* (`null` in `mcp.json`) — and a line without `=` is the
/// sentence of the toast that stops the save.
pub fn env_pairs(
    text: &str,
    editing: Option<&Server>,
) -> Result<BTreeMap<String, Option<String>>, String> {
    let mut out = BTreeMap::new();
    for line in text.split('\n') {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let i = t.find('=');
        let i = match i {
            Some(i) if i >= 1 => i,
            _ => return Err(format!("Environment line \"{t}\" needs the form NAME=value")),
        };
        let key = t[..i].trim().to_string();
        let value = t[i + 1..].to_string();
        let keeps = value.is_empty()
            && editing.is_some_and(|s| s.env_keys.iter().any(|k| k == &key));
        out.insert(key, (!keeps).then_some(value));
    }
    Ok(out)
}

/// What *Save and start* sends, or the sentence of the toast that stopped it — the environment is
/// read first, the way `mcpSave` reads it, so its sentence comes before the one of the headers.
pub fn save_body(
    name: &str,
    url: &str,
    command: &str,
    args_text: &str,
    env_text: &str,
    headers_text: &str,
    editing: Option<&Server>,
) -> Result<(serde_json::Value, String), String> {
    let env = env_pairs(env_text, editing)?;
    let headers = match header_pairs(headers_text) {
        Some(headers) => headers,
        None => return Err("Header lines need the form NAME=value".into()),
    };
    let name = name.trim().to_string();
    let body = serde_json::json!({
        "name": name,
        "url": url.trim(),
        "command": command.trim(),
        "args": args_text
            .split('\n')
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .collect::<Vec<_>>(),
        "env": env,
        "headers": headers,
    });
    Ok((body, name))
}

/// One variable of a registry entry (`variable()` of `server/api/mcp_registry.rs`).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Variable {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub value: String,
}

/// One way fLLM could run a server of the registry (`candidates` of an entry).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Candidate {
    pub kind: String,
    pub label: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: Vec<Variable>,
    #[serde(default)]
    pub headers: Vec<Variable>,
}

/// One server of the registry.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct RegServer {
    pub title: String,
    #[serde(default)]
    pub verified: bool,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub candidates: Vec<Candidate>,
}

/// The answer of `GET /api/studio/mcp/registry`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct RegistryAnswer {
    #[serde(default)]
    pub servers: Vec<RegServer>,
    #[serde(default)]
    pub next: Option<String>,
}

/// What *Use* puts into the form, and what the line under the list says afterwards.
#[derive(Debug, Clone, PartialEq)]
pub struct FormFill {
    pub name: String,
    pub url: String,
    pub command: String,
    pub args: String,
    pub env: String,
    pub headers: String,
    pub status: String,
}

/// The `{name}: {description}` line of a value the person still has to fill in
/// (`${h.name}: ${h.description || 'fill this in'}`).
fn note(v: &Variable) -> String {
    format!("{}: {}", v.name, if v.description.is_empty() { "fill this in" } else { &v.description })
}

/// *Use* on a candidate of the registry: the form it fills and the line it writes
/// (`list.addEventListener('click')` of `mcpRegistry`).
pub fn use_fill(entry: &RegServer, index: usize) -> Option<FormFill> {
    let candidate = entry.candidates.get(index)?;
    let mut notes: Vec<String> = Vec::new();
    let (url, command, args, env, headers) = if candidate.kind == "remote" {
        let headers: Vec<String> = candidate
            .headers
            .iter()
            .map(|h| format!("{}={}", h.name, h.value))
            .collect();
        for h in &candidate.headers {
            // a header that is required, or one whose value holds a `{placeholder}`, is filled in
            // by hand afterwards
            if h.required || h.value.contains('{') {
                notes.push(note(h));
            }
        }
        (
            candidate.url.clone().unwrap_or_default(),
            String::new(),
            String::new(),
            String::new(),
            headers.join("\n"),
        )
    } else {
        let env: Vec<String> = candidate
            .env
            .iter()
            .filter(|v| v.required || !v.value.is_empty())
            .map(|v| format!("{}={}", v.name, v.value))
            .collect();
        for v in &candidate.env {
            if v.required {
                notes.push(note(v));
            }
        }
        if candidate.args.iter().any(|a| is_angle(a)) {
            notes.push("replace the <...> arguments with real values".into());
        }
        (
            String::new(),
            candidate.command.clone().unwrap_or_default(),
            candidate.args.join("\n"),
            env.join("\n"),
            String::new(),
        )
    };
    let fill = FormFill {
        name: candidate.name.clone().unwrap_or_default(),
        url,
        command,
        args,
        env,
        headers,
        status: format!(
            "Filled in the form below with \u{201c}{}\u{201d}. {}Check it, then press Save.",
            entry.title,
            if notes.is_empty() {
                String::new()
            } else {
                format!("Still to do: {}. ", notes.join("; "))
            }
        ),
    };
    Some(fill)
}

/// Whether an argument is a `<...>` that has to be replaced by a real value (`/^<.+>$/`).
pub fn is_angle(arg: &str) -> bool {
    arg.len() > 2 && arg.starts_with('<') && arg.ends_with('>')
}

/// One resource a server publishes (`POST /api/studio/mcp/resources`).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Resource {
    pub uri: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

/// The answer of `POST /api/studio/mcp/resources`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ResourcesAnswer {
    #[serde(default)]
    pub resources: Vec<Resource>,
}

/// What a box on this page holds: nothing, the sentence while it is read, what came, or the
/// reason it could not be read.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Filled<T> {
    /// Nothing was read for this box yet, which is what the page shows before any button.
    #[default]
    Nothing,
    Busy,
    Done(T),
    Failed(String),
}

impl<T> Filled<T> {
    pub fn is_busy(&self) -> bool {
        matches!(self, Filled::Busy)
    }
}

/// What the Resources button of one row has read so far: the list of its resources and the text
/// of the last *Read* under it (`data-mcp-resources` of the row, which the page rebuilds whole
/// for every step of it).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResourcesState {
    pub list: Filled<Vec<Resource>>,
    pub read: Filled<String>,
    /// The resource the text under the list belongs to.
    pub read_uri: String,
}

/// The question before a server is removed (`confirmDialog` of `mcp.js`), with its name.
pub fn remove_window<'a>(t: Tokens, name: &str) -> Element<'a, Message> {
    let question = format!("Remove the MCP server \"{name}\"?");
    let window = container(
        column![
            text(question).size(16).font(bold()),
            row![
                Space::new().width(Length::Fill),
                button(text("Cancel").size(13))
                    .on_press(Message::McpRemoveCancel)
                    .padding([6, 12])
                    .style(theme::ghost_button(t)),
                button(text("Remove").size(13))
                    .on_press(Message::McpRemoveOk)
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
            ]
            .spacing(8),
        ]
        .spacing(10),
    )
    .padding(16)
    .width(460)
    .style(theme::card(t, theme::R_LG, false));
    container(iced::widget::opaque(window))
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(24)
        .into()
}

/// The description of the first card (`app-item-desc` of `spane-mcp`).
pub const TOOLS_HELP: &str = "Model Context Protocol servers give the model extra tools (files, search, databases, ...). Each server is a program that fLLM starts on this computer.";

/// The warning over the list (`color:var(--destructive)`): its first word is `<strong>` on the
/// page, so it is its own span here.
pub const WARNING: &str = "Warning: an MCP server runs with your user rights and can read, change and delete what you can. Only add servers you trust, and check the command and arguments.";

/// The description of *Find servers*, the two link labels with the rest of the sentence
/// (`<a data-mcp-site>` of the card).
pub const FIND_HELP: &str = "Search the official MCP registry or Smithery. \u{201c}Use\u{201d} fills in the form below; read the command, fill in the values it asks for, and save. Anybody can publish to these lists: only run what you trust.";

/// The two sources of the search (`#mcpRegSource`): the name in the request and the label of the
/// option.
pub const SOURCES: [(&str, &str); 2] = [("official", "Official registry"), ("smithery", "Smithery")];

/// The label of a source of the select.
pub fn source_label(id: &str) -> Option<&'static str> {
    SOURCES.iter().find(|(name, _)| *name == id).map(|(_, label)| *label)
}

/// The name in the request for a label of the select.
pub fn source_id(label: &str) -> &'static str {
    SOURCES.iter().find(|(_, name)| *name == label).map(|(id, _)| *id).unwrap_or("official")
}

fn warning_spans() -> Vec<Span<'static, String>> {
    vec![
        Span::new("Warning:"),
        Span::new(
            " an MCP server runs with your user rights and can read, change and delete what you can. Only add servers you trust, and check the command and arguments.",
        ),
    ]
}

fn find_spans(link_colour: Option<iced::Color>) -> Vec<Span<'static, String>> {
    // the two links are underlined in the colour of the theme, the way `<a>` of the page is
    let registry = Span::new("official MCP registry");
    let smithery = Span::new("Smithery");
    vec![
        Span::new("Search the "),
        Span {
            link: Some("https://registry.modelcontextprotocol.io/".into()),
            underline: true,
            color: link_colour,
            ..registry
        },
        Span::new(" or "),
        Span {
            link: Some("https://smithery.ai/servers".into()),
            underline: true,
            color: link_colour,
            ..smithery
        },
        Span::new(
            ". \u{201c}Use\u{201d} fills in the form below; read the command, fill in the values it asks for, and save. Anybody can publish to these lists: only run what you trust.",
        ),
    ]
}

/// The line between two rows (`border-bottom` of the page).
fn divider(t: Tokens) -> Element<'static, Message> {
    container(Space::new())
        .width(Length::Fill)
        .height(Length::Fixed(1.0))
        .style(move |_| container::Style {
            background: Some(iced::Background::Color(t.separator)),
            ..Default::default()
        })
        .into()
}

/// A text box of the form (`ui-input mono`).
fn input<'a>(
    t: Tokens,
    placeholder: &'a str,
    value: &'a str,
    typed: fn(String) -> Message,
    submit: Option<Message>,
) -> Element<'a, Message> {
    let mut field = text_input(placeholder, value).on_input(typed).size(13).padding([6, 10]).font(Font::MONOSPACE);
    if let Some(submit) = submit {
        field = field.on_submit(submit);
    }
    field
        .style(move |_, status| text_input::Style {
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

fn editor_style<'a>(t: Tokens) -> impl Fn(&iced::Theme, text_editor::Status) -> text_editor::Style + 'a {
    move |_, status| text_editor::Style {
        background: iced::Background::Color(t.surface_2),
        border: iced::Border {
            radius: 8.0.into(),
            width: 1.0,
            color: if matches!(status, text_editor::Status::Focused { .. }) { t.fg_soft } else { t.separator },
        },
        placeholder: t.muted,
        value: t.fg,
        selection: iced::Color { a: 0.35, ..t.primary },
    }
}

/// One of the three boxes over the buttons (`textarea` of the page).
fn area<'a>(
    t: Tokens,
    content: &'a text_editor::Content,
    placeholder: &'a str,
    action: fn(text_editor::Action) -> Message,
    height: f32,
) -> Element<'a, Message> {
    text_editor(content)
        .on_action(action)
        .placeholder(placeholder)
        .padding(8)
        .size(13)
        .font(Font::MONOSPACE)
        .height(Length::Fixed(height))
        .style(editor_style(t))
        .into()
}

/// A labelled field of the form (`app-card-item column` of the page).
fn field<'a>(t: Tokens, title: &'a str, control: Element<'a, Message>) -> Element<'a, Message> {
    column![
        text(title)
            .size(13)
            .color(t.fg)
            .font(Font { weight: iced::font::Weight::Semibold, ..Font::DEFAULT }),
        control
    ]
    .spacing(6)
    .into()
}

/// The part of a resource under its address: `name [mime] description`, each part only when it
/// is there (`${r.name || ''}${mime ? ' [' + mime + ']' : ''} ${description || ''}`).
fn resource_line(r: &Resource) -> String {
    let mut out = r.name.clone().unwrap_or_default();
    if let Some(mime) = &r.mime_type {
        out.push_str(&format!(" [{mime}]"));
    }
    out.push(' ');
    out.push_str(r.description.as_deref().unwrap_or(""));
    out
}

/// The box the Resources button fills (`data-mcp-resources` of the row).
fn resources_box<'a>(
    t: Tokens,
    server: &'a str,
    state: Option<&'a ResourcesState>,
) -> Element<'a, Message> {
    let state = match state {
        // the box is empty until the button was pressed
        None => return Space::new().height(Length::Fixed(0.0)).into(),
        Some(state) => state,
    };
    let mut body = column![].spacing(4);
    match &state.list {
        Filled::Nothing => {}
        Filled::Busy => body = body.push(text("Reading the resources ...").size(12).color(t.fg_soft)),
        Filled::Failed(reason) => body = body.push(text(reason.clone()).size(12).color(t.fg_soft)),
        Filled::Done(items) if items.is_empty() => {
            body = body.push(text("This server publishes no resources.").size(12).color(t.fg_soft))
        }
        Filled::Done(items) => {
            body = body.push(text(format!("{} resource(s)", items.len())).size(12).color(t.fg));
            for item in items {
                let mut line = row![
                    text(item.uri.clone()).size(12).font(Font::MONOSPACE).color(t.fg),
                    text(resource_line(item)).size(12).color(t.fg_soft),
                    Space::new().width(Length::Fill),
                ]
                .spacing(6)
                .align_y(Alignment::Center);
                let uri = item.uri.clone();
                let owner = server.to_string();
                line = line.push(
                    button(text("Read").size(12))
                        .on_press(Message::McpRead(owner, uri))
                        .padding([2, 8])
                        .style(theme::ghost_button(t)),
                );
                body = body.push(line);
            }
        }
    }
    match &state.read {
        Filled::Nothing => {}
        Filled::Busy => body = body.push(text("Reading ...").size(12).color(t.fg_soft)),
        Filled::Failed(reason) => body = body.push(text(reason.clone()).size(12).color(t.fg_soft)),
        Filled::Done(content) => body = body.push(
            scrollable(
                container(text(content.clone()).size(12).font(Font::MONOSPACE).color(t.fg_soft))
                    .width(Length::Fill)
                    .padding(6),
            )
            .height(Length::Fixed(220.0)),
        ),
    }
    body.into()
}

/// `models::card` with a title that is built on the spot: the title of the third card changes
/// with the form (*Add a server*, *Edit <name>*), so it is owned here instead of borrowed from
/// the page.
fn titled_card<'a>(t: Tokens, title: String, body: Element<'a, Message>) -> Element<'a, Message> {
    container(column![text(title).size(13).font(bold()).color(t.fg_soft), body].spacing(12))
        .padding(16)
        .width(Length::Fill)
        .style(theme::card(t, theme::R_LG, false))
        .into()
}

/// One server of the list with its four buttons (`mcpRender` of `mcp.js`).
fn server_block<'a>(
    t: Tokens,
    s: &'a Server,
    state: Option<&'a ResourcesState>,
) -> Element<'a, Message> {
    let state_colour = if s.ready() {
        t.primary
    } else if s.failed() {
        t.danger
    } else {
        t.muted
    };
    let mut head = row![
        text(s.name.clone()).size(13).color(t.fg),
        text(format!("{}{}", s.state, ready_tools(s))).size(12).color(state_colour),
        Space::new().width(Length::Fill),
    ]
    .spacing(8)
    .align_y(Alignment::Center);
    for (label, message) in [
        ("Resources", Message::McpResources(s.name.clone())),
        ("Restart", Message::McpRestart(s.name.clone())),
        ("Edit", Message::McpEdit(s.name.clone())),
        ("Remove", Message::McpRemove(s.name.clone())),
    ] {
        head = head.push(
            button(text(label).size(12))
                .on_press(message)
                .padding([2, 8])
                .style(theme::ghost_button(t)),
        );
    }
    let mut body = column![
        head,
        text(line_of(s)).size(12).font(Font::MONOSPACE).color(t.fg_soft),
    ]
    .spacing(4);
    if let Some(message) = s.message.as_deref().filter(|m| !m.is_empty()) {
        body = body.push(text(message.to_string()).size(12).color(t.danger));
    }
    if !s.tools.is_empty() {
        // `<details>` of the page: its tools are drawn open, the way the group of *Other servers
        // on your network* is drawn open in the providers card
        let mut tools = column![].spacing(2);
        for tool in &s.tools {
            tools = tools.push(
                row![
                    text(tool.name.clone()).size(12).font(Font::MONOSPACE).color(t.fg),
                    text(tool.description.clone().unwrap_or_default()).size(12).color(t.fg_soft),
                ]
                .spacing(6),
            );
        }
        body = body.push(column![text("Tools").size(12).color(t.fg_soft), tools].spacing(4));
    }
    if let Some(stderr) = s.stderr.as_deref().filter(|s| !s.is_empty()) {
        body = body.push(
            column![
                text("Error output of the server").size(12).color(t.fg_soft),
                text(stderr.to_string()).size(12).font(Font::MONOSPACE).color(t.fg_soft),
            ]
            .spacing(4),
        );
    }
    body = body.push(resources_box(t, &s.name, state));
    column![body, divider(t)].spacing(10).into()
}

/// One server of the registry with the buttons that fill the form with it.
fn registry_row<'a>(t: Tokens, entry: &'a RegServer, index: usize) -> Element<'a, Message> {
    let mut head = row![text(entry.title.clone()).size(13).color(t.fg)]
        .spacing(6)
        .align_y(Alignment::Center);
    if entry.verified {
        head = head.push(
            container(text("verified").size(11).color(t.fg_soft)).padding([1, 6]).style(theme::chip(t)),
        );
    }
    if !entry.version.is_empty() {
        head = head.push(text(entry.version.clone()).size(11).color(t.muted));
    }
    let mut body = column![head, text(entry.description.clone()).size(12).color(t.fg_soft)].spacing(4);
    if entry.candidates.is_empty() {
        body = body.push(
            text("fLLM cannot run this one yet (no npm, Python or container package and no HTTP address).")
                .size(12)
                .color(t.fg_soft),
        );
    } else {
        let mut buttons = row![].spacing(6);
        for (k, candidate) in entry.candidates.iter().enumerate() {
            buttons = buttons.push(
                button(text(format!("Use: {}", candidate.label)).size(12))
                    .on_press(Message::RegUse(index, k))
                    .padding([3, 8])
                    .style(theme::secondary_button(t)),
            );
        }
        body = body.push(buttons);
    }
    column![body, divider(t)].spacing(6).padding([4, 0]).into()
}

/// The data of the three cards of `spane-mcp`.
#[derive(Clone, Copy)]
pub struct McpView<'a> {
    /// The list, its line under the form and the sentence that replaces it when the read failed.
    pub servers: &'a [Server],
    pub file_info: &'a str,
    pub list_say: &'a str,
    /// The six fields, the hint under the environment box and whether a server is being edited.
    pub name: &'a str,
    pub url: &'a str,
    pub command: &'a str,
    pub args: &'a text_editor::Content,
    pub env: &'a text_editor::Content,
    pub headers: &'a text_editor::Content,
    pub env_hint: &'a str,
    pub editing: bool,
    /// What the Resources buttons read, and what *Find servers* holds.
    pub resources: &'a HashMap<String, ResourcesState>,
    pub reg_source: &'a str,
    pub reg_query: &'a str,
    pub reg_status: &'a str,
    pub reg_found: &'a [RegServer],
    pub reg_has_next: bool,
}

/// The three cards of `spane-mcp`: what is running, where to find one and the form that writes
/// one into `mcp.json`.
pub fn section<'a>(t: Tokens, v: McpView<'a>) -> Element<'a, Message> {
    // ---- Tools (MCP) --------------------------------------------------------------------
    let mut tools = column![].spacing(10);
    tools = tools.push(text(TOOLS_HELP).size(13).color(t.fg_soft));
    tools = tools.push(
        rich_text(warning_spans())
            .size(13)
            .color(t.danger),
    );
    if !v.list_say.is_empty() {
        tools = tools.push(text(v.list_say).size(13).color(t.fg_soft));
    } else if v.servers.is_empty() {
        tools = tools.push(
            text("No MCP servers yet. Add one below.").size(13).color(t.fg_soft),
        );
    } else {
        let mut list = column![].spacing(0);
        for server in v.servers {
            list = list.push(server_block(t, server, v.resources.get(&server.name)));
        }
        tools = tools.push(list);
    }

    // ---- Find servers (inserted before the form by `mcpRegistry`) ------------------------
    let mut find = column![].spacing(10);
    find = find.push(rich_text(find_spans(Some(t.primary))).size(13).color(t.fg_soft));
    find = find.push(
        row![
            pick_list(
                SOURCES.iter().map(|(_, label)| *label).collect::<Vec<_>>(),
                source_label(v.reg_source),
                |label: &'static str| Message::RegSource(source_id(label).to_string()),
            )
            .padding([6, 10]),
            input(t, "Search (for example github, postgres, files)", v.reg_query, Message::RegQuery, Some(Message::RegSearch)),
            button(text("Search").size(13))
                .on_press(Message::RegSearch)
                .padding([6, 12])
                .style(theme::secondary_button(t)),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    );
    if !v.reg_found.is_empty() {
        let mut list = column![].spacing(0);
        for (index, entry) in v.reg_found.iter().enumerate() {
            list = list.push(registry_row(t, entry, index));
        }
        find = find.push(list);
    }
    if v.reg_has_next {
        find = find.push(
            button(text("Show more").size(13))
                .on_press(Message::RegMore)
                .padding([6, 12])
                .style(theme::ghost_button(t)),
        );
    }
    if !v.reg_status.is_empty() {
        find = find.push(text(v.reg_status).size(11).font(Font::MONOSPACE).color(t.fg_soft));
    }

    // ---- Add a server --------------------------------------------------------------------
    let mut form = column![].spacing(10);
    form = form.push(field(t, "Name", input(t, "files", v.name, Message::McpName, None)));
    form = form.push(field(
        t,
        "URL (a remote server instead of a command)",
        input(t, "https://mcp.example.com/mcp", v.url, Message::McpUrl, None),
    ));
    form = form.push(
        text("A server reached over HTTP (Streamable HTTP). Fill either the command or the URL.")
            .size(12)
            .color(t.fg_soft),
    );
    form = form.push(field(t, "Command", input(t, "npx", v.command, Message::McpCommand, None)));
    form = form.push(field(
        t,
        "Arguments (one per line)",
        area(t, v.args, "-y", Message::McpArgs, 74.0),
    ));
    form = form.push(field(
        t,
        "Environment variables (NAME=value, one per line)",
        area(t, v.env, "API_KEY=...", Message::McpEnv, 74.0),
    ));
    if !v.env_hint.is_empty() {
        form = form.push(text(v.env_hint).size(12).color(t.fg_soft));
    }
    form = form.push(field(
        t,
        "HTTP headers for a remote server (NAME=value, one per line)",
        area(t, v.headers, "Authorization=Bearer ", Message::McpHeaders, 56.0),
    ));
    form = form.push(
        text("A value may reference an environment variable as ${NAME}, so a token does not have to be written into the file.")
            .size(12)
            .color(t.fg_soft),
    );
    form = form.push(
        row![
            button(text("Save and start").size(13))
                .on_press(Message::McpSave)
                .padding([6, 12])
                .style(theme::primary_button(t)),
            button(text("Clear").size(13))
                .on_press(Message::McpClear)
                .padding([6, 12])
                .style(theme::ghost_button(t)),
        ]
        .spacing(6),
    );
    if !v.file_info.is_empty() {
        form = form.push(text(v.file_info).size(11).font(Font::MONOSPACE).color(t.fg_soft));
    }

    let title = if v.editing {
        format!("Edit {}", v.name)
    } else {
        "Add a server".into()
    };
    column![
        card(t, "Tools (MCP)", tools.into()),
        card(t, "Find servers", find.into()),
        titled_card(t, title, form.into()),
    ]
    .spacing(14)
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn server() -> Server {
        Server {
            name: "files".into(),
            command: "npx".into(),
            args: vec!["-y".into(), "@modelcontextprotocol/server-filesystem".into()],
            url: String::new(),
            headers: vec![],
            env_keys: vec!["API_KEY".into(), "TOKEN".into()],
            state: "ready".into(),
            message: None,
            tool_count: 1,
            tools: vec![Tool { name: "read".into(), description: Some("Read a file".into()) }],
            stderr: None,
        }
    }

    /// The two cards that are drawn as `rich_text` keep exactly the sentences of the page: the
    /// warning with its bold word in front, and the description with its two links in the middle.
    /// The window cannot look inside a `rich_text`, so this is where they are held to it.
    #[test]
    fn the_rich_sentences_are_the_ones_of_the_page() {
        assert_eq!(
            warning_spans()
                .into_iter()
                .map(|s| s.text.to_string())
                .collect::<Vec<_>>()
                .join(""),
            WARNING
        );
        let links: Vec<String> = find_spans(None)
            .into_iter()
            .filter_map(|s| s.link.clone())
            .collect();
        assert_eq!(
            links,
            vec![
                "https://registry.modelcontextprotocol.io/".to_string(),
                "https://smithery.ai/servers".to_string(),
            ],
            "the two links of the description, in the order of the sentence"
        );
        assert_eq!(
            find_spans(None)
                .into_iter()
                .map(|s| s.text.to_string())
                .collect::<Vec<_>>()
                .join(""),
            FIND_HELP
        );
    }

    /// The line under a server: its address when it has one, its command with its arguments when
    /// not — joined by a space, exactly as the page joins them.
    #[test]
    fn the_line_under_a_server_is_its_command_with_its_arguments() {
        let mut s = server();
        assert_eq!(line_of(&s), "npx -y @modelcontextprotocol/server-filesystem");
        s.command = String::new();
        s.args = vec![];
        assert_eq!(line_of(&s), "", "nothing at all joins to nothing");
        s.url = "https://mcp.example.com/mcp".into();
        assert_eq!(line_of(&s), "HTTP https://mcp.example.com/mcp");
    }

    /// What follows the state: nothing until a server is ready, then its tools — and `tool` when
    /// there is exactly one.
    #[test]
    fn the_state_of_a_ready_server_counts_its_tools() {
        let mut s = server();
        assert_eq!(ready_tools(&s), " - 1 tool", "one tool is not tools");
        s.tool_count = 3;
        assert_eq!(ready_tools(&s), " - 3 tools");
        s.state = "starting".into();
        assert_eq!(ready_tools(&s), "", "a server that is still starting says nothing");
        s.state = "failed".into();
        assert_eq!(ready_tools(&s), "");
        assert!(s.failed() && !s.ready());
    }

    /// The line under the form: where the file is, and the problem it has when one was found.
    #[test]
    fn the_line_under_the_form_names_the_file_and_its_problem() {
        let answer: ServersAnswer = serde_json::from_value(json!({
            "servers": [],
            "config_file": "/home/m/.fllm/mcp.json",
            "config_error": null
        }))
        .expect("an answer of the servers");
        assert_eq!(file_info(&answer), "Saved in /home/m/.fllm/mcp.json");
        let broken: ServersAnswer = serde_json::from_value(json!({
            "servers": [],
            "config_file": "/home/m/.fllm/mcp.json",
            "config_error": "line 3 is no JSON"
        }))
        .expect("an answer of the servers");
        assert_eq!(
            file_info(&broken),
            "Saved in /home/m/.fllm/mcp.json - problem: line 3 is no JSON"
        );
    }

    /// The environment box while a server is edited: the names without their values, which stay
    /// on disk, and the sentence that says so.
    #[test]
    fn editing_shows_the_names_and_the_sentence_under_them() {
        let s = server();
        assert_eq!(env_lines_for_edit(&s), "API_KEY=\nTOKEN=");
        assert_eq!(
            env_hint(Some(&s)),
            "Saved values are not shown. A variable left without a value keeps its saved value."
        );
        assert_eq!(env_hint(None), "", "adding a server has nothing to hint");
    }

    /// `NAME=value` lines: the headers as they are, and the environment where an empty value for
    /// a saved variable means *keep what is there*.
    #[test]
    fn the_boxes_read_lines_of_name_and_value() {
        assert_eq!(
            header_pairs("A=1\nB = two ").expect("headers"),
            BTreeMap::from([("A".into(), "1".into()), ("B".into(), " two".into())]),
            "the whole line is trimmed, the key is trimmed again, and the value is what stands \
             after the = as it stands there"
        );
        assert!(header_pairs("no equals").is_none(), "a line without = is refused");
        assert!(header_pairs("=at the start").is_none(), "a name needs a character before the =");
        assert!(header_pairs("").is_some_and(|m| m.is_empty()), "an empty box is no error");

        let env = env_pairs("API_KEY=\nNEW=abc", Some(&server())).expect("the environment");
        assert_eq!(env.get("API_KEY"), Some(&None), "an empty value for a saved variable keeps it");
        assert_eq!(env.get("NEW"), Some(&Some("abc".into())), "a value that is there is sent");
        assert_eq!(
            env_pairs("BROKEN", None).unwrap_err(),
            "Environment line \"BROKEN\" needs the form NAME=value"
        );
        assert_eq!(
            env_pairs("=no name", None).unwrap_err(),
            "Environment line \"=no name\" needs the form NAME=value"
        );
    }

    /// *Save and start* sends what the six fields hold, in the shape of `mcpSave`, or the toast
    /// that stopped it.
    #[test]
    fn the_body_is_built_the_way_the_button_builds_it() {
        let (body, name) = save_body(
            "  files  ",
            "",
            " npx ",
            "-y\n\n  /path  \n",
            "API_KEY=\nNEW=abc",
            "Authorization=Bearer x",
            Some(&server()),
        )
        .expect("a body");
        assert_eq!(name, "files", "the name comes back trimmed");
        assert_eq!(body["name"], "files");
        assert_eq!(body["command"], "npx");
        assert_eq!(body["url"], "");
        assert_eq!(
            body["args"],
            json!(["-y", "/path"]),
            "the arguments are trimmed and empty lines are dropped"
        );
        assert_eq!(body["env"]["API_KEY"], Value::Null, "the saved variable keeps its value");
        assert_eq!(body["env"]["NEW"], json!("abc"));
        assert_eq!(body["headers"]["Authorization"], json!("Bearer x"));

        let bad_env = save_body("a", "", "b", "", "=x", "", None).unwrap_err();
        assert!(
            bad_env.starts_with("Environment line"),
            "the environment is read first: {bad_env}"
        );
        let bad_header = save_body("a", "", "b", "", "", "no equals", None).unwrap_err();
        assert_eq!(bad_header, "Header lines need the form NAME=value");
    }

    /// *Use* fills the form of the candidate and writes the line under the list — a remote one
    /// with its address and headers, a local one with its command, arguments and variables.
    #[test]
    fn use_fills_the_form_and_says_what_is_left() {
        let entry: RegServer = serde_json::from_value(json!({
            "title": "GitHub MCP",
            "verified": true,
            "version": "1.2.0",
            "description": "GitHub tools",
            "candidates": [
                {
                    "kind": "local",
                    "label": "npm github-mcp",
                    "name": "github",
                    "command": "npx",
                    "args": ["-y", "@github/mcp-server", "<GITHUB_TOKEN>"],
                    "env": [
                        {"name": "GITHUB_PERSONAL_ACCESS_TOKEN", "description": "a token", "required": true, "secret": true, "value": ""},
                        {"name": "OPTIONAL", "description": "", "required": false, "secret": false, "value": ""}
                    ]
                },
                {
                    "kind": "remote",
                    "label": "remote server https://mcp.github.com",
                    "name": "github-remote",
                    "url": "https://mcp.github.com",
                    "headers": [
                        {"name": "Authorization", "description": "", "required": false, "secret": true, "value": "Bearer {TOKEN}"},
                        {"name": "X-Org", "description": "optional", "required": false, "secret": false, "value": ""},
                        {"name": "X-Required", "description": "an org to send", "required": true, "secret": false, "value": ""}
                    ]
                }
            ]
        }))
        .expect("an entry of the registry");

        let fill = use_fill(&entry, 0).expect("the first candidate");
        assert_eq!(fill.name, "github");
        assert_eq!(fill.command, "npx");
        assert_eq!(fill.args, "-y\n@github/mcp-server\n<GITHUB_TOKEN>");
        assert_eq!(fill.env, "GITHUB_PERSONAL_ACCESS_TOKEN=", "an optional variable with no value is left out");
        assert_eq!(fill.headers, "", "a local candidate has no headers");
        assert_eq!(
            fill.status,
            "Filled in the form below with \u{201c}GitHub MCP\u{201d}. Still to do: GITHUB_PERSONAL_ACCESS_TOKEN: a token; replace the <...> arguments with real values. Check it, then press Save."
        );

        let remote = use_fill(&entry, 1).expect("the second candidate");
        assert_eq!(remote.url, "https://mcp.github.com");
        assert_eq!(remote.command, "");
        assert_eq!(
            remote.headers,
            "Authorization=Bearer {TOKEN}\nX-Org=\nX-Required=",
            "every header of the candidate goes into the box, in the order it came"
        );
        assert_eq!(
            remote.status,
            "Filled in the form below with \u{201c}GitHub MCP\u{201d}. Still to do: Authorization: fill this in; X-Required: an org to send. Check it, then press Save.",
            "a header whose value holds a placeholder is a note, a required one is a note, and a header that is neither is not"
        );
        assert!(use_fill(&entry, 9).is_none(), "an index that is not there fills nothing");
    }

    /// An argument is a placeholder when the whole of it stands in angle brackets.
    #[test]
    fn an_argument_in_angle_brackets_is_one_to_replace() {
        assert!(is_angle("<TOKEN>"));
        assert!(!is_angle("a<TOKEN>"), "the whole argument has to stand in brackets");
        assert!(!is_angle("<>"), "there has to be something inside");
        assert!(!is_angle("-y"));
    }
}
