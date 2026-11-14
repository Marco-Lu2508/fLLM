//! Settings > Logs, Server and API, API keys and the tokenizer: what the server does for other
//! programs, the keys it hands out, its latest messages and the way it splits text into tokens
//! (`spane-logs`, `spane-local-api-server` and `spane-tokenizer` of `shell.html`).
//!
//! This file grows with the row: the Logs card is the first of the four.

use crate::app::Message;
use crate::models::{bold, card};
use crate::theme::{self, Tokens};
use iced::widget::{button, column, container, pick_list, row, scrollable, svg, text, text_editor, text_input, Space};
use iced::{Alignment, Element, Font, Length};

/// The answer of `GET /api/studio/logs?lines=300`.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Logs {
    #[serde(default)]
    pub lines: Vec<String>,
}

/// What the `<pre>` of the Logs card holds (`refreshLogs` of `models.js`): the lines joined, or
/// the sentence for a log that has nothing in it yet.
pub fn logs_text(lines: &[String]) -> String {
    if lines.is_empty() {
        "Nothing has been logged yet.".into()
    } else {
        lines.join("\n")
    }
}

/// `GET /api/studio/logs?lines=300`.
pub async fn fetch_logs(base: String) -> Result<Logs, String> {
    crate::models::get_json(&base, "/api/studio/logs?lines=300")
        .await
        .and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
}

/// The line under the title of the Logs card (`shell.html`).
pub const LOGS_HELP: &str = "The latest messages of the server. Useful when something does not work; copy them into a bug report.";

/// The card `spane-logs` of `shell.html`: what the server wrote lately, with the two buttons of
/// its header. The box is a scrollable of the height the Studio gives it (`max-height:480px`),
/// and it wraps the lines the way `white-space: pre-wrap` does.
pub fn logs_card<'a>(t: Tokens, body: &'a str) -> Element<'a, Message> {
    let mut b = column![].spacing(10);
    b = b.push(text(LOGS_HELP).size(13).color(t.fg_soft));
    b = b.push(
        row![
            button(text("Refresh").size(12))
                .on_press(Message::LogsRefresh)
                .padding([5, 10])
                .style(crate::app::starter_style(t)),
            button(text("Copy").size(12))
                .on_press(Message::CopyText(body.to_string()))
                .padding([5, 10])
                .style(crate::app::starter_style(t)),
        ]
        .spacing(6),
    );
    b = b.push(
        scrollable(
            container(text(body.to_string()).size(12).font(Font::MONOSPACE).color(t.fg_soft))
                .width(Length::Fill)
                .padding(10)
                .style(theme::inset(t)),
        )
        .height(Length::Fixed(480.0)),
    );
    card(t, "Logs", b.into())
}

/// The limits the server accepts for everyone (`rate_limit` of `GET /api/studio/keys`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct RateLimit {
    #[serde(default)]
    pub requests_per_minute: Option<u64>,
    #[serde(default)]
    pub tokens_per_minute: Option<u64>,
}

/// One key of the server (`list_keys` of `security.rs`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Key {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub masked: String,
    #[serde(default)]
    pub admin: bool,
    #[serde(default)]
    pub rpm_limit: Option<u64>,
    #[serde(default)]
    pub tpm_limit: Option<u64>,
    #[serde(default)]
    pub requests_accepted: u64,
    #[serde(default)]
    pub tokens_used: u64,
    #[serde(default)]
    pub rejected: u64,
}

/// The answer of `GET /api/studio/keys`.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct KeyList {
    #[serde(default)]
    pub auth_required: bool,
    #[serde(default)]
    pub rate_limit: RateLimit,
    #[serde(default)]
    pub keys: Vec<Key>,
}

/// The answer of `POST /api/studio/keys`: the secret, shown once.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct CreatedKey {
    #[serde(default)]
    pub secret: String,
    #[serde(default)]
    pub key: Key,
}

/// A number as JavaScript writes one into a request body (`parseFloat(x) || null` of
/// `listeners-panels.js`): nothing at all for a box that holds no number, a whole number without
/// a dot and a fraction with one, the way `JSON.stringify` writes a JavaScript number. An empty
/// box therefore reaches the server as `null`, which it reads as *no limit*.
pub fn js_number(typed: &str) -> Option<serde_json::Value> {
    let n: f64 = typed.trim().parse().ok()?;
    if !n.is_finite() {
        return None;
    }
    if n.fract() == 0.0 && n.abs() < 1e15 {
        Some(serde_json::json!(n as i64))
    } else {
        Some(serde_json::json!(n))
    }
}

/// The limit of a row of the table (`{n} per minute`, or *Default* without one).
pub fn limit_line(rpm: Option<u64>) -> String {
    match rpm {
        Some(n) => format!("{n} per minute"),
        None => "Default".into(),
    }
}

/// `GET /api/studio/keys`: the keys and the limits of the server.
pub async fn fetch_keys(base: String) -> Result<KeyList, String> {
    crate::models::get_json(&base, "/api/studio/keys")
        .await
        .and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
}

/// `POST /api/studio/keys` with `{label, rpm, tpm, admin}` — the secret comes back once
/// (`btnCreateKey` of `listeners-panels.js`).
pub async fn create_key(base: String, body: serde_json::Value) -> Result<CreatedKey, String> {
    crate::models::post_fallback(&base, "/api/studio/keys", body, "The key could not be created")
        .await
        .and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
}

/// `DELETE /api/studio/keys/{id}` (`data-revoke-key` of `refreshApiKeys`).
pub async fn delete_key(base: String, id: String) -> Result<serde_json::Value, String> {
    crate::models::delete_json(&base, &format!("/api/studio/keys/{id}"), serde_json::json!({}), "The key could not be revoked").await
}

/// The routes of the tester, as the select of `shell.html` offers them: the value
/// (`METHOD:path`, what the window sends) and the label (what the row shows).
pub const ROUTES: [(&str, &str); 8] = [
    ("POST:/v1/chat/completions", "POST /v1/chat/completions (OpenAI Chat)"),
    ("POST:/v1/messages", "POST /v1/messages (Anthropic Messages)"),
    ("POST:/v1/responses", "POST /v1/responses (OpenAI Responses)"),
    ("POST:/api/chat", "POST /api/chat (Ollama Chat)"),
    ("POST:/v1/embeddings", "POST /v1/embeddings (Embeddings)"),
    ("POST:/v1/rerank", "POST /v1/rerank (Document Reranker)"),
    ("GET:/v1/models", "GET /v1/models (Models List)"),
    ("GET:/props", "GET:/props (llama.cpp Server Properties)"),
];

/// What the request box holds for one of those routes when nothing was typed yet
/// (`updateRouteTesterSample` of `panels.js`); the two GET routes take no body.
pub fn route_sample(route: &str, model: &str) -> String {
    let m = if model.is_empty() { "your-model" } else { model };
    let body = match route {
        "POST:/v1/chat/completions" => serde_json::json!({"model": m, "messages": [{"role": "user", "content": "Hello! Answer briefly."}], "max_tokens": 48}),
        "POST:/v1/messages" => serde_json::json!({"model": m, "max_tokens": 48, "messages": [{"role": "user", "content": "Hello via Anthropic API!"}]}),
        "POST:/v1/responses" => serde_json::json!({"model": m, "input": "Three facts about Rust.", "max_output_tokens": 48}),
        "POST:/api/chat" => serde_json::json!({"model": m, "stream": false, "messages": [{"role": "user", "content": "Hello via Ollama API!"}]}),
        "POST:/v1/embeddings" => serde_json::json!({"model": m, "input": ["Local AI in Rust", "SIMD Kernels"]}),
        "POST:/v1/rerank" => serde_json::json!({"model": m, "query": "SIMD CPU acceleration", "documents": ["AVX-512 accelerates dot products.", "Apple pie recipe."]}),
        _ => return String::new(),
    };
    serde_json::to_string_pretty(&body).unwrap_or_default()
}

/// One answer of the tester: pretty when it is JSON, the text as it came when not
/// (`JSON.stringify(JSON.parse(txt), null, 2)` of `btnRunRouteTest`). One difference to that
/// line: the JSON of Rust keeps the keys of an object in the order they were written in the file,
/// JavaScript keeps the order they came in — so a pretty answer lists them in the order of the
/// parser (sorted), which says nothing about the answer itself.
pub fn pretty_answer(text: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| text.to_string()),
        Err(_) => text.to_string(),
    }
}

/// `selRouteTest` + `Send Request`: one call to the route of the select. The answer is shown
/// whatever status it came with, the way the window shows it.
pub async fn run_route(base: String, route: String, body: Option<serde_json::Value>) -> Result<String, String> {
    let (method, path) = route.split_once(':').unwrap_or(("GET", route.as_str()));
    let client = reqwest::Client::new();
    let request = if method == "POST" {
        client
            .post(format!("{base}{path}"))
            .json(&body.unwrap_or_else(|| serde_json::json!({})))
    } else {
        client.get(format!("{base}{path}"))
    };
    let text = request.send().await.map_err(|e| e.to_string())?.text().await.map_err(|e| e.to_string())?;
    Ok(pretty_answer(&text))
}

/// `GET /metrics`: the text of the page as it comes (`refreshPrometheus` of `panels.js`).
pub async fn fetch_metrics(base: String) -> Result<String, String> {
    crate::models::get_text(&base, "/metrics").await
}

/// The labels of the tester's select, in the order `shell.html` offers them.
pub fn route_labels() -> Vec<&'static str> {
    ROUTES.iter().map(|(_, label)| *label).collect()
}

/// The label of the route the select shows for its value.
pub fn route_label(value: &str) -> &'static str {
    ROUTES.iter().find(|(v, _)| *v == value).map(|(_, l)| *l).unwrap_or(ROUTES[0].1)
}

/// The value of a row the user chose in the select.
pub fn route_value(label: &str) -> &'static str {
    ROUTES.iter().find(|(_, l)| *l == label).map(|(v, _)| *v).unwrap_or(ROUTES[0].0)
}

/// One token of `POST /tokenize` as the card shows it: its id and the piece of text it is
/// (the answer holds an id alone when the window did not ask for pieces).
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(untagged)]
pub enum Token {
    Id(u64),
    Piece { id: u64, piece: String },
}

impl Token {
    pub fn id(&self) -> u64 {
        match self {
            Token::Id(id) => *id,
            Token::Piece { id, .. } => *id,
        }
    }

    /// The text of the piece: what the model sees for this token. An answer without pieces is
    /// its id as text (`String(tok)` of `runTokenize`).
    pub fn piece(&self) -> String {
        match self {
            Token::Id(id) => id.to_string(),
            Token::Piece { piece, .. } => piece.clone(),
        }
    }
}

/// The answer of `POST /tokenize`.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Tokenized {
    #[serde(default)]
    pub tokens: Vec<Token>,
}

/// The line over the wall (`runTokenize` of `panels.js`): how many tokens, and how many bytes of
/// text they are (a byte in Rust is a byte in the browser's `Blob`).
pub fn token_stats(tokens: usize, bytes: usize) -> String {
    format!("{tokens} tokens · {bytes} bytes")
}

/// The hues the wall cycles through (`hues` of `runTokenize`), and the colour of a pill:
/// `hsla(H, 65%, 45%, 0.18)`.
pub const TOKEN_HUES: [u16; 6] = [145, 200, 265, 25, 330, 175];

/// One of those hues as a colour with the alpha of a pill (`hsl` of the Studio's `hsla`).
pub fn token_colour(hue: u16) -> iced::Color {
    let h = (hue as f32 % 360.0) / 60.0;
    let s = 0.65f32;
    let l = 0.45f32;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    iced::Color { r: r + m, g: g + m, b: b + m, a: 0.18 }
}

/// `POST /tokenize` with the body of `runTokenize`.
pub async fn tokenize(base: String, content: String, add_special: bool) -> Result<Tokenized, String> {
    let body = serde_json::json!({ "content": content, "add_special": add_special, "with_pieces": true });
    crate::models::post_fallback(&base, "/tokenize", body, "The text could not be tokenized.")
        .await
        .and_then(|v| serde_json::from_value(v).map_err(|e| e.to_string()))
}

/// `POST /apply-template` with the messages of the box (`runJinjaTemplate` of `panels.js`).
pub async fn apply_template(base: String, messages: serde_json::Value) -> Result<String, String> {
    let body = serde_json::json!({ "messages": messages });
    let answer: serde_json::Value = crate::models::post_fallback(&base, "/apply-template", body, "The text could not be rendered.")
        .await?;
    Ok(match answer.get("prompt").and_then(serde_json::Value::as_str) {
        Some(prompt) => prompt.to_string(),
        None => serde_json::to_string_pretty(&answer).unwrap_or_default(),
    })
}

/// What the settings page needs for the two cards of `spane-local-api-server`.
#[derive(Clone, Copy)]
pub struct ServerView<'a> {
    /// The address the server answers on; the address of the card is it with `/v1` behind it
    /// (`window.location.origin + "/v1"` of `panels.js`).
    pub base: &'a str,
    /// The chip *Running* is only there when the server answers (the Studio's page comes from the
    /// server, so its chip is static).
    pub running: bool,
    /// The limits as the server sent them (the boxes are filled from them).
    pub rpm: &'a str,
    pub tpm: &'a str,
    /// The keys of the table, and whether the answer came.
    pub keys: &'a [Key],
    pub keys_loaded: bool,
    /// The boxes of the form above the table.
    pub label: &'a str,
    pub key_rpm: &'a str,
    pub key_tpm: &'a str,
    pub admin: bool,
    /// The secret that was made, shown once (`newKeyBanner`).
    pub new_key: Option<&'a str>,
    /// Without a model the server has no key routes (`state.setup` of the Studio).
    pub no_model: bool,
    /// The model the sample bodies of the tester name (`state.selectedModel` of the Studio).
    pub model: &'a str,
    /// The route of the tester (`selRouteTest`) and what is in its request box.
    pub route: &'a str,
    pub request: &'a text_editor::Content,
    /// The answer of the last request, or the line the page starts with (`preRouteRes`).
    pub answer: &'a str,
    /// The statistics behind their own line, and what `/metrics` answered (`prePrometheusMetrics`).
    pub metrics_open: bool,
    pub metrics: &'a str,
}

/// The first card of `spane-local-api-server`: where the server listens and what it accepts.
pub fn server_card<'a>(t: Tokens, v: ServerView<'a>) -> Element<'a, Message> {
    let url = format!("{}/v1", v.base);
    let mut body = column![].spacing(14);
    let mut head = column![text(SERVER_HELP).size(13).color(t.fg_soft)].spacing(8);
    if v.running {
        head = head.push(
            container(
                row![
                    container(Space::new().width(6).height(6)).style(move |_| container::Style {
                        background: Some(iced::Background::Color(t.ok)),
                        border: iced::border::rounded(99),
                        ..Default::default()
                    }),
                    text("Running").size(12).color(t.fg_soft),
                ]
                .spacing(6)
                .align_y(Alignment::Center),
            )
            .padding([3, 8])
            .style(move |_| container::Style {
                background: Some(iced::Background::Color(t.surface_2)),
                border: iced::Border { radius: 999.0.into(), width: 1.0, color: t.separator },
                ..Default::default()
            }),
        );
    }
    body = body.push(head);

    // Address: the line of the Studio, in the monospace font it gives it
    body = body.push(
        row![
            column![text("Address").size(14), text(url.clone()).size(12).font(Font::MONOSPACE).color(t.muted)]
                .spacing(3)
                .width(Length::Fill),
            container(
                button(text("Copy URL").size(12))
                    .on_press(Message::CopyText(url.clone()))
                    .padding([5, 10])
                    .style(crate::app::starter_style(t)),
            )
            .width(280)
            .align_x(iced::alignment::Horizontal::Right),
        ]
        .spacing(20)
        .align_y(Alignment::Center),
    );

    // Request limits: the two boxes and the button that writes them
    let control = row![
        number_box(t, v.rpm, Message::RateRpm, "RPM"),
        number_box(t, v.tpm, Message::RateTpm, "TPM"),
        button(text("Apply").size(12))
            .on_press(Message::RateApply)
            .padding([5, 10])
            .style(crate::app::starter_style(t)),
    ]
    .spacing(6);
    body = body.push(
        row![
            column![text("Request limits").size(14), text(RATE_HELP).size(12).color(t.muted)]
                .spacing(3)
                .width(Length::Fill),
            container(control).width(320).align_x(iced::alignment::Horizontal::Right),
        ]
        .spacing(20)
        .align_y(Alignment::Center),
    );
    card(t, "Server and API", body.into())
}

/// One of the two boxes of the limits (`inpGlobalRpm` and `inpGlobalTpm`).
fn number_box<'a>(t: Tokens, value: &'a str, typed: fn(String) -> Message, hint: &'a str) -> Element<'a, Message> {
    text_input(hint, value)
        .on_input(typed)
        .on_submit(Message::RateApply)
        .size(13)
        .padding([6, 8])
        .width(110)
        .font(Font::MONOSPACE)
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

/// The description of the first card (`shell.html`).
pub const SERVER_HELP: &str = "Other apps can use your model through this address. It speaks the OpenAI, Anthropic and Ollama APIs.";

/// The help of the limits row (`shell.html`).
pub const RATE_HELP: &str = "The most requests and tokens per minute the server accepts. Empty means no limit.";

/// The description of the keys card (`shell.html`).
pub const KEYS_HELP: &str = "fLLM makes the key for you; you never type one. Make one key per app, copy it into that app, and revoke it when you stop using it. As soon as one key exists, every app must send a key.";

/// The second card of `spane-local-api-server`: the form that makes a key, the secret that was
/// made (shown once) and the table of the keys the server knows (`refreshApiKeys` of `panels.js`).
pub fn keys_card<'a>(t: Tokens, v: ServerView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(12);
    body = body.push(text(KEYS_HELP).size(13).color(t.fg_soft));

    // the row above the table: name, the two limits, admin rights, and the button
    let form = row![
        text_input("Name, for example Cursor or Claude Code", v.label)
            .on_input(Message::KeyLabel)
            .size(13)
            .padding([6, 10])
            .width(Length::Fill)
            .style(input_style(t)),
        number_box(t, v.key_rpm, Message::KeyRpm, "Requests/min"),
        number_box(t, v.key_tpm, Message::KeyTpm, "Tokens/min"),
        crate::settings::segmented(
            t,
            vec![("Admin rights", v.admin, Message::KeyAdmin(true)), ("Off", !v.admin, Message::KeyAdmin(false))],
        ),
        button(text("Create key").size(13))
            .on_press(Message::KeyCreate)
            .padding([6, 14])
            .style(theme::primary_button(t)),
    ]
    .spacing(6)
    .wrap();
    body = body.push(form);

    // the secret of a key that was just made (`newKeyBanner`, shown once)
    if let Some(secret) = v.new_key {
        body = body.push(
            container(
                row![
                    column![
                        text("Your new key. Copy it now; it is shown only once and cannot be looked up later:")
                            .size(11)
                            .font(bold())
                            .color(t.primary),
                        text(secret.to_string()).size(13).font(Font::MONOSPACE).color(t.fg),
                    ]
                    .spacing(4)
                    .width(Length::Fill),
                    button(text("Copy key").size(12))
                        .on_press(Message::CopyText(secret.to_string()))
                        .padding([5, 10])
                        .style(crate::app::starter_style(t)),
                ]
                .spacing(10)
                .align_y(Alignment::Center),
            )
            .padding([10, 12])
            .style(move |_| container::Style {
                background: Some(iced::Background::Color(t.primary_soft)),
                border: iced::Border { radius: 8.0.into(), width: 1.0, color: t.primary },
                ..Default::default()
            }),
        );
    }

    // the table, or the sentence for what is missing
    if v.no_model {
        body = body.push(
            text("Keys are made by the running server. Load a model first (Models page), then create a key here.")
                .size(13)
                .color(t.muted),
        );
    } else if !v.keys_loaded {
        // the answer of `GET /api/studio/keys` has not come: nothing of the table is drawn
    } else if v.keys.is_empty() {
        body = body.push(
            text("No API key is set up. The server is open, and it accepts changes only from this computer. Create a key above to require a Bearer token for requests.")
                .size(13)
                .color(t.muted),
        );
    } else {
        body = body.push(key_table(t, v.keys));
    }
    card(t, "API keys", body.into())
}

/// The widths of the eight columns of the table (`apiKeysTableContainer` of `panels.js`).
const KEY_COLS: [(&str, f32); 8] = [
    ("ID", 110.0),
    ("Name", 170.0),
    ("Key", 150.0),
    ("Limit", 90.0),
    ("Requests", 66.0),
    ("Tokens", 66.0),
    ("Refused", 66.0),
    ("", 70.0),
];

fn key_table<'a>(t: Tokens, keys: &'a [Key]) -> Element<'a, Message> {
    let mut table = column![].spacing(6);
    let head = KEY_COLS.iter().fold(row![].spacing(6), |r, (label, w)| {
        r.push(text(*label).size(12).font(bold()).color(t.muted).width(Length::Fixed(*w)))
    });
    table = table.push(head);
    for k in keys {
        let cells = row![
            text(k.id.clone()).size(12).font(Font::MONOSPACE).color(t.fg_soft).width(Length::Fixed(KEY_COLS[0].1)),
            row![
                text(k.label.clone()).size(13).font(bold()),
                if k.admin { text("admin").size(11).color(t.primary) } else { text("").size(11) },
            ]
            .spacing(4)
            .width(Length::Fixed(KEY_COLS[1].1)),
            text(k.masked.clone()).size(12).font(Font::MONOSPACE).color(t.fg_soft).width(Length::Fixed(KEY_COLS[2].1)),
            text(limit_line(k.rpm_limit)).size(12).font(Font::MONOSPACE).color(t.fg_soft).width(Length::Fixed(KEY_COLS[3].1)),
            text(k.requests_accepted.to_string()).size(12).font(Font::MONOSPACE).color(t.fg_soft).width(Length::Fixed(KEY_COLS[4].1)),
            text(k.tokens_used.to_string()).size(12).font(Font::MONOSPACE).color(t.fg_soft).width(Length::Fixed(KEY_COLS[5].1)),
            text(k.rejected.to_string()).size(12).font(Font::MONOSPACE).color(t.fg_soft).width(Length::Fixed(KEY_COLS[6].1)),
            button(text("Revoke").size(12))
                .on_press(Message::KeyRevoke(k.id.clone()))
                .padding([3, 8])
                .style(crate::app::starter_style(t))
                .width(Length::Fixed(KEY_COLS[7].1)),
        ]
        .spacing(6)
        .align_y(Alignment::Center);
        // the row with the hairline under it (the `border-bottom` of the table of the Studio)
        table = table
            .push(container(cells).width(Length::Fill).padding([4, 0]))
            .push(crate::models::rule(t));
    }
    table.into()
}

/// The boxes of the form (`ui-input` of the Studio).
fn input_style<'a>(t: Tokens) -> impl Fn(&iced::Theme, text_input::Status) -> text_input::Style + 'a {
    move |_, status| text_input::Style {
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
    }
}

/// The third card of `spane-local-api-server`: one request to any of the eight routes and the
/// answer of the server beside it (`Try the API` of `shell.html`).
pub fn tester_card<'a>(t: Tokens, v: ServerView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(10);
    body = body.push(
        row![
            pick_list(route_labels(), Some(route_label(v.route)), |label| Message::RouteSelect(route_value(label)))
                .padding([6, 10])
                .style(select_style(t)),
            button(text("Send Request").size(13))
                .on_press(Message::RouteSend)
                .padding([6, 14])
                .style(theme::primary_button(t)),
        ]
        .spacing(8),
    );
    let left = text_editor(v.request)
        .on_action(Message::RouteReq)
        .padding(10)
        .size(12)
        .height(Length::Fixed(175.0))
        .style(editor_style(t));
    let right = scrollable(
        container(text(v.answer.to_string()).size(11).font(Font::MONOSPACE).color(t.fg_soft))
            .width(Length::Fill)
            .padding(10)
            .style(theme::inset(t)),
    )
    .height(Length::Fixed(175.0));
    body = body.push(row![left, right].spacing(10));
    card(t, "Try the API", body.into())
}

/// The last card of `spane-local-api-server`: the statistics of the server behind their own line
/// (`<details id="detailsPrometheusLog">` of `shell.html` — the line is its summary, so the card
/// is drawn with the line as its header).
pub fn metrics_card<'a>(t: Tokens, v: ServerView<'a>) -> Element<'a, Message> {
    let head = row![
        button(
            row![
                svg(crate::app::icon(if v.metrics_open {
                    crate::app::ICON_CHEVRON
                } else {
                    crate::app::ICON_CHEVRON_RIGHT
                }))
                .width(14)
                .height(14)
                .style(crate::app::muted_svg(t)),
                text("Server statistics (/metrics)").size(13).font(bold()).color(t.fg_soft),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        )
        .on_press(Message::MetricsOpen)
        .padding([3, 4])
        .style(theme::ghost_button(t)),
        Space::new().width(Length::Fill),
        button(text("Refresh").size(12))
            .on_press(Message::MetricsRefresh)
            .padding([5, 10])
            .style(crate::app::starter_style(t)),
    ]
    .align_y(Alignment::Center);
    let mut col = column![head].spacing(10);
    if v.metrics_open {
        col = col.push(
            scrollable(
                container(text(v.metrics.to_string()).size(12).font(Font::MONOSPACE).color(t.fg_soft))
                    .width(Length::Fill)
                    .padding(10)
                    .style(theme::inset(t)),
            )
            .height(Length::Fixed(220.0)),
        );
    }
    container(col)
        .padding(16)
        .width(Length::Fill)
        .style(theme::card(t, theme::R_LG, false))
        .into()
}

/// The select of the routes (`ui-select` of the Studio).
fn select_style<'a>(t: Tokens) -> impl Fn(&iced::Theme, pick_list::Status) -> pick_list::Style + 'a {
    move |_, status| pick_list::Style {
        background: iced::Background::Color(t.surface_2),
        border: iced::Border {
            radius: 8.0.into(),
            width: 1.0,
            color: if matches!(status, pick_list::Status::Hovered) { t.fg_soft } else { t.separator },
        },
        text_color: t.fg,
        placeholder_color: t.muted,
        handle_color: t.muted,
    }
}

/// The request box (`ui-textarea` of the Studio).
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

/// What the settings page needs for the two cards of `spane-tokenizer`.
#[derive(Clone, Copy)]
pub struct TokenizerView<'a> {
    /// What is typed into the first box (`tokTextInput`, prefilled the way the page prefills it).
    pub text: &'a text_editor::Content,
    /// The line over the wall (`tokStatsLabel`), and the tokens behind it.
    pub stats: &'a str,
    pub tokens: &'a [Token],
    pub special: bool,
    /// The messages of the second box (`jinjaMessagesInput`) and what *Show the text* answered.
    pub messages: &'a text_editor::Content,
    pub output: &'a str,
}

/// The sentence the first box starts with (`shell.html`).
pub const TOKEN_TEXT: &str = "Hello! How does the fLLM Rust engine tokenize this sentence?";

/// The messages the second box starts with (`shell.html`).
pub const JINJA_MESSAGES: &str = "[\n  {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},\n  {\"role\": \"user\", \"content\": \"Explain SIMD in one sentence.\"}\n]";

/// The first card of `spane-tokenizer`: how the model splits text into tokens, and the wall of
/// the pieces with their ids (`runTokenize` of `panels.js`).
pub fn token_card<'a>(t: Tokens, v: TokenizerView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(10);
    body = body.push(
        text_editor(v.text)
            .on_action(Message::TokReq)
            .padding(10)
            .size(13)
            .height(Length::Fixed(74.0))
            .style(editor_style(t)),
    );
    body = body.push(
        row![
            row![
                crate::settings::segmented(
                    t,
                    vec![
                        ("On", v.special, Message::TokSpecial(true)),
                        ("Off", !v.special, Message::TokSpecial(false)),
                    ],
                ),
                text("Include the special start token").size(12).color(t.fg_soft),
            ]
            .spacing(8)
            .align_y(Alignment::Center)
            .width(Length::Fill),
            text(v.stats.to_string()).size(12).font(Font::MONOSPACE).color(t.muted),
        ]
        .align_y(Alignment::Center),
    );

    // the wall of the pieces (`.token-wall`): wrapped pills of the hues the Studio cycles through
    let mut wall = row![].spacing(4);
    for (i, tok) in v.tokens.iter().enumerate() {
        wall = wall.push(token_pill(t, tok, &TOKEN_HUES[i % TOKEN_HUES.len()]));
    }
    if v.tokens.is_empty() {
        // an empty wall keeps the height the Studio gives it (`.token-wall { min-height: 80px }`)
        body = body.push(container(Space::new()).width(Length::Fill).height(Length::Fixed(80.0)));
    } else {
        // the wall grows with its pills; the Studio clamps it at 80px from below, which this
        // renderer has no length for — a wall with one row is its own height plus its padding
        body = body.push(
            container(wall.wrap())
                .width(Length::Fill)
                .padding(12)
                .style(move |_| container::Style {
                    background: Some(iced::Background::Color(t.bg)),
                    border: iced::Border { radius: 8.0.into(), width: 1.0, color: t.separator },
                    ..Default::default()
                }),
        );
    }
    card(t, "See how the model splits text into tokens", body.into())
}

/// One piece with its id (`.tok-pill` of `style.css`).
fn token_pill<'a>(t: Tokens, tok: &Token, hue: &u16) -> Element<'a, Message> {
    let colour = token_colour(*hue);
    let piece = tok.piece();
    container(
        column![
            text(piece).size(12).font(Font::MONOSPACE).color(t.fg),
            text(format!("#{}", tok.id())).size(9).color(iced::Color { a: 0.65, ..t.fg }),
        ]
        .spacing(0),
    )
    .padding([2, 6])
    .style(move |_| container::Style {
        background: Some(iced::Background::Color(colour)),
        border: iced::Border { radius: 5.0.into(), width: 1.0, color: iced::Color { a: 0.08, ..iced::Color::WHITE } },
        ..Default::default()
    })
    .into()
}

/// The second card of `spane-tokenizer`: the messages of a chat and the text the model receives
/// for them (`runJinjaTemplate` of `panels.js`).
pub fn jinja_card<'a>(t: Tokens, v: TokenizerView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(10);
    body = body.push(
        row![
            text_editor(v.messages)
                .on_action(Message::JinjaReq)
                .padding(10)
                .size(12)
                .font(Font::MONOSPACE)
                .height(Length::Fixed(150.0))
                .style(editor_style(t)),
            scrollable(
                container(text(v.output.to_string()).size(12).font(Font::MONOSPACE).color(t.fg_soft))
                    .width(Length::Fill)
                    .padding(10)
                    .style(theme::inset(t)),
            )
            .height(Length::Fixed(150.0)),
        ]
        .spacing(10),
    );
    body = body.push(
        row![
            Space::new().width(Length::Fill),
            button(text("Show the text").size(13))
                .on_press(Message::JinjaRender)
                .padding([6, 14])
                .style(theme::primary_button(t)),
        ],
    );
    card(t, "See the text the model receives for a chat", body.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The text of the box (`refreshLogs` of `models.js`): the lines joined, and the sentence
    /// for a log that has nothing in it yet.
    #[test]
    fn the_lines_are_joined_and_an_empty_log_says_so() {
        assert_eq!(logs_text(&[]), "Nothing has been logged yet.");
        assert_eq!(logs_text(&["one".into()]), "one");
        assert_eq!(logs_text(&["one".into(), "two".into()]), "one\ntwo");
    }

    /// A limit is written into a request the way JavaScript writes a number, and an empty box
    /// reaches the server as no limit at all.
    #[test]
    fn a_limit_is_written_the_way_javascript_writes_a_number() {
        assert_eq!(js_number("60"), Some(serde_json::json!(60)));
        assert_eq!(js_number(" 30 "), Some(serde_json::json!(30)));
        assert_eq!(js_number("1e2"), Some(serde_json::json!(100)), "1e2 is 100, without a dot");
        assert_eq!(js_number("0.5"), Some(serde_json::json!(0.5)));
        assert_eq!(js_number(""), None, "an empty box means no limit");
        assert_eq!(js_number("abc"), None, "a text sends nothing");
    }

    /// The limit of a row of the table (`panels.js` writes `{n} per minute` or *Default*).
    #[test]
    fn a_key_without_a_limit_of_its_own_says_default() {
        assert_eq!(limit_line(None), "Default");
        assert_eq!(limit_line(Some(60)), "60 per minute");
    }

    /// The label of a route and the value behind it go together, and the select offers all eight.
    #[test]
    fn the_select_offers_the_eight_routes_of_the_studio() {
        let labels = route_labels();
        assert_eq!(labels.len(), ROUTES.len());
        assert_eq!(labels[0], "POST /v1/chat/completions (OpenAI Chat)");
        assert_eq!(labels[7], "GET:/props (llama.cpp Server Properties)");
        for (value, label) in ROUTES {
            assert_eq!(route_label(value), label, "{value}");
            assert_eq!(route_value(label), value, "{label}");
        }
        // something the select does not know falls back on the first row, as the map of
        // `updateRouteTesterSample` falls back on an empty body
        assert_eq!(route_value("no such route"), ROUTES[0].0);
    }

    /// The sample of a route (`updateRouteTesterSample`): the body of the Studio, and nothing for
    /// the two routes that take no body.
    #[test]
    fn a_sample_is_the_body_of_the_studio_for_that_route() {
        let rerank = route_sample("POST:/v1/rerank", "llama3");
        assert!(rerank.contains("\"query\": \"SIMD CPU acceleration\""), "{rerank}");
        assert!(rerank.contains("\"model\": \"llama3\""), "{rerank}");
        // without a model the sample says what the Studio says
        assert!(route_sample("POST:/v1/chat/completions", "").contains("\"model\": \"your-model\""));
        let chat = route_sample("POST:/v1/chat/completions", "m");
        assert!(chat.contains("\"max_tokens\": 48"), "{chat}");
        assert!(chat.contains("Hello! Answer briefly."), "{chat}");
        // the two GET routes take no body
        assert_eq!(route_sample("GET:/v1/models", "m"), "");
        assert_eq!(route_sample("GET:/props", "m"), "");
        assert!(route_sample("POST:/v1/embeddings", "m").contains("SIMD Kernels"), "the two inputs of the Studio");
    }

    /// An answer carries an id with its piece, or an id alone (`runTokenize` reads both).
    #[test]
    fn a_token_is_read_with_its_piece_or_as_an_id() {
        let with_piece: Token = serde_json::from_value(serde_json::json!({"id": 7, "piece": " a"}))
            .expect("a token with its piece");
        assert_eq!(with_piece.id(), 7);
        assert_eq!(with_piece.piece(), " a");
        let only: Token = serde_json::from_value(serde_json::json!(9)).expect("an id alone");
        assert_eq!(only.id(), 9);
        // `String(tok)` of the Studio: an id alone is its own text
        assert_eq!(only.piece(), "9");
    }

    /// The line over the wall: how many tokens and how many bytes of text they are.
    #[test]
    fn the_line_counts_the_tokens_and_the_bytes() {
        assert_eq!(token_stats(0, 52), "0 tokens · 52 bytes");
        // `runTokenize` writes `tokens` for one too — the line is not declined anywhere
        assert_eq!(token_stats(1, 1), "1 tokens · 1 bytes");
        assert_eq!(token_stats(12, 4243), "12 tokens · 4243 bytes");
    }

    /// The hue of a pill: `hsla(H, 65%, 45%, 0.18)` for that hue, and every hue of the wall is a
    /// colour of its own.
    #[test]
    fn a_pill_takes_the_hue_of_the_studio() {
        let red = token_colour(0);
        assert!((red.r - 0.7425).abs() < 0.001, "{}", red.r);
        assert!((red.g - 0.1575).abs() < 0.001, "{}", red.g);
        assert!((red.b - 0.1575).abs() < 0.001, "{}", red.b);
        assert!((red.a - 0.18).abs() < 0.001, "the alpha of the pill");
        let all: Vec<[u32; 3]> = TOKEN_HUES
            .iter()
            .map(|h| {
                let c = token_colour(*h);
                [(c.r * 255.0) as u32, (c.g * 255.0) as u32, (c.b * 255.0) as u32]
            })
            .collect();
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert!(a != b, "two hues of the wall give the same colour: {a:?} {b:?}");
            }
        }
    }

    /// An answer of the tester is pretty when it is JSON, and the text as it came when not
    /// (`JSON.stringify(JSON.parse(txt), null, 2)` of `btnRunRouteTest`).
    #[test]
    fn an_answer_is_pretty_when_it_is_json_and_raw_when_not() {
        // the keys come out in the order of Rust's JSON parser, not in the order JavaScript keeps
        // (`JSON.parse` of the Studio would answer `b` before `a` here)
        assert_eq!(pretty_answer("{\"b\":1,\"a\":2}"), "{\n  \"a\": 2,\n  \"b\": 1\n}");
        assert_eq!(pretty_answer("not json at all"), "not json at all");
        assert_eq!(pretty_answer(""), "");
        // a plain number is JSON too, and a broken object falls back on the text
        assert_eq!(pretty_answer("42"), "42");
        assert_eq!(pretty_answer("{broken"), "{broken");
    }
}
