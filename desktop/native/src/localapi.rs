//! Settings > Logs, Server and API, API keys and the tokenizer: what the server does for other
//! programs, the keys it hands out, its latest messages and the way it splits text into tokens
//! (`spane-logs`, `spane-local-api-server` and `spane-tokenizer` of `shell.html`).
//!
//! This file grows with the row: the Logs card is the first of the four.

use crate::app::Message;
use crate::models::{bold, card};
use crate::theme::{self, Tokens};
use iced::widget::{button, column, container, row, scrollable, text, text_input, Space};
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
}
