//! Settings > Logs, Server and API, API keys and the tokenizer: what the server does for other
//! programs, the keys it hands out, its latest messages and the way it splits text into tokens
//! (`spane-logs`, `spane-local-api-server` and `spane-tokenizer` of `shell.html`).
//!
//! This file grows with the row: the Logs card is the first of the four.

use crate::app::Message;
use crate::models::card;
use crate::theme::{self, Tokens};
use iced::widget::{button, column, container, row, scrollable, text};
use iced::{Element, Font, Length};

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
}
