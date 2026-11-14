//! The **Plugins** card: one folder that brings skills, sub-agents, commands and hooks at once.
//!
//! Port of `pluginsSettings` in `rust/engine/src/webui/plugins.js` (the whole file) and of
//! `GET|POST|DELETE /api/studio/plugins` — field for field, message for message.

use iced::{
    Font, Length,
    alignment::Alignment,
    widget::{button, column, container, row, text, text_input, Space},
    Element,
};

use crate::app::Message;
use crate::models::card;
use crate::theme::{self, Tokens};

/// The one card of the pane, as the page writes it (`app-card-title` + its description). The file
/// names inside the description are in `mono` on the page; the window draws the whole line as its
/// card text, the way the other descriptions here are drawn.
pub const DESC: &str = "A plugin is a folder that brings several things at once: skills (skills/<name>/SKILL.md), sub-agents (agents/<name>.md), slash commands (commands/<name>.md) and hooks (hooks.json), described by plugin.json. Nothing is copied into your own files: the contributions stay in the plugin folder, so removing the plugin removes everything it added.";

/// One installed plugin (`p` of `refresh` in the page). A plugin that does not name one of the
/// fields leaves it out of the answer, which the page reads as nothing — `#[serde(default)]`
/// keeps that the same here.
#[derive(Debug, Clone, PartialEq, Default, serde::Deserialize)]
pub struct Plugin {
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub skills: usize,
    #[serde(default)]
    pub agents: usize,
    #[serde(default)]
    pub commands: usize,
    #[serde(default)]
    pub hooks: usize,
    #[serde(default)]
    pub author: String,
    pub path: String,
}

/// The answer of `GET /api/studio/plugins`.
#[derive(Debug, Clone, PartialEq, Default, serde::Deserialize)]
pub struct PluginsAnswer {
    pub folder: String,
    pub plugins: Vec<Plugin>,
}

/// The line under a plugin: what it contributes (`parts.join(' · ')` in the page), or the words
/// for nothing, then the author when there is one.
///
/// A count is a part only when it is set — the page keeps `p.skills` when it is not `0`, so a
/// plugin with none of them says *nothing contributed yet*.
pub fn parts_line(p: &Plugin) -> String {
    let mut parts: Vec<String> = Vec::new();
    if p.skills != 0 {
        parts.push(format!("{} skill(s)", p.skills));
    }
    if p.agents != 0 {
        parts.push(format!("{} sub-agent(s)", p.agents));
    }
    if p.commands != 0 {
        parts.push(format!("{} command(s)", p.commands));
    }
    if p.hooks != 0 {
        parts.push(format!("{} hook(s)", p.hooks));
    }
    let mut line = if parts.is_empty() {
        "nothing contributed yet".to_string()
    } else {
        parts.join(" · ")
    };
    if !p.author.is_empty() {
        line.push_str(" · by ");
        line.push_str(&p.author);
    }
    line
}

/// The folder line: `Plugins live in {folder}`, empty when the answer has no folder.
pub fn folder_line(folder: &str) -> String {
    if folder.is_empty() {
        String::new()
    } else {
        format!("Plugins live in {folder}")
    }
}

/// The references the card draws.
#[derive(Default)]
pub struct PluginsView<'a> {
    pub plugins: &'a [Plugin],
    pub folder: &'a str,
    pub status: &'a str,
    pub path: &'a str,
    pub url: &'a str,
    pub ref_: &'a str,
}

/// One plugin of the list: its head with the Remove button, its description, what it contributes
/// and where it lies.
fn plugin_row<'a>(t: Tokens, p: &'a Plugin) -> Element<'a, Message> {
    let name = p.name.clone();
    let head = if p.version.is_empty() {
        row![text(p.name.clone()).size(14).color(t.fg)]
    } else {
        row![
            text(p.name.clone()).size(14).color(t.fg),
            text(format!("v{}", p.version)).size(13).color(t.fg_soft),
        ]
    }
    .spacing(8)
    .align_y(Alignment::Center);

    let mut item = column![row![
        head,
        Space::new().width(Length::Fill),
        button(text("Remove").size(12))
            .on_press(Message::PluginRemoveAsk(name))
            .padding([2, 8])
            .style(theme::ghost_button(t)),
    ]
    .spacing(8)
    .align_y(Alignment::Center)]
    .spacing(3);
    if !p.description.is_empty() {
        item = item.push(text(p.description.clone()).size(13).color(t.fg_soft));
    }
    item = item.push(text(parts_line(p)).size(13).color(t.fg_soft));
    item = item.push(
        text(p.path.clone())
            .size(11)
            .font(Font::MONOSPACE)
            .color(t.fg_soft),
    );
    column![item, divider(t)].padding([4, 0]).spacing(4).into()
}

/// The card of `spane-plugins`.
pub fn section<'a>(t: Tokens, v: PluginsView<'a>) -> Element<'a, Message> {
    let mut list = column![].spacing(0);
    if v.plugins.is_empty() {
        // `No plugins installed.`
        list = list.push(text("No plugins installed.").size(13).color(t.fg_soft));
    } else {
        for p in v.plugins {
            list = list.push(plugin_row(t, p));
        }
    }

    let folder_line_text = folder_line(v.folder);

    let mut body = column![text(DESC).size(13).color(t.fg_soft)].spacing(10);
    body = body.push(list);
    body = body.push(
        row![
            input(
                t,
                "/home/you/plugins/my-plugin",
                v.path,
                Message::PluginPath
            ),
            button(text("Install from folder").size(13))
                .on_press(Message::PluginInstall)
                .padding([6, 12])
                .style(theme::secondary_button(t)),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    );
    body = body.push(
        row![
            input(t, "https://github.com/you/your-plugin", v.url, Message::PluginUrl),
            input(t, "branch (optional)", v.ref_, Message::PluginRef),
            button(text("Install from repository").size(13))
                .on_press(Message::PluginClone)
                .padding([6, 12])
                .style(theme::secondary_button(t)),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    );
    if !folder_line_text.is_empty() {
        body = body.push(Space::new().height(8));
        body = body.push(
            text(folder_line_text)
                .size(11)
                .font(Font::MONOSPACE)
                .color(t.fg_soft),
        );
    }
    if !v.status.is_empty() {
        body = body.push(Space::new().height(6));
        body = body.push(
            text(v.status.to_string())
                .size(11)
                .font(Font::MONOSPACE)
                .color(t.fg_soft),
        );
    }
    card(t, "Plugins", body.into())
}

/// The `ui-input` of the page.
fn input<'a>(
    t: Tokens,
    placeholder: &'a str,
    value: &'a str,
    typed: fn(String) -> Message,
) -> Element<'a, Message> {
    text_input(placeholder, value)
        .on_input(typed)
        .size(13)
        .padding([6, 10])
        .font(Font::MONOSPACE)
        .style(move |_, status| text_input::Style {
            background: iced::Background::Color(t.surface_2),
            border: iced::Border {
                radius: 8.0.into(),
                width: 1.0,
                color: if matches!(status, text_input::Status::Focused { .. }) {
                    t.fg_soft
                } else {
                    t.separator
                },
            },
            icon: t.muted,
            placeholder: t.muted,
            value: t.fg,
            selection: t.primary_soft,
        })
        .into()
}

/// The line between the rows of a list (the border of `.app-card-item`).
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

#[cfg(test)]
mod tests {
    use super::*;

    fn plugin() -> Plugin {
        Plugin {
            name: "review-tools".to_string(),
            version: "1.2.0".to_string(),
            description: "Checks a change.".to_string(),
            skills: 2,
            agents: 1,
            commands: 0,
            hooks: 0,
            author: "ada".to_string(),
            path: "/home/m/plugins/review-tools".to_string(),
        }
    }

    /// `parts.join(' · ')`, with only the counts the page keeps, and the author behind it.
    #[test]
    fn a_plugin_tells_what_it_brings_and_who_wrote_it() {
        assert_eq!(
            parts_line(&plugin()),
            "2 skill(s) · 1 sub-agent(s) · by ada"
        );
        let mut none = plugin();
        none.skills = 0;
        none.agents = 0;
        none.author = String::new();
        assert_eq!(parts_line(&none), "nothing contributed yet");
    }

    /// `Plugins live in {folder}` — and nothing at all without a folder.
    #[test]
    fn the_folder_line_names_the_folder_only_when_there_is_one() {
        assert_eq!(
            folder_line("/home/m/.fllm/plugins"),
            "Plugins live in /home/m/.fllm/plugins"
        );
        assert_eq!(folder_line(""), "");
    }
}
