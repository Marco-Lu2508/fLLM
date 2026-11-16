//! Settings > General of the Studio (`spane-general` of shell.html): the version, the models,
//! the welcome tour, the keyboard shortcuts, *Delete all chats*, *Name new chats automatically*,
//! and the card *Start it from a terminal* with the command and the `fllm.toml` of the model that
//! runs now (`updateCliAndSdkSnippets` of panels.js).

use crate::app::Message;
use crate::settings::setting;
use crate::theme::{self, Tokens};
use iced::widget::{button, checkbox, column, container, row, text, Space};
use iced::{Alignment, Element, Font, Length};

/// What the card shows and reads.
pub struct GeneralView<'a> {
    /// `build_info` of the server's status, empty until it answered.
    pub build_info: &'a str,
    pub auto_title: bool,
    pub rich_answers: bool,
    /// The model that runs: its name, context, parallel sequences and aliases (`state.status.model`).
    pub model: ModelFacts,
}

/// The facts of the running model the two snippets are built from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelFacts {
    pub name: String,
    pub max_context: u64,
    pub max_sequences: u64,
    pub aliases: Vec<String>,
}

/// `genCliPre` and `genTomlPre`: the command and the settings file of the model that runs now.
pub fn snippets(m: &ModelFacts) -> (String, String) {
    let model = if m.name.is_empty() { "fllm-studio-demo" } else { m.name.as_str() };
    let ctx = if m.max_context == 0 { 512 } else { m.max_context };
    let seqs = if m.max_sequences == 0 { 8 } else { m.max_sequences };
    let mut names = vec![model.to_string()];
    names.extend(m.aliases.iter().cloned());
    let cli = [
        format!("fllm-rs serve --model \"{model}.gguf\""),
        "  --host 127.0.0.1 --port 8000".to_string(),
        format!("  --ctx {ctx} --max-seqs {seqs}"),
        format!("  --model-name \"{}\"", names.join(",")),
    ]
    .join(" \\\n");
    let toml = format!("[server]\nmodel = \"{model}.gguf\"\nhost = \"127.0.0.1\"\nport = 8000\nctx = {ctx}\nmax_seqs = {seqs}\n");
    (cli, toml)
}

fn action<'a>(t: Tokens, label: &'a str, msg: Message, danger: bool) -> Element<'a, Message> {
    let b = button(text(label).size(13)).on_press(msg).padding([5, 12]);
    if danger {
        b.style(move |_, status| {
            let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
            button::Style { background: Some(iced::Background::Color(if hovered { t.fill_hover } else { t.fill })), text_color: t.danger, border: iced::border::rounded(8), ..Default::default() }
        })
        .into()
    } else {
        b.style(theme::secondary_button(t)).into()
    }
}

fn code_card<'a>(t: Tokens, language: &'a str, body: String, toast: &'static str) -> Element<'a, Message> {
    let copy = Message::CopyWithToast(body.clone(), toast);
    column![
        row![text(language).size(12).color(t.muted), Space::new().width(Length::Fill), button(text("Copy").size(12)).on_press(copy).padding([2, 8]).style(theme::ghost_button(t))].align_y(Alignment::Center),
        container(text(body).size(12).font(Font::MONOSPACE)).padding(10).width(Length::Fill).style(move |_| container::Style {
            background: Some(iced::Background::Color(t.surface_2)),
            border: iced::Border { width: 1.0, color: t.separator, radius: 8.0.into() },
            ..Default::default()
        }),
    ]
    .spacing(6)
    .into()
}

/// The two cards: *General* and *Start it from a terminal*.
pub fn view<'a>(t: Tokens, v: &GeneralView<'a>) -> Vec<(&'static str, Element<'a, Message>)> {
    let version = if v.build_info.is_empty() { concat!("fllm-app ", env!("CARGO_PKG_VERSION")).to_string() } else { v.build_info.to_string() };
    let general = column![
        setting(t, "Version", "", row![text(version).size(12).font(Font::MONOSPACE), action(t, "Refresh", Message::GeneralRefresh, false)].spacing(10).align_y(Alignment::Center).into()),
        setting(t, "Models", "Download, load and remove models.", action(t, "Open Models", Message::Go(crate::app::View::Models), false)),
        setting(t, "Welcome tour", "A short introduction to fLLM, your computer and the first steps.", action(t, "Show the welcome tour", Message::TourShow, false)),
        setting(t, "Keyboard shortcuts", "See and change the keys.", action(t, "Keyboard shortcuts", Message::ShortcutsOpen, false)),
        setting(t, "Delete all chats", "Deletes all saved chats on this computer.", action(t, "Delete chats", Message::ChatsDeleteAsk, true)),
        setting(
            t,
            "Name new chats automatically",
            "After the first answer the model gives a new chat a short title.",
            checkbox(v.auto_title).size(18).on_toggle(Message::AutoTitle).into()
        ),
        setting(
            t,
            "Charts, diagrams and math in answers",
            "The model is told it can write math, charts, Mermaid diagrams and SVG pictures, and the app draws them. Turn it off to save a few lines of every request.",
            checkbox(v.rich_answers).size(18).on_toggle(Message::RichAnswers).into()
        ),
    ]
    .spacing(14);
    let (cli, toml) = snippets(&v.model);
    let terminal = column![
        column![text("Command").size(14), text("Starts the server the way it runs now, without this app.").size(12).color(t.muted), code_card(t, "bash", cli, "Command copied")].spacing(6),
        column![text("Settings file (fllm.toml)").size(14), code_card(t, "toml", toml, "fllm.toml copied")].spacing(6),
    ]
    .spacing(14);
    vec![("General", general.into()), ("Start it from a terminal", terminal.into())]
}

/// The dialog that asks before every chat goes (`askDialog` of `btnDeleteAllThreads`).
pub fn delete_all_window<'a>(t: Tokens, chats: usize) -> Element<'a, Message> {
    let window = container(
        column![
            text("Delete all chats?").size(16).font(Font { weight: iced::font::Weight::Bold, ..Font::default() }),
            text(format!("{chats} chats will be deleted for good. Export them first if you want to keep them.")).size(13).color(t.fg_soft),
            row![Space::new().width(Length::Fill), button(text("Cancel").size(13)).on_press(Message::ChatsDeleteCancel).padding([6, 12]).style(theme::ghost_button(t)), action(t, "Delete all", Message::ChatsDeleteOk, true)].spacing(8),
        ]
        .spacing(10),
    )
    .padding(16)
    .width(400)
    .style(theme::card(t, theme::R_LG, false));
    container(iced::widget::opaque(window)).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into()
}
