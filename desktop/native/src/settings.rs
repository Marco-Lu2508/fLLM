//! The Settings page. Everything here is automatic until you switch it to manual: a manual value is sent with every request, an automatic one
//! is left to the server and the model.
//!
//! This file draws the settings of the answers (sampling, system prompt, longest answer), which the window keeps itself.

use crate::app::Message;
use crate::models::{bold, card, rule};
use crate::store::{Prefs, BUILT_IN_SYSTEM_PROMPT};
use crate::theme::{self, Tokens};
use iced::widget::text_editor::{Content, Status};
use iced::widget::{button, column, container, row, scrollable, slider, text, text_editor, text_input, Space};
use iced::{Alignment, Element, Length, Padding};

/// Two or three choices side by side; the selected one is filled.
pub fn segmented<'a>(t: Tokens, options: Vec<(&'a str, bool, Message)>) -> Element<'a, Message> {
    let mut r = row![].spacing(2);
    for (label, selected, message) in options {
        r = r.push(
            button(text(label).size(13).width(Length::Fill).align_x(Alignment::Center))
                .on_press(message)
                .width((label.chars().count() as f32 * 7.6 + 28.0).max(80.0))
                .padding([5, 10])
                .style(move |_, status| {
                    let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                    button::Style {
                        background: Some(iced::Background::Color(if selected { t.primary_soft } else if hovered { t.fill_hover } else { iced::Color::TRANSPARENT })),
                        text_color: if selected { t.primary } else { t.fg_soft },
                        border: iced::border::rounded(8),
                        ..Default::default()
                    }
                }),
        );
    }
    container(r).padding(2).style(move |_| container::Style { background: Some(iced::Background::Color(t.fill)), border: iced::border::rounded(10), ..Default::default() }).into()
}

/// A setting with a title and a sentence of help on the left and its control on the right.
pub fn setting<'a>(t: Tokens, title: &'a str, help: &'a str, control: Element<'a, Message>) -> Element<'a, Message> {
    row![column![text(title).size(14), text(help).size(12).color(t.muted)].spacing(3).width(Length::Fill), container(control).width(280).align_x(iced::alignment::Horizontal::Right)]
        .spacing(20)
        .align_y(Alignment::Center)
        .into()
}

fn value_slider<'a>(t: Tokens, range: std::ops::RangeInclusive<f32>, value: f32, step: f32, on_change: fn(f32) -> Message) -> Element<'a, Message> {
    let s = slider(range, value, on_change).step(step).on_release(Message::SavePrefs).width(190).style(move |_, _| slider::Style {
        rail: slider::Rail { backgrounds: (iced::Background::Color(t.primary), iced::Background::Color(t.fill_hover)), width: 4.0, border: iced::border::rounded(2) },
        handle: slider::Handle { shape: slider::HandleShape::Circle { radius: 9.0 }, background: iced::Background::Color(iced::Color::WHITE), border_width: 1.0, border_color: iced::Color { a: 0.35, ..iced::Color::BLACK } },
    });
    row![s, text(format!("{value:.2}")).size(13).font(iced::Font::MONOSPACE).width(44)].spacing(10).align_y(Alignment::Center).into()
}

pub fn page<'a>(t: Tokens, prefs: &'a Prefs, system_editor: &'a Content, max_tokens_text: &'a str, engine: Element<'a, Message>) -> Element<'a, Message> {
    let mut answers = column![].spacing(14);
    answers = answers.push(setting(
        t,
        "Sampling",
        "Automatic uses the values the model ships with. Manual sends your values with every message.",
        segmented(t, vec![("Automatic", !prefs.manual_sampling, Message::SamplingManual(false)), ("Manual", prefs.manual_sampling, Message::SamplingManual(true))]),
    ));
    if prefs.manual_sampling {
        answers = answers
            .push(setting(t, "Temperature", "Low: steady and predictable. High: varied and surprising.", value_slider(t, 0.0..=2.0, prefs.temperature, 0.05, Message::Temperature)))
            .push(setting(t, "Top-p", "Only the most likely words that together make up this share are considered.", value_slider(t, 0.05..=1.0, prefs.top_p, 0.05, Message::TopP)));
    }
    answers = answers.push(rule(t));
    answers = answers.push(setting(
        t,
        "System prompt",
        "What the model is told before every chat. Automatic is a short built-in prompt.",
        segmented(t, vec![("Automatic", !prefs.custom_system, Message::SystemCustom(false)), ("Custom", prefs.custom_system, Message::SystemCustom(true))]),
    ));
    if prefs.custom_system {
        let editor = text_editor(system_editor)
            .placeholder(BUILT_IN_SYSTEM_PROMPT)
            .on_action(Message::SystemEdit)
            .padding(12)
            .size(14)
            .height(Length::Shrink)
            .min_height(90)
            .max_height(240)
            .style(move |_, status| text_editor::Style {
                background: iced::Background::Color(t.surface_2),
                border: iced::Border { radius: 10.0.into(), width: 1.0, color: if matches!(status, Status::Focused { .. }) { t.fg_soft } else { t.separator } },
                placeholder: t.muted,
                value: t.fg,
                selection: iced::Color { a: 0.35, ..t.primary },
            });
        answers = answers.push(editor);
        answers = answers.push(text("An empty text falls back to the automatic prompt.").size(12).color(t.muted));
    }
    answers = answers.push(rule(t));
    let limit = text_input("automatic", max_tokens_text)
        .on_input(Message::MaxTokensText)
        .on_submit(Message::SavePrefs)
        .size(14)
        .padding([6, 10])
        .width(110)
        .style(move |_, status| text_input::Style {
            background: iced::Background::Color(t.surface_2),
            border: iced::Border { radius: 8.0.into(), width: 1.0, color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator } },
            icon: t.muted,
            placeholder: t.muted,
            value: t.fg,
            selection: t.primary_soft,
        });
    answers = answers.push(setting(t, "Longest answer", "In tokens. Empty lets the server decide.", row![limit, text("tokens").size(12).color(t.muted).width(60)].spacing(8).align_y(Alignment::Center).into()));
    answers = answers.push(
        row![
            Space::new().width(Length::Fill),
            button(text("Reset all answer settings").size(13)).on_press(Message::ResetPrefs).padding([5, 14]).style(theme::secondary_button(t))
        ],
    );

    let content = column![
        container(text("Settings").size(26).font(bold())).padding(Padding { top: 14.0, right: 0.0, bottom: 4.0, left: 2.0 }),
        card(t, "Answers", answers.into()),
        engine,
    ]
    .spacing(14);
    let inner = container(content).max_width(780).width(Length::Fill).padding(Padding { top: 8.0, right: 24.0, bottom: 32.0, left: 24.0 });
    scrollable(container(inner).center_x(Length::Fill).width(Length::Fill)).height(Length::Fill).into()
}
