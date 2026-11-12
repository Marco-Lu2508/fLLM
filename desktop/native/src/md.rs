//! The Markdown of an answer, drawn with the tokens of the design system: code blocks and quotes as calm inset surfaces, inline code as a
//! small quiet chip (iced's default is white text on a black slab).

use crate::app::Message;
use crate::theme::{self, Tokens};
use iced::widget::{column, container, markdown, row, rich_text, scrollable, text, Space};
use iced::{font, Element, Font, Length, Padding, Pixels};

pub struct Viewer {
    pub t: Tokens,
}

/// The settings for an answer: sizes in proportion (the default makes a heading twice the body), spacing between the blocks.
pub fn settings(t: Tokens, theme: &iced::Theme) -> markdown::Settings {
    let mut style = markdown::Style::from(theme);
    style.font = crate::app::UI_FONT;
    style.inline_code_highlight = markdown::Highlight {
        background: iced::Background::Color(t.fill_hover),
        border: iced::Border { color: t.separator, width: 0.5, radius: 5.0.into() },
    };
    style.inline_code_padding = Padding { top: 1.0, right: 4.0, bottom: 1.0, left: 4.0 };
    style.inline_code_color = t.fg;
    style.link_color = t.primary;
    let base = 15.0;
    let mut s = markdown::Settings::with_text_size(base, style);
    s.h1_size = Pixels(base * 1.5);
    s.h2_size = Pixels(base * 1.3);
    s.h3_size = Pixels(base * 1.15);
    s.h4_size = Pixels(base * 1.05);
    s.h5_size = Pixels(base);
    s.h6_size = Pixels(base);
    s.code_size = Pixels(13.0);
    s.spacing = Pixels(12.0);
    s
}

pub fn view<'a>(content: &'a markdown::Content, t: Tokens, theme: &iced::Theme) -> Element<'a, Message> {
    markdown::view_with(content.items(), settings(t, theme), &Viewer { t })
}

impl<'a> markdown::Viewer<'a, Message> for Viewer {
    fn on_link_click(url: markdown::Uri) -> Message {
        Message::LinkClicked(url)
    }

    fn code_block(&self, settings: markdown::Settings, language: Option<&'a str>, code: &'a str, lines: &'a [markdown::Text]) -> Element<'a, Message> {
        let t = self.t;
        let label = language.filter(|l| !l.is_empty()).unwrap_or("text");
        let header = row![
            text(label).size(12).color(t.muted).font(Font { weight: font::Weight::Medium, ..crate::app::UI_FONT }),
            Space::new().width(Length::Fill),
            iced::widget::button(text("Copy code").size(12)).on_press(Message::CopyText(code.to_string())).padding([2, 8]).style(theme::ghost_button(t)),
        ]
        .align_y(iced::Alignment::Center)
        .padding(Padding { top: 4.0, right: 6.0, bottom: 0.0, left: 14.0 });
        let body = scrollable(
            container(column(lines.iter().map(|line| {
                rich_text(line.spans(settings.style)).on_link_click(Message::LinkClicked).font(settings.style.code_block_font).size(settings.code_size).into()
            })))
            .padding(Padding { top: 6.0, right: 14.0, bottom: 14.0, left: 14.0 }),
        )
        .direction(scrollable::Direction::Horizontal(scrollable::Scrollbar::default().width(6).scroller_width(6)));
        container(column![header, body]).width(Length::Fill).style(theme::inset(t)).into()
    }

    fn quote(&self, settings: markdown::Settings, contents: &'a [markdown::Item]) -> Element<'a, Message> {
        let t = self.t;
        let bar = container(Space::new().width(3).height(Length::Fill)).style(move |_| iced::widget::container::Style {
            background: Some(iced::Background::Color(iced::Color { a: 0.55, ..t.muted })),
            border: iced::border::rounded(2),
            ..Default::default()
        });
        row![bar, container(markdown::items(self, settings, contents)).padding(Padding { top: 0.0, right: 0.0, bottom: 0.0, left: 4.0 })].spacing(10).height(Length::Shrink).into()
    }
}
