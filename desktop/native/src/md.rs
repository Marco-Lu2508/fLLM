//! The Markdown of an answer, drawn with the tokens of the design system: code blocks and quotes as calm inset surfaces, inline code as a
//! small quiet chip (iced's default is white text on a black slab).

use crate::app::Message;
use crate::theme::{self, Tokens};
use iced::widget::{column, container, markdown, row, rich_text, scrollable, text, Space};
use iced::{font, Element, Font, Length, Padding, Pixels};

pub struct Viewer<'a> {
    pub t: Tokens,
    /// The runs that stand under a code block of this answer, keyed by the code that runs.
    pub runs: Option<&'a std::collections::HashMap<String, crate::artifacts::CodeRun>>,
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
    markdown::view_with(content.items(), settings(t, theme), &Viewer { t, runs: None })
}

/// The same, with the runs of the chat: every code block draws the output of its own run under
/// itself (the `.run-out` of a code card in the Studio).
pub fn view_runs<'a>(
    content: &'a markdown::Content,
    t: Tokens,
    theme: &iced::Theme,
    runs: &'a std::collections::HashMap<String, crate::artifacts::CodeRun>,
) -> Element<'a, Message> {
    markdown::view_with(content.items(), settings(t, theme), &Viewer { t, runs: Some(runs) })
}

/// The Studio typesets the math of an answer: a display paragraph (`$$x$$`, `\[..\]`) gets its
/// own line with the typeset source, an inline piece (`$x$`) is typeset between the words.
/// Everything else is drawn exactly as the paragraph is today.
impl<'a> markdown::Viewer<'a, Message> for Viewer<'a> {
    fn paragraph(
        &self,
        settings: markdown::Settings,
        text: &markdown::Text,
    ) -> Element<'a, Message> {
        let spans = text.spans(settings.style);
        // the character stream with the boundaries of the spans, so a piece of math is found
        // even when it begins in one span and ends in the next
        let mut stream = String::new();
        let mut bounds: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
        for s in spans.iter() {
            let start = stream.chars().count();
            stream.push_str(&s.text);
            bounds.push((start, stream.chars().count()));
        }
        let hits = crate::tex::math_hits(&stream);
        if hits.is_empty() {
            return markdown::paragraph(settings, text, Self::on_link_click);
        }
        let last = stream.chars().count();
        // a piece of the paragraph between two character positions, with the styles it had, and
        // optionally one text in the style the piece began with (a piece of math is typeset,
        // the Studio swaps `$x$` for the typeset source between the words)
        let slice = |from: usize, to: usize, replace_by: Option<&str>| -> Option<Element<'a, Message>> {
            let mut out: Vec<iced::widget::text::Span<'static, markdown::Uri>> = Vec::new();
            for (span, (start, end)) in spans.iter().zip(bounds.iter()) {
                let s = (*start).max(from);
                let e = (*end).min(to);
                if s >= e {
                    continue;
                }
                let cut: String = span.text.chars().skip(s - start).take(e - s).collect();
                if cut.is_empty() {
                    continue;
                }
                let mut piece = span.clone();
                piece.text = cut.into();
                out.push(piece);
            }
            if let Some(replacement) = replace_by {
                if let Some(first) = out.first_mut() {
                    first.text = replacement.to_string().into();
                    out.truncate(1);
                }
            }
            if out.is_empty() {
                return None;
            }
            Some(rich_text(out).on_link_click(Self::on_link_click).into())
        };

        let mut pieces: Vec<Element<'a, Message>> = Vec::new();
        let mut cursor = 0usize;
        for hit in hits {
            if hit.start > cursor {
                if let Some(el) = slice(cursor, hit.start, None) {
                    pieces.push(el);
                }
            }
            let source = match crate::tex::typeset(&hit.body, hit.display) {
                Some(node) => crate::tex::html_plain(&crate::tex::to_html(&node, hit.display)),
                None if hit.display => hit.body.clone(),
                None => format!("${}$", hit.body),
            };
            if hit.display {
                // a display paragraph is a line of its own
                pieces.push(
                    container(iced::widget::text(source))
                        .width(Length::Fill)
                        .padding([6, 8])
                        .into(),
                );
            } else if let Some(el) = slice(hit.start, hit.end, Some(&source)) {
                // an inline piece keeps its place between the words: the typeset source of the
                // Studio, in the style of the paragraph around it
                pieces.push(el);
            }
            cursor = hit.end;
        }
        if cursor < last {
            if let Some(el) = slice(cursor, last, None) {
                pieces.push(el);
            }
        }
        if pieces.is_empty() {
            return markdown::paragraph(settings, text, Self::on_link_click);
        }
        if pieces.len() == 1 {
            return pieces.remove(0);
        }
        column(pieces).spacing(4).into()
    }

    fn on_link_click(url: markdown::Uri) -> Message {
        Message::LinkClicked(url)
    }

    fn code_block(&self, settings: markdown::Settings, language: Option<&'a str>, code: &'a str, lines: &'a [markdown::Text]) -> Element<'a, Message> {
        let t = self.t;
        let info = language.unwrap_or("");
        let lang = info.split_whitespace().next().unwrap_or("");
        let label = if lang.is_empty() { "text" } else { lang };
        let kind = crate::artifacts::artifact_kind(lang);
        let title = crate::artifacts::parse_info(info);
        // the buttons of the Studio: Open/Preview for a block that is one, Run for JavaScript
        let mut actions = row![].spacing(4);
        if kind.is_previewable() || !title.is_empty() {
            let word = if kind.is_previewable() { "Preview" } else { "Open" };
            actions = actions.push(
                iced::widget::button(text(word).size(12))
                    .on_press(Message::ArtOpen(code.to_string(), info.to_string()))
                    .padding([2, 8])
                    .style(theme::ghost_button(t)),
            );
        }
        let run = self.runs.and_then(|runs| runs.get(code));
        if crate::artifacts::is_js_lang(lang) {
            let word = if run.is_some_and(|r| matches!(r.status, crate::artifacts::RunStatus::Running)) { "Stop" } else { "Run" };
            let msg = if word == "Stop" { Message::RunStop(code.to_string()) } else { Message::RunCode(code.to_string()) };
            actions = actions.push(iced::widget::button(text(word).size(12)).on_press(msg).padding([2, 8]).style(theme::ghost_button(t)));
        }
        let header = row![
            text(label).size(12).color(t.muted).font(Font { weight: font::Weight::Medium, ..crate::app::UI_FONT }),
            Space::new().width(Length::Fill),
            actions,
            iced::widget::button(text("Copy code").size(12)).on_press(Message::CopyText(code.to_string())).padding([2, 8]).style(theme::ghost_button(t)),
        ]
        .spacing(6)
        .align_y(iced::Alignment::Center)
        .padding(Padding { top: 4.0, right: 6.0, bottom: 0.0, left: 14.0 });
        let body = scrollable(
            container(column(lines.iter().map(|line| {
                rich_text(line.spans(settings.style)).on_link_click(Message::LinkClicked).font(settings.style.code_block_font).size(settings.code_size).into()
            })))
            .padding(Padding { top: 6.0, right: 14.0, bottom: 14.0, left: 14.0 }),
        )
        .direction(scrollable::Direction::Horizontal(scrollable::Scrollbar::default().width(6).scroller_width(6)));
        let mut col = column![header, body];
        if let Some(run) = run {
            let status = match &run.status {
                crate::artifacts::RunStatus::Running => text("Running...").size(12).color(if run.warn { t.danger } else { t.muted }),
                crate::artifacts::RunStatus::Finished { line, .. } => text(line.clone()).size(12).color(if run.warn { t.danger } else { t.muted }),
                crate::artifacts::RunStatus::Stopped(why) => text(why.clone()).size(12).color(if run.warn { t.danger } else { t.muted }),
            };
            let mut out = column![status].spacing(3);
            if !run.lines.is_empty() {
                let mut body = column![].spacing(1);
                for line in &run.lines {
                    let color = match line.level.as_str() {
                        "error" => t.danger,
                        "warn" | "info" => t.primary,
                        "value" => t.primary,
                        _ => t.fg_soft,
                    };
                    body = body.push(text(line.text.clone()).size(12).color(color).font(Font { weight: font::Weight::Normal, ..crate::app::UI_FONT }));
                }
                out = out.push(body);
            }
            col = col.push(container(out).padding([6, 10]).style(theme::inset(t)));
        }
        container(col).width(Length::Fill).style(theme::inset(t)).into()
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

/// The visible text of an answer: what the plain copy of it carries, without the Markdown marks
/// (the Studio's `plainTextOf` reads the rendered text; this builds the same text from the
/// source — the code stays, because it is visible in the answer, and web addresses stay, while
/// `stripForSpeech` takes both out).
pub fn plain(markdown: &str) -> String {
    // fenced code: only the fences are not visible
    let opener = crate::speech::re(r"^[ \t]*(```|~~~)");
    let mut kept: Vec<&str> = Vec::new();
    let mut fence = String::new();
    for line in markdown.split('\n') {
        let found = opener.captures(line);
        if !fence.is_empty() {
            if let Some(c) = found {
                if c[1] == fence {
                    // the closing fence is not visible, the code above it is
                    fence.clear();
                    continue;
                }
            }
            kept.push(line);
            continue;
        }
        if let Some(c) = found {
            fence = c[1].to_string();
            continue;
        }
        kept.push(line);
    }
    let mut s = kept.join("\n");
    s = crate::speech::re(r"<[^>\n]+>").replace_all(&s, " ").into_owned();
    s = crate::speech::re(r"!\[([^\]]*)\]\([^)]*\)").replace_all(&s, "$1").into_owned();
    s = crate::speech::re(r"\[([^\]]+)\]\([^)]*\)").replace_all(&s, "$1").into_owned();
    s = crate::speech::re(r"`+([^`\n]*)`+").replace_all(&s, "$1").into_owned();
    s = crate::speech::re(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+").replace_all(&s, "").into_owned();
    s = crate::speech::re(r"(?m)^[ \t]{0,3}>[ \t]?").replace_all(&s, "").into_owned();
    s = crate::speech::re(r"(?m)^[ \t]*([-*+]|\d+[.)])[ \t]+").replace_all(&s, "").into_owned();
    s = crate::speech::re(r"(?m)^[ \t]*([-*_][ \t]*){3,}$").replace_all(&s, "").into_owned();
    s = crate::speech::re(r"(?m)^\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$")
        .replace_all(&s, "")
        .into_owned();
    s = crate::speech::re(r"\|").replace_all(&s, " ").into_owned();
    s = crate::speech::re(r"\*\*(.+?)\*\*").replace_all(&s, "$1").into_owned();
    s = crate::speech::re(r"__(.+?)__").replace_all(&s, "$1").into_owned();
    s = crate::speech::re(r"(?m)(^|[\s(])[*_]([^*_\n]+?)[*_]([\s).,!?;:]|$)").replace_all(&s, "$1$2$3").into_owned();
    s = s.replace("~~", "");
    s.split('\n')
        .map(|l| crate::speech::re(r"[ \t]+").replace_all(l, " ").trim().to_string())
        .filter(|l| !l.is_empty())
        .collect::<Vec<String>>()
        .join("\n")
}

#[cfg(test)]
mod plain_tests {
    use super::*;

    /// What the copy of an answer carries: the text you can see, without the marks.
    #[test]
    fn the_plain_copy_keeps_the_visible_text_only() {
        assert_eq!(plain("# Title\n\nSome **bold** here."), "Title\nSome bold here.");
        assert_eq!(plain("A [link](https://example.com) and `code`."), "A link and code.");
        assert_eq!(plain("```js\nrun();\n```\nAfter."), "run();\nAfter.", "the code is visible, the fences are not");
        assert_eq!(plain("| a | b |\n|---|---|\n| 1 | 2 |"), "a b\n1 2");
        assert_eq!(plain("Keep https://example.com here."), "Keep https://example.com here.", "the address is part of the text you see");
        assert_eq!(plain("> quoted\n- item"), "quoted\nitem");
        assert_eq!(plain("~~gone~~ *low* __und__"), "gone low und");
        assert!(plain("").is_empty());
    }
}
