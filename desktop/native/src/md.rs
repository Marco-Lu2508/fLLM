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

/// The Studio typesets the math of an answer: a display paragraph (`$$x$$`, `\\[..\\]`) gets its own
/// line, typeset with the real layout of `rich::math`; an inline piece (`$x$`) stands between the
/// words on the baseline of the line, which breaks where the words break. Everything else is drawn
/// exactly as the paragraph is today.
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
        // the spans between two character positions, with the styles they had
        let slice = |from: usize, to: usize| -> Vec<Span> {
            let mut out: Vec<Span> = Vec::new();
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
            out
        };
        let size = settings.text_size.0;
        let t = self.t;
        let mut blocks: Vec<Element<'a, Message>> = Vec::new();
        let mut flow: Vec<Atom> = Vec::new();
        let mut cursor = 0usize;
        for hit in hits {
            if hit.start > cursor {
                flow.push(Atom::Words(slice(cursor, hit.start)));
            }
            let art = crate::rich::math(&hit.body, hit.display, if hit.display { size * 1.2 } else { size }, t.fg);
            match (art, hit.display) {
                (Ok(art), true) => {
                    if !flow.is_empty() {
                        blocks.push(inline_flow(std::mem::take(&mut flow), size, t, Self::on_link_click));
                    }
                    blocks.push(display_math(art));
                }
                (Ok(art), false) => flow.push(Atom::Math(art)),
                // what is not understood stays visible as the source, in a quiet chip
                (Err(_), display) => {
                    let source = if display { hit.body.clone() } else { format!("${}$", hit.body) };
                    let mut chip = slice(hit.start, hit.end).into_iter().next().unwrap_or_else(|| Span::new(source.clone()));
                    chip.text = source.into();
                    chip = chip.font(Font::MONOSPACE).color(t.muted);
                    if display {
                        if !flow.is_empty() {
                            blocks.push(inline_flow(std::mem::take(&mut flow), size, t, Self::on_link_click));
                        }
                        blocks.push(container(rich_text([chip]).on_link_click(Self::on_link_click)).padding([6, 8]).into());
                    } else {
                        flow.push(Atom::Words(vec![chip]));
                    }
                }
            }
            cursor = hit.end;
        }
        if cursor < last {
            flow.push(Atom::Words(slice(cursor, last)));
        }
        if !flow.is_empty() {
            blocks.push(inline_flow(flow, size, t, Self::on_link_click));
        }
        if blocks.is_empty() {
            return markdown::paragraph(settings, text, Self::on_link_click);
        }
        if blocks.len() == 1 {
            return blocks.remove(0);
        }
        column(blocks).spacing(6).into()
    }

    fn on_link_click(url: markdown::Uri) -> Message {
        Message::LinkClicked(url)
    }

    fn code_block(&self, settings: markdown::Settings, language: Option<&'a str>, code: &'a str, lines: &'a [markdown::Text]) -> Element<'a, Message> {
        let t = self.t;
        let info = language.unwrap_or("");
        let lang = info.split_whitespace().next().unwrap_or("");
        // a chart, a diagram, a picture or a formula is drawn; when it cannot be (yet: the answer
        // may still be coming in), the block stays the code it is
        let mut note: Option<String> = None;
        match rich_block(lang, code, t, settings.text_size.0) {
            Ok(Some(block)) => return block,
            Ok(None) => {}
            Err(why) => note = why,
        }
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
        if let Some(why) = note {
            col = col.push(container(text(why).size(12).color(t.danger)).padding(Padding { top: 0.0, right: 14.0, bottom: 10.0, left: 14.0 }));
        }
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

/// The block of a chart, a diagram, an SVG picture or a formula: `Ok(Some)` when it is drawn,
/// `Ok(None)` when the block is plain code, `Err` when it is one of them but cannot be drawn (with a
/// sentence for the reader when that is not just an answer that is still being written).
fn rich_block<'a>(lang: &str, code: &str, t: Tokens, size: f32) -> Result<Option<Element<'a, Message>>, Option<String>> {
    let lang = lang.to_ascii_lowercase();
    let (label, art) = match lang.as_str() {
        "chart" | "chartjs" | "chart.js" => {
            let art = crate::rich::chart::render(code, t);
            if art.is_err() && crate::rich::chart::unfinished(code) {
                return Ok(None);
            }
            ("chart", art.map_err(|e| Some(format!("This chart could not be drawn: {e}"))))
        }
        "mermaid" => ("diagram", crate::rich::diagram(code, t, t.dark).map_err(|_| None)),
        "svg" if code.trim_start().starts_with("<svg") || code.trim_start().starts_with("<?xml") => ("svg", crate::rich::picture(code).map_err(|_| None)),
        "math" | "latex" | "tex" | "katex" => {
            let body = code.trim().trim_start_matches("$$").trim_end_matches("$$").trim_start_matches("\\[").trim_end_matches("\\]");
            ("math", crate::rich::math(body, true, size * 1.25, t.fg).map_err(|_| None))
        }
        _ => return Ok(None),
    };
    let art = art?;
    let name = format!("{label}.svg");
    let header = row![
        text(label).size(12).color(t.muted).font(Font { weight: font::Weight::Medium, ..crate::app::UI_FONT }),
        Space::new().width(Length::Fill),
        iced::widget::button(text("Save SVG").size(12)).on_press(Message::SaveRich { name, svg: art.source.clone() }).padding([2, 8]).style(theme::ghost_button(t)),
        iced::widget::button(text("Copy code").size(12)).on_press(Message::CopyText(code.to_string())).padding([2, 8]).style(theme::ghost_button(t)),
    ]
    .spacing(6)
    .align_y(iced::Alignment::Center)
    .padding(Padding { top: 4.0, right: 6.0, bottom: 0.0, left: 14.0 });
    let picture = iced::widget::svg(art.handle.clone()).width(Length::Fill).height(Length::Shrink);
    let picture: Element<'a, Message> = if label == "math" {
        // a formula is centred and as big as it is; a wide one scrolls
        display_math(art)
    } else {
        container(picture).padding(Padding { top: 6.0, right: 14.0, bottom: 14.0, left: 14.0 }).center_x(Length::Fill).into()
    };
    Ok(Some(container(column![header, picture]).width(Length::Fill).style(theme::inset(t)).into()))
}

type Span = iced::widget::text::Span<'static, markdown::Uri>;

/// A piece of a line: words, or a piece of typeset math.
enum Atom {
    Words(Vec<Span>),
    Math(crate::rich::Art),
}

/// A display formula on a line of its own, centred, as big as it is; a formula wider than the
/// answer is made smaller to fit.
fn display_math<'a>(art: crate::rich::Art) -> Element<'a, Message> {
    let w = art.width;
    let picture = iced::widget::svg(art.handle).width(Length::Fill).height(Length::Shrink);
    container(container(picture).max_width(w)).padding([6, 8]).center_x(Length::Fill).into()
}

/// Words and inline math as one wrapping line: the words are cut at their spaces, a piece of math
/// that touches a word (`$x$,`) stays glued to it, and the math sits on the baseline of the text.
/// What a unit of a line is made of.
enum Part {
    Words(Vec<Span>),
    Math(crate::rich::Art),
}

/// The pieces of a line that wrap as one: a word, or a piece of math with the punctuation that
/// touches it (`$x$,`). Words are cut at spaces, whatever span the space is in.
fn units_of(atoms: Vec<Atom>) -> Vec<Vec<Part>> {
    let mut units: Vec<Vec<Part>> = Vec::new();
    let mut unit: Vec<Part> = Vec::new();
    for atom in atoms {
        match atom {
            Atom::Math(art) => unit.push(Part::Math(art)),
            Atom::Words(spans) => {
                for span in spans {
                    let text = span.text.to_string();
                    let mut word = String::new();
                    let flush = |word: &mut String, unit: &mut Vec<Part>| {
                        if word.is_empty() {
                            return;
                        }
                        let mut piece = span.clone();
                        piece.text = std::mem::take(word).into();
                        match unit.last_mut() {
                            Some(Part::Words(w)) => w.push(piece),
                            _ => unit.push(Part::Words(vec![piece])),
                        }
                    };
                    for ch in text.chars() {
                        if ch.is_whitespace() {
                            flush(&mut word, &mut unit);
                            if !unit.is_empty() {
                                units.push(std::mem::take(&mut unit));
                            }
                        } else {
                            word.push(ch);
                        }
                    }
                    flush(&mut word, &mut unit);
                }
            }
        }
    }
    if !unit.is_empty() {
        units.push(unit);
    }
    units
}

/// Words and inline math as one wrapping line: the math sits on the baseline of the text.
fn inline_flow<'a>(atoms: Vec<Atom>, size: f32, t: Tokens, on_link: fn(markdown::Uri) -> Message) -> Element<'a, Message> {
    let _ = t;
    // the line of the text: Inter's ascent and descent in a line of 1.3 times the size
    let line = size * 1.3;
    let below_centre = size * (0.969 + (1.3 - 1.21) / 2.0) - line / 2.0;
    let elements: Vec<Element<'a, Message>> = units_of(atoms)
        .into_iter()
        .map(|parts| {
            let items: Vec<Element<'a, Message>> = parts
                .into_iter()
                .map(|part| match part {
                    Part::Words(spans) => rich_text(spans).size(size).on_link_click(on_link).into(),
                    Part::Math(art) => {
                        let (w, h, b) = (art.width, art.height, art.baseline);
                        // a box that is as tall on both sides of the centre of the text as the math needs
                        let half = (b - below_centre).max(h - b + below_centre).max(line / 2.0);
                        let top = half + below_centre - b;
                        let bottom = (2.0 * half - h - top).max(0.0);
                        container(iced::widget::svg(art.handle).width(Length::Fixed(w)).height(Length::Fixed(h)))
                            .padding(Padding { top: top.max(0.0), right: 1.0, bottom, left: 1.0 })
                            .into()
                    }
                })
                .collect();
            row(items).align_y(iced::Alignment::Center).into()
        })
        .collect();
    row(elements).spacing(size * 0.4).align_y(iced::Alignment::Center).wrap().vertical_spacing(2).into()
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

#[cfg(test)]
mod flow_tests {
    use super::*;

    fn words(s: &str) -> Atom {
        Atom::Words(vec![Span::new(s.to_string())])
    }

    fn shape(units: &[Vec<Part>]) -> Vec<String> {
        units
            .iter()
            .map(|u| {
                u.iter()
                    .map(|p| match p {
                        Part::Words(w) => w.iter().map(|s| s.text.to_string()).collect::<String>(),
                        Part::Math(_) => "<math>".to_string(),
                    })
                    .collect::<String>()
            })
            .collect()
    }

    #[test]
    fn words_wrap_one_by_one_and_math_stays_glued_to_the_punctuation_that_touches_it() {
        let m = crate::rich::math("x^2", false, 15.0, iced::Color::BLACK).unwrap();
        let units = units_of(vec![words("The roots of "), Atom::Math(m.clone()), words(", and  so on")]);
        assert_eq!(shape(&units), ["The", "roots", "of", "<math>,", "and", "so", "on"]);
        // math with a space on both sides is a unit of its own
        assert_eq!(shape(&units_of(vec![words("a "), Atom::Math(m.clone()), words(" b")])), ["a", "<math>", "b"]);
        // a word that is cut by a change of style stays one word: **bo**ld
        let bold = Atom::Words(vec![Span::new("bo".to_string()), Span::new("ld text".to_string())]);
        assert_eq!(shape(&units_of(vec![bold])), ["bold", "text"]);
        // math at the start and at the end
        assert_eq!(shape(&units_of(vec![Atom::Math(m.clone()), words(" is"), Atom::Math(m)])), ["<math>", "is<math>"]);
        assert!(units_of(vec![words("   ")]).is_empty());
    }
}
