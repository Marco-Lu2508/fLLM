//! The Settings page. Everything here is automatic until you switch it to manual: a manual value is sent with every request, an automatic one
//! is left to the server and the model.
//!
//! This file draws the settings of the answers (sampling, system prompt, longest answer), which the window keeps itself.

use crate::app::Message;
use crate::models::{bold, card, rule};
use crate::store::{Prefs, Sampling, BUILT_IN_SYSTEM_PROMPT};
use crate::theme::{self, Tokens};
use iced::widget::text_editor::{Content, Status};
use iced::widget::{button, column, container, pick_list, row, scrollable, slider, text, text_editor, text_input, Space};
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

/// The advanced sampling values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    TopK,
    MinP,
    RepetitionPenalty,
    PresencePenalty,
    FrequencyPenalty,
    DryMultiplier,
    Mirostat,
    Seed,
    Stop,
}

/// The texts of the boxes of those values: what you type is kept as typed, and applied to the settings when it is a valid number.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingTexts {
    pub top_k: String,
    pub min_p: String,
    pub repetition_penalty: String,
    pub presence_penalty: String,
    pub frequency_penalty: String,
    pub dry_multiplier: String,
    pub mirostat: String,
    pub seed: String,
    pub stop: String,
}

impl SamplingTexts {
    pub fn from(s: &Sampling) -> SamplingTexts {
        let num = |v: f32, neutral: f32| if (v - neutral).abs() < 1e-6 { String::new() } else { format!("{v}") };
        SamplingTexts {
            top_k: if s.top_k == 0 { String::new() } else { s.top_k.to_string() },
            min_p: num(s.min_p, 0.0),
            repetition_penalty: num(s.repetition_penalty, 1.0),
            presence_penalty: num(s.presence_penalty, 0.0),
            frequency_penalty: num(s.frequency_penalty, 0.0),
            dry_multiplier: num(s.dry_multiplier, 0.0),
            mirostat: if s.mirostat == 0 { String::new() } else { s.mirostat.to_string() },
            seed: s.seed.map(|v| v.to_string()).unwrap_or_default(),
            stop: s.stop.clone(),
        }
    }

    pub fn text(&self, f: Field) -> &str {
        match f {
            Field::TopK => &self.top_k,
            Field::MinP => &self.min_p,
            Field::RepetitionPenalty => &self.repetition_penalty,
            Field::PresencePenalty => &self.presence_penalty,
            Field::FrequencyPenalty => &self.frequency_penalty,
            Field::DryMultiplier => &self.dry_multiplier,
            Field::Mirostat => &self.mirostat,
            Field::Seed => &self.seed,
            Field::Stop => &self.stop,
        }
    }

    /// Takes what was typed into the box and the settings: an empty box is the neutral value, a text that is not a number changes nothing.
    pub fn apply(&mut self, f: Field, typed: String, s: &mut Sampling) {
        let float = |t: &str, neutral: f32, min: f32, max: f32| -> Option<f32> {
            if t.trim().is_empty() {
                Some(neutral)
            } else {
                t.trim().parse::<f32>().ok().filter(|v| v.is_finite() && (min..=max).contains(v))
            }
        };
        let ok = match f {
            Field::TopK => (typed.trim().is_empty() || typed.trim().parse::<u32>().is_ok_and(|v| v <= 100_000)).then(|| s.top_k = typed.trim().parse().unwrap_or(0)).is_some(),
            Field::MinP => float(&typed, 0.0, 0.0, 1.0).map(|v| s.min_p = v).is_some(),
            Field::RepetitionPenalty => float(&typed, 1.0, 0.0, 4.0).map(|v| s.repetition_penalty = v).is_some(),
            Field::PresencePenalty => float(&typed, 0.0, -2.0, 2.0).map(|v| s.presence_penalty = v).is_some(),
            Field::FrequencyPenalty => float(&typed, 0.0, -2.0, 2.0).map(|v| s.frequency_penalty = v).is_some(),
            Field::DryMultiplier => float(&typed, 0.0, 0.0, 10.0).map(|v| s.dry_multiplier = v).is_some(),
            Field::Mirostat => (typed.trim().is_empty() || typed.trim().parse::<u8>().is_ok_and(|v| v <= 2)).then(|| s.mirostat = typed.trim().parse().unwrap_or(0)).is_some(),
            Field::Seed => (typed.trim().is_empty() || typed.trim().parse::<i64>().is_ok()).then(|| s.seed = typed.trim().parse().ok().filter(|v: &i64| *v >= 0)).is_some(),
            Field::Stop => {
                s.stop = typed.clone();
                true
            }
        };
        // a box keeps what was typed even when it is not (yet) valid, like "0." on the way to "0.5"
        let _ = ok;
        match f {
            Field::TopK => self.top_k = typed,
            Field::MinP => self.min_p = typed,
            Field::RepetitionPenalty => self.repetition_penalty = typed,
            Field::PresencePenalty => self.presence_penalty = typed,
            Field::FrequencyPenalty => self.frequency_penalty = typed,
            Field::DryMultiplier => self.dry_multiplier = typed,
            Field::Mirostat => self.mirostat = typed,
            Field::Seed => self.seed = typed,
            Field::Stop => self.stop = typed,
        }
    }
}

fn small_input<'a>(t: Tokens, placeholder: &'a str, value: &'a str, field: Field, width: f32) -> Element<'a, Message> {
    text_input(placeholder, value)
        .on_input(move |v| Message::SamplingField(field, v))
        .on_submit(Message::SavePrefs)
        .size(14)
        .padding([6, 10])
        .width(width)
        .style(move |_, status| text_input::Style {
            background: iced::Background::Color(t.surface_2),
            border: iced::Border { radius: 8.0.into(), width: 1.0, color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator } },
            icon: t.muted,
            placeholder: t.muted,
            value: t.fg,
            selection: t.primary_soft,
        })
        .into()
}

/// The picker of the output format (the Studio's selConstraint with its five options).
fn constraint_picker<'a>(t: Tokens, constraint: &'a str) -> Element<'a, Message> {
    const OPTIONS: [&str; 5] = ["Free text", "JSON object", "JSON schema", "GBNF grammar", "Regular expression"];
    let current = match constraint {
        "json_object" => "JSON object",
        "json_schema" => "JSON schema",
        "gbnf" => "GBNF grammar",
        "regex" => "Regular expression",
        _ => "Free text",
    };
    pick_list(OPTIONS, Some(current), |label: &'static str| {
        Message::SamplingConstraint(
            match label {
                "JSON object" => "json_object",
                "JSON schema" => "json_schema",
                "GBNF grammar" => "gbnf",
                "Regular expression" => "regex",
                _ => "none",
            }
            .to_string(),
        )
    })
    .padding([6, 10])
    .style(move |_, status| pick_list::Style {
        background: iced::Background::Color(t.surface_2),
        border: iced::Border {
            radius: 8.0.into(),
            width: 1.0,
            color: if matches!(status, pick_list::Status::Hovered) { t.fg_soft } else { t.separator },
        },
        text_color: t.fg,
        placeholder_color: t.muted,
        handle_color: t.muted,
    })
    .into()
}

/// What the Settings page draws of the personalization (the state lives in the window).
#[derive(Debug, Clone)]
pub struct PersonalView<'a> {
    pub about: &'a Content,
    pub use_about: bool,
    pub use_memory: bool,
    pub memories: &'a [crate::memory::Memory],
    /// The line under the list: how many entries there are, whether all of them fit into the prompt.
    pub usage: String,
    pub new_text: &'a str,
    /// The memory whose text is being changed: its id and the text in the box.
    pub edit: Option<(&'a str, &'a str)>,
    /// The saved prompts of the library, and the box being typed in per id.
    pub prompts: &'a [crate::promptlib::SavedPrompt],
    pub plib_texts: &'a std::collections::HashMap<String, Content>,
}

/// A box for a text of the personalization (the input of a memory row, the field to add one).
fn mem_input<'a>(t: Tokens, placeholder: &'a str, value: &'a str, on_input: fn(String) -> Message, on_submit: Message) -> Element<'a, Message> {
    text_input(placeholder, value)
        .on_input(on_input)
        .on_submit(on_submit)
        .size(14)
        .padding([6, 10])
        .width(Length::Fill)
        .style(move |_, status| text_input::Style {
            background: iced::Background::Color(t.surface_2),
            border: iced::Border { radius: 8.0.into(), width: 1.0, color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator } },
            icon: t.muted,
            placeholder: t.muted,
            value: t.fg,
            selection: t.primary_soft,
        })
        .into()
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

#[allow(clippy::too_many_arguments)] // a page draws from eight pieces of state
pub fn page<'a>(
    t: Tokens,
    prefs: &'a Prefs,
    system_editor: &'a Content,
    spec_editor: &'a Content,
    max_tokens_text: &'a str,
    sampling: &'a SamplingTexts,
    engine: Element<'a, Message>,
    personal: PersonalView<'a>,
) -> Element<'a, Message> {
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
            .push(setting(t, "Top-p", "Only the most likely words that together make up this share are considered.", value_slider(t, 0.05..=1.0, prefs.top_p, 0.05, Message::TopP)))
            .push(setting(t, "Top-k", "Only this many most likely words are considered. Empty: no limit.", small_input(t, "off", &sampling.top_k, Field::TopK, 110.0)))
            .push(setting(t, "Min-p", "Drops words that are much less likely than the best one (0 to 1). Empty: off.", small_input(t, "off", &sampling.min_p, Field::MinP, 110.0)))
            .push(setting(t, "Repetition penalty", "Above 1 makes the model repeat itself less. Empty: 1.", small_input(t, "1", &sampling.repetition_penalty, Field::RepetitionPenalty, 110.0)))
            .push(setting(t, "Presence penalty", "Positive: new topics. Negative: stay on topic (-2 to 2). Empty: 0.", small_input(t, "0", &sampling.presence_penalty, Field::PresencePenalty, 110.0)))
            .push(setting(t, "Frequency penalty", "Positive: fewer repeated words (-2 to 2). Empty: 0.", small_input(t, "0", &sampling.frequency_penalty, Field::FrequencyPenalty, 110.0)))
            .push(setting(t, "DRY multiplier", "Penalizes repeated phrases, not only words. Empty: off.", small_input(t, "off", &sampling.dry_multiplier, Field::DryMultiplier, 110.0)))
            .push(setting(t, "Mirostat", "0 off, 1 or 2: the model aims at a steady level of surprise. Empty: off.", small_input(t, "off", &sampling.mirostat, Field::Mirostat, 110.0)))
            .push(setting(t, "Seed", "The same seed and settings give the same answer. Empty: random.", small_input(t, "random", &sampling.seed, Field::Seed, 140.0)))
            .push(setting(t, "Stop words", "The answer ends when one of these appears; separate them with commas.", small_input(t, "none", &sampling.stop, Field::Stop, 240.0)))
            .push(setting(
                t,
                "Output format",
                "The answer is generated under this shape. A schema, grammar or pattern is written below when one needs it.",
                constraint_picker(t, &prefs.sampling.constraint),
            ))
            .push(setting(
                t,
                "Show token probabilities",
                "Every answer shows its words with how likely the model thought them: the warmer the color, the more certain.",
                segmented(t, vec![
                    ("On", prefs.sampling.logprobs, Message::SamplingLogprobs(true)),
                    ("Off", !prefs.sampling.logprobs, Message::SamplingLogprobs(false)),
                ]),
            ));
        if matches!(prefs.sampling.constraint.as_str(), "json_schema" | "gbnf" | "regex") {
            answers = answers.push(setting(
                t,
                "Format specification",
                "The JSON schema, GBNF grammar or regular expression the answer is generated under.",
                text_editor(spec_editor)
                    .placeholder("JSON schema, GBNF grammar or regular expression")
                    .on_action(Message::SamplingSpecEdit)
                    .padding(12)
                    .size(13)
                    .height(Length::Shrink)
                    .min_height(74)
                    .max_height(180)
                    .style(move |_, status| text_editor::Style {
                        background: iced::Background::Color(t.surface_2),
                        border: iced::Border {
                            radius: 10.0.into(),
                            width: 1.0,
                            color: if matches!(status, Status::Focused { .. }) { t.fg_soft } else { t.separator }
                        },
                        placeholder: t.muted,
                        value: t.fg,
                        selection: iced::Color { a: 0.35, ..t.primary },
                    })
                    .into(),
            ));
        }
    }
    answers = answers.push(rule(t));
    answers = answers.push(setting(
        t,
        "System prompt",
        "What the model is told before every chat. Automatic is built from the date, your language, the model and what it can do.",
        segmented(t, vec![("Automatic", !prefs.custom_system, Message::SystemCustom(false)), ("Custom", prefs.custom_system, Message::SystemCustom(true))]),
    ));
    if !prefs.custom_system {
        answers = answers.push(setting(
            t,
            "Length of the automatic prompt",
            "Short saves reading time on a slow computer. Small models and small contexts always get the short one.",
            segmented(
                t,
                vec![
                    ("Full", prefs.prompt_length == crate::prompts::Length::Full, Message::PromptLength(crate::prompts::Length::Full)),
                    ("Short", prefs.prompt_length == crate::prompts::Length::Short, Message::PromptLength(crate::prompts::Length::Short)),
                    ("By speed", prefs.prompt_length == crate::prompts::Length::Auto, Message::PromptLength(crate::prompts::Length::Auto)),
                ],
            ),
        ));
    }
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
        answers = answers.push(setting(
            t,
            "Your text",
            "Replace the automatic prompt, or add yours after it. {{date}}, {{time}} and {{language}} are filled in.",
            segmented(t, vec![("Replace", !prefs.system_append, Message::SystemAppend(false)), ("Add after", prefs.system_append, Message::SystemAppend(true))]),
        ));
        answers = answers.push(editor);
        answers = answers.push(text("An empty text leaves the automatic prompt as it is.").size(12).color(t.muted));
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

    // ---- personalization (the Studio's Settings > Personalization: about me and memory) ----
    let mut pers = column![].spacing(12);
    pers = pers.push(
        column![
            text("Custom instructions").size(14),
            text("Put in front of the system prompt of your chats (switch it off for one chat in the chat menu). Everything on this page stays on this computer; it is sent only to the model you are chatting with.").size(12).color(t.muted),
        ]
        .spacing(3),
    );
    pers = pers.push(
        text_editor(personal.about)
            .placeholder("For example: I am a nurse. Keep answers short, use metric units, and explain medical terms.")
            .on_action(Message::AboutEdit)
            .padding(10)
            .size(14)
            .height(Length::Shrink)
            .min_height(72)
            .max_height(180)
            .style(move |_, status| text_editor::Style {
                background: iced::Background::Color(t.surface_2),
                border: iced::Border { radius: 10.0.into(), width: 1.0, color: if matches!(status, Status::Focused { .. }) { t.fg_soft } else { t.separator } },
                placeholder: t.muted,
                value: t.fg,
                selection: iced::Color { a: 0.35, ..t.primary },
            }),
    );
    pers = pers.push(
        row![
            text(format!("{} / {}", personal.about.text().chars().count(), crate::memory::limits::ABOUT))
                .size(11)
                .font(iced::Font::MONOSPACE)
                .color(t.muted),
            Space::new().width(Length::Fill),
            text("Use these instructions").size(12).color(t.fg_soft),
            segmented(
                t,
                vec![
                    ("On", personal.use_about, Message::UseAbout(true)),
                    ("Off", !personal.use_about, Message::UseAbout(false)),
                ],
            ),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    );
    pers = pers.push(rule(t));
    pers = pers.push(
        column![
            text("Remembered facts").size(14),
            text("The assistant saves durable facts about you on its own (a card with Undo appears in the chat). You can also start a message with \"remember that ...\" or press Remember under a message. Memories are added to the system prompt of every chat (up to 3,000 characters, fewer for small models), kept on this computer with a copy on the server, and can be switched off for one chat in the chat menu.").size(12).color(t.muted),
        ]
        .spacing(3),
    );
    pers = pers.push(
        row![
            text("Use memory").size(13).color(t.fg_soft),
            Space::new().width(Length::Fill),
            segmented(
                t,
                vec![
                    ("On", personal.use_memory, Message::UseMemory(true)),
                    ("Off", !personal.use_memory, Message::UseMemory(false)),
                ],
            ),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    );
    for m in personal.memories {
        let editing = personal.edit.is_some_and(|(id, _)| id == m.id);
        if editing {
            let text_box = personal.edit.map(|(_, text)| text).unwrap_or("");
            pers = pers.push(
                row![
                    mem_input(t, "Remember", text_box, Message::MemEditInput, Message::MemEditEnd(true)),
                    button(text("OK").size(12)).on_press(Message::MemEditEnd(true)).padding([4, 8]).style(theme::secondary_button(t)),
                    button(text("×").size(14)).on_press(Message::MemEditEnd(false)).padding([2, 6]).style(theme::ghost_button(t)),
                ]
                .spacing(4)
                .align_y(Alignment::Center),
            );
        } else {
            let mut line = row![
                button(text(m.text.clone()).size(13).width(Length::Fill).align_x(iced::alignment::Horizontal::Left))
                    .on_press(Message::MemEditStart(m.id.clone()))
                    .padding([5, 8])
                    .style(theme::plain_button(t)),
            ];
            if let Some(cat) = &m.category {
                line = line.push(
                    container(text(cat.clone()).size(11).color(t.muted))
                        .padding([1, 7])
                        .style(move |_| container::Style {
                            background: Some(iced::Background::Color(t.fill)),
                            border: iced::border::rounded(999),
                            ..Default::default()
                        }),
                );
            }
            line = line.push(
                button(text("×").size(14)).on_press(Message::MemDelete(m.id.clone())).padding([2, 6]).style(theme::ghost_button(t)),
            );
            pers = pers.push(line.spacing(6).align_y(Alignment::Center));
        }
    }
    pers = pers.push(
        row![
            mem_input(t, "Add something to remember", personal.new_text, Message::MemNewInput, Message::MemAddNow),
            button(text("Add").size(13)).on_press(Message::MemAddNow).padding([5, 12]).style(theme::secondary_button(t)),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    );
    pers = pers.push(
        row![
            text(personal.usage.clone()).size(12).color(t.muted).width(Length::Fill),
            button(text("Forget everything").size(13))
                .on_press(Message::MemClear)
                .padding([5, 12])
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
        .spacing(8)
        .align_y(Alignment::Center),
    );

    // ---- the prompt library (the Studio's "Saved prompts" in the same card) ----
    pers = pers.push(rule(t));
    pers = pers.push(
        column![
            text("Saved prompts").size(14),
            text("Type / in the message box to list them, pick one with the arrow keys and Enter, or keep typing to filter by title.").size(12).color(t.muted),
        ]
        .spacing(3),
    );
    if personal.prompts.is_empty() {
        pers = pers.push(text("No saved prompts yet.").size(13).color(t.muted));
    }
    for p in personal.prompts {
        // the box of this row lives in the window (every saved prompt gets one when the program
        // starts and when it is added; a row whose box is somehow missing is left out)
        let Some(held) = personal.plib_texts.get(&p.id) else { continue };
        let title_of = p.id.clone();
        let text_of = p.id.clone();
        pers = pers.push(
            container(
                row![
                    column![
                        text_input("Title", &p.title)
                            .on_input(move |v| Message::PromptTitle(title_of.clone(), v))
                            .size(13)
                            .padding([6, 8])
                            .width(Length::Fill)
                            .style(move |_, status| text_input::Style {
                                background: iced::Background::Color(t.surface_2),
                                border: iced::Border { radius: 8.0.into(), width: 1.0, color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { t.separator } },
                                icon: t.muted,
                                placeholder: t.muted,
                                value: t.fg,
                                selection: t.primary_soft,
                            }),
                        text_editor(held)
                            .placeholder("Prompt")
                            .on_action(move |a| Message::PromptText(text_of.clone(), a))
                            .size(13)
                            .padding([6, 8])
                            .height(Length::Shrink)
                            .min_height(56)
                            .max_height(160)
                            .style(move |_, status| text_editor::Style {
                                background: iced::Background::Color(t.surface_2),
                                border: iced::Border { radius: 8.0.into(), width: 1.0, color: if matches!(status, Status::Focused { .. }) { t.fg_soft } else { t.separator } },
                                placeholder: t.muted,
                                value: t.fg,
                                selection: iced::Color { a: 0.35, ..t.primary },
                            }),
                    ]
                    .spacing(6)
                    .width(Length::Fill),
                    button(text("×").size(14)).on_press(Message::PromptDel(p.id.clone())).padding([2, 6]).style(theme::ghost_button(t)),
                ]
                .spacing(4)
                .align_y(Alignment::Start),
            )
            .width(Length::Fill)
            .padding(8)
            .style(move |_| container::Style {
                border: iced::border::rounded(8).width(1.0).color(t.separator),
                ..Default::default()
            }),
        );
    }
    pers = pers.push(
        button(text("Add a prompt").size(13))
            .on_press(Message::PromptAdd)
            .padding([5, 12])
            .style(theme::secondary_button(t)),
    );

    // the General card of the Studio (shell.html): the welcome tour comes back from here
    let general = column![setting(
        t,
        "Welcome tour",
        "A short introduction to fLLM, your computer and the first steps.",
        button(text("Show the welcome tour").size(13))
            .on_press(Message::TourShow)
            .padding([5, 12])
            .style(theme::secondary_button(t))
            .into(),
    )]
    .push(setting(
        t,
        "Keyboard shortcuts",
        "See and change the keys.",
        button(text("Keyboard shortcuts").size(13))
            .on_press(Message::ShortcutsOpen)
            .padding([5, 12])
            .style(theme::secondary_button(t))
            .into(),
    ))
    .spacing(14);

    let content = column![
        container(text("Settings").size(26).font(bold())).padding(Padding { top: 14.0, right: 0.0, bottom: 4.0, left: 2.0 }),
        card(t, "General", general.into()),
        card(t, "Answers", answers.into()),
        card(t, "Personalization", pers.into()),
        engine,
    ]
    .spacing(14);
    let inner = container(content).max_width(780).width(Length::Fill).padding(Padding { top: 8.0, right: 24.0, bottom: 32.0, left: 24.0 });
    scrollable(container(inner).center_x(Length::Fill).width(Length::Fill)).height(Length::Fill).into()
}
