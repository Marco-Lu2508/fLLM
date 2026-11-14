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

/// What the hardware card shows: the devices of this machine, the notice of what runs on the CPU,
/// the rows over them and what every card's check answered (`Settings > Hardware` of the Studio).
pub struct HardwareView<'a> {
    /// The whole report of `GET /api/studio/hardware`; `None` while it is still being read.
    pub report: Option<&'a crate::tour::Hardware>,
    pub notice: Option<&'a crate::gpucheck::FallbackNotice>,
    pub busy: &'a [String],
    pub results: &'a [(String, Vec<crate::gpucheck::GpuResult>)],
    /// The answer of `GET /api/npu`; `None` while the card is still checking.
    pub npu: Option<&'a serde_json::Value>,
    /// The `--device` value of the loaded model (`auto` when the server chose).
    pub load_device: &'a str,
    /// Whether a model is loaded (the roles of `deviceRole` say *running* then).
    pub model_loaded: bool,
    /// The answer under *Run speed test*, and whether it is running right now.
    pub speed_test: Option<&'a String>,
    pub speed_test_busy: bool,
    /// The instruction set cap of the server (`config.cpu_tier_cap` of the status).
    pub tier_cap: &'a str,
    /// Whether the weights are repacked for long prompts (`config.cpu_repack`).
    pub cpu_repack: bool,
    /// The report of *Check the processor* (`btnRunSimdSelftest`); `None` until it answered.
    pub selftest: Option<&'a crate::gpucheck::Selftest>,
    /// The state of the AMD NPU card (`GET /api/studio/npu/amd`).
    pub amd_npu: Option<&'a serde_json::Value>,
    /// The model the list of that card holds.
    pub amd_chosen: &'a str,
    /// The slots of the running server (`GET /slots`) for the table of *Chats answered at the
    /// same time*.
    pub slots: &'a [crate::models::Slot],
}

/// What the appearance card shows: the four choices of the Studio's `selThemeMode`,
/// `accentPickerRow`, `selFontSize` and the two switches.
pub struct LookView<'a> {
    pub theme: &'a str,
    pub accent: &'a str,
    pub font_size: u16,
    pub tint: bool,
    pub speed: bool,
}

/// What the usage card shows: the counters, the price rows and the line under its buttons.
pub struct UsageView<'a> {
    pub data: &'a crate::usage::UsageData,
    pub prices: &'a [crate::usage::Price],
    pub status: Option<&'a str>,
}

/// What the settings show about this machine in the voice cards: its voices and whether it can
/// notify (the `fillVoices` and `notifyNote` of `voice.js`).
#[derive(Debug, Clone, Default)]
pub struct VoiceView<'a> {
    /// The voices of this computer; empty = no speech program was found.
    pub voices: &'a [String],
    /// Whether a program for system notifications is installed.
    pub can_notify: bool,
}

/// One voice of the select: the empty one is the default voice (`<option value="">Default</option>).
/// One language of the select: the tag is what is kept, the name is what is shown (`LANGUAGES` of
/// `voice.js`, where the option value is the tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LangChoice(&'static str, &'static str);

impl std::fmt::Display for LangChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VoiceChoice(String);

impl std::fmt::Display for VoiceChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            f.write_str("Default")
        } else {
            f.write_str(&self.0)
        }
    }
}

// the page of the settings carries every block of it (the Studio reads its rows from the page)
#[allow(clippy::too_many_arguments)]
/// The hardware card of the Studio (`spane-hardware` of `shell.html`): what this machine has —
/// the two buttons of the header, the speed test, the instruction set, the repack switch and the
/// system memory over the devices, the notice of what runs on the CPU, and a check button on every
/// card that can be tested.
fn hardware_card<'a>(t: Tokens, view: HardwareView<'a>) -> Element<'a, Message> {
    let empty = crate::tour::Hardware::default();
    let report = view.report.unwrap_or(&empty);
    // one cell of the two tables of this card: the width is the column of the Studio
    let cell = |value: String, width: f32, colour: iced::Color| {
        text(value).size(13).color(colour).width(Length::Fixed(width))
    };
    let mut body = column![].spacing(14);
    body = body.push(
        text("The processor and graphics cards fLLM can use, and a few ways to check that they work.")
            .size(13)
            .color(t.muted),
    );

    // the buttons of the card header (`btnMeasureBandwidth` and `btnRunSimdSelftest`)
    body = body.push(
        row![
            button(text("Measure memory speed").size(12).color(t.fg_soft))
                .on_press(Message::MeasureBandwidth)
                .padding([5, 10])
                .style(crate::app::starter_style(t)),
            button(text("Check the processor").size(12).color(t.fg_soft))
                .on_press(Message::SelftestCpu)
                .padding([5, 10])
                .style(crate::app::starter_style(t)),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    );

    // Speed test: the line is what `speedTestResult` holds — the help of the Studio until the
    // first run, then what the answer said
    let speed_line = view
        .speed_test
        .map(String::as_str)
        .unwrap_or("Runs a short answer with the loaded model and measures how fast it reads and writes.");
    let speed_button = {
        let b = button(text("Run speed test").size(12).color(t.fg_soft))
            .padding([5, 10])
            .style(crate::app::starter_style(t));
        if view.speed_test_busy {
            // the button is off while the test runs (`button.disabled = true`)
            b
        } else {
            b.on_press(Message::SpeedTest)
        }
    };
    body = body.push(setting(t, "Speed test", speed_line, speed_button.into()));

    // Processor feature limit: `auto` and every instruction set the report names, with the value
    // the server runs with (`selSimdTierCap` of `refreshHardware`)
    let tiers = crate::tour::tier_options(report);
    let labels: Vec<String> = tiers.iter().map(|tier| crate::tour::tier_label(tier)).collect();
    let wanted = if tiers.iter().any(|tier| tier == view.tier_cap) {
        view.tier_cap
    } else {
        "auto"
    };
    let current = crate::tour::tier_label(wanted);
    let tier_picker = pick_list(labels, Some(current), |label: String| {
        Message::TierCap(if label.starts_with("Auto (highest") {
            String::from("auto")
        } else {
            label
        })
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
    });
    body = body.push(setting(
        t,
        "Processor feature limit",
        "Stops the processor from using its newest features. Only useful when something misbehaves; leave it on Auto.",
        tier_picker.into(),
    ));

    // Faster reading of long prompts (`chkCpuRepack`)
    body = body.push(setting(
        t,
        "Faster reading of long prompts",
        "Prepares the model so that long texts are read faster. Uses some extra memory.",
        segmented(
            t,
            vec![
                ("On", view.cpu_repack, Message::CpuRepack(true)),
                ("Off", !view.cpu_repack, Message::CpuRepack(false)),
            ],
        ),
    ));

    // System memory: the line of the Studio under the title, the bar and the numbers beside it —
    // before the report arrives the row holds what `shell.html` writes into it
    let (memory_line, memory_free, memory_pct) = match view.report {
        None => (String::from("Measuring ..."), String::from("—"), 35),
        Some(_) => {
            let memory = crate::tour::system_memory(report);
            (memory.desc, memory.free_text, memory.pct)
        }
    };
    let used = memory_pct.clamp(2, 100) as u16;
    let rest = 100 - used;
    let bar = row![
        container(Space::new())
            .width(Length::FillPortion(used))
            .height(6.0)
            .style(move |_| iced::widget::container::Style {
                background: Some(iced::Background::Color(t.primary)),
                border: iced::border::rounded(3),
                ..Default::default()
            }),
        container(Space::new())
            .width(Length::FillPortion(rest.max(1)))
            .height(6.0)
            .style(move |_| iced::widget::container::Style {
                background: Some(iced::Background::Color(t.separator)),
                border: iced::border::rounded(3),
                ..Default::default()
            }),
        text(memory_free).size(12).color(t.muted),
    ]
    .spacing(8)
    .align_y(Alignment::Center);
    // the row of `setting`, with the line owned: it is built from the report, not a literal
    body = body.push(
        row![
            column![
                text("System memory").size(14),
                text(memory_line).size(12).color(t.muted)
            ]
            .spacing(3)
            .width(Length::Fill),
            container(bar).width(280).align_x(iced::alignment::Horizontal::Right),
        ]
        .spacing(20)
        .align_y(Alignment::Center),
    );

    // the amber notice of what runs on the CPU (`hwFallbackNotice` of the Studio)
    if let Some(notice) = view.notice {
        let mut notice_body = column![].spacing(6);
        notice_body = notice_body.push(text(notice.title.clone()).size(13).font(bold()));
        notice_body = notice_body.push(text(notice.summary.clone()).size(13).color(t.muted));
        for line in &notice.lines {
            notice_body = notice_body.push(text(line.clone()).size(13).color(t.fg_soft));
        }
        let hint = if notice.detail.is_empty() {
            String::from(
                "Run fllm-rs doctor --gpu in a terminal, or press \"Test this GPU\" below, to see in one line whether the card does the work.",
            )
        } else {
            format!(
                "Run fllm-rs doctor --gpu in a terminal, or press \"Test this GPU\" below, to see in one line whether the card does the work. {}",
                notice.detail
            )
        };
        notice_body = notice_body.push(text(hint).size(12).color(t.muted));
        body = body.push(
            container(notice_body)
                .width(Length::Fill)
                .padding(12)
                .style(move |_| iced::widget::container::Style {
                    background: Some(iced::Background::Color(if t.dark {
                        iced::Color { r: 0.35, g: 0.25, b: 0.05, a: 1.0 }
                    } else {
                        iced::Color { r: 1.0, g: 0.95, b: 0.8, a: 1.0 }
                    })),
                    text_color: Some(t.fg),
                    border: iced::Border { radius: theme::R_MD.into(), width: 1.0, color: t.separator },
                    ..Default::default()
                }),
        );
    }

    // the devices: one card per physical device, grouped the way `mergeDevices` of the Studio
    // groups them, with the role of each one and the sentence of where the model runs
    let target = if view.load_device == "auto" {
        report.auto_choice.as_str()
    } else {
        view.load_device
    };
    let groups = crate::npu::merge_devices(&report.devices, target);
    let mut devices = column![].spacing(10);
    if !groups.is_empty() {
        // what the Studio writes above the list (`hwChoiceText`)
        let chosen = groups
            .iter()
            .find(|g| g.members.iter().any(|m| m.id.eq_ignore_ascii_case(report.auto_choice.as_str())));
        let choice = match chosen {
            Some(g) => format!(
                "With the automatic choice fLLM runs models on: {} ({}). The processor is the fallback.",
                g.primary.name,
                crate::npu::device_backend_label(&g.primary)
            ),
            None => String::from("The processor runs the models: no usable graphics device was found."),
        };
        body = body.push(text(choice).size(13).color(t.muted));
    }
    for group in &groups {
        let device = &group.primary;
        // a device that reports no memory cannot hold a model: it stays out of the list
        if device.backend != "cpu" && device.total_memory_bytes == 0 {
            continue;
        }
        let busy = view.busy.contains(&device.id);
        let results = view
            .results
            .iter()
            .find(|(key, _)| key == &device.id)
            .map(|(_, lines)| lines.as_slice())
            .unwrap_or(&[]);
        let check = crate::gpucheck::gpu_check_view(&device.backend, busy, results);
        let member_ids: Vec<String> = group.members.iter().map(|m| m.id.clone()).collect();
        let role = crate::npu::device_role(
            &member_ids,
            &device.id,
            view.load_device,
            report.auto_choice.as_str(),
            view.model_loaded,
        );
        let mut card_row = column![].spacing(4);
        let mut head = row![].spacing(8).align_y(Alignment::Center);
        head = head.push(text(device.name.clone()).size(14).font(bold()));
        head = head.push(text(crate::npu::device_backend_label(device)).size(12).color(t.muted));
        if device.total_memory_bytes > 0 {
            head = head.push(
                text(format!("· {}", crate::app::format_bytes(device.total_memory_bytes)))
                    .size(12)
                    .color(t.muted),
            );
        }
        if !role.is_empty() {
            head = head.push(
                text(if role == "running" { "· the model runs here" } else { "· the automatic choice" })
                    .size(12)
                    .color(t.ok),
            );
        }
        if group.others.len() == 1 {
            head = head.push(text(format!("+ {}", group.others[0].1)).size(12).color(t.muted));
        } else if !group.others.is_empty() {
            head = head.push(text(format!("+ {} more interfaces", group.others.len())).size(12).color(t.muted));
        }
        card_row = card_row.push(head);
        // the bar of what the card has used (`deviceMemory`: ok, a warning, high): the used part
        // over the track, never nothing (the card would show no bar at all)
        let memory = crate::npu::device_memory(device);
        if memory.known {
            let colour = match memory.level {
                "high" => t.danger,
                "warn" => t.primary,
                _ => t.ok,
            };
            // the used part over the track: two portions add up to the whole width
            let used = memory.pct.clamp(4, 100) as u16;
            let rest = 100 - used;
            let bar = row![
                container(Space::new())
                    .width(Length::FillPortion(used))
                    .height(4.0)
                    .style(move |_| iced::widget::container::Style {
                        background: Some(iced::Background::Color(colour)),
                        border: iced::Border { radius: 2.0.into(), ..Default::default() },
                        ..Default::default()
                    }),
                container(Space::new())
                    .width(Length::FillPortion(rest.max(1)))
                    .height(4.0)
                    .style(move |_| iced::widget::container::Style {
                        background: Some(iced::Background::Color(t.separator)),
                        border: iced::Border { radius: 2.0.into(), ..Default::default() },
                        ..Default::default()
                    }),
            ];
            card_row = card_row.push(bar);
        }
        if check.show_button && !device.id.is_empty() {
            let id = device.id.clone();
            card_row = card_row.push(
                button(text(check.label).size(12).color(t.fg_soft))
                    .on_press(Message::GpuCheck(id))
                    .padding([5, 10])
                    .style(crate::app::starter_style(t)),
            );
        }
        for line in &check.lines {
            let colour = match line.cls {
                "ok" => t.ok,
                "warn" => t.primary,
                _ => t.danger,
            };
            card_row = card_row.push(text(line.text.clone()).size(13).color(colour));
        }
        devices = devices.push(
            container(card_row)
                .width(Length::Fill)
                .padding(10)
                .style(theme::card(t, theme::R_MD, false)),
        );
    }
    if report.devices.is_empty() {
        body = body.push(text("Reading the devices of this machine …").size(13).color(t.muted));
    } else {
        body = body.push(text("Compute devices").size(15).font(bold()));
        body = body.push(devices);
    }

    // The NPU card of `npu.js`: the summary decides what of it shows, the four paragraphs of
    // NPU_EXPLANATION stand under it in any case.
    let npu = crate::npu::npu_card_model(view.npu);
    let mut npu_body = column![].spacing(8);
    npu_body = npu_body.push(text("NPU (neural processing unit)").size(15).font(bold()));
    let summary_colour = match npu.tone {
        "good" => t.ok,
        "warn" => t.primary,
        _ => t.muted,
    };
    npu_body = npu_body.push(text(npu.summary.clone()).size(13).color(summary_colour));
    if !npu.offload.is_empty() {
        npu_body = npu_body.push(text(npu.offload.clone()).size(13).color(t.fg_soft));
    }
    for row in &npu.hardware {
        let badge = if row.supported { t.ok } else if row.badge.starts_with("AMD") { t.primary } else { t.muted };
        npu_body = npu_body.push(
            row![text(row.label.clone()).size(13), Space::new().width(8.0), text(row.badge.clone()).size(12).color(badge)]
                .align_y(Alignment::Center),
        );
        if let Some(problem) = &row.problem {
            npu_body = npu_body.push(text(problem.clone()).size(12).color(t.muted));
        }
    }
    if npu.progress.visible {
        npu_body = npu_body.push(text(npu.progress.text.clone()).size(13).color(t.fg_soft));
    }
    if let Some(error) = &npu.error {
        npu_body = npu_body.push(text(error.clone()).size(13).color(t.danger));
    }
    let mut npu_actions = row![].spacing(8).align_y(Alignment::Center);
    if let Some(button_data) = &npu.button {
        // `npuSetupBtn`: the button of the card always installs (`canInstall` gives action install)
        let action: &'static str = if button_data.action == "cancel" { "cancel" } else { "install" };
        npu_actions = npu_actions.push(
            button(text(button_data.label.clone()).size(13).color(t.fg_soft))
                .on_press(Message::NpuSetup(action))
                .padding([8, 14])
                .style(crate::app::starter_style(t)),
        );
    }
    if npu.cancel {
        npu_actions = npu_actions.push(
            button(text("Cancel").size(13).color(t.fg_soft))
                .on_press(Message::NpuSetup("cancel"))
                .padding([8, 14])
                .style(crate::app::starter_style(t)),
        );
    }
    if npu.remove {
        // the Studio asks first (`confirmDialog` in front of the DELETE)
        npu_actions = npu_actions.push(
            button(text("Remove the runtime").size(13).color(t.fg_soft))
                .on_press(Message::NpuRemove)
                .padding([8, 14])
                .style(crate::app::starter_style(t)),
        );
    }
    if npu.button.is_some() || npu.cancel || npu.remove {
        npu_body = npu_body.push(npu_actions);
    }
    for format in &npu.formats {
        npu_body = npu_body.push(
            column![text(format.title).size(12).color(t.muted), text(format.items.join(", ")).size(13)]
                .spacing(3),
        );
    }
    npu_body = npu_body.push(text("Explanation").size(12).font(bold()).color(t.muted));
    for paragraph in npu.explanation {
        npu_body = npu_body.push(text(paragraph).size(12).color(t.muted));
    }
    body = body.push(
        container(npu_body).width(Length::Fill).padding(14).style(theme::card(t, theme::R_LG, false)),
    );

    // The AMD NPU card (`amdNpuCard` of `amdnpu.js`): away when this machine has no AMD NPU and
    // no FastFlowLM — the card only shows itself (`visible`).
    let amd = crate::amdnpu::amd_npu_model(view.amd_npu, view.amd_chosen);
    if amd.visible {
        let mut amd_body = column![].spacing(8);
        amd_body = amd_body.push(text("AMD NPU (Ryzen AI): native kernels and FastFlowLM").size(15).font(bold()));
        let amd_tone = if amd.tone == "good" {
            t.ok
        } else if amd.tone == "warn" {
            t.primary
        } else {
            t.muted
        };
        amd_body = amd_body.push(text(amd.summary.clone()).size(13).color(amd_tone));
        if !amd.native_text.is_empty() {
            let colour = if amd.native_ready { t.ok } else { t.muted };
            amd_body = amd_body.push(text(amd.native_text.clone()).size(13).color(colour));
        }
        for h in &amd.hardware {
            amd_body = amd_body.push(match &h.problem {
                Some(problem) => text(format!("{} - {problem}", h.label)).size(13).color(t.fg_soft),
                None => text(h.label.clone()).size(13).color(t.fg_soft),
            });
        }
        if let Some(install) = &amd.install {
            amd_body = amd_body.push(text(install.text.clone()).size(13).color(t.fg_soft));
            amd_body = amd_body.push(
                button(text(install.url.clone()).size(12).color(t.primary))
                    .on_press(Message::OpenUrl(install.url.clone()))
                    .padding([0, 0])
                    .style(theme::ghost_button(t)),
            );
        }
        for problem in &amd.problems {
            amd_body = amd_body.push(text(problem.clone()).size(13).color(t.danger));
        }

        // the list of models and the buttons of what to do with the chosen one
        let mut amd_actions = row![].spacing(8).align_y(Alignment::Center);
        if !amd.models.is_empty() {
            let options: Vec<String> = amd.models.iter().map(|m| m.label.clone()).collect();
            let selected = amd
                .models
                .iter()
                .find(|m| m.tag == amd.selected)
                .map(|m| m.label.clone());
            let picker = pick_list(options, selected, |label: String| {
                // the option reads `tag` or `tag (not downloaded)`, the action wants the tag
                Message::AmdNpuChosen(
                    label
                        .strip_suffix(" (not downloaded)")
                        .unwrap_or(&label)
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
            });
            amd_actions = amd_actions.push(picker);
        }
        let amd_button = |label: &'static str, action: &'static str| {
            button(text(label).size(13).color(t.fg_soft))
                .on_press(Message::AmdNpuAction(action))
                .padding([8, 14])
                .style(crate::app::starter_style(t))
        };
        if amd.start.unwrap_or(false) {
            amd_actions = amd_actions.push(amd_button("Run on the NPU", "start"));
        }
        if amd.stop {
            amd_actions = amd_actions.push(amd_button("Stop", "stop"));
        }
        if amd.pull {
            amd_actions = amd_actions.push(amd_button("Download", "pull"));
        }
        if amd.cancel_pull {
            amd_actions = amd_actions.push(amd_button("Cancel the download", "cancel_pull"));
        }
        if !amd.models.is_empty() || amd.start.unwrap_or(false) || amd.stop || amd.pull || amd.cancel_pull {
            amd_body = amd_body.push(amd_actions);
        }
        if !amd.pull_text.is_empty() {
            amd_body = amd_body.push(text(amd.pull_text.clone()).size(13).color(t.fg_soft));
        }
        if let Some(error) = &amd.error {
            amd_body = amd_body.push(text(error.clone()).size(13).color(t.danger));
        }
        amd_body = amd_body.push(text(String::from(crate::amdnpu::AMD_NPU_NOTE)).size(12).color(t.muted));
        body = body.push(
            container(amd_body).width(Length::Fill).padding(14).style(theme::card(t, theme::R_LG, false)),
        );
    }

    // The table of the scheduler (`hwSlotsTableWrap` of `panels.js`): which of the slots is
    // working on an answer right now.
    let mut slots_card = column![].spacing(8);
    slots_card = slots_card.push(text("Chats answered at the same time").size(15).font(bold()));
    if !view.slots.is_empty() {
        let mut head = row![].spacing(12);
        head = head.push(cell(String::from("Slot"), 80.0, t.muted));
        head = head.push(cell(String::from("State"), 200.0, t.muted));
        head = head.push(cell(String::from("Max Context"), 140.0, t.muted));
        head = head.push(cell(String::from("Task"), 140.0, t.muted));
        slots_card = slots_card.push(head);
        for slot in view.slots {
            let mut line = row![].spacing(12);
            line = line.push(cell(format!("#{}", slot.id), 80.0, t.fg));
            line = line.push(cell(slot.state.clone(), 200.0, t.fg));
            line = line.push(cell(format!("{} tok", slot.n_ctx), 140.0, t.fg));
            line = line.push(cell(
                if slot.running { String::from("Running") } else { String::from("Idle") },
                140.0,
                if slot.running { t.ok } else { t.muted },
            ));
            slots_card = slots_card.push(line);
        }
    }
    body = body.push(
        container(slots_card).width(Length::Fill).padding(14).style(theme::card(t, theme::R_LG, false)),
    );

    // The report of *Check the processor* (`simdSelftestCard` of `listeners-panels.js`)
    if let Some(report) = view.selftest {
        let mut check = column![].spacing(8);
        check = check.push(text("Processor check").size(15).font(bold()));
        check = check.push(text(report.title.clone()).size(13).color(t.fg_soft));
        let mut head = row![].spacing(12);
        head = head.push(cell(String::from("Feature level"), 170.0, t.muted));
        head = head.push(cell(String::from("Checks"), 70.0, t.muted));
        head = head.push(cell(String::from("Passed"), 70.0, t.muted));
        head = head.push(cell(String::from("Failed"), 70.0, t.muted));
        head = head.push(cell(String::from("Result"), 70.0, t.muted));
        check = check.push(head);
        for row_data in &report.rows {
            let mut line = row![].spacing(12);
            line = line.push(cell(row_data.tier.clone(), 170.0, t.fg));
            line = line.push(cell(row_data.checks.to_string(), 70.0, t.fg));
            line = line.push(cell(row_data.passed.to_string(), 70.0, t.fg));
            line = line.push(cell(row_data.failed.to_string(), 70.0, t.fg));
            line = line.push(cell(
                if row_data.ok { String::from("PASS") } else { String::from("FAIL") },
                70.0,
                if row_data.ok { t.ok } else { t.danger },
            ));
            check = check.push(line);
        }
        check = check.push(text(report.note.clone()).size(12).color(t.muted));
        body = body.push(
            container(check).width(Length::Fill).padding(14).style(theme::card(t, theme::R_LG, false)),
        );
    }

    body.into()
}

/// The appearance card of the Studio (the `Appearance` card of `shell.html`): the theme, the accent
/// people pick, the base size of the text and the two switches under it.
fn appearance_card<'a>(t: Tokens, look: LookView<'a>) -> Element<'a, Message> {
    /// The style every select of the page wears.
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

    let mut body = column![].spacing(16);

    // Theme: what the page follows when it does not decide itself
    const THEME_LABELS: [&str; 3] = ["Same as the system", "Dark", "Light"];
    let theme_now = match look.theme {
        "dark" => "Dark",
        "light" => "Light",
        _ => "Same as the system",
    };
    let theme_control = pick_list(THEME_LABELS, Some(theme_now), |label: &'static str| {
        Message::LookTheme(match label {
            "Dark" => String::from("dark"),
            "Light" => String::from("light"),
            _ => String::new(),
        })
    })
    .padding([6, 10])
    .style(select_style(t));
    body = body.push(setting(t, "Theme", "By default the page follows the light or dark setting of your system.", theme_control.into()));

    // Accent color: the eight of the Studio, the chosen one ringed like `.accent-swatch.active`
    let mut swatches = row![].spacing(8).align_y(Alignment::Center);
    for accent in theme::ACCENTS {
        let chosen = accent.value == look.accent;
        let colour = accent.thumb;
        let label = accent.name;
        let text_colour = theme::on_accent(colour);
        swatches = swatches.push(
            button(text(label).size(11).color(text_colour))
                .on_press(Message::LookAccent(accent.value.to_string()))
                .padding([7, 10])
                .style(move |_, status| button::Style {
                    background: Some(iced::Background::Color(match status {
                        button::Status::Hovered | button::Status::Pressed => iced::Color { a: 0.85, ..colour },
                        _ => colour,
                    })),
                    text_color: text_colour,
                    border: iced::Border {
                        radius: 999.0.into(),
                        width: if chosen { 2.0 } else { 1.0 },
                        color: if chosen { t.primary } else { secondary_ring(t) },
                    },
                    ..Default::default()
                }),
        );
    }
    body = body.push(setting(t, "Accent color", "Colors the sidebar and the controls.", swatches.into()));

    // Font size: the base the whole interface draws with
    let size_now = look.font_size;
    const SIZE_LABELS: [&str; 4] = ["Small (14px)", "Medium (15px)", "Large (16px)", "Extra Large (18px)"];
    let size_label = match size_now {
        14 => SIZE_LABELS[0],
        16 => SIZE_LABELS[2],
        18 => SIZE_LABELS[3],
        _ => SIZE_LABELS[1],
    };
    let size_control = pick_list(SIZE_LABELS, Some(size_label), |label: &'static str| {
        Message::LookFontSize(match label {
            "Small (14px)" => 14,
            "Large (16px)" => 16,
            "Extra Large (18px)" => 18,
            _ => 15,
        })
    })
    .padding([6, 10])
    .style(select_style(t));
    body = body.push(setting(t, "Font size", "Base size of the text in the whole interface.", size_control.into()));

    // Tint your messages: the bubble of the accent, or the Studio's secondary
    body = body.push(setting(
        t,
        "Tint your messages",
        "Tints your messages in the accent color.",
        segmented(t, vec![("On", look.tint, Message::LookTint(true)), ("Off", !look.tint, Message::LookTint(false))]),
    ));

    // Show generation speed: the line under every answer
    body = body.push(setting(
        t,
        "Show generation speed",
        "Shows tokens per second and the time to the first token below every answer.",
        segmented(t, vec![("On", look.speed, Message::LookSpeed(true)), ("Off", !look.speed, Message::LookSpeed(false))]),
    ));

    body.into()
}

/// The ring around the swatch that is not chosen (`--secondary` of the Studio).
fn secondary_ring(t: Tokens) -> iced::Color {
    theme::secondary(t.dark)
}

/// The usage card of the Studio (`usage.js`): what has been counted, what it costs with the
/// prices entered, and the rows themselves.
fn usage_card<'a>(t: Tokens, view: UsageView<'a>) -> Element<'a, Message> {
    let mut body = column![].spacing(12);
    body = body.push(
        text("Tokens are counted from the usage block of every answer. A local model costs nothing; for a model behind a provider, add its price below and the cost appears here and in /cost.")
            .size(13)
            .color(t.muted),
    );
    body = body.push(text(crate::usage::totals_text(view.data, view.prices)).size(13));
    // the table: model, answers, input, output, cost, and the row of everything together
    if view.data.by_model.is_empty() {
        body = body.push(
            text("Nothing counted yet in this browser. Ask something in Chat or run the Code agent.")
                .size(13)
                .color(t.muted),
        );
    } else {
        let mut table = column![].spacing(4);
        table = table.push(
            row![
                text("Model").size(12).color(t.muted).width(240),
                text("Answers").size(12).color(t.muted).width(80),
                text("Input").size(12).color(t.muted).width(90),
                text("Output").size(12).color(t.muted).width(90),
                text("Cost").size(12).color(t.muted).width(90),
            ]
            .spacing(8),
        );
        for (model, entry) in crate::usage::rows_sorted(view.data) {
            let cost = crate::usage::cost_of(model, entry, view.prices);
            table = table.push(
                row![
                    text(model.clone()).size(13).width(240),
                    text(entry.calls.to_string()).size(13).width(80),
                    text(crate::usage::fmt_tokens(entry.prompt)).size(13).width(90),
                    text(crate::usage::fmt_tokens(entry.completion)).size(13).width(90),
                    text(crate::usage::fmt_money(cost)).size(13).width(90),
                ]
                .spacing(8),
            );
        }
        let all = crate::usage::totals(view.data, view.prices);
        table = table.push(
            row![
                text("All").size(13).font(bold()).width(240),
                text(all.calls.to_string()).size(13).font(bold()).width(80),
                text(crate::usage::fmt_tokens(all.prompt)).size(13).font(bold()).width(90),
                text(crate::usage::fmt_tokens(all.completion)).size(13).font(bold()).width(90),
                text(crate::usage::fmt_money(if all.priced > 0 { Some(all.cost) } else { None }))
                    .size(13)
                    .font(bold())
                    .width(90),
            ]
            .spacing(8),
        );
        body = body.push(table);
    }
    body = body.push(
        button(text("Reset counters").size(13).color(t.fg_soft))
            .on_press(Message::UsageReset)
            .padding([8, 14])
            .style(crate::app::starter_style(t)),
    );
    // the prices themselves
    body = body.push(
        text("Prices (dollars per million tokens)").size(15).font(bold()),
    ).push(
        text("A pattern is matched against the model name; the longest match wins, * matches any tail (for example gpt-6*). Models without a row show tokens only.")
            .size(13)
            .color(t.muted),
    );
    if view.prices.is_empty() {
        body = body.push(text("No price rows. Local models need none.").size(13).color(t.muted));
    }
    for (i, row) in view.prices.iter().enumerate() {
        let mut fields = row![].spacing(6).align_y(Alignment::Center);
        fields = fields.push(price_input(t, "model pattern, e.g. gpt-6*", &row.pattern, i, "pattern", 200.0));
        fields = fields.push(price_input(t, "0", &row.input_text, i, "input", 110.0));
        fields = fields.push(price_input(t, "0", &row.output_text, i, "output", 110.0));
        fields = fields.push(price_input(t, "note (optional)", &row.note, i, "note", 140.0));
        fields = fields.push(
            button(text("Remove").size(13).color(t.fg_soft))
                .on_press(Message::PriceRemove(i))
                .padding([6, 12])
                .style(crate::app::starter_style(t)),
        );
        body = body.push(fields);
    }
    let mut actions = row![].spacing(8).align_y(Alignment::Center);
    actions = actions.push(
        button(text("+ Add price").size(13).color(t.fg_soft))
            .on_press(Message::PriceAdd)
            .padding([8, 14])
            .style(crate::app::starter_style(t)),
    ).push(
        button(text("Save prices").size(13).font(bold()))
            .on_press(Message::UsageSave)
            .padding([8, 14])
            .style(crate::app::primary_style(t)),
    );
    if let Some(status) = view.status {
        actions = actions.push(text(status).size(12).color(t.muted));
    }
    body = body.push(actions);
    body.into()
}

/// One field of a price row (the `data-f` inputs of the Studio).
fn price_input<'a>(
    t: Tokens,
    placeholder: &'a str,
    value: &'a str,
    index: usize,
    field: &'static str,
    width: f32,
) -> Element<'a, Message> {
    text_input(placeholder, value)
        .on_input(move |v| Message::PriceField(index, field, v))
        .size(14)
        .padding([6, 10])
        .width(width)
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

#[allow(clippy::too_many_arguments)] // a page draws from ten pieces of state
pub fn page<'a>(
    t: Tokens,
    prefs: &'a Prefs,
    system_editor: &'a Content,
    spec_editor: &'a Content,
    max_tokens_text: &'a str,
    sampling: &'a SamplingTexts,
    engine: Element<'a, Message>,
    personal: PersonalView<'a>,
    voice: VoiceView<'a>,
    usage: UsageView<'a>,
    look: LookView<'a>,
    hardware: HardwareView<'a>,
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

    // ---- the Voice and alerts pane of voice.js: dictation, reading aloud, alerts ----
    let v = &prefs.voice;
    let dictation_note = "Dictation is not available here: this program, or the web view of the desktop app on this system, has no speech recognition, so the microphone button is hidden. Chrome, Edge and Safari have it.";
    let dictation = column![
        setting(
            t,
            "Microphone",
            dictation_note,
            Space::new().width(180.0).into(),
        ),
        setting(
            t,
            "Language",
            "Used for dictation, and for reading aloud when the voice does not set one.",
            pick_list(
                crate::speech::LANGUAGES
                    .iter()
                    .map(|(tag, name)| LangChoice(tag, name))
                    .collect::<Vec<LangChoice>>(),
                Some(
                    crate::speech::LANGUAGES
                        .iter()
                        .find(|(tag, _)| *tag == v.lang)
                        .map(|(tag, name)| LangChoice(tag, name))
                        .unwrap_or(LangChoice("", "Browser language")),
                ),
                |c: LangChoice| Message::VoiceLang(c.0.to_string()),
            )
            .width(180)
            .into(),
        ),
    ]
    .spacing(14);

    let voice_note = if voice.voices.is_empty() {
        "This computer cannot read aloud."
    } else {
        "The voices come from your system."
    };
    let mut voice_items: Vec<VoiceChoice> = vec![VoiceChoice(String::new())];
    voice_items.extend(voice.voices.iter().cloned().map(VoiceChoice));
    let reading = column![
        setting(
            t,
            "Voice",
            voice_note,
            pick_list(
                voice_items,
                Some(VoiceChoice(v.voice.clone())),
                |c: VoiceChoice| Message::VoiceVoice(c.0),
            )
            .width(180)
            .into(),
        ),
        setting(
            t,
            "Speed",
            "How fast the answer is read.",
            value_slider(t, 0.5..=2.0, v.rate, 0.1, Message::VoiceRate),
        ),
        setting(
            t,
            "Pitch",
            "How high the voice speaks.",
            value_slider(t, 0.5..=2.0, v.pitch, 0.1, Message::VoicePitch),
        ),
        setting(
            t,
            "Read answers automatically",
            "Reads each finished answer aloud. The speaker button under an answer reads one answer.",
            segmented(
                t,
                vec![
                    ("On", v.auto, Message::VoiceAuto(true)),
                    ("Off", !v.auto, Message::VoiceAuto(false)),
                ],
            ),
        ),
        setting(
            t,
            "Try it",
            "Code and Markdown marks are left out when reading.",
            button(text("Read a sample").size(13))
                .on_press(Message::VoiceTest)
                .padding([5, 12])
                .style(theme::secondary_button(t))
                .into(),
        ),
    ]
    .spacing(14);

    let notify_status = if voice.can_notify {
        String::new()
    } else {
        String::from("This computer has no system notifications.")
    };
    let mut alerts = column![
        setting(
            t,
            "System notification",
            "When a long answer (over 20 seconds) or an agent run finishes while this window is not in front.",
            segmented(
                t,
                vec![
                    ("On", v.notify, Message::VoiceNotify(true)),
                    ("Off", !v.notify, Message::VoiceNotify(false)),
                ],
            ),
        ),
        setting(
            t,
            "Sound",
            "A short tone at the same moment.",
            row![
                segmented(
                    t,
                    vec![
                        ("On", v.sound, Message::VoiceSound(true)),
                        ("Off", !v.sound, Message::VoiceSound(false)),
                    ],
                ),
                button(text("Play").size(13))
                    .on_press(Message::VoiceChimeTest)
                    .padding([5, 12])
                    .style(theme::ghost_button(t)),
            ]
            .spacing(8)
            .into(),
        ),
    ]
    .spacing(14);
    if !notify_status.is_empty() {
        alerts = alerts.push(text(notify_status).size(12).color(t.muted));
    }

    let content = column![
        container(text("Settings").size(26).font(bold())).padding(Padding { top: 14.0, right: 0.0, bottom: 4.0, left: 2.0 }),
        card(t, "General", general.into()),
        card(t, "Answers", answers.into()),
        card(t, "Personalization", pers.into()),
        engine,
        // the Voice and alerts pane sits last so the cards before it keep their place (the pane
        // of voice.js has its own place in the Studio's menu; the drawing of the settings takes
        // its own height for these)
        card(t, "Appearance", appearance_card(t, look)),
        hardware_card(t, hardware),
        card(t, "Usage and cost", usage_card(t, usage)),
        card(t, "Dictation", dictation.into()),
        card(t, "Reading aloud", reading.into()),
        card(t, "Alerts", alerts.into()),
    ]
    .spacing(14);
    let inner = container(content).max_width(780).width(Length::Fill).padding(Padding { top: 8.0, right: 24.0, bottom: 32.0, left: 24.0 });
    scrollable(container(inner).center_x(Length::Fill).width(Length::Fill)).height(Length::Fill).into()
}
