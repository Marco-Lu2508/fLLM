//! The system prompt of the chat: the default prompt is built from facts (time, language, model family and size, the tools that are offered),
//! then your own text is applied to it (replace or append), and the chat's assistant and project add theirs. Port of the Studio's `prompts.js`.
//!
//! The prompt describes only what the engine really provides: a tool is mentioned only when it is offered.

use chrono::{DateTime, Datelike, FixedOffset, Local, TimeZone, Timelike};

/// Limits that decide how long the prompt is.
pub const CUSTOM_MAX: usize = 6000;
const COMPACT_CONTEXT: u64 = 12288;
const COMPACT_PARAMS: f64 = 3.5e9;
const TINY_CONTEXT: u64 = 2048;
const SLOW_PREFILL: f64 = 300.0;

/// What the app knows about the loaded model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelDetail {
    pub name: String,
    pub arch: String,
    pub parameters: f64,
    pub chat_template: String,
    /// The model can think (a switch or an effort): the thinking chip is shown.
    pub thinks: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    GptOss,
    SmolLm,
    DeepSeek,
    Qwen,
    Gemma,
    Llama,
    Mistral,
    Phi,
    Generic,
}

/// The model family, from the name, the architecture and the chat template.
pub fn family(m: &ModelDetail) -> Family {
    let name = format!("{} {}", m.name, m.arch).to_lowercase();
    let tpl = &m.chat_template;
    let has = |words: &[&str]| words.iter().any(|w| name.contains(w));
    if has(&["gpt-oss", "gpt_oss", "gpt oss", "gptoss"]) || tpl.contains("<|channel|>") {
        return Family::GptOss;
    }
    if has(&["smollm"]) {
        return Family::SmolLm;
    }
    if has(&["deepseek"]) || (tpl.contains("<｜User｜>") || tpl.contains("<|User|>")) {
        return Family::DeepSeek;
    }
    if has(&["qwen", "qwq"]) {
        return Family::Qwen;
    }
    if has(&["gemma"]) || tpl.contains("<start_of_turn>") {
        return Family::Gemma;
    }
    if has(&["llama"]) {
        return Family::Llama;
    }
    if has(&["mistral", "mixtral", "ministral", "devstral", "magistral"]) || tpl.contains("[INST]") {
        return Family::Mistral;
    }
    if has(&["phi-", "phi_", "phi "]) || name.split_whitespace().any(|w| w == "phi") || name.contains("phi3") || name.contains("phi4") {
        return Family::Phi;
    }
    if tpl.contains("<|im_start|>") {
        return Family::Qwen;
    }
    Family::Generic
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Size {
    /// A context so small that a full prompt would fill it.
    Tiny,
    Compact,
    Full,
}

/// How the prompt length is chosen: always the full prompt, the short one, or by the speed at which the model read its last prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Length {
    #[default]
    Full,
    Short,
    Auto,
}

pub fn size(max_context: u64, parameters: f64, length: Length, prompt_per_second: f64) -> Size {
    if max_context < TINY_CONTEXT {
        return Size::Tiny;
    }
    if max_context < COMPACT_CONTEXT || (parameters > 0.0 && parameters < COMPACT_PARAMS) {
        return Size::Compact;
    }
    match length {
        Length::Short => Size::Compact,
        Length::Auto if prompt_per_second > 0.0 && prompt_per_second < SLOW_PREFILL => Size::Compact,
        _ => Size::Full,
    }
}

fn family_note(f: Family) -> &'static str {
    match f {
        Family::Qwen => "Keep any reasoning brief and do not repeat it in the final answer. Write a tool call only in the format of the tool interface, never as ordinary text.",
        Family::Llama => "Do not apologize and do not add disclaimers. Get to the point.",
        Family::Mistral => "Answer directly without a long introduction. Write a tool call only in the format of the tool interface.",
        Family::Gemma => "Follow these instructions for the whole conversation, also when the user message is short.",
        Family::GptOss => "Keep the final answer short; your reasoning stays out of it.",
        Family::DeepSeek => "Keep your reasoning short. The final answer is in the language of the user, even when you reasoned in another one.",
        Family::Phi => "Answer in the language of the user, even when you reason in English. Stay on topic and never write the next message of the user.",
        Family::SmolLm => "You are a very small model: keep answers short and simple, say \"I am not sure\" instead of guessing, and never write the next message of the user.",
        Family::Generic => "",
    }
}

/// The name of a language from its tag (`de-DE` is German).
pub fn language_name(locale: &str) -> String {
    let tag = locale.trim();
    let base = tag.split(['-', '_']).next().unwrap_or("en").to_lowercase();
    let name = match base.as_str() {
        "en" => "English",
        "de" => "German",
        "fr" => "French",
        "es" => "Spanish",
        "it" => "Italian",
        "pt" => "Portuguese",
        "nl" => "Dutch",
        "pl" => "Polish",
        "cs" => "Czech",
        "sv" => "Swedish",
        "da" => "Danish",
        "nb" | "no" => "Norwegian",
        "fi" => "Finnish",
        "tr" => "Turkish",
        "ru" => "Russian",
        "uk" => "Ukrainian",
        "ar" => "Arabic",
        "he" => "Hebrew",
        "hi" => "Hindi",
        "zh" => "Chinese",
        "ja" => "Japanese",
        "ko" => "Korean",
        "el" => "Greek",
        "hu" => "Hungarian",
        "ro" => "Romanian",
        "id" => "Indonesian",
        "vi" => "Vietnamese",
        "th" => "Thai",
        _ => return if tag.is_empty() { "English".into() } else { tag.to_string() },
    };
    name.to_string()
}

const WEEKDAYS: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/// "Monday, 26 October 2026"
pub fn date_text<T: TimeZone>(t: &DateTime<T>) -> String {
    format!("{}, {} {} {}", WEEKDAYS[t.weekday().num_days_from_monday() as usize], t.day(), MONTHS[t.month0() as usize], t.year())
}

/// "Monday, 26 October 2026, 14:05 (UTC+02:00)"
pub fn now_text(t: &DateTime<FixedOffset>) -> String {
    let off = t.offset().local_minus_utc() / 60;
    format!("{}, {:02}:{:02} (UTC{}{:02}:{:02})", date_text(t), t.hour(), t.minute(), if off < 0 { '-' } else { '+' }, off.abs() / 60, off.abs() % 60)
}

/// The clock of a chat is frozen for 30 minutes: a prompt that changes every minute would make the server drop its cached start of the
/// conversation on every message.
pub fn frozen_clock(held: Option<i64>, now: i64) -> i64 {
    match held {
        Some(h) if h <= now && now - h <= 30 * 60 => h,
        _ => now,
    }
}

pub fn local_time(seconds: i64) -> DateTime<FixedOffset> {
    let local = Local.timestamp_opt(seconds, 0).single().unwrap_or_else(|| Local.timestamp_opt(0, 0).unwrap());
    local.with_timezone(local.offset())
}

/// Which tools the chat offers; the prompt mentions exactly these.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tools {
    pub web: bool,
    pub memory: bool,
    pub chats: bool,
}

const WEB: &str = "Web tools (web_search, web_fetch): use them only for current or changing facts (news, prices, versions, events after your training), for specific facts you do not know or are unsure about, or when the user asks you to look something up or to open a link. For everything else (explanations, writing, code, math, advice, small talk) answer directly without searching. When you used the web, cite the sources as [n] with the numbers from the results and never invent a source. If a search fails, say so.";
const WEB_COMPACT: &str = "Web tools: search only for current or unknown facts, otherwise answer directly. Cite sources as [n]; never invent sources.";
const MEMORY: &str = "Memory tool (memory): save a short fact when you learn something durable about the user (name, preferences, projects, constraints) or when they ask you to remember something. Never save trivia, one-off requests or secrets such as passwords, keys or card numbers. Use update or remove when a fact changed or the user wants it forgotten. Mention a saved fact only briefly, if at all.";
const MEMORY_COMPACT: &str = "Memory tool: save durable facts about the user (name, preferences, projects); never trivia or secrets.";
const CHATS: &str = "Chat search tool (search_chats): use it when the user refers to an earlier conversation that you cannot see (\"as we discussed\", \"last time\").";
const CHATS_COMPACT: &str = "search_chats finds earlier conversations the user refers to.";
const CALL: &str = "Call tools with valid JSON arguments. After a tool result, answer the user; do not repeat the same call.";
const NONE: &str = "You have no tools in this chat: you cannot browse the web, open links, run code or read files except what the user shares here. Say so instead of pretending.";

/// The response styles of the Studio (artifacts.js): a chip in the composer, and an extra
/// instruction at the end of the system prompt of the chat.
pub mod style {
    /// One style: what the chip says and what it adds to the prompt.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Style {
        pub id: &'static str,
        pub label: &'static str,
        pub prompt: &'static str,
    }

    /// The styles of the Studio (`STYLES` in artifacts.js).
    pub const STYLES: [Style; 5] = [
        Style { id: "default", label: "Default", prompt: "" },
        Style { id: "concise", label: "Concise", prompt: "Keep answers short and to the point. Skip preamble and repetition." },
        Style { id: "explanatory", label: "Explanatory", prompt: "Explain your reasoning step by step with clear examples, and define terms that may be unfamiliar." },
        Style { id: "formal", label: "Formal", prompt: "Use a formal, professional tone and precise wording." },
        Style { id: "casual", label: "Casual", prompt: "Use a relaxed, friendly, conversational tone." },
    ];

    /// What a chat's stored style means: empty or unknown falls back to the default (`styleOf`).
    pub fn style_of(style: &str) -> &'static str {
        STYLES.iter().find(|s| s.id == style).map(|s| s.id).unwrap_or("default")
    }

    /// The label of a style: the chip says "Style: Concise", the default keeps just "Style".
    pub fn label_of(style: &str) -> &'static str {
        STYLES.iter().find(|s| s.id == style_of(style)).map(|s| s.label).unwrap_or("Default")
    }

    /// What the style adds to the prompt; nothing for the default (`withStylePrompt`).
    pub fn instruction(style: &str) -> Option<&'static str> {
        let s = STYLES.iter().find(|s| s.id == style_of(style))?;
        if s.prompt.is_empty() { None } else { Some(s.prompt) }
    }

    /// The base of a request with the instruction of the style behind it (`withStylePrompt`).
    pub fn with_style_prompt(base: &str, style: &str) -> String {
        let Some(add) = instruction(style) else { return base.to_string() };
        if base.is_empty() {
            add.to_string()
        } else {
            format!("{base}\n\n{add}")
        }
    }
}

/// What the default prompt is built from.
#[derive(Debug, Clone)]
pub struct Facts {
    pub size: Size,
    pub family: Family,
    pub now: DateTime<FixedOffset>,
    pub locale: String,
    /// The model runs on this computer (not with a provider).
    pub local: bool,
    pub model_name: String,
    pub thinking: bool,
    pub tools: Tools,
}

/// The default system prompt.
pub fn build(o: &Facts) -> String {
    let compact = o.size == Size::Compact;
    let lang = language_name(&o.locale);
    let running = if o.local { " running locally on the user's computer" } else { "" };
    if o.size == Size::Tiny {
        return format!("You are a helpful assistant{running}. The date is {}. Answer in the language of the user ({lang} if unclear), be brief, and say so when you are unsure.", date_text(&o.now));
    }
    let mut lines: Vec<String> = Vec::new();
    lines.push(if compact {
        format!("You are a helpful assistant{running}. Today is {}.", now_text(&o.now))
    } else {
        let based = if o.local && !o.model_name.is_empty() { format!(", based on the language model {}", o.model_name.chars().take(60).collect::<String>()) } else { String::new() };
        let place = if o.local { format!(" running locally on the user's own computer{based}") } else { String::new() };
        format!("You are a helpful, honest assistant{place}. The current date and time is {}.", now_text(&o.now))
    });
    lines.push(if compact {
        format!("Answer in the language of the user ({lang} if unclear). Be concise. Use Markdown only where it helps.")
    } else {
        format!("Answer in the language the user writes in ({lang} if it is unclear). Be concise by default: give the answer first, then only the detail that is needed. Go into depth when the user asks for it.")
    });
    if !compact {
        lines.push("Format with Markdown only where it helps: short lists for steps or options, fenced code blocks with a language for code, tables only for real comparisons. Write plain paragraphs otherwise. Write math as plain text or LaTeX.".into());
    }
    lines.push(if compact {
        "If you are not sure, say so. Never invent facts, quotes, links or sources.".into()
    } else {
        "Be honest about what you do not know: say when you are unsure, and never invent facts, quotes, numbers, links or sources. Your knowledge ends at your training date, so you may not know recent events. If a question is ambiguous, ask a short question before you answer.".into()
    });
    if o.local {
        lines.push(if compact {
            "If asked, you run locally on the user's computer.".into()
        } else {
            "If you are asked where you run: you run locally on the user's own computer, and the conversation stays on it unless the user chose an external provider.".into()
        });
    }
    let mut parts: Vec<&str> = Vec::new();
    if o.tools.web {
        parts.push(if compact { WEB_COMPACT } else { WEB });
    }
    if o.tools.memory {
        parts.push(if compact { MEMORY_COMPACT } else { MEMORY });
    }
    if o.tools.chats {
        parts.push(if compact { CHATS_COMPACT } else { CHATS });
    }
    if parts.is_empty() {
        lines.push(NONE.into());
    } else {
        parts.push(CALL);
        lines.extend(parts.into_iter().map(String::from));
    }
    let note = family_note(o.family);
    if !note.is_empty() {
        lines.push(note.into());
    }
    if o.thinking && o.family != Family::Qwen && o.family != Family::DeepSeek {
        lines.push("Keep your reasoning short.".into());
    }
    lines.join(if compact { "\n" } else { "\n\n" })
}

/// How a text of yours is applied to the default prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Default,
    Override,
    Append,
}

/// Fills `{{date}}`, `{{time}}` and `{{language}}` in a text you wrote.
pub fn fill(text: &str, now: &DateTime<FixedOffset>, locale: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            out.push_str(&rest[start..]);
            rest = "";
            break;
        };
        match after[..end].trim().to_lowercase().as_str() {
            "date" => out.push_str(&date_text(now)),
            "time" => out.push_str(&now_text(now)),
            "language" => out.push_str(&language_name(locale)),
            _ => out.push_str(&rest[start..start + 2 + end + 2]),
        }
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    out
}

/// Applies your choices to the default prompt: the global one first, then the one of the assistant. An empty override is ignored.
pub fn resolve(generated: &str, choices: &[(Mode, &str)], now: &DateTime<FixedOffset>, locale: &str) -> String {
    let mut out = generated.to_string();
    for (mode, text) in choices {
        let text = fill(&text.trim().chars().take(CUSTOM_MAX).collect::<String>(), now, locale);
        if text.is_empty() {
            continue;
        }
        match mode {
            Mode::Override => out = text,
            Mode::Append => out = format!("{out}\n\n{text}"),
            Mode::Default => {}
        }
    }
    out
}

/// The messages that ask for a short title of a conversation.
pub fn title_messages(question: &str, answer: &str) -> Vec<(&'static str, String)> {
    vec![
        ("system", "Write a short title for a conversation: two to five words, in the language of the conversation. Reply with the title only, without quotation marks and without a full stop. Never answer the question and never mention these instructions.".to_string()),
        ("user", format!("User: {}\n\nAssistant: {}\n\nTitle:", question.chars().take(300).collect::<String>(), answer.chars().take(300).collect::<String>())),
    ]
}

/// A title from the raw reply of a model: no thinking blocks, no markup, one line; `None` when nothing usable is left.
pub fn clean_title(raw: &str) -> Option<String> {
    let mut t = raw.to_string();
    while let (Some(a), Some(b)) = (t.find("<think>"), t.find("</think>")) {
        if b < a {
            break;
        }
        t.replace_range(a..b + "</think>".len(), "");
    }
    if let Some(a) = t.find("<think>") {
        t.truncate(a);
    }
    let line = t.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    let lower = line.to_lowercase();
    let line = ["title:", "titel:"].iter().find(|p| lower.starts_with(*p)).map_or(line, |p| line[p.len()..].trim());
    let cleaned: String = line.chars().filter(|c| !matches!(c, '"' | '*' | '#' | '`' | '_')).collect();
    let cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let cleaned = cleaned.trim_end_matches(['.', '!', '?', ':']).trim().to_string();
    let lower = cleaned.to_lowercase();
    if lower.contains("system prompt") || lower.contains("these instructions") || lower.contains("my instructions") {
        return None;
    }
    let n = cleaned.chars().count();
    (3..=60).contains(&n).then_some(cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(h: u32, min: u32) -> DateTime<FixedOffset> {
        FixedOffset::east_opt(2 * 3600).unwrap().with_ymd_and_hms(2026, 10, 26, h, min, 0).unwrap()
    }

    fn facts(size: Size, family: Family, tools: Tools) -> Facts {
        Facts { size, family, now: at(14, 5), locale: "de-DE".into(), local: true, model_name: "Qwen3.5-9B".into(), thinking: false, tools }
    }

    fn model(name: &str, arch: &str, tpl: &str) -> ModelDetail {
        ModelDetail { name: name.into(), arch: arch.into(), chat_template: tpl.into(), ..Default::default() }
    }

    #[test]
    fn the_family_comes_from_the_name_the_architecture_or_the_template() {
        assert_eq!(family(&model("Qwen3.5-9B-IQ3_XXS", "qwen35", "")), Family::Qwen);
        assert_eq!(family(&model("unknown", "x", "<|im_start|>user")), Family::Qwen, "ChatML is a Qwen-style template");
        assert_eq!(family(&model("gpt-oss-20b", "", "")), Family::GptOss);
        assert_eq!(family(&model("x", "", "...<|channel|>...")), Family::GptOss);
        assert_eq!(family(&model("DeepSeek-R1-Distill", "", "")), Family::DeepSeek);
        assert_eq!(family(&model("gemma-3-4b", "", "")), Family::Gemma);
        assert_eq!(family(&model("Llama-3.2-3B", "", "")), Family::Llama);
        assert_eq!(family(&model("Mistral-7B", "", "")), Family::Mistral);
        assert_eq!(family(&model("x", "", "[INST] hi [/INST]")), Family::Mistral);
        assert_eq!(family(&model("Phi-4-mini", "", "")), Family::Phi);
        assert_eq!(family(&model("SmolLM2-360M", "", "")), Family::SmolLm);
        assert_eq!(family(&model("tiny-llm", "", "")), Family::Generic, "llm is not llama");
        assert_eq!(family(&model("", "", "")), Family::Generic);
    }

    #[test]
    fn the_size_follows_the_context_the_parameters_and_your_choice() {
        assert_eq!(size(1024, 9e9, Length::Full, 0.0), Size::Tiny);
        assert_eq!(size(8192, 9e9, Length::Full, 0.0), Size::Compact, "a small context");
        assert_eq!(size(32768, 1e9, Length::Full, 0.0), Size::Compact, "a small model");
        assert_eq!(size(32768, 9e9, Length::Full, 0.0), Size::Full);
        assert_eq!(size(32768, 9e9, Length::Short, 0.0), Size::Compact);
        assert_eq!(size(32768, 9e9, Length::Auto, 66.0), Size::Compact, "a slow reader gets the short prompt");
        assert_eq!(size(32768, 9e9, Length::Auto, 0.0), Size::Full, "nothing measured yet: the full prompt");
        assert_eq!(size(32768, 9e9, Length::Auto, 900.0), Size::Full);
        assert_eq!(size(32768, 0.0, Length::Full, 0.0), Size::Full, "unknown size is not small");
    }

    #[test]
    fn the_clock_reads_like_the_studio_and_is_frozen_for_half_an_hour() {
        assert_eq!(now_text(&at(14, 5)), "Monday, 26 October 2026, 14:05 (UTC+02:00)");
        assert_eq!(date_text(&at(0, 0)), "Monday, 26 October 2026");
        let west = FixedOffset::west_opt(5 * 3600 + 1800).unwrap().with_ymd_and_hms(2026, 1, 2, 3, 4, 0).unwrap();
        assert_eq!(now_text(&west), "Friday, 2 January 2026, 03:04 (UTC-05:30)");
        assert_eq!(frozen_clock(None, 1000), 1000);
        assert_eq!(frozen_clock(Some(1000), 1000 + 29 * 60), 1000, "inside the half hour the old time is kept");
        assert_eq!(frozen_clock(Some(1000), 1000 + 31 * 60), 1000 + 31 * 60);
        assert_eq!(frozen_clock(Some(5000), 1000), 1000, "a clock from the future is not trusted");
    }

    #[test]
    fn the_full_prompt_says_what_the_engine_really_offers() {
        let none = build(&facts(Size::Full, Family::Qwen, Tools::default()));
        assert!(none.starts_with("You are a helpful, honest assistant running locally on the user's own computer, based on the language model Qwen3.5-9B. The current date and time is Monday, 26 October 2026, 14:05 (UTC+02:00)."), "{none}");
        assert!(none.contains("(German if it is unclear)"));
        assert!(none.contains("You have no tools in this chat"), "no tool is offered: it says so");
        assert!(!none.contains("web_search") && !none.contains("Call tools with valid JSON"));
        assert!(none.contains("Keep any reasoning brief"), "the note of the family");
        let tools = build(&facts(Size::Full, Family::Generic, Tools { web: true, memory: true, chats: true }));
        for word in ["web_search", "Memory tool (memory)", "search_chats", "Call tools with valid JSON arguments"] {
            assert!(tools.contains(word), "{word}");
        }
        assert!(!tools.contains("You have no tools"));
        assert!(tools.contains("\n\n"), "paragraphs in the full prompt");
        let web_only = build(&facts(Size::Full, Family::Generic, Tools { web: true, ..Default::default() }));
        assert!(web_only.contains("web_search") && !web_only.contains("Memory tool") && !web_only.contains("search_chats"));
    }

    #[test]
    fn the_short_prompts_are_short() {
        let full = build(&facts(Size::Full, Family::Qwen, Tools { web: true, memory: true, chats: true }));
        let compact = build(&facts(Size::Compact, Family::Qwen, Tools { web: true, memory: true, chats: true }));
        let tiny = build(&facts(Size::Tiny, Family::Qwen, Tools { web: true, memory: true, chats: true }));
        assert!(tiny.len() < compact.len() && compact.len() < full.len(), "{} {} {}", tiny.len(), compact.len(), full.len());
        assert!(!tiny.contains("web_search") && !tiny.contains("Keep any reasoning"), "a tiny context has room for two lines");
        assert!(tiny.contains("The date is Monday, 26 October 2026."));
        assert!(compact.contains("Today is Monday, 26 October 2026, 14:05"));
        assert!(compact.contains("Web tools: search only") && !compact.contains("Web tools (web_search"));
        assert!(!compact.contains("\n\n"), "one line per instruction");
        let remote = build(&Facts { local: false, ..facts(Size::Full, Family::Generic, Tools::default()) });
        assert!(!remote.contains("locally") && !remote.contains("based on the language model"), "{remote}");
        let thinking = build(&Facts { thinking: true, ..facts(Size::Full, Family::Llama, Tools::default()) });
        assert!(thinking.ends_with("Keep your reasoning short."));
        let qwen = build(&Facts { thinking: true, ..facts(Size::Full, Family::Qwen, Tools::default()) });
        assert!(!qwen.ends_with("Keep your reasoning short."), "Qwen has its own note");
    }

    #[test]
    fn your_texts_replace_or_extend_the_prompt_and_get_the_date_filled_in() {
        let now = at(14, 5);
        let g = "GENERATED";
        assert_eq!(resolve(g, &[(Mode::Default, "ignored")], &now, "en"), "GENERATED");
        assert_eq!(resolve(g, &[(Mode::Override, "Only this. {{language}}")], &now, "de"), "Only this. German");
        assert_eq!(resolve(g, &[(Mode::Append, "Also {{date}}.")], &now, "en"), "GENERATED\n\nAlso Monday, 26 October 2026.");
        assert_eq!(resolve(g, &[(Mode::Override, "   ")], &now, "en"), "GENERATED", "an empty override is ignored");
        assert_eq!(resolve(g, &[(Mode::Override, "Global"), (Mode::Append, "Assistant")], &now, "en"), "Global\n\nAssistant", "the assistant comes after the global choice");
        assert_eq!(fill("{{ TIME }} and {{unknown}} and {{date", &now, "en"), "Monday, 26 October 2026, 14:05 (UTC+02:00) and {{unknown}} and {{date");
        assert_eq!(resolve(g, &[(Mode::Override, &"x".repeat(7000))], &now, "en").len(), CUSTOM_MAX, "a very long text is cut");
        assert_eq!(language_name("de-DE"), "German");
        assert_eq!(language_name("xx-YY"), "xx-YY");
        assert_eq!(language_name(""), "English");
    }

    #[test]
    fn titles_are_cleaned_from_what_models_write() {
        assert_eq!(clean_title("Title: \"Cooking pasta fast\"."), Some("Cooking pasta fast".into()));
        assert_eq!(clean_title("<think>hmm let me think</think>\n**Fix my Rust build**"), Some("Fix my Rust build".into()));
        assert_eq!(clean_title("<think>never closed"), None);
        assert_eq!(clean_title("Titel: Reise nach Wien!"), Some("Reise nach Wien".into()));
        assert_eq!(clean_title("ok"), None, "too short");
        assert_eq!(clean_title(&"word ".repeat(30)), None, "too long");
        assert_eq!(clean_title("My system prompt says"), None, "a title that talks about the instructions is not one");
        assert_eq!(clean_title("\n\n  Second line wins \nthird"), Some("Second line wins".into()));
        let m = title_messages(&"q".repeat(500), "answer");
        assert_eq!(m[0].0, "system");
        assert!(m[1].1.starts_with("User: qqq") && m[1].1.ends_with("Title:"));
        assert_eq!(m[1].1.matches('q').count(), 300, "the question is cut");
    }
}

#[cfg(test)]
mod tool_texts_and_style {
    use super::*;

    /// The tool texts of the prompt, word for word like SP_TOOL_TEXT in the Studio's prompts.js
    /// (the round "Use the web / Research mode" brings the web tools; the texts wait for them here).
    #[test]
    fn the_tool_texts_are_the_studios_word_for_word() {
        assert_eq!(
            WEB,
            "Web tools (web_search, web_fetch): use them only for current or changing facts (news, prices, versions, events after your training), for specific facts you do not know or are unsure about, or when the user asks you to look something up or to open a link. For everything else (explanations, writing, code, math, advice, small talk) answer directly without searching. When you used the web, cite the sources as [n] with the numbers from the results and never invent a source. If a search fails, say so."
        );
        assert_eq!(
            MEMORY,
            "Memory tool (memory): save a short fact when you learn something durable about the user (name, preferences, projects, constraints) or when they ask you to remember something. Never save trivia, one-off requests or secrets such as passwords, keys or card numbers. Use update or remove when a fact changed or the user wants it forgotten. Mention a saved fact only briefly, if at all."
        );
        assert_eq!(WEB_COMPACT, "Web tools: search only for current or unknown facts, otherwise answer directly. Cite sources as [n]; never invent sources.");
        assert_eq!(MEMORY_COMPACT, "Memory tool: save durable facts about the user (name, preferences, projects); never trivia or secrets.");
        assert_eq!(CHATS, "Chat search tool (search_chats): use it when the user refers to an earlier conversation that you cannot see (\"as we discussed\", \"last time\").");
        assert_eq!(CHATS_COMPACT, "search_chats finds earlier conversations the user refers to.");
        assert_eq!(CALL, "Call tools with valid JSON arguments. After a tool result, answer the user; do not repeat the same call.");
        assert_eq!(NONE, "You have no tools in this chat: you cannot browse the web, open links, run code or read files except what the user shares here. Say so instead of pretending.");
    }

    /// The styles of the prompt against the Studio's own output (the fixtures come from
    /// artifacts.js, see fixtures/style_fixtures.js).
    #[test]
    fn the_styles_are_the_studios() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/style_fixtures.json")).expect("the fixtures are json");
        // the catalog itself: id, label and instruction of every style
        let styles = data["styles"].as_array().expect("the styles of the Studio");
        assert_eq!(styles.len(), style::STYLES.len(), "the five styles of the Studio");
        for (i, s) in styles.iter().enumerate() {
            assert_eq!(s["id"].as_str(), Some(style::STYLES[i].id), "id {i}");
            assert_eq!(s["label"].as_str(), Some(style::STYLES[i].label), "label {i}");
            assert_eq!(s["prompt"].as_str(), Some(style::STYLES[i].prompt), "prompt {i}");
        }
        // withStylePrompt for every combination the fixtures hold
        let cases = data["cases"].as_array().expect("the cases");
        assert_eq!(cases.len(), 18, "three bases over five styles, plus unknown, empty and missing");
        for case in cases {
            let base = case["base"].as_str().expect("a base");
            let style = case["style"].as_str().unwrap_or("default");
            assert_eq!(style::with_style_prompt(base, style), case["out"], "for {base:?} + {style:?}");
        }
        // what a stored style means and what the chip says
        assert_eq!(style::style_of(""), "default");
        assert_eq!(style::style_of("nope"), "default");
        assert_eq!(style::label_of("concise"), "Concise");
        assert_eq!(style::label_of("nope"), "Default");
        assert_eq!(style::instruction("default"), None, "the default adds nothing");
        assert_eq!(style::instruction("formal"), Some(style::STYLES[3].prompt));
    }
}
