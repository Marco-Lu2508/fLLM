//! Reasoning parsers: split a model's text into its reasoning ("thinking") and the answer.
//!
//! The server calls [`ReasoningParser::split`] with the text produced so far after every
//! token. A split is a pure function of that text, and while generation runs (`complete ==
//! false`) it never contains half a tag, so the reasoning and the content of a later call
//! always start with the ones of an earlier call; streamed deltas never need to be undone.
//!
//! Families (see `docs/server-api.md`, "Tools und Reasoning"):
//!
//! | id | tags | source |
//! | --- | --- | --- |
//! | `think` | `<think>` … `</think>` | Qwen3, QwQ, DeepSeek R1/V3.x, GLM, MiniMax-M2, SmolLM3, LFM2.5 |
//! | `ifm` | `<ifm|think>`, `<ifm|think_fast>`, `<ifm|think_faster>` … matching end tags | K2-Horizon template |
//! | `mistral` | `[THINK]` … `[/THINK]` | Ministral reasoning template |
//! | `seed` | `<seed:think>` … `</seed:think>` | Seed-OSS template |
//! | `minimax_m3` | `<mm:think>` … `</mm:think>` | MiniMax-M3 template |
//! | `gemma4` | `<|channel>thought` … `<channel|>` | Gemma 4 template |
//! | `gpt_oss` | Harmony channels: `analysis` is reasoning, `final` the answer | gpt-oss template |
//! | `granite` | `Here is my thought process:` ... `Here is my response:` | Granite 3.2/3.3 template |
//! | `hunyuan` | `<think>` ... `</think>` then the answer inside `<answer>` ... `</answer>` | Hunyuan-A13B |

use std::sync::Arc;

/// Reasoning and answer of the text seen so far.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReasoningParts {
    pub reasoning: String,
    pub content: String,
    /// `true` once the text contained (or, for prompts that open the block, implied) reasoning.
    pub reasoning_found: bool,
}

/// A reasoning format.
pub trait ReasoningParser: Send + Sync {
    /// Stable id, used by `--reasoning-parser`.
    fn id(&self) -> &'static str;

    /// Literals the server must keep when detokenizing: the vocabulary stores most of them
    /// as control tokens that plain decoding drops. Literals that are not tokens are ignored.
    fn markers(&self) -> Vec<String>;

    /// Split the text produced so far. `complete` is `true` for the final call.
    fn split(&self, text: &str, complete: bool) -> ReasoningParts;
}

/// Length of the prefix of `piece` that cannot be the start of one of `tags` at its end.
fn safe_prefix_end<S: AsRef<str>>(piece: &str, tags: &[S]) -> usize {
    let mut hold = 0usize;
    for tag in tags {
        let tag = tag.as_ref();
        for len in (1..tag.len()).rev() {
            if len <= hold {
                break;
            }
            if tag.is_char_boundary(len) && piece.ends_with(&tag[..len]) {
                hold = len;
                break;
            }
        }
    }
    piece.len() - hold
}

/// Earliest occurrence of any of `tags`: `(index, tag length)`.
fn earliest<S: AsRef<str>>(text: &str, tags: &[S]) -> Option<(usize, usize)> {
    tags.iter()
        .filter_map(|tag| {
            let tag = tag.as_ref();
            (!tag.is_empty())
                .then(|| text.find(tag).map(|at| (at, tag.len())))
                .flatten()
        })
        .min_by_key(|&(at, _)| at)
}

/// Reasoning between an opening and a closing tag.
///
/// With `starts_open` the prompt already opened the block (`…<think>\n`), so the reasoning
/// runs from the start of the text to the first closing tag. Otherwise the model opens the
/// block itself; text before an opening tag is content.
pub struct TagReasoning {
    id: &'static str,
    opens: Vec<String>,
    closes: Vec<String>,
    keep: Vec<String>,
    starts_open: bool,
}

impl TagReasoning {
    pub fn new(
        id: &'static str,
        opens: &[&str],
        closes: &[&str],
        keep: &[&str],
        starts_open: bool,
    ) -> Self {
        let own = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect();
        Self {
            id,
            opens: own(opens),
            closes: own(closes),
            keep: own(keep),
            starts_open,
        }
    }

    pub fn starts_open(&self) -> bool {
        self.starts_open
    }

    /// The closing tags that end the reasoning.
    pub fn closes(&self) -> &[String] {
        &self.closes
    }

    fn all_tags(&self) -> Vec<&String> {
        self.opens.iter().chain(&self.closes).collect()
    }
}

impl ReasoningParser for TagReasoning {
    fn id(&self) -> &'static str {
        self.id
    }

    fn markers(&self) -> Vec<String> {
        self.keep.clone()
    }

    fn split(&self, text: &str, complete: bool) -> ReasoningParts {
        let tags = self.all_tags();
        let hold = |piece: &str| -> usize {
            if complete {
                piece.len()
            } else {
                safe_prefix_end(piece, &tags)
            }
        };
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut reasoning_found = false;
        let mut closed = false;
        let mut rest = text;
        if self.starts_open {
            reasoning_found = true;
            // A model that repeats the opening tag the prompt already wrote.
            let lead = rest.trim_start();
            if let Some(after) = self
                .opens
                .iter()
                .find_map(|open| lead.strip_prefix(open.as_str()))
            {
                rest = after;
            }
        } else if let Some((index, len)) = earliest(rest, &self.opens) {
            content.push_str(&rest[..index]);
            rest = &rest[index + len..];
            reasoning_found = true;
        } else {
            let keep = hold(rest);
            content.push_str(&rest[..keep]);
            return ReasoningParts {
                reasoning,
                content,
                reasoning_found,
            };
        }
        if let Some((index, len)) = earliest(rest, &self.closes) {
            reasoning.push_str(&rest[..index]);
            content.push_str(&rest[index + len..]);
            closed = true;
        } else {
            let keep = hold(rest);
            reasoning.push_str(&rest[..keep]);
        }
        let reasoning = reasoning.trim().to_owned();
        let content = if closed || complete {
            content.trim().to_owned()
        } else {
            content.trim_start().to_owned()
        };
        ReasoningParts {
            reasoning,
            content,
            reasoning_found,
        }
    }
}

/// Reasoning in `<think>` tags followed by an answer wrapped in `<answer>` tags (Hunyuan-A13B). The answer tags are not
/// part of the content; a model that skips them still has its text returned.
pub struct AnswerTagged {
    inner: TagReasoning,
}

impl AnswerTagged {
    const OPEN: &'static str = "<answer>";
    const CLOSE: &'static str = "</answer>";

    pub fn new(starts_open: bool) -> Self {
        Self {
            inner: TagReasoning::new("hunyuan", &["<think>"], &["</think>"], &["<think>", "</think>", Self::OPEN, Self::CLOSE], starts_open),
        }
    }
}

impl ReasoningParser for AnswerTagged {
    fn id(&self) -> &'static str {
        "hunyuan"
    }

    fn markers(&self) -> Vec<String> {
        self.inner.markers()
    }

    fn split(&self, text: &str, complete: bool) -> ReasoningParts {
        let mut parts = self.inner.split(text, complete);
        let lead = parts.content.trim_start();
        let tagged = lead.starts_with(Self::OPEN);
        let mut content = lead.strip_prefix(Self::OPEN).unwrap_or(lead).to_owned();
        if !complete && Self::OPEN.starts_with(lead) && !lead.is_empty() {
            // `<ans` of a tag that is still being written
            content.clear();
        }
        let trimmed = content.trim_end();
        if let Some(body) = trimmed.strip_suffix(Self::CLOSE) {
            content = body.to_owned();
        } else if !complete {
            content.truncate(safe_prefix_end(&content, &[Self::CLOSE]));
        }
        // inside answer tags the whitespace before `</answer>` is dropped at the end, so it is not streamed early
        parts.content = if complete || tagged { content.trim().to_owned() } else { content.trim_start().to_owned() };
        parts
    }
}

/// No reasoning format: everything is the answer.
pub struct NoReasoning;

impl ReasoningParser for NoReasoning {
    fn id(&self) -> &'static str {
        "none"
    }
    fn markers(&self) -> Vec<String> {
        Vec::new()
    }
    fn split(&self, text: &str, _complete: bool) -> ReasoningParts {
        ReasoningParts {
            reasoning: String::new(),
            content: text.to_owned(),
            reasoning_found: false,
        }
    }
}

/// gpt-oss "Harmony" channels.
///
/// ```text
/// <|channel|>analysis<|message|>…reasoning…<|end|>
/// <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{…}<|call|>
/// <|start|>assistant<|channel|>final<|message|>…answer…<|return|>
/// ```
///
/// `analysis` is the reasoning, `final` and recipient-less `commentary` (a preamble) the
/// answer. A message addressed to `functions.NAME` — in the channel header or, as in the
/// template, in the role header — is passed on in the canonical spelling
/// `<|channel|>commentary to=functions.NAME<|message|>BODY<|call|>`, which
/// `tool_parsers::HarmonyParser` reads. Messages to other recipients (built-in tools) are
/// not shown and count as reasoning. Text without any Harmony marker is plain content.
pub struct HarmonyReasoning;

const HARMONY_TOKENS: [&str; 7] = [
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<|constrain|>",
    "<|end|>",
    "<|call|>",
    "<|return|>",
];

enum Channel {
    Analysis,
    Answer,
    Tool(String),
    Hidden,
}

impl HarmonyReasoning {
    fn classify(header: &str) -> Channel {
        let header = header
            .replace("<|start|>", " ")
            .replace("<|channel|>", " ")
            .replace("<|constrain|>", " ");
        if let Some(at) = header.find("to=") {
            let recipient: String = header[at + 3..]
                .chars()
                .take_while(|c| !c.is_whitespace() && *c != '<')
                .collect();
            return match recipient.strip_prefix("functions.") {
                Some(name) if !name.is_empty() => Channel::Tool(name.to_owned()),
                _ => Channel::Hidden,
            };
        }
        let channel = header
            .split_whitespace()
            .find(|word| *word != "assistant")
            .unwrap_or("");
        match channel {
            "analysis" => Channel::Analysis,
            // `final`, a recipient-less `commentary` preamble and anything unknown are shown.
            _ => Channel::Answer,
        }
    }
}

impl ReasoningParser for HarmonyReasoning {
    fn id(&self) -> &'static str {
        "gpt_oss"
    }

    fn markers(&self) -> Vec<String> {
        HARMONY_TOKENS.iter().map(|t| (*t).to_owned()).collect()
    }

    fn split(&self, text: &str, complete: bool) -> ReasoningParts {
        let hold = |piece: &str| -> usize {
            if complete {
                piece.len()
            } else {
                safe_prefix_end(piece, &HARMONY_TOKENS)
            }
        };
        let mut reasoning: Vec<String> = Vec::new();
        let mut content = String::new();
        let mut found = false;

        // Text before the first marker is plain content.
        let first = text.find("<|").unwrap_or(text.len());
        let lead = &text[..first];
        content.push_str(
            &lead[..if first == text.len() {
                hold(lead)
            } else {
                first
            }],
        );
        if first == text.len() {
            return ReasoningParts {
                reasoning: String::new(),
                content: if complete {
                    content.trim().to_owned()
                } else {
                    content.trim_start().to_owned()
                },
                reasoning_found: false,
            };
        }
        let mut pos = first;
        loop {
            let Some(message) = text[pos..].find("<|message|>") else {
                // An incomplete header: nothing to show yet; at the end, show it as it is.
                if complete {
                    content.push_str(&text[pos..]);
                }
                break;
            };
            let header = &text[pos..pos + message];
            let body_start = pos + message + "<|message|>".len();
            let terminator = [
                "<|end|>",
                "<|call|>",
                "<|return|>",
                "<|start|>",
                "<|channel|>",
            ]
            .iter()
            .filter_map(|t| text[body_start..].find(t).map(|at| (at, *t)))
            .min_by_key(|&(at, _)| at);
            let (body, ended, next) = match terminator {
                Some((at, term)) => {
                    let consumed = if matches!(term, "<|end|>" | "<|call|>" | "<|return|>") {
                        term.len()
                    } else {
                        0
                    };
                    (
                        &text[body_start..body_start + at],
                        true,
                        body_start + at + consumed,
                    )
                }
                None => {
                    let rest = &text[body_start..];
                    (&rest[..hold(rest)], false, text.len())
                }
            };
            match Self::classify(header) {
                Channel::Analysis => {
                    found = true;
                    reasoning.push(body.to_owned());
                }
                Channel::Answer => content.push_str(body),
                Channel::Hidden => {
                    found = true;
                    reasoning.push(body.to_owned());
                }
                Channel::Tool(name) => {
                    content.push_str("<|channel|>commentary to=functions.");
                    content.push_str(&name);
                    content.push_str("<|message|>");
                    content.push_str(body);
                    if ended && text[..next].ends_with("<|call|>") {
                        content.push_str("<|call|>");
                    }
                }
            }
            if !ended {
                break;
            }
            pos = next;
            // Only markers can follow a message; stray text between messages is content.
            let tail = &text[pos..];
            let gap = tail.find("<|").unwrap_or(tail.len());
            if gap > 0 {
                let shown = if gap == tail.len() { hold(tail) } else { gap };
                content.push_str(&tail[..shown]);
                pos += gap;
            }
            if pos >= text.len() {
                break;
            }
        }
        let reasoning = reasoning
            .iter()
            .map(|r| r.trim())
            .filter(|r| !r.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        ReasoningParts {
            reasoning,
            content: if complete {
                content.trim().to_owned()
            } else {
                content.trim_start().to_owned()
            },
            reasoning_found: found,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------------------------

/// The reasoning format of a model; resolved once at startup, turned into a parser per request
/// (whether the prompt already opened the thinking block depends on the prompt).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Think,
    Ifm,
    Mistral,
    Seed,
    MiniMaxM3,
    Gemma4,
    GptOss,
    Granite,
    Hunyuan,
    None,
}

const NAMES: &[(&str, Family, &[&str])] = &[
    (
        "think",
        Family::Think,
        &["deepseek_r1", "qwen3", "qwq", "glm", "minimax", "default"],
    ),
    (
        "ifm",
        Family::Ifm,
        &["k2horizon", "k2-horizon", "k2_horizon"],
    ),
    ("mistral", Family::Mistral, &["ministral", "magistral"]),
    ("seed", Family::Seed, &["seed_oss", "seed-oss"]),
    ("minimax_m3", Family::MiniMaxM3, &["minimax-m3", "mm"]),
    ("gemma4", Family::Gemma4, &["gemma", "gemma-4"]),
    (
        "gpt_oss",
        Family::GptOss,
        &["gpt-oss", "gptoss", "harmony", "openai"],
    ),
    ("granite", Family::Granite, &["granite3", "granite-3.3", "granite3.3"]),
    ("hunyuan", Family::Hunyuan, &["hunyuan_a13b", "hunyuan-a13b"]),
    ("none", Family::None, &["off", "disabled"]),
];

/// Canonical ids, for help texts and error messages.
pub fn family_ids() -> Vec<&'static str> {
    NAMES.iter().map(|(id, _, _)| *id).collect()
}

impl Family {
    /// `--reasoning-parser` value. `auto` is not a family: use [`Family::select`].
    pub fn parse(name: &str) -> Result<Self, String> {
        let lower = name.trim().to_ascii_lowercase();
        NAMES
            .iter()
            .find(|(id, _, aliases)| *id == lower || aliases.contains(&lower.as_str()))
            .map(|(_, family, _)| *family)
            .ok_or_else(|| {
                format!(
                    "unknown reasoning parser '{name}' (available: auto, {})",
                    family_ids().join(", ")
                )
            })
    }

    /// `true` when `--reasoning-parser` accepts the value (`auto` included).
    pub fn is_known(name: &str) -> bool {
        name.trim().eq_ignore_ascii_case("auto") || Self::parse(name).is_ok()
    }

    /// The family of a model: an explicit flag wins, then the markers of the chat template,
    /// then `<think>`, which almost every reasoning model uses.
    pub fn select(chat_template: &str, flag: Option<&str>) -> Result<Self, String> {
        if let Some(name) = flag.filter(|n| !n.trim().eq_ignore_ascii_case("auto")) {
            return Self::parse(name);
        }
        const BY_TEMPLATE: &[(&str, Family)] = &[
            ("<ifm|think", Family::Ifm),
            ("<|start|>assistant<|channel|>", Family::GptOss),
            ("<|channel|>analysis", Family::GptOss),
            ("<|channel>thought", Family::Gemma4),
            ("<seed:think>", Family::Seed),
            ("<mm:think>", Family::MiniMaxM3),
            ("[THINK]", Family::Mistral),
            ("Here is my thought process:", Family::Granite),
        ];
        for (marker, family) in BY_TEMPLATE {
            if chat_template.contains(marker) {
                return Ok(*family);
            }
        }
        Ok(Family::Think)
    }

    /// The parser for a request whose rendered prompt is `prompt`.
    pub fn parser(self, prompt: &str) -> Arc<dyn ReasoningParser> {
        let tail = prompt.trim_end();
        match self {
            Family::None => Arc::new(NoReasoning),
            Family::GptOss => Arc::new(HarmonyReasoning),
            Family::Think => {
                // Templates such as QwQ, DeepSeek-R1 or K2-Horizon end the prompt with an open
                // tag whose name contains "think"; its closing tag derives from it.
                if tail.ends_with('>') {
                    if let Some(start) = tail.rfind('<') {
                        let name = &tail[start + 1..tail.len() - 1];
                        if !name.starts_with('/')
                            && name.contains("think")
                            && !name.contains(char::is_whitespace)
                        {
                            let open = tail[start..].to_owned();
                            let close = format!("</{name}>");
                            return Arc::new(TagReasoning::new(
                                "think",
                                &[&open],
                                &[&close],
                                &[&open, &close],
                                true,
                            ));
                        }
                    }
                }
                Arc::new(TagReasoning::new(
                    "think",
                    &["<think>"],
                    &["</think>"],
                    &["<think>", "</think>"],
                    false,
                ))
            }
            Family::Ifm => {
                let opens = ["<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>"];
                let closes = ["</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"];
                let keep: Vec<&str> = opens.iter().chain(&closes).copied().collect();
                let starts_open = opens.iter().any(|open| tail.ends_with(open));
                Arc::new(TagReasoning::new(
                    "ifm",
                    &opens,
                    &closes,
                    &keep,
                    starts_open,
                ))
            }
            Family::Mistral => tags("mistral", "[THINK]", "[/THINK]", tail),
            Family::Granite => Arc::new(TagReasoning::new(
                "granite",
                &["Here is my thought process:"],
                &["Here is my response:"],
                &[],
                false,
            )),
            Family::Hunyuan => Arc::new(AnswerTagged::new(tail.ends_with("<think>"))),
            Family::Seed => tags("seed", "<seed:think>", "</seed:think>", tail),
            Family::MiniMaxM3 => tags("minimax_m3", "<mm:think>", "</mm:think>", tail),
            Family::Gemma4 => {
                // The block opens with the `<|channel>` token followed by the word "thought".
                let starts_open = tail.ends_with("<|channel>thought");
                Arc::new(TagReasoning::new(
                    "gemma4",
                    &["<|channel>thought"],
                    &["<channel|>"],
                    &["<|channel>", "<channel|>"],
                    starts_open,
                ))
            }
        }
    }
}

fn tags(id: &'static str, open: &str, close: &str, tail: &str) -> Arc<dyn ReasoningParser> {
    Arc::new(TagReasoning::new(
        id,
        &[open],
        &[close],
        &[open, close],
        tail.ends_with(open),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every prefix of `text`: reasoning and content only grow, no tag is ever shown half.
    fn stream(parser: &dyn ReasoningParser, text: &str) -> ReasoningParts {
        let mut previous = ReasoningParts::default();
        for end in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
            let step = parser.split(&text[..end], false);
            assert!(
                step.reasoning.starts_with(&previous.reasoning),
                "{}: reasoning shrank at {end}: {:?} -> {:?}",
                parser.id(),
                previous.reasoning,
                step.reasoning
            );
            assert!(
                step.content.starts_with(&previous.content),
                "{}: content shrank at {end}: {:?} -> {:?}",
                parser.id(),
                previous.content,
                step.content
            );
            previous = step;
        }
        let last = parser.split(text, true);
        assert!(last.reasoning.starts_with(&previous.reasoning));
        assert!(last.content.starts_with(&previous.content));
        last
    }

    #[test]
    fn granite_thought_process_and_response_markers() {
        let parser = Family::Granite.parser("<|start_of_role|>assistant<|end_of_role|>");
        let text = "Here is my thought process: The user wants 6*7, which is 42. Here is my response: 6 times 7 is 42.";
        let parts = stream(parser.as_ref(), text);
        assert_eq!(parts.reasoning, "The user wants 6*7, which is 42.");
        assert_eq!(parts.content, "6 times 7 is 42.");
        assert!(parts.reasoning_found);
        // no thinking requested: the answer stays untouched
        let plain = stream(parser.as_ref(), "6 times 7 is 42.");
        assert_eq!((plain.reasoning.as_str(), plain.content.as_str(), plain.reasoning_found), ("", "6 times 7 is 42.", false));
        // cut off while thinking
        let cut = parser.split("Here is my thought process: still thi", true);
        assert_eq!((cut.reasoning.as_str(), cut.content.as_str()), ("still thi", ""));
        let selected = Family::select("... write your response after 'Here is my thought process:' ...", None).unwrap();
        assert_eq!(selected, Family::Granite);
    }

    #[test]
    fn hunyuan_think_then_answer_tags() {
        let parser = Family::Hunyuan.parser("<|startoftext|>user<|extra_0|>");
        let text = "<think>
Compute 6*7.
</think>
<answer>
42
</answer>";
        let parts = stream(parser.as_ref(), text);
        assert_eq!((parts.reasoning.as_str(), parts.content.as_str()), ("Compute 6*7.", "42"));
        // a model without answer tags
        let bare = stream(parser.as_ref(), "<think>x</think>
plain answer");
        assert_eq!(bare.content, "plain answer");
        // no reasoning at all, with answer tags
        let only = stream(parser.as_ref(), "<answer>
hi
</answer>");
        assert_eq!((only.reasoning.as_str(), only.content.as_str()), ("", "hi"));
        // a prompt that already opened the block
        let open = Family::Hunyuan.parser("...<think>");
        let parts = stream(open.as_ref(), "thinking</think><answer>done</answer>");
        assert_eq!((parts.reasoning.as_str(), parts.content.as_str()), ("thinking", "done"));
        // incomplete answer tag is never shown
        let streamed = parser.split("<think>a</think>
<ans", false);
        assert_eq!(streamed.content, "");
        let streamed = parser.split("<think>a</think>
<answer>text</ans", false);
        assert_eq!(streamed.content, "text");
    }

    #[test]
    fn prompt_that_opens_the_think_block() {
        let parser = Family::Think.parser("<|ifm|im_start|>assistant\n<ifm|think>\n");
        let parts = stream(parser.as_ref(), "Let me think.\n</ifm|think>\n391");
        assert_eq!(parts.reasoning, "Let me think.");
        assert_eq!(parts.content, "391");
        let parts = parser.split("Step one\n</ifm|th", false);
        assert_eq!(
            (parts.reasoning.as_str(), parts.content.as_str()),
            ("Step one", "")
        );
        let parts = parser.split("cut off while thinking", true);
        assert_eq!(
            (parts.reasoning.as_str(), parts.content.as_str()),
            ("cut off while thinking", "")
        );
        // The model repeats the opening tag.
        let parts = parser.split("<ifm|think>\nagain\n</ifm|think>\nanswer", true);
        assert_eq!(
            (parts.reasoning.as_str(), parts.content.as_str()),
            ("again", "answer")
        );
    }

    #[test]
    fn model_opens_the_tag_itself() {
        let parser = Family::Think.parser("<|im_start|>assistant\n");
        let parts = stream(parser.as_ref(), "<think>\nhmm\n</think>\n\nAnswer");
        assert_eq!(
            (parts.reasoning.as_str(), parts.content.as_str()),
            ("hmm", "Answer")
        );
        let plain = stream(parser.as_ref(), "No thinking here");
        assert!(!plain.reasoning_found);
        assert_eq!(plain.content, "No thinking here");
        // A closed block in the prompt (thinking disabled) leaves the answer alone.
        let closed = Family::Think.parser("x<think>\n\n</think>\n\n");
        let parts = closed.split("Answer", true);
        assert!(!parts.reasoning_found);
        assert_eq!(parts.content, "Answer");
    }

    #[test]
    fn k2_horizon_accepts_every_end_tag() {
        for (open_tag, close_tag) in [
            ("<ifm|think>", "</ifm|think>"),
            ("<ifm|think_fast>", "</ifm|think_fast>"),
            ("<ifm|think_faster>", "</ifm|think_faster>"),
            // The effort in the prompt and the end tag the model writes may differ.
            ("<ifm|think>", "</ifm|think_fast>"),
            ("<ifm|think_fast>", "</ifm|think_faster>"),
        ] {
            let prompt = format!("<|ifm|im_start|>assistant\n{open_tag}\n");
            let parser = Family::Ifm.parser(&prompt);
            let text = format!("Reason.\n{close_tag}\nAnswer");
            let parts = stream(parser.as_ref(), &text);
            assert_eq!(
                (parts.reasoning.as_str(), parts.content.as_str()),
                ("Reason.", "Answer"),
                "{open_tag} {close_tag}"
            );
        }
        // Thinking disabled: the prompt already closed the block.
        let disabled = Family::Ifm.parser("<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>\n");
        assert_eq!(
            disabled.split("Just the answer", true).content,
            "Just the answer"
        );
        // The model opens the block itself.
        let parser = Family::Ifm.parser("<|ifm|im_start|>assistant\n");
        let parts = stream(
            parser.as_ref(),
            "<ifm|think_fast>\nquick\n</ifm|think_fast>\nA",
        );
        assert_eq!(
            (parts.reasoning.as_str(), parts.content.as_str()),
            ("quick", "A")
        );
        assert!(parser.markers().contains(&"</ifm|think_faster>".to_owned()));
    }

    #[test]
    fn other_tag_families() {
        let mistral = Family::Mistral.parser("[INST]x[/INST]");
        let parts = stream(mistral.as_ref(), "[THINK]plan[/THINK]Done");
        assert_eq!(
            (parts.reasoning.as_str(), parts.content.as_str()),
            ("plan", "Done")
        );
        let seed = Family::Seed.parser("");
        let parts = stream(seed.as_ref(), "<seed:think>why</seed:think>because");
        assert_eq!(
            (parts.reasoning.as_str(), parts.content.as_str()),
            ("why", "because")
        );
        let gemma = Family::Gemma4.parser("<|turn>model\n");
        let parts = stream(gemma.as_ref(), "<|channel>thought\nhmm<channel|>The answer");
        assert_eq!(
            (parts.reasoning.as_str(), parts.content.as_str()),
            ("hmm", "The answer")
        );
        let none = Family::None.parser("<think>");
        assert_eq!(
            none.split("<think>x</think>y", true).content,
            "<think>x</think>y"
        );
    }

    #[test]
    fn harmony_channels() {
        let parser = HarmonyReasoning;
        let text = "<|channel|>analysis<|message|>User wants the weather.<|end|><|start|>assistant<|channel|>final<|message|>It is sunny.<|return|>";
        let parts = stream(&parser, text);
        assert_eq!(parts.reasoning, "User wants the weather.");
        assert_eq!(parts.content, "It is sunny.");
        assert!(parts.reasoning_found);
        // Function calls are handed on in the canonical spelling, from either header form.
        let channel_form = "<|channel|>analysis<|message|>Need a tool.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Berlin\"}<|call|>";
        let parts = stream(&parser, channel_form);
        assert_eq!(parts.reasoning, "Need a tool.");
        assert_eq!(
            parts.content,
            "<|channel|>commentary to=functions.get_weather<|message|>{\"city\":\"Berlin\"}<|call|>"
        );
        let role_form = "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"city\":\"Berlin\"}<|call|>";
        assert_eq!(stream(&parser, role_form).content, parts.content);
        // A preamble is shown; other recipients are not.
        let preamble = "<|channel|>commentary<|message|>Checking.<|end|><|start|>assistant<|channel|>commentary to=browser.search<|message|>{}<|call|>";
        let parts = stream(&parser, preamble);
        assert_eq!(parts.content, "Checking.");
        // No markers at all: plain content, and a trailing `<` is held while streaming.
        let plain = stream(&parser, "just text");
        assert_eq!(plain.content, "just text");
        assert!(!plain.reasoning_found);
        assert_eq!(parser.split("text <", false).content, "text ");
        // A cut-off call: the body is passed on so the tool parser can refuse it.
        let cut = parser.split(
            "<|channel|>commentary to=functions.f<|message|>{\"a\":",
            true,
        );
        assert_eq!(
            cut.content,
            "<|channel|>commentary to=functions.f<|message|>{\"a\":"
        );
    }

    #[test]
    fn harmony_pipeline_reaches_the_tool_parser() {
        let text = "<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Berlin\"}";
        let parts = HarmonyReasoning.split(text, true);
        let tools = crate::tool_parsers::by_name("gpt_oss").unwrap();
        let out = crate::tool_parsers::extract(tools.as_ref(), &parts.content, true);
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].name, "get_weather");
        assert_eq!(out.content, "");
    }

    #[test]
    fn family_selection() {
        assert_eq!(Family::select("", None).unwrap(), Family::Think);
        assert_eq!(
            Family::select("... <ifm|think> ...", None).unwrap(),
            Family::Ifm
        );
        assert_eq!(
            Family::select("<|start|>assistant<|channel|>", None).unwrap(),
            Family::GptOss
        );
        assert_eq!(Family::select("[THINK]", None).unwrap(), Family::Mistral);
        assert_eq!(
            Family::select("<ifm|think>", Some("think")).unwrap(),
            Family::Think
        );
        assert_eq!(
            Family::select("<ifm|think>", Some("auto")).unwrap(),
            Family::Ifm
        );
        assert_eq!(Family::select("", Some("none")).unwrap(), Family::None);
        assert!(Family::select("", Some("bogus")).is_err());
        assert!(Family::is_known("auto") && Family::is_known("harmony") && !Family::is_known("x"));
        for id in family_ids() {
            assert!(Family::parse(id).is_ok());
        }
    }

    #[test]
    fn hostile_text_never_panics() {
        let families = [
            Family::Think,
            Family::Ifm,
            Family::Mistral,
            Family::Seed,
            Family::MiniMaxM3,
            Family::Gemma4,
            Family::GptOss,
            Family::None,
        ];
        let inputs = [
            "",
            "<",
            "<|",
            "<|channel|>",
            "<|channel|>final",
            "<|message|><|message|>",
            "</think>",
            "<think>",
            "<think></think></think>",
            "€<think",
            "[THINK]",
            "<|channel>thought",
            "<|start|><|start|><|end|>",
            "\u{FFFD}<ifm|think",
        ];
        for family in families {
            for prompt in [
                "",
                "<think>\n",
                "<ifm|think>\n",
                "[THINK]",
                "<|channel>thought",
            ] {
                let parser = family.parser(prompt);
                for text in inputs {
                    for complete in [false, true] {
                        let _ = parser.split(text, complete);
                    }
                    for (cut, _) in text.char_indices() {
                        let _ = parser.split(&text[..cut], false);
                    }
                }
            }
        }
    }
}
