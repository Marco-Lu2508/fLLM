//! **Settings search**: the box over the settings pages that keeps the ones whose words every
//! word of the question is in, best first.
//!
//! Port of `settingssearch.js` — the two functions of its `@pure` block with their rule intact,
//! and the box, the empty line, the hiding and the mark of the page. The page reads its words
//! out of its own document (`pane.textContent`); this window keeps a list of its settings and
//! their words, which is the one difference said in PARITY.md.

use iced::{
    Length,
    widget::{column, container, text, text_input, Space},
    Element,
};

use crate::app::Message;
use crate::theme::Tokens;

/// One settings page of the index: what it is called and the words that find it (`{id, label,
/// text}` of `buildIndex`).
#[derive(Debug, Clone, PartialEq)]
pub struct Page {
    pub id: &'static str,
    pub label: &'static str,
    pub text: &'static str,
}

/// One line of the answer (`{id, score}` of `rankSettingsPages`, with the order it came in for
/// the tie the page breaks by).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ranked {
    pub id: &'static str,
    pub score: i64,
    pub order: usize,
}

/// The settings of this window, in the order the column draws them. The page fills its `text`
/// from the document it has; here every line is written down — the words of the boxes this
/// window draws, which it cannot read back (no document to read).
pub const PAGES: &[Page] = &[
    Page {
        id: "general",
        label: "General",
        text: "startup send key name new chat where chats are saved open the folder language of the app update",
    },
    Page {
        id: "answers",
        label: "Answers",
        text: "sampling automatic manual temperature top-p top-k min-p token limit reply length stream steps server endpoint answers",
    },
    Page {
        id: "personalization",
        label: "Personalization",
        text: "about me instructions memory remember facts learn from this session lookback personal",
    },
    Page {
        id: "engine",
        label: "Processor",
        text: "processor feature limit faster reading long prompts engine threads cache quantisation speed features",
    },
    Page {
        id: "appearance",
        label: "Appearance",
        text: "theme dark light accent colour font size density look style",
    },
    Page {
        id: "hardware",
        label: "Hardware",
        text: "processor check graphics gpu npu amd nvidia device memory test report simd tier",
    },
    Page {
        id: "local-api",
        label: "Local API server",
        text: "server and api keys try the api statistics metrics openai anthropic ollama port rate limit token count requests",
    },
    Page {
        id: "logs",
        label: "Logs",
        text: "log lines refresh copy server output errors",
    },
    Page {
        id: "model",
        label: "Model details",
        text: "model details facts architecture parameters quantization context length vocabulary word pieces other names aliases chat format template",
    },
    Page {
        id: "tokenizer",
        label: "Tokenizer",
        text: "tokenize tokens bytes jinja template messages chat template count",
    },
    Page {
        id: "connect",
        label: "Connect other apps",
        text: "claude code codex cursor cli sdk python typescript curl key plan shell environment",
    },
    Page {
        id: "search",
        label: "Search",
        text: "document search embeddings vector web search engines searxng brave test query hybrid keyword",
    },
    Page {
        id: "mcp",
        label: "MCP servers",
        text: "model context protocol tools servers registry find add url command headers resources",
    },
    Page {
        id: "skills",
        label: "Skills",
        text: "skill improvements scheduled tasks new task library finder template workspace markdown learning consolidate",
    },
    Page {
        id: "plugins",
        label: "Plugins",
        text: "plugin folder install repository clone remove contributions skills agents commands hooks",
    },
    Page {
        id: "hooks",
        label: "Hooks",
        text: "hook shell commands before a tool call after a file edit prompt submit stop compact matcher timeout",
    },
    Page {
        id: "agents",
        label: "Sub-agents",
        text: "sub agent helper instructions tools model step limit scope built in explore general",
    },
    Page {
        id: "usage",
        label: "Usage and cost",
        text: "usage cost tokens price minutes spend cache",
    },
    Page {
        id: "runtime",
        label: "Runtime",
        text: "sandbox computer use turbo limits restricted agent commands runtime",
    },
    Page {
        id: "providers",
        label: "Providers",
        text: "model providers cloud openai anthropic google ollama address key test connection models list",
    },
    Page {
        id: "dictation",
        label: "Dictation",
        text: "dictation speech to text microphone voice input transcribe",
    },
    Page {
        id: "reading",
        label: "Reading aloud",
        text: "reading aloud text to speech voice sound play chime",
    },
    Page {
        id: "alerts",
        label: "Alerts",
        text: "alerts notify sound desktop notification permission",
    },
];

/// Splits a question into lower-case words (`searchWords`): everything that is neither a letter
/// nor a digit separates, and nothing empty is kept. The page splits on `[^\p{L}\p{N}]+`; Rust's
/// `is_alphanumeric` is letters and all numbers, which is what those two classes hold.
pub fn search_words(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

/// Ranks the pages for a question (`rankSettingsPages`): every word must be in the label or the
/// text, a word at the start of the label counts most, and a page without a word for one of them
/// is not part of the answer at all. Ties keep the order the pages came in.
pub fn rank_settings_pages(query: &str, pages: &[Page]) -> Vec<Ranked> {
    let words = search_words(query);
    if words.is_empty() {
        return pages
            .iter()
            .enumerate()
            .map(|(order, page)| Ranked {
                id: page.id,
                score: 0,
                order,
            })
            .collect();
    }
    let mut out: Vec<Ranked> = Vec::new();
    for (order, page) in pages.iter().enumerate() {
        let label = page.label.to_lowercase();
        let text = page.text.to_lowercase();
        let mut score: i64 = 0;
        for word in &words {
            let points = if label.starts_with(word.as_str()) {
                100
            } else if label.contains(word.as_str()) {
                60
            } else if text.contains(&format!(" {word}")) || text.starts_with(word.as_str()) {
                10
            } else if text.contains(word.as_str()) {
                4
            } else {
                // a word without a place on this page takes the page out of the answer
                score = 0;
                break;
            };
            score += points;
        }
        if score != 0 {
            out.push(Ranked {
                id: page.id,
                score,
                order,
            });
        }
    }
    out.sort_by(|a, b| b.score.cmp(&a.score).then(a.order.cmp(&b.order)));
    out
}

/// Whether the empty line stands (`empty.hidden = !searching || ranked.length > 0`): a word is
/// asked and no page has it.
pub fn shows_empty(query: &str) -> bool {
    !search_words(query).is_empty() && rank_settings_pages(query, PAGES).is_empty()
}

/// The id of the best match, which is the page Enter opens (`rankSettingsPages(...)[0]`).
pub fn best_match(query: &str) -> Option<&'static str> {
    rank_settings_pages(query, PAGES).first().map(|rank| rank.id)
}

/// Whether a page stands in the column while a question is asked (`btn.hidden = searching &&
/// !rank.has(id)`): with nothing asked, everything stands; a word that finds nothing hides every
/// page and the empty line says so.
pub fn keep(id: &'static str, query: &str) -> bool {
    if search_words(query).is_empty() {
        return true;
    }
    rank_settings_pages(query, PAGES).iter().any(|rank| rank.id == id)
}

/// The box over the settings pages with the line under it (`settings-search` of the page).
pub fn search_box<'a>(t: Tokens, query: &'a str, showing_empty: bool) -> Element<'a, Message> {
    let mut body = column![text_input("Search settings", query)
        .on_input(Message::SettingsQuery)
        .on_submit(Message::SettingsEnter)
        .size(13)
        .padding([6, 10])
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
        })]
    .spacing(6);
    if showing_empty {
        // `settings-search-empty` of the page, which is no longer hidden
        body = body.push(text("No setting matches.").size(13).color(t.muted));
    }
    body.into()
}

/// The mark of a page the question jumped to (`search-hit` of the page): a line over the card
/// that finds the eye, the way the class colours the row it lands on. The page puts no words on
/// a mark, so neither has this one.
pub fn marker<'a>(t: Tokens) -> Element<'a, Message> {
    iced::widget::container(Space::new().width(Length::Fill).height(Length::Fixed(2.0)))
        .width(Length::Fill)
        .style(move |_: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(t.primary)),
            ..Default::default()
        })
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages() -> Vec<Page> {
        vec![
            Page {
                id: "one",
                label: "Theme",
                text: "dark light accent colour",
            },
            Page {
                id: "two",
                label: "Models",
                text: "the theme of the answer is not here token",
            },
            Page {
                id: "three",
                label: "Logs",
                text: "server output",
            },
        ]
    }

    /// `searchWords`: lower case, on every character that is no letter or digit, nothing empty.
    #[test]
    fn a_question_becomes_its_words() {
        assert_eq!(search_words("Top-P & min_p"), vec!["top", "p", "min", "p"]);
        assert_eq!(search_words("  "), Vec::<String>::new());
        assert_eq!(search_words("MCP-servers 2"), vec!["mcp", "servers", "2"]);
        assert_eq!(search_words("Ünïcode Ärger"), vec!["ünïcode", "ärger"]);
    }

    /// The four places a word can have: the start of the label, inside the label, the start of a
    /// word of the text, and inside the text.
    #[test]
    fn a_word_counts_where_it_stands() {
        let ranked = rank_settings_pages("answer", &pages());
        assert_eq!(ranked[0].id, "two", "the start of a word of the text is worth 10");
        let ranked = rank_settings_pages("odels", &pages());
        assert_eq!(ranked[0].id, "two", "inside the label is worth 60");
        assert_eq!(ranked[0].score, 60, "and that is what it is worth");
        let ranked = rank_settings_pages("answer", &pages());
        assert_eq!(ranked[0].score, 10, "the start of a word of the text is worth 10");
        let ranked = rank_settings_pages("theme", &pages());
        assert_eq!(ranked[0].id, "one", "the start of the label is worth 100");
        assert_eq!(ranked[0].score, 100, "and that is what it is worth");
        assert_eq!(ranked[1].score, 10, "the page that only has it in its text");
        let ranked = rank_settings_pages("colour", &pages());
        assert_eq!(ranked[0].id, "one", "a word of the text is worth 10");
        assert_eq!(ranked[0].score, 10, "and that is what it is worth");
        let ranked = rank_settings_pages("ark", &pages());
        assert_eq!(
            ranked[0].id, "one",
            "inside a word of the text is worth 4"
        );
        assert_eq!(ranked[0].score, 4, "and that is what it is worth");
        let ranked = rank_settings_pages("the", &pages());
        assert_eq!(ranked[0].id, "one", "the start of the label is worth more than the text");
    }

    /// A word that is nowhere takes the whole page out; the scores of the words add up; ties keep
    /// the order of the pages.
    #[test]
    fn a_page_without_a_word_is_not_part_of_the_answer() {
        let ranked = rank_settings_pages("theme log", &pages());
        assert_eq!(
            ranked.iter().map(|rank| rank.id).collect::<Vec<_>>(),
            Vec::<&str>::new(),
            "one word without a place is enough"
        );
        let ranked = rank_settings_pages("logs theme", &pages());
        assert!(
            ranked.is_empty(),
            "the same from the other side, word by word"
        );
        let ranked = rank_settings_pages("log", &pages());
        assert_eq!(ranked.len(), 1, "the page that has it stays");
        let tied = rank_settings_pages("", &pages());
        assert_eq!(
            tied.iter().map(|rank| rank.id).collect::<Vec<_>>(),
            vec!["one", "two", "three"],
            "without a word every page stands, as it came"
        );
        let equal = vec![
            Page {
                id: "a",
                label: "Alpha",
                text: "first",
            },
            Page {
                id: "b",
                label: "Alps",
                text: "second",
            },
        ];
        let ranked = rank_settings_pages("alp", &equal);
        assert_eq!(
            ranked.iter().map(|rank| rank.id).collect::<Vec<_>>(),
            vec!["a", "b"],
            "the same score keeps the order the pages came in"
        );
        assert_eq!(ranked[0].score, ranked[1].score, "both are worth 100");
    }

    /// The question of the box hides what does not match and keeps what does — and with no word
    /// asked, nothing is hidden. These two stand on the index of the window itself, as the box
    /// asks it.
    #[test]
    fn the_column_shows_what_the_question_finds() {
        assert!(keep("general", ""));
        assert!(keep("general", "   "));
        assert!(keep("appearance", "theme"), "the page that has the word");
        assert!(!keep("mcp", "theme"));
        assert!(!keep("general", "zzqqxx"));
    }

    /// The best match is what Enter opens.
    #[test]
    fn the_best_match_comes_first() {
        let best = |query: &str| rank_settings_pages(query, &pages()).first().map(|rank| rank.id);
        assert_eq!(best("logs"), Some("three"));
        assert_eq!(best("theme"), Some("one"), "the label that starts with it");
        assert_eq!(best("nothing here at all"), None);
        assert_eq!(
            best_match("alerts"),
            Some("alerts"),
            "of the index of the window, the label that starts with the word"
        );
        assert_eq!(
            best_match(""),
            Some("general"),
            "the first page of the column stands first, as the page opens the first pane"
        );
        assert_eq!(best_match("zzqqxx"), None);
    }

    /// The line over the box stands when a word is asked and no page has it.
    #[test]
    fn the_line_of_the_box_stands_only_for_a_word_nothing_has() {
        assert!(!shows_empty(""), "nothing is asked");
        assert!(!shows_empty("   "));
        assert!(!shows_empty("alerts"), "a page has the word");
        assert!(shows_empty("zzqqxx"), "no page has it");
    }

    /// The index of this window covers every card of the column, each with its own words.
    #[test]
    fn the_index_has_no_id_twice() {
        let mut ids: Vec<&str> = PAGES.iter().map(|page| page.id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), before, "an id gates one card only");
        for page in PAGES {
            assert!(!page.label.is_empty() && !page.text.is_empty());
        }
        assert!(PAGES.iter().any(|page| page.id == "alerts"), "the last card of the column");
    }
}
