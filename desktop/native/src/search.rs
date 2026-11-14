//! Settings > Search: the embeddings server for the document search and the engines of the web
//! search (`searchSettings` of `rag.js`, which builds the pane, and `searchengines.js`, which
//! appends its card to it).

use crate::app::Message;
use crate::models::card;
use crate::theme::{self, Tokens};
use iced::widget::{button, column, container, pick_list, row, text, text_input, Space};
use iced::{Alignment, Element, Font, Length};

/// The description of the *Document search* card (its `<div class="app-item-desc">`, with the
/// `mono` words set like the rest of the sentence).
pub const DOC_HELP: &str = "Documents are cut into passages and ranked by keyword (BM25) in this browser. With an embeddings server the ranking becomes hybrid: half keyword, half meaning, so a passage is found even when the question uses another word for the same thing. Nothing to install: run fllm-rs serve --encoder <model> or point it at any OpenAI-compatible /v1/embeddings. Leave the field empty to keep keyword-only search.";

/// The description of the *Web search engines* card.
pub const ENGINES_HELP: &str = "DuckDuckGo and Wikipedia need no key and no server, so search works the moment the app starts. Add your own engine if you run SearXNG or have a Brave key: every engine is asked, the results are merged, and the tool reports which engine failed.";

/// What a template literal writes for a value the answer does not carry (`${data.file}` of
/// `searchengines.js`, `${r.found} Treffer`, `${r.error}`): the word `undefined`.
fn js_undefined(v: Option<String>) -> String {
    v.unwrap_or_else(|| "undefined".into())
}

/// The engines the page offers in its select (`#seKind`): the name in the request and the label
/// of the option.
pub const KINDS: [(&str, &str); 4] = [
    ("ddg", "DuckDuckGo"),
    ("wikipedia", "Wikipedia"),
    ("searxng", "SearXNG"),
    ("brave", "Brave"),
];

/// The label of a kind of engine: nothing is selected for a kind the page does not offer.
pub fn kind_label(id: &str) -> Option<&'static str> {
    KINDS.iter().find(|(name, _)| *name == id).map(|(_, label)| *label)
}

/// The name in the request for a label of the select.
pub fn kind_id(label: &str) -> &'static str {
    KINDS.iter().find(|(_, name)| *name == label).map(|(id, _)| *id).unwrap_or("ddg")
}

/// One engine as it is stored (`engines` of `GET /api/studio/search/engines`).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Engine {
    pub kind: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub key: String,
}

impl Engine {
    /// What the row writes as its title: the name it was given, or the kind when it has none
    /// (`e.name || e.kind`).
    pub fn title(&self) -> String {
        if self.name.is_empty() {
            self.kind.clone()
        } else {
            self.name.clone()
        }
    }

    /// What the row writes under it: the kind, then the address, then `key set` when there is a
    /// key — each part only when it is there.
    pub fn detail(&self) -> String {
        let mut out = self.kind.clone();
        if !self.url.is_empty() {
            out.push_str(&format!(" · {}", self.url));
        }
        if !self.key.is_empty() {
            out.push_str(" · key set");
        }
        out
    }
}

/// What the server answers for the engines: the engines it has and where it keeps them.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct EngineAnswer {
    #[serde(default)]
    pub engines: Vec<Engine>,
    #[serde(default)]
    pub file: Option<String>,
}

/// The line under the list after a read (`say(load)` of `searchengines.js`).
pub fn engines_line(answer: &EngineAnswer) -> String {
    format!(
        "{} engine(s) · saved in {}",
        answer.engines.len(),
        js_undefined(answer.file.clone())
    )
}

/// What an engine of the list needs before it can be added (the two checks of `#seAdd`).
pub fn add_engine(
    kind: &str,
    name: &str,
    url: &str,
    key: &str,
) -> Result<Engine, String> {
    if kind == "searxng" && url.is_empty() {
        return Err("A SearXNG engine needs its address.".into());
    }
    if kind == "brave" && key.is_empty() {
        return Err("A Brave engine needs an API key.".into());
    }
    Ok(Engine {
        kind: kind.to_string(),
        name: name.to_string(),
        url: url.to_string(),
        key: key.to_string(),
    })
}

/// What `GET` and `POST /api/studio/embeddings` answer (`route_status` and `route_save`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct EmbedConfig {
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub configured: bool,
}

/// What `POST /api/studio/embed` answers: the vectors of the two sentences, how many numbers
/// each holds and the model behind them.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Vectors {
    #[serde(default)]
    pub vectors: Vec<Vec<f32>>,
    #[serde(default)]
    pub dims: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
}

/// The similarity of two vectors, the way `ragCosine` of `rag.js` figures it: not two vectors,
/// vectors of different lengths and vectors with nothing in them all give `0`.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb).sqrt()
}

/// The line *Test* writes on a server that answered (the sentence of `btnEmbedTest`).
pub fn embed_line(v: &Vectors) -> String {
    let close = if v.vectors.len() == 2 {
        cosine(&v.vectors[0], &v.vectors[1])
    } else {
        0.0
    };
    format!(
        "{} vector(s), {} numbers each{}. Similarity of two similar sentences: {:.3}.",
        v.vectors.len(),
        v.dims.unwrap_or(0),
        match &v.model {
            Some(model) => format!(", model {model}"),
            None => String::new(),
        },
        close
    )
}

/// What `POST /api/studio/search` answers for the test: how every engine did, and what came back.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct TestAnswer {
    #[serde(default)]
    pub engines: Vec<EngineLine>,
    #[serde(default)]
    pub results: Vec<ResultLine>,
    #[serde(default)]
    pub error: Option<serde_json::Value>,
}

/// One engine of the answer of the test (`r.ok ? ... : ...`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct EngineLine {
    #[serde(default)]
    pub ok: bool,
    #[serde(default)]
    pub engine: String,
    #[serde(default)]
    pub found: Option<u64>,
    #[serde(default)]
    pub error: Option<String>,
}

/// One result of the answer of the test (the first three are listed by their title).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct ResultLine {
    #[serde(default)]
    pub title: String,
}

/// What the *Test a search* button writes: one line per engine — the count of its hits with the
/// word *Treffer* as the Studio writes it — a blank line and the first three titles, or the
/// sentence of the answer when it holds nothing (`#seTest` of `searchengines.js`).
pub fn test_text(a: &TestAnswer) -> String {
    // `[...lines, '', ...hits].join('\n')` of the page: the blank line is there in every case,
    // so an answer with engines and one with results both keep the line the Studio gives them
    let mut parts: Vec<String> = a
        .engines
        .iter()
        .map(|line| {
            if line.ok {
                format!("{}: {} Treffer", line.engine, js_undefined(line.found.map(|n| n.to_string())))
            } else {
                format!("{}: {}", line.engine, js_undefined(line.error.clone()))
            }
        })
        .collect();
    parts.push(String::new());
    for (i, hit) in a.results.iter().take(3).enumerate() {
        parts.push(format!("{}. {}", i + 1, hit.title));
    }
    let joined = parts.join("\n");
    if !joined.is_empty() {
        return joined;
    }
    match &a.error {
        Some(serde_json::Value::Object(map)) => map
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("No answer.")
            .to_string(),
        Some(serde_json::Value::String(s)) => s.clone(),
        _ => "No answer.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The rows of the list: the name it was given, or the kind when it has none, and what
    /// follows it (`e.name || e.kind` and the three parts of the second span).
    #[test]
    fn a_row_reads_the_way_the_page_writes_it() {
        let full = Engine {
            kind: "searxng".into(),
            name: "mine".into(),
            url: "http://127.0.0.1:8888".into(),
            key: String::new(),
        };
        assert_eq!(full.title(), "mine");
        assert_eq!(full.detail(), "searxng · http://127.0.0.1:8888");

        let bare = Engine { kind: "brave".into(), key: "k".into(), ..Default::default() };
        assert_eq!(bare.title(), "brave", "no name, so the kind is the title");
        assert_eq!(bare.detail(), "brave · key set");

        let plain = Engine { kind: "ddg".into(), ..Default::default() };
        assert_eq!(plain.detail(), "ddg", "nothing follows the kind when there is nothing");
    }

    /// The two checks of Add: a SearXNG without an address and a Brave without a key are not
    /// added and say so; everything else is.
    #[test]
    fn an_engine_is_added_when_what_it_needs_is_there() {
        assert_eq!(
            add_engine("searxng", "", "", "").unwrap_err(),
            "A SearXNG engine needs its address."
        );
        assert_eq!(
            add_engine("brave", "", "https://api.example.com", "").unwrap_err(),
            "A Brave engine needs an API key."
        );
        let ddg = add_engine("ddg", "mine", "", "").expect("DuckDuckGo needs nothing");
        assert_eq!(ddg.kind, "ddg");
        assert_eq!(ddg.name, "mine");
        assert!(ddg.url.is_empty() && ddg.key.is_empty());
    }

    /// The line under the list after a read (`say(load)` of `searchengines.js`), and the word a
    /// template literal writes when the answer carries no file.
    #[test]
    fn the_line_counts_the_engines_and_names_the_file() {
        let answer: EngineAnswer = serde_json::from_value(json!({
            "engines": [{"kind": "ddg"}],
            "file": "/home/m/.fllm/search.json"
        }))
        .expect("an answer of the engines");
        assert_eq!(engines_line(&answer), "1 engine(s) · saved in /home/m/.fllm/search.json");
        assert_eq!(
            engines_line(&EngineAnswer::default()),
            "0 engine(s) · saved in undefined",
            "what a template literal writes for a file that is not there"
        );
    }

    /// The similarity of two vectors, as `ragCosine` of `rag.js` figures it, and the line the
    /// Test button writes with it.
    #[test]
    fn the_similarity_is_the_one_of_the_studio() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-9);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0, "vectors of different lengths");
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 2.0]), 0.0, "a vector with nothing in it");

        let vectors: Vectors = serde_json::from_value(json!({
            "vectors": [[1.0, 0.0], [1.0, 0.0]],
            "dims": 2,
            "model": "nomic-embed-text"
        }))
        .expect("an answer of /api/studio/embed");
        assert_eq!(
            embed_line(&vectors),
            "2 vector(s), 2 numbers each, model nomic-embed-text. Similarity of two similar sentences: 1.000."
        );
        let without = Vectors { vectors: vec![vec![1.0]], dims: None, model: None };
        assert_eq!(
            embed_line(&without),
            "1 vector(s), 0 numbers each. Similarity of two similar sentences: 0.000.",
            "one vector is not two, and no model leaves its half of the sentence out"
        );
    }

    /// What the Test button writes: the lines of the engines with the word *Treffer* as the page
    /// writes it, the blank line, the first three titles — and the sentence of an answer that
    /// holds nothing.
    #[test]
    fn the_test_writes_the_lines_of_the_page() {
        let answer: TestAnswer = serde_json::from_value(json!({
            "engines": [
                {"ok": true, "engine": "ddg", "found": 3},
                {"ok": false, "engine": "brave", "error": "429 rate limited"}
            ],
            "results": [
                {"title": "First"}, {"title": "Second"}, {"title": "Third"}, {"title": "Fourth"}
            ]
        }))
        .expect("an answer of the search");
        assert_eq!(
            test_text(&answer),
            "ddg: 3 Treffer\nbrave: 429 rate limited\n\n1. First\n2. Second\n3. Third",
            "the blank line is the one the page puts between them, and only three results are listed"
        );

        // an answer with results and no engine lines keeps the blank line the page puts first
        let hits: TestAnswer = serde_json::from_value(json!({"results": [{"title": "Only"}]}))
            .expect("an answer without engines");
        assert_eq!(test_text(&hits), "\n1. Only");

        let nothing: TestAnswer = serde_json::from_value(json!({"error": {"message": "no search key"}}))
            .expect("an answer with only an error");
        assert_eq!(test_text(&nothing), "no search key", "`data.error.message`");
        assert_eq!(test_text(&TestAnswer::default()), "No answer.");
    }

    /// The select of the kinds offers what the page offers, and its labels come back to their
    /// names.
    #[test]
    fn the_kinds_of_the_select_are_the_four_of_the_page() {
        assert_eq!(KINDS.len(), 4);
        assert_eq!(kind_label("ddg"), Some("DuckDuckGo"));
        assert_eq!(kind_label("wikipedia"), Some("Wikipedia"));
        assert_eq!(kind_label("searxng"), Some("SearXNG"));
        assert_eq!(kind_label("brave"), Some("Brave"));
        assert_eq!(kind_label("yahoo"), None, "a kind the page does not offer");
        assert_eq!(kind_id("bash / zsh"), "ddg", "an unknown label starts at the first one");
        assert_eq!(kind_id("Brave"), "brave");
    }
}

/// What the pane holds for both cards.
#[derive(Clone, Copy)]
pub struct SearchView<'a> {
    /// The box of the embeddings server and the line under it.
    pub embed_url: &'a str,
    pub embed_status: &'a str,
    /// The list and the four fields over its buttons.
    pub engines: &'a [Engine],
    pub kind: &'a str,
    pub name: &'a str,
    pub url: &'a str,
    pub key: &'a str,
    pub status: &'a str,
}

/// A text box of the cards (`ui-input mono` of the page).
fn input<'a>(
    t: Tokens,
    placeholder: &'a str,
    value: &'a str,
    typed: fn(String) -> Message,
) -> Element<'a, Message> {
    text_input(placeholder, value)
        .on_input(typed)
        .size(13)
        .padding([6, 10])
        .font(Font::MONOSPACE)
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
        })
        .into()
}

/// The line between two engines (`border-bottom: 1px solid var(--border)`).
fn divider(t: Tokens) -> Element<'static, Message> {
    container(Space::new())
        .width(Length::Fill)
        .height(Length::Fixed(1.0))
        .style(move |_| container::Style {
            background: Some(iced::Background::Color(t.separator)),
            ..Default::default()
        })
        .into()
}

/// One engine of the list with its Remove button.
fn engine_row<'a>(t: Tokens, e: &Engine, index: usize) -> Element<'a, Message> {
    let title = e.title();
    let detail = e.detail();
    column![
        row![
            text(title).size(12).color(t.fg),
            text(detail).size(11).font(Font::MONOSPACE).color(t.fg_soft),
            Space::new().width(Length::Fill),
            button(text("Remove").size(12))
                .on_press(Message::EngRemove(index))
                .padding([2, 8])
                .style(theme::ghost_button(t)),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
        divider(t),
    ]
    .padding([4, 0])
    .spacing(4)
    .into()
}

/// The two cards of `spane-search`: what the document search reaches, and the engines of the web
/// search.
pub fn section<'a>(t: Tokens, v: SearchView<'a>) -> Element<'a, Message> {
    // ---- Document search (`searchSettings` of `rag.js`) --------------------------------
    let mut doc = column![].spacing(10);
    doc = doc.push(text(DOC_HELP).size(13).color(t.fg_soft));
    doc = doc.push(
        row![
            input(t, "http://127.0.0.1:8100", v.embed_url, Message::EmbedUrl),
            button(text("Save").size(13))
                .on_press(Message::EmbedSave)
                .padding([6, 12])
                .style(theme::secondary_button(t)),
            button(text("Test").size(13))
                .on_press(Message::EmbedTest)
                .padding([6, 12])
                .style(theme::ghost_button(t)),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    );
    doc = doc.push(
        text(v.embed_status.to_string())
            .size(11)
            .font(Font::MONOSPACE)
            .color(t.fg_soft),
    );

    // ---- Web search engines (`searchengines.js`) ---------------------------------------
    let mut engines = column![].spacing(10);
    engines = engines.push(text(ENGINES_HELP).size(13).color(t.fg_soft));
    if v.engines.is_empty() {
        engines = engines.push(text("No engine yet.").size(13).color(t.fg_soft));
    } else {
        let mut list = column![].spacing(0);
        for (i, engine) in v.engines.iter().enumerate() {
            list = list.push(engine_row(t, engine, i));
        }
        engines = engines.push(list);
    }

    // the row over the buttons: the kind to add, its name, its address, its key, and Add
    engines = engines.push(
        row![
            pick_list(
                KINDS.iter().map(|(_, label)| *label).collect::<Vec<_>>(),
                kind_label(v.kind),
                |label: &'static str| Message::EngKind(kind_id(label).to_string()),
            )
            .padding([6, 10]),
            input(t, "name (optional)", v.name, Message::EngName),
            input(t, "http://127.0.0.1:8888", v.url, Message::EngUrl),
            input(t, "API key (Brave)", v.key, Message::EngKey),
            button(text("Add").size(13))
                .on_press(Message::EngAdd)
                .padding([6, 12])
                .style(theme::ghost_button(t)),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    );
    engines = engines.push(
        row![
            button(text("Save engines").size(13))
                .on_press(Message::EngSave)
                .padding([6, 12])
                .style(theme::primary_button(t)),
            button(text("Test a search").size(13))
                .on_press(Message::EngTest)
                .padding([6, 12])
                .style(theme::secondary_button(t)),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    );
    engines = engines.push(
        text(v.status.to_string())
            .size(11)
            .font(Font::MONOSPACE)
            .color(t.fg_soft),
    );

    column![
        card(t, "Document search", doc.into()),
        card(t, "Web search engines", engines.into()),
    ]
    .spacing(14)
    .into()
}
