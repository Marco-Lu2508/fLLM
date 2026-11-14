//! Settings > Tokenizer in the window: the two cards of `spane-tokenizer` — how the model splits
//! text into tokens with the wall of its pieces, and the text it receives for a chat.

use fllm_app::app::{App, Message, View};
use fllm_app::localapi::{Tokenized, JINJA_MESSAGES, TOKEN_TEXT};
use fllm_app::store::Store;
use iced::Size;
use iced_test::Simulator;
use serde_json::json;

fn settings() -> iced::Settings {
    iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    }
}

fn app(theme: &str) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Err("no server at http://127.0.0.1:9: connection refused".into())));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), Size::new(1280.0, 6000.0), app.view())
}

/// The two cards start the way the page starts them, and nothing is sent until something is typed.
#[test]
fn the_two_cards_start_the_way_the_page_starts_them() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));

    let mut ui = sim(&app);
    ui.find("See how the model splits text into tokens").expect("the title of the first card");
    ui.find("Include the special start token").expect("the switch under the box");
    ui.find("0 tokens").expect("the line over the wall before anything was sent");
    ui.find("See the text the model receives for a chat").expect("the title of the second card");
    ui.find("Show the text").expect("the button of the second card");
    assert_eq!(app.tok_text.text(), TOKEN_TEXT, "the sentence the box starts with");
    assert_eq!(app.jinja_messages.text(), JINJA_MESSAGES, "the messages the box starts with");
    assert!(app.tok_special, "the switch starts on");
    assert!(app.tok_tokens.is_empty(), "the wall is empty until the server answered");
}

/// An answer of the server fills the line with tokens and bytes and the wall with its pieces.
#[test]
fn an_answer_fills_the_line_and_the_wall() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    let answer: Tokenized = serde_json::from_value(json!({
        "tokens": [
            {"id": 151643, "piece": "<bos>"},
            {"id": 9906, "piece": "Hello"},
            {"id": 0, "piece": "!"},
            42
        ],
        "count": 4,
        "max_model_len": 4096
    }))
    .expect("the answer of /tokenize");
    let _ = app.update(Message::TokLoaded(Ok(answer)));

    // 23 bytes of the sentence the box holds (the line counts bytes, not characters)
    let bytes = TOKEN_TEXT.len();
    let mut ui = sim(&app);
    ui.find(format!("4 tokens · {bytes} bytes").as_str()).expect("the line of `runTokenize`");
    ui.find("<bos>").expect("the first piece in its pill");
    ui.find("Hello").expect("the second one");
    ui.find("#151643").expect("the id under a piece");
    ui.find("#42").expect("an answer without pieces shows its id as text");
    drop(ui);

    // the switch is off now: the next call carries it
    let _ = app.update(Message::TokSpecial(false));
    assert!(!app.tok_special, "the switch followed the click");
}

/// The second card: a box that is no JSON says so, and the answer of the server is drawn.
#[test]
fn the_second_card_says_what_went_wrong_and_draws_the_answer() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    // the page parses the box inside its `try`, so a box that is no JSON says so and sends nothing
    app.jinja_messages = iced::widget::text_editor::Content::with_text("not json at all");
    let _ = app.update(Message::JinjaRender);
    assert!(
        app.jinja_out.starts_with("Invalid JSON:"),
        "the line of the Studio for a box that is no JSON: {}",
        app.jinja_out
    );

    // the answer of the server is drawn as it came
    let _ = app.update(Message::JinjaRendered(Ok("<bos> You are a helpful assistant.".into())));
    let mut ui = sim(&app);
    ui.find("<bos> You are a helpful assistant.").expect("the prompt of the template");
    drop(ui);

    // a server that does not answer lands in the same line (the `catch` of `runJinjaTemplate`
    // wraps the fetch too)
    let _ = app.update(Message::JinjaRendered(Err("no server at http://127.0.0.1:9".into())));
    let mut ui = sim(&app);
    ui.find("Invalid JSON: no server at http://127.0.0.1:9").expect("the line of a failed render");
}
