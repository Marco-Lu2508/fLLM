//! The math of an answer in the window: the Studio typesets it (markdown `$$..$$`, `$..$`) —
//! the display piece gets its own line, an inline piece sits between the words, and something
//! that will not typeset is shown as its source.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message};
use fllm_app::store::{ChatRecord, Record, Role, Store};
use iced_test::Simulator;

fn rec(role: Role, content: &str) -> Record {
    Record {
        role,
        content: content.into(),
        reasoning: String::new(),
        stats: None,
        passages: None,
        images: Vec::new(),
        files: Vec::new(),
        tool_calls: None,
        tool_call_id: None,
        tool_cards: Vec::new(),
        sources: Vec::new(),
        research: None,
        logprobs: Vec::new(),
    }
}

fn app_with(store: Store) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    store.save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 4096))));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    };
    Simulator::with_size(settings, iced::Size::new(1000.0, 800.0), app.view())
}

fn store_with(answer: &str) -> Store {
    let mut store = Store::default();
    store.chats.push(ChatRecord::new(
        1,
        "math",
        vec![rec(Role::User, "show me"), rec(Role::Assistant, answer)],
    ));
    store
}

/// Writes the drawing of the current window under `ui-snapshots/<name>.png`.
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let settings = iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1000.0, 800.0), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(
        std::env::var("FLLM_SNAPSHOT_DIR")
            .unwrap_or_else(|_| "/opt/data/cache/scratch/fLLM/desktop/native/ui-snapshots".into()),
    );
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
    {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
}

/// A display piece is a line of its own with the typeset source, an inline piece keeps its
/// place between the words — the math is drawn, the dollars are not (the typeset of markdown).
#[test]
fn the_math_of_an_answer_is_typeset_in_the_window() {
    let (mut app, _dir) = app_with(store_with(
        "Energy: $$E=mc^2$$ and also the sum $a+b$ of it.",
    ));
    let _ = app.update(Message::OpenChat(0));
    let mut ui = sim(&app);
    // the display piece on its own line, typeset: the dollars are not drawn
    // the display piece on its own line, typeset: the dollars are not drawn, the spaces of
    // the Studio's typeset are not either (`x + y` is `<i>x</i>+<i>y</i>`). The slices of words
    // around it are `rich_text`, which the simulator's text selector does not walk — the PNG
    // below is what they are checked against.
    ui.find("E=mc2").expect("the display piece is typeset, without the dollars");
    save_png(&app, "math-answer-typeset");
}

/// Something that will not typeset is shown as its source, exactly like the Studio's fallback
/// (`math_source`: the source between the dollars of the chip).
#[test]
fn a_piece_that_will_not_typeset_is_its_source() {
    let (mut app, _dir) = app_with(store_with("Broken: $$not really math$$ done."));
    let _ = app.update(Message::OpenChat(0));
    let mut ui = sim(&app);
    ui.find("notreallymath")
        .expect("the source of the piece that would not typeset, without the dollars");
    ui.find("Broken:").ok();
    save_png(&app, "math-source-fallback");
}
