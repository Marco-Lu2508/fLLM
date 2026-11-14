//! Settings search in the window: the box over the column of settings pages keeps what every
//! word of the question is in, the line when nothing does, the mark after Enter and the row the
//! palette offers for it (`settingssearch.js`).

use fllm_app::app::{App, Message, View};
use fllm_app::store::Store;
use iced::Size;
use iced::keyboard::{key::Named, Key, Modifiers};
use iced_test::Simulator;

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
    Store { theme: theme.into(), ..Default::default() }
        .save(&path)
        .unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Err(
        "no server at http://127.0.0.1:9: connection refused".into(),
    )));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), Size::new(1280.0, 12000.0), app.view())
}

/// The box sits over the column and every card stands while no question is asked.
#[test]
fn the_column_starts_whole_with_the_box_over_it() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));

    assert!(app.settings_query.is_empty(), "the box starts empty");
    let mut ui = sim(&app);
    for title in [
        "Settings",
        "General",
        "Answers",
        "Personalization",
        "Appearance",
        "Document search",
        "Tools (MCP)",
        "Skills",
        "Plugins",
        "Hooks",
        "Sub-agents",
        "Dictation",
        "Reading aloud",
        "Alerts",
    ] {
        ui.find(title).expect("a card of the column");
    }
    assert!(
        ui.find("No setting matches.").is_err(),
        "the line of the page stands only while a question is asked"
    );
}

/// The question keeps what it finds — every word of it — and says so when nothing has them.
#[test]
fn the_question_keeps_what_it_finds() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));

    let _ = app.update(Message::SettingsQuery("plugins".into()));
    let mut ui = sim(&app);
    ui.find("Plugins").expect("the card whose page has the word");
    assert!(ui.find("Alerts").is_err(), "the card without it is not drawn");
    assert!(
        ui.find("Skills").is_err(),
        "a word no page of these has takes the others out too"
    );
    drop(ui);

    // another question, other cards
    let _ = app.update(Message::SettingsQuery("theme".into()));
    let mut ui = sim(&app);
    ui.find("Appearance").expect("the card whose page has the word");
    assert!(ui.find("General").is_err(), "the card before it is not drawn");
    drop(ui);

    // a question nothing has says the one line of the page
    let _ = app.update(Message::SettingsQuery("zzqqxx".into()));
    let mut ui = sim(&app);
    ui.find("No setting matches.").expect("the line of the page");
    assert!(ui.find("Alerts").is_err(), "and no card stands");
    drop(ui);

    // an emptied box brings the column back whole
    let _ = app.update(Message::SettingsQuery(String::new()));
    let mut ui = sim(&app);
    ui.find("Alerts").expect("every card stands again");
    assert!(ui.find("No setting matches.").is_err());
    drop(ui);
}

/// Enter marks the best page for 3200 ms, Escape empties the box on this page, and the palette
/// asks the same question of the same box.
#[test]
fn enter_marks_the_best_page_and_the_palette_asks_the_box() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));

    let _ = app.update(Message::SettingsQuery("alerts".into()));
    let _ = app.update(Message::SettingsEnter);
    assert_eq!(
        app.settings_hit,
        Some("alerts"),
        "the page whose label starts with the word is the best one"
    );
    // the mark stands until its own time is up, and a mark of another question stays
    let _ = app.update(Message::SettingsHitClear("general"));
    assert_eq!(app.settings_hit, Some("alerts"), "the time of another mark");
    let _ = app.update(Message::SettingsHitClear("alerts"));
    assert!(app.settings_hit.is_none(), "the time of this one is up");

    // Escape empties the box while this page is open
    let _ = app.update(Message::Key(Key::Named(Named::Escape), Modifiers::empty()));
    assert!(app.settings_query.is_empty(), "the box of the page");
    let _ = app.update(Message::SettingsQuery("alerts".into()));
    let _ = app.update(Message::Go(View::Chat));
    let _ = app.update(Message::Key(Key::Named(Named::Escape), Modifiers::empty()));
    assert_eq!(
        app.settings_query, "alerts",
        "the box of another page is not the box this key empties"
    );

    // the weak row of the palette asks the box, and what it asks lands in the settings
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::PaletteOpen);
    let _ = app.update(Message::PaletteQuery("theme".into()));
    let mut ui = sim(&app);
    ui.find("Search the settings for \"theme\"")
        .expect("the weak row the palette offers for a setting");
    drop(ui);
    let _ = app.update(Message::SettingsSearch("mcp servers".into()));
    assert_eq!(
        app.page(),
        View::Settings,
        "the surface of the settings shows it"
    );
    assert_eq!(app.settings_query, "mcp servers", "the box holds the question");
    assert_eq!(
        app.settings_hit,
        Some("mcp"),
        "the best page of the question is marked"
    );
}
