//! The usage card of the settings window: what has been counted, what the prices do, and the
//! rows themselves — the `usage.js` pane of the Studio. The counters and prices themselves are
//! `usage.rs`'s own tests.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::usage::{record, Price};
use iced_test::Simulator;

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    fllm_app::store::Store::default().save(&path).unwrap();
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
    Simulator::with_size(settings, iced::Size::new(1100.0, 900.0), app.view())
}

/// One drawing of the current window under `ui-snapshots/<name>.png`.
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
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
    let mut ui = Simulator::with_size(
        iced::Settings {
            fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
            default_font: fllm_app::app::UI_FONT,
            ..Default::default()
        },
        iced::Size::new(1100.0, 2400.0),
        app.view(),
    );
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
}

/// With counters and a price: the card says what the Studio says, the rows stand in its order,
/// the table carries every model and the totals line ends with the dollars.
#[test]
fn the_card_says_what_the_studio_says() {
    let (mut app, _dir) = app();
    record(&mut app.store_mut().usage, "gpt-6-mini", 1_000_000, 500_000);
    record(&mut app.store_mut().usage, "local-model", 1_500, 10);
    let mut row = Price {
        pattern: "gpt*".into(),
        input_per_million: 1.0,
        output_per_million: 2.0,
        note: String::new(),
        input_text: String::new(),
        output_text: String::new(),
    };
    row.refresh_text();
    let _ = app.update(Message::PricesLoaded(Ok(vec![row])));

    let _ = app.update(Message::Go(View::Settings));
    let mut ui = sim(&app);
    ui.find("Usage and cost").expect("the card of the Studio");
    ui.find(
        "Tokens are counted from the usage block of every answer. A local model costs nothing; \
         for a model behind a provider, add its price below and the cost appears here and in /cost.",
    )
    .expect("the words under the title");
    // 1,000,000 + 1,500 input tokens, 500,000 + 10 output, and only the priced model's dollars
    ui.find("1.00M in / 500.0k out in 2 answer(s) · about $2.00")
        .expect("the totals line of renderUsage");
    ui.find("Reset counters").expect("the reset button");
    ui.find("Prices (dollars per million tokens)").expect("the prices title");
    ui.find("No price rows. Local models need none.").ok();
    ui.find("gpt-6-mini").expect("the model in the table");
    ui.find("local-model").expect("the second model");
    ui.find("2").ok();
    ui.find("Save prices").expect("the save button");
    ui.find("+ Add price").expect("the add button");
    save_png(&app, "settings-usage");

    // the row's fields hold what the Studio's inputs hold
    ui.find("gpt*").expect("the pattern field shows its value");
    ui.find("1").expect("the input price as text (String(1) of the Studio)");
    ui.find("Remove").expect("the row can be removed");
}

/// Without counters the card says that nothing has been counted yet, and the totals line has no
/// dollar amount (no prices were entered).
#[test]
fn without_anything_counted_it_says_so() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    let mut ui = sim(&app);
    ui.find("0 in / 0 out in 0 answer(s) · no prices entered, so no dollar amount")
        .expect("the totals line without counters");
    ui.find("Nothing counted yet in this browser. Ask something in Chat or run the Code agent.")
        .expect("the empty table of the Studio");
    ui.find("No price rows. Local models need none.").expect("the empty prices of the Studio");
}

/// Adding a row puts the empty one of the Studio into the card, and removing it says what to do
/// next (`Remove` of usage.js).
#[test]
fn a_price_row_can_be_added_and_removed() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::PriceAdd);
    {
        let mut ui = sim(&app);
        ui.find("model pattern, e.g. gpt-6*").expect("the pattern placeholder of the Studio");
        ui.find("0").ok();
    }
    let _ = app.update(Message::PriceField(0, "pattern", "gpt-6*".into()));
    let _ = app.update(Message::PriceField(0, "input", "0.5".into()));
    let _ = app.update(Message::PriceField(0, "output", "1.5".into()));
    assert_eq!(app.prices[0].input_per_million, 0.5, "the field is a number too");
    let _ = app.update(Message::PriceRemove(0));
    assert!(app.prices.is_empty());
    assert_eq!(
        app.usage_status.as_deref(),
        Some("Removed. Press Save prices to write it."),
        "the words of the Studio after removing"
    );
    // Reset counters says so as well
    let _ = app.update(Message::UsageReset);
    assert_eq!(app.usage_status.as_deref(), Some("Counters reset."));
}
