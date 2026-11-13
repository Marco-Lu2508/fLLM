//! The appearance card in the settings window: the theme of the Studio, its eight accents with the
//! chosen one ringed, the base size of the text and the two switches — the `Appearance` card of
//! `shell.html`. The colours themselves are `theme.rs`'s own tests.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::theme::{ACCENTS, Tokens};
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

/// One drawing of the window under `ui-snapshots/<name>.png`.
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
        iced::Size::new(1100.0, 2600.0),
        app.view(),
    );
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
}

/// The card says what the Studio's card says: its title, the five rows with their descriptions,
/// and the eight accents by the names the Studio gives them.
#[test]
fn the_card_says_what_the_studio_says() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::Go(View::Settings));
    let mut ui = sim(&app);
    ui.find("Appearance").expect("the card of shell.html");
    ui.find("By default the page follows the light or dark setting of your system.")
        .expect("the words of the Theme row");
    ui.find("Colors the sidebar and the controls.").expect("the words of the Accent row");
    ui.find("Base size of the text in the whole interface.").expect("the words of the Font size row");
    ui.find("Tints your messages in the accent color.").expect("the words of the tint row");
    ui.find("Shows tokens per second and the time to the first token below every answer.")
        .expect("the words of the speed row");
    // the selects show their defaults; the choices themselves are what the store says (the
    // simulator does not read a closed select as text)
    assert_eq!(app.store().theme.as_str(), "", "Same as the system is the default of selThemeMode");
    assert_eq!(app.store().font_size, 15, "Medium (15px) is the default of selFontSize");
    for accent in ACCENTS {
        ui.find(accent.name).expect("the eight accents of core.js");
    }
    save_png(&app, "settings-appearance");
}

/// Choosing an accent is what the whole page draws with: the accent of the dark theme is its own
/// colour, the light one its lighter twin, and the text on it follows the Studio's luminance.
#[test]
fn choosing_an_accent_recolours_the_page() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::LookTheme(String::from("dark")));
    let _ = app.update(Message::LookAccent(String::from("emerald")));
    assert_eq!(app.store().accent, "emerald", "the choice is kept");
    assert!(app.dark(), "the theme of the card");
    assert_eq!(app.tokens().primary, ACCENTS[1].primary, "the dark theme takes the accent itself");
    assert_eq!(
        app.tokens().primary_soft,
        fllm_app::theme::Tokens::new(true, "emerald").primary_soft,
        "the whole app draws with it (tokens() is the app's own)"
    );

    let _ = app.update(Message::LookTheme(String::from("light")));
    assert_eq!(app.tokens().primary, ACCENTS[1].primary_light, "the light theme takes the lighter twin");

    // an unknown value falls back to Copper, as the Studio does
    let _ = app.update(Message::LookAccent(String::from("nope")));
    assert_eq!(app.store().accent, "nope", "what was clicked is kept");
    assert_eq!(
        Tokens::new(false, app.store().accent.as_str()).primary,
        ACCENTS[0].primary_light,
        "but the drawing falls back to Copper"
    );
}

/// The three other choices of the card land in the store and change what they say: the base size
/// of the text, the tint of the messages and the speed under an answer.
#[test]
fn the_three_other_choices_reach_the_store_and_the_view() {
    let (mut app, _dir) = app();
    assert_eq!(app.store().font_size, 15, "the medium of the Studio");
    assert!(app.store().colored_bubble, "tint is on by default (chkColoredBubble)");
    assert!(app.store().show_speed, "the speed shows by default (chkShowSpeed)");

    let _ = app.update(Message::LookFontSize(18));
    let _ = app.update(Message::LookTint(false));
    let _ = app.update(Message::LookSpeed(false));
    assert_eq!(app.store().font_size, 18, "the base size of the Studio's selFontSize");
    assert!(!app.store().colored_bubble, "the bubble falls back to the Studio's secondary");
    assert!(!app.store().show_speed, "the line under an answer goes away");

    // with the tint off the bubble is the `--secondary` of the Studio, with it on the accent
    let tinted = fllm_app::theme::bubble(app.tokens(), true);
    let plain = fllm_app::theme::bubble(app.tokens(), false);
    let style_on = (tinted)(&iced::Theme::Light);
    let style_off = (plain)(&iced::Theme::Light);
    assert_ne!(
        style_on.background, style_off.background,
        "the two switches give the bubble two colours"
    );

    // the card's switches follow the store (the labels of segmented())
    let _ = app.update(Message::Go(View::Settings));
    let mut ui = sim(&app);
    ui.find("Show generation speed").expect("the speed row is where the switch lives");
}
