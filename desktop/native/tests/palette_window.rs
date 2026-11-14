//! The command palette in the window: it opens on its keys, lists the commands in their groups
//! with the chats of the store, walks with the arrow keys, runs a row and closes, says *No results*
//! and *Nothing found for "…"* when nothing matches (the `searchModal` of `shell.html`).

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::store::{ChatRecord, Record, Role, Store};
use iced::keyboard::{key::Named, Key, Modifiers};
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
            .unwrap_or_else(|_| format!("{}/ui-snapshots", env!("CARGO_TARGET_TMPDIR"))),
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

/// Two chats: one that a title matches and one that only its text matches.
fn store_with_chats() -> Store {
    let mut store = Store::default();
    store.chats.push(ChatRecord::new(
        1,
        "Docker networks",
        vec![rec(Role::User, "How do I bridge a container?")],
    ));
    store.chats.push(ChatRecord::new(
        2,
        "Weekend plans",
        vec![rec(Role::User, "Let us talk about docker compose later")],
    ));
    store
}

/// The palette opens with the commands of the Studio in their groups, the foot with its help and
/// the count of the rows (the `open` of `palette.js`).
#[test]
fn the_palette_opens_with_the_commands_in_their_groups() {
    let (mut app, _dir) = app_with(store_with_chats());
    let _ = app.update(Message::PaletteOpen);
    assert!(app.palette_open, "the window is open");
    assert_eq!(app.palette_query, "", "the field starts empty");
    assert_eq!(app.palette_active, 0, "the first row is active");

    let mut ui = sim(&app);
    ui.find("Commands").expect("the group of the commands");
    // without a query the first seven commands come, then the chats (rankPalette of palette.js);
    // with a chat open that is New chat, Regenerate, the two exports and the three pages
    ui.find("New chat").expect("the first command");
    ui.find("Ctrl+N").expect("the keys of the shortcut on the row");
    ui.find("Regenerate the last answer").expect("a chat is open, so this row is there");
    ui.find("Export this chat as Markdown").expect("the export rows of an open chat");
    ui.find("Go to Code").expect("another command");
    ui.find("Open Models").expect("the seventh command");
    ui.find("Chats").expect("the group of the chats");
    ui.find("Docker networks").expect("a chat of the store");
    ui.find("Up Down to move, Enter to open, Esc to close").expect("the help of the foot");
    // seven commands and the two chats of the store are the rows, and the count says so
    ui.find("9 results").expect("the count says how many rows there are");
    save_png(&app, "palette-open");
    drop(ui);

    // a query brings the rows the seven-command window leaves out
    let _ = app.update(Message::PaletteQuery("tour".into()));
    {
        let mut ui = sim(&app);
        ui.find("Show the welcome tour").expect("the tour command by its query");
    }

    // the theme command says where it would go (the ternary of palette.js); it sits behind the
    // first seven commands while a chat is open, so the query brings it into view
    let theme_title = if app.dark() {
        "Switch to the light theme"
    } else {
        "Switch to the dark theme"
    };
    let _ = app.update(Message::PaletteQuery("theme".into()));
    let titles: Vec<&str> = app.palette_rows.iter().map(|(i, _)| i.title.as_str()).collect();
    println!("rows for theme: {titles:?} (dark={})", app.dark());
    {
        let mut ui2 = sim(&app);
        ui2.find(theme_title).expect("the title says where it would go");
    }
    let _ = app.update(Message::PaletteClose);
}

/// A query keeps what matches, the chats show the text around a hit, and nothing at all says so
/// (the `build` and the empty line of `palette.js`).
#[test]
fn a_query_keeps_what_matches_and_nothing_says_so() {
    let (mut app, _dir) = app_with(store_with_chats());
    let _ = app.update(Message::PaletteOpen);

    // the title matches: the row is kept with its keys
    let _ = app.update(Message::PaletteQuery("models".into()));
    assert_eq!(app.palette_rows.len(), 1, "only the command about models");
    assert_eq!(app.palette_rows[0].0.title, "Open Models");
    assert_eq!(app.palette_active, 0, "the first row is active after every query");

    // only the text of a chat matches: it is appended with the text around the hit
    let _ = app.update(Message::PaletteQuery("compose".into()));
    assert!(
        app.palette_rows.iter().any(|(i, _)| i.id == "chat-2"),
        "the chat whose text matched is there"
    );
    let row = app
        .palette_rows
        .iter()
        .find(|(i, _)| i.id == "chat-2")
        .expect("the chat row");
    assert!(
        row.0.sub.contains("docker compose"),
        "the snippet shows the hit: {:?}",
        row.0.sub
    );

    // nothing at all
    let _ = app.update(Message::PaletteQuery("qqqzzz".into()));
    assert!(app.palette_rows.is_empty(), "nothing matched");
    assert_eq!(app.palette_active, -1, "no row is active");
    {
        let mut ui = sim(&app);
        ui.find("Nothing found for \"qqqzzz\".")
            .expect("the empty line of the Studio");
        ui.find("No results").expect("the count of the Studio");
        save_png(&app, "palette-empty");
    }
    let _ = app.update(Message::PaletteClose);
    assert!(!app.palette_open, "the window closes");
}

/// The arrow keys walk the rows and wrap around (`paletteStep` of `palette.js`), a row runs and
/// closes the window, and Esc does the same (`closeOverlays` of `boot.js`).
#[test]
fn the_arrow_keys_walk_the_rows_and_a_row_runs() {
    let (mut app, _dir) = app_with(store_with_chats());
    let _ = app.update(Message::PaletteOpen);
    let count = app.palette_rows.len();

    let _ = app.update(Message::Key(Key::Named(Named::ArrowDown), Modifiers::empty()));
    assert_eq!(app.palette_active, 1, "ArrowDown takes the next row");
    let _ = app.update(Message::Key(Key::Named(Named::ArrowDown), Modifiers::empty()));
    let _ = app.update(Message::Key(Key::Named(Named::ArrowUp), Modifiers::empty()));
    assert_eq!(app.palette_active, 1, "ArrowUp walks back");
    let _ = app.update(Message::Key(Key::Named(Named::ArrowUp), Modifiers::empty()));
    assert_eq!(app.palette_active, 0, "up from the top of the first block");
    let _ = app.update(Message::Key(Key::Named(Named::ArrowUp), Modifiers::empty()));
    assert_eq!(app.palette_active, (count - 1) as i32, "it wraps to the last row");
    let _ = app.update(Message::Key(Key::Named(Named::Home), Modifiers::empty()));
    assert_eq!(app.palette_active, 0, "Home goes to the top");
    let _ = app.update(Message::Key(Key::Named(Named::End), Modifiers::empty()));
    assert_eq!(app.palette_active, (count - 1) as i32, "End goes to the bottom");

    // a plain key runs nothing: only a row does
    let _ = app.update(Message::Key(Key::Character("x".into()), Modifiers::empty()));
    assert!(app.palette_open, "a plain key leaves the window alone");

    // find the row that opens Models and run it
    let index = app
        .palette_rows
        .iter()
        .position(|(i, _)| i.id == "models")
        .expect("the command is in the list");
    let _ = app.update(Message::PaletteRun(index));
    assert_eq!(app.page(), View::Models, "the row ran and the Models page opened");
    assert!(!app.palette_open, "the window closed before the row ran");

    // Esc closes the palette (closeOverlays before anything else happens)
    let _ = app.update(Message::PaletteOpen);
    let _ = app.update(Message::Esc);
    assert!(!app.palette_open, "Esc closes the palette");
    assert_eq!(app.page(), View::Models, "and stops nothing else");
}

/// `Mod+K` opens the palette (the `search` shortcut of the Studio) and its rows carry the keys in
/// force — a changed shortcut is what the row shows.
#[test]
fn the_search_shortcut_opens_the_palette_with_the_keys_in_force() {
    let (mut app, _dir) = app_with(store_with_chats());
    let _ = app.update(Message::Key(Key::Character("k".into()), Modifiers::CTRL));
    assert!(app.palette_open, "Mod+K opens the palette");
    // the window takes the keys while it is open (like the input of palette.js), so the change
    // happens with it closed and the rows are built anew after it
    let _ = app.update(Message::Esc);
    assert!(!app.palette_open, "Esc closes it");

    let _ = app.update(Message::ScChange("newChat".into()));
    let _ = app.update(Message::Key(Key::Character("j".into()), Modifiers::CTRL));
    let _ = app.update(Message::PaletteOpen);
    {
        let mut ui = sim(&app);
        ui.find("Ctrl+J").expect("the row shows the keys in force");
        save_png(&app, "palette-changed-keys");
    }
    let _ = app.update(Message::Esc);
    assert!(!app.palette_open, "Esc closes it");
}
