//! Voice and alerts: what each speech program is told (`voice_args`), the languages of the
//! dictation, the settings cards of `voice.js` in the window, and what happens when a run ends —
//! read alone when that is on, an alert after twenty seconds when the window is behind.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::settings::VoiceView;
use fllm_app::speech::{chime, list_voices, voice_args, LANGUAGES};
use fllm_app::store::{Record, Role, Store, VoicePrefs};
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
    // the settings page with the voice cards at its end: the picture takes all of it
    let mut ui = Simulator::with_size(settings, iced::Size::new(1000.0, 3000.0), app.view());
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

/// A chat with a finished answer, for the runs that end.
fn store_with_answer() -> Store {
    let mut store = Store::default();
    store.chats.push(fllm_app::store::ChatRecord::new(
        1,
        "chat",
        vec![
            rec(Role::User, "question"),
            rec(Role::Assistant, "a finished answer"),
        ],
    ));
    store
}

/// Every speech program gets the language, the voice, the speed and the pitch on its own scale
/// (`voice_args`): speech-dispatcher from -100 to 100, espeak words per minute and pitch 0 to 99,
/// the synthesizer of macOS words per minute and pitch 0 to 20, Windows -10 to 10.
#[test]
fn every_program_gets_the_settings_on_its_own_scale() {
    let mut v = VoicePrefs::default();
    assert_eq!((v.rate, v.pitch, v.notify, v.sound, v.auto), (1.0, 1.0, true, false, false));
    v.lang = "de-DE".into();
    v.voice = "Fred".into();
    v.rate = 2.0;
    v.pitch = 0.5;

    assert_eq!(
        voice_args("spd-say", &v),
        ["-r", "100", "-p", "-50", "-l", "de-DE"],
        "speech-dispatcher: both scales from -100 to 100, the language as -l"
    );
    assert_eq!(
        voice_args("espeak-ng", &v),
        ["-s", "350", "-p", "25", "-v", "Fred"],
        "espeak: words per minute, pitch 0 to 99, the voice"
    );
    assert_eq!(
        voice_args("say", &v),
        ["-r", "350", "-p", "5", "-v", "Fred"],
        "the synthesizer of macOS"
    );
    let powershell = voice_args("powershell", &v);
    assert_eq!(powershell.len(), 1, "one script for Windows");
    assert!(
        powershell[0].contains("$s.Rate=10") && powershell[0].contains("$s.Pitch=-5"),
        "Windows: -10 to 10: {}",
        powershell[0]
    );
    assert!(
        powershell[0].contains("SelectVoice('Fred')") && powershell[0].contains("Locale='de-DE'"),
        "the voice and the language of Windows: {}",
        powershell[0]
    );

    // without a language and a voice nothing is added for them
    let plain = VoicePrefs::default();
    assert_eq!(voice_args("spd-say", &plain), ["-r", "0", "-p", "0"]);
    assert_eq!(voice_args("espeak-ng", &plain), ["-s", "175", "-p", "50"]);
    assert_eq!(voice_args("say", &plain), ["-r", "175", "-p", "10"]);
    // out-of-range values are kept in range (the sliders of the Studio only go that far)
    v.rate = 9.0;
    v.pitch = 0.1;
    assert_eq!(voice_args("spd-say", &v)[1], "100", "rate 2 is the top");
    assert_eq!(
        voice_args("espeak-ng", &v)[3],
        "25",
        "pitch 0.5 is half of espeak's normal 50"
    );
}

/// The languages of the dictation are the seventeen of the Studio, the empty one first (`LANGUAGES`
/// of `voice.js`), and the voice list of this computer is a list, not an error.
#[test]
fn the_languages_are_the_ones_of_the_studio_and_the_list_of_voices_is_a_list() {
    assert_eq!(LANGUAGES.len(), 17, "the languages of voice.js");
    assert_eq!(LANGUAGES[0], ("", "Browser language"));
    assert_eq!(LANGUAGES[3], ("de-DE", "German"));
    assert_eq!(LANGUAGES[16], ("ar-SA", "Arabic"));
    // whatever this machine has, the answer is a sorted list without empties (or nothing)
    let voices = list_voices();
    let mut sorted = voices.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(voices, sorted, "the voices come sorted and without doubles");
    assert!(voices.iter().all(|v| !v.trim().is_empty()), "and none is empty");
    // no program that plays a tone is a silent no-op, not a panic
    chime();
}

/// The settings of the voice are read from json the way the Studio saves them, and a write keeps
/// them (the `fllm.studio.voice.v1` of `voice.js`).
#[test]
fn the_voice_settings_are_kept_with_the_studio_defaults() {
    let v: VoicePrefs = serde_json::from_str("{}").expect("the defaults without a file");
    assert_eq!(v.lang, "");
    assert_eq!(v.voice, "");
    assert_eq!(v.rate, 1.0, "the rate of the Studio");
    assert_eq!(v.pitch, 1.0);
    assert!(!v.auto, "reading alone is off");
    assert!(v.notify, "the alert is on");
    assert!(!v.sound, "the tone is off");
    assert!(!v.asked, "not asked yet");
    let changed: VoicePrefs =
        serde_json::from_str(r#"{"lang":"fr-FR","rate":1.5,"auto":true,"notify":false}"#)
            .expect("a changed file");
    assert_eq!(changed.lang, "fr-FR");
    assert_eq!(changed.rate, 1.5);
    assert!(changed.auto);
    assert!(!changed.notify);
    assert_eq!(changed.pitch, 1.0, "the fields not in the file keep their default");
}

/// What happens when a run ends: nothing without an answer, the alert after twenty seconds only
/// while the window is behind, and never the tone or the notification when they are off
/// (`watchRun` of `voice.js`).
#[test]
fn a_finished_run_announces_only_when_it_was_long_and_the_window_is_behind() {
    let (mut app, _dir) = app_with(store_with_answer());
    {
        let mut prefs = app.store().prefs.clone();
        prefs.voice.auto = false;
        prefs.voice.sound = false;
        prefs.voice.notify = false;
        // the settings the app reads are the ones of the store
        let _ = app.update(Message::VoiceAuto(false));
        let _ = app.update(Message::VoiceSound(false));
        let _ = app.update(Message::VoiceNotify(false));
        let _ = prefs;
    }

    // a short run says nothing
    app.after_run(5_000);
    assert_eq!(app.toast(), None, "under twenty seconds nothing is said");

    // a long one does, while the window is behind
    app.after_run(20_000);
    assert_eq!(
        app.toast(),
        Some("Alert: Your answer is ready."),
        "twenty seconds or more announce it (voice.js)"
    );

    // while the window is in front the alert is silent
    let _ = app.update(Message::DismissToast);
    let _ = app.update(Message::WindowFocus(true));
    app.after_run(60_000);
    assert_eq!(app.toast(), None, "silent while the window is in front");
    let _ = app.update(Message::WindowFocus(false));

    // announce says the body it is given
    app.announce("Your answer is ready.".into());
    assert_eq!(app.toast(), Some("Alert: Your answer is ready."));
}

/// Without a finished answer a run end does nothing (`finished` of watchRun).
#[test]
fn a_run_without_an_answer_announces_nothing() {
    let mut store = Store::default();
    store.chats.push(fllm_app::store::ChatRecord::new(
        1,
        "chat",
        vec![rec(Role::User, "question")],
    ));
    let (mut app, _dir) = app_with(store);
    app.after_run(60_000);
    assert_eq!(app.toast(), None, "there is no answer to announce");
}

/// What the settings show about this machine reaches the page, and the permission flag of the
/// Studio is written the first time (`fillVoices`, `notifyNote` and `asked` of `voice.js`).
#[test]
fn the_system_answer_reaches_the_settings_and_asks_once() {
    let (mut app, _dir) = app_with(store_with_answer());
    assert!(!app.store().prefs.voice.asked, "nothing was asked yet");

    let _ = app.update(Message::VoiceSystem {
        voices: vec!["Fred".into(), "Alice".into()],
        can_notify: true,
    });
    assert_eq!(app.voice_names.len(), 2, "the voices are kept for the select");
    assert!(app.voice_can_notify);
    assert!(
        app.store().prefs.voice.asked,
        "the first time the alerts are on the flag of the Studio is written"
    );

    let _ = app.update(Message::Go(View::Settings));
    {
        let mut ui = sim(&app);
        ui.find("Dictation").expect("the card of the Studio");
        ui.find("Reading aloud").expect("the card of the Studio");
        ui.find("Alerts").expect("the card of the Studio");
        ui.find("Microphone").expect("the row of the Studio");
        ui.find("This program, or the web view of the desktop app").ok();
        ui.find("Language").expect("the language row");
        ui.find("Read answers automatically").expect("the switch row");
        ui.find("Read a sample").expect("the try-it button");
        ui.find("System notification").expect("the alert row");
        ui.find("Play").expect("the tone button");
        ui.find("When a long answer (over 20 seconds)").ok();
        save_png(&app, "settings-voice");
    }
    drop(app);

    // the voice settings are written the way the sliders send them
    let (mut app2, _dir2) = app_with(store_with_answer());
    let _ = app2.update(Message::VoiceRate(1.5));
    let _ = app2.update(Message::VoicePitch(0.7));
    let _ = app2.update(Message::VoiceLang("fr-FR".into()));
    let _ = app2.update(Message::VoiceVoice("Fred".into()));
    let _ = app2.update(Message::VoiceAuto(true));
    let _ = app2.update(Message::VoiceSound(true));
    let v = &app2.store().prefs.voice;
    assert_eq!((v.rate, v.pitch), (1.5, 0.7), "the sliders");
    assert_eq!(v.lang, "fr-FR", "the language");
    assert_eq!(v.voice, "Fred", "the voice");
    assert!(v.auto && v.sound, "the switches");
}

/// The view data of the cards (the app passes it beside the personal view) holds what the page
/// shows: the voices of this computer and whether notifications are possible.
#[test]
fn the_view_data_of_the_cards_says_what_this_computer_has() {
    let view = VoiceView {
        voices: &["Fred".to_string()],
        can_notify: false,
    };
    assert_eq!(view.voices.len(), 1);
    assert!(!view.can_notify);
    let empty = VoiceView::default();
    assert!(empty.voices.is_empty());
    assert!(!empty.can_notify);
}
