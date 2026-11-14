//! Settings > Server and API in the window: where the server listens, what it accepts, the keys
//! it hands out and the secret it shows once (`spane-local-api-server` of `shell.html`).

use fllm_app::app::{App, Message, Server, View};
use fllm_app::localapi::{CreatedKey, KeyList};
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
    Simulator::with_size(settings(), Size::new(1280.0, 3000.0), app.view())
}

/// The answer of `GET /api/studio/keys`: one key with a limit, one without.
fn list(with_keys: bool) -> KeyList {
    let keys = if with_keys {
        json!([
            {"id": "k-7f2a", "label": "Cursor", "masked": "fllm-sk-…a91c", "admin": true,
             "rpm_limit": 60, "tpm_limit": null, "requests_accepted": 12,
             "tokens_used": 3456, "rejected": 1},
            {"id": "k-11b0", "label": "Claude Code", "masked": "fllm-sk-…02de", "admin": false,
             "rpm_limit": null, "tpm_limit": null, "requests_accepted": 0,
             "tokens_used": 0, "rejected": 0}
        ])
    } else {
        json!([])
    };
    serde_json::from_value(json!({
        "auth_required": with_keys,
        "rate_limit": {"requests_per_minute": 60, "tokens_per_minute": 30000},
        "keys": keys
    }))
    .expect("the answer of the server")
}

/// The card says where the server listens, and the chip is only there when it answers.
#[test]
fn the_server_card_says_where_it_listens_and_what_it_accepts() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));

    // the window knows a server that does not answer: no chip, the address is there anyway
    let mut ui = sim(&app);
    ui.find("Server and API").expect("the title of the card");
    ui.find(fllm_app::localapi::SERVER_HELP).expect("its description");
    ui.find("Address").expect("the first row");
    ui.find("http://127.0.0.1:9/v1").expect("the address with /v1 behind it");
    ui.find("Copy URL").expect("its button");
    ui.find("Request limits").expect("the second row");
    ui.find(fllm_app::localapi::RATE_HELP).expect("its description");
    ui.find("Apply").expect("the button under the boxes");
    assert!(ui.find("Running").is_err(), "no chip while the server does not answer");
    drop(ui);

    // a server that answers brings the chip (`backend-chip` of the Studio)
    let _ = app.update(Message::Connected(Ok(fllm_app::api::ModelInfo::new("m", 4096))));
    assert!(matches!(app.server_state(), Server::Ready(_)));
    let mut ui = sim(&app);
    ui.find("Running").expect("the chip of a server that answers");
    drop(ui);

    // *Copy URL* takes the address to the clipboard
    let mut ui = sim(&app);
    ui.click("Copy URL").expect("the button of the row");
    let copied: Vec<String> = ui
        .into_messages()
        .filter_map(|m| match m {
            Message::CopyText(s) => Some(s),
            _ => None,
        })
        .collect();
    assert_eq!(copied, ["http://127.0.0.1:9/v1"], "the address of the card");
}

/// The keys card: the sentence without keys, the table with them, and the secret that was made.
#[test]
fn the_keys_card_says_what_the_server_has() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    // until the answer came nothing of the table is drawn
    let mut ui = sim(&app);
    ui.find("API keys").expect("the title of the card");
    ui.find(fllm_app::localapi::KEYS_HELP).expect("its description");
    ui.find("Create key").expect("the button of the form");
    ui.find("Admin rights").expect("the switch of the form");
    assert!(ui.find("ID").is_err(), "no table before the answer came");
    drop(ui);

    // an empty list says what an open server is
    let _ = app.update(Message::KeysLoaded(Ok(list(false))));
    let mut ui = sim(&app);
    ui.find("No API key is set up. The server is open, and it accepts changes only from this computer. Create a key above to require a Bearer token for requests.")
        .expect("the sentence for a server without keys");
    drop(ui);

    // with keys the table comes, with the line of every row
    let _ = app.update(Message::KeysLoaded(Ok(list(true))));
    let mut ui = sim(&app);
    ui.find("Refused").expect("a column of the table");
    ui.find("Cursor").expect("the name of the first key");
    ui.find("admin").expect("its rights");
    ui.find("fllm-sk-…a91c").expect("the masked key");
    ui.find("60 per minute").expect("the limit of the row");
    ui.find("Revoke").expect("the button of the row");
    ui.find("Claude Code").expect("the second key");
    ui.find("Default").expect("a key without a limit of its own");
    drop(ui);

    // the answer fills the boxes of the limits (`inpGlobalRpm.value = rl.requests_per_minute`)
    let _ = app.update(Message::KeysLoaded(Ok(list(true))));
    assert_eq!(app.rate_rpm, "60", "the requests per minute of the answer");
    assert_eq!(app.rate_tpm, "30000", "and the tokens per minute");
}

/// Without a model the server has no key routes, and the card says so (`state.setup`).
#[test]
fn without_a_model_the_card_says_so_and_the_button_refuses() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::Connected(Err(fllm_app::models::NO_MODEL.into())));

    let mut ui = sim(&app);
    ui.find("Keys are made by the running server. Load a model first (Models page), then create a key here.")
        .expect("the sentence for a server without a model");
    drop(ui);

    let _ = app.update(Message::KeyCreate);
    let toast = app.toast.clone().expect("the button refuses with a toast");
    assert_eq!(toast, "Load a model first: keys are made by the running server.");

    // the secret of a key that was made is shown once, and the form is cleared again
    let _ = app.update(Message::Connected(Ok(fllm_app::api::ModelInfo::new("m", 4096))));
    let _ = app.update(Message::KeyLabel("Cursor".into()));
    let created: CreatedKey = serde_json::from_value(json!({
        "secret": "fllm-sk-TESTTESTTESTTEST",
        "key": {"id": "k-new", "label": "Cursor", "masked": "fllm-sk-…cdef", "admin": false,
                "rpm_limit": null, "tpm_limit": null, "requests_accepted": 0,
                "tokens_used": 0, "rejected": 0}
    }))
    .expect("the answer of the server");
    let _ = app.update(Message::KeyCreated(Ok(created)));
    let mut ui = sim(&app);
    ui.find("Your new key. Copy it now; it is shown only once and cannot be looked up later:")
        .expect("the banner of the new key");
    ui.find("fllm-sk-TESTTESTTESTTEST").expect("the secret in it");
    ui.find("Copy key").expect("its button");
    drop(ui);
    assert_eq!(app.toast.as_deref(), Some("API key \"Cursor\" created"), "the toast of `btnCreateKey`");
}
