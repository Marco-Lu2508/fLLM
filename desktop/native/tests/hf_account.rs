//! The card *Hugging Face account* of the Models page: the token for private and gated models is typed here, the server checks and
//! keeps it, and the window does not.

use fllm_app::app::{App, Message, View};
use fllm_app::models::HfAccount;
use fllm_app::store::Store;
use iced_test::Simulator;
use serde_json::json;

fn app() -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store::default().save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Go(View::Models));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    Simulator::with_size(settings, iced::Size::new(1000.0, 3000.0), app.view())
}

#[test]
fn the_page_asks_for_the_account_and_says_what_the_server_says() {
    assert_eq!(HfAccount::from_json(&json!({"signed_in": true, "name": "marco", "source": "saved"})).line(), "Signed in as marco.");
    assert_eq!(
        HfAccount::from_json(&json!({"signed_in": true, "name": "marco", "source": "environment"})).line(),
        "Signed in as marco (token from the environment)."
    );
    assert_eq!(HfAccount::from_json(&json!({"signed_in": false})).line(), "Not signed in. Public models need no account.");
    assert_eq!(HfAccount::from_json(&json!({"signed_in": false, "error": "invalid token"})).line(), "The saved token does not work: invalid token");
    // only a saved token can be taken away from here
    assert!(HfAccount::from_json(&json!({"signed_in": true, "name": "m", "source": "saved"})).can_sign_out());
    assert!(!HfAccount::from_json(&json!({"signed_in": true, "name": "m", "source": "environment"})).can_sign_out());
    assert!(!HfAccount::from_json(&json!({"signed_in": false})).can_sign_out());

    let (mut app, _dir) = app();
    let mut ui = sim(&app);
    ui.find("Hugging Face account").expect("the card");
    ui.find("Checking …").expect("before the server has answered");
    drop(ui);
    let _ = app.update(Message::HfAccount(Ok(HfAccount::from_json(&json!({"signed_in": false})))));
    let mut ui = sim(&app);
    ui.find("Not signed in. Public models need no account.").expect("the server's words");
    ui.find("Create a token").expect("the page that makes tokens");
    ui.find("Sign out").expect_err("nobody to sign out");
}

#[test]
fn a_token_is_sent_once_and_does_not_stay_in_the_window() {
    let (mut app, _dir) = app();
    let none = app.update(Message::HfSignIn);
    assert_eq!(none.units(), 0, "an empty field sends nothing");
    let _ = app.update(Message::HfToken("  hf_secret  ".into()));
    let sent = app.update(Message::HfSignIn);
    assert_eq!(sent.units(), 1);
    assert!(app.models().hf_busy);
    let twice = app.update(Message::HfSignIn);
    assert_eq!(twice.units(), 0, "not twice while the first is checked");
    let _ = app.update(Message::HfSaved(Ok(HfAccount::from_json(&json!({"signed_in": true, "name": "marco", "source": "saved"})))));
    assert!(!app.models().hf_busy);
    assert_eq!(app.models().hf_token, "", "the token is on the server, not in the window");
    let mut ui = sim(&app);
    ui.find("Signed in as marco.").expect("signed in");
    ui.find("Sign out").expect("a saved token can be taken away");
}

#[test]
fn a_token_that_is_not_accepted_says_why_and_stays_to_be_corrected() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::HfToken("hf_wrong".into()));
    let _ = app.update(Message::HfSignIn);
    let _ = app.update(Message::HfSaved(Err("Invalid user token.".into())));
    assert_eq!(app.models().hf_token, "hf_wrong", "what was typed stays, to be corrected");
    assert_eq!(app.models().hf_error.as_deref(), Some("Invalid user token."));
    let mut ui = sim(&app);
    ui.find("Invalid user token.").expect("the reason");
    drop(ui);
    let _ = app.update(Message::HfToken("hf_right".into()));
    assert_eq!(app.models().hf_error, None, "typing again clears it");
}

#[test]
fn signing_out_sends_an_empty_token() {
    let (mut app, _dir) = app();
    let _ = app.update(Message::HfAccount(Ok(HfAccount::from_json(&json!({"signed_in": true, "name": "m", "source": "saved"})))));
    let out = app.update(Message::HfSignOut);
    assert_eq!(out.units(), 1);
    let _ = app.update(Message::HfSaved(Ok(HfAccount::from_json(&json!({"signed_in": false})))));
    assert_eq!(app.models().hf.as_ref().map(|a| a.signed_in), Some(false));
}
