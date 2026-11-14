//! The answer settings: automatic until switched to manual, remembered on the disk, and what each one puts into the request.

use fllm_app::api::{chat_body, ModelInfo};
use fllm_app::app::{App, Message, View};
use fllm_app::store::Store;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use std::sync::Arc;

fn app(theme: &str) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("tiny", 4096))));
    (app, dir)
}

fn reload(dir: &tempfile::TempDir) -> Store {
    Store::load(&dir.path().join("chats.json"))
}

fn paste(text: &str) -> Message {
    Message::SystemEdit(Action::Edit(Edit::Paste(Arc::new(text.to_string()))))
}

fn settings() -> iced::Settings {
    iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() }
}

#[test]
fn everything_is_automatic_at_the_start_and_sends_nothing_extra() {
    let (app, _dir) = app("light");
    assert_eq!(app.chat_params(), Default::default());
    assert!(app.system_prompt().starts_with("You are a helpful assistant running locally on the user's computer. Today is "), "{}", app.system_prompt());
    let body = chat_body("tiny", &app.wire_messages(), app.chat_params());
    assert!(body.get("temperature").is_none() && body.get("top_p").is_none() && body.get("max_tokens").is_none(), "{body}");
    assert_eq!(body["messages"][0]["role"], "system");
}

#[test]
fn manual_sampling_is_sent_remembered_and_switched_off_again() {
    let (mut app, dir) = app("light");
    let _ = app.update(Message::SamplingManual(true));
    let _ = app.update(Message::Temperature(0.3));
    let _ = app.update(Message::TopP(0.8));
    let _ = app.update(Message::SavePrefs);
    let body = chat_body("tiny", &app.wire_messages(), app.chat_params());
    assert_eq!((body["temperature"].as_f64().map(|t| (t * 100.0).round()), body["top_p"].as_f64().map(|t| (t * 100.0).round())), (Some(30.0), Some(80.0)));
    let saved = reload(&dir).prefs;
    assert!(saved.manual_sampling && (saved.temperature - 0.3).abs() < 1e-6 && (saved.top_p - 0.8).abs() < 1e-6);
    // back to automatic: the chosen values stay in the file for the next time, but are not sent
    let _ = app.update(Message::SamplingManual(false));
    assert_eq!(app.chat_params(), Default::default());
    assert!((reload(&dir).prefs.temperature - 0.3).abs() < 1e-6);
}

/// The output format and the token probabilities of the manual sampling (the Studio's
/// selConstraint, txtConstraintSpec and chkLogprobs): what they send and what is remembered.
#[test]
fn the_output_format_and_the_probabilities_are_sent_as_the_studio_sends_them() {
    let (mut app, dir) = app("light");
    let _ = app.update(Message::SamplingManual(true));
    let _ = app.update(Message::SamplingConstraint("json_schema".into()));
    let _ = app.update(Message::SamplingSpec("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},\"required\":[\"a\"],\"additionalProperties\":false}".into()));
    let _ = app.update(Message::SamplingLogprobs(true));
    let body = chat_body("tiny", &app.wire_messages(), app.chat_params());
    assert_eq!(body["response_format"]["type"], "json_schema", "{body}");
    assert_eq!(body["response_format"]["json_schema"]["name"], "schema");
    assert!(body["response_format"]["json_schema"]["schema"]["properties"]["a"].is_object(), "{body}");
    assert_eq!(body["logprobs"], serde_json::json!(true), "the switch of the Studio");
    assert_eq!(body["top_logprobs"], serde_json::json!(5), "five alternatives, like the Studio");
    let saved = reload(&dir).prefs;
    assert_eq!(saved.sampling.constraint, "json_schema", "kept with the chat file");
    assert!(saved.sampling.logprobs);

    // an invalid schema box is ignored, the answer stays free (chat.js)
    let _ = app.update(Message::SamplingSpec("{not json".into()));
    let body = chat_body("tiny", &app.wire_messages(), app.chat_params());
    assert!(body.get("response_format").is_none(), "a broken schema is not sent: {body}");

    // a GBNF grammar and a regular expression ride as their own fields
    let _ = app.update(Message::SamplingConstraint("gbnf".into()));
    let _ = app.update(Message::SamplingSpec("root ::= \"hi\"".into()));
    let body = chat_body("tiny", &app.wire_messages(), app.chat_params());
    assert_eq!(body["grammar"], "root ::= \"hi\"", "{body}");
    assert_eq!(body["grammar_type"], "gbnf", "{body}");
    let _ = app.update(Message::SamplingConstraint("regex".into()));
    let _ = app.update(Message::SamplingSpec("^hi".into()));
    let body = chat_body("tiny", &app.wire_messages(), app.chat_params());
    assert_eq!(body["regex"], "^hi", "{body}");
    assert!(body.get("grammar").is_none(), "{body}");

    // a JSON object needs no specification
    let _ = app.update(Message::SamplingConstraint("json_object".into()));
    let body = chat_body("tiny", &app.wire_messages(), app.chat_params());
    assert_eq!(body["response_format"]["type"], "json_object", "{body}");
    assert!(body.get("regex").is_none(), "{body}");

    // automatic sends none of it
    let _ = app.update(Message::SamplingManual(false));
    let body = chat_body("tiny", &app.wire_messages(), app.chat_params());
    assert!(body.get("logprobs").is_none() && body.get("response_format").is_none() && body.get("grammar").is_none(), "{body}");
}

/// The panel draws the picker, the switch, and the box for the specification when one is needed.
#[test]
fn the_sampling_panel_draws_the_format_box_and_the_switch() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::SamplingManual(true));
    let _ = app.update(Message::Go(fllm_app::app::View::Settings));
    {
        let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 900.0), app.view());
        ui.find("Output format").expect("the picker of the five formats");
        ui.find("Show token probabilities").expect("the switch");
        let mut ui2 = Simulator::with_size(settings(), iced::Size::new(1280.0, 900.0), app.view());
        assert!(ui2.find("Format specification").is_err(), "no box without a format that needs one");
    }
    let _ = app.update(Message::SamplingConstraint("gbnf".into()));
    {
        let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 900.0), app.view());
        ui.find("Format specification").expect("the box for the grammar");
    }
    let _ = app.update(Message::SamplingConstraint("none".into()));
    {
        let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 900.0), app.view());
        assert!(ui.find("Format specification").is_err(), "the box is gone with free text");
    }
    let _ = app.update(Message::SamplingConstraint("regex".into()));
    let _ = app.update(Message::SamplingLogprobs(true));
    save_png(&app, "sampling-format");
}

/// Writes one PNG of the given view into the snapshot dir of the run (the pattern of `tests/ui.rs`).
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 900.0), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(std::env::var("FLLM_SNAPSHOT_DIR").unwrap_or_else(|_| format!("{}/ui-snapshots", env!("CARGO_TARGET_TMPDIR"))));
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&prefix)) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
}

#[test]
fn a_custom_system_prompt_replaces_the_built_in_one_and_an_empty_one_falls_back() {
    let (mut app, dir) = app("light");
    let _ = app.update(Message::SystemCustom(true));
    let automatic = app.system_prompt();
    assert!(automatic.contains("Be concise"), "custom but empty: the automatic prompt");
    let _ = app.update(paste("Du bist ein knapper Assistent. Antworte immer auf Deutsch."));
    assert_eq!(app.system_prompt(), "Du bist ein knapper Assistent. Antworte immer auf Deutsch.");
    assert_eq!(app.wire_messages()[0].content, "Du bist ein knapper Assistent. Antworte immer auf Deutsch.");
    assert_eq!(reload(&dir).prefs.system_prompt, "Du bist ein knapper Assistent. Antworte immer auf Deutsch.", "written after the edit");
    let _ = app.update(Message::SystemCustom(false));
    assert!(app.system_prompt().contains("Be concise") && !app.system_prompt().contains("Du bist"), "switched back: the text is kept but not used");
    assert!(!reload(&dir).prefs.system_prompt.is_empty());
}

#[test]
fn the_longest_answer_takes_digits_only_and_empty_means_automatic() {
    let (mut app, dir) = app("light");
    let _ = app.update(Message::MaxTokensText("abc 12x3".into()));
    assert_eq!(app.chat_params().max_tokens, Some(123));
    let _ = app.update(Message::SavePrefs);
    assert_eq!(reload(&dir).prefs.max_tokens, 123);
    let _ = app.update(Message::MaxTokensText(String::new()));
    assert_eq!(app.chat_params().max_tokens, None);
    let _ = app.update(Message::MaxTokensText("99999999999".into()));
    assert!(app.chat_params().max_tokens.unwrap() <= 999_999, "a very long number is cut");
}

#[test]
fn back_to_automatic_resets_every_answer_setting() {
    let (mut app, dir) = app("light");
    let _ = app.update(Message::SamplingManual(true));
    let _ = app.update(Message::SystemCustom(true));
    let _ = app.update(paste("x"));
    let _ = app.update(Message::MaxTokensText("50".into()));
    let _ = app.update(Message::ResetPrefs);
    assert_eq!(app.prefs(), &Default::default());
    assert_eq!(reload(&dir).prefs, Default::default());
}

#[test]
fn settings_from_an_older_file_without_them_load_with_the_automatic_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    std::fs::write(&path, r#"{"chats":[],"next_id":3,"theme":"dark"}"#).unwrap();
    let store = Store::load(&path);
    assert_eq!(store.prefs, Default::default());
    assert_eq!((store.next_id, store.theme.as_str()), (3, "dark"));
    // a file that has only some of them keeps those and defaults the rest
    std::fs::write(&path, r#"{"chats":[],"prefs":{"manual_sampling":true,"temperature":1.2}}"#).unwrap();
    let p = Store::load(&path).prefs;
    assert!(p.manual_sampling && (p.temperature - 1.2).abs() < 1e-6 && (p.top_p - 0.95).abs() < 1e-6 && p.max_tokens == 0);
}

#[test]
fn the_settings_page_draws_in_both_modes_and_its_buttons_send_their_messages() {
    for theme in ["light", "dark"] {
        let (mut app, _dir) = app(theme);
        let _ = app.update(Message::Go(View::Settings));
        let _ = app.update(Message::SamplingManual(true));
        let _ = app.update(Message::SystemCustom(true));
        let _ = app.update(paste("Be brief."));
        let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 900.0), app.view());
        let shot = ui.snapshot(&app.theme()).unwrap();
        let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots");
        std::fs::create_dir_all(&dir).unwrap();
        let name = format!("settings-{theme}");
        for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}-"))) {
            let _ = std::fs::remove_file(old.path());
        }
        shot.matches_image(dir.join(format!("{name}.png"))).unwrap();
    }
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 900.0), app.view());
    ui.click("Manual").unwrap();
    assert!(ui.into_messages().any(|m| matches!(m, Message::SamplingManual(true))));
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 900.0), app.view());
    ui.click("Custom").unwrap();
    assert!(ui.into_messages().any(|m| matches!(m, Message::SystemCustom(true))));
}

#[test]
fn the_automatic_prompt_knows_the_model_and_your_text_can_be_added_to_it() {
    use fllm_app::api::ModelInfo;
    use fllm_app::prompts::ModelDetail;
    let (mut app, _dir) = app("light");
    let mut model = ModelInfo::new("Qwen3.5-9B", 32768);
    model.detail = ModelDetail { name: "Qwen3.5-9B".into(), arch: "qwen35".into(), parameters: 9.0e9, ..Default::default() };
    let _ = app.update(Message::Connected(Ok(model)));
    let full = app.system_prompt();
    assert!(full.contains("based on the language model Qwen3.5-9B") && full.contains("Format with Markdown only where it helps"), "a big model with room gets the full prompt: {full}");
    // the memory of the personalization is offered by default and described in the prompt
    assert!(full.contains("Memory tool (memory)"), "the memory tool is offered and described: {full}");
    let _ = app.update(Message::UseMemory(false));
    assert!(app.system_prompt().contains("You have no tools in this chat"), "switched off, it says what it cannot do");
    let _ = app.update(Message::UseMemory(true));
    let _ = app.update(Message::PromptLength(fllm_app::prompts::Length::Short));
    let short = app.system_prompt();
    assert!(short.len() < full.len() && !short.contains("based on the language model"), "{short}");
    // your text is added after the automatic one, with the placeholders filled
    let _ = app.update(Message::SystemCustom(true));
    let _ = app.update(Message::SystemAppend(true));
    let _ = app.update(paste("Today is {{date}}. Reply in {{language}}."));
    let added = app.system_prompt();
    assert!(added.starts_with(&short) && added.contains("Reply in "), "{added}");
    assert!(!added.contains("{{"), "{added}");
    let _ = app.update(Message::SystemAppend(false));
    assert!(app.system_prompt().starts_with("Today is "), "replaced");
}

#[test]
fn the_clock_in_the_prompt_stays_the_same_within_a_chat() {
    use fllm_app::app::Message as M;
    let (mut app, _dir) = app("light");
    let _ = app.update(M::Edit(Action::Edit(Edit::Paste(Arc::new("Hi".to_string())))));
    let _ = app.update(M::Send);
    let first = app.system_prompt();
    std::thread::sleep(std::time::Duration::from_millis(1100));
    assert_eq!(app.system_prompt(), first, "the same text, so the server can reuse what it already read");
}

#[test]
fn the_thinking_chip_follows_the_model_and_is_sent_with_the_request() {
    use fllm_app::api::{ModelInfo, Thinking};
    let (mut app, _dir) = app("light");
    // a model that cannot think: no chip, nothing sent
    assert!(!app.model_info().thinking.supported());
    assert!(app.chat_params().extra.is_empty());
    let mut model = ModelInfo::new("qwen", 8192);
    model.thinking = Thinking::from_caps(Some(&serde_json::json!({"supported": true, "control": "toggle", "variable": "enable_thinking", "levels": ["off", "low", "medium", "high"], "default": "off"})));
    let _ = app.update(Message::Connected(Ok(model)));
    assert_eq!(app.thinking_level(), "off");
    assert_eq!(app.chat_params().extra["chat_template_kwargs"], serde_json::json!({"enable_thinking": false}));
    assert!(app.chat_params().extra.get("reasoning_effort").is_none());
    let _ = app.update(Message::CycleThinking);
    assert_eq!(app.thinking_level(), "low");
    let extra = app.chat_params().extra;
    assert_eq!((extra["reasoning_effort"].as_str(), &extra["chat_template_kwargs"]), (Some("low"), &serde_json::json!({"enable_thinking": true})));
    for _ in 0..3 {
        let _ = app.update(Message::CycleThinking);
    }
    assert_eq!(app.thinking_level(), "off", "after high comes off");
    let body = chat_body("qwen", &app.wire_messages(), app.chat_params());
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    // a model with an effort takes the level as it is; one that always thinks has nothing to switch
    assert_eq!(Thinking::from_caps(Some(&serde_json::json!({"supported": true, "control": "effort", "levels": ["low", "high"], "default": "low"}))).fields("high")["reasoning_effort"], "high");
    let always = Thinking::from_caps(Some(&serde_json::json!({"supported": true, "control": "always", "levels": ["high"], "default": "high"})));
    assert!(always.fields("high").is_empty() && always.chip("high").0 == "Thinking");
    // an older server that only said true: a switch
    assert_eq!(Thinking::from_caps(Some(&serde_json::json!(true))).levels, ["off", "high"]);
    assert!(!Thinking::from_caps(Some(&serde_json::json!({"supported": false}))).supported());
    // a saved level the new model does not have falls back to its default
    let t = Thinking::from_caps(Some(&serde_json::json!({"supported": true, "control": "toggle", "levels": ["off", "high"], "default": "off"})));
    assert_eq!(t.level_for("medium"), "off");
}

#[test]
fn the_advanced_sampling_values_are_sent_only_when_they_are_not_neutral_and_manual() {
    use fllm_app::settings::Field;
    let (mut app, dir) = app("light");
    let _ = app.update(Message::SamplingField(Field::TopK, "40".into()));
    assert!(app.chat_params().extra.is_empty(), "automatic sampling sends nothing, whatever is typed");
    let _ = app.update(Message::SamplingManual(true));
    for (f, v) in [(Field::TopK, "40"), (Field::MinP, "0.05"), (Field::RepetitionPenalty, "1.1"), (Field::Seed, "42"), (Field::Stop, "###, END ,")] {
        let _ = app.update(Message::SamplingField(f, v.into()));
    }
    let extra = app.chat_params().extra;
    assert_eq!(extra["top_k"], 40);
    assert!((extra["min_p"].as_f64().unwrap() - 0.05).abs() < 1e-6 && (extra["repetition_penalty"].as_f64().unwrap() - 1.1).abs() < 1e-6);
    assert_eq!((extra["seed"].as_i64(), &extra["stop"]), (Some(42), &serde_json::json!(["###", "END"])));
    assert!(extra.get("presence_penalty").is_none() && extra.get("mirostat").is_none(), "{extra:?}");
    // a text that is not a number changes nothing; an empty box is the neutral value again
    let _ = app.update(Message::SamplingField(Field::TopK, "4x".into()));
    assert_eq!(app.chat_params().extra["top_k"], 40);
    let _ = app.update(Message::SamplingField(Field::MinP, "7".into()));
    assert!((app.chat_params().extra["min_p"].as_f64().unwrap() - 0.05).abs() < 1e-6, "min-p is at most 1");
    let _ = app.update(Message::SamplingField(Field::TopK, String::new()));
    assert!(app.chat_params().extra.get("top_k").is_none());
    let _ = app.update(Message::SavePrefs);
    let saved = reload(&dir).prefs.sampling;
    assert_eq!((saved.seed, saved.top_k, saved.stop.as_str()), (Some(42), 0, "###, END ,"));
    // the request carries them
    let body = chat_body("tiny", &app.wire_messages(), app.chat_params());
    assert_eq!((body["seed"].as_i64(), body["stop"][1].as_str()), (Some(42), Some("END")));
}
