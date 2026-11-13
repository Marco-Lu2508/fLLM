//! The keyboard shortcuts against what `boot.js` of the Studio answers: the fixtures are produced
//! by the `@pure-begin`..`@pure-end` block of that file itself (the parse, the normal form, how a
//! combination is shown, the keys in force, whether a new one may be given, which shortcut a
//! combination triggers, and the combination of a keyboard event).

use fllm_app::shortcuts::{
    check_combo, combo_of_key, effective_combos, norm_combo, parse_combo, shortcut_for,
};
use fllm_app::tour::format_combo;
use iced::keyboard::{key::Named, Key, Modifiers};
use serde_json::Value;
use std::collections::HashMap;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/shortcuts_fixtures.json"
    ))
    .expect("the fixtures of boot.js are there");
    serde_json::from_str(&raw).expect("the fixtures are json")
}

fn combos_of(v: &Value) -> HashMap<String, String> {
    v.as_object()
        .expect("a combos object")
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
        .collect()
}

/// The combination of a text as `parseCombo` of the Studio parses it.
#[test]
fn the_combination_of_a_text_is_parsed_the_way_boot_js_parses_it() {
    let f = fixtures();
    for case in f["parse"].as_array().expect("parse cases") {
        let fields = case.as_array().expect("a case array");
        let [input, expected] = fields.as_slice() else { panic!("the case has its fields") };
        let input = input.as_str().unwrap();
        let got = parse_combo(input).map(|p| {
            serde_json::json!({ "mods": p.mods, "key": p.key })
        });
        assert_eq!(
            got.unwrap_or(serde_json::Value::Null),
            *expected,
            "parseCombo({input:?}) of boot.js"
        );
    }
}

/// The normal form is what `normCombo` of the Studio builds.
#[test]
fn the_normal_form_is_the_one_of_boot_js() {
    let f = fixtures();
    for case in f["norm"].as_array().expect("norm cases") {
        let fields = case.as_array().expect("a case array");
        let [input, expected] = fields.as_slice() else { panic!("the case has its fields") };
        let input = input.as_str().unwrap();
        assert_eq!(
            norm_combo(input),
            expected.as_str().unwrap(),
            "normCombo({input:?}) of boot.js"
        );
    }
}

/// The combination is shown to people the way `formatCombo` shows it, off macOS and on it.
#[test]
fn a_combination_is_shown_the_way_boot_js_shows_it() {
    let f = fixtures();
    for case in f["format"].as_array().expect("format cases") {
        let fields = case.as_array().expect("a case array");
        let [combo, off, mac] = fields.as_slice() else { panic!("the case has its fields") };
        let combo = combo.as_str().unwrap();
        assert_eq!(
            format_combo(combo, false),
            off.as_str().unwrap(),
            "formatCombo({combo:?}, false) of boot.js"
        );
        assert_eq!(
            format_combo(combo, true),
            mac.as_str().unwrap(),
            "formatCombo({combo:?}, true) of boot.js"
        );
    }
}

/// The keys in force are the defaults with the saved changes on top, unknown ids ignored
/// (`effectiveCombos` of boot.js).
#[test]
fn the_keys_in_force_are_the_ones_of_boot_js() {
    let f = fixtures();
    for case in f["effective"].as_array().expect("effective cases") {
        let fields = case.as_array().expect("a case array");
        let [overrides, expected] = fields.as_slice() else { panic!("the case has its fields") };
        let mut map = HashMap::new();
        for (k, v) in overrides.as_object().expect("overrides") {
            map.insert(k.clone(), v.as_str().unwrap_or_default().to_string());
        }
        let got = effective_combos(&map);
        let want = combos_of(expected);
        assert_eq!(got, want, "effectiveCombos({overrides}) of boot.js");
    }
}

/// Whether a combination may be given to a shortcut — on yes its normal form, on no the exact
/// reason the Studio says (`checkCombo` of boot.js).
#[test]
fn a_new_combination_is_refused_the_way_boot_js_refuses_it() {
    let f = fixtures();
    for case in f["check"].as_array().expect("check cases") {
        let fields = case.as_array().expect("a case array");
        let [combos, id, text, expected] = fields.as_slice() else { panic!("the case has its fields") };
        let id = id.as_str().unwrap();
        let text = text.as_str().unwrap();
        let combos = combos_of(combos);
        let got = match check_combo(&combos, id, text) {
            Ok(combo) => serde_json::json!({ "ok": true, "combo": combo }),
            Err(error) => serde_json::json!({ "ok": false, "error": error }),
        };
        assert_eq!(
            &got,
            expected,
            "checkCombo for {id} with {text:?} against boot.js"
        );
    }
}

/// The shortcut a combination triggers (`shortcutFor` of boot.js: the id, or nothing).
#[test]
fn a_combination_triggers_the_shortcut_of_boot_js() {
    let f = fixtures();
    for case in f["for"].as_array().expect("for cases") {
        let fields = case.as_array().expect("a case array");
        let [combos, combo, expected] = fields.as_slice() else { panic!("the case has its fields") };
        let combos = combos_of(combos);
        let combo = combo.as_str().unwrap();
        let got = shortcut_for(&combos, combo).unwrap_or_default();
        assert_eq!(
            got,
            expected.as_str().unwrap(),
            "shortcutFor({combo:?}) of boot.js"
        );
    }
}

/// One keyboard event as the window sees it gives the combination `comboFromEvent` of the Studio
/// builds — the events come as boot.js sees them (the flags it reads), the key as iced reports it.
#[test]
fn a_keyboard_event_gives_the_combination_of_boot_js() {
    let f = fixtures();
    for case in f["fromevent"].as_array().expect("fromevent cases") {
        let fields = case.as_array().expect("a case array");
        let [event, is_mac, expected] = fields.as_slice() else { panic!("the case has its fields") };
        let is_mac = is_mac.as_bool().unwrap();
        let key_json = event["key"].as_str().unwrap_or_default();
        let key = match key_json {
            "" => Key::Unidentified,
            "Escape" => Key::Named(Named::Escape),
            "ArrowDown" => Key::Named(Named::ArrowDown),
            "ArrowUp" => Key::Named(Named::ArrowUp),
            "Enter" => Key::Named(Named::Enter),
            "Shift" => Key::Named(Named::Shift),
            "Control" => Key::Named(Named::Control),
            "Alt" => Key::Named(Named::Alt),
            "Meta" => Key::Named(Named::Meta),
            other => Key::Character(other.into()),
        };
        let mut modifiers = Modifiers::empty();
        let ctrl = event["ctrlKey"].as_bool().unwrap_or(false);
        let meta = event["metaKey"].as_bool().unwrap_or(false);
        if meta {
            modifiers |= Modifiers::LOGO;
        }
        if ctrl {
            modifiers |= Modifiers::CTRL;
        }
        if event["altKey"].as_bool().unwrap_or(false) {
            modifiers |= Modifiers::ALT;
        }
        if event["shiftKey"].as_bool().unwrap_or(false) {
            modifiers |= Modifiers::SHIFT;
        }
        let got = combo_of_key(&key, modifiers, is_mac);
        assert_eq!(
            got,
            expected.as_str().unwrap(),
            "comboFromEvent({event}, {is_mac}) of boot.js"
        );
    }
}

/// The list itself: fifteen shortcuts with their keys and four fixed rows, and the `?` that the
/// overview leaves out (the filter of `renderShortcutsPanel`).
#[test]
fn the_list_is_the_one_of_boot_js() {
    use fllm_app::shortcuts::{Shortcut, FIXED_KEYS, MOD_ORDER, SHORTCUTS};
    assert_eq!(SHORTCUTS.len(), 15, "the fifteen shortcuts of the Studio");
    assert_eq!(SHORTCUTS[0].id, "newChat");
    assert_eq!(SHORTCUTS[0].keys, "Mod+N");
    assert_eq!(
        SHORTCUTS[0].desc,
        "New chat (new session in Code)",
        "what the list says about the first one"
    );
    assert_eq!(SHORTCUTS[14].id, "codeSessions");
    // every id is unique and every default parses
    let mut ids: Vec<&str> = SHORTCUTS.iter().map(|s: &Shortcut| s.id).collect();
    ids.sort_unstable();
    let before = ids.clone();
    ids.dedup();
    assert_eq!(ids, before, "no id twice");
    for s in SHORTCUTS.iter() {
        assert!(parse_combo(s.keys).is_some(), "{} must parse", s.keys);
    }
    assert_eq!(FIXED_KEYS.len(), 4, "the fixed keys of the Studio");
    assert_eq!(FIXED_KEYS[3].0, "?", "the row the overview leaves out");
    assert_eq!(MOD_ORDER, ["Mod", "Ctrl", "Alt", "Shift"]);
}
