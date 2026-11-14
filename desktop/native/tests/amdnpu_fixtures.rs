//! The AMD NPU card against the Studio's own: thirty-seven cases from the `@pure` zone of
//! `amdnpu.js` (loaded by `amdnpu_fixtures.js`), each compared field by field. The card that
//! draws them is `hardware_window.rs`.

use fllm_app::amdnpu::{amd_npu_model, amd_npu_native_text, amd_npu_text};
use serde_json::{json, Value};

fn fixtures() -> Vec<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/amdnpu_fixtures.json");
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("run `node amdnpu_fixtures.js` first: {e}"));
    serde_json::from_str(&raw).expect("the fixtures are a list")
}

/// A JSON value as the argument the ports take: null is nothing, anything else is the object.
fn obj(v: &Value) -> Option<&Value> {
    if v.is_null() { None } else { Some(v) }
}

#[test]
fn every_case_of_the_studio_gives_the_same_answer() {
    let cases = fixtures();
    assert!(cases.len() >= 37, "the generator found the pure zone: {} cases", cases.len());
    for case in &cases {
        let function = case["fn"].as_str().unwrap();
        let note = case["note"].as_str().unwrap_or("");
        let args = case["args"].as_array().unwrap();
        let theirs = case["got"].clone();

        let mine = match function {
            "amdNpuText" => {
                let problems: Vec<String> = args[2]
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                    .unwrap_or_default();
                json!(amd_npu_text(&args[0], &args[1], &problems))
            }
            "amdNpuNativeText" => json!(amd_npu_native_text(obj(&args[0]))),
            "amdNpuModel" => {
                let chosen = args[1].as_str().unwrap_or("");
                let card = amd_npu_model(obj(&args[0]), chosen);
                if !card.visible {
                    // the Studio answers with the bare `{ visible: false }`: the flag is all the
                    // two sides share, so the rest of the fields is compared only when it is true
                    assert_eq!(theirs.get("visible"), Some(&json!(false)), "{note}");
                    json!({ "visible": card.visible })
                } else {
                    serde_json::to_value(&card).expect("the card is serialisable")
                }
            }
            other => panic!("the fixtures name an unknown function: {other}"),
        };

        assert_eq!(
            theirs, mine,
            "{function} ({note}): the Studio says {theirs}, the port says {mine}"
        );
    }
}
