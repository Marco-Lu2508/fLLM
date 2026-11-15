//! Row 88: the pure block of the page's `learn.js` against the port. The fixtures are the page's
//! own `lnSkillsRead`, `lnRunMessages`, `lnRunFailed` and `lnToastText` run in a vm over a few
//! dozen cases (`tests/fixtures/learn_fixtures.js`).

use fllm_app::api::WireMessage;
use fllm_app::learn::*;
use serde_json::Value;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/learn_fixtures.json")).expect("the fixtures of the page");
    serde_json::from_str(&raw).expect("fixtures as json")
}

fn wire(v: &Value) -> Vec<WireMessage> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|m| {
            let mut w = WireMessage::new(m["role"].as_str().unwrap(), m["content"].as_str().unwrap_or_default());
            w.tool_calls = m.get("tool_calls").cloned();
            w.tool_call_id = m.get("tool_call_id").and_then(Value::as_str).map(str::to_string);
            w
        })
        .collect()
}

#[test]
fn the_pages_own_functions_and_the_port_agree() {
    let data = fixtures();
    let cases = data["cases"].as_array().unwrap();
    assert!(cases.len() >= 24, "the generator found the functions: {} cases", cases.len());
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let messages = wire(&case["messages"]);
        match case["kind"].as_str().unwrap() {
            "skills" => {
                // the page pushes `args.name` as it came: a number stays a number there
                let page: Vec<String> = case["result"].as_array().unwrap().iter().map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string)).collect();
                assert_eq!(skills_read(&messages), page, "skills: {name}");
            }
            "run" => {
                let max = case["max"].as_u64().unwrap() as usize;
                let from = case["from"].as_u64().unwrap() as usize;
                assert_eq!(Value::Array(run_messages(&messages, from, max, 3000)), case["result"], "run: {name}");
            }
            other => panic!("{other}"),
        }
    }
    for row in data["failed"].as_array().unwrap() {
        assert_eq!(run_failed(row["error"].as_bool().unwrap(), row["hitLimit"].as_bool().unwrap()), row["result"].as_bool().unwrap());
    }
    for row in data["toasts"].as_array().unwrap() {
        let last = row["last"].as_object().map(|o| LastRun { summary: o.get("summary").and_then(Value::as_str).unwrap_or_default().to_string(), ..Default::default() });
        assert_eq!(toast_text(last.as_ref()), row["result"].as_str().unwrap(), "toast of {}", row["last"]);
    }
}
