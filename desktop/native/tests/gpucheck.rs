//! The two pure functions of the Studio's GPU check against the Studio's own: eleven cases from
//! the `@pure` zone of `gpucheck.js` (loaded by `gpucheck_fixtures.js`), each compared field by
//! field. The card that shows them is `hardware_window.rs`.

use fllm_app::gpucheck::{fallback_notice, gpu_verdict_line, GpuResult};
use serde_json::Value;

fn fixtures() -> Vec<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gpucheck_fixtures.json");
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("run `node gpucheck_fixtures.js` first: {e}"));
    serde_json::from_str(&raw).expect("the fixtures are a list")
}

#[test]
fn every_case_of_the_studio_gives_the_same_answer() {
    let cases = fixtures();
    assert!(cases.len() >= 11, "the generator found the pure zone: {} cases", cases.len());
    for case in &cases {
        let function = case["fn"].as_str().unwrap();
        let note = case["note"].as_str().unwrap_or("");
        let args = case["args"].as_array().unwrap();
        if function == "fallbackNotice" {
            let df = match &args[0] {
                Value::Null => None,
                object => Some(object),
            };
            let mine = fallback_notice(df);
            let theirs = case["got"].clone();
            if theirs.get("show").and_then(Value::as_bool) == Some(false) {
                assert!(!mine.show, "fallbackNotice shows nothing — {note}");
                continue;
            }
            // the Studio returns camelCase keys and its own order; the fields are compared one by one
            let get = |key: &str| theirs.get(key).cloned().unwrap_or(Value::Null);
            assert!(mine.show, "show — {note}");
            assert_eq!(Value::String(mine.level.clone()), get("level"), "level — {note}");
            assert_eq!(Value::String(mine.title.clone()), get("title"), "title — {note}");
            assert_eq!(Value::String(mine.short.clone()), get("short"), "short — {note}");
            assert_eq!(Value::String(mine.summary.clone()), get("summary"), "summary — {note}");
            assert_eq!(Value::String(mine.detail.clone()), get("detail"), "detail — {note}");
            let lines: Vec<Value> = mine.lines.iter().cloned().map(Value::String).collect();
            assert_eq!(Value::Array(lines), get("lines"), "lines — {note}");
            assert_eq!(
                Value::String(mine.footer_title.clone()),
                get("footerTitle"),
                "footerTitle — {note}"
            );
        } else {
            let input = serde_json::from_value::<GpuResult>(args[0].clone())
                .unwrap_or_else(|e| panic!("a verdict case with {e}"));
            let mine = gpu_verdict_line(&input);
            let theirs = case["got"].as_object().expect("an object");
            assert_eq!(Value::String(mine.text.clone()), theirs["text"], "text — {note}");
            assert_eq!(Value::String(mine.cls.to_string()), theirs["cls"], "cls — {note}");
        }
    }
}

/// The counts of the card: with the flag on, every line of the Studio is there in its order.
#[test]
fn the_lines_come_in_the_order_of_the_studio() {
    let df = serde_json::json!({
        "fallback": true, "mode": "layer_offload",
        "device": "GPU", "backend": "vulkan",
        "operations_on_cpu": 12, "operations_total": 48, "layers_on_device": 30,
        "reasons": [{"reason": "no kernel", "operations": "attention"}],
        "operation_fallbacks": [
            {"operation": "rms_norm", "reason": "unsupported", "calls": 4, "by_design": false},
            {"operation": "rope", "reason": "by design", "calls": 9, "by_design": true},
        ],
        "headline": "Some blocks run on the processor.",
        "notices": ["The card has 4 GB."],
    });
    let notice = fallback_notice(Some(&df));
    assert_eq!(
        notice.lines,
        vec![
            "12 of 48 operations (attention or feed-forward blocks of the layers) run on the CPU. 30 layers run completely on GPU (vulkan).",
            "Why: no kernel (attention)",
            "Operation on the CPU: rms_norm: unsupported (4 calls)",
        ],
        "the counts first, then the reasons, then the operations (join of the Studio)"
    );
}
