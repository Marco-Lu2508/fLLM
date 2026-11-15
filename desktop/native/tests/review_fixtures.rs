//! Row 81: what the page's own review functions answered, replayed in Rust — the self-review
//! pair of `codeprompts.js`, the check after an edit of `code.js` and the reply it appends.
//! The records come from running the page's code in `tests/fixtures/command_fixtures.js`.

use fllm_app::review as review_port;
use serde_json::Value;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/command_fixtures.json"
    ))
    .expect("the fixtures of the page");
    serde_json::from_str(&raw).expect("fixtures as json")
}

fn report_of(value: &Value) -> review_port::Report {
    let problems = value["problems"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|row| review_port::Problem {
                    path: row["path"].as_str().unwrap_or_default().to_string(),
                    output: row["output"].as_str().unwrap_or_default().to_string(),
                })
                .collect()
        })
        .unwrap_or_default();
    review_port::Report {
        review: value["review"].as_str().unwrap_or_default().to_string(),
        problems,
    }
}

/// `isCleanDiagnostic` and `reviewFeedback`, called with the page's own arguments.
#[test]
fn the_self_review_pair_answers_what_the_page_answers() {
    let data = fixtures();
    let rows = data["review"].as_array().expect("the review records");
    assert!(rows.len() >= 17, "the generator ran the pair: {} rows", rows.len());
    let mut checked = 0;
    for row in rows {
        let name = row["fn"].as_str().expect("the function of the record");
        let args = row["args"].as_array().expect("its arguments");
        let got = match name {
            "isCleanDiagnostic" => {
                let output = match &args[0] {
                    Value::Null => None,
                    other => other.as_str(),
                };
                Value::from(review_port::is_clean_diagnostic(output))
            }
            "reviewFeedback" => {
                let report = report_of(&args[0]);
                let round = args[1].as_u64().unwrap_or_default() as u32;
                review_port::review_feedback(&report, round)
                    .map(Value::from)
                    .unwrap_or(Value::Null)
            }
            other => panic!("{other} is not a function the fixtures carry"),
        };
        assert_eq!(got, row["result"], "{name} with {args:?}: other answer than the page");
        checked += 1;
    }
    assert_eq!(checked, rows.len());
}

/// Which paths `autoCheckEdit` asks diagnostics for — the four it looks at, one report per
/// extension (the `seen` set holds extensions), the rule of `bash: diagnostics <path>` and the
/// two answers it throws away.
#[test]
fn the_check_after_an_edit_walks_the_files_the_page_walks() {
    let data = fixtures();
    let rows = data["autocheck"].as_array().expect("the autocheck records");
    assert!(rows.len() >= 9, "the generator ran the check: {} rows", rows.len());
    for row in rows {
        let name = row["name"].as_str().expect("the tool of the record");
        let args = &row["args"];
        let rules = fllm_app::commands::Rules {
            allow: String::new(),
            deny: row["deny"].as_str().unwrap_or_default().to_string(),
        };
        let parsed = rules.parsed();
        let denied = |path: &str| -> bool {
            fllm_app::commands::evaluate_rules(&parsed, "bash", &format!("diagnostics {path}"), "build") == Some("deny")
        };
        let targets = review_port::auto_check_targets(name, args, &denied);
        // the calls the page really made: ['tool', name, { path, timeout_s }]
        let calls: Vec<Value> = row["calls"]
            .as_array()
            .expect("the recorded calls")
            .iter()
            .map(|entry| entry[2].clone())
            .collect();
        let want: Vec<Value> = targets.iter().map(|p| review_port::auto_check_args(p)).collect();
        assert_eq!(want, calls, "{name} {args:?}: the page asked for other paths");
        // and the answers it kept, in the order of the canned outputs
        let tools = row["tools"].as_array().expect("the canned answers");
        let kept: Vec<&str> = tools
            .iter()
            .zip(targets.iter())
            .filter_map(|(tool, _)| {
                let output = tool["output"].as_str().unwrap_or_default();
                review_port::auto_check_keep(output).then_some(output)
            })
            .collect();
        let joined = if kept.is_empty() { String::new() } else { kept.join("\n") };
        assert_eq!(Value::from(joined.clone()), row["result"], "{name} {args:?}: other report than the page ({joined:?})");
    }
}

/// The reply the model gets back when the check found something, clipped the way `clipOutput`
/// clips it (the cap over the model's context of the record).
#[test]
fn the_reply_carries_both_halves_and_the_finding_between_them() {
    let data = fixtures();
    let rows = data["reply"].as_array().expect(" the reply records");
    assert!(rows.len() >= 3, "the generator built replies: {} rows", rows.len());
    for row in rows {
        let ctx = row["ctx"].as_u64().unwrap_or(8192) as usize;
        let got = review_port::auto_check_reply(
            row["reply"].as_str().unwrap_or_default(),
            row["problems"].as_str().unwrap_or_default(),
            ctx,
        );
        assert_eq!(Value::from(got), row["result"], "other reply than the page's template");
    }
}
