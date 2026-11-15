//! Row 87: the background jobs against the page's own `parseJobId`, `jobArgName`, `trackJob`,
//! `jobOutput` and `jobKill` (`tests/fixtures/jobs_fixtures.js` runs them in a vm).

use fllm_app::jobs::*;
use serde_json::Value;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jobs_fixtures.json")).expect("the fixtures of the page");
    serde_json::from_str(&raw).expect("fixtures as json")
}

fn page_jobs(v: &Value) -> Vec<Job> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|j| Job {
            id: j["id"].as_str().unwrap().to_string(),
            command: j["command"].as_str().unwrap().to_string(),
            status: j["status"].as_str().unwrap().to_string(),
            out: j["out"].as_str().unwrap().to_string(),
            open: j["open"].as_bool().unwrap(),
        })
        .collect()
}

#[test]
fn the_id_of_a_job_is_read_the_way_the_page_reads_it() {
    let data = fixtures();
    assert!(data["parse"].as_array().unwrap().len() >= 26);
    for row in data["parse"].as_array().unwrap() {
        let text = match &row["text"] {
            Value::String(s) if s == "__undefined__" => String::new(),
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            other => other.to_string(),
        };
        assert_eq!(parse_job_id(&text), row["id"].as_str().map(str::to_string), "parseJobId({:?})", row["text"]);
    }
}

#[test]
fn the_list_follows_the_page_step_by_step() {
    let data = fixtures();
    for sc in data["scenarios"].as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let props: Vec<String> = sc["registry"]["props"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
        let has_tool = !sc["registry"].is_null();
        let arg_name = if has_tool { job_arg_name(&props) } else { "id".to_string() };
        let mut jobs: Vec<Job> = Vec::new();
        let mut calls: Vec<Value> = Vec::new();
        let mut toasts: Vec<String> = Vec::new();
        for (step, after) in sc["steps"].as_array().unwrap().iter().zip(sc["after"].as_array().unwrap()) {
            let args = step["args"].as_array().unwrap();
            match step["fn"].as_str().unwrap() {
                "trackJob" => track_job(&mut jobs, args[0].as_str().unwrap(), &args[1], args[2].as_str().unwrap_or_default()),
                "jobOutput" => {
                    let id = args[0].as_str().unwrap();
                    if toggle_output(&mut jobs, id) == Some(true) {
                        calls.push(serde_json::json!(["bash_output", { arg_name.clone(): id }]));
                        output_arrived(&mut jobs, id, answer(&step["answer"]));
                    }
                }
                "jobKill" => {
                    let id = args[0].as_str().unwrap();
                    calls.push(serde_json::json!(["bash_kill", { arg_name.clone(): id }]));
                    toasts.extend(kill_answered(&mut jobs, id, answer(&step["answer"])));
                }
                other => panic!("{other}"),
            }
            assert_eq!(jobs, page_jobs(&after["jobs"]), "{name}: the list after {}", step["fn"]);
            assert_eq!(Value::Array(calls.clone()), after["calls"], "{name}: what was asked of the server");
            let page_toasts: Vec<String> = after["toasts"].as_array().unwrap().iter().filter_map(Value::as_str).map(str::to_string).collect();
            assert_eq!(toasts, page_toasts, "{name}: the toasts");
        }
    }
}

/// What the tool call answered: `ok`/`output`, a refusal with its `error`, or a call that threw.
fn answer(v: &Value) -> Result<String, String> {
    if let Some(msg) = v.as_str().and_then(|s| s.strip_prefix("throw:")) {
        return Err(msg.to_string());
    }
    if v["ok"].as_bool() == Some(true) {
        Ok(v["output"].as_str().unwrap_or_default().to_string())
    } else {
        Err(v["error"].as_str().unwrap_or_default().to_string())
    }
}
