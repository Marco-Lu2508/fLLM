//! The commands of the Code window against the answers of the page itself:
//! `tests/fixtures/command_fixtures.js` runs the real `runCodeCommand` of `code.js` inside a vm
//! (its own `@pure` blocks, its table, its `toggleAgent`, recorders where it talks to the world)
//! and writes what it did to `command_fixtures.json`. Every command this window builds has to
//! plan the same steps, and every command it does not build has to stay unhandled here — the row
//! of PARITY.md lists those with their reason.

use fllm_app::commands::{
    dispatch, expand_command, help_text, insights_notice, sandbox_saved_notice, search_hits_text, session_insights, turbo_notice, CmdState, CustomCommand,
    InsightMessage, SearchHit, Step, TURBO_CONFIRM,
};
use serde_json::{json, Value};

/// The `/cost` summary of the harness (`window.fllmUsageSummary` of `command_fixtures.js`).
const USAGE: &str = "42 in / 7 out in 3 answer(s)";

/// The body `readCustomCommand` of the harness answers with.
const CUSTOM_BODY: &str = "Do it\n\nARGUMENTS: x";

/// What the page answers and this window does not build (row 79 of PARITY.md). While a command
/// stands here, `dispatch` must return `None` for it — when it gets built, take it out of this
/// list, and the record pins the behaviour from then on.
const NOT_BUILT: &[&str] = &[
    "/skill",
    "/rewind",
    "/compact",
    "/spec a b",
    "/resume",
    "/fork",
    "/revert",
    "/rename New title",
    "/rename",
    "/export",
    "/model",
    "/sessions",
    "/memory",
    "/check src",
    "/review staged",
    "/security-review",
    "/worktree",
    "/batch",
    "/learn",
    "/good",
    "/bad",
    "/retry",
    // the switch would claim the self-review run, which is row 81's work
    "/autoreview on",
    "/autoreview",
];

/// The `code` state the page stood in **before** the command ran (`pre` of the record: every case
/// starts from the harness seed, a setup script may change one field).
fn base_state(case: &Value) -> CmdState {
    let pre = &case["pre"];
    let custom = pre["custom"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|row| CustomCommand {
                    name: row["name"].as_str().unwrap_or_default().to_string(),
                    description: row["description"].as_str().unwrap_or_default().to_string(),
                    path: None,
                    plugin: None,
                })
                .collect()
        })
        .unwrap_or_default();
    let str = |key: &str, fallback: &str| pre[key].as_str().unwrap_or(fallback).to_string();
    CmdState {
        running: pre["running"].as_bool().unwrap_or(false),
        perm: str("perm", "ask"),
        last_build_perm: str("lastBuildPerm", "ask"),
        auto_approve: pre["autoApprove"].as_bool().unwrap_or(true),
        turbo: pre["turbo"].as_bool().unwrap_or(false),
        sandbox: pre["sandbox"].as_str().map(str::to_string),
        sandbox_available: pre["sandboxAvailable"].as_bool().unwrap_or(true),
        personality: str("personality", "default"),
        rules: fllm_app::commands::Rules {
            allow: pre["rules"]["allow"].as_str().unwrap_or_default().to_string(),
            deny: pre["rules"]["deny"].as_str().unwrap_or_default().to_string(),
        },
        custom,
        usage: Some(USAGE.to_string()),
    }
}

/// The events of the page's record, without the ones this window has no step for: drawing and
/// the writes into the browser's own storage.
fn page_events(case: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    for event in case["record"].as_array().expect("a record") {
        let name = event[0].as_str().unwrap_or_default();
        if matches!(name, "render" | "renderSide" | "saveSession" | "saveSettings" | "busy" | "pickMention" | "hook") {
            continue;
        }
        out.push(event.clone());
    }
    out
}

/// What this window does for the steps, in the shape the page writes its record.
fn native_events(steps: &[Step], case: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    for step in steps {
        match step {
            Step::Notice(text) => out.push(json!(["notice", text])),
            Step::Toast(text) => out.push(json!(["toast", text])),
            Step::NewSession => out.push(json!(["newSession"])),
            Step::Undo(steps) => out.push(json!(["undo", steps])),
            Step::AgentRun(prompt, source) => out.push(json!(["agentRun", prompt, source])),
            Step::SetPerm(perm, notice) => {
                out.push(json!(["setPerm", perm]));
                out.push(json!(["notice", notice]));
            }
            Step::OpenFolder(Some(path)) => out.push(json!(["setWorkspace", path])),
            Step::OpenFolder(None) => out.push(json!(["pickWorkspace"])),
            Step::Stop => out.push(json!(["stopAgent"])),
            Step::UserRun(command, share) => out.push(json!(["userRun", command, share])),
            Step::Remember { scope, entry, notice } => {
                out.push(json!(["memory", entry, scope]));
                out.push(json!(["notice", notice]));
            }
            Step::SaveRules { allow, deny, notice } => {
                out.push(json!(["saveRules", allow, deny]));
                out.push(json!(["notice", notice]));
            }
            Step::Turbo(next) => {
                if *next {
                    out.push(json!(["confirm", TURBO_CONFIRM]));
                }
                let confirmed = case["canned"]["turboConfirm"].as_bool().unwrap_or(true);
                if *next && !confirmed {
                    continue;
                }
                out.push(json!(["api", "/api/studio/runtime", {"method": "POST", "body": {"turbo": next}}]));
                // the notice uses what the runtime answered, as the page reads `data.turbo`
                let on = case["canned"]["runtime"]["turbo"].as_bool().unwrap_or(false);
                out.push(json!(["notice", turbo_notice(on)]));
            }
            Step::SetSandbox(mode) => {
                out.push(json!(["api", "/api/studio/runtime", {"method": "POST", "body": {"sandbox": mode}}]));
                let runtime = &case["canned"]["runtime"];
                let available = runtime["sandbox_available"].as_bool().unwrap_or(false);
                let answered = runtime["sandbox"].as_str().unwrap_or("unknown");
                out.push(json!(["notice", sandbox_saved_notice(answered, available)]));
            }
            Step::AutoApprove(_want, notice) => out.push(json!(["notice", notice])),
            Step::Search(query) => {
                // `encodeURIComponent` of the page: this window's own queries are plain words
                out.push(json!(["api", format!("/api/studio/sessions/search?q={query}&limit=8")]));
                let hits: Vec<SearchHit> = serde_json::from_value(case["canned"]["search"]["hits"].clone()).unwrap_or_default();
                out.push(json!(["notice", search_hits_text(&hits, query)]));
            }
            Step::Insights => {
                // the page counts its other sessions of the folder plus the live one; the live
                // conversation of the harness is empty, so it is one empty session at the end
                let mut sessions: Vec<Vec<InsightMessage>> = case["canned"]["savedSessions"]
                    .as_array()
                    .map(|rows| rows.iter().map(insight_messages).collect())
                    .unwrap_or_default();
                sessions.push(Vec::new());
                out.push(json!(["notice", insights_notice(&session_insights(&sessions))]));
            }
            Step::SetPersonality(_key, notice) => out.push(json!(["notice", notice])),
            Step::CustomCommand { name, arg_text, source } => {
                out.push(json!(["readCustom", name]));
                out.push(json!(["agentRun", expand_command(CUSTOM_BODY, arg_text), source]));
            }
        }
    }
    out
}

fn insight_messages(session: &Value) -> Vec<InsightMessage> {
    session["messages"]
        .as_array()
        .map(|messages| {
            messages
                .iter()
                .map(|m| InsightMessage {
                    role: m["role"].as_str().unwrap_or_default().to_string(),
                    tool_calls: m["tool_calls"]
                        .as_array()
                        .map(|calls| {
                            calls
                                .iter()
                                .filter_map(|c| c["function"]["name"].as_str().map(str::to_string))
                                .collect()
                        })
                        .unwrap_or_default(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Numbers are compared as numbers: the page writes `3`, Rust builds `3.0`.
fn norm(value: &Value) -> Value {
    match value {
        Value::Number(n) => Value::String(n.as_f64().map(|f| format!("{f}")).unwrap_or_default()),
        Value::Array(items) => Value::Array(items.iter().map(norm).collect()),
        Value::Object(map) => Value::Object(map.iter().map(|(k, v)| (k.clone(), norm(v))).collect()),
        other => other.clone(),
    }
}

#[test]
fn every_command_of_the_page_is_planned_the_same_way() {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/command_fixtures.json")).expect("the fixtures of the page");
    let data: Value = serde_json::from_str(&raw).expect("fixtures as json");
    let empty: Vec<Value> = Vec::new();
    let mut checked = 0;
    for case in data["cases"].as_array().expect("the cases") {
        let text = case["text"].as_str().expect("words");
        let state = base_state(case);
        if NOT_BUILT.contains(&text) {
            assert!(
                dispatch(text, &state).is_none(),
                "{text} is on the not-built list of row 79 but the window plans it — take it out of NOT_BUILT"
            );
            continue;
        }
        let planned = dispatch(text, &state);
        if let Some(steps) = &planned {
            assert_eq!(case["ret"].as_bool(), Some(true), "{text}: the page answers it, so the window must too");
            let want = norm(&Value::Array(page_events(case)));
            let got = norm(&Value::Array(native_events(steps, case)));
            assert_eq!(got, want, "{text}: the window plans other steps than the page performs");
        } else {
            assert_eq!(case["ret"].as_bool(), Some(false), "{text}: the page lets it go to the model, the window too");
            assert_eq!(page_events(case), empty, "{text}: the page does nothing with it");
        }
        checked += 1;
    }
    assert!(checked >= 40, "only {checked} commands were checked — the fixtures grew quiet");
}

/// One call from the plan check down to the table: what the page decides with its own
/// `authorize`, `permissionInfo`, `modeFor` and `autoVerdict` (`__decision` of the fixture
/// harness walks `runToolCall` 1955-2010 over those functions), and the window must decide the
/// same for the same state.
#[test]
fn decisions_of_one_call_match_the_page() {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/command_fixtures.json")).expect("the fixtures of the page");
    let data: Value = serde_json::from_str(&raw).expect("fixtures as json");
    let mut checked = 0;
    for case in data["decisions"].as_array().expect("the decisions") {
        let name = case["name"].as_str().expect("a name");
        let entry = &case["entry"];
        let args = case["args"].clone();
        let permission = entry["permission"].as_str().unwrap_or("read");
        let kind_mcp = (entry["source"].as_str() == Some("mcp"))
            .then(|| (entry["server"].as_str().unwrap_or(""), entry["tool"].as_str().unwrap_or("")));
        let mode = match case["perm"].as_str().unwrap_or("ask") {
            "plan" => fllm_app::code::Mode::Plan,
            "edit" => fllm_app::code::Mode::AutoEdit,
            _ => fllm_app::code::Mode::Ask,
        };
        let rules = fllm_app::commands::Rules {
            allow: case["rules"]["allow"].as_str().unwrap_or_default().to_string(),
            deny: case["rules"]["deny"].as_str().unwrap_or_default().to_string(),
        };
        let always: std::collections::HashSet<String> = case["always"]
            .as_array()
            .map(|keys| keys.iter().filter_map(|k| k.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        // the category and the subject are the ones `permissionInfo` builds — production reads
        // them the same way through `category_of`, which this unit test above pins
        let tool_name = entry["name"].as_str().unwrap_or_default();
        let (category, _) = fllm_app::commands::permission_info(permission, kind_mcp, tool_name, &args);
        let outcome = fllm_app::commands::call_decision(
            case["turbo"].as_bool().unwrap_or(false),
            &rules,
            mode == fllm_app::code::Mode::Plan,
            fllm_app::code::CodeState::mode_allows_for(&mode, &category),
            permission,
            kind_mcp,
            tool_name,
            &args,
            case["autoApprove"].as_bool().unwrap_or(true),
            if mode == fllm_app::code::Mode::Plan { "plan" } else { "build" },
            &always,
        );
        let (decision, auto_note) = match outcome {
            fllm_app::commands::Outcome::Plan => ("plan", None),
            fllm_app::commands::Outcome::Blocked => ("deny", None),
            fllm_app::commands::Outcome::Running(note) => ("allow", note),
            fllm_app::commands::Outcome::Ask => ("ask", None),
        };
        let want = &case["result"];
        assert_eq!(Value::from(decision), want["decision"], "{name}: other decision than the page");
        assert_eq!(
            auto_note.as_deref().map(Value::from),
            want["autoNote"].as_str().map(Value::from),
            "{name}: other words under the card than the page"
        );
        checked += 1;
    }
    assert!(checked >= 20, "only {checked} decisions were checked");
}

/// The parts the record cannot reach on its own: the help with custom commands, the counting of
/// `/insights` with data, and the two text builders the cases above only exercise in one shape.
#[test]
fn the_texts_behind_the_steps() {
    let custom = vec![CustomCommand {
        name: "deploy".into(),
        description: "Ship it".into(),
        path: Some(".fllm/commands/deploy.md".into()),
        plugin: None,
    }];
    let plain = help_text(&[]);
    let with_custom = help_text(&custom);
    assert_eq!(
        with_custom.replace(" Custom commands from .fllm/commands: /deploy (Ship it).", ""),
        plain,
        "the custom section sits between the table and the fixed tail",
    );
    assert!(with_custom.ends_with("Alt+B switches between the build and plan agent."), "the tail of the page");
    // the counting itself, with the plural rules of the page
    let sessions = vec![
        vec![
            InsightMessage { role: "user".into(), tool_calls: vec![] },
            InsightMessage {
                role: "assistant".into(),
                tool_calls: vec!["bash".into(), "write_file".into()],
            },
        ],
        vec![InsightMessage { role: "user".into(), tool_calls: vec![] }],
    ];
    let ins = session_insights(&sessions);
    assert_eq!((ins.sessions, ins.user_messages, ins.steps, ins.edits), (2, 2, 1, 1));
    assert_eq!(ins.top, vec![("bash".to_string(), 1), ("write_file".to_string(), 1)]);
    assert_eq!(
        insights_notice(&ins),
        "This folder, 2 saved sessions (the browser keeps the newest 30): 2 messages from you, 1 agent step, 1 file edit.\n\
Most used tools: bash 1, write_file 1.\nTokens and cost: /cost. The terminal agent has the same overview over its own sessions: fllm-rs insights."
    );
}
