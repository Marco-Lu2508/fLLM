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
const NOT_BUILT: &[&str] = &[];

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
        auto_review: pre["autoReview"].as_bool().unwrap_or(false),
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
        specs: specs_of(pre),
        // `modelNames()` of the harness
        models: vec!["qwen".to_string(), "llama".to_string()],
    }
}

/// The events of the page's record, without the ones this window has no step for: drawing and
/// the writes into the browser's own storage. For the worktree commands everything that is the
/// ANSWER's doing (the folder switch, the agent run and the notice that follows the request)
/// drops out here — the fixtures pin the request those steps plan, and `tests/worktree_window.rs`
/// pins what the answer then does, with the same canned data.
/// The `specs` of a fixture's pre-state (`code.specs`, as the snapshot writes it).
fn specs_of(pre: &Value) -> Vec<fllm_app::specs::Spec> {
    pre.get("specs")
        .and_then(Value::as_array)
        .map(|rows| rows.iter().filter_map(|row| serde_json::from_value(row.clone()).ok()).collect())
        .unwrap_or_default()
}

fn page_events(case: &Value) -> Vec<Value> {
    let text = case["text"].as_str().unwrap_or_default();
    let worktree_family = text.starts_with("/worktree") || text.starts_with("/wt") || text.starts_with("/batch");
    // a spec's run answers after the prompt: `agentRun` and the reload behind it happen when the
    // window performs them, so the record is cut where the answer begins (as row 82 cut it)
    let spec_family = text.starts_with("/spec");
    let mut out = Vec::new();
    let mut asked = false;
    let mut answered = false;
    for event in case["record"].as_array().expect("a record") {
        let name = event[0].as_str().unwrap_or_default();
        if matches!(name, "render" | "renderSide" | "saveSession" | "saveSettings" | "busy" | "pickMention" | "hook") {
            continue;
        }
        if worktree_family && (name == "setWorkspace" || name == "agentRun" || (asked && name == "notice")) {
            continue;
        }
        // after the run is handed over, what the page does is the window's reload
        if spec_family && answered {
            continue;
        }
        if spec_family && name == "agentRun" {
            answered = true;
        }
        // a notice that follows a request is the answer's doing (`specs_window.rs` holds it)
        if spec_family && asked && name == "notice" {
            answered = true;
            continue;
        }
        if name == "api" {
            asked = true;
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
            Step::AutoReview(_want, notice) => out.push(json!(["notice", notice])),
            Step::UserTool { name, args } => out.push(json!(["userTool", name, args])),
            Step::Worktree { action, name } => {
                // only the request is planned here; what its answer does is the window's own
                // machinery, pinned by `tests/worktree_window.rs`
                let mut body = json!({ "action": action });
                if let Some(n) = name {
                    body["name"] = json!(n);
                }
                out.push(json!(["api", "/api/studio/agent/worktree", { "method": "POST", "body": body }]));
            }
            Step::Batch { name, task: _ } => {
                out.push(json!(["api", "/api/studio/agent/worktree", { "method": "POST", "body": { "action": "new", "name": name } }]));
            }
            Step::SpecList => out.push(json!(["api", "/api/studio/agent/specs"])),
            Step::SpecNew { name, description, display } => {
                out.push(json!(["api", "/api/studio/agent/specs", {
                    "method": "POST",
                    "body": { "action": "new_prompt", "name": name, "description": description }
                }]));
                // the answer of a refusal is a notice the window pins (`specs_window.rs`)
                if case["canned"]["spec"]["ok"] != json!(false) {
                    out.push(json!(["agentRun", case["canned"]["spec"]["prompt"], display]));
                }
            }
            Step::SpecRun { name, asked } => {
                // the page finds the spec in the list it has and reloads only when the name is
                // missing — both legs, against the fixture's own state and canned answer
                use fllm_app::specs::{decide_run, index_json, index_text, parse_specs, RunDecision};
                let pre = specs_of(&case["pre"]);
                let mut decision = decide_run(&pre, name, *asked);
                let mut reloaded = false;
                if matches!(decision, RunDecision::NeedReload) {
                    reloaded = true;
                    out.push(json!(["api", "/api/studio/agent/specs"]));
                    decision = match decide_run(&parse_specs(&case["canned"]["spec"]["list"].as_array().map_or(Value::Null, |_| json!({ "specs": case["canned"]["spec"]["list"] }))), name, *asked) {
                        RunDecision::NeedReload => RunDecision::NotFound,
                        other => other,
                    };
                }
                match decision {
                    RunDecision::Act { name, index } => {
                        out.push(json!(["api", "/api/studio/agent/specs", {
                            "method": "POST",
                            "body": { "action": "task_prompt", "name": name, "index": index_json(index) }
                        }]));
                        if case["canned"]["spec"]["ok"] != json!(false) {
                            out.push(json!(["agentRun", case["canned"]["spec"]["prompt"], format!("/spec run {name} {}", index_text(index))]));
                        }
                    }
                    // behind a reload the page's notice comes after its GET and the record cuts
                    // it there (`asked`); without one it is the whole answer and stays
                    RunDecision::NotFound if !reloaded => out.push(json!(["notice", fllm_app::specs::no_spec_notice(name)])),
                    RunDecision::AllDone if !reloaded => out.push(json!(["notice", fllm_app::specs::ALL_DONE_NOTICE])),
                    RunDecision::NotFound | RunDecision::AllDone => {}
                    RunDecision::NeedReload => unreachable!("folded above"),
                }
            }
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
            Step::Learn(arg) => {
                use fllm_app::learn::*;
                let status: Option<Status> = serde_json::from_value(case["canned"]["learn"]["status"].clone()).ok().flatten();
                let learn_before = case["pre"]["learn"].as_bool().unwrap_or(false);
                match plan_learn(arg, true) {
                    LearnPlan::ProjectUsage => out.push(json!(["notice", PROJECT_USAGE])),
                    LearnPlan::Project(project) => {
                        out.push(json!(["api", "/api/studio/learn/settings", { "method": "POST", "body": { "workspace": "/home/marco/p", "project": project } }]));
                        out.push(json!(["learnStatus"]));
                        out.push(json!(["notice", project_notice(project)]));
                    }
                    LearnPlan::Show(set) => {
                        if let Some(on) = set {
                            out.push(json!(["learnSet", on]));
                        }
                        out.push(json!(["learnStatus"]));
                        out.push(json!(["notice", learn_notice(set.unwrap_or(learn_before), status.as_ref())]));
                    }
                }
            }
            Step::Thumb(value) => out.push(json!(["learnFeedback", value])),
            Step::SetPersonality(_key, notice) => out.push(json!(["notice", notice])),
            // what the page's harness has no turns for: the sessions list of the case, an empty list
            // of your messages (`turnChoices()` of the harness is empty) and the id `s1`; the labels
            // of the sessions carry the locale's date, so the rows are the page's own
            Step::Sessions => {
                let rows = case["record"].as_array().and_then(|r| r.first()).map(|e| e[3].clone()).unwrap_or(json!([]));
                out.push(json!(["choices", "Pick a session to continue:", "resume", rows]));
            }
            Step::Fork => out.push(json!(["fork", ""])),
            Step::Revert => out.push(json!(["notice", "There is no message to revert to."])),
            Step::Retry => out.push(json!(["notice", "There is no message to retry."])),
            Step::Rename(Some(_)) => {}
            Step::Rename(None) => out.push(json!(["rename", "s1"])),
            Step::Export => out.push(json!(["export", "s1"])),
            Step::Rewind => out.push(json!(["rewind"])),
            Step::Compact(focus) => out.push(json!(["compact", { "system": "sys", "tools": [], "signal": {} }, { "force": true, "focus": focus }])),
            Step::PickModel(names) => {
                let rows: Vec<Value> = names.iter().map(|n| json!({ "label": n, "data": n })).collect();
                out.push(json!(["choices", "Pick the model for the agent:", "model", rows]));
            }
            Step::Memory => {}
            Step::Skill => {}
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
        // the harness has no `window.fllmSkillPropose`, so the page answers `false` for `/skill` there;
        // the window plans the command and `tests/sessions_window.rs` holds what it does
        if text == "/skill" {
            assert_eq!(dispatch(text, &state), Some(vec![Step::Skill]));
            continue;
        }
        if NOT_BUILT.contains(&text) {
            assert!(
                dispatch(text, &state).is_none(),
                "{text} is on the not-built list of row 79 but the window plans it — take it out of NOT_BUILT"
            );
            continue;
        }
        let planned = dispatch(text, &state);
        if let Some(steps) = &planned {
            // `/memory` opens the block with `$('codeSide')`, which the harness has no stub for: it
            // stops there (`ret` null) after asking for the side to be drawn
            let answered = case["ret"].as_bool() == Some(true) || (text == "/memory" && case["ret"].is_null());
            assert!(answered, "{text}: the page answers it, so the window must too");
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
