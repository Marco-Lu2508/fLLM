//! Row 77: the sub-agents in the window — a `task` call starts a conversation of its own with its
//! own stream, its calls ask like the conversation's, at most four work at a time, and its
//! report is what the model of the conversation reads. The decisions before the loop starts are
//! pinned against the page's own function in `tests/subagent_fixtures.rs`; here is the loop.

use fllm_app::api::{ModelInfo, StreamEvent};
use fllm_app::app::{App, Message, View};
use fllm_app::code::{client_tools, CallState, Call, CodeMsg, Item, KidItem, Mode, Permission, Saved, Tool, Workspace};
use fllm_app::store::Store;
use fllm_app::subagents::CustomAgent;
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use serde_json::{json, Value};
use std::sync::Arc;

const FOLDER: &str = "/home/marco/projects/fllm";

fn settings() -> iced::Settings {
    iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    }
}

fn app(theme: &str) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("qwen", 8192))));
    let _ = app.update(Message::Go(View::Code));
    (app, dir)
}

fn code(app: &mut App, m: CodeMsg) {
    let _ = app.update(Message::Code(m));
}

fn def(name: &str) -> Value {
    json!({ "type": "function", "function": { "name": name, "description": name, "parameters": { "type": "object", "properties": {} } } })
}

/// The folder is open with the server's tools and the window's own (todo, ask, task, create_agent).
fn open_folder(app: &mut App) {
    code(app, CodeMsg::Opened(Ok(Workspace { root: FOLDER.into(), name: "fllm".into(), branch: Some("main".into()) })));
    let mut tools: Vec<Tool> = [("read_file", Permission::Read), ("grep", Permission::Read), ("edit_file", Permission::Edit), ("bash", Permission::Command)]
        .into_iter()
        .map(|(n, p)| Tool::server(n, p))
        .collect();
    let mut defs: Vec<Value> = tools.iter().map(|t| def(&t.name)).collect();
    let (client, client_defs) = client_tools();
    tools.extend(client);
    defs.extend(client_defs);
    code(app, CodeMsg::ToolsLoaded(Ok((tools, Value::Array(defs)))));
}

fn say(app: &mut App, text: &str) {
    code(app, CodeMsg::Edit(Action::Edit(Edit::Paste(Arc::new(text.to_string())))));
    code(app, CodeMsg::Send);
}

fn parent_calls(app: &mut App, calls: &[(&str, &str, &str)]) {
    for (i, (id, name, args)) in calls.iter().enumerate() {
        code(app, CodeMsg::Stream(StreamEvent::ToolCall { index: i, id: Some((*id).into()), name: Some((*name).into()), arguments: (*args).into() }));
    }
    code(app, CodeMsg::Stream(StreamEvent::Done));
}

fn kid(app: &mut App, id: &str, ev: StreamEvent) {
    code(app, CodeMsg::KidStream(id.into(), ev));
}

fn kid_says(app: &mut App, id: &str, words: &str) {
    kid(app, id, StreamEvent::Delta(words.into()));
    kid(app, id, StreamEvent::Done);
}

fn kid_calls(app: &mut App, id: &str, words: &str, calls: &[(&str, &str)]) {
    if !words.is_empty() {
        kid(app, id, StreamEvent::Delta(words.into()));
    }
    for (i, (name, args)) in calls.iter().enumerate() {
        kid(app, id, StreamEvent::ToolCall { index: i, id: Some(format!("model-{i}")), name: Some((*name).into()), arguments: (*args).into() });
    }
    kid(app, id, StreamEvent::Done);
}

fn find_call(app: &App, id: &str) -> Option<Call> {
    for item in &app.code().items {
        if let Item::Call(c) = item {
            if c.id == id {
                return Some(c.clone());
            }
            for k in c.kid.iter().flat_map(|k| &k.items) {
                if let KidItem::Call(cc) = k {
                    if cc.id == id {
                        return Some((**cc).clone());
                    }
                }
            }
        }
    }
    None
}

fn task(app: &App, id: &str) -> Call {
    find_call(app, id).unwrap_or_else(|| panic!("the task call {id}"))
}

fn kid_items(app: &App, id: &str) -> Vec<KidItem> {
    task(app, id).kid.map(|k| k.items).unwrap_or_default()
}

fn nested_ids(app: &App, id: &str) -> Vec<String> {
    kid_items(app, id).into_iter().filter_map(|i| if let KidItem::Call(c) = i { Some(c.id) } else { None }).collect()
}

fn progress(app: &App, id: &str) -> String {
    task(app, id).kid.map(|k| k.progress).unwrap_or_default()
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), iced::Size::new(1280.0, 2400.0), app.view())
}

/// The conversation starts a task: its call runs at once (the page asks nothing before
/// `runSubAgent`), and the sub-agent starts with the first message of its type.
fn start(app: &mut App, args: &str) {
    say(app, "look into it");
    parent_calls(app, &[("t1", "task", args)]);
}

#[test]
fn a_task_starts_a_sub_agent_with_its_own_conversation() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    start(&mut app, r#"{"description":"Find the parser","prompt":"Where is the SSE parser?"}"#);

    let call = task(&app, "t1");
    assert_eq!(call.state, CallState::Running, "the task runs while its sub-agent works");
    assert_eq!(progress(&app, "t1"), "step 1", "the progress line of the page");
    assert_eq!(call.kid.as_ref().unwrap().agent_type, "explore", "explore is the default type");
    assert_eq!(app.code().kids_working_and_waiting(), (1, 0));

    let wire = app.code().kid_wire("t1").expect("the sub-agent's conversation");
    assert_eq!(wire.len(), 2, "its instructions and the job");
    assert_eq!(wire[0].role, "system");
    assert!(wire[0].content.starts_with("You are a coding agent"), "it is told what the conversation is told");
    assert_eq!(wire[1].role, "user");
    assert!(wire[1].content.starts_with("You are a sub-agent."), "the first message is the page's");
    assert!(wire[1].content.ends_with("Job: Find the parser\n\nWhere is the SSE parser?"));

    // the conversation itself waits: its round has an open call
    assert!(app.code().running);
    assert_eq!(app.code().kid_tools("t1").unwrap().as_array().unwrap().len(), 9, "every tool the conversation has is offered; the refusals come per call");
}

#[test]
fn the_report_is_what_the_model_of_the_conversation_reads() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    start(&mut app, r#"{"description":"Find the parser","prompt":"Where is the SSE parser?"}"#);
    kid_says(&mut app, "t1", "  It is in src/api.rs:120.\n");

    let call = task(&app, "t1");
    assert_eq!(call.state, CallState::Done { ok: true, output: "It is in src/api.rs:120.".into() }, "the card carries the report, trimmed");
    assert_eq!(call.sent.as_deref(), Some("Sub-agent report (Find the parser):\nIt is in src/api.rs:120."), "the model gets the page's wording");
    assert_eq!(call.kid.as_ref().unwrap().progress, "", "the progress line is gone");
    assert!(app.code().kid_wire("t1").is_none(), "the sub-agent is over");
    // the conversation goes on with the reply as the answer of the call
    let wire = app.code().wire(None);
    let tool = wire.iter().find(|m| m.role == "tool").expect("the answer of the task call");
    assert_eq!(tool.content, "Sub-agent report (Find the parser):\nIt is in src/api.rs:120.");
    assert_eq!(tool.tool_call_id.as_deref(), Some("t1"));
    assert!(matches!(app.code().items.last(), Some(Item::Assistant { .. })), "the next request of the conversation started");
}

#[test]
fn a_sub_agent_reads_with_its_calls_and_is_told_the_results() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    start(&mut app, r#"{"description":"Find it","prompt":"p"}"#);
    kid_calls(&mut app, "t1", "Looking.", &[("grep", r#"{"pattern":"parse"}"#), ("read_file", r#"{"path":"src/api.rs"}"#)]);

    let ids = nested_ids(&app, "t1");
    assert_eq!(ids, vec!["t1~0.0".to_string(), "t1~0.1".to_string()], "its calls have ids of their own");
    for id in &ids {
        assert_eq!(find_call(&app, id).unwrap().state, CallState::Running, "a read runs without a question");
    }
    assert_eq!(progress(&app, "t1"), "step 1");

    // the answers arrive in any order; the model gets them in the order of the calls
    code(&mut app, CodeMsg::ToolDone("t1~0.1".into(), Ok("1\tfn parse() {}".into())));
    assert_eq!(progress(&app, "t1"), "step 1", "it waits for the other call");
    code(&mut app, CodeMsg::ToolDone("t1~0.0".into(), Ok("src/api.rs:1: fn parse".into())));
    assert_eq!(progress(&app, "t1"), "step 2", "the next step started");
    let wire = app.code().kid_wire("t1").unwrap();
    let roles: Vec<&str> = wire.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["system", "user", "assistant", "tool", "tool"]);
    assert_eq!(wire[2].content, "Looking.");
    let announced: Vec<String> = wire[2].tool_calls.as_ref().unwrap().as_array().unwrap().iter().map(|c| c["id"].as_str().unwrap().to_string()).collect();
    assert_eq!(announced, ids, "the assistant message announces the same ids the answers carry");
    assert_eq!(wire[3].tool_call_id.as_deref(), Some("t1~0.0"));
    assert_eq!(wire[3].content, "src/api.rs:1: fn parse");
    assert_eq!(wire[4].tool_call_id.as_deref(), Some("t1~0.1"));

    kid_says(&mut app, "t1", "Found: src/api.rs.");
    assert_eq!(task(&app, "t1").state, CallState::Done { ok: true, output: "Found: src/api.rs.".into() });
    // the items of the card: its words, its calls, its last answer
    let items = kid_items(&app, "t1");
    assert!(matches!(&items[0], KidItem::Assistant(t) if t == "Looking."));
    assert!(matches!(&items[1], KidItem::Call(c) if c.name == "grep"));
    assert!(matches!(items.last(), Some(KidItem::Assistant(t)) if t == "Found: src/api.rs."));
}

#[test]
fn an_explore_agent_is_refused_every_change_and_told_so() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::SetMode(Mode::Auto));
    start(&mut app, r#"{"description":"Edit","prompt":"p"}"#);
    kid_calls(&mut app, "t1", "", &[("edit_file", r#"{"path":"a"}"#), ("bash", r#"{"command":"ls"}"#), ("todo_write", r#"{"todos":[]}"#), ("task", r#"{"prompt":"x"}"#)]);

    let expected = [
        ("edit_file", "This sub-agent is read-only.", "Denied: this sub-agent is read-only. Describe the change in your report instead."),
        ("bash", "This sub-agent is read-only.", "Denied: this sub-agent is read-only. Describe the change in your report instead."),
        ("todo_write", "Not available to sub-agents.", "Denied: todo_write is not available to a sub-agent. Do the work yourself and put the answer in your report."),
        ("task", "Not available to sub-agents.", "Denied: task is not available to a sub-agent. Do the work yourself and put the answer in your report."),
    ];
    for (id, (name, card, reply)) in nested_ids(&app, "t1").iter().zip(expected) {
        let c = find_call(&app, id).unwrap();
        assert_eq!(c.name, name);
        assert_eq!(c.state, CallState::Done { ok: false, output: card.into() }, "{name}: the card");
        assert_eq!(c.sent.as_deref(), Some(reply), "{name}: the model's reply");
    }
    // all four are answered: the sub-agent is on its second step already, told every refusal
    assert_eq!(progress(&app, "t1"), "step 2");
    let wire = app.code().kid_wire("t1").unwrap();
    let tools: Vec<&str> = wire.iter().filter(|m| m.role == "tool").map(|m| m.content.as_str()).collect();
    assert_eq!(tools.len(), 4);
    assert_eq!(tools[0], expected[0].2);
}

#[test]
fn a_general_agent_asks_for_what_it_changes_like_the_conversation_does() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    start(&mut app, r#"{"description":"Fix it","prompt":"p","agent":"general"}"#);
    assert_eq!(task(&app, "t1").kid.unwrap().agent_type, "general");
    kid_calls(&mut app, "t1", "", &[("edit_file", r#"{"path":"a.rs","old_string":"1","new_string":"2"}"#), ("read_file", r#"{"path":"a.rs"}"#)]);

    let edit = find_call(&app, "t1~0.0").unwrap();
    assert_eq!(edit.state, CallState::Pending, "an edit waits for you in the mode Ask");
    assert_eq!(find_call(&app, "t1~0.1").unwrap().state, CallState::Running, "a read does not");
    assert_eq!(task(&app, "t1").state, CallState::Running);

    // the card of the sub-agent has the buttons: Allow runs it, and the sub-agent goes on
    code(&mut app, CodeMsg::Approve("t1~0.0".into()));
    assert_eq!(find_call(&app, "t1~0.0").unwrap().state, CallState::Running);
    code(&mut app, CodeMsg::ToolDone("t1~0.0".into(), Ok("Edited.".into())));
    code(&mut app, CodeMsg::ToolDone("t1~0.1".into(), Ok("1\tx".into())));
    assert_eq!(progress(&app, "t1"), "step 2");

    // a denial goes back to the sub-agent as it goes back to the model of the conversation
    kid_calls(&mut app, "t1", "", &[("bash", r#"{"command":"git push"}"#)]);
    assert_eq!(find_call(&app, "t1~1.0").unwrap().state, CallState::Pending);
    code(&mut app, CodeMsg::Deny("t1~1.0".into()));
    let c = find_call(&app, "t1~1.0").unwrap();
    assert!(matches!(&c.state, CallState::Done { ok: false, output } if output.starts_with("The user declined this call.")));
    assert_eq!(progress(&app, "t1"), "step 3");
}

#[test]
fn switching_to_plan_mode_refuses_what_a_sub_agent_waits_to_do() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    start(&mut app, r#"{"description":"Fix it","prompt":"p","agent":"general"}"#);
    kid_calls(&mut app, "t1", "", &[("edit_file", r#"{"path":"a.rs"}"#)]);
    assert_eq!(find_call(&app, "t1~0.0").unwrap().state, CallState::Pending);
    code(&mut app, CodeMsg::SetMode(Mode::Plan));
    assert!(matches!(find_call(&app, "t1~0.0").unwrap().state, CallState::Done { ok: false, ref output } if output.starts_with("Plan mode only allows reading.")));
    assert_eq!(progress(&app, "t1"), "step 2", "the sub-agent was told and goes on");

    // a general agent asked for in plan mode is an explore agent
    let (mut app, _dir) = self::app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::SetMode(Mode::Plan));
    start(&mut app, r#"{"description":"Fix it","prompt":"p","agent":"general"}"#);
    assert_eq!(task(&app, "t1").kid.unwrap().agent_type, "explore");
}

#[test]
fn a_task_that_cannot_start_is_answered_at_once() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "go");
    parent_calls(&mut app, &[("t1", "task", r#"{"description":"x"}"#), ("t2", "task", r#"{"description":"x","prompt":"p","agent":"ghost"}"#), ("t3", "task", "{not json")]);

    let a = task(&app, "t1");
    assert_eq!(a.state, CallState::Done { ok: false, output: "No prompt.".into() });
    assert_eq!(a.sent.as_deref(), Some("Error: task needs a prompt."));
    let b = task(&app, "t2");
    assert_eq!(b.state, CallState::Done { ok: false, output: "There is no sub-agent \"ghost\". Known: explore, general.".into() });
    assert_eq!(b.sent.as_deref(), Some("Error: there is no sub-agent \"ghost\". Known: explore, general."));
    let c = task(&app, "t3");
    assert_eq!(c.state, CallState::Done { ok: false, output: "The arguments are not valid JSON.".into() });
    assert_eq!(app.code().kids_working_and_waiting(), (0, 0), "nothing started");
    assert!(matches!(app.code().items.last(), Some(Item::Assistant { .. })), "the conversation got its answers and went on");
}

#[test]
fn custom_agents_are_listed_for_the_model_and_run_with_their_own_rules() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    let reviewer = CustomAgent {
        name: "code-reviewer".into(),
        description: "Reads a diff.".into(),
        tools: vec!["read_file".into(), "grep".into()],
        body: "You review code.".into(),
        max_steps: Some(3.0),
        ..Default::default()
    };
    code(&mut app, CodeMsg::AgentsLoaded(Some(vec![reviewer.clone()])));
    let listed = app.code().tools_for_model();
    let task_def = listed.as_array().unwrap().iter().find(|d| d["function"]["name"] == "task").unwrap();
    assert_eq!(task_def["function"]["parameters"]["properties"]["agent"]["enum"], json!(["explore", "general", "code-reviewer"]), "the task tool names the agents of the folder");
    code(&mut app, CodeMsg::AgentsLoaded(None));
    assert_eq!(app.code().agents, vec![reviewer], "a route that did not answer leaves the list as it was");

    start(&mut app, r#"{"description":"Review","prompt":"Look at the diff.","agent":"code-reviewer"}"#);
    let tools: Vec<String> = app.code().kid_tools("t1").unwrap().as_array().unwrap().iter().map(|d| d["function"]["name"].as_str().unwrap().to_string()).collect();
    assert_eq!(tools, ["read_file", "grep"], "only the tools the agent was given");
    let wire = app.code().kid_wire("t1").unwrap();
    assert!(wire[0].content.ends_with("\n\n[sub-agent \"code-reviewer\"]\nYou review code."), "its own instructions follow the conversation's");
    assert_eq!(wire[1].content, "Sub-agent \"code-reviewer\" — Reads a diff.\n\nLook at the diff.");
}

#[test]
fn the_last_step_asks_for_the_report_and_ignores_calls() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    let agent = CustomAgent { name: "short".into(), max_steps: Some(2.0), ..Default::default() };
    code(&mut app, CodeMsg::AgentsLoaded(Some(vec![agent])));
    start(&mut app, r#"{"description":"x","prompt":"p","agent":"short"}"#);
    assert_eq!(progress(&app, "t1"), "step 1");
    kid_calls(&mut app, "t1", "", &[("read_file", r#"{"path":"a"}"#)]);
    code(&mut app, CodeMsg::ToolDone("t1~0.0".into(), Ok("x".into())));

    assert_eq!(progress(&app, "t1"), "step 2, writing report", "the last step is the report");
    let wire = app.code().kid_wire("t1").unwrap();
    assert_eq!(wire.last().unwrap().content, "Stop investigating now. Write your final report with what you found so far, without calling tools.");
    // it still calls a tool: the call is ignored and the words are the report
    kid_calls(&mut app, "t1", "Report so far.", &[("read_file", r#"{"path":"b"}"#)]);
    assert_eq!(task(&app, "t1").state, CallState::Done { ok: true, output: "Report so far.".into() });
    assert_eq!(nested_ids(&app, "t1"), vec!["t1~0.0".to_string()], "no card for the ignored call");
}

#[test]
fn four_work_at_a_time_and_the_rest_wait_for_a_place() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "go");
    let calls: Vec<(String, String)> = (1..=5).map(|n| (format!("t{n}"), format!(r#"{{"description":"job {n}","prompt":"p{n}"}}"#))).collect();
    let refs: Vec<(&str, &str, &str)> = calls.iter().map(|(id, args)| (id.as_str(), "task", args.as_str())).collect();
    parent_calls(&mut app, &refs);
    assert_eq!(app.code().kids_working_and_waiting(), (4, 1), "four work, one waits");
    assert_eq!(task(&app, "t5").state, CallState::Running);
    assert!(app.code().kid_wire("t5").is_some());
    assert_eq!(progress(&app, "t5"), "starting", "the one that waits has not begun");
    assert!(sim(&app).find("Starting ...").is_ok(), "its card says so");

    kid_says(&mut app, "t2", "two");
    assert_eq!(app.code().kids_working_and_waiting(), (4, 0), "the waiting one took the place");
    assert_eq!(progress(&app, "t5"), "step 1");
    // the conversation goes on only when all five are over
    assert!(!matches!(app.code().items.last(), Some(Item::Assistant { text, .. }) if text.is_empty() && task(&app, "t1").state != CallState::Running));
    for id in ["t1", "t3", "t4", "t5"] {
        kid_says(&mut app, id, id);
    }
    assert_eq!(app.code().kids_working_and_waiting(), (0, 0));
    let wire = app.code().wire(None);
    let answers: Vec<&str> = wire.iter().filter(|m| m.role == "tool").map(|m| m.tool_call_id.as_deref().unwrap()).collect();
    assert_eq!(answers, ["t1", "t2", "t3", "t4", "t5"], "answered in the order of the calls");
}

#[test]
fn a_failed_request_is_the_failure_of_the_task() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    start(&mut app, r#"{"description":"Find it","prompt":"p"}"#);
    kid(&mut app, "t1", StreamEvent::Error("The server answered with error 500".into()));
    let call = task(&app, "t1");
    assert_eq!(call.state, CallState::Done { ok: false, output: "The server answered with error 500".into() });
    assert_eq!(call.sent.as_deref(), Some("Sub-agent failed: The server answered with error 500"));
}

#[test]
fn stopping_ends_the_sub_agents_too() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    start(&mut app, r#"{"description":"Fix it","prompt":"p","agent":"general"}"#);
    kid_calls(&mut app, "t1", "", &[("edit_file", r#"{"path":"a"}"#)]);
    assert_eq!(find_call(&app, "t1~0.0").unwrap().state, CallState::Pending);
    code(&mut app, CodeMsg::Stop);
    assert!(!app.code().running);
    assert_eq!(app.code().kids_working_and_waiting(), (0, 0), "no sub-agent is left");
    assert_eq!(find_call(&app, "t1~0.0").unwrap().state, CallState::Done { ok: false, output: "Stopped.".into() }, "its waiting call is closed");
    assert_eq!(task(&app, "t1").state, CallState::Done { ok: false, output: "Stopped.".into() });
    // a late word of its stream changes nothing
    kid(&mut app, "t1", StreamEvent::Delta("late".into()));
    assert!(!kid_items(&app, "t1").iter().any(|i| matches!(i, KidItem::Assistant(t) if t == "late")), "its stream is gone");
}

#[test]
fn an_agent_is_created_after_a_question_when_it_is_for_every_folder() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::SetMode(Mode::Auto));
    say(&mut app, "make an agent");
    parent_calls(&mut app, &[
        ("c1", "create_agent", r#"{"name":"a","description":"d","instructions":"i"}"#),
        ("c2", "create_agent", r#"{"name":"b","description":"d","instructions":"i","scope":"user"}"#),
    ]);
    assert_eq!(find_call(&app, "c1").unwrap().state, CallState::Running, "for this folder it runs in the mode Everything");
    assert_eq!(find_call(&app, "c2").unwrap().state, CallState::Pending, "for every folder it always asks");
    // plan mode does not offer it
    code(&mut app, CodeMsg::SetMode(Mode::Plan));
    let offered: Vec<String> = app.code().tools_for_model().as_array().unwrap().iter().map(|d| d["function"]["name"].as_str().unwrap().to_string()).collect();
    assert!(offered.contains(&"task".to_string()), "a task reads only, so the plan agent has it");
    assert!(!offered.contains(&"create_agent".to_string()), "writing an agent is an edit");
}

#[test]
fn a_session_keeps_the_conversation_of_its_sub_agents() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    start(&mut app, r#"{"description":"Find it","prompt":"p"}"#);
    kid_calls(&mut app, "t1", "Looking.", &[("read_file", r#"{"path":"a"}"#)]);
    code(&mut app, CodeMsg::ToolDone("t1~0.0".into(), Ok("x".into())));
    kid_says(&mut app, "t1", "Found.");
    let saved = app.code().saved();
    let kept = saved.iter().find_map(|s| match s { Saved::Call { name, children, agent, .. } if name == "task" => Some((children.clone(), agent.clone())), _ => None }).expect("the task is kept");
    assert_eq!(kept.1.as_deref(), Some("explore"));
    assert_eq!(kept.0.len(), 3, "its words, its call and its last answer");
    let text = serde_json::to_string(&saved).unwrap();
    let back: Vec<Saved> = serde_json::from_str(&text).unwrap();
    assert_eq!(back, saved, "the file reads back as it was written");

    let (mut again, _dir) = self::app("light");
    open_folder(&mut again);
    code(&mut again, CodeMsg::Opened(Ok(Workspace { root: FOLDER.into(), name: "fllm".into(), branch: None })));
    let mut state = fllm_app::code::CodeState::default();
    state.restore(back);
    let call = state.items.iter().find_map(|i| if let Item::Call(c) = i { (c.name == "task").then(|| c.clone()) } else { None }).unwrap();
    let view = call.kid.expect("the card is a task card again");
    assert_eq!((view.agent_type.as_str(), view.description.as_str()), ("explore", "Find it"));
    assert_eq!(view.items.len(), 3);
    assert!(matches!(&view.items[1], KidItem::Call(c) if c.name == "read_file" && c.state == CallState::Done { ok: true, output: "x".into() }));
    assert_eq!(call.sent.as_deref(), Some("Sub-agent report (Find it):\nFound."), "the model's reply is kept with the call");
}

#[test]
fn the_card_shows_the_job_the_sub_agent_and_its_report() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    start(&mut app, r#"{"description":"Find the parser","prompt":"Where is the SSE parser?"}"#);
    {
        let mut ui = sim(&app);
        assert!(ui.find("Sub-agent (explore)").is_ok(), "the verb of the page");
        assert!(ui.find("Find the parser").is_ok(), "and the words of the job");
        assert!(ui.find("Where is the SSE parser?").is_ok(), "the job as it was given");
        assert!(ui.find("…").is_ok(), "its first answer is on its way");
        assert!(ui.find("step 1").is_ok(), "the progress line while it works");
    }
    kid_calls(&mut app, "t1", "Looking around.", &[("grep", r#"{"pattern":"parse"}"#)]);
    {
        let mut ui = sim(&app);
        assert!(ui.find("Looking around.").is_ok(), "its words stand in the card");
        assert!(ui.find("parse").is_ok(), "and its calls as cards of their own");
    }
    code(&mut app, CodeMsg::ToolDone("t1~0.0".into(), Ok("src/api.rs:1: parse".into())));
    kid_says(&mut app, "t1", "It is in src/api.rs.");
    // a finished card is closed as the page closes it; its header opens it
    code(&mut app, CodeMsg::ToggleCard("t1".into()));
    let mut ui = sim(&app);
    assert!(ui.find("Report").is_ok(), "the report has its heading");
    assert!(ui.find("step 2").is_err(), "the progress is gone when it is over");
}

#[test]
fn a_new_agent_is_offered_at_once_wherever_it_was_made() {
    // made by the model: the list is read again when the file is written, not when it failed
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    code(&mut app, CodeMsg::SetMode(Mode::Auto));
    say(&mut app, "make one");
    parent_calls(&mut app, &[("c1", "create_agent", r#"{"name":"a","description":"d","instructions":"i"}"#), ("c2", "create_agent", r#"{"name":"b","description":"d","instructions":"i"}"#)]);
    let failed = app.update(Message::Code(CodeMsg::ToolDone("c1".into(), Err("an agent with this name exists".into()))));
    let written = app.update(Message::Code(CodeMsg::ToolDone("c2".into(), Ok("The agent \"b\" was created.".into()))));
    assert!(
        written.units() >= failed.units() + 2,
        "the changes of the folder and the list of agents are read after a write: {} units after it, {} after the refusal",
        written.units(),
        failed.units()
    );

    // made on the Settings page: its list is read again, and so is the Code page's
    let saved = app.update(Message::AgentSaved("mine".into(), true, Ok(json!({}))));
    let refused = app.update(Message::AgentSaved("mine".into(), true, Err("no".into())));
    assert_eq!(refused.units(), 0);
    assert!(saved.units() >= 2, "the Settings list and the Code page's: {} units", saved.units());
    let deleted = app.update(Message::AgentDeleted("mine".into(), Ok(json!({}))));
    assert!(deleted.units() >= 2, "a deleted agent is gone from the Code page too: {} units", deleted.units());
}

fn picture(app: &App, name: &str) {
    let mut ui = Simulator::with_size(settings(), iced::Size::new(1280.0, 1000.0), app.view());
    let shot = ui.snapshot(&app.theme()).unwrap();
    let out = std::env::var("FLLM_SNAPSHOT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots"));
    std::fs::create_dir_all(&out).unwrap();
    for old in std::fs::read_dir(&out).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{name}-"))) {
        let _ = std::fs::remove_file(old.path());
    }
    assert!(shot.matches_image(out.join(format!("{name}.png"))).unwrap(), "the picture of {name}");
}

/// Two sub-agents drawn, one working and one done, for the eye of the vision check (the PNGs are
/// artifacts, not a pixel gate).
#[test]
fn the_cards_of_two_sub_agents_are_drawn() {
    let (mut app, _dir) = app("dark");
    open_folder(&mut app);
    say(&mut app, "look into it");
    parent_calls(&mut app, &[
        ("t1", "task", r#"{"description":"Find the parser","prompt":"Where is the SSE parser? Name the file and the line."}"#),
        ("t2", "task", r#"{"description":"Fix the typo","prompt":"Fix the typo in the README.","agent":"general"}"#),
    ]);
    kid_calls(&mut app, "t1", "Looking around.", &[("grep", r#"{"pattern":"parse"}"#)]);
    code(&mut app, CodeMsg::ToolDone("t1~0.0".into(), Ok("src/api.rs:120: fn parse_sse".into())));
    kid_says(&mut app, "t1", "The SSE parser is parse_sse in src/api.rs:120.");
    kid_calls(&mut app, "t2", "", &[("edit_file", r#"{"path":"README.md","old_string":"teh","new_string":"the"}"#)]);
    picture(&app, "code-subagents");
}

/// A round of the conversation with one call, answered at once: what the model does when it is
/// stuck.
fn parent_round(app: &mut App, id: &str, name: &str, args: &str) {
    parent_calls(app, &[(id, name, args)]);
    if matches!(find_call(app, id).map(|c| c.state), Some(CallState::Running)) {
        code(app, CodeMsg::ToolDone(id.into(), Ok("same".into())));
    }
}

#[test]
fn the_same_call_three_times_in_a_row_is_refused_the_third_time() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "go");
    parent_round(&mut app, "a", "read_file", r#"{"path":"x","start":1}"#);
    parent_round(&mut app, "b", "read_file", r#"{"start":1,"path":"x"}"#);
    assert_eq!(find_call(&app, "b").unwrap().state, CallState::Done { ok: true, output: "same".into() }, "two are fine, in whatever order the keys come");
    parent_round(&mut app, "c", "read_file", r#"{"path":"x","start":1}"#);
    let third = find_call(&app, "c").unwrap();
    assert_eq!(third.state, CallState::Done { ok: false, output: "The same call was repeated three times.".into() });
    assert!(third.sent.as_deref().unwrap().starts_with("Error: you have made this exact call three times in a row."));
    // another call in between starts the count again
    parent_round(&mut app, "d", "grep", r#"{"pattern":"p"}"#);
    parent_round(&mut app, "e", "read_file", r#"{"path":"x","start":1}"#);
    assert_eq!(find_call(&app, "e").unwrap().state, CallState::Done { ok: true, output: "same".into() });
}

#[test]
fn polling_and_bookkeeping_may_repeat() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "go");
    for id in ["a", "b", "c", "d"] {
        parent_round(&mut app, id, "todo_read", "{}");
        assert!(matches!(find_call(&app, id).unwrap().state, CallState::Done { ok: true, .. }), "{id}");
    }
}

#[test]
fn a_new_run_forgets_what_was_called() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "one");
    parent_round(&mut app, "a", "read_file", r#"{"path":"x"}"#);
    parent_round(&mut app, "b", "read_file", r#"{"path":"x"}"#);
    code(&mut app, CodeMsg::Stop);
    say(&mut app, "two");
    parent_round(&mut app, "c", "read_file", r#"{"path":"x"}"#);
    assert!(matches!(find_call(&app, "c").unwrap().state, CallState::Done { ok: true, .. }), "the count starts with the run");
}

#[test]
fn a_sub_agent_has_a_history_of_its_own() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "go");
    parent_round(&mut app, "a", "read_file", r#"{"path":"x"}"#);
    parent_round(&mut app, "b", "read_file", r#"{"path":"x"}"#);
    parent_calls(&mut app, &[("t1", "task", r#"{"description":"d","prompt":"p"}"#)]);
    // the sub-agent reads the same file: its first call, not the conversation's third
    kid_calls(&mut app, "t1", "", &[("read_file", r#"{"path":"x"}"#)]);
    assert_eq!(find_call(&app, "t1~0.0").unwrap().state, CallState::Running);
    code(&mut app, CodeMsg::ToolDone("t1~0.0".into(), Ok("same".into())));
    kid_calls(&mut app, "t1", "", &[("read_file", r#"{"path":"x"}"#)]);
    code(&mut app, CodeMsg::ToolDone("t1~1.0".into(), Ok("same".into())));
    kid_calls(&mut app, "t1", "", &[("read_file", r#"{"path":"x"}"#)]);
    let third = find_call(&app, "t1~2.0").unwrap();
    assert_eq!(third.state, CallState::Done { ok: false, output: "The same call was repeated three times.".into() }, "its own third time");
    assert!(third.sent.unwrap().contains("three times in a row"));
}

#[test]
fn what_a_sub_agent_did_is_not_counted_against_the_conversation() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    say(&mut app, "go");
    parent_calls(&mut app, &[("t1", "task", r#"{"description":"d","prompt":"p"}"#)]);
    kid_calls(&mut app, "t1", "", &[("read_file", r#"{"path":"x"}"#)]);
    code(&mut app, CodeMsg::ToolDone("t1~0.0".into(), Ok("same".into())));
    kid_calls(&mut app, "t1", "", &[("read_file", r#"{"path":"x"}"#)]);
    code(&mut app, CodeMsg::ToolDone("t1~1.0".into(), Ok("same".into())));
    kid_says(&mut app, "t1", "done");
    // two reads of the sub-agent stand before this one; it is the conversation's first
    parent_round(&mut app, "a", "read_file", r#"{"path":"x"}"#);
    assert!(matches!(find_call(&app, "a").unwrap().state, CallState::Done { ok: true, .. }), "{:?}", find_call(&app, "a").unwrap().state);
}

#[test]
fn a_sub_agent_may_not_use_the_screen() {
    let (mut app, _dir) = app("light");
    open_folder(&mut app);
    // the server offers a screen tool when the setting is on
    let mut all = vec![Tool::server("read_file", Permission::Read), Tool::server("computer_click", Permission::Computer)];
    let mut defs: Vec<Value> = all.iter().map(|t| def(&t.name)).collect();
    let (client, client_defs) = client_tools();
    all.extend(client);
    defs.extend(client_defs);
    code(&mut app, CodeMsg::ToolsLoaded(Ok((all, Value::Array(defs)))));
    code(&mut app, CodeMsg::SetMode(Mode::Auto));
    start(&mut app, r#"{"description":"d","prompt":"p","agent":"general"}"#);
    kid_calls(&mut app, "t1", "", &[("computer_click", r#"{"x":1,"y":2}"#)]);
    let c = find_call(&app, "t1~0.0").unwrap();
    assert_eq!(c.state, CallState::Done { ok: false, output: "Not available to sub-agents.".into() }, "even a general agent in the mode Everything");
    assert_eq!(c.sent.as_deref(), Some("Denied: a sub-agent may not use the screen, mouse or keyboard. Do that work yourself."));
}
