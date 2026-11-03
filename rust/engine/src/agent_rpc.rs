//! `fllm-rs agent --rpc`: the coding agent as a library for other programs. JSON-RPC 2.0 over standard input and output, one JSON
//! message per line, so an editor extension, a script or a web service can start the process and drive it (the role Pi's RPC mode
//! and Claude Code's Agent SDK have). Unlike `fllm-rs acp`, which follows the editor protocol, this one exposes what the agent can
//! do directly: a folder per session, the permission categories, extra allow/deny/ask rules, plan mode, a model and a system text.
//!
//! Client to agent (requests unless marked as a notification):
//! * `initialize` -> `{name, version, protocol: 1, methods, categories}`
//! * `session/new` `{folder, allow?: ["edit", "command", ...], plan?, model?, system?, max_steps?, allow_rules?, deny_rules?, ask_rules?}` -> `{session}`
//! * `session/prompt` `{session, text}` -> `{answer, status, steps, edited, tool_calls}` after `session/event` notifications
//! * `session/cancel` `{session}` (notification) ends the running prompt
//! * `session/history` `{session}` -> `{messages}`; `session/close` `{session}`; `tools/list` -> `{tools}`; `shutdown`
//!
//! Agent to client: `session/event` notifications `{session, type: "text" | "tool_call" | "tool_result", ...}` and, for an action the
//! permissions do not cover, the request `session/approve` `{session, call_id, tool, category, args}` that the client answers with
//! `{allow: bool, always?: bool}` (no answer within the prompt means refused). Reading is always allowed.
//! Errors use the JSON-RPC codes (-32602 invalid params, -32601 unknown method, -32603 failure).
use crate::agent_cli::{self, LoopConfig, Status};
use crate::agent_rules::{self, Rules};
use crate::acp::AcpBackend;
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

pub const HELP: &str = "usage: fllm-rs agent --rpc [--url http://127.0.0.1:8000] [--api-key KEY] [--model ID]
Runs the coding agent as a JSON-RPC 2.0 server on standard input and output (one JSON message per line) so that editors, scripts
and services can embed it. Methods: initialize, session/new (folder, allow, plan, model, rules ...), session/prompt, session/cancel,
session/history, session/close, tools/list, shutdown. Progress arrives as session/event notifications; actions that are not
allowed are put to the client with session/approve. Clients: scripts/sdk/fllm_agent.py and scripts/sdk/fllm-agent.mjs.";

pub const CATEGORIES: [&str; 5] = ["read", "edit", "command", "network", "memory"];

struct Session {
    folder: String,
    history: Vec<Value>,
    allow: BTreeSet<String>,
    always: BTreeSet<String>,
    plan: bool,
    model: Option<String>,
    system: Option<String>,
    max_steps: usize,
    rules: Rules,
}

pub struct Defaults {
    pub model: Option<String>,
    pub max_steps: usize,
}

pub struct Rpc<'a> {
    backend: &'a mut dyn AcpBackend,
    out: RefCell<&'a mut dyn Write>,
    rx: &'a Receiver<Value>,
    next_request: Cell<u64>,
    defaults: Defaults,
    sessions: HashMap<String, Session>,
    next_session: u64,
    pub shutdown: bool,
}

struct Wire<'w, 'a> {
    out: &'w RefCell<&'a mut dyn Write>,
}

impl Wire<'_, '_> {
    fn send(&self, message: &Value) {
        let mut out = self.out.borrow_mut();
        let _ = writeln!(out, "{message}");
        let _ = out.flush();
    }
}

fn string_list(params: &Value, key: &str) -> Vec<String> {
    params[key].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(str::to_owned)).collect()
}

impl<'a> Rpc<'a> {
    pub fn new(backend: &'a mut dyn AcpBackend, out: &'a mut dyn Write, rx: &'a Receiver<Value>, defaults: Defaults) -> Self {
        Self { backend, out: RefCell::new(out), rx, next_request: Cell::new(5000), defaults, sessions: HashMap::new(), next_session: 0, shutdown: false }
    }

    fn wire(&self) -> Wire<'_, 'a> {
        Wire { out: &self.out }
    }

    fn reply(&self, id: &Value, result: Value) {
        self.wire().send(&json!({"jsonrpc": "2.0", "id": id, "result": result}));
    }

    fn fail(&self, id: &Value, code: i64, message: &str) {
        self.wire().send(&json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}));
    }

    pub fn handle(&mut self, message: &Value) {
        let id = message.get("id").cloned();
        let method = message["method"].as_str().unwrap_or("");
        let params = &message["params"];
        match (method, id) {
            ("initialize", Some(id)) => self.reply(&id, json!({
                "name": "fllm-agent",
                "version": env!("CARGO_PKG_VERSION"),
                "protocol": 1,
                "methods": ["initialize", "session/new", "session/prompt", "session/cancel", "session/history", "session/close", "tools/list", "shutdown"],
                "categories": CATEGORIES,
            })),
            ("session/new", Some(id)) => self.new_session(&id, params),
            ("session/prompt", Some(id)) => self.prompt(&id, params),
            ("session/cancel", None) => agent_cli::CANCEL.store(true, std::sync::atomic::Ordering::SeqCst),
            ("session/history", Some(id)) => match self.sessions.get(params["session"].as_str().unwrap_or("")) {
                Some(s) => self.reply(&id, json!({"messages": s.history})),
                None => self.fail(&id, -32602, "unknown session"),
            },
            ("session/close", Some(id)) => {
                let removed = self.sessions.remove(params["session"].as_str().unwrap_or("")).is_some();
                self.reply(&id, json!({"closed": removed}));
            }
            ("tools/list", Some(id)) => match self.backend.tools() {
                Ok(tools) => {
                    let (schemas, categories) = agent_cli::prepare_tools(tools);
                    let list: Vec<Value> = schemas.iter().map(|t| json!({"name": t["function"]["name"], "description": t["function"]["description"], "category": categories.get(t["function"]["name"].as_str().unwrap_or("")).cloned().unwrap_or_default()})).collect();
                    self.reply(&id, json!({"tools": list}));
                }
                Err(e) => self.fail(&id, -32603, &e),
            },
            ("shutdown", Some(id)) => {
                self.reply(&id, json!({}));
                self.shutdown = true;
            }
            (_, Some(id)) => self.fail(&id, -32601, &format!("method not found: {method}")),
            _ => {}
        }
    }

    fn new_session(&mut self, id: &Value, params: &Value) {
        let Some(folder) = params["folder"].as_str().filter(|c| std::path::Path::new(c).is_absolute()) else {
            return self.fail(id, -32602, "folder must be an absolute path");
        };
        let mut allow = BTreeSet::from(["read".to_owned()]);
        for c in string_list(params, "allow") {
            if c == "all" {
                allow.extend(CATEGORIES.iter().map(|c| (*c).to_owned()));
            } else if CATEGORIES.contains(&c.as_str()) {
                allow.insert(c);
            } else {
                return self.fail(id, -32602, &format!("allow: unknown category '{c}'; use {} or all", CATEGORIES.join(", ")));
            }
        }
        if let Err(e) = self.backend.set_folder(folder) {
            return self.fail(id, -32603, &format!("the folder cannot be used: {e}"));
        }
        let mut rules = agent_rules::load_hierarchy(&agent_cli::studio_folder(), std::path::Path::new(folder)).unwrap_or_default();
        rules.merge(Rules::from_json(&json!({"allow": string_list(params, "allow_rules"), "deny": string_list(params, "deny_rules"), "ask": string_list(params, "ask_rules")})));
        let plan = params["plan"].as_bool().unwrap_or(false) || rules.mode.as_deref() == Some("plan");
        if plan {
            allow = BTreeSet::from(["read".to_owned()]);
            rules.merge(Rules::from_json(&json!({"deny": ["bash: *", "edit: **", "network: *", "memory: *"]})));
        }
        self.next_session += 1;
        let sid = format!("rpc_{}_{}", std::process::id(), self.next_session);
        let max_steps = params["max_steps"].as_u64().map_or(self.defaults.max_steps, |n| (n as usize).clamp(1, 200));
        self.sessions.insert(
            sid.clone(),
            Session {
                folder: folder.to_owned(),
                history: Vec::new(),
                allow,
                always: BTreeSet::new(),
                plan,
                model: params["model"].as_str().map(str::to_owned).or_else(|| self.defaults.model.clone()),
                system: params["system"].as_str().map(str::to_owned),
                max_steps,
                rules,
            },
        );
        self.reply(id, json!({"session": sid, "plan": plan}));
    }

    fn prompt(&mut self, id: &Value, params: &Value) {
        let sid = params["session"].as_str().unwrap_or("").to_owned();
        let text = params["text"].as_str().unwrap_or("").to_owned();
        let Some(session) = self.sessions.get(&sid) else { return self.fail(id, -32602, "unknown session") };
        if text.trim().is_empty() {
            return self.fail(id, -32602, "text is empty");
        }
        let folder = session.folder.clone();
        let mut history = agent_cli::trim_history(session.history.clone(), 120);
        let always = RefCell::new(session.always.clone());
        let (allow, rules, plan, system_extra, max_steps) = (session.allow.clone(), session.rules.clone(), session.plan, session.system.clone(), session.max_steps);
        let model_choice = session.model.clone();
        if let Err(e) = self.backend.set_folder(&folder) {
            return self.fail(id, -32603, &format!("the folder cannot be used: {e}"));
        }
        let (model, context) = match model_choice {
            Some(m) => (m, None),
            None => match self.backend.model() {
                Ok(m) => m,
                Err(e) => return self.fail(id, -32603, &e),
            },
        };
        // hooks on the task: a user_prompt_submit hook may refuse it or add notes
        let submitted = self.backend.fire_hook("user_prompt_submit", "", &json!({"prompt": text}));
        if submitted.blocked {
            return self.fail(id, -32603, &format!("a user_prompt_submit hook blocked the prompt: {}", submitted.reason));
        }
        history.push(json!({"role": "user", "content": agent_cli::with_hook_context(&text, &submitted)}));
        let mut extra = system_extra.unwrap_or_default();
        if plan {
            extra.push_str("\n\nPLAN MODE: you may only read and search; every other action is refused. Investigate, then finish with a numbered plan. Do not claim that anything was changed.");
        }
        let cfg = LoopConfig { model: Some(model), allow: allow.clone(), rules: rules.clone(), max_steps, max_tokens: agent_cli::effective_max_tokens(4096, context), temperature: 0.2 };
        let system = agent_cli::full_system(&mut *self.backend, &folder, &allow, Some(extra.as_str()), None);
        agent_cli::CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);

        let announced: RefCell<Vec<(String, String, Value)>> = RefCell::new(Vec::new());
        let wire = Wire { out: &self.out };
        let rx = self.rx;
        let next_request = &self.next_request;
        let sid_ref = sid.as_str();
        let event = |body: Value| {
            let mut params = body;
            params["session"] = json!(sid_ref);
            wire.send(&json!({"jsonrpc": "2.0", "method": "session/event", "params": params}));
        };
        let mut observe = |m: &Value| match m["role"].as_str() {
            Some("assistant") => {
                if let Some(t) = m["content"].as_str().filter(|t| !t.trim().is_empty()) {
                    event(json!({"type": "text", "text": t}));
                }
                for call in m["tool_calls"].as_array().into_iter().flatten() {
                    let name = call["function"]["name"].as_str().unwrap_or("").to_owned();
                    let args = agent_cli::parse_call_args(&call["function"]["arguments"]).unwrap_or_else(|_| json!({}));
                    let cid = call["id"].as_str().unwrap_or("").to_owned();
                    announced.borrow_mut().push((cid.clone(), name.clone(), args.clone()));
                    event(json!({"type": "tool_call", "id": cid, "name": name, "args": args}));
                }
            }
            Some("tool") => {
                let text = m["content"].as_str().unwrap_or("");
                event(json!({"type": "tool_result", "id": m["tool_call_id"], "ok": !text.starts_with("Error:"), "text": agent_cli::clip(text, 4000)}));
            }
            _ => {}
        };
        let mut approve = |name: &str, category: &str, args: &Value| -> bool {
            if always.borrow().contains(category) {
                return true;
            }
            let call_id = announced.borrow().iter().rev().find(|(_, n, a)| n == name && a == args).map(|(i, _, _)| i.clone()).unwrap_or_default();
            let request_id = next_request.get();
            next_request.set(request_id + 1);
            wire.send(&json!({"jsonrpc": "2.0", "id": request_id, "method": "session/approve", "params": {"session": sid_ref, "call_id": call_id, "tool": name, "category": category, "args": args}}));
            loop {
                match rx.recv_timeout(Duration::from_millis(200)) {
                    Ok(reply) if reply["id"] == json!(request_id) && reply.get("method").is_none() => {
                        let allowed = reply["result"]["allow"].as_bool().unwrap_or(false);
                        if allowed && reply["result"]["always"].as_bool().unwrap_or(false) {
                            always.borrow_mut().insert(category.to_owned());
                        }
                        return allowed;
                    }
                    Ok(other) if other["method"] == "session/cancel" => {
                        agent_cli::CANCEL.store(true, std::sync::atomic::Ordering::SeqCst);
                        return false;
                    }
                    Ok(_) => {}
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return false,
                }
            }
        };
        let outcome = agent_cli::run_loop_full(&mut *self.backend, &cfg, &system, history, &mut |_| {}, &mut approve, &mut observe);
        let always_after = always.into_inner();
        match outcome {
            Ok(mut out) => {
                // a stop hook may send the agent back for one more round
                let stop = self.backend.fire_hook("stop", "", &json!({"answer": out.answer, "edited": out.edited}));
                if stop.blocked {
                    let mut next = out.messages.clone();
                    next.push(json!({"role": "user", "content": format!("A stop hook asks you to continue before you finish: {}", stop.reason)}));
                    let wire = Wire { out: &self.out };
                    let mut observe_again = |m: &Value| {
                        if m["role"] == "assistant" {
                            if let Some(t) = m["content"].as_str().filter(|t| !t.trim().is_empty()) {
                                wire.send(&json!({"jsonrpc": "2.0", "method": "session/event", "params": {"session": sid_ref, "type": "text", "text": t}}));
                            }
                        }
                    };
                    if let Ok(again) = agent_cli::run_loop_full(&mut *self.backend, &cfg, &system, next, &mut |_| {}, &mut |_, _, _| false, &mut observe_again) {
                        out = agent_cli::merge_rounds(out, again);
                    }
                }
                if let Some(s) = self.sessions.get_mut(&sid) {
                    s.history = out.messages.clone();
                    s.always = always_after;
                }
                let status = match out.status {
                    Status::Done => "done",
                    Status::OutOfSteps => "out_of_steps",
                    Status::Cancelled => "cancelled",
                };
                let calls: Vec<Value> = out.tool_calls.iter().map(|c| json!({"tool": c.name, "permission": c.category, "status": c.status, "detail": c.detail})).collect();
                self.reply(id, json!({"answer": out.answer, "status": status, "steps": out.steps, "edited": out.edited, "tool_calls": calls}));
            }
            Err(e) => self.fail(id, -32603, &e),
        }
    }
}

pub fn run(args: &[String]) -> Result<i32, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let mut agent_args: Vec<String> = args.iter().filter(|a| *a != "--rpc").cloned().collect();
    agent_args.extend(["-p".to_owned(), "rpc".to_owned()]);
    let parsed = agent_cli::parse_args(&agent_args)?;
    let mut http = agent_cli::HttpBackend::new(&parsed.url, parsed.api_key.clone())?;
    let rx = crate::acp::spawn_reader(Box::new(std::io::BufReader::new(std::io::stdin())));
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let mut rpc = Rpc::new(&mut http, &mut lock, &rx, Defaults { model: parsed.model.clone(), max_steps: parsed.max_steps });
    while let Ok(message) = rx.recv() {
        rpc.handle(&message);
        if rpc.shutdown {
            break;
        }
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_cli::Backend;
    use std::collections::VecDeque;
    use std::sync::mpsc;

    struct Fake {
        replies: VecDeque<Value>,
        ran: Vec<String>,
        hook_blocks: usize,
    }

    impl Backend for Fake {
        fn tools(&mut self) -> Result<Vec<Value>, String> {
            let t = |n: &str, p: &str| json!({"type": "function", "function": {"name": n, "description": "d", "parameters": {"type": "object", "properties": {}}}, "permission": p});
            Ok(vec![t("read_file", "read"), t("write_file", "edit"), t("bash", "command")])
        }
        fn complete(&mut self, _: Value) -> Result<Value, String> {
            self.replies.pop_front().ok_or_else(|| "script ended".to_owned())
        }
        fn run_tool(&mut self, name: &str, args: &Value) -> Result<String, String> {
            self.ran.push(format!("{name} {args}"));
            Ok(format!("ran {name}"))
        }
        fn fire_hook(&mut self, event: &str, _: &str, _: &Value) -> agent_cli::HookReply {
            if event == "stop" && self.hook_blocks > 0 {
                self.hook_blocks -= 1;
                return agent_cli::HookReply { blocked: true, reason: "run the tests".into(), ..Default::default() };
            }
            agent_cli::HookReply::default()
        }
    }

    impl AcpBackend for Fake {
        fn set_folder(&mut self, _: &str) -> Result<(), String> {
            Ok(())
        }
        fn model(&mut self) -> Result<(String, Option<u64>), String> {
            Ok(("m".into(), Some(8192)))
        }
    }

    fn call(id: &str, name: &str, args: Value) -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": "working", "tool_calls": [{"id": id, "type": "function", "function": {"name": name, "arguments": args.to_string()}}]}}]})
    }

    fn text(t: &str) -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": t}}]})
    }

    fn lines(out: &[u8]) -> Vec<Value> {
        String::from_utf8_lossy(out).lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }

    fn abs() -> String {
        std::env::temp_dir().display().to_string()
    }

    fn sid() -> String {
        format!("rpc_{}_1", std::process::id())
    }

    fn drive(fake: &mut Fake, inbox: Vec<Value>, messages: Vec<Value>) -> Vec<Value> {
        let mut out = Vec::new();
        let (tx, rx) = mpsc::channel();
        for m in inbox {
            tx.send(m).unwrap();
        }
        {
            let mut rpc = Rpc::new(fake, &mut out, &rx, Defaults { model: None, max_steps: 6 });
            for m in messages {
                rpc.handle(&m);
            }
        }
        lines(&out)
    }

    #[test]
    fn a_session_runs_a_prompt_streams_events_and_returns_the_report() {
        let mut fake = Fake { replies: VecDeque::from([call("c1", "read_file", json!({"path": "a.rs"})), text("All done.")]), ran: vec![], hook_blocks: 0 };
        let msgs = drive(
            &mut fake,
            vec![],
            vec![
                json!({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
                json!({"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"folder": abs()}}),
                json!({"jsonrpc": "2.0", "id": 3, "method": "session/new", "params": {"folder": "rel"}}),
                json!({"jsonrpc": "2.0", "id": 4, "method": "session/new", "params": {"folder": abs(), "allow": ["bogus"]}}),
                json!({"jsonrpc": "2.0", "id": 5, "method": "session/prompt", "params": {"session": sid(), "text": "look"}}),
                json!({"jsonrpc": "2.0", "id": 6, "method": "session/history", "params": {"session": sid()}}),
                json!({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}),
                json!({"jsonrpc": "2.0", "id": 8, "method": "nothing"}),
            ],
        );
        let by_id = |id: u64| msgs.iter().find(|m| m["id"] == id).unwrap();
        assert_eq!(by_id(1)["result"]["protocol"], 1);
        assert_eq!(by_id(2)["result"]["session"], sid().as_str());
        assert_eq!(by_id(3)["error"]["code"], -32602);
        assert_eq!(by_id(4)["error"]["code"], -32602);
        let report = &by_id(5)["result"];
        assert_eq!((report["answer"].as_str(), report["status"].as_str(), report["steps"].as_u64()), (Some("All done."), Some("done"), Some(2)));
        assert_eq!(report["tool_calls"][0]["tool"], "read_file");
        let kinds: Vec<&str> = msgs.iter().filter(|m| m["method"] == "session/event").map(|m| m["params"]["type"].as_str().unwrap()).collect();
        assert_eq!(kinds, vec!["text", "tool_call", "tool_result", "text"]);
        assert!(by_id(6)["result"]["messages"].as_array().unwrap().len() >= 4);
        assert_eq!(by_id(7)["result"]["tools"][1]["category"], "edit");
        assert_eq!(by_id(8)["error"]["code"], -32601);
        assert_eq!(fake.ran.len(), 1);
    }

    #[test]
    fn actions_outside_the_allowed_categories_are_put_to_the_client() {
        let mut fake = Fake { replies: VecDeque::from([call("c1", "bash", json!({"command": "ls"})), text("listed")]), ran: vec![], hook_blocks: 0 };
        // the client approves request 5000
        let msgs = drive(
            &mut fake,
            vec![json!({"jsonrpc": "2.0", "id": 5000, "result": {"allow": true}})],
            vec![
                json!({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"folder": abs()}}),
                json!({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"session": sid(), "text": "list files"}}),
            ],
        );
        let ask = msgs.iter().find(|m| m["method"] == "session/approve").expect("an approval request");
        assert_eq!((ask["params"]["tool"].as_str(), ask["params"]["category"].as_str(), ask["params"]["call_id"].as_str()), (Some("bash"), Some("command"), Some("c1")));
        assert_eq!(fake.ran.len(), 1);

        // a plan-mode session refuses even when the client would approve; allow rules and categories lift the question
        let mut fake = Fake { replies: VecDeque::from([call("c1", "write_file", json!({"path": "x", "content": "y"})), text("could not")]), ran: vec![], hook_blocks: 0 };
        let msgs = drive(
            &mut fake,
            vec![json!({"jsonrpc": "2.0", "id": 5000, "result": {"allow": true}})],
            vec![
                json!({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"folder": abs(), "plan": true, "allow": ["edit"]}}),
                json!({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"session": sid(), "text": "write"}}),
            ],
        );
        assert!(fake.ran.is_empty());
        assert_eq!(msgs.iter().find(|m| m["id"] == 1).unwrap()["result"]["plan"], true);
        let mut fake = Fake { replies: VecDeque::from([call("c1", "write_file", json!({"path": "x", "content": "y"})), text("written")]), ran: vec![], hook_blocks: 0 };
        let msgs = drive(
            &mut fake,
            vec![],
            vec![
                json!({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"folder": abs(), "allow": ["edit"]}}),
                json!({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"session": sid(), "text": "write"}}),
            ],
        );
        assert!(msgs.iter().all(|m| m["method"] != "session/approve"));
        assert_eq!(fake.ran.len(), 1);
    }

    #[test]
    fn a_stop_hook_sends_the_agent_back_once_and_shutdown_ends_the_loop() {
        let mut fake = Fake { replies: VecDeque::from([text("first answer"), text("after the tests")]), ran: vec![], hook_blocks: 1 };
        let msgs = drive(
            &mut fake,
            vec![],
            vec![
                json!({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"folder": abs()}}),
                json!({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"session": sid(), "text": "go"}}),
                json!({"jsonrpc": "2.0", "id": 3, "method": "shutdown"}),
            ],
        );
        let report = &msgs.iter().find(|m| m["id"] == 2).unwrap()["result"];
        assert_eq!(report["answer"], "after the tests");
        assert_eq!(report["steps"], 2);
        assert!(msgs.iter().any(|m| m["id"] == 3));
    }
}
