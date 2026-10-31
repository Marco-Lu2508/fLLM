//! `fllm-rs acp`: fLLM's coding agent as an Agent Client Protocol agent (agentclientprotocol.com), so an editor that speaks ACP
//! (Zed and others) can start it as a subprocess and talk to it over standard input and output: JSON-RPC 2.0, one message per
//! line. Methods: `initialize`, `session/new`, `session/prompt` (with `session/update` notifications: message chunks, tool calls
//! and their results), `session/cancel`, and `session/request_permission` towards the editor for actions the permissions do not
//! allow. The agent loop, the tools, the sandbox and the permission rules are those of `fllm-rs agent`.
//!
//! Written from the public protocol pages (initialization, session setup, prompt turn, tool calls). It was tested against a scripted
//! client in the unit tests, not against a real editor.
use crate::agent_cli::{self, Backend, LoopConfig, Status};
use crate::agent_rules;
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, Write};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Duration;

pub const HELP: &str = "usage: fllm-rs acp [--url http://127.0.0.1:8000] [--api-key KEY] [--model ID]
                [--allow-edit] [--allow-command] [--allow-network] [--allow-memory] [--yes] [--max-steps N]
Runs the coding agent as an Agent Client Protocol agent on standard input and output (for editors such as Zed). Register it in the
editor as an external agent whose command is `fllm-rs acp --url ...`. Actions that the --allow-* flags and the permission rules do
not cover are put to the editor with session/request_permission; reading is always allowed. The folder of a session is the cwd the
editor sends. Needs a running fllm-rs server with a model that can call tools.";

/// What the adapter needs besides the agent loop: choosing the folder the tools work in.
pub trait AcpBackend: Backend {
    fn set_folder(&mut self, folder: &str) -> Result<(), String>;
    fn model(&mut self) -> Result<(String, Option<u64>), String>;
}

impl AcpBackend for agent_cli::HttpBackend {
    fn set_folder(&mut self, folder: &str) -> Result<(), String> {
        agent_cli::HttpBackend::set_folder(self, folder)
    }
    fn model(&mut self) -> Result<(String, Option<u64>), String> {
        self.model_info()
    }
}

pub struct Settings {
    pub allow: BTreeSet<String>,
    pub max_steps: usize,
    pub model: Option<String>,
}

/// The wire: where messages go and where the editor's answers come from. Separate from the backend so that both can be borrowed at once.
struct Io<'a> {
    out: RefCell<&'a mut dyn Write>,
    rx: &'a Receiver<Value>,
    next_request: Cell<u64>,
}

pub struct Acp<'a> {
    backend: &'a mut dyn AcpBackend,
    io: Io<'a>,
    settings: Settings,
    sessions: HashMap<String, Session>,
    next_session: u64,
}

struct Session {
    folder: String,
    history: Vec<Value>,
    always: BTreeSet<String>,
}

fn kind_of(category: &str, tool: &str) -> &'static str {
    match (category, tool) {
        (_, "glob" | "grep" | "lsp" | "session_search") => "search",
        ("read", _) => "read",
        ("edit", _) => "edit",
        ("command", _) => "execute",
        ("network", _) => "fetch",
        _ => "other",
    }
}

fn title_of(name: &str, args: &Value) -> String {
    let detail = ["command", "path", "url", "query", "pattern"].iter().find_map(|k| args[*k].as_str()).unwrap_or("");
    let detail: String = detail.replace('\n', " ").chars().take(80).collect();
    if detail.is_empty() { name.to_owned() } else { format!("{name} {detail}") }
}

/// The text of a prompt: text blocks, links and embedded resources. Pictures are not supported (the agent says so in `initialize`).
pub fn prompt_text(blocks: &[Value]) -> String {
    let mut out = Vec::new();
    for b in blocks {
        match b["type"].as_str() {
            Some("text") => out.push(b["text"].as_str().unwrap_or("").to_owned()),
            Some("resource_link") => out.push(format!("(the user referred to {})", b["uri"].as_str().or_else(|| b["name"].as_str()).unwrap_or("a file"))),
            Some("resource") => {
                let r = &b["resource"];
                if let Some(text) = r["text"].as_str() {
                    out.push(format!("Contents of {}:\n```\n{text}\n```", r["uri"].as_str().unwrap_or("a file")));
                }
            }
            Some("image") => out.push("(the user attached a picture, which this agent cannot see)".to_owned()),
            _ => {}
        }
    }
    out.join("\n\n")
}

impl<'a> Acp<'a> {
    pub fn new(backend: &'a mut dyn AcpBackend, out: &'a mut dyn Write, rx: &'a Receiver<Value>, settings: Settings) -> Self {
        Self { backend, io: Io { out: RefCell::new(out), rx, next_request: Cell::new(1000) }, settings, sessions: HashMap::new(), next_session: 0 }
    }

    fn reply(&self, id: &Value, result: Value) {
        self.io.reply(id, result);
    }

    fn fail(&self, id: &Value, code: i64, message: &str) {
        self.io.fail(id, code, message);
    }
}

impl Io<'_> {
    fn send(&self, message: &Value) {
        let mut out = self.out.borrow_mut();
        let _ = writeln!(out, "{message}");
        let _ = out.flush();
    }

    fn reply(&self, id: &Value, result: Value) {
        self.send(&json!({"jsonrpc": "2.0", "id": id, "result": result}));
    }

    fn fail(&self, id: &Value, code: i64, message: &str) {
        self.send(&json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}));
    }

    fn update(&self, session: &str, update: Value) {
        self.send(&json!({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session, "update": update}}));
    }
}

impl Acp<'_> {
    /// Handles one message from the editor.
    pub fn handle(&mut self, message: &Value) {
        let id = message.get("id").cloned();
        let method = message["method"].as_str().unwrap_or("");
        let params = &message["params"];
        match (method, id) {
            ("initialize", Some(id)) => {
                // the editor sends the newest version it knows; this adapter speaks version 1
                self.reply(&id, json!({
                    "protocolVersion": 1,
                    "agentCapabilities": {"loadSession": false, "promptCapabilities": {"image": false, "audio": false, "embeddedContext": true}, "mcpCapabilities": {"http": false, "sse": false}},
                    "agentInfo": {"name": "fllm", "title": "fLLM Code", "version": env!("CARGO_PKG_VERSION")},
                    "authMethods": [],
                }));
            }
            ("session/new", Some(id)) => {
                let Some(cwd) = params["cwd"].as_str().filter(|c| std::path::Path::new(c).is_absolute()) else { return self.fail(&id, -32602, "cwd must be an absolute path") };
                match self.backend.set_folder(cwd) {
                    Ok(()) => {
                        self.next_session += 1;
                        let sid = format!("fllm_{}_{}", std::process::id(), self.next_session);
                        self.sessions.insert(sid.clone(), Session { folder: cwd.to_owned(), history: Vec::new(), always: BTreeSet::new() });
                        self.reply(&id, json!({"sessionId": sid}));
                    }
                    Err(e) => self.fail(&id, -32603, &format!("the folder cannot be used: {e}")),
                }
            }
            ("session/prompt", Some(id)) => self.prompt(&id, params),
            ("session/cancel", None) => agent_cli::CANCEL.store(true, std::sync::atomic::Ordering::SeqCst),
            ("session/load" | "session/set_mode" | "authenticate", Some(id)) => self.fail(&id, -32601, &format!("{method} is not supported")),
            (_, Some(id)) => self.fail(&id, -32601, &format!("method not found: {method}")),
            _ => {}
        }
    }

    fn prompt(&mut self, id: &Value, params: &Value) {
        let sid = params["sessionId"].as_str().unwrap_or("").to_owned();
        let text = prompt_text(params["prompt"].as_array().map_or(&[][..], Vec::as_slice));
        let Some(session) = self.sessions.get(&sid) else { return self.fail(id, -32602, "unknown sessionId") };
        if text.trim().is_empty() {
            return self.fail(id, -32602, "the prompt is empty");
        }
        let folder = session.folder.clone();
        let mut history = agent_cli::trim_history(session.history.clone(), 120);
        let always = RefCell::new(session.always.clone());
        history.push(json!({"role": "user", "content": text}));
        if let Err(e) = self.backend.set_folder(&folder) {
            return self.fail(id, -32603, &format!("the folder cannot be used: {e}"));
        }
        let (model, context) = match self.settings.model.clone() {
            Some(m) => (m, None),
            None => match self.backend.model() {
                Ok(m) => m,
                Err(e) => return self.fail(id, -32603, &e),
            },
        };
        let rules = agent_rules::load_hierarchy(&agent_cli::studio_folder(), std::path::Path::new(&folder)).unwrap_or_default();
        let cfg = LoopConfig {
            model: Some(model),
            allow: self.settings.allow.clone(),
            rules: rules.clone(),
            max_steps: self.settings.max_steps,
            max_tokens: agent_cli::effective_max_tokens(4096, context),
            temperature: 0.2,
        };
        let system = agent_cli::full_system(&mut *self.backend, &folder, &cfg.allow, None, None);
        agent_cli::CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);

        // the loop borrows the backend; everything else the callbacks need is shared through cells
        let announced: RefCell<Vec<(String, String, Value)>> = RefCell::new(Vec::new());
        let acp: &Io = &self.io;
        let sid_ref = sid.as_str();
        let mut observe = |m: &Value| match m["role"].as_str() {
            Some("assistant") => {
                if let Some(t) = m["content"].as_str().filter(|t| !t.trim().is_empty()) {
                    acp.update(sid_ref, json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": t}}));
                }
                for call in m["tool_calls"].as_array().into_iter().flatten() {
                    let name = call["function"]["name"].as_str().unwrap_or("").to_owned();
                    let args = agent_cli::parse_call_args(&call["function"]["arguments"]).unwrap_or_else(|_| json!({}));
                    let cid = call["id"].as_str().unwrap_or("").to_owned();
                    announced.borrow_mut().push((cid.clone(), name.clone(), args.clone()));
                    let category = match name.as_str() {
                        "bash" | "bash_kill" => "command",
                        "write_file" | "edit_file" | "multi_edit" | "patch" | "notebook_edit" => "edit",
                        "web_fetch" | "web_search" => "network",
                        "memory" => "memory",
                        _ => "read",
                    };
                    acp.update(sid_ref, json!({"sessionUpdate": "tool_call", "toolCallId": cid, "title": title_of(&name, &args), "kind": kind_of(category, &name), "status": "pending", "rawInput": args}));
                }
            }
            Some("tool") => {
                let text = m["content"].as_str().unwrap_or("");
                let failed = text.starts_with("Error:");
                acp.update(sid_ref, json!({
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": m["tool_call_id"],
                    "status": if failed { "failed" } else { "completed" },
                    "content": [{"type": "content", "content": {"type": "text", "text": agent_cli::clip(text, 4000)}}],
                }));
            }
            _ => {}
        };
        let mut approve = |name: &str, category: &str, args: &Value| -> bool {
            if always.borrow().contains(category) {
                return true;
            }
            let call_id = announced.borrow().iter().rev().find(|(_, n, a)| n == name && a == args).map(|(i, _, _)| i.clone()).unwrap_or_default();
            let request_id = acp.next_request.get();
            acp.next_request.set(request_id + 1);
            acp.send(&json!({"jsonrpc": "2.0", "id": request_id, "method": "session/request_permission", "params": {
                "sessionId": sid_ref,
                "toolCall": {"toolCallId": call_id, "title": title_of(name, args), "kind": kind_of(category, name), "status": "pending"},
                "options": [
                    {"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
                    {"optionId": "allow-always", "name": format!("Allow {category} actions in this session"), "kind": "allow_always"},
                    {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"},
                ],
            }}));
            // wait for the answer; a cancel ends the wait
            loop {
                match acp.rx.recv_timeout(Duration::from_millis(200)) {
                    Ok(reply) if reply["id"] == json!(request_id) && reply.get("method").is_none() => {
                        let outcome = &reply["result"]["outcome"];
                        return match (outcome["outcome"].as_str(), outcome["optionId"].as_str()) {
                            (Some("selected"), Some("allow-once")) => true,
                            (Some("selected"), Some("allow-always")) => {
                                always.borrow_mut().insert(category.to_owned());
                                true
                            }
                            _ => false,
                        };
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
            Ok(out) => {
                if let Some(s) = self.sessions.get_mut(&sid) {
                    s.history = out.messages.clone();
                    s.always = always_after;
                }
                let stop = match out.status {
                    Status::Done => "end_turn",
                    Status::OutOfSteps => "max_turn_requests",
                    Status::Cancelled => "cancelled",
                };
                self.reply(id, json!({"stopReason": stop}));
            }
            Err(e) => self.fail(id, -32603, &e),
        }
    }
}

/// Reads the lines of `input` on a thread and hands the parsed messages over; the main loop and a waiting permission request share the channel.
pub fn spawn_reader(input: Box<dyn BufRead + Send>) -> Receiver<Value> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in input.lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<Value>(line.trim()) {
                // a cancel must not wait behind a running prompt: set the flag at once as well as queueing it
                if v["method"] == "session/cancel" && v.get("id").is_none() {
                    agent_cli::CANCEL.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                if tx.send(v).is_err() {
                    break;
                }
            }
        }
    });
    rx
}

pub fn run(args: &[String]) -> Result<i32, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let mut agent_args = args.to_vec();
    agent_args.extend(["-p".to_owned(), "acp".to_owned()]);
    let parsed = agent_cli::parse_args(&agent_args)?;
    let mut http = agent_cli::HttpBackend::new(&parsed.url, parsed.api_key.clone())?;
    let rx = spawn_reader(Box::new(std::io::BufReader::new(std::io::stdin())));
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let mut acp = Acp::new(&mut http, &mut lock, &rx, Settings { allow: parsed.allow.clone(), max_steps: parsed.max_steps, model: parsed.model.clone() });
    while let Ok(message) = rx.recv() {
        // cancel notifications that arrive between prompts are harmless: the next prompt resets the flag
        acp.handle(&message);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::mpsc;

    struct Fake {
        replies: VecDeque<Value>,
        ran: Vec<String>,
        folder: String,
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
    }

    impl AcpBackend for Fake {
        fn set_folder(&mut self, folder: &str) -> Result<(), String> {
            self.folder = folder.to_owned();
            Ok(())
        }
        fn model(&mut self) -> Result<(String, Option<u64>), String> {
            Ok(("m".into(), Some(8192)))
        }
    }

    fn assistant_call(id: &str, name: &str, args: Value) -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": "working", "tool_calls": [{"id": id, "type": "function", "function": {"name": name, "arguments": args.to_string()}}]}}]})
    }

    fn text(t: &str) -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": t}}]})
    }

    fn lines(out: &[u8]) -> Vec<Value> {
        String::from_utf8_lossy(out).lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }

    fn settings(allow: &[&str]) -> Settings {
        Settings { allow: allow.iter().map(|a| (*a).to_owned()).collect(), max_steps: 5, model: None }
    }

    fn abs() -> String {
        std::env::temp_dir().display().to_string()
    }

    #[test]
    fn initialize_and_a_prompt_turn_stream_updates_and_end_with_a_stop_reason() {
        let mut fake = Fake { replies: VecDeque::from([assistant_call("c1", "read_file", json!({"path": "a.rs"})), text("All done.")]), ran: vec![], folder: String::new() };
        let mut out = Vec::new();
        let (tx, rx) = mpsc::channel();
        {
            let mut acp = Acp::new(&mut fake, &mut out, &rx, settings(&["read"]));
            acp.handle(&json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1}}));
            acp.handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": abs(), "mcpServers": []}}));
            acp.handle(&json!({"jsonrpc": "2.0", "id": 9, "method": "session/new", "params": {"cwd": "relative/path"}}));
            let sid = "fllm_".to_owned() + &std::process::id().to_string() + "_1";
            acp.handle(&json!({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "look at a.rs"}]}}));
            acp.handle(&json!({"jsonrpc": "2.0", "id": 3, "method": "session/prompt", "params": {"sessionId": "nope", "prompt": [{"type": "text", "text": "x"}]}}));
            acp.handle(&json!({"jsonrpc": "2.0", "id": 4, "method": "nothing/here"}));
        }
        drop(tx);
        let msgs = lines(&out);
        assert_eq!(msgs[0]["result"]["protocolVersion"], 1);
        assert_eq!(msgs[0]["result"]["agentCapabilities"]["promptCapabilities"]["image"], false);
        assert!(msgs[1]["result"]["sessionId"].as_str().unwrap().starts_with("fllm_"));
        assert_eq!(msgs[2]["error"]["code"], -32602, "a relative cwd is refused");
        let kinds: Vec<(&str, &str)> = msgs[3..].iter().filter(|m| m["method"] == "session/update").map(|m| (m["params"]["update"]["sessionUpdate"].as_str().unwrap(), m["params"]["update"]["status"].as_str().unwrap_or(""))).collect();
        assert_eq!(kinds, vec![("agent_message_chunk", ""), ("tool_call", "pending"), ("tool_call_update", "completed"), ("agent_message_chunk", "")]);
        let call = msgs.iter().find(|m| m["params"]["update"]["sessionUpdate"] == "tool_call").unwrap();
        assert_eq!((call["params"]["update"]["kind"].as_str(), call["params"]["update"]["toolCallId"].as_str()), (Some("read"), Some("c1")));
        let done = msgs.iter().find(|m| m["id"] == 2).unwrap();
        assert_eq!(done["result"]["stopReason"], "end_turn");
        assert_eq!(msgs.iter().find(|m| m["id"] == 3).unwrap()["error"]["code"], -32602);
        assert_eq!(msgs.iter().find(|m| m["id"] == 4).unwrap()["error"]["code"], -32601);
        assert_eq!(fake.ran, vec![r#"read_file {"path":"a.rs"}"#]);
        assert_eq!(fake.folder, abs());
    }

    #[test]
    fn an_action_the_flags_do_not_allow_is_put_to_the_editor() {
        let mut fake = Fake { replies: VecDeque::from([assistant_call("c1", "bash", json!({"command": "ls"})), text("listed")]), ran: vec![], folder: String::new() };
        let mut out = Vec::new();
        let (tx, rx) = mpsc::channel();
        // the editor will answer the first permission request (id 1000) with "allow once"
        tx.send(json!({"jsonrpc": "2.0", "id": 1000, "result": {"outcome": {"outcome": "selected", "optionId": "allow-once"}}})).unwrap();
        {
            let mut acp = Acp::new(&mut fake, &mut out, &rx, settings(&["read"]));
            acp.handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": abs()}}));
            let sid = "fllm_".to_owned() + &std::process::id().to_string() + "_1";
            acp.handle(&json!({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "list files"}]}}));
        }
        let msgs = lines(&out);
        let ask = msgs.iter().find(|m| m["method"] == "session/request_permission").expect("a permission request");
        assert_eq!(ask["params"]["toolCall"]["toolCallId"], "c1", "it names the call the editor already shows");
        assert_eq!(ask["params"]["toolCall"]["kind"], "execute");
        assert_eq!(ask["params"]["options"][0]["kind"], "allow_once");
        assert_eq!(fake.ran.len(), 1, "allowed once: the command ran");
    }

    #[test]
    fn a_rejected_or_cancelled_permission_keeps_the_action_from_running() {
        let mut fake = Fake { replies: VecDeque::from([assistant_call("c1", "write_file", json!({"path": "x", "content": "y"})), text("could not write")]), ran: vec![], folder: String::new() };
        let mut out = Vec::new();
        let (tx, rx) = mpsc::channel();
        tx.send(json!({"jsonrpc": "2.0", "id": 1000, "result": {"outcome": {"outcome": "selected", "optionId": "reject-once"}}})).unwrap();
        {
            let mut acp = Acp::new(&mut fake, &mut out, &rx, settings(&["read"]));
            acp.handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": abs()}}));
            let sid = "fllm_".to_owned() + &std::process::id().to_string() + "_1";
            acp.handle(&json!({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {"sessionId": sid, "prompt": [{"type": "text", "text": "write"}]}}));
        }
        assert!(fake.ran.is_empty());
        let msgs = lines(&out);
        assert!(msgs.iter().any(|m| m["params"]["update"]["status"] == "failed"), "the refusal is shown as a failed call");
    }

    #[test]
    fn prompt_blocks_become_one_text() {
        let blocks = vec![
            json!({"type": "text", "text": "fix this"}),
            json!({"type": "resource", "resource": {"uri": "file:///p/main.py", "text": "print(1)"}}),
            json!({"type": "resource_link", "uri": "file:///p/other.py", "name": "other.py"}),
            json!({"type": "image", "data": "AAAA", "mimeType": "image/png"}),
        ];
        let t = prompt_text(&blocks);
        assert!(t.starts_with("fix this") && t.contains("Contents of file:///p/main.py:") && t.contains("print(1)") && t.contains("file:///p/other.py") && t.contains("cannot see"));
        assert_eq!(prompt_text(&[]), "");
        assert_eq!(kind_of("read", "grep"), "search");
        assert_eq!(title_of("bash", &json!({"command": "ls\n-la"})), "bash ls -la");
    }
}
