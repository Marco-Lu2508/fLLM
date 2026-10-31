//! `fllm-rs mcp-server`: fLLM's agent tools (and the agent itself) as an MCP server on standard input and output, so another
//! agent or an editor (Claude Code, Codex CLI, Cursor, Zed ...) can use them. The transport is the MCP stdio one: one JSON-RPC
//! message per line. The tools run through the same server routes as the Studio, so the sandbox, the hooks, `.gitignore`
//! handling and the permission rules apply; reading is always allowed, every other kind of action needs its `--allow-*` flag.
use crate::agent_cli::{self, Backend};
use crate::agent_rules::{Decision, Rules};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, Write};

pub const HELP: &str = "usage: fllm-rs mcp-server [--folder DIR] [--url http://127.0.0.1:8000] [--api-key KEY]
                      [--allow-edit] [--allow-command] [--allow-network] [--allow-memory] [--yes] [--no-agent]
Speaks MCP (JSON-RPC, one message per line) on standard input and output. Tools: the agent tools of a running fllm-rs server
(read_file, grep, edit_file, bash ...; only those whose kind of action is allowed) and `fllm_agent`, which runs the whole coding
agent on a task with the local model and returns its report (switch it off with --no-agent).
Register it in an MCP client, for example: claude mcp add fllm -- fllm-rs mcp-server --folder /path/to/project --allow-edit
Reading and searching is always allowed; deny rules of permissions.json apply; an ask rule cannot be answered here and refuses.";

/// Runs the coding agent on a task with a step limit and returns its report.
pub type AgentRunner<'a> = &'a mut dyn FnMut(&str, usize) -> Result<String, String>;

pub struct McpServer<'a> {
    pub backend: &'a mut dyn Backend,
    pub allow: BTreeSet<String>,
    pub rules: Rules,
    pub agent: Option<AgentRunner<'a>>,
    tools: Vec<Value>,
    categories: HashMap<String, String>,
}

const PROTOCOLS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

fn rpc_result(id: &Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn rpc_error(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn text_result(text: &str, is_error: bool) -> Value {
    json!({"content": [{"type": "text", "text": text}], "isError": is_error})
}

impl<'a> McpServer<'a> {
    pub fn new(backend: &'a mut dyn Backend, allow: BTreeSet<String>, rules: Rules, agent: Option<AgentRunner<'a>>) -> Self {
        Self { backend, allow, rules, agent, tools: Vec::new(), categories: HashMap::new() }
    }

    fn load_tools(&mut self) -> Result<(), String> {
        let (schemas, categories) = agent_cli::prepare_tools(self.backend.tools()?);
        self.tools = schemas;
        self.categories = categories;
        Ok(())
    }

    fn visible(&self, name: &str) -> bool {
        self.categories.get(name).is_some_and(|c| self.allow.contains(c))
    }

    /// The MCP `tools/list` answer: the allowed agent tools plus `fllm_agent`.
    pub fn tool_list(&mut self) -> Result<Value, String> {
        self.load_tools()?;
        let mut list: Vec<Value> = self
            .tools
            .iter()
            .filter(|t| t["function"]["name"].as_str().is_some_and(|n| self.visible(n)))
            .map(|t| {
                let name = t["function"]["name"].as_str().unwrap_or_default();
                let category = self.categories.get(name).map_or("read", String::as_str);
                json!({
                    "name": name,
                    "description": t["function"]["description"],
                    "inputSchema": t["function"]["parameters"],
                    "annotations": {"readOnlyHint": category == "read", "destructiveHint": category == "edit" || category == "command", "openWorldHint": category == "network"},
                })
            })
            .collect();
        if self.agent.is_some() {
            list.push(json!({
                "name": "fllm_agent",
                "description": "Run fLLM's coding agent (a local model with file, search and shell tools) on a task in the project folder and return its final report. Use it to hand over a self-contained job.",
                "inputSchema": {"type": "object", "properties": {"task": {"type": "string", "description": "What to do, with everything the agent needs to know."}, "max_steps": {"type": "integer", "description": "Model calls the agent may use (default 30, at most 100)."}}, "required": ["task"]},
                "annotations": {"readOnlyHint": !(self.allow.contains("edit") || self.allow.contains("command")), "destructiveHint": self.allow.contains("edit") || self.allow.contains("command"), "openWorldHint": self.allow.contains("network")},
            }));
        }
        Ok(json!({"tools": list}))
    }

    pub fn call_tool(&mut self, name: &str, args: &Value) -> Value {
        if name == "fllm_agent" {
            let Some(task) = args["task"].as_str().filter(|t| !t.trim().is_empty()) else { return text_result("the argument 'task' is missing", true) };
            let steps = args["max_steps"].as_u64().unwrap_or(30).clamp(1, 100) as usize;
            return match self.agent.as_mut() {
                Some(run) => match run(task, steps) {
                    Ok(report) => text_result(&report, false),
                    Err(e) => text_result(&e, true),
                },
                None => text_result("the agent is switched off (--no-agent)", true),
            };
        }
        if self.categories.is_empty() {
            if let Err(e) = self.load_tools() {
                return text_result(&e, true);
            }
        }
        let Some(category) = self.categories.get(name).cloned() else { return text_result(&format!("there is no tool '{name}'"), true) };
        if !self.allow.contains(&category) {
            return text_result(&format!("'{name}' needs the permission '{category}', which this server was not started with (--allow-{category})"), true);
        }
        match self.rules.decide(name, &category, args) {
            Some(Decision::Deny) => return text_result(&format!("'{name}' is blocked by a deny rule of the permission settings"), true),
            Some(Decision::Ask) => return text_result(&format!("'{name}' matches an ask rule and an MCP server cannot ask; run it in the Studio or with `fllm-rs agent --ask`"), true),
            _ => {}
        }
        match self.backend.run_tool(name, args) {
            Ok(out) => text_result(&out, false),
            Err(e) => text_result(&e, true),
        }
    }

    /// One incoming message; `None` for notifications (no answer).
    pub fn handle(&mut self, request: &Value) -> Option<Value> {
        let id = request.get("id")?.clone();
        let method = request["method"].as_str().unwrap_or("");
        Some(match method {
            "initialize" => {
                let asked = request["params"]["protocolVersion"].as_str().unwrap_or("");
                let version = PROTOCOLS.iter().find(|p| **p == asked).unwrap_or(&PROTOCOLS[0]);
                rpc_result(&id, json!({"protocolVersion": version, "capabilities": {"tools": {"listChanged": false}}, "serverInfo": {"name": "fllm", "version": env!("CARGO_PKG_VERSION")}, "instructions": "fLLM agent tools. Reading is always allowed; edits, commands and network need the matching server flag."}))
            }
            "ping" => rpc_result(&id, json!({})),
            "tools/list" => match self.tool_list() {
                Ok(v) => rpc_result(&id, v),
                Err(e) => rpc_error(&id, -32603, &e),
            },
            "tools/call" => {
                let name = request["params"]["name"].as_str().unwrap_or("");
                let args = if request["params"]["arguments"].is_object() { request["params"]["arguments"].clone() } else { json!({}) };
                rpc_result(&id, self.call_tool(name, &args))
            }
            "resources/list" => rpc_result(&id, json!({"resources": []})),
            "prompts/list" => rpc_result(&id, json!({"prompts": []})),
            other => rpc_error(&id, -32601, &format!("method not found: {other}")),
        })
    }
}

/// Reads lines from `input` and writes the answers to `output` until the input ends.
pub fn serve(server: &mut McpServer, input: &mut dyn BufRead, output: &mut dyn Write) -> std::io::Result<()> {
    let mut line = String::new();
    while input.read_line(&mut line)? > 0 {
        let text = line.trim().to_owned();
        line.clear();
        if text.is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&text) {
            Ok(request) if request.is_array() => Some(json!(request.as_array().into_iter().flatten().filter_map(|r| server.handle(r)).collect::<Vec<_>>())),
            Ok(request) => server.handle(&request),
            Err(_) => Some(rpc_error(&Value::Null, -32700, "parse error")),
        };
        if let Some(reply) = reply {
            writeln!(output, "{reply}")?;
            output.flush()?;
        }
    }
    Ok(())
}

pub fn run(args: &[String]) -> Result<i32, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let mut allow = BTreeSet::from(["read".to_owned()]);
    let mut folder: Option<std::path::PathBuf> = None;
    let mut url = "http://127.0.0.1:8000".to_owned();
    let mut key = std::env::var("FLLM_API_KEY").ok().filter(|k| !k.is_empty());
    let mut no_agent = false;
    let mut i = 0;
    while i < args.len() {
        let next = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or_else(|| format!("missing value for {}", args[*i - 1]))
        };
        match args[i].as_str() {
            "--folder" | "--cwd" => folder = Some(next(&mut i)?.into()),
            "--url" => url = next(&mut i)?,
            "--api-key" => key = Some(next(&mut i)?),
            "--allow-edit" => {
                allow.insert("edit".into());
            }
            "--allow-command" => {
                allow.insert("command".into());
            }
            "--allow-network" => {
                allow.insert("network".into());
            }
            "--allow-memory" => {
                allow.insert("memory".into());
            }
            "--yes" | "-y" => allow.extend(["edit", "command", "network", "memory"].map(str::to_owned)),
            "--no-agent" => no_agent = true,
            other => return Err(format!("unknown mcp-server option: {other}")),
        }
        i += 1;
    }
    let folder = folder.map_or_else(std::env::current_dir, Ok).map_err(|e| e.to_string())?.canonicalize().map_err(|e| format!("the folder cannot be used: {e}"))?;
    let folder_text = folder.display().to_string();
    let rules = crate::agent_rules::load_hierarchy(&agent_cli::studio_folder(), &folder)?;
    let mut http = agent_cli::HttpBackend::new(&url, key.clone())?;
    http.set_folder(&folder_text)?;

    // the agent tool: a second connection, so the loop does not borrow the one the plain tools use
    let mut agent_http = agent_cli::HttpBackend::new(&url, key)?;
    let (allow_for_agent, rules_for_agent, folder_for_agent) = (allow.clone(), rules.clone(), folder_text.clone());
    let mut run_agent = move |task: &str, steps: usize| -> Result<String, String> {
        let (model, context) = agent_http.model_info()?;
        agent_http.set_folder(&folder_for_agent)?;
        let system = agent_cli::full_system(&mut agent_http, &folder_for_agent, &allow_for_agent, None, None);
        let cfg = agent_cli::LoopConfig { model: Some(model), allow: allow_for_agent.clone(), rules: rules_for_agent.clone(), max_steps: steps, max_tokens: agent_cli::effective_max_tokens(4096, context), temperature: 0.2 };
        let outcome = agent_cli::run_loop(&mut agent_http, &cfg, &system, vec![json!({"role": "user", "content": task})], &mut |line| eprintln!("{line}"));
        outcome.map(|o| o.answer)
    };
    let agent: Option<AgentRunner> = if no_agent { None } else { Some(&mut run_agent) };
    let mut server = McpServer::new(&mut http, allow, rules, agent);
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    serve(&mut server, &mut stdin.lock(), &mut stdout.lock()).map_err(|e| e.to_string())?;
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        ran: Vec<String>,
    }

    impl Backend for Fake {
        fn tools(&mut self) -> Result<Vec<Value>, String> {
            let t = |n: &str, p: &str| json!({"type": "function", "function": {"name": n, "description": "d", "parameters": {"type": "object", "properties": {}}}, "permission": p});
            Ok(vec![t("read_file", "read"), t("write_file", "edit"), t("bash", "command"), t("computer", "computer")])
        }
        fn complete(&mut self, _: Value) -> Result<Value, String> {
            Err("no model".into())
        }
        fn run_tool(&mut self, name: &str, args: &Value) -> Result<String, String> {
            self.ran.push(format!("{name} {args}"));
            if name == "read_file" && args["path"] == "missing" { Err("no such file".into()) } else { Ok(format!("ran {name}")) }
        }
    }

    fn call(server: &mut McpServer, method: &str, params: Value) -> Value {
        server.handle(&json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})).unwrap()
    }

    #[test]
    fn it_negotiates_lists_only_allowed_tools_and_runs_them() {
        let mut fake = Fake { ran: vec![] };
        let mut agent = |task: &str, steps: usize| -> Result<String, String> { Ok(format!("report for {task} in {steps}")) };
        let mut server = McpServer::new(&mut fake, BTreeSet::from(["read".to_owned()]), Rules::default(), Some(&mut agent));
        let init = call(&mut server, "initialize", json!({"protocolVersion": "2025-03-26"}));
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(call(&mut server, "initialize", json!({"protocolVersion": "1999"}))["result"]["protocolVersion"], PROTOCOLS[0]);
        assert!(server.handle(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).is_none());
        let list = call(&mut server, "tools/list", json!({}));
        let names: Vec<&str> = list["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["read_file", "fllm_agent"], "no edit, command or computer tool without flags");
        assert_eq!(list["result"]["tools"][0]["annotations"]["readOnlyHint"], true);
        let ok = call(&mut server, "tools/call", json!({"name": "read_file", "arguments": {"path": "a"}}));
        assert_eq!(ok["result"]["isError"], false);
        assert_eq!(ok["result"]["content"][0]["text"], "ran read_file");
        assert_eq!(call(&mut server, "tools/call", json!({"name": "read_file", "arguments": {"path": "missing"}}))["result"]["isError"], true);
        let refused = call(&mut server, "tools/call", json!({"name": "bash", "arguments": {"command": "ls"}}));
        assert_eq!(refused["result"]["isError"], true);
        assert!(refused["result"]["content"][0]["text"].as_str().unwrap().contains("--allow-command"));
        let agent_call = call(&mut server, "tools/call", json!({"name": "fllm_agent", "arguments": {"task": "fix it", "max_steps": 500}}));
        assert_eq!(agent_call["result"]["content"][0]["text"], "report for fix it in 100");
        assert_eq!(call(&mut server, "tools/call", json!({"name": "fllm_agent", "arguments": {}}))["result"]["isError"], true);
        assert_eq!(call(&mut server, "nope", json!({}))["error"]["code"], -32601);
        drop(server);
        assert_eq!(fake.ran.len(), 2, "the refused call never reached the tools");
    }

    #[test]
    fn rules_and_flags_both_apply() {
        let mut fake = Fake { ran: vec![] };
        let rules = Rules::from_json(&json!({"deny": ["bash: rm *"], "ask": ["edit: src/**"], "allow": ["bash: ls*"]}));
        let mut server = McpServer::new(&mut fake, BTreeSet::from(["read".to_owned(), "edit".to_owned(), "command".to_owned()]), rules, None);
        let names: Vec<String> = call(&mut server, "tools/list", json!({}))["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap().to_owned()).collect();
        assert_eq!(names, vec!["read_file", "write_file", "bash"]);
        assert!(call(&mut server, "tools/call", json!({"name": "bash", "arguments": {"command": "rm -rf x"}}))["result"]["content"][0]["text"].as_str().unwrap().contains("deny rule"));
        assert!(call(&mut server, "tools/call", json!({"name": "write_file", "arguments": {"path": "src/a.rs", "content": "x"}}))["result"]["content"][0]["text"].as_str().unwrap().contains("ask rule"));
        assert_eq!(call(&mut server, "tools/call", json!({"name": "bash", "arguments": {"command": "ls"}}))["result"]["isError"], false);
        assert_eq!(call(&mut server, "tools/call", json!({"name": "write_file", "arguments": {"path": "docs/a.md", "content": "x"}}))["result"]["isError"], false);
    }

    #[test]
    fn the_line_loop_answers_requests_and_survives_garbage() {
        let mut fake = Fake { ran: vec![] };
        let mut server = McpServer::new(&mut fake, BTreeSet::from(["read".to_owned()]), Rules::default(), None);
        let input = "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\nnot json\n\n{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n[{\"jsonrpc\":\"2.0\",\"id\":8,\"method\":\"ping\"}]\n";
        let mut out = Vec::new();
        serve(&mut server, &mut input.as_bytes(), &mut out).unwrap();
        let lines: Vec<Value> = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["id"], 7);
        assert_eq!(lines[1]["error"]["code"], -32700);
        assert_eq!(lines[2][0]["id"], 8);
    }
}
