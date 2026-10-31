//! Headless coding agent: `fllm-rs agent -p "task"` runs the agent loop of the Studio without a browser.
//!
//! The model and the tools stay where they are: the loop talks to a running `fllm-rs` server over HTTP, takes the
//! tool catalog from `GET /api/studio/agent/tools`, lets the model call tools through `/v1/chat/completions` and runs
//! them with `POST /api/studio/agent/tool`, so the sandbox, the hooks, `.gitignore` handling and the edit log are
//! the same as in the Studio. Nobody can answer questions in a script, so permissions are decided up front by flags:
//! reading is always allowed, every other kind of action (`edit`, `command`, `network`, `memory`) only when the
//! matching `--allow-*` flag (or `--yes`) is given. A refused call is not run; the model reads why and can adapt.
//!
//! The loop itself is written against the [`Backend`] trait, so it is tested with a scripted model and fake tools.
use std::collections::{BTreeSet, HashMap};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Duration;

use serde_json::{json, Value};

use crate::agent_extras::{self, OutputFormat};
use crate::agent_rules::{self, Decision, Rules};

pub const DEFAULT_MAX_STEPS: usize = 30;
const MAX_TOOL_OUTPUT: usize = 30_000;
/// The permission categories the headless agent knows. A tool in any other category (computer use) is never offered.
const CATEGORIES: [&str; 5] = ["read", "edit", "command", "network", "memory"];
/// A call that is repeated this many times in a row with the same arguments is not run again.
const REPEAT_LIMIT: usize = 3;
/// A continued session keeps at most this many messages (cut at a user message) so it cannot outgrow the context for ever.
const MAX_SESSION_MESSAGES: usize = 120;

pub const HELP: &str = "usage: fllm-rs agent -p \"TASK\" [--folder DIR] [--url http://127.0.0.1:8000] [--model ID] [--api-key KEY]
                  [--allow-edit] [--allow-command] [--allow-network] [--allow-memory] [--yes]
                  [--max-steps N] [--max-tokens N] [--temperature T] [--system TEXT]
                  [--session NAME] [--continue] [--resume NAME] [--profile NAME] [--ask] [--trajectory FILE] [--json] [--quiet]
                  [--output-format text|json|stream-json] [--plan] [--allow-rule RULE] [--deny-rule RULE] [--ask-rule RULE]
                  [--review] [--review-rounds N] [--verify COMMAND] [--propose-skills] [--auto-skills] [--image FILE]...
Runs the coding agent of the Studio without a browser against a running fllm-rs server on this machine.
TASK: the text after -p; `-p -` reads it from standard input. Reading and searching files is always allowed.
Everything else is refused unless you allow it: --allow-edit (write_file, edit_file, patch ...), --allow-command (bash),
--allow-network (web_fetch, web_search, browser), --allow-memory (memory tool); --yes allows all four. A refused call is
not run; the model is told which flag would allow it.
--folder: the folder the agent works in (default: the current folder). --session NAME keeps the conversation in the
Studio folder (agent-sessions/NAME.json) and continues it next time. --profile NAME keeps sessions and a persona
(SOUL.md) under profiles/NAME of the Studio folder; without it SOUL.md of the Studio folder is used if it exists.
--ask asks in the terminal before a call that the flags do not allow (y/n/a); without a terminal it stays refused. --trajectory FILE appends one JSON line per run
(time, folder, model, status, tool calls and the whole conversation) for later analysis or training data. --json prints one JSON object (answer, status,
steps, tool calls) instead of the plain answer. Progress goes to standard error unless --quiet.
--output-format json is --json; stream-json prints one event per line while the run goes on (system init, assistant, user tool results, result).
--continue continues the newest session, --resume NAME a named one (every run is saved as run-<time> unless --session names it).
--plan: read-only planning; the answer is a plan, and on a terminal you are asked whether to run it with the allowed actions.
Rules (allow / deny / ask, `bash: npm test*`, `edit: src/**`, or `Bash(npm test:*)`) come from permissions.json in the Studio folder, then
.fllm/permissions.json, then .fllm/permissions.local.json of the folder, then these flags. A deny rule always wins.
--review: when the run changed files, check them (review tool, diagnostics, --verify COMMAND when commands are allowed) and let the agent
fix what is found before the run counts as done (at most --review-rounds, default 2). --image FILE (repeatable) attaches a png/jpg/gif/webp picture to the task; the loaded model must be able to see.
--propose-skills: after a long run the model may
write a skill; it waits in skills-proposed/ (fllm-rs skills approve NAME), --auto-skills saves it at once.
The answer is printed on standard output. Exit codes: 0 finished, 1 error, 3 stopped at --max-steps (a report is still printed).
The agent tools only accept requests from the machine the server runs on, so --url must name this machine.";

#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub url: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub folder: Option<PathBuf>,
    pub prompt: String,
    pub allow: BTreeSet<String>,
    pub max_steps: usize,
    pub max_tokens: usize,
    pub temperature: f64,
    pub system: Option<String>,
    pub session: Option<String>,
    pub profile: Option<String>,
    pub ask: bool,
    pub trajectory: Option<PathBuf>,
    pub json: bool,
    pub quiet: bool,
    pub format: OutputFormat,
    pub continue_last: bool,
    pub plan: bool,
    pub rule_flags: Vec<(&'static str, String)>,
    pub review: bool,
    pub review_rounds: usize,
    pub verify: Option<String>,
    pub propose_skills: bool,
    pub auto_skills: bool,
    pub images: Vec<PathBuf>,
}

fn value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a String, String> {
    *i += 1;
    args.get(*i).ok_or_else(|| format!("missing value for {flag}"))
}

/// A session name is a slug, because it becomes a file name.
pub fn valid_session_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && name.len() <= 64
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

pub fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        url: "http://127.0.0.1:8000".into(),
        model: None,
        api_key: std::env::var("FLLM_API_KEY").ok().filter(|k| !k.is_empty()),
        folder: None,
        prompt: String::new(),
        allow: BTreeSet::from(["read".to_owned()]),
        max_steps: DEFAULT_MAX_STEPS,
        max_tokens: 4096,
        temperature: 0.2,
        system: None,
        session: None,
        profile: None,
        ask: false,
        trajectory: None,
        json: false,
        quiet: false,
        format: OutputFormat::Text,
        continue_last: false,
        plan: false,
        rule_flags: Vec::new(),
        review: false,
        review_rounds: 2,
        verify: None,
        propose_skills: false,
        auto_skills: false,
        images: Vec::new(),
    };
    let mut loose: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "-p" | "--prompt" => o.prompt = value(args, &mut i, flag)?.clone(),
            "--url" => o.url = value(args, &mut i, flag)?.clone(),
            "--model" => o.model = Some(value(args, &mut i, flag)?.clone()),
            "--api-key" => o.api_key = Some(value(args, &mut i, flag)?.clone()),
            "--folder" | "--cwd" => o.folder = Some(PathBuf::from(value(args, &mut i, flag)?)),
            "--system" => o.system = Some(value(args, &mut i, flag)?.clone()),
            "--session" => {
                let name = value(args, &mut i, flag)?.clone();
                if !valid_session_name(&name) {
                    return Err("--session needs a name of lowercase letters, digits, - and _ (at most 64)".into());
                }
                o.session = Some(name);
            }
            "--profile" => {
                let name = value(args, &mut i, flag)?.clone();
                if !valid_session_name(&name) {
                    return Err("--profile needs a name of lowercase letters, digits, - and _ (at most 64)".into());
                }
                o.profile = Some(name);
            }
            "--ask" => o.ask = true,
            "--trajectory" => o.trajectory = Some(PathBuf::from(value(args, &mut i, flag)?)),
            "--allow-edit" => {
                o.allow.insert("edit".into());
            }
            "--allow-command" => {
                o.allow.insert("command".into());
            }
            "--allow-network" => {
                o.allow.insert("network".into());
            }
            "--allow-memory" => {
                o.allow.insert("memory".into());
            }
            "--yes" | "-y" => o.allow.extend(CATEGORIES.iter().map(|c| (*c).to_owned())),
            "--max-steps" => {
                o.max_steps = value(args, &mut i, flag)?.parse().ok().filter(|n| (1..=500).contains(n)).ok_or("--max-steps is a number from 1 to 500")?;
            }
            "--max-tokens" => {
                o.max_tokens = value(args, &mut i, flag)?.parse().ok().filter(|n| *n > 0).ok_or("--max-tokens is a positive number")?;
            }
            "--temperature" => {
                o.temperature = value(args, &mut i, flag)?.parse().ok().filter(|t: &f64| *t >= 0.0).ok_or("--temperature is a number >= 0")?;
            }
            "--json" => {
                o.json = true;
                o.format = OutputFormat::Json;
            }
            "--output-format" => {
                o.format = OutputFormat::parse(value(args, &mut i, flag)?).ok_or("--output-format is text, json or stream-json")?;
                o.json = o.format == OutputFormat::Json;
            }
            "--continue" | "-c" => o.continue_last = true,
            "--resume" | "-r" => {
                let name = value(args, &mut i, flag)?.clone();
                if !valid_session_name(&name) {
                    return Err("--resume needs the name of a session (lowercase letters, digits, - and _)".into());
                }
                o.session = Some(name);
            }
            "--plan" => o.plan = true,
            "--allow-rule" => o.rule_flags.push(("allow", value(args, &mut i, flag)?.clone())),
            "--deny-rule" => o.rule_flags.push(("deny", value(args, &mut i, flag)?.clone())),
            "--ask-rule" => o.rule_flags.push(("ask", value(args, &mut i, flag)?.clone())),
            "--review" => o.review = true,
            "--review-rounds" => {
                o.review = true;
                o.review_rounds = value(args, &mut i, flag)?.parse().ok().filter(|n| (1..=5).contains(n)).ok_or("--review-rounds is a number from 1 to 5")?;
            }
            "--verify" => {
                o.review = true;
                o.verify = Some(value(args, &mut i, flag)?.clone());
            }
            "--image" | "-i" => o.images.push(PathBuf::from(value(args, &mut i, flag)?)),
            "--propose-skills" => o.propose_skills = true,
            "--auto-skills" => {
                o.propose_skills = true;
                o.auto_skills = true;
            }
            "--quiet" | "-q" => o.quiet = true,
            "--help" | "-h" => return Err(HELP.to_owned()),
            other if other.starts_with('-') && other != "-" => return Err(format!("unknown agent option: {other}")),
            other => loose.push(other.to_owned()),
        }
        i += 1;
    }
    if o.prompt.is_empty() && !loose.is_empty() {
        o.prompt = loose.join(" ");
    }
    if o.prompt.trim().is_empty() {
        return Err("the task is missing: fllm-rs agent -p \"what to do\"".into());
    }
    Ok(o)
}

// ---------------------------------------------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------------------------------------------

/// What the loop needs from the outside: the tool catalog, one model call and one tool run.
pub trait Backend {
    /// The tool catalog as the server serves it: OpenAI function schemas, each with a `permission` category.
    fn tools(&mut self) -> Result<Vec<Value>, String>;
    /// One chat-completions request (non-streaming); returns the response body.
    fn complete(&mut self, body: Value) -> Result<Value, String>;
    /// Runs one tool; `Ok` is the output, `Err` the tool's own error text.
    fn run_tool(&mut self, name: &str, args: &Value) -> Result<String, String>;
    /// The project instruction files and the memories of the folder (`{files: [{file, text}], memory: {workspace, global}}`).
    fn instructions(&mut self) -> Result<Value, String> {
        Ok(json!({}))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolRecord {
    pub name: String,
    pub category: String,
    pub status: &'static str,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Done,
    OutOfSteps,
    /// Ended by [`CANCEL`] (the editor pressed stop in the ACP adapter).
    Cancelled,
}

/// Set by a front end (the ACP adapter) to end the running loop before its next model call; the caller resets it.
pub static CANCEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[derive(Debug)]
pub struct Outcome {
    pub answer: String,
    pub status: Status,
    pub steps: usize,
    pub tool_calls: Vec<ToolRecord>,
    /// The whole conversation without the system message, to continue it later.
    pub messages: Vec<Value>,
    /// The files that edit tools changed successfully (for the self-review).
    pub edited: Vec<String>,
}

pub struct LoopConfig {
    pub model: Option<String>,
    pub allow: BTreeSet<String>,
    /// allow / deny / ask rules; a deny rule beats the flags, an allow rule beats a missing flag, an ask rule asks.
    pub rules: Rules,
    pub max_steps: usize,
    pub max_tokens: usize,
    pub temperature: f64,
}

/// Splits the catalog into the schemas the model sees (without the `permission` key) and the category of each tool.
/// Tools outside the known categories (computer use) are dropped: a script has nobody to stop them.
pub fn prepare_tools(catalog: Vec<Value>) -> (Vec<Value>, HashMap<String, String>) {
    let mut schemas = Vec::new();
    let mut categories = HashMap::new();
    for mut tool in catalog {
        let name = tool["function"]["name"].as_str().unwrap_or_default().to_owned();
        let category = tool["permission"].as_str().unwrap_or_default().to_owned();
        if name.is_empty() || !CATEGORIES.contains(&category.as_str()) {
            continue;
        }
        if let Some(object) = tool.as_object_mut() {
            object.remove("permission");
        }
        categories.insert(name, category);
        schemas.push(tool);
    }
    (schemas, categories)
}

/// The arguments of a tool call: a JSON string (OpenAI) or an object (some servers). `Err` names the problem so the
/// model can fix its call instead of the tool running with empty arguments.
pub fn parse_call_args(arguments: &Value) -> Result<Value, String> {
    match arguments {
        Value::String(s) if s.trim().is_empty() => Ok(json!({})),
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(v) if v.is_object() => Ok(v),
            Ok(_) => Err("the arguments must be a JSON object".into()),
            Err(e) => Err(format!("the arguments are not valid JSON ({e})")),
        },
        Value::Object(_) => Ok(arguments.clone()),
        Value::Null => Ok(json!({})),
        _ => Err("the arguments must be a JSON object".into()),
    }
}

fn message_text(message: &Value) -> String {
    match &message["content"] {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

pub(crate) fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let head: String = text.chars().take(max * 2 / 3).collect();
    let tail: String = text.chars().rev().take(max / 3).collect::<Vec<_>>().into_iter().rev().collect();
    format!("{head}\n[... {} characters cut ...]\n{tail}", text.chars().count() - head.chars().count() - tail.chars().count())
}

fn flag_for(category: &str) -> &'static str {
    match category {
        "edit" => "--allow-edit",
        "command" => "--allow-command",
        "network" => "--allow-network",
        "memory" => "--allow-memory",
        _ => "--yes",
    }
}

pub fn system_prompt(folder: &str, allow: &BTreeSet<String>, extra: Option<&str>) -> String {
    let now = chrono::Local::now().format("%A, %Y-%m-%d %H:%M");
    let allowed: Vec<&str> = CATEGORIES.iter().copied().filter(|c| allow.contains(*c)).collect();
    let mut text = format!(
        "You are a coding agent that runs without a person watching: nobody can answer questions, so never ask for confirmation. \
         Work in the folder {folder} (system: {}, local time: {now}). Read a file before you change it, make the smallest change that solves the task and \
         check your work when a check is available. Allowed kinds of action: {}. A tool call outside them is refused with a message; do not retry it, \
         choose another way or say in the report what could not be done. Finish with a short report of what you found or changed.",
        std::env::consts::OS,
        allowed.join(", ")
    );
    if let Some(extra) = extra.filter(|e| !e.trim().is_empty()) {
        text.push_str("\n\n");
        text.push_str(extra.trim());
    }
    text
}

const MAX_INSTRUCTIONS_CHARS: usize = 12_000;
const MAX_SOUL_CHARS: usize = 4_000;

/// The project instruction files (`AGENTS.md`, `FLLM.md`, `CLAUDE.md`) and the memories as a block for the system prompt.
/// Empty when there is nothing; capped so that a huge file cannot fill the context.
pub fn instructions_block(answer: &Value) -> String {
    let mut text = String::new();
    for file in answer["files"].as_array().into_iter().flatten() {
        let name = file["file"].as_str().unwrap_or("instructions");
        let body = file["text"].as_str().unwrap_or_default().trim();
        if !body.is_empty() {
            text.push_str(&format!("\n\n# Project instructions ({name})\n{body}"));
        }
    }
    for (key, title) in [("workspace", "Memory of this folder"), ("global", "Memory")] {
        let entries: Vec<String> = answer["memory"][key]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|e| e.as_str().map(|t| t.trim().to_owned()))
            .filter(|t| !t.is_empty())
            .collect();
        if !entries.is_empty() {
            text.push_str(&format!("\n\n# {title}\n{}", entries.iter().map(|e| format!("- {e}")).collect::<Vec<_>>().join("\n")));
        }
    }
    clip(&text, MAX_INSTRUCTIONS_CHARS)
}

/// The Studio folder, or `profiles/NAME` inside it: where the sessions and the persona (`SOUL.md`) of a profile live.
pub fn profile_base(profile: Option<&str>) -> Result<PathBuf, String> {
    let base = studio_folder();
    match profile {
        None => Ok(base),
        Some(name) if valid_session_name(name) => Ok(base.join("profiles").join(name)),
        Some(_) => Err("invalid profile name".into()),
    }
}

/// `SOUL.md` of a profile folder: the persona that is added to the system prompt. `None` when missing or empty.
pub fn load_soul(base: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(base.join("SOUL.md")).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| clip(text, MAX_SOUL_CHARS))
}

/// The whole system prompt of a run: the rules, the persona, the project instructions and the memory.
pub fn full_system(backend: &mut dyn Backend, folder: &str, allow: &BTreeSet<String>, extra: Option<&str>, soul: Option<&str>) -> String {
    let mut text = system_prompt(folder, allow, extra);
    if let Some(soul) = soul.filter(|s| !s.trim().is_empty()) {
        text.push_str("\n\n# Persona\n");
        text.push_str(soul.trim());
    }
    if let Ok(answer) = backend.instructions() {
        text.push_str(&instructions_block(&answer));
    }
    text
}

#[derive(Debug, PartialEq)]
pub enum Answer {
    Yes,
    No,
    Always,
}

pub fn parse_answer(line: &str) -> Answer {
    match line.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" | "j" | "ja" => Answer::Yes,
        "a" | "always" | "immer" => Answer::Always,
        _ => Answer::No,
    }
}

/// One line that shows what a call would do, for the question.
pub fn describe_call(name: &str, category: &str, args: &Value) -> String {
    let shown = ["command", "path", "url", "query", "pattern", "text"].iter().find_map(|k| args[*k].as_str()).map_or_else(|| args.to_string(), str::to_owned);
    format!("Allow {name} ({category}): {}", clip(&shown.replace('\n', " "), 160))
}

/// Asks in the terminal. Without a terminal on standard input nothing can be asked, so the call stays refused.
#[derive(Default)]
pub struct TerminalApprover {
    always: BTreeSet<String>,
    warned: bool,
}

impl TerminalApprover {
    pub fn ask(&mut self, name: &str, category: &str, args: &Value) -> bool {
        use std::io::IsTerminal;
        if self.always.contains(category) {
            return true;
        }
        if !io::stdin().is_terminal() {
            if !self.warned {
                eprintln!("[ask] standard input is not a terminal: calls that need permission stay refused");
                self.warned = true;
            }
            return false;
        }
        eprint!("{} [y]es / [n]o / [a]lways for {category}: ", describe_call(name, category, args));
        io::stderr().flush().ok();
        let mut line = String::new();
        if io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
            return false;
        }
        match parse_answer(&line) {
            Answer::Yes => true,
            Answer::Always => {
                self.always.insert(category.to_owned());
                true
            }
            Answer::No => false,
        }
    }
}

/// Runs the agent until the model answers without a tool call or `max_steps` model calls are used.
/// `progress` receives one line per tool call and result.
pub fn run_loop(
    backend: &mut dyn Backend,
    cfg: &LoopConfig,
    system: &str,
    history: Vec<Value>,
    progress: &mut dyn FnMut(&str),
) -> Result<Outcome, String> {
    run_loop_with(backend, cfg, system, history, progress, &mut |_, _, _| false)
}

/// Like [`run_loop`], but a call that the permissions do not allow is first shown to `approve(tool, category, args)`; `true` runs
/// it once (status `approved`), `false` refuses it as before.
pub fn run_loop_with(
    backend: &mut dyn Backend,
    cfg: &LoopConfig,
    system: &str,
    history: Vec<Value>,
    progress: &mut dyn FnMut(&str),
    approve: &mut dyn FnMut(&str, &str, &Value) -> bool,
) -> Result<Outcome, String> {
    run_loop_full(backend, cfg, system, history, progress, approve, &mut |_| {})
}

/// Like [`run_loop_with`], and `observe` sees every message the loop appends (assistant messages and tool results) as it happens.
pub fn run_loop_full(
    backend: &mut dyn Backend,
    cfg: &LoopConfig,
    system: &str,
    history: Vec<Value>,
    progress: &mut dyn FnMut(&str),
    approve: &mut dyn FnMut(&str, &str, &Value) -> bool,
    observe: &mut dyn FnMut(&Value),
) -> Result<Outcome, String> {
    let (schemas, categories) = prepare_tools(backend.tools()?);
    let mut messages = history;
    let mut records: Vec<ToolRecord> = Vec::new();
    let mut edited: Vec<String> = Vec::new();
    let mut last_call = String::new();
    let mut repeats = 0usize;
    let mut next_id = 0usize;

    // like the Studio: the server constrains the reply to valid tool calls (`tool_call_grammar`); a server that does not know the field is asked
    // again without it. Without the grammar a model that writes a number as text ("1") makes the server refuse the whole reply (HTTP 422).
    let mut grammar = true;
    let mut bad_calls = 0usize;
    let request = |messages: &[Value], with_tools: bool, grammar: bool| {
        let mut all = vec![json!({"role": "system", "content": system})];
        all.extend_from_slice(messages);
        let mut body = json!({"messages": all, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature, "stream": false});
        if with_tools && !schemas.is_empty() {
            body["tools"] = json!(schemas);
            if grammar {
                body["tool_call_grammar"] = json!(true);
            }
        }
        if let Some(model) = &cfg.model {
            body["model"] = json!(model);
        }
        body
    };

    for step in 1..=cfg.max_steps {
        if CANCEL.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(Outcome { answer: "Stopped.".into(), status: Status::Cancelled, steps: step - 1, tool_calls: records, messages, edited });
        }
        let reply = match backend.complete(request(&messages, true, grammar)) {
            Ok(reply) => reply,
            Err(e) if grammar && e.contains("HTTP 400") => {
                grammar = false;
                backend.complete(request(&messages, true, false))?
            }
            // the server refused tool arguments that do not fit the schema: tell the model what was wrong and let it try again (twice at most)
            Err(e) if e.contains("tool arguments do not match") && bad_calls < 2 => {
                bad_calls += 1;
                messages.push(json!({"role": "user", "content": format!("Your last tool call was refused by the server: {e}. Call the tool again with arguments that match its schema (numbers as numbers).")}));
                continue;
            }
            Err(e) => return Err(e),
        };
        let message = reply["choices"][0]["message"].clone();
        if message.is_null() {
            return Err(format!("the server answer has no message: {}", clip(&reply.to_string(), 300)));
        }
        let calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
        let text = message_text(&message);
        if calls.is_empty() {
            if text.trim().is_empty() {
                return Err("the model gave an empty answer".into());
            }
            let final_message = json!({"role": "assistant", "content": text});
            observe(&final_message);
            messages.push(final_message);
            return Ok(Outcome { answer: text, status: Status::Done, steps: step, tool_calls: records, messages, edited });
        }
        // every call needs an id so that the tool results can be matched to it
        let calls: Vec<Value> = calls
            .into_iter()
            .map(|mut call| {
                if call["id"].as_str().is_none_or(str::is_empty) {
                    next_id += 1;
                    call["id"] = json!(format!("call_{next_id}"));
                }
                call
            })
            .collect();
        let assistant = json!({"role": "assistant", "content": text, "tool_calls": calls});
        observe(&assistant);
        messages.push(assistant);
        for call in &calls {
            let name = call["function"]["name"].as_str().unwrap_or_default().to_owned();
            let id = call["id"].as_str().unwrap_or_default().to_owned();
            let category = categories.get(&name).cloned().unwrap_or_default();
            let parsed = parse_call_args(&call["function"]["arguments"]);
            let key = format!("{name} {}", call["function"]["arguments"]);
            repeats = if key == last_call { repeats + 1 } else { 1 };
            last_call = key;

            // the rules first: deny beats everything, ask puts the call to the question, allow lifts a missing flag
            let mut approved = false;
            let mut blocked: Option<String> = None;
            let known = categories.contains_key(&name);
            let decision = if known { parsed.as_ref().ok().and_then(|a| cfg.rules.decide(&name, &category, a)) } else { None };
            match decision {
                Some(Decision::Deny) => {
                    blocked = Some(format!("Error: '{name}' is blocked by a deny rule of the permission settings. Do not retry; use another way or report it."));
                }
                Some(Decision::Ask) => {
                    if let Ok(shown) = &parsed {
                        approved = approve(&name, &category, shown);
                    }
                    if !approved {
                        blocked = Some(format!("Error: '{name}' matches an ask rule and nobody approved it. Do not retry; use another way or report it."));
                    }
                }
                Some(Decision::Allow) => approved = !cfg.allow.contains(&category),
                None => {
                    if known && !cfg.allow.contains(&category) {
                        if let Ok(shown) = &parsed {
                            approved = approve(&name, &category, shown);
                        }
                        if !approved {
                            blocked = Some(format!("Error: '{name}' needs the permission '{category}', which is not granted in this headless run (it would need {}). Do not retry; use another way or report it.", flag_for(&category)));
                        }
                    }
                }
            }
            let (status, output) = if !known {
                ("unknown", format!("Error: there is no tool '{name}'. Tools: {}", { let mut n: Vec<_> = categories.keys().cloned().collect(); n.sort(); n.join(", ") }))
            } else if let Some(reason) = blocked {
                ("denied", reason)
            } else if repeats >= REPEAT_LIMIT {
                ("repeated", "Error: this exact call was already made several times in a row; it is not run again. Try something different.".to_owned())
            } else {
                match parsed {
                    Err(e) => ("bad_args", format!("Error: {e}")),
                    Ok(args) => match backend.run_tool(&name, &args) {
                        Ok(out) => {
                            if category == "edit" {
                                for path in agent_rules::subjects_of(&name, &category, &args).1 {
                                    if !edited.contains(&path) {
                                        edited.push(path);
                                    }
                                }
                            }
                            (if approved { "approved" } else { "ok" }, out)
                        }
                        Err(e) => ("error", format!("Error: {e}")),
                    },
                }
            };
            progress(&format!("[{step}] {name} -> {status}"));
            records.push(ToolRecord { name: name.clone(), category, status, detail: clip(&output, 200) });
            let result = json!({"role": "tool", "tool_call_id": id, "content": clip(&output, MAX_TOOL_OUTPUT)});
            observe(&result);
            messages.push(result);
        }
    }
    // out of steps: ask for the report with what was found, without tools
    messages.push(json!({"role": "user", "content": "Stop using tools now and write the final report with what you found or changed so far."}));
    // a failing request or an empty reply must not throw away what the run did: the fallback lists the real tool calls
    let text = backend
        .complete(request(&messages, false, false))
        .map(|reply| message_text(&reply["choices"][0]["message"]))
        .ok()
        .filter(|t| !t.trim().is_empty())
        .unwrap_or_else(|| fallback_report(cfg.max_steps, &records));
    let final_message = json!({"role": "assistant", "content": text});
    observe(&final_message);
    messages.push(final_message);
    Ok(Outcome { answer: text, status: Status::OutOfSteps, steps: cfg.max_steps, tool_calls: records, messages, edited })
}

/// The report of a run that used all its steps and got no written report from the model. It only states facts the
/// loop knows (the tool calls and how they ended); it never claims the task was solved.
pub fn fallback_report(max_steps: usize, records: &[ToolRecord]) -> String {
    let mut text = format!("The agent stopped after {max_steps} steps without writing a report. Tool calls so far ({}):\n", records.len());
    for r in records.iter().take(60) {
        text.push_str(&format!("- {} [{}]: {}\n", r.name, r.status, r.detail.lines().next().unwrap_or_default().chars().take(100).collect::<String>()));
    }
    if records.len() > 60 {
        text.push_str(&format!("- ... and {} more\n", records.len() - 60));
    }
    text
}

// ---------------------------------------------------------------------------------------------------------------
// Sessions kept in the Studio folder
// ---------------------------------------------------------------------------------------------------------------

pub(crate) fn session_file(base: &std::path::Path, name: &str) -> PathBuf {
    base.join("agent-sessions").join(format!("{name}.json"))
}

pub(crate) fn studio_folder() -> PathBuf {
    crate::hub::studio_data_file("")
}

pub(crate) fn load_session_in(base: &std::path::Path, name: &str) -> Result<Vec<Value>, String> {
    if !valid_session_name(name) {
        return Err("invalid session name".into());
    }
    match std::fs::read_to_string(session_file(base, name)) {
        Ok(text) => serde_json::from_str::<Value>(&text)
            .map_err(|e| format!("session '{name}' is damaged: {e}"))?["messages"]
            .as_array()
            .cloned()
            .ok_or_else(|| format!("session '{name}' has no messages")),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("cannot read session '{name}': {e}")),
    }
}

pub(crate) fn save_session_in(base: &std::path::Path, name: &str, messages: &[Value]) -> Result<(), String> {
    save_session_with(base, name, messages, json!({}))
}

/// Like [`save_session_in`], plus facts about the run (`folder`, `model`, `updated`, `edited` ...) that `fllm-rs insights` reads.
/// Loading ignores them, so older builds still read the file.
pub(crate) fn save_session_with(base: &std::path::Path, name: &str, messages: &[Value], meta: Value) -> Result<(), String> {
    if !valid_session_name(name) {
        return Err("invalid session name".into());
    }
    let file = session_file(base, name);
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let tmp = file.with_extension("json.tmp");
    let mut body = meta;
    body["messages"] = json!(messages);
    std::fs::write(&tmp, body.to_string()).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &file).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------------------------------------------
// The real server connection
// ---------------------------------------------------------------------------------------------------------------

pub(crate) struct HttpBackend {
    client: reqwest::blocking::Client,
    base: String,
    key: Option<String>,
}

impl HttpBackend {
    pub(crate) fn new(url: &str, key: Option<String>) -> Result<Self, String> {
        let base = url.trim_end_matches('/').trim_end_matches("/v1").to_owned();
        if !(base.starts_with("http://") || base.starts_with("https://")) {
            return Err("--url must begin with http:// or https://".into());
        }
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(3600))
            .user_agent("fllm-agent-cli/0.1")
            .build()
            .map_err(|e| format!("cannot initialize HTTP client: {e}"))?;
        Ok(Self { client, base, key })
    }

    fn call(&self, method: reqwest::Method, path: &str, body: Option<&Value>) -> Result<Value, String> {
        let mut request = self.client.request(method, format!("{}{path}", self.base));
        if let Some(key) = &self.key {
            request = request.bearer_auth(key);
        }
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request.send().map_err(|e| format!("cannot reach the server at {}: {e}", self.base))?;
        let status = response.status();
        let text = response.text().map_err(|e| e.to_string())?;
        let parsed = serde_json::from_str::<Value>(&text).unwrap_or(Value::Null);
        if !status.is_success() {
            let message = parsed["error"]["message"].as_str().map_or_else(|| clip(&text, 300), str::to_owned);
            return Err(format!("{path}: HTTP {status}: {message}"));
        }
        Ok(parsed)
    }

    pub(crate) fn get_json(&self, path: &str) -> Result<Value, String> {
        self.call(reqwest::Method::GET, path, None)
    }

    /// The served model's id and its context length (`max_model_len`), when the server tells it.
    pub(crate) fn model_info(&self) -> Result<(String, Option<u64>), String> {
        let models = self.call(reqwest::Method::GET, "/v1/models", None)?;
        let first = &models["data"][0];
        let id = first["id"].as_str().map(str::to_owned).ok_or_else(|| "the server lists no model; load one first".to_owned())?;
        Ok((id, first["max_model_len"].as_u64()))
    }

    pub(crate) fn set_folder(&self, folder: &str) -> Result<(), String> {
        self.call(reqwest::Method::POST, "/api/studio/agent/workspace", Some(&json!({"path": folder}))).map(|_| ())
    }

    pub(crate) fn end(&self) {
        let _ = self.call(reqwest::Method::POST, "/api/studio/agent/end", Some(&json!({})));
    }
}

impl Backend for HttpBackend {
    fn tools(&mut self) -> Result<Vec<Value>, String> {
        let answer = self.call(reqwest::Method::GET, "/api/studio/agent/tools", None)?;
        answer["tools"].as_array().cloned().ok_or_else(|| "the server sent no tool list".to_owned())
    }

    fn complete(&mut self, body: Value) -> Result<Value, String> {
        self.call(reqwest::Method::POST, "/v1/chat/completions", Some(&body))
    }

    fn instructions(&mut self) -> Result<Value, String> {
        self.call(reqwest::Method::POST, "/api/studio/agent/instructions", Some(&json!({})))
    }

    fn run_tool(&mut self, name: &str, args: &Value) -> Result<String, String> {
        let answer = self.call(reqwest::Method::POST, "/api/studio/agent/tool", Some(&json!({"name": name, "args": args})))?;
        if answer["ok"].as_bool().unwrap_or(false) {
            Ok(answer["output"].as_str().unwrap_or_default().to_owned())
        } else {
            Err(answer["error"].as_str().unwrap_or("the tool failed").to_owned())
        }
    }
}

/// The answer length to ask for: what the user wants, but at most a quarter of the model's context window (the prompt and the
/// tool schemas need the rest). Without a known window the request is used as it is. Never below 64.
pub fn effective_max_tokens(requested: usize, context: Option<u64>) -> usize {
    match context {
        Some(window) => requested.min(((window / 4) as usize).max(64)),
        None => requested,
    }
}

/// A warning when the tool schemas alone are likely to fill the window (about 3.5 characters per token), else `None`.
pub fn context_warning(tool_schema_chars: usize, context: Option<u64>) -> Option<String> {
    let window = context?;
    let tokens = (tool_schema_chars as f64 / 3.5) as u64;
    (tokens * 2 > window).then(|| format!("the tool list needs about {tokens} tokens and the model's context is {window}: requests will probably fail; use a model with a larger context (--ctx)"))
}

/// One line of a trajectory file: everything needed to replay or study a run. The system message is included because it
/// is part of what the model saw; API keys are never part of any of it.
pub fn trajectory_line(folder: &str, model: &str, system: &str, outcome: &Outcome, time: i64) -> String {
    let mut messages = vec![json!({"role": "system", "content": system})];
    messages.extend(agent_extras::strip_images(&outcome.messages));
    let calls: Vec<Value> = outcome.tool_calls.iter().map(|c| json!({"tool": c.name, "permission": c.category, "status": c.status})).collect();
    json!({
        "time": time,
        "folder": folder,
        "model": model,
        "status": if outcome.status == Status::Done { "done" } else { "out_of_steps" },
        "steps": outcome.steps,
        "tool_calls": calls,
        "messages": messages,
    })
    .to_string()
}

fn append_trajectory(file: &std::path::Path, line: &str) -> Result<(), String> {
    use std::io::Write as _;
    if let Some(dir) = file.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(file).map_err(|e| format!("cannot write {}: {e}", file.display()))?;
    writeln!(f, "{line}").map_err(|e| e.to_string())
}

/// Keeps a long conversation inside a bound without breaking a tool call from its result: the kept part starts at a user
/// message. A conversation that is short enough is returned unchanged.
pub fn trim_history(messages: Vec<Value>, max: usize) -> Vec<Value> {
    if messages.len() <= max {
        return messages;
    }
    let from = messages.len() - max;
    match messages[from..].iter().position(|m| m["role"] == "user") {
        Some(offset) => messages[from + offset..].to_vec(),
        None => messages.iter().rposition(|m| m["role"] == "user").map(|i| messages[i..].to_vec()).unwrap_or_default(),
    }
}

/// The checks of `--review`: the diff with the findings of the fixed rules (`review` tool), `diagnostics` for each changed file and the
/// verification command. The command is typed by the user, so it runs without `--allow-command`, but a deny rule still stops it.
pub fn self_review(backend: &mut dyn Backend, edited: &[String], verify: Option<&str>, rules: &Rules) -> agent_extras::ReviewReport {
    let mut report = agent_extras::ReviewReport { review: backend.run_tool("review", &json!({})).unwrap_or_default(), ..Default::default() };
    for path in edited.iter().take(12) {
        if let Ok(out) = backend.run_tool("diagnostics", &json!({"path": path})) {
            if !agent_extras::is_clean_diagnostic(&out) {
                report.problems.push((path.clone(), out));
            }
        }
    }
    if let Some(command) = verify {
        let args = json!({"command": command});
        report.verify = Some(if rules.decide("bash", "command", &args) == Some(Decision::Deny) {
            (false, "the verification command is blocked by a deny rule".to_owned())
        } else {
            match backend.run_tool("bash", &args) {
                Ok(out) => (true, out),
                Err(e) => (false, e),
            }
        });
    }
    report
}

/// After a long run: ask the model for a skill (without tools) and store it as a proposal, or as a skill when `auto` is set.
fn propose_skill(http: &mut HttpBackend, cfg: &LoopConfig, system: &str, outcome: &Outcome, base: &std::path::Path, auto: bool, progress: &mut dyn FnMut(&str)) {
    let mut messages = vec![json!({"role": "system", "content": system})];
    messages.extend(outcome.messages.iter().cloned());
    messages.push(json!({"role": "user", "content": agent_extras::SKILL_PROMPT}));
    let mut body = json!({"messages": messages, "max_tokens": cfg.max_tokens.min(1500), "temperature": 0.2, "stream": false});
    if let Some(model) = &cfg.model {
        body["model"] = json!(model);
    }
    let Ok(reply) = http.complete(body) else { return };
    let Some((name, file)) = agent_extras::parse_skill(&message_text(&reply["choices"][0]["message"])) else {
        progress("[skills] the model found nothing worth saving");
        return;
    };
    match agent_extras::save_skill(base, &name, &file, auto) {
        Ok(path) if auto => progress(&format!("[skills] saved skill '{name}' at {}", path.display())),
        Ok(path) => progress(&format!("[skills] proposed skill '{name}' at {} (fllm-rs skills approve {name})", path.display())),
        Err(e) => progress(&format!("[skills] {e}")),
    }
}

/// Copies the text of a run into the search index of the Studio (best effort: an older server or a missing route is ignored).
fn index_session(http: &HttpBackend, id: &str, messages: &[Value], now: i64) {
    let (title, text) = agent_extras::session_text(messages);
    if text.trim().is_empty() {
        return;
    }
    let _ = http.call(reqwest::Method::POST, "/api/studio/sessions/index", Some(&json!({"id": format!("cli:{id}"), "title": title, "text": text, "kind": "code", "updated": now * 1000})));
}

/// `fllm-rs agent ...`: returns the exit code.
pub fn run(args: &[String]) -> Result<i32, String> {
    let mut options = match parse_args(args) {
        Ok(o) => o,
        Err(e) if e == HELP => {
            println!("{HELP}");
            return Ok(0);
        }
        Err(e) => return Err(e),
    };
    if options.prompt == "-" {
        let mut text = String::new();
        io::stdin().read_to_string(&mut text).map_err(|e| e.to_string())?;
        if text.trim().is_empty() {
            return Err("the task on standard input is empty".into());
        }
        options.prompt = text;
    }
    let folder = match &options.folder {
        Some(f) => f.clone(),
        None => std::env::current_dir().map_err(|e| e.to_string())?,
    };
    let folder = folder.canonicalize().map_err(|e| format!("the folder {} cannot be used: {e}", folder.display()))?;
    if !folder.is_dir() {
        return Err(format!("{} is not a folder", folder.display()));
    }
    let folder_text = folder.display().to_string();

    let mut http = HttpBackend::new(&options.url, options.api_key.clone())?;
    let (served, context) = http.model_info()?;
    let model = options.model.clone().unwrap_or(served);
    http.set_folder(&folder_text)?;
    if let Ok(tools) = http.tools() {
        if let Some(warning) = context_warning(tools.iter().map(|t| t.to_string().len()).sum(), context) {
            eprintln!("[warning] {warning}");
        }
    }

    let base = profile_base(options.profile.as_deref())?;
    let started = std::time::Instant::now();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    if options.continue_last && options.session.is_none() {
        options.session = agent_extras::latest_session(&base);
        if options.session.is_none() {
            return Err("--continue: there is no earlier session to continue".into());
        }
    }
    let session_id = options.session.clone().unwrap_or_else(|| agent_extras::fresh_session_name(now));
    let mut history = match &options.session {
        Some(name) => trim_history(load_session_in(&base, name)?, MAX_SESSION_MESSAGES),
        None => Vec::new(),
    };
    let pictures: Vec<Value> = options.images.iter().map(|p| agent_extras::image_part(p)).collect::<Result<_, _>>()?;
    history.push(json!({"role": "user", "content": agent_extras::user_content(&options.prompt, &pictures)}));

    // permission rules: the settings files (user < project < local), then the flags; plan mode is read-only whatever else is allowed
    let mut rules = agent_rules::load_hierarchy(&studio_folder(), &folder)?;
    for (kind, text) in &options.rule_flags {
        rules.merge(Rules::from_json(&json!({ *kind: [text] })));
    }
    let base_rules = rules.clone();
    let plan = options.plan || rules.mode.as_deref() == Some("plan");
    let mut allow = options.allow.clone();
    let mut extra = options.system.clone().unwrap_or_default();
    if plan {
        allow = BTreeSet::from(["read".to_owned()]);
        rules.merge(Rules::from_json(&json!({"deny": ["bash: *", "edit: **", "network: *", "memory: *"]})));
        extra.push_str("\n\nPLAN MODE: you may only read and search; every other action is refused. Investigate, then finish with a numbered plan: the files to change, what to change in each, and how to check it. Do not claim that anything was changed.");
    }
    let system = full_system(&mut http, &folder_text, &allow, Some(extra.as_str()), load_soul(&base).as_deref());
    let mut cfg = LoopConfig {
        model: Some(model),
        allow: allow.clone(),
        rules: rules.clone(),
        max_steps: options.max_steps,
        max_tokens: effective_max_tokens(options.max_tokens, context),
        temperature: options.temperature,
    };
    let quiet = options.quiet;
    let streaming = options.format == OutputFormat::StreamJson;
    let mut progress = |line: &str| {
        if !quiet {
            eprintln!("{line}");
        }
    };
    if streaming {
        let names: Vec<String> = http.tools().unwrap_or_default().iter().filter_map(|t| t["function"]["name"].as_str().map(str::to_owned)).collect();
        println!("{}", agent_extras::init_event(&session_id, cfg.model.as_deref().unwrap_or_default(), &folder_text, &names, if plan { "plan" } else { "default" }));
    }
    let mut approver = TerminalApprover::default();
    let ask = options.ask;
    let rules_for_ask = rules.clone();
    let mut approve = |name: &str, category: &str, args: &Value| {
        let rule_ask = rules_for_ask.decide(name, category, args) == Some(Decision::Ask);
        (ask || rule_ask) && approver.ask(name, category, args)
    };
    let mut observe = |m: &Value| {
        if streaming {
            for event in agent_extras::message_events(m, &session_id) {
                println!("{event}");
            }
        }
    };
    let mut outcome = run_loop_full(&mut http, &cfg, &system, history, &mut progress, &mut approve, &mut observe);

    // self-review: look at the work before it counts as done, and let the agent fix what the checks find
    if options.review {
        let mut round = 1;
        while let Ok(done) = &outcome {
            if done.status != Status::Done || done.edited.is_empty() || round > options.review_rounds {
                break;
            }
            progress(&format!("[review {round}] checking {} changed file(s)", done.edited.len()));
            let report = self_review(&mut http, &done.edited, options.verify.as_deref(), &rules);
            let Some(feedback) = agent_extras::review_feedback(&report, round) else { break };
            let mut next = done.messages.clone();
            next.push(json!({"role": "user", "content": feedback}));
            let (calls_before, steps_before, edited_before) = (done.tool_calls.clone(), done.steps, done.edited.clone());
            outcome = run_loop_full(&mut http, &cfg, &system, next, &mut progress, &mut approve, &mut observe).map(|mut again| {
                again.tool_calls.splice(0..0, calls_before);
                again.steps += steps_before;
                for p in edited_before {
                    if !again.edited.contains(&p) {
                        again.edited.push(p);
                    }
                }
                again
            });
            round += 1;
        }
    }

    // plan mode on a terminal: show the plan and run it after a yes
    let mut planned = plan;
    if plan {
        if let Ok(done) = &outcome {
            use std::io::IsTerminal;
            if done.status == Status::Done && io::stdin().is_terminal() && io::stdout().is_terminal() && !streaming && !options.json {
                println!("{}\n", done.answer.trim_end());
                eprint!("Run this plan now? Each action will be asked first. [y/N]: ");
                io::stderr().flush().ok();
                let mut line = String::new();
                if io::stdin().read_line(&mut line).unwrap_or(0) > 0 && parse_answer(&line) == Answer::Yes {
                    cfg.allow = options.allow.clone();
                    cfg.rules = base_rules.clone();
                    let mut next = done.messages.clone();
                    next.push(json!({"role": "user", "content": "The plan is approved. Carry it out now."}));
                    let build_system = full_system(&mut http, &folder_text, &options.allow, options.system.as_deref(), load_soul(&base).as_deref());
                    let mut ask_each = |name: &str, category: &str, args: &Value| approver.ask(name, category, args);
                    outcome = run_loop_full(&mut http, &cfg, &build_system, next, &mut progress, &mut ask_each, &mut observe);
                    planned = false;
                }
            }
        }
    }
    http.end();
    if let Err(e) = &outcome {
        if streaming {
            println!("{}", agent_extras::result_event("", "error", 0, &session_id, started.elapsed().as_millis(), Some(e)));
        }
    }
    let outcome = outcome?;

    // a long run may leave a skill behind: proposed (waits for approval) or saved at once
    if options.propose_skills && outcome.status == Status::Done && outcome.tool_calls.len() >= 8 {
        propose_skill(&mut http, &cfg, &system, &outcome, &base, options.auto_skills, &mut progress);
    }

    // every run is kept (sessions of the same name continue), so `--resume`, `--continue` and the search can find it again
    if !outcome.messages.is_empty() {
        save_session_with(&base, &session_id, &agent_extras::strip_images(&outcome.messages), json!({"folder": folder_text, "model": cfg.model, "updated": now, "steps": outcome.steps, "edited": outcome.edited}))?;
        if options.session.is_none() {
            agent_extras::prune_auto_sessions(&base, 200);
        }
        index_session(&http, &session_id, &outcome.messages, now);
    }
    if let Some(file) = &options.trajectory {
        append_trajectory(file, &trajectory_line(&folder_text, cfg.model.as_deref().unwrap_or_default(), &system, &outcome, now))?;
    }
    let out_of_steps = outcome.status == Status::OutOfSteps;
    let status_text = if out_of_steps { "out_of_steps" } else { "done" };
    if streaming {
        println!("{}", agent_extras::result_event(&outcome.answer, status_text, outcome.steps, &session_id, started.elapsed().as_millis(), None));
    } else if options.format == OutputFormat::Json {
        let calls: Vec<Value> = outcome
            .tool_calls
            .iter()
            .map(|c| json!({"tool": c.name, "permission": c.category, "status": c.status, "detail": c.detail}))
            .collect();
        let report = json!({
            "answer": outcome.answer,
            "status": status_text,
            "steps": outcome.steps,
            "folder": folder_text,
            "session": session_id,
            "plan": planned,
            "edited": outcome.edited,
            "tool_calls": calls,
        });
        println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    } else {
        println!("{}", outcome.answer.trim_end());
    }
    io::stdout().flush().ok();
    if out_of_steps {
        eprintln!("[stopped at --max-steps {}; the report above is what was found so far]", options.max_steps);
        return Ok(3);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|i| (*i).to_owned()).collect()
    }

    /// A model that follows a script and tools that record what ran.
    struct Scripted {
        replies: VecDeque<Value>,
        bodies: Vec<Value>,
        ran: Vec<(String, Value)>,
        catalog: Vec<Value>,
        fail_tool: bool,
    }

    fn tool(name: &str, permission: &str) -> Value {
        json!({"type": "function", "function": {"name": name, "description": "d", "parameters": {"type": "object", "properties": {}}}, "permission": permission})
    }

    fn scripted(replies: Vec<Value>) -> Scripted {
        Scripted {
            replies: replies.into(),
            bodies: Vec::new(),
            ran: Vec::new(),
            catalog: vec![tool("read_file", "read"), tool("write_file", "edit"), tool("bash", "command"), tool("computer", "computer")],
            fail_tool: false,
        }
    }

    impl Backend for Scripted {
        fn tools(&mut self) -> Result<Vec<Value>, String> {
            Ok(self.catalog.clone())
        }
        fn complete(&mut self, body: Value) -> Result<Value, String> {
            self.bodies.push(body);
            let reply = self.replies.pop_front().ok_or_else(|| "script ended".to_owned())?;
            // a scripted failure of the request: {"__error": "text"}
            match reply.get("__error").and_then(Value::as_str) {
                Some(e) => Err(e.to_owned()),
                None => Ok(reply),
            }
        }
        fn run_tool(&mut self, name: &str, args: &Value) -> Result<String, String> {
            self.ran.push((name.to_owned(), args.clone()));
            if self.fail_tool { Err("boom".into()) } else { Ok(format!("ran {name}")) }
        }
    }

    fn text(t: &str) -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": t}}]})
    }

    fn calls(items: &[(&str, Value)]) -> Value {
        let calls: Vec<Value> = items
            .iter()
            .enumerate()
            .map(|(i, (n, a))| json!({"id": format!("c{i}"), "type": "function", "function": {"name": n, "arguments": a.to_string()}}))
            .collect();
        json!({"choices": [{"message": {"role": "assistant", "content": null, "tool_calls": calls}}]})
    }

    fn cfg(allow: &[&str], max_steps: usize) -> LoopConfig {
        LoopConfig {
            model: Some("m".into()),
            allow: allow.iter().map(|a| (*a).to_owned()).collect(),
            rules: Rules::default(),
            max_steps,
            max_tokens: 100,
            temperature: 0.0,
        }
    }

    fn run_result(b: &mut Scripted, allow: &[&str], max_steps: usize) -> Result<Outcome, String> {
        run_loop(b, &cfg(allow, max_steps), "sys", vec![json!({"role": "user", "content": "task"})], &mut |_| {})
    }

    fn run(b: &mut Scripted, allow: &[&str], max_steps: usize) -> Outcome {
        run_result(b, allow, max_steps).unwrap()
    }

    #[test]
    fn a_read_call_runs_and_the_answer_ends_the_loop() {
        let mut b = scripted(vec![calls(&[("read_file", json!({"path": "a.txt"}))]), text("done: a.txt read")]);
        let out = run(&mut b, &["read"], 5);
        assert_eq!(out.answer, "done: a.txt read");
        assert_eq!(out.status, Status::Done);
        assert_eq!(out.steps, 2);
        assert_eq!(b.ran, vec![("read_file".to_owned(), json!({"path": "a.txt"}))]);
        // the tool result went back to the model under the id of its call
        let tool_msg = b.bodies[1]["messages"].as_array().unwrap().iter().find(|m| m["role"] == "tool").unwrap().clone();
        assert_eq!(tool_msg["tool_call_id"], "c0");
        assert_eq!(tool_msg["content"], "ran read_file");
    }

    #[test]
    fn an_edit_is_refused_without_the_flag_and_runs_with_it() {
        let call = || calls(&[("write_file", json!({"path": "x", "content": "y"}))]);
        let mut denied = scripted(vec![call(), text("could not write")]);
        let out = run(&mut denied, &["read"], 5);
        assert!(denied.ran.is_empty(), "a refused call must not reach the tool");
        assert_eq!(out.tool_calls[0].status, "denied");
        let told = denied.bodies[1]["messages"].as_array().unwrap().iter().find(|m| m["role"] == "tool").unwrap()["content"].as_str().unwrap().to_owned();
        assert!(told.contains("--allow-edit"), "the model is told which flag would allow it: {told}");

        let mut allowed = scripted(vec![call(), text("written")]);
        let out = run(&mut allowed, &["read", "edit"], 5);
        assert_eq!(allowed.ran.len(), 1);
        assert_eq!(out.tool_calls[0].status, "ok");
        // but a command is still not allowed by --allow-edit
        let mut cmd = scripted(vec![calls(&[("bash", json!({"command": "ls"}))]), text("no")]);
        run(&mut cmd, &["read", "edit"], 5);
        assert!(cmd.ran.is_empty());
    }

    #[test]
    fn computer_use_is_never_offered_and_the_permission_key_is_not_sent() {
        let mut b = scripted(vec![text("hi")]);
        run(&mut b, &["read"], 5);
        let tools = b.bodies[0]["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["read_file", "write_file", "bash"]);
        assert!(tools.iter().all(|t| t.get("permission").is_none()));
        assert_eq!(b.bodies[0]["messages"][0]["role"], "system");
        assert_eq!(b.bodies[0]["stream"], false);
    }

    #[test]
    fn the_tool_call_grammar_is_asked_for_and_dropped_for_a_server_that_does_not_know_it() {
        let mut b = scripted(vec![json!({"__error": "/v1/chat/completions: HTTP 400 Bad Request: unknown field tool_call_grammar"}), text("done")]);
        let out = run(&mut b, &["read"], 5);
        assert_eq!(out.answer, "done");
        assert_eq!(b.bodies[0]["tool_call_grammar"], true);
        assert!(b.bodies[1].get("tool_call_grammar").is_none(), "asked again without the field");
        assert!(b.bodies[1].get("tools").is_some(), "the tools are still offered");
    }

    #[test]
    fn arguments_the_server_refuses_are_told_to_the_model_twice_at_most() {
        let refused = || json!({"__error": "/v1/chat/completions: HTTP 422 Unprocessable Entity: tool arguments do not match read_file: \"1\" is not of type \"integer\""});
        let mut b = scripted(vec![refused(), text("fixed")]);
        let out = run(&mut b, &["read"], 5);
        assert_eq!(out.answer, "fixed");
        let last = b.bodies[1]["messages"].as_array().unwrap().last().unwrap().clone();
        assert!(last["content"].as_str().unwrap().contains("refused by the server"), "{last}");

        let mut c = scripted(vec![refused(), refused(), refused(), text("never")]);
        assert!(run_result(&mut c, &["read"], 8).is_err(), "the third refusal ends the run");
    }

    #[test]
    fn unknown_tools_and_bad_arguments_are_reported_to_the_model_not_run() {
        let bad = json!({"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "a", "function": {"name": "nope", "arguments": "{}"}},
            {"id": "b", "function": {"name": "read_file", "arguments": "{not json"}},
        ]}}]});
        let mut b = scripted(vec![bad, text("ok")]);
        let out = run(&mut b, &["read"], 5);
        assert!(b.ran.is_empty());
        assert_eq!(out.tool_calls.iter().map(|c| c.status).collect::<Vec<_>>(), vec!["unknown", "bad_args"]);
    }

    #[test]
    fn a_tool_error_is_passed_on_and_calls_without_an_id_get_one() {
        let no_id = json!({"choices": [{"message": {"content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "q"}}}]}}]});
        let mut b = scripted(vec![no_id, text("fine")]);
        b.fail_tool = true;
        let out = run(&mut b, &["read"], 5);
        assert_eq!(out.tool_calls[0].status, "error");
        let msgs = b.bodies[1]["messages"].as_array().unwrap();
        let assistant = msgs.iter().find(|m| m["role"] == "assistant").unwrap();
        let id = assistant["tool_calls"][0]["id"].as_str().unwrap();
        assert!(!id.is_empty());
        let tool_msg = msgs.iter().find(|m| m["role"] == "tool").unwrap();
        assert_eq!(tool_msg["tool_call_id"], id);
        assert_eq!(tool_msg["content"], "Error: boom");
    }

    #[test]
    fn the_same_call_three_times_in_a_row_is_not_run_a_third_time() {
        let same = || calls(&[("read_file", json!({"path": "a"}))]);
        let mut b = scripted(vec![same(), same(), same(), text("gave up")]);
        let out = run(&mut b, &["read"], 10);
        assert_eq!(b.ran.len(), 2);
        assert_eq!(out.tool_calls[2].status, "repeated");
    }

    #[test]
    fn running_out_of_steps_asks_for_a_report_without_tools() {
        let call = || calls(&[("read_file", json!({"path": "a"})), ("read_file", json!({"path": "b"}))]);
        let mut b = scripted(vec![call(), call(), text("partial report")]);
        let out = run(&mut b, &["read"], 2);
        assert_eq!(out.status, Status::OutOfSteps);
        assert_eq!(out.answer, "partial report");
        let last = b.bodies.last().unwrap();
        assert!(last.get("tools").is_none(), "the report request must not offer tools");
        assert!(last["messages"].as_array().unwrap().last().unwrap()["content"].as_str().unwrap().contains("Stop using tools"));
    }

    #[test]
    fn running_out_of_steps_without_any_report_still_returns_the_tool_history() {
        // the report request is answered with another tool call (no text), then with an error: both must give the fallback
        let call = || calls(&[("read_file", json!({"path": "a"}))]);
        let mut b = scripted(vec![call(), call()]);
        let out = run(&mut b, &["read"], 1);
        assert_eq!(out.status, Status::OutOfSteps);
        assert!(out.answer.contains("stopped after 1 steps") && out.answer.contains("read_file [ok]"), "{}", out.answer);
        let mut gone = scripted(vec![call()]); // the server has no answer left for the report request
        let out = run(&mut gone, &["read"], 1);
        assert_eq!(out.status, Status::OutOfSteps);
        assert!(out.answer.contains("read_file"));
    }

    #[test]
    fn an_empty_answer_and_a_dead_server_are_errors() {
        let mut empty = scripted(vec![text("  ")]);
        assert!(run_loop(&mut empty, &cfg(&["read"], 3), "s", vec![], &mut |_| {}).is_err());
        let mut dead = scripted(vec![]);
        assert!(run_loop(&mut dead, &cfg(&["read"], 3), "s", vec![], &mut |_| {}).unwrap_err().contains("script ended"));
    }

    #[test]
    fn call_arguments_accept_strings_and_objects_only_when_they_are_objects() {
        assert_eq!(parse_call_args(&json!("{\"a\":1}")).unwrap(), json!({"a": 1}));
        assert_eq!(parse_call_args(&json!({"a": 1})).unwrap(), json!({"a": 1}));
        assert_eq!(parse_call_args(&json!("")).unwrap(), json!({}));
        assert!(parse_call_args(&json!("[1]")).is_err());
        assert!(parse_call_args(&json!("{oops")).is_err());
        assert!(parse_call_args(&json!(5)).is_err());
    }

    #[test]
    fn long_tool_output_keeps_its_start_and_end() {
        let long = format!("{}{}{}", "A".repeat(5000), "M".repeat(100_000), "Z".repeat(5000));
        let cut = clip(&long, MAX_TOOL_OUTPUT);
        assert!(cut.starts_with("AAAA") && cut.ends_with("ZZZZ"));
        assert!(cut.contains("characters cut"));
        assert!(cut.chars().count() < MAX_TOOL_OUTPUT + 100);
        assert_eq!(clip("short", 100), "short");
    }

    #[test]
    fn arguments_default_to_read_only_and_flags_open_categories() {
        let o = parse_args(&s(&["-p", "fix it"])).unwrap();
        assert_eq!(o.allow, BTreeSet::from(["read".to_owned()]));
        assert_eq!(o.prompt, "fix it");
        assert_eq!(o.max_steps, DEFAULT_MAX_STEPS);
        let o = parse_args(&s(&["-p", "x", "--allow-edit", "--allow-command", "--max-steps", "7", "--json", "--session", "nightly-1"])).unwrap();
        assert!(o.allow.contains("edit") && o.allow.contains("command") && !o.allow.contains("network"));
        assert_eq!((o.max_steps, o.json, o.session.as_deref()), (7, true, Some("nightly-1")));
        let all = parse_args(&s(&["--yes", "do", "the", "thing"])).unwrap();
        assert_eq!(all.allow.len(), 5);
        assert_eq!(all.prompt, "do the thing", "loose words form the task when -p is absent");
    }

    #[test]
    fn bad_arguments_are_refused_with_a_reason() {
        for bad in [
            s(&[]),
            s(&["-p", " "]),
            s(&["-p", "x", "--bogus"]),
            s(&["-p", "x", "--max-steps", "0"]),
            s(&["-p", "x", "--max-steps", "501"]),
            s(&["-p", "x", "--session", "../etc"]),
            s(&["-p", "x", "--session", "Upper"]),
            s(&["-p", "x", "--temperature", "-1"]),
            s(&["-p"]),
        ] {
            assert!(parse_args(&bad).is_err(), "{bad:?} must be refused");
        }
        assert_eq!(parse_args(&s(&["--help"])).unwrap_err(), HELP);
    }

    #[test]
    fn a_trajectory_line_holds_the_whole_run_and_no_key() {
        let mut b = scripted(vec![calls(&[("read_file", json!({"path": "a"}))]), text("report")]);
        let out = run(&mut b, &["read"], 5);
        let line = trajectory_line("/w", "m1", "SYSTEM TEXT", &out, 9);
        assert!(!line.contains('\n'), "one run is one line");
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!((v["status"].as_str(), v["model"].as_str(), v["time"].as_i64()), (Some("done"), Some("m1"), Some(9)));
        let roles: Vec<&str> = v["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool", "assistant"]);
        assert_eq!(v["tool_calls"][0]["tool"], "read_file");
        let dir = std::env::temp_dir().join(format!("fllm-traj-{}", std::process::id()));
        let file = dir.join("sub").join("t.jsonl");
        append_trajectory(&file, &line).unwrap();
        append_trajectory(&file, &line).unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap().lines().count(), 2, "runs are appended");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn trimming_never_starts_in_the_middle_of_a_tool_exchange() {
        let u = |t: &str| json!({"role": "user", "content": t});
        let a = |t: &str| json!({"role": "assistant", "content": t});
        let tool = || json!({"role": "tool", "tool_call_id": "c", "content": "r"});
        let short = vec![u("1"), a("2")];
        assert_eq!(trim_history(short.clone(), 10), short);
        let long = vec![u("1"), a("2"), tool(), a("3"), u("4"), a("5"), tool(), a("6")];
        let kept = trim_history(long.clone(), 5);
        assert_eq!(kept[0]["role"], "user");
        assert_eq!(kept, long[4..].to_vec());
        // no user message inside the window: fall back to the last user message
        let odd = vec![u("1"), a("2"), tool(), a("3"), tool(), a("4")];
        assert_eq!(trim_history(odd, 3), vec![u("1"), a("2"), tool(), a("3"), tool(), a("4")]);
    }

    #[test]
    fn the_answer_length_follows_the_context_window() {
        assert_eq!(effective_max_tokens(4096, Some(512)), 128, "a quarter of a small window");
        assert_eq!(effective_max_tokens(4096, Some(32768)), 4096, "a big window leaves the request alone");
        assert_eq!(effective_max_tokens(4096, Some(100)), 64, "never below 64");
        assert_eq!(effective_max_tokens(4096, None), 4096);
        assert_eq!(effective_max_tokens(100, Some(512)), 100, "a small request stays small");
        assert!(context_warning(14_000, Some(512)).unwrap().contains("4000"));
        assert!(context_warning(14_000, Some(32768)).is_none());
        assert!(context_warning(14_000, None).is_none());
    }

    #[test]
    fn instructions_and_memory_become_a_prompt_block() {
        let answer = json!({"files": [{"file": "AGENTS.md", "text": "Use tabs.\n"}, {"file": "empty.md", "text": "  "}],
                            "memory": {"workspace": ["tests run with make test"], "global": ["prefers short answers", ""]}});
        let block = instructions_block(&answer);
        assert!(block.contains("# Project instructions (AGENTS.md)\nUse tabs."), "{block}");
        assert!(!block.contains("empty.md"));
        assert!(block.contains("# Memory of this folder\n- tests run with make test") && block.contains("# Memory\n- prefers short answers"));
        assert_eq!(instructions_block(&json!({})), "", "nothing to add: no empty headings");
        let huge = json!({"files": [{"file": "A.md", "text": "x".repeat(50_000)}]});
        assert!(instructions_block(&huge).chars().count() < MAX_INSTRUCTIONS_CHARS + 100);
    }

    #[test]
    fn the_system_prompt_contains_persona_and_instructions_when_the_server_has_them() {
        struct WithInstructions;
        impl Backend for WithInstructions {
            fn tools(&mut self) -> Result<Vec<Value>, String> { Ok(vec![]) }
            fn complete(&mut self, _: Value) -> Result<Value, String> { Err("no".into()) }
            fn run_tool(&mut self, _: &str, _: &Value) -> Result<String, String> { Err("no".into()) }
            fn instructions(&mut self) -> Result<Value, String> { Ok(json!({"files": [{"file": "AGENTS.md", "text": "Run make test."}]})) }
        }
        let allow = BTreeSet::from(["read".to_owned()]);
        let text = full_system(&mut WithInstructions, "/w", &allow, Some("extra"), Some("You are Hermes, short and direct."));
        for part in ["/w", "extra", "# Persona\nYou are Hermes, short and direct.", "Run make test."] {
            assert!(text.contains(part), "missing {part:?}");
        }
        // a server that cannot give instructions (older build, error) does not break the run
        let mut plain = scripted(vec![]);
        assert!(!full_system(&mut plain, "/w", &allow, None, None).contains("# Persona"));
    }

    #[test]
    fn soul_and_profiles_come_from_files_and_names_are_checked() {
        let dir = std::env::temp_dir().join(format!("fllm-soul-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(load_soul(&dir), None);
        std::fs::write(dir.join("SOUL.md"), "  \n").unwrap();
        assert_eq!(load_soul(&dir), None, "an empty file is no persona");
        std::fs::write(dir.join("SOUL.md"), "Be direct.\n").unwrap();
        assert_eq!(load_soul(&dir).as_deref(), Some("Be direct."));
        std::fs::write(dir.join("SOUL.md"), "y".repeat(10_000)).unwrap();
        assert!(load_soul(&dir).unwrap().chars().count() <= MAX_SOUL_CHARS + 60, "the cut marker adds about 30 characters");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(profile_base(Some("work")).unwrap().ends_with("profiles/work"));
        assert!(profile_base(None).is_ok());
        for bad in ["../x", "A", "a/b", ""] {
            assert!(profile_base(Some(bad)).is_err(), "{bad}");
        }
        assert!(parse_args(&s(&["-p", "x", "--profile", "../x"])).is_err());
        assert_eq!(parse_args(&s(&["-p", "x", "--profile", "work", "--ask"])).map(|o| (o.profile, o.ask)).unwrap(), (Some("work".to_owned()), true));
    }

    #[test]
    fn profiles_keep_their_sessions_apart() {
        let dir = std::env::temp_dir().join(format!("fllm-prof-{}", std::process::id()));
        let (a, b) = (dir.join("profiles/a"), dir.join("profiles/b"));
        save_session_in(&a, "chat", &[json!({"role": "user", "content": "in a"})]).unwrap();
        assert!(load_session_in(&b, "chat").unwrap().is_empty(), "profile b does not see profile a");
        assert_eq!(load_session_in(&a, "chat").unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_approved_call_runs_once_and_a_refused_one_does_not() {
        let call = || calls(&[("write_file", json!({"path": "x", "content": "y"}))]);
        let mut seen: Vec<(String, String, Value)> = Vec::new();
        let mut b = scripted(vec![call(), text("done")]);
        let out = run_loop_with(&mut b, &cfg(&["read"], 5), "s", vec![json!({"role": "user", "content": "t"})], &mut |_| {}, &mut |n, c, a| {
            seen.push((n.to_owned(), c.to_owned(), a.clone()));
            true
        })
        .unwrap();
        assert_eq!(b.ran.len(), 1, "an approved call reaches the tool");
        assert_eq!(out.tool_calls[0].status, "approved");
        assert_eq!(seen, vec![("write_file".to_owned(), "edit".to_owned(), json!({"path": "x", "content": "y"}))]);
        // refused: not run, status denied, and a call that is already allowed is never put to the question
        let mut b = scripted(vec![call(), calls(&[("read_file", json!({"path": "q"}))]), text("done")]);
        let mut asked = 0;
        let out = run_loop_with(&mut b, &cfg(&["read"], 5), "s", vec![], &mut |_| {}, &mut |_, _, _| {
            asked += 1;
            false
        })
        .unwrap();
        assert_eq!((b.ran.len(), asked), (1, 1), "only the read ran; only the edit was asked about");
        assert_eq!(out.tool_calls.iter().map(|c| c.status).collect::<Vec<_>>(), vec!["denied", "ok"]);
        // unknown tools and bad arguments are never put to the question
        let bad = json!({"choices": [{"message": {"content": "", "tool_calls": [{"id": "a", "function": {"name": "write_file", "arguments": "{oops"}}]}}]});
        let mut b = scripted(vec![bad, text("x")]);
        let mut asked = 0;
        run_loop_with(&mut b, &cfg(&["read"], 5), "s", vec![], &mut |_| {}, &mut |_, _, _| {
            asked += 1;
            true
        })
        .unwrap();
        assert_eq!((asked, b.ran.len()), (0, 0), "a call with broken arguments is not offered for approval");
    }

    #[test]
    fn answers_and_call_descriptions_are_read_sensibly() {
        for (line, want) in [("y", Answer::Yes), ("YES\n", Answer::Yes), ("ja", Answer::Yes), ("a", Answer::Always), ("always", Answer::Always), ("n", Answer::No), ("", Answer::No), ("maybe", Answer::No)] {
            assert_eq!(parse_answer(line), want, "{line:?}");
        }
        let d = describe_call("bash", "command", &json!({"command": "rm -r build\nls"}));
        assert!(d.contains("bash (command): rm -r build ls") && !d.contains('\n'));
        assert!(describe_call("x", "edit", &json!({"a": 1})).contains("{\"a\":1}"));
    }

    #[test]
    fn session_names_are_slugs() {
        for ok in ["a", "nightly-1", "x_y", "0day"] {
            assert!(valid_session_name(ok), "{ok}");
        }
        for bad in ["", "-a", "_a", "A", "a/b", "a.b", "a b", &"x".repeat(65)] {
            assert!(!valid_session_name(bad), "{bad}");
        }
    }

    #[test]
    fn a_session_is_saved_and_continued() {
        let dir = std::env::temp_dir().join(format!("fllm-agent-session-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let name = "unit-test-session";
        assert!(load_session_in(&dir, name).unwrap().is_empty(), "a new session starts empty");
        let first = vec![json!({"role": "user", "content": "one"}), json!({"role": "assistant", "content": "two"})];
        save_session_in(&dir, name, &first).unwrap();
        assert_eq!(load_session_in(&dir, name).unwrap(), first);
        // a damaged file is reported, not silently replaced
        std::fs::write(session_file(&dir, name), "{broken").unwrap();
        assert!(load_session_in(&dir, name).unwrap_err().contains("damaged"));
        assert!(load_session_in(&dir, "../x").is_err());
        assert!(save_session_in(&dir, "../x", &first).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_system_prompt_names_the_folder_and_the_allowed_actions() {
        let allow = BTreeSet::from(["read".to_owned(), "edit".to_owned()]);
        let p = system_prompt("/work/app", &allow, Some("Use tabs."));
        assert!(p.contains("/work/app") && p.contains("read, edit") && !p.contains("command") && p.ends_with("Use tabs."));
    }

    fn cfg_with(allow: &[&str], rules: Rules) -> LoopConfig {
        LoopConfig { rules, ..cfg(allow, 5) }
    }

    fn rules_from(value: Value) -> Rules {
        Rules::from_json(&value)
    }

    #[test]
    fn a_deny_rule_beats_yes_and_an_allow_rule_lifts_a_missing_flag() {
        let history = || vec![json!({"role": "user", "content": "t"})];
        let all = ["read", "edit", "command", "network", "memory"];
        let mut b = scripted(vec![calls(&[("bash", json!({"command": "git push origin main"}))]), text("ok")]);
        let out = run_loop(&mut b, &cfg_with(&all, rules_from(json!({"deny": ["bash: git push*"]}))), "s", history(), &mut |_| {}).unwrap();
        assert!(b.ran.is_empty(), "the denied call did not run even with every flag");
        assert_eq!(out.tool_calls[0].status, "denied");
        assert!(b.bodies[1]["messages"].to_string().contains("deny rule"));

        let mut b = scripted(vec![calls(&[("bash", json!({"command": "npm test"}))]), text("ok")]);
        let out = run_loop(&mut b, &cfg_with(&["read"], rules_from(json!({"allow": ["bash: npm test*"]}))), "s", history(), &mut |_| {}).unwrap();
        assert_eq!(b.ran.len(), 1, "an allow rule runs a command no flag allows");
        assert_eq!(out.tool_calls[0].status, "approved");

        let mut b = scripted(vec![calls(&[("bash", json!({"command": "npm test && rm -rf /"}))]), text("ok")]);
        let out = run_loop(&mut b, &cfg_with(&["read"], rules_from(json!({"allow": ["bash: npm test*"]}))), "s", history(), &mut |_| {}).unwrap();
        assert!(b.ran.is_empty());
        assert_eq!(out.tool_calls[0].status, "denied", "an allow rule does not cover a compound line");
    }

    #[test]
    fn an_ask_rule_asks_even_when_the_flag_allows_it() {
        let history = vec![json!({"role": "user", "content": "t"})];
        let all = ["read", "edit", "command"];
        let rules = rules_from(json!({"ask": ["edit: src/**"]}));
        let mut b = scripted(vec![calls(&[("write_file", json!({"path": "src/a.rs", "content": "x"}))]), text("ok")]);
        let mut asked = Vec::new();
        let out = run_loop_with(&mut b, &cfg_with(&all, rules.clone()), "s", history.clone(), &mut |_| {}, &mut |n, c, _| {
            asked.push(format!("{n}/{c}"));
            true
        })
        .unwrap();
        assert_eq!(asked, vec!["write_file/edit"]);
        assert_eq!(out.tool_calls[0].status, "approved");
        assert_eq!(out.edited, vec!["src/a.rs"]);

        let mut b = scripted(vec![calls(&[("write_file", json!({"path": "src/a.rs", "content": "x"}))]), text("ok")]);
        let out = run_loop(&mut b, &cfg_with(&all, rules), "s", history, &mut |_| {}).unwrap();
        assert!(b.ran.is_empty(), "nobody to ask: it stays refused");
        assert_eq!(out.tool_calls[0].status, "denied");
    }

    #[test]
    fn the_observer_sees_every_message_in_order() {
        let mut b = scripted(vec![calls(&[("read_file", json!({"path": "a"}))]), text("done")]);
        let mut roles = Vec::new();
        run_loop_full(&mut b, &cfg(&["read"], 5), "s", vec![json!({"role": "user", "content": "t"})], &mut |_| {}, &mut |_, _, _| false, &mut |m| roles.push(m["role"].as_str().unwrap().to_owned())).unwrap();
        assert_eq!(roles, vec!["assistant", "tool", "assistant"]);
    }

    #[test]
    fn the_new_flags_parse() {
        let o = parse_args(&s(&["-p", "x", "--output-format", "stream-json", "--continue", "--plan", "--allow-rule", "bash: ls*", "--deny-rule", "Read(.env)", "--review-rounds", "3", "--verify", "cargo test", "--auto-skills"])).unwrap();
        assert_eq!(o.format, OutputFormat::StreamJson);
        assert!(o.continue_last && o.plan && o.review && o.propose_skills && o.auto_skills);
        assert_eq!(o.review_rounds, 3);
        assert_eq!(o.verify.as_deref(), Some("cargo test"));
        assert_eq!(o.rule_flags, vec![("allow", "bash: ls*".to_owned()), ("deny", "Read(.env)".to_owned())]);
        assert_eq!(parse_args(&s(&["-p", "x", "--resume", "my-run"])).unwrap().session.as_deref(), Some("my-run"));
        assert!(parse_args(&s(&["-p", "x", "--output-format", "xml"])).is_err());
        assert!(parse_args(&s(&["-p", "x", "--resume", "../x"])).is_err());
        assert!(parse_args(&s(&["-p", "x", "--review-rounds", "9"])).is_err());
        assert_eq!(parse_args(&s(&["-p", "x", "--json"])).unwrap().format, OutputFormat::Json);
    }

    #[test]
    fn the_self_review_collects_findings_diagnostics_and_the_verify_result() {
        struct Checks;
        impl Backend for Checks {
            fn tools(&mut self) -> Result<Vec<Value>, String> {
                Ok(vec![])
            }
            fn complete(&mut self, _: Value) -> Result<Value, String> {
                Err("no model".into())
            }
            fn run_tool(&mut self, name: &str, args: &Value) -> Result<String, String> {
                match name {
                    "review" => Ok("Review (diff): 1 file(s), +1/-0 lines, 0 finding(s)".into()),
                    "diagnostics" if args["path"] == "bad.py" => Ok("Problems found (python -m py_compile, exit code 1):\nSyntaxError".into()),
                    "diagnostics" => Ok("No problems found (python -m py_compile).".into()),
                    "bash" => Err("exit code 1\n2 tests failed".into()),
                    _ => Err("unknown".into()),
                }
            }
        }
        let report = self_review(&mut Checks, &["good.py".into(), "bad.py".into()], Some("pytest"), &Rules::default());
        assert_eq!(report.problems.len(), 1);
        assert_eq!(report.problems[0].0, "bad.py");
        assert_eq!(report.verify.as_ref().map(|v| v.0), Some(false));
        assert!(report.objective_problems());
        let blocked = self_review(&mut Checks, &[], Some("pytest"), &rules_from(json!({"deny": ["bash: pytest*"]})));
        assert!(blocked.verify.unwrap().1.contains("deny rule"));
    }
}
