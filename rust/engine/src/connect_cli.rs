//! `fllm-rs connect TOOL`: the exact environment and configuration that points a coding tool at an fLLM server.
//!
//! The command asks the running server for the model name and the context length (a server that is not running yet
//! is fine: the values come from the flags), turns them into the settings of the chosen tool and prints them. It
//! changes a file only with `--write`, always shows the difference first, and keeps a copy of a file it changes.
//! Everything that builds text is a pure function of [`Settings`], so it is covered by unit tests; the part that
//! touches the disk and the network is small.
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Map, Value};

pub const HELP: &str = "usage: fllm-rs connect TOOL [--url http://127.0.0.1:8000] [--model ID] [--api-key KEY] [--ctx N]
                         [--shell powershell|sh|cmd] [--folder DIR] [--global] [--write]
       fllm-rs connect --list
TOOL: claude-code, codex, opencode, aider, continue, cline, openai-sdk, anthropic-sdk or env.
Prints the environment variables and the config of the tool for the server at --url (default http://127.0.0.1:8000).
The model name and the context length come from the server when it runs; --model and --ctx fill in for one that is
not started yet. The key is --api-key or FLLM_API_KEY; a server without a key accepts any value, so a placeholder is used.
Nothing is changed on disk unless you add --write. Without it the command shows what --write would do as a diff; with it the
file is created or updated, and the old content of a file that changes is kept next to it as FILE.fllm-backup.
--folder: the project folder for project-level files (default: the current folder). --global: use the user-level file of the
tool where it has one (claude-code, codex, opencode, continue). --shell: the syntax of the environment lines (default: the shell of this system).";

/// The tools that have a setup page, with the one-line description of `--list`.
pub const TOOLS: [(&str, &str); 8] = [
    ("claude-code", "Claude Code (Anthropic Messages API): environment variables or .claude/settings.local.json"),
    ("codex", "Codex CLI (OpenAI Responses API): a provider and a profile in config.toml"),
    ("opencode", "OpenCode (OpenAI-compatible): opencode.json"),
    ("aider", "Aider (OpenAI-compatible): .aider.conf.yml and a model metadata file"),
    ("continue", "Continue (VS Code, JetBrains): config.yaml with a chat and an autocomplete model"),
    ("cline", "Cline, Roo Code, Kilo Code (VS Code extensions): the three fields to fill in"),
    ("openai-sdk", "OpenAI Python/JS SDK and anything that reads OPENAI_BASE_URL"),
    ("anthropic-sdk", "Anthropic Python/JS SDK and anything that reads ANTHROPIC_BASE_URL"),
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shell {
    Powershell,
    Sh,
    Cmd,
}

impl Shell {
    pub fn system_default() -> Self {
        if cfg!(windows) {
            Shell::Powershell
        } else {
            Shell::Sh
        }
    }

    fn parse(text: &str) -> Result<Self, String> {
        match text.to_ascii_lowercase().as_str() {
            "powershell" | "pwsh" | "ps" => Ok(Shell::Powershell),
            "sh" | "bash" | "zsh" | "fish" => Ok(Shell::Sh),
            "cmd" => Ok(Shell::Cmd),
            other => Err(format!("--shell must be powershell, sh or cmd, not {other:?}")),
        }
    }

    /// One line that sets `name` to `value` in this shell.
    pub fn set(self, name: &str, value: &str) -> String {
        match self {
            Shell::Powershell => format!("$env:{name} = \"{}\"", value.replace('`', "``").replace('"', "`\"").replace('$', "`$")),
            Shell::Sh => format!("export {name}='{}'", value.replace('\'', "'\\''")),
            Shell::Cmd => format!("set {name}={value}"),
        }
    }
}

/// What is known about the server the tool is to talk to.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// `http://host:port` without a path.
    pub base: String,
    /// The served model name, which the tools must send as `model`.
    pub model: String,
    pub key: String,
    /// `true` when the key came from the user (a placeholder is not worth a "set your key" note).
    pub key_given: bool,
    /// The context length of the server, when it is known.
    pub context: Option<u64>,
    /// `true` when the values come from the running server.
    pub live: bool,
}

impl Settings {
    fn openai_url(&self) -> String {
        format!("{}/v1", self.base)
    }

    /// A good ceiling for one reply: a quarter of the context, between 2,048 and 16,384 tokens.
    fn output_limit(&self) -> u64 {
        self.context.map_or(8192, |c| (c / 4).clamp(2048, 16384))
    }
}

/// Everything a tool needs: environment variables, files to create or change, and the commands that start it.
#[derive(Default)]
pub struct Plan {
    pub env: Vec<(String, String)>,
    pub files: Vec<FileEdit>,
    pub notes: Vec<String>,
    pub start: Vec<String>,
    pub snippet: Option<(String, String)>,
}

type Apply = Box<dyn Fn(&str) -> Result<String, String>>;

/// A file the setup would create or change: `apply` maps the current content (empty when the file is missing) to the new one.
pub struct FileEdit {
    pub path: PathBuf,
    pub apply: Apply,
}

#[derive(Debug, Clone)]
pub struct Targets {
    pub folder: PathBuf,
    pub home: PathBuf,
    pub global: bool,
}

fn json_edit(update: impl Fn(&mut Map<String, Value>) + 'static) -> Apply {
    Box::new(move |current| {
        let mut root = if current.trim().is_empty() {
            Value::Object(Map::new())
        } else {
            serde_json::from_str::<Value>(current).map_err(|e| format!("it is not plain JSON ({e}); add the settings by hand"))?
        };
        let object = root.as_object_mut().ok_or("it does not hold a JSON object; add the settings by hand")?;
        update(object);
        serde_json::to_string_pretty(&root).map(|text| text + "\n").map_err(|e| e.to_string())
    })
}

fn object_entry<'a>(object: &'a mut Map<String, Value>, key: &str) -> &'a mut Map<String, Value> {
    let slot = object.entry(key.to_owned()).or_insert_with(|| Value::Object(Map::new()));
    if !slot.is_object() {
        *slot = Value::Object(Map::new());
    }
    slot.as_object_mut().expect("just made an object")
}

const BLOCK_BEGIN: &str = "# >>> fllm connect (written by `fllm-rs connect codex`) >>>";
const BLOCK_END: &str = "# <<< fllm connect <<<";

/// `current` with the block between the fLLM markers replaced by `block` (appended when there is none).
fn replace_block(current: &str, block: &str) -> String {
    let mut kept = current.to_owned();
    if let (Some(begin), Some(end)) = (current.find(BLOCK_BEGIN), current.find(BLOCK_END)) {
        if begin < end {
            let mut after = end + BLOCK_END.len();
            if current[after..].starts_with("\r\n") {
                after += 2;
            } else if current[after..].starts_with('\n') {
                after += 1;
            }
            kept = format!("{}{}", &current[..begin], &current[after..]);
        }
    }
    let kept = kept.trim_end().to_owned();
    if kept.is_empty() {
        format!("{BLOCK_BEGIN}\n{block}{BLOCK_END}\n")
    } else {
        format!("{kept}\n\n{BLOCK_BEGIN}\n{block}{BLOCK_END}\n")
    }
}

/// `key: value` lines of a flat YAML file: an existing line of the key is replaced, a missing one is appended.
fn set_yaml_keys(current: &str, pairs: &[(&str, String)]) -> String {
    let mut lines: Vec<String> = current.lines().map(str::to_owned).collect();
    for (key, value) in pairs {
        let prefix = format!("{key}:");
        let line = format!("{key}: {value}");
        match lines.iter().position(|l| l.starts_with(&prefix)) {
            Some(at) => lines[at] = line,
            None => lines.push(line),
        }
    }
    lines.join("\n") + "\n"
}

fn toml_string(text: &str) -> String {
    Value::String(text.to_owned()).to_string()
}

/// The setup of one tool for the server described by `s`.
pub fn plan(tool: &str, s: &Settings, t: &Targets) -> Result<Plan, String> {
    let mut plan = Plan::default();
    match tool {
        "claude-code" => {
            let limit = s.output_limit().to_string();
            plan.env = vec![
                ("ANTHROPIC_BASE_URL".into(), s.base.clone()),
                ("ANTHROPIC_AUTH_TOKEN".into(), s.key.clone()),
                ("ANTHROPIC_MODEL".into(), s.model.clone()),
                ("ANTHROPIC_DEFAULT_OPUS_MODEL".into(), s.model.clone()),
                ("ANTHROPIC_DEFAULT_SONNET_MODEL".into(), s.model.clone()),
                ("ANTHROPIC_DEFAULT_HAIKU_MODEL".into(), s.model.clone()),
                ("CLAUDE_CODE_SUBAGENT_MODEL".into(), s.model.clone()),
                ("CLAUDE_CODE_MAX_OUTPUT_TOKENS".into(), limit),
                ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".into(), "1".into()),
            ];
            let path = if t.global { t.home.join(".claude").join("settings.json") } else { t.folder.join(".claude").join("settings.local.json") };
            let env = plan.env.clone();
            plan.files.push(FileEdit {
                path,
                apply: json_edit(move |root| {
                    let slot = object_entry(root, "env");
                    for (name, value) in &env {
                        slot.insert(name.clone(), json!(value));
                    }
                }),
            });
            plan.start.push("claude".into());
            plan.notes.push("Claude Code sends the tool list and instructions of a full coding session (roughly 15,000 to 25,000 tokens), so start the server with --ctx 32768 or more.".into());
            plan.notes.push("Requests for any claude-* model name that the server does not serve are answered by the loaded model.".into());
            plan.notes.push("A key in the file is stored in clear text; the environment lines keep it out of the file.".into());
        }
        "codex" => {
            plan.env = vec![("FLLM_API_KEY".into(), s.key.clone())];
            let window = s.context.map(|c| format!("model_context_window = {c}\nmodel_auto_compact_token_limit = {}\n", c * 8 / 10)).unwrap_or_default();
            let block = format!(
                "[model_providers.fllm]\nname = \"fLLM\"\nbase_url = {}\nwire_api = \"responses\"\nenv_key = \"FLLM_API_KEY\"\n\n[profiles.fllm]\nmodel = {}\nmodel_provider = \"fllm\"\n{window}",
                toml_string(&s.openai_url()),
                toml_string(&s.model),
            );
            let home = std::env::var_os("CODEX_HOME").map(PathBuf::from).unwrap_or_else(|| t.home.join(".codex"));
            plan.files.push(FileEdit {
                path: home.join("config.toml"),
                apply: Box::new(move |current| {
                    let outside = match (current.find(BLOCK_BEGIN), current.find(BLOCK_END)) {
                        (Some(a), Some(b)) if a < b => format!("{}{}", &current[..a], &current[b..]),
                        _ => current.to_owned(),
                    };
                    for table in ["[model_providers.fllm]", "[profiles.fllm]"] {
                        if outside.contains(table) {
                            return Err(format!("it already defines {table} outside the block of this command; remove it or edit it by hand"));
                        }
                    }
                    Ok(replace_block(current, &block))
                }),
            });
            plan.start.push("codex --profile fllm".into());
            plan.start.push("codex exec --profile fllm \"explain this repository\"".into());
            plan.notes.push("The profile leaves your default provider alone. Codex warns that it has no metadata for the model name; that is harmless.".into());
            plan.notes.push("Codex sends about 6,000 tokens of instructions and tool descriptions with every request; the first turn is the slowest, later turns reuse the cached prefix.".into());
        }
        "opencode" => {
            let model = s.model.clone();
            let url = s.openai_url();
            let key = s.key.clone();
            let context = s.context;
            let output = s.output_limit();
            let project = !t.global;
            let path = if t.global { t.home.join(".config").join("opencode").join("opencode.json") } else { t.folder.join("opencode.json") };
            plan.files.push(FileEdit {
                path,
                apply: json_edit(move |root| {
                    root.entry("$schema".to_owned()).or_insert_with(|| json!("https://opencode.ai/config.json"));
                    let providers = object_entry(root, "provider");
                    let mut limit = json!({"output": output});
                    if let Some(context) = context {
                        limit["context"] = json!(context);
                    }
                    providers.insert(
                        "fllm".into(),
                        json!({
                            "npm": "@ai-sdk/openai-compatible",
                            "name": "fLLM",
                            "options": {"baseURL": url, "apiKey": key},
                            "models": {model.clone(): {"name": model, "limit": limit, "tool_call": true}},
                        }),
                    );
                    if project {
                        root.insert("model".into(), json!(format!("fllm/{model}")));
                    }
                }),
            });
            plan.start.push(format!("opencode --model fllm/{}", s.model));
            plan.notes.push("With --global the default model of OpenCode is left alone; pick fllm/MODEL with --model or /models.".into());
        }
        "aider" => {
            plan.env = vec![("OPENAI_API_BASE".into(), s.openai_url()), ("OPENAI_API_KEY".into(), s.key.clone())];
            let pairs = vec![("openai-api-base", s.openai_url()), ("openai-api-key", s.key.clone()), ("model", format!("openai/{}", s.model))];
            plan.files.push(FileEdit { path: t.folder.join(".aider.conf.yml"), apply: Box::new(move |current| Ok(set_yaml_keys(current, &pairs))) });
            let name = format!("openai/{}", s.model);
            let mut meta = json!({"max_output_tokens": s.output_limit(), "input_cost_per_token": 0, "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"});
            if let Some(context) = s.context {
                meta["max_input_tokens"] = json!(context);
                meta["max_tokens"] = json!(context);
            }
            plan.files.push(FileEdit {
                path: t.folder.join(".aider.model.metadata.json"),
                apply: json_edit(move |root| {
                    root.insert(name.clone(), meta.clone());
                }),
            });
            plan.start.push(format!("aider --model openai/{} --model-metadata-file .aider.model.metadata.json", s.model));
            plan.notes.push("The metadata file tells Aider the context window of the model, so it neither warns nor assumes a size.".into());
        }
        "continue" => {
            let context = s.context.map(|c| format!("    defaultCompletionOptions:\n      contextLength: {c}\n")).unwrap_or_default();
            let yaml = format!(
                "name: fLLM\nversion: 1.0.0\nschema: v1\nmodels:\n  - name: fLLM chat\n    provider: openai\n    model: {model}\n    apiBase: {url}\n    apiKey: {key}\n    roles:\n      - chat\n      - edit\n      - apply\n{context}  - name: fLLM autocomplete\n    provider: llama.cpp\n    model: {model}\n    apiBase: {base}\n    roles:\n      - autocomplete\n",
                model = s.model,
                url = s.openai_url(),
                base = s.base,
                key = s.key,
            );
            plan.snippet = Some((if t.global { "~/.continue/config.yaml".into() } else { "config.yaml".into() }, yaml.clone()));
            plan.files.push(FileEdit {
                path: t.home.join(".continue").join("config.yaml"),
                apply: Box::new(move |current| {
                    if current.trim().is_empty() {
                        Ok(yaml.clone())
                    } else {
                        Err("it exists already; Continue config files are not merged by this command, copy the models from the snippet above".into())
                    }
                }),
            });
            plan.notes.push("Autocomplete uses the llama.cpp provider, which calls POST /infill, and needs a model with fill-in-the-middle tokens (Qwen2.5-Coder, StarCoder2, CodeLlama, DeepSeek-Coder, Codestral); chat and edit use the OpenAI provider.".into());
        }
        "cline" => {
            plan.notes.push(format!(
                "Cline, Roo Code and Kilo Code: open the settings of the extension, choose the provider \"OpenAI Compatible\" and fill in\n    Base URL   {}\n    API Key    {}\n    Model ID   {}\n    Context window  {}\n    Max output tokens  {}\n  Turn off \"Enable streaming\" only if a proxy in between breaks event streams.",
                s.openai_url(),
                s.key,
                s.model,
                s.context.map_or("the --ctx of the server".to_owned(), |c| c.to_string()),
                s.output_limit(),
            ));
        }
        "openai-sdk" => {
            plan.env = vec![("OPENAI_BASE_URL".into(), s.openai_url()), ("OPENAI_API_KEY".into(), s.key.clone())];
            plan.snippet = Some((
                "Python".into(),
                format!(
                    "from openai import OpenAI\nclient = OpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY\nreply = client.chat.completions.create(model=\"{m}\", messages=[{{\"role\": \"user\", \"content\": \"Hello\"}}])\nprint(reply.choices[0].message.content)\n# the Responses API works too: client.responses.create(model=\"{m}\", input=\"Hello\")\n",
                    m = s.model
                ),
            ));
        }
        "anthropic-sdk" => {
            plan.env = vec![("ANTHROPIC_BASE_URL".into(), s.base.clone()), ("ANTHROPIC_API_KEY".into(), s.key.clone())];
            plan.snippet = Some((
                "Python".into(),
                format!(
                    "import anthropic\nclient = anthropic.Anthropic()  # reads ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY\nmessage = client.messages.create(model=\"{m}\", max_tokens=256, messages=[{{\"role\": \"user\", \"content\": \"Hello\"}}])\nprint(message.content[0].text)\n",
                    m = s.model
                ),
            ));
            plan.notes.push("The Anthropic SDK takes the address without /v1.".into());
        }
        "env" => {
            plan.env = vec![
                ("OPENAI_BASE_URL".into(), s.openai_url()),
                ("OPENAI_API_BASE".into(), s.openai_url()),
                ("OPENAI_API_KEY".into(), s.key.clone()),
                ("ANTHROPIC_BASE_URL".into(), s.base.clone()),
                ("ANTHROPIC_API_KEY".into(), s.key.clone()),
            ];
        }
        other => return Err(format!("unknown tool {other:?}; `fllm-rs connect --list` names them")),
    }
    Ok(plan)
}

/// A unified diff of two texts, line by line (the files involved are small, so a plain longest-common-subsequence is enough).
pub fn diff(old: &str, new: &str) -> String {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    let mut table = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            table[i][j] = if a[i] == b[j] { table[i + 1][j + 1] + 1 } else { table[i + 1][j].max(table[i][j + 1]) };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out = String::new();
    while i < a.len() || j < b.len() {
        if i < a.len() && j < b.len() && a[i] == b[j] {
            out.push_str(&format!("  {}\n", a[i]));
            i += 1;
            j += 1;
        } else if i < a.len() && (j == b.len() || table[i + 1][j] >= table[i][j + 1]) {
            out.push_str(&format!("- {}\n", a[i]));
            i += 1;
        } else {
            out.push_str(&format!("+ {}\n", b[j]));
            j += 1;
        }
    }
    out
}

/// The diff with unchanged lines cut down to three lines of context around each change.
fn short_diff(old: &str, new: &str) -> String {
    let lines: Vec<String> = diff(old, new).lines().map(str::to_owned).collect();
    let changed: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| !l.starts_with("  ")).map(|(i, _)| i).collect();
    if changed.is_empty() {
        return String::new();
    }
    let mut keep = vec![false; lines.len()];
    for at in changed {
        for slot in keep.iter_mut().take((at + 4).min(lines.len())).skip(at.saturating_sub(3)) {
            *slot = true;
        }
    }
    let mut out = String::new();
    let mut gap = false;
    for (line, keep) in lines.iter().zip(&keep) {
        if *keep {
            out.push_str(line);
            out.push('\n');
            gap = false;
        } else if !gap {
            out.push_str("  ...\n");
            gap = true;
        }
    }
    out
}

fn hide_key(text: &str, key: &str, key_given: bool) -> String {
    if key_given && !key.is_empty() {
        text.replace(key, "<your key>")
    } else {
        text.to_owned()
    }
}

#[derive(Debug, Default)]
struct Options {
    tool: String,
    url: String,
    model: Option<String>,
    key: Option<String>,
    ctx: Option<u64>,
    shell: Option<Shell>,
    folder: Option<PathBuf>,
    global: bool,
    write: bool,
    list: bool,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options { url: "http://127.0.0.1:8000".into(), key: std::env::var("FLLM_API_KEY").ok().filter(|k| !k.is_empty()), ..Options::default() };
    let mut i = 0;
    let value = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("missing value for {flag}"))
    };
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--help" | "-h" => return Err(HELP.to_owned()),
            "--list" => o.list = true,
            "--url" => o.url = value(&mut i, flag)?,
            "--model" => o.model = Some(value(&mut i, flag)?),
            "--api-key" => o.key = Some(value(&mut i, flag)?),
            "--ctx" => o.ctx = Some(value(&mut i, flag)?.parse().map_err(|_| "--ctx needs a number of tokens".to_owned())?),
            "--shell" => o.shell = Some(Shell::parse(&value(&mut i, flag)?)?),
            "--folder" => o.folder = Some(PathBuf::from(value(&mut i, flag)?)),
            "--global" => o.global = true,
            "--write" => o.write = true,
            other if other.starts_with('-') => return Err(format!("unknown option {other}\n{HELP}")),
            other => {
                if !o.tool.is_empty() {
                    return Err(format!("one tool at a time, got {:?} and {other:?}", o.tool));
                }
                o.tool = other.to_owned();
            }
        }
        i += 1;
    }
    if !o.list && o.tool.is_empty() {
        return Err(HELP.to_owned());
    }
    Ok(o)
}

/// Asks the server for the served model and its context. `Err` carries the reason it could not be asked.
fn ask_server(url: &str, key: Option<&str>) -> Result<(String, Option<u64>), String> {
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(10))
        .user_agent("fllm-connect/0.1")
        .build()
        .map_err(|e| e.to_string())?;
    let mut request = client.get(format!("{url}/v1/models"));
    if let Some(key) = key {
        request = request.bearer_auth(key);
    }
    let response = request.send().map_err(|e| format!("cannot reach {url}: {e}"))?;
    let status = response.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(format!("{url} wants an API key (use --api-key or FLLM_API_KEY)"));
    }
    if !status.is_success() {
        return Err(format!("{url}/v1/models answered HTTP {status}"));
    }
    let body: Value = response.json().map_err(|e| e.to_string())?;
    let first = &body["data"][0];
    let id = first["id"].as_str().ok_or("the server lists no model; load one first")?;
    Ok((id.to_owned(), first["max_model_len"].as_u64()))
}

fn home_dir() -> Result<PathBuf, String> {
    ["USERPROFILE", "HOME"]
        .iter()
        .find_map(|name| std::env::var_os(name).filter(|v| !v.is_empty()))
        .map(PathBuf::from)
        .ok_or_else(|| "cannot find the home folder".to_owned())
}

fn show_path(path: &Path) -> String {
    path.display().to_string()
}

pub fn run(args: &[String]) -> Result<i32, String> {
    let o = match parse(args) {
        Ok(o) => o,
        Err(text) if text == HELP => {
            println!("{HELP}");
            return Ok(0);
        }
        Err(text) => return Err(text),
    };
    if o.list {
        for (name, what) in TOOLS {
            println!("{name:<14}{what}");
        }
        return Ok(0);
    }
    let base = o.url.trim_end_matches('/').trim_end_matches("/v1").to_owned();
    if !(base.starts_with("http://") || base.starts_with("https://")) {
        return Err("--url must begin with http:// or https://".into());
    }
    if o.tool != "env" && !TOOLS.iter().any(|(name, _)| *name == o.tool) {
        return Err(format!("unknown tool {:?}; `fllm-rs connect --list` names them", o.tool));
    }
    let asked = ask_server(&base, o.key.as_deref());
    let mut notes: Vec<String> = Vec::new();
    let (served, served_ctx, live) = match asked {
        Ok((id, ctx)) => (Some(id), ctx, true),
        Err(why) => {
            notes.push(format!("The server was not asked: {why}. The values below come from the flags; start it with `fllm-rs serve --model MODEL --model-name NAME` and run this command again to be sure."));
            (None, None, false)
        }
    };
    let model = o.model.clone().or(served).unwrap_or_else(|| "fllm".to_owned());
    if live && o.model.as_deref().is_some_and(|m| m != model) {
        notes.push("--model differs from the model the server reports; the server must serve (or alias) that name.".into());
    }
    let settings = Settings {
        base,
        model,
        key: o.key.clone().unwrap_or_else(|| "fllm".to_owned()),
        key_given: o.key.is_some(),
        context: o.ctx.or(served_ctx),
        live,
    };
    let folder = match &o.folder {
        Some(f) => f.clone(),
        None => std::env::current_dir().map_err(|e| e.to_string())?,
    };
    let targets = Targets { folder, home: home_dir()?, global: o.global };
    let plan = plan(&o.tool, &settings, &targets)?;
    let shell = o.shell.unwrap_or_else(Shell::system_default);

    println!("fLLM at {} serving {}{}", settings.base, settings.model, settings.context.map_or(String::new(), |c| format!(" (context {c} tokens)")));
    for note in notes.iter().chain(&plan.notes) {
        println!("note: {note}");
    }
    if !plan.env.is_empty() {
        println!("\nEnvironment ({shell:?} syntax):");
        for (name, value) in &plan.env {
            println!("  {}", hide_key(&shell.set(name, value), &settings.key, settings.key_given));
        }
    }
    if let Some((title, text)) = &plan.snippet {
        println!("\n{title}:");
        for line in hide_key(text, &settings.key, settings.key_given).lines() {
            println!("  {line}");
        }
    }
    let mut failed = false;
    for file in &plan.files {
        let current = std::fs::read_to_string(&file.path).unwrap_or_default();
        let exists = file.path.exists();
        println!("\n{} {}", if exists { "File" } else { "New file" }, show_path(&file.path));
        let new = match (file.apply)(&current) {
            Ok(new) => new,
            Err(why) => {
                println!("  not touched: {why}");
                failed |= o.write;
                continue;
            }
        };
        if new == current {
            println!("  already up to date");
            continue;
        }
        let text = if exists { short_diff(&current, &new) } else { new.lines().map(|l| format!("+ {l}\n")).collect() };
        for line in hide_key(&text, &settings.key, settings.key_given).lines() {
            println!("  {line}");
        }
        if o.write {
            if let Some(parent) = file.path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
            }
            if exists {
                let mut backup = file.path.clone().into_os_string();
                backup.push(".fllm-backup");
                std::fs::write(&backup, &current).map_err(|e| format!("cannot write the backup of {}: {e}", file.path.display()))?;
            }
            std::fs::write(&file.path, &new).map_err(|e| format!("cannot write {}: {e}", file.path.display()))?;
            println!("  written{}", if exists { " (the old content is in the .fllm-backup file next to it)" } else { "" });
        } else {
            println!("  (not written; add --write)");
        }
    }
    if !plan.start.is_empty() {
        println!("\nStart:");
        for line in &plan.start {
            println!("  {line}");
        }
    }
    Ok(i32::from(failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> Settings {
        Settings { base: "http://127.0.0.1:8000".into(), model: "coder".into(), key: "k1".into(), key_given: true, context: Some(32768), live: true }
    }

    fn targets() -> Targets {
        Targets { folder: PathBuf::from("proj"), home: PathBuf::from("home"), global: false }
    }

    fn edit(plan: &Plan, index: usize, current: &str) -> Result<String, String> {
        (plan.files[index].apply)(current)
    }

    #[test]
    fn every_listed_tool_has_a_plan() {
        for (name, _) in TOOLS {
            plan(name, &settings(), &targets()).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        assert!(plan("nope", &settings(), &targets()).is_err());
    }

    #[test]
    fn claude_code_gets_the_anthropic_environment_and_a_merged_settings_file() {
        let p = plan("claude-code", &settings(), &targets()).unwrap();
        let get = |name: &str| p.env.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str());
        assert_eq!(get("ANTHROPIC_BASE_URL"), Some("http://127.0.0.1:8000"));
        assert_eq!(get("ANTHROPIC_MODEL"), Some("coder"));
        assert_eq!(get("ANTHROPIC_DEFAULT_HAIKU_MODEL"), Some("coder"));
        assert_eq!(get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"), Some("8192"));
        let merged = edit(&p, 0, "{\"permissions\":{\"allow\":[\"Bash(ls)\"]},\"env\":{\"KEEP\":\"1\"}}").unwrap();
        let value: Value = serde_json::from_str(&merged).unwrap();
        assert_eq!(value["permissions"]["allow"][0], "Bash(ls)");
        assert_eq!(value["env"]["KEEP"], "1");
        assert_eq!(value["env"]["ANTHROPIC_AUTH_TOKEN"], "k1");
        assert!(p.files[0].path.ends_with("settings.local.json"));
        assert!(edit(&p, 0, "{ not json").is_err());
        assert!(plan("claude-code", &settings(), &Targets { global: true, ..targets() }).unwrap().files[0].path.ends_with("settings.json"));
    }

    #[test]
    fn codex_block_is_replaced_not_duplicated_and_foreign_tables_are_respected() {
        let p = plan("codex", &settings(), &targets()).unwrap();
        let once = edit(&p, 0, "model = \"x\"\n\n[tui]\nnotifications = true\n").unwrap();
        assert!(once.starts_with("model = \"x\""));
        assert!(once.contains("[model_providers.fllm]") && once.contains("wire_api = \"responses\""));
        assert!(once.contains("base_url = \"http://127.0.0.1:8000/v1\""));
        assert!(once.contains("model_context_window = 32768"));
        let twice = edit(&p, 0, &once).unwrap();
        assert_eq!(once, twice, "running it again changes nothing");
        assert_eq!(twice.matches("[profiles.fllm]").count(), 1);
        let err = edit(&p, 0, "[profiles.fllm]\nmodel = \"mine\"\n").unwrap_err();
        assert!(err.contains("already defines"));
    }

    #[test]
    fn opencode_config_keeps_other_providers() {
        let p = plan("opencode", &settings(), &targets()).unwrap();
        let merged = edit(&p, 0, "{\"provider\":{\"other\":{\"name\":\"o\"}},\"theme\":\"dark\"}").unwrap();
        let v: Value = serde_json::from_str(&merged).unwrap();
        assert_eq!(v["provider"]["other"]["name"], "o");
        assert_eq!(v["theme"], "dark");
        assert_eq!(v["provider"]["fllm"]["options"]["baseURL"], "http://127.0.0.1:8000/v1");
        assert_eq!(v["provider"]["fllm"]["models"]["coder"]["limit"]["context"], 32768);
        assert_eq!(v["model"], "fllm/coder");
        let global = plan("opencode", &settings(), &Targets { global: true, ..targets() }).unwrap();
        let v: Value = serde_json::from_str(&edit(&global, 0, "").unwrap()).unwrap();
        assert!(v.get("model").is_none(), "a global file does not change the default model");
    }

    #[test]
    fn aider_yaml_keys_are_replaced_in_place() {
        let p = plan("aider", &settings(), &targets()).unwrap();
        let out = edit(&p, 0, "dark-mode: true\nmodel: gpt-4o\n").unwrap();
        assert!(out.contains("dark-mode: true"));
        assert!(out.contains("model: openai/coder"));
        assert!(!out.contains("gpt-4o"));
        assert!(out.contains("openai-api-base: http://127.0.0.1:8000/v1"));
        let meta: Value = serde_json::from_str(&edit(&p, 1, "").unwrap()).unwrap();
        assert_eq!(meta["openai/coder"]["max_input_tokens"], 32768);
    }

    #[test]
    fn continue_config_is_created_but_never_merged() {
        let p = plan("continue", &settings(), &targets()).unwrap();
        let fresh = edit(&p, 0, "").unwrap();
        assert!(fresh.contains("provider: llama.cpp") && fresh.contains("contextLength: 32768"));
        assert!(edit(&p, 0, "name: mine\n").is_err());
    }

    #[test]
    fn shells_quote_their_values() {
        assert_eq!(Shell::Sh.set("A", "it's"), "export A='it'\\''s'");
        assert_eq!(Shell::Powershell.set("A", "a\"b$c"), "$env:A = \"a`\"b`$c\"");
        assert_eq!(Shell::Cmd.set("A", "b"), "set A=b");
        assert!(Shell::parse("tcsh").is_err());
    }

    #[test]
    fn the_diff_marks_added_and_removed_lines_and_keeps_the_key_out() {
        let d = diff("a\nb\nc\n", "a\nx\nc\n");
        assert_eq!(d, "  a\n- b\n+ x\n  c\n");
        assert_eq!(short_diff("same\n", "same\n"), "");
        assert_eq!(hide_key("token abc", "abc", true), "token <your key>");
        assert_eq!(hide_key("token fllm", "fllm", false), "token fllm");
    }

    #[test]
    fn arguments_are_parsed_and_misuse_is_refused() {
        let args = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        let o = parse(&args(&["codex", "--url", "http://h:1", "--ctx", "4096", "--write", "--shell", "cmd"])).unwrap();
        assert_eq!((o.tool.as_str(), o.ctx, o.write, o.shell), ("codex", Some(4096), true, Some(Shell::Cmd)));
        assert!(parse(&args(&["codex", "aider"])).is_err());
        assert!(parse(&args(&["codex", "--bogus"])).is_err());
        assert!(parse(&args(&["codex", "--ctx", "many"])).is_err());
        assert!(parse(&args(&[])).is_err());
        assert!(parse(&args(&["--list"])).unwrap().list);
    }
}
