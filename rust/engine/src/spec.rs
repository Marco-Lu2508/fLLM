//! Spec-driven development: a feature is planned in three files before any code is written, then built one task at a time
//! (the workflow of Kiro's specs).
//!
//! ```text
//! .fllm/specs/<name>/requirements.md   what must be true when it is done (user stories, acceptance criteria)
//! .fllm/specs/<name>/design.md         how it fits the code (files, data, interfaces, tests)
//! .fllm/specs/<name>/tasks.md          a checklist: `- [ ] 1. Add the parser` ...
//! ```
//!
//! `fllm-rs spec new NAME "what you want"` asks the agent for the three files in turn (reading the code, no edits; the runner writes
//! the files, so a small model only has to answer with a document). `fllm-rs spec run NAME` then does the unchecked tasks one by one,
//! each in its own agent run that sees the requirements, the design and the whole list; the runner ticks the box when the run
//! finished (and the `--verify` command passes), never the model, so a box is never ticked for work that did not happen.
//! The Studio shows the same files in its Code side panel.
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const HELP: &str = "usage: fllm-rs spec new NAME [\"what you want\"] [--phase requirements|design|tasks] [--folder DIR] [AGENT OPTIONS]
       fllm-rs spec list [--folder DIR]
       fllm-rs spec show NAME [--folder DIR]
       fllm-rs spec run NAME [--task N | --all] [--verify COMMAND] [--allow-command] [--folder DIR] [AGENT OPTIONS]
       fllm-rs spec tick NAME N [--undo] [--folder DIR]
Plans a feature in three files under .fllm/specs/NAME/ (requirements.md, design.md, tasks.md) and builds it task by task.
new: asks the agent for the files one after the other; the files are plain Markdown, edit them before the next step. --phase makes
one file only (the earlier ones are read), so you can review in between. On a terminal each file is shown and you are asked before the next.
run: does the next unchecked task (--task N a given one, --all every unchecked one, stopping at the first that fails). The task's agent
may edit files (and run commands with --allow-command); with --verify COMMAND the box is only ticked when the command succeeds.
AGENT OPTIONS are passed on to `fllm-rs agent`: --url --api-key --model --max-steps --max-tokens --temperature --profile --allow-network --allow-memory.";

const DOC_LIMIT: usize = 24_000;
const CONTEXT_LIMIT: usize = 9_000;

pub fn specs_dir(root: &Path) -> PathBuf {
    root.join(".fllm").join("specs")
}

/// A spec name is a short slug.
pub fn slug(name: &str) -> Result<String, String> {
    let s: String = name
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if s.is_empty() || s.len() > 48 {
        return Err("the spec name must be 1 to 48 letters, digits or dashes".into());
    }
    Ok(s)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    /// 1-based position among all checkbox lines of the file.
    pub index: usize,
    pub text: String,
    pub done: bool,
    /// Nesting level (0 for a top-level task, one per two spaces or one tab of indentation).
    pub depth: usize,
    /// 0-based line in the file.
    pub line: usize,
}

fn checkbox(line: &str) -> Option<(usize, bool, &str)> {
    let indent_width: usize = line.chars().take_while(|c| *c == ' ' || *c == '\t').map(|c| if c == '\t' { 2 } else { 1 }).sum();
    let rest = line.trim_start();
    let rest = rest.strip_prefix("- ").or_else(|| rest.strip_prefix("* "))?;
    let (done, text) = match rest.strip_prefix("[ ]") {
        Some(t) => (false, t),
        None => (true, rest.strip_prefix("[x]").or_else(|| rest.strip_prefix("[X]"))?),
    };
    // `[ ]x` is not a checkbox
    if !(text.is_empty() || text.starts_with(' ')) {
        return None;
    }
    Some((indent_width / 2, done, text.trim()))
}

pub fn parse_tasks(markdown: &str) -> Vec<Task> {
    markdown
        .lines()
        .enumerate()
        .filter_map(|(line, l)| checkbox(l).map(|(depth, done, text)| (line, depth, done, text.to_owned())))
        .enumerate()
        .map(|(i, (line, depth, done, text))| Task { index: i + 1, text, done, depth, line })
        .collect()
}

/// The file with checkbox number `index` ticked (or unticked). `None` when there is no such checkbox.
pub fn set_done(markdown: &str, index: usize, done: bool) -> Option<String> {
    let task = parse_tasks(markdown).into_iter().find(|t| t.index == index)?;
    let crlf = markdown.contains("\r\n");
    let mut lines: Vec<String> = markdown.split('\n').map(|l| l.trim_end_matches('\r').to_owned()).collect();
    let line = lines.get_mut(task.line)?;
    let at = line.find('[')?;
    line.replace_range(at..at + 3, if done { "[x]" } else { "[ ]" });
    Some(lines.join(if crlf { "\r\n" } else { "\n" }))
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpecDocs {
    pub requirements: String,
    pub design: String,
    pub tasks: String,
}

pub fn read_docs(root: &Path, name: &str) -> SpecDocs {
    let dir = specs_dir(root).join(name);
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap_or_default();
    SpecDocs { requirements: read("requirements.md"), design: read("design.md"), tasks: read("tasks.md") }
}

/// Every spec of the workspace as JSON: name, which files exist, and the tasks.
pub fn list_json(root: &Path) -> Value {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(specs_dir(root)) else { return json!({"specs": out}) };
    let mut names: Vec<String> = entries.filter_map(Result::ok).filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().into_string().ok()).collect();
    names.sort();
    for name in names.into_iter().take(100) {
        let docs = read_docs(root, &name);
        let tasks = parse_tasks(&docs.tasks);
        let done = tasks.iter().filter(|t| t.done).count();
        out.push(json!({
            "name": name,
            "requirements": !docs.requirements.trim().is_empty(),
            "design": !docs.design.trim().is_empty(),
            "tasks_file": !docs.tasks.trim().is_empty(),
            "total": tasks.len(),
            "done": done,
            "tasks": tasks.iter().map(|t| json!({"index": t.index, "text": t.text, "done": t.done, "depth": t.depth})).collect::<Vec<_>>(),
        }));
    }
    json!({"specs": out})
}

/// Ticks one task of a spec on disk; returns the updated list entry.
pub fn tick(root: &Path, name: &str, index: usize, done: bool) -> Result<(), String> {
    let name = slug(name)?;
    let file = specs_dir(root).join(&name).join("tasks.md");
    let text = std::fs::read_to_string(&file).map_err(|_| format!("spec '{name}' has no tasks.md"))?;
    let updated = set_done(&text, index, done).ok_or_else(|| format!("tasks.md of '{name}' has no task {index}"))?;
    std::fs::write(&file, updated).map_err(|e| format!("cannot write {}: {e}", file.display()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Requirements,
    Design,
    Tasks,
}

impl Phase {
    pub fn parse(text: &str) -> Option<Phase> {
        match text {
            "requirements" | "req" => Some(Phase::Requirements),
            "design" => Some(Phase::Design),
            "tasks" | "task" => Some(Phase::Tasks),
            _ => None,
        }
    }
    pub fn file(self) -> &'static str {
        match self {
            Phase::Requirements => "requirements.md",
            Phase::Design => "design.md",
            Phase::Tasks => "tasks.md",
        }
    }
}

fn clip_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_owned()
    } else {
        let head: String = text.chars().take(max).collect();
        format!("{head}\n[... cut]")
    }
}

/// The prompt that asks for one file. The answer is the file itself: a small model is much better at writing one document than at
/// calling a write tool with the right path.
pub fn phase_prompt(phase: Phase, name: &str, description: &str, docs: &SpecDocs) -> String {
    let want = description.trim();
    let mut p = format!("You are planning the feature \"{name}\" before any code is written.\nWhat the user wants: {want}\n\n");
    if phase != Phase::Requirements {
        p.push_str(&format!("Requirements (approved):\n{}\n\n", clip_chars(&docs.requirements, CONTEXT_LIMIT)));
    }
    if phase == Phase::Tasks {
        p.push_str(&format!("Design (approved):\n{}\n\n", clip_chars(&docs.design, CONTEXT_LIMIT)));
    }
    p.push_str("First look at the code with the read-only tools if that helps (list_dir, grep, read_file); do not change anything. ");
    p.push_str(match phase {
        Phase::Requirements => "Then reply with the complete requirements.md and nothing else. Structure: a first line `# Requirements: <title>`, a short `## Introduction`, then `## Requirements` with numbered requirements. Each requirement has a user story (`As a <role>, I want <goal>, so that <benefit>`) and `#### Acceptance criteria` as a list written like `WHEN <situation> THEN the system SHALL <behaviour>`. Cover the normal case, the error cases and the limits. Do not describe the implementation.",
        Phase::Design => "Then reply with the complete design.md and nothing else. Structure: `# Design: <title>`, `## Overview`, `## Files and components` (which existing files change, which new ones appear, one line each, with real paths you found), `## Data and interfaces` (types, functions, formats), `## Error handling`, `## Testing` (which tests prove each requirement). Keep to what the requirements need.",
        Phase::Tasks => "Then reply with the complete tasks.md and nothing else. Structure: `# Tasks: <title>` and then a checklist where every line is `- [ ] 1. Short imperative title` followed by indented detail lines (the files to touch and `_Requirements: 1.1, 2_`). Order the tasks so that each can be done and checked after the one before; put the test for a piece in the same task or right after it. 4 to 12 tasks, each small enough for one focused change. Every task starts with an unticked box `- [ ]`.",
    });
    p.push_str("\nYour reply is saved as the file as it is: no introduction, no closing remarks, no code fence around the whole file.");
    p
}

/// The prompt for one task of the checklist.
pub fn task_prompt(name: &str, docs: &SpecDocs, task: &Task) -> String {
    let all: Vec<Task> = parse_tasks(&docs.tasks);
    let list = all.iter().map(|t| format!("{}- [{}] {}", "  ".repeat(t.depth), if t.done { "x" } else { " " }, t.text)).collect::<Vec<_>>().join("\n");
    format!(
        "You are building the feature \"{name}\" task by task from a written spec.\n\nRequirements:\n{}\n\nDesign:\n{}\n\nAll tasks (x = done):\n{list}\n\nYour task now, and only this one: {}\n\nRead the files you need, make the change, and check it with the project's tests or a quick check when you can run commands. Do not work on other tasks and do not edit tasks.md; the box is ticked for you when this run is done. Finish with two or three lines: what you changed and how you checked it, or what stopped you.",
        clip_chars(&docs.requirements, CONTEXT_LIMIT),
        clip_chars(&docs.design, CONTEXT_LIMIT),
        task.text
    )
}

/// The same instructions for the Studio, where the agent runs in the browser: it writes the files itself and ticks the boxes.
pub fn studio_new_prompt(name: &str, description: &str) -> String {
    format!(
        "Start a spec named \"{name}\" for this feature: {}\n\nWork in phases and stop after each one so the user can review and edit the file:\n1. Look at the code, then write .fllm/specs/{name}/requirements.md: numbered requirements, each with a user story and acceptance criteria like `WHEN ... THEN the system SHALL ...`. Stop and ask the user to approve.\n2. After approval write .fllm/specs/{name}/design.md: overview, files and components with real paths, data and interfaces, error handling, testing. Stop and ask.\n3. After approval write .fllm/specs/{name}/tasks.md: a checklist of 4 to 12 small tasks, each line `- [ ] 1. Title`, ordered so each can be checked after the previous one. Stop.\nDo not change any code in these phases.",
        description.trim()
    )
}

pub fn studio_task_prompt(root: &Path, name: &str, index: usize) -> Result<String, String> {
    let docs = read_docs(root, name);
    let task = parse_tasks(&docs.tasks).into_iter().find(|t| t.index == index).ok_or_else(|| format!("spec '{name}' has no task {index}"))?;
    let mut p = task_prompt(name, &docs, &task);
    p = p.replace(
        "do not edit tasks.md; the box is ticked for you when this run is done.",
        &format!("when the task is done and checked, tick its box in .fllm/specs/{name}/tasks.md (change `[ ]` to `[x]` on that line, nothing else)."),
    );
    Ok(p)
}

/// Cuts the chatter around a document the model returned: a wrapping code fence and text before the first heading.
pub fn clean_doc(answer: &str) -> String {
    let mut text = answer.trim().to_owned();
    if let Some(rest) = text.strip_prefix("```") {
        // ```markdown\n ... \n```
        let body = rest.split_once('\n').map_or("", |(_, b)| b);
        text = body.trim_end().strip_suffix("```").unwrap_or(body).trim().to_owned();
    }
    if let Some(at) = text.find("\n# ") {
        if !text.starts_with('#') {
            text = text[at + 1..].to_owned();
        }
    }
    clip_chars(text.trim(), DOC_LIMIT)
}

/// A document is acceptable when it has the shape the later steps rely on. `Err` says what is missing, for a second try.
pub fn validate(phase: Phase, doc: &str) -> Result<(), String> {
    if doc.trim().chars().count() < 40 {
        return Err("the answer is too short to be the file".into());
    }
    if phase == Phase::Tasks {
        let tasks = parse_tasks(doc);
        if tasks.is_empty() {
            return Err("the file has no checklist; every task must be a line starting with `- [ ] `".into());
        }
        if tasks.iter().all(|t| t.done) {
            return Err("every task is already ticked; new tasks start with `- [ ]`".into());
        }
    }
    Ok(())
}

/// Runs the agent for the given arguments and returns (exit code, the JSON report of `--output-format json`).
pub type AgentRunner<'a> = dyn FnMut(&[String]) -> Result<(i32, Value), String> + 'a;

pub fn run_agent(args: &[String]) -> Result<(i32, Value), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut all: Vec<String> = vec!["agent".into()];
    all.extend_from_slice(args);
    let out = std::process::Command::new(exe).args(&all).stdin(std::process::Stdio::null()).stderr(std::process::Stdio::inherit()).output().map_err(|e| format!("cannot start the agent: {e}"))?;
    let code = out.status.code().unwrap_or(1);
    let text = String::from_utf8_lossy(&out.stdout);
    let report = serde_json::from_str::<Value>(text.trim()).unwrap_or_else(|_| json!({"answer": text.trim(), "status": "done"}));
    Ok((code, report))
}

#[derive(Debug, Default, Clone)]
struct Common {
    folder: Option<PathBuf>,
    forward: Vec<String>,
}

const FORWARD_VALUE: [&str; 8] = ["--url", "--api-key", "--model", "--max-steps", "--max-tokens", "--temperature", "--profile", "--system"];
const FORWARD_FLAG: [&str; 3] = ["--allow-network", "--allow-memory", "--quiet"];

/// Writes one phase file: asks, cleans, validates (one retry with the reason), saves.
pub fn make_phase(root: &Path, name: &str, phase: Phase, description: &str, forward: &[String], runner: &mut AgentRunner) -> Result<PathBuf, String> {
    let dir = specs_dir(root).join(name);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let docs = read_docs(root, name);
    if phase == Phase::Design && docs.requirements.trim().is_empty() {
        return Err("design.md needs requirements.md first".into());
    }
    if phase == Phase::Tasks && docs.design.trim().is_empty() {
        return Err("tasks.md needs design.md first".into());
    }
    let mut prompt = phase_prompt(phase, name, description, &docs);
    let mut last_problem = String::new();
    for attempt in 0..2 {
        // a few reading steps are enough to plan; options the user gave come later and win
        let mut args: Vec<String> = vec!["-p".into(), prompt.clone(), "--folder".into(), root.display().to_string(), "--output-format".into(), "json".into(), "--quiet".into(), "--max-steps".into(), "6".into()];
        args.extend_from_slice(forward);
        let (code, report) = runner(&args)?;
        if code != 0 && code != 3 {
            // most often the context ran full while the agent read files: one more try from the given text alone
            if attempt == 0 {
                prompt = format!("{}\n\nDo not read any files this time: write the file from the text above.", phase_prompt(phase, name, description, &docs));
                last_problem = format!("the agent ended with exit code {code}");
                continue;
            }
            return Err(format!("the agent ended with exit code {code} (a model with a larger context may be needed; the tool list alone takes about 4,000 tokens)"));
        }
        let doc = clean_doc(report["answer"].as_str().unwrap_or_default());
        match validate(phase, &doc) {
            Ok(()) => {
                let file = dir.join(phase.file());
                std::fs::write(&file, format!("{doc}\n")).map_err(|e| format!("cannot write {}: {e}", file.display()))?;
                return Ok(file);
            }
            Err(problem) => {
                last_problem = problem.clone();
                if attempt == 0 {
                    prompt = format!("{}\n\nYour last reply could not be used: {problem}. Reply again with the complete file.", phase_prompt(phase, name, description, &docs));
                }
            }
        }
    }
    Err(format!("the model did not produce a usable {}: {last_problem}", phase.file()))
}

/// Does one task: the agent run, the optional check, then the box. Returns whether the task is done.
pub fn do_task(root: &Path, name: &str, index: usize, verify: Option<&str>, allow_command: bool, forward: &[String], runner: &mut AgentRunner) -> Result<(bool, String), String> {
    let docs = read_docs(root, name);
    let task = parse_tasks(&docs.tasks).into_iter().find(|t| t.index == index).ok_or_else(|| format!("spec '{name}' has no task {index}"))?;
    let prompt = task_prompt(name, &docs, &task);
    let mut args: Vec<String> = vec!["-p".into(), prompt, "--folder".into(), root.display().to_string(), "--allow-edit".into(), "--output-format".into(), "json".into(), "--quiet".into(), "--session".into(), format!("spec-{name}-t{index}")];
    // the plan files are the user's: the task run may not rewrite them
    args.extend(["--deny-rule".into(), "edit: .fllm/specs/**".into()]);
    if allow_command || verify.is_some() {
        args.push("--allow-command".into());
    }
    if let Some(command) = verify {
        args.extend(["--review".into(), "--verify".into(), command.to_owned()]);
    }
    args.extend_from_slice(forward);
    let (code, report) = runner(&args)?;
    let answer = report["answer"].as_str().unwrap_or_default().trim().to_owned();
    // exit 4 = the --verify command still fails; 3 = out of steps; both leave the box open
    let ok = code == 0;
    if ok {
        tick(root, name, index, true)?;
    }
    Ok((ok, answer))
}

type OwnOptions = Vec<(String, Option<String>)>;

fn parse_common(args: &[String], extra_value: &[&str], extra_flag: &[&str]) -> Result<(Common, OwnOptions, Vec<String>), String> {
    let mut common = Common::default();
    let mut own: OwnOptions = Vec::new();
    let mut loose: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--folder" {
            i += 1;
            common.folder = Some(PathBuf::from(args.get(i).ok_or("missing value for --folder")?));
        } else if FORWARD_VALUE.contains(&a) {
            i += 1;
            common.forward.push(a.to_owned());
            common.forward.push(args.get(i).ok_or_else(|| format!("missing value for {a}"))?.clone());
        } else if FORWARD_FLAG.contains(&a) {
            common.forward.push(a.to_owned());
        } else if extra_value.contains(&a) {
            i += 1;
            own.push((a.to_owned(), Some(args.get(i).ok_or_else(|| format!("missing value for {a}"))?.clone())));
        } else if extra_flag.contains(&a) {
            own.push((a.to_owned(), None));
        } else if a.starts_with("--") {
            return Err(format!("unknown option {a}; see fllm-rs spec --help"));
        } else {
            loose.push(a.to_owned());
        }
        i += 1;
    }
    Ok((common, own, loose))
}

fn folder_of(common: &Common) -> Result<PathBuf, String> {
    let f = match &common.folder {
        Some(f) => f.clone(),
        None => std::env::current_dir().map_err(|e| e.to_string())?,
    };
    let f = f.canonicalize().map_err(|e| format!("the folder {} cannot be used: {e}", f.display()))?;
    Ok(crate::agent_cli::plain_path(&f))
}

fn print_progress(root: &Path, name: &str) {
    let docs = read_docs(root, name);
    let tasks = parse_tasks(&docs.tasks);
    println!("{name}: {}/{} tasks done", tasks.iter().filter(|t| t.done).count(), tasks.len());
}

pub fn run(args: &[String]) -> Result<i32, String> {
    run_with(args, &mut run_agent, &mut |q| ask_terminal(q))
}

fn ask_terminal(question: &str) -> bool {
    use std::io::{IsTerminal, Write};
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        return true;
    }
    eprint!("{question} [Y/n]: ");
    std::io::stderr().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).is_ok() && !matches!(line.trim().to_ascii_lowercase().as_str(), "n" | "no")
}

pub fn run_with(args: &[String], runner: &mut AgentRunner, confirm: &mut dyn FnMut(&str) -> bool) -> Result<i32, String> {
    let Some(sub) = args.first().map(String::as_str) else {
        println!("{HELP}");
        return Ok(0);
    };
    let rest = &args[1..];
    match sub {
        "--help" | "-h" | "help" => {
            println!("{HELP}");
            Ok(0)
        }
        "new" => {
            let (common, own, loose) = parse_common(rest, &["--phase"], &[])?;
            let root = folder_of(&common)?;
            let name = slug(loose.first().ok_or("usage: fllm-rs spec new NAME \"what you want\"")?)?;
            let phase = own.iter().find(|(k, _)| k == "--phase").and_then(|(_, v)| v.as_deref()).map(|v| Phase::parse(v).ok_or_else(|| "--phase is requirements, design or tasks".to_owned())).transpose()?;
            let mut description = loose[1..].join(" ");
            let phases: Vec<Phase> = match phase {
                Some(p) => vec![p],
                None => vec![Phase::Requirements, Phase::Design, Phase::Tasks],
            };
            if description.trim().is_empty() {
                // a later phase can work from the saved requirements
                description = read_docs(&root, &name).requirements.lines().next().unwrap_or("").trim_start_matches('#').trim().to_owned();
                if description.is_empty() {
                    return Err("describe what you want: fllm-rs spec new NAME \"what you want\"".into());
                }
            }
            for (n, p) in phases.iter().enumerate() {
                if n > 0 && !confirm(&format!("{} written. Continue with the next file?", phases[n - 1].file())) {
                    println!("Stopped. Edit the files, then run: fllm-rs spec new {name} --phase {}", match p { Phase::Design => "design", _ => "tasks" });
                    return Ok(0);
                }
                eprintln!("[spec] writing {} ...", p.file());
                let file = make_phase(&root, &name, *p, &description, &common.forward, runner)?;
                println!("{}", file.display());
            }
            if phases.contains(&Phase::Tasks) {
                print_progress(&root, &name);
                println!("Next: fllm-rs spec run {name}");
            }
            Ok(0)
        }
        "list" => {
            let (common, _, _) = parse_common(rest, &[], &[])?;
            let root = folder_of(&common)?;
            let v = list_json(&root);
            let specs = v["specs"].as_array().cloned().unwrap_or_default();
            if specs.is_empty() {
                println!("No specs yet. Start one: fllm-rs spec new NAME \"what you want\"");
            }
            for s in specs {
                println!("{}  {}/{} tasks done", s["name"].as_str().unwrap_or(""), s["done"], s["total"]);
            }
            Ok(0)
        }
        "show" => {
            let (common, _, loose) = parse_common(rest, &[], &[])?;
            let root = folder_of(&common)?;
            let name = slug(loose.first().ok_or("usage: fllm-rs spec show NAME")?)?;
            let docs = read_docs(&root, &name);
            if docs.tasks.trim().is_empty() && docs.requirements.trim().is_empty() {
                return Err(format!("there is no spec '{name}'"));
            }
            for t in parse_tasks(&docs.tasks) {
                println!("{}[{}] {}. {}", "  ".repeat(t.depth), if t.done { "x" } else { " " }, t.index, t.text);
            }
            Ok(0)
        }
        "tick" => {
            let (common, own, loose) = parse_common(rest, &[], &["--undo"])?;
            let root = folder_of(&common)?;
            let name = loose.first().ok_or("usage: fllm-rs spec tick NAME N [--undo]")?;
            let n: usize = loose.get(1).and_then(|n| n.parse().ok()).ok_or("usage: fllm-rs spec tick NAME N [--undo]")?;
            tick(&root, name, n, own.is_empty())?;
            print_progress(&root, &slug(name)?);
            Ok(0)
        }
        "run" => {
            let (common, own, loose) = parse_common(rest, &["--task", "--verify"], &["--all", "--allow-command"])?;
            let root = folder_of(&common)?;
            let name = slug(loose.first().ok_or("usage: fllm-rs spec run NAME")?)?;
            let flag = |k: &str| own.iter().any(|(a, _)| a == k);
            let value = |k: &str| own.iter().find(|(a, _)| a == k).and_then(|(_, v)| v.clone());
            let docs = read_docs(&root, &name);
            let tasks = parse_tasks(&docs.tasks);
            if tasks.is_empty() {
                return Err(format!("spec '{name}' has no tasks; create them with: fllm-rs spec new {name} --phase tasks"));
            }
            let todo: Vec<usize> = if let Some(t) = value("--task") {
                vec![t.parse().map_err(|_| "--task is a number".to_owned())?]
            } else {
                let open: Vec<usize> = tasks.iter().filter(|t| !t.done).map(|t| t.index).collect();
                if flag("--all") { open } else { open.into_iter().take(1).collect() }
            };
            if todo.is_empty() {
                println!("Every task of {name} is done.");
                return Ok(0);
            }
            let verify = value("--verify");
            for index in todo {
                eprintln!("[spec] task {index} ...");
                let (ok, answer) = do_task(&root, &name, index, verify.as_deref(), flag("--allow-command"), &common.forward, runner)?;
                println!("{}\n", answer);
                if !ok {
                    println!("Task {index} is not done (the agent did not finish or the check failed); its box stays open.");
                    print_progress(&root, &name);
                    return Ok(1);
                }
                println!("Task {index} done.");
            }
            print_progress(&root, &name);
            Ok(0)
        }
        other => Err(format!("unknown spec command '{other}'; see fllm-rs spec --help")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TASKS: &str = "# Tasks: Export\n\n- [ ] 1. Add the parser\n  - touch src/parse.rs\n  - [x] sub step done\n- [x] 2. Wire the CLI\n- [ ] 3. Test it\n  _Requirements: 1_\nNot - [ ] a task\n";

    #[test]
    fn tasks_are_the_checkbox_lines() {
        let t = parse_tasks(TASKS);
        assert_eq!(t.len(), 4);
        assert_eq!((t[0].index, t[0].text.as_str(), t[0].done, t[0].depth), (1, "1. Add the parser", false, 0));
        assert_eq!((t[1].done, t[1].depth), (true, 1));
        assert!(t[2].done && !t[3].done);
        assert!(parse_tasks("- [ ]x\n-[ ] y\n").is_empty());
    }

    #[test]
    fn ticking_changes_one_box_and_keeps_line_endings() {
        let out = set_done(TASKS, 1, true).unwrap();
        assert!(out.contains("- [x] 1. Add the parser"));
        assert_eq!(out.matches("[x]").count(), 3);
        assert!(set_done(&out, 1, false).unwrap().contains("- [ ] 1. Add"));
        assert!(set_done(TASKS, 9, true).is_none());
        let crlf = TASKS.replace('\n', "\r\n");
        let out = set_done(&crlf, 4, true).unwrap();
        assert!(out.contains("- [x] 3. Test it\r\n") && !out.contains("\r\r"));
    }

    #[test]
    fn names_documents_and_validation() {
        assert_eq!(slug("  Export to CSV! ").unwrap(), "export-to-csv");
        assert!(slug("!!!").is_err());
        assert_eq!(clean_doc("Sure! Here it is:\n# Tasks: x\n- [ ] 1. a\n"), "# Tasks: x\n- [ ] 1. a");
        assert_eq!(clean_doc("```markdown\n# Design: x\ntext\n```"), "# Design: x\ntext");
        assert!(validate(Phase::Tasks, "# Tasks: Something long enough to pass the length check\nno boxes at all here").is_err());
        assert!(validate(Phase::Tasks, "# Tasks: Something long enough to pass the length check\n- [ ] 1. a").is_ok());
        assert!(validate(Phase::Design, "short").is_err());
    }

    fn temp_root(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fllm-spec-{tag}-{}-{}", std::process::id(), crate::agent_extras::fresh_session_name(7)));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn new_writes_three_files_and_run_ticks_only_finished_tasks() {
        let root = temp_root("flow");
        let calls: std::cell::RefCell<Vec<Vec<String>>> = std::cell::RefCell::new(Vec::new());
        let mut runner = |args: &[String]| -> Result<(i32, Value), String> {
            calls.borrow_mut().push(args.to_vec());
            let prompt = &args[1];
            let answer = if prompt.contains("complete requirements.md") {
                "Here you go:\n# Requirements: Export\n\n1. As a user, I want CSV, so that I can open it.\n   WHEN I export THEN the system SHALL write a file."
            } else if prompt.contains("complete design.md") {
                "# Design: Export\n\n## Overview\nA writer module in src/export.rs and a CLI flag in main."
            } else if prompt.contains("complete tasks.md") {
                "# Tasks: Export\n\n- [ ] 1. Add the writer\n- [ ] 2. Add the flag\n"
            } else if prompt.contains("only this one: 1. Add the writer") {
                "Wrote src/export.rs."
            } else {
                return Ok((3, json!({"answer": "ran out of steps"})));
            };
            Ok((0, json!({"answer": answer, "status": "done"})))
        };
        let args: Vec<String> = ["new", "export", "CSV", "export", "--folder", root.to_str().unwrap(), "--model", "m"].iter().map(|s| (*s).to_owned()).collect();
        assert_eq!(run_with(&args, &mut runner, &mut |_| true).unwrap(), 0);
        let docs = read_docs(&root, "export");
        assert!(docs.requirements.starts_with("# Requirements: Export") && docs.design.contains("src/export.rs"));
        assert_eq!(parse_tasks(&docs.tasks).len(), 2);
        // phase prompts carry the earlier documents; the agent options are passed on
        assert!(calls.borrow()[1][1].contains("Requirements (approved)") && calls.borrow()[2][1].contains("Design (approved)"));
        assert!(calls.borrow()[0].contains(&"--model".to_owned()) && !calls.borrow()[0].contains(&"--allow-edit".to_owned()));

        // run: first task finishes and is ticked, the second run stops (exit 3) and stays open
        let run_args: Vec<String> = ["run", "export", "--all", "--folder", root.to_str().unwrap()].iter().map(|s| (*s).to_owned()).collect();
        assert_eq!(run_with(&run_args, &mut runner, &mut |_| true).unwrap(), 1);
        let tasks = parse_tasks(&read_docs(&root, "export").tasks);
        assert!(tasks[0].done && !tasks[1].done);
        let all_calls = calls.borrow();
        let last = all_calls.last().unwrap();
        assert!(last.contains(&"--allow-edit".to_owned()) && last.contains(&"edit: .fllm/specs/**".to_owned()));
        assert_eq!(list_json(&root)["specs"][0]["done"], 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn an_unusable_answer_is_asked_for_once_more_then_refused() {
        let root = temp_root("retry");
        let n = std::cell::Cell::new(0);
        let mut runner = |_: &[String]| -> Result<(i32, Value), String> {
            n.set(n.get() + 1);
            Ok((0, json!({"answer": "I cannot do that."})))
        };
        let err = make_phase(&root, "x", Phase::Requirements, "thing", &[], &mut runner).unwrap_err();
        assert!(err.contains("requirements.md"), "{err}");
        assert_eq!(n.get(), 2);
        assert!(!specs_dir(&root).join("x").join("requirements.md").exists());
        // the design needs the requirements first
        assert!(make_phase(&root, "x", Phase::Design, "thing", &[], &mut runner).unwrap_err().contains("requirements.md first"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_failed_run_is_tried_once_more_without_reading_files() {
        let root = temp_root("fail");
        let prompts = std::cell::RefCell::new(Vec::<String>::new());
        let mut runner = |args: &[String]| -> Result<(i32, Value), String> {
            prompts.borrow_mut().push(args[1].clone());
            if prompts.borrow().len() == 1 {
                return Ok((1, json!({"answer": ""})));
            }
            Ok((0, json!({"answer": "# Requirements: X\n\n1. As a user, I want it, so that it works.\n   WHEN a THEN the system SHALL b."})))
        };
        let file = make_phase(&root, "x", Phase::Requirements, "thing", &["--max-steps".into(), "3".into()], &mut runner).unwrap();
        assert!(file.exists());
        assert!(prompts.borrow()[1].contains("Do not read any files this time"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn the_studio_prompt_asks_the_agent_to_tick_and_the_cli_prompt_does_not() {
        let root = temp_root("studio");
        let dir = specs_dir(&root).join("e");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tasks.md"), TASKS).unwrap();
        let p = studio_task_prompt(&root, "e", 1).unwrap();
        assert!(p.contains("tick its box in .fllm/specs/e/tasks.md") && !p.contains("do not edit tasks.md"));
        assert!(studio_task_prompt(&root, "e", 99).is_err());
        tick(&root, "e", 1, true).unwrap();
        assert!(read_docs(&root, "e").tasks.contains("- [x] 1. Add the parser"));
        let _ = std::fs::remove_dir_all(root);
    }
}
