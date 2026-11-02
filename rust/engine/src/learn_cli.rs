//! `fllm-rs learn` and the learning backend of `fllm-rs agent --learn`.
//!
//! * `fllm-rs learn status [--json]` shows what waits for a decision: skill revisions, memory entries, profile lines.
//! * `fllm-rs learn review` shows each item and, on a terminal, asks accept / reject / skip.
//! * `fllm-rs learn accept ID... | --all` and `reject ID... | --all` decide in bulk (`--skills`, `--memory`, `--profile` narrow `--all`).
//! * `fllm-rs learn memory`, `profile`, `forget`, `consolidate` look at, clean and tidy up what was learned.
//! * `fllm-rs skills improve|history|rollback|usage|diff|approve-revision|reject-revision NAME` (see [`skills_verb`]).
use crate::agent_cli::{Backend, HttpBackend, Outcome, Status};
use crate::learn::{self, SkillRef};
use crate::learn_flow::{self, Options, RunInfo};
use serde_json::{json, Value};
use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

pub const HELP: &str = "usage: fllm-rs learn status [--json]
       fllm-rs learn review [--folder DIR]
       fllm-rs learn accept ID... | --all [--skills|--memory|--profile]
       fllm-rs learn reject ID... | --all [--skills|--memory|--profile]
       fllm-rs learn memory [--folder DIR]
       fllm-rs learn profile [show | delete | export FILE | on | off]
       fllm-rs learn forget WORDS... [--all-learned] [--folder DIR]
       fllm-rs learn consolidate [--folder DIR] [--url http://127.0.0.1:8000] [--api-key KEY] [--model ID]
What `fllm-rs agent --learn` (and the Studio's 'Learn from this session') found waits here until you decide: revised skills with a diff and the
reason, memory entries with the session they came from, and lines for your profile. Nothing is applied before you accept it
(--auto-memory on the agent writes memory entries and profile lines at once). `forget` removes memory entries, profile lines and proposals that
contain the words; --all-learned removes everything the loop wrote itself. `consolidate` drafts the skill revisions that are due and proposes
to remove repeated memory entries (needs a running server); schedule it in the Studio (Settings > Skills and schedules) to run nightly.";

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// The words of a command without the options (and the values of the options that take one).
fn positional(args: &[String], with_value: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip = false;
    for a in args {
        if skip {
            skip = false;
        } else if with_value.contains(&a.as_str()) {
            skip = true;
        } else if !a.starts_with("--") {
            out.push(a.clone());
        }
    }
    out
}

fn workspace_of(args: &[String]) -> Option<PathBuf> {
    flag(args, "--folder").map(PathBuf::from).and_then(|p| p.canonicalize().ok()).map(|p| crate::agent_cli::plain_path(&p))
}

// ---------------------------------------------------------------------------------------------------------------
// the backend of a learning run
// ---------------------------------------------------------------------------------------------------------------

pub struct LearnConfig {
    pub learn: bool,
    pub skills: bool,
    pub auto_memory: bool,
    pub nudge_every: usize,
    pub quiet: bool,
    pub secrets: Vec<String>,
    pub small_context: bool,
    /// The model the server serves; named in the requests of the reflective steps.
    pub model: Option<String>,
}

/// Wraps the connection to the server: adds the `skill` tool and the skill index (`--skills`), notes how skills fared, runs the
/// reflective step every few tool steps of a long run and after the run (`--learn`). Everything else passes through.
pub struct LearnBackend<'a> {
    inner: &'a mut HttpBackend,
    studio: PathBuf,
    workspace: PathBuf,
    cfg: LearnConfig,
    matched: Vec<String>,
    nudge_seen: usize,
    fresh: bool,
}

fn tool_messages(messages: &[Value]) -> usize {
    messages.iter().filter(|m| m["role"] == "tool").count()
}

impl<'a> LearnBackend<'a> {
    pub(crate) fn new(inner: &'a mut HttpBackend, studio: &Path, workspace: &Path, task: &str, cfg: LearnConfig) -> LearnBackend<'a> {
        let studio = studio.to_path_buf();
        let matched = if cfg.skills && !task.trim().is_empty() {
            let skills = learn::list_skills(&studio, Some(workspace));
            learn::match_skills(task, &skills, 2).into_iter().map(|m| m.0).collect()
        } else {
            Vec::new()
        };
        LearnBackend { inner, studio, workspace: workspace.to_path_buf(), cfg, matched, nudge_seen: 0, fresh: true }
    }

    pub(crate) fn http(&mut self) -> &mut HttpBackend {
        self.inner
    }

    fn say(&self, line: &str) {
        if !self.cfg.quiet {
            eprintln!("{line}");
        }
    }

    fn options(&self) -> Options {
        Options {
            auto_memory: self.cfg.auto_memory,
            reflect: true,
            improve: true,
            transcript_chars: if self.cfg.small_context { 3500 } else { 6000 },
            secrets: self.cfg.secrets.clone(),
        }
    }

    /// After the run was answered: note the skills it used, draft revisions that are due, reflect on memory and profile.
    pub fn finish(&mut self, outcome: &Outcome, session: &str, verify: Option<bool>, from: usize) -> Vec<String> {
        if !self.cfg.learn {
            return Vec::new();
        }
        let failed = outcome.status != Status::Done || verify == Some(false);
        let errors: Vec<&crate::agent_cli::ToolRecord> = outcome.tool_calls.iter().filter(|c| matches!(c.status, "error" | "bad_args" | "repeated")).collect();
        let mut note = String::new();
        if outcome.status == Status::OutOfSteps {
            note.push_str("stopped at the step limit; ");
        }
        if verify == Some(false) {
            note.push_str("the verify command still failed; ");
        }
        for e in errors.iter().take(3) {
            note.push_str(&format!("{}: {}; ", e.name, e.detail.lines().next().unwrap_or_default()));
        }
        let run = &outcome.messages[from.min(outcome.messages.len())..];
        let info = RunInfo {
            session: session.to_owned(),
            workspace: Some(self.workspace.clone()),
            outcome: if failed { "failed" } else { "done" }.to_owned(),
            verify,
            retries: errors.len() as u32,
            note,
            skills: learn::skills_read_in(run),
            matched: self.matched.clone(),
            feedback: None,
        };
        let opts = self.options();
        let studio = self.studio.clone();
        let inner: &HttpBackend = self.inner;
        let model = self.cfg.model.clone();
        let mut complete = |mut body: Value| {
            body["model"] = json!(model);
            inner.complete_within(&body, 240)
        };
        let lines = learn_flow::after_run(&studio, &info, run, &opts, &mut complete);
        self.fresh = true;
        lines
    }
}

impl Backend for LearnBackend<'_> {
    fn tools(&mut self) -> Result<Vec<Value>, String> {
        let mut tools = self.inner.tools()?;
        if self.cfg.skills && !learn::list_skills(&self.studio, Some(&self.workspace)).is_empty() {
            tools.push(learn_flow::skill_tool_schema());
        }
        Ok(tools)
    }

    fn complete(&mut self, body: Value) -> Result<Value, String> {
        if self.cfg.learn && self.cfg.nudge_every > 0 {
            let messages: Vec<Value> = body["messages"].as_array().cloned().unwrap_or_default();
            let tools = tool_messages(&messages);
            if self.fresh {
                // what an earlier run left in a continued session does not count
                self.nudge_seen = tools;
                self.fresh = false;
            } else if tools >= self.nudge_seen + self.cfg.nudge_every {
                self.nudge_seen = tools;
                self.say("[learn] a long run: reflecting on what is worth remembering so far");
                let opts = self.options();
                let studio = self.studio.clone();
                let ws = self.workspace.clone();
                let inner: &HttpBackend = self.inner;
                let mut complete = |mut b: Value| {
                    b["model"] = body["model"].clone();
                    inner.complete_within(&b, 120)
                };
                match learn_flow::reflect(&studio, Some(&ws), "nudge", &messages, &opts, &mut complete) {
                    Ok((m, p)) => {
                        for t in m.added.iter().chain(&m.proposed).chain(&p.added).chain(&p.proposed) {
                            self.say(&format!("[learn] noted: {}", learn::clip_chars(t, 120)));
                        }
                    }
                    Err(e) => self.say(&format!("[learn] reflection skipped: {e}")),
                }
            }
        }
        self.inner.complete(body)
    }

    fn run_tool(&mut self, name: &str, args: &Value) -> Result<String, String> {
        if name == "skill" && self.cfg.skills {
            return learn_flow::skill_tool(&self.studio, Some(&self.workspace), args);
        }
        self.inner.run_tool(name, args)
    }

    fn instructions(&mut self) -> Result<Value, String> {
        let mut answer = self.inner.instructions().unwrap_or_else(|_| json!({}));
        if !answer.is_object() {
            answer = json!({});
        }
        if self.cfg.skills {
            let skills = learn::list_skills(&self.studio, Some(&self.workspace));
            let index = learn_flow::skills_index(&skills, &self.matched, 1800);
            if !index.is_empty() {
                answer["skills"] = json!(index);
            }
        }
        // a server that does not know the profile (an older one) is covered by reading the file here
        if answer.get("profile").is_none() {
            answer["profile"] = json!(learn::profile_for_prompt(&self.studio));
        }
        Ok(answer)
    }

    fn after_run(&mut self, outcome: &Outcome, session: &str, from: usize) {
        for line in self.finish(outcome, session, None, from) {
            self.say(&line);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// the command
// ---------------------------------------------------------------------------------------------------------------

/// A server connection for the commands that need a model: `--url`, `--api-key` (or FLLM_API_KEY), `--model`.
fn connect(args: &[String]) -> Result<(HttpBackend, Option<String>), String> {
    let url = flag(args, "--url").unwrap_or_else(|| "http://127.0.0.1:8000".to_owned());
    let key = flag(args, "--api-key").or_else(|| std::env::var("FLLM_API_KEY").ok().filter(|k| !k.is_empty()));
    let http = HttpBackend::new(&url, key)?;
    let (served, _) = http.model_info()?;
    Ok((http, Some(flag(args, "--model").unwrap_or(served))))
}

fn with_model<T>(args: &[String], f: impl FnOnce(&mut dyn FnMut(Value) -> Result<Value, String>) -> T) -> Result<T, String> {
    let (http, model) = connect(args)?;
    let mut complete = |mut body: Value| {
        body["model"] = json!(model);
        http.complete_within(&body, 300)
    };
    Ok(f(&mut complete))
}

fn print_overview(o: &Value) {
    let list = |k: &str| o[k].as_array().cloned().unwrap_or_default();
    let (skills, memory, profile) = (list("skills"), list("memory"), list("profile"));
    if skills.is_empty() && memory.is_empty() && profile.is_empty() {
        println!("nothing waits for a decision");
    }
    for s in &skills {
        println!("{}  revised skill '{}': {}", s["id"].as_str().unwrap_or("?"), s["name"].as_str().unwrap_or("?"), s["reason"].as_str().unwrap_or_default());
    }
    for m in &memory {
        let what = if m["action"] == "remove" { "forget" } else { "remember" };
        println!("{}  {what} ({}, {}): {}", m["id"].as_str().unwrap_or("?"), m["scope"].as_str().unwrap_or("?"), m["kind"].as_str().unwrap_or("fact"), m["text"].as_str().unwrap_or_default());
    }
    for p in &profile {
        println!("{}  profile / {}: {}", p["id"].as_str().unwrap_or("?"), p["section"].as_str().unwrap_or("?"), p["text"].as_str().unwrap_or_default());
    }
    if let Some(lines) = o["last_run"]["lines"].as_array() {
        if !lines.is_empty() {
            println!("\nlast learning run ({}):", o["last_run"]["session"].as_str().unwrap_or("?"));
            for l in lines {
                println!("  {}", l.as_str().unwrap_or_default());
            }
        }
    }
}

/// Applies a decision to one item by id. Returns a line for the user.
pub fn decide(studio: &Path, id: &str, accept: bool) -> Result<String, String> {
    if let Some(key) = id.strip_prefix("skill:") {
        let rev = learn::pending_revisions(studio).into_iter().find(|r| r.skill.key() == key).ok_or_else(|| format!("no revision '{id}'"))?;
        return if accept {
            let v = learn::approve_revision(studio, &rev.skill, false)?;
            Ok(format!("skill '{}' revised (the previous text is kept as version {v}; undo with: fllm-rs skills rollback {})", rev.skill.name, rev.skill.name))
        } else {
            learn::reject_revision(studio, &rev.skill)?;
            Ok(format!("revision of '{}' dropped", rev.skill.name))
        };
    }
    if id.starts_with("prof-") {
        return learn::decide_profile(studio, id, accept);
    }
    learn::decide_memory(studio, id, accept)
}

fn all_ids(o: &Value, args: &[String]) -> Vec<String> {
    let only = ["--skills", "--memory", "--profile"].iter().filter(|f| has(args, f)).count() > 0;
    let want = |f: &str| !only || has(args, f);
    let mut ids = Vec::new();
    for (key, f) in [("skills", "--skills"), ("memory", "--memory"), ("profile", "--profile")] {
        if want(f) {
            ids.extend(o[key].as_array().into_iter().flatten().filter_map(|i| i["id"].as_str().map(str::to_owned)));
        }
    }
    ids
}

fn review(studio: &Path) -> Result<i32, String> {
    let o = learn::overview(studio);
    let mut items: Vec<(String, String)> = Vec::new();
    for s in o["skills"].as_array().into_iter().flatten() {
        items.push((
            s["id"].as_str().unwrap_or_default().to_owned(),
            format!("Revised skill '{}'\nWhy: {}\n{}", s["name"].as_str().unwrap_or("?"), s["reason"].as_str().unwrap_or_default(), s["diff"].as_str().unwrap_or_default()),
        ));
    }
    for m in o["memory"].as_array().into_iter().flatten() {
        let src = m["session"].as_str().filter(|s| !s.is_empty()).map_or(String::new(), |s| format!(" (from session {s})"));
        let what = if m["action"] == "remove" { format!("Forget this memory entry ({})", m["reason"].as_str().unwrap_or_default()) } else { format!("Remember ({})", m["scope"].as_str().unwrap_or("?")) };
        items.push((m["id"].as_str().unwrap_or_default().to_owned(), format!("{what}{src}:\n  {}", m["text"].as_str().unwrap_or_default())));
    }
    for p in o["profile"].as_array().into_iter().flatten() {
        items.push((p["id"].as_str().unwrap_or_default().to_owned(), format!("Add to your profile / {}:\n  {}", p["section"].as_str().unwrap_or("?"), p["text"].as_str().unwrap_or_default())));
    }
    if items.is_empty() {
        println!("nothing waits for a decision");
        return Ok(0);
    }
    let ask = std::io::stdin().is_terminal();
    let stdin = std::io::stdin();
    for (n, (id, text)) in items.iter().enumerate() {
        println!("[{}/{}] {id}\n{text}", n + 1, items.len());
        if !ask {
            println!();
            continue;
        }
        loop {
            eprint!("[a]ccept / [r]eject / [s]kip / [q]uit: ");
            std::io::stderr().flush().ok();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
                return Ok(0);
            }
            match line.trim().to_lowercase().as_str() {
                "a" | "accept" | "y" | "yes" => {
                    println!("{}", decide(studio, id, true).unwrap_or_else(|e| format!("error: {e}")));
                    break;
                }
                "r" | "reject" | "n" | "no" => {
                    println!("{}", decide(studio, id, false).unwrap_or_else(|e| format!("error: {e}")));
                    break;
                }
                "s" | "skip" | "" => break,
                "q" | "quit" => return Ok(0),
                _ => {}
            }
        }
        println!();
    }
    if !ask {
        println!("decide with: fllm-rs learn accept ID... | reject ID... (or --all)");
    }
    Ok(0)
}

pub fn run(args: &[String]) -> Result<i32, String> {
    let studio = crate::agent_cli::studio_folder();
    let rest = &args[args.len().min(1)..];
    let ws = workspace_of(rest);
    match args.first().map(String::as_str) {
        Some("status") => {
            let o = learn::overview(&studio);
            if has(rest, "--json") {
                println!("{}", serde_json::to_string_pretty(&o).map_err(|e| e.to_string())?);
            } else {
                print_overview(&o);
                let profile = learn::read_profile(&studio);
                println!(
                    "\nprofile: {} line(s), {} in the system prompt; memory: {} global entr{}",
                    learn::profile_bullets_count(&profile),
                    if learn::profile_in_prompt(&studio) { "used" } else { "not used" },
                    learn::read_memory(&studio, "global", None).len(),
                    if learn::read_memory(&studio, "global", None).len() == 1 { "y" } else { "ies" }
                );
            }
            Ok(0)
        }
        Some("review") => review(&studio),
        Some(verb @ ("accept" | "reject")) => {
            let accept = verb == "accept";
            let o = learn::overview(&studio);
            let ids = if has(rest, "--all") { all_ids(&o, rest) } else { positional(rest, &["--folder"]) };
            if ids.is_empty() {
                return Err(format!("give the ids to {verb} (see: fllm-rs learn status) or --all"));
            }
            let mut bad = 0;
            for id in ids {
                match decide(&studio, &id, accept) {
                    Ok(line) => println!("{line}"),
                    Err(e) => {
                        bad += 1;
                        eprintln!("{id}: {e}");
                    }
                }
            }
            Ok(i32::from(bad > 0))
        }
        Some("memory") => {
            let sources = learn::memory_sources(&studio);
            for (scope, label) in [("global", "for every project"), ("workspace", "this project")] {
                if scope == "workspace" && ws.is_none() {
                    continue;
                }
                let entries = learn::read_memory(&studio, scope, ws.as_deref());
                println!("{label} ({} entr{}):", entries.len(), if entries.len() == 1 { "y" } else { "ies" });
                for (i, e) in entries.iter().enumerate() {
                    let from = sources.get(&learn::normalize(e)).map_or(String::new(), |s| format!("   [learned in session {}]", s["session"].as_str().unwrap_or("?")));
                    println!("  {}. {e}{from}", i + 1);
                }
            }
            Ok(0)
        }
        Some("profile") => match rest.first().map(String::as_str) {
            None | Some("show") => {
                let text = learn::read_profile(&studio);
                if text.trim().is_empty() {
                    println!("no profile yet ({} will hold it)", learn::profile_path(&studio).display());
                } else {
                    println!("{}", text.trim_end());
                }
                Ok(0)
            }
            Some("delete") => {
                learn::delete_profile(&studio)?;
                println!("profile and waiting profile lines deleted");
                Ok(0)
            }
            Some("export") => {
                let to = rest.get(1).ok_or("give a file to write")?;
                std::fs::write(to, learn::read_profile(&studio)).map_err(|e| e.to_string())?;
                println!("profile written to {to}");
                Ok(0)
            }
            Some(v @ ("on" | "off")) => {
                learn::set_profile_in_prompt(&studio, v == "on")?;
                println!("the profile is {} in the system prompt", if v == "on" { "put" } else { "not put" });
                Ok(0)
            }
            Some(other) => Err(format!("unknown profile command '{other}'")),
        },
        Some("forget") => {
            let words = positional(rest, &["--folder"]).join(" ");
            let gone = learn::forget(&studio, ws.as_deref(), &words, has(rest, "--all-learned"))?;
            if gone.is_empty() {
                println!("nothing matched");
            }
            for g in gone {
                println!("forgot: {g}");
            }
            Ok(0)
        }
        Some("consolidate") => {
            let lines = with_model(rest, |complete| learn_flow::consolidate(&studio, ws.as_deref(), complete))?;
            for l in lines {
                println!("{l}");
            }
            Ok(0)
        }
        _ => {
            println!("{HELP}");
            Ok(if args.is_empty() || args[0] == "--help" || args[0] == "-h" { 0 } else { 1 })
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// `fllm-rs skills improve|history|rollback|usage|diff|approve-revision|reject-revision`
// ---------------------------------------------------------------------------------------------------------------

fn find_skill(studio: &Path, ws: Option<&Path>, name: &str) -> Result<SkillRef, String> {
    if !crate::agent_cli::valid_session_name(name) {
        return Err(format!("'{name}' is not a skill name"));
    }
    learn::list_skills(studio, ws).into_iter().find(|s| s.skill.name == name).map(|s| s.skill).ok_or_else(|| format!("no skill named '{name}'"))
}

fn when(t: i64) -> String {
    chrono::DateTime::from_timestamp(t, 0).map_or_else(|| "?".to_owned(), |d| d.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string())
}

/// The new verbs of `fllm-rs skills`; `None` when `verb` is not one of them.
pub fn skills_verb(verb: &str, args: &[String]) -> Option<Result<i32, String>> {
    if !matches!(verb, "usage" | "improve" | "diff" | "approve-revision" | "reject-revision" | "history" | "rollback") {
        return None;
    }
    Some(skills_verb_inner(verb, args))
}

fn skills_verb_inner(verb: &str, args: &[String]) -> Result<i32, String> {
    let studio = crate::agent_cli::studio_folder();
    let ws = workspace_of(args);
    let words = positional(args, &["--folder", "--url", "--api-key", "--model"]);
    if verb == "usage" && words.is_empty() {
        let overview = learn_flow::skill_overview(&studio, ws.as_deref());
        if overview.is_empty() {
            println!("no skills");
        }
        for s in overview {
            println!(
                "{:<26} {} use(s), {} failed, thumbs {}/{}{}{}",
                s["name"].as_str().unwrap_or("?"),
                s["uses"],
                s["failed"],
                s["up"],
                s["down"],
                if s["pending"] == true { "   [revision waiting]" } else { "" },
                s["due"].as_str().map_or(String::new(), |d| format!("   [review due: {d}]")),
            );
        }
        return Ok(0);
    }
    let name = words.first().ok_or("give the name of the skill")?;
    let skill = find_skill(&studio, ws.as_deref(), name)?;
    match verb {
        "usage" => {
            let usage = learn::read_usage(&studio, &skill);
            let st = learn::stats_of(&studio, &skill);
            println!("{name}: {} use(s), {} ok, {} failed, thumbs up {} / down {}, {} retries", st.uses, st.ok, st.failed, st.up, st.down, st.retries);
            for u in usage.iter().rev().take(10).rev() {
                println!("  {}  {:<8} {}{}{}", when(u.t), u.outcome, u.session, u.feedback.as_ref().map_or(String::new(), |f| format!("  thumbs {f}")), if u.note.is_empty() { String::new() } else { format!("  {}", u.note) });
            }
            Ok(0)
        }
        "improve" => {
            let trigger = learn::improve_due(&studio, &skill).unwrap_or_else(|| "requested by hand".to_owned());
            if learn::pending_revision(&studio, &skill).is_some() {
                return Err(format!("a revision of '{name}' waits already (fllm-rs skills diff {name})"));
            }
            let drafted = with_model(args, |complete| learn_flow::improve_skill(&studio, &skill, &trigger, "", complete))??;
            if drafted {
                println!("drafted a revision of '{name}' ({trigger}); the skill is unchanged until you approve it:");
                if let Some(r) = learn::pending_revision(&studio, &skill) {
                    println!("why: {}\n{}", r.reason, learn::diff(&r.old, &r.new));
                }
                println!("fllm-rs skills approve-revision {name}   |   fllm-rs skills reject-revision {name}");
            } else {
                println!("the model found nothing to change in '{name}'");
            }
            Ok(0)
        }
        "diff" => {
            let r = learn::pending_revision(&studio, &skill).ok_or_else(|| format!("no revision waits for '{name}'"))?;
            println!("why: {}\n{}", r.reason, learn::diff(&r.old, &r.new));
            Ok(0)
        }
        "approve-revision" => {
            let v = learn::approve_revision(&studio, &skill, has(args, "--force"))?;
            println!("'{name}' revised; the previous text is version {v} (fllm-rs skills rollback {name} {v})");
            Ok(0)
        }
        "reject-revision" => {
            learn::reject_revision(&studio, &skill)?;
            println!("revision of '{name}' dropped");
            Ok(0)
        }
        "history" => {
            let h = learn::read_history(&studio, &skill);
            if h.is_empty() {
                println!("'{name}' has no history yet");
            }
            for e in h {
                println!("  {}  v{:<3} {:<9} {}", when(e["t"].as_i64().unwrap_or(0)), e["version"], e["action"].as_str().unwrap_or("?"), e["reason"].as_str().unwrap_or_default());
            }
            Ok(0)
        }
        _ => {
            let version = words.get(1).map(|v| v.trim_start_matches('v').parse::<u32>().map_err(|_| "the version is a number".to_owned())).transpose()?;
            let kept = learn::rollback(&studio, &skill, version)?;
            println!("'{name}' rolled back; the text it had before is kept as version {kept}");
            Ok(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fllm-learncli-{tag}-{}-{}", std::process::id(), learn::now()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn decisions_apply_by_id_and_bulk_selection_follows_the_filters() {
        let studio = temp("decide");
        let dir = studio.join("skills/build");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), "---\nname: build\ndescription: d\n---\n1. old step\n").unwrap();
        let s = SkillRef::user("build");
        learn::propose_revision(&studio, &s, "---\nname: build\ndescription: d\n---\n1. new step\n", "because", "t").unwrap();
        learn::ingest_memory(&studio, None, "s1", &[learn::MemoryCandidate { text: "Staging runs on port 9001 for tests".into(), kind: "fact".into(), scope: "global".into() }], false);
        learn::ingest_profile(&studio, "s1", &[("languages".into(), "Writes Rust every day".into())], false);
        let o = learn::overview(&studio);
        assert_eq!(o["pending"], 3);
        let args = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert_eq!(all_ids(&o, &args(&["--all"])).len(), 3);
        assert_eq!(all_ids(&o, &args(&["--all", "--memory"])).len(), 1);
        assert_eq!(all_ids(&o, &args(&["--all", "--skills", "--profile"])).len(), 2);
        let sid = o["skills"][0]["id"].as_str().unwrap();
        assert_eq!(sid, "skill:build");
        assert!(o["skills"][0]["diff"].as_str().unwrap().contains("+ 1. new step"));
        assert!(decide(&studio, sid, true).unwrap().contains("revised"));
        assert!(std::fs::read_to_string(dir.join("SKILL.md")).unwrap().contains("new step"));
        let mid = o["memory"][0]["id"].as_str().unwrap();
        assert!(decide(&studio, mid, true).unwrap().contains("remembered"));
        assert!(learn::read_memory(&studio, "global", None)[0].contains("9001"));
        let pid = o["profile"][0]["id"].as_str().unwrap();
        assert!(decide(&studio, pid, false).unwrap().contains("dropped"));
        assert_eq!(learn::overview(&studio)["pending"], 0);
        assert!(decide(&studio, "mem-00000000", true).is_err());
        assert!(decide(&studio, "skill:nothing", true).is_err());
        let _ = std::fs::remove_dir_all(&studio);
    }

    /// A minimal server: answers `count` requests, one connection each, with whatever `handler(request line, body)` returns.
    fn serve(count: usize, mut handler: impl FnMut(&str, &Value) -> Value + Send + 'static) -> (String, std::thread::JoinHandle<Vec<(String, Value)>>) {
        use std::io::Read;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for _ in 0..count {
                let (mut stream, _) = listener.accept().unwrap();
                stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
                let mut got = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let n = stream.read(&mut buf).unwrap_or(0);
                    got.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&got).into_owned();
                    if let Some((head, body)) = text.split_once("\r\n\r\n") {
                        let len = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0))).unwrap_or(0);
                        if body.len() >= len {
                            break;
                        }
                    }
                    if n == 0 {
                        break;
                    }
                }
                let text = String::from_utf8_lossy(&got).into_owned();
                let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
                let line = head.lines().next().unwrap_or_default().to_owned();
                let body: Value = serde_json::from_str(body).unwrap_or(Value::Null);
                let payload = handler(&line, &body).to_string();
                stream.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}", payload.len()).as_bytes()).unwrap();
                seen.push((line, body));
            }
            seen
        });
        (url, handle)
    }

    #[test]
    fn a_learning_run_end_to_end_against_a_stub_server() {
        let studio = temp("e2e");
        let ws = temp("e2ews");
        let dir = studio.join("skills/build");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), "---\nname: build\ndescription: Build the project\nwhen: the user asks to build\n---\n1. run make\n").unwrap();
        std::fs::write(studio.join("user-profile.md"), "## Languages\n- Writes Rust\n").unwrap();
        let reflection = json!({"memory": [{"text": "The project builds with cargo build, not make", "kind": "command", "scope": "workspace"}], "profile": [{"section": "tool_habits", "text": "Runs cargo test before committing"}]}).to_string();
        let r = reflection.clone();
        let (url, server) = serve(6, move |line, body| {
            if line.contains("/api/studio/agent/tools") {
                json!({"tools": [{"type": "function", "function": {"name": "bash", "description": "d", "parameters": {"type": "object", "properties": {}}}, "permission": "command"}]})
            } else if line.contains("/api/studio/agent/instructions") {
                json!({"files": [], "memory": {"workspace": [], "global": []}})
            } else if body.get("response_format").is_some() {
                json!({"choices": [{"message": {"role": "assistant", "content": r}}]})
            } else {
                json!({"choices": [{"message": {"role": "assistant", "content": "done"}}]})
            }
        });
        let mut http = HttpBackend::new(&url, None).unwrap();
        let cfg = LearnConfig { learn: true, skills: true, auto_memory: false, nudge_every: 2, quiet: true, secrets: Vec::new(), small_context: false, model: Some("m".into()) };
        let mut lb = LearnBackend::new(&mut http, &studio, &ws, "please build the project", cfg);
        // the skill tool is offered next to the server's tools
        let names: Vec<String> = lb.tools().unwrap().iter().filter_map(|t| t["function"]["name"].as_str().map(str::to_owned)).collect();
        assert_eq!(names, ["bash", "skill"]);
        // the system prompt gets the skill index (the task matches the skill) and the profile
        let answer = lb.instructions().unwrap();
        assert!(answer["skills"].as_str().unwrap().contains("- build: Build the project") && answer["skills"].as_str().unwrap().contains("read them first: build"), "{answer}");
        assert_eq!(answer["profile"], "- Languages: Writes Rust");
        let block = crate::agent_cli::instructions_block(&answer);
        assert!(block.contains("# Skills") && block.contains("# About the user (profile)\n"), "{block}");
        // the skill tool runs locally
        assert!(lb.run_tool("skill", &json!({"action": "read", "name": "build"})).unwrap().contains("1. run make"));
        // first request: passes through; the second has two more tool results: a reflection runs before it
        let first = json!({"model": "m", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "please build the project"}]});
        assert_eq!(lb.complete(first).unwrap()["choices"][0]["message"]["content"], "done");
        let messages = vec![
            json!({"role": "system", "content": "s"}),
            json!({"role": "user", "content": "please build the project"}),
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "skill", "arguments": "{\"action\":\"read\",\"name\":\"build\"}"}}, {"id": "2", "function": {"name": "bash", "arguments": "{\"command\":\"cargo build\"}"}}]}),
            json!({"role": "tool", "tool_call_id": "1", "content": "1. run make"}),
            json!({"role": "tool", "tool_call_id": "2", "content": "Finished dev profile"}),
        ];
        lb.complete(json!({"model": "m", "messages": messages})).unwrap();
        assert_eq!(learn::memory_proposals(&studio).len(), 1, "the nudge proposed an entry");
        // after the run: the use of the skill is noted, the reflection finds the same entry again (a duplicate) and a new profile line
        let outcome = Outcome {
            answer: "built".into(),
            status: Status::Done,
            steps: 2,
            tool_calls: Vec::new(),
            messages: messages[1..].iter().cloned().chain([json!({"role": "assistant", "content": "built with cargo"})]).collect(),
            edited: Vec::new(),
        };
        let lines = lb.finish(&outcome, "run-7", Some(true), 0);
        assert!(lines.iter().any(|l| l.contains("skill 'build' used")), "{lines:?}");
        let seen = server.join().unwrap();
        assert_eq!(seen.len(), 6);
        assert!(seen[3].1.get("response_format").is_some() && seen[4].1.get("response_format").is_none(), "nudge, then the real request: {:?}", seen.iter().map(|s| s.0.clone()).collect::<Vec<_>>());
        assert_eq!(seen[3].1["model"], "m");
        let usage = learn::read_usage(&studio, &SkillRef::user("build"));
        assert_eq!(usage.len(), 1);
        assert!(usage[0].matched && usage[0].verify == Some(true) && usage[0].session == "run-7" && usage[0].outcome == "done");
        assert_eq!(learn::memory_proposals(&studio).len(), 1, "no second copy of the entry");
        assert_eq!(learn::profile_proposals(&studio).len(), 1);
        assert!(learn::read_memory(&studio, "workspace", Some(&ws)).is_empty(), "nothing is written without approval");
        // a run without --learn does nothing
        let mut off = LearnBackend::new(&mut http, &studio, &ws, "", LearnConfig { learn: false, skills: false, auto_memory: false, nudge_every: 0, quiet: true, secrets: Vec::new(), small_context: false, model: None });
        assert!(off.finish(&outcome, "x", None, 0).is_empty());
        let _ = std::fs::remove_dir_all(&studio);
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn words_are_separated_from_options() {
        let a: Vec<String> = ["old", "--folder", "x", "stuff", "--all-learned"].iter().map(|s| (*s).to_owned()).collect();
        assert_eq!(positional(&a, &["--folder"]), ["old", "stuff"]);
        assert_eq!(flag(&a, "--folder").as_deref(), Some("x"));
        assert!(has(&a, "--all-learned"));
        assert!(skills_verb("list", &[]).is_none());
    }
}
