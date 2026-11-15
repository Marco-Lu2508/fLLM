//! Row 87 of PARITY.md: the *Background jobs* of the Code page (`parseJobId`, `trackJob`,
//! `jobOutput`, `jobKill` and `jobsHtml` of `code.js`) — the commands the agent started with
//! `bash` and `background: true`, kept in the side with their output and a button to stop them.
//!
//! This file is the logic that is compared with the page case by case
//! (`tests/jobs_fixtures.rs`); the window only holds the list and draws it.

use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;

/// One background job (`code.jobs[i]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    pub id: String,
    pub command: String,
    /// `running`, `done` or `killed`.
    pub status: String,
    pub out: String,
    pub open: bool,
}

/// JavaScript's truthiness of a value the model wrote.
pub fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::Bool(true)) | Some(Value::Array(_)) | Some(Value::Object(_)) => true,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()),
        Some(Value::String(s)) => !s.is_empty(),
    }
}

/// `String(value)` of the values a model writes into an argument.
fn js_string(v: &Value) -> String {
    crate::subagents::js_text(Some(v), "")
}

fn id_words() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // JavaScript's `\b` knows only ASCII word characters
    RE.get_or_init(|| Regex::new(r#"(?i)(?-u:\b)(?:job|shell|process|task|id)(?-u:\b)[\s:=#"'(]*([A-Za-z0-9_-]+)"#).expect("the first pattern of parseJobId"))
}

fn id_shapes() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?-u:\b)(bash_[A-Za-z0-9]+|[a-f0-9]{6,}|[0-9]+)(?-u:\b)").expect("the second pattern of parseJobId"))
}

fn ended_words() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)(?-u:\b)(?:exited|finished|completed|exit code)(?-u:\b)").expect("the words of an ended job"))
}

/// `parseJobId(text)`: the id of a job from the answer of the bash tool — the word after
/// *job*, *shell*, *process*, *task* or *id*, else anything that looks like an id.
pub fn parse_job_id(text: &str) -> Option<String> {
    id_words().captures(text).or_else(|| id_shapes().captures(text)).map(|m| m[1].to_string())
}

/// Whether the output says the job is over (`/\b(exited|finished|completed|exit code)\b/i`).
pub fn output_says_ended(output: &str) -> bool {
    ended_words().is_match(output)
}

/// `jobArgName()`: what the server calls the id in `bash_output` — the first argument that
/// looks like an id, else the first, else `id`.
pub fn job_arg_name(properties: &[String]) -> String {
    let looks = Regex::new(r"(?i)id|job|shell").expect("the pattern of jobArgName");
    properties.iter().find(|k| looks.is_match(k)).or_else(|| properties.first()).cloned().unwrap_or_else(|| "id".to_string())
}

/// `trackJob(name, args, output)`: the answer of a finished tool call changes the list.
pub fn track_job(jobs: &mut Vec<Job>, name: &str, args: &Value, output: &str) {
    if name == "bash" && truthy(args.get("background")) {
        if let Some(id) = parse_job_id(output) {
            let command = js_string(args.get("command").filter(|_| truthy(args.get("command"))).unwrap_or(&Value::String(String::new())));
            jobs.push(Job { id, command, status: "running".into(), out: String::new(), open: false });
        }
    } else if name == "bash_output" || name == "bash_kill" {
        // `String(Object.values(args)[0] ?? '')`
        let id = args.as_object().and_then(|o| o.values().next()).filter(|v| !v.is_null()).map(js_string).unwrap_or_default();
        if let Some(job) = jobs.iter_mut().find(|j| j.id == id) {
            if name == "bash_kill" {
                job.status = "killed".into();
            } else {
                job.out = output.to_string();
                if output_says_ended(output) {
                    job.status = "done".into();
                }
            }
        }
    }
}

/// The first half of `jobOutput(id)`: the box opens or closes; `true` when it opened and the
/// output has to be read.
pub fn toggle_output(jobs: &mut [Job], id: &str) -> Option<bool> {
    let job = jobs.iter_mut().find(|j| j.id == id)?;
    job.open = !job.open;
    Some(job.open)
}

/// The second half: what `bash_output` answered (`Err` is the message of a call that failed).
pub fn output_arrived(jobs: &mut [Job], id: &str, answer: Result<String, String>) {
    if let Some(job) = jobs.iter_mut().find(|j| j.id == id) {
        match answer {
            Ok(output) => {
                job.out = output.clone();
                if output_says_ended(&output) {
                    job.status = "done".into();
                }
            }
            Err(e) => job.out = e,
        }
    }
}

/// What `jobKill(id)` makes of the answer of `bash_kill`: the toast when it failed.
pub fn kill_answered(jobs: &mut [Job], id: &str, answer: Result<String, String>) -> Option<String> {
    match answer {
        Ok(_) => {
            if let Some(job) = jobs.iter_mut().find(|j| j.id == id) {
                job.status = "killed".into();
            }
            None
        }
        Err(e) => Some(if e.is_empty() { "The job could not be stopped.".to_string() } else { e }),
    }
}

/// The text of a job's box (`j.out || '(no output yet)'`).
pub fn shown_output(job: &Job) -> &str {
    if job.out.is_empty() { "(no output yet)" } else { &job.out }
}

/// The paragraph of the empty block.
pub const EMPTY: &str = "Commands started in the background show up here.";
