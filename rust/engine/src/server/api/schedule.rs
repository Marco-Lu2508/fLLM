//! Scheduled tasks. A task is a prompt that runs on a clock inside the server: as a plain chat answer or as an agent
//! run in a folder with a fixed read-only tool set (list_dir, read_file, grep, glob, web_fetch, web_search). The
//! tasks live in `schedule.json` and the last results of each task in `schedule-results.json`, both next to
//! `studio.json`. The clock is a thread of the server, so tasks run only while the app (the server) runs; a run that
//! was missed while it was off is not made up. One task never runs twice at once.
//!
//! The schedule text and the next-run computation are pure functions (local time, daylight saving safe): a wall-clock
//! time that does not exist on a day (the hour skipped in spring) runs at the first moment after the gap, one that
//! exists twice (the hour repeated in autumn) runs once, at its first occurrence.
use super::*;
use axum::extract::Query;
use chrono::{Datelike, Local, LocalResult, NaiveDate, NaiveDateTime, TimeZone, Timelike};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{atomic::AtomicBool, Mutex, OnceLock, Weak};

const MAX_TASKS: usize = 50;
const KEEP_RESULTS: usize = 20;
const MAX_RESULT_CHARS: usize = 20_000;
const MAX_PROMPT_CHARS: usize = 8_000;
const MAX_AGENT_STEPS: usize = 12;
const TICK_SECS: u64 = 5;
/// A run that is later than this when the clock notices it (the computer was asleep) is skipped, not made up.
const MISSED_GRACE_SECS: i64 = 3600;
const READ_ONLY_TOOLS: [&str; 6] = ["list_dir", "read_file", "grep", "glob", "web_fetch", "web_search"];
const NET_TOOLS: [&str; 2] = ["web_fetch", "web_search"];

// ---------------------------------------------------------------------------------------------------------------
// Schedules (pure)
// ---------------------------------------------------------------------------------------------------------------

/// Every day of the month (bits 1 to 31) and every month (bits 1 to 12), as a `*` field gives them.
const ALL_DAYS: u32 = 0xFFFF_FFFE;
const ALL_MONTHS: u16 = 0x1FFE;

/// The fields of a cron line as bit sets. Day of month and weekday follow the usual cron rule: when both are
/// restricted, a day matches if either does.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Cron {
    minutes: u64,
    hours: u32,
    days: u32,
    months: u16,
    weekdays: u8,
    any_day: bool,
    any_weekday: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Schedule {
    /// A fixed interval; it ignores the clock of the wall.
    Every { minutes: u32 },
    Cron(Cron),
}

fn parse_clock(text: &str) -> Result<(u32, u32), String> {
    let t: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let (t, pm, am) = match (t.strip_suffix("pm"), t.strip_suffix("am")) {
        (Some(rest), _) => (rest.to_owned(), true, false),
        (_, Some(rest)) => (rest.to_owned(), false, true),
        _ => (t.clone(), false, false),
    };
    let (h, m) = t.split_once(':').unwrap_or((t.as_str(), "0"));
    let hour: u32 = h.parse().map_err(|_| format!("'{text}' is not a time like 08:30"))?;
    let minute: u32 = m.parse().map_err(|_| format!("'{text}' is not a time like 08:30"))?;
    let hour = match (am, pm) {
        (false, false) if hour < 24 => hour,
        (true, _) if (1..=12).contains(&hour) => hour % 12,
        (_, true) if (1..=12).contains(&hour) => hour % 12 + 12,
        _ => return Err(format!("'{text}' is not a valid time")),
    };
    if minute > 59 {
        return Err(format!("'{text}' is not a valid time"));
    }
    Ok((hour, minute))
}

fn name_number(word: &str) -> Option<u32> {
    const DAYS: [&str; 7] = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];
    const MONTHS: [&str; 12] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
    let short = word.get(..3)?;
    DAYS.iter().position(|d| *d == short).map(|i| i as u32).or_else(|| MONTHS.iter().position(|m| *m == short).map(|i| i as u32 + 1))
}

/// One cron field as a bit set and whether it was a plain `*`.
fn parse_field(field: &str, min: u32, max: u32, names: bool) -> Result<(u64, bool), String> {
    let number = |s: &str| -> Result<u32, String> {
        match s.parse::<u32>() {
            Ok(n) => Ok(n),
            Err(_) if names => name_number(s).ok_or_else(|| format!("'{s}' is not a number")),
            Err(_) => Err(format!("'{s}' is not a number")),
        }
    };
    let mut mask = 0u64;
    for part in field.split(',') {
        let (range, step) = match part.split_once('/') {
            Some((r, s)) => (r, s.parse::<u32>().map_err(|_| format!("'{s}' is not a step"))?),
            None => (part, 1),
        };
        if step == 0 {
            return Err("a step must be at least 1".into());
        }
        let (lo, hi) = if range == "*" {
            (min, max)
        } else if let Some((a, b)) = range.split_once('-') {
            (number(a)?, number(b)?)
        } else {
            let n = number(range)?;
            (n, if part.contains('/') { max } else { n })
        };
        if lo < min || hi > max || lo > hi {
            return Err(format!("'{part}' is outside {min}-{max}"));
        }
        let mut n = lo;
        while n <= hi {
            mask |= 1 << n;
            n += step;
        }
    }
    Ok((mask, field == "*"))
}

fn parse_cron(text: &str) -> Result<Cron, String> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() != 5 {
        return Err("write the schedule like 'every 30 minutes', 'daily at 08:30', 'weekdays at 09:00' or as a cron line with 5 fields (minute hour day month weekday)".into());
    }
    let (minutes, _) = parse_field(fields[0], 0, 59, false)?;
    let (hours, _) = parse_field(fields[1], 0, 23, false)?;
    let (days, any_day) = parse_field(fields[2], 1, 31, false)?;
    let (months, _) = parse_field(fields[3], 1, 12, true)?;
    let (mut weekdays, any_weekday) = parse_field(fields[4], 0, 7, true)?;
    if weekdays & (1 << 7) != 0 {
        weekdays = (weekdays | 1) & !(1 << 7);
    }
    Ok(Cron { minutes, hours: hours as u32, days: days as u32, months: months as u16, weekdays: weekdays as u8, any_day, any_weekday })
}

fn daily_like(hour: u32, minute: u32, weekdays: u8) -> Cron {
    Cron { minutes: 1 << minute, hours: 1 << hour, days: ALL_DAYS, months: ALL_MONTHS, weekdays, any_day: true, any_weekday: weekdays == 0x7f }
}

/// Reads the text of a schedule: `every 30 minutes`, `every 2 hours`, `hourly`, `hourly at :15`, `daily at 08:30`,
/// `weekdays at 9am`, `weekly on mon,thu at 18:00`, or a cron line with five fields.
pub(super) fn parse(text: &str) -> Result<Schedule, String> {
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    let every = |minutes: u32| -> Result<Schedule, String> {
        if (1..=10_080).contains(&minutes) {
            Ok(Schedule::Every { minutes })
        } else {
            Err("an interval is between 1 minute and 7 days".into())
        }
    };
    if t == "every minute" {
        return every(1);
    }
    if t == "hourly" || t == "every hour" {
        return Ok(Schedule::Cron(Cron { minutes: 1, hours: (1 << 24) - 1, days: ALL_DAYS, months: ALL_MONTHS, weekdays: 0x7f, any_day: true, any_weekday: true }));
    }
    if let Some(rest) = t.strip_prefix("hourly at ") {
        let minute: u32 = rest.trim_start_matches(':').parse().map_err(|_| format!("'{rest}' is not a minute"))?;
        if minute > 59 {
            return Err("the minute is 0 to 59".into());
        }
        return Ok(Schedule::Cron(Cron { minutes: 1 << minute, hours: (1 << 24) - 1, days: ALL_DAYS, months: ALL_MONTHS, weekdays: 0x7f, any_day: true, any_weekday: true }));
    }
    if let Some(rest) = t.strip_prefix("every ") {
        let mut words = rest.split(' ');
        if let (Some(n), Some(unit), None) = (words.next(), words.next(), words.next()) {
            if let Ok(n) = n.parse::<u32>() {
                return match unit {
                    "minute" | "minutes" | "min" | "mins" | "m" => every(n),
                    "hour" | "hours" | "h" => every(n.saturating_mul(60)),
                    "day" | "days" | "d" => every(n.saturating_mul(1440)),
                    _ => Err(format!("unknown unit '{unit}' (minutes, hours or days)")),
                };
            }
        }
    }
    for prefix in ["daily at ", "every day at "] {
        if let Some(rest) = t.strip_prefix(prefix) {
            let (h, m) = parse_clock(rest)?;
            return Ok(Schedule::Cron(daily_like(h, m, 0x7f)));
        }
    }
    for prefix in ["weekdays at ", "every weekday at "] {
        if let Some(rest) = t.strip_prefix(prefix) {
            let (h, m) = parse_clock(rest)?;
            return Ok(Schedule::Cron(daily_like(h, m, 0b0111110)));
        }
    }
    if let Some(rest) = t.strip_prefix("weekly on ") {
        let (days, time) = rest.split_once(" at ").ok_or("write it like 'weekly on mon,thu at 18:00'")?;
        let (mask, _) = parse_field(days, 0, 7, true)?;
        let mut weekdays = mask as u8;
        if mask & (1 << 7) != 0 {
            weekdays = (weekdays | 1) & 0x7f;
        }
        let (h, m) = parse_clock(time)?;
        return Ok(Schedule::Cron(daily_like(h, m, weekdays)));
    }
    parse_cron(&t).map(Schedule::Cron)
}

impl Cron {
    fn day_matches(&self, date: NaiveDate) -> bool {
        if self.months & (1 << date.month()) == 0 {
            return false;
        }
        let day = self.days & (1 << date.day()) != 0;
        let weekday = self.weekdays & (1 << date.weekday().num_days_from_sunday()) != 0;
        match (self.any_day, self.any_weekday) {
            (true, true) => true,
            (false, true) => day,
            (true, false) => weekday,
            (false, false) => day || weekday,
        }
    }
}

/// The first wall-clock minute strictly after `after` that the cron line allows (no time zone involved).
pub(super) fn next_wall(cron: &Cron, after: NaiveDateTime) -> Option<NaiveDateTime> {
    let mut date = after.date();
    for _ in 0..(366 * 8) {
        if cron.day_matches(date) {
            for hour in 0..24u32 {
                if cron.hours & (1 << hour) == 0 {
                    continue;
                }
                for minute in 0..60u32 {
                    if cron.minutes & (1 << minute) == 0 {
                        continue;
                    }
                    let candidate = date.and_hms_opt(hour, minute, 0)?;
                    if candidate > after {
                        return Some(candidate);
                    }
                }
            }
        }
        date = date.succ_opt()?;
    }
    None
}

/// What the local zone makes of a wall-clock time: one instant, two (autumn) or none (spring).
pub(super) type Zone<'a> = &'a dyn Fn(NaiveDateTime) -> LocalResult<i64>;

fn resolve(zone: Zone, wall: NaiveDateTime) -> Option<i64> {
    let mut wall = wall;
    for _ in 0..4 {
        match zone(wall) {
            LocalResult::Single(t) => return Some(t),
            LocalResult::Ambiguous(a, b) => return Some(a.min(b)),
            LocalResult::None => wall += chrono::Duration::hours(1),
        }
    }
    None
}

/// The first run after `after` (unix seconds). `wall_of` gives the local wall clock of an instant.
pub(super) fn next_run(schedule: &Schedule, after: i64, zone: Zone, wall_of: &dyn Fn(i64) -> Option<NaiveDateTime>) -> Option<i64> {
    match schedule {
        Schedule::Every { minutes } => Some(after + i64::from(*minutes) * 60),
        Schedule::Cron(cron) => {
            let mut wall = wall_of(after)?.with_second(0)?;
            for _ in 0..8 {
                wall = next_wall(cron, wall)?;
                if let Some(at) = resolve(zone, wall).filter(|at| *at > after) {
                    return Some(at);
                }
            }
            None
        }
    }
}

fn local_zone(wall: NaiveDateTime) -> LocalResult<i64> {
    match Local.from_local_datetime(&wall) {
        LocalResult::Single(t) => LocalResult::Single(t.timestamp()),
        LocalResult::Ambiguous(a, b) => LocalResult::Ambiguous(a.timestamp(), b.timestamp()),
        LocalResult::None => LocalResult::None,
    }
}

fn local_wall(at: i64) -> Option<NaiveDateTime> {
    Local.timestamp_opt(at, 0).single().map(|d| d.naive_local())
}

pub(super) fn next_run_local(schedule: &Schedule, after: i64) -> Option<i64> {
    next_run(schedule, after, &local_zone, &local_wall)
}

fn unix_now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

// ---------------------------------------------------------------------------------------------------------------
// Tasks, the run gate and the clock (pure parts)
// ---------------------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Task {
    pub id: String,
    pub name: String,
    pub prompt: String,
    pub when: String,
    /// `chat` or `agent`.
    pub mode: String,
    /// The folder of an agent run.
    pub folder: String,
    pub notify: bool,
    pub enabled: bool,
    pub created: i64,
    /// Where the result is sent besides the page (`webhook:URL`, `ntfy:URL`, `telegram:CHAT`); empty: nowhere.
    pub deliver: String,
}

impl Task {
    fn to_json(&self) -> Value {
        json!({"id": self.id, "name": self.name, "prompt": self.prompt, "when": self.when, "mode": self.mode, "folder": self.folder,
               "notify": self.notify, "enabled": self.enabled, "created": self.created, "deliver": self.deliver})
    }

    fn from_json(v: &Value) -> Option<Task> {
        Some(Task {
            id: v["id"].as_str().filter(|s| !s.is_empty())?.to_owned(),
            name: v["name"].as_str()?.to_owned(),
            prompt: v["prompt"].as_str()?.to_owned(),
            when: v["when"].as_str()?.to_owned(),
            mode: if v["mode"].as_str() == Some("agent") { "agent" } else { "chat" }.to_owned(),
            folder: v["folder"].as_str().unwrap_or_default().to_owned(),
            notify: v["notify"].as_bool().unwrap_or(false),
            enabled: v["enabled"].as_bool().unwrap_or(true),
            created: v["created"].as_i64().unwrap_or(0),
            deliver: v["deliver"].as_str().unwrap_or_default().to_owned(),
        })
    }
}

/// Checks the body of a create or update request; `existing` is the task with the same id, if any.
pub(super) fn task_from_request(body: &Value, existing: Option<&Task>, id: String, now: i64) -> Result<Task, String> {
    let name = body["name"].as_str().unwrap_or_default().trim().to_owned();
    if name.is_empty() || name.chars().count() > 80 {
        return Err("the name has 1 to 80 characters".into());
    }
    let prompt = body["prompt"].as_str().unwrap_or_default().trim().to_owned();
    if prompt.is_empty() || prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(format!("the prompt has 1 to {MAX_PROMPT_CHARS} characters"));
    }
    let when = body["when"].as_str().unwrap_or_default().trim().to_owned();
    parse(&when)?;
    let mode = match body["mode"].as_str().unwrap_or("chat") {
        "chat" => "chat",
        "agent" => "agent",
        other => return Err(format!("unknown mode '{other}' (chat or agent)")),
    };
    let folder = body["folder"].as_str().unwrap_or_default().trim().to_owned();
    if mode == "agent" {
        let path = Path::new(&folder);
        if !path.is_absolute() || !path.is_dir() {
            return Err("an agent task needs an existing absolute folder".into());
        }
    }
    let deliver = body["deliver"].as_str().unwrap_or_default().trim().to_owned();
    super::schedule_deliver::parse(&deliver)?;
    Ok(Task {
        id,
        name,
        prompt,
        when,
        mode: mode.to_owned(),
        folder: if mode == "agent" { folder } else { String::new() },
        notify: body["notify"].as_bool().unwrap_or(false),
        enabled: body["enabled"].as_bool().unwrap_or(true),
        created: existing.map_or(now, |t| t.created),
        deliver,
    })
}

/// Which tasks are running. A task starts only if it is not running already.
pub(super) struct Gate(Mutex<Vec<String>>);

impl Gate {
    pub const fn new() -> Gate {
        Gate(Mutex::new(Vec::new()))
    }
    pub fn begin(&self, id: &str) -> bool {
        let mut running = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if running.iter().any(|r| r == id) {
            return false;
        }
        running.push(id.to_owned());
        true
    }
    pub fn end(&self, id: &str) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).retain(|r| r != id);
    }
    pub fn is_running(&self, id: &str) -> bool {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|r| r == id)
    }
}

/// The clock: given the tasks and the time, which are due now. `next` remembers the next run of each task (with the
/// schedule text it was computed for); a task whose schedule changed is planned again.
pub(super) fn due(tasks: &[Task], next: &mut HashMap<String, (String, i64)>, now: i64, next_fn: &dyn Fn(&Schedule, i64) -> Option<i64>) -> Vec<String> {
    next.retain(|id, _| tasks.iter().any(|t| &t.id == id && t.enabled));
    let mut out = Vec::new();
    for task in tasks.iter().filter(|t| t.enabled) {
        let Ok(schedule) = parse(&task.when) else { continue };
        let planned = match next.get(&task.id) {
            Some((when, at)) if *when == task.when => Some(*at),
            _ => None,
        };
        match planned {
            Some(at) if now >= at => {
                if now - at <= MISSED_GRACE_SECS {
                    out.push(task.id.clone());
                }
                next.insert(task.id.clone(), (task.when.clone(), next_fn(&schedule, now).unwrap_or(i64::MAX)));
            }
            Some(_) => {}
            None => {
                next.insert(task.id.clone(), (task.when.clone(), next_fn(&schedule, now).unwrap_or(i64::MAX)));
            }
        }
    }
    out
}

/// Adds a result to the list of a task and keeps the newest `keep`.
pub(super) fn push_result(results: &mut Vec<Value>, result: Value, keep: usize) {
    results.insert(0, result);
    results.truncate(keep);
}

// ---------------------------------------------------------------------------------------------------------------
// Running a task (the model is reached through closures, so the loop is tested without one)
// ---------------------------------------------------------------------------------------------------------------

type NetTool<'a> = &'a dyn Fn(&str, &Value) -> Option<Result<String, String>>;

pub(super) struct Env<'a> {
    /// Sends a chat-completions body and returns the reply.
    pub complete: &'a dyn Fn(Value) -> Result<Value, String>,
    /// Runs a network tool; `None` when the name is not one.
    pub net: NetTool<'a>,
}

fn system_prompt(task: &Task) -> String {
    let now = Local::now().format("%A, %Y-%m-%d %H:%M");
    let mut text = format!(
        "You run as a scheduled task named \"{}\" of the fLLM app. Nobody is watching and nobody can answer questions, so do the task completely and finish with the result as a short report in Markdown. Local time now: {now}.",
        task.name
    );
    if task.mode == "agent" {
        text.push_str(" You may only read and search (list_dir, read_file, grep, glob) inside the folder and look things up on the web (web_fetch, web_search). You cannot change files or run commands.");
    }
    text
}

fn message_text(message: &Value) -> String {
    match &message["content"] {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

fn run_chat(task: &Task, env: &Env) -> Result<String, String> {
    let body = json!({"messages": [{"role": "system", "content": system_prompt(task)}, {"role": "user", "content": task.prompt}]});
    let reply = (env.complete)(body)?;
    let text = message_text(&reply["choices"][0]["message"]);
    if text.trim().is_empty() {
        return Err("the model gave an empty answer".into());
    }
    Ok(text)
}

fn tool_schemas() -> Vec<Value> {
    super::agent_tools::catalog()
        .into_iter()
        .filter(|t| t["function"]["name"].as_str().is_some_and(|n| READ_ONLY_TOOLS.contains(&n)))
        .map(|mut t| {
            if let Some(o) = t.as_object_mut() {
                o.remove("permission");
            }
            t
        })
        .collect()
}

fn run_agent(task: &Task, env: &Env) -> Result<String, String> {
    if super::agent::agent_disabled() {
        return Err("the agent is switched off on this server (FLLM_NO_AGENT)".into());
    }
    let session = super::agent::AgentSession::default();
    session.set_root(&task.folder)?;
    let tools = tool_schemas();
    let mut messages = vec![json!({"role": "system", "content": system_prompt(task)}), json!({"role": "user", "content": task.prompt})];
    for _ in 0..MAX_AGENT_STEPS {
        let reply = (env.complete)(json!({"messages": messages, "tools": tools}))?;
        let message = &reply["choices"][0]["message"];
        let calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
        if calls.is_empty() {
            let text = message_text(message);
            if text.trim().is_empty() {
                return Err("the model gave an empty answer".into());
            }
            return Ok(text);
        }
        messages.push(json!({"role": "assistant", "content": message_text(message), "tool_calls": calls}));
        for call in &calls {
            let name = call["function"]["name"].as_str().unwrap_or_default();
            let args: Value = match &call["function"]["arguments"] {
                Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
                other if other.is_object() => other.clone(),
                _ => json!({}),
            };
            let output = if !READ_ONLY_TOOLS.contains(&name) {
                format!("Error: the tool '{name}' is not available in a scheduled task (read-only: {})", READ_ONLY_TOOLS.join(", "))
            } else {
                let result = if NET_TOOLS.contains(&name) { (env.net)(name, &args).unwrap_or_else(|| Err("no such tool".into())) } else { session.run_tool(name, &args) };
                match result {
                    Ok(text) => text,
                    Err(e) => format!("Error: {e}"),
                }
            };
            messages.push(json!({"role": "tool", "tool_call_id": call["id"], "content": output.chars().take(30_000).collect::<String>()}));
        }
    }
    // out of steps: ask for the report with what was found
    messages.push(json!({"role": "user", "content": "Stop using tools now and write the final report with what you found."}));
    let text = message_text(&(env.complete)(json!({"messages": messages}))?["choices"][0]["message"]);
    if text.trim().is_empty() {
        return Err("the model did not finish the task".into());
    }
    Ok(text)
}

pub(super) fn execute(task: &Task, env: &Env) -> Result<String, String> {
    if task.mode == "agent" { run_agent(task, env) } else { run_chat(task, env) }
}

// ---------------------------------------------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------------------------------------------

static IO: Mutex<()> = Mutex::new(());
static GATE: Gate = Gate::new();
static CLOCK_RUNNING: AtomicBool = AtomicBool::new(false);

fn tasks_file() -> PathBuf {
    crate::hub::studio_data_file("schedule.json")
}

fn results_file() -> PathBuf {
    crate::hub::studio_data_file("schedule-results.json")
}

fn read_json_file(file: &Path, key: &str) -> Vec<Value> {
    fs::read_to_string(file)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v[key].as_array().cloned())
        .unwrap_or_default()
}

fn write_json_file(file: &Path, key: &str, items: Vec<Value>) -> Result<(), String> {
    if let Some(dir) = file.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let tmp = file.with_extension("json.tmp");
    fs::write(&tmp, json!({ key: items }).to_string()).map_err(|e| e.to_string())?;
    fs::rename(&tmp, file).map_err(|e| e.to_string())
}

fn load_tasks() -> Vec<Task> {
    read_json_file(&tasks_file(), "tasks").iter().filter_map(Task::from_json).collect()
}

fn save_tasks(tasks: &[Task]) -> Result<(), String> {
    write_json_file(&tasks_file(), "tasks", tasks.iter().map(Task::to_json).collect())
}

fn all_results() -> Vec<Value> {
    read_json_file(&results_file(), "results")
}

fn store_result(task: &Task, ok: bool, text: String, seconds: u64, delivery: Option<String>) {
    let _guard = IO.lock().unwrap_or_else(|e| e.into_inner());
    let all = all_results();
    let (mut mine, others): (Vec<Value>, Vec<Value>) = all.into_iter().partition(|r| r["task_id"] == task.id.as_str());
    push_result(
        &mut mine,
        json!({"id": new_id("res"), "task_id": task.id, "task_name": task.name, "mode": task.mode, "notify": task.notify, "time": unix_now(),
               "ok": ok, "seconds": seconds, "delivery": delivery, "text": text.chars().take(MAX_RESULT_CHARS).collect::<String>()}),
        KEEP_RESULTS,
    );
    let mut merged = others;
    merged.extend(mine);
    let _ = write_json_file(&results_file(), "results", merged);
}

// ---------------------------------------------------------------------------------------------------------------
// The clock thread and the real model connection
// ---------------------------------------------------------------------------------------------------------------

struct Context {
    state: Weak<AppState>,
    handle: tokio::runtime::Handle,
}

static CONTEXT: OnceLock<Context> = OnceLock::new();

/// Starts the clock of the server. `state` is held weakly, so the clock never keeps a stopping server alive.
pub(in crate::server) fn start(state: Weak<AppState>, handle: tokio::runtime::Handle) {
    if CONTEXT.set(Context { state, handle }).is_err() {
        return;
    }
    CLOCK_RUNNING.store(true, Ordering::Relaxed);
    let _ = thread::Builder::new().name("fllm-schedule".into()).spawn(|| {
        let mut next: HashMap<String, (String, i64)> = HashMap::new();
        loop {
            thread::sleep(std::time::Duration::from_secs(TICK_SECS));
            let tasks = load_tasks();
            for id in due(&tasks, &mut next, unix_now(), &next_run_local) {
                if let Some(task) = tasks.iter().find(|t| t.id == id) {
                    let _ = launch(task.clone());
                }
            }
        }
    });
}

fn run_with_server(task: &Task) -> Result<String, String> {
    let ctx = CONTEXT.get().ok_or("the scheduler is not running (no model is loaded)")?;
    let state = ctx.state.upgrade().ok_or("the server is stopping")?;
    let complete = |mut body: Value| -> Result<Value, String> {
        body["model"] = json!(state.model_name);
        body["stream"] = json!(false);
        ctx.handle.block_on(async {
            let response = chat_completions(State(state.clone()), Json(body)).await;
            read_json(response).await.map_err(|(_, message)| message)
        })
    };
    let net = |name: &str, args: &Value| ctx.handle.block_on(super::agent_net::run_tool(name, args));
    execute(task, &Env { complete: &complete, net: &net })
}

/// Sends the result to the task's `deliver` target; `None` when it has none, else `"sent"` or `"failed: reason"`.
fn deliver_result(task: &Task, ok: bool, text: &str) -> Option<String> {
    let target = super::schedule_deliver::parse(&task.deliver).ok().flatten()?;
    Some(super::schedule_deliver::deliver(&target, &task.name, ok, text, unix_now()))
}

/// Runs a task on its own thread. Fails if it is running already or the clock is not running.
fn launch(task: Task) -> Result<(), String> {
    if CONTEXT.get().is_none() {
        return Err("the scheduler is not running (no model is loaded)".into());
    }
    if !GATE.begin(&task.id) {
        return Err("this task is running already".into());
    }
    let id = task.id.clone();
    let spawned = thread::Builder::new().name("fllm-schedule-task".into()).spawn(move || {
        let started = Instant::now();
        let result = std::panic::catch_unwind(|| run_with_server(&task)).unwrap_or_else(|_| Err("the task stopped unexpectedly".into()));
        let (ok, text) = match result {
            Ok(text) => (true, text),
            Err(e) => (false, e),
        };
        let delivery = deliver_result(&task, ok, &text);
        store_result(&task, ok, text, started.elapsed().as_secs(), delivery);
        GATE.end(&task.id);
    });
    if spawned.is_err() {
        GATE.end(&id);
        return Err("cannot start a thread".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------------------------------------------

fn task_view(task: &Task, last: Option<&Value>) -> Value {
    let mut v = task.to_json();
    let next = if task.enabled { parse(&task.when).ok().and_then(|s| next_run_local(&s, unix_now())) } else { None };
    v["next_run"] = json!(next);
    v["running"] = json!(GATE.is_running(&task.id));
    v["last"] = last.map_or(Value::Null, |r| json!({"time": r["time"], "ok": r["ok"], "delivery": r["delivery"]}));
    v
}

/// `GET /api/studio/schedule`: `{tasks, clock_running, note}`.
pub(in crate::server) async fn list() -> Response {
    let tasks = load_tasks();
    let results = all_results();
    let views: Vec<Value> = tasks
        .iter()
        .map(|t| {
            let last = results.iter().filter(|r| r["task_id"] == t.id.as_str()).max_by_key(|r| r["time"].as_i64().unwrap_or(0));
            task_view(t, last)
        })
        .collect();
    Json(json!({"tasks": views, "clock_running": CLOCK_RUNNING.load(Ordering::Relaxed),
                "note": "Tasks run only while the app is running; a run missed while it was closed is not made up."}))
    .into_response()
}

/// `POST /api/studio/schedule` `{id?, name, prompt, when, mode?, folder?, notify?, enabled?}` creates or updates a task.
pub(in crate::server) async fn upsert(Json(body): Json<Value>) -> Response {
    let _guard = IO.lock().unwrap_or_else(|e| e.into_inner());
    let mut tasks = load_tasks();
    let given = body["id"].as_str().filter(|s| !s.is_empty());
    let existing = given.and_then(|id| tasks.iter().find(|t| t.id == id)).cloned();
    if given.is_some() && existing.is_none() {
        return api_error(StatusCode::NOT_FOUND, "no such task");
    }
    if existing.is_none() && tasks.len() >= MAX_TASKS {
        return api_error(StatusCode::BAD_REQUEST, &format!("at most {MAX_TASKS} tasks"));
    }
    let id = existing.as_ref().map_or_else(|| new_id("task"), |t| t.id.clone());
    let task = match task_from_request(&body, existing.as_ref(), id, unix_now()) {
        Ok(task) => task,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
    };
    match tasks.iter_mut().find(|t| t.id == task.id) {
        Some(slot) => *slot = task.clone(),
        None => tasks.push(task.clone()),
    }
    match save_tasks(&tasks) {
        Ok(()) => Json(json!({"ok": true, "task": task_view(&task, None)})).into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
    }
}

/// `DELETE /api/studio/schedule?id=` removes a task and its results.
pub(in crate::server) async fn remove(Query(query): Query<HashMap<String, String>>) -> Response {
    let id = query.get("id").cloned().unwrap_or_default();
    let _guard = IO.lock().unwrap_or_else(|e| e.into_inner());
    let mut tasks = load_tasks();
    let before = tasks.len();
    tasks.retain(|t| t.id != id);
    if tasks.len() == before {
        return api_error(StatusCode::NOT_FOUND, "no such task");
    }
    let kept: Vec<Value> = all_results().into_iter().filter(|r| r["task_id"] != id.as_str()).collect();
    let _ = write_json_file(&results_file(), "results", kept);
    match save_tasks(&tasks) {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e),
    }
}

/// `POST /api/studio/schedule/run` `{id}` starts a task now.
pub(in crate::server) async fn run_now(Json(body): Json<Value>) -> Response {
    let id = body["id"].as_str().unwrap_or_default();
    let Some(task) = load_tasks().into_iter().find(|t| t.id == id) else { return api_error(StatusCode::NOT_FOUND, "no such task") };
    match launch(task) {
        Ok(()) => (StatusCode::ACCEPTED, Json(json!({"ok": true}))).into_response(),
        Err(e) if e.contains("running already") => api_error(StatusCode::CONFLICT, &e),
        Err(e) => api_error(StatusCode::SERVICE_UNAVAILABLE, &e),
    }
}

/// `GET /api/studio/schedule/results?id=&since=` answers the results, newest first, of one task or of all.
pub(in crate::server) async fn results(Query(query): Query<HashMap<String, String>>) -> Response {
    let id = query.get("id").cloned().unwrap_or_default();
    let since = query.get("since").and_then(|s| s.parse::<i64>().ok()).unwrap_or(0);
    let mut out: Vec<Value> = all_results()
        .into_iter()
        .filter(|r| (id.is_empty() || r["task_id"] == id.as_str()) && r["time"].as_i64().unwrap_or(0) > since)
        .collect();
    out.sort_by_key(|r| std::cmp::Reverse(r["time"].as_i64().unwrap_or(0)));
    Json(json!({"results": out})).into_response()
}

/// `GET /api/studio/schedule/check?when=` tells whether a schedule text is understood and when it runs next.
pub(in crate::server) async fn check(Query(query): Query<HashMap<String, String>>) -> Response {
    match parse(query.get("when").map_or("", String::as_str)) {
        Err(e) => Json(json!({"ok": false, "error": e})).into_response(),
        Ok(schedule) => {
            let mut at = unix_now();
            let mut next = Vec::new();
            for _ in 0..3 {
                match next_run_local(&schedule, at) {
                    Some(t) => {
                        next.push(t);
                        at = t;
                    }
                    None => break,
                }
            }
            Json(json!({"ok": true, "next": next})).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wall(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(y, mo, d).unwrap().and_hms_opt(h, mi, 0).unwrap()
    }

    fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> i64 {
        wall(y, mo, d, h, mi).and_utc().timestamp()
    }

    // A zone like US Eastern in 2026: UTC-5, UTC-4 from 2026-03-08 07:00 UTC (02:00 -> 03:00) to 2026-11-01 06:00 UTC (02:00 -> 01:00).
    fn zone(w: NaiveDateTime) -> LocalResult<i64> {
        let as_utc = w.and_utc().timestamp();
        let (gap_start, gap_end) = (wall(2026, 3, 8, 2, 0), wall(2026, 3, 8, 3, 0));
        let (twice_start, twice_end) = (wall(2026, 11, 1, 1, 0), wall(2026, 11, 1, 2, 0));
        if w >= gap_start && w < gap_end {
            LocalResult::None
        } else if w >= twice_start && w < twice_end {
            LocalResult::Ambiguous(as_utc + 4 * 3600, as_utc + 5 * 3600)
        } else if w >= gap_end && w < twice_start {
            LocalResult::Single(as_utc + 4 * 3600)
        } else {
            LocalResult::Single(as_utc + 5 * 3600)
        }
    }

    fn wall_of(at: i64) -> Option<NaiveDateTime> {
        let offset = if at >= utc(2026, 3, 8, 7, 0) && at < utc(2026, 11, 1, 6, 0) { -4 } else { -5 };
        chrono::DateTime::from_timestamp(at + offset * 3600, 0).map(|d| d.naive_utc())
    }

    fn next(text: &str, after: i64) -> Option<i64> {
        next_run(&parse(text).unwrap(), after, &zone, &wall_of)
    }

    #[test]
    fn friendly_forms_parse() {
        assert_eq!(parse("every 30 minutes").unwrap(), Schedule::Every { minutes: 30 });
        assert_eq!(parse("Every  2 hours").unwrap(), Schedule::Every { minutes: 120 });
        assert_eq!(parse("every minute").unwrap(), Schedule::Every { minutes: 1 });
        assert_eq!(parse("every 1 day").unwrap(), Schedule::Every { minutes: 1440 });
        assert_eq!(parse("daily at 8:30pm").unwrap(), parse("daily at 20:30").unwrap());
        assert_eq!(parse("daily at 12am").unwrap(), parse("daily at 00:00").unwrap());
        assert_eq!(parse("daily at 12pm").unwrap(), parse("daily at 12:00").unwrap());
        assert_eq!(parse("weekdays at 9am").unwrap(), parse("0 9 * * 1-5").unwrap());
        assert_eq!(parse("weekly on mon,thu at 18:00").unwrap(), parse("0 18 * * 1,4").unwrap());
        assert_eq!(parse("hourly at :15").unwrap(), parse("15 * * * *").unwrap());
        assert_eq!(parse("hourly").unwrap(), parse("0 * * * *").unwrap());
    }

    #[test]
    fn bad_schedules_are_refused() {
        for text in ["", "sometimes", "every 0 minutes", "every 20000 minutes", "every 5 parsecs", "daily at 25:00", "daily at 10:75", "daily", "* * * *",
                     "61 * * * *", "* 24 * * *", "*/0 * * * *", "5-1 * * * *", "* * 32 * *", "* * * 13 *", "* * * * 8", "weekly on mon", "hourly at 99", "a b c d e"] {
            assert!(parse(text).is_err(), "{text:?} should be refused");
        }
    }

    #[test]
    fn cron_fields_cover_lists_ranges_and_steps() {
        let Schedule::Cron(c) = parse("*/15 9-17 * * mon-fri").unwrap() else { panic!() };
        assert_eq!(c.minutes, 1 | 1 << 15 | 1 << 30 | 1 << 45);
        assert_eq!(c.hours.count_ones(), 9);
        assert_eq!(c.weekdays, 0b0111110);
        let Schedule::Cron(c) = parse("0 0 * * 7").unwrap() else { panic!() };
        assert_eq!(c.weekdays, 1, "7 is Sunday like 0");
        let Schedule::Cron(c) = parse("5/20 * * * *").unwrap() else { panic!() };
        assert_eq!(c.minutes, 1 << 5 | 1 << 25 | 1 << 45);
    }

    #[test]
    fn next_run_follows_days_and_weekdays() {
        // Friday evening: the next run of a work-hours line is Monday morning
        let friday = utc(2026, 10, 23, 21, 50); // 17:50 local (UTC-4)
        assert_eq!(next("*/15 9-17 * * mon-fri", friday), Some(utc(2026, 10, 26, 13, 0)));
        // day of month or weekday when both are given
        let monday = utc(2026, 10, 26, 4, 0);
        assert_eq!(next("0 0 13 * fri", monday), Some(utc(2026, 10, 30, 4, 0)));
        // a line that never matches (31 February) gives no run
        assert_eq!(next("0 0 31 2 *", monday), None);
        // every N minutes counts from now
        assert_eq!(next("every 45 minutes", 1000), Some(1000 + 2700));
        // strictly after
        assert_eq!(next("daily at 08:00", utc(2026, 10, 26, 12, 0)), Some(utc(2026, 10, 27, 12, 0)));
    }

    #[test]
    fn the_hour_skipped_in_spring_runs_after_the_gap() {
        let before = utc(2026, 3, 7, 17, 0);
        // 02:30 does not exist on 8 March: the task runs at 03:30 local that day, not never and not twice
        assert_eq!(next("daily at 02:30", before), Some(utc(2026, 3, 8, 7, 30)));
        assert_eq!(next("daily at 02:30", utc(2026, 3, 8, 7, 30)), Some(utc(2026, 3, 9, 6, 30)));
    }

    #[test]
    fn the_hour_repeated_in_autumn_runs_once() {
        let before = utc(2026, 10, 31, 16, 0);
        let first = next("daily at 01:30", before).unwrap();
        assert_eq!(first, utc(2026, 11, 1, 5, 30), "the first 01:30");
        assert_eq!(next("daily at 01:30", first), Some(utc(2026, 11, 2, 6, 30)), "the second 01:30 of that night is skipped");
        // an hourly line always moves forward
        let mut at = utc(2026, 11, 1, 3, 0);
        for _ in 0..8 {
            let n = next("hourly", at).unwrap();
            assert!(n > at && n - at <= 2 * 3600);
            at = n;
        }
    }

    #[test]
    fn requests_are_validated() {
        let dir = std::env::temp_dir();
        let ok = json!({"name": " Morning ", "prompt": "Summarise the news", "when": "daily at 08:00"});
        let t = task_from_request(&ok, None, "task_1".into(), 5).unwrap();
        assert_eq!((t.name.as_str(), t.mode.as_str(), t.enabled, t.created), ("Morning", "chat", true, 5));
        for bad in [
            json!({"name": "", "prompt": "x", "when": "hourly"}),
            json!({"name": "n", "prompt": " ", "when": "hourly"}),
            json!({"name": "n", "prompt": "x", "when": "whenever"}),
            json!({"name": "n", "prompt": "x", "when": "hourly", "mode": "shell"}),
            json!({"name": "n", "prompt": "x", "when": "hourly", "mode": "agent"}),
            json!({"name": "n", "prompt": "x", "when": "hourly", "mode": "agent", "folder": "relative"}),
        ] {
            assert!(task_from_request(&bad, None, "i".into(), 0).is_err(), "{bad}");
        }
        let agent = json!({"name": "n", "prompt": "x", "when": "hourly", "mode": "agent", "folder": dir.to_str().unwrap()});
        assert_eq!(task_from_request(&agent, Some(&t), "task_1".into(), 99).unwrap().created, 5);
        assert_eq!(Task::from_json(&t.to_json()).unwrap(), t);
        // the delivery target is validated and kept
        let with = json!({"name": "n", "prompt": "x", "when": "hourly", "deliver": " ntfy:http://10.7.0.5:8080/t "});
        let t2 = task_from_request(&with, None, "task_2".into(), 0).unwrap();
        assert_eq!(t2.deliver, "ntfy:http://10.7.0.5:8080/t");
        assert_eq!(Task::from_json(&t2.to_json()).unwrap(), t2);
        assert!(task_from_request(&json!({"name": "n", "prompt": "x", "when": "hourly", "deliver": "mail:a@b.c"}), None, "i".into(), 0).is_err());
        assert_eq!(t.deliver, "", "old tasks without the field deliver nowhere");
    }

    #[test]
    fn one_task_never_runs_twice_at_once() {
        let gate = Gate::new();
        assert!(gate.begin("a"));
        assert!(!gate.begin("a"));
        assert!(gate.begin("b"));
        assert!(gate.is_running("a"));
        gate.end("a");
        assert!(!gate.is_running("a"));
        assert!(gate.begin("a"));
    }

    fn task(id: &str, when: &str, enabled: bool) -> Task {
        Task { id: id.into(), name: id.into(), prompt: "p".into(), when: when.into(), mode: "chat".into(), folder: String::new(), notify: false, enabled, created: 0, deliver: String::new() }
    }

    #[test]
    fn the_clock_fires_each_task_once_per_slot_and_skips_missed_runs() {
        let plan = |_: &Schedule, now: i64| Some(now + 600);
        let mut map = HashMap::new();
        let tasks = vec![task("a", "every 10 minutes", true), task("off", "every 10 minutes", false), task("bad", "nonsense", true)];
        assert!(due(&tasks, &mut map, 1000, &plan).is_empty(), "the first look only plans");
        assert!(due(&tasks, &mut map, 1300, &plan).is_empty());
        assert_eq!(due(&tasks, &mut map, 1600, &plan), vec!["a".to_owned()]);
        assert!(due(&tasks, &mut map, 1601, &plan).is_empty(), "planned again, not due twice");
        // the computer slept for a day: the run that was due is skipped and planned again
        assert!(due(&tasks, &mut map, 1600 + 86_400, &plan).is_empty());
        assert_eq!(map["a"].1, 1600 + 86_400 + 600);
        // a changed schedule is planned again
        let changed = vec![task("a", "every 5 minutes", true)];
        assert!(due(&changed, &mut map, 1600 + 86_400 + 700, &plan).is_empty());
        assert_eq!(map["a"].0, "every 5 minutes");
        // a deleted task is forgotten
        due(&[], &mut map, 0, &plan);
        assert!(map.is_empty());
    }

    #[test]
    fn only_the_last_results_are_kept() {
        let mut list = Vec::new();
        for i in 0..30 {
            push_result(&mut list, json!({"n": i}), KEEP_RESULTS);
        }
        assert_eq!(list.len(), KEEP_RESULTS);
        assert_eq!(list[0]["n"], 29);
    }

    #[test]
    fn a_finished_task_delivers_its_result_and_records_how_it_went() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/hook", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
            let mut got = vec![0u8; 8192];
            let n = stream.read(&mut got).unwrap_or(0);
            stream.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n").unwrap();
            String::from_utf8_lossy(&got[..n]).into_owned()
        });
        let mut t = task("d", "hourly", true);
        assert_eq!(deliver_result(&t, true, "x"), None, "a task without a target delivers nothing");
        t.deliver = format!("webhook:{url}");
        assert_eq!(deliver_result(&t, true, "the report").as_deref(), Some("sent"));
        assert!(server.join().unwrap().contains("the report"));
        t.deliver = "webhook:http://127.0.0.1:1/x".into();
        assert!(deliver_result(&t, true, "x").unwrap().starts_with("failed:"), "a dead target is recorded, the task result stays");
    }

    #[test]
    fn a_chat_task_returns_the_answer() {
        let t = task("c", "hourly", true);
        let complete = |body: Value| -> Result<Value, String> {
            assert_eq!(body["messages"][1]["content"], "p");
            assert!(body.get("tools").is_none());
            Ok(json!({"choices": [{"message": {"role": "assistant", "content": "Done."}}]}))
        };
        let net = |_: &str, _: &Value| None;
        assert_eq!(execute(&t, &Env { complete: &complete, net: &net }).unwrap(), "Done.");
        let empty = |_: Value| -> Result<Value, String> { Ok(json!({"choices": [{"message": {"content": ""}}]})) };
        assert!(execute(&t, &Env { complete: &empty, net: &net }).is_err());
    }

    #[test]
    fn an_agent_task_reads_but_cannot_write() {
        let dir = std::env::temp_dir().join(format!("fllm-sched-{}-{}", std::process::id(), new_id("t")));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("notes.txt"), "the secret word is plum\n").unwrap();
        let mut t = task("g", "hourly", true);
        t.mode = "agent".into();
        t.folder = dir.to_str().unwrap().into();
        let step = std::cell::Cell::new(0);
        let complete = |body: Value| -> Result<Value, String> {
            let n = step.get();
            step.set(n + 1);
            let names: Vec<&str> = body["tools"].as_array().map(|a| a.iter().filter_map(|t| t["function"]["name"].as_str()).collect()).unwrap_or_default();
            if n == 0 {
                assert_eq!(names.len(), READ_ONLY_TOOLS.len(), "{names:?}");
                assert!(!names.contains(&"write_file") && !names.contains(&"bash"));
                assert!(body["tools"][0].get("permission").is_none());
                return Ok(json!({"choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"notes.txt\"}"}},
                    {"id": "2", "type": "function", "function": {"name": "write_file", "arguments": "{\"path\": \"x.txt\", \"content\": \"no\"}"}}]}}]}));
            }
            let messages = body["messages"].as_array().unwrap();
            let read = messages[messages.len() - 2]["content"].as_str().unwrap();
            let refused = messages[messages.len() - 1]["content"].as_str().unwrap();
            assert!(read.contains("plum"), "{read}");
            assert!(refused.contains("not available"), "{refused}");
            Ok(json!({"choices": [{"message": {"role": "assistant", "content": "The word is plum."}}]}))
        };
        let net = |_: &str, _: &Value| None;
        let text = execute(&t, &Env { complete: &complete, net: &net }).unwrap();
        assert_eq!(text, "The word is plum.");
        assert!(!dir.join("x.txt").exists());
        // the tool loop is bounded
        let endless = |body: Value| -> Result<Value, String> {
            if body.get("tools").is_none() {
                return Ok(json!({"choices": [{"message": {"content": "Gave up."}}]}));
            }
            Ok(json!({"choices": [{"message": {"content": "", "tool_calls": [{"id": "x", "function": {"name": "list_dir", "arguments": "{}"}}]}}]}))
        };
        assert_eq!(execute(&t, &Env { complete: &endless, net: &net }).unwrap(), "Gave up.");
    }
}
