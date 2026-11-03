//! Proactive wake-ups for the messaging gateway (the heartbeat idea of OpenClaw): every so often the agent reads a checklist,
//! `HEARTBEAT.md` in its folder, checks what the list asks for with its tools, and writes to you only when something needs
//! attention. When all is well it answers `HEARTBEAT_OK`, which is dropped, so a quiet system produces no messages at all.
//!
//! ```markdown
//! # Heartbeat
//! - Is the disk of /srv more than 85% full? (run `df -h /srv`)
//! - Did the nightly backup log (backup.log) end with an error?
//! - Any new line with "ERROR" in app.log since yesterday?
//! ```
//!
//! `fllm-rs gateway telegram --chat 123 --heartbeat 30m --heartbeat-chat 123` runs the checklist every 30 minutes (the first run one
//! interval after the start) and sends a message to the listed chats only when there is something to say. Options:
//! `--heartbeat-hours 08:00-22:00` (local time; outside it the heartbeat sleeps), `--heartbeat-file PATH`, `--heartbeat-act`
//! (let the run use the permissions of the gateway; by default a heartbeat run may only read), `--heartbeat-idle 10m` (wait until
//! nobody has written for that long). A heartbeat run has no memory of earlier ones: every wake-up starts from the checklist.
//! A file with no list items counts as empty and costs no model call.
use crate::agent_cli::{self, Backend, LoopConfig, Status};
use crate::gateway::Chat;
use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant};

pub const OK_TOKEN: &str = "HEARTBEAT_OK";
const MAX_CHECKLIST_CHARS: usize = 6_000;
/// An answer that only acknowledges (the token plus at most this many other characters) is not worth a message.
const ACK_SLACK: usize = 40;

#[derive(Debug, Clone, PartialEq)]
pub struct Heartbeat {
    pub every: Duration,
    /// Chats that receive an alert.
    pub chats: Vec<String>,
    pub file: Option<String>,
    /// Minutes since midnight: the window in which wake-ups happen (`start` may be after `end` for a night window).
    pub hours: Option<(u32, u32)>,
    pub act: bool,
    pub idle: Duration,
}

/// `30m`, `2h`, `45s`, `1d`; a bare number is minutes. 1 minute at least.
pub fn parse_interval(text: &str) -> Result<Duration, String> {
    let t = text.trim().to_ascii_lowercase();
    let (num, unit) = match t.chars().last() {
        Some(c) if c.is_ascii_alphabetic() => (&t[..t.len() - 1], c),
        _ => (t.as_str(), 'm'),
    };
    let n: u64 = num.parse().map_err(|_| format!("'{text}' is not an interval like 30m, 2h or 1d"))?;
    let secs = match unit {
        's' => n,
        'm' => n * 60,
        'h' => n * 3600,
        'd' => n * 86_400,
        _ => return Err(format!("'{text}' is not an interval like 30m, 2h or 1d")),
    };
    if !(60..=7 * 86_400).contains(&secs) {
        return Err("the interval is between 1 minute and 7 days".into());
    }
    Ok(Duration::from_secs(secs))
}

/// `08:00-22:00` as minutes since midnight.
pub fn parse_hours(text: &str) -> Result<(u32, u32), String> {
    let bad = || format!("'{text}' is not a time window like 08:00-22:00");
    let (a, b) = text.trim().split_once('-').ok_or_else(bad)?;
    let minutes = |s: &str| -> Option<u32> {
        let (h, m) = s.trim().split_once(':')?;
        let (h, m): (u32, u32) = (h.parse().ok()?, m.parse().ok()?);
        (h < 24 && m < 60).then_some(h * 60 + m)
    };
    match (minutes(a), minutes(b)) {
        (Some(s), Some(e)) if s != e => Ok((s, e)),
        _ => Err(bad()),
    }
}

/// Is `minute` (since midnight) inside the window? A window that ends before it starts crosses midnight.
pub fn in_window(window: Option<(u32, u32)>, minute: u32) -> bool {
    match window {
        None => true,
        Some((s, e)) if s < e => minute >= s && minute < e,
        Some((s, e)) => minute >= s || minute < e,
    }
}

/// The checklist, or `None` when the text has no list item or heading content that asks for anything (an empty file, only headings
/// and comments). A heartbeat with nothing to check must cost nothing.
pub fn checklist(text: &str) -> Option<String> {
    let body: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#') && !l.trim_start().starts_with("<!--")).collect();
    if body.is_empty() {
        return None;
    }
    Some(text.trim().chars().take(MAX_CHECKLIST_CHARS).collect())
}

pub fn prompt(list: &str, now: &str) -> String {
    format!(
        "This is a scheduled check, not a message from a person. The time is {now}. Work through this checklist with your tools, one item after the other, and look at the real state (files, commands, logs); do not guess.\n\n{list}\n\nIf nothing needs attention, reply with exactly {OK_TOKEN} and nothing else. If something needs attention, reply with a short message for the owner: what is wrong, the evidence in one line each, and what you suggest. Do not list items that are fine."
    )
}

/// What to send to the chats, or `None` when the answer only says that all is well.
pub fn alert_from(answer: &str) -> Option<String> {
    let t = answer.trim();
    if t.is_empty() {
        return None;
    }
    if t.contains(OK_TOKEN) {
        let rest: String = t.replace(OK_TOKEN, "").chars().filter(|c| c.is_alphanumeric()).collect();
        if rest.chars().count() <= ACK_SLACK {
            return None;
        }
        // the token amid a long text: the text is the news, the token is noise
        return Some(t.replace(OK_TOKEN, "").trim().to_owned());
    }
    Some(t.to_owned())
}

/// Tracks when the next wake-up is due.
pub struct Clock {
    next: Instant,
}

impl Clock {
    pub fn new(every: Duration, now: Instant) -> Self {
        Self { next: now + every }
    }
    pub fn due(&self, now: Instant) -> bool {
        now >= self.next
    }
    pub fn done(&mut self, every: Duration, now: Instant) {
        self.next = now + every;
    }
}

/// One wake-up. Returns what happened, for the log.
pub fn run_once(hb: &Heartbeat, folder: &Path, cfg: &LoopConfig, system: &str, backend: &mut dyn Backend, chat: &mut dyn Chat, now_text: &str) -> String {
    let file = hb.file.as_deref().map_or_else(|| folder.join("HEARTBEAT.md"), |f| folder.join(f));
    let Some(list) = std::fs::read_to_string(&file).ok().and_then(|t| checklist(&t)) else {
        return format!("skipped: {} is missing or has nothing to check", file.display());
    };
    let mut run_cfg = cfg.clone();
    if !hb.act {
        run_cfg.allow = BTreeSet::from(["read".to_owned()]);
    }
    let history = vec![serde_json::json!({"role": "user", "content": prompt(&list, now_text)})];
    match agent_cli::run_loop(backend, &run_cfg, system, history, &mut |_| {}) {
        Ok(outcome) => {
            let note = if outcome.status == Status::OutOfSteps { " (stopped at the step limit)" } else { "" };
            match alert_from(&outcome.answer) {
                None => format!("all clear{note}"),
                Some(text) => {
                    let mut sent = 0;
                    for key in &hb.chats {
                        for part in crate::gateway::split_message(&format!("Heartbeat: {text}"), chat.max_len()) {
                            if chat.send(key, &part).is_ok() {
                                sent += 1;
                            }
                        }
                    }
                    format!("alert sent to {} chat(s){note}", hb.chats.len().min(sent))
                }
            }
        }
        Err(e) => format!("failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::{Incoming, Chat};
    use serde_json::{json, Value};
    use std::collections::VecDeque;

    #[test]
    fn intervals_and_windows() {
        assert_eq!(parse_interval("30m").unwrap(), Duration::from_secs(1800));
        assert_eq!(parse_interval("2h").unwrap(), Duration::from_secs(7200));
        assert_eq!(parse_interval("15").unwrap(), Duration::from_secs(900));
        assert!(parse_interval("10s").is_err() && parse_interval("x").is_err() && parse_interval("9d").is_err());
        assert_eq!(parse_hours("08:00-22:30").unwrap(), (480, 1350));
        assert!(parse_hours("8-9").is_err() && parse_hours("08:00-08:00").is_err() && parse_hours("25:00-26:00").is_err());
        assert!(in_window(None, 3) && in_window(Some((480, 1350)), 600) && !in_window(Some((480, 1350)), 1400));
        // a night window crosses midnight
        assert!(in_window(Some((1320, 360)), 1400) && in_window(Some((1320, 360)), 100) && !in_window(Some((1320, 360)), 700));
    }

    #[test]
    fn an_empty_checklist_costs_nothing_and_acknowledgements_are_dropped() {
        assert_eq!(checklist("# Heartbeat\n\n<!-- nothing yet -->\n"), None);
        assert_eq!(checklist(""), None);
        assert!(checklist("# H\n- check disk\n").unwrap().contains("check disk"));
        assert_eq!(alert_from("HEARTBEAT_OK"), None);
        assert_eq!(alert_from("  **HEARTBEAT_OK**.\n"), None);
        assert_eq!(alert_from(""), None);
        assert_eq!(alert_from("The disk is 91% full.").as_deref(), Some("The disk is 91% full."));
        let long = "HEARTBEAT_OK but the backup log ends with an error: exit 12 at 03:10, the target is not mounted.";
        assert!(alert_from(long).unwrap().starts_with("but the backup log"));
    }

    #[test]
    fn the_clock_is_due_after_one_interval() {
        let t0 = Instant::now();
        let mut clock = Clock::new(Duration::from_secs(60), t0);
        assert!(!clock.due(t0 + Duration::from_secs(59)) && clock.due(t0 + Duration::from_secs(60)));
        clock.done(Duration::from_secs(60), t0 + Duration::from_secs(61));
        assert!(!clock.due(t0 + Duration::from_secs(100)));
    }

    struct Model(VecDeque<Value>, Vec<Value>);
    impl Backend for Model {
        fn tools(&mut self) -> Result<Vec<Value>, String> {
            Ok(vec![json!({"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {"type": "object", "properties": {}}}, "permission": "read"}),
                    json!({"type": "function", "function": {"name": "bash", "description": "d", "parameters": {"type": "object", "properties": {}}}, "permission": "command"})])
        }
        fn complete(&mut self, body: Value) -> Result<Value, String> {
            self.1.push(body);
            self.0.pop_front().ok_or_else(|| "script ended".to_owned())
        }
        fn run_tool(&mut self, name: &str, _: &Value) -> Result<String, String> {
            Ok(format!("ran {name}"))
        }
    }

    struct Sink(Vec<(String, String)>);
    impl Chat for Sink {
        fn updates(&mut self, c: &str) -> Result<(Vec<Incoming>, String), String> {
            Ok((vec![], c.to_owned()))
        }
        fn send(&mut self, key: &str, text: &str) -> Result<(), String> {
            self.0.push((key.to_owned(), text.to_owned()));
            Ok(())
        }
        fn session_name(&self, key: &str) -> String {
            key.to_owned()
        }
        fn max_len(&self) -> usize {
            4000
        }
    }

    fn reply(text: &str) -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": text}}]})
    }

    fn setup(tag: &str, checklist_text: &str) -> (std::path::PathBuf, Heartbeat, LoopConfig) {
        let dir = std::env::temp_dir().join(format!("fllm-hb-{tag}-{}-{}", std::process::id(), crate::agent_extras::fresh_session_name(9)));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("HEARTBEAT.md"), checklist_text).unwrap();
        let hb = Heartbeat { every: Duration::from_secs(60), chats: vec!["42".into()], file: None, hours: None, act: false, idle: Duration::ZERO };
        let cfg = LoopConfig { model: Some("m".into()), allow: BTreeSet::from(["read".to_owned(), "command".to_owned()]), rules: Default::default(), max_steps: 4, max_tokens: 100, temperature: 0.0 };
        (dir, hb, cfg)
    }

    #[test]
    fn a_quiet_check_sends_nothing_and_an_alert_goes_to_the_listed_chats() {
        let (dir, hb, cfg) = setup("run", "- is the disk full?\n");
        let mut quiet = Model(VecDeque::from([reply("HEARTBEAT_OK")]), vec![]);
        let mut sink = Sink(vec![]);
        assert_eq!(run_once(&hb, &dir, &cfg, "sys", &mut quiet, &mut sink, "Monday 10:00"), "all clear");
        assert!(sink.0.is_empty());
        let asked = quiet.1[0]["messages"][1]["content"].as_str().unwrap().to_owned();
        assert!(asked.contains("is the disk full?") && asked.contains("exactly HEARTBEAT_OK") && asked.contains("Monday 10:00"));
        // the run is read-only by default: the bash tool is not callable even though the gateway allows commands
        let mut loud = Model(VecDeque::from([reply("The disk is 91% full on /srv.")]), vec![]);
        assert!(run_once(&hb, &dir, &cfg, "sys", &mut loud, &mut sink, "t").starts_with("alert sent"));
        assert_eq!(sink.0, vec![("42".to_owned(), "Heartbeat: The disk is 91% full on /srv.".to_owned())]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn without_a_list_there_is_no_model_call_and_acting_needs_the_flag() {
        let (dir, mut hb, cfg) = setup("empty", "# only a heading\n");
        let mut model = Model(VecDeque::new(), vec![]);
        let mut sink = Sink(vec![]);
        assert!(run_once(&hb, &dir, &cfg, "sys", &mut model, &mut sink, "t").starts_with("skipped"));
        assert!(model.1.is_empty());
        std::fs::write(dir.join("HEARTBEAT.md"), "- restart the thing if it hangs\n").unwrap();
        let bash = json!({"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}}]});
        let mut read_only = Model(VecDeque::from([bash.clone(), reply("HEARTBEAT_OK")]), vec![]);
        run_once(&hb, &dir, &cfg, "sys", &mut read_only, &mut sink, "t");
        assert!(read_only.1[1]["messages"].to_string().contains("needs the permission"), "the command was refused");
        hb.act = true;
        let mut acting = Model(VecDeque::from([bash, reply("HEARTBEAT_OK")]), vec![]);
        run_once(&hb, &dir, &cfg, "sys", &mut acting, &mut sink, "t");
        assert!(acting.1[1]["messages"].to_string().contains("ran bash"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
