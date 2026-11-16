//! Slash commands of the Code page: the pure parts of `runCodeCommand` of `code.js`, the
//! permission rules behind `/allow` and `/deny`, and the command table the completion and
//! `/help` read. Everything here is the page's own code ported word for word; the stateful
//! halves (notices, routes, session work) run in `code.rs`, which executes [`Action`]s.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::sync::OnceLock;

/// The categories a rule may name (`RULE_CATEGORIES` of `code.js`).
pub const RULE_CATEGORIES: [&str; 9] = ["bash", "edit", "network", "mcp", "command", "read", "memory", "computer", "*"];

/// Programs an allow rule never covers by pattern (`RISKY_COMMANDS`).
const RISKY_COMMANDS: [&str; 26] = [
    "rm", "rmdir", "del", "erase", "rd", "format", "shutdown", "reboot", "mkfs", "dd", "sudo", "su", "curl", "wget", "iwr", "invoke-webrequest",
    "powershell", "pwsh", "cmd", "sh", "bash", "chmod", "chown", "kill", "taskkill", "git",
];

/// The characters that make a shell command a compound one (`SHELL_OPS`).
fn shell_ops() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[;&|><`$(){}\n]").expect("the shell operators of the page"))
}

/// `normPath` of `code.js`: backslashes are slashes, a leading `./` and a leading `/` are dropped.
pub fn norm_path(p: &str) -> String {
    let p = p.replace('\\', "/");
    let p = p.strip_prefix("./").unwrap_or(&p);
    p.trim_start_matches('/').to_string()
}

/// `globToRegex` of `code.js`: mode "path" stops `*` at `/`, `**` does not; mode "text" lets `*`
/// match anything.
pub fn glob_to_regex(pattern: &str, mode: &str) -> Regex {
    let chars: Vec<char> = pattern.chars().collect();
    let path_mode = mode == "path";
    let mut re = String::from("(?s)^");
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '*' {
            if path_mode {
                if chars.get(i + 1) == Some(&'*') {
                    i += 1;
                    if chars.get(i + 1) == Some(&'/') {
                        i += 1;
                        re.push_str("(?:.*/)?");
                    } else {
                        re.push_str(".*");
                    }
                } else {
                    re.push_str("[^/]*");
                }
            } else {
                while chars.get(i + 1) == Some(&'*') {
                    i += 1;
                }
                re.push_str(".*");
            }
        } else if c == '?' {
            re.push_str(if path_mode { "[^/]" } else { "." });
        } else if ".+^${}()|[]\\".contains(c) {
            re.push('\\');
            re.push(c);
        } else {
            re.push(c);
        }
        i += 1;
    }
    re.push('$');
    Regex::new(&re).unwrap_or_else(|_| Regex::new("(?s)^$").expect("an empty pattern"))
}

/// One parsed rule (`parseRules` of `code.js`): the category, the pattern and the agent the
/// optional `[agent]` in front names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rule {
    pub category: String,
    pub pattern: String,
    #[serde(default)]
    pub agent: String,
}

/// The two sides of the permission rules the window keeps per folder (`code.rules`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rules {
    #[serde(default)]
    pub allow: String,
    #[serde(default)]
    pub deny: String,
}

impl Rules {
    /// `parseRules` of `code.js`: one rule per line, `#` comments are skipped, the category must
    /// be one of the nine above.
    pub fn parse_side(text: &str) -> Vec<Rule> {
        static LINE: OnceLock<Regex> = OnceLock::new();
        let line = LINE.get_or_init(|| Regex::new(r"^\s*(?:\[([A-Za-z]+)\]\s*)?([A-Za-z*]+)\s*:\s*(.+?)\s*$").expect("a rule line"));
        let mut rules = Vec::new();
        for raw in text.split('\n') {
            if raw.trim().starts_with('#') {
                continue;
            }
            let Some(caps) = line.captures(raw) else { continue };
            let category = caps[2].to_lowercase();
            if !RULE_CATEGORIES.contains(&category.as_str()) {
                continue;
            }
            rules.push(Rule {
                category,
                pattern: caps[3].to_string(),
                agent: caps.get(1).map(|m| m.as_str().to_lowercase()).unwrap_or_default(),
            });
        }
        rules
    }

    /// What `evaluateRules` reads: both sides parsed, side by side.
    pub fn parsed(&self) -> ParsedRules {
        ParsedRules {
            deny: Self::parse_side(&self.deny),
            allow: Self::parse_side(&self.allow),
        }
    }
}

/// Both sides, parsed (`parsedRules` of `code.js`).
#[derive(Debug, Clone, Default)]
pub struct ParsedRules {
    pub deny: Vec<Rule>,
    pub allow: Vec<Rule>,
}

fn rule_applies_to(rule: &Rule, category: &str) -> bool {
    rule.category == "*" || rule.category == category || (rule.category == "command" && (category == "bash" || category == "mcp"))
}

fn pattern_matches(rule: &Rule, category: &str, subject: &str) -> bool {
    let path = category == "edit" || category == "read";
    let (pattern, subject) = if path {
        (norm_path(&rule.pattern), norm_path(subject))
    } else {
        (rule.pattern.clone(), subject.trim().to_string())
    };
    let mode = if path { "path" } else { "text" };
    glob_to_regex(&pattern, mode).is_match(&subject)
}

/// The program of one command, without a path and without an extension (`commandBase`).
pub fn command_base(command: &str) -> String {
    let first = command.split_whitespace().next().unwrap_or("");
    let no_path = first.rsplit(['/', '\\']).next().unwrap_or(first);
    let lower = no_path.to_lowercase();
    for ext in [".exe", ".cmd", ".bat"] {
        if lower.ends_with(ext) {
            return lower[..lower.len() - ext.len()].to_string();
        }
    }
    lower
}

fn has_shell_ops(command: &str) -> bool {
    shell_ops().is_match(command)
}

/// The parts of a command joined by a shell operator (`commandSegments`).
pub fn command_segments(command: &str) -> Vec<String> {
    static SPLIT: OnceLock<Regex> = OnceLock::new();
    let split = SPLIT.get_or_init(|| Regex::new(r"&&|\|\||[;&|\n]").expect("the segments of the page"));
    split.split(command).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect()
}

/// The key an "always" remembers for a command, or none when it is risky or compound
/// (`commandKey` of `code.js`).
pub fn command_key(command: &str) -> Option<String> {
    let base = command_base(command);
    if base.is_empty() || RISKY_COMMANDS.contains(&base.as_str()) || has_shell_ops(command) {
        return None;
    }
    Some(base)
}

fn mine(rule: &Rule, agent: &str) -> bool {
    rule.agent.is_empty() || agent.is_empty() || rule.agent == agent
}

/// `'deny'`, `'allow'` or none (no rule: ask) — deny wins, and an allow rule never covers a
/// compound shell command unless it is that exact string, nor a risky program unless the rule's
/// first word is that program (`evaluateRules` of `code.js`).
pub fn evaluate_rules(rules: &ParsedRules, category: &str, subject: &str, agent: &str) -> Option<&'static str> {
    let deny: Vec<&Rule> = rules.deny.iter().filter(|r| mine(r, agent) && rule_applies_to(r, category)).collect();
    let allow: Vec<&Rule> = rules.allow.iter().filter(|r| mine(r, agent) && rule_applies_to(r, category)).collect();
    if category == "bash" {
        let mut parts = vec![subject.trim().to_string()];
        parts.extend(command_segments(subject));
        if deny.iter().any(|r| parts.iter().any(|s| pattern_matches(r, category, s))) {
            return Some("deny");
        }
        let compound = has_shell_ops(subject);
        let base = command_base(subject);
        let risky = RISKY_COMMANDS.contains(&base.as_str());
        let hit = allow.iter().any(|r| {
            if !pattern_matches(r, category, subject) {
                return false;
            }
            if compound && r.pattern.trim() != subject.trim() {
                return false;
            }
            if risky {
                let head = r.pattern.split_whitespace().next().unwrap_or("");
                let head_base = command_base(head);
                if head.contains('*') || head.contains('?') || head_base != base {
                    return false;
                }
            }
            true
        });
        return hit.then_some("allow");
    }
    if deny.iter().any(|r| pattern_matches(r, category, subject)) {
        return Some("deny");
    }
    allow.iter().any(|r| pattern_matches(r, category, subject)).then_some("allow")
}

/// The categories the page keeps a mode per (`MODE_CATEGORIES`), in its order.
pub const MODE_CATEGORIES: [&str; 7] = ["read", "edit", "bash", "network", "mcp", "memory", "computer"];

/// What each agent does without being told otherwise (`DEFAULT_MODES` of `code.js`).
pub fn default_modes(agent: &str) -> &'static [(&'static str, &'static str)] {
    match agent {
        "plan" => &[
            ("read", "allow"),
            ("edit", "deny"),
            ("bash", "deny"),
            ("network", "ask"),
            ("mcp", "ask"),
            ("memory", "deny"),
            ("computer", "deny"),
        ],
        _ => &[
            ("read", "allow"),
            ("edit", "ask"),
            ("bash", "ask"),
            ("network", "ask"),
            ("mcp", "ask"),
            ("memory", "allow"),
            ("computer", "ask"),
        ],
    }
}

/// The mode of one category for one agent: the user's table, else the default — and `autoEdit`
/// makes `edit` allowed for the build agent (`modeFor` of `code.js`).
pub fn mode_for(modes: &BTreeMap<String, BTreeMap<String, String>>, agent: &str, category: &str, auto_edit: bool) -> String {
    if let Some(own) = modes.get(agent).and_then(|m| m.get(category)) {
        if matches!(own.as_str(), "allow" | "ask" | "deny") {
            return own.clone();
        }
    }
    if agent == "build" && category == "edit" && auto_edit {
        return "allow".into();
    }
    default_modes(agent)
        .iter()
        .find(|(c, _)| *c == category)
        .map(|(_, v)| (*v).to_string())
        .unwrap_or_else(|| "ask".into())
}

/// What one tool call is, as the page reads it (`permissionInfo`): the category and the subject
/// the rules and the modes judge.
pub fn permission_info(permission: &str, kind_mcp: Option<(&str, &str)>, name: &str, args: &serde_json::Value) -> (String, Vec<String>) {
    let cat = if kind_mcp.is_some() {
        "mcp".to_string()
    } else {
        match permission {
            "edit" => "edit".into(),
            "network" => "network".into(),
            "command" => "bash".into(),
            "memory" => "memory".into(),
            "computer" => "computer".into(),
            _ => "read".into(),
        }
    };
    let mut subjects: Vec<String> = Vec::new();
    let subject = if cat == "computer" {
        // `computerSubject`: the one line the rules of `computer` match
        crate::computer::computer_subject(name, args)
    } else if cat == "bash" {
        match args.get("command").and_then(|v| v.as_str()) {
            Some(command) => command.to_string(),
            None => {
                let values: Vec<String> = args.as_object().map(|o| o.values().map(value_word).collect()).unwrap_or_default();
                format!("{name} {}", values.join(" ")).trim().to_string()
            }
        }
    } else if cat == "edit" && args.get("patch").and_then(|v| v.as_str()).is_some() {
        let found = patch_paths(args["patch"].as_str().unwrap_or(""));
        let first = found.first().cloned().unwrap_or_default();
        subjects.extend(found);
        first
    } else if cat == "edit" || cat == "read" {
        args.get("path")
            .or_else(|| args.get("pattern"))
            .map(value_word)
            .unwrap_or_default()
    } else if cat == "network" {
        args.get("url")
            .or_else(|| args.get("query"))
            .map(value_word)
            .unwrap_or_default()
    } else if cat == "memory" {
        args.get("scope").map(value_word).unwrap_or_else(|| "workspace".into())
    } else if let Some((server, tool)) = kind_mcp {
        format!("{server}/{tool}")
    } else {
        String::new()
    };
    if subjects.is_empty() {
        subjects.push(subject);
    }
    (cat, subjects)
}

fn value_word(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// The files a unified diff names (`patchPaths` of `code.js`): the new side of every header.
pub fn patch_paths(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out: Vec<String> = Vec::new();
    for i in 0..lines.len().saturating_sub(1) {
        if !lines[i].starts_with("--- ") || !lines[i + 1].starts_with("+++ ") {
            continue;
        }
        let raw = lines[i + 1].strip_prefix("+++ ").unwrap_or("");
        let clean = raw.split('\t').next().unwrap_or("");
        let clean = clean.strip_prefix("a/").or_else(|| clean.strip_prefix("b/")).unwrap_or(clean);
        if clean != "/dev/null" && !out.iter().any(|p| p == clean) {
            out.push(clean.to_string());
        }
    }
    out
}

/// What `"Always"` would remember for a call, or none when it can only be allowed once
/// (`alwaysKey` of `code.js`).
pub fn always_key(permission: &str, kind_mcp: Option<(&str, &str)>, name: &str, args: &serde_json::Value) -> Option<String> {
    let (cat, subjects) = permission_info(permission, kind_mcp, name, args);
    let subject = subjects.first().map(String::as_str).unwrap_or("");
    match cat.as_str() {
        "edit" => Some("edit".into()),
        "bash" => {
            if name == "bash" {
                command_key(subject).map(|k| format!("bash:{k}"))
            } else {
                Some(name.to_string())
            }
        }
        "network" => Some("network".into()),
        "mcp" => Some(format!("mcp:{subject}")),
        // only looking can be allowed for the rest of the session, and only on the screenshot card
        "computer" => (name == "computer_screenshot").then(|| "computer:look".to_string()),
        _ => None,
    }
}

/// `$ARGUMENTS` is the whole text, `$1` to `$9` single words; without `$ARGUMENTS` the text is
/// appended as its own line (`expandCommand` of `code.js`).
pub fn expand_command(body: &str, args_text: &str) -> String {
    let args = split_args(args_text);
    let has_all = body.contains("$ARGUMENTS");
    let mut out = body.replace("$ARGUMENTS", args_text.trim());
    for n in 1..=9usize {
        let token = format!("${n}");
        if out.contains(&token) {
            let replacement = args.get(n - 1).cloned().unwrap_or_default();
            out = out.replace(&token, &replacement);
        }
    }
    if !has_all && !args_text.trim().is_empty() {
        out.push_str("\n\nARGUMENTS: ");
        out.push_str(args_text.trim());
    }
    out
}

/// The words of a command line, double or single quotes kept together (`splitArgs`).
pub fn split_args(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '"' || c == '\'' {
            // `"…"` or `'…'` only when the same quote closes it, else the word runs on
            if let Some(end) = (i + 1..chars.len()).find(|&j| chars[j] == c) {
                out.push(chars[i + 1..end].iter().collect());
                i = end + 1;
                continue;
            }
        }
        let start = i;
        while i < chars.len() && !chars[i].is_whitespace() {
            i += 1;
        }
        out.push(chars[start..i].iter().collect());
    }
    out
}

/// A custom command file: optional front matter with the description, then the prompt body
/// (`parseCommandFile` of `code.js`).
pub fn parse_command_file(text: &str) -> (String, String) {
    let mut body = text.replace("\r\n", "\n");
    let mut description = String::new();
    if body.starts_with("---\n") {
        if let Some(end) = body[4..].find("\n---") {
            let block_end = 4 + end + 4;
            let block = &body[4..4 + end];
            let full_end = body[block_end..].find('\n').map(|i| block_end + i).unwrap_or(body.len());
            for line in block.split('\n') {
                if let Some(rest) = line.strip_prefix("description:") {
                    description = rest.trim().trim_matches(['"', '\'']).to_string();
                }
            }
            body = body[full_end..].to_string();
        }
    }
    let body = body.trim().to_string();
    if description.is_empty() {
        let line = body.split('\n').find(|l| !l.trim().is_empty()).unwrap_or("");
        let hashes = line.chars().take_while(|c| *c == '#').count();
        let rest = line[hashes.min(line.len())..].trim_start_matches([' ', '\t']);
        description = rest.chars().take(80).collect();
    }
    (description, body)
}

/// The names of the custom commands under `.fllm/commands` (`commandNamesFromPaths`).
pub fn command_names_from_paths(paths: &[String]) -> Vec<(String, String)> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<(String, String)> = Vec::new();
    for raw in paths {
        let norm = norm_path(raw);
        let Some(rest) = norm.strip_prefix(".fllm/commands/") else { continue };
        let Some(name) = rest.strip_suffix(".md") else { continue };
        let name = name.replace('/', ":");
        if !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == ':' || c == '.')
            && seen.insert(name.clone())
        {
            out.push((name, norm));
        }
    }
    out
}

/// `clipMiddle` of `code.js`: a long output keeps its head (40 %) and its tail.
pub fn clip_middle(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head = max * 4 / 10;
    let tail = max - head;
    let cut = text.chars().count() - max;
    let head_text: String = text.chars().take(head).collect();
    let tail_text: String = text.chars().skip(text.chars().count() - tail).collect();
    format!("{head_text}\n... [{cut} characters cut from the middle] ...\n{tail_text}")
}

/// What `userRun(command, true)` puts in front of the next message (`pending.push` of `code.js`).
pub fn share_text(command: &str, output: &str) -> String {
    format!("I ran this command myself:\n$ {command}\n```\n{}\n```", clip_middle(output, 4000))
}

/// The text of `/cost` (`fllmUsageSummary` of `usage.js`, wrapped as the command wraps it).
pub fn cost_notice(summary: &str) -> String {
    format!("{summary}. Prices and the full table: Settings > Usage and cost.")
}

/// The hits of the session search as a notice (`searchHitsText` of `codeprompts.js`).
pub fn search_hits_text(hits: &[SearchHit], query: &str) -> String {
    if hits.is_empty() {
        return format!("No stored session mentions \"{query}\".");
    }
    let plural = if hits.len() == 1 { "" } else { "s" };
    let rows: Vec<String> = hits
        .iter()
        .map(|h| {
            let kind = if h.kind == "code" { "code" } else { "chat" };
            let when = match &h.updated {
                Some(updated) => format_date(*updated),
                None => "undated".into(),
            };
            format!("- {} ({kind}, {when})\n  {}", h.title.as_deref().unwrap_or(&h.id), h.snippet)
        })
        .collect();
    format!("{} stored session{plural} for \"{query}\":\n{}", hits.len(), rows.join("\n"))
}

/// The date of a hit in the machine's locale (`new Date(...).toLocaleDateString()`).
fn format_date(ms: i64) -> String {
    // the page shows a locale date; the window writes the calendar form of it
    let secs = ms / 1000;
    let days = secs.div_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!("{month}/{day}/{year}")
}

/// Howard Hinnant's `civil_from_days`, so the date needs no chrono.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// One hit of `GET /api/studio/sessions/search`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub updated: Option<i64>,
    #[serde(default)]
    pub snippet: String,
}

/// The counts `/insights` shows (`sessionInsights` of `codeprompts.js`): sessions, messages from
/// the user, agent steps, file edits, the most used tools.
pub fn session_insights(sessions: &[Vec<InsightMessage>]) -> Insights {
    let mut out = Insights::default();
    for session in sessions {
        out.sessions += 1;
        for message in session {
            if message.role == "user" {
                out.user_messages += 1;
                continue;
            }
            if message.role != "assistant" {
                continue;
            }
            out.steps += 1;
            for name in &message.tool_calls {
                *out.tools.entry(name.clone()).or_insert(0) += 1;
                if ["write_file", "edit_file", "multi_edit", "patch", "notebook_edit"].contains(&name.as_str()) {
                    out.edits += 1;
                }
            }
        }
    }
    out.top = {
        let mut rows: Vec<(String, usize)> = out.tools.iter().map(|(k, v)| (k.clone(), *v)).collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        rows.into_iter().take(8).collect()
    };
    out
}

/// One message as the counting reads it: the role and the names of the tool calls on it.
#[derive(Debug, Clone, Default)]
pub struct InsightMessage {
    pub role: String,
    pub tool_calls: Vec<String>,
}

/// What `session_insights` counts.
#[derive(Debug, Clone, Default)]
pub struct Insights {
    pub sessions: usize,
    pub user_messages: usize,
    pub steps: usize,
    pub tools: BTreeMap<String, usize>,
    pub edits: usize,
    pub top: Vec<(String, usize)>,
}

/// The five styles of `/personality` (`PERSONALITIES` of `codeprompts.js`).
pub const PERSONALITIES: [(&str, &str, &str); 5] = [
    ("default", "Default", ""),
    ("concise", "Concise", "Style: use as few words as the job allows. No preamble and no recap of the request. Report what changed, what you checked and what is left."),
    ("teacher", "Teacher", "Style: explain your reasoning in a sentence or two as you go and name the idea behind each change, so the user learns the codebase. Never lecture."),
    ("reviewer", "Strict reviewer", "Style: be critical of your own work. Before you call something done, look for the bug you did not think of: edge cases, error paths, missing tests. Say plainly what is weak."),
    ("pair", "Pair programmer", "Style: think aloud in short lines, say what you will do next when the choice is not obvious, and ask one question instead of guessing."),
];

/// The system prompt line for a personality (`personalityNote`): empty for the default and for
/// names the table does not have.
pub fn personality_note(name: &str) -> &'static str {
    PERSONALITIES.iter().find(|(key, _, _)| *key == name).map(|(_, _, note)| *note).unwrap_or("")
}

// ---- the dispatcher ------------------------------------------------------------------------

/// The commands the page knows (`BUILTIN_COMMANDS` of `code.js`), with the hint its `/` menu
/// shows. `sandbox` stands twice, exactly as there — both entries are kept, and `/help` lists
/// every name in this order, `/sandbox` twice.
pub const BUILTIN_COMMANDS: [(&str, &str); 46] = [
    ("clear", "Start a new session"),
    ("compact", "Summarise older turns to free context"),
    ("init", "Let the agent write AGENTS.md"),
    ("spec", "Plan a feature in files, then build it task by task (/spec name what you want, /spec run name)"),
    ("resume", "Pick an earlier session"),
    ("fork", "New session from an earlier message"),
    ("revert", "Go back to before a message, files included"),
    ("rename", "Rename this session"),
    ("export", "Save this session as Markdown"),
    ("rewind", "Undo file changes back to a chosen point"),
    ("undo", "Undo the last file change (/undo 3 for three)"),
    ("plan", "Plan agent: read only"),
    ("build", "Build agent"),
    ("ask", "Build agent, ask before every change"),
    ("edit", "Build agent, auto-edit"),
    ("agent", "Switch between build and plan"),
    ("model", "Pick the model"),
    ("skill", "Turn this session into a reusable skill"),
    ("remember", "Save a fact to memory (/remember global ...)"),
    ("memory", "Show and edit the memory"),
    ("check", "Check the code for errors (diagnostics)"),
    ("diff", "git diff"),
    ("status", "git status"),
    ("worktree", "Work in a separate git worktree (/worktree new <name>, leave, remove)"),
    ("batch", "Run a task in its own worktree (/batch <name> <task>)"),
    ("cost", "Tokens and cost counted in this browser"),
    ("sandbox", "Show whether agent commands run sandboxed"),
    ("autoapprove", "Harmless commands run without a question (/autoapprove on|off)"),
    ("turbo", "No questions at all, sandbox off (/turbo on|off)"),
    ("sandbox", "Show or set the sandbox mode (/sandbox off|workspace|strict)"),
    ("review", "Review the changes: diff plus rule findings (/review staged)"),
    ("security-review", "Security-only review of the changes"),
    ("cd", "Open another folder"),
    ("allow", "Add an allow rule, e.g. /allow bash: npm test*"),
    ("deny", "Add a deny rule"),
    ("rules", "Show the permission rules"),
    ("search", "Search earlier chats and code sessions (/search words)"),
    ("retry", "Take the last message back, files included, and send it again"),
    ("stop", "Stop the agent"),
    ("insights", "What the agent did in your saved sessions of this folder"),
    ("personality", "How the agent talks (/personality concise)"),
    ("autoreview", "Review my work before it counts as done (/autoreview on|off)"),
    ("learn", "Learn from my sessions: on by default (/learn on|off, /learn project off|on|default)"),
    ("good", "The skills of the last run helped"),
    ("bad", "The skills of the last run did not help"),
    ("help", "List the commands"),
];

/// The prompt `/init` sends (`INIT_PROMPT` of `code.js`, word for word).
pub const INIT_PROMPT: &str = "Explore this workspace and write a file named AGENTS.md in its root with instructions for future coding sessions. Cover: what the project is, how to build, test and run it (exact commands, checked against the build files), the layout of the important folders, code style and conventions you can see, and pitfalls. Keep it under 80 lines, concrete, with no filler. If AGENTS.md, FLLM.md or CLAUDE.md already exists, read it and improve that file instead of starting over.";

/// The notice `/cost` writes when nothing was counted (`fllmUsageSummary` returning nothing).
pub const NO_USAGE: &str = "No usage counted yet in this browser.";

/// A custom command as the window keeps it (a file under `.fllm/commands` or one an installed
/// plugin brings).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomCommand {
    pub name: String,
    pub description: String,
    /// The path of the file, none for a command of a plugin.
    pub path: Option<String>,
    pub plugin: Option<String>,
}

/// The fields of the page's `code` object the switch reads (`CmdState`), so every command can be
/// planned without touching the window.
#[derive(Debug, Clone, Default)]
pub struct CmdState {
    pub running: bool,
    /// `plan` | `ask` | `edit` (`code.perm`).
    pub perm: String,
    /// The build mode to go back to from plan (`code.lastBuildPerm`), empty like an unset one.
    pub last_build_perm: String,
    pub auto_approve: bool,
    /// `code.autoReview`: what the notice of `/autoreview` without an argument reads.
    pub auto_review: bool,
    /// `code.specs`: the listing of `/spec` is built from it (the page reloads before it).
    pub specs: Vec<crate::specs::Spec>,
    pub turbo: bool,
    pub sandbox: Option<String>,
    pub sandbox_available: bool,
    pub personality: String,
    pub rules: Rules,
    pub custom: Vec<CustomCommand>,
    /// The `/cost` summary of the window, none when nothing was counted.
    pub usage: Option<String>,
    /// The models `/model` offers (`modelNames()` of the page).
    pub models: Vec<String>,
}

// ---- the decision of one call ----------------------------------------------------------------

/// What the answer to one tool call is after the rules, the mode and the remembered keys
/// (`authorize` of `code.js`, with the plan check of `runToolCall` — line 1967 — in front).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Ask,
    Deny,
}

/// What the window makes of one call: the plan refusal, the block of a rule, a run (with the
/// table's reason when it lifted a question), or the question itself.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Plan,
    Blocked,
    Running(Option<String>),
    Ask,
}

/// What `Always` remembered for this call, the way `authorize` looks it up (code.js 1724-1728).
pub fn remembered(category: &str, tool_name: &str, subject: &str, always: &std::collections::HashSet<String>) -> bool {
    match category {
        "bash" => {
            let key = if tool_name == "bash" { command_key(subject) } else { Some(tool_name.to_string()) };
            key.is_some_and(|key| always.contains(&format!("bash:{key}")))
        }
        "network" => always.contains("network"),
        "mcp" => always.contains(&format!("mcp:{subject}")),
        // only looking can be remembered, and only by the screenshot card
        "computer" => crate::computer::is_look(tool_name) && always.contains("computer:look"),
        _ => false,
    }
}

/// `authorize` of the page over every subject a call has (a patch brings one per file): deny wins
/// immediately, mode and rule and the remembered keys allow, and anything else is a question.
/// The plan refusal runs before this on the page (`runToolCall`, line 1967) and stays in
/// `code.rs`; a mode that *denies* a category comes only from plan mode, which never gets here.
// the page calls this with a handful of pieces of the caller's state, and the window keeps
// that shape so the fixtures of the page can drive it one to one
#[allow(clippy::too_many_arguments)]
pub fn authorize(
    turbo: bool,
    rules: &ParsedRules,
    mode_allows: bool,
    category: &str,
    subjects: &[String],
    agent: &str,
    tool_name: &str,
    always: &std::collections::HashSet<String>,
) -> Decision {
    if turbo {
        // turbo: no question, no rule
        return Decision::Allow;
    }
    let mut any_ask = false;
    for subject in subjects {
        if evaluate_rules(rules, category, subject, agent) == Some("deny") {
            return Decision::Deny;
        }
        // operating the computer (click, type, key, scroll, drag, move) is asked every single time:
        // neither a mode, nor an allow rule, nor "Always" lifts that (`decidePermission`)
        if category == "computer" && !crate::computer::is_look(subject.split(' ').next().unwrap_or_default()) {
            any_ask = true;
            continue;
        }
        if mode_allows {
            continue;
        }
        if evaluate_rules(rules, category, subject, agent) == Some("allow") {
            continue;
        }
        if remembered(category, tool_name, subject, always) {
            continue;
        }
        any_ask = true;
    }
    if any_ask { Decision::Ask } else { Decision::Allow }
}

/// One call from the plan check down to the table of `/autoapprove` — the chain of `runToolCall`
/// (code.js 1955-2010) with everything the runner already knows passed in: whether plan mode is
/// on and refuses this, and whether the mode allows it outright.
#[allow(clippy::too_many_arguments)]
pub fn call_decision(
    turbo: bool,
    rules: &Rules,
    in_plan: bool,
    mode_allows: bool,
    permission: &str,
    kind_mcp: Option<(&str, &str)>,
    tool_name: &str,
    args: &serde_json::Value,
    auto_approve: bool,
    agent: &str,
    always: &std::collections::HashSet<String>,
) -> Outcome {
    let read_only = permission == "read";
    if in_plan && !read_only {
        // the window's tools are its own (no `source === 'client'`), as on the page
        return Outcome::Plan;
    }
    let (category, subjects) = permission_info(permission, kind_mcp, tool_name, args);
    let decision = authorize(turbo, &rules.parsed(), mode_allows, &category, &subjects, agent, tool_name, always);
    match decision {
        Decision::Deny => Outcome::Blocked,
        Decision::Allow => Outcome::Running(None),
        Decision::Ask => {
            if !auto_approve {
                return Outcome::Ask;
            }
            let subject = subjects.first().map(String::as_str).unwrap_or_default();
            let verdict = crate::autoapprove::auto_verdict(&category, tool_name, subject);
            if verdict.safe {
                Outcome::Running(Some(verdict.reason))
            } else {
                Outcome::Ask
            }
        }
    }
}

/// What a command asks the window to do (`runCodeCommand` of `code.js`, every case this window
/// builds). The steps stand in the order the page performs them; `None` from
/// [`dispatch`] means the words are no command here, so they go to the model.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// `newCodeSession()` — stop a running agent first, as the page does.
    NewSession,
    /// `undoChange(steps)` of the page: the count as `Number(...)` read it.
    Undo(f64),
    /// `agentRun(prompt, source)`
    AgentRun(String, String),
    /// `setPerm(perm)` followed by the notice of the case (or of `toggleAgent`).
    SetPerm(&'static str, String),
    /// `setWorkspace(path)`, or `pickWorkspace()` with no path.
    OpenFolder(Option<String>),
    /// `stopAgent()` — only when the agent runs; otherwise the case notices it.
    Stop,
    /// `userRun(command, share)`: a card of a command the user ran, its output shared with the
    /// model with the next message when `share` is set.
    UserRun(String, bool),
    /// `saveMemoryEntry(entry, scope)` with the notice that follows a save.
    Remember { scope: &'static str, entry: String, notice: String },
    /// `saveRules(allow, deny)` with `Rule saved: ...`.
    SaveRules { allow: String, deny: String, notice: String },
    /// `/turbo [on|off]`: the switch, behind the page's question when it turns on.
    Turbo(bool),
    /// `POST /api/studio/runtime { sandbox }` with the notice of the answer.
    SetSandbox(String),
    /// `/autoapprove [on|off]`: `Some(want)` changes the switch, the notice always follows.
    AutoApprove(Option<bool>, String),
    /// `/autoreview [on|off]`: the same shape — no argument changes nothing and the notice
    /// reads the switch as it stands (`case 'autoreview'` of `runCodeCommand`).
    AutoReview(Option<bool>, String),
    /// `userTool(name, args)`: a card of a tool the user ran (`/check`, `/review`,
    /// `/security-review`) — nothing of it is shared with the model, unlike `userRun(…, true)`.
    UserTool { name: String, args: serde_json::Value },
    /// `/worktree [new|leave|remove|list] [name]`: the request of the page, its answer coming
    /// back through `WorktreeDone`.
    Worktree { action: &'static str, name: Option<String> },
    /// `/batch <name> <task>`: a new worktree, the notice of it, and then the task as a run of
    /// its own (`agentRun(task, '/batch')` there).
    Batch { name: String, task: String },
    /// `/learn [on|off|project …]`: the words as typed; `learn::plan_learn` reads them.
    Learn(String),
    /// `/good` (`up`) and `/bad` (`down`): a thumb for the skills of the last run.
    Thumb(&'static str),
    /// `/spec` with no argument: reload the list and write the listing (`specCommand`'s
    /// empty branch).
    SpecList,
    /// `/spec run [NAME] [N]`: find the spec against the fresh list (the page reloads when it
    /// does not have the name) and run the task it picks.
    SpecRun { name: String, asked: Option<f64> },
    /// `/spec NAME what you want`: the prompt of `new_prompt`, run with the words as they came.
    SpecNew { name: String, description: String, display: String },
    /// `/search words`: the search over the stored sessions of this window.
    Search(String),
    /// The counts of `/insights` over this folder's session(s).
    Insights,
    /// `/personality <key>`: the chosen style with its notice.
    SetPersonality(&'static str, String),
    /// Read a custom command file (or the plugin's) and run what it holds.
    CustomCommand { name: String, arg_text: String, source: String },
    /// `/resume` and `/sessions`: the earlier sessions of this folder to pick from.
    Sessions,
    /// `/fork`: pick a message to fork before, or fork at the end when there is none.
    Fork,
    /// `/revert`: pick a message to go back to.
    Revert,
    /// `/rename <title>` sets the title; `/rename` asks for one.
    Rename(Option<String>),
    /// `/export`: the session as a Markdown file.
    Export,
    /// `/rewind`: pick the oldest file change to take back.
    Rewind,
    /// `/compact [focus]`: summarise the older turns.
    Compact(String),
    /// `/model`: pick the model for the agent among these.
    PickModel(Vec<String>),
    /// `/memory`: the memory block of the side, open.
    Memory,
    /// `/retry`: go back to your last message and send it again.
    Retry,
    /// `/skill`: the model writes a skill from the session.
    Skill,
    /// `codeNotice(text)` — everything the cases only say.
    Notice(String),
    /// `toast(text)` — only `/agent` while the agent runs.
    Toast(String),
}

/// The words of a entered line, split the way the page splits them (`split(/\s+/)` keeps empty
/// strings at the borders, an internal run collapses to one).
fn split_ws_js(body: &str) -> Vec<&str> {
    if body.is_empty() {
        return vec![""];
    }
    // the words between the runs of whitespace; empty strings only at the borders, as a split on
    // a run of whitespace keeps them there and collapses them inside
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in body.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                runs.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        runs.push((s, body.len()));
    }
    let mut out: Vec<&str> = Vec::new();
    if body.starts_with(char::is_whitespace) {
        out.push("");
    }
    out.extend(runs.into_iter().map(|(s, e)| &body[s..e]));
    if body.ends_with(char::is_whitespace) {
        out.push("");
    }
    out
}

/// `Number(x) || 1` of `/undo`: what JavaScript reads from the word, with NaN and 0 falling back
/// to 1.
pub fn undo_steps(word: Option<&str>) -> f64 {
    let Some(word) = word else { return 1.0 };
    let word = word.trim();
    let parsed = if let Some(hex) = word.strip_prefix("0x").or_else(|| word.strip_prefix("0X")) {
        i128::from_str_radix(hex, 16).ok().map(|v| v as f64)
    } else if word.eq_ignore_ascii_case("infinity") {
        Some(f64::INFINITY)
    } else {
        word.parse::<f64>().ok().filter(|v| v.is_finite())
    };
    match parsed {
        Some(v) if v != 0.0 => v,
        _ => 1.0,
    }
}

/// The text of `/help` (`case 'help'` of `code.js`): the table above, the custom commands when
/// there are any, and the tail the page always adds.
pub fn help_text(custom: &[CustomCommand]) -> String {
    let names: Vec<String> = BUILTIN_COMMANDS.iter().map(|(name, _)| format!("/{name}")).collect();
    let mut text = format!("Commands: {}.", names.join(", "));
    if !custom.is_empty() {
        let list: Vec<String> = custom
            .iter()
            .map(|c| if c.description.is_empty() { format!("/{}", c.name) } else { format!("/{} ({})", c.name, c.description) })
            .collect();
        text.push_str(&format!(" Custom commands from .fllm/commands: {}.", list.join(", ")));
    }
    text.push_str(" @file attaches a file (fuzzy search), !command runs a shell command and shows the model its output, Enter while the agent works queues a message, Alt+B switches between the build and plan agent.");
    text
}

/// The usage of `/worktree` when the action needs a name (`case 'worktree'`).
pub const WORKTREE_USAGE: &str = "Usage: /worktree new <name> · /worktree leave · /worktree remove <name> · /worktree (list)";
/// The usage of `/batch` (`case 'batch'`).
pub const BATCH_USAGE: &str = "Usage: /batch <name> <task> — the task runs in a new worktree on branch <name>.";

/// The three notices of `/plan`, `/ask` and `/edit` (`case 'plan': case 'ask': case 'edit':`).
pub fn perm_notice(perm: &str) -> &'static str {
    match perm {
        "plan" => "Plan agent: it only reads and writes a plan.",
        "ask" => "Build agent: it asks before every change and every command.",
        _ => "Build agent, auto-edit: it changes files itself and asks before commands.",
    }
}

/// The notice `toggleAgent` writes after it switched (`Alt+B`).
pub fn toggle_notice(perm_after: &str) -> &'static str {
    if perm_after == "plan" {
        "Plan agent: it only reads and writes a plan."
    } else {
        "Build agent: it can change files and run commands (with the permissions you set)."
    }
}

/// The guard of `/agent` while the agent works.
pub const AGENT_BUSY: &str = "Wait until the agent has finished or stop it.";

/// The notice of `/stop` when nothing runs.
pub const STOP_IDLE: &str = "The agent is not running.";

/// The two lines of `/remember` when the words hold no fact.
pub const REMEMBER_USAGE: &str = "Usage: /remember a short fact, or /remember global a fact for every workspace.";

/// The notice after `/remember` saved (`saveMemoryEntry`).
pub fn remember_notice(global: bool, entry: &str) -> String {
    format!("Remembered ({}): {entry}", if global { "for every workspace" } else { "for this workspace" })
}

/// The words of `/allow` and `/deny` when what is typed is no rule.
pub fn rule_usage(name: &str) -> String {
    format!("Usage: /{name} category: pattern, for example /{name} bash: npm test*")
}

/// The notice after a rule was appended.
pub fn rule_saved_notice(name: &str, arg_text: &str) -> String {
    format!("Rule saved: {name} {arg_text}")
}

/// `/rules`: both sides, `(none)` when a side holds nothing.
pub fn rules_notice(rules: &Rules) -> String {
    format!("Allow:
{}

Deny:
{}", if rules.allow.is_empty() { "(none)" } else { &rules.allow }, if rules.deny.is_empty() { "(none)" } else { &rules.deny })
}

/// The question `/turbo on` asks first.
pub const TURBO_CONFIRM: &str = "Turbo removes every permission question and the sandbox. Continue?";

/// The notice after the runtime answered the switch of `/turbo`.
pub fn turbo_notice(on: bool) -> &'static str {
    if on {
        "Turbo is on: every tool call runs, nothing asks, sandbox off."
    } else {
        "Turbo is off: the usual permission questions are back."
    }
}

/// The notice when the runtime refused `/turbo`.
pub const RUNTIME_FAIL: &str = "The runtime could not be changed.";

/// The notice of `/sandbox` with a word that is no mode.
pub fn sandbox_invalid_notice(sandbox: Option<&str>, available: bool) -> String {
    let note = if available { "" } else { " (the sandbox does not work on this machine)" };
    format!("Sandbox is {}{note}. /sandbox off | workspace | strict", sandbox.unwrap_or("unknown"))
}

/// The notice after the runtime accepted a sandbox mode.
pub fn sandbox_saved_notice(mode: &str, available: bool) -> String {
    let note = if available { "" } else { " — but the sandbox does not work here, so commands run unwrapped" };
    format!("Sandbox: {mode}{note}.")
}

/// The notice when the runtime refused the mode of `/sandbox`.
pub const SANDBOX_FAIL: &str = "The sandbox mode could not be changed.";

/// The words of `/search` without words to search for.
pub const SEARCH_USAGE: &str = "Usage: /search words. It looks through the chats and code sessions this app has stored (Settings > Skills and schedules keeps the index).";

/// The notice of `/insights` (`case 'insights'` of `code.js`) — `MAX_SESSIONS` of the page is 30.
pub fn insights_notice(ins: &Insights) -> String {
    if ins.sessions == 0 || ins.steps == 0 {
        return "No agent work saved for this folder yet.".to_string();
    }
    let one = |n: usize| if n == 1 { "" } else { "s" };
    let tools: Vec<String> = ins.top.iter().map(|(name, count)| format!("{name} {count}")).collect();
    format!(
        "This folder, {} saved session{} (the browser keeps the newest 30): {} message{} from you, {} agent step{}, {} file edit{}.\nMost used tools: {}.\nTokens and cost: /cost. The terminal agent has the same overview over its own sessions: fllm-rs insights.",
        ins.sessions,
        one(ins.sessions),
        ins.user_messages,
        one(ins.user_messages),
        ins.steps,
        one(ins.steps),
        ins.edits,
        one(ins.edits),
        tools.join(", "),
    )
}

/// The keys of `/personality`, as the case lists them.
pub fn personality_keys() -> String {
    PERSONALITIES.iter().map(|(key, _, _)| *key).collect::<Vec<_>>().join(", ")
}

/// The notice of `/personality` with no word, and the one for a style the table does not know.
pub fn personality_info_notice() -> String {
    let label = PERSONALITIES[0].1;
    format!("Style: {label}. Choose one of: {}.", personality_keys())
}

pub fn personality_unknown_notice(arg: &str) -> String {
    format!("There is no style \"{arg}\". Choose one of: {}.", personality_keys())
}

/// The notice after `/personality` picked a style: the label and the note without its `Style: `
/// prefix, its first letter upper-cased (`case 'personality'` of `code.js`).
pub fn personality_notice(key: &str) -> String {
    let (_, label, note) = PERSONALITIES.iter().find(|(k, _, _)| *k == key).expect("a key of the table");
    let detail = note.strip_prefix("Style: ").unwrap_or(note);
    if detail.is_empty() {
        format!("Style: {label}.")
    } else {
        let mut chars = detail.chars();
        let head: String = chars.next().map(|c| c.to_uppercase().collect()).unwrap_or_default();
        format!("Style: {label}. {head}{}", chars.as_str())
    }
}

/// One line of entered text as the page parses it for its switch (`runCodeCommand`, first lines).
struct Parsed {
    cmd: String,
    name: String,
    arg_text: String,
    rest: Vec<String>,
}

fn parse_line(text: &str) -> Option<Parsed> {
    if !text.starts_with('/') {
        return None;
    }
    let body: String = text.chars().skip(1).collect();
    let words = split_ws_js(&body);
    let cmd = words.first().copied().unwrap_or("").to_string();
    let rest: Vec<String> = words.iter().skip(1).map(|w| (*w).to_string()).collect();
    let arg_text = body[cmd.len().min(body.len())..].trim().to_string();
    Some(Parsed {
        name: cmd.to_lowercase(),
        cmd,
        arg_text,
        rest,
    })
}

/// The commands this window builds. `None` means: not a command here — the page either has no
/// such case either (`return false`: the words go to the model) or it has one this window does
/// not build, which PARITY.md lists for row 79 with the reason.
pub fn dispatch(text: &str, state: &CmdState) -> Option<Vec<Step>> {
    let parsed = parse_line(text)?;
    let Parsed { cmd, name, arg_text, rest } = parsed;
    let steps = match name.as_str() {
        "clear" | "new" => vec![Step::NewSession],
        "undo" => vec![Step::Undo(undo_steps(rest.first().map(String::as_str)))],
        "init" => vec![Step::AgentRun(INIT_PROMPT.to_string(), "/init".to_string())],
        "plan" | "ask" | "edit" => {
            let perm: &'static str = if name == "plan" { "plan" } else if name == "ask" { "ask" } else { "edit" };
            vec![Step::SetPerm(perm, perm_notice(perm).to_string())]
        }
        "build" => {
            let perm: &'static str = if state.last_build_perm == "edit" { "edit" } else { "ask" };
            vec![Step::SetPerm(perm, "Build agent.".to_string())]
        }
        "agent" => {
            if state.running {
                vec![Step::Toast(AGENT_BUSY.to_string())]
            } else {
                let next: &'static str = if state.perm == "plan" {
                    if state.last_build_perm == "edit" { "edit" } else { "ask" }
                } else {
                    "plan"
                };
                vec![Step::SetPerm(next, toggle_notice(next).to_string())]
            }
        }
        "remember" => {
            let global = regex::Regex::new(r"(?i)^global\b").expect("global").is_match(&arg_text);
            let entry = if global {
                regex::Regex::new(r"(?i)^global\b\s*").expect("global entry").replace(&arg_text, "").to_string()
            } else {
                arg_text.clone()
            };
            if entry.is_empty() {
                vec![Step::Notice(REMEMBER_USAGE.to_string())]
            } else {
                let scope: &'static str = if global { "global" } else { "workspace" };
                vec![Step::Remember {
                    scope,
                    entry: entry.clone(),
                    notice: remember_notice(global, &entry),
                }]
            }
        }
        "learn" => vec![Step::Learn(arg_text.clone())],
        "good" => vec![Step::Thumb("up")],
        "bad" => vec![Step::Thumb("down")],
        "autoreview" => {
            // only `on` and `off` change anything, and the notice follows either way
            let want: Option<bool> = match arg_text.to_ascii_lowercase().as_str() {
                "on" => Some(true),
                "off" => Some(false),
                _ => None,
            };
            let on = want.unwrap_or(state.auto_review);
            vec![Step::AutoReview(want, crate::review::autoreview_notice(on).to_string())]
        }
        "check" => vec![Step::UserTool {
            name: "diagnostics".into(),
            args: if arg_text.is_empty() {
                serde_json::json!({})
            } else {
                serde_json::json!({ "path": arg_text })
            },
        }],
        "review" => vec![Step::UserTool { name: "review".into(), args: crate::review::review_args(&arg_text) }],
        "security-review" | "security" => vec![Step::UserTool { name: "review".into(), args: crate::review::security_review_args() }],
        "worktree" | "wt" => {
            // `const [sub, name] = argText.split(/\s+/).filter(Boolean)`: the words of the
            // argument, the second one the name
            let mut words = arg_text.split_whitespace();
            let sub = words.next().unwrap_or("");
            let name = words.next().map(str::to_string);
            let action: &'static str = match sub {
                "new" => "new",
                "leave" => "leave",
                "remove" => "remove",
                _ => "list",
            };
            if matches!(action, "new" | "remove") && name.is_none() {
                vec![Step::Notice(WORKTREE_USAGE.to_string())]
            } else {
                vec![Step::Worktree { action, name }]
            }
        }
        "batch" => {
            // the first word is the branch, everything behind it the task (`argText.slice(name
            // .length).trim()`)
            let (name, rest) = match arg_text.split_once(char::is_whitespace) {
                Some((n, r)) => (n, r.trim()),
                None => (arg_text.as_str(), ""),
            };
            if name.is_empty() || rest.is_empty() {
                vec![Step::Notice(BATCH_USAGE.to_string())]
            } else {
                vec![Step::Batch { name: name.to_string(), task: rest.to_string() }]
            }
        }
        "spec" => {
            let words: Vec<&str> = arg_text.split_whitespace().collect();
            if words.is_empty() {
                vec![Step::SpecList]
            } else if words[0] == "run" {
                // the task is picked against the list the state has, reloaded when the name is
                // missing — as the page does, only across the round trip
                let name = words.get(1).copied().unwrap_or("").to_string();
                let asked = crate::specs::asked_task(words.get(2).copied());
                vec![Step::SpecRun { name, asked }]
            } else {
                let name = words[0].to_string();
                let description = words[1..].join(" ");
                if description.is_empty() {
                    vec![Step::Notice(crate::specs::SPEC_USAGE.to_string())]
                } else {
                    vec![Step::SpecNew { name, description, display: format!("/spec {}", arg_text) }]
                }
            }
        }
        "diff" => vec![Step::UserRun("git diff --stat && git diff".to_string(), true)],
        "status" => vec![Step::UserRun("git status -sb".to_string(), true)],
        "turbo" => {
            let want = arg_text.to_lowercase();
            let next = if want == "on" { true } else if want == "off" { false } else { !state.turbo };
            vec![Step::Turbo(next)]
        }
        "sandbox" => {
            let want = arg_text.to_lowercase();
            match want.as_str() {
                "off" | "workspace" | "strict" => vec![Step::SetSandbox(want)],
                _ => vec![Step::Notice(sandbox_invalid_notice(state.sandbox.as_deref(), state.sandbox_available))],
            }
        }
        "autoapprove" => {
            let want = crate::autoapprove::command(text)?;
            let after = want.unwrap_or(state.auto_approve);
            vec![Step::AutoApprove(want, crate::autoapprove::notice(after))]
        }
        "cost" | "usage" => {
            let text = match &state.usage {
                Some(summary) => cost_notice(summary),
                None => NO_USAGE.to_string(),
            };
            vec![Step::Notice(text)]
        }
        "search" => {
            if arg_text.is_empty() {
                vec![Step::Notice(SEARCH_USAGE.to_string())]
            } else {
                vec![Step::Search(arg_text.clone())]
            }
        }
        "stop" => {
            if state.running {
                vec![Step::Stop]
            } else {
                vec![Step::Notice(STOP_IDLE.to_string())]
            }
        }
        "insights" => vec![Step::Insights],
        "skill" => vec![Step::Skill],
        "rewind" => vec![Step::Rewind],
        "compact" => {
            if state.running {
                vec![Step::Toast("Wait until the agent has finished.".to_string())]
            } else {
                vec![Step::Compact(arg_text.clone())]
            }
        }
        "resume" | "sessions" => vec![Step::Sessions],
        "fork" => vec![Step::Fork],
        "revert" => vec![Step::Revert],
        "rename" => vec![Step::Rename((!arg_text.is_empty()).then(|| arg_text.clone()))],
        "export" => vec![Step::Export],
        "model" => vec![Step::PickModel(state.models.clone())],
        "memory" => vec![Step::Memory],
        "retry" => vec![Step::Retry],
        "personality" => {
            if arg_text.is_empty() {
                vec![Step::Notice(personality_info_notice())]
            } else if let Some((key, _, _)) = PERSONALITIES.iter().find(|(k, _, _)| *k == arg_text.to_lowercase()) {
                vec![Step::SetPersonality(key, personality_notice(key))]
            } else {
                vec![Step::Notice(personality_unknown_notice(&arg_text))]
            }
        }
        "cd" | "folder" => {
            if rest.is_empty() {
                vec![Step::OpenFolder(None)]
            } else {
                vec![Step::OpenFolder(Some(rest.join(" ")))]
            }
        }
        "allow" | "deny" => {
            if Rules::parse_side(&arg_text).is_empty() {
                vec![Step::Notice(rule_usage(&name))]
            } else {
                let (allow, deny) = if name == "allow" {
                    (format!("{}
{arg_text}", state.rules.allow).trim().to_string(), state.rules.deny.clone())
                } else {
                    (state.rules.allow.clone(), format!("{}
{arg_text}", state.rules.deny).trim().to_string())
                };
                vec![Step::SaveRules {
                    allow,
                    deny,
                    notice: rule_saved_notice(&name, &arg_text),
                }]
            }
        }
        "rules" => vec![Step::Notice(rules_notice(&state.rules))],
        "help" => vec![Step::Notice(help_text(&state.custom))],
        _ => {
            // a command of the table this window does not build keeps its place: a custom file
            // of the same name never runs on the page either (its case is answered first)
            if BUILTIN_COMMANDS.iter().any(|(n, _)| *n == name || *n == cmd) {
                return None;
            }
            // a custom command of `.fllm/commands` or one an installed plugin brings; the page
            // matches the lower-cased name first and the word as typed second
            let found = state.custom.iter().find(|c| c.name == name || c.name == cmd)?;
            vec![Step::CustomCommand {
                name: found.name.clone(),
                arg_text,
                source: text.to_string(),
            }]
        }
    };
    Some(steps)
}
