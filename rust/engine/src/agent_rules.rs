//! Permission rules for the headless agent: allow / deny / ask patterns per kind of action, read from a settings
//! hierarchy (user < project < local), in the same `category: pattern` form the Studio uses, plus the spelling of
//! Claude Code (`Bash(npm test:*)`, `Edit(src/**)`, `Read(.env)`, `WebFetch(domain:example.com)`).
//!
//! Order of decision: a **deny** rule always wins, then an **ask** rule (the call is put to the question; without a
//! person it stays refused), then an **allow** rule (the call runs even when no `--allow-*` flag covers it). A call no
//! rule mentions falls back to the flags. An allow rule never covers a shell command with operators (`;`, `&&`, `|`,
//! redirects, substitutions) unless the rule is that exact command, and it only covers a risky program (`rm`, `curl`,
//! `sudo`, `git`, shells ...) when the rule names the program.
use regex::Regex;
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub category: String,
    pub pattern: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rules {
    pub allow: Vec<Rule>,
    pub deny: Vec<Rule>,
    pub ask: Vec<Rule>,
    /// `plan` when a settings file asks for read-only planning.
    pub mode: Option<String>,
}

const RISKY: [&str; 26] = [
    "rm", "rmdir", "del", "erase", "rd", "format", "shutdown", "reboot", "mkfs", "dd", "sudo", "su", "curl", "wget", "iwr", "invoke-webrequest", "powershell", "pwsh", "cmd", "sh", "bash", "chmod", "chown", "kill",
    "taskkill", "git",
];
const CATEGORIES: [&str; 8] = ["bash", "edit", "network", "mcp", "command", "read", "memory", "*"];

fn is_risky(base: &str) -> bool {
    RISKY.contains(&base)
}

fn has_shell_ops(command: &str) -> bool {
    command.chars().any(|c| matches!(c, ';' | '&' | '|' | '>' | '<' | '`' | '$' | '(' | ')' | '{' | '}' | '\n'))
}

fn command_segments(command: &str) -> Vec<String> {
    let mut parts = vec![String::new()];
    let chars: Vec<char> = command.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if matches!(c, ';' | '&' | '|' | '\n') {
            while i + 1 < chars.len() && matches!(chars[i + 1], '&' | '|') {
                i += 1;
            }
            parts.push(String::new());
        } else if let Some(last) = parts.last_mut() {
            last.push(c);
        }
        i += 1;
    }
    parts.into_iter().map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect()
}

/// The program name of a command: first word without folder and `.exe`, lower case.
pub fn command_base(command: &str) -> String {
    let first = command.split_whitespace().next().unwrap_or("");
    let name = first.rsplit(['/', '\\']).next().unwrap_or("").to_ascii_lowercase();
    for ext in [".exe", ".cmd", ".bat"] {
        if let Some(stripped) = name.strip_suffix(ext) {
            return stripped.to_owned();
        }
    }
    name
}

fn norm_path(p: &str) -> String {
    let p = p.replace('\\', "/");
    let p = p.strip_prefix("./").unwrap_or(&p);
    p.trim_start_matches('/').to_owned()
}

/// `*` and `?` patterns. Path mode: `*` stops at `/`, `**` does not. Text mode: `*` matches anything.
fn glob(pattern: &str, path_mode: bool) -> Option<Regex> {
    let p: Vec<char> = pattern.chars().collect();
    let mut re = String::from("(?s)^");
    let mut i = 0;
    while i < p.len() {
        let c = p[i];
        if c == '*' {
            if path_mode {
                if p.get(i + 1) == Some(&'*') {
                    i += 1;
                    if p.get(i + 1) == Some(&'/') {
                        i += 1;
                        re.push_str("(?:.*/)?");
                    } else {
                        re.push_str(".*");
                    }
                } else {
                    re.push_str("[^/]*");
                }
            } else {
                while p.get(i + 1) == Some(&'*') {
                    i += 1;
                }
                re.push_str(".*");
            }
        } else if c == '?' {
            re.push_str(if path_mode { "[^/]" } else { "." });
        } else {
            re.push_str(&regex::escape(&c.to_string()));
        }
        i += 1;
    }
    re.push('$');
    Regex::new(&re).ok()
}

fn pattern_matches(rule: &Rule, rule_category: &str, subject: &str) -> bool {
    let path = rule_category == "edit" || rule_category == "read";
    let (pattern, subject) = if path { (norm_path(&rule.pattern), norm_path(subject)) } else { (rule.pattern.clone(), subject.trim().to_owned()) };
    glob(&pattern, path).is_some_and(|re| re.is_match(&subject))
}

fn applies(rule: &Rule, category: &str) -> bool {
    rule.category == "*" || rule.category == category || (rule.category == "command" && (category == "bash" || category == "mcp"))
}

fn bash_allowed(rule: &Rule, subject: &str) -> bool {
    if !pattern_matches(rule, "bash", subject) {
        return false;
    }
    if has_shell_ops(subject) && rule.pattern.trim() != subject.trim() {
        return false;
    }
    let base = command_base(subject);
    if is_risky(&base) {
        let head = rule.pattern.split_whitespace().next().unwrap_or("");
        if head.contains(['*', '?']) || command_base(head) != base {
            return false;
        }
    }
    true
}

/// Parses one rule line: `bash: npm test*` or the Claude Code spelling `Bash(npm test:*)`.
pub fn parse_rule(line: &str) -> Option<Rule> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    if let Some(open) = line.find('(').filter(|_| line.ends_with(')')) {
        let tool = &line[..open];
        let inner = &line[open + 1..line.len() - 1];
        let category = match tool {
            "Bash" | "PowerShell" => "bash",
            "Edit" | "Write" | "MultiEdit" | "NotebookEdit" => "edit",
            "Read" | "Glob" | "Grep" | "LS" => "read",
            "WebFetch" | "WebSearch" => "network",
            _ => "",
        };
        if !category.is_empty() {
            let pattern = match category {
                "bash" => inner.strip_suffix(":*").map_or_else(|| inner.to_owned(), |head| format!("{head}*")),
                "network" => inner.strip_prefix("domain:").map_or_else(|| inner.to_owned(), |d| format!("*{d}*")),
                _ => inner.to_owned(),
            };
            return Some(Rule { category: category.into(), pattern: if pattern.is_empty() { "*".into() } else { pattern } });
        }
    }
    // a bare Claude Code tool name covers everything of that kind
    match line {
        "Bash" => return Some(Rule { category: "bash".into(), pattern: "*".into() }),
        "Edit" | "Write" | "MultiEdit" => return Some(Rule { category: "edit".into(), pattern: "**".into() }),
        "Read" => return Some(Rule { category: "read".into(), pattern: "**".into() }),
        "WebFetch" | "WebSearch" => return Some(Rule { category: "network".into(), pattern: "*".into() }),
        _ => {}
    }
    let (category, pattern) = line.split_once(':')?;
    let category = category.trim().to_ascii_lowercase();
    let pattern = pattern.trim();
    (CATEGORIES.contains(&category.as_str()) && !pattern.is_empty()).then(|| Rule { category, pattern: pattern.to_owned() })
}

fn parse_list(value: &Value) -> Vec<Rule> {
    match value {
        Value::Array(items) => items.iter().filter_map(|v| v.as_str()).filter_map(parse_rule).collect(),
        Value::String(text) => text.lines().filter_map(parse_rule).collect(),
        _ => Vec::new(),
    }
}

impl Rules {
    pub fn is_empty(&self) -> bool {
        self.allow.is_empty() && self.deny.is_empty() && self.ask.is_empty()
    }

    /// `{"allow": [...], "deny": [...], "ask": [...], "mode": "plan"}`, or the same under a `permissions` key (the shape of a Claude Code
    /// `settings.json`, whose `defaultMode: "plan"` is understood too).
    pub fn from_json(value: &Value) -> Rules {
        let v = if value["permissions"].is_object() { &value["permissions"] } else { value };
        let mode = v["mode"].as_str().or_else(|| v["defaultMode"].as_str()).map(str::to_owned);
        Rules { allow: parse_list(&v["allow"]), deny: parse_list(&v["deny"]), ask: parse_list(&v["ask"]), mode }
    }

    /// Adds the rules of a higher-priority file. Deny, ask and allow lists are joined; the later mode wins.
    pub fn merge(&mut self, other: Rules) {
        self.allow.extend(other.allow);
        self.deny.extend(other.deny);
        self.ask.extend(other.ask);
        if other.mode.is_some() {
            self.mode = other.mode;
        }
    }

    /// The rule decision for a subject of a rule category (`bash`, `edit`, `read`, `network`, `memory`), or `None` when no rule speaks.
    pub fn evaluate(&self, category: &str, subject: &str) -> Option<Decision> {
        let hit = |list: &[Rule], bash_mode: bool| -> bool {
            list.iter().filter(|r| applies(r, category)).any(|r| {
                if category == "bash" {
                    if bash_mode {
                        // deny and ask match the whole line or any part of a compound line
                        std::iter::once(subject.trim().to_owned()).chain(command_segments(subject)).any(|part| pattern_matches(r, "bash", &part))
                    } else {
                        bash_allowed(r, subject)
                    }
                } else {
                    pattern_matches(r, category, subject)
                }
            })
        };
        if hit(&self.deny, true) {
            Some(Decision::Deny)
        } else if hit(&self.ask, true) {
            Some(Decision::Ask)
        } else if hit(&self.allow, false) {
            Some(Decision::Allow)
        } else {
            None
        }
    }

    /// The decision for a tool call of the catalog. A call with several subjects (a patch that touches many files) is denied when any is
    /// denied, asked when any is asked, and allowed only when every subject is allowed.
    pub fn decide(&self, tool: &str, permission: &str, args: &Value) -> Option<Decision> {
        let (category, subjects) = subjects_of(tool, permission, args);
        if subjects.is_empty() || self.is_empty() {
            return None;
        }
        let decisions: Vec<Option<Decision>> = subjects.iter().map(|s| self.evaluate(category, s)).collect();
        if decisions.contains(&Some(Decision::Deny)) {
            Some(Decision::Deny)
        } else if decisions.contains(&Some(Decision::Ask)) {
            Some(Decision::Ask)
        } else if decisions.iter().all(|d| *d == Some(Decision::Allow)) {
            Some(Decision::Allow)
        } else {
            None
        }
    }
}

fn first_str<'a>(args: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| args[*k].as_str())
}

/// The rule category and the subjects (command, paths, urls) a tool call acts on.
pub fn subjects_of(tool: &str, permission: &str, args: &Value) -> (&'static str, Vec<String>) {
    match permission {
        "command" => ("bash", first_str(args, &["command", "cmd"]).map(str::to_owned).into_iter().collect()),
        "edit" => {
            let mut paths: Vec<String> = first_str(args, &["path", "file_path", "file"]).map(str::to_owned).into_iter().collect();
            if let Some(patch) = first_str(args, &["patch", "diff"]) {
                for line in patch.lines() {
                    if let Some(rest) = line.strip_prefix("+++ ").or_else(|| line.strip_prefix("--- ")) {
                        let name = rest.split('\t').next().unwrap_or("").trim();
                        let name = name.strip_prefix("b/").or_else(|| name.strip_prefix("a/")).unwrap_or(name);
                        if !name.is_empty() && name != "/dev/null" && !paths.iter().any(|p| p == name) {
                            paths.push(name.to_owned());
                        }
                    }
                }
            }
            ("edit", paths)
        }
        "network" => ("network", first_str(args, &["url", "query"]).map(str::to_owned).into_iter().collect()),
        "memory" => ("memory", vec![first_str(args, &["text", "op"]).unwrap_or("*").to_owned()]),
        _ => {
            let _ = tool;
            ("read", first_str(args, &["path", "pattern", "file_path", "query"]).map(str::to_owned).into_iter().collect())
        }
    }
}

/// Loads the settings hierarchy: the Studio folder (user), then the project (`.fllm/permissions.json`), then the local, uncommitted
/// file (`.fllm/permissions.local.json`). A missing file is fine; a damaged one is reported as an error so a typo cannot silently drop
/// a deny rule.
pub fn load_hierarchy(studio: &Path, folder: &Path) -> Result<Rules, String> {
    let mut rules = Rules::default();
    for file in [studio.join("permissions.json"), folder.join(".fllm").join("permissions.json"), folder.join(".fllm").join("permissions.local.json")] {
        match std::fs::read_to_string(&file) {
            Ok(text) => {
                let value: Value = serde_json::from_str(&text).map_err(|e| format!("{} is not valid JSON: {e}", file.display()))?;
                rules.merge(Rules::from_json(&value));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("cannot read {}: {e}", file.display())),
        }
    }
    Ok(rules)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rules(allow: &[&str], deny: &[&str], ask: &[&str]) -> Rules {
        let l = |v: &[&str]| v.iter().filter_map(|s| parse_rule(s)).collect::<Vec<_>>();
        Rules { allow: l(allow), deny: l(deny), ask: l(ask), mode: None }
    }

    #[test]
    fn both_spellings_parse() {
        assert_eq!(parse_rule("bash: npm test*"), Some(Rule { category: "bash".into(), pattern: "npm test*".into() }));
        assert_eq!(parse_rule("Bash(npm test:*)"), Some(Rule { category: "bash".into(), pattern: "npm test*".into() }));
        assert_eq!(parse_rule("Read(./.env)").unwrap().category, "read");
        assert_eq!(parse_rule("WebFetch(domain:example.com)").unwrap().pattern, "*example.com*");
        assert_eq!(parse_rule("Edit").unwrap().pattern, "**");
        assert!(parse_rule("# comment").is_none());
        assert!(parse_rule("nonsense: x").is_none());
        assert!(parse_rule("bash:").is_none());
    }

    #[test]
    fn deny_beats_ask_beats_allow() {
        let r = rules(&["bash: git *"], &["bash: git push*"], &["bash: git commit*"]);
        assert_eq!(r.evaluate("bash", "git push origin main"), Some(Decision::Deny));
        assert_eq!(r.evaluate("bash", "git commit -m x"), Some(Decision::Ask));
        assert_eq!(r.evaluate("bash", "git status"), Some(Decision::Allow));
        assert_eq!(r.evaluate("bash", "ls"), None);
    }

    #[test]
    fn an_allow_rule_never_covers_a_compound_or_a_risky_program_by_wildcard() {
        let r = rules(&["bash: npm test*", "bash: *"], &[], &[]);
        assert_eq!(r.evaluate("bash", "npm test && rm -rf /"), None);
        let r = rules(&["bash: npm test*"], &["bash: rm *"], &[]);
        assert_eq!(r.evaluate("bash", "npm test && rm -rf x"), Some(Decision::Deny), "a deny rule sees every part");
        let r = rules(&["bash: npm test*"], &[], &[]);
        assert_eq!(r.evaluate("bash", "npm test; echo hi"), None);
        let r = rules(&["bash: *"], &[], &[]);
        assert_eq!(r.evaluate("bash", "rm -rf x"), None, "a wildcard head does not cover a risky program");
        let r = rules(&["bash: rm -rf build"], &[], &[]);
        assert_eq!(r.evaluate("bash", "rm -rf build"), Some(Decision::Allow));
    }

    #[test]
    fn path_rules_use_double_star_for_folders() {
        let r = rules(&["edit: src/**"], &["edit: **/.env", "read: secrets/**"], &[]);
        assert_eq!(r.evaluate("edit", "src/a/b.rs"), Some(Decision::Allow));
        assert_eq!(r.evaluate("edit", "./src/a.rs"), Some(Decision::Allow));
        assert_eq!(r.evaluate("edit", "docs/a.md"), None);
        assert_eq!(r.evaluate("edit", "src/deep/.env"), Some(Decision::Deny));
        assert_eq!(r.evaluate("read", "secrets/key.txt"), Some(Decision::Deny));
        assert_eq!(r.evaluate("read", "src/key.txt"), None);
    }

    #[test]
    fn a_call_is_judged_by_all_its_subjects() {
        let r = rules(&["edit: src/**"], &["edit: .github/**"], &[]);
        let patch = "--- a/src/x.rs\n+++ b/src/x.rs\n@@ -1 +1 @@\n-a\n+b\n--- a/.github/ci.yml\n+++ b/.github/ci.yml\n@@ -1 +1 @@\n-a\n+b\n";
        assert_eq!(r.decide("patch", "edit", &json!({"patch": patch})), Some(Decision::Deny));
        let only_src = "--- a/src/x.rs\n+++ b/src/x.rs\n@@ -1 +1 @@\n-a\n+b\n";
        assert_eq!(r.decide("patch", "edit", &json!({"patch": only_src})), Some(Decision::Allow));
        let mixed = "--- a/src/x.rs\n+++ b/src/x.rs\n--- a/docs/y.md\n+++ b/docs/y.md\n";
        assert_eq!(r.decide("patch", "edit", &json!({"patch": mixed})), None);
        assert_eq!(r.decide("bash", "command", &json!({"command": "ls"})), None);
    }

    #[test]
    fn settings_json_shapes_and_the_hierarchy() {
        let user = Rules::from_json(&json!({"permissions": {"allow": ["Bash(cargo test:*)"], "deny": ["Read(.env)"], "defaultMode": "plan"}}));
        assert_eq!(user.mode.as_deref(), Some("plan"));
        let mut all = Rules::default();
        all.merge(user);
        all.merge(Rules::from_json(&json!({"allow": "bash: make*\n# note\nedit: src/**", "ask": ["bash: make deploy*"]})));
        assert_eq!(all.allow.len(), 3);
        assert_eq!(all.evaluate("bash", "make deploy prod"), Some(Decision::Ask));
        assert_eq!(all.evaluate("bash", "cargo test -p x"), Some(Decision::Allow));
        assert_eq!(all.evaluate("read", ".env"), Some(Decision::Deny));
    }

    #[test]
    fn files_are_merged_in_order_and_a_broken_one_is_an_error() {
        let dir = std::env::temp_dir().join(format!("fllm-rules-{}", std::process::id()));
        let studio = dir.join("studio");
        let project = dir.join("proj");
        std::fs::create_dir_all(project.join(".fllm")).unwrap();
        std::fs::create_dir_all(&studio).unwrap();
        std::fs::write(studio.join("permissions.json"), r#"{"deny": ["bash: rm *"]}"#).unwrap();
        std::fs::write(project.join(".fllm/permissions.json"), r#"{"allow": ["bash: make*"]}"#).unwrap();
        std::fs::write(project.join(".fllm/permissions.local.json"), r#"{"ask": ["bash: make install*"]}"#).unwrap();
        let r = load_hierarchy(&studio, &project).unwrap();
        assert_eq!((r.deny.len(), r.allow.len(), r.ask.len()), (1, 1, 1));
        std::fs::write(project.join(".fllm/permissions.local.json"), "{oops").unwrap();
        assert!(load_hierarchy(&studio, &project).unwrap_err().contains("not valid JSON"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
