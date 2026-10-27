//! The `review` tool: the changes of the workspace as a unified diff plus a deterministic scan of the added lines.
//!
//! The model gets both halves at once: the diff it should read (like a reviewer would) and the findings a fixed set of
//! rules already made, with file and line. That keeps a review honest on small models too, and `/security-review` can
//! ask for the security rules only. Nothing here changes a file; the command line is fixed and never comes from the
//! model (only the base of a branch comparison does, and it is checked).
use super::agent::{collect_output, AgentSession};
use serde_json::Value;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

const MAX_DIFF_CHARS: usize = 40_000;
const MAX_FINDINGS: usize = 200;
const TIMEOUT: Duration = Duration::from_secs(20);

struct Rule {
    name: &'static str,
    severity: &'static str,
    security: bool,
    pattern: &'static str,
}

/// The rules. `security` marks the ones `/security-review` keeps.
const RULES: [Rule; 12] = [
    Rule { name: "possible secret in an assignment", severity: "high", security: true, pattern: r#"(?i)(api[_-]?key|secret|token|passw(or)?d|private[_-]?key)\s*[:=]\s*["'][^"']{8,}"# },
    Rule { name: "known key or token format", severity: "high", security: true, pattern: r"(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)" },
    Rule { name: "eval or exec of a string", severity: "high", security: true, pattern: r"\b(eval|exec)\s*\(" },
    Rule { name: "shell call with interpolation", severity: "high", security: true, pattern: r"(subprocess\.(run|call|Popen)\([^)]*shell\s*=\s*True|os\.system\(|child_process\.exec\()" },
    Rule { name: "SQL built by concatenation", severity: "medium", security: true, pattern: r#"(?i)(select|insert|update|delete)\b[^;"']*(\+\s*|\$\{)(req|params|input|args|query)"# },
    Rule { name: "merge conflict marker", severity: "high", security: true, pattern: r"^(<<<<<<< |>>>>>>> |=======$)" },
    Rule { name: "plain http URL", severity: "low", security: true, pattern: r"http://[A-Za-z0-9.-]+" },
    Rule { name: "leftover TODO/FIXME", severity: "low", security: false, pattern: r"\b(TODO|FIXME|XXX|HACK)\b" },
    Rule { name: "debug output left in", severity: "low", security: false, pattern: r"(console\.log\(|dbg!\(|println!\(|System\.out\.print)" },
    Rule { name: "unwrap() without a message", severity: "low", security: false, pattern: r"\.unwrap\(\)" },
    Rule { name: "unsafe block", severity: "medium", security: false, pattern: r"\bunsafe\s*\{" },
    Rule { name: "catch-all or ignored error", severity: "low", security: false, pattern: r"(catch\s*\{\s*\}|except\s*:\s*pass|let\s+_\s*=\s*.*\.ok\(\))" },
];

fn compiled() -> Vec<(usize, regex::Regex)> {
    RULES
        .iter()
        .enumerate()
        .filter_map(|(i, rule)| regex::Regex::new(rule.pattern).ok().map(|re| (i, re)))
        .collect()
}

/// Only a branch or commit name may reach the command line.
fn safe_ref(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= 100
        && !text.starts_with('-')
        && text.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-'))
}

fn run_command(dir: &Path, line: &str) -> Result<String, String> {
    let mut cmd = super::agent_jobs::shell_command(line);
    let mut child = cmd
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run git: {e}"))?;
    let (out, err) = (collect_output(child.stdout.take()), collect_output(child.stderr.take()));
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if started.elapsed() >= TIMEOUT => {
                super::agent_jobs::kill_tree(child.id());
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => return Err(e.to_string()),
        }
    };
    let stdout = String::from_utf8_lossy(&out.join().unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&err.join().unwrap_or_default()).into_owned();
    if status.is_none() {
        return Err("git did not answer within 20 s".to_owned());
    }
    if !status.is_some_and(|s| s.success()) {
        let text = if stderr.trim().is_empty() { stdout } else { stderr };
        return Err(format!("git failed: {}", text.trim().chars().take(300).collect::<String>()));
    }
    Ok(stdout)
}

/// A finding of the scan.
#[derive(Debug, PartialEq)]
struct Finding {
    path: String,
    line: u32,
    rule: &'static str,
    severity: &'static str,
    snippet: String,
}

/// Scans the added lines of a unified diff. Pure, so the rules are tested without git.
fn scan(diff: &str, security_only: bool) -> Vec<Finding> {
    let compiled = compiled();
    let mut findings = Vec::new();
    let mut path = String::new();
    let mut line = 0u32;
    for raw in diff.lines() {
        if let Some(rest) = raw.strip_prefix("+++ ") {
            path = rest.trim_start_matches("b/").to_owned();
            if path == "/dev/null" {
                path.clear();
            }
            continue;
        }
        if raw.starts_with("@@") {
            // @@ -a,b +c,d @@ -> the next added line is c
            if let Some(plus) = raw.split('+').nth(1) {
                let start: String = plus.chars().take_while(char::is_ascii_digit).collect();
                line = start.parse().unwrap_or(0);
            }
            continue;
        }
        if let Some(text) = raw.strip_prefix('+') {
            if raw.starts_with("+++") {
                continue;
            }
            for (index, re) in &compiled {
                let rule = &RULES[*index];
                if security_only && !rule.security {
                    continue;
                }
                if re.is_match(text) {
                    findings.push(Finding {
                        path: if path.is_empty() { "-".to_owned() } else { path.clone() },
                        line,
                        rule: rule.name,
                        severity: rule.severity,
                        snippet: text.trim().chars().take(160).collect(),
                    });
                    if findings.len() >= MAX_FINDINGS {
                        return findings;
                    }
                }
            }
            line = line.saturating_add(1);
        } else if !raw.starts_with('-') {
            line = line.saturating_add(1);
        }
    }
    findings
}

fn count_lines(diff: &str) -> (usize, usize, usize) {
    let mut files = 0;
    let (mut added, mut removed) = (0, 0);
    for raw in diff.lines() {
        if raw.starts_with("+++ ") {
            files += 1;
        } else if raw.starts_with('+') && !raw.starts_with("+++") {
            added += 1;
        } else if raw.starts_with('-') && !raw.starts_with("---") {
            removed += 1;
        }
    }
    (files, added, removed)
}

pub(super) fn run(session: &AgentSession, args: &Value) -> Result<String, String> {
    let scope = args.get("scope").and_then(Value::as_str).unwrap_or("diff");
    let security = args.get("security").and_then(Value::as_bool).unwrap_or(false);
    let root = session.root_dir()?;
    let line = match scope {
        "staged" => "git diff --no-color --staged".to_owned(),
        "branch" => {
            let base = args.get("base").and_then(Value::as_str).unwrap_or("HEAD");
            if !safe_ref(base) {
                return Err("the base of a branch comparison must be a branch, tag or commit name".to_owned());
            }
            format!("git diff --no-color {base}...HEAD")
        }
        _ => "git diff --no-color".to_owned(),
    };
    let diff = run_command(&root, &line)?;
    if diff.trim().is_empty() {
        return Ok("No changes to review: the working tree is clean for this scope.".to_owned());
    }
    let (files, added, removed) = count_lines(&diff);
    let findings = scan(&diff, security);
    let mut text = format!(
        "Review ({}{}): {files} file(s), +{added}/-{removed} lines, {} finding(s) from the fixed rules.\n",
        if security { "security" } else { "diff" },
        match scope {
            "staged" => ", staged",
            "branch" => ", against the base branch",
            _ => "",
        },
        findings.len()
    );
    if findings.is_empty() {
        text.push_str("The rules found nothing; read the diff yourself and judge.\n");
    } else {
        text.push_str("\nFindings of the fixed rules (severity, rule, file:line, the added line):\n");
        for f in &findings {
            text.push_str(&format!("- [{}] {} at {}:{}: `{}`\n", f.severity, f.rule, f.path, f.line, f.snippet));
        }
        text.push_str("\nThese are leads, not verdicts: open the file and say whether each one is a real problem.\n");
    }
    let clipped: String = diff.chars().take(MAX_DIFF_CHARS).collect();
    text.push_str("\n--- diff ---\n");
    text.push_str(&clipped);
    if diff.chars().count() > MAX_DIFF_CHARS {
        text.push_str("\n[diff cut: read the files for the rest]");
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIFF: &str = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1,3 +1,4 @@\n x = 1\n+api_key = \"sk-supersecretvalue1234\"\n+print(x)  # TODO remove\n y = 2\n";

    #[test]
    fn added_lines_are_scanned_with_their_line_number() {
        let findings = scan(DIFF, false);
        let paths: Vec<(String, u32)> = findings.iter().map(|f| (f.path.clone(), f.line)).collect();
        assert!(paths.contains(&("src/app.py".to_owned(), 2)));
        assert!(paths.iter().any(|(_, line)| *line == 3));
        assert!(findings.iter().any(|f| f.rule == "known key or token format"));
        assert!(findings.iter().any(|f| f.rule == "leftover TODO/FIXME"));
    }

    #[test]
    fn the_security_only_scan_drops_the_style_rules() {
        let all = scan(DIFF, false);
        let security = scan(DIFF, true);
        assert!(security.len() < all.len());
        assert!(security.iter().all(|f| f.rule == "known key or token format" || f.severity == "high"));
        assert!(!security.iter().any(|f| f.rule == "leftover TODO/FIXME"));
    }

    #[test]
    fn removed_lines_are_not_reported() {
        let diff = "--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n-api_key = \"sk-abcdefghijklmnop\"\n+safe = 1\n";
        assert!(scan(diff, true).iter().all(|f| f.snippet.contains("safe")));
    }

    #[test]
    fn the_line_counter_and_the_ref_check_are_strict() {
        assert_eq!(count_lines(DIFF), (1, 2, 0));
        assert!(safe_ref("main"));
        assert!(safe_ref("origin/release-1.2"));
        assert!(!safe_ref("--upload-pack=evil"));
        assert!(!safe_ref("main; rm -rf /"));
        assert!(!safe_ref(""));
    }
}
