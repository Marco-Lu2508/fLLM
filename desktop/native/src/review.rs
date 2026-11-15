//! Row 81 of PARITY.md: *Review, self-review* — the pure half of the page's feature.
//!
//! Everything here comes from the page's own functions and is pinned by the fixtures that
//! running them produced (`tests/fixtures/command_fixtures.json`, arrays `review`, `autocheck`
//! and `reply`): `isCleanDiagnostic` and `reviewFeedback` of `codeprompts.js` (inside its
//! `@pure` block), `autoCheckEdit`, `patchPaths`, `clipOutput` and the reply template of
//! `code.js`, and the scope of `/review`.
//!
//! A JavaScript `.slice(0, n)` counts UTF-16 units, this port counts chars (`clip_middle` of
//! `commands.rs` is shared): identical for everything the fixtures carry, an astral character
//! near a boundary would land one position apart.

use serde_json::Value;

/// What the review of one run found: the text of the review tool and the diagnostics of the
/// changed files that are not clean (`report` of `selfReview`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    pub review: String,
    pub problems: Vec<Problem>,
}

/// One diagnostics result worth showing (`{ path, output }`).
#[derive(Debug, Clone, PartialEq)]
pub struct Problem {
    pub path: String,
    pub output: String,
}

/// `isCleanDiagnostic` (codeprompts.js 259-262): true when a diagnostics result says there is
/// nothing to fix (or nothing could be checked). `None` is what the page gets for `null`.
pub fn is_clean_diagnostic(output: Option<&str>) -> bool {
    let o = output.unwrap_or_default().trim_start();
    o.starts_with("No problems found")
        || o.contains("is not available here")
        || o.contains("no automatic check is known")
        || o.starts_with("No ")
}

/// `reviewFeedback` (codeprompts.js 265-286): the message that goes back to the model after its
/// edits. Round 1 is always sent (the model reads its own diff once more); later rounds only
/// when a check found something a machine can state. `None` when nothing is to be sent.
pub fn review_feedback(report: &Report, round: u32) -> Option<String> {
    let problems = &report.problems;
    let review = report.review.trim();
    let objective = !problems.is_empty() || review.contains("[high]");
    if round > 1 && !objective {
        return None;
    }
    let mut parts = vec!["Self-review before you finish. Check your own work against the task now.".to_string()];
    if !problems.is_empty() {
        let lines: Vec<String> = problems
            .iter()
            .map(|p| format!("- {}: {}", p.path, take_chars(p.output.trim(), 1500)))
            .collect();
        parts.push(format!("Diagnostics of the files you changed:\n{}", lines.join("\n")));
    }
    if !review.is_empty() {
        parts.push(format!(
            "Your changes and the findings of a fixed rule set:\n{}",
            take_chars(review, 12000)
        ));
    }
    parts.push(
        if objective {
            "Fix every real problem above, check again, then give the final report. If a finding is a false alarm, say so in one line."
        } else {
            "Nothing was flagged automatically. Re-read the diff: does it do exactly what the task asked, with no leftover debug code, no unrelated change and no missing case? Fix what is wrong, otherwise give the final report."
        }
        .to_string(),
    );
    Some(parts.join("\n\n"))
}

/// What the page's `.slice(0, n)` keeps (a char count, see the module note).
fn take_chars(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

/// The notice of the first review round (`selfReview`, round 1).
pub fn reviewing_notice(files: usize) -> String {
    format!(
        "Reviewing my own changes ({files} file{}) before finishing.",
        if files == 1 { "" } else { "s" }
    )
}

/// The notice of the second review round (`selfReview`, round 2) — it counts the files with
/// diagnostics, not the files of the run.
pub fn another_round_notice(problems: usize) -> String {
    format!(
        "The review found problems: another round ({problems} file{} with diagnostics).",
        if problems == 1 { "" } else { "s" }
    )
}

/// The answer of `/autoreview` (`case 'autoreview'` of `runCodeCommand`): the argument changes
/// the switch, and the notice follows either way.
pub fn autoreview_notice(on: bool) -> &'static str {
    if on {
        "Review before done: on. When a run changes files the agent reads its own diff and the diagnostics once more and fixes what is wrong."
    } else {
        "Review before done: off."
    }
}

/// `patchPaths` (code.js 236-250): the files a unified diff names, `b/` side first, without the
/// timestamps after a tab and without the quotes some tools wrap a name in.
pub fn patch_paths(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let lines: Vec<String> = text.replace("\r\n", "\n").split('\n').map(str::to_string).collect();
    // `.replace(/^"|"$/g, '')`: a quote at either edge goes, one at a time
    let clean = |s: &str| -> String {
        let mut t = s[4..].split('\t').next().unwrap_or("").trim().to_string();
        if t.starts_with('"') {
            t = t[1..].to_string();
        }
        if t.ends_with('"') && !t.is_empty() {
            t = t[..t.len() - 1].to_string();
        }
        for prefix in ["a/", "b/"] {
            if let Some(rest) = t.strip_prefix(prefix) {
                t = rest.to_string();
            }
        }
        t
    };
    for i in 0..lines.len().saturating_sub(1) {
        if !lines[i].starts_with("--- ") || !lines[i + 1].starts_with("+++ ") {
            continue;
        }
        let a = clean(&lines[i]);
        let b = clean(&lines[i + 1]);
        let p = if b == "/dev/null" { a } else { b };
        if !p.is_empty() && p != "/dev/null" && !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// The edit tools of the page (code.js 1358): after these, and only after these, the automatic
/// check of an edit may run.
pub const EDIT_TOOLS: [&str; 4] = ["write_file", "edit_file", "multi_edit", "patch"];

/// The extension set of `autoCheckEdit` (code.js 1937): only these are checked after an edit.
pub const CHECKED_EXTENSIONS: [&str; 9] = ["py", "js", "mjs", "cjs", "ts", "tsx", "rs", "go", "json"];

/// The paths `autoCheckEdit` would ask diagnostics for, in order: `paths.slice(0, 4)` of the
/// page, one path per extension (the `seen` set holds extensions, and it is filled *after* the
/// rule check — a denied path does not claim its extension), with a rule of
/// `bash: diagnostics <path>` skipping the file. `denied` answers that rule for one path.
pub fn auto_check_targets(name: &str, args: &Value, denied: &dyn Fn(&str) -> bool) -> Vec<String> {
    let paths: Vec<String> = if name == "patch" {
        args.get("patch").and_then(Value::as_str).map(patch_paths).unwrap_or_default()
    } else if let Some(path) = args.get("path").and_then(Value::as_str) {
        vec![path.to_string()]
    } else {
        Vec::new()
    };
    let mut seen: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for p in paths.iter().take(4) {
        let ext = p.rsplit('.').next().unwrap_or("").to_lowercase();
        if !CHECKED_EXTENSIONS.contains(&ext.as_str()) || seen.contains(&ext) {
            continue;
        }
        if denied(p) {
            continue;
        }
        seen.push(ext);
        out.push(p.clone());
    }
    out
}

/// What `autoCheckEdit` keeps of a diagnostics answer: not the clean line, and neither of the
/// two answers that say nothing could be done (`/^No problems found/`,
/// `/is not available here|did not finish within/`).
pub fn auto_check_keep(output: &str) -> bool {
    !output.starts_with("No problems found") && !output.contains("is not available here") && !output.contains("did not finish within")
}

/// The diagnostics call `autoCheckEdit` makes for one path (`{ path, timeout_s: 60 }`).
pub fn auto_check_args(path: &str) -> Value {
    serde_json::json!({ "path": path, "timeout_s": 60 })
}

/// `clipOutput` (code.js 1884-1887): the start and the end of an answer are kept — build and
/// test output has its errors and its summary at the end. The cap reads the model's context the
/// way the page does, in the page's own order of operations.
pub fn clip_output(text: &str, ctx_max: usize) -> String {
    let cap = ((ctx_max as f64 * 3.4) * 0.22).floor();
    let max = 3000f64.max(cap) as usize;
    crate::commands::clip_middle(text, max)
}

/// The reply the model gets back when the check after an edit found something (the template of
/// `runToolCall`, code.js 2044-2048): both halves clipped, the finding in between.
pub fn auto_check_reply(reply: &str, problems: &str, ctx_max: usize) -> String {
    format!(
        "{}\n\n[automatic check found problems]\n{}\nFix them if your change caused them.",
        clip_output(reply, ctx_max),
        clip_output(problems, ctx_max)
    )
}

/// The scope of `/review` (`case 'review'`): the first word picks it, the second is the base of
/// a branch review, and anything else is the plain diff.
pub fn review_scope(arg: &str) -> (&'static str, Option<String>) {
    let words: Vec<&str> = arg.split_whitespace().collect();
    let scope_word = words.first().copied().unwrap_or("");
    let scope = if scope_word == "staged" {
        "staged"
    } else if scope_word == "branch" {
        "branch"
    } else {
        "diff"
    };
    let base = if scope == "branch" {
        words.get(1).map(|s| s.to_string()).filter(|s| !s.is_empty())
    } else {
        None
    };
    (scope, base)
}

/// The arguments of `/review` (`case 'review'`): a branch carries its base, the rest only the
/// scope.
pub fn review_args(arg: &str) -> Value {
    let (scope, base) = review_scope(arg);
    match base {
        Some(base) => serde_json::json!({ "scope": scope, "base": base }),
        None => serde_json::json!({ "scope": scope }),
    }
}

/// The arguments of `/security-review` and `/security` (`case 'security-review'`).
pub fn security_review_args() -> Value {
    serde_json::json!({ "scope": "diff", "security": true })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two notices of `selfReview` with their plurals, as the page words them.
    #[test]
    fn the_review_notices_count_the_way_the_page_counts() {
        assert_eq!(
            reviewing_notice(1),
            "Reviewing my own changes (1 file) before finishing."
        );
        assert_eq!(
            reviewing_notice(2),
            "Reviewing my own changes (2 files) before finishing."
        );
        assert_eq!(
            another_round_notice(1),
            "The review found problems: another round (1 file with diagnostics)."
        );
        assert_eq!(
            another_round_notice(3),
            "The review found problems: another round (3 files with diagnostics)."
        );
    }
}
