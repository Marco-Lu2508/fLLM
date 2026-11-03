//! `fllm-rs review`: a code review of a pull request, a branch or the working tree by the coding agent, for a terminal or for CI.
//!
//! ```text
//! fllm-rs review --pr 42                 review pull request 42 of this repository (needs the GitHub CLI `gh`, signed in)
//! fllm-rs review --pr 42 --comment       ... and post the review as a comment on the pull request
//! fllm-rs review --base main             review what this branch changes against main (plain git)
//! fllm-rs review --staged | (nothing)    review the staged changes, or all uncommitted changes
//! ```
//!
//! The agent gets the diff and the pull request text and may read the rest of the repository with its read-only tools, so it can
//! check a call against the function it calls. It cannot change anything: the review run has no edit, command or network rights.
//! The answer is a list of findings (`path:line - severity - problem - suggestion`) and a last line `VERDICT: ok` or `VERDICT: issues`.
//! Exit codes: 0 finished (also with findings), 1 error, 5 with `--fail-on-issues` when the verdict is `issues`.
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const HELP: &str = "usage: fllm-rs review [--pr N | --base BRANCH | --staged] [--comment] [--security] [--fail-on-issues] [--folder DIR] [--max-diff-chars N] [AGENT OPTIONS]
Reviews a pull request (--pr N, uses the GitHub CLI gh), the changes of this branch against BRANCH, the staged changes or all uncommitted changes.
The agent reads the diff and may read the repository (read-only: it cannot edit, run commands or use the network). The review lists findings as
`path:line - severity - problem - suggestion` and ends with `VERDICT: ok` or `VERDICT: issues`. --comment posts it on the pull request
(gh pr comment); the comment starts with a hidden marker so a CI job can find and replace its earlier one. --security looks for security problems only.
--fail-on-issues makes the exit code 5 when the verdict is issues (for CI). AGENT OPTIONS are passed on to `fllm-rs agent`: --url --api-key --model --max-steps --profile.
Setup for GitHub Actions: docs/user/coding.md and .github/workflows/fllm-review.yml.example.";

pub const MARKER: &str = "<!-- fllm-review -->";
const DEFAULT_MAX_DIFF: usize = 40_000;

pub struct Target {
    pub title: String,
    pub body: String,
    pub diff: String,
}

pub fn clip_diff(diff: &str, max: usize) -> (String, bool) {
    if diff.chars().count() <= max {
        return (diff.to_owned(), false);
    }
    // cut at a line break so the last file is not torn in the middle of a line
    let head: String = diff.chars().take(max).collect();
    let cut = head.rfind('\n').unwrap_or(head.len());
    (head[..cut].to_owned(), true)
}

pub fn build_prompt(target: &Target, security: bool, cut: bool) -> String {
    let focus = if security {
        "Look only for security problems: injection, unchecked input at a boundary, secrets in the code, unsafe file or command handling, missing authorization, unsafe defaults. Ignore style and everything that is not a security matter."
    } else {
        "Look for real problems: wrong logic, missed edge cases and error paths, broken callers of changed functions, missing or wrong tests, security problems, leftover debugging code. Ignore formatting and taste."
    };
    let mut p = format!("You are reviewing a code change. {focus}\n\nTitle: {}\n", if target.title.is_empty() { "(none)" } else { &target.title });
    if !target.body.trim().is_empty() {
        p.push_str(&format!("Description:\n{}\n", target.body.trim()));
    }
    p.push_str(&format!("\nThe diff:\n```diff\n{}\n```\n", target.diff.trim_end()));
    if cut {
        p.push_str("\nThe diff was cut because it is long; read the other changed files with the tools if you need them.\n");
    }
    p.push_str("\nCheck a suspicion in the repository before you report it (read the function that is called, grep for other callers). Report only findings you can point to. One finding per line: `path:line - severity (high, medium, low) - the problem - what to change`. If there is nothing to report, say so in one sentence. Finish with a last line `VERDICT: ok` (nothing that must be fixed) or `VERDICT: issues`.");
    p
}

/// True when the last verdict line says `issues`.
pub fn verdict_is_issues(answer: &str) -> bool {
    answer.lines().rev().find_map(|l| l.trim().strip_prefix("VERDICT:").map(|v| v.trim().to_ascii_lowercase())).is_some_and(|v| v.starts_with("issues"))
}

pub fn format_comment(answer: &str, security: bool) -> String {
    let title = if security { "fLLM security review" } else { "fLLM review" };
    format!("{MARKER}\n### {title}\n\n{}\n\n<sub>Written by a language model reading the diff; check each finding before acting on it.</sub>\n", answer.trim())
}

/// How `gh` and `git` are run: arguments in, text out (tests supply a fake).
pub type Tool<'a> = dyn FnMut(&str, &[String], &Path) -> Result<String, String> + 'a;

fn real_tool(program: &str, args: &[String], cwd: &Path) -> Result<String, String> {
    let out = std::process::Command::new(program).args(args).current_dir(cwd).stdin(std::process::Stdio::null()).output().map_err(|e| {
        if program == "gh" { format!("cannot start gh ({e}); install the GitHub CLI and run `gh auth login`") } else { format!("cannot start {program}: {e}") }
    })?;
    if !out.status.success() {
        return Err(format!("{program} {} failed: {}", args.first().map_or("", String::as_str), String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn fetch_target(pr: Option<u64>, base: Option<&str>, staged: bool, cwd: &Path, tool: &mut Tool) -> Result<Target, String> {
    if let Some(n) = pr {
        let n = n.to_string();
        let diff = tool("gh", &["pr".into(), "diff".into(), n.clone()], cwd)?;
        let meta = tool("gh", &["pr".into(), "view".into(), n.clone(), "--json".into(), "title,body".into()], cwd).unwrap_or_default();
        let v: Value = serde_json::from_str(&meta).unwrap_or(Value::Null);
        return Ok(Target { title: v["title"].as_str().unwrap_or_default().to_owned(), body: v["body"].as_str().unwrap_or_default().to_owned(), diff });
    }
    let args: Vec<String> = match (base, staged) {
        (Some(b), _) => vec!["diff".into(), format!("{b}...HEAD")],
        (None, true) => vec!["diff".into(), "--staged".into()],
        (None, false) => vec!["diff".into(), "HEAD".into()],
    };
    let diff = tool("git", &args, cwd)?;
    Ok(Target { title: base.map_or_else(|| "uncommitted changes".to_owned(), |b| format!("changes against {b}")), body: String::new(), diff })
}

#[derive(Debug, Default)]
struct Parsed {
    pr: Option<u64>,
    base: Option<String>,
    staged: bool,
    comment: bool,
    security: bool,
    fail: bool,
    folder: Option<PathBuf>,
    max_diff: usize,
    forward: Vec<String>,
}

fn parse(args: &[String]) -> Result<Parsed, String> {
    let mut p = Parsed { max_diff: DEFAULT_MAX_DIFF, ..Parsed::default() };
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let val = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or_else(|| format!("missing value for {a}"))
        };
        match a {
            "--pr" => p.pr = Some(val(&mut i)?.trim_start_matches('#').parse().map_err(|_| "--pr is a number".to_owned())?),
            "--base" => p.base = Some(val(&mut i)?),
            "--staged" => p.staged = true,
            "--comment" => p.comment = true,
            "--security" => p.security = true,
            "--fail-on-issues" => p.fail = true,
            "--folder" => p.folder = Some(PathBuf::from(val(&mut i)?)),
            "--max-diff-chars" => p.max_diff = val(&mut i)?.parse().ok().filter(|n| *n >= 2000).ok_or("--max-diff-chars is a number of at least 2000")?,
            "--url" | "--api-key" | "--model" | "--max-steps" | "--max-tokens" | "--temperature" | "--profile" => {
                p.forward.push(a.to_owned());
                p.forward.push(val(&mut i)?);
            }
            "--help" | "-h" => return Err(HELP.to_owned()),
            other => return Err(format!("unknown option {other}; see fllm-rs review --help")),
        }
        i += 1;
    }
    if p.comment && p.pr.is_none() {
        return Err("--comment needs --pr N: a comment belongs to a pull request".into());
    }
    Ok(p)
}

pub fn run(args: &[String]) -> Result<i32, String> {
    let mut agent = |a: &[String]| crate::spec::run_agent(a);
    run_with(args, &mut real_tool, &mut agent)
}

pub fn run_with(args: &[String], tool: &mut Tool, agent: &mut crate::spec::AgentRunner) -> Result<i32, String> {
    let p = match parse(args) {
        Ok(p) => p,
        Err(e) if e == HELP => {
            println!("{HELP}");
            return Ok(0);
        }
        Err(e) => return Err(e),
    };
    let cwd = match &p.folder {
        Some(f) => f.clone(),
        None => std::env::current_dir().map_err(|e| e.to_string())?,
    };
    let target = fetch_target(p.pr, p.base.as_deref(), p.staged, &cwd, tool)?;
    if target.diff.trim().is_empty() {
        println!("Nothing to review: the diff is empty.");
        return Ok(0);
    }
    let (diff, cut) = clip_diff(&target.diff, p.max_diff);
    let prompt = build_prompt(&Target { diff, ..target }, p.security, cut);
    let mut agent_args: Vec<String> = vec!["-p".into(), prompt, "--folder".into(), cwd.display().to_string(), "--output-format".into(), "json".into(), "--quiet".into(), "--plan".into()];
    agent_args.extend(p.forward.clone());
    let (code, report) = agent(&agent_args)?;
    if code != 0 && code != 3 {
        return Err(format!("the agent ended with exit code {code}"));
    }
    let answer = report["answer"].as_str().unwrap_or_default().trim().to_owned();
    if answer.is_empty() {
        return Err("the agent gave no review".into());
    }
    println!("{answer}");
    if p.comment {
        if let Some(n) = p.pr {
            let body = format_comment(&answer, p.security);
            tool("gh", &["pr".into(), "comment".into(), n.to_string(), "--body".into(), body], &cwd)?;
            eprintln!("[review] comment posted on pull request {n}");
        }
    }
    if p.fail && verdict_is_issues(&answer) {
        return Ok(5);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|i| (*i).to_owned()).collect()
    }

    #[test]
    fn the_prompt_has_the_diff_the_text_and_the_focus() {
        let t = Target { title: "Add export".into(), body: "Adds CSV.".into(), diff: "+let x = 1;\n".into() };
        let p = build_prompt(&t, false, false);
        assert!(p.contains("Title: Add export") && p.contains("Adds CSV.") && p.contains("+let x = 1;") && p.contains("VERDICT: issues") && !p.contains("was cut"));
        assert!(build_prompt(&t, true, true).contains("only for security problems") && build_prompt(&t, true, true).contains("was cut"));
        let (short, cut) = clip_diff("aaaa\nbbbb\ncccc\n", 7);
        assert_eq!((short.as_str(), cut), ("aaaa", true));
        assert!(!clip_diff("abc", 10).1);
    }

    #[test]
    fn verdicts_and_comments() {
        assert!(verdict_is_issues("1. a.rs:3 - high - x\nVERDICT: issues"));
        assert!(!verdict_is_issues("Looks fine.\nVERDICT: ok\n"));
        assert!(!verdict_is_issues("no verdict at all"));
        let c = format_comment("  text  ", false);
        assert!(c.starts_with(MARKER) && c.contains("### fLLM review") && c.contains("text") && c.contains("check each finding"));
    }

    #[test]
    fn a_pull_request_is_reviewed_and_commented_and_the_run_cannot_edit() {
        let calls = std::cell::RefCell::new(Vec::<String>::new());
        let mut tool = |program: &str, args: &[String], _: &Path| -> Result<String, String> {
            calls.borrow_mut().push(format!("{program} {}", args.join(" ")));
            match (program, args.get(1).map(String::as_str)) {
                ("gh", Some("diff")) => Ok("diff --git a/x b/x\n+bad()\n".into()),
                ("gh", Some("view")) => Ok(json!({"title": "T", "body": "B"}).to_string()),
                _ => Ok(String::new()),
            }
        };
        let seen = std::cell::RefCell::new(Vec::<Vec<String>>::new());
        let mut agent = |a: &[String]| -> Result<(i32, Value), String> {
            seen.borrow_mut().push(a.to_vec());
            Ok((0, json!({"answer": "x:1 - high - bad call - remove\nVERDICT: issues"})))
        };
        let args = s(&["--pr", "#7", "--comment", "--fail-on-issues", "--folder", ".", "--model", "m"]);
        assert_eq!(run_with(&args, &mut tool, &mut agent).unwrap(), 5);
        let c = calls.borrow();
        assert_eq!(c[0], "gh pr diff 7");
        assert!(c[1].starts_with("gh pr view 7") && c[2].starts_with("gh pr comment 7 --body <!-- fllm-review -->"), "{c:?}");
        let a = &seen.borrow()[0];
        assert!(a.contains(&"--plan".to_owned()) && !a.contains(&"--allow-edit".to_owned()) && a.contains(&"--model".to_owned()));
        assert!(a[1].contains("+bad()") && a[1].contains("Title: T"));
    }

    #[test]
    fn an_empty_diff_needs_no_model_and_comment_needs_a_pull_request() {
        let mut tool = |_: &str, _: &[String], _: &Path| -> Result<String, String> { Ok("  \n".into()) };
        let mut agent = |_: &[String]| -> Result<(i32, Value), String> { panic!("no model call for an empty diff") };
        assert_eq!(run_with(&s(&["--staged"]), &mut tool, &mut agent).unwrap(), 0);
        assert!(run_with(&s(&["--comment"]), &mut tool, &mut agent).unwrap_err().contains("--pr"));
        assert!(run_with(&s(&["--bogus"]), &mut tool, &mut agent).is_err());
        // a plain branch review uses git, not gh
        let used = std::cell::RefCell::new(String::new());
        let mut git = |p: &str, a: &[String], _: &Path| -> Result<String, String> {
            *used.borrow_mut() = format!("{p} {}", a.join(" "));
            Ok(String::new())
        };
        run_with(&s(&["--base", "main"]), &mut git, &mut agent).unwrap();
        assert_eq!(*used.borrow(), "git diff main...HEAD");
    }
}
