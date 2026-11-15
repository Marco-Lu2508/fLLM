//! Automatic approval of the Code page: a fixed table of harmless programs decides whether a
//! command runs without a question.
//!
//! Port of the `@pure-begin`…`@pure-end` block of `code.js` (lines 19–560): `HARMLESS`,
//! `GIT_DANGER`, `DANGER_WORDS`, `commandParts`, `commandHead`, `classifyCommand` and
//! `autoVerdict`, with their rule order and their sentences. The cases the page answers for this
//! round are in `tests/fixtures/autoapprove_fixtures.json`, written by `autoapprove_fixtures.js`
//! running the page's own code — the DIFF of that file was empty before any of this was wired.

use regex::Regex;
use std::sync::OnceLock;

/// The table of harmless programs and the subcommands they are harmless with (`HARMLESS` of the
/// page): an empty list means every arguments are fine — the other rules still judge them.
const HARMLESS: &[(&str, &[&str])] = &[
    // reading
    (
        "ls",
        &[],
    ),
    ("cat", &[]),
    ("head", &[]),
    ("tail", &[]),
    ("wc", &[]),
    ("pwd", &[]),
    ("echo", &[]),
    ("printf", &[]),
    ("date", &[]),
    ("whoami", &[]),
    ("uname", &[]),
    ("which", &[]),
    ("type", &[]),
    ("sort", &[]),
    ("uniq", &[]),
    ("cut", &[]),
    ("tr", &[]),
    ("basename", &[]),
    ("dirname", &[]),
    ("realpath", &[]),
    ("stat", &[]),
    ("file", &[]),
    ("du", &[]),
    ("df", &[]),
    ("tree", &[]),
    ("jq", &[]),
    ("grep", &[]),
    ("rg", &[]),
    ("find", &[]),
    ("sed", &[]),
    ("awk", &[]),
    ("diff", &[]),
    ("cmp", &[]),
    ("column", &[]),
    ("seq", &[]),
    ("hostname", &[]),
    ("python3", &[]),
    ("python", &[]),
    ("node", &[]),
    // building and testing
    (
        "cargo",
        &[
            "build",
            "check",
            "test",
            "clippy",
            "fmt",
            "doc",
            "metadata",
            "tree",
            "bench",
            "locate-project",
            "--version",
            "-V",
        ],
    ),
    ("rustc", &["--version", "-V", "--explain"]),
    ("rustfmt", &["--check", "--version"]),
    ("npm", &["run", "test", "ls", "--version", "-v"]),
    ("npx", &["tsc", "prettier"]),
    ("tsc", &[]),
    ("prettier", &["--check"]),
    ("eslint", &[]),
    ("pytest", &[]),
    ("pyright", &[]),
    ("go", &["build", "test", "vet", "fmt", "version", "list", "env"]),
    ("make", &[]),
    ("cmake", &["--build", "--version"]),
    ("deno", &["check", "test", "fmt", "lint", "--version"]),
    ("bun", &["test", "run", "--version"]),
    // git, reading and staging only
    (
        "git",
        &[
            "status",
            "diff",
            "log",
            "show",
            "add",
            "blame",
            "describe",
            "rev-parse",
            "ls-files",
            "shortlog",
        ],
    ),
];

/// The subcommands of git that change history or the working tree (`GIT_DANGER`).
const GIT_DANGER: &[&str] = &[
    "push",
    "reset",
    "clean",
    "rebase",
    "merge",
    "cherry-pick",
    "revert",
    "checkout",
    "switch",
    "restore",
    "rm",
    "mv",
    "gc",
    "prune",
    "filter-branch",
    "am",
];

/// The words that make a harmless program look dangerous (`DANGER_WORDS` of the page).
const DANGER_WORDS: &str = r"\b(rm|rmdir|mv|cp|chmod|chown|chgrp|dd|mkfs\w*|truncate|shred|shutdown|reboot|halt|poweroff|kill|pkill|killall|mount|umount|systemctl|service|docker|podman|kubectl|helm|terraform|ansible|apt|apt-get|yum|dnf|pacman|brew|snap|flatpak|pip|pip3|npm|yarn|pnpm|curl|wget|nc|netcat|ncat|ssh|scp|sftp|rsync|tee|sudo|su|doas|chroot|insmod|modprobe|ln|useradd|passwd|crontab|at|nohup|setsid|eval|exec|source)\b";

fn danger_words() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(DANGER_WORDS).expect("the words of the page are a regular expression"))
}

/// The answer of the judge: `safe` when the command only reads, builds or tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub safe: bool,
    pub reason: String,
}

impl Verdict {
    fn safe() -> Self {
        Verdict {
            safe: true,
            reason: "reads, builds or tests only".to_string(),
        }
    }

    fn risky(reason: impl Into<String>) -> Self {
        Verdict {
            safe: false,
            reason: reason.into(),
        }
    }
}

/// Splits a command line into the parts that run on their own (`commandParts`): `&&`, `||`, `;`
/// and both sides of a pipe, trimmed, nothing empty kept.
pub fn command_parts(command: &str) -> Vec<String> {
    static SPLIT: OnceLock<Regex> = OnceLock::new();
    let split = SPLIT.get_or_init(|| Regex::new(r"&&|\|\||;|\|").expect("the separators of the page"));
    split
        .split(command)
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

/// The program of one part, without leading `NAME=value` pairs and without a path
/// (`commandHead`): the words after those pairs are its arguments.
pub fn command_head(part: &str) -> (String, Vec<String>) {
    let words: Vec<&str> = part.split_whitespace().collect();
    static ASSIGN: OnceLock<Regex> = OnceLock::new();
    let assign = ASSIGN.get_or_init(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*=").expect("a name="));
    let mut i = 0;
    while i < words.len() && assign.is_match(words[i]) {
        i += 1;
    }
    let word = words.get(i).copied().unwrap_or("");
    let head = word.rsplit('/').next().unwrap_or(word).to_string();
    // `slice(i + 1)` of the page never runs past the end
    let rest: Vec<String> = words
        .get(i + 1..)
        .unwrap_or(&[])
        .iter()
        .map(|word| (*word).to_string())
        .collect();
    (head, rest)
}

/// The quotes are taken out of a line before it is judged: a word inside `"…"` cannot hide
/// another one there (the page strips them the same way).
fn without_quotes(text: &str) -> String {
    static QUOTES: OnceLock<Regex> = OnceLock::new();
    let quotes = QUOTES.get_or_init(|| Regex::new(r#""[^"]*"|'[^']*'"#).expect("a quoted run"));
    quotes.replace_all(text, "").into_owned()
}

/// The substitution runs inside the command are judged themselves (`$(…)` and backticks), with
/// their own recursion — no nesting of brackets is looked at, as on the page.
fn substitutions_safe(text: &str) -> Result<(), String> {
    static SUBS: OnceLock<Regex> = OnceLock::new();
    let subs = SUBS.get_or_init(|| Regex::new(r"\$\(([^()]*)\)|`([^`]*)`").expect("a substitution"));
    for captures in subs.captures_iter(text) {
        let inner = captures.get(1).or_else(|| captures.get(2)).map(|m| m.as_str()).unwrap_or("");
        let verdict = classify_command(inner);
        if !verdict.safe {
            return Err(format!("inside the substitution: {}", verdict.reason));
        }
    }
    Ok(())
}

/// The `sed` of the page is approved for printing a range of lines and nothing else
/// (`sed -n '3,9p' file`): its scripts can write files with `w` and run programs with `e`.
/// The pattern of the page holds a backreference for its quote, which Rust's expressions cannot
/// carry, so the two quotes are asked one after the other — the same words either way.
fn sed_prints(part: &str) -> bool {
    static SINGLE: OnceLock<Regex> = OnceLock::new();
    static DOUBLE: OnceLock<Regex> = OnceLock::new();
    static LEAD: OnceLock<Regex> = OnceLock::new();
    let single = SINGLE.get_or_init(|| Regex::new(r"^sed\s+-n\s+'(\d+(,\d+|\$)?)p'(\s+[^\s-]\S*)+$").expect("a sed printing"));
    let double = DOUBLE.get_or_init(|| Regex::new(r#"^sed\s+-n\s+"(\d+(,\d+|\$)?)p"(\s+[^\s-]\S*)+$"#).expect("a sed printing"));
    let lead = LEAD.get_or_init(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*=\S*\s+").expect("a name="));
    let line = lead.replace(part.trim(), "").into_owned();
    single.is_match(&line) || double.is_match(&line)
}

/// Is this command harmless? (`classifyCommand` of the page.) Everything is decided by the table
/// plus four hard rules — a redirect that is not `/dev/null`, a heredoc, a substitution with
/// something risky inside, and any part whose program is not on the list — with the checks of the
/// special heads in the order the page puts them.
pub fn classify_command(command: &str) -> Verdict {
    let text = command.trim();
    if text.is_empty() {
        return Verdict::risky("the command is empty");
    }
    // one simple line: a second line, a background job, a subshell, a group or a process
    // substitution can hide a program
    if text.contains('\r') || text.contains('\n') {
        return Verdict::risky("it has more than one line");
    }
    let bare = without_quotes(text);
    static BACKGROUND: OnceLock<Regex> = OnceLock::new();
    let background = BACKGROUND.get_or_init(|| Regex::new(r"(^|[^&|>\d])&($|[^&>])").expect("a background job"));
    if background.is_match(&bare) {
        return Verdict::risky("it starts a background job");
    }
    static GROUP: OnceLock<Regex> = OnceLock::new();
    let group = GROUP.get_or_init(|| Regex::new(r"\$\{|<\(|>\(|(^|[\s;|&])[({]").expect("a group"));
    if group.is_match(&bare) {
        return Verdict::risky("it uses a group, a subshell or an expansion");
    }
    static REDIRECT: OnceLock<Regex> = OnceLock::new();
    let redirect = REDIRECT.get_or_init(|| Regex::new(r"(^|\s)(\d?)>>?\s*([^\s|&;]+)").expect("a redirect"));
    if let Some(captures) = redirect.captures(text) {
        let target = captures.get(3).map(|m| m.as_str()).unwrap_or("");
        if target != "/dev/null" {
            return Verdict::risky(format!("it writes into {target} through a redirect"));
        }
    }
    static HEREDOC: OnceLock<Regex> = OnceLock::new();
    let heredoc = HEREDOC.get_or_init(|| Regex::new(r#"<<-?\s*['"]?[A-Za-z_]"#).expect("a heredoc"));
    if heredoc.is_match(text) {
        return Verdict::risky("it uses a heredoc, which writes a file or feeds a program");
    }
    if let Err(reason) = substitutions_safe(text) {
        return Verdict::risky(reason);
    }
    let parts = command_parts(text);
    if parts.is_empty() {
        return Verdict::risky("there is nothing to run");
    }
    // the regex of `awk` is built once, outside the loop over the parts (clippy reads the
    // `get_or_init` of a loop body as a compilation in a loop)
    static AWK: OnceLock<Regex> = OnceLock::new();
    let awk = AWK.get_or_init(|| Regex::new(r#"system\(|getline|\|"|print\s*>|printf\s*>"#).expect("awk"));
    for part in &parts {
        let (head, args) = command_head(part);
        if head.is_empty() {
            continue;
        }
        let allowed = match HARMLESS.iter().find(|(name, _)| *name == head) {
            Some((_, allowed)) => *allowed,
            None => return Verdict::risky(format!("`{head}` is not on the list of harmless programs")),
        };
        if head == "git" {
            // options before the subcommand (`-c`, `-C`, `--exec-path`, `--git-dir`) can
            // configure or redirect it
            if args.first().is_some_and(|arg| arg.starts_with('-')) {
                return Verdict::risky("`git` is given an option before its command");
            }
            let sub = args.iter().find(|arg| !arg.starts_with('-')).map(String::as_str).unwrap_or("");
            if GIT_DANGER.contains(&sub) {
                return Verdict::risky(format!("`git {sub}` changes history or the working tree"));
            }
            let writes = args.iter().any(|arg| {
                arg.starts_with("--output")
                    || arg.starts_with("--ext-diff")
                    || arg.starts_with("--textconv")
                    || arg.starts_with("--open-files-in-pager")
                    || arg.starts_with("--exec-path")
                    || arg.starts_with("--upload-pack")
                    || arg.starts_with("--receive-pack")
                    || arg == "-o"
            });
            if writes {
                return Verdict::risky("a `git` option writes a file or starts a program");
            }
        }
        // sed is allowed to print a range of lines, nothing else
        if head == "sed" && !sed_prints(part) {
            return Verdict::risky("`sed` is only approved for `sed -n '3,9p' file`");
        }
        if head == "sort"
            && args
                .iter()
                .any(|arg| arg == "-o" || arg.starts_with("-o") || arg.starts_with("--output") || arg.starts_with("--compress-program"))
        {
            return Verdict::risky("`sort` would write a file or start a program");
        }
        if head == "rg" && args.iter().any(|arg| arg.starts_with("--pre") || arg == "--hostname-bin") {
            return Verdict::risky("`rg` would start a program");
        }
        let find_writes = ["-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"];
        if head == "find" && args.iter().any(|arg| find_writes.contains(&arg.as_str())) {
            return Verdict::risky("`find` would run a program or write or delete files");
        }
        if head == "awk" {
            let joined = args.join(" ");
            if awk.is_match(&joined) {
                return Verdict::risky("`awk` would start a program or write a file");
            }
        }
        // interpreters run whatever they are given, so only a test runner and --version pass
        if head == "python3" || head == "python" {
            let at = args.iter().position(|arg| arg == "-m");
            let module = at.and_then(|at| args.get(at + 1)).map(String::as_str).unwrap_or("");
            let test = ["pytest", "unittest", "json.tool", "py_compile", "compileall"].contains(&module);
            let version = args.iter().any(|arg| arg == "--version" || arg == "-V");
            if !test && !version {
                return Verdict::risky(format!("`{head}` would run code that is no test"));
            }
        }
        if head == "node" {
            let version = args.iter().any(|arg| arg == "--version" || arg == "-v");
            let check = args.iter().any(|arg| arg == "--check");
            if !version && !check {
                return Verdict::risky("`node` would run code that is no syntax check");
            }
        }
        if !allowed.is_empty() {
            let sub = args
                .iter()
                .find(|arg| !arg.starts_with('-'))
                .or_else(|| args.iter().find(|arg| arg.starts_with('-')))
                .map(String::as_str)
                .unwrap_or("");
            let ok = allowed.iter().any(|allowed| {
                sub == *allowed || sub.starts_with(&format!("{allowed}=")) || (allowed.starts_with('-') && sub == *allowed)
            });
            if !ok {
                return Verdict::risky(format!("`{head} {sub}` is not on the list of harmless subcommands"));
            }
        }
        if danger_words().is_match(&without_quotes(&args.join(" "))) {
            return Verdict::risky(format!("`{head}` is given something that changes files or the system"));
        }
    }
    Verdict::safe()
}

/// The verdict for a whole call (`autoVerdict`): only shell commands can be approved
/// automatically, and only when they are harmless. `category` is the permission category of the
/// call and `name` the tool.
pub fn auto_verdict(category: &str, name: &str, subject: &str) -> Verdict {
    if category != "bash" {
        return Verdict::risky(format!("a {category} call is never approved on its own"));
    }
    if name != "bash" {
        return Verdict::risky(format!("`{name}` is not a plain shell command"));
    }
    classify_command(subject)
}

/// The command in the box (`runCodeCommand` of `code.js`): the first word after the slash is
/// lower-cased and asked for `autoapprove`, and only the words `on` and `off` change the switch —
/// everything else (including nothing at all) leaves it standing. `None` when the words are no
/// such command, `Some(want)` when they are.
pub fn command(text: &str) -> Option<Option<bool>> {
    let rest = text.trim().strip_prefix('/')?;
    let word = rest.split_whitespace().next()?;
    if word.to_lowercase() != "autoapprove" {
        return None;
    }
    let arg = rest[word.len()..].trim().to_lowercase();
    Some(match arg.as_str() {
        "on" => Some(true),
        "off" => Some(false),
        _ => None,
    })
}

/// The words of the `/autoapprove` notice of the page (`codeNotice` of the command): what the
/// switch means now, and the two ways to say it.
pub fn notice(on: bool) -> String {
    let sentence = if on {
        "Automatic approval is on: harmless commands (reading, building, testing) run without a question; everything else still asks."
    } else {
        "Automatic approval is off: every command asks again."
    };
    format!("{sentence}\n/autoapprove on · /autoapprove off")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cases the page itself answered: `tests/fixtures/autoapprove_fixtures.json`, written by
    /// `autoapprove_fixtures.js` running the block of `code.js`. The script prints a DIFF line
    /// for every case where this round's belief and the page disagree; it was empty.
    #[test]
    fn fixtures_from_the_page() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/autoapprove_fixtures.json");
        let fixtures: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("the fixtures of the page")).expect("json");
        for case in fixtures["classify"].as_array().expect("the classify cases") {
            let command = case[0].as_str().expect("the command");
            let verdict = classify_command(command);
            assert_eq!(
                (verdict.safe, verdict.reason.as_str()),
                (case[1].as_str() == Some("safe"), case[2].as_str().expect("the reason")),
                "classify {command:?}"
            );
        }
        for case in fixtures["auto"].as_array().expect("the autoVerdict cases") {
            let verdict = auto_verdict(
                case[0].as_str().expect("the category"),
                case[1].as_str().expect("the tool"),
                case[2].as_str().expect("the subject"),
            );
            assert_eq!(
                (verdict.safe, verdict.reason.as_str()),
                (case[3].as_str() == Some("safe"), case[4].as_str().expect("the reason")),
                "autoVerdict {:?}",
                case[2]
            );
        }
        for case in fixtures["parts"].as_array().expect("the commandParts cases") {
            let parts = command_parts(case[0].as_str().expect("the command"));
            let expected: Vec<String> = case[1].as_array().expect("the parts").iter().map(|p| p.as_str().expect("a part").to_string()).collect();
            assert_eq!(parts, expected, "commandParts {:?}", case[0]);
        }
        for case in fixtures["head"].as_array().expect("the commandHead cases") {
            let (head, args) = command_head(case[0].as_str().expect("the part"));
            let expected: Vec<String> = case[1][1].as_array().expect("the args").iter().map(|a| a.as_str().expect("an arg").to_string()).collect();
            assert_eq!(head, case[1][0].as_str().expect("the head"), "commandHead head {:?}", case[0]);
            assert_eq!(args, expected, "commandHead args {:?}", case[0]);
        }
    }

    /// The notice of the command, both ways, with the line the page puts behind the sentence.
    #[test]
    fn the_notice_says_what_the_switch_means() {
        assert_eq!(
            notice(true),
            "Automatic approval is on: harmless commands (reading, building, testing) run without a question; everything else still asks.\n/autoapprove on · /autoapprove off"
        );
        assert_eq!(
            notice(false),
            "Automatic approval is off: every command asks again.\n/autoapprove on · /autoapprove off"
        );
    }
}
