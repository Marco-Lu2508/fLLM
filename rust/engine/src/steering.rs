//! Steering files: standing instructions in `.fllm/steering/*.md`, next to `AGENTS.md`, with a small front matter that says when
//! each file applies (the idea and the key names come from Kiro's steering files).
//!
//! ```markdown
//! ---
//! inclusion: fileMatch
//! fileMatchPattern: ["**/*.tsx", "src/ui/**"]
//! ---
//! Components are function components. Styles live in the .module.css next to them.
//! ```
//!
//! `inclusion` is one of
//! * `always` (the default): the file is part of every prompt, like `AGENTS.md`;
//! * `fileMatch`: the text is added to the result of a tool call that reads or changes a file matching `fileMatchPattern`, once per session;
//! * `manual`: only listed by name; the agent reads it when the task needs it, or you ask for it (`--steering NAME`);
//! * `auto`: like `manual`, and the one-line `description` tells the agent when it applies.
//!
//! Files in the Studio folder under `steering/` apply to every workspace; the workspace file of the same name wins.
use regex::Regex;
use std::path::{Path, PathBuf};

pub const MAX_FILES: usize = 40;
const MAX_BODY_CHARS: usize = 6_000;
const MAX_ALWAYS_TOTAL: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inclusion {
    Always,
    FileMatch,
    Manual,
    Auto,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Steering {
    /// The file name without `.md`.
    pub name: String,
    pub inclusion: Inclusion,
    pub patterns: Vec<String>,
    pub description: String,
    pub body: String,
    /// Where it was read from, as shown to the agent (`.fllm/steering/api.md`, or the full path for a user file).
    pub file: String,
}

/// Splits `---` front matter from the body. A file without front matter is all body.
fn split_front_matter(text: &str) -> (Vec<(String, String)>, &str) {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = text.split_inclusive('\n');
    let Some(first) = lines.next() else { return (Vec::new(), text) };
    if first.trim_end() != "---" {
        return (Vec::new(), text);
    }
    let mut consumed = first.len();
    let mut pairs: Vec<(String, String)> = Vec::new();
    for line in lines {
        consumed += line.len();
        let trimmed = line.trim_end();
        if trimmed == "---" {
            return (pairs, &text[consumed..]);
        }
        if let Some(item) = trimmed.trim_start().strip_prefix("- ") {
            // a block list continues the value of the key before it
            if let Some(last) = pairs.last_mut() {
                if last.1.is_empty() {
                    last.1 = format!("[{}]", item.trim());
                } else if last.1.starts_with('[') && last.1.ends_with(']') {
                    last.1.insert_str(last.1.len() - 1, &format!(", {}", item.trim()));
                }
            }
            continue;
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            pairs.push((key.trim().to_owned(), value.trim().to_owned()));
        }
    }
    // never closed: it was not front matter
    (Vec::new(), text)
}

fn unquote(text: &str) -> String {
    let t = text.trim();
    for q in ['"', '\''] {
        if t.len() >= 2 && t.starts_with(q) && t.ends_with(q) {
            return t[1..t.len() - 1].to_owned();
        }
    }
    t.to_owned()
}

/// `"a"`, `a`, `["a", "b"]`, `[a, b]` -> the list of patterns.
fn parse_list(value: &str) -> Vec<String> {
    let v = value.trim();
    let inner = v.strip_prefix('[').and_then(|r| r.strip_suffix(']')).unwrap_or(v);
    if !v.starts_with('[') {
        let one = unquote(v);
        return if one.is_empty() { Vec::new() } else { vec![one] };
    }
    inner.split(',').map(unquote).filter(|s| !s.is_empty()).collect()
}

pub fn parse(name: &str, file: &str, text: &str) -> Steering {
    let (pairs, body) = split_front_matter(text);
    let get = |key: &str| pairs.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)).map(|(_, v)| v.as_str());
    let patterns = get("fileMatchPattern").or_else(|| get("fileMatchPatterns")).map(parse_list).unwrap_or_default();
    let inclusion = match get("inclusion").map(|v| unquote(v).to_ascii_lowercase()).as_deref() {
        Some("filematch") => Inclusion::FileMatch,
        Some("manual") => Inclusion::Manual,
        Some("auto") => Inclusion::Auto,
        // an unknown word or a missing key: the safe reading is "always", the file does what its author most likely wanted
        _ => Inclusion::Always,
    };
    // a fileMatch file without a pattern could never apply
    let inclusion = if inclusion == Inclusion::FileMatch && patterns.is_empty() { Inclusion::Manual } else { inclusion };
    let body: String = body.trim().chars().take(MAX_BODY_CHARS).collect();
    Steering {
        name: get("name").map(unquote).filter(|n| !n.is_empty()).unwrap_or_else(|| name.to_owned()),
        inclusion,
        patterns,
        description: get("description").map(unquote).unwrap_or_default(),
        body,
        file: file.to_owned(),
    }
}

fn read_dir(dir: &Path, shown: &dyn Fn(&str) -> String) -> Vec<Steering> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut names: Vec<(String, PathBuf)> = entries
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("md")) && e.path().is_file())
        .filter_map(|e| Some((e.path().file_stem()?.to_str()?.to_owned(), e.path())))
        .collect();
    names.sort();
    names
        .into_iter()
        .take(MAX_FILES)
        .filter_map(|(name, path)| {
            let text = std::fs::read_to_string(&path).ok()?;
            let item = parse(&name, &shown(&format!("{name}.md")), &text);
            (!item.body.is_empty()).then_some(item)
        })
        .collect()
}

/// The steering files of a workspace: the user's own (`user_dir`) first, then those of `.fllm/steering/`, which win by name.
pub fn load(root: &Path, user_dir: Option<&Path>) -> Vec<Steering> {
    let mut all: Vec<Steering> = Vec::new();
    if let Some(dir) = user_dir {
        let base = dir.display().to_string().replace('\\', "/");
        all.extend(read_dir(dir, &|n| format!("{base}/{n}")));
    }
    for item in read_dir(&root.join(".fllm").join("steering"), &|n| format!(".fllm/steering/{n}")) {
        all.retain(|x| x.name != item.name);
        all.push(item);
    }
    all.truncate(MAX_FILES);
    all
}

/// `**`, `*`, `?` against a path with `/` separators. A pattern without a `/` matches the file name at any depth (`*.rs`).
pub fn glob_matches(pattern: &str, path: &str) -> bool {
    let path = path.replace('\\', "/");
    let path = path.trim_start_matches("./");
    let pattern = pattern.trim().trim_start_matches("./");
    if pattern.is_empty() {
        return false;
    }
    let mut re = String::from("^");
    if !pattern.contains('/') {
        re.push_str("(?:.*/)?");
    }
    let chars: Vec<char> = pattern.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '*' if chars.get(i + 1) == Some(&'*') => {
                i += 1;
                // `**/` also matches nothing, so `**/x` covers `x`
                if chars.get(i + 1) == Some(&'/') {
                    i += 1;
                    re.push_str("(?:.*/)?");
                } else {
                    re.push_str(".*");
                }
            }
            '*' => re.push_str("[^/]*"),
            '?' => re.push_str("[^/]"),
            '{' => {
                // {a,b} alternatives
                if let Some(close) = chars[i..].iter().position(|c| *c == '}') {
                    let alts: Vec<String> = chars[i + 1..i + close].iter().collect::<String>().split(',').map(regex::escape).collect();
                    re.push_str(&format!("(?:{})", alts.join("|")));
                    i += close;
                } else {
                    re.push_str("\\{");
                }
            }
            c => re.push_str(&regex::escape(&c.to_string())),
        }
        i += 1;
    }
    re.push('$');
    Regex::new(&re).is_ok_and(|r| r.is_match(path))
}

impl Steering {
    pub fn applies_to(&self, path: &str) -> bool {
        self.inclusion == Inclusion::FileMatch && self.patterns.iter().any(|p| glob_matches(p, path))
    }
}

/// The `always` files as `(shown name, text)` pairs for the instruction block, within a total cap.
pub fn always_files(all: &[Steering]) -> Vec<(String, String)> {
    let mut total = 0usize;
    let mut out = Vec::new();
    for s in all.iter().filter(|s| s.inclusion == Inclusion::Always) {
        if total + s.body.len() > MAX_ALWAYS_TOTAL {
            break;
        }
        total += s.body.len();
        out.push((s.file.clone(), s.body.clone()));
    }
    out
}

/// A short list of the files the agent may read when a task needs them (manual, auto and file-matching files), or empty.
pub fn index_text(all: &[Steering]) -> String {
    let rows: Vec<String> = all
        .iter()
        .filter(|s| s.inclusion != Inclusion::Always)
        .map(|s| {
            let when = match s.inclusion {
                Inclusion::FileMatch => format!("added automatically when you read or change {}", s.patterns.join(", ")),
                Inclusion::Auto => "read it when the task fits the description".to_owned(),
                _ => "read it when the task needs it".to_owned(),
            };
            let desc = if s.description.is_empty() { String::new() } else { format!(": {}", s.description) };
            format!("- {}{} ({when})", s.file, desc)
        })
        .collect();
    if rows.is_empty() {
        String::new()
    } else {
        format!("More standing instructions exist; they are not in this prompt.\n{}", rows.join("\n"))
    }
}

/// The text added to a tool result when `path` matches file-matching steering that was not shown yet. `seen` holds the names already shown.
pub fn notes_for_path(all: &[Steering], path: &str, seen: &mut std::collections::HashSet<String>) -> String {
    let mut out = String::new();
    for s in all.iter().filter(|s| s.applies_to(path)) {
        if seen.insert(s.name.clone()) {
            out.push_str(&format!("- steering {} ({}):\n{}\n", s.name, s.file, s.body));
        }
    }
    out.trim_end().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_matter_modes_and_defaults() {
        let s = parse("api", "x/api.md", "---\ninclusion: fileMatch\nfileMatchPattern: [\"**/*.ts\", 'src/api/**']\n---\nUse zod.\n");
        assert_eq!(s.inclusion, Inclusion::FileMatch);
        assert_eq!(s.patterns, vec!["**/*.ts", "src/api/**"]);
        assert_eq!(s.body, "Use zod.");
        let plain = parse("tech", "x/tech.md", "Rust only.\n");
        assert_eq!(plain.inclusion, Inclusion::Always);
        assert_eq!(plain.body, "Rust only.");
        let manual = parse("m", "m.md", "---\ninclusion: manual\ndescription: \"For releases\"\n---\nTag first.");
        assert_eq!((manual.inclusion, manual.description.as_str()), (Inclusion::Manual, "For releases"));
        // a fileMatch file without a pattern cannot apply: it becomes manual
        assert_eq!(parse("p", "p.md", "---\ninclusion: fileMatch\n---\nbody").inclusion, Inclusion::Manual);
        // front matter that is never closed is part of the body
        assert_eq!(parse("u", "u.md", "---\ninclusion: manual\nno end").inclusion, Inclusion::Always);
    }

    #[test]
    fn block_lists_and_single_patterns() {
        let s = parse("a", "a.md", "---\ninclusion: fileMatch\nfileMatchPattern:\n  - \"*.py\"\n  - tests/**\n---\nx");
        assert_eq!(s.patterns, vec!["*.py", "tests/**"]);
        let one = parse("b", "b.md", "---\ninclusion: fileMatch\nfileMatchPattern: components/**/*.tsx\n---\nx");
        assert_eq!(one.patterns, vec!["components/**/*.tsx"]);
    }

    #[test]
    fn globs() {
        assert!(glob_matches("**/*.tsx", "src/ui/Button.tsx"));
        assert!(glob_matches("**/*.tsx", "Button.tsx"));
        assert!(glob_matches("*.rs", "src/deep/lib.rs"));
        assert!(!glob_matches("src/*.rs", "src/deep/lib.rs"));
        assert!(glob_matches("src/**", "src/deep/lib.rs"));
        assert!(glob_matches("components/**/*.tsx", "components/a/b/C.tsx"));
        assert!(glob_matches("*.{ts,tsx}", "x/y.tsx"));
        assert!(!glob_matches("*.{ts,tsx}", "x/y.js"));
        assert!(glob_matches("src\\a\\b.rs", "src/a/b.rs") || !glob_matches("src/a/b.rs", "other/b.rs"));
        assert!(!glob_matches("", "a"));
    }

    #[test]
    fn load_orders_user_then_workspace_and_workspace_wins() {
        let base = std::env::temp_dir().join(format!("fllm-steer-{}-{}", std::process::id(), crate::agent_extras::fresh_session_name(1)));
        let user = base.join("user");
        let ws = base.join("ws");
        std::fs::create_dir_all(&user).unwrap();
        std::fs::create_dir_all(ws.join(".fllm").join("steering")).unwrap();
        std::fs::write(user.join("style.md"), "user style").unwrap();
        std::fs::write(user.join("mine.md"), "---\ninclusion: manual\n---\nonly mine").unwrap();
        std::fs::write(ws.join(".fllm").join("steering").join("style.md"), "project style").unwrap();
        std::fs::write(ws.join(".fllm").join("steering").join("empty.md"), "  ").unwrap();
        let all = load(&ws, Some(&user));
        let names: Vec<&str> = all.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["mine", "style"]);
        assert_eq!(all[1].body, "project style");
        assert_eq!(always_files(&all), vec![(".fllm/steering/style.md".to_owned(), "project style".to_owned())]);
        assert!(index_text(&all).contains("mine.md"));
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn file_match_notes_come_once_per_session() {
        let all = vec![parse("ts", ".fllm/steering/ts.md", "---\ninclusion: fileMatch\nfileMatchPattern: \"**/*.ts\"\n---\nStrict mode.")];
        let mut seen = std::collections::HashSet::new();
        assert!(notes_for_path(&all, "src/a.ts", &mut seen).contains("Strict mode."));
        assert_eq!(notes_for_path(&all, "src/b.ts", &mut seen), "");
        assert_eq!(notes_for_path(&all, "src/a.py", &mut std::collections::HashSet::new()), "");
    }
}
