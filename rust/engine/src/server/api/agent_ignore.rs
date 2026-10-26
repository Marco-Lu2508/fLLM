//! Ignore rules (`.gitignore`, `.ignore`) for the file walks of the coding agent, and the fuzzy ranking of
//! file names behind `@` completion.
use regex::Regex;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

/// One line of an ignore file.
struct Rule {
    re: Regex,
    negate: bool,
    dir_only: bool,
}

/// The rules of one ignore file; paths are matched relative to the folder the file lives in.
pub(super) struct Scope {
    base: PathBuf,
    rules: Vec<Rule>,
}

fn translate(pattern: &str) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '*' if chars.get(i + 1) == Some(&'*') => {
                i += 1;
                if chars.get(i + 1) == Some(&'/') {
                    i += 1;
                    out.push_str("(?:.*/)?");
                } else {
                    out.push_str(".*");
                }
            }
            '*' => out.push_str("[^/]*"),
            '?' => out.push_str("[^/]"),
            '\\' if i + 1 < chars.len() => {
                i += 1;
                out.push_str(&regex::escape(&chars[i].to_string()));
            }
            c => out.push_str(&regex::escape(&c.to_string())),
        }
        i += 1;
    }
    out
}

fn parse_rule(line: &str) -> Option<Rule> {
    let mut line = line.trim_end_matches(['\r', ' ']);
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let negate = line.starts_with('!');
    if negate {
        line = &line[1..];
    }
    let dir_only = line.ends_with('/');
    let line = line.trim_end_matches('/');
    if line.is_empty() {
        return None;
    }
    let anchored = line.starts_with('/') || line.contains('/');
    let body = translate(line.trim_start_matches('/'));
    let source = if anchored { format!("^{body}$") } else { format!("^(?:.*/)?{body}$") };
    Regex::new(&source).ok().map(|re| Rule { re, negate, dir_only })
}

impl Scope {
    /// The ignore files of `dir`, or `None` when it has none.
    pub(super) fn load(dir: &Path) -> Option<Arc<Scope>> {
        let mut rules = Vec::new();
        for name in [".gitignore", ".ignore"] {
            if let Ok(text) = fs::read_to_string(dir.join(name)) {
                rules.extend(text.lines().filter_map(parse_rule));
            }
        }
        (!rules.is_empty()).then(|| Arc::new(Scope { base: dir.to_owned(), rules }))
    }
}

/// The chain of ignore files that apply below a folder (outermost first).
#[derive(Clone, Default)]
pub(super) struct Chain(Vec<Arc<Scope>>);

impl Chain {
    /// Every ignore file from `root` down to `dir` (inclusive).
    pub(super) fn for_dir(root: &Path, dir: &Path) -> Chain {
        let mut dirs = vec![dir.to_owned()];
        let mut at = dir;
        while at != root {
            match at.parent() {
                Some(p) if p.starts_with(root) => {
                    dirs.push(p.to_owned());
                    at = p;
                }
                _ => break,
            }
        }
        dirs.reverse();
        Chain(dirs.iter().filter_map(|d| Scope::load(d)).collect())
    }

    /// This chain plus the ignore files of `dir` (a child folder).
    pub(super) fn entering(&self, dir: &Path) -> Chain {
        let mut next = self.clone();
        if let Some(scope) = Scope::load(dir) {
            next.0.push(scope);
        }
        next
    }

    /// Whether the file or folder `path` is ignored. The last matching rule wins, deeper files over outer ones.
    pub(super) fn ignores(&self, path: &Path, is_dir: bool) -> bool {
        let mut ignored = false;
        for scope in &self.0 {
            let Ok(rel) = path.strip_prefix(&scope.base) else { continue };
            let rel = rel.to_string_lossy().replace('\\', "/");
            for rule in &scope.rules {
                if (!rule.dir_only || is_dir) && rule.re.is_match(&rel) {
                    ignored = !rule.negate;
                }
            }
        }
        ignored
    }
}

/// A score for how well `query` matches `path` as a fuzzy subsequence, or `None` when it does not.
/// Matches in the file name, at word starts and in a row score higher; shorter paths win ties.
pub(super) fn fuzzy_score(query: &str, path: &str) -> Option<i32> {
    let q: Vec<char> = query.to_lowercase().chars().filter(|c| !c.is_whitespace()).collect();
    if q.is_empty() {
        return Some(-(path.len() as i32));
    }
    let p: Vec<char> = path.to_lowercase().chars().collect();
    let name_start = path.rfind('/').map_or(0, |i| path[..i + 1].chars().count());
    let mut score = 0i32;
    let mut at = 0usize;
    let mut previous: Option<usize> = None;
    for &c in &q {
        let found = (at..p.len()).find(|&i| p[i] == c)?;
        score += 10;
        if previous == Some(found.wrapping_sub(1)) {
            score += 15;
        }
        if found == 0 || matches!(p[found - 1], '/' | '_' | '-' | '.' | ' ') {
            score += 12;
        }
        if found >= name_start {
            score += 6;
        }
        previous = Some(found);
        at = found + 1;
    }
    let joined: String = q.iter().collect();
    let name: String = p[name_start..].iter().collect();
    if name == joined {
        score += 200;
    } else if name.starts_with(&joined) {
        score += 80;
    } else if name.contains(&joined) {
        score += 50;
    } else if path.to_lowercase().contains(&joined) {
        score += 20;
    }
    Some(score * 4 - path.len() as i32)
}

/// The best `limit` of `paths` for `query`, best first.
pub(super) fn rank(query: &str, paths: &[String], limit: usize) -> Vec<String> {
    let mut scored: Vec<(i32, &String)> = paths.iter().filter_map(|p| fuzzy_score(query, p).map(|s| (s, p))).collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().take(limit).map(|(_, p)| p.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(text: &str) -> Chain {
        let rules: Vec<Rule> = text.lines().filter_map(parse_rule).collect();
        Chain(vec![Arc::new(Scope { base: PathBuf::from("/w"), rules })])
    }

    #[test]
    fn gitignore_patterns() {
        let c = chain("target/\n*.log\n/build\n!keep.log\ndocs/*.tmp\n# comment\nsrc/**/gen\n");
        assert!(c.ignores(Path::new("/w/target"), true));
        assert!(!c.ignores(Path::new("/w/target"), false), "a folder rule does not match a file");
        assert!(c.ignores(Path::new("/w/a/b/x.log"), false));
        assert!(!c.ignores(Path::new("/w/keep.log"), false));
        assert!(c.ignores(Path::new("/w/build"), true));
        assert!(!c.ignores(Path::new("/w/sub/build"), true), "anchored");
        assert!(c.ignores(Path::new("/w/docs/a.tmp"), false));
        assert!(!c.ignores(Path::new("/w/docs/x/a.tmp"), false));
        assert!(c.ignores(Path::new("/w/src/a/b/gen"), true));
        assert!(!c.ignores(Path::new("/w/src/main.rs"), false));
    }

    #[test]
    fn fuzzy_ranking_prefers_names_and_word_starts() {
        let paths: Vec<String> = ["src/server/api/agent.rs", "src/agent_notes.txt", "docs/user/studio.md", "src/webui/code.js", "README.md"].iter().map(|s| (*s).to_owned()).collect();
        assert_eq!(rank("agent.rs", &paths, 3)[0], "src/server/api/agent.rs");
        assert_eq!(rank("code", &paths, 1)[0], "src/webui/code.js");
        assert_eq!(rank("stdmd", &paths, 1)[0], "docs/user/studio.md");
        assert!(fuzzy_score("zzz", "src/a.rs").is_none());
        assert_eq!(rank("", &paths, 2).len(), 2);
    }
}
