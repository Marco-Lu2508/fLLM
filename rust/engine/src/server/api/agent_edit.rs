//! Text changes of the coding agent as pure functions: the exact-then-tolerant search and replace of
//! `edit_file` (with messages that help the model recover) and the unified-diff applier of `patch`.
//! Line endings are handled here: a file that uses CRLF throughout is edited as if it used LF and written
//! back with CRLF, so a model that never sees the carriage returns still matches.

/// A text split into the form the edits work on.
pub(super) struct Normalized {
    /// The text with LF line endings.
    pub text: String,
    /// The file used CRLF throughout and must get it back.
    pub crlf: bool,
}

pub(super) fn normalize_eol(raw: &str) -> Normalized {
    let crlf_count = raw.matches("\r\n").count();
    let lf_count = raw.matches('\n').count();
    if crlf_count > 0 && crlf_count == lf_count {
        Normalized { text: raw.replace("\r\n", "\n"), crlf: true }
    } else {
        Normalized { text: raw.to_owned(), crlf: false }
    }
}

pub(super) fn restore_eol(text: String, crlf: bool) -> String {
    if crlf {
        text.replace('\n', "\r\n")
    } else {
        text
    }
}

fn line_of(text: &str, offset: usize) -> usize {
    text[..offset].matches('\n').count() + 1
}

fn lead(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

fn clip(line: &str) -> String {
    let s: String = line.chars().take(160).collect();
    if line.chars().count() > 160 {
        format!("{s} ...")
    } else {
        s
    }
}

/// The result of one replacement.
#[derive(Debug)]
pub(super) struct Edited {
    pub text: String,
    pub count: usize,
    /// Set when the match needed tolerance (indentation / trailing spaces).
    pub note: &'static str,
}

/// Replace `old` by `new` in `text` (LF line endings). Exact matches first; when there is none, a match that
/// ignores leading and trailing white space of each line is accepted if it is unique (or `replace_all`).
pub(super) fn replace(text: &str, old: &str, new: &str, replace_all: bool, shown: &str) -> Result<Edited, String> {
    if old.is_empty() {
        return Err("old_string must not be empty; use write_file to create a file".into());
    }
    if old == new {
        return Err("old_string and new_string are the same; nothing would change".into());
    }
    let hits: Vec<usize> = text.match_indices(old).map(|(at, _)| at).collect();
    match hits.len() {
        0 => {}
        1 => return Ok(Edited { text: text.replacen(old, new, 1), count: 1, note: "" }),
        n if replace_all => return Ok(Edited { text: text.replace(old, new), count: n, note: "" }),
        n => {
            let lines: Vec<String> = hits.iter().take(8).map(|&at| line_of(text, at).to_string()).collect();
            return Err(format!(
                "old_string occurs {n} times in {shown} (at lines {}{}); include more surrounding lines to make it unique, or set replace_all",
                lines.join(", "),
                if n > 8 { ", ..." } else { "" }
            ));
        }
    }
    // tolerant: compare line by line without surrounding white space
    let lines: Vec<&str> = text.split('\n').collect();
    let want: Vec<&str> = old.trim_matches('\n').split('\n').collect();
    if want.iter().any(|l| !l.trim().is_empty()) && want.len() <= lines.len() {
        let windows: Vec<usize> = (0..=lines.len() - want.len())
            .filter(|&i| want.iter().enumerate().all(|(k, w)| lines[i + k].trim() == w.trim()))
            .collect();
        if windows.len() == 1 || (replace_all && !windows.is_empty()) {
            let new_lines: Vec<&str> = new.trim_matches('\n').split('\n').collect();
            let mut out: Vec<String> = lines.iter().map(|l| (*l).to_owned()).collect();
            for &i in windows.iter().rev() {
                let file_indent = lead(lines[i]);
                let want_indent = lead(want.iter().find(|l| !l.trim().is_empty()).copied().unwrap_or(""));
                let replacement: Vec<String> = new_lines.iter().map(|l| reindent(l, want_indent, file_indent)).collect();
                out.splice(i..i + want.len(), replacement);
            }
            return Ok(Edited { text: out.join("\n"), count: windows.len(), note: " (matched ignoring differences in indentation and trailing spaces)" });
        }
        if windows.len() > 1 {
            let at: Vec<String> = windows.iter().take(8).map(|i| (i + 1).to_string()).collect();
            return Err(format!(
                "old_string matches {} places in {shown} when white space is ignored (lines {}); include more surrounding lines, or set replace_all",
                windows.len(),
                at.join(", ")
            ));
        }
    }
    Err(not_found(&lines, &want, shown))
}

/// Moves a line from the indentation the model wrote to the one the file has.
fn reindent(line: &str, from: &str, to: &str) -> String {
    if line.trim().is_empty() || from == to {
        return line.to_owned();
    }
    if let Some(rest) = line.strip_prefix(from) {
        return format!("{to}{rest}");
    }
    line.to_owned()
}

/// The message for an old_string that was not found: where the closest text is and how it differs.
fn not_found(lines: &[&str], want: &[&str], shown: &str) -> String {
    let base = format!("old_string was not found in {shown}.");
    let wanted: Vec<&str> = want.iter().map(|l| l.trim()).collect();
    let useful = wanted.iter().filter(|l| l.len() > 2).count();
    if useful == 0 || lines.is_empty() {
        return format!("{base} Read the file again and copy the text exactly, including indentation.");
    }
    // best window: the most lines (trimmed) that are equal at the same offset
    let n = want.len().min(lines.len());
    let mut best = (0usize, 0usize);
    for i in 0..=lines.len() - n {
        let score = (0..n).filter(|&k| wanted[k].len() > 2 && lines[i + k].trim() == wanted[k]).count();
        if score > best.0 {
            best = (score, i);
        }
    }
    if best.0 == 0 {
        // no equal line: look for the line that contains the longest wanted line
        let key = wanted.iter().max_by_key(|l| l.len()).copied().unwrap_or("");
        if let Some(i) = lines.iter().position(|l| l.contains(key)) {
            return format!(
                "{base} The line `{}` exists at line {} but the lines around it differ. Read lines {}-{} and copy the text exactly.",
                clip(key),
                i + 1,
                (i + 1).saturating_sub(2).max(1),
                i + 1 + want.len() + 2
            );
        }
        return format!("{base} No similar text was found; the file may have changed. Read it again (read_file) before editing.");
    }
    let (score, start) = best;
    let mut msg = format!("{base} The closest text is at lines {}-{} ({score} of {} lines are the same):\n", start + 1, start + n, want.len());
    for k in 0..n.min(10) {
        msg.push_str(&format!("{:>6}\t{}\n", start + k + 1, clip(lines[start + k])));
    }
    if let Some(k) = (0..n).find(|&k| lines[start + k].trim() != wanted[k]) {
        msg.push_str(&format!("First difference at line {}: you wrote `{}`, the file has `{}`.\n", start + k + 1, clip(wanted[k]), clip(lines[start + k].trim())));
    }
    msg.push_str("Copy the text from the file exactly (read_file shows it), or use a smaller old_string.");
    msg
}

// ---- unified diffs ---------------------------------------------------------------------------------

/// One hunk: the lines it expects and the lines it leaves behind.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Hunk {
    /// 1-based line in the old file the hunk claims to start at (`usize::MAX` when the header has no numbers).
    pub hint: usize,
    pub old: Vec<String>,
    pub new: Vec<String>,
}

/// The changes to one file.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct FilePatch {
    pub path: String,
    pub create: bool,
    pub delete: bool,
    pub hunks: Vec<Hunk>,
}

fn clean_path(raw: &str) -> Option<String> {
    let p = raw.split('\t').next().unwrap_or(raw).trim().trim_matches('"');
    if p == "/dev/null" {
        return None;
    }
    let p = p.strip_prefix("a/").or_else(|| p.strip_prefix("b/")).unwrap_or(p);
    Some(p.to_owned())
}

fn hunk_start(header: &str) -> usize {
    // @@ -12,5 +12,6 @@
    header
        .trim_start_matches('@')
        .trim()
        .strip_prefix('-')
        .and_then(|rest| rest.split([',', ' ']).next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(usize::MAX)
}

/// Reads a unified diff (git or plain `diff -u` style, several files allowed).
pub(super) fn parse_patch(patch: &str) -> Result<Vec<FilePatch>, String> {
    let text = patch.replace("\r\n", "\n");
    let mut files: Vec<FilePatch> = Vec::new();
    let mut current: Option<Hunk> = None;
    let finish = |files: &mut Vec<FilePatch>, hunk: &mut Option<Hunk>| {
        if let (Some(h), Some(f)) = (hunk.take(), files.last_mut()) {
            f.hunks.push(h);
        }
    };
    let all: Vec<&str> = text.lines().collect();
    let mut index = 0;
    while index < all.len() {
        let line = all[index];
        index += 1;
        // a file header is "--- old" directly followed by "+++ new" (a removed line "-- x" is never followed by one)
        if let (Some(old), Some(new)) = (line.strip_prefix("--- "), all.get(index).and_then(|n| n.strip_prefix("+++ "))) {
            index += 1;
            finish(&mut files, &mut current);
            let (old, new) = (clean_path(old), clean_path(new));
            let (path, create, delete) = match (old, new) {
                (None, Some(n)) => (n, true, false),
                (Some(o), None) => (o, false, true),
                (Some(o), Some(n)) => {
                    if o != n {
                        return Err(format!("renaming ({o} to {n}) is not supported by patch; use bash with git mv"));
                    }
                    (n, false, false)
                }
                (None, None) => return Err("a patch cannot have /dev/null on both sides".into()),
            };
            files.push(FilePatch { path, create, delete, hunks: Vec::new() });
            continue;
        }
        if line.starts_with("@@") {
            finish(&mut files, &mut current);
            if files.is_empty() {
                return Err("the patch has a hunk before any '--- path' / '+++ path' header".into());
            }
            current = Some(Hunk { hint: hunk_start(line), old: Vec::new(), new: Vec::new() });
            continue;
        }
        let Some(h) = current.as_mut() else { continue };
        match line.chars().next() {
            Some('+') => h.new.push(line[1..].to_owned()),
            Some('-') => h.old.push(line[1..].to_owned()),
            Some(' ') => {
                h.old.push(line[1..].to_owned());
                h.new.push(line[1..].to_owned());
            }
            None => {
                // an empty line inside a hunk is an empty context line that lost its space
                h.old.push(String::new());
                h.new.push(String::new());
            }
            Some('\\') => {}
            Some(_) => {
                if line.starts_with("diff ") || line.starts_with("index ") {
                    finish(&mut files, &mut current);
                } else {
                    return Err(format!("unexpected line in a hunk (every line must start with ' ', '+' or '-'): {}", clip(line)));
                }
            }
        }
    }
    finish(&mut files, &mut current);
    if files.is_empty() {
        return Err("no file changes found: a patch needs '--- path', '+++ path' and '@@' hunks (unified diff)".into());
    }
    for f in &files {
        if f.hunks.is_empty() && !f.delete {
            return Err(format!("the patch for {} has no hunks", f.path));
        }
    }
    Ok(files)
}

/// A way of comparing lines: the note for the model and how a line is normalised before it is compared.
type Tier = (&'static str, fn(&str) -> &str);

/// Where `old` starts in `lines`, searching outwards from `hint` (0-based). Exact, then without trailing
/// space, then without any surrounding white space.
fn locate(lines: &[String], old: &[String], hint: usize, from: usize) -> Option<(usize, &'static str)> {
    if old.is_empty() {
        return Some((hint.min(lines.len()), ""));
    }
    if old.len() > lines.len() {
        return None;
    }
    let tiers: [Tier; 3] = [("", |s| s), (" (matched ignoring trailing spaces)", str::trim_end), (" (matched ignoring indentation)", str::trim)];
    for (note, norm) in tiers {
        let hits: Vec<usize> = (0..=lines.len() - old.len())
            .filter(|&i| old.iter().enumerate().all(|(k, o)| norm(&lines[i + k]) == norm(o)))
            .collect();
        // prefer the hit at or after the previous hunk, nearest to the hint
        let best = hits.iter().copied().filter(|&i| i >= from).min_by_key(|&i| i.abs_diff(hint)).or_else(|| hits.iter().copied().min_by_key(|&i| i.abs_diff(hint)));
        if let Some(i) = best {
            return Some((i, note));
        }
    }
    None
}

/// Applies the hunks of one file to LF text. Returns the new text and a note about tolerance used.
pub(super) fn apply_hunks(text: &str, hunks: &[Hunk], shown: &str) -> Result<(String, &'static str), String> {
    let had_final_newline = text.ends_with('\n');
    let mut lines: Vec<String> = if text.is_empty() { Vec::new() } else { text.strip_suffix('\n').unwrap_or(text).split('\n').map(str::to_owned).collect() };
    let mut note = "";
    let mut from = 0usize;
    for (index, hunk) in hunks.iter().enumerate() {
        // a hunk that only adds lines names the line after which they go; one with context names its first line
        let hint = if hunk.old.is_empty() { hunk.hint } else { hunk.hint.saturating_sub(1) };
        let (at, n) = locate(&lines, &hunk.old, hint, from).ok_or_else(|| {
            let first = hunk.old.iter().find(|l| !l.trim().is_empty()).map(|l| clip(l.trim())).unwrap_or_default();
            format!(
                "hunk {} of {shown} does not apply: the lines it expects were not found (first expected line: `{first}`). Read the file again and make the hunk from its current text. Nothing was changed.",
                index + 1
            )
        })?;
        if !n.is_empty() {
            note = n;
        }
        lines.splice(at..at + hunk.old.len(), hunk.new.iter().cloned());
        from = at + hunk.new.len();
    }
    let mut out = lines.join("\n");
    if had_final_newline || text.is_empty() {
        out.push('\n');
    }
    Ok((out, note))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_unique_and_ambiguous_matches() {
        let text = "a\nb\nc\nb\n";
        assert_eq!(replace(text, "a", "x", false, "f").unwrap().text, "x\nb\nc\nb\n");
        let err = replace(text, "b", "x", false, "f").unwrap_err();
        assert!(err.contains("2 times") && err.contains("lines 2, 4"), "{err}");
        assert_eq!(replace(text, "b", "x", true, "f").unwrap().count, 2);
        assert!(replace(text, "", "x", false, "f").is_err());
        assert!(replace(text, "a", "a", false, "f").is_err());
    }

    #[test]
    fn indentation_differences_are_tolerated_and_reindented() {
        let text = "fn main() {\n    let a = 1;\n    let b = 2;\n}\n";
        let e = replace(text, "let a = 1;\nlet b = 2;", "let a = 10;\nlet b = 20;", false, "f").unwrap();
        assert_eq!(e.text, "fn main() {\n    let a = 10;\n    let b = 20;\n}\n");
        assert!(!e.note.is_empty());
    }

    #[test]
    fn a_missing_text_names_the_closest_lines() {
        let text = "one\ntwo\nthree\nfour\n";
        let err = replace(text, "two\nTHREE\nfour", "x", false, "f.txt").unwrap_err();
        assert!(err.contains("closest text is at lines 2-4") && err.contains("First difference at line 3"), "{err}");
        let err = replace(text, "completely different content", "x", false, "f.txt").unwrap_err();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn crlf_files_round_trip() {
        let n = normalize_eol("a\r\nb\r\n");
        assert!(n.crlf && n.text == "a\nb\n");
        assert_eq!(restore_eol("a\nX\n".into(), n.crlf), "a\r\nX\r\n");
        let mixed = normalize_eol("a\r\nb\n");
        assert!(!mixed.crlf);
    }

    #[test]
    fn unified_diffs_are_parsed_and_applied() {
        let patch = "--- a/x.txt\n+++ b/x.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n";
        let files = parse_patch(patch).unwrap();
        assert_eq!(files.len(), 1);
        let (out, _) = apply_hunks("one\ntwo\nthree\n", &files[0].hunks, "x.txt").unwrap();
        assert_eq!(out, "one\nTWO\nthree\n");
        // a wrong line number is fine when the context is found elsewhere
        let patch = "--- x\n+++ x\n@@ -40,2 +40,2 @@\n two\n-three\n+3\n";
        let files = parse_patch(patch).unwrap();
        assert_eq!(apply_hunks("one\ntwo\nthree\n", &files[0].hunks, "x").unwrap().0, "one\ntwo\n3\n");
        // missing context is refused
        let patch = "--- x\n+++ x\n@@ -1 +1 @@\n-nothing\n+x\n";
        let files = parse_patch(patch).unwrap();
        assert!(apply_hunks("one\n", &files[0].hunks, "x").unwrap_err().contains("does not apply"));
    }

    #[test]
    fn new_and_deleted_files_and_two_hunks() {
        let files = parse_patch("--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,2 @@\n+a\n+b\n--- a/old.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-gone\n").unwrap();
        assert!(files[0].create && files[1].delete);
        assert_eq!(apply_hunks("", &files[0].hunks, "new.txt").unwrap().0, "a\nb\n");
        let two = parse_patch("--- x\n+++ x\n@@ -1 +1 @@\n-a\n+A\n@@ -3 +3 @@\n-c\n+C\n").unwrap();
        assert_eq!(apply_hunks("a\nb\nc\n", &two[0].hunks, "x").unwrap().0, "A\nb\nC\n");
        assert!(parse_patch("nothing useful").is_err());
    }
}
