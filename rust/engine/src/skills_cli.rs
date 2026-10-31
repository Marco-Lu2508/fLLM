//! `fllm-rs skills`: list, approve and exchange skills as files. A skill is a folder with a `SKILL.md`, the format of the
//! agentskills.io standard and of Claude Code, so the same folder can be exchanged with those tools in both directions.
//!
//! * `fllm-rs skills list` shows your skills (the Studio folder and, with `--folder`, the project).
//! * `fllm-rs skills proposed` lists skills the agent proposed (`fllm-rs agent --propose-skills`); `approve NAME` moves one into
//!   your skills, `reject NAME` deletes it.
//! * `fllm-rs skills export NAME --to claude|codex|hermes|agentskills|DIR` writes a copy in the other tool's layout.
//! * `fllm-rs skills import PATH` / `install OWNER/REPO[/PATH]` reads every `SKILL.md` below a folder (or a git repository).
use std::path::{Path, PathBuf};

pub const HELP: &str = "usage: fllm-rs skills list [--folder DIR]
       fllm-rs skills proposed
       fllm-rs skills approve NAME | reject NAME
       fllm-rs skills export NAME --to claude|codex|hermes|agentskills|DIR [--folder DIR] [--out DIR]
       fllm-rs skills import PATH [--folder DIR] [--dry-run] [--force]
       fllm-rs skills install OWNER/REPO[/SUBFOLDER] | GIT-URL [--folder DIR] [--dry-run] [--force]
Skills are folders with a SKILL.md (agentskills.io / Claude Code format). Skills of the Studio folder are yours in every project,
--folder DIR uses DIR/.fllm/skills instead. import and install copy; a skill that already exists is kept unless --force is given.";

const MAX_SKILL_FILES_BYTES: u64 = 2 << 20;

/// `name` as an agentskills.io name: lower case letters, digits and single hyphens, 1 to 64 characters, no hyphen at either end.
pub fn standard_name(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().to_lowercase().chars() {
        let c = if c.is_ascii_alphanumeric() { c } else { '-' };
        if c == '-' && (out.is_empty() || out.ends_with('-')) {
            continue;
        }
        out.push(c);
    }
    let out = out.trim_end_matches('-');
    out.chars().take(64).collect::<String>().trim_end_matches('-').to_owned()
}

/// A name fLLM accepts (`a-z0-9`, `-`, `_`).
pub fn local_name(name: &str) -> String {
    let n: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' { c } else { '-' })
        .collect();
    n.trim_matches('-').chars().take(64).collect()
}

/// The front matter lines and the body of a SKILL.md (`None` when there is no front matter block).
fn split(text: &str) -> Option<(Vec<String>, String)> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text).replace("\r\n", "\n");
    let rest = text.strip_prefix("---\n")?;
    let end = rest.find("\n---")?;
    let head: Vec<String> = rest[..end].lines().map(str::to_owned).collect();
    let after = &rest[end + 4..];
    Some((head, after.strip_prefix('\n').unwrap_or(after).to_owned()))
}

/// The value of a one-line `key: value` or a YAML block (`key: >` / `|` followed by indented lines).
fn head_value(head: &[String], key: &str) -> String {
    let prefix = format!("{key}:");
    let Some(at) = head.iter().position(|l| l.starts_with(&prefix)) else { return String::new() };
    let first = head[at][prefix.len()..].trim();
    if matches!(first, ">" | "|" | ">-" | "|-" | ">+" | "|+") {
        return head[at + 1..].iter().take_while(|l| l.starts_with(' ') || l.starts_with('\t') || l.is_empty()).map(|l| l.trim()).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" ");
    }
    first.trim_matches(['"', '\'']).to_owned()
}

/// `(name, description, when)` of a skill file.
pub fn read_meta(text: &str) -> (String, String, String) {
    split(text).map_or_else(Default::default, |(head, _)| (head_value(&head, "name"), head_value(&head, "description"), head_value(&head, "when")))
}

/// Rewrites a SKILL.md for another tool. `standard` true: agentskills.io rules (hyphen name, `when` folded into the description,
/// description at most 1024 characters). `standard` false: the names fLLM accepts, text otherwise unchanged.
pub fn rewrite(text: &str, fallback_name: &str, standard: bool) -> (String, String) {
    let (name, description, when) = read_meta(text);
    let body = split(text).map_or_else(|| text.trim().to_owned(), |(_, b)| b.trim_start_matches('\n').to_owned());
    let name = if name.is_empty() { fallback_name.to_owned() } else { name };
    // the other front matter keys of the standard travel with the skill (fLLM ignores them, the other tools use them)
    let mut extra = String::new();
    if let Some((head, _)) = split(text) {
        for key in ["license", "compatibility", "allowed-tools"] {
            let v = head_value(&head, key);
            if !v.is_empty() {
                extra.push_str(&format!("{key}: {}\n", one_line(&v)));
            }
        }
    }
    if standard {
        let mut description = description;
        if !when.is_empty() {
            description = format!("{}. Use when: {}", description.trim_end_matches('.'), when.trim_end_matches('.'));
        }
        if description.trim().is_empty() {
            description = format!("Skill {name}.");
        }
        let description: String = description.chars().take(1024).collect();
        let slug = standard_name(&name);
        (slug.clone(), format!("---\nname: {slug}\ndescription: {}\n{extra}---\n{}\n", one_line(&description), body.trim_end()))
    } else {
        let slug = local_name(&name);
        let mut file = format!("---\nname: {slug}\ndescription: {}\n", one_line(if description.is_empty() { &slug } else { &description }));
        if !when.is_empty() {
            file.push_str(&format!("when: {}\n", one_line(&when)));
        }
        file.push_str(&extra);
        file.push_str(&format!("---\n{}\n", body.trim_end()));
        (slug, file)
    }
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every folder below `root` (at most 6 levels) that holds a `SKILL.md`.
pub fn find_skills(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        if dir.join("SKILL.md").is_file() {
            out.push(dir.to_path_buf());
            return;
        }
        if depth == 0 {
            return;
        }
        let Ok(read) = std::fs::read_dir(dir) else { return };
        let mut entries: Vec<_> = read.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let name = e.file_name().to_string_lossy().into_owned();
            let is_dir = e.file_type().is_ok_and(|t| t.is_dir() && !t.is_symlink());
            if is_dir && !name.starts_with('.') && name != "node_modules" {
                walk(&e.path(), depth - 1, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, 6, &mut out);
    out
}

fn copy_extras(from: &Path, to: &Path) -> Result<(), String> {
    let mut total = 0u64;
    fn walk(from: &Path, to: &Path, base: &Path, total: &mut u64) -> Result<(), String> {
        for e in std::fs::read_dir(from).map_err(|e| e.to_string())?.flatten() {
            let path = e.path();
            let Ok(kind) = e.file_type() else { continue };
            if kind.is_symlink() {
                continue; // links could point anywhere
            }
            let rel = path.strip_prefix(base).map_err(|e| e.to_string())?;
            if kind.is_dir() {
                walk(&path, to, base, total)?;
            } else if rel.as_os_str() != "SKILL.md" {
                *total += e.metadata().map_or(0, |m| m.len());
                if *total > MAX_SKILL_FILES_BYTES {
                    return Err("the skill folder is larger than 2 MB".into());
                }
                let target = to.join(rel);
                if let Some(dir) = target.parent() {
                    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                }
                std::fs::copy(&path, &target).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
    walk(from, to, from, &mut total)
}

/// Where your skills live: `<folder>/.fllm/skills` for a project, else `skills` in the Studio folder.
pub fn skills_root(studio: &Path, folder: Option<&Path>) -> PathBuf {
    folder.map_or_else(|| studio.join("skills"), |f| f.join(".fllm").join("skills"))
}

/// Copies the skills found below `from` into `dest_root` (names made valid for fLLM). Returns one line per skill.
pub fn import_skills(from: &Path, dest_root: &Path, dry_run: bool, force: bool) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    // a tool that has no skills folder simply has nothing to bring over
    if !from.is_dir() {
        return Ok(lines);
    }
    for dir in find_skills(from) {
        let text = std::fs::read_to_string(dir.join("SKILL.md")).map_err(|e| format!("{}: {e}", dir.display()))?;
        if text.len() > 64 * 1024 {
            lines.push(format!("skipped {}: SKILL.md is larger than 64 KB", dir.display()));
            continue;
        }
        let fallback = dir.file_name().and_then(|n| n.to_str()).unwrap_or("skill");
        let (name, file) = rewrite(&text, fallback, false);
        if !crate::agent_cli::valid_session_name(&name) {
            lines.push(format!("skipped {}: '{name}' is not a usable name", dir.display()));
            continue;
        }
        let target = dest_root.join(&name);
        if target.exists() && !force {
            lines.push(format!("kept   {name}: already exists (use --force to replace)"));
            continue;
        }
        if !dry_run {
            std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
            std::fs::write(target.join("SKILL.md"), file).map_err(|e| e.to_string())?;
            copy_extras(&dir, &target)?;
        }
        lines.push(format!("{} {name}  <- {}", if dry_run { "would import" } else { "imported   " }, dir.display()));
    }
    if lines.is_empty() {
        lines.push(format!("no SKILL.md found below {}", from.display()));
    }
    Ok(lines)
}

fn home() -> Result<PathBuf, String> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).ok_or_else(|| "cannot find the home folder".to_owned())
}

/// Exports one skill into `dest_root/<name>` in the layout of another tool.
pub fn export_skill(src_dir: &Path, dest_root: &Path, standard: bool) -> Result<PathBuf, String> {
    let text = std::fs::read_to_string(src_dir.join("SKILL.md")).map_err(|e| format!("{}: {e}", src_dir.display()))?;
    let (name, file) = rewrite(&text, src_dir.file_name().and_then(|n| n.to_str()).unwrap_or("skill"), standard);
    if name.is_empty() {
        return Err("the skill has no usable name".into());
    }
    let target = dest_root.join(&name);
    std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
    std::fs::write(target.join("SKILL.md"), file).map_err(|e| e.to_string())?;
    copy_extras(src_dir, &target)?;
    Ok(target)
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

pub fn run(args: &[String]) -> Result<i32, String> {
    let studio = crate::agent_cli::studio_folder();
    let folder = flag(args, "--folder").map(PathBuf::from);
    let root = skills_root(&studio, folder.as_deref());
    let dry = args.iter().any(|a| a == "--dry-run");
    let force = args.iter().any(|a| a == "--force");
    match args.first().map(String::as_str) {
        Some("list") => {
            let found = find_skills(&root);
            if found.is_empty() {
                println!("no skills in {}", root.display());
            }
            for dir in found {
                let (name, description, _) = read_meta(&std::fs::read_to_string(dir.join("SKILL.md")).unwrap_or_default());
                println!("{:<28} {}", if name.is_empty() { dir.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_owned() } else { name }, description);
            }
            Ok(0)
        }
        Some("proposed") => {
            let dir = crate::agent_extras::proposals_dir(&studio);
            let found = find_skills(&dir);
            if found.is_empty() {
                println!("no proposed skills ({} is empty)", dir.display());
            }
            for d in found {
                let (_, description, _) = read_meta(&std::fs::read_to_string(d.join("SKILL.md")).unwrap_or_default());
                println!("{:<28} {}", d.file_name().and_then(|n| n.to_str()).unwrap_or("?"), description);
            }
            Ok(0)
        }
        Some(verb @ ("approve" | "reject")) => {
            let name = args.get(1).filter(|n| crate::agent_cli::valid_session_name(n)).ok_or("give the name of a proposed skill")?;
            let from = crate::agent_extras::proposals_dir(&studio).join(name);
            if !from.join("SKILL.md").is_file() {
                return Err(format!("no proposed skill '{name}'"));
            }
            if verb == "reject" {
                std::fs::remove_dir_all(&from).map_err(|e| e.to_string())?;
                println!("deleted proposal {name}");
                return Ok(0);
            }
            let target = root.join(name);
            if target.exists() && !force {
                return Err(format!("a skill named '{name}' already exists in {} (use --force to replace it)", root.display()));
            }
            std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
            if target.exists() {
                std::fs::remove_dir_all(&target).map_err(|e| e.to_string())?;
            }
            std::fs::rename(&from, &target).map_err(|e| e.to_string()).or_else(|_| {
                std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
                copy_extras(&from, &target)?;
                std::fs::copy(from.join("SKILL.md"), target.join("SKILL.md")).map(|_| ()).map_err(|e| e.to_string())
            })?;
            println!("approved {name} -> {}", target.display());
            Ok(0)
        }
        Some("export") => {
            let name = args.get(1).filter(|n| !n.starts_with("--")).ok_or("give the name of the skill")?;
            let to = flag(args, "--to").ok_or("--to claude|codex|hermes|agentskills|DIR is missing")?;
            let src = root.join(name);
            if !src.join("SKILL.md").is_file() {
                return Err(format!("no skill '{name}' in {}", root.display()));
            }
            let (dest, standard) = match to.as_str() {
                "claude" => (home()?.join(".claude").join("skills"), true),
                "codex" => (home()?.join(".codex").join("skills"), true),
                "hermes" => (home()?.join(".hermes").join("skills"), true),
                "agentskills" => (flag(args, "--out").map_or_else(|| PathBuf::from("."), PathBuf::from), true),
                other => (PathBuf::from(other), true),
            };
            let target = export_skill(&src, &dest, standard)?;
            println!("exported {name} -> {}", target.display());
            Ok(0)
        }
        Some("import") => {
            let from = args.get(1).filter(|a| !a.starts_with("--")).ok_or("give a folder with skills")?;
            if !Path::new(from).is_dir() {
                return Err(format!("{from} is not a folder"));
            }
            for line in import_skills(Path::new(from), &root, dry, force)? {
                println!("{line}");
            }
            Ok(0)
        }
        Some("install") => {
            let spec = args.get(1).filter(|a| !a.starts_with("--")).ok_or("give OWNER/REPO[/SUBFOLDER] or a git URL")?;
            let (url, sub) = repo_spec(spec)?;
            let tmp = std::env::temp_dir().join(format!("fllm-skills-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&tmp);
            let status = std::process::Command::new("git")
                .args(["clone", "--depth", "1", "--quiet", &url])
                .arg(&tmp)
                .env("GIT_TERMINAL_PROMPT", "0")
                .status()
                .map_err(|e| format!("cannot run git: {e}"))?;
            if !status.success() {
                return Err(format!("git clone of {url} failed"));
            }
            let scan = if sub.is_empty() { tmp.clone() } else { tmp.join(&sub) };
            if !scan.starts_with(&tmp) || sub.contains("..") {
                return Err("the subfolder must stay inside the repository".into());
            }
            let result = import_skills(&scan, &root, dry, force);
            let _ = std::fs::remove_dir_all(&tmp);
            for line in result? {
                println!("{line}");
            }
            Ok(0)
        }
        _ => {
            println!("{HELP}");
            Ok(if args.is_empty() || args[0] == "--help" || args[0] == "-h" { 0 } else { 1 })
        }
    }
}

/// `owner/repo[/sub/folder]` or a full git URL, as `(clone url, subfolder)`.
pub fn repo_spec(spec: &str) -> Result<(String, String), String> {
    if spec.contains("://") || spec.starts_with("git@") {
        if !(spec.starts_with("https://") || spec.starts_with("http://") || spec.starts_with("ssh://") || spec.starts_with("git@")) {
            return Err("only https, http and ssh git addresses are accepted".into());
        }
        return Ok((spec.to_owned(), String::new()));
    }
    let parts: Vec<&str> = spec.split('/').collect();
    let ok = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if parts.len() < 2 || !ok(parts[0]) || !ok(parts[1]) {
        return Err("expected OWNER/REPO[/SUBFOLDER]".into());
    }
    Ok((format!("https://github.com/{}/{}.git", parts[0], parts[1]), parts[2..].join("/")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCAL: &str = "---\nname: run_tests\ndescription: Run the tests\nwhen: after a change\n---\n1. `cargo test`\n";

    #[test]
    fn names_are_made_valid_for_each_side() {
        assert_eq!(standard_name("Run_Tests v2"), "run-tests-v2");
        assert_eq!(standard_name("--a--b--"), "a-b");
        assert_eq!(local_name("Deploy Staging!"), "deploy-staging");
        assert_eq!(local_name("run_tests"), "run_tests");
    }

    #[test]
    fn exporting_follows_the_agentskills_rules() {
        let (name, file) = rewrite(LOCAL, "x", true);
        assert_eq!(name, "run-tests");
        assert!(file.starts_with("---\nname: run-tests\ndescription: Run the tests. Use when: after a change\n---\n"), "{file}");
        assert!(!file.contains("\nwhen:"));
        let long = format!("---\nname: a\ndescription: {}\n---\nbody\n", "word ".repeat(400));
        assert!(read_meta(&rewrite(&long, "a", true).1).1.chars().count() <= 1024);
    }

    #[test]
    fn importing_a_claude_style_skill_keeps_its_meaning() {
        let claude = "---\nname: PDF Processing\ndescription: >\n  Extract text\n  from PDFs\nlicense: MIT\nallowed-tools: Read, Bash\n---\n# Steps\nUse the script.\n";
        let (name, file) = rewrite(claude, "pdf", false);
        assert_eq!(name, "pdf-processing");
        let (n, d, _) = read_meta(&file);
        assert_eq!((n.as_str(), d.as_str()), ("pdf-processing", "Extract text from PDFs"));
        assert!(file.ends_with("# Steps\nUse the script.\n"));
        assert!(file.contains("license: MIT\n") && file.contains("allowed-tools: Read, Bash\n"), "an import keeps the keys other tools use: {file}");
        // no front matter at all: the folder name is used
        let (name, file) = rewrite("Just steps\n", "from-folder", false);
        assert_eq!(name, "from-folder");
        assert!(file.contains("description: from-folder") && file.contains("Just steps"));
        // the standard export keeps license and allowed-tools
        let standard = rewrite(claude, "pdf", true).1;
        assert!(standard.contains("license: MIT") && standard.contains("allowed-tools: Read, Bash"));
    }

    #[test]
    fn import_export_round_trip_with_extra_files_and_no_overwrite() {
        let dir = std::env::temp_dir().join(format!("fllm-skills-cli-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = dir.join("claude/skills/Deploy It");
        std::fs::create_dir_all(src.join("scripts")).unwrap();
        std::fs::write(src.join("SKILL.md"), "---\nname: deploy-it\ndescription: Deploy\n---\nrun scripts/go.sh\n").unwrap();
        std::fs::write(src.join("scripts/go.sh"), "echo go").unwrap();
        let dest = dir.join("studio/skills");
        let dry = import_skills(&dir.join("claude"), &dest, true, false).unwrap();
        assert!(dry[0].starts_with("would import") && !dest.exists());
        let done = import_skills(&dir.join("claude"), &dest, false, false).unwrap();
        assert!(done[0].starts_with("imported"), "{done:?}");
        assert!(dest.join("deploy-it/SKILL.md").is_file() && dest.join("deploy-it/scripts/go.sh").is_file());
        assert!(import_skills(&dir.join("claude"), &dest, false, false).unwrap()[0].starts_with("kept"));
        let out = export_skill(&dest.join("deploy-it"), &dir.join("out"), true).unwrap();
        assert!(out.join("SKILL.md").is_file() && out.join("scripts/go.sh").is_file());
        std::fs::create_dir_all(dir.join("empty")).unwrap();
        assert!(import_skills(&dir.join("empty"), &dest, true, false).unwrap()[0].starts_with("no SKILL.md"));
        assert!(import_skills(&dir.join("nothing-here"), &dest, true, false).unwrap().is_empty(), "a missing folder is nothing to import");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repository_specs_are_checked() {
        assert_eq!(repo_spec("anthropics/skills/skills").unwrap(), ("https://github.com/anthropics/skills.git".to_owned(), "skills".to_owned()));
        assert!(repo_spec("file:///etc").is_err());
        assert!(repo_spec("justone").is_err());
        assert!(repo_spec("a b/c").is_err());
        assert!(repo_spec("https://example.com/x.git").is_ok());
    }
}
