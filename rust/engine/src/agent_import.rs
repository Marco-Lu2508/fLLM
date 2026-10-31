//! `fllm-rs import claude|codex|hermes`: brings the settings, memories, skills, commands, sub-agents and command allowlists of
//! another coding agent into fLLM. It only reads the other tool's files and shows what it would do; nothing is written without
//! `--apply`, and an existing file or skill is kept unless `--force` is given. API keys and login tokens are never read.
//!
//! The file layouts come from the documentation of those tools (Claude Code: `~/.claude/{settings.json,CLAUDE.md,skills,commands,
//! agents}` and the same folders in a project; Codex CLI: `~/.codex/{AGENTS.md,prompts,skills,rules}`; Hermes Agent:
//! `~/.hermes/{SOUL.md,memories,skills,config.yaml}`). They were not checked against a real installation of each tool.
use crate::agent_rules::{Rule, Rules};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const HELP: &str = "usage: fllm-rs import claude|codex|hermes [--from HOME] [--folder PROJECT] [--apply] [--force]
Shows what would be imported from the other agent's files (skills, commands, sub-agents, permission rules / command allowlists,
persona and memories). Nothing is written unless --apply is given; existing files and skills are kept unless --force is given.
--from HOME reads the other tool's folders below HOME instead of your home folder; --folder PROJECT also imports the project
files of that tool (.claude/ settings, commands, agents) into PROJECT/.fllm. API keys and tokens are never read or copied.";

const MAX_ENTRIES: usize = 100;
const MAX_ENTRY_CHARS: usize = 400;

pub struct Ctx {
    pub apply: bool,
    pub force: bool,
    pub studio: PathBuf,
    pub project: Option<PathBuf>,
    pub lines: Vec<String>,
}

impl Ctx {
    fn say(&mut self, text: impl Into<String>) {
        self.lines.push(text.into());
    }

    fn verb(&self) -> &'static str {
        if self.apply { "wrote" } else { "would write" }
    }

    /// Writes `content` to `dest` unless it exists (and `--force` is missing).
    fn put(&mut self, dest: &Path, content: &str, what: &str) -> Result<(), String> {
        if dest.exists() && !self.force {
            self.say(format!("kept   {what}: {} already exists", dest.display()));
            return Ok(());
        }
        if self.apply {
            if let Some(dir) = dest.parent() {
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            }
            std::fs::write(dest, content).map_err(|e| e.to_string())?;
        }
        let verb = self.verb();
        self.say(format!("{verb} {what} -> {}", dest.display()));
        Ok(())
    }
}

fn read(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > 2 << 20 {
        return None;
    }
    std::fs::read_to_string(path).ok()
}

/// Memory entries from a text: bullets or lines, or blocks separated by `§` (Hermes). Headings and empty lines are dropped.
pub fn entries_from_text(text: &str) -> Vec<String> {
    let blocks: Vec<String> = if text.contains('§') {
        text.split('§').map(|b| b.split_whitespace().collect::<Vec<_>>().join(" ")).collect()
    } else {
        text.lines().map(|l| l.trim().trim_start_matches(['-', '*', '+']).trim().to_owned()).collect()
    };
    let mut out: Vec<String> = Vec::new();
    for b in blocks {
        let b = b.trim();
        if b.is_empty() || b.starts_with('#') || b.starts_with("```") || b.starts_with("---") {
            continue;
        }
        let entry: String = b.chars().take(MAX_ENTRY_CHARS).collect();
        if !out.iter().any(|e| e.eq_ignore_ascii_case(&entry)) {
            out.push(entry);
        }
    }
    out
}

fn add_memory(ctx: &mut Ctx, entries: Vec<String>, what: &str) -> Result<(), String> {
    if entries.is_empty() {
        return Ok(());
    }
    let file = ctx.studio.join("agent-memory.md");
    let existing: Vec<String> = read(&file).map(|t| entries_from_text(&t)).unwrap_or_default();
    let mut merged = existing.clone();
    let mut added = 0;
    for e in entries {
        if merged.len() >= MAX_ENTRIES {
            break;
        }
        if !merged.iter().any(|m| m.eq_ignore_ascii_case(&e)) {
            merged.push(e);
            added += 1;
        }
    }
    if added == 0 {
        ctx.say(format!("kept   memory from {what}: nothing new (or the memory of 100 entries is full)"));
        return Ok(());
    }
    if ctx.apply {
        std::fs::create_dir_all(&ctx.studio).map_err(|e| e.to_string())?;
        let text: String = merged.iter().map(|e| format!("- {e}\n")).collect();
        std::fs::write(&file, text).map_err(|e| e.to_string())?;
    }
    let verb = ctx.verb();
    ctx.say(format!("{verb} {added} memory entr{} from {what} -> {}", if added == 1 { "y" } else { "ies" }, file.display()));
    Ok(())
}

pub fn rules_to_json(rules: &Rules) -> Value {
    let list = |v: &[Rule]| v.iter().map(|r| format!("{}: {}", r.category, r.pattern)).collect::<Vec<_>>();
    let mut v = json!({"allow": list(&rules.allow), "deny": list(&rules.deny), "ask": list(&rules.ask)});
    if let Some(mode) = &rules.mode {
        v["mode"] = json!(mode);
    }
    v
}

fn import_permissions(ctx: &mut Ctx, rules: &Rules, dest: &Path, what: &str) -> Result<(), String> {
    if rules.is_empty() && rules.mode.is_none() {
        return Ok(());
    }
    let text = serde_json::to_string_pretty(&rules_to_json(rules)).map_err(|e| e.to_string())?;
    ctx.put(dest, &format!("{text}\n"), &format!("{} rule(s) from {what}", rules.allow.len() + rules.deny.len() + rules.ask.len()))
}

fn import_markdown_dir(ctx: &mut Ctx, from: &Path, to: &Path, what: &str) -> Result<(), String> {
    let Ok(read_dir) = std::fs::read_dir(from) else { return Ok(()) };
    let mut files: Vec<_> = read_dir.flatten().filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("md")).collect();
    files.sort_by_key(|e| e.file_name());
    for e in files {
        let name = e.path().file_stem().and_then(|s| s.to_str()).map(crate::skills_cli::local_name).unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        if let Some(text) = read(&e.path()) {
            ctx.put(&to.join(format!("{name}.md")), &text, &format!("{what} /{name}"))?;
        }
    }
    Ok(())
}

/// Claude Code tool names to the fLLM tools of the same job.
pub fn map_tools(claude: &str) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for t in claude.split(',').map(str::trim) {
        let mapped: &[&str] = match t {
            "Read" | "LS" => &["read_file", "list_dir"],
            "Glob" => &["glob"],
            "Grep" => &["grep"],
            "Edit" | "MultiEdit" => &["edit_file", "multi_edit", "patch"],
            "Write" => &["write_file"],
            "Bash" => &["bash", "bash_output", "bash_kill"],
            "WebFetch" => &["web_fetch"],
            "WebSearch" => &["web_search"],
            "NotebookEdit" => &["notebook_edit"],
            _ => &[],
        };
        for m in mapped {
            if !out.contains(m) {
                out.push(m);
            }
        }
    }
    out
}

/// A Claude Code sub-agent file as an fLLM one: same name, description and instructions; the tool list is translated and the model
/// is dropped (it names a Claude model).
pub fn convert_agent(text: &str, fallback: &str) -> Option<(String, String)> {
    let (name, description, _) = crate::skills_cli::read_meta(text);
    let rest = text.strip_prefix("---\n")?;
    let end = rest.find("\n---")?;
    let head = &rest[..end];
    let body = rest[end + 4..].trim();
    let tools = head.lines().find_map(|l| l.strip_prefix("tools:")).map(str::trim).unwrap_or("");
    let name = crate::skills_cli::local_name(if name.is_empty() { fallback } else { &name });
    if name.is_empty() || body.is_empty() {
        return None;
    }
    let mapped = map_tools(tools);
    // a Claude agent without a tool list can use everything; an fLLM agent without one inherits the session
    let tool_line = if tools.is_empty() { String::new() } else if mapped.is_empty() { "tools: read_file, list_dir, glob, grep\n".to_owned() } else { format!("tools: {}\n", mapped.join(", ")) };
    Some((name.clone(), format!("---\nname: {name}\ndescription: {}\n{tool_line}---\n{body}\n", if description.is_empty() { name.clone() } else { description })))
}

fn import_agents(ctx: &mut Ctx, from: &Path, to: &Path, what: &str) -> Result<(), String> {
    let Ok(read_dir) = std::fs::read_dir(from) else { return Ok(()) };
    let mut files: Vec<_> = read_dir.flatten().filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("md")).collect();
    files.sort_by_key(|e| e.file_name());
    for e in files {
        let stem = e.path().file_stem().and_then(|s| s.to_str()).unwrap_or("agent").to_owned();
        match read(&e.path()).and_then(|t| convert_agent(&t, &stem)) {
            Some((name, text)) => ctx.put(&to.join(format!("{name}.md")), &text, &format!("{what} sub-agent {name}"))?,
            None => ctx.say(format!("skipped {what} agent file {stem}: no front matter or no instructions")),
        }
    }
    Ok(())
}

/// `prefix_rule(pattern=["git", "status"], decision="allow")` lines of a Codex rules file.
pub fn parse_codex_rules(text: &str) -> Rules {
    let mut rules = Rules::default();
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with("prefix_rule(") {
            continue;
        }
        let pattern = line.split("pattern=[").nth(1).and_then(|s| s.split(']').next()).unwrap_or("");
        let words: Vec<String> = pattern.split(',').map(|w| w.trim().trim_matches(['"', '\'']).to_owned()).filter(|w| !w.is_empty()).collect();
        if words.is_empty() {
            continue;
        }
        let rule = Rule { category: "bash".into(), pattern: format!("{}*", words.join(" ")) };
        match line.split("decision=").nth(1).map(|d| d.trim_start_matches(['"', '\''])) {
            Some(d) if d.starts_with("forbidden") => rules.deny.push(rule),
            Some(d) if d.starts_with("prompt") => rules.ask.push(rule),
            _ => rules.allow.push(rule),
        }
    }
    rules
}

/// Items of a `command_allowlist:` list in Hermes' `config.yaml`.
pub fn parse_hermes_allowlist(yaml: &str) -> Rules {
    let mut rules = Rules::default();
    let mut inside = false;
    for line in yaml.lines() {
        if line.starts_with("command_allowlist:") || line.starts_with("allowed_commands:") {
            inside = true;
            continue;
        }
        if inside {
            let t = line.trim();
            if let Some(item) = t.strip_prefix("- ") {
                let item = item.trim().trim_matches(['"', '\'']);
                if !item.is_empty() {
                    rules.allow.push(Rule { category: "bash".into(), pattern: format!("{item}*") });
                }
            } else if !t.is_empty() && !t.starts_with('#') {
                inside = false;
            }
        }
    }
    rules
}

fn project_fllm(ctx: &Ctx) -> Option<PathBuf> {
    ctx.project.as_ref().map(|p| p.join(".fllm"))
}

pub fn import_claude(ctx: &mut Ctx, home: &Path) -> Result<(), String> {
    let root = home.join(".claude");
    let skills = ctx.studio.join("skills");
    for line in crate::skills_cli::import_skills(&root.join("skills"), &skills, !ctx.apply, ctx.force)? {
        ctx.say(line);
    }
    if let Some(text) = read(&root.join("CLAUDE.md")) {
        add_memory(ctx, entries_from_text(&text), "~/.claude/CLAUDE.md")?;
    }
    if let Some(v) = read(&root.join("settings.json")).and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
        let dest = ctx.studio.join("permissions.json");
        import_permissions(ctx, &Rules::from_json(&v), &dest, "~/.claude/settings.json")?;
        if v["hooks"].is_object() {
            ctx.say("note   hooks of ~/.claude/settings.json are not converted (the event names and the stdin format differ); see Settings > Hooks");
        }
    }
    let studio_agents = ctx.studio.join("agents");
    import_agents(ctx, &root.join("agents"), &studio_agents, "Claude Code")?;
    if let Some(project) = ctx.project.clone() {
        let pc = project.join(".claude");
        let fllm = project.join(".fllm");
        for line in crate::skills_cli::import_skills(&pc.join("skills"), &fllm.join("skills"), !ctx.apply, ctx.force)? {
            ctx.say(line);
        }
        import_markdown_dir(ctx, &pc.join("commands"), &fllm.join("commands"), "Claude Code command")?;
        import_agents(ctx, &pc.join("agents"), &fllm.join("agents"), "Claude Code")?;
        for (src, dst) in [("settings.json", "permissions.json"), ("settings.local.json", "permissions.local.json")] {
            if let Some(v) = read(&pc.join(src)).and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
                import_permissions(ctx, &Rules::from_json(&v), &fllm.join(dst), &format!(".claude/{src}"))?;
            }
        }
        if project.join("CLAUDE.md").is_file() {
            ctx.say("note   CLAUDE.md of the project is read by fLLM as it is (AGENTS.md, FLLM.md or CLAUDE.md)");
        }
    } else if root.join("commands").is_dir() {
        ctx.say("note   ~/.claude/commands exist; custom commands belong to a project: run again with --folder PROJECT to copy them into PROJECT/.fllm/commands");
    }
    Ok(())
}

pub fn import_codex(ctx: &mut Ctx, home: &Path) -> Result<(), String> {
    let root = home.join(".codex");
    let skills = ctx.studio.join("skills");
    for line in crate::skills_cli::import_skills(&root.join("skills"), &skills, !ctx.apply, ctx.force)? {
        ctx.say(line);
    }
    if let Some(text) = read(&root.join("AGENTS.md")) {
        add_memory(ctx, entries_from_text(&text), "~/.codex/AGENTS.md")?;
    }
    if let Some(fllm) = project_fllm(ctx) {
        import_markdown_dir(ctx, &root.join("prompts"), &fllm.join("commands"), "Codex prompt")?;
    } else if root.join("prompts").is_dir() {
        ctx.say("note   ~/.codex/prompts exist; run again with --folder PROJECT to copy them into PROJECT/.fllm/commands");
    }
    let mut rules = Rules::default();
    if let Ok(read_dir) = std::fs::read_dir(root.join("rules")) {
        let mut files: Vec<_> = read_dir.flatten().filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("rules")).collect();
        files.sort_by_key(|e| e.file_name());
        for e in files {
            if let Some(text) = read(&e.path()) {
                rules.merge(parse_codex_rules(&text));
            }
        }
    }
    let dest = ctx.studio.join("permissions.json");
    import_permissions(ctx, &rules, &dest, "~/.codex/rules")?;
    if root.join("config.toml").is_file() {
        ctx.say("note   config.toml (approval policy, sandbox mode, profiles, model) is not converted: use the --allow-* flags, the sandbox setting and Settings > Providers");
    }
    Ok(())
}

pub fn import_hermes(ctx: &mut Ctx, home: &Path) -> Result<(), String> {
    let root = home.join(".hermes");
    let skills = ctx.studio.join("skills");
    for line in crate::skills_cli::import_skills(&root.join("skills"), &skills, !ctx.apply, ctx.force)? {
        ctx.say(line);
    }
    if let Some(text) = read(&root.join("SOUL.md")) {
        let dest = ctx.studio.join("SOUL.md");
        ctx.put(&dest, &text, "persona (SOUL.md)")?;
    }
    for file in ["MEMORY.md", "USER.md"] {
        if let Some(text) = read(&root.join("memories").join(file)).or_else(|| read(&root.join(file))) {
            add_memory(ctx, entries_from_text(&text), &format!("~/.hermes/{file}"))?;
        }
    }
    if let Some(yaml) = read(&root.join("config.yaml")) {
        let dest = ctx.studio.join("permissions.json");
        import_permissions(ctx, &parse_hermes_allowlist(&yaml), &dest, "the command allowlist of ~/.hermes/config.yaml")?;
        ctx.say("note   the rest of config.yaml (model, platforms, terminal backend, cron) is not converted; messaging tokens are never read");
    }
    Ok(())
}

pub fn run(args: &[String]) -> Result<i32, String> {
    let kind = match args.first().map(String::as_str) {
        Some(k @ ("claude" | "codex" | "hermes")) => k.to_owned(),
        _ => {
            println!("{HELP}");
            return Ok(if args.first().is_some_and(|a| a == "--help" || a == "-h") { 0 } else { 1 });
        }
    };
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let home = match flag("--from") {
        Some(h) => PathBuf::from(h),
        None => std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).ok_or("cannot find the home folder; use --from")?,
    };
    let mut ctx = Ctx {
        apply: args.iter().any(|a| a == "--apply"),
        force: args.iter().any(|a| a == "--force"),
        studio: crate::agent_cli::studio_folder(),
        project: flag("--folder").map(PathBuf::from),
        lines: Vec::new(),
    };
    match kind.as_str() {
        "claude" => import_claude(&mut ctx, &home)?,
        "codex" => import_codex(&mut ctx, &home)?,
        _ => import_hermes(&mut ctx, &home)?,
    }
    if ctx.lines.is_empty() {
        println!("nothing to import: no {kind} files found below {}", home.display());
    }
    for line in &ctx.lines {
        println!("{line}");
    }
    if !ctx.apply && !ctx.lines.is_empty() {
        println!("\nThis was a dry run. Run again with --apply to write these files.");
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fllm-import-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn ctx(dir: &Path, apply: bool) -> Ctx {
        Ctx { apply, force: false, studio: dir.join("studio"), project: Some(dir.join("proj")), lines: Vec::new() }
    }

    #[test]
    fn memory_entries_come_from_bullets_blocks_and_hermes_separators() {
        assert_eq!(entries_from_text("# Title\n- use tabs\n* run `make`\n\nplain line\n- use tabs\n"), vec!["use tabs", "run `make`", "plain line"]);
        assert_eq!(entries_from_text("first fact\nstill first§second fact§"), vec!["first fact still first", "second fact"]);
        assert_eq!(entries_from_text(&format!("- {}", "x".repeat(900)))[0].len(), 400);
    }

    #[test]
    fn claude_agents_are_converted_with_translated_tools() {
        let (name, text) = convert_agent("---\nname: Code Reviewer\ndescription: Reviews code\ntools: Read, Grep, Bash\nmodel: sonnet\n---\nYou review code.\n", "x").unwrap();
        assert_eq!(name, "code-reviewer");
        assert!(text.contains("tools: read_file, list_dir, grep, bash, bash_output, bash_kill\n"));
        assert!(!text.contains("model:") && text.ends_with("You review code.\n"));
        let (_, none) = convert_agent("---\nname: all\ndescription: d\n---\nbody\n", "x").unwrap();
        assert!(!none.contains("tools:"), "no tool line keeps the session's tools");
        assert!(convert_agent("no front matter", "x").is_none());
    }

    #[test]
    fn codex_rules_and_hermes_allowlists_become_rules() {
        let rules = parse_codex_rules("prefix_rule(pattern=[\"git\", \"status\"], decision=\"allow\")\nprefix_rule(pattern=[\"rm\"], decision=\"forbidden\")\nprefix_rule(pattern=[\"docker\",\"push\"], decision=\"prompt\")\n# c\n");
        assert_eq!((rules.allow.len(), rules.deny.len(), rules.ask.len()), (1, 1, 1));
        assert_eq!(rules.allow[0].pattern, "git status*");
        let h = parse_hermes_allowlist("model: x\ncommand_allowlist:\n  - ls\n  - \"git status\"\nother: 1\n  - not\n");
        assert_eq!(h.allow.iter().map(|r| r.pattern.as_str()).collect::<Vec<_>>(), vec!["ls*", "git status*"]);
    }

    #[test]
    fn a_dry_run_writes_nothing_and_apply_writes_without_overwriting() {
        let dir = temp("claude");
        let home = dir.join("home");
        std::fs::create_dir_all(home.join(".claude/skills/review")).unwrap();
        std::fs::create_dir_all(home.join(".claude/agents")).unwrap();
        std::fs::write(home.join(".claude/skills/review/SKILL.md"), "---\nname: review\ndescription: Review\n---\nsteps\n").unwrap();
        std::fs::write(home.join(".claude/CLAUDE.md"), "- prefer small commits\n").unwrap();
        std::fs::write(home.join(".claude/settings.json"), r#"{"permissions": {"allow": ["Bash(npm test:*)"], "deny": ["Read(.env)"]}, "env": {"ANTHROPIC_API_KEY": "sk-secret"}}"#).unwrap();
        std::fs::write(home.join(".claude/agents/r.md"), "---\nname: r\ndescription: d\ntools: Read\n---\nbody\n").unwrap();
        let mut dry = ctx(&dir, false);
        import_claude(&mut dry, &home).unwrap();
        assert!(dry.lines.iter().any(|l| l.starts_with("would")), "{:?}", dry.lines);
        assert!(!dir.join("studio").exists(), "a dry run touches nothing");
        let mut real = ctx(&dir, true);
        import_claude(&mut real, &home).unwrap();
        let studio = dir.join("studio");
        assert!(studio.join("skills/review/SKILL.md").is_file() && studio.join("agents/r.md").is_file());
        assert!(std::fs::read_to_string(studio.join("agent-memory.md")).unwrap().contains("prefer small commits"));
        let perms = std::fs::read_to_string(studio.join("permissions.json")).unwrap();
        assert!(perms.contains("bash: npm test*") && perms.contains("read: .env") && !perms.contains("sk-secret"));
        let mut again = ctx(&dir, true);
        import_claude(&mut again, &home).unwrap();
        assert!(again.lines.iter().any(|l| l.starts_with("kept")), "{:?}", again.lines);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hermes_persona_and_memories_are_imported() {
        let dir = temp("hermes");
        let home = dir.join("home");
        std::fs::create_dir_all(home.join(".hermes/memories")).unwrap();
        std::fs::write(home.join(".hermes/SOUL.md"), "You are terse.").unwrap();
        std::fs::write(home.join(".hermes/memories/MEMORY.md"), "likes rust§uses linux").unwrap();
        let mut c = ctx(&dir, true);
        import_hermes(&mut c, &home).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("studio/SOUL.md")).unwrap(), "You are terse.");
        let mem = std::fs::read_to_string(dir.join("studio/agent-memory.md")).unwrap();
        assert!(mem.contains("- likes rust\n") && mem.contains("- uses linux\n"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
