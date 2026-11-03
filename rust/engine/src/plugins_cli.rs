//! `fllm-rs plugins`: install plugins from git repositories, and use a repository as a small marketplace.
//!
//! ```text
//! fllm-rs plugins add owner/repo            a repository with plugin.json at its root: installs it
//! fllm-rs plugins add owner/repo@v1.2       ... at a branch or tag
//! fllm-rs plugins add owner/repo            a repository with marketplace.json: lists what it offers
//! fllm-rs plugins add owner/repo --plugin rust-review      ... installs one entry (or --all)
//! fllm-rs plugins list | remove NAME
//! ```
//!
//! A marketplace is any git repository with a `marketplace.json` at the root:
//! `{"name": "my-plugins", "plugins": [{"name": "rust-review", "source": "plugins/rust-review", "description": "..."},
//! {"name": "other", "source": "someone/other-plugin"}]}`. A `source` is a folder inside that repository or another repository
//! (`owner/repo`, a full git address). The result is the same as installing from the Settings page: a folder under `plugins/`
//! of the Studio folder that is read in place (skills, commands, agents, hooks). A plugin can carry hooks, which are commands that
//! run with your rights, so the command shows what the plugin contains before it asks (with `--yes` it does not ask).
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const HELP: &str = "usage: fllm-rs plugins add SOURCE [--plugin NAME | --all] [--yes]
       fllm-rs plugins list
       fllm-rs plugins remove NAME
SOURCE: owner/repo[@branch-or-tag] (GitHub), a full git address (https://, ssh://, git@, file://), or a local folder.
A repository with plugin.json at its root is installed; a repository with marketplace.json lists its plugins, --plugin NAME installs one,
--all every one. Before installing, the plugin's skills, commands, agents and hooks are listed (hooks run commands with your rights) and
you are asked, unless --yes. Plugins live in the plugins folder of the Studio folder and are read in place.";

const MAX_BYTES: u64 = 50 * 1024 * 1024;
const MAX_ENTRIES: usize = 5_000;

pub fn plugins_dir() -> PathBuf {
    crate::hub::studio_data_file("plugins")
}

pub fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 64 && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_') && !name.starts_with('-')
}

/// `owner/repo[@ref]` or a git address, as (address, ref). A local folder is returned as it is.
pub fn resolve_source(source: &str) -> Result<(String, Option<String>), String> {
    let s = source.trim();
    if s.is_empty() || s.contains(char::is_whitespace) || s.contains('\u{0}') {
        return Err("the source is empty or holds a space".into());
    }
    if ["https://", "http://", "ssh://", "git@", "file://"].iter().any(|p| s.starts_with(p)) {
        return Ok((s.to_owned(), None));
    }
    if Path::new(s).is_absolute() || s.starts_with("./") || s.starts_with("../") || s.starts_with('.') {
        return Ok((s.to_owned(), None));
    }
    let (repo, reference) = match s.split_once('@') {
        Some((r, v)) => (r, Some(v.to_owned())),
        None => (s, None),
    };
    let mut parts = repo.split('/');
    let (owner, name, rest) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next());
    let ok = |p: &str| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if rest.is_some() || !ok(owner) || !ok(name) {
        return Err(format!("'{s}' is not owner/repo, a git address or an absolute folder"));
    }
    if reference.as_deref().is_some_and(|r| r.is_empty() || r.starts_with('-')) {
        return Err("the branch or tag after @ looks wrong".into());
    }
    Ok((format!("https://github.com/{owner}/{name}"), reference))
}

fn git_clone(address: &str, reference: Option<&str>, to: &Path) -> Result<(), String> {
    let mut c = std::process::Command::new("git");
    c.args(["clone", "--quiet", "--depth", "1"]);
    if let Some(r) = reference {
        c.args(["--branch", r]);
    }
    c.arg(address).arg(to).env("GIT_TERMINAL_PROMPT", "0").env("GIT_ASKPASS", "true");
    let out = c.output().map_err(|e| format!("cannot run git: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!("git clone failed: {}", String::from_utf8_lossy(&out.stderr).trim().chars().take(300).collect::<String>()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub name: String,
    pub source: String,
    pub description: String,
}

pub fn marketplace_entries(text: &str) -> Result<Vec<Entry>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("marketplace.json is not valid JSON: {e}"))?;
    let list = v["plugins"].as_array().ok_or("marketplace.json needs a \"plugins\" list")?;
    let mut out = Vec::new();
    for item in list {
        let name = item["name"].as_str().unwrap_or("").trim().to_owned();
        let source = item["source"].as_str().unwrap_or("").trim().to_owned();
        if !valid_name(&name) || source.is_empty() {
            return Err(format!("a marketplace entry needs a valid name and a source ({})", item));
        }
        out.push(Entry { name, source, description: item["description"].as_str().unwrap_or("").trim().to_owned() });
    }
    Ok(out)
}

/// What a plugin folder contains, one line per kind, for the question before the install.
pub fn describe(folder: &Path) -> String {
    let count = |sub: &str, file: Option<&str>| -> usize {
        let Ok(rd) = std::fs::read_dir(folder.join(sub)) else { return 0 };
        rd.filter_map(Result::ok).filter(|e| match file {
            Some(f) => e.path().join(f).is_file(),
            None => e.path().extension().is_some_and(|x| x == "md"),
        }).count()
    };
    let hooks = std::fs::read_to_string(folder.join("hooks.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).map_or(0, |v| {
        v["hooks"].as_object().map_or(0, |m| m.values().filter_map(Value::as_array).map(Vec::len).sum())
    });
    let mut lines = vec![format!("  skills: {}, commands: {}, agents: {}, hooks: {}", count("skills", Some("SKILL.md")), count("commands", None), count("agents", None), hooks)];
    if hooks > 0 {
        lines.push("  WARNING: the hooks run commands on this computer with your rights; read hooks.json first.".to_owned());
    }
    lines.join("\n")
}

fn copy_dir(from: &Path, to: &Path, bytes: &mut u64, entries: &mut usize) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| format!("cannot create {}: {e}", to.display()))?;
    for entry in std::fs::read_dir(from).map_err(|e| e.to_string())?.filter_map(Result::ok) {
        let name = entry.file_name();
        if name == ".git" {
            continue;
        }
        let meta = std::fs::symlink_metadata(entry.path()).map_err(|e| e.to_string())?;
        // a link could point out of the plugin: it is skipped
        if meta.file_type().is_symlink() {
            continue;
        }
        *entries += 1;
        if *entries > MAX_ENTRIES {
            return Err("the plugin has too many files".into());
        }
        if meta.is_dir() {
            copy_dir(&entry.path(), &to.join(&name), bytes, entries)?;
        } else {
            *bytes += meta.len();
            if *bytes > MAX_BYTES {
                return Err("the plugin is larger than 50 MB".into());
            }
            std::fs::copy(entry.path(), to.join(&name)).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Copies a folder with `plugin.json` into the plugins folder `into`. Returns the plugin name.
pub fn install_folder(from: &Path, into: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(from.join("plugin.json")).map_err(|_| format!("{} holds no plugin.json", from.display()))?;
    let meta: Value = serde_json::from_str(&text).map_err(|e| format!("plugin.json is not valid JSON: {e}"))?;
    let name = meta["name"].as_str().unwrap_or("").trim().to_owned();
    if !valid_name(&name) {
        return Err("plugin.json needs a name of 1 to 64 characters: lowercase letters, digits, '-' and '_'".into());
    }
    let target = into.join(&name);
    if target.exists() {
        return Err(format!("a plugin named '{name}' is installed already; remove it first (fllm-rs plugins remove {name})"));
    }
    let staging = into.join(format!(".{name}.incoming"));
    let _ = std::fs::remove_dir_all(&staging);
    let (mut bytes, mut entries) = (0u64, 0usize);
    if let Err(e) = copy_dir(from, &staging, &mut bytes, &mut entries) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }
    std::fs::rename(&staging, &target).map_err(|e| {
        let _ = std::fs::remove_dir_all(&staging);
        format!("cannot move the plugin into place: {e}")
    })?;
    Ok(name)
}

/// Gets the source into a folder: clones a repository into `scratch`, or uses the local folder directly.
fn fetch(source: &str, scratch: &Path) -> Result<PathBuf, String> {
    let (address, reference) = resolve_source(source)?;
    if Path::new(&address).is_absolute() {
        return Ok(PathBuf::from(address));
    }
    if address.starts_with('.') {
        return Err("a local folder must be given as an absolute path".into());
    }
    let to = scratch.join(format!("clone-{}", scratch.read_dir().map_or(0, std::iter::Iterator::count)));
    git_clone(&address, reference.as_deref(), &to)?;
    Ok(to)
}

pub fn run(args: &[String]) -> Result<i32, String> {
    run_with(args, &plugins_dir(), &mut |q| ask(q))
}

fn ask(question: &str) -> bool {
    use std::io::{IsTerminal, Write};
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        eprintln!("{question} - not installed: there is no terminal to ask on; use --yes to install without asking.");
        return false;
    }
    eprint!("{question} [y/N]: ");
    std::io::stderr().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).is_ok() && matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

pub fn run_with(args: &[String], dir: &Path, confirm: &mut dyn FnMut(&str) -> bool) -> Result<i32, String> {
    let Some(sub) = args.first().map(String::as_str) else {
        println!("{HELP}");
        return Ok(0);
    };
    match sub {
        "--help" | "-h" | "help" => {
            println!("{HELP}");
            Ok(0)
        }
        "list" => {
            let mut found = 0;
            if let Ok(rd) = std::fs::read_dir(dir) {
                let mut names: Vec<_> = rd.filter_map(Result::ok).filter(|e| e.path().join("plugin.json").is_file()).map(|e| e.path()).collect();
                names.sort();
                for p in names {
                    let meta: Value = std::fs::read_to_string(p.join("plugin.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
                    println!("{} {}  {}", p.file_name().and_then(|n| n.to_str()).unwrap_or(""), meta["version"].as_str().unwrap_or(""), meta["description"].as_str().unwrap_or(""));
                    found += 1;
                }
            }
            if found == 0 {
                println!("No plugins installed. Install one: fllm-rs plugins add owner/repo");
            }
            Ok(0)
        }
        "remove" => {
            let name = args.get(1).ok_or("usage: fllm-rs plugins remove NAME")?;
            if !valid_name(name) {
                return Err("a plugin name has lowercase letters, digits, '-' and '_'".into());
            }
            let path = dir.join(name);
            if !path.is_dir() {
                return Err(format!("there is no plugin named '{name}'"));
            }
            std::fs::remove_dir_all(&path).map_err(|e| format!("cannot remove {}: {e}", path.display()))?;
            println!("Removed {name}.");
            Ok(0)
        }
        "add" => {
            let mut source: Option<&String> = None;
            let (mut pick, mut all, mut yes) = (None::<String>, false, false);
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--plugin" => {
                        i += 1;
                        pick = Some(args.get(i).ok_or("missing value for --plugin")?.clone());
                    }
                    "--all" => all = true,
                    "--yes" | "-y" => yes = true,
                    o if o.starts_with("--") => return Err(format!("unknown option {o}")),
                    _ => source = Some(&args[i]),
                }
                i += 1;
            }
            let source = source.ok_or("usage: fllm-rs plugins add SOURCE")?;
            std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
            let scratch = dir.join(format!(".fetch-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&scratch);
            std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
            let result = add(source, &scratch, dir, pick.as_deref(), all, yes, confirm);
            let _ = std::fs::remove_dir_all(&scratch);
            result
        }
        other => Err(format!("unknown plugins command '{other}'; see fllm-rs plugins --help")),
    }
}

fn add(source: &str, scratch: &Path, dir: &Path, pick: Option<&str>, all: bool, yes: bool, confirm: &mut dyn FnMut(&str) -> bool) -> Result<i32, String> {
    let root = fetch(source, scratch)?;
    let mut targets: Vec<(String, PathBuf)> = Vec::new();
    if root.join("plugin.json").is_file() && pick.is_none() {
        targets.push((source.to_owned(), root.clone()));
    } else if let Ok(text) = std::fs::read_to_string(root.join("marketplace.json")) {
        let entries = marketplace_entries(&text)?;
        if pick.is_none() && !all {
            println!("{source} offers {} plugin(s):", entries.len());
            for e in &entries {
                println!("  {}  {}", e.name, e.description);
            }
            println!("Install one with: fllm-rs plugins add {source} --plugin NAME   (or --all)");
            return Ok(0);
        }
        for e in entries.iter().filter(|e| all || pick == Some(e.name.as_str())) {
            // a source is a folder of this repository, or another repository
            let inside = root.join(&e.source);
            let folder = if !e.source.starts_with("http") && !e.source.starts_with("git@") && !e.source.starts_with("ssh://") && !e.source.contains("..") && inside.join("plugin.json").is_file() {
                inside
            } else {
                fetch(&e.source, scratch)?
            };
            targets.push((e.name.clone(), folder));
        }
        if targets.is_empty() {
            return Err(format!("{source} has no plugin named '{}'", pick.unwrap_or("")));
        }
    } else {
        return Err(format!("{source} has neither plugin.json nor marketplace.json at its root{}", if root.join("SKILL.md").is_file() || root.join("skills").is_dir() { " (it looks like a skills repository: use fllm-rs skills install)" } else { "" }));
    }
    for (label, folder) in targets {
        println!("{label}:\n{}", describe(&folder));
        if !yes && !confirm(&format!("Install {label}?")) {
            println!("Skipped {label}.");
            continue;
        }
        let name = install_folder(&folder, dir)?;
        println!("Installed {name}. Its skills, commands, agents and hooks are active from the next agent run.");
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fllm-plug-{tag}-{}-{}", std::process::id(), crate::agent_extras::fresh_session_name(11)));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn make_plugin(dir: &Path, name: &str, hooks: bool) {
        std::fs::create_dir_all(dir.join("skills").join("s1")).unwrap();
        std::fs::write(dir.join("plugin.json"), format!(r#"{{"name": "{name}", "version": "1.0.0", "description": "d"}}"#)).unwrap();
        std::fs::write(dir.join("skills").join("s1").join("SKILL.md"), "---\nname: s1\n---\nbody").unwrap();
        if hooks {
            std::fs::write(dir.join("hooks.json"), r#"{"hooks": {"on_file_edit": [{"command": "true"}]}}"#).unwrap();
        }
    }

    #[test]
    fn sources_are_github_shorthand_addresses_or_absolute_folders() {
        assert_eq!(resolve_source("acme/tools").unwrap(), ("https://github.com/acme/tools".to_owned(), None));
        assert_eq!(resolve_source("acme/tools@v1.2").unwrap(), ("https://github.com/acme/tools".to_owned(), Some("v1.2".to_owned())));
        assert_eq!(resolve_source("git@host:a/b.git").unwrap().0, "git@host:a/b.git");
        assert!(resolve_source("a/b/c").is_err() && resolve_source("a/b@-x").is_err() && resolve_source("a b").is_err() && resolve_source("").is_err());
        assert!(valid_name("rust-review") && !valid_name("Bad") && !valid_name("-x") && !valid_name(""));
    }

    #[test]
    fn marketplace_entries_are_checked() {
        let e = marketplace_entries(r#"{"plugins": [{"name": "a", "source": "plugins/a", "description": "A"}, {"name": "b", "source": "o/b"}]}"#).unwrap();
        assert_eq!((e.len(), e[0].description.as_str(), e[1].source.as_str()), (2, "A", "o/b"));
        assert!(marketplace_entries(r#"{"plugins": [{"name": "Bad Name", "source": "x"}]}"#).is_err());
        assert!(marketplace_entries(r#"{"nothing": 1}"#).is_err());
    }

    #[test]
    fn a_local_plugin_is_described_asked_about_and_installed_once() {
        let base = temp("local");
        let (src, dir) = (base.join("src"), base.join("plugins"));
        make_plugin(&src, "demo", true);
        assert!(describe(&src).contains("skills: 1") && describe(&src).contains("hooks: 1") && describe(&src).contains("WARNING"));
        let asked = std::cell::RefCell::new(Vec::<String>::new());
        let a = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        // a "no" installs nothing
        let r = run_with(&a(&["add", src.to_str().unwrap()]), &dir, &mut |q| { asked.borrow_mut().push(q.to_owned()); false }).unwrap();
        assert_eq!(r, 0);
        assert!(!dir.join("demo").exists() && asked.borrow()[0].contains("Install"));
        // --yes installs, a second time is refused
        run_with(&a(&["add", src.to_str().unwrap(), "--yes"]), &dir, &mut |_| false).unwrap();
        assert!(dir.join("demo").join("plugin.json").is_file() && dir.join("demo").join("skills").join("s1").join("SKILL.md").is_file());
        assert!(run_with(&a(&["add", src.to_str().unwrap(), "--yes"]), &dir, &mut |_| false).unwrap_err().contains("installed already"));
        run_with(&a(&["remove", "demo"]), &dir, &mut |_| false).unwrap();
        assert!(!dir.join("demo").exists());
        assert!(run_with(&a(&["remove", "demo"]), &dir, &mut |_| false).is_err());
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn a_marketplace_folder_lists_and_installs_entries() {
        let base = temp("market");
        let (repo, dir) = (base.join("repo"), base.join("plugins"));
        make_plugin(&repo.join("plugins").join("one"), "one", false);
        make_plugin(&repo.join("plugins").join("two"), "two", false);
        std::fs::write(repo.join("marketplace.json"), r#"{"plugins": [{"name": "one", "source": "plugins/one", "description": "First"}, {"name": "two", "source": "plugins/two"}]}"#).unwrap();
        let a = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        // without --plugin only the list is shown
        run_with(&a(&["add", repo.to_str().unwrap()]), &dir, &mut |_| true).unwrap();
        assert!(!dir.join("one").exists());
        run_with(&a(&["add", repo.to_str().unwrap(), "--plugin", "two", "--yes"]), &dir, &mut |_| true).unwrap();
        assert!(dir.join("two").is_dir() && !dir.join("one").exists());
        assert!(run_with(&a(&["add", repo.to_str().unwrap(), "--plugin", "zzz", "--yes"]), &dir, &mut |_| true).unwrap_err().contains("no plugin named"));
        run_with(&a(&["add", repo.to_str().unwrap(), "--all", "--yes"]), &dir, &mut |_| true).unwrap_err();
        assert!(dir.join("one").is_dir(), "--all installs the rest and reports the one that is there already");
        let _ = std::fs::remove_dir_all(base);
    }
}
