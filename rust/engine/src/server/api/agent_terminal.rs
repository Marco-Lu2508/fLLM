//! Terminal backends of the `bash` tool: where the shell commands of the agent run.
//!
//! * `local` (default): on this machine, in the sandbox of [`super::agent_sandbox`] when there is one.
//! * `docker:IMAGE`: every command is `docker run --rm` of that image with the workspace bind-mounted at `/workspace`; the
//!   container is the isolation (the local sandbox is not stacked on top). `FLLM_TERMINAL_NETWORK=none` cuts the network.
//! * `ssh://[user@]host[:port]/REMOTE/ROOT`: every command is `ssh host 'cd REMOTE/ROOT/<folder> && sh -c ...'`. The file tools
//!   still read and write the local workspace, so the remote folder must be the same files (a shared or synchronised folder);
//!   the backend maps the folder, it does not copy anything.
//!
//! Chosen by `FLLM_TERMINAL`, or by `terminal.json` in the Studio folder: `{"default": "docker:python:3.12",
//! "projects": {"/home/me/app": "ssh://me@box/srv/app"}}`. A file inside the project is deliberately **not** read: a downloaded
//! repository must not be able to send the agent's commands to a host of its choice. The working directory the command reports is
//! mapped back to the local path (`/workspace/x` -> `<root>/x`), so `cd` keeps working.
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Terminal {
    Local,
    Docker { image: String, network: bool },
    Ssh { target: String, port: Option<u16>, root: String },
}

const CONTAINER_ROOT: &str = "/workspace";

/// The container image names Docker accepts, narrowed: no option can hide in it.
fn valid_image(image: &str) -> bool {
    !image.is_empty() && image.len() <= 200 && image.chars().next().is_some_and(|c| c.is_ascii_alphanumeric()) && image.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | ':' | '@' | '-'))
}

fn valid_host(host: &str) -> bool {
    !host.is_empty() && host.len() <= 200 && !host.starts_with('-') && host.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '@' | '-' | ':' | '[' | ']'))
}

/// `docker:IMAGE`, `ssh://[user@]host[:port]/root` or `local`.
pub(super) fn parse(spec: &str, network: bool) -> Result<Terminal, String> {
    let spec = spec.trim();
    if spec.is_empty() || spec == "local" {
        return Ok(Terminal::Local);
    }
    if let Some(image) = spec.strip_prefix("docker:") {
        return if valid_image(image) { Ok(Terminal::Docker { image: image.to_owned(), network }) } else { Err(format!("'{image}' is not a usable container image name")) };
    }
    if let Some(rest) = spec.strip_prefix("ssh://") {
        let (authority, root) = rest.split_once('/').ok_or("ssh://host/REMOTE/ROOT needs the remote folder")?;
        let root = format!("/{}", root.trim_matches('/'));
        if root.len() < 2 || root.chars().any(|c| c.is_control()) || root.contains("..") {
            return Err("the remote folder must be an absolute path without '..'".into());
        }
        let (target, port) = match authority.rsplit_once(':') {
            Some((t, p)) if p.chars().all(|c| c.is_ascii_digit()) && !p.is_empty() => (t.to_owned(), Some(p.parse::<u16>().map_err(|_| "the ssh port is not a number")?)),
            _ => (authority.to_owned(), None),
        };
        return if valid_host(&target) { Ok(Terminal::Ssh { target, port, root }) } else { Err(format!("'{target}' is not a usable ssh target")) };
    }
    Err(format!("unknown terminal '{spec}' (local, docker:IMAGE, ssh://host/ROOT)"))
}

fn same_folder(a: &str, b: &str) -> bool {
    let norm = |s: &str| {
        let s = s.replace('\\', "/");
        let s = s.trim_end_matches('/').to_owned();
        if cfg!(windows) { s.to_lowercase() } else { s }
    };
    norm(a) == norm(b)
}

/// The backend for a workspace: `FLLM_TERMINAL` wins, then the project entry of `terminal.json`, then its default.
pub(super) fn select(root: &Path) -> Terminal {
    select_with(root, std::env::var("FLLM_TERMINAL").ok().as_deref(), &std::fs::read_to_string(crate::hub::studio_data_file("terminal.json")).unwrap_or_default(), std::env::var("FLLM_TERMINAL_NETWORK").ok().as_deref())
}

pub(super) fn select_with(root: &Path, env_spec: Option<&str>, file: &str, network_env: Option<&str>) -> Terminal {
    let v: Value = serde_json::from_str(file).unwrap_or(Value::Null);
    let network = !matches!(network_env.or_else(|| v["network"].as_str()), Some("none" | "off" | "false"));
    let root_text = root.to_string_lossy();
    let spec = env_spec.filter(|s| !s.trim().is_empty()).map(str::to_owned).or_else(|| {
        v["projects"].as_object().and_then(|m| m.iter().find(|(k, _)| same_folder(k, &root_text)).and_then(|(_, s)| s.as_str().map(str::to_owned))).or_else(|| v["default"].as_str().map(str::to_owned))
    });
    match spec {
        // a spec that does not parse must not fall back to running on this machine unnoticed: the caller reports it
        Some(s) => parse(&s, network).unwrap_or_else(|e| {
            eprintln!("[terminal] {e}; commands run locally");
            Terminal::Local
        }),
        None => Terminal::Local,
    }
}

pub(super) fn is_remote(t: &Terminal) -> bool {
    !matches!(t, Terminal::Local)
}

/// The path of `cwd` below `root` with forward slashes (`""` for the root itself).
fn relative(root: &Path, cwd: &Path) -> String {
    cwd.strip_prefix(root).map(|r| r.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/")).unwrap_or_default()
}

fn join_remote(base: &str, rel: &str) -> String {
    if rel.is_empty() { base.to_owned() } else { format!("{}/{rel}", base.trim_end_matches('/')) }
}

/// A word for a POSIX shell: single quotes, with the quote itself escaped.
pub(super) fn sh_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

/// The remote working directory for a local one.
pub(super) fn remote_cwd(t: &Terminal, root: &Path, cwd: &Path) -> String {
    let rel = relative(root, cwd);
    match t {
        Terminal::Docker { .. } => join_remote(CONTAINER_ROOT, &rel),
        Terminal::Ssh { root: remote, .. } => join_remote(remote, &rel),
        Terminal::Local => cwd.to_string_lossy().into_owned(),
    }
}

/// A path the command reported (container or remote) as a local path, or the text unchanged when it lies outside the mapped root.
pub(super) fn to_local(t: &Terminal, reported: &str, root: &Path) -> PathBuf {
    let base = match t {
        Terminal::Docker { .. } => CONTAINER_ROOT,
        Terminal::Ssh { root, .. } => root.as_str(),
        Terminal::Local => return PathBuf::from(reported),
    };
    let reported = reported.trim();
    match reported.strip_prefix(base.trim_end_matches('/')) {
        Some("") => root.to_path_buf(),
        Some(rest) if rest.starts_with('/') => rest.trim_start_matches('/').split('/').filter(|p| !p.is_empty() && *p != "." && *p != "..").fold(root.to_path_buf(), |acc, part| acc.join(part)),
        // outside the mapped root: an impossible local path, so the caller ignores the `cd`
        _ => PathBuf::from("/nonexistent-outside-the-workspace"),
    }
}

/// The argument list that runs `line` in the backend. `uid_gid` is the owner of the workspace on Unix, so files made in the container
/// belong to the user.
pub(super) fn argv(t: &Terminal, line: &str, root: &Path, cwd: &Path, container: &str, uid_gid: Option<(u32, u32)>) -> Option<Vec<String>> {
    match t {
        Terminal::Local => None,
        Terminal::Docker { image, network } => {
            let mut a: Vec<String> = ["docker", "run", "--rm", "--name", container, "--init", "--workdir"].map(str::to_owned).to_vec();
            a.push(remote_cwd(t, root, cwd));
            a.push("--mount".into());
            // --mount reads its value as CSV: a folder name with a comma or a quote needs the CSV quoting
            let source = format!("source={}", root.display());
            let source = if source.contains([',', '"']) { format!("\"{}\"", source.replace('"', "\"\"")) } else { source };
            a.push(format!("type=bind,{source},target={CONTAINER_ROOT}"));
            if !network {
                a.extend(["--network".into(), "none".into()]);
            }
            if let Some((uid, gid)) = uid_gid {
                a.extend(["--user".into(), format!("{uid}:{gid}")]);
            }
            a.extend(["--cap-drop".into(), "ALL".into(), "--security-opt".into(), "no-new-privileges".into(), image.clone(), "sh".into(), "-c".into(), line.to_owned()]);
            Some(a)
        }
        Terminal::Ssh { target, port, .. } => {
            let mut a: Vec<String> = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"].map(str::to_owned).to_vec();
            if let Some(p) = port {
                a.extend(["-p".into(), p.to_string()]);
            }
            a.push("--".into());
            a.push(target.clone());
            a.push(format!("cd {} && sh -c {}", sh_quote(&remote_cwd(t, root, cwd)), sh_quote(line)));
            Some(a)
        }
    }
}

#[cfg(unix)]
fn owner(root: &Path) -> Option<(u32, u32)> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(root).ok().map(|m| (m.uid(), m.gid()))
}

#[cfg(not(unix))]
fn owner(_root: &Path) -> Option<(u32, u32)> {
    None
}

static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CONTAINERS: Mutex<Option<HashMap<u32, String>>> = Mutex::new(None);

/// The command to start for `line`, when the workspace uses a remote backend.
pub(super) fn command(line: &str, root: &Path, cwd: &Path) -> Option<Command> {
    let t = select(root);
    let name = format!("fllm-agent-{}-{}", std::process::id(), COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let argv = argv(&t, line, root, cwd, &name, owner(root))?;
    let mut c = Command::new(&argv[0]);
    c.args(&argv[1..]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        c.process_group(0);
    }
    Some(c)
}

/// Remembers which container belongs to a started process, so that stopping the process stops the container too.
pub(super) fn note_spawn(cmd: &Command, pid: u32) {
    if cmd.get_program() != "docker" {
        return;
    }
    let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
    if let Some(name) = args.iter().position(|a| a == "--name").and_then(|i| args.get(i + 1)) {
        CONTAINERS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(pid, name.clone());
    }
}

/// Called when a process is killed: removes its container (killing the docker client alone would leave it running).
pub(super) fn stop_container_of(pid: u32) {
    let name = CONTAINERS.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|m| m.remove(&pid));
    if let Some(name) = name {
        let _ = Command::new("docker").args(["rm", "-f", &name]).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status();
    }
}

/// What the page shows next to the sandbox state.
pub(crate) fn status_json() -> Value {
    let t = select_with(Path::new(""), std::env::var("FLLM_TERMINAL").ok().as_deref(), &std::fs::read_to_string(crate::hub::studio_data_file("terminal.json")).unwrap_or_default(), std::env::var("FLLM_TERMINAL_NETWORK").ok().as_deref());
    match t {
        Terminal::Local => json!({"backend": "local"}),
        Terminal::Docker { image, network } => json!({"backend": "docker", "image": image, "network": network, "note": "commands run in a container; the local sandbox is not used"}),
        Terminal::Ssh { target, port, root } => json!({"backend": "ssh", "target": target, "port": port, "remote_root": root, "note": "commands run on the remote host in the mapped folder; file tools still use the local workspace"}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_are_parsed_and_dangerous_ones_are_refused() {
        assert_eq!(parse("", true), Ok(Terminal::Local));
        assert_eq!(parse("docker:python:3.12", false), Ok(Terminal::Docker { image: "python:3.12".into(), network: false }));
        assert_eq!(parse("ssh://me@box:2222/srv/app/", true), Ok(Terminal::Ssh { target: "me@box".into(), port: Some(2222), root: "/srv/app".into() }));
        assert!(parse("docker:--privileged", true).is_err());
        assert!(parse("docker:img --rm", true).is_err());
        assert!(parse("ssh://-oProxyCommand=x/srv", true).is_err());
        assert!(parse("ssh://box", true).is_err());
        assert!(parse("ssh://box/../etc", true).is_err());
        assert!(parse("telnet://x", true).is_err());
    }

    #[test]
    fn the_environment_beats_the_file_and_a_project_entry_beats_the_default() {
        let file = r#"{"default": "docker:alpine", "network": "none", "projects": {"/work/app": "ssh://me@box/srv/app"}}"#;
        assert_eq!(select_with(Path::new("/work/other"), None, file, None), Terminal::Docker { image: "alpine".into(), network: false });
        assert!(matches!(select_with(Path::new("/work/app"), None, file, None), Terminal::Ssh { .. }));
        assert!(matches!(select_with(Path::new("/work/app/"), None, file, None), Terminal::Ssh { .. }), "a trailing slash is the same folder");
        assert_eq!(select_with(Path::new("/work/app"), Some("local"), file, None), Terminal::Local, "'local' in the environment switches it off");
        assert_eq!(select_with(Path::new("/x"), Some("docker:busybox"), file, Some("on")), Terminal::Docker { image: "busybox".into(), network: true });
        assert_eq!(select_with(Path::new("/x"), None, "{broken", None), Terminal::Local);
        assert_eq!(select_with(Path::new("/x"), Some("docker:--bad"), "", None), Terminal::Local);
    }

    #[test]
    fn paths_are_mapped_both_ways() {
        let root = Path::new("/home/me/app");
        let docker = Terminal::Docker { image: "alpine".into(), network: true };
        let ssh = Terminal::Ssh { target: "box".into(), port: None, root: "/srv/app".into() };
        assert_eq!(remote_cwd(&docker, root, root), "/workspace");
        assert_eq!(remote_cwd(&docker, root, &root.join("src/a")), "/workspace/src/a");
        assert_eq!(remote_cwd(&ssh, root, &root.join("src")), "/srv/app/src");
        assert_eq!(to_local(&docker, "/workspace", root), root);
        assert_eq!(to_local(&docker, "/workspace/src/a", root), root.join("src").join("a"));
        assert_eq!(to_local(&ssh, "/srv/app/x", root), root.join("x"));
        assert_eq!(to_local(&docker, "/workspace/../etc", root), root.join("etc"), "'..' cannot climb out");
        assert_eq!(to_local(&docker, "/etc", root), PathBuf::from("/nonexistent-outside-the-workspace"));
        assert_eq!(to_local(&docker, "/workspace2/x", root), PathBuf::from("/nonexistent-outside-the-workspace"), "a sibling with the same prefix is outside");
        assert_eq!(to_local(&Terminal::Local, "/a/b", root), PathBuf::from("/a/b"));
    }

    #[test]
    fn the_docker_command_mounts_only_the_workspace_and_drops_privileges() {
        let t = Terminal::Docker { image: "python:3.12".into(), network: false };
        let a = argv(&t, "ls -la", Path::new("/home/me/app"), Path::new("/home/me/app/src"), "fllm-agent-1-0", Some((1000, 1000))).unwrap();
        let text = a.join(" ");
        assert!(text.starts_with("docker run --rm --name fllm-agent-1-0 --init --workdir /workspace/src --mount type=bind,source=/home/me/app,target=/workspace --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges python:3.12 sh -c ls -la"), "{text}");
        assert_eq!(a.last().unwrap(), "ls -la", "the command is one argument, never joined into a shell string");
        assert_eq!(a.iter().filter(|x| x.as_str() == "--mount").count(), 1);
        assert!(!text.contains("--privileged") && !text.contains("-v "));
        let net = argv(&Terminal::Docker { image: "alpine".into(), network: true }, "x", Path::new("/r"), Path::new("/r"), "n", None).unwrap();
        assert!(!net.iter().any(|x| x == "--network") && !net.iter().any(|x| x == "--user"));
        let odd = argv(&Terminal::Docker { image: "alpine".into(), network: true }, "x", Path::new("/r/a,b"), Path::new("/r/a,b"), "n", None).unwrap();
        assert!(odd.contains(&"type=bind,\"source=/r/a,b\",target=/workspace".to_owned()), "{odd:?}");
    }

    #[test]
    fn the_ssh_command_quotes_the_folder_and_the_command() {
        let t = Terminal::Ssh { target: "me@box".into(), port: Some(2222), root: "/srv/my app".into() };
        let a = argv(&t, "echo 'hi'; ls", Path::new("/home/me/app"), Path::new("/home/me/app/src"), "n", None).unwrap();
        assert_eq!(&a[..9], ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", "2222", "--"]);
        assert_eq!(a[9], "me@box");
        assert_eq!(a[10], "cd '/srv/my app/src' && sh -c 'echo '\\''hi'\\''; ls'");
        assert_eq!(sh_quote("it's"), "'it'\\''s'");
        assert!(argv(&Terminal::Local, "x", Path::new("/r"), Path::new("/r"), "n", None).is_none());
    }

    /// Runs against a real Docker: `FLLM_TEST_DOCKER_DIR=/a/folder/that/the/docker/daemon/sees cargo test real_docker -- --ignored`.
    #[test]
    #[ignore = "needs a Docker daemon and FLLM_TEST_DOCKER_DIR"]
    fn real_docker_runs_a_command_in_the_mapped_folder() {
        let Some(dir) = std::env::var_os("FLLM_TEST_DOCKER_DIR").map(PathBuf::from) else { panic!("set FLLM_TEST_DOCKER_DIR") };
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("hello.txt"), "from the host").unwrap();
        let t = Terminal::Docker { image: std::env::var("FLLM_TEST_DOCKER_IMAGE").unwrap_or_else(|_| "alpine:3".into()), network: false };
        let line = "pwd; cat hello.txt; echo made > new.txt; cd sub; printf '\\n__MARK__%s\\n' \"$(pwd)\"; (wget -q -T 2 -O- http://example.com >/dev/null 2>&1 && echo net-ok || echo net-blocked)";
        let a = argv(&t, line, &dir, &dir, "fllm-agent-test-real", owner(&dir)).unwrap();
        let out = Command::new(&a[0]).args(&a[1..]).output().expect("docker could not be started");
        let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{text}");
        assert!(text.contains("/workspace\nfrom the host"), "the folder is mounted at /workspace and the file is there: {text}");
        assert_eq!(std::fs::read_to_string(dir.join("new.txt")).unwrap().trim(), "made", "a file made in the container is in the host folder");
        assert!(text.contains("net-blocked"), "no network: {text}");
        let reported = text.lines().find_map(|l| l.strip_prefix("__MARK__")).unwrap_or_else(|| panic!("no folder report in: {text}"));
        assert_eq!(to_local(&t, reported, &dir), dir.join("sub"), "the reported folder maps back");
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(std::fs::metadata(dir.join("new.txt")).unwrap().uid(), std::fs::metadata(&dir).unwrap().uid(), "files belong to the workspace owner");
        }
    }

    #[test]
    fn a_container_is_tracked_by_the_pid_of_its_client() {
        let mut c = Command::new("docker");
        c.args(["run", "--name", "fllm-agent-9-9", "alpine"]);
        note_spawn(&c, 424_242);
        assert_eq!(CONTAINERS.lock().unwrap().as_ref().unwrap().get(&424_242).map(String::as_str), Some("fllm-agent-9-9"));
        // an unrelated program is not tracked
        let mut other = Command::new("sh");
        other.args(["--name", "x"]);
        note_spawn(&other, 424_243);
        assert!(!CONTAINERS.lock().unwrap().as_ref().unwrap().contains_key(&424_243));
        CONTAINERS.lock().unwrap().as_mut().unwrap().remove(&424_242);
    }
}
