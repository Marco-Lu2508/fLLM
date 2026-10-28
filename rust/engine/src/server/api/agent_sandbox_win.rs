//! The decisions of the Windows sandbox, as plain portable functions.
//!
//! Everything that can be reasoned about without a Windows machine lives here and is unit-tested on every OS: the policy
//! of each mode, the layout of the private temp folder, the command line for `CreateProcess` (quoting), the scrubbed
//! environment and its UTF-16 block, which token groups are dropped, which folders may be labelled, the arguments of the
//! launcher and the self-test that proves the sandbox really denies a write. The system calls themselves are in
//! `agent_sandbox_native` (`cfg(windows)`) and only apply what this file decided.
//!
//! The mechanism (see `agent_sandbox_native`): every sandboxed command is started through `fllm-rs sandbox-exec`, a small
//! launcher that builds a restricted, low-integrity token, puts the process into a job object (kill on close, process and
//! memory limit, no clipboard/desktop access) and runs the command line from there. A launcher keeps the callers on
//! `std::process::Command`, so cancel, timeout and background jobs work as on the other systems.
#![cfg_attr(not(windows), allow(dead_code))]
use super::agent_sandbox::Mode;
use std::path::{Path, PathBuf};

/// Why `strict` does not exist on Windows. Shown in the status and returned as the error of a command.
pub(super) const STRICT_UNAVAILABLE: &str = "strict mode needs a network cut-off, which fLLM can only do on Windows with an AppContainer; \
that part is not implemented (it could not be verified), so strict is unavailable on Windows. Use workspace (writes confined, network \
allowed) or off";

/// Most processes the commands of one job may have at the same time (builds start many compilers, hence generous).
pub(super) const MAX_PROCESSES: u32 = 512;

/// What a mode means on Windows. Pure data: the native part applies it step by step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Policy {
    /// The process gets a low mandatory integrity level: it cannot write to anything that is not labelled low.
    pub low_integrity: bool,
    /// All privileges are removed from the token and administrator-like groups become deny-only.
    pub drop_privileges: bool,
    /// The job kills every process of the command when the launcher ends (stop, timeout, crash).
    pub kill_on_close: bool,
    pub max_processes: Option<u32>,
    /// The job may use this fraction (numerator, denominator) of the physical memory.
    pub memory_fraction: Option<(u64, u64)>,
    /// Job UI restrictions: no clipboard, desktop switching, global atoms, handles of other windows, system settings.
    pub ui_limits: bool,
    /// Environment variables that look like secrets are removed, TEMP/TMP point to the private folder.
    pub scrub_environment: bool,
}

/// The policy of `mode` on Windows, or the reason there is none.
pub(super) fn policy_for(mode: Mode) -> Result<Policy, String> {
    match mode {
        Mode::Off => Err("the sandbox is off".to_owned()),
        Mode::Strict => Err(STRICT_UNAVAILABLE.to_owned()),
        Mode::Workspace => Ok(Policy {
            low_integrity: true,
            drop_privileges: true,
            kill_on_close: true,
            max_processes: Some(MAX_PROCESSES),
            memory_fraction: Some((3, 4)),
            ui_limits: true,
            scrub_environment: true,
        }),
    }
}

/// The memory the job may use in total: the fraction of the physical memory, never less than 1 GiB.
pub(super) fn memory_limit_bytes(physical: u64, fraction: (u64, u64)) -> u64 {
    let (num, den) = (fraction.0, fraction.1.max(1));
    (physical / den).saturating_mul(num).max(1 << 30)
}

// ---------------------------------------------------------------------------------------------------------------------
// folders

/// `%USERPROFILE%\AppData\LocalLow`: Windows labels this folder (and everything created in it) low integrity, so a low
/// process can write there without anybody changing a label.
pub(super) fn local_low_dir(user_profile: &str) -> PathBuf {
    PathBuf::from(format!("{}\\AppData\\LocalLow", user_profile.trim_end_matches(['\\', '/'])))
}

/// The private temp folder of one server run, inside the low folder. `session` makes runs independent.
pub(super) fn private_temp_dir(local_low: &Path, session: &str) -> PathBuf {
    local_low.join("fLLM").join("sandbox-temp").join(session)
}

fn components(path: &str) -> Vec<String> {
    let stripped = path.strip_prefix("\\\\?\\").unwrap_or(path);
    stripped.split(['\\', '/']).filter(|c| !c.is_empty()).map(str::to_lowercase).collect()
}

/// Whether `root` may become the writable workspace of the sandbox. The workspace gets a low integrity label (that is
/// how a low process is allowed to write there), so a drive root, the user profile, an ancestor of it, a well-known
/// personal folder or a system folder is refused: labelling those would hand a large part of the machine to every low
/// process.
pub(super) fn workspace_label_allowed(root: &str, user_profile: &str) -> Result<(), String> {
    let root_parts = components(root);
    let profile = components(user_profile);
    if root_parts.len() < 2 {
        return Err("the workspace is a drive root; choose a project folder as the workspace for the sandbox".to_owned());
    }
    if !profile.is_empty() && profile.starts_with(&root_parts) {
        return Err("the workspace is the user profile or contains it; choose a project folder as the workspace for the sandbox".to_owned());
    }
    if ["windows", "program files", "program files (x86)", "programdata"].contains(&root_parts[1].as_str()) {
        return Err("the workspace is a system folder or inside one; choose a project folder for the sandbox".to_owned());
    }
    if !profile.is_empty() && root_parts.len() == profile.len() + 1 && root_parts.starts_with(&profile) {
        let name = root_parts[profile.len()].as_str();
        if ["desktop", "documents", "downloads", "appdata", "pictures", "music", "videos", "onedrive", ".ssh", ".gnupg", ".aws"].contains(&name) {
            return Err(format!("the workspace is the personal folder '{name}'; choose a project folder inside it for the sandbox"));
        }
    }
    Ok(())
}

/// The arguments of `icacls` that give `root` (and, with `/T`, everything below it) the low integrity label.
pub(super) fn icacls_label_args(root: &str) -> Vec<String> {
    vec![root.to_owned(), "/setintegritylevel".into(), "(OI)(CI)L".into(), "/T".into(), "/C".into(), "/Q".into()]
}

/// The folders that already got the label, one per line (case-insensitive on Windows).
pub(super) fn labelled_contains(list: &str, root: &str) -> bool {
    let wanted = root.to_lowercase();
    list.lines().any(|line| line.trim().to_lowercase() == wanted)
}

// ---------------------------------------------------------------------------------------------------------------------
// command lines

/// One argument quoted for `CreateProcess` the way `CommandLineToArgvW` and the C runtime read it back.
pub(super) fn quote_arg(arg: &str) -> String {
    if !arg.is_empty() && !arg.contains([' ', '\t', '\n', '\x0b', '"']) {
        return arg.to_owned();
    }
    let mut out = String::from("\"");
    let mut backslashes = 0usize;
    for c in arg.chars() {
        match c {
            '\\' => backslashes += 1,
            '"' => {
                out.push_str(&"\\".repeat(backslashes * 2 + 1));
                out.push('"');
                backslashes = 0;
            }
            _ => {
                out.push_str(&"\\".repeat(backslashes));
                backslashes = 0;
                out.push(c);
            }
        }
    }
    out.push_str(&"\\".repeat(backslashes * 2));
    out.push('"');
    out
}

/// The program followed by its arguments as one command line (used to prove [`quote_arg`] by reading the line back).
#[cfg(test)]
pub(super) fn join_command_line(program: &str, args: &[String]) -> String {
    let mut line = quote_arg(program);
    for arg in args {
        line.push(' ');
        line.push_str(&quote_arg(arg));
    }
    line
}

/// The command line of the shell that runs `line`: the same `cmd /V:ON /S /C "..."` as the unsandboxed commands, where
/// `/S` strips exactly the outer quotes and the text in between is passed on untouched.
pub(super) fn cmd_command_line(comspec: &str, line: &str) -> Result<String, String> {
    if line.contains('\0') {
        return Err("the command contains a NUL character".to_owned());
    }
    Ok(format!("{} /V:ON /S /C \"{line}\"", quote_arg(comspec)))
}

/// A NUL-terminated UTF-16 copy for the wide API calls.
pub(super) fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

// ---------------------------------------------------------------------------------------------------------------------
// environment

/// Whether an environment variable looks like it holds a secret that a sandboxed command has no business with.
pub(super) fn is_secret_env(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    const EXACT: [&str; 10] = [
        "TOKEN", "SECRET", "PASSWORD", "API_KEY", "APIKEY", "AUTH_TOKEN", "SSH_AUTH_SOCK", "GITHUB_PAT", "AWS_ACCESS_KEY_ID", "NPM_CONFIG__AUTH",
    ];
    const SUFFIXES: [&str; 11] = [
        "_TOKEN", "_SECRET", "_PASSWORD", "_PASSWD", "_API_KEY", "_APIKEY", "_ACCESS_KEY", "_SECRET_KEY", "_PRIVATE_KEY", "_CREDENTIALS", "_AUTH",
    ];
    const PREFIXES: [&str; 3] = ["AWS_SECRET", "AWS_SESSION", "AZURE_CLIENT_SECRET"];
    EXACT.contains(&upper.as_str()) || SUFFIXES.iter().any(|s| upper.ends_with(s)) || PREFIXES.iter().any(|p| upper.starts_with(p))
}

/// The environment of the sandboxed command: the given variables without the secrets, TEMP/TMP/TMPDIR replaced by the private
/// folder, sorted by name as `CreateProcess` wants it. Names are compared without regard to case.
pub(super) fn scrubbed_env(vars: impl IntoIterator<Item = (String, String)>, temp: &str, scrub: bool) -> Vec<(String, String)> {
    let mut kept: Vec<(String, String)> = vars
        .into_iter()
        .filter(|(name, _)| !name.is_empty() && !name.contains('='))
        .filter(|(name, _)| !scrub || !is_secret_env(name))
        .filter(|(name, _)| !["TEMP", "TMP", "TMPDIR"].contains(&name.to_ascii_uppercase().as_str()))
        .collect();
    for name in ["TEMP", "TMP"] {
        kept.push((name.to_owned(), temp.to_owned()));
    }
    kept.sort_by_key(|(name, _)| name.to_uppercase());
    kept.dedup_by(|a, b| a.0.eq_ignore_ascii_case(&b.0));
    kept
}

/// The `lpEnvironment` block for `CreateProcess` with `CREATE_UNICODE_ENVIRONMENT`: `NAME=value\0` for each variable and
/// an extra `\0` at the end (two for an empty block).
pub(super) fn environment_block(vars: &[(String, String)]) -> Vec<u16> {
    let mut block = Vec::new();
    for (name, value) in vars {
        block.extend(format!("{name}={value}").encode_utf16());
        block.push(0);
    }
    if block.is_empty() {
        block.push(0);
    }
    block.push(0);
    block
}

// ---------------------------------------------------------------------------------------------------------------------
// token groups

/// Whether a token group (given as the identifier authority and sub-authorities of its SID) is administrator-like and
/// has to become deny-only in the sandbox token: BUILTIN\Administrators, Power Users, the operator groups, Backup
/// Operators, Replicator and, in a domain, Domain/Schema/Enterprise Admins.
pub(super) fn should_disable_group(authority: [u8; 6], sub_authorities: &[u32]) -> bool {
    let nt = authority == [0, 0, 0, 0, 0, 5];
    if !nt {
        return false;
    }
    match sub_authorities {
        [32, rid] => [544, 547, 548, 549, 550, 551, 552].contains(rid),
        [21, .., last] if sub_authorities.len() >= 4 => [512, 518, 519].contains(last),
        _ => false,
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// the launcher

/// What the parent tells the launcher (`fllm-rs sandbox-exec`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LauncherRequest {
    pub mode: Mode,
    pub root: String,
    pub cwd: String,
    pub temp: String,
    pub line: String,
}

/// The arguments after the program name: `sandbox-exec --mode M --root R --cwd C --temp T -- LINE`. `Command::arg` quotes
/// each of them, and the launcher's own argument parser reads them back unchanged.
pub(super) fn launcher_args(request: &LauncherRequest) -> Vec<String> {
    vec![
        "sandbox-exec".into(),
        "--mode".into(),
        request.mode.name().into(),
        "--root".into(),
        request.root.clone(),
        "--cwd".into(),
        request.cwd.clone(),
        "--temp".into(),
        request.temp.clone(),
        "--".into(),
        request.line.clone(),
    ]
}

/// Reads what [`launcher_args`] wrote (without the leading `sandbox-exec`).
pub(super) fn parse_launcher_args(args: &[String]) -> Result<LauncherRequest, String> {
    let (mut mode, mut root, mut cwd, mut temp) = (None, None, None, None);
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--" {
            let line = args.get(i + 1).cloned().ok_or("sandbox-exec: the command line is missing")?;
            if args.len() > i + 2 {
                return Err("sandbox-exec: unexpected arguments after the command line".to_owned());
            }
            let need = |value: Option<String>, name: &str| value.ok_or_else(|| format!("sandbox-exec: {name} is missing"));
            let mode = match need(mode, "--mode")?.as_str() {
                "workspace" => Mode::Workspace,
                "strict" => Mode::Strict,
                other => return Err(format!("sandbox-exec: unknown mode '{other}'")),
            };
            return Ok(LauncherRequest { mode, root: need(root, "--root")?, cwd: need(cwd, "--cwd")?, temp: need(temp, "--temp")?, line });
        }
        let value = args.get(i + 1).cloned().ok_or_else(|| format!("sandbox-exec: {flag} needs a value"))?;
        match flag {
            "--mode" => mode = Some(value),
            "--root" => root = Some(value),
            "--cwd" => cwd = Some(value),
            "--temp" => temp = Some(value),
            other => return Err(format!("sandbox-exec: unknown option {other}")),
        }
        i += 2;
    }
    Err("sandbox-exec: the command line is missing".to_owned())
}

// ---------------------------------------------------------------------------------------------------------------------
// proof that it works

/// The proof the sandbox really sandboxes, used by Windows and macOS before they call themselves available. `run` runs
/// one shell line inside the sandbox and says whether it exited with 0. A command must run, a write to the allowed file
/// must work and a write to the denied file must fail: a sandbox that fails the third check is not a sandbox, and one that
/// fails the first two would break every command.
pub(super) fn enforcement_selftest(
    run: &dyn Fn(&str) -> Result<bool, String>,
    write_line: &dyn Fn(&str) -> String,
    allowed_file: &str,
    denied_file: &str,
) -> Result<(), String> {
    if !run("echo fllm-sandbox-ok")? {
        return Err("a plain command does not start inside the sandbox".to_owned());
    }
    if !run(&write_line(allowed_file))? {
        return Err("the sandbox does not let a command write to the workspace".to_owned());
    }
    if run(&write_line(denied_file))? {
        return Err("the sandbox does not stop a write outside the workspace".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reference reader of command lines (the documented rules of `CommandLineToArgvW` / the C runtime).
    fn parse_command_line(line: &str) -> Vec<String> {
        let chars: Vec<char> = line.chars().collect();
        let (mut args, mut i) = (Vec::new(), 0);
        while i < chars.len() {
            while i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
                i += 1;
            }
            if i >= chars.len() {
                break;
            }
            let (mut arg, mut in_quotes) = (String::new(), false);
            while i < chars.len() && (in_quotes || (chars[i] != ' ' && chars[i] != '\t')) {
                let mut backslashes = 0;
                while i < chars.len() && chars[i] == '\\' {
                    backslashes += 1;
                    i += 1;
                }
                if i < chars.len() && chars[i] == '"' {
                    arg.push_str(&"\\".repeat(backslashes / 2));
                    if backslashes % 2 == 1 {
                        arg.push('"');
                    } else {
                        in_quotes = !in_quotes;
                    }
                    i += 1;
                } else {
                    arg.push_str(&"\\".repeat(backslashes));
                    if i < chars.len() && (in_quotes || (chars[i] != ' ' && chars[i] != '\t')) {
                        arg.push(chars[i]);
                        i += 1;
                    }
                }
            }
            args.push(arg);
        }
        args
    }

    #[test]
    fn strict_has_no_policy_on_windows_and_says_why() {
        let reason = policy_for(Mode::Strict).unwrap_err();
        assert!(reason.contains("AppContainer") && reason.contains("unavailable"), "{reason}");
        assert!(policy_for(Mode::Off).is_err());
    }

    #[test]
    fn workspace_policy_drops_rights_and_limits_the_job() {
        let p = policy_for(Mode::Workspace).unwrap();
        assert!(p.low_integrity && p.drop_privileges && p.kill_on_close && p.ui_limits && p.scrub_environment);
        assert_eq!(p.max_processes, Some(512));
        assert_eq!(p.memory_fraction, Some((3, 4)));
    }

    #[test]
    fn the_memory_limit_is_a_fraction_with_a_floor() {
        assert_eq!(memory_limit_bytes(16 << 30, (3, 4)), 12 << 30);
        assert_eq!(memory_limit_bytes(1 << 30, (3, 4)), 1 << 30);
        assert_eq!(memory_limit_bytes(100, (3, 0)), 1 << 30);
    }

    #[test]
    fn the_private_temp_lives_in_the_low_folder() {
        let low = local_low_dir("C:\\Users\\ann\\");
        assert_eq!(low.to_string_lossy(), "C:\\Users\\ann\\AppData\\LocalLow");
        let temp = private_temp_dir(&low, "s1");
        assert!(temp.starts_with(&low) && temp.ends_with("s1"), "{temp:?}");
    }

    #[test]
    fn dangerous_workspaces_are_refused_for_labelling() {
        let profile = "C:\\Users\\ann";
        for bad in ["C:\\", "c:", "C:\\Users\\ann", "C:\\Users", "\\\\?\\C:\\Users\\Ann", "C:\\Windows", "C:\\Windows\\System32", "C:\\Program Files\\x", "C:\\Users\\ann\\Documents", "C:\\Users\\ann\\.ssh", "C:\\Users"] {
            assert!(workspace_label_allowed(bad, profile).is_err(), "{bad} must be refused");
        }
        for good in ["C:\\Users\\ann\\work\\proj", "D:\\code", "D:\\code\\proj", "C:\\Users\\ann\\Documents\\proj", "C:\\dev"] {
            assert!(workspace_label_allowed(good, profile).is_ok(), "{good} must be accepted");
        }
    }

    #[test]
    fn the_label_command_and_the_done_list() {
        let args = icacls_label_args("D:\\code\\my proj");
        assert_eq!(args[0], "D:\\code\\my proj");
        assert!(args.contains(&"(OI)(CI)L".to_owned()) && args.contains(&"/setintegritylevel".to_owned()));
        assert!(labelled_contains("d:\\CODE\\a\nD:\\code\\my proj\n", "d:\\code\\MY PROJ"));
        assert!(!labelled_contains("D:\\code\\a\n", "D:\\code"));
    }

    #[test]
    fn arguments_survive_the_round_trip_through_a_command_line() {
        let cases = [
            "plain", "", "with space", "tab\there", "say \"hi\"", "trailing\\", "trailing space \\", "back\\\\slash\\\"quote", "C:\\Program Files\\x y\\", "\\\\server\\share", "a\"b\"c", "\"", "\\\"", "ünï côde", "100% & (a|b) > c",
        ];
        for case in cases {
            let line = join_command_line("C:\\Windows\\System32\\cmd.exe", &[case.to_owned(), "tail".to_owned()]);
            let parsed = parse_command_line(&line);
            assert_eq!(parsed, vec!["C:\\Windows\\System32\\cmd.exe".to_owned(), case.to_owned(), "tail".to_owned()], "{line}");
        }
    }

    #[test]
    fn the_shell_line_is_cmd_with_the_outer_quotes_only() {
        let line = cmd_command_line("C:\\Windows\\System32\\cmd.exe", "echo \"a b\" && dir").unwrap();
        assert_eq!(line, "C:\\Windows\\System32\\cmd.exe /V:ON /S /C \"echo \"a b\" && dir\"");
        let spaced = cmd_command_line("C:\\Program Files\\sh\\cmd.exe", "dir").unwrap();
        assert!(spaced.starts_with("\"C:\\Program Files\\sh\\cmd.exe\" /V:ON"), "{spaced}");
        assert!(cmd_command_line("cmd.exe", "a\0b").is_err());
        assert_eq!(wide("ab"), vec![97, 98, 0]);
    }

    #[test]
    fn secrets_are_recognised_and_ordinary_variables_are_not() {
        for secret in ["GITHUB_TOKEN", "aws_secret_access_key", "AWS_SESSION_TOKEN", "OPENAI_API_KEY", "DB_PASSWORD", "SSH_AUTH_SOCK", "FLLM_API_KEY", "token"] {
            assert!(is_secret_env(secret), "{secret}");
        }
        for plain in ["PATH", "USERPROFILE", "SystemRoot", "COMSPEC", "KEYBOARD_LAYOUT", "TOKENIZERS_PARALLELISM", "HOME", "CARGO_HOME", "RUST_LOG"] {
            assert!(!is_secret_env(plain), "{plain}");
        }
    }

    #[test]
    fn the_environment_is_scrubbed_sorted_and_gets_the_private_temp() {
        let vars = vec![
            ("Path".to_owned(), "C:\\bin".to_owned()),
            ("GITHUB_TOKEN".to_owned(), "x".to_owned()),
            ("temp".to_owned(), "C:\\Users\\a\\AppData\\Local\\Temp".to_owned()),
            ("SystemRoot".to_owned(), "C:\\Windows".to_owned()),
            ("BROKEN=NAME".to_owned(), "v".to_owned()),
        ];
        let env = scrubbed_env(vars.clone(), "D:\\low\\t", true);
        let names: Vec<&str> = env.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["Path", "SystemRoot", "TEMP", "TMP"]);
        assert!(env.contains(&("TEMP".to_owned(), "D:\\low\\t".to_owned())));
        // without scrubbing the secrets stay, the temp folder is still private
        assert!(scrubbed_env(vars, "D:\\low\\t", false).iter().any(|(n, _)| n == "GITHUB_TOKEN"));
    }

    #[test]
    fn the_environment_block_is_double_nul_terminated() {
        let block = environment_block(&[("A".to_owned(), "1".to_owned()), ("B".to_owned(), "".to_owned())]);
        assert_eq!(String::from_utf16(&block).unwrap(), "A=1\0B=\0\0");
        assert_eq!(environment_block(&[]), vec![0, 0]);
    }

    #[test]
    fn administrator_like_groups_are_dropped_and_ordinary_ones_kept() {
        let nt = [0, 0, 0, 0, 0, 5];
        assert!(should_disable_group(nt, &[32, 544]), "Administrators");
        assert!(should_disable_group(nt, &[32, 551]), "Backup Operators");
        assert!(should_disable_group(nt, &[21, 1, 2, 3, 512]), "Domain Admins");
        assert!(!should_disable_group(nt, &[32, 545]), "Users");
        assert!(!should_disable_group(nt, &[11]), "Authenticated Users");
        assert!(!should_disable_group(nt, &[21, 1, 2, 3, 513]), "Domain Users");
        assert!(!should_disable_group([0, 0, 0, 0, 0, 1], &[32, 544]), "another authority");
    }

    #[test]
    fn the_launcher_request_survives_its_arguments() {
        let request = LauncherRequest {
            mode: Mode::Workspace,
            root: "D:\\my code\\p".into(),
            cwd: "D:\\my code\\p\\src".into(),
            temp: "C:\\Users\\a\\AppData\\LocalLow\\fLLM\\sandbox-temp\\1".into(),
            line: "echo \"--root\" -- x & exit /b !errorlevel!".into(),
        };
        let args = launcher_args(&request);
        assert_eq!(args[0], "sandbox-exec");
        assert_eq!(parse_launcher_args(&args[1..]).unwrap(), request);
        // a command line that looks like an option is still only the command line
        assert!(parse_launcher_args(&["--mode".into(), "workspace".into()]).is_err());
        assert!(parse_launcher_args(&["--bogus".into(), "1".into()]).is_err());
        let mut off = args[1..].to_vec();
        off[1] = "off".into();
        assert!(parse_launcher_args(&off).is_err());
    }

    #[test]
    fn the_selftest_demands_a_working_command_an_allowed_write_and_a_denied_write() {
        let write = |path: &str| format!("write {path}");
        let sandbox = |allow_denied: bool, allow_allowed: bool, plain: bool| {
            move |line: &str| -> Result<bool, String> {
                Ok(match line {
                    "echo fllm-sandbox-ok" => plain,
                    "write ok.txt" => allow_allowed,
                    "write no.txt" => allow_denied,
                    _ => false,
                })
            }
        };
        assert!(enforcement_selftest(&sandbox(false, true, true), &write, "ok.txt", "no.txt").is_ok());
        let leaking = enforcement_selftest(&sandbox(true, true, true), &write, "ok.txt", "no.txt").unwrap_err();
        assert!(leaking.contains("does not stop a write"), "{leaking}");
        assert!(enforcement_selftest(&sandbox(false, false, true), &write, "ok.txt", "no.txt").unwrap_err().contains("does not let"));
        assert!(enforcement_selftest(&sandbox(false, true, false), &write, "ok.txt", "no.txt").unwrap_err().contains("does not start"));
        let failing = |_: &str| -> Result<bool, String> { Err("spawn failed".into()) };
        assert_eq!(enforcement_selftest(&failing, &write, "a", "b").unwrap_err(), "spawn failed");
    }
}
