//! The macOS sandbox: Seatbelt (`/usr/bin/sandbox-exec`) with a profile written for the three modes.
//!
//! UNVERIFIED: the profile text and the argument list are unit-tested (on every OS), but no macOS machine was available
//! to run them. Apple marks `sandbox-exec` as deprecated; it still ships, and it is what other agent tools use. Before
//! fLLM calls the sandbox available on a Mac it runs a self-test there (a command runs, a write into the workspace works,
//! a write outside is refused); when that fails the sandbox is reported as unavailable with the reason.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]
use super::agent_sandbox::Mode;

pub(super) const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// The Seatbelt profile of a mode. Paths are not spliced into the text: the profile reads `(param "ROOT")` and
/// `(param "TMP")`, which `sandbox-exec -D` fills in, so no path can change the profile.
///
/// Everything is denied first. Reading is allowed everywhere (as with the read-only root of the Linux sandbox), writing
/// only inside the workspace, the private temp folder and the terminal devices. `workspace` keeps the network; `strict`
/// has no `network*` rule at all, so sockets fail.
pub(super) fn profile(mode: Mode) -> String {
    let mut text = String::from(
        "(version 1)\n\
         (deny default)\n\
         (allow process-exec)\n\
         (allow process-fork)\n\
         (allow signal (target self))\n\
         (allow sysctl-read)\n\
         (allow mach-lookup)\n\
         (allow ipc-posix-shm)\n\
         (allow pseudo-tty)\n\
         (allow file-read*)\n\
         (allow file-ioctl)\n\
         (allow file-write*\n\
         \x20   (subpath (param \"ROOT\"))\n\
         \x20   (subpath (param \"TMP\"))\n\
         \x20   (literal \"/dev/null\")\n\
         \x20   (literal \"/dev/dtracehelper\")\n\
         \x20   (regex #\"^/dev/(tty|pty)\"))\n",
    );
    if mode == Mode::Workspace {
        text.push_str("(allow network*)\n");
    }
    text
}

/// The argument list that runs `line` under the profile of `mode`. `root` and `tmp` must be real paths (`/tmp` is
/// `/private/tmp` for Seatbelt), `cwd` is where the command starts.
pub(super) fn argv(mode: Mode, line: &str, root: &str, tmp: &str) -> Vec<String> {
    vec![
        SANDBOX_EXEC.to_owned(),
        "-p".to_owned(),
        profile(mode),
        "-D".to_owned(),
        format!("ROOT={root}"),
        "-D".to_owned(),
        format!("TMP={tmp}"),
        "--".to_owned(),
        "/usr/bin/env".to_owned(),
        format!("TMPDIR={tmp}"),
        "/bin/sh".to_owned(),
        "-c".to_owned(),
        line.to_owned(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_are_limited_to_the_workspace_and_the_private_temp() {
        let text = profile(Mode::Workspace);
        assert!(text.starts_with("(version 1)\n(deny default)"), "{text}");
        assert!(text.contains("(subpath (param \"ROOT\"))") && text.contains("(subpath (param \"TMP\"))"));
        assert!(text.contains("(allow file-read*)"));
        // no unconditional write rule: every write allowance is inside the one file-write* form
        assert_eq!(text.matches("file-write*").count(), 1, "{text}");
    }

    #[test]
    fn only_workspace_mode_has_the_network() {
        assert!(profile(Mode::Workspace).contains("(allow network*)"));
        assert!(!profile(Mode::Strict).contains("network"), "strict must not mention the network at all");
        // parentheses balance in both profiles
        for mode in [Mode::Workspace, Mode::Strict] {
            let text = profile(mode);
            assert_eq!(text.matches('(').count(), text.matches(')').count(), "{text}");
        }
    }

    #[test]
    fn paths_are_parameters_and_the_command_line_stays_one_argument() {
        let evil = "/w\") (allow default) (\"";
        let args = argv(Mode::Strict, "echo $HOME; rm -rf x", evil, "/private/tmp/t");
        assert!(!args[2].contains("allow default"), "a path must never end up in the profile");
        assert!(args.contains(&format!("ROOT={evil}")));
        assert_eq!(args.last().map(String::as_str), Some("echo $HOME; rm -rf x"));
        assert_eq!(args[7], "--");
    }
}
