#!/usr/bin/env python3
"""One word at a time: the tests of row 82 must go red.

Covers the two commands of the worktree row: the action the page picks, the usage lines, the
batch's two words, and everything the answer does — the four notices with their spacing, the
list's fallback, the two failure families, and the pieces that wait for the folder and its
tools.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # desktop/native
SRC = ROOT / "src"
# the whole environment is kept: rustup finds its toolchain only over RUSTUP_HOME/CARGO_HOME
CARGO_ENV = dict(__import__("os").environ)
CARGO_ENV["PATH"] = "/opt/data/home/.cargo/bin:" + CARGO_ENV.get("PATH", "")
CARGO_ENV["CARGO_TARGET_DIR"] = "/opt/data/cache/scratch/fllm-app-target"
CARGO_ENV["FLLM_SNAPSHOT_DIR"] = str(ROOT / "ui-snapshots")

# (file, old text, new text, which tests must go red)
MUTATIONS = [
    # ---- what the dispatch plans (the fixtures carry the page's own records)
    (
        "commands.rs",
        """            let action: &'static str = match sub {
                "new" => "new",
                "leave" => "leave",
                "remove" => "remove",
                _ => "list",
            };""",
        """            let action: &'static str = match sub {
                "new" => "list",
                "leave" => "leave",
                "remove" => "remove",
                _ => "list",
            };""",
        "the fixtures",
    ),
    (
        "commands.rs",
        "            if matches!(action, \"new\" | \"remove\") && name.is_none() {",
        "            if false {",
        "the fixtures (the usage lines)",
    ),
    (
        "commands.rs",
        'pub const WORKTREE_USAGE: &str = "Usage: /worktree new <name> · /worktree leave · /worktree remove <name> · /worktree (list)";',
        'pub const WORKTREE_USAGE: &str = "Usage: /worktree new <name>";',
        "the fixtures (the usage lines)",
    ),
    (
        "commands.rs",
        'pub const BATCH_USAGE: &str = "Usage: /batch <name> <task> — the task runs in a new worktree on branch <name>.";',
        'pub const BATCH_USAGE: &str = "Usage: /batch <name> <task>.";',
        "the fixtures (the usage lines)",
    ),
    (
        "commands.rs",
        "            if name.is_empty() || rest.is_empty() {",
        "            if name.is_empty() {",
        "the fixtures (a batch without a task)",
    ),
    # ---- what the answer does (the window tests carry the same canned data)
    (
        "code.rs",
        'None => format!(\n                                "Worktree {branch} at {path}. The agent works there now; /worktree leave brings you back, /worktree remove {name} deletes it."\n                            ),',
        'None => format!("Worktree {branch} at {path}."),',
        "the window: a new worktree",
    ),
    (
        "code.rs",
        'Some(b) if !b.is_empty() => format!(", branch {b}"),',
        'Some(b) if !b.is_empty() => format!(", {b}"),',
        "the window: back to the main worktree",
    ),
    (
        "code.rs",
        'self.items.push(Item::Note(format!(\n                            "Removed the worktree {removed}. Its branch stays until you delete it."\n                        )));',
        'self.items.push(Item::Note(format!("Removed {removed}.")));',
        "the window: remove and the failures",
    ),
    (
        "code.rs",
        """                                            if w.get("current").and_then(Value::as_bool).unwrap_or(false) { "*" } else { " " },
                                            w.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()).unwrap_or("(main)"),
                                            w.get("branch").and_then(Value::as_str).unwrap_or(""),
                                            w.get("path").and_then(Value::as_str).unwrap_or("")""",
        """                                            if w.get("current").and_then(Value::as_bool).unwrap_or(false) { "*" } else { "" },
                                            w.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()).unwrap_or("(main)"),
                                            w.get("branch").and_then(Value::as_str).unwrap_or(""),
                                            w.get("path").and_then(Value::as_str).unwrap_or("")""",
        "the window: the rows of the list",
    ),
    (
        "code.rs",
        '                            "No worktrees. /worktree new <name> creates one.".to_string()',
        '                            "No worktrees.".to_string()',
        "the window: the rows of the list",
    ),
    (
        "code.rs",
        '                            "The worktree could not be created.".to_string()',
        '                            "The worktree action failed.".to_string()',
        "the window: remove and the failures",
    ),
    (
        "code.rs",
        """                        if e.starts_with("transport:") {
                            format!("The batch could not start: {}", e.trim_start_matches("transport:"))
                        } else {""",
        """                        if false {
                            format!("The batch could not start: {}", e.trim_start_matches("transport:"))
                        } else {""",
        "the window: remove and the failures",
    ),
    # ---- the pieces that wait for the folder and its tools
    (
        "code.rs",
        "                        self.pending_notice = Some(notice);\n                        self.pending_batch = batch;",
        "                        self.pending_batch = batch;",
        "the window: a new worktree / a batch end to end",
    ),
    (
        "code.rs",
        "        let Some(task) = self.pending_batch.take() else {",
        "        let Some(task) = (None::<String>).take() else {",
        "the window: a batch end to end",
    ),
    (
        "code.rs",
        '        self.run_prompt(&task, "/batch", env)',
        '        self.run_prompt(&task, "task", env)',
        "the window: a batch end to end",
    ),
]

TESTS = [
    ["cargo", "test", "--locked", "--test", "worktree_window"],
    ["cargo", "test", "--locked", "--test", "commands_fixtures"],
]


def run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=ROOT, env=CARGO_ENV).returncode


original: dict[str, str] = {}
red: list[str] = []
problems: list[str] = []
for file, old, new, must in MUTATIONS:
    path = SRC / file
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        problems.append(f"{file}: {count} matches for {old[:60]!r} — fix the script")
        continue
    original[file] = text
    path.write_text(text.replace(old, new, 1))
    failing = []
    for cmd in TESTS:
        if run(cmd) != 0:
            failing.append(cmd[cmd.index("--") + 1] if "--" in cmd else cmd[-1])
    path.write_text(original[file])
    if failing:
        red.append(f"{file}: {old[:60]!r} → {failing}")
        print(f"RED   {old[:78]!r}", flush=True)
    else:
        problems.append(f"{file}: {old[:60]!r} ({must})")
        print(f"GREEN {old[:78]!r}", flush=True)

for file, text in original.items():
    (SRC / file).write_text(text)

print(f"{len(red)}/{len(MUTATIONS)} mutations red", flush=True)
if problems:
    print("PROBLEMS:", problems, flush=True)
    sys.exit(1)
print("ALL MUTATIONS BIT", flush=True)
