#!/usr/bin/env python3
"""One word at a time: the tests of row 59 must go red.

Covers the auto-approve table of the page, the words of the notice, the command in the box of
the Code page, and the way the switch reaches the store of the window.
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
    (
        "autoapprove.rs",
        '"`{head}` is not on the list of harmless programs"',
        '"`{head}` is not on the list of harmless tools"',
        "autoapprove module",
    ),
    (
        "autoapprove.rs",
        '"`{head} {sub}` is not on the list of harmless subcommands"',
        '"`{head} {sub}` is not on the list of harmless subcommand"',
        "autoapprove module",
    ),
    (
        "autoapprove.rs",
        'return Verdict::risky("it has more than one line");',
        'return Verdict::risky("it has more than one line of script");',
        "autoapprove module",
    ),
    (
        "autoapprove.rs",
        "    if word.to_lowercase() != \"autoapprove\" {",
        "    if word != \"autoapprove\" {",
        "autoapprove_window / the case of the word",
    ),
    (
        "autoapprove.rs",
        '        "on" => Some(true),',
        '        "on" => Some(false),',
        "autoapprove_window / on and off",
    ),
    (
        "autoapprove.rs",
        '"Automatic approval is off: every command asks again."',
        '"Automatic approval is off: every command must be asked for."',
        "autoapprove_window / the notice",
    ),
    (
        "code.rs",
        "        } else if self.auto_approve {",
        "        } else if false {",
        "code::tests::the_switch",
    ),
    (
        "code.rs",
        """                if let Some(want) = crate::autoapprove::command(&typed) {
                    self.editor = Content::new();""",
        """                if let Some(want) = crate::autoapprove::command(&typed) {""",
        "autoapprove_window / the box goes away",
    ),
    (
        "code.rs",
        "                    self.items.push(Item::Note(note));",
        "                    let _ = note;",
        "autoapprove_window / the notice stands",
    ),
    (
        "code.rs",
        "                        self.pending_auto_approve = Some(self.auto_approve);",
        "",
        "autoapprove_window / the words are saved",
    ),
    (
        "app.rs",
        "                        self.store.prefs.auto_approve = on;",
        "                        self.store.prefs.auto_approve = on && false;",
        "autoapprove_window / the store keeps it",
    ),
]

TESTS = [
    ["cargo", "test", "--locked", "--lib", "--", "code::tests::the_switch", "code::tests::the_category", "autoapprove"],
    ["cargo", "test", "--locked", "--test", "autoapprove_window"],
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
