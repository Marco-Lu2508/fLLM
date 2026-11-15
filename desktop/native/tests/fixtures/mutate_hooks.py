#!/usr/bin/env python3
"""One word at a time: the tests of row 78 must go red.

Covers what the hook list changes in the run: the context of `user_prompt_submit` with its
block, the stop hook after the review with its two texts and its once-only, the gates the list
opens (own events and the plugins'), the answer the page reads, and the tail phase the review
starts from.
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
    # ---- the submit hook: its context rides along with the message
    (
        "code.rs",
        """                let prompt = if answer.context.is_empty() {
                    prompt
                } else {
                    format!("{prompt}\\n\\n[hooks]\\n{}", answer.context)
                };""",
        "                let prompt = prompt;",
        "the submit hook puts its context before the model",
    ),
    (
        "code.rs",
        '                    format!("{prompt}\\n\\n[hooks]\\n{}", answer.context)',
        '                    format!("{prompt}\\n\\n[hook]\\n{}", answer.context)',
        "the submit hook puts its context before the model",
    ),
    (
        "code.rs",
        "        if env.hooks.submit {",
        "        if false {",
        "the submit hook puts its context before the model / without a known hook",
    ),
    # ---- the stop hook: its two texts, its block, its once-only
    (
        "code.rs",
        'self.items.push(Item::Note(format!("A stop hook asks the agent to continue: {}", answer.reason)));',
        'self.items.push(Item::Note(format!("A stop hook will continue: {}", answer.reason)));',
        "a blocked stop hook sends the model back exactly once",
    ),
    (
        "code.rs",
        '"A stop hook asks you to continue before you finish: {}",',
        '"A stop hook wants you to go on before you finish: {}",',
        "a blocked stop hook sends the model back exactly once",
    ),
    (
        "code.rs",
        "                if answer.blocked && self.running {",
        "                if answer.blocked && self.running && false {",
        "a blocked stop hook sends the model back exactly once",
    ),
    (
        "code.rs",
        "        if matches!(self.tail, TailPhase::Hook) && self.main_run && env.hooks.stop && self.running {",
        "        if false && self.main_run && env.hooks.stop && self.running {",
        "a blocked stop hook sends the model back exactly once / plan mode still asks it",
    ),
    # ---- the list is the gate, own events and the plugins' alike
    (
        # the gates are built in app.rs where the pane's answer lands
        "app.rs",
        "                    .chain(answer.from_plugins.iter().map(|h| h.event.clone()))",
        "                    .chain(std::iter::empty::<String>())",
        "a blocked stop hook sends the model back exactly once (the plugin's event)",
    ),
    # ---- the answer the page reads
    (
        "code.rs",
        "        blocked: value.get(\"blocked\").and_then(Value::as_bool).unwrap_or(false),",
        "        blocked: value.get(\"blocked\").and_then(Value::as_bool).unwrap_or(true),",
        "the hook answer reads what the page reads",
    ),
    # ---- the tail: the review is reached through its phase, and its end falls into the hook
    (
        "code.rs",
        "        if matches!(self.tail, TailPhase::Review) {",
        "        if false {",
        "after the review the stop hook is asked (the review never starts)",
    ),
    (
        "code.rs",
        """                    self.review = None;
                    self.tail = TailPhase::Hook;
                    self.tail_hook_or_end(env)
                }
            };
        }
        if matches!(self.tail, TailPhase::Review) {""",
        """                    self.review = None;
                    self.tail_hook_or_end(env)
                }
            };
        }
        if matches!(self.tail, TailPhase::Review) {""",
        "after the review the stop hook is asked",
    ),
]

TESTS = [
    ["cargo", "test", "--locked", "--test", "hooks_window"],
    ["cargo", "test", "--locked", "--test", "review_window"],
    ["cargo", "test", "--locked", "--lib", "--", "fire_answer", "code::tests"],
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
