#!/usr/bin/env python3
"""One word at a time: the tests of row 79 must go red.

Covers the dispatcher of the page's `runCodeCommand` (its notices, the help, the tail of the
box), and the decision of one call — the plan check, `authorize` with its rules, mode, remembered
keys and turbo, the harmless table, the words under the card, the *Always* button and the way the
window keeps what it remembers until a new session.
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
    # ---- what `Always` remembers (authorize of the page, code.js 1724-1728)
    (
        "commands.rs",
        '"network" => always.contains("network"),',
        '"network" => always.contains("networks"),',
        "decisions of the page",
    ),
    (
        "commands.rs",
        'always.contains(&format!("mcp:{subject}"))',
        'always.contains(&format!("mcp:{tool_name}"))',
        "decisions of the page",
    ),
    # ---- authorize: deny first, the mode and the rules and the keys, and the question last
    (
        "commands.rs",
        """        if evaluate_rules(rules, category, subject, agent) == Some("deny") {
            return Decision::Deny;
        }""",
        """        if evaluate_rules(rules, category, subject, agent) == Some("deny") {
            return Decision::Allow;
        }""",
        "decisions of the page / the window's denied card",
    ),
    (
        "commands.rs",
        "    if any_ask { Decision::Ask } else { Decision::Allow }",
        "    if any_ask { Decision::Allow } else { Decision::Ask }",
        "decisions of the page",
    ),
    (
        "commands.rs",
        """    if turbo {
        // turbo: no question, no rule
        return Decision::Allow;
    }""",
        """    if turbo && false {
        // turbo: no question, no rule
        return Decision::Allow;
    }""",
        "decisions of the page / the window's turbo test",
    ),
    # ---- the plan check in front of authorize (runToolCall, line 1967)
    (
        "commands.rs",
        "    if in_plan && !read_only {",
        "    if in_plan && false {",
        "decisions of the page / the window's plan test",
    ),
    # ---- what a mode allows per category (the defaults of the page's build agent)
    (
        "code.rs",
        "            _ => *mode == Mode::Auto,",
        "            _ => true,",
        "decisions of the page",
    ),
    (
        "code.rs",
        '            "edit" => matches!(mode, Mode::AutoEdit | Mode::Auto),',
        '            "edit" => matches!(mode, Mode::Auto),',
        "decisions of the page",
    ),
    # ---- the harmless table gets the last word on a question
    (
        "commands.rs",
        """            if !auto_approve {
                return Outcome::Ask;
            }""",
        """            if auto_approve {
                return Outcome::Ask;
            }""",
        "decisions of the page",
    ),
    (
        "commands.rs",
        """            if verdict.safe {
                Outcome::Running(Some(verdict.reason))""",
        """            if !verdict.safe {
                Outcome::Running(Some(verdict.reason))""",
        "decisions of the page",
    ),
    # ---- the words of the commands
    (
        "commands.rs",
        'pub const TURBO_CONFIRM: &str = "Turbo removes every permission question and the sandbox. Continue?";',
        'pub const TURBO_CONFIRM: &str = "Turbo removes every permission question and the sandbox?";',
        "the window's turbo dialog",
    ),
    (
        "commands.rs",
        "Enter while the agent works queues a message",
        "Enter while the agent works queues nothing",
        "the help of the fixtures",
    ),
    # ---- what the card says and the button it offers
    (
        "code.rs",
        'Some(head) => format!("Always ({head})"),',
        'Some(head) => format!("Always {head}"),',
        "the window's Always button",
    ),
    (
        "code.rs",
        'output: "Plan mode only allows reading. Write the plan instead.".into(),',
        'output: "Plan mode only reads. Write the plan instead.".into(),',
        "the window's plan test / code::tests",
    ),
    (
        "code.rs",
        'output: "Blocked by a permission rule.".into(),',
        'output: "Blocked.".into(),',
        "the window's denied card",
    ),
    # ---- the window forgets what Always remembered with the session
    (
        "code.rs",
        """                self.todos.clear();
                self.steps = 0;
                self.always_allow.clear();""",
        """                self.todos.clear();
                self.steps = 0;""",
        "the window's Always test",
    ),
    # ---- what the window keeps between sessions
    (
        "app.rs",
        "self.store.prefs.code_personality = personality;",
        'self.store.prefs.code_personality = "default".into();',
        "the window's personality test",
    ),
    (
        "app.rs",
        "if self.store.prefs.code_rules.get(&w.root) != Some(&rules) {",
        "if false {",
        "the window's rules test",
    ),
]

TESTS = [
    ["cargo", "test", "--locked", "--test", "commands_fixtures"],
    ["cargo", "test", "--locked", "--test", "commands_window"],
    ["cargo", "test", "--locked", "--lib", "--", "code::tests"],
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
