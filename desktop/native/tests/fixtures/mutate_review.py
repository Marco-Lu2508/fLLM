#!/usr/bin/env python3
"""One word at a time: the tests of row 81 must go red.

Covers the pure half of the page's review feature (`isCleanDiagnostic`, `reviewFeedback`, the
notices, `patchPaths`, the check after an edit with its extension set and its reply, the scope
of `/review`), the dispatcher's four commands, the self-review machine in `finish_round`, the
check in `ToolDone` and the two switches with their store.
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
    # ---- the self-review pair of codeprompts.js
    (
        # the first arm is subsumed by this one (`No problems found` starts with `No `), so the
        # mutation has to go against the arm every fixture case really runs through
        "review.rs",
        '        || o.starts_with("No ")',
        "        || o.starts_with(\"No x\")",
        "the clean shapes of the fixtures",
    ),
    (
        "review.rs",
        "    if round > 1 && !objective {",
        "    if round > 2 && !objective {",
        "round two stops for a clean report",
    ),
    (
        "review.rs",
        '    let objective = !problems.is_empty() || review.contains("[high]");',
        "    let objective = !problems.is_empty();",
        "a review that says [high] is a problem itself",
    ),
    (
        "review.rs",
        ".map(|p| format!(\"- {}: {}\", p.path, take_chars(p.output.trim(), 1500)))",
        ".map(|p| format!(\"- {}: {}\", p.path, take_chars(p.output.trim(), 1499)))",
        "the diagnostics are cut at 1500",
    ),
    (
        "review.rs",
        '        "Reviewing my own changes ({files} file{}) before finishing.",',
        '        "Reviewing my own changes ({files} file{}s) before finishing.",',
        "the two notices",
    ),
    (
        "review.rs",
        '        "The review found problems: another round ({problems} file{} with diagnostics).",',
        '        "The review found problems: another round ({problems} file{} problems).",',
        "the two notices",
    ),
    (
        "review.rs",
        '        "Review before done: on. When a run changes files the agent reads its own diff and the diagnostics once more and fixes what is wrong."',
        '        "Review before done: on. The agent looks at its own work again."',
        "the notice of /autoreview",
    ),
    # ---- patchPaths and the check after an edit
    (
        "review.rs",
        '        let p = if b == "/dev/null" { a } else { b };',
        "        let p = b;",
        "a deleted file keeps its a-side name",
    ),
    (
        "review.rs",
        "    for p in paths.iter().take(4) {",
        "    for p in paths.iter().take(5) {",
        "the check looks at four paths",
    ),
    (
        "review.rs",
        "        if !CHECKED_EXTENSIONS.contains(&ext.as_str()) || seen.contains(&ext) {",
        "        if !CHECKED_EXTENSIONS.contains(&ext.as_str()) {",
        "one report per extension",
    ),
    (
        "review.rs",
        "        if denied(p) {\n            continue;\n        }",
        "        if denied(p) && false {\n            continue;\n        }",
        "the rule of `bash: diagnostics <path>` skips the file",
    ),
    (
        "review.rs",
        '    !output.starts_with("No problems found") && !output.contains("is not available here") && !output.contains("did not finish within")',
        '    !output.starts_with("No problems found") && !output.contains("is not available here")',
        "the answer that says nothing could be checked",
    ),
    (
        "review.rs",
        "    let max = 3000f64.max(cap) as usize;",
        "    let max = cap as usize;",
        "the clip never sinks below 3000",
    ),
    (
        "review.rs",
        '\\nFix them if your change caused them.',
        '\\nFix what the check found.', 
        "the reply template",
    ),
    (
        "review.rs",
        "    let base = if scope == \"branch\" {",
        "    let base = if false {",
        "a branch review carries its base",
    ),
    # ---- the four commands of row 81
    (
        "commands.rs",
        "            let on = want.unwrap_or(state.auto_review);",
        "            let on = false;",
        "the notice reads the switch without an argument",
    ),
    (
        "commands.rs",
        '            name: "diagnostics".into(),',
        '            name: "diagnostics_check".into(),',
        "/check runs the diagnostics tool",
    ),
    # ---- the machine in code.rs
    (
        "code.rs",
        "            if self.main_run && self.auto_review && self.mode != Mode::Plan {",
        "            if self.main_run && self.auto_review {",
        "plan mode ends without a review",
    ),
    (
        "code.rs",
        "                    ReviewStage::Run if rv.round < 2 => {",
        "                    ReviewStage::Run if rv.round < 3 => {",
        "the second round is the last",
    ),
    (
        "code.rs",
        "            if c.id > mark && !files.contains(&c.path) {",
        "            if c.id >= mark && !files.contains(&c.path) {",
        "only what the run changed is reviewed",
    ),
    (
        "code.rs",
        "                        .filter(|(_, output)| !crate::review::is_clean_diagnostic(Some(output)))",
        "                        .filter(|_| true)",
        "a clean diagnostics answer stays out of the feedback",
    ),
    (
        "code.rs",
        "                        c.sent = Some(crate::review::auto_check_reply(&reply, &problems, env.ctx_max));",
                        "                        c.sent = None;",
        "the model gets the findings with its answer",
    ),
    # ---- the two switches with the store
    (
        "code.rs",
        "                    if changed {\n                        self.pending_auto_review = Some(self.auto_review);\n                    }",
        "                    if false {\n                        self.pending_auto_review = Some(self.auto_review);\n                    }",
        "the window keeps the switch",
    ),
    (
        "app.rs",
        "        app.code.auto_review = app.store.prefs.auto_review;",
        "        app.code.auto_review = false;",
        "the switch comes back from the store",
    ),
]

TESTS = [
    ["cargo", "test", "--locked", "--test", "review_fixtures"],
    ["cargo", "test", "--locked", "--test", "review_window"],
    ["cargo", "test", "--locked", "--test", "commands_fixtures"],
    ["cargo", "test", "--locked", "--lib", "--", "review::", "code::tests"],
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
