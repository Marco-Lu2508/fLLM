#!/usr/bin/env python3
"""One word at a time: the tests of row 80 must go red.

Covers the pure layer of the specs (the task picker and the `Number` of the page, the notices,
the bodies of the actions), the dispatch of `/spec`, the answers of the window (listing, refusal,
reload, tick, run) and the Specs block of the side. A run is red only when a test reports
`FAILED`; one that does not compile or reports nothing is broken and its tail is printed.
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # desktop/native
SRC = ROOT / "src"
# the whole environment is kept: rustup finds its toolchain only over RUSTUP_HOME/CARGO_HOME
CARGO_ENV = dict(os.environ)
CARGO_ENV["PATH"] = "/opt/data/home/.cargo/bin:" + CARGO_ENV.get("PATH", "")
CARGO_ENV["CARGO_TARGET_DIR"] = "/opt/data/cache/scratch/fllm-app-target"
CARGO_ENV["FLLM_SNAPSHOT_DIR"] = str(ROOT / "ui-snapshots")

# (file, old text, new text, which tests must go red)
MUTATIONS = [
    # ---- the picker and the Number of the page
    ("specs.rs", "if let Some(n) = asked {", "if let Some(n) = asked.filter(|_| false) {", "the picker takes the number asked for"),
    ("specs.rs", "spec.tasks.iter().find(|t| !t.done).map(|t| t.index as f64)", "spec.tasks.iter().find(|t| t.done).map(|t| t.index as f64)", "the first open task"),
    ("specs.rs", "word.map(js_number).filter(|n| *n != 0.0 && !n.is_nan())", "word.map(js_number).filter(|n| !n.is_nan())", "zero is falsy"),
    ("specs.rs", "word.map(js_number).filter(|n| *n != 0.0 && !n.is_nan())", "word.map(js_number).filter(|n| *n != 0.0)", "NaN is falsy"),
    ("specs.rs", 'if rest == "Infinity" {', "if false {", "Number('Infinity')"),
    ("specs.rs", '[("0x", 16), ("0o", 8), ("0b", 2)]', '[("0x", 16), ("0o", 8)]', "Number('0b11')"),
    ("specs.rs", "if !rest.starts_with(|c: char| c.is_ascii_digit() || c == '.') || ", "if ", "Number('--1') is NaN"),
    ("specs.rs", "} else if index.fract() == 0.0 && index.abs() < 9e15 {", "} else if false {", "a whole number is an integer in JSON"),
    ("specs.rs", '"Infinity".into() } else { "-Infinity".into() }', '"inf".into() } else { "-inf".into() }', "Infinity in a template literal"),
    ("specs.rs", "serde_json::json!({ \"action\": \"task_prompt\", \"name\": name, \"index\": index_json(index) })", "serde_json::json!({ \"action\": \"task_prompt\", \"name\": name, \"index\": index })", "the body of task_prompt writes a whole number as an integer"),
    ("specs.rs", 'serde_json::json!({ "action": "new_prompt", "name": name, "description": description })', 'serde_json::json!({ "action": "new_prompt", "name": name, "description": name })', "the body of new_prompt"),
    ("specs.rs", '        "tick"\n    } else {\n        "untick"', '        "untick"\n    } else {\n        "tick"', "the action of a box"),
    ("specs.rs", 'format!("/spec run {name} {}", index_text(index))', 'format!("/spec run {name}")', "the display of a run"),
    ("specs.rs", "(depth * 12) as f32", "(depth * 24) as f32", "twelve pixels a level"),
    ("specs.rs", 'format!("{file} (not written yet)")', 'format!("{file}")', "the title of a missing file"),
    # ---- what the page decides and says
    ("specs.rs", "let Some(sp) = current.iter().find(|s| s.name == name) else {\n        return RunDecision::NeedReload;", "let Some(sp) = current.iter().find(|s| s.name == name) else {\n        return RunDecision::NotFound;", "a missing name is read for first"),
    ("specs.rs", "None => RunDecision::AllDone,", "None => RunDecision::NotFound,", "a spec with nothing left"),
    ("specs.rs", 'format!("{}: {}/{} tasks done", s.name, s.done, s.total)', 'format!("{}: {}/{}", s.name, s.done, s.total)', "a line of the listing"),
    ("specs.rs", 'format!("There is no spec \\"{name}\\". /spec lists them.")', 'format!("There is no spec \\"{name}\\".")', "the notice of a missing spec"),
    ("specs.rs", 'pub const ALL_DONE_NOTICE: &str = "Every task of this spec is done.";', 'pub const ALL_DONE_NOTICE: &str = "Every task is done.";', "the notice of a finished spec"),
    ("specs.rs", 'Usage: /spec NAME what you want. Example: /spec csv-export Export the report as CSV.', 'Usage: /spec NAME what you want.', "the usage line"),
    ("specs.rs", "Plan a feature before coding it: /spec name", "Plan a feature: /spec name", "the pitch"),
    ("specs.rs", 'Err(e) => Err(if e.starts_with("the server answered") {', "Err(e) => Err(if false {", "the page's words for a refusal without a message"),
    # ---- the dispatch of /spec
    ("commands.rs", "let asked = crate::specs::asked_task(words.get(2).copied());", "let asked = crate::specs::asked_task(words.get(3).copied());", "the word that names the task"),
    ("commands.rs", 'words[0] == "run" {', 'words[0] == "runs" {', "the run branch"),
    ("commands.rs", 'let description = words[1..].join(" ");', 'let description = words[1..].join("");', "the words of a description"),
    ("commands.rs", 'display: format!("/spec {}", arg_text) }', "display: String::new() }", "the display of a new spec"),
    ("commands.rs", "vec![Step::Notice(crate::specs::SPEC_USAGE.to_string())]", "vec![Step::SpecList]", "the usage without a description"),
    # ---- what the window does with the answers
    ("code.rs", "CodeMsg::SpecPrompt { prompt, display } => self.begin_run(prompt, display, env),", "CodeMsg::SpecPrompt { prompt, display } => self.run_prompt(&prompt, &display, env),", "a spec's run fires the submit hook"),
    ("code.rs", "Step::AgentRun(prompt, display) => tasks.push(self.begin_run(prompt, display, env)),", "Step::AgentRun(prompt, display) => tasks.push(self.run_prompt(&prompt, &display, env)),", "`/init` fires the submit hook"),
    ("code.rs", 'self.begin_run(task, "/batch".to_string(), env)', 'self.run_prompt(&task, "/batch", env)', "`/batch` fires the submit hook"),
    ("code.rs", "if self.workspace.is_none() || env.model.is_none() || self.running {\n            return Task::none();\n        }\n        self.change_mark", "if self.workspace.is_none() || env.model.is_none() {\n            return Task::none();\n        }\n        self.change_mark", "a run cannot start beside a run"),
    ("code.rs", "        self.running = false;\n        self.handle = None;\n        Task::perform(\n            crate::specs::fetch_specs(env.base.to_string()),\n            |r| Message::Code(CodeMsg::SpecsLoaded(r.unwrap_or_default())),\n        )", "        self.running = false;\n        self.handle = None;\n        Task::none()", "the end of a run reads the specs"),
    ("code.rs", "CodeMsg::SpecsLoaded(specs) => {\n                self.specs = specs;", "CodeMsg::SpecsLoaded(specs) => {\n                let _ = specs;", "a loaded list is kept"),
    ("code.rs", "self.specs = specs.clone();\n                self.items.push", "self.items.push", "the listing refreshes the side"),
    ("code.rs", "self.items.push(Item::Note(crate::specs::listing_notice(&specs)));", "let _ = &specs;", "the listing is written"),
    ("code.rs", "CodeMsg::SpecReplaced(specs) => {\n                self.specs = specs;", "CodeMsg::SpecReplaced(specs) => {\n                let _ = specs;", "a tick's answer replaces the list"),
    ("code.rs", "self.pending_toast = Some(message);\n                    Task::perform(crate::specs::fetch_specs", "let _ = message;\n                    Task::perform(crate::specs::fetch_specs", "a refused tick toasts"),
    ("code.rs", "self.items.push(Item::Note(message));\n                    Task::none()\n                }\n            }\n            CodeMsg::SpecTick", "Task::none()\n                }\n            }\n            CodeMsg::SpecTick", "a refusal is a note"),
    ("code.rs", "Task::perform(crate::specs::fetch_specs(base), |r| Message::Code(CodeMsg::SpecsLoaded(r.unwrap_or_default()))),\n                ])", "Task::none(),\n                ])", "opening a folder reads the specs"),
    ("code.rs", "RunDecision::NotFound => Err(no_spec_notice(&name)),", "RunDecision::NotFound => Err(String::new()),", "the notice of a missing spec reaches the list"),
    ("code.rs", "RunDecision::AllDone => Err(ALL_DONE_NOTICE.to_string()),", "RunDecision::AllDone => Ok((String::new(), String::new())),", "a finished spec says so"),
    ("code.rs", "None => RunDecision::NotFound,\n        },\n        other => other,", "None => RunDecision::NeedReload,\n        },\n        other => other,", "a failed reload is a list of nothing"),
    # ---- the side
    ("code.rs", ".on_press_maybe((!s.running).then_some(Message::Code(CodeMsg::SpecRunTask { name, index })))", ".on_press_maybe(Some(Message::Code(CodeMsg::SpecRunTask { name, index })))", "the button waits for a run"),
    ("code.rs", "if let Some(next) = sp.tasks.iter().find(|task| !task.done) {", "if let Some(next) = sp.tasks.iter().find(|task| task.done) {", "the button is for the first open task"),
    ("code.rs", "} else if sp.total > 0 {", "} else if true {", "a spec without tasks is not done"),
    ("code.rs", "if !s.specs.is_empty() {\n        head = head.push", "if true {\n        head = head.push", "the count beside the heading"),
    ("code.rs", "if s.specs.is_empty() {\n        specs = specs.push(text(crate::specs::PITCH)", "if false {\n        specs = specs.push(text(crate::specs::PITCH)", "the pitch of an empty folder"),
    ("code.rs", "CodeMsg::SpecTick { name: name.clone(), index, done }", "CodeMsg::SpecTick { name: name.clone(), index, done: true }", "the box sends its new state"),
    ("code.rs", "let tick = checkbox(task.done)", "let tick = checkbox(false)", "a done task is drawn ticked"),
]

TEST = ["cargo", "test", "--locked", "--no-fail-fast", "--lib", "--test", "specs_window", "--test", "specs_server", "--test", "commands_fixtures"]


def run() -> tuple[bool, list[str], str]:
    out = subprocess.run(TEST, cwd=ROOT, env=CARGO_ENV, capture_output=True, text=True)
    text = out.stdout + out.stderr
    failed = [line.split()[1] for line in text.splitlines() if line.startswith("test ") and line.endswith("FAILED")]
    ran = "test result:" in text
    return (ran and bool(failed)) or ("test result: FAILED" in text), failed, text


if len(sys.argv) > 1 and sys.argv[1] == "--check":
    bad = 0
    for file, old, new, _ in MUTATIONS:
        count = (SRC / file).read_text().count(old)
        if count != 1:
            bad += 1
            print(f"{file}: {count} matches for {old[:70]!r}")
    print("anchors ok" if not bad else f"{bad} bad anchors")
    sys.exit(1 if bad else 0)

# `ONLY=words python3 mutate_specs.py` runs the entries whose test words contain `words`
ONLY = os.environ.get("ONLY")
if ONLY:
    MUTATIONS[:] = [m for m in MUTATIONS if ONLY in m[3]]

original: dict[str, str] = {}
red: list[str] = []
problems: list[str] = []
try:
    for file, old, new, must in MUTATIONS:
        path = SRC / file
        text = path.read_text()
        count = text.count(old)
        if count != 1:
            problems.append(f"{file}: {count} matches for {old[:60]!r} — fix the script (check the FILE named first)")
            continue
        original[file] = text
        path.write_text(text.replace(old, new, 1))
        try:
            is_red, failed, tail = run()
        finally:
            path.write_text(original[file])
        if is_red:
            red.append(must)
            print(f"RED   {must}: {sorted(set(failed))[:3]}", flush=True)
        elif "test result:" not in tail:
            problems.append(f"BROKEN {must}")
            print(f"BROKEN {must} — it did not compile or ran nothing; tail:\n" + "\n".join(tail.splitlines()[-12:]), flush=True)
        else:
            problems.append(f"GREEN {must}")
            print(f"GREEN {must}", flush=True)
finally:
    for file, text in original.items():
        (SRC / file).write_text(text)

print(f"{len(red)}/{len(MUTATIONS)} mutations red", flush=True)
if problems:
    print("PROBLEMS:", problems, flush=True)
    sys.exit(1)
print("ALL MUTATIONS BIT", flush=True)
