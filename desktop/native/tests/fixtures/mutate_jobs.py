#!/usr/bin/env python3
"""One word at a time: the tests of row 87 must go red.

Covers the logic of the background jobs (the id read from an answer, what a finished call changes, the
buttons) and the Code page that holds them. A run is red only when a test reports `FAILED`; one that
does not compile or reports nothing is broken and its tail is printed.
"""
import os
import pathlib
import signal
import subprocess
import sys

# a kill must not leave a mutation in the source: the `finally` below puts every file back
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

ROOT = pathlib.Path(__file__).resolve().parents[2]  # desktop/native
SRC = ROOT / "src"
# the whole environment is kept: rustup finds its toolchain only over RUSTUP_HOME/CARGO_HOME
CARGO_ENV = dict(os.environ)
CARGO_ENV["PATH"] = "/opt/data/home/.cargo/bin:" + CARGO_ENV.get("PATH", "")
# no debug info and its own target directory: the build is what each mutation waits for
CARGO_ENV["CARGO_TARGET_DIR"] = "/opt/data/cache/scratch/fllm-mutation-target"
CARGO_ENV["CARGO_PROFILE_DEV_DEBUG"] = "0"
CARGO_ENV["FLLM_SNAPSHOT_DIR"] = str(ROOT / "ui-snapshots")

# (file, old text, new text, which tests must go red)
MUTATIONS = [
    ('jobs.rs', '(?:job|shell|process|task|id)(?-u:\\b)[\\s:=#"\'(]*([A-Za-z0-9_-]+)', '(?:job|shell|process|task)(?-u:\\b)[\\s:=#"\'(]*([A-Za-z0-9_-]+)', 'the word id names a job'),
    ('jobs.rs', '(bash_[A-Za-z0-9]+|[a-f0-9]{6,}|[0-9]+)', '(bash_[A-Za-z0-9]+|[a-f0-9]{6,})', 'a bare number is an id'),
    ('jobs.rs', '(bash_[A-Za-z0-9]+|[a-f0-9]{6,}|[0-9]+)', '([a-f0-9]{6,}|[0-9]+)', 'bash_ ids'),
    ('jobs.rs', '(bash_[A-Za-z0-9]+|[a-f0-9]{6,}|[0-9]+)', '(bash_[A-Za-z0-9]+|[a-f0-9]{7,}|[0-9]+)', 'six hex digits are an id'),
    ('jobs.rs', 'Regex::new(r#"(?i)(?-u:\\b)(?:job', 'Regex::new(r#"(?-u:\\b)(?:job', 'the words of an id do not care for case'),
    ('jobs.rs', '(?i)(?-u:\\b)(?:exited|finished|completed|exit code)(?-u:\\b)', '(?-u:\\b)(?:exited|finished|completed|exit code)(?-u:\\b)', 'an ended job in any case'),
    ('jobs.rs', '(?i)(?-u:\\b)(?:exited|finished|completed|exit code)(?-u:\\b)', '(?i)(?:exited|finished|completed|exit code)', 'ended words need word borders'),
    ('jobs.rs', '(?i)(?-u:\\b)(?:exited|finished|completed|exit code)(?-u:\\b)', '(?i)(?-u:\\b)(?:exited|finished|exit code)(?-u:\\b)', 'completed ends a job'),
    ('jobs.rs', 'looks = Regex::new(r"(?i)id|job|shell")', 'looks = Regex::new(r"(?i)^id$|job|shell")', 'an argument that looks like an id'),
    ('jobs.rs', 'looks = Regex::new(r"(?i)id|job|shell")', 'looks = Regex::new(r"id|job|shell")', 'the argument name in any case'),
    ('jobs.rs', 'looks = Regex::new(r"(?i)id|job|shell")', 'looks = Regex::new(r"(?i)id|job")', 'shell as the argument name'),
    ('jobs.rs', '.or_else(|| properties.first()).cloned().unwrap_or_else(|| "id".to_string())', '.cloned().unwrap_or_else(|| "id".to_string())', 'the first argument when none looks like an id'),
    ('jobs.rs', 'unwrap_or_else(|| "id".to_string())', 'unwrap_or_else(|| "job_id".to_string())', 'id when the tool is unknown'),
    ('jobs.rs', 'if name == "bash" && truthy(args.get("background")) {', 'if name == "bash" {', 'only a background command is a job'),
    ('jobs.rs', 'if name == "bash" && truthy(args.get("background")) {', 'if name == "bash" && args.get("background").and_then(Value::as_bool) == Some(true) {', 'a truthy background'),
    ('jobs.rs', 'status: "running".into(), out: String::new(), open: false', 'status: "running".into(), out: String::new(), open: true', 'a new job is closed'),
    ('jobs.rs', '} else if name == "bash_output" || name == "bash_kill" {', '} else if name == "bash_output" {', 'stop is tracked too'),
    ('jobs.rs', '        if let Some(job) = jobs.iter_mut().find(|j| j.id == id) {\n            if name == "bash_kill" {', '        if let Some(job) = jobs.last_mut() {\n            if name == "bash_kill" {', 'the job of the id'),
    ('jobs.rs', '.and_then(|o| o.values().next()).filter(|v| !v.is_null())', '.and_then(|o| o.values().next())', 'a null id is none'),
    ('jobs.rs', '                job.out = output.to_string();\n                if output_says_ended(output) {', '                if output_says_ended(output) {', 'the output is kept'),
    ('jobs.rs', '                job.status = "killed".into();\n            } else {', '                job.status = "done".into();\n            } else {', 'a stopped job is killed'),
    ('jobs.rs', '    job.open = !job.open;\n    Some(job.open)', '    job.open = true;\n    Some(job.open)', 'the output button closes the box again'),
    ('jobs.rs', '            Err(e) => job.out = e,', '            Err(_) => {}', 'a refusal is what the box shows'),
    ('jobs.rs', '                job.out = output.clone();\n                if output_says_ended(&output) {\n                    job.status = "done".into();\n                }', '                job.out = output.clone();', 'an answer that says the job ended'),
    ('jobs.rs', 'Err(e) => Some(if e.is_empty() { "The job could not be stopped.".to_string() } else { e }),', 'Err(e) => Some(e),', 'an empty refusal has a text'),
    ('jobs.rs', '        Ok(_) => {\n            if let Some(job) = jobs.iter_mut().find(|j| j.id == id) {\n                job.status = "killed".into();\n            }\n            None\n        }', '        Ok(_) => None,', 'stop marks the job'),
    ('jobs.rs', 'if job.out.is_empty() { "(no output yet)" } else { &job.out }', '&job.out', 'an empty box says so'),
    ('jobs.rs', 'Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()),', 'Some(Value::Number(_)) => true,', 'zero is no truth'),
    ('jobs.rs', 'Some(Value::String(s)) => !s.is_empty(),', 'Some(Value::String(_)) => true,', 'an empty text is no truth'),
    ('jobs.rs', 'Some(Value::Bool(true)) | Some(Value::Array(_)) | Some(Value::Object(_)) => true,', 'Some(Value::Bool(true)) => true,\n        Some(Value::Array(_)) | Some(Value::Object(_)) => false,', 'a list is true'),
    ('code.rs', '                if ok {\n                    // `trackJob(call.name, args, r.output)`', '                if !ok {\n                    // `trackJob(call.name, args, r.output)`', 'only a call that worked is tracked'),
    ('code.rs', 'crate::jobs::track_job(&mut self.jobs, &name, &value, &reply);', 'let _ = (&name, &value, &reply);', 'the code page tracks the jobs'),
    ('code.rs', '                    self.always_allow.clear();\n                    self.jobs.clear();\n                }', '                    self.always_allow.clear();\n                }', '/new forgets the jobs'),
    ('code.rs', '                self.always_allow.clear();\n                self.jobs.clear();\n                Task::none()', '                self.always_allow.clear();\n                Task::none()', 'a cleared session forgets the jobs'),
    ('code.rs', 'Task::perform(run_tool(base, "bash_output".into(), args)', 'Task::perform(run_tool(base, "bash_kill".into(), args)', 'output asks for the output'),
    ('code.rs', 'Task::perform(run_tool(base, "bash_kill".into(), args)', 'Task::perform(run_tool(base, "bash_output".into(), args)', 'stop asks to kill'),
    ('code.rs', 'let args = json!({ self.job_arg_name(): id });\n                    let base', 'let args = json!({ "id": id });\n                    let base', 'the output call names the id the way the server does'),
    ('code.rs', 'let args = json!({ self.job_arg_name(): id });\n                let base', 'let args = json!({ "id": id });\n                let base', 'the stop call names the id the way the server does'),
    ('code.rs', '.find(|d| d.pointer("/function/name").and_then(Value::as_str) == Some("bash_output"))', '.find(|d| d.pointer("/function/name").and_then(Value::as_str) == Some("bash"))', 'the argument comes from bash_output'),
    ('code.rs', 'self.pending_toast = Some(toast);\n                }\n                Task::none()\n            }\n            CodeMsg::LearnStatus(None)', 'let _ = toast;\n                }\n                Task::none()\n            }\n            CodeMsg::LearnStatus(None)', 'a refusal to stop is a toast'),
    ('code.rs', 'if running {\n            actions = actions.push(button(text("Stop")', 'if true {\n            actions = actions.push(button(text("Stop")', 'a finished job has no stop'),
    ('code.rs', '.on_press(Message::Code(CodeMsg::JobOutput(j.id.clone())))', '.on_press(Message::Code(CodeMsg::JobKill(j.id.clone())))', 'the output button'),
    ('code.rs', '.on_press(Message::Code(CodeMsg::JobKill(j.id.clone()))).padding([2, 8]).style(theme::ghost_button(t)));\n        }', '.on_press(Message::Code(CodeMsg::JobOutput(j.id.clone()))).padding([2, 8]).style(theme::ghost_button(t)));\n        }', 'the stop button'),
    ('code.rs', '        if j.open {\n            entry = entry.push(', '        if !j.open {\n            entry = entry.push(', 'an open box shows the output'),
    ('code.rs', '        block = block.push(text(crate::jobs::EMPTY)', '        block = block.push(text("")', 'the empty block says what it is for'),
    ('code.rs', 'if !s.jobs.is_empty() {\n        head = head.push(', 'if s.jobs.is_empty() {\n        head = head.push(', 'the count of jobs'),
]

TEST = ["cargo", "test", "--locked", "--no-fail-fast", "--lib", "--test", "jobs_fixtures", "--test", "jobs_window"]


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
