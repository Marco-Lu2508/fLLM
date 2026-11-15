#!/usr/bin/env python3
"""One word at a time: the tests of row 88 must go red.

Covers the pure layer of the learning loop (what is read from a run, the toast, the poll, the
requests), the Code page (the switch, the hand-over, the thumbs, the watch, `/learn`) and the
three cards of Settings > Personalization. A run is red only when a test reports `FAILED`; one that
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
    ('learn.rs', '.and_then(Value::as_str) != Some("skill") {', '.and_then(Value::as_str) != Some("skills") {', 'only the skill tool counts'),
    ('learn.rs', 'args.get("action").and_then(Value::as_str) != Some("read")', 'args.get("action").and_then(Value::as_str) != Some("reads")', 'only a read counts'),
    ('learn.rs', 'if crate::skills::valid_name(&name) && !out.contains(&name) {', 'if !out.contains(&name) {', 'a skill name is valid'),
    ('learn.rs', 'if crate::skills::valid_name(&name) && !out.contains(&name) {', 'if crate::skills::valid_name(&name) {', 'each skill once'),
    ('learn.rs', 'let start = list.len().saturating_sub(max_messages);', 'let start = 0;', 'the last three hundred'),
    ('learn.rs', '.skip(from)\n        .filter(|m| matches!(m.role.as_str(), "user" | "assistant" | "tool"))', '.skip(0)\n        .filter(|m| matches!(m.role.as_str(), "user" | "assistant" | "tool"))', 'the run begins at from'),
    ('learn.rs', 'matches!(m.role.as_str(), "user" | "assistant" | "tool")', 'matches!(m.role.as_str(), "user" | "assistant")', 'tool messages are kept'),
    ('learn.rs', 'json!({ "role": m.role, "content": slice_units(&m.content, max_chars) })', 'json!({ "role": m.role, "content": m.content })', 'the text is cut at three thousand'),
    ('learn.rs', 'Some(Value::String(s)) => slice_units(s, 800),', 'Some(Value::String(s)) => s.clone(),', 'the arguments are cut at eight hundred'),
    ('learn.rs', 'error || hit_limit', 'error && hit_limit', 'a run failed by error or by limit'),
    ('learn.rs', 'format!("Learned: {}", l.summary)', 'format!("Learned {}", l.summary)', 'the toast of a learned step'),
    ('learn.rs', 'status.is_some_and(|o| !o.running &&', 'status.is_some_and(|o| o.running &&', 'the step is over'),
    ('learn.rs', 'l.t >= since', 'l.t > since', 'the step began after the run'),
    ('learn.rs', 'pub const MAX_POLLS: u32 = 60;', 'pub const MAX_POLLS: u32 = 61;', 'after sixty polls the page gives up'),
    ('learn.rs', 'format!("Noted: {} worked.", skills.join(", "))', 'format!("Noted: {} worked", skills.join(", "))', 'a thumb up is noted'),
    ('learn.rs', 'if notes.is_empty() { String::new() } else { format!("; {}", notes.join(" ")) }', 'String::new()', 'a thumb down carries the notes'),
    ('learn.rs', '"session": if session.is_empty() { "studio" } else { session },', '"session": session,', 'a conversation without a name is called studio'),
    ('learn.rs', '"outcome": if failed { "failed" } else { "done" },', '"outcome": if failed { "done" } else { "failed" },', 'the outcome of a run'),
    ('learn.rs', 'json!({ "value": value, "skills": last.skills, "session": last.session, "workspace": last.workspace })', 'json!({ "value": value, "skills": last.skills, "session": last.session })', 'the body of a thumb'),
    ('learn.rs', 'let one = n == 1.0;', 'let one = n == 2.0;', 'one proposal waits'),
    ('learn.rs', 'if st.ask_first {\n                t.push_str("It proposes and waits for your yes.");', 'if !st.ask_first {\n                t.push_str("It proposes and waits for your yes.");', 'the notice of /learn says when it asks first'),
    ('learn.rs', '"on" => LearnPlan::Show(Some(true)),', '"on" => LearnPlan::Show(Some(false)),', '/learn on'),
    ('learn.rs', 'if word == "project" {', 'if word == "projects" {', '/learn project'),
    ('learn.rs', 'None => "follows the global setting",', 'None => "off",', 'the folder follows the global setting'),
    ('learn.rs', '"enabled": on, "workspace": workspace, "project": Value::Null', '"enabled": on, "workspace": workspace', "the global switch clears the folder's own setting"),
    ('learn.rs', '    if !st.notice_pending {\n        return String::new();\n    }', '    if st.notice_pending {\n        return String::new();\n    }', 'the notice is taken once'),
    ('learn.rs', 'Some(json!({ "notice": true }))', 'Some(json!({ "notice": false }))', 'the server is told the notice was shown'),
    ('code.rs', 'if st.enabled != Some(self.learn) {\n                    self.learn = st.enabled != Some(false);', 'if st.enabled != Some(true) {\n                    self.learn = st.enabled != Some(false);', 'the switch of the server is the truth'),
    ('code.rs', 'self.learn = st.enabled != Some(false);\n                    self.pending_learn', 'self.learn = st.enabled == Some(true);\n                    self.pending_learn', 'a server that does not say is on'),
    ('code.rs', 'if st.enabled == Some(false) {\n                    return Task::none();\n                }\n                let root', 'if st.enabled == Some(true) {\n                    return Task::none();\n                }\n                let root', 'the notice is asked for only while learning is on'),
    ('code.rs', 'CodeMsg::LearnToggle(on) => {\n                self.learn = on;', 'CodeMsg::LearnToggle(on) => {\n                self.learn = !on;', 'the checkbox sets the switch'),
    ('code.rs', '        if !self.learn {\n            return Task::none();\n        }\n        let wire = self.wire(None);', '        let wire = self.wire(None);', 'nothing is handed over while learning is off'),
    ('code.rs', 'self.learn_from = Some(self.wire(None).len().saturating_sub(1));', 'self.learn_from = Some(0);', 'a second run hands over only its own skills'),
    ('code.rs', '                        // a run that ended in an error is still learned from, as a failed one\n                        self.learn_after_run(env, true)', '                        Task::none()', 'a failed run is learned from too'),
    ('code.rs', '        let learn = self.learn_after_run(env, false);\n        Task::batch([', '        let learn = Task::none();\n        Task::batch([', 'a run is handed over'),
    ('code.rs', 'if tries < crate::learn::MAX_POLLS {', 'if tries <= crate::learn::MAX_POLLS {', 'after sixty polls the page gives up'),
    ('code.rs', 'let Some(last) = self.learn_last.clone().filter(|l| !l.skills.is_empty()) else {', 'let Some(last) = self.learn_last.clone() else {', 'the thumbs need a skill of the last run'),
    ('code.rs', '                        if let Some(st) = &status {\n                            self.learn = st.enabled == Some(true);', '                        if let Some(st) = &status {\n                            self.learn = true;', '/learn project reads the answer'),
    ('code.rs', '                if let Some(on) = set {\n                    self.learn = on;\n                    self.pending_learn = Some(on);\n                }\n                Task::perform(', '                if let Some(on) = set {\n                    self.learn = !on;\n                    self.pending_learn = Some(on);\n                }\n                Task::perform(', '/learn off changes the switch at once'),
    ('code.rs', '                    self.learn_refresh = true;\n                    return Task::none();\n                }\n                if tries <', '                    return Task::none();\n                }\n                if tries <', 'the lists are read again when the step ended'),
    ('code.rs', '            CodeMsg::LearnSwitched(Err(e)) => {\n                self.pending_toast = Some(e);', '            CodeMsg::LearnSwitched(Err(e)) => {\n                let _ = e;', 'a refusal is a toast'),
    ('code.rs', '.on_toggle(|value| Message::Code(CodeMsg::LearnToggle(value))),', '.on_toggle(|value| Message::Code(CodeMsg::LearnToggle(!value))),', 'the side has the switch'),
    ('code.rs', 'CodeMsg::LearnThumb("down"))).padding', 'CodeMsg::LearnThumb("up"))).padding', 'the thumbs of the side'),
    ('learncards.rs', 'LpMsg::Save => {\n                let text = self.editor.text();\n                post("/api/studio/learn/profile", json!({ "action": "save"', 'LpMsg::Save => {\n                let text = self.editor.text();\n                post("/api/studio/learn/profile", json!({ "action": "prompt"', 'every button asks the server'),
    ('learncards.rs', 'json!({ "action": "forget", "all_learned": true, "text": "", "workspace": workspace })', 'json!({ "action": "forget", "all_learned": false, "text": "", "workspace": workspace })', 'forget everything'),
    ('learncards.rs', 'let action = if input.trim().is_empty() { "delete" } else { "edit" };', 'let action = if input.is_empty() { "delete" } else { "edit" };', 'an empty text deletes the entry'),
    ('learncards.rs', 'if let Some(st) = self.status.as_mut() {\n                    st.ask_first = *on;\n                }', '', 'the switch that asks first'),
    ('learncards.rs', '(Err(e), _) => *toast = Some(e.clone()),', '(Err(_), _) => {}', 'an answer is a toast'),
    ('learncards.rs', 'let place = if scope == "global" { "every project" } else { "this project" };\n    match &entry.source', 'let place = if scope == "global" { "this project" } else { "every project" };\n    match &entry.source', 'the lines of the cards'),
    ('learncards.rs', 'p.section.replace(\'_\', " ")', 'p.section.clone()', 'the section of a profile line'),
    ('learncards.rs', 'text.encode_utf16().count()', 'text.chars().count()', 'the page counts UTF-16 units'),
    ('learncards.rs', 'if on { "Learning is on for the open project (its .fllm/learn.json)." } else { "Learning is off for the open project (its .fllm/learn.json)." }', 'if on { "Learning is off for the open project (its .fllm/learn.json)." } else { "Learning is on for the open project (its .fllm/learn.json)." }', 'the project note'),
    ('learncards.rs', 'if remove { "Forget it" } else { "Remember" }', 'if remove { "Remember" } else { "Forget it" }', 'a removal is forgotten'),
]

TEST = ["cargo", "test", "--locked", "--no-fail-fast", "--lib", "--test", "learn_fixtures", "--test", "learn_window", "--test", "learn_server", "--test", "learncards_window", "--test", "commands_fixtures"]


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
