#!/usr/bin/env python3
"""One word at a time: the tests of row 86 (computer use) must go red.

Covers the pure part (`computer.rs`: the lines that name an action, the questions, the marks, the pictures kept,
the panic keys, the labels) and its hooks in the Code page (`code.rs`: the poll, the stop, `computerBegin`, the banner). A run is red only when a test reports `FAILED`; one that
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

MUTATIONS = [
    ('computer.rs', 'if n.is_finite() { round(n * 10.0) / 10.0 } else { 0.0 }', 'if n.is_finite() { round(n) } else { 0.0 }', 'a number is rounded to a tenth'),
    ('computer.rs', 'if n.is_finite() { round(n * 10.0) / 10.0 } else { 0.0 }', 'if n.is_finite() { round(n * 10.0) / 10.0 } else { n }', 'a number that is not finite is zero'),
    ('computer.rs', '(x + 0.5).floor()', 'x.round()', 'halves go up'),
    ('computer.rs', '_ => "?".into(),\n    }\n}\n\n/// JavaScript truthiness', '_ => "-".into(),\n    }\n}\n\n/// JavaScript truthiness', 'a point that is no object is a question mark'),
    ('computer.rs', 'format!(" {}", js_string(b))', 'format!("{}", js_string(b))', 'the button stands after the point'),
    ('computer.rs', 'if truthy(a("double")) { " double" } else { "" }', 'if truthy(a("double")) { " dbl" } else { "" }', 'the subject of a double click'),
    ('computer.rs', '"{name} {} -> {}"', '"{name} {} => {}"', 'the subject of a drag'),
    ('computer.rs', '"computer_type" => format!("{name} {}", serde_json::to_string(&text_or_empty(a("text"))).unwrap_or_default()),', '"computer_type" => format!("{name} {}", text_or_empty(a("text"))),', 'the subject of a typed text is quoted'),
    ('computer.rs', 'None | Some(Value::Null) => 1000.0,', 'None | Some(Value::Null) => 500.0,', 'a wait without time is a second'),
    ('computer.rs', '"Let the agent look at your screen?"', '"Let the agent look at the screen?"', 'the question of a screenshot'),
    ('computer.rs', '"Double-click".to_string()', '"Double click".to_string()', 'the question of a double click'),
    ('computer.rs', 'format!("{}-click", js_string(b))', 'format!("{} click", js_string(b))', 'the question of a right click'),
    ('computer.rs', 'let shown = serde_json::to_string(&slice_units(&text, 80))', 'let shown = serde_json::to_string(&slice_units(&text, 70))', 'the question shows 80 units of a text'),
    ('computer.rs', 'if text.encode_utf16().count() > 80 { " ..." }', 'if text.encode_utf16().count() > 81 { " ..." }', 'a longer text ends in dots'),
    ('computer.rs', '"computer_drag" => "Drag with the mouse along the marked line?".into(),', '"computer_drag" => "Drag with the mouse?".into(),', 'the question of a drag'),
    ('computer.rs', 'if (name == "computer_click" || name == "computer_move") && ok {', 'if name == "computer_click" && ok {', 'a move is marked'),
    ('computer.rs', 'return vec![Mark::Cross { x, y, double: None }, Mark::Arrow', 'return vec![Mark::Arrow', 'a scroll is a cross and an arrow'),
    ('computer.rs', '.iter().all(|n| n.is_finite())', '.iter().any(|n| n.is_finite())', 'a drag needs four numbers'),
    ('computer.rs', 'let s = 40f64.min(14f64.max(big / 50.0));', 'let s = 40f64.min(14f64.max(big / 40.0));', 'the size of a mark follows the screen'),
    ('computer.rs', 'let w = 2f64.max(round(big / 450.0));', 'let w = 2f64.max(round(big / 400.0));', 'the width of a mark follows the screen'),
    ('computer.rs', 'let r = s * 0.55;', 'let r = s * 0.5;', 'the ring of a cross'),
    ('computer.rs', 'f(r * 1.7)', 'f(r * 1.5)', 'the second ring of a double click'),
    ('computer.rs', 'layer("#fff", 3.0)', 'layer("#fff", 2.0)', 'white underneath'),
    ('computer.rs', 'layer("#e5231b", 0.0)', 'layer("#e5231c", 0.0)', 'red on top'),
    ('computer.rs', '|| width <= 0.0 || height <= 0.0 {', '|| width <= 0.0 {', 'a screen without a height has no marks'),
    ('computer.rs', 'let ex = x + (dx / len) * s * 3.0;', 'let ex = x + (dx / len) * s * 2.0;', 'the length of the arrow'),
    ('computer.rs', 'carrying.len().saturating_sub(keep)', 'carrying.len().saturating_sub(keep + 1)', 'the last pictures are kept'),
    ('computer.rs', '        messages[i].images.clear();\n        messages[i].content', '        messages[i].content', 'a dropped picture is gone'),
    ('computer.rs', 'if m.images.len() == 1 { "a screenshot was"', 'if m.images.len() == 2 { "a screenshot was"', 'one screenshot is said in the singular'),
    ('computer.rs', '"type": "image_url"', '"type": "image"', 'a picture goes to the model as an image part'),
    ('computer.rs', 'now - t <= window_ms', 'now - t < window_ms', 'a press at the edge of the time counts'),
    ('computer.rs', 'if recent.len() >= needed {', 'if recent.len() > needed {', 'the third press stops'),
    ('computer.rs', 'escape_streak(times, now, 1500.0, 3)', 'escape_streak(times, now, 1400.0, 3)', 'three presses within a second and a half'),
    ('computer.rs', 'key == "Escape" && ctrl && alt && shift', 'key == "Escape" && ctrl && alt', 'the panic chord has Shift'),
    ('computer.rs', '"computer_screenshot" | "computer_cursor" | "computer_displays" | "computer_wait"', '"computer_screenshot" | "computer_cursor" | "computer_displays"', 'a wait only looks'),
    ('computer.rs', 't("a part of the screen")', 't("part of the screen")', 'the label of a region screenshot'),
    ('computer.rs', '(if truthy(a("double")) { t("Double-click") }', '(if truthy(a("double")) { t("Double click") }', 'the label of a double click'),
    ('computer.rs', '"{}, {} to {}, {}"', '"{}, {} -> {}, {}"', 'the label of a drag'),
    ('computer.rs', 'slice_units(&text_or_empty(a("text")), 60)', 'slice_units(&text_or_empty(a("text")), 50)', 'the label of a typed text is clipped at 60'),
    ('computer.rs', 'format!("{} right ", raw("dx"))', 'format!("{} left ", raw("dx"))', 'the label of a sideways scroll'),
    ('computer.rs', 'format!("{} ms", match a("ms")', 'format!("{}ms", match a("ms")', 'the label of a wait'),
    ('computer.rs', '"computer_cursor" => (t("Pointer position")', '"computer_cursor" => (t("Pointer")', 'the label of the cursor tool'),
    ('code.rs', 'if was && !now && !self.computer.stopping {', 'if was && !now {', 'a stop you pressed is not a surprise'),
    ('code.rs', 'if !now {\n                    self.computer.busy.clear();', 'if now {\n                    self.computer.busy.clear();', 'switched off, nothing is going on'),
    ('code.rs', 'computer_poll(env.base.to_string(), epoch, if now { 1500 } else { 15000 })', 'computer_poll(env.base.to_string(), epoch, if now { 15000 } else { 1500 })', 'a poll every 1.5 seconds while it is on'),
    ('code.rs', '                self.computer.last_shot = None;\n                self.computer.busy.clear();\n                self.items.push(Item::Note(format!("{why}', '                self.computer.busy.clear();\n                self.items.push(Item::Note(format!("{why}', 'a stop forgets the screenshot'),
    ('code.rs', 'self.pending_toast = Some("Computer use stopped and switched off".into());', 'let _ = 0;', 'a stop is a toast'),
    ('code.rs', 'if !std::mem::take(&mut self.beginning) {\n                    return Task::none();', 'if false {\n                    return Task::none();', 'a stop in the meantime takes the run back'),
    ('code.rs', 'if !self.computer_on() {\n            return self.run_prompt(&prompt, &display, env);', 'if self.computer_on() {\n            return self.run_prompt(&prompt, &display, env);', 'a run waits for computerBegin only when it is on'),
    ('code.rs', '        self.beginning = true;\n        let base = env.base.to_string();', '        let base = env.base.to_string();', 'nothing else starts a run while it begins'),
    ('code.rs', '|| self.running || self.beginning {\n            return Task::none();', '|| self.running {\n            return Task::none();', 'a second message does not start a second run'),
    ('code.rs', 'if !self.computer_on() {\n            return false;\n        }\n        let (times, fire)', 'let (times, fire)', 'Esc counts only while it is on'),
    ('code.rs', 'if waiting { "Waiting for your answer:" } else { "Operating your computer:" }', 'if waiting { "Operating your computer:" } else { "Waiting for your answer:" }', 'the banner says what the agent does'),
    ('code.rs', 'if crate::computer::is_look(name) {\n            return Ok(None);', 'if false {\n            return Ok(None);', 'a look needs no earlier screenshot'),
    ('code.rs', '"Take a screenshot first (computer_screenshot): coordinates refer to the last screenshot you took."', '"Take a screenshot first."', 'an action before the first screenshot is an error'),
    ('code.rs', 'crate::computer::drop_old_images(&mut msgs, 2);', 'crate::computer::drop_old_images(&mut msgs, 3);', 'the last two screenshots are kept'),
    ('code.rs', 'format!(" {} of {} actions used in this task.", st.steps_used, st.max_steps)', 'format!(" {} of {} actions used.", st.steps_used, st.max_steps)', 'the banner counts the actions'),
    ('code.rs', 'if !s.computer_on() {\n        return None;\n    }\n    let bold', 'let bold', 'no banner while it is off'),
    ('code.rs', 'CodeMsg::ComputerStop("You pressed STOP.".into())', 'CodeMsg::ComputerStop("You pressed stop.".into())', 'the words of the stop button'),
    ('code.rs', 'let marks = if cu.image.is_some() { String::new() }', 'let marks = if cu.image.is_none() { String::new() }', 'the marks are on the screenshot an action refers to'),
]

TEST = ["cargo", "test", "--locked", "--no-fail-fast", "--lib", "--test", "computer_fixtures", "--test", "computer_window", "--test", "computer_begin"]


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

# `ONLY=words python3 mutate_x.py` runs the entries whose test words contain `words` (`a|b` for several)
ONLY = os.environ.get("ONLY")
if ONLY:
    MUTATIONS[:] = [m for m in MUTATIONS if any(o in m[3] for o in ONLY.split("|"))]

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
