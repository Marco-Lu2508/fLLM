#!/usr/bin/env python3
"""One word at a time: the tests of row 85 (the browser panel) must go red.

Covers the panel's logic (`browser.rs`: ticks, the page point, wheel, keys, comments), its hooks in the Code
page and the look (`code.rs`: the panel, the agent's activity, the wide panel, Ctrl+L). A run is red only when a test reports `FAILED`; one that
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

# Left out on purpose, because the mutated code behaves the same (the guard is also in the next function): the
# modifier list of `key` (`KEY_NAMES` refuses them anyway), `Ctrl+L` with a closed panel (the handler asks for an
# open one), `!env.visible` in `BrMsg::Key` (`br_pump` asks again), and the lock icon for https (a picture).
MUTATIONS = [
    ('browser.rs', 'first || n.is_multiple_of(12)', 'first || n.is_multiple_of(11)', 'the status is read on every twelfth tick'),
    ('browser.rs', 'first || n.is_multiple_of(12)', 'n.is_multiple_of(12)', 'the status is read at the first tick'),
    ('browser.rs', 'if self.running { 650 } else { 2000 }', 'if self.running { 2000 } else { 650 }', 'a tick every 650 ms while it runs'),
    ('browser.rs', 'if self.running { 650 } else { 2000 }', 'if self.running { 650 } else { 1000 }', 'a tick every 2000 ms while it does not'),
    ('browser.rs', '/ rect.width * self.view.0, (client.1', '/ rect.width * self.view.1, (client.1', 'x of a click in the pixels of the page'),
    ('browser.rs', '(client.1 - rect.top) / rect.height * self.view.1', '(client.1 - rect.top) / rect.width * self.view.1', 'y of a click in the pixels of the page'),
    ('browser.rs', '(client.0 - rect.left) / rect.width', '(client.0) / rect.width', 'the stage does not start at the corner'),
    ('browser.rs', 'if on_button || !self.live() {\n            return None;', 'if !self.live() {\n            return None;', 'a click on a button is not a click on the page'),
    ('browser.rs', 'if on_button || !self.live() {\n            return None;', 'if on_button {\n            return None;', 'a click needs a live page'),
    ('browser.rs', 'Some(if self.commenting { Input::Locate { x, y } } else { Input::Click { x, y } })', 'Some(if !self.commenting { Input::Locate { x, y } } else { Input::Click { x, y } })', 'comment mode asks which element'),
    ('browser.rs', 'if self.commenting || !self.live() {\n            return None;\n        }\n        self.page_point(client, rect).map(|(x, y)| Input::Dblclick', 'if !self.live() {\n            return None;\n        }\n        self.page_point(client, rect).map(|(x, y)| Input::Dblclick', 'no double click in comment mode'),
    ('browser.rs', 'WheelUnit::Lines => 16.0,', 'WheelUnit::Lines => 15.0,', 'a line of the wheel is 16 pixels'),
    ('browser.rs', 'WheelUnit::Pages => self.view.1,', 'WheelUnit::Pages => self.view.0,', 'a page of the wheel is the height of the view'),
    ('browser.rs', 'let start = !w.armed;', 'let start = true;', 'the wheel timer starts once'),
    ('browser.rs', '        if !self.wheel.armed {\n            return None;\n        }\n        let w = std::mem::take', '        let w = std::mem::take', 'nothing to flush when nothing turned'),
    ('browser.rs', 'w.dx += delta.0 * k;', 'w.dx = delta.0 * k;', 'the turns of the wheel add up'),
    ('browser.rs', '|| press.composing ||', '||', 'a key of a composition is not sent'),
    ('browser.rs', 'press.key.to_lowercase() == \"v\" {\n            return None;', 'press.key.to_lowercase() == \"x\" {\n            return None;', 'the paste shortcut is left to the paste event'),
    ('browser.rs', 'if press.shift && (!single || !mods.is_empty()) {', 'if press.shift {', 'Shift is only named with other keys'),
    ('browser.rs', 'if press.key == \" \" { \"Space\" }', 'if press.key == \" \" { \"Spacebar\" }', 'the space key'),
    ('browser.rs', '(1..=12).contains(&n)', '(1..=11).contains(&n)', 'F12 is a key'),
    ('browser.rs', '&& b.get(1) != Some(&b\'0\')', '', 'F0x is no key'),
    ('browser.rs', '(!text.is_empty() && self.running).then(|| Input::Text', '(self.running).then(|| Input::Text', 'an empty paste is nothing'),
    ('browser.rs', 'let text = self.url_input.trim();', 'let text = self.url_input.as_str();', 'the address is trimmed'),
    ('browser.rs', 'if matches!(input, Input::Locate { .. }) && data.get("ok")', 'if data.get("ok")', 'only a locate makes a draft'),
    ('browser.rs', '        self.running = false;\n        self.key.clear();\n        self.image = None;\n    }\n\n    // ---- comments', '        self.running = false;\n        self.key.clear();\n    }\n\n    // ---- comments', 'a closed browser shows no picture'),
    ('browser.rs', '        if !self.commenting {\n            self.draft = None;\n        }', '        if self.commenting {\n            self.draft = None;\n        }', 'leaving comment mode drops the draft'),
    ('browser.rs', 'let note = note.trim();\n        if note.is_empty() {', 'let note = note;\n        if note.is_empty() {', 'a note of spaces is no note'),
    ('browser.rs', 'slice_units(&collapse(&member(&d, "text")), 160)', 'slice_units(&collapse(&member(&d, "text")), 150)', 'the text of a comment is clipped at 160'),
    ('browser.rs', '        self.comments.clear();\n        format!("[The user pointed', '        format!("[The user pointed', 'comments are sent once'),
    ('browser.rs', 'if box_text.trim().is_empty() {', 'if box_text.is_empty() {', 'a box of spaces is empty'),
    ('browser.rs', 'self.comments.remove(i);', 'self.comments.remove(0);', 'the comment that is removed'),
    ('browser.rs', 'if !self.url.is_empty() {\n            return self.url.clone();', 'if self.url.is_empty() {\n            return self.url.clone();', 'the page the browser is on'),
    ('browser.rs', 'if !self.url_focused {\n                self.url_input = if self.url == "about:blank"', 'if self.url_focused {\n                self.url_input = if self.url == "about:blank"', 'a frame does not overwrite the address you type'),
    ('browser.rs', 'self.url_input = if self.url == "about:blank" { String::new() } else { self.url.clone() };', 'self.url_input = self.url.clone();', 'a blank page has an empty address box'),
    ('browser.rs', '            self.undrawable = None;\n            if let Some(image)', '            if let Some(image)', 'a page that can be drawn again'),
    ('browser.rs', '"running, window shown"', '"running, shown"', 'the state says the window is shown'),
    ('browser.rs', 'if s.browser.is_none() {\n            return Empty', 'if s.browser.is_some() {\n            return Empty', 'no browser found'),
    ('browser.rs', 'pub fn show_image(&self) -> bool {\n        self.running && self.image.is_some()', 'pub fn show_image(&self) -> bool {\n        self.image.is_some()', 'no picture of a browser that stopped'),
    ('browser.rs', 'url_enabled: self.status.is_some() && self.enabled(),', 'url_enabled: self.status.is_some(),', 'the address box needs the browser switched on'),
    ('browser.rs', '(w > 0.0 && h > 0.0 && k > 0.0)', '(w > 0.0 && k > 0.0)', 'no frame around an element without a height'),
    ('browser.rs', '} else if self.running {\n            "Click, type', '} else if !self.running {\n            "Click, type', 'the foot says what you can do'),
    ('browser.rs', 'slice_units(&or_empty("text"), 60)', 'slice_units(&or_empty("text"), 59)', 'the label of a typed text is clipped at 60'),
    ('browser.rs', 't("Open in browser")', 't("Open in browsr")', 'the label of the open tool'),
    ('code.rs', '            BrMsg::Wide => {\n                self.br_wide = !self.br_wide;', '            BrMsg::Wide => {\n                self.br_wide = true;', 'the panel gets narrow again'),
    ('code.rs', '            BrMsg::Quick(address) => {\n                self.browser.url_input = address;', '            BrMsg::Quick(address) => {\n                let _ = address;', 'a quick address is sent'),
    ('app.rs', '&& press.key.to_lowercase() == "l" {', '&& press.key.to_lowercase() == "k" {', 'Ctrl+L'),
    ('app.rs', '(press.ctrl || press.meta) && !press.alt && press.key.to_lowercase() == "l"', '!press.alt && press.key.to_lowercase() == "l"', 'a plain l is only a letter'),
    ('code.rs', 'Item::Call(c) if c.name.starts_with("browser_") &&', 'Item::Call(c) if c.name.starts_with("bash") &&', 'what the agent does in the browser'),
    ('code.rs', 'matches!(c.state, CallState::Running | CallState::Pending) => {', 'matches!(c.state, CallState::Running) => {', 'a call that waits for you is shown'),
    ('code.rs', 'Some((matches!(c.state, CallState::Pending), line))', 'Some((false, line))', 'waiting for you is said as waiting'),
    ('code.rs', 'text(if v.show_image { format!("page {}×{}"', 'text(if true { format!("page {}×{}"', 'no size of a page that is not there'),
    ('code.rs', 'if running && b.url.is_empty() && v.url_enabled {', 'if running && v.url_enabled {', 'the usual addresses only while nothing is open'),
    ('code.rs', 'let chip = container(text("Comment mode: click an element")', 'let chip = container(text("Comment mode")', 'the chip of comment mode'),
    ('code.rs', 'if !b.title.is_empty() || !host.is_empty() {', 'if false {', 'the caption under the picture'),
    ('code.rs', 'rest.split([\'/\', \'?\', \'#\']).next().unwrap_or("")', 'rest', 'only the host in the caption'),
    ('code.rs', 'BrMsg::Hide => {\n                self.browser.open = false;\n                self.br_epoch += 1;', 'BrMsg::Hide => {\n                self.browser.open = false;', 'a hidden panel ends its ticks'),
    ('code.rs', 'self.browser.url_input = v;\n                self.browser.url_focused = true;\n                self.br_focus = false;', 'self.browser.url_input = v;\n                self.browser.url_focused = true;', 'typing in the address leaves the page'),
    ('code.rs', '            BrMsg::Submit => {\n                self.browser.url_focused = false;', '            BrMsg::Submit => {', 'a sent address lets frames in again'),
    ('code.rs', 'key.to_lowercase() == "v" && self.browser.running {', 'key.to_lowercase() == "v" {', 'paste needs a running browser'),
]

TEST = ["cargo", "test", "--locked", "--no-fail-fast", "--lib", "--test", "browser_fixtures", "--test", "browser_window", "--test", "browser_look", "--test", "browser_panel_logic"]


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
