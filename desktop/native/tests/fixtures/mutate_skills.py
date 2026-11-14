#!/usr/bin/env python3
"""Mutate one anchor of the skills cards, require the named test to go red, restore."""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/opt/data/cache/scratch/fLLM/desktop/native")
ENV = {
    **dict(__import__("os").environ),
    "PATH": "/opt/data/home/.cargo/bin:" + __import__("os").environ["PATH"],
    "CARGO_TARGET_DIR": "/opt/data/cache/scratch/fllm-app-target",
    "FLLM_SNAPSHOT_DIR": str(ROOT / "ui-snapshots"),
}

MUTATIONS = [
    (
        # an underscore survives the slug instead of becoming a dash
        "src/skills.rs",
        "        if c.is_ascii_lowercase() || c.is_ascii_digit() {",
        "        if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' {",
        ["cargo", "test", "--locked", "--lib", "a_name_is_lowercase"],
    ),
    (
        # one use is written the way several are
        "src/skills.rs",
        'let mut parts = vec![format!("{} use{}", skill.uses, if skill.uses == 1 { "" } else { "s" })];',
        'let mut parts = vec![format!("{} uses", skill.uses)];',
        ["cargo", "test", "--locked", "--lib", "the_row_counts"],
    ),
    (
        # a step that runs now says nothing about it
        "src/skills.rs",
        '        text.push_str(" A learning step is running now.");',
        "",
        ["cargo", "test", "--locked", "--lib", "the_counters"],
    ),
    (
        # a task that never ran says something else
        "src/skills.rs",
        'None => out.push_str("Has not run yet."),',
        'None => out.push_str("Has not run."),',
        ["cargo", "test", "--locked", "--lib", "a_task_is_told"],
    ),
    (
        # the seconds of an answer lose their unit
        "src/skills.rs",
        '"{} - {} ({}s)",',
        '"{} - {} ({})",',
        ["cargo", "test", "--locked", "--lib", "an_answer_says"],
    ),
    (
        # consolidating is called by its other name
        "src/skills.rs",
        '        "learn" => "consolidate",',
        '        "learn" => "learn",',
        ["cargo", "test", "--locked", "--lib", "a_task_is_told"],
    ),
]


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=900)


def main():
    failures = []
    for path_rel, old, new, cmd in MUTATIONS:
        path = ROOT / path_rel
        raw = path.read_bytes()
        old_b, new_b = old.encode(), new.encode()
        count = raw.count(old_b)
        name = old.strip().splitlines()[0][:64]
        if count != 1:
            print(f"SKIP  {name} — anchor occurs {count} times")
            failures.append(name)
            continue
        path.write_bytes(raw.replace(old_b, new_b, 1))
        try:
            result = run(cmd)
        finally:
            path.write_bytes(raw)
        red = "test result: FAILED" in (result.stdout + result.stderr)
        restored = path.read_bytes() == raw
        status = "RED  " if red else "GREEN"
        if not red or not restored:
            failures.append(name)
        print(f"{status} {name} (restored={restored})")
        if not red:
            tail = (result.stdout + result.stderr).strip().splitlines()[-6:]
            for line in tail:
                print(f"      {line}")
    print("ALL MUTATIONS BIT" if not failures else f"PROBLEMS: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
