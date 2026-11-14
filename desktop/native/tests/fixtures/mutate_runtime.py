#!/usr/bin/env python3
"""Mutate one anchor of the Runtime cards, require the named test to go red, restore."""
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
        "src/runtime.rs",
        'parts.join(" · ")',
        'parts.join(", ")',
        ["cargo", "test", "--locked", "--lib", "the_state_line_reads"],
    ),
    (
        "src/runtime.rs",
        '" (commands are refused in this mode)"',
        '" (commands run unwrapped)"',
        ["cargo", "test", "--locked", "--lib", "the_state_line_reads"],
    ),
    (
        "src/runtime.rs",
        'lines.join("\\n")',
        'lines.join(" · ")',
        ["cargo", "test", "--locked", "--lib", "the_details_are_the_lines"],
    ),
    (
        "src/runtime.rs",
        '"This build has no computer use (it was built without the computer-use feature).".into()',
        '"Computer use is forbidden on this server (FLLM_NO_COMPUTER_USE).".into()',
        ["cargo", "test", "--locked", "--lib", "the_computer_line_says"],
    ),
    (
        "src/runtime.rs",
        "c.forbidden || !c.compiled",
        "c.forbidden",
        ["cargo", "test", "--locked", "--lib", "the_computer_switch_is_dead"],
    ),
    (
        "src/runtime.rs",
        "let n = n.round();",
        "let n = n.trunc();",
        ["cargo", "test", "--locked", "--lib", "a_limit_is_rounded"],
    ),
    (
        "src/runtime.rs",
        'Confirm::Turbo => "Turbo removes every permission question and the sandbox. Continue?"',
        'Confirm::Turbo => "Turbo removes every permission question and the sandbox. Go on?"',
        ["cargo", "test", "--locked", "--test", "runtime_window", "a_switch_asks"],
    ),
    (
        "src/app.rs",
        'self.runtime_say = "Saved.".into();',
        "self.runtime_say = String::new();",
        ["cargo", "test", "--locked", "--test", "runtime_window", "a_switch_asks"],
    ),
]


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    failures = []
    for i, (rel, old, new, cmd) in enumerate(MUTATIONS, 1):
        name = f"{i}. {rel}: {old[:40]!r}"
        if only and only not in name:
            continue
        path = ROOT / rel
        raw = path.read_bytes()
        old_b, new_b = old.encode(), new.encode()
        count = raw.count(old_b)
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
