#!/usr/bin/env python3
"""Mutate one anchor of the native app, require the named test to go red, restore."""
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
        "src/tour.rs",
        "let pct = used.clamp(2.0, 100.0).round() as u32;",
        "let pct = used.clamp(0.0, 100.0).round() as u32;",
        ["cargo", "test", "--locked", "--test", "tour", "system_memory"],
    ),
    (
        "src/tour.rs",
        'tier.to_string()',
        'format!("{}!", tier)',
        ["cargo", "test", "--locked", "--test", "tour", "feature_limit"],
    ),
    (
        "src/models.rs",
        "let tenth = (value * 10.0).round() / 10.0;",
        "let tenth = value;",
        ["cargo", "test", "--locked", "--lib", "speed_test"],
    ),
    (
        "src/gpucheck.rs",
        "            return group.members.iter().map(|m| m.id.clone()).collect();",
        "            return vec![id.to_string()];",
        ["cargo", "test", "--locked", "--lib", "the_check_of_a_card"],
    ),
    (
        "src/gpucheck.rs",
        'checks.saturating_sub(failed),',
        "failed,",
        ["cargo", "test", "--locked", "--lib", "processor_check_says"],
    ),
    (
        "src/amdnpu.rs",
        "a.chars().map(rank).cmp(b.chars().map(rank))",
        "a.cmp(b)",
        ["cargo", "test", "--locked", "--test", "amdnpu_fixtures"],
    ),
    (
        "src/amdnpu.rs",
        "        start: if ready && picked_known && !picked_says {\n            None",
        "        start: if ready && picked_known && !picked_says {\n            Some(false)",
        ["cargo", "test", "--locked", "--test", "amdnpu_fixtures"],
    ),
    (
        "src/models.rs",
        "running: s.get(\"next_token\").is_some_and(|t| !t.is_null()),",
        "running: false,",
        ["cargo", "test", "--locked", "--lib", "the_slots_table_reads"],
    ),
    (
        "src/settings.rs",
        "None => (String::from(\"Measuring ...\"), String::from(\"—\"), 35),",
        "None => (String::from(\"Loading ...\"), String::from(\"—\"), 35),",
        ["cargo", "test", "--locked", "--test", "hardware_window", "the_slots_table"],
    ),
    (
        "src/app.rs",
        "self.toast = Some(toast.to_string());",
        "let _ = toast;",
        ["cargo", "test", "--locked", "--test", "hardware_window", "a_setting_that_was_written"],
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
