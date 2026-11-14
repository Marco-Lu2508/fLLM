#!/usr/bin/env python3
"""Mutate one anchor of the three cards of row 66, require the named test to go red, restore."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/opt/data/cache/scratch/fLLM/desktop/native")
ENV = {
    **os.environ,
    "PATH": "/opt/data/home/.cargo/bin:" + os.environ["PATH"],
    "CARGO_TARGET_DIR": "/opt/data/cache/scratch/fllm-app-target",
    "FLLM_SNAPSHOT_DIR": str(ROOT / "ui-snapshots"),
}

MUTATIONS = [
    (
        "src/plugins.rs",
        '        parts.push(format!("{} skill(s)", p.skills));',
        '        parts.push(format!("{} skills", p.skills));',
        ["cargo", "test", "--locked", "--lib", "a_plugin_tells"],
    ),
    (
        "src/plugins.rs",
        '        format!("Plugins live in {folder}")',
        '        format!("The plugins live in {folder}")',
        ["cargo", "test", "--locked", "--lib", "the_folder_line"],
    ),
    (
        "src/hooks.rs",
        "    let entries: Vec<serde_json::Value> = rows.iter().filter(|row| row.is_kept()).map(HookRow::entry).collect();",
        "    let entries: Vec<serde_json::Value> = rows.iter().map(HookRow::entry).collect();",
        ["cargo", "test", "--locked", "--lib", "a_row_that_says"],
    ),
    (
        "src/hooks.rs",
        "        let value = if self.timeout_seconds == 0.0 {\n            30.0",
        "        let value = if self.timeout_seconds == 0.0 {\n            45.0",
        ["cargo", "test", "--locked", "--lib", "the_timeout_of_the_box"],
    ),
    (
        "src/hooks.rs",
        '        format!("BLOCKED: {reason}\\n")',
        '        format!("BLOCKED: {reason} ")',
        ["cargo", "test", "--locked", "--lib", "a_test_says"],
    ),
    (
        "src/hooks.rs",
        '            serde_json::json!({ "path": "README.md", "kind": "modified" }),',
        '            serde_json::json!({ "path": "README.md", "kind": "created" }),',
        ["cargo", "test", "--locked", "--lib", "the_sample_of_a_test"],
    ),
    (
        "src/agents.rs",
        "    let first = bytes[0];\n    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {",
        "    let first = bytes[0];\n    if !first.is_ascii_alphanumeric() {",
        ["cargo", "test", "--locked", "--lib", "a_name_is_what"],
    ),
    (
        "src/agents.rs",
        '        format!("can change files or run commands ({} tools)", tools.len())',
        '        format!("can change files or run commands ({} tools)", writes)',
        ["cargo", "test", "--locked", "--lib", "the_line_of_a_row"],
    ),
    (
        "src/agents.rs",
        "    let steps = steps.clamp(0.0, 60.0) as i64;",
        "    let steps = steps.clamp(0.0, 90.0) as i64;",
        ["cargo", "test", "--locked", "--lib", "the_form_sends"],
    ),
    (
        "src/agents.rs",
        "    match agent_tools {\n        Some(list) if !list.is_empty() => list.to_vec(),\n        _ => read.iter().map(|tool| tool.name.clone()).collect(),\n    }",
        "    match agent_tools {\n        Some(list) if !list.is_empty() => list.to_vec(),\n        _ => Vec::new(),\n    }",
        ["cargo", "test", "--locked", "--lib", "the_boxes_start"],
    ),
]


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=900)


def main():
    problems = []
    for path_rel, old, new, cmd in MUTATIONS:
        path = ROOT / path_rel
        raw = path.read_bytes()
        old_b, new_b = old.encode(), new.encode()
        count = raw.count(old_b)
        name = old.strip().splitlines()[0][:64]
        if count != 1:
            print(f"SKIP  {name} — anchor occurs {count} times")
            problems.append(name)
            continue
        path.write_bytes(raw.replace(old_b, new_b, 1))
        try:
            result = run(cmd)
        finally:
            path.write_bytes(raw)
        red = "test result: FAILED" in (result.stdout + result.stderr)
        restored = path.read_bytes() == raw
        if not red or not restored:
            problems.append(name)
        print(f"{'RED  ' if red else 'GREEN'} {name} (restored={restored})")
        if not red:
            for line in (result.stdout + result.stderr).strip().splitlines()[-6:]:
                print(f"      {line}")
    print("ALL MUTATIONS BIT" if not problems else f"PROBLEMS: {problems}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
