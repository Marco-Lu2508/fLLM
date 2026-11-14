#!/usr/bin/env python3
"""Mutate one anchor of the MCP cards, require the named test to go red, restore."""
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
        # one tool is written the way several are
        "src/mcp.rs",
        'format!(" - {} tool{}", s.tool_count, if s.tool_count == 1 { "" } else { "s" })',
        'format!(" - {} tools", s.tool_count)',
        ["cargo", "test", "--locked", "--lib", "the_state_of_a_ready"],
    ),
    (
        # the problem of the file is written another way
        "src/mcp.rs",
        'Some(problem) => format!("Saved in {} - problem: {}", answer.config_file, problem),',
        'Some(problem) => format!("Saved in {} (problem) {}", answer.config_file, problem),',
        ["cargo", "test", "--locked", "--lib", "the_line_under_the_form"],
    ),
    (
        # an empty value for a saved variable is sent instead of keeping what is there
        "src/mcp.rs",
        "out.insert(key, (!keeps).then_some(value));",
        "out.insert(key, Some(value));",
        ["cargo", "test", "--locked", "--lib", "the_boxes_read"],
    ),
    (
        # the value of a header comes out trimmed
        "src/mcp.rs",
        "out.insert(t[..i].trim().to_string(), t[i + 1..].to_string());",
        "out.insert(t[..i].trim().to_string(), t[i + 1..].trim().to_string());",
        ["cargo", "test", "--locked", "--lib", "the_boxes_read"],
    ),
    (
        # a header whose value holds a placeholder is not a note
        "src/mcp.rs",
        "if h.required || h.value.contains('{') {",
        "if h.required {",
        ["cargo", "test", "--locked", "--lib", "use_fills_the_form"],
    ),
    (
        # the address of a remote server is written without the word the page puts in front of it
        "src/mcp.rs",
        'return format!("HTTP {}", s.url);',
        "return s.url.clone();",
        ["cargo", "test", "--locked", "--lib", "the_line_under_a_server"],
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
