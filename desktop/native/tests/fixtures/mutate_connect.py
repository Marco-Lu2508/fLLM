#!/usr/bin/env python3
"""Mutate one anchor of the connect cards, require the named test to go red, restore."""
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
        # the temperature of the example is the one in the box, not a fixed one
        "src/connect.rs",
        "    temperature={temperature},",
        "    temperature=0.7,",
        ["cargo", "test", "--locked", "--lib", "the_four_examples"],
    ),
    (
        # the sentence of `statusLine` has its two halves the other way round
        "src/connect.rs",
        "    if auth_required {",
        "    if !auth_required {",
        ["cargo", "test", "--locked", "--lib", "the_line_about_the_key"],
    ),
    (
        # the snippet of the plan is left out of the blocks
        "src/connect.rs",
        "    if let Some(snippet) = &plan.snippet {",
        "    if let Some(snippet) = &None::<Snippet> {",
        ["cargo", "test", "--locked", "--lib", "the_blocks_come_in"],
    ),
    (
        # a shell of the select comes back as the wrong one
        "src/connect.rs",
        ".map(|(id, _)| *id).unwrap_or(\"sh\")",
        ".map(|(id, _)| if *id == \"sh\" { \"powershell\" } else { *id }).unwrap_or(\"sh\")",
        ["cargo", "test", "--locked", "--lib", "the_shell_selects"],
    ),
    (
        # the file that cannot be written loses the reason in its title
        "src/connect.rs",
        'format!("{} (cannot be created here)", self.path)',
        "self.path.clone()",
        ["cargo", "test", "--locked", "--lib", "the_blocks_come_in"],
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
