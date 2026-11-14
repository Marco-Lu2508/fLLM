#!/usr/bin/env python3
"""Mutate one anchor of the search cards, require the named test to go red, restore."""
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
        # the line under the list counts the engines another way
        "src/search.rs",
        '"{} engine(s) · saved in {}"',
        '"{} engines in {}"',
        ["cargo", "test", "--locked", "--lib", "the_line_counts"],
    ),
    (
        # SearXNG is allowed in without an address
        "src/search.rs",
        'return Err("A SearXNG engine needs its address.".into());',
        'return Err("A SearXNG engine needs a URL.".into());',
        ["cargo", "test", "--locked", "--lib", "an_engine_is_added"],
    ),
    (
        # a vector with nothing in it is counted as similar
        "src/search.rs",
        "    if na == 0.0 || nb == 0.0 {\n        return 0.0;\n    }",
        "    if na == 0.0 || nb == 0.0 {\n        return 1.0;\n    }",
        ["cargo", "test", "--locked", "--lib", "the_similarity"],
    ),
    (
        # the count of hits is not written the way the Studio writes it
        "src/search.rs",
        'format!("{}: {} Treffer", line.engine, js_undefined(line.found.map(|n| n.to_string())))',
        'format!("{}: {} hits", line.engine, js_undefined(line.found.map(|n| n.to_string())))',
        ["cargo", "test", "--locked", "--lib", "the_test_writes"],
    ),
    (
        # the similarity is written with another precision than `toFixed(3)` gives
        "src/search.rs",
        "Similarity of two similar sentences: {:.3}.",
        "Similarity of two similar sentences: {:.1}.",
        ["cargo", "test", "--locked", "--lib", "the_similarity"],
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
