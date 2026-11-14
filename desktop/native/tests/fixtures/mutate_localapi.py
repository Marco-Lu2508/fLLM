#!/usr/bin/env python3
"""Mutate one anchor of the logs and server cards, require the named test to go red, restore."""
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
        "src/localapi.rs",
        "Some(serde_json::json!(n as i64))",
        "Some(serde_json::json!((n as i64) + 1))",
        ["cargo", "test", "--locked", "--lib", "a_limit_is_written"],
    ),
    (
        "src/localapi.rs",
        'lines.join("\\n")',
        'lines.join(", ")',
        ["cargo", "test", "--locked", "--lib", "the_lines_are_joined"],
    ),
    (
        "src/localapi.rs",
        'Some(n) => format!("{n} per minute"),',
        'Some(n) => format!("{n}/min"),',
        ["cargo", "test", "--locked", "--lib", "a_key_without_a_limit"],
    ),
    (
        "src/localapi.rs",
        "if lines.is_empty() {",
        "if lines.len() > 99 {",
        ["cargo", "test", "--locked", "--lib", "the_lines_are_joined"],
    ),
    (
        "src/localapi.rs",
        '"query": "SIMD CPU acceleration"',
        '"query": "something else"',
        ["cargo", "test", "--locked", "--lib", "a_sample_is_the_body"],
    ),
    (
        "src/localapi.rs",
        "Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| text.to_string()),",
        "Ok(_) => text.to_string(),",
        ["cargo", "test", "--locked", "--lib", "an_answer_is_pretty"],
    ),
    (
        "src/localapi.rs",
        "ROUTES.iter().find(|(_, l)| *l == label).map(|(v, _)| *v).unwrap_or(ROUTES[0].0)",
        "ROUTES.iter().find(|(_, l)| *l == label).map(|(v, _)| *v).unwrap_or(&ROUTES[1].0)",
        ["cargo", "test", "--locked", "--lib", "the_select_offers"],
    ),
    (
        "src/localapi.rs",
        'format!("{tokens} tokens · {bytes} bytes")',
        'format!("{tokens} tokens")',
        ["cargo", "test", "--locked", "--lib", "the_line_counts"],
    ),
    (
        "src/localapi.rs",
        "iced::Color { r: r + m, g: g + m, b: b + m, a: 0.18 }",
        "iced::Color { r: r + m, g: g + m, b: b + m, a: 1.0 }",
        ["cargo", "test", "--locked", "--lib", "a_pill_takes_the_hue"],
    ),
    (
        "src/localapi.rs",
        "Token::Piece { piece, .. } => piece.clone(),",
        "Token::Piece { id, .. } => id.to_string(),",
        ["cargo", "test", "--locked", "--lib", "a_token_is_read"],
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
