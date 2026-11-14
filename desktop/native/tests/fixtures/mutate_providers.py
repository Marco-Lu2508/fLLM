#!/usr/bin/env python3
"""Mutate one anchor of the provider functions, require the named test to go red, restore."""
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
        "src/providers.rs",
        "if slash < 1 || slash + 1 == rest.len() {",
        "if slash < 1 {",
        ["cargo", "test", "--locked", "--lib", "the_id_of_a_provider_model"],
    ),
    (
        "src/providers.rs",
        'let label = format!("{} · {}", m.provider_name.as_deref().filter(|n| !n.is_empty()).unwrap_or(&m.provider), m.id);',
        'let label = format!("{} · {}", m.provider, m.id);',
        ["cargo", "test", "--locked", "--lib", "the_picker_lists"],
    ),
    (
        "src/providers.rs",
        "        fallback.to_string()\n    }\n}",
        '        selected.to_string()\n    }\n}',
        ["cargo", "test", "--locked", "--lib", "a_selection_that_is_gone"],
    ),
    (
        "src/providers.rs",
        'if p.group.as_deref() == Some("network") || (p.group.is_none() && p.local)',
        "if p.local",
        ["cargo", "test", "--locked", "--lib", "the_templates_come"],
    ),
    (
        "src/providers.rs",
        'if url.scheme() != "https" {',
        'if url.scheme() == "https" {',
        ["cargo", "test", "--locked", "--lib", "only_an_https_page"],
    ),
    (
        "src/providers.rs",
        "let shown: Vec<&str> = models.iter().take(8).map(String::as_str).collect();",
        "let shown: Vec<&str> = models.iter().take(5).map(String::as_str).collect();",
        ["cargo", "test", "--locked", "--lib", "the_line_of_a_test"],
    ),
    (
        "src/providers.rs",
        'format!("{name} · {}", r.model)',
        'format!("{}", r.model)',
        ["cargo", "test", "--locked", "--lib", "the_label_of_the_picker"],
    ),
    (
        "src/providers.rs",
        'out.insert("model".into(), json!(model));',
        'let _ = model;',
        ["cargo", "test", "--locked", "--lib", "a_request_for_a_provider"],
    ),
    (
        "src/providers.rs",
        'None => ChatEndpoint { url: "/v1/chat/completions", remote: false, provider: None, model: None },',
        'None => ChatEndpoint { url: "/api/studio/providers/chat", remote: true, provider: None, model: None },',
        ["cargo", "test", "--locked", "--lib", "a_chat_of_a_provider"],
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
