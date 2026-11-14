#!/usr/bin/env python3
"""Mutate one anchor of the settings search, require the named test to go red, restore."""
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
        "src/settingssearch.rs",
        "        .split(|c: char| !c.is_alphanumeric())",
        "        .split(|c: char| !c.is_ascii_alphanumeric())",
        ["cargo", "test", "--locked", "--lib", "a_question_becomes"],
    ),
    (
        "src/settingssearch.rs",
        """            } else {
                // a word without a place on this page takes the page out of the answer
                score = 0;
                break;
            };""",
        """            } else {
                4
            };""",
        ["cargo", "test", "--locked", "--lib", "a_page_without_a_word"],
    ),
    (
        "src/settingssearch.rs",
        "    out.sort_by(|a, b| b.score.cmp(&a.score).then(a.order.cmp(&b.order)));",
        "    out.sort_by(|a, b| a.score.cmp(&b.score).then(a.order.cmp(&b.order)));",
        ["cargo", "test", "--locked", "--lib", "a_word_counts_where"],
    ),
    (
        "src/settingssearch.rs",
        """            let points = if label.starts_with(word.as_str()) {
                100""",
        """            let points = if label.starts_with(word.as_str()) {
                60""",
        ["cargo", "test", "--locked", "--lib", "a_word_counts_where"],
    ),
    (
        "src/settingssearch.rs",
        """    if search_words(query).is_empty() {
        return true;
    }""",
        """    if search_words(query).is_empty() {
        return false;
    }""",
        ["cargo", "test", "--locked", "--lib", "the_column_shows"],
    ),
    (
        "src/settingssearch.rs",
        "    !search_words(query).is_empty() && rank_settings_pages(query, PAGES).is_empty()",
        "    !search_words(query).is_empty()",
        ["cargo", "test", "--locked", "--lib", "the_line_of_the_box"],
    ),
    (
        "src/settingssearch.rs",
        "    rank_settings_pages(query, PAGES).first().map(|rank| rank.id)",
        "    rank_settings_pages(query, PAGES).last().map(|rank| rank.id)",
        ["cargo", "test", "--locked", "--lib", "the_best_match_comes"],
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
