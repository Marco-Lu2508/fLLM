#!/usr/bin/env python3
"""Mechanical checks for the documentation (no third-party packages).

    python3 scripts/check_docs.py             # all checks, exit 1 on any error
    python3 scripts/check_docs.py --only links,jargon
    python3 scripts/check_docs.py --list      # print the checks

Checks
  links    relative Markdown links point at an existing file and, with #anchor, at an existing heading
  paths    repository paths written in `backticks` (rust/..., docs/..., scripts/..., .github/... and so on) exist
  german   Markdown files must be English: files with many German function words outside code blocks are reported
  jargon   internal development wording (work-package labels, agent names, private host names, ...) must not appear
  yaml     workflow and issue-template files are parsed with PyYAML when it is installed (skipped otherwise)

Files under target*/, third_party/, .git/ and .venv/ are ignored. Generated files may be listed in GENERATED below.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "target", "third_party", ".venv", "node_modules"}
GENERATED = {"docs/cli-reference.md", "bench/baseline.json"}  # created on demand
PATH_PREFIXES = ("rust/", "docs/", "scripts/", "bench/", ".github/", "docker/", "licenses/", "benchmarks/")
PATH_SUFFIXES = (
    ".rs", ".md", ".py", ".sh", ".ps1", ".yml", ".yaml", ".toml", ".cu", ".cuh", ".cpp", ".hpp", ".h", ".glsl", ".metal",
    ".json", ".patch", ".txt", ".html",
)
GERMAN_WORDS = {
    "und", "nicht", "wird", "werden", "für", "oder", "eine", "einen", "einer", "sind", "dem", "auch", "aber", "wenn", "dann",
    "noch", "nur", "bei", "nach", "zum", "zur", "über", "können", "müssen", "gegen", "ohne", "damit", "weil", "diese", "dieser",
    "wurde", "wurden", "sowie", "jeden", "jede", "keine", "kein",
}
GERMAN_LIMIT = 6  # distinct hits outside code blocks before a file is reported
JARGON = [
    r"\bpackages? p\d{1,2}\b", r"\bp\d{2}\b report", r"\bMega-?Runde\b", r"\brounds? \d+\b", r"\bthis package\b",
    r"\bbranch of this package\b", r"task-provided",
]
JARGON_SKIP = {"scripts/check_docs.py"}
JARGON_SKIP_DIRS = ("docs/changes/",)


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True)
        files = [ROOT / line for line in out.stdout.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        files = [p for p in ROOT.rglob("*") if p.is_file()]
    return [p for p in files if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS)]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_code(text: str) -> str:
    """Remove fenced code blocks, keeping line numbers (blank lines)."""
    out, fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            fence = not fence
            out.append("")
        else:
            out.append("" if fence else line)
    return "\n".join(out)


def slug(heading: str) -> str:
    h = re.sub(r"`([^`]*)`", r"\1", heading)
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", h)
    h = h.strip().lower()
    h = re.sub(r"[^\w\- ]", "", h, flags=re.UNICODE)
    return h.replace(" ", "-")


def anchors_of(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    if path in cache:
        return cache[path]
    seen: dict[str, int] = {}
    result: set[str] = set()
    for line in strip_code(read(path)).splitlines():
        m = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        s = slug(m.group(1))
        n = seen.get(s, 0)
        seen[s] = n + 1
        result.add(s if n == 0 else f"{s}-{n}")
    for m in re.finditer(r'<a\s+(?:name|id)="([^"]+)"', read(path)):
        result.add(m.group(1))
    cache[path] = result
    return result


def check_links(files: list[Path], errors: list[str]) -> None:
    cache: dict[Path, set[str]] = {}
    link = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
    for f in files:
        if f.suffix != ".md":
            continue
        for no, line in enumerate(strip_code(read(f)).splitlines(), 1):
            line = re.sub(r"`[^`]*`", "", line)
            for m in link.finditer(line):
                target = m.group(1)
                if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("//"):
                    continue
                path_part, _, anchor = target.partition("#")
                dest = f if not path_part else (f.parent / path_part).resolve()
                rel = f.relative_to(ROOT)
                if path_part and not dest.exists():
                    if str(dest.relative_to(ROOT)).replace("\\", "/") in GENERATED:
                        continue
                    errors.append(f"{rel}:{no}: broken link target '{path_part}'")
                    continue
                if anchor and dest.suffix == ".md" and dest.is_file():
                    if anchor.lower() not in anchors_of(dest, cache):
                        errors.append(f"{rel}:{no}: no heading '#{anchor}' in {dest.relative_to(ROOT)}")


def check_paths(files: list[Path], errors: list[str]) -> None:
    span = re.compile(r"`([^`\n]+)`")
    for f in files:
        if f.suffix != ".md" or str(f.relative_to(ROOT)).replace("\\", "/").startswith(JARGON_SKIP_DIRS):
            continue
        for no, line in enumerate(read(f).splitlines(), 1):
            for m in span.finditer(line):
                token = m.group(1).strip()
                if not token.startswith(PATH_PREFIXES) or " " in token:
                    continue
                if any(c in token for c in "*<>{}$|()[]") or token.endswith("/"):
                    continue
                token = token.split("#")[0].split(":")[0].rstrip(".,;")
                if not token.endswith(PATH_SUFFIXES):
                    continue
                if token in GENERATED or (ROOT / token).exists():
                    continue
                errors.append(f"{f.relative_to(ROOT)}:{no}: path `{token}` does not exist")


def check_german(files: list[Path], errors: list[str]) -> None:
    word = re.compile(r"[A-Za-zÄÖÜäöüß]+")
    for f in files:
        if f.suffix != ".md":
            continue
        text = strip_code(read(f))
        text = re.sub(r"`[^`]*`", "", text)
        hits = {w.lower() for w in word.findall(text) if w.lower() in GERMAN_WORDS}
        if len(hits) >= GERMAN_LIMIT:
            errors.append(f"{f.relative_to(ROOT)}: looks German ({len(hits)} German function words: {', '.join(sorted(hits)[:8])})")


def check_jargon(files: list[Path], errors: list[str]) -> None:
    rx = re.compile("|".join(JARGON), re.I)
    for f in files:
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        if rel in JARGON_SKIP or rel.startswith(JARGON_SKIP_DIRS):
            continue
        if f.suffix not in {".md", ".yml", ".yaml", ".sh", ".ps1", ".py"} or f.stat().st_size > 1_000_000:
            continue
        for no, line in enumerate(read(f).splitlines(), 1):
            m = rx.search(line)
            if m:
                errors.append(f"{rel}:{no}: internal wording '{m.group(0)}'")


def check_yaml(files: list[Path], errors: list[str]) -> None:
    try:
        import yaml  # type: ignore
    except ImportError:
        print("yaml: PyYAML not installed, skipped (use actionlint for workflows)")
        return
    for f in files:
        rel = f.relative_to(ROOT)
        if f.suffix in {".yml", ".yaml"} and (".github" in rel.parts):
            try:
                yaml.safe_load(read(f))
            except yaml.YAMLError as e:
                errors.append(f"{rel}: invalid YAML: {e}")


CHECKS = {"links": check_links, "paths": check_paths, "german": check_german, "jargon": check_jargon, "yaml": check_yaml}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma separated list of checks")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(CHECKS))
        return 0
    names = args.only.split(",") if args.only else list(CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        ap.error("unknown check: " + ", ".join(unknown))
    files = tracked_files()
    failed = 0
    for name in names:
        errors: list[str] = []
        CHECKS[name](files, errors)
        print(f"{name}: {'ok' if not errors else str(len(errors)) + ' problem(s)'}")
        for e in errors:
            print("  " + e)
        failed += bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
