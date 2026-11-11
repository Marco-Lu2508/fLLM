#!/usr/bin/env python3
"""Write the body of a GitHub release from CHANGELOG.md.

    python3 scripts/release_notes.py v0.2.2 > release-notes.md

The release gets what a reader wants first: the intro and the Highlights of the version's changelog section, how to
install it, and a link to the full section. The workflow appends GitHub's list of merged changes and, at the end, the
checksum table. Exits with 1 when the changelog has no section for the tag, so a release can not go out with empty notes.
"""
import re
import sys
from pathlib import Path

REPO = "Marco-Lu2508/fLLM"


def section(changelog: str, version: str) -> str:
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", changelog, re.S | re.M)
    if not m:
        raise SystemExit(f"CHANGELOG.md has no section for {version}")
    return m.group(1).strip("\n")


def intro_and_highlights(body: str) -> tuple[str, str]:
    """The text before the first '###' and the bullets of every '### Highlights' block (a version may have two)."""
    parts = re.split(r"^### ", body, flags=re.M)
    intro = parts[0].strip()
    bullets = []
    for p in parts[1:]:
        head, _, rest = p.partition("\n")
        if head.strip().lower() == "highlights":
            bullets.append(rest.strip("\n"))
    return intro, "\n".join(bullets)


def main() -> None:
    tag = sys.argv[1]
    version = tag.lstrip("v")
    root = Path(__file__).resolve().parent.parent
    body = section((root / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    intro, highlights = intro_and_highlights(body)
    out = [f"# fLLM {tag}", ""]
    if intro:
        out += [intro, ""]
    if highlights:
        out += ["## Highlights", "", highlights, ""]
    out += [
        "## Install",
        "",
        "```sh",
        f"curl -fsSL https://raw.githubusercontent.com/{REPO}/{tag}/install.sh | sh   # Linux, macOS",
        "```",
        "",
        "Windows: `install.ps1` from the same tag; desktop installers (fLLM and fLLM-cuda), the Docker images and every archive are in the assets below, "
        "each with a `.sha256` file and a keyless cosign bundle (`.sigstore.json`).",
        "",
        f"The complete list of changes of this version is in the [changelog](https://github.com/{REPO}/blob/{tag}/CHANGELOG.md), section {version}.",
        "",
    ]
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
