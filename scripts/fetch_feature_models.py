#!/usr/bin/env python3
"""Download small public test models, pin revision and verify Hub SHA-256.

Explicit opt-in test assets, stored outside the repository/production models.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ASSETS = {
    "vision": (
        "ggml-org/SmolVLM-256M-Instruct-GGUF",
        ["SmolVLM-256M-Instruct-Q8_0.gguf", "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"],
    ),
    "embedding": ("ggml-org/bge-small-en-v1.5-Q8_0-GGUF", ["bge-small-en-v1.5-q8_0.gguf"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset", choices=ASSETS, action="append")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for asset in args.asset or ASSETS:
        repo, files = ASSETS[asset]
        with urllib.request.urlopen(
            f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=60
        ) as response:
            model = json.load(response)
        for name in files:
            entry = next(item for item in model["siblings"] if item["rfilename"] == name)
            expected = entry["lfs"]["sha256"]
            path = args.output_dir / name
            if not path.exists():
                temporary = path.with_suffix(".partial")
                url = f"https://huggingface.co/{repo}/resolve/{model['sha']}/{name}?download=true"
                digest = hashlib.sha256()
                with (
                    urllib.request.urlopen(url, timeout=180) as response,
                    temporary.open("wb") as destination,
                ):
                    while block := response.read(1024 * 1024):
                        digest.update(block)
                        destination.write(block)
                if digest.hexdigest() != expected:
                    raise RuntimeError(
                        f"SHA-256 mismatch for {name}; retained partial file for inspection"
                    )
                temporary.rename(path)
            else:
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise RuntimeError(
                        f"Existing asset {name} has a different checksum; refusing to replace it"
                    )
            manifest.append(
                {
                    "repo": repo,
                    "revision": model["sha"],
                    "file": name,
                    "sha256": expected,
                    "license": model.get("cardData", {}).get("license"),
                }
            )
            print(f"Verified {name}: {path.stat().st_size} bytes", flush=True)
    (args.output_dir / "feature-models.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
