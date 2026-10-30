#!/usr/bin/env python3
"""Reads only the safetensors header of a file over HTTP (Range requests) and prints tensor names (layer numbers collapsed) with
dtype and shape. Used to check what real repositories store without downloading the weights.

    python scripts/st_peek.py https://huggingface.co/<owner>/<repo>/resolve/main/<file>.safetensors [--limit 60] [--grep SUBSTRING]
    python scripts/st_peek.py local/file.safetensors"""
import json
import os
import re
import struct
import sys
import urllib.request


def read(url, start, size):
    if os.path.exists(url):
        with open(url, "rb") as f:
            f.seek(start)
            return f.read(size)
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + size - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    url = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 60
    needle = sys.argv[sys.argv.index("--grep") + 1] if "--grep" in sys.argv else None
    (length,) = struct.unpack("<Q", read(url, 0, 8))
    header = json.loads(read(url, 8, length))
    meta = header.pop("__metadata__", None)
    if meta:
        print("metadata:", meta)
    seen = {}
    for name, info in header.items():
        key = re.sub(r"\.\d+\.", ".N.", name)
        if needle and needle not in key:
            continue
        seen.setdefault(key, (info["dtype"], info["shape"]))
    print(f"{len(header)} tensors")
    for key, (dtype, shape) in list(seen.items())[:limit]:
        print(f"  {key} {dtype} {shape}")


if __name__ == "__main__":
    main()
