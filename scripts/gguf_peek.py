#!/usr/bin/env python3
"""Reads only the header of a GGUF file over HTTP (Range requests) and prints its architecture, a few metadata keys and the
tensor names (layer numbers collapsed). Used to check what real llama.cpp files of a model family contain without downloading them.

    python scripts/gguf_peek.py https://huggingface.co/<owner>/<repo>/resolve/main/<file>.gguf [--keys] [--limit 60]
"""
from __future__ import annotations

import re
import struct
import sys
import urllib.request

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Reader:
    def __init__(self, url: str):
        self.url, self.buf, self.pos, self.start = url, b"", 0, 0
        self.fetch(0, 1 << 22)

    def fetch(self, start: int, size: int) -> None:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + size - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.buf, self.start = r.read(), start

    def take(self, n: int) -> bytes:
        while self.pos + n > self.start + len(self.buf):
            self.fetch(self.pos, max(1 << 22, n))
        out = self.buf[self.pos - self.start : self.pos - self.start + n]
        self.pos += n
        return out

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.unpack("<Q")).decode("utf8", "replace")

    def value(self, kind: int):
        if kind in SCALARS:
            return self.unpack(SCALARS[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            inner, count = self.unpack("<I"), self.unpack("<Q")
            if inner in SCALARS and count > 64:
                self.pos += struct.calcsize(SCALARS[inner]) * count
                return f"<{count} values>"
            if inner == 8 and count > 64:
                for _ in range(count):
                    self.string()
                return f"<{count} strings>"
            return [self.value(inner) for _ in range(count)]
        raise ValueError(f"unknown value type {kind}")


def main() -> int:
    url = sys.argv[1]
    show_keys = "--keys" in sys.argv
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 60
    r = Reader(url)
    assert r.take(4) == b"GGUF"
    version, tensors, kv = r.unpack("<I"), r.unpack("<Q"), r.unpack("<Q")
    meta = {}
    for _ in range(kv):
        key = r.string()
        meta[key] = r.value(r.unpack("<I"))
    arch = meta.get("general.architecture")
    print(f"version {version}, {tensors} tensors, architecture {arch!r}, name {meta.get('general.name')!r}")
    for k, v in meta.items():
        if show_keys or (arch and k.startswith(f"{arch}.")):
            print(f"  {k} = {str(v)[:100]}")
    names = {}
    for _ in range(tensors):
        name = r.string()
        dims = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
        kind = r.unpack("<I")
        r.unpack("<Q")
        names.setdefault(re.sub(r"\.\d+\.", ".N.", name), (dims, kind))
    for n, (dims, kind) in list(names.items())[:limit]:
        print(f"  {n} {dims} type {kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
