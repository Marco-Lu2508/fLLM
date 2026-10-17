#!/usr/bin/env python3
"""Compare two GGUF files tensor by tensor: type agreement and byte-exact payloads.

    python scripts/compare_gguf.py ours.gguf reference.gguf

Used to check `fllm-rs quantize` against files written by llama.cpp's llama-quantize.
"""
import mmap
import struct
import sys

SCALARS = {0: 1, 1: 2, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read(path):
    f = open(path, "rb")
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    at = 0

    def u(fmt):
        nonlocal at
        v = struct.unpack_from("<" + fmt, m, at)
        at += struct.calcsize("<" + fmt)
        return v[0]

    def string():
        nonlocal at
        n = u("Q")
        s = m[at : at + n].decode()
        at += n
        return s

    def skip(t):
        nonlocal at
        if t in SCALARS:
            at += SCALARS[t]
        elif t == 8:
            string()
        elif t == 9:
            inner = u("I")
            for _ in range(u("Q")):
                skip(inner)
        else:
            raise SystemExit(f"unknown metadata type {t}")

    assert m[:4] == b"GGUF"
    at = 4
    u("I")
    n_tensors = u("Q")
    n_kv = u("Q")
    align = 32
    for _ in range(n_kv):
        key = string()
        t = u("I")
        if key == "general.alignment":
            align = u("I")
        else:
            skip(t)
    infos = []
    for _ in range(n_tensors):
        name = string()
        dims = [u("Q") for _ in range(u("I"))]
        type_id = u("I")
        off = u("Q")
        infos.append((name, dims, type_id, off))
    base = -(-at // align) * align
    out = {}
    for i, (name, dims, type_id, off) in enumerate(infos):
        end = infos[i + 1][3] if i + 1 < len(infos) else len(m) - base
        out[name] = (dims, type_id, m[base + off : base + end])
    return out


a, b = read(sys.argv[1]), read(sys.argv[2])
same = diff = typed = missing = 0
for name, (dims, t, data) in a.items():
    if name not in b:
        missing += 1
        continue
    d2, t2, data2 = b[name]
    if t != t2:
        typed += 1
        print(f"type differs: {name}: {t} vs {t2}")
        continue
    n = min(len(data), len(data2))
    if data[:n].rstrip(b"\0") == data2[:n].rstrip(b"\0") and abs(len(data) - len(data2)) < 64:
        same += 1
    else:
        diff += 1
        if diff <= 5:
            bad = next((i for i in range(n) if data[i] != data2[i]), None)
            print(f"bytes differ: {name} type {t} first difference at byte {bad}")
print(f"identical {same}, different {diff}, type mismatches {typed}, missing {missing}")
