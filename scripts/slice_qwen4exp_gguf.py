#!/usr/bin/env python3
"""Cut the first N layers (without the n-gram layer) out of a real `qwen4exp` GGUF into a small GGUF with the same tensor bytes.

The slice keeps the tokenizer, the embedding, the output head and the hyper-connection mixers of the full model and the real quantized
tensors of layers 0..N-1 (types and bytes untouched), drops the n-gram embedding (`ple.*` keys and tensors), and sets `block_count` to N,
so llama.cpp and fLLM can run the same few layers of the real weights and their logits can be compared (`llama_cpp_logits.cpp`). The result
is not a language model: it is the real weights of the first layers followed by the real output head.

    python slice_qwen4exp_gguf.py URL_OR_PATH_OF_FIRST_SHARD OUT.gguf [--layers 4]

The source is read with HTTP range requests (only the tensors of the slice are transferred) or from a local file. Needs nothing but Python.
"""
from __future__ import annotations

import argparse
import struct
import sys
import urllib.request


BLOCKS = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74),
          18: (256, 98), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 42: (64, 18), 10: (256, 84), 11: (256, 110)}


class Source:
    def __init__(self, where: str):
        self.where = where
        self.local = not where.startswith("http")
        self.f = open(where, "rb") if self.local else None

    def read(self, start: int, length: int) -> bytes:
        if self.local:
            self.f.seek(start)
            return self.f.read(length)
        out = bytearray()
        while len(out) < length:
            want = min(length - len(out), 256 << 20)
            req = urllib.request.Request(self.where, headers={"Range": f"bytes={start + len(out)}-{start + len(out) + want - 1}"})
            out += urllib.request.urlopen(req, timeout=600).read()
        return bytes(out)


class Reader:
    def __init__(self, data: bytes):
        self.d, self.o = data, 0

    def take(self, n):
        b = self.d[self.o:self.o + n]
        if len(b) != n:
            raise EOFError
        self.o += n
        return b

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf8")

    def value(self, t):
        if t == 8: return self.string()
        if t == 9:
            et = self.u32()
            n = self.u64()
            return et, [self.value(et) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--header-bytes", type=int, default=48 << 20)
    args = ap.parse_args()
    src = Source(args.source)
    rd = Reader(src.read(0, args.header_bytes))
    assert rd.take(4) == b"GGUF"
    rd.u32()
    n_tensors, n_kv = rd.u64(), rd.u64()
    kvs = []
    for _ in range(n_kv):
        key, t = rd.string(), rd.u32()
        kvs.append((key, t, rd.value(t)))
    infos = []
    for _ in range(n_tensors):
        name = rd.string()
        dims = [rd.u64() for _ in range(rd.u32())]
        ty, off = rd.u32(), rd.u64()
        infos.append((name, dims, ty, off))
    data_start = (rd.o + 31) // 32 * 32

    n = args.layers
    keep_kv = []
    for key, t, val in kvs:
        if key.startswith(("general.architecture", "split.", "GGUF.", "qwen4exp.ple.")) or key == "qwen4exp.embedding_length_per_layer_input":
            continue
        if key == "qwen4exp.block_count":
            val = n
        elif key == "qwen4exp.attention.compress_ratios":
            val = (val[0], val[1][:n])
        keep_kv.append((key, t, val))
    keep_kv.insert(0, ("general.architecture", 8, "qwen4exp"))
    keep = lambda name: not name.startswith("per_layer_token_embd") and (not name.startswith("blk.") or (int(name.split(".")[1]) < n and ".ple_" not in name))
    tensors = []
    for name, dims, ty, off in infos:
        if not keep(name):
            continue
        elems = 1
        for d in dims:
            elems *= d
        block, size = BLOCKS[ty]
        length = elems // block * size
        tensors.append((name, dims, ty, src.read(data_start + off, length)))
        print(f"{name} {dims} type {ty} {length >> 10} KiB", file=sys.stderr)

    def enc_string(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def enc_value(t: int, v) -> bytes:
        if t == 8:
            return enc_string(v)
        if t == 9:
            et, items = v
            return struct.pack("<IQ", et, len(items)) + b"".join(enc_value(et, x) for x in items)
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.pack(fmt, v)

    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(keep_kv))
    for key, t, v in keep_kv:
        head += enc_string(key) + struct.pack("<I", t) + enc_value(t, v)
    offset = 0
    offsets = []
    for name, dims, ty, data in tensors:
        offsets.append(offset)
        offset += (len(data) + 31) // 32 * 32
    for (name, dims, ty, data), off in zip(tensors, offsets):
        head += enc_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, off)
    with open(args.output, "wb") as out:
        out.write(head)
        out.write(b"\0" * ((-len(head)) % 32))
        for name, dims, ty, data in tensors:
            out.write(data)
            out.write(b"\0" * ((-len(data)) % 32))
    print(f"{args.output}: {sum(len(t[3]) for t in tensors) >> 20} MiB of tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
