#!/usr/bin/env python3
"""Write a `qwen4exp` GGUF with the tensor names, shapes and quantization types of a real file but random weights, for speed work.

The header (metadata, tokenizer, tensor list) is read from a real `qwen4exp` GGUF (a URL or a local first shard; only the first tens of
megabytes are transferred) and the first `--layers` layers are written with random but valid bytes: every block of a quantized tensor has a
finite small scale, floats are small normal numbers. The n-gram table is written as a sparse hole (its bytes read as zero and take no disk
space), because the model reads a handful of its rows per token. The file is not a language model; it has the byte sizes, the access pattern
and the per-token work of the real one, so decode and verification speed, memory behaviour and the expert cache can be measured without the
75 GB download. Routing is random (every expert equally likely), which is the worst case for an expert cache.

    python synth_qwen4exp_gguf.py URL_OR_PATH_OF_FIRST_SHARD OUT.gguf [--layers 12] [--no-ple]

Needs nothing but Python 3.
"""
from __future__ import annotations

import argparse
import struct
import sys
import urllib.request

import array
import os
import random

# GGML type -> (values per block, bytes per block, offset of the fp16 scale `d` or None for a float type)
BLOCKS = {0: (1, 4, None), 1: (1, 2, None), 30: (1, 2, None), 8: (32, 34, 0), 10: (256, 84, 80), 11: (256, 110, 108), 12: (256, 144, 0),
          13: (256, 176, 0), 14: (256, 210, 208), 16: (256, 66, 0), 17: (256, 74, 0), 18: (256, 98, 0), 20: (32, 18, 0), 21: (256, 110, 0),
          22: (256, 82, 0), 23: (256, 136, 0), 42: (64, 18, 0)}
Q4K_LIKE = (12, 13)  # d and dmin
FP16_SCALE = struct.pack("<e", 0.004)


class Source:
    def __init__(self, where: str):
        self.where = where
        self.local = not where.startswith("http")
        self.f = open(where, "rb") if self.local else None

    def read(self, start: int, length: int) -> bytes:
        if self.local:
            self.f.seek(start)
            return self.f.read(length)
        req = urllib.request.Request(self.where, headers={"Range": f"bytes={start}-{start + length - 1}"})
        return urllib.request.urlopen(req, timeout=600).read()


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
    def string(self): return self.take(self.u64()).decode("utf8")

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et = self.u32()
            n = self.u64()
            return et, [self.value(et) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]


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


def pool_for(ty: int, rng: random.Random, blocks: int = 1 << 14) -> bytes:
    """`blocks` random blocks of GGML type `ty` with finite scales."""
    values, size, d_at = BLOCKS[ty]
    if d_at is None:
        n = blocks * size // (4 if ty == 0 else 2)
        x = [rng.gauss(0.0, 0.05) for _ in range(n)]
        if ty == 0:
            return array.array("f", x).tobytes()
        if ty == 1:
            return b"".join(struct.pack("<e", v) for v in x)
        return b"".join(struct.pack("<f", v)[2:] for v in x)  # bf16: the high half of the float
    raw = bytearray(os.urandom(blocks * size))
    for at in ([d_at] + ([2] if ty in Q4K_LIKE else []) + ([82] if ty == 10 else [])):
        for k in range(2):
            raw[at + k::size] = FP16_SCALE[k:k + 1] * blocks
    return bytes(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--header-bytes", type=int, default=48 << 20)
    ap.add_argument("--no-ple", action="store_true", help="leave out the n-gram layer and its table")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    n = args.layers
    rng = random.Random(args.seed)
    rd = Reader(Source(args.source).read(0, args.header_bytes))
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
        ty, _off = rd.u32(), rd.u64()
        infos.append((name, dims, ty))
    meta = {k: v for k, _, v in kvs}
    ple_rows = max(o + s for o, s in zip(meta["qwen4exp.ple.head_offsets"][1], meta["qwen4exp.ple.head_vocab_sizes"][1]))
    ple_dim = meta["qwen4exp.embedding_length_per_layer_input"]

    keep_kv = []
    for key, t, val in kvs:
        if key.startswith(("split.", "GGUF.")):
            continue
        if key == "qwen4exp.block_count":
            val = n
        elif key == "qwen4exp.attention.compress_ratios":
            val = (val[0], val[1][:n])
        elif args.no_ple and key.startswith("qwen4exp.ple.") or (args.no_ple and key == "qwen4exp.embedding_length_per_layer_input"):
            continue
        elif key == "qwen4exp.ple.layers" and val[1] and val[1][0] >= n:
            continue
        keep_kv.append((key, t, val))
    keep = lambda name: not (args.no_ple and (".ple_" in name)) and (not name.startswith("blk.") or int(name.split(".")[1]) < n)
    tensors = [(name, dims, ty) for name, dims, ty in infos if keep(name)]
    if not args.no_ple:
        tensors.append(("per_layer_token_embd.weight", [ple_dim, ple_rows], 20))

    def nbytes(dims, ty):
        e = 1
        for d in dims:
            e *= d
        v, s, _ = BLOCKS[ty]
        return e // v * s

    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(keep_kv))
    for key, t, v in keep_kv:
        head += enc_string(key) + struct.pack("<I", t) + enc_value(t, v)
    offset, offsets = 0, []
    for name, dims, ty in tensors:
        offsets.append(offset)
        offset += (nbytes(dims, ty) + 31) // 32 * 32
    # the layout of the tensor infos does not depend on the offsets values, so the header size is known after one pass
    for (name, dims, ty), off in zip(tensors, offsets):
        head += enc_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, off)
    head += b"\0" * ((-len(head)) % 32)
    pools = {}
    total = 0
    with open(args.output, "wb") as out:
        out.write(head)
        for (name, dims, ty), off in zip(tensors, offsets):
            length = nbytes(dims, ty)
            out.seek(len(head) + off)
            if name == "per_layer_token_embd.weight":
                out.seek(len(head) + off + length - 1)
                out.write(b"\0")
                continue
            if ty not in pools:
                pools[ty] = pool_for(ty, rng)
            pool = pools[ty]
            _, size, _ = BLOCKS[ty]
            # whole blocks from the pool, starting at a random block, so that tensors do not all hold the same bytes
            pos = rng.randrange(len(pool) // size) * size
            left = length
            while left > 0:
                take = min(left, len(pool) - pos)
                out.write(memoryview(pool)[pos:pos + take])
                left -= take
                pos = 0
            total += length
        out.seek(0, 2)
        end = len(head) + offsets[-1] + nbytes(tensors[-1][1], tensors[-1][2])
        out.truncate(end)
    print(f"{args.output}: {n} layers, {total / 2**30:.2f} GiB written, {end / 2**30:.2f} GiB file size (n-gram table is a sparse hole: {not args.no_ple})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
