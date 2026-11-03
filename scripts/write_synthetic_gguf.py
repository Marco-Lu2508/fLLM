#!/usr/bin/env python3
"""Write a synthetic llama-architecture GGUF whose size and quantization mix you choose, with random weights.

The file is for load, memory-plan and upload tests (does a model of this size and these formats load, how long does the first token take,
what happens when it is larger than the memory), not for output quality: the weights are random bytes with a small valid block scale, the
tokenizer has 256 byte tokens plus a few specials. Needs only numpy (no torch, no gguf package).

    python write_synthetic_gguf.py OUT.gguf --layers 24 --hidden 2048 --ffn 8192 --gate-up iq3_xxs --down q2_0

Formats (name: values per block, bytes per block): f16, q8_0 (32, 34), q4_k (256, 144), q2_k (256, 84), iq3_xxs (256, 98), iq3_s (256, 110),
iq2_xxs (256, 66), iq2_xs (256, 74), iq2_s (256, 82), iq1_s (256, 50), iq1_m (256, 56), q2_0 (64, 18). Every row length must be a multiple of
the block size. A file is as large as the weights: sparse files do not exist on every system, so the size you ask for is the size you get.
"""
import argparse
import struct
import sys

import numpy as np

# name: (ggml type id, values per block, bytes per block, offset of the f16 block scale or None)
FORMATS = {
    "f32": (0, 1, 4, None),
    "f16": (1, 1, 2, None),
    "q8_0": (8, 32, 34, 0),
    "q2_k": (10, 256, 84, 80),
    "q4_k": (12, 256, 144, 0),
    "iq2_xxs": (16, 256, 66, 0),
    "iq2_xs": (17, 256, 74, 0),
    "iq3_xxs": (18, 256, 98, 0),
    "iq1_s": (19, 256, 50, 0),
    "iq3_s": (21, 256, 110, 0),
    "iq2_s": (22, 256, 82, 0),
    "iq1_m": (29, 256, 56, None),
    "q2_0": (42, 64, 18, 0),
}


def kv_bytes(key, kind, value):
    k = key.encode()
    out = struct.pack("<Q", len(k)) + k
    if kind == "u32":
        return out + struct.pack("<II", 4, value)
    if kind == "f32":
        return out + struct.pack("<If", 6, value)
    if kind == "str":
        v = value.encode()
        return out + struct.pack("<IQ", 8, len(v)) + v
    if kind == "arr_str":
        body = b"".join(struct.pack("<Q", len(s.encode())) + s.encode() for s in value)
        return out + struct.pack("<IIQ", 9, 8, len(value)) + body
    if kind == "arr_f32":
        return out + struct.pack("<IIQ", 9, 6, len(value)) + struct.pack(f"<{len(value)}f", *value)
    if kind == "arr_i32":
        return out + struct.pack("<IIQ", 9, 5, len(value)) + struct.pack(f"<{len(value)}i", *value)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--ffn", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--attn", default="q4_k", choices=sorted(FORMATS))
    ap.add_argument("--gate-up", default="iq3_xxs", choices=sorted(FORMATS))
    ap.add_argument("--down", default="q2_0", choices=sorted(FORMATS))
    ap.add_argument("--embd", default="q8_0", choices=sorted(FORMATS))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    head_dim = a.hidden // a.heads
    kv_dim = head_dim * a.kv_heads

    # (name, shape as in the file = [columns, rows], format)
    tensors = [("token_embd.weight", [a.hidden, a.vocab], a.embd)]
    for i in range(a.layers):
        p = f"blk.{i}."
        tensors += [
            (p + "attn_norm.weight", [a.hidden], "f32"),
            (p + "attn_q.weight", [a.hidden, a.hidden], a.attn),
            (p + "attn_k.weight", [a.hidden, kv_dim], a.attn),
            (p + "attn_v.weight", [a.hidden, kv_dim], a.attn),
            (p + "attn_output.weight", [a.hidden, a.hidden], a.attn),
            (p + "ffn_norm.weight", [a.hidden], "f32"),
            (p + "ffn_gate.weight", [a.hidden, a.ffn], a.gate_up),
            (p + "ffn_up.weight", [a.hidden, a.ffn], a.gate_up),
            (p + "ffn_down.weight", [a.ffn, a.hidden], a.down),
        ]
    tensors += [("output_norm.weight", [a.hidden], "f32"), ("output.weight", [a.hidden, a.vocab], a.embd)]

    def nbytes(shape, fmt):
        _, block, width, _ = FORMATS[fmt]
        n = int(np.prod(shape))
        if shape[0] % block:
            sys.exit(f"row length {shape[0]} is not a multiple of {block} ({fmt})")
        return n // block * width

    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    tokens += [f"tok{i}" for i in range(a.vocab - len(tokens))]
    types = [2, 3, 3] + [6] * 256 + [1] * (a.vocab - 259)
    meta = [
        kv_bytes("general.architecture", "str", "llama"),
        kv_bytes("general.name", "str", "synthetic"),
        kv_bytes("llama.context_length", "u32", 4096),
        kv_bytes("llama.embedding_length", "u32", a.hidden),
        kv_bytes("llama.block_count", "u32", a.layers),
        kv_bytes("llama.feed_forward_length", "u32", a.ffn),
        kv_bytes("llama.attention.head_count", "u32", a.heads),
        kv_bytes("llama.attention.head_count_kv", "u32", a.kv_heads),
        kv_bytes("llama.attention.layer_norm_rms_epsilon", "f32", 1e-5),
        kv_bytes("llama.rope.freq_base", "f32", 10000.0),
        kv_bytes("llama.vocab_size", "u32", a.vocab),
        kv_bytes("tokenizer.ggml.model", "str", "llama"),
        kv_bytes("tokenizer.ggml.tokens", "arr_str", tokens),
        kv_bytes("tokenizer.ggml.scores", "arr_f32", [0.0] * a.vocab),
        kv_bytes("tokenizer.ggml.token_type", "arr_i32", types),
        kv_bytes("tokenizer.ggml.bos_token_id", "u32", 1),
        kv_bytes("tokenizer.ggml.eos_token_id", "u32", 2),
        kv_bytes("tokenizer.ggml.unknown_token_id", "u32", 0),
    ]
    align = 32
    infos, offset = [], 0
    for name, shape, fmt in tensors:
        n = name.encode()
        infos.append(struct.pack("<Q", len(n)) + n + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
                     + struct.pack("<IQ", FORMATS[fmt][0], offset))
        offset += -(-nbytes(shape, fmt) // align) * align
    rng = np.random.default_rng(a.seed)
    pool = rng.integers(0, 256, size=1 << 24, dtype=np.uint8)
    with open(a.output, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)))
        for m in meta:
            f.write(m)
        for i in infos:
            f.write(i)
        f.write(b"\0" * (-f.tell() % align))
        written = 0
        for name, shape, fmt in tensors:
            tid, block, width, scale_at = FORMATS[fmt]
            total = nbytes(shape, fmt)
            if fmt == "f32":
                f.write(np.ones(int(np.prod(shape)), dtype="<f4").tobytes())
            else:
                done = 0
                step = (len(pool) // width) * width
                while done < total:
                    take = min(step, total - done)
                    chunk = np.roll(pool, int(rng.integers(0, 1 << 20)))[:take].copy()
                    if scale_at is not None and width > 2:
                        view = chunk[: take // width * width].reshape(-1, width)
                        halves = 2 if fmt in ("q2_k", "q4_k") else 1  # d and dmin
                        view[:, scale_at:scale_at + 2 * halves] = np.tile(np.frombuffer(np.float16(0.02).tobytes(), dtype=np.uint8), halves)
                    f.write(chunk.tobytes())
                    done += take
            pad = -total % align
            f.write(b"\0" * pad)
            written += total + pad
    print(f"{a.output}: {len(tensors)} tensors, {written / 2**30:.2f} GiB of weights")


if __name__ == "__main__":
    main()
