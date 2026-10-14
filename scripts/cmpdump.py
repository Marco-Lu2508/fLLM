#!/usr/bin/env python3
"""Compare two directories written with FLLM_DUMP_LAYERS (CPU decoder, one run each).

usage: cmpdump.py DIR_A DIR_B [FILE_PREFIX]

For every `l<layer>_<tag>.f32` file present in DIR_A, prints the relative L2 difference and the
largest absolute difference against the same file in DIR_B. Tags: normed (attention-norm output),
q, k, v (projection results), attnout (attention result), branch (output projection), attn
(hidden state after the attention branch), ffn (hidden state after the FFN branch).

Typical use, the layer-by-layer growth of the difference between two CPU tiers:
  FLLM_DUMP_LAYERS=/tmp/a FLLM_CPU_ISA=avx2 fllm-rs serve ...      (one short request)
  FLLM_DUMP_LAYERS=/tmp/b FLLM_CPU_ISA=avx512-vnni fllm-rs serve ... (the same request)
  cmpdump.py /tmp/a /tmp/b l0_
"""
import array
import math
import os
import sys


def load(path):
    values = array.array("f")
    with open(path, "rb") as handle:
        values.frombytes(handle.read())
    return values


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, b = sys.argv[1:3]
    prefix = sys.argv[3] if len(sys.argv) > 3 else ""
    names = [n for n in os.listdir(a) if n.startswith(prefix) and n.endswith(".f32")]
    for name in sorted(names, key=lambda n: (int(n[1 : n.index("_")]), n)):
        x, y = load(os.path.join(a, name)), load(os.path.join(b, name))
        if len(x) != len(y):
            print(f"{name:16s} length differs: {len(x)} against {len(y)}")
            continue
        diff = sum((p - q) ** 2 for p, q in zip(x, y))
        norm = sum(p * p for p in x) or 1.0
        print(
            f"{name:16s} rel-L2 {math.sqrt(diff / norm):.3e}  "
            f"max abs {max(abs(p - q) for p, q in zip(x, y)):.3e}  "
            f"(reference max {max(abs(p) for p in x):.2f})"
        )


if __name__ == "__main__":
    main()
