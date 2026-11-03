#!/usr/bin/env python3
"""Prints the HIP kernel sources of rust/engine/build.rs (hip_kernels), one per line, relative to rust/engine: cuda/api.cu, hip/hip_gemm.cu, ..."""
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
body = text[text.index("fn hip_kernels()"):]
out = []
for name, folder in (("let sources", "cuda"), ("let hip_sources", "hip")):
    m = re.search(name + r"\s*=\s*\[(.*?)\];", body, re.S)
    out += [f"{folder}/{x}" for x in re.findall(r'"([^"]+\.cu)"', m.group(1))]
print("\n".join(out))
