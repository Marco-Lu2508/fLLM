#!/usr/bin/env python3
"""Tiny MLX checkpoints in the floating-point quantization modes (`mxfp4`, `nvfp4`, `mxfp8`), written by `mlx.core.quantize`, with the
values `mlx.core.dequantize` returns. Output as in make_ct_int_fixtures.py (cases merged into <dir>/cases.json).

    pip install "mlx[cpu]" safetensors torch numpy
    python make_mlx_fp_fixtures.py <output dir>"""
import json
import os
import sys

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import save_file

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
cases_path = os.path.join(out, "cases.json")
cases = json.load(open(cases_path)) if os.path.exists(cases_path) else {}
mx.random.seed(11)
for mode, group, bits in (("mxfp4", 32, 4), ("nvfp4", 16, 4), ("mxfp8", 32, 8)):
    w = mx.random.normal((8, 128)) * 0.3
    q, s = mx.quantize(w, group_size=group, bits=bits, mode=mode)
    d = mx.dequantize(q, s, group_size=group, bits=bits, mode=mode).astype(mx.float32)
    tensors = {
        "layer.weight": torch.from_numpy(np.array(q)),
        "layer.scales": torch.from_numpy(np.array(s)),
        "expected": torch.from_numpy(np.array(d)),
    }
    save_file(tensors, os.path.join(out, f"mlx_{mode}.safetensors"))
    cases[f"mlx_{mode}"] = {"format": "mlx_fp", "mode": mode, "bits": bits, "group_size": group, "shape": [8, 128]}
    print(mode, {k: (str(v.dtype), tuple(v.shape)) for k, v in tensors.items()})
json.dump(cases, open(cases_path, "w"), indent=1, sort_keys=True)
