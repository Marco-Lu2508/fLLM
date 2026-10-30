#!/usr/bin/env python3
"""Tiny compressed-tensors integer checkpoints (asymmetric zero points, 4 and 8 bits, group and channel strategies),
produced by the compressed-tensors library itself, with the dequantization the library computes.

Output: <dir>/<case>.safetensors (tensors under the module prefix `layer.` plus `expected`, float32 [out, in]) and the case
list merged into <dir>/cases.json. Usage: make_ct_int_fixtures.py <output dir>   (the checked-in copy is
rust/checkpoint/testdata/quant)."""
import json
import os
import sys

import torch
from safetensors.torch import save_file
from compressed_tensors.compressors import PackedQuantizationCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.lifecycle.forward import dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
cases_path = os.path.join(out, "cases.json")
cases = json.load(open(cases_path)) if os.path.exists(cases_path) else {}
torch.manual_seed(77)

ROWS, COLS = 12, 64
for bits in (4, 8):
    for strategy, group in (("group", 16), ("channel", None)):
        w = torch.randn(ROWS, COLS) * 0.2 + 0.05
        w[1, 3] = 1.7
        args = QuantizationArgs(num_bits=bits, type="int", symmetric=False, strategy=strategy, group_size=group)
        scheme = QuantizationScheme(targets=["Linear"], weights=args)
        grouped = w.reshape(ROWS, COLS // group, group) if group else w.reshape(ROWS, 1, COLS)
        scale, zp = calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)
        state = PackedQuantizationCompressor.compress({"weight": w, "weight_scale": scale, "weight_zero_point": zp}, scheme)
        # the library's own decompression of what it just wrote
        dec = PackedQuantizationCompressor.decompress({k: v for k, v in state.items()}, scheme)["weight"].float()
        assert dec.shape == w.shape and (dec - w).abs().max() < 0.2, "fixture sanity"
        tensors = {"layer." + k: v.contiguous() for k, v in state.items()}
        tensors["expected"] = dec.contiguous()
        case = f"ct_asym_int{bits}_{strategy}"
        save_file(tensors, os.path.join(out, case + ".safetensors"))
        cases[case] = {"format": "ct_asym", "bits": bits, "strategy": strategy, "group_size": group or COLS, "shape": [ROWS, COLS]}
        print(case, {k: (str(v.dtype), tuple(v.shape)) for k, v in tensors.items()})

json.dump(cases, open(cases_path, "w"), indent=1, sort_keys=True)
