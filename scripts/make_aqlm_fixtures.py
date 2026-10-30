#!/usr/bin/env python3
"""Tiny AQLM layers (random codes and codebooks in the layout of `aqlm.QuantizedLinear`) with the dequantization of the aqlm library
(`aqlm.utils._dequantize_weight`), for the converter's unit test (rust/checkpoint/src/dequant_bits.rs).

    pip install aqlm torch safetensors
    python make_aqlm_fixtures.py <output dir>      (merges into <dir>/cases.json)"""
import json
import os
import sys

import torch
from aqlm.utils import _dequantize_weight, get_int_dtype, pack_int_data, unpack_int_data
from safetensors.torch import save_file

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
cases_path = os.path.join(out, "cases.json")
cases = json.load(open(cases_path)) if os.path.exists(cases_path) else {}
torch.manual_seed(21)
for name, (in_f, out_f, ig, og, ncb, nbits) in {
    "aqlm_2x8": (64, 16, 8, 1, 2, 8),
    "aqlm_1x12_og2": (32, 8, 4, 2, 1, 12),
}.items():
    size = 2**nbits
    codes = torch.randint(0, size, (out_f // og, in_f // ig, ncb))
    codebooks = (torch.randn(ncb, size, og, ig) * 0.1).half()
    scales = (torch.rand(out_f // og, 1, 1, 1) + 0.5).half()
    stored = pack_int_data(codes.clone(), nbits).to(get_int_dtype(nbits))
    # the library computes in the dtype of the codebooks: use float32 copies of the stored half-precision values
    expected = _dequantize_weight(unpack_int_data(stored, nbits), codebooks.float(), scales.float())
    save_file({"layer.codes": stored.contiguous(), "layer.codebooks": codebooks.contiguous(), "layer.scales": scales.contiguous(), "expected": expected.contiguous()}, os.path.join(out, name + ".safetensors"))
    cases[name] = {"format": "aqlm", "in_features": in_f, "out_features": out_f, "in_group_size": ig, "out_group_size": og, "num_codebooks": ncb, "nbits": nbits}
    print(name, str(stored.dtype), tuple(stored.shape), tuple(expected.shape))
json.dump(cases, open(cases_path, "w"), indent=1, sort_keys=True)
