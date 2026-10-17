#!/usr/bin/env python3
"""Writes tiny quantized checkpoints produced by the real libraries, with the dequantized values the library itself
computes, for the converter's unit tests (rust/checkpoint/src/dequant.rs and convert.rs).

Output: <dir>/<case>.safetensors holding the tensors a checkpoint of that format stores (Hugging Face names under the
module prefix `layer.`) plus `expected` (float32 [out, in], the library's dequantization). `cases.json` lists the cases.

Libraries: bitsandbytes (CPU), compressed-tensors, torch. Usage: make_quant_fixtures.py <output dir>"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
cases = {}
torch.manual_seed(1234)


def save(case, tensors, expected, meta):
    t = {k: v.contiguous() for k, v in tensors.items()}
    t["expected"] = expected.to(torch.float32).contiguous()
    save_file(t, os.path.join(out, case + ".safetensors"))
    cases[case] = meta
    print(case, {k: (str(v.dtype), tuple(v.shape)) for k, v in t.items()})


# ---------------------------------------------------------------- bitsandbytes
import bitsandbytes.functional as F  # noqa: E402

for qt in ("nf4", "fp4"):
    for dq in (False, True):
        for bs in (64, 128):
            w = torch.randn(16, 256) * 0.05
            w[3, 7] = 1.9  # an outlier that dominates its block
            packed, state = F.quantize_4bit(w, blocksize=bs, compress_statistics=dq, quant_type=qt)
            expected = F.dequantize_4bit(packed, state)
            sd = state.as_dict(packed=True)
            tensors = {"layer.weight": packed}
            for k, v in sd.items():
                tensors["layer.weight." + k] = v
            save(f"bnb_{qt}{'_dq' if dq else ''}_bs{bs}", tensors, expected, {"format": "bnb4", "quant_type": qt, "double_quant": dq, "blocksize": bs, "shape": [16, 256]})

# LLM.int8(): int8 weight + per-row absmax statistics (SCB)
w = (torch.randn(12, 96) * 0.1).half()
w[2, 5] = 3.0
q, stats, _ = F.int8_vectorwise_quant(w)
expected = F.int8_vectorwise_dequant(q, stats)
save("bnb_int8", {"layer.weight": q, "layer.SCB": stats.float()}, expected, {"format": "bnb8", "shape": [12, 96]})

# ---------------------------------------------------------------- compressed-tensors NVFP4 / MXFP4
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor  # noqa: E402
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor  # noqa: E402
from compressed_tensors.quantization import QuantizationScheme  # noqa: E402
from compressed_tensors.quantization.quant_scheme import MXFP4, NVFP4  # noqa: E402
from compressed_tensors.quantization.utils.helpers import calculate_qparams, generate_gparam  # noqa: E402

rows, cols = 8, 128
w = torch.randn(rows, cols) * 0.2
w[1, 3] = 2.5

scheme = QuantizationScheme(**NVFP4, targets=["Linear"])
args = scheme.weights
grouped = w.reshape(rows, cols // 16, 16)
gmin, gmax = torch.aminmax(w)
global_scale = generate_gparam(gmin.reshape(1), gmax.reshape(1))
mins, maxs = grouped.amin(-1), grouped.amax(-1)
scale, zp = calculate_qparams(mins, maxs, args, global_scale=global_scale)
state = NVFP4PackedCompressor.compress({"weight": w, "weight_scale": scale, "weight_global_scale": global_scale}, scheme)
# the library's own dequantization at float32 precision (its decompress() rounds to bfloat16)
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8  # noqa: E402
from compressed_tensors.quantization.lifecycle.forward import dequantize  # noqa: E402

unpacked = unpack_fp4_from_uint8(state["weight_packed"], rows, cols, dtype=torch.float32)
dec = {"weight": dequantize(x_q=unpacked, scale=state["weight_scale"].to(torch.float32), args=scheme.weights, global_scale=global_scale.float(), dtype=torch.float32)}
tensors = {"layer.weight_packed": state["weight_packed"], "layer.weight_scale": state["weight_scale"], "layer.weight_global_scale": global_scale.float()}
inp = torch.tensor([448.0 * 6.0 / 3.1]).float()  # input_global_scale = (448 * 6) / amax
tensors["layer.input_global_scale"] = inp
save("ct_nvfp4", tensors, dec["weight"].float(), {"format": "ct_nvfp4", "shape": [rows, cols]})

scheme = QuantizationScheme(**MXFP4, targets=["Linear"])
args = scheme.weights
grouped = w.reshape(rows, cols // 32, 32)
mins, maxs = grouped.amin(-1), grouped.amax(-1)
from compressed_tensors.quantization.utils.mxfp_utils import generate_mx_scales  # noqa: E402

scale = generate_mx_scales(torch.maximum(mins.abs(), maxs.abs())) if True else None
state = MXFP4PackedCompressor.compress({"weight": w, "weight_scale": scale}, scheme)
from compressed_tensors.compressors.mx_utils import decompress_mx_scale  # noqa: E402

unpacked = unpack_fp4_from_uint8(state["weight_packed"], rows, cols, dtype=torch.float32)
dec = {"weight": dequantize(x_q=unpacked, scale=decompress_mx_scale(state["weight_scale"]).to(torch.float32), args=scheme.weights, dtype=torch.float32)}
save("ct_mxfp4", {"layer.weight_packed": state["weight_packed"], "layer.weight_scale": state["weight_scale"]}, dec["weight"].float(), {"format": "ct_mxfp4", "shape": [rows, cols]})

# ---------------------------------------------------------------- GPTQ (gptqmodel packing, 2/3/4/8 bits, v1 and v2 zero points)
from gptqmodel.nn_modules.qlinear.torch import TorchLinear  # noqa: E402

IN, OUT, GROUP = 128, 32, 32
for bits in (2, 3, 4, 8):
    maxq = 2**bits - 1
    for sym, desc, name in ((False, True, "asym_desc"), (True, False, "sym")):
        W = torch.randn(OUT, IN) * 0.1
        g_idx = torch.arange(IN) // GROUP
        if desc:
            g_idx = g_idx[torch.randperm(IN)]
        # per-group quantization parameters over the columns that the group owns
        scales = torch.zeros(OUT, IN // GROUP)
        zeros = torch.zeros(OUT, IN // GROUP)
        for g in range(IN // GROUP):
            cols = (g_idx == g).nonzero().flatten()
            lo = torch.minimum(W[:, cols].amin(1), torch.zeros(OUT))
            hi = torch.maximum(W[:, cols].amax(1), torch.zeros(OUT))
            if sym:
                s = torch.maximum(-lo, hi) * 2 / maxq
                z = torch.full((OUT,), float((maxq + 1) // 2))
            else:
                s = (hi - lo) / maxq
                z = torch.round(-lo / s)
            scales[:, g], zeros[:, g] = s, z
        scales = scales.half().float()
        lin = torch.nn.Linear(IN, OUT, bias=False)
        lin.weight.data = W.clone()
        layer = TorchLinear(bits=bits, group_size=GROUP, sym=sym, desc_act=desc, in_features=IN, out_features=OUT, bias=False, pack_dtype=torch.int32)
        layer.pack(lin, scales, zeros, g_idx.int())
        # the stored v1 format subtracts one from every zero point; pack a second layer to get those qzeros
        layer1 = TorchLinear(bits=bits, group_size=GROUP, sym=sym, desc_act=desc, in_features=IN, out_features=OUT, bias=False, pack_dtype=torch.int32)
        layer1.pack(lin, scales, torch.remainder(zeros - 1, maxq + 1), g_idx.int())
        q = torch.clamp(torch.round(W / scales[:, g_idx.long()] + zeros[:, g_idx.long()]), 0, maxq)
        expected = ((q - zeros[:, g_idx.long()]) * scales[:, g_idx.long()]).t().contiguous()  # [in, out] like the library
        for v2, qz in ((True, layer.qzeros), (False, layer1.qzeros)):
            case = f"gptq_{bits}bit_{name}_{'v2' if v2 else 'v1'}"
            tensors = {"layer.qweight": layer.qweight.int(), "layer.qzeros": qz.int(), "layer.scales": layer.scales.half(), "layer.g_idx": layer.g_idx.int()}
            save(case, tensors, expected, {"format": "gptq", "bits": bits, "group_size": GROUP, "v2": v2, "in_features": IN, "out_features": OUT})

# ---------------------------------------------------------------- MLX affine quantization
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

for bits in (2, 3, 4, 5, 6, 8):
    for group in (32, 64):
        w = mx.random.normal((8, 128)) * 0.3
        q, s, b = mx.quantize(w, group_size=group, bits=bits)
        d = mx.dequantize(q, s, b, group_size=group, bits=bits)
        tensors = {"layer.weight": torch.from_numpy(np.array(q)), "layer.scales": torch.from_numpy(np.array(s)), "layer.biases": torch.from_numpy(np.array(b))}
        save(f"mlx_{bits}bit_g{group}", tensors, torch.from_numpy(np.array(d)), {"format": "mlx", "bits": bits, "group_size": group})

json.dump(cases, open(os.path.join(out, "cases.json"), "w"), indent=1, sort_keys=True)
