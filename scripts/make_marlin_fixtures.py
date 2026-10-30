#!/usr/bin/env python3
"""Tiny GPTQ-Marlin checkpoints (symmetric, 4 and 8 bits) for the converter's unit test (rust/checkpoint/src/dequant_bits.rs).

The Marlin layout is produced by the permutation code of vLLM (`marlin_weights`, `get_weight_perm` of
vllm/model_executor/layers/quantization/utils/marlin_utils_test.py and `get_scale_perms`, `marlin_permute_scales` of marlin_utils.py,
Apache-2.0, Copyright contributors to the vLLM project), reproduced here with torch and numpy only. `expected` is the plain
dequantization `(q - 2^(bits-1)) * scale`, float32 [out, in].

    python make_marlin_fixtures.py <output dir>      (merges into <dir>/cases.json)"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

TILE = 16


def get_weight_perm(num_bits):
    perm_list = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])
    perm = np.array(perm_list)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7]) if num_bits == 4 else np.array([0, 2, 1, 3])
    return torch.from_numpy(perm.reshape((-1, len(interleave)))[:, interleave].ravel())


def marlin_weights(q_w, size_k, size_n, num_bits, perm):
    q_w = q_w.reshape((size_k // TILE, TILE, size_n // TILE, TILE)).permute((0, 2, 1, 3)).reshape((size_k // TILE, size_n * TILE))
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)
    pack_factor = 32 // num_bits
    q = q_w.cpu().numpy().astype(np.uint32)
    packed = np.zeros((q.shape[0], q.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        packed |= q[:, i::pack_factor] << num_bits * i
    return torch.from_numpy(packed.astype(np.int32))


def get_scale_perms():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s, size_k, size_n, group_size):
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
cases_path = os.path.join(out, "cases.json")
cases = json.load(open(cases_path)) if os.path.exists(cases_path) else {}
torch.manual_seed(5)
K, N = 256, 128
for name, bits, group, names in (("marlin_4bit_g128", 4, 128, ("B", "s")), ("marlin_4bit_channel", 4, -1, ("B", "s")), ("marlin_8bit_g64", 8, 64, ("qweight", "scales"))):
    w = torch.randn(K, N) * 0.1  # [in, out] like vLLM
    g = K if group == -1 else group
    maxq, offset = 2**bits - 1, 2 ** (bits - 1)
    wg = w.reshape(K // g, g, N)
    s = (wg.abs().amax(1) / (offset - 1 if bits == 4 else offset - 1)).clamp(min=1e-6).half()  # [groups, N]
    q = torch.clamp(torch.round(wg / s.float().unsqueeze(1)) + offset, 0, maxq).to(torch.int32).reshape(K, N)
    expected = ((q.float().reshape(K // g, g, N) - offset) * s.float().unsqueeze(1)).reshape(K, N).t().contiguous()  # [out, in]
    marlin_q = marlin_weights(q, K, N, bits, get_weight_perm(bits))
    marlin_s = marlin_permute_scales(s, K, N, group)
    save_file({f"layer.{names[0]}": marlin_q.contiguous(), f"layer.{names[1]}": marlin_s, "expected": expected}, os.path.join(out, name + ".safetensors"))
    cases[name] = {"format": "gptq_marlin", "bits": bits, "group_size": group, "names": list(names), "shape": [N, K]}
    print(name, tuple(marlin_q.shape), tuple(marlin_s.shape))
json.dump(cases, open(cases_path, "w"), indent=1, sort_keys=True)
