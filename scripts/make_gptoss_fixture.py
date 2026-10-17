#!/usr/bin/env python3
"""Writes a tiny gpt-oss checkpoint in the layout of the published `quant_method: mxfp4` repositories (experts as
`*_blocks` / `*_scales` U8 tensors, everything else bf16) plus the expert weights as `transformers` dequantizes them
(`convert_moe_packed_tensors`), for the converter test in rust/checkpoint/src/dequant.rs.

Usage: make_gptoss_fixture.py <output dir>"""
import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import GptOssConfig, GptOssForCausalLM
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(7)
E, H, F, L = 2, 64, 64, 2
cfg = GptOssConfig(
    vocab_size=128, hidden_size=H, intermediate_size=F, num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_local_experts=E, num_experts_per_tok=2, sliding_window=8, max_position_embeddings=256, layer_types=["sliding_attention", "full_attention"],
    rope_scaling={"rope_type": "yarn", "factor": 4.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False, "original_max_position_embeddings": 64},
)
model = GptOssForCausalLM(cfg).to(torch.bfloat16)
sd = {k: v.contiguous() for k, v in model.state_dict().items()}
tensors, expected = {}, {}
for name, value in sd.items():
    if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.down_proj"):
        base = name
        # input-major [E, in, out] in the dense layout; the quantized layout stores [E, out, in/32, 16] blocks
        e, in_dim, out_dim = value.shape
        blocks = torch.randint(0, 256, (e, out_dim, in_dim // 32, 16), dtype=torch.uint8)
        scales = torch.randint(118, 130, (e, out_dim, in_dim // 32), dtype=torch.uint8)
        tensors[base + "_blocks"] = blocks
        tensors[base + "_scales"] = scales
        expected[base] = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16).contiguous()  # [E, in, out]
    else:
        tensors[name] = value
save_file(tensors, os.path.join(out, "model.safetensors"))
save_file(expected, os.path.join(out, "expected.safetensors"))
config = cfg.to_dict()
config["quantization_config"] = {"modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"], "quant_method": "mxfp4"}
config["architectures"] = ["GptOssForCausalLM"]
config["torch_dtype"] = "bfloat16"
json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=1)
vocab = {f"t{i}": i for i in range(128)}
json.dump({"version": "1.0", "model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": []}, open(os.path.join(out, "tokenizer.json"), "w"))
json.dump({"eos_token": "t1", "bos_token": "t0"}, open(os.path.join(out, "tokenizer_config.json"), "w"))
print(sorted(tensors)[:12], {k: tuple(v.shape) for k, v in expected.items()})
