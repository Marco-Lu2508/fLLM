#!/usr/bin/env python3
"""Regenerate the tiny Gemma 4 / Gemma 3n fixtures that the Rust tests read.

    python scripts/make_gemma_fixtures.py rust/engine/tests/fixtures

Each fixture directory holds `config.json`, `model.safetensors` (float32, a few tens of kilobytes) and `golden.json`
(token ids and the float32 logits of every position computed by transformers with eager attention). The Rust tests
(`rust/engine/src/gemma_tests.rs`) write a tokenizer next to them, convert the directory with the checkpoint importer,
run the engine on the CPU and compare with the golden logits, so the architectures are checked without Python and
without any download. Requires torch and transformers >= 5.19.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import transformers as T

SEED = 20260101
VOCAB = 32


def init(model: torch.nn.Module) -> None:
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                p.copy_(0.35 * torch.randn(p.shape, generator=gen))
            elif name.endswith("norm.weight") or "layernorm" in name or name.endswith("router.scale") or name.endswith("per_expert_scale"):
                p.copy_(0.8 + 0.4 * torch.rand(p.shape, generator=gen))
            elif name.endswith("correct_output_scale"):
                p.copy_(0.5 + torch.rand(p.shape, generator=gen))
            else:
                p.copy_(0.2 * torch.randn(p.shape, generator=gen))
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.copy_(0.7 + 0.6 * torch.rand(b.shape, generator=gen))


def gemma4() -> T.PreTrainedModel:
    # all features at once: two kinds of layers with different head width and KV heads, K = V on the global layers,
    # shared KV cache in the last two layers (double-wide MLP there), expert block, per-layer embeddings
    cfg = T.Gemma4TextConfig(
        vocab_size=VOCAB, vocab_size_per_layer_input=VOCAB, hidden_size=16, intermediate_size=16, num_hidden_layers=5,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, global_head_dim=16, num_global_key_value_heads=1,
        attention_k_eq_v=True, layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        sliding_window=3, hidden_size_per_layer_input=4, num_kv_shared_layers=2, use_double_wide_mlp=True,
        enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=8, final_logit_softcapping=30.0,
        max_position_embeddings=64, rms_norm_eps=1e-6, pad_token_id=0, eos_token_id=1, bos_token_id=2,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
        },
    )
    return T.Gemma4ForCausalLM._from_config(cfg, dtype=torch.float32, attn_implementation="eager")


def gemma3n() -> T.PreTrainedModel:
    layers = 5
    cfg = T.Gemma3nTextConfig(
        vocab_size=VOCAB, vocab_size_per_layer_input=VOCAB, hidden_size=16, hidden_size_per_layer_input=4,
        intermediate_size=[16, 16, 16, 24, 24], num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, sliding_window=3,
        layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        num_kv_shared_layers=2, laurel_rank=3, altup_num_inputs=4, altup_active_idx=0, altup_correct_scale=True,
        activation_sparsity_pattern=[0.9, 0.9, 0.0, 0.0, 0.0], final_logit_softcapping=30.0, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=1, bos_token_id=2,
    )
    return T.Gemma3nForCausalLM._from_config(cfg, dtype=torch.float32, attn_implementation="eager")


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "rust/engine/tests/fixtures")
    for name, make in (("gemma4-tiny", gemma4), ("gemma3n-tiny", gemma3n)):
        torch.manual_seed(SEED)
        model = make()
        init(model)
        model.eval()
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target, safe_serialization=True)
        for extra in ("generation_config.json",):
            (target / extra).unlink(missing_ok=True)
        ids = [3, 17, 5, 9, 28, 4, 21, 12, 7, 30]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
        (target / "golden.json").write_text(json.dumps({"ids": ids, "shape": list(logits.shape), "logits": logits.reshape(-1).tolist()}))
        size = sum(p.stat().st_size for p in target.iterdir())
        print(f"{name}: {size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
