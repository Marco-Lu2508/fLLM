#!/usr/bin/env python3
"""Compare fLLM with the Kimi K3 text model's own code on a tiny random checkpoint (latent attention layers only).

The reference is the model repository's `modeling_kimi_linear.py` (moonshotai/Kimi-K3; it needs `fla` and an older transformers API, so
only the pure-torch classes are taken out of the file with `ast`): SiTU activation, attention residual blocks, latent experts with their norm, the
sigmoid router, latent attention without rotary embeddings and with an output gate. The Kimi delta layers need `fla` and are not part of
this check (their maths is covered by the Kimi-Linear, Ling-3.0 and GLM-5-Next fixtures; the full-rank output gate of K3 is the `ssm_g`
projection of Ling-3.0).

    python scripts/verify_kimi_k3.py --modeling path/to/modeling_kimi_linear.py \
        --fllm target/release/fllm-rs --example target/release/examples/verify_checkpoint
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import hf_golden  # noqa: E402

WANTED = {
    "SituAndMul", "_get_situ_activation_params", "KimiRMSNorm", "KimiBlockSparseMLP", "KimiMLP", "repeat_kv", "eager_attention_forward",
    "KimiMLAAttention", "KimiMoEGate", "KimiSparseMoeBlock", "KimiDecoderLayer", "_apply_attn_res",
}


class Config(types.SimpleNamespace):
    """The attributes the extracted classes read."""

    def is_kda_layer(self, layer_idx):
        return False

    @property
    def is_mla(self):
        return True


def load_reference(path: Path) -> dict:
    from transformers.activations import ACT2FN

    tree = ast.parse(path.read_text())
    picked = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in WANTED]
    missing = WANTED - {n.name for n in picked}
    if missing:
        raise SystemExit(f"{path}: classes not found: {sorted(missing)}")
    namespace = {
        "math": math, "torch": torch, "nn": nn, "F": F, "ACT2FN": ACT2FN, "Unpack": lambda x: x, "TransformersKwargs": dict,
        "FlashAttentionKwargs": dict, "Cache": object, "Callable": object, "ALL_ATTENTION_FUNCTIONS": {}, "KimiLinearConfig": Config,
        "KimiDeltaAttention": nn.Module, "ALL_LAYERNORM_LAYERS": [],
    }
    # `ALL_LAYERNORM_LAYERS.append(...)` and the `ACT2FN["situ"] = ...` statement of the file
    module = ast.Module(body=picked, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    ACT2FN["situ"] = namespace["SituAndMul"]
    return namespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modeling", required=True, type=Path)
    parser.add_argument("--fllm", required=True)
    parser.add_argument("--example", required=True)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    ref = load_reference(args.modeling)
    torch.manual_seed(7)
    layers, hidden, vocab = 6, 64, 320
    cfg = Config(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=24, kv_lora_rank=16,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, mla_use_nope=True, mla_use_output_gate=True, intermediate_size=128,
        moe_intermediate_size=32, num_experts=8, num_experts_per_token=2, num_shared_experts=1, first_k_dense_replace=1, moe_layer_freq=1,
        routed_scaling_factor=1.5, moe_router_activation_func="sigmoid", num_expert_group=1, topk_group=1, moe_renormalize=True,
        routed_expert_hidden_size=32, latent_moe_use_norm=True, hidden_act="situ", activation_situ_beta=2.0, activation_situ_linear_beta=3.0,
        rms_norm_eps=1e-5, attn_res_block_size=2, _attn_implementation="eager", attention_dropout=0.0, vocab_size=vocab,
        num_hidden_layers_=layers,
    )
    embed = nn.Embedding(vocab, hidden)
    blocks = nn.ModuleList([ref["KimiDecoderLayer"](cfg, i) for i in range(layers)])
    norm = ref["KimiRMSNorm"](hidden, eps=cfg.rms_norm_eps)
    out_norm = ref["KimiRMSNorm"](hidden, eps=cfg.rms_norm_eps)
    out_proj = nn.Linear(hidden, 1, bias=False)
    head = nn.Linear(hidden, vocab, bias=False)
    modules = nn.ModuleDict({"embed": embed, "blocks": blocks, "norm": norm, "out_norm": out_norm, "out_proj": out_proj, "head": head})
    with torch.no_grad():
        for name, p in modules.named_parameters():
            if p.dim() >= 2:
                p.normal_(std=0.15)
            elif name.endswith("weight"):
                p.copy_(0.8 + 0.4 * torch.rand_like(p))
            else:
                p.normal_(std=0.3)
        for name, p in modules.named_parameters():
            if "e_score_correction_bias" in name:
                p.normal_(std=0.2)
        for layer in blocks:
            # the scoring projections decide how the blocks mix: make them matter
            for key in ("self_attention_res_proj", "mlp_res_proj"):
                getattr(layer, key).weight.normal_(std=0.6)
        out_proj.weight.normal_(std=0.6)
    modules.eval()
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(4, vocab, (1, args.tokens), generator=g)

    def forward(ids):
        hs = embed(ids)
        seq = ids.shape[1]
        mask = torch.full((seq, seq), float("-inf")).triu(1)[None, None]
        block_residual = hs.new_zeros(hs.shape[0] * hs.shape[1], 0, hs.shape[2])
        for layer in blocks:
            hs, block_residual = layer(hs, attention_mask=mask, block_residual=block_residual)
        batch, length, size = hs.shape
        hs = ref["_apply_attn_res"](hs.view(-1, size), block_residual, out_proj, out_norm).view(batch, length, size)
        return head(norm(hs))

    with torch.no_grad():
        logits = forward(ids)[0].float()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        hf_golden.make_tokenizer(out, vocab)
        tensors = {"model.embed_tokens.weight": embed.weight, "model.norm.weight": norm.weight, "lm_head.weight": head.weight,
                   "model.output_attn_res_norm.weight": out_norm.weight, "model.output_attn_res_proj.weight": out_proj.weight}
        for i, layer in enumerate(blocks):
            for name, p in layer.state_dict().items():
                tensors[f"model.layers.{i}.{name}"] = p
        save_file({k: v.detach().clone().contiguous() for k, v in tensors.items()}, str(out / "model.safetensors"))
        config = {
            "model_type": "kimi_linear", "architectures": ["KimiLinearForCausalLM"], "hidden_size": hidden, "num_hidden_layers": layers,
            "num_attention_heads": 4, "num_key_value_heads": 4, "intermediate_size": 128, "moe_intermediate_size": 32, "vocab_size": vocab,
            "kv_lora_rank": 16, "q_lora_rank": 24, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16, "mla_use_nope": True,
            "mla_use_output_gate": True, "num_experts": 8, "num_experts_per_token": 2, "num_shared_experts": 1, "first_k_dense_replace": 1,
            "moe_layer_freq": 1, "routed_scaling_factor": 1.5, "moe_router_activation_func": "sigmoid", "num_expert_group": 1, "topk_group": 1,
            "moe_renormalize": True, "routed_expert_hidden_size": 32, "latent_moe_use_norm": True, "hidden_act": "situ",
            "activation_situ_beta": 2.0, "activation_situ_linear_beta": 3.0, "rms_norm_eps": 1e-5, "attn_res_block_size": 2,
            "max_position_embeddings": 256, "tie_word_embeddings": False, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 3,
            "linear_attn_config": {"kda_layers": [], "full_attn_layers": list(range(1, layers + 1)), "head_dim": 16, "num_heads": 4,
                                   "short_conv_kernel_size": 4, "use_full_rank_gate": True, "gate_lower_bound": -5.0},
        }
        (out / "config.json").write_text(json.dumps(config))
        gguf = out / "k3.gguf"
        subprocess.run([args.fllm, "convert", str(out), str(gguf), "--outtype", "f32"], check=True)
        idfile = out / "ids.json"
        idfile.write_text(json.dumps({"ids": ids[0].tolist(), "text": "hello"}))
        # a float32 KV cache: with the default half-precision cache the sigmoid router of a random model flips an expert now and then
        env = {**os.environ, "FLLM_VERIFY_KV": os.environ.get("FLLM_VERIFY_KV", "f32")}
        result = subprocess.run([args.example, str(gguf), str(idfile)], check=True, capture_output=True, text=True, env=env)
        mine = torch.tensor(json.loads(result.stdout.strip().splitlines()[-1])["logits"]).reshape(logits.shape)
    span = float(logits.max() - logits.min())
    worst = float((mine - logits).abs().max()) / span
    if args.verbose:
        print("per position:", [round(float((mine[i] - logits[i]).abs().max()) / span, 5) for i in range(args.tokens)])
    argmax = int((mine.argmax(-1) != logits.argmax(-1)).sum())
    print(json.dumps({"arch": "kimi_k3 (latent attention layers)", "relative": worst, "argmax_mismatches": argmax, "positions": args.tokens, "ok": worst < 2e-3 and argmax == 0}))
    return 0 if worst < 2e-3 and argmax == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
