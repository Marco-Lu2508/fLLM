#!/usr/bin/env python3
"""Logit comparison of the hybrid state-space / linear-attention architectures with transformers.

For every fixture a tiny random checkpoint is built with the official transformers implementation, converted with
`fllm-rs convert` and run through the CPU decoder (`examples/verify_checkpoint`); the logits of every position must agree
with transformers (relative difference below 2e-3 of the logit range, same arg-max).

    python scripts/verify_hybrid.py --fllm target/release/fllm-rs \
        --example target/release/examples/verify_checkpoint falcon_h1 falcon_h1_norm jamba

    python scripts/verify_hybrid.py --list

Needs torch, transformers and tokenizers (see docs/architectures.md). Some fixtures need a recent transformers (the class is
looked up by name); a missing class is reported as "skipped" and does not count as a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if not os.environ.get("HYBRID_USE_FLA"):
    # flash-linear-attention is only needed by gen_kda_reference.py; its Triton kernels cannot run on the CPU, so keep
    # transformers on its pure PyTorch implementations of the linear-attention layers
    sys.modules["fla"] = None  # type: ignore[assignment]

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Die Hauptstadt von Frankreich ist Paris, und 17 mal 23 ergibt 391.",
    "def add(a, b):\n    return a + b\n",
    "fLLM runs language models in pure Rust, on CPU and GPU.",
    "Zahlen: 0123456789 und Sonderzeichen: !?;:,.-_()[]{}",
] * 8
SPECIALS = ["<unk>", "<s>", "</s>", "<pad>", "<|im_start|>", "<|im_end|>"]
TOKENS = 12


def make_tokenizer(out: Path, vocab: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    tok.train_from_iterator(CORPUS, trainer)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
            }
        )
    )
    return tok


def common(vocab: int) -> dict:
    return dict(
        vocab_size=vocab, hidden_size=64, num_hidden_layers=3, bos_token_id=1, eos_token_id=2, pad_token_id=3,
        max_position_embeddings=256, initializer_range=0.08, tie_word_embeddings=False,
    )


def falcon_h1(vocab: int, **over):
    import transformers as T

    args = dict(
        **common(vocab), intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-5,
        mamba_d_ssm=64, mamba_n_heads=8, mamba_d_head=8, mamba_n_groups=2, mamba_d_state=16, mamba_d_conv=4,
        mamba_chunk_size=8, mamba_conv_bias=True, mamba_proj_bias=False, mamba_norm_before_gate=True, mamba_rms_norm=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        lm_head_multiplier=0.7, embedding_multiplier=1.5, mlp_multipliers=[0.8, 1.3], key_multiplier=0.9,
        attention_out_multiplier=1.2, attention_in_multiplier=0.6, ssm_multipliers=[1.1, 0.9, 1.3, 0.7, 1.2],
        ssm_in_multiplier=0.8, ssm_out_multiplier=1.4,
    )
    args.update(over)
    return T.FalconH1Config(**args)


def zamba2(vocab: int, **over):
    import transformers as T

    args = dict(
        **{**common(vocab), "num_hidden_layers": 6, "tie_word_embeddings": True}, intermediate_size=96, num_attention_heads=4,
        num_key_value_heads=4, rms_norm_eps=1e-5, layers_block_type=["mamba", "hybrid", "mamba", "hybrid", "mamba", "hybrid"],
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, mamba_ngroups=2, n_mamba_heads=8, chunk_size=8,
        num_mem_blocks=2, use_shared_attention_adapter=True, adapter_rank=8, use_mem_rope=False, time_step_min=0.001,
        hidden_act="gelu", rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    args.update(over)
    return T.Zamba2Config(**args)


def kimi_linear(vocab: int, **over):
    import transformers as T

    args = dict(
        **{**common(vocab), "num_hidden_layers": 4}, intermediate_size=96, moe_intermediate_size=32, num_attention_heads=4,
        num_key_value_heads=4, n_shared_experts=1, routed_scaling_factor=1.7, kv_lora_rank=16, q_lora_rank=None,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16, n_group=1, topk_group=1, num_experts_per_tok=2,
        norm_topk_prob=True, rms_norm_eps=1e-5, num_local_experts=4,
        layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        linear_head_dim=16, linear_num_heads=4, linear_conv_kernel_dim=4,
    )
    args.update(over)
    return T.KimiLinearConfig(**args)


def jamba(vocab: int, **over):
    import transformers as T

    args = dict(
        **{**common(vocab), "num_hidden_layers": 4}, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-6, num_experts=4, num_experts_per_tok=2, expert_layer_period=2, expert_layer_offset=1,
        attn_layer_period=4, attn_layer_offset=2, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_dt_rank=8,
        mamba_conv_bias=True, mamba_proj_bias=False,
    )
    args.update(over)
    return T.JambaConfig(**args)


# name -> callable(vocab) returning a transformers config
FIXTURES = {
    "falcon_h1": lambda v: falcon_h1(v),
    "falcon_h1_norm": lambda v: falcon_h1(v, mamba_rms_norm=True, mamba_norm_before_gate=True),
    "falcon_h1_norm_after": lambda v: falcon_h1(v, mamba_rms_norm=True, mamba_norm_before_gate=False),
    "falcon_h1_tied": lambda v: falcon_h1(v, tie_word_embeddings=True, projectors_bias=True, attention_bias=True),
    "zamba2": lambda v: zamba2(v),
    "zamba2_rope": lambda v: zamba2(v, use_mem_rope=True),
    "zamba2_one_block": lambda v: zamba2(v, num_mem_blocks=1, use_shared_attention_adapter=False),
    "kimi_linear": lambda v: kimi_linear(v),
    "kimi_linear_mla_first": lambda v: kimi_linear(
        v, layer_types=["full_attention", "linear_attention", "linear_attention", "full_attention"],
        mlp_layer_types=["dense", "dense", "sparse", "sparse"], q_lora_rank=None,
    ),
    "jamba": lambda v: jamba(v),
    "jamba_fused_experts": lambda v: jamba(v),
    "jamba_biased": lambda v: jamba(v, mamba_proj_bias=True, num_hidden_layers=6, attn_layer_period=3, attn_layer_offset=1),
}


def stack_experts(out: Path):
    """Rewrite per-expert MoE tensors as the stacked `experts.gate_up_proj` / `experts.down_proj` layout of transformers 5."""
    import re

    import torch
    from safetensors.torch import load_file, save_file

    path = out / "model.safetensors"
    tensors = load_file(str(path))
    groups: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for key in list(tensors):
        m = re.match(r"(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", key)
        if m:
            groups.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = tensors.pop(key)
    for prefix, experts in groups.items():
        order = sorted(experts)
        tensors[prefix + ".gate_up_proj"] = torch.stack([torch.cat([experts[e]["gate_proj"], experts[e]["up_proj"]], 0) for e in order])
        tensors[prefix + ".down_proj"] = torch.stack([experts[e]["down_proj"] for e in order])
    save_file(tensors, str(path), metadata={"format": "pt"})


def build(name: str, out: Path, vocab: int = 320, seed: int = 1234) -> dict:
    import torch
    import transformers as T

    out.mkdir(parents=True, exist_ok=True)
    tok = make_tokenizer(out, vocab)
    vocab = tok.get_vocab_size()
    cfg = FIXTURES[name](vocab)
    torch.manual_seed(seed)
    cls = getattr(T, type(cfg).__name__.replace("Config", "ForCausalLM"))
    model = cls._from_config(cfg, dtype=torch.float32, attn_implementation="eager")
    with torch.no_grad():
        for pname, p in model.named_parameters():
            if pname.endswith("norm.weight") or pname.endswith("layernorm.weight"):
                p.copy_(0.8 + 0.4 * torch.rand_like(p))
            if "adapter_list" in pname:
                p.normal_(std=0.25)
            if pname.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(0.3, 3.0)))
            if pname.endswith(".D"):
                p.normal_(std=1.0)
            if pname.endswith("dt_bias"):
                p.copy_(torch.empty_like(p).uniform_(-1.0, 0.5))
    model.eval()
    model.save_pretrained(out, safe_serialization=True)
    module = __import__("importlib").import_module(type(model).__module__)
    if os.environ.get("HYBRID_REF_KV16", "1") != "0" and hasattr(module, "eager_attention_forward"):
        # the example runs the engine with an f16 KV cache: round keys and values in the reference too, otherwise the
        # f16 rounding (up to ~1.5e-3 of the logit range for Zamba2's wide shared attention) hides real differences
        original = module.eager_attention_forward

        def rounded(mod, query, key, value, *a, **k):
            return original(mod, query, key.half().float(), value.half().float(), *a, **k)

        module.eager_attention_forward = rounded
    if name.endswith("_fused_experts"):
        stack_experts(out)
    gen = torch.Generator().manual_seed(seed + 1)
    ids = torch.randint(4, vocab, (1, TOKENS), generator=gen)
    with torch.no_grad():
        if os.environ.get("HYBRID_REF_FP64"):
            # an exact reference tells numeric conditioning apart from real differences
            model = model.double()
        logits = model(input_ids=ids).logits[0].float()
    return {"ids": ids[0].tolist(), "logits": logits.flatten().tolist(), "shape": list(logits.shape)}


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def verify(name: str, fllm: str, example: str, workdir: Path, outtype: str, keep: bool) -> dict:
    src = workdir / f"{name}-hf"
    try:
        golden = build(name, src)
    except (AttributeError, ImportError) as e:
        return {"arch": name, "ok": True, "skipped": f"transformers has no implementation ({e})"}
    gguf = workdir / f"{name}.gguf"
    run([fllm, "convert", str(src), str(gguf), "--outtype", outtype])
    ref = workdir / f"{name}-ids.json"
    ref.write_text(json.dumps({"ids": golden["ids"], "text": "hello"}))
    out = run([example, str(gguf), str(ref)])
    mine = json.loads(out.stdout.strip().splitlines()[-1])["logits"]
    want = golden["logits"]
    if len(mine) != len(want):
        return {"arch": name, "ok": False, "error": f"logit count {len(mine)} != {len(want)}"}
    positions, vocab = golden["shape"]
    span = max(want) - min(want)
    worst, mismatches = 0.0, 0
    for p in range(positions):
        a, b = mine[p * vocab : (p + 1) * vocab], want[p * vocab : (p + 1) * vocab]
        diff = max(abs(x - y) for x, y in zip(a, b))
        worst = max(worst, diff)
        mine_top = max(range(vocab), key=a.__getitem__)
        ref_top = max(range(vocab), key=b.__getitem__)
        if mine_top != ref_top and b[ref_top] - b[mine_top] > 4 * diff:
            mismatches += 1
    result = {
        "arch": name, "max_abs": worst, "relative": worst / span if span else 0.0,
        "argmax_mismatches": mismatches, "positions": positions,
    }
    result["ok"] = result["relative"] < 2e-3 and mismatches == 0
    if not keep:
        for path in (gguf, ref):
            path.unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archs", nargs="*")
    parser.add_argument("--fllm")
    parser.add_argument("--example")
    parser.add_argument("--outtype", default="f32")
    parser.add_argument("--workdir")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(sorted(FIXTURES)))
        return 0
    if not args.archs or not args.fllm or not args.example:
        parser.error("archs, --fllm and --example are required")
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.workdir or tmp)
        work.mkdir(parents=True, exist_ok=True)
        for name in args.archs:
            try:
                r = verify(name, args.fllm, args.example, work, args.outtype, args.keep)
            except subprocess.CalledProcessError as e:
                r = {"arch": name, "ok": False, "error": (e.stderr or e.stdout or str(e))[-800:]}
            failed += not r["ok"]
            print(json.dumps(r))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
