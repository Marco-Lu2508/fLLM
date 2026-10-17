#!/usr/bin/env python3
"""Per-layer geometry architectures (Gemma 4, Gemma 3n and relatives) against transformers.

For every fixture a tiny random checkpoint (hidden 64, 2-5 layers) is built with the official
transformers implementation, converted with `fllm-rs convert` to an F32 GGUF and run through the
CPU backend of the engine (`examples/verify_checkpoint`). The logits of every position are compared
with the transformers logits (eager attention, float32): the relative error is
`max |a - b| / (max b - min b)` and must stay below `--tol` (default 1e-3); the arg-max of every
position has to agree unless the reference itself has a near tie.

    cargo build --release -p fllm-engine --bin fllm-rs --example verify_checkpoint
    python scripts/verify_gemma4.py --fllm target/release/fllm-rs \
        --example target/release/examples/verify_checkpoint            # all fixtures
    python scripts/verify_gemma4.py --list
    python scripts/verify_gemma4.py gemma4-moe --fllm ... --example ...

Requires torch, transformers (>= 5.19 for Gemma 4), tokenizers and safetensors. Nothing is
downloaded: the checkpoints are random.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>"]
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Die Hauptstadt von Frankreich ist Paris, und 17 mal 23 ergibt 391.",
    "def add(a, b):\n    return a + b\n",
    "fLLM runs language models in pure Rust, on CPU and GPU.",
    "Zahlen: 0123456789 und Sonderzeichen: !?;:,.-_()[]{}",
] * 8
TOKENS = 14
GENERATED = 10
# the example keeps the KV cache in f32 unless --kv f16: the f16 cache of the real runtime costs about 1e-3 of relative logit error
ENV = dict(os.environ, FLLM_VERIFY_KV="f32")
SEED = 1234

# name -> (family, description)
FIXTURES = {
    "gemma4-dense": "31B-like: sliding and global layers with different head width and KV heads, K = V on the global layers",
    "gemma4-moe": "26B-A4B-like: dense MLP and a routed expert block side by side, K = V, proportional rope",
    "gemma4-ple": "E2B-like: per-layer embeddings, KV sharing across layers, double-wide MLP in the shared layers",
    "gemma4-vlm": "gemma4-dense inside Gemma4ForConditionalGeneration (model.language_model.* names)",
    "gemma3": "Gemma 3 (existing single-geometry architecture, regression baseline)",
    "mellum": "Mellum: Qwen3-MoE with sliding-window layers (own rope base) and dense / sparse MLP layers mixed",
    "laguna": "Laguna: per-layer head counts, softplus output gate per head, partial rope per layer type, sigmoid router with logit soft cap and shared expert",
    "mimo": "MiMo-V2-Flash: query/key width 24 and value width 16, twice the KV heads and attention sinks on the sliding layers, value scale, partial rope, grouped sigmoid router",
    "granite_swa": "Granite SWA: sliding-window layers, learnable per-head sinks, own rope base per layer (0 = no position encoding), Granite multipliers",
    "granitemoe_swa": "GraniteMoE SWA: the same attention with a mixture-of-experts feed-forward",
    "gemma3n": "Gemma 3n: AltUp, LAuReL, per-layer embeddings, activation sparsity, KV sharing",
}


def make_tokenizer(out: Path, vocab: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS, trainer)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "add_bos_token": False,
                "add_eos_token": False,
            }
        )
    )
    return tok


def gemma4_text_config(variant: str, vocab: int):
    import transformers as T

    common = dict(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        num_global_key_value_heads=1,
        sliding_window=4,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        initializer_range=0.1,
        final_logit_softcapping=30.0,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=vocab,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
        },
    )
    layers5 = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    if variant in ("gemma4-dense", "gemma4-vlm"):
        return T.Gemma4TextConfig(**common, num_hidden_layers=5, layer_types=layers5, attention_k_eq_v=True)
    if variant == "gemma4-moe":
        return T.Gemma4TextConfig(
            **common,
            num_hidden_layers=5,
            layer_types=layers5,
            attention_k_eq_v=True,
            enable_moe_block=True,
            num_experts=6,
            top_k_experts=2,
            moe_intermediate_size=32,
        )
    if variant == "gemma4-ple":
        common.update(hidden_size_per_layer_input=16)
        return T.Gemma4TextConfig(
            **common,
            num_hidden_layers=6,
            layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"],
            attention_k_eq_v=False,
            num_kv_shared_layers=3,
            use_double_wide_mlp=True,
        )
    raise ValueError(variant)


def gemma3n_text_config(vocab: int):
    import transformers as T

    layers = 6
    return T.Gemma3nTextConfig(
        vocab_size=vocab,
        vocab_size_per_layer_input=vocab,
        hidden_size=64,
        hidden_size_per_layer_input=16,
        intermediate_size=[96] * layers,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=4,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        num_kv_shared_layers=2,
        laurel_rank=8,
        altup_num_inputs=4,
        altup_active_idx=0,
        altup_coef_clip=120.0,
        altup_correct_scale=True,
        activation_sparsity_pattern=[0.95, 0.95, 0.0, 0.0, 0.0, 0.0],
        final_logit_softcapping=30.0,
        max_position_embeddings=256,
        initializer_range=0.1,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        rope_theta=1000000.0,
        rope_local_base_freq=10000.0,
    )


def mellum_config(vocab: int):
    import transformers as T

    return T.MellumConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, sliding_window=4, num_experts=6, num_experts_per_tok=2,
        moe_intermediate_size=32, norm_topk_prob=True, max_position_embeddings=256, initializer_range=0.1,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "dense"], pad_token_id=0, eos_token_id=1, bos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )


def laguna_config(vocab: int):
    import transformers as T

    return T.LagunaConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_attention_heads_per_layer=[4, 6, 2, 4], num_key_value_heads=2, head_dim=16, sliding_window=4,
        num_experts=6, num_experts_per_tok=3, moe_intermediate_size=32, shared_expert_intermediate_size=48,
        moe_routed_scaling_factor=1.7, moe_router_logit_softcapping=5.0, max_position_embeddings=256,
        initializer_range=0.1, gating=True, layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"], pad_token_id=0, eos_token_id=1, bos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 500000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        },
    )


def mimo_config(vocab: int):
    import transformers as T

    return T.MiMoV2FlashConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=1, head_dim=24, v_head_dim=16, sliding_window=4, n_routed_experts=6, num_experts_per_tok=3,
        moe_intermediate_size=32, n_group=2, topk_group=1, norm_topk_prob=True, routed_scaling_factor=1.5,
        attention_value_scale=0.707, max_position_embeddings=256, initializer_range=0.1,
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"], pad_token_id=0, eos_token_id=1, bos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 5000000.0, "partial_rotary_factor": 0.334},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.334},
        },
    )


def granite_swa_config(vocab: int, moe: bool):
    import transformers as T

    common = dict(
        vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, sliding_window=4, max_position_embeddings=256, initializer_range=0.1,
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
        layer_rope_theta=[10000.0, 500000.0, 500000.0, 0, 10000.0, 20000.0], embedding_multiplier=3.0, attention_multiplier=0.3,
        residual_multiplier=0.4, logits_scaling=2.5, pad_token_id=0, eos_token_id=1, bos_token_id=2, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    if moe:
        return T.GraniteMoeSWAConfig(**common, num_local_experts=4, num_experts_per_tok=2, shared_intermediate_size=48)
    return T.GraniteSWAConfig(**common)


def build(variant: str, out: Path, vocab: int = 320):
    import torch
    import transformers as T

    out.mkdir(parents=True, exist_ok=True)
    tok = make_tokenizer(out, vocab)
    vocab = tok.get_vocab_size()
    torch.manual_seed(SEED)
    if variant == "gemma3":
        cfg = T.Gemma3TextConfig(
            vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=6, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, sliding_window=4, sliding_window_pattern=3, max_position_embeddings=256,
            initializer_range=0.1, final_logit_softcapping=None, query_pre_attn_scalar=16, pad_token_id=0, eos_token_id=1,
            bos_token_id=2, rope_local_base_freq=10000.0, rope_theta=1000000.0,
        )
        model = T.Gemma3ForCausalLM._from_config(cfg, dtype=torch.float32, attn_implementation="eager")
    elif variant == "mellum":
        cfg = mellum_config(vocab)
        model = T.MellumForCausalLM._from_config(cfg, dtype=torch.float32, attn_implementation="eager")
    elif variant == "laguna":
        cfg = laguna_config(vocab)
        model = T.LagunaForCausalLM._from_config(cfg, dtype=torch.float32, attn_implementation="eager")
    elif variant == "mimo":
        cfg = mimo_config(vocab)
        model = T.MiMoV2FlashForCausalLM._from_config(cfg, dtype=torch.float32, attn_implementation="eager")
    elif variant in ("granite_swa", "granitemoe_swa"):
        cfg = granite_swa_config(vocab, variant == "granitemoe_swa")
        cls = T.GraniteMoeSWAForCausalLM if variant == "granitemoe_swa" else T.GraniteSWAForCausalLM
        model = cls._from_config(cfg, dtype=torch.float32, attn_implementation="eager")
    elif variant == "gemma3n":
        cfg = gemma3n_text_config(vocab)
        model = T.Gemma3nForCausalLM._from_config(cfg, dtype=torch.float32, attn_implementation="eager")
    else:
        cfg = gemma4_text_config(variant, vocab)
        if variant == "gemma4-vlm":
            full = T.Gemma4Config(text_config=cfg.to_dict())
            model = T.Gemma4ForConditionalGeneration._from_config(full, dtype=torch.float32, attn_implementation="eager")
        else:
            model = T.Gemma4ForCausalLM._from_config(cfg, dtype=torch.float32, attn_implementation="eager")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("layernorm.weight") or "layernorm_" in name:
                p.copy_(0.8 + 0.4 * torch.rand_like(p))
            if name.endswith("per_expert_scale") or name.endswith("router.scale"):
                p.copy_(0.7 + 0.6 * torch.rand_like(p))
            if name.endswith("e_score_correction_bias"):
                p.copy_(0.2 * torch.randn_like(p))
            if name.endswith("sinks"):
                p.copy_(torch.randn_like(p))
            if p.dim() == 2 and float(p.abs().sum()) == 0.0:
                p.normal_(std=0.3)  # routers may start from zeros (all ties)
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(0.2 * torch.randn_like(b))
            if name.endswith("layer_scalar"):
                b.copy_(0.7 + 0.6 * torch.rand_like(b))
        if variant == "gemma3n":
            for name, p in model.named_parameters():
                if "altup" in name and name.endswith("correct_output_scale"):
                    p.copy_(0.5 + torch.rand_like(p))
                if name.endswith("router_norm.weight") or name.endswith("norm.weight"):
                    p.copy_(0.8 + 0.4 * torch.rand_like(p))
    model.eval()
    model.save_pretrained(out, safe_serialization=True)
    generator = torch.Generator().manual_seed(SEED + 1)
    ids = torch.randint(4, vocab, (1, TOKENS), generator=generator)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
    with torch.no_grad():
        greedy = model.generate(ids[:, :6], max_new_tokens=GENERATED, do_sample=False, use_cache=True, pad_token_id=0)[0, 6:].tolist()
    golden = dict(arch=variant, ids=ids[0].tolist(), logits=logits.reshape(-1).tolist(), shape=list(logits.shape), greedy=greedy)
    (out / "golden.json").write_text(json.dumps(golden))
    return golden


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=ENV)


def served_tokens(fllm: str, gguf: Path, prompt: list, count: int, kv: str) -> str:
    """Greedy continuation of `prompt` through the real server (scheduler, paged cache) on the CPU."""
    import socket
    import time
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen(
        [fllm, "serve", "--model", str(gguf), "--device", "cpu", "--ctx", "128", "--port", str(port), "--max-seqs", "2", "--cache-type-k", kv, "--cache-type-v", kv],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        body = json.dumps({"model": "x", "prompt": prompt, "max_tokens": count, "temperature": 0}).encode()
        for _ in range(100):
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
                reply = json.load(urllib.request.urlopen(req, timeout=60))
                break
            except OSError:
                time.sleep(0.3)
        else:
            raise RuntimeError("server did not answer")
        return reply["choices"][0]["text"]
    finally:
        server.kill()
        server.wait()


def verify(variant: str, fllm: str, example: str, workdir: Path, outtype: str, tol: float, keep: bool, generate: bool = False, kv: str = "f32") -> dict:
    src = workdir / f"{variant}-hf"
    golden = build(variant, src)
    gguf = workdir / f"{variant}.gguf"
    run([fllm, "convert", str(src), str(gguf), "--outtype", outtype, "--overwrite"])
    ref = workdir / f"{variant}-ids.json"
    ref.write_text(json.dumps({"ids": golden["ids"], "text": "hello"}))
    out = run([example, str(gguf), str(ref)])
    mine = json.loads(out.stdout.strip().splitlines()[-1])["logits"]
    want = golden["logits"]
    if len(mine) != len(want):
        return {"arch": variant, "ok": False, "error": f"logit count {len(mine)} != {len(want)}"}
    positions, vocab = golden["shape"]
    span = max(want) - min(want)
    worst = 0.0
    worst_at = (0, 0)
    mismatches = 0
    for p in range(positions):
        a = mine[p * vocab : (p + 1) * vocab]
        b = want[p * vocab : (p + 1) * vocab]
        diffs = [abs(x - y) for x, y in zip(a, b)]
        position_worst = max(diffs)
        if position_worst > worst:
            worst, worst_at = position_worst, (p, diffs.index(position_worst))
        mine_top = max(range(vocab), key=a.__getitem__)
        ref_top = max(range(vocab), key=b.__getitem__)
        if mine_top != ref_top and b[ref_top] - b[mine_top] > 4 * position_worst:
            mismatches += 1
    result = {
        "arch": variant,
        "max_abs": worst,
        "worst_position_token": worst_at,
        "relative": worst / span if span else 0.0,
        "argmax_mismatches": mismatches,
        "positions": positions,
    }
    result["ok"] = result["relative"] <= tol and mismatches == 0
    if generate:
        from tokenizers import Tokenizer

        want_text = Tokenizer.from_file(str(src / "tokenizer.json")).decode(golden["greedy"], skip_special_tokens=False)
        got_text = served_tokens(fllm, gguf, golden["ids"][:6], len(golden["greedy"]), kv)
        result["generation_matches"] = got_text == want_text
        result["ok"] = result["ok"] and got_text == want_text
    if not keep:
        for path in (gguf, ref):
            path.unlink(missing_ok=True)
    return result


def main() -> int:
    global ENV
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help="fixtures to run (default: all)")
    parser.add_argument("--fllm")
    parser.add_argument("--example")
    parser.add_argument("--outtype", default="f32")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--kv", choices=["f32", "f16"], default="f32", help="KV cache precision of the engine run (f16 is what the server uses)")
    parser.add_argument("--workdir")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--generate", action="store_true", help="also compare a greedy continuation served by the real server (CPU, paged cache) with transformers")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    ENV["FLLM_VERIFY_KV"] = args.kv
    if args.list:
        for name, text in FIXTURES.items():
            print(f"{name:14} {text}")
        return 0
    if not args.fllm or not args.example:
        parser.error("--fllm and --example are required")
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.workdir or tmp)
        work.mkdir(parents=True, exist_ok=True)
        for variant in args.variants or list(FIXTURES):
            try:
                r = verify(variant, args.fllm, args.example, work, args.outtype, args.tol, args.keep, args.generate, args.kv)
            except subprocess.CalledProcessError as e:
                r = {"arch": variant, "ok": False, "error": (e.stderr or e.stdout or str(e))[-800:]}
            failed += not r["ok"]
            print(json.dumps(r))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
