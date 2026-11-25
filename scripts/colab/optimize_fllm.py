#!/usr/bin/env python3
"""Inference-optimisation ceiling calculator and bench.py config generator for fLLM.

Implements the roofline & hill-climb methodology from the `inference-optimisation`
skill for Google Colab GPUs (T4 / L4 / A100-40G / A100-80G) running `fllm-rs serve`
with DSpine / DFlash parallel speculative decoding and GPU-accelerated MoE / Hybrid models.

Examples:
    python3 scripts/colab/optimize_fllm.py --preset qwen3.5-27b-dspine --gpu a100-80g
    python3 scripts/colab/optimize_fllm.py --preset qwen3-8b-dspine --gpu l4 --write-bench /tmp/fllm_bench.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GPUS: dict[str, dict[str, float]] = {
    "t4": {"bandwidth_gbs": 300.0, "peak_tflops": 65.0, "vram_gib": 16.0, "mfu": 0.45},
    "l4": {"bandwidth_gbs": 300.0, "peak_tflops": 121.0, "vram_gib": 24.0, "mfu": 0.50},
    "a100-40g": {"bandwidth_gbs": 1555.0, "peak_tflops": 312.0, "vram_gib": 40.0, "mfu": 0.55},
    "a100-80g": {"bandwidth_gbs": 1935.0, "peak_tflops": 312.0, "vram_gib": 80.0, "mfu": 0.55},
}

PRESETS: dict[str, dict[str, Any]] = {
    "qwen3-8b-dspine": {
        "target_hf": "Qwen/Qwen3-8B",
        "draft_hf": "tencent/Qwen3-8B-DSpine-b16",
        "quant": "q4_k_m",
        "dense_gb": 4.3,
        "experts_gb": 0.0,
        "top_k": 0,
        "n_experts": 0,
        "lm_head_gb": 0.48,
        "active_params_b": 8.2,
        "attn_layers": 36,
        "attn_heads": 32,
        "head_dim": 128,
        "kv_bytes_per_token": 36 * 8 * 128 * 2 * 2,  # f16 KV across 36 GQA layers
        "draft_layer_gb": 0.28,
        "draft_head_gb": 0.015,  # partial head rows via bridge head_topk=64
        "spec": [
            {"drafts": 0, "accept": 1.0},
            {"drafts": 1, "accept": 4.2},  # block_size=16 one-shot verification
        ],
        "extra_serve_args": ["--num-speculative-tokens", "15"],
        "env": {"FLLM_ACT_DTYPE": "bf16"},
    },
    "qwen3.5-27b-dspine": {
        "target_hf": "Qwen/Qwen3.5-27B",
        "draft_hf": "tencent/Qwen3.8-27B-DSpine-b16",
        "quant": "q4_k_m",
        "dense_gb": 15.2,
        "experts_gb": 0.0,
        "top_k": 0,
        "n_experts": 0,
        "lm_head_gb": 0.95,
        "active_params_b": 27.0,
        "attn_layers": 16,  # hybrid GatedDeltaNet + 16 full-attn layers
        "attn_heads": 24,
        "head_dim": 256,
        "kv_bytes_per_token": 16 * 4 * 256 * 2 * 2,
        "draft_layer_gb": 0.65,
        "draft_head_gb": 0.025,
        "spec": [
            {"drafts": 0, "accept": 1.0},
            {"drafts": 1, "accept": 4.6},
        ],
        "extra_serve_args": ["--num-speculative-tokens", "15"],
        "env": {"FLLM_ACT_DTYPE": "bf16"},
    },
    "lfm2moe": {
        "target_hf": "LiquidAI/LFM2-8B-A1B",
        "draft_hf": None,
        "quant": "q4_k_m",
        "dense_gb": 0.9,
        "experts_gb": 3.8,
        "top_k": 4,
        "n_experts": 32,
        "lm_head_gb": 0.25,
        "active_params_b": 1.5,
        "attn_layers": 6,
        "attn_heads": 32,
        "head_dim": 64,
        "kv_bytes_per_token": 6 * 8 * 64 * 2 * 2,
        "draft_layer_gb": 0.0,
        "draft_head_gb": 0.0,
        "spec": [{"drafts": 0, "accept": 1.0}],
        "extra_serve_args": [],
        "env": {"FLLM_ACT_DTYPE": "bf16"},
    },
    "dbrx": {
        "target_hf": "databricks/dbrx-instruct",
        "draft_hf": None,
        "quant": "q4_k_m",
        "dense_gb": 5.5,
        "experts_gb": 60.0,
        "top_k": 4,
        "n_experts": 16,
        "lm_head_gb": 0.35,
        "active_params_b": 36.0,
        "attn_layers": 40,
        "attn_heads": 48,
        "head_dim": 128,
        "kv_bytes_per_token": 40 * 8 * 128 * 2 * 2,
        "draft_layer_gb": 0.0,
        "draft_head_gb": 0.0,
        "spec": [{"drafts": 0, "accept": 1.0}],
        "extra_serve_args": [],
        "env": {},
    },
}


def decode_ceiling(model: dict[str, Any], gpu: dict[str, float], drafts: int, accept: float, ctx: int, conc: int) -> float:
    experts_gb = model.get("experts_gb", 0.0)
    union = (
        1.0 - (1.0 - model["top_k"] / model["n_experts"]) ** ((drafts + 1) * conc)
        if experts_gb and model.get("n_experts")
        else 0.0
    )
    verify = model["dense_gb"] + model["lm_head_gb"] + experts_gb * union
    verify += conc * model["kv_bytes_per_token"] * ctx / 1e9
    draft = drafts * conc * (model["draft_layer_gb"] + model["draft_head_gb"])
    return gpu["bandwidth_gbs"] / ((verify + draft) / (accept * conc))


def prefill_ceiling(model: dict[str, Any], gpu: dict[str, float], length: int) -> float:
    linear = 2.0 * model["active_params_b"] * 1e9
    attn = 4.0 * model["attn_layers"] * model["attn_heads"] * model["head_dim"] * (length / 2.0)
    return gpu["mfu"] * gpu["peak_tflops"] * 1e12 / (linear + attn)


def build_bench_config(preset_name: str, preset: dict[str, Any], gpu_name: str, gpu: dict[str, float], fllm_bin: str) -> dict[str, Any]:
    env_prefix = ["env", *(f"{k}={v}" for k, v in preset.get("env", {}).items())] if preset.get("env") else []
    cmd = [
        *env_prefix,
        fllm_bin,
        "serve",
        "--model",
        preset["target_hf"],
        "--quant",
        preset["quant"],
        "--device",
        "cuda:0",
        "--port",
        "8000",
        *preset["extra_serve_args"],
    ]
    if preset.get("draft_hf"):
        cmd.extend(["--speculative", "dspine", "--speculative-model", preset["draft_hf"]])
    return {
        "name": f"fllm-{preset_name}-{gpu_name}",
        "base_url": "http://127.0.0.1:8000",
        "model": preset["target_hf"],
        "launch": cmd,
        "health_timeout_s": 900,
        "concurrency": 4,
        "prefill_lengths": [4096, 32768],
        "temperature": 0.0,
        "memory": {
            "cap_gib": gpu["vram_gib"],
            "probe": [
                "bash",
                "-c",
                "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | awk '{printf \"%.0f\\n\", $1 * 1048576}'",
            ],
            "interval_s": 3,
        },
        "require_ac": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="qwen3.5-27b-dspine")
    ap.add_argument("--gpu", choices=sorted(GPUS), default="a100-80g")
    ap.add_argument("--fllm-bin", default="./target/release/fllm-rs")
    ap.add_argument("--write-ceiling", type=Path, help="Write ceiling.py-compatible JSON")
    ap.add_argument("--write-bench", type=Path, help="Write bench.py-compatible JSON")
    args = ap.parse_args()

    model = PRESETS[args.preset]
    gpu = GPUS[args.gpu]
    contexts = [4096, 32768, 131072]

    print(f"Preset: {args.preset} | GPU: {args.gpu} ({gpu['bandwidth_gbs']:.0f} GB/s, {gpu['peak_tflops']:.0f} TFLOPS, {gpu['vram_gib']:.0f} GiB VRAM)")
    fits_weights = model["dense_gb"] + model.get("experts_gb", 0.0) + model["lm_head_gb"] + model["draft_layer_gb"]
    print(f"Estimated weights footprint ({model['quant']}): {fits_weights:.2f} GB ({'fits' if fits_weights < gpu['vram_gib'] else 'NEEDS OFFLOAD / SMALLER QUANT'})")
    for conc in (1, 4):
        print(f"\nDecode ceiling (concurrency {conc}, aggregate tok/s):")
        print("  drafts accept" + "".join(f"{c // 1024:>8}K" for c in contexts))
        for spec in model["spec"]:
            row = "".join(
                f"{decode_ceiling(model, gpu, spec['drafts'], spec['accept'], ctx, conc):9.0f}"
                for ctx in contexts
            )
            print(f"  {spec['drafts']:>6} {spec['accept']:>6}{row}")
    print("\nPrefill ceiling (tok/s):")
    for ctx in contexts[:2]:
        print(f"  {ctx // 1024:>4}K {prefill_ceiling(model, gpu, ctx):8.0f}")

    if args.write_ceiling:
        payload = {**gpu, **model, "contexts": contexts, "concurrency": [1, 4]}
        args.write_ceiling.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote ceiling JSON -> {args.write_ceiling}")
    if args.write_bench:
        bench_cfg = build_bench_config(args.preset, model, args.gpu, gpu, args.fllm_bin)
        args.write_bench.write_text(json.dumps(bench_cfg, indent=2) + "\n")
        print(f"Wrote bench JSON -> {args.write_bench}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
