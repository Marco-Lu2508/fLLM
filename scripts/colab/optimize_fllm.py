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
import shlex
from pathlib import Path
from typing import Any

GPUS: dict[str, dict[str, Any]] = {
    "t4": {"bandwidth_gbs": 300.0, "peak_tflops": 65.0, "vram_gib": 16.0, "mfu": 0.45, "sm": 75, "act_dtype": "fp16"},
    "l4": {"bandwidth_gbs": 300.0, "peak_tflops": 121.0, "vram_gib": 24.0, "mfu": 0.50, "sm": 89, "act_dtype": "bf16"},
    "a100-40g": {"bandwidth_gbs": 1555.0, "peak_tflops": 312.0, "vram_gib": 40.0, "mfu": 0.55, "sm": 80, "act_dtype": "bf16"},
    "a100-80g": {"bandwidth_gbs": 1935.0, "peak_tflops": 312.0, "vram_gib": 80.0, "mfu": 0.55, "sm": 80, "act_dtype": "bf16"},
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
        "attn_layers": 16,  # 64 layers total: 48 GatedDeltaNet + 16 full-attn layers
        "attn_heads": 24,
        "head_dim": 256,
        "gdn_layers": 48,
        "gdn_k_heads": 16,
        "gdn_v_heads": 48,
        "gdn_head_dim": 128,
        # 48 GatedDeltaNet layers: (48 * 128 * 128 matrix state + 3 * 10240 conv1d history) * 4B f32
        "recurrent_state_gb": round(48 * (48 * 128 * 128 + 3 * 10240) * 4 / 1e9, 4),
        # Chunked GatedDeltaNet state-update FLOPs/token across 48 recurrent layers
        "extra_flops_per_token": 48 * 6 * 48 * 128 * 128,
        "kv_bytes_per_token": 16 * 4 * 256 * 2 * 2,
        "kv_per_verify_row": True,
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
    "bailing-moe": {
        "target_hf": "inclusionAI/Ling-lite",
        "arch": "bailing_moe",
        "draft_hf": None,
        "quant": "q4_k_m",
        "dense_gb": 1.3,
        "experts_gb": 7.8,
        "top_k": 6,
        "n_experts": 64,
        "lm_head_gb": 0.45,
        "active_params_b": 2.75,
        "attn_layers": 28,
        "attn_heads": 16,
        "head_dim": 128,
        "kv_bytes_per_token": 28 * 4 * 128 * 2 * 2,  # f16 KV across 28 GQA layers (4 KV heads)
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
    "qwen2.5-7b-awq": {
        "target_hf": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "draft_hf": None,
        "quant": "awq",
        "dense_gb": 4.3,
        "experts_gb": 0.0,
        "top_k": 0,
        "n_experts": 0,
        "lm_head_gb": 1.09,
        "active_params_b": 7.61,
        "attn_layers": 28,
        "attn_heads": 28,
        "head_dim": 128,
        "kv_bytes_per_token": 28 * 4 * 128 * 2 * 2,  # f16 KV across 28 GQA layers (4 KV heads)
        "draft_layer_gb": 0.0,
        "draft_head_gb": 0.0,
        "spec": [{"drafts": 0, "accept": 1.0}],
        "extra_serve_args": [],
        "env": {"FLLM_ACT_DTYPE": "bf16"},
    },
    "qwen1.5-moe-gptq": {
        "target_hf": "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
        "arch": "qwen2_moe",
        "draft_hf": None,
        "quant": "gptq",
        "dense_gb": 1.2,
        "experts_gb": 6.4,
        "top_k": 4,
        "n_experts": 60,
        "lm_head_gb": 0.62,
        "active_params_b": 2.7,
        "attn_layers": 24,
        "attn_heads": 16,
        "head_dim": 128,
        "kv_bytes_per_token": 24 * 16 * 128 * 2 * 2,  # f16 KV across 24 MHA layers (16 KV heads)
        "draft_layer_gb": 0.0,
        "draft_head_gb": 0.0,
        "spec": [{"drafts": 0, "accept": 1.0}],
        "extra_serve_args": [],
        "env": {"FLLM_ACT_DTYPE": "bf16"},
    },
    "qwen3-30b-a3b-gptq": {
        "target_hf": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        "arch": "qwen3_moe",
        "draft_hf": None,
        "quant": "gptq",
        "dense_gb": 0.85,
        "experts_gb": 15.4,
        "top_k": 8,
        "n_experts": 128,
        "lm_head_gb": 0.62,
        "active_params_b": 3.3,
        "attn_layers": 48,
        "attn_heads": 32,
        "head_dim": 128,
        "kv_bytes_per_token": 48 * 4 * 128 * 2 * 2,  # f16 KV across 48 GQA layers (4 KV heads)
        "draft_layer_gb": 0.0,
        "draft_head_gb": 0.0,
        "spec": [{"drafts": 0, "accept": 1.0}],
        "extra_serve_args": [],
        "env": {"FLLM_ACT_DTYPE": "bf16"},
    },
}


def act_dtype_for_gpu(gpu_name: str, gpu: dict[str, Any] | None = None) -> str:
    """Select FLLM_ACT_DTYPE for a GPU: fp16 on T4 (sm_75), bf16 on L4/A100 (sm_80+)."""
    if gpu and gpu.get("act_dtype"):
        return str(gpu["act_dtype"])
    if gpu and gpu.get("sm") is not None:
        return "fp16" if int(gpu["sm"]) < 80 else "bf16"
    return "fp16" if "t4" in gpu_name.lower() else "bf16"


def decode_ceiling(model: dict[str, Any], gpu: dict[str, Any], drafts: int, accept: float, ctx: int, conc: int) -> float:
    experts_gb = model.get("experts_gb", 0.0)
    union = (
        1.0 - (1.0 - model["top_k"] / model["n_experts"]) ** ((drafts + 1) * conc)
        if experts_gb and model.get("n_experts")
        else 0.0
    )
    verify = model["dense_gb"] + model["lm_head_gb"] + experts_gb * union
    kv_mult = (drafts + 1) if model.get("kv_per_verify_row") else 1
    verify += conc * model["kv_bytes_per_token"] * ctx / 1e9 * kv_mult
    recurrent_gb = model.get("recurrent_state_gb", 0.0)
    if recurrent_gb:
        # Read + write recurrent GDN state per decode step; verification also snapshots/restores state.
        verify += conc * recurrent_gb * (4.0 if drafts > 0 else 2.0)
    draft = drafts * conc * (model["draft_layer_gb"] + model["draft_head_gb"])
    return gpu["bandwidth_gbs"] / ((verify + draft) / (accept * conc))


def prefill_ceiling(model: dict[str, Any], gpu: dict[str, Any], length: int) -> float:
    linear = 2.0 * model["active_params_b"] * 1e9
    attn = 4.0 * model["attn_layers"] * model["attn_heads"] * model["head_dim"] * (length / 2.0)
    extra = model.get("extra_flops_per_token", 0.0)
    return gpu["mfu"] * gpu["peak_tflops"] * 1e12 / (linear + attn + extra)


def build_bench_config(
    preset_name: str,
    preset: dict[str, Any],
    gpu_name: str,
    gpu: dict[str, Any],
    fllm_bin: str,
    *,
    port: int = 8000,
    pid_file: str = "/tmp/fllm_serve.pid",
    log_file: str = "/tmp/fllm_serve.log",
    extra_env: dict[str, str | None] | None = None,
    extra_args: list[str] | None = None,
    name: str | None = None,
    concurrency: int = 4,
    prefill_lengths: list[int] | None = None,
    include_draft: bool = True,
) -> dict[str, Any]:
    env_map = dict(preset.get("env", {}))
    if "FLLM_ACT_DTYPE" in env_map or preset_name != "dbrx":
        env_map["FLLM_ACT_DTYPE"] = act_dtype_for_gpu(gpu_name, gpu)
    if extra_env:
        for key, val in extra_env.items():
            if val is None:
                env_map.pop(key, None)
            else:
                env_map[key] = val
    env_prefix = ["env", *(f"{k}={v}" for k, v in sorted(env_map.items()))] if env_map else []
    serve_args = list(preset["extra_serve_args"]) if extra_args is None else list(extra_args)
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
        str(port),
        *serve_args,
    ]
    if include_draft and preset.get("draft_hf"):
        cmd.extend(["--speculative", "dspine", "--speculative-model", preset["draft_hf"]])

    quoted_cmd = shlex.join(cmd)
    q_pid = shlex.quote(pid_file)
    q_log = shlex.quote(log_file)
    launch_script = (
        f"rm -f {q_pid}; "
        f"nohup {quoted_cmd} </dev/null >{q_log} 2>&1 & "
        f"pid=$!; echo \"$pid\" > {q_pid}; "
        f"sleep 0.2; tail -n 20 {q_log} 2>/dev/null || true"
    )
    stop_script = (
        f"if [ -f {q_pid} ]; then "
        f"pid=$(cat {q_pid}); "
        f"if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then "
        f"kill \"$pid\" 2>/dev/null || true; "
        f"for _ in $(seq 1 30); do kill -0 \"$pid\" 2>/dev/null || break; sleep 0.5; done; "
        f"kill -9 \"$pid\" 2>/dev/null || true; "
        f"fi; "
        f"rm -f {q_pid}; "
        f"fi"
    )
    return {
        "name": name or f"fllm-{preset_name}-{gpu_name}",
        "base_url": f"http://127.0.0.1:{port}",
        "model": preset["target_hf"],
        "launch": ["bash", "-c", launch_script],
        "stop": ["bash", "-c", stop_script],
        "pid_file": pid_file,
        "log_file": log_file,
        "health_timeout_s": 900,
        "concurrency": concurrency,
        "prefill_lengths": list(prefill_lengths) if prefill_lengths is not None else [4096, 32768],
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="qwen3.5-27b-dspine")
    ap.add_argument("--gpu", choices=sorted(GPUS), default="a100-80g")
    ap.add_argument("--fllm-bin", default="./target/release/fllm-rs")
    ap.add_argument("--concurrency", default="1,4", help="comma-separated user counts, e.g. 1,2,5,10,50")
    ap.add_argument("--contexts", default="4096,32768,131072", help="comma-separated contexts, e.g. 2048,4096,8192")
    ap.add_argument("--write-ceiling", type=Path, help="Write ceiling.py-compatible JSON")
    ap.add_argument("--write-bench", type=Path, help="Write bench.py-compatible JSON")
    args = ap.parse_args(argv)

    model = PRESETS[args.preset]
    gpu = GPUS[args.gpu]
    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    concurrencies = [int(x) for x in args.concurrency.split(",") if x.strip()]

    print(
        f"Preset: {args.preset} | GPU: {args.gpu} ({gpu['bandwidth_gbs']:.0f} GB/s, "
        f"{gpu['peak_tflops']:.0f} TFLOPS, {gpu['vram_gib']:.0f} GiB VRAM, "
        f"FLLM_ACT_DTYPE={act_dtype_for_gpu(args.gpu, gpu)})"
    )
    fits_weights = model["dense_gb"] + model.get("experts_gb", 0.0) + model["lm_head_gb"] + model["draft_layer_gb"]
    print(f"Estimated weights footprint ({model['quant']}): {fits_weights:.2f} GB ({'fits' if fits_weights < gpu['vram_gib'] else 'NEEDS OFFLOAD / SMALLER QUANT'})")
    if model.get("recurrent_state_gb"):
        print(f"Recurrent GDN state ({model.get('gdn_layers', 0)} layers): {model['recurrent_state_gb']:.3f} GB/stream")
    for conc in concurrencies:
        print(f"\nDecode ceiling (concurrency {conc}, aggregate tok/s):")
        print("  drafts accept" + "".join(f"{(c // 1024) if c >= 1024 else c:>7}{'K' if c >= 1024 else 't'}" for c in contexts))
        for spec in model["spec"]:
            row = "".join(
                f"{decode_ceiling(model, gpu, spec['drafts'], spec['accept'], ctx, conc):9.0f}"
                for ctx in contexts
            )
            print(f"  {spec['drafts']:>6} {spec['accept']:>6}{row}")
    print("\nPrefill ceiling (tok/s):")
    for ctx in contexts[:3]:
        label = f"{ctx // 1024}K" if ctx >= 1024 else f"{ctx}t"
        print(f"  {label:>5} {prefill_ceiling(model, gpu, ctx):8.0f}")

    if args.write_ceiling:
        payload = {**gpu, **model, "contexts": contexts, "concurrency": concurrencies}
        args.write_ceiling.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote ceiling JSON -> {args.write_ceiling}")
    if args.write_bench:
        bench_cfg = build_bench_config(args.preset, model, args.gpu, gpu, args.fllm_bin, concurrency=concurrencies[-1])
        args.write_bench.write_text(json.dumps(bench_cfg, indent=2) + "\n")
        print(f"Wrote bench JSON -> {args.write_bench}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
