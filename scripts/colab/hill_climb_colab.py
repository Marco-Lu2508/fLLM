#!/usr/bin/env python3
"""Colab hill-climb orchestrator compliant with the `inference-optimisation` skill.

Computes theoretical ceilings via `optimize_fllm.py`, generates `bench.py`-compatible
configs across ordered fLLM optimization candidates:
  1. baseline         (FP32 activations, f16 KV, CUDA graphs off, speculation off)
  2. +act_dtype       (FLLM_ACT_DTYPE=fp16 on T4 sm_75, bf16 on L4/A100 sm_80+)
  3. +kv_q8_0         (--kv-cache-type q8_0)
  4. +cuda_graph      (FLLM_CUDA_GRAPH=1)
  5. +dspine          (--speculative dspine --speculative-model <draft_hf> --num-speculative-tokens 15)
  6. +spec_tokens_tune (--num-speculative-tokens <tuned>, default 8)

Each candidate is derived one-change-at-a-time from the current best configuration,
evaluated with interleaved controls (control -> candidate -> control) and >= 6 decode
samples via `bench.py`, logged to `PLAN.md` + `ledger.jsonl`, and checked against
regression/output-sanity gates before updating `best-config.json`.

Examples:
    python3 scripts/colab/hill_climb_colab.py --dry-run
    python3 scripts/colab/hill_climb_colab.py --preset qwen3-8b-dspine --gpu l4 --dry-run
    python3 scripts/colab/hill_climb_colab.py --preset qwen3.5-27b-dspine --gpu a100-80g --bench-py /path/to/bench.py
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import optimize_fllm  # noqa: E402

CANDIDATE_ORDER: tuple[str, ...] = (
    "baseline",
    "+act_dtype",
    "+kv_q8_0",
    "+cuda_graph",
    "+dspine",
    "+spec_tokens_tune",
)

BAD_FLAGS = frozenset({"LOOP", "REQ_FAIL", "MEM_CAP", "ON_BATTERY"})


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def detect_gpu() -> str | None:
    """Detect the Colab GPU preset key using nvidia-smi, or None if no GPU is found."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    line = proc.stdout.strip().splitlines()[0]
    upper = line.upper()
    if "T4" in upper:
        return "t4"
    if "L4" in upper:
        return "l4"
    if "A100" in upper:
        digits = "".join(ch for ch in line.split(",", 1)[-1] if ch.isdigit())
        mem_mib = int(digits) if digits else 0
        return "a100-80g" if mem_mib >= 60000 else "a100-40g"
    return None


def find_bench_py(explicit: Path | None = None) -> Path | None:
    """Locate inference-optimisation's bench.py."""
    if explicit is not None:
        p = explicit.expanduser().resolve()
        return p if p.is_file() else None
    candidates = [
        SCRIPT_DIR / "bench.py",
        Path.cwd() / "bench.py",
        Path.home() / ".gemini/skills/inference-optimisation/scripts/bench.py",
        Path.home() / ".agents/skills/inference-optimisation/scripts/bench.py",
        Path.home() / ".codex/skills/inference-optimisation/scripts/bench.py",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    found = shutil.which("bench.py")
    return Path(found).resolve() if found else None


def initial_state(preset: dict[str, Any]) -> dict[str, Any]:
    return {
        "act_dtype": "f32",
        "kv_cache_type": "f16",
        "cuda_graph": "0",
        "speculative": "off",
        "num_speculative_tokens": 15,
    }


def apply_candidate(
    base_state: dict[str, Any],
    candidate: str,
    preset: dict[str, Any],
    gpu_name: str,
    gpu: dict[str, Any],
    tune_spec_tokens: int = 8,
) -> tuple[dict[str, Any], bool, str]:
    """Return (new_state, applicable, description) for a single hill-climb lever."""
    state = copy.deepcopy(base_state)
    if candidate == "baseline":
        return state, True, "Baseline control (act_dtype=f32, kv=f16, cuda_graph=0, spec=off)"
    if candidate == "+act_dtype":
        target = optimize_fllm.act_dtype_for_gpu(gpu_name, gpu)
        state["act_dtype"] = target
        return state, True, f"Enable narrow activations FLLM_ACT_DTYPE={target} for {gpu_name}"
    if candidate == "+kv_q8_0":
        state["kv_cache_type"] = "q8_0"
        return state, True, "Quantize KV cache (--kv-cache-type q8_0)"
    if candidate == "+cuda_graph":
        state["cuda_graph"] = "1"
        return state, True, "Enable decode CUDA graphs (FLLM_CUDA_GRAPH=1)"
    if candidate == "+dspine":
        if not preset.get("draft_hf"):
            return state, False, "Skipped: preset has no DSpine draft_hf"
        state["speculative"] = "dspine"
        state["num_speculative_tokens"] = 15
        return state, True, f"Enable DSpine parallel speculation ({preset['draft_hf']}, k=15)"
    if candidate == "+spec_tokens_tune":
        if not preset.get("draft_hf"):
            return state, False, "Skipped: preset has no DSpine draft_hf"
        if state.get("speculative") != "dspine":
            return state, False, "Skipped: +dspine is not active in current best"
        state["num_speculative_tokens"] = tune_spec_tokens
        return state, True, f"Tune draft block length (--num-speculative-tokens {tune_spec_tokens})"
    raise ValueError(f"unknown candidate: {candidate}")


def state_to_bench_config(
    state: dict[str, Any],
    preset_name: str,
    preset: dict[str, Any],
    gpu_name: str,
    gpu: dict[str, Any],
    fllm_bin: str,
    *,
    run_name: str,
    port: int = 8000,
    pid_file: str = "/tmp/fllm_serve.pid",
    log_file: str = "/tmp/fllm_serve.log",
    concurrency: int = 4,
    prefill_lengths: list[int] | None = None,
) -> dict[str, Any]:
    extra_env = {
        "FLLM_ACT_DTYPE": state["act_dtype"],
        "FLLM_CUDA_GRAPH": state["cuda_graph"],
    }
    extra_args = ["--kv-cache-type", state["kv_cache_type"]]
    include_draft = state["speculative"] == "dspine" and bool(preset.get("draft_hf"))
    if include_draft:
        extra_args.extend(["--num-speculative-tokens", str(state["num_speculative_tokens"])])
    else:
        extra_args.extend(["--speculative", "off"])

    return optimize_fllm.build_bench_config(
        preset_name,
        preset,
        gpu_name,
        gpu,
        fllm_bin,
        port=port,
        pid_file=pid_file,
        log_file=log_file,
        extra_env=extra_env,
        extra_args=extra_args,
        name=run_name,
        concurrency=concurrency,
        prefill_lengths=prefill_lengths or [4096, 32768],
        include_draft=include_draft,
    )


def validate_bench_config(cfg: dict[str, Any]) -> list[str]:
    """Validate that a config matches bench.py's contract and uses non-blocking launch/stop."""
    errors: list[str] = []
    for key in ("name", "base_url", "model", "launch", "stop", "health_timeout_s",
                "concurrency", "prefill_lengths", "temperature", "memory", "require_ac"):
        if key not in cfg:
            errors.append(f"missing required key: {key}")
    if not isinstance(cfg.get("name"), str) or not cfg["name"]:
        errors.append("name must be a non-empty string")
    if not isinstance(cfg.get("base_url"), str) or not cfg["base_url"].startswith("http"):
        errors.append("base_url must start with http")
    for cmd_key in ("launch", "stop"):
        cmd = cfg.get(cmd_key)
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            errors.append(f"{cmd_key} must be a non-empty list of strings")
    launch_str = " ".join(cfg.get("launch") or [])
    if "/tmp/fllm_serve.pid" not in launch_str and cfg.get("pid_file", "") not in launch_str:
        errors.append("launch must record a PID file")
    if "&" not in launch_str:
        errors.append("launch must start fllm-rs serve asynchronously in background")
    stop_str = " ".join(cfg.get("stop") or [])
    if "kill" not in stop_str:
        errors.append("stop must kill the recorded PID")
    mem = cfg.get("memory") or {}
    if not isinstance(mem, dict) or not mem.get("cap_gib") or not mem.get("probe"):
        errors.append("memory must define cap_gib and probe")
    if not isinstance(cfg.get("prefill_lengths"), list) or not cfg["prefill_lengths"]:
        errors.append("prefill_lengths must be a non-empty list")
    return errors


def compute_ceilings(preset_name: str, gpu_name: str) -> dict[str, Any]:
    model = optimize_fllm.PRESETS[preset_name]
    gpu = optimize_fllm.GPUS[gpu_name]
    contexts = [4096, 32768, 131072]
    decode = {}
    for conc in (1, 4):
        decode[conc] = [
            {
                "drafts": s["drafts"],
                "accept": s["accept"],
                "by_ctx": {
                    ctx: round(optimize_fllm.decode_ceiling(model, gpu, s["drafts"], s["accept"], ctx, conc), 1)
                    for ctx in contexts
                },
            }
            for s in model["spec"]
        ]
    prefill = {ctx: round(optimize_fllm.prefill_ceiling(model, gpu, ctx), 1) for ctx in contexts[:2]}
    weights_gb = model["dense_gb"] + model.get("experts_gb", 0.0) + model["lm_head_gb"] + model["draft_layer_gb"]
    return {
        "preset": preset_name,
        "gpu": gpu_name,
        "act_dtype": optimize_fllm.act_dtype_for_gpu(gpu_name, gpu),
        "weights_gb": round(weights_gb, 2),
        "vram_gib": gpu["vram_gib"],
        "fits": weights_gb < gpu["vram_gib"],
        "recurrent_state_gb": model.get("recurrent_state_gb", 0.0),
        "contexts": contexts,
        "decode": decode,
        "prefill": prefill,
    }


def format_theory_md(ceilings: dict[str, Any]) -> str:
    lines = [
        f"# Theoretical Ceilings: {ceilings['preset']} on {ceilings['gpu']}",
        "",
        f"- Target `FLLM_ACT_DTYPE`: `{ceilings['act_dtype']}`",
        f"- Weights footprint: `{ceilings['weights_gb']:.2f} GB` / `{ceilings['vram_gib']:.0f} GiB` VRAM "
        f"({'fits' if ceilings['fits'] else 'EXCEEDS VRAM'})",
    ]
    if ceilings["recurrent_state_gb"]:
        lines.append(f"- GatedDeltaNet recurrent state: `{ceilings['recurrent_state_gb']:.4f} GB/stream` (read+write accounted)")
    for conc in (1, 4):
        lines.extend([
            "",
            f"## Decode ceiling (concurrency {conc}, aggregate tok/s)",
            "",
            "| drafts | accept | " + " | ".join(f"{c // 1024}K" for c in ceilings["contexts"]) + " |",
            "|---:|---:|" + "|".join("---:" for _ in ceilings["contexts"]) + "|",
        ])
        for row in ceilings["decode"][conc]:
            vals = " | ".join(f"{row['by_ctx'][c]:.0f}" for c in ceilings["contexts"])
            lines.append(f"| {row['drafts']} | {row['accept']} | {vals} |")
    lines.extend([
        "",
        "## Prefill ceiling (practical, tok/s)",
        "",
        "| context | ceiling tok/s |",
        "|---:|---:|",
    ])
    for ctx, tps in ceilings["prefill"].items():
        lines.append(f"| {ctx // 1024}K | {tps:.0f} |")
    return "\n".join(lines) + "\n"


def should_accept(cand: dict[str, Any], ctl_before: dict[str, Any], ctl_after: dict[str, Any]) -> tuple[bool, str]:
    """Enforce inference-optimisation acceptance & non-regression gates."""
    for label, row in (("ctl_before", ctl_before), ("candidate", cand), ("ctl_after", ctl_after)):
        if not row.get("ok"):
            return False, f"{label} failed (ok=False, error={row.get('error', 'unknown')})"
        bad = set(row.get("flags") or []) & BAD_FLAGS
        if bad:
            return False, f"{label} tripped flags: {','.join(sorted(bad))}"

    c_dec = cand.get("c1_decode")
    b_dec = ctl_before.get("c1_decode")
    a_dec = ctl_after.get("c1_decode")
    if c_dec is None or b_dec is None or a_dec is None:
        return False, "missing c1_decode measurement"
    if not (c_dec > b_dec and c_dec > a_dec):
        return False, f"c1_decode {c_dec} did not beat both controls ({b_dec}, {a_dec})"

    for key in sorted({k for k in cand if k == "c4_aggregate" or (k.startswith("c") and k.endswith("_aggregate"))
                       or (k.startswith("prefill_") and k[8:].isdigit())}):
        cv = cand.get(key)
        refs = [r.get(key) for r in (ctl_before, ctl_after) if isinstance(r.get(key), (int, float))]
        if isinstance(cv, (int, float)) and refs:
            floor = 0.97 * min(refs)
            if cv < floor:
                return False, f"{key} regressed >3% ({cv} < 0.97 * {min(refs):.2f})"

    gain_vs_max = (c_dec / max(b_dec, a_dec) - 1.0) * 100.0
    return True, f"c1_decode {c_dec} beat controls ({b_dec}, {a_dec}) [+{gain_vs_max:.1f}%]"


def run_bench_once(
    bench_py: Path,
    cfg_path: Path,
    work_dir: Path,
    reps: int,
    long_ctx: int | None,
    skip_cn: bool,
) -> dict[str, Any]:
    ledger_path = work_dir / "ledger.jsonl"
    before_lines = len(ledger_path.read_text().splitlines()) if ledger_path.exists() else 0
    cmd = [sys.executable, str(bench_py), "run", str(cfg_path), "--reps", str(reps)]
    if long_ctx:
        cmd.extend(["--long", str(long_ctx)])
    if skip_cn:
        cmd.append("--skip-cN")
    proc = subprocess.run(cmd, cwd=work_dir)
    if not ledger_path.exists():
        raise RuntimeError(f"bench.py exited rc={proc.returncode} without writing {ledger_path}")
    lines = [line for line in ledger_path.read_text().splitlines() if line.strip()]
    if len(lines) <= before_lines:
        raise RuntimeError(f"bench.py exited rc={proc.returncode} without appending a ledger row")
    return json.loads(lines[-1])


def print_dry_run(
    preset_name: str,
    gpu_name: str,
    fllm_bin: str,
    bench_py: Path | None,
    ceilings: dict[str, Any],
    tune_spec_tokens: int,
    port: int,
    reps: int,
) -> int:
    preset = optimize_fllm.PRESETS[preset_name]
    gpu = optimize_fllm.GPUS[gpu_name]
    print(f"=== fLLM Colab Hill-Climb Plan (--dry-run) ===")
    print(f"Preset:     {preset_name} ({preset['target_hf']}, quant={preset['quant']})")
    print(f"GPU:        {gpu_name} ({gpu['vram_gib']:.0f} GiB VRAM, FLLM_ACT_DTYPE={ceilings['act_dtype']})")
    print(f"Binary:     {fllm_bin}")
    print(f"bench.py:   {bench_py or 'not found locally (pass --bench-py on Colab)'}")
    print(f"Protocol:   interleaved controls (control -> candidate -> control), reps={reps}")
    print()
    print(format_theory_md(ceilings).rstrip())
    print()
    print("Candidate Sequence & Config Validation:")

    cur_state = initial_state(preset)
    all_valid = True
    for idx, cand in enumerate(CANDIDATE_ORDER, start=1):
        # For dry-run preview, assume prior applicable levers were accepted so we validate every config.
        next_state, applicable, desc = apply_candidate(
            cur_state, cand, preset, gpu_name, gpu, tune_spec_tokens=tune_spec_tokens
        )
        run_name = f"{idx:02d}-{slugify(cand)}"
        cfg = state_to_bench_config(
            next_state, preset_name, preset, gpu_name, gpu, fllm_bin,
            run_name=run_name, port=port,
        )
        errs = validate_bench_config(cfg)
        if errs:
            all_valid = False
        status = "VALID" if not errs else f"INVALID ({'; '.join(errs)})"
        tag = "APPLY" if applicable else "SKIP "
        schedule = "single baseline" if cand == "baseline" else ("ctl -> cand -> ctl" if applicable else "skipped")
        print(f"  {idx}. [{tag}] {cand:<18} | {schedule:<19} | {status} | {desc}")
        if applicable:
            cur_state = next_state
    print()
    print(f"Config validation: {'ALL PASSED' if all_valid else 'FAILED'} ({len(CANDIDATE_ORDER)} candidates checked)")
    return 0 if all_valid else 1


def run_live(args: argparse.Namespace, gpu_name: str, bench_py: Path) -> int:
    preset = optimize_fllm.PRESETS[args.preset]
    gpu = optimize_fllm.GPUS[gpu_name]
    work_dir = Path(args.work).resolve()
    configs_dir = work_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    ceilings = compute_ceilings(args.preset, gpu_name)
    (work_dir / "THEORY.md").write_text(format_theory_md(ceilings))
    plan_path = work_dir / "PLAN.md"
    if not plan_path.exists():
        plan_path.write_text(
            f"# Hill-Climb Log: {args.preset} on {gpu_name}\n\n"
            f"Started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n"
        )

    def append_plan(text: str) -> None:
        with plan_path.open("a") as fh:
            fh.write(text + "\n")

    def write_cfg(name: str, state: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        cfg = state_to_bench_config(
            state, args.preset, preset, gpu_name, gpu, args.fllm_bin,
            run_name=name, port=args.port,
        )
        errs = validate_bench_config(cfg)
        if errs:
            raise RuntimeError(f"invalid config {name}: {'; '.join(errs)}")
        p = configs_dir / f"{slugify(name)}.json"
        p.write_text(json.dumps(cfg, indent=2) + "\n")
        return p, cfg

    c1_ceiling = ceilings["decode"][1][0]["by_ctx"][4096]
    best_state = initial_state(preset)
    last_cfg: dict[str, Any] | None = None
    consecutive_rejects = 0

    try:
        base_path, last_cfg = write_cfg("01-baseline", best_state)
        base_res = run_bench_once(bench_py, base_path, work_dir, args.reps, args.long, args.skip_cn)
        (work_dir / "best-config.json").write_text(json.dumps(last_cfg, indent=2) + "\n")
        c1 = base_res.get("c1_decode")
        pct = f" ({100.0 * c1 / c1_ceiling:.1f}% of baseline c1 ceiling)" if c1 and c1_ceiling else ""
        append_plan(f"- `{time.strftime('%H:%M:%S')}` **baseline**: c1={c1}{pct}, flags={base_res.get('flags')}")

        for idx, cand in enumerate(CANDIDATE_ORDER[1:], start=2):
            cand_state, applicable, desc = apply_candidate(
                best_state, cand, preset, gpu_name, gpu, tune_spec_tokens=args.tune_spec_tokens
            )
            if not applicable:
                append_plan(f"- `{time.strftime('%H:%M:%S')}` **{cand}**: SKIP ({desc})")
                continue

            prefix = f"{idx:02d}-{slugify(cand)}"
            ctl_a_path, last_cfg = write_cfg(f"{prefix}-ctl-a", best_state)
            ctl_a = run_bench_once(bench_py, ctl_a_path, work_dir, args.reps, args.long, args.skip_cn)

            cand_path, last_cfg = write_cfg(prefix, cand_state)
            cand_res = run_bench_once(bench_py, cand_path, work_dir, args.reps, args.long, args.skip_cn)

            ctl_b_path, last_cfg = write_cfg(f"{prefix}-ctl-b", best_state)
            ctl_b = run_bench_once(bench_py, ctl_b_path, work_dir, args.reps, args.long, args.skip_cn)

            accepted, reason = should_accept(cand_res, ctl_a, ctl_b)
            verdict = "ACCEPT" if accepted else "REJECT"
            append_plan(
                f"- `{time.strftime('%H:%M:%S')}` **{cand}** ({desc}): **{verdict}** — {reason} "
                f"(c4={cand_res.get('c4_aggregate')}, flags={cand_res.get('flags')})"
            )
            if accepted:
                best_state = cand_state
                _, best_cfg = write_cfg(f"best-{slugify(cand)}", best_state)
                (work_dir / "best-config.json").write_text(json.dumps(best_cfg, indent=2) + "\n")
                consecutive_rejects = 0
            else:
                consecutive_rejects += 1
                if consecutive_rejects >= 3:
                    append_plan("\n## STATUS: Plateau\nThree consecutive candidates landed inside noise or regressed.\n")
                    break
    finally:
        if last_cfg and last_cfg.get("stop"):
            subprocess.run(last_cfg["stop"], capture_output=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_work = "/content/fllm_hill_climb" if Path("/content").is_dir() else "fllm_hill_climb"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(optimize_fllm.PRESETS), default="qwen3.5-27b-dspine")
    ap.add_argument("--gpu", choices=["auto", *sorted(optimize_fllm.GPUS)], default="auto")
    ap.add_argument("--fllm-bin", default="./target/release/fllm-rs")
    ap.add_argument("--work", default=default_work)
    ap.add_argument("--bench-py", type=Path, help="Path to inference-optimisation scripts/bench.py")
    ap.add_argument("--reps", type=int, default=6, help="Decode samples per run (inference-optimisation >= 6)")
    ap.add_argument("--long", type=int, default=None, help="Optional long-context needle gate token length")
    ap.add_argument("--skip-cn", action="store_true", help="Pass --skip-cN to bench.py")
    ap.add_argument("--tune-spec-tokens", type=int, default=8, help="Draft length for +spec_tokens_tune")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dry-run", action="store_true", help="Print ceiling table, A/B plan, and validate configs without a GPU")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reps < 6:
        print(f"warning: inference-optimisation recommends --reps >= 6 (got {args.reps})", file=sys.stderr)

    gpu_name = detect_gpu() if args.gpu == "auto" else args.gpu
    if gpu_name is None:
        if args.dry_run:
            gpu_name = "a100-80g"
        else:
            sys.exit("Could not detect Colab GPU via nvidia-smi; pass --gpu explicitly (t4, l4, a100-40g, a100-80g).")

    ceilings = compute_ceilings(args.preset, gpu_name)
    bench_py = find_bench_py(args.bench_py)
    if args.dry_run:
        return print_dry_run(
            args.preset, gpu_name, args.fllm_bin, bench_py, ceilings,
            args.tune_spec_tokens, args.port, args.reps,
        )

    if bench_py is None:
        sys.exit("bench.py not found; pass --bench-py /path/to/inference-optimisation/scripts/bench.py")
    return run_live(args, gpu_name, bench_py)


if __name__ == "__main__":
    raise SystemExit(main())
