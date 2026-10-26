#!/usr/bin/env python3
"""Which serving setting makes fLLM fastest under load on this GPU: one load test (scripts/colab/compare.py) per setting, one table.

    python3 scripts/colab/tune_load.py --fllm /content/fllm_compare/fllm-src/target/release/fllm-rs --model /content/fllm_compare/X.gguf

Each setting starts its own server (extra `serve` arguments and environment variables) and measures prompts of 100 and 2000 tokens with
16 and 4 simultaneous users. Next to the speeds the table shows where the time of a step goes, from the server's own counters
(/api/studio/status): model forward per call and the host work around it (scheduler, batch building, post-processing).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402

SETTINGS = [
    ("default", "", {}),
    ("prefill-first", "--chunked-prefill-policy prefill-first", {}),
    ("batch 1024", "--max-batched-tokens 1024", {}),
    ("batch 4096", "--max-batched-tokens 4096", {}),
    ("no CUDA graphs", "", {"FLLM_CUDA_GRAPH": "0"}),
    ("graphs for prefill", "", {"FLLM_CUDA_GRAPH_PREFILL": "1"}),
]


def status(port):
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/studio/status", timeout=5)).get("metrics", {})
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fllm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", default="100,2000")
    ap.add_argument("--users", default="16,4")
    ap.add_argument("--only", help="comma-separated setting names")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    users = [int(x) for x in args.users.split(",")]
    ctx = (max(sizes) + 464) * max(users)
    rows = []
    for name, extra, env in SETTINGS:
        if args.only and name not in args.only.split(","):
            continue
        port = 9300 + len(rows)
        cmd = [args.fllm, "serve", "--model", args.model, "--device", "cuda", "--port", str(port), "--ctx", str(ctx), "--max-seqs", str(max(users)), "--model-name", "m"] + extra.split()
        server = compare.Server(cmd, f"http://127.0.0.1:{port}/health", "m", env=env, logfile=f"/tmp/tune-{len(rows)}.log")
        if not server.ok:
            print(f"{name}: did not start: " + open(server.logfile).read()[-300:].replace("\n", " | "))
            server.stop()
            continue
        before = status(port)
        start = time.time()
        result = compare.measure(server, port, sizes, users, 64, {"chat_template_kwargs": {"enable_thinking": False}})
        wall = time.time() - start
        after = status(port)
        calls = max(1, after.get("forward_calls_total", 0) - before.get("forward_calls_total", 0))
        per = {k: (after.get(k, 0) - before.get(k, 0)) / calls for k in ("model_forward_ms", "scheduler_ms", "batch_build_ms", "postprocess_ms")}
        server.stop()
        time.sleep(3)
        rows.append((name, result, per, calls, wall))
        print(f"{name}: {calls} steps in {wall:.0f} s; per step forward {per['model_forward_ms']:.2f} ms, scheduler {per['scheduler_ms']:.3f}, "
              f"build {per['batch_build_ms']:.3f}, post {per['postprocess_ms']:.3f}", flush=True)
    print("\n=== TUNE RESULT ===  (total tok/s of all users | first token s)")
    head = "  ".join(f"{s:>5}x{u:<3}" + " " * 10 for s in sizes for u in users)
    print(f"{'setting':<20} {head}")
    for name, result, *_ in rows:
        cells = [f"{r.get('total_tok_s', 'err'):>7} | {r.get('first_s', '-'):>5}" for r in result]
        print(f"{name:<20} " + "  ".join(f"{c:>19}" for c in cells))


if __name__ == "__main__":
    main()
