#!/usr/bin/env python3
"""Where does a decode or verification step of a big MoE GGUF go on the CPU?

Starts `fllm-rs serve` on the CPU with `FLLM_CPU_OPS=1` (per-operation wall-clock totals of every forward pass, see docs/cpu-kernels.md),
runs two probes and prints a table:

* decode: a short prompt and `--gen` new tokens, one row per pass (what a plain decode step costs);
* chunks: a long prompt processed with `--max-batched-tokens` set to each of `--rows` (1, 2, 4, 8, 16 ...), so that every pass carries exactly
  that many consecutive rows of one sequence. That is the shape of the verification pass of speculative decoding (the last accepted token and
  the draft tokens), so "ms per row" against rows shows how much the experts, which several rows share, are read once.

    python3 bench_cpu_moe.py --binary fllm-rs --model MODEL.gguf [--threads N] [--gen 24] [--rows 1,2,4,8,16] [--env KEY=VAL ...] [--json OUT]

Use a real model, or `scripts/synth_qwen4exp_gguf.py` for a file with the sizes of Qwen3.8-Flash-Next but random weights. The median over the
passes of each probe is reported, the first pass of a probe is dropped. Unless `--cold` is given a long prompt is run first, which
routes to nearly every expert, so the page cache holds the weights (if they fit) and the numbers are compute time, not storage time.
"""
import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

PORT = 18211
WORDS = ("river stone window garden copper winter harvest lantern meadow compass velvet thunder orchard marble whisper anchor canyon ember "
         "falcon glacier harbor island jasmine kettle lagoon monsoon nectar oyster pebble quartz ripple saffron timber umbra violet willow zephyr").split()


def prompt(seed, words):
    x = seed * 2654435761 % (1 << 32)
    out = []
    for _ in range(words):
        x = (x * 1103515245 + 12345) % (1 << 31)
        out.append(WORDS[x % len(WORDS)])
    return "List: " + " ".join(out)


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def passes(log, start):
    with open(log, errors="replace") as f:
        f.seek(start)
        text = f.read()
    out = []
    for m in re.finditer(r'\{"cpu_ops":.*\}\}', text):
        try:
            out.append(json.loads(m.group(0))["cpu_ops"])
        except ValueError:
            pass
    return out, start + len(text.encode())


def summarize(items, rows):
    sel = [p for p in items if p["rows"] == rows][1:] or [p for p in items if p["rows"] == rows]
    if not sel:
        return None
    keys = sel[0]["ms"].keys()
    ms = {k: statistics.median(p["ms"][k] for p in sel) for k in keys}
    return {"passes": len(sel), "ms": ms}


def run_server(args, max_batched):
    log = os.path.join(tempfile.gettempdir(), f"bench_cpu_moe_{max_batched}.log")
    cmd = [args.binary, "serve", "--model", args.model, "--model-name", "m", "--device", "cpu", "--port", str(PORT), "--ctx", "4096",
           "--max-batched-tokens", str(max_batched)] + (["--threads", str(args.threads)] if args.threads else []) + args.extra
    env = dict(os.environ, FLLM_CPU_OPS="1")
    for kv in args.env:
        k, v = kv.split("=", 1)
        env[k] = v
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
    end = time.time() + 1800
    while time.time() < end:
        if proc.poll() is not None:
            raise SystemExit(f"server exited, see {log}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/ready", timeout=3).read()
            return proc, log
        except Exception:
            time.sleep(2)
    raise SystemExit("server did not become ready")


def stop(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(2)


def completion(text, n):
    return post("/v1/completions", {"model": "m", "prompt": text, "max_tokens": n, "temperature": 0, "ignore_eos": True, "min_tokens": n})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--rows", default="1,2,4,8,16")
    ap.add_argument("--chunk-words", type=int, default=300, help="length of the long prompt of the chunk probe")
    ap.add_argument("--cold", action="store_true", help="do not warm the page cache with a long prompt first")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL for the server")
    ap.add_argument("--json", default="")
    ap.add_argument("extra", nargs="*", help="more arguments for `fllm-rs serve` (after --)")
    args = ap.parse_args()
    result = {"model": os.path.basename(args.model), "threads": args.threads, "env": args.env, "extra": args.extra}

    # decode probe
    proc, log = run_server(args, 512)
    try:
        completion(prompt(1, 8), 2)  # warm-up (thread pools)
        if not args.cold:
            completion(prompt(5, args.chunk_words), 1)  # a long prompt routes to nearly every expert: the weights are in the page cache afterwards
        pos = os.path.getsize(log)
        t0 = time.time()
        r = completion(prompt(2, 8), args.gen)
        wall = time.time() - t0
        items, pos = passes(log, pos)
        one = summarize(items, 1)
        n = r["usage"]["completion_tokens"]
        result["decode"] = {"tokens": n, "wall_s": round(wall, 2), "tok_s_wall": round(n / wall, 3), "pass": one}
        if one:
            total = sum(one["ms"].values())
            result["decode"]["sum_ms"] = round(total, 1)
    finally:
        stop(proc)

    # chunk probes
    result["chunks"] = {}
    for rows in [int(x) for x in args.rows.split(",") if x]:
        if rows == 1:
            continue
        proc, log = run_server(args, rows)
        try:
            completion(prompt(3, 8), 2)
            if not args.cold:
                completion(prompt(5, args.chunk_words), 1)
            pos = os.path.getsize(log)
            completion(prompt(10 + rows, args.chunk_words), 1)
            items, pos = passes(log, pos)
            s = summarize(items, rows)
            if s:
                s["sum_ms"] = round(sum(s["ms"].values()), 1)
                s["ms_per_row"] = round(s["sum_ms"] / rows, 1)
            result["chunks"][rows] = s
        finally:
            stop(proc)

    d = result["decode"]
    print(f"model {result['model']}  threads {args.threads or 'auto'}  env {args.env}")
    if d.get("pass"):
        print(f"decode: {d['tok_s_wall']} tok/s wall over {d['tokens']} tokens; one pass (1 row) = {d['sum_ms']} ms")
        print("   " + "  ".join(f"{k} {v:.1f}" for k, v in d["pass"]["ms"].items()))
    base = d.get("sum_ms")
    for rows, s in result["chunks"].items():
        if not s:
            print(f"rows {rows}: no passes seen")
            continue
        rel = f"  = {s['sum_ms'] / base:.2f} x decode" if base else ""
        print(f"rows {rows:>2}: {s['sum_ms']} ms per pass, {s['ms_per_row']} ms per row{rel}")
        print("   " + "  ".join(f"{k} {v:.1f}" for k, v in s["ms"].items()))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
