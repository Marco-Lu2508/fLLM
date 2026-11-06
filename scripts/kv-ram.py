#!/usr/bin/env python3
"""The host-memory tier read back WHILE the server runs, on a real model (no disk folder at all).

A small KV pool (--ctx 2048 --max-seqs 2: about 256 blocks of 16 tokens) is filled with 12 other documents of ~45 blocks each, so the prefix
cache must have dropped the first document (the oldest leaves go first; the first document is NOT asked again in between, which would
refresh it). Asked again, the server with the memory tier must read the dropped blocks back (cached tokens > 0, reloaded_blocks > 0, same
text); the control without a tier must compute it again (cached 0). Every figure printed is a measurement, nothing is assumed.

Usage: kv-ram.py <fllm-rs> <model.gguf>
"""
import json, os, shutil, signal, subprocess, sys, time, urllib.request

BIN, MODEL = sys.argv[1:3]
PORT = 8171
DIR = "/tmp/fllm-kvram-unused"
OTHERS = 12


def doc(k, sections=22):
    return " ".join(f"Document {k} section {i}: unit {k}-{i} records value {(i * 37 + k * 11) % 101} and reports it every {i % 7 + 2} seconds." for i in range(1, sections))


def serve(tag, disk):
    env = {**os.environ, "FLLM_OUTPUT_CACHE": "off"}
    env.pop("FLLM_KV_DISK_DIR", None)
    env.pop("FLLM_KV_RAM_BYTES", None)
    if disk:  # in this script "disk" means: the memory tier is on
        env["FLLM_KV_RAM_BYTES"] = str(512 << 20)
    proc = subprocess.Popen([BIN, "serve", "--model", MODEL, "--device", "cpu", "--port", str(PORT), "--speculative", "off", "--ctx", "2048",
                             "--max-seqs", "2"], env=env, stdout=subprocess.DEVNULL, stderr=open(f"/tmp/fllm-kvram-{tag}.log", "w"),
                            start_new_session=True)
    for _ in range(120):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise SystemExit(f"{tag}: server exited, see /tmp/fllm-kvram-{tag}.log")
    raise SystemExit(f"{tag}: not ready")


def stop(proc):
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=60)


def get(path):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=30))


def ask(text, question="What does the first unit record?"):
    mid = get("/v1/models")["data"][0]["id"]
    body = {"model": mid, "max_tokens": 16, "temperature": 0, "messages": [{"role": "system", "content": text}, {"role": "user", "content": question}]}
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                                                  headers={"Content-Type": "application/json"}), timeout=600))
    cached = (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return time.perf_counter() - t, r["usage"]["prompt_tokens"], cached, r["choices"][0]["message"]["content"]


def run(tag, disk):
    if disk:
        shutil.rmtree(DIR, ignore_errors=True)
        os.makedirs(DIR)
    p = serve(tag, disk)
    try:
        a = doc(0)
        t0, n, c0, text0 = ask(a)
        print(f"[{tag}] A cold: {t0:.2f} s, {n} prompt tokens, cached {c0}", flush=True)
        time.sleep(3)  # the spill writer is a background thread
        for k in range(1, OTHERS + 1):
            ask(doc(k))
        blocks = get("/admin/prefix-cache").get("entries")
        t2, _, c2, text2 = ask(a)
        print(f"[{tag}] after {OTHERS} other documents the cache held {blocks} blocks; A again: {t2:.2f} s, cached {c2} of {n}, text equal: {text2 == text0}", flush=True)
        if disk:
            tier = get("/admin/kv-tier")
            print(f"[{tag}] tier: reloaded_blocks {tier.get('reloaded_blocks')}, memory hits {tier.get('ram_hits')}, misses {tier.get('misses')}, written {tier.get('spills')}, memory used {tier.get('ram_used_bytes')} B in {tier.get('ram_blocks')} blocks", flush=True)
        t3, _, c3, _ = ask(a)
        print(f"[{tag}] A once more (now in the cache again): {t3:.2f} s, cached {c3}", flush=True)
    finally:
        stop(p)


for tag, disk in (("ram", True), ("control", False)):
    run(tag, disk)
shutil.rmtree(DIR, ignore_errors=True)
