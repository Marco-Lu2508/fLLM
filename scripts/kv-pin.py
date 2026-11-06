#!/usr/bin/env python3
"""Pinning a shared prefix, on a real model.

A small KV pool (--ctx 2048 --max-seqs 2) is pushed through with twelve other documents. Three runs, same prompts:
  A. no pin, no tier        -> the shared prefix is gone (cached 0)           the control
  B. pinned, no tier        -> the pin kept it (cached > 0)
  C. disk tier, the server is restarted; the pin is the first call (before any request), then the pool is pushed through; with a control that is not pinned.
Every figure is printed as measured.

Usage: kv-pin.py <fllm-rs> <model.gguf>
"""
import json, os, shutil, signal, subprocess, sys, time, urllib.error, urllib.request

BIN, MODEL = sys.argv[1:3]
PORT = 8181
DIR = "/tmp/fllm-kvpin"
OTHERS = 12


def doc(k, sections=22):
    return " ".join(f"Document {k} section {i}: unit {k}-{i} records value {(i * 37 + k * 11) % 101} and reports it every {i % 7 + 2} seconds." for i in range(1, sections))


SHARED = doc(0)


def serve(tag, disk):
    env = {**os.environ, "FLLM_OUTPUT_CACHE": "off"}
    for k in ("FLLM_KV_DISK_DIR", "FLLM_KV_RAM_BYTES"):
        env.pop(k, None)
    if disk:
        env["FLLM_KV_DISK_DIR"] = DIR
        env["FLLM_KV_DISK_BYTES"] = str(512 << 20)
    proc = subprocess.Popen([BIN, "serve", "--model", MODEL, "--device", "cpu", "--port", str(PORT), "--speculative", "off", "--ctx", "2048", "--max-seqs", "2"],
                            env=env, stdout=subprocess.DEVNULL, stderr=open(f"/tmp/fllm-kvpin-{tag}.log", "w"), start_new_session=True)
    for _ in range(120):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise SystemExit(f"{tag}: server exited, see /tmp/fllm-kvpin-{tag}.log")
    raise SystemExit(f"{tag}: not ready")


def stop(proc):
    os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=60)


def call(path, body=None, method=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method=method)
    try:
        return json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{path}: HTTP {e.code} {e.read()[:300]}")


def ask(text, question="What does the first unit record?"):
    mid = call("/v1/models")["data"][0]["id"]
    t = time.perf_counter()
    r = call("/v1/chat/completions", {"model": mid, "max_tokens": 16, "temperature": 0, "messages": [{"role": "system", "content": text}, {"role": "user", "content": question}]})
    return time.perf_counter() - t, r["usage"]["prompt_tokens"], (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0), r["choices"][0]["message"]["content"]


def pin():
    # only the shared system part: what every developer sends first
    return call("/admin/prefix-cache/pin", {"messages": [{"role": "system", "content": SHARED}]})


def push_through():
    for k in range(1, OTHERS + 1):
        ask(doc(k))


def run(tag, disk, pinned):
    p = serve(tag, disk)
    try:
        t0, n, c0, text0 = ask(SHARED)
        print(f"[{tag}] shared prompt cold: {t0:.2f} s, {n} tokens, cached {c0}", flush=True)
        time.sleep(3)
        if pinned:
            print(f"[{tag}] pin: {pin()}", flush=True)
        push_through()
        t1, _, c1, text1 = ask(SHARED)
        print(f"[{tag}] after {OTHERS} other documents: {t1:.2f} s, cached {c1} of {n}, text equal: {text1 == text0}, cache holds {call('/admin/prefix-cache').get('entries')} blocks, pinned {call('/admin/prefix-cache').get('pinned')}", flush=True)
        return text0
    finally:
        stop(p)


def restart_case(pinned):
    """Process 1 sees the shared prompt (written to the disk tier); process 2 starts empty. With `pinned` the pin is the FIRST call, before any
    request, so the only way the prefix can be in the cache is that the pin read it from the disk; then twelve other documents go through."""
    tag = "C-pin-first" if pinned else "C-control"
    shutil.rmtree(DIR, ignore_errors=True)
    os.makedirs(DIR)
    p = serve(tag + "-1", True)
    try:
        _, n, _, text0 = ask(SHARED)
        time.sleep(3)
    finally:
        stop(p)
    p = serve(tag + "-2", True)
    try:
        print(f"[{tag}] after the restart: cache holds {call('/admin/prefix-cache').get('entries')} blocks", flush=True)
        if pinned:
            print(f"[{tag}] pin (first call): {pin()}  reloaded_blocks {call('/admin/kv-tier').get('reloaded_blocks')}", flush=True)
        push_through()
        t, _, c, text = ask(SHARED)
        tier = call("/admin/kv-tier")
        print(f"[{tag}] after {OTHERS} other documents: {t:.2f} s, cached {c} of {n}, text equal: {text == text0}, reloaded_blocks {tier.get('reloaded_blocks')}, pinned {call('/admin/prefix-cache').get('pinned')}", flush=True)
    finally:
        stop(p)
    shutil.rmtree(DIR, ignore_errors=True)


restart_case(True)
restart_case(False)
