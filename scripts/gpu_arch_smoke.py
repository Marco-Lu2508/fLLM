#!/usr/bin/env python3
"""Greedy completions of tiny random-weight GGUF files on the CPU and on an accelerator, compared token by token.

Random models have near-tied logits, so the check is "same text for the first tokens" and a report of where they diverge; it
shows that an architecture loads and runs on a device (including the host fallback of layers the device has no kernel for).

    python scripts/gpu_arch_smoke.py --fllm ./fllm-rs --device vulkan:1 a.gguf b.gguf ...
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.request


def complete(fllm: str, gguf: str, device: str, port: int, tokens: int, prompt: str) -> str:
    log = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen([fllm, "serve", "--model", gguf, "--device", device, "--port", str(port), "--ctx", "256", "--threads", "4"],
                            stdout=subprocess.DEVNULL, stderr=log, text=True, preexec_fn=os.setsid)
    try:
        url = f"http://127.0.0.1:{port}/v1/completions"
        for _ in range(120):
            if proc.poll() is not None:
                log.seek(0)
                return "FAILED: " + log.read()[-300:].replace("\n", " ")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
                    name = json.loads(r.read())["data"][0]["id"]
                body = json.dumps({"model": name, "prompt": prompt, "max_tokens": tokens, "temperature": 0, "seed": 1}).encode()
                req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read())["choices"][0]["text"]
            except Exception:
                time.sleep(1)
        return "FAILED: no response"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fllm", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--prompt", default="the cat sat on the mat and", help="a short prompt (\"the cat\") keeps the prefill small")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    port = 19000 + (os.getpid() % 500) * 2
    for m in args.models:
        cpu = complete(args.fllm, m, "cpu", port, args.tokens, args.prompt)
        gpu = complete(args.fllm, m, args.device, port + 1, args.tokens, args.prompt)
        same = cpu == gpu and not cpu.startswith("FAILED")
        print(json.dumps({"model": os.path.basename(m), "device": args.device, "same_text": same, "cpu": cpu[:80], "device_text": gpu[:80]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
