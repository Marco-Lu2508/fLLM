#!/usr/bin/env python3
"""Greedy continuation of a real (small) checkpoint: transformers on the CPU against `fllm-rs serve` on the CPU.

    python scripts/verify_real_model.py --fllm target/release/fllm-rs --model DIR_OR_GGUF [--reference DIR] [--tokens 24] [--threads 4]

* `--model` is what fLLM loads (a checkpoint directory or a GGUF file).
* `--reference` is the Hugging Face directory transformers loads (default: `--model` when it is a directory); pass `none`
  to skip the transformers run and only print fLLM's text.
The prompts are plain completions (no chat template). The comparison reports the number of identical leading tokens per
prompt; a divergence at a position where the reference has a near tie is not an error, so read the printed texts too.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request

PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5, 6, 7,",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Water boils at a temperature of",
]


def reference_texts(directory: str, tokens: int) -> list[list[int]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(directory)
    try:
        model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(directory, dtype=torch.float32)
    model.eval()
    out = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=tokens, do_sample=False)
        out.append(gen[0][ids["input_ids"].shape[1]:].tolist())
    return out, tok


def fllm_texts(fllm: str, model: str, tokens: int, threads: int) -> list[str]:
    port = 19500 + os.getpid() % 400
    proc = subprocess.Popen([fllm, "serve", "--model", model, "--device", "cpu", "--port", str(port), "--ctx", "512", "--threads", str(threads)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    try:
        url = f"http://127.0.0.1:{port}/v1/completions"
        texts = []
        for p in PROMPTS:
            body = json.dumps({"model": "m", "prompt": p, "max_tokens": tokens, "temperature": 0, "seed": 1}).encode()
            for _ in range(600):
                try:
                    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=600) as r:
                        texts.append(json.loads(r.read())["choices"][0]["text"])
                    break
                except Exception:
                    time.sleep(1)
            else:
                texts.append("FAILED")
        return texts
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fllm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    ref_dir = args.reference or (args.model if os.path.isdir(args.model) else None)
    mine = fllm_texts(args.fllm, args.model, args.tokens, args.threads)
    if not ref_dir or ref_dir == "none":
        for p, t in zip(PROMPTS, mine):
            print(json.dumps({"prompt": p, "fllm": t}))
        return 0
    ref_ids, tok = reference_texts(ref_dir, args.tokens)
    for p, t, ids in zip(PROMPTS, mine, ref_ids):
        want = tok.decode(ids, skip_special_tokens=True)
        common = 0
        for a, b in zip(t, want):
            if a != b:
                break
            common += 1
        print(json.dumps({"prompt": p, "same_leading_chars": common, "of": len(want), "identical": t == want, "fllm": t, "transformers": want}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
