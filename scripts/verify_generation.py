#!/usr/bin/env python3
"""Greedy generation through the HTTP server must match transformers, also with concurrent
requests (this exercises incremental decoding and, for hybrid models, the recurrent state).

    python scripts/verify_generation.py --fllm target/release/fllm-rs qwen3_5 qwen3_next
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hf_golden  # noqa: E402

PROMPTS = ["The quick brown fox", "Die Hauptstadt von", "def add(a, b):", "fLLM runs"]
NEW = 14


def post(port, prompt):
    body = json.dumps({"model": "x", "prompt": prompt, "max_tokens": NEW, "temperature": 0}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["choices"][0]["text"]


def verify(arch, fllm, work, port):
    import transformers as T
    from tokenizers import Tokenizer

    src = work / f"{arch}-hf"
    hf_golden.build(arch, src)
    gguf = work / f"{arch}.gguf"
    subprocess.run([fllm, "convert", str(src), str(gguf), "--outtype", "f32", "--overwrite"], check=True, capture_output=True)
    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    model = T.AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    want = []
    for prompt in PROMPTS:
        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids])
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=NEW, do_sample=False, min_new_tokens=NEW)
        want.append(tok.decode(out[0, ids.shape[1] :].tolist(), skip_special_tokens=True))
    server = subprocess.Popen(
        [fllm, "serve", "--model", str(gguf), "--port", str(port), "--ctx", "128", "--max-seqs", "4", "--threads", "2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                if server.poll() is not None:
                    return {"arch": arch, "ok": False, "error": server.stdout.read().decode()[-400:]}
                time.sleep(0.5)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            got = list(pool.map(lambda p: post(port, p), PROMPTS))
    finally:
        server.terminate()
    # a random model may emit the end-of-sequence token that transformers was told to suppress
    same = [g == w or (len(g) > 0 and w.startswith(g)) for g, w in zip(got, want)]
    bad = [(g, w) for g, w, s in zip(got, want, same) if not s]
    return {"arch": arch, "ok": all(same), "identical": sum(same), "of": len(same), "mismatches": bad[:2]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archs", nargs="+")
    parser.add_argument("--fllm", required=True)
    parser.add_argument("--port", type=int, default=8741)
    args = parser.parse_args()
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for arch in args.archs:
            r = verify(arch, args.fllm, Path(tmp), args.port)
            failed += not r["ok"]
            print(json.dumps(r, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
