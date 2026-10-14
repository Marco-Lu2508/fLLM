#!/usr/bin/env python3
"""NVFP4 quality check: perplexity and token-by-token logprob agreement of fLLM (fast path, exact path, built-in kernel, activation-scale
variants) and optionally vLLM on one fixed text. Standard library only.

What it does
  1. downloads a text (default: Pride and Prejudice from Project Gutenberg, public domain) and cuts N tokens worth of it,
  2. scores the text with `fllm-rs ppl --device auto` once per fLLM configuration (an environment per configuration, see CONFIGS), every run on
     the same token ids (the first run tokenizes, `--dump-tokens` writes the ids, the others read them with `--token-ids`),
  3. scores the same ids with vLLM through its OpenAI API (`prompt_logprobs`) if a server is given (--vllm-url) or started (--vllm-serve MODEL),
  4. prints a table: mean logprob, perplexity, and for each configuration the difference of its per-token logprobs to the exact fLLM path and
     to vLLM (mean |d|, RMS d, max |d|), and writes everything to --out (results.json, one logprob file per configuration).

Reading the table
  * `exact` keeps activations in FP32 (FLLM_CUDA_EXACT=1); it is NOT the reference of a W4A4 checkpoint, vLLM is closer to what the model was
    calibrated for. A correct fast path should be about as close to vLLM as `exact` is, and its perplexity should not be worse than vLLM's.
  * `fast` vs `fast-act-off`: the effect of the per-tensor activation factor (FLLM_NVFP4_ACT_SCALE). `fast` vs `nocutlass`: CUTLASS prompt GEMM
    against the built-in kernel (they should agree to noise level).

Example (Colab, fLLM built with --features cuda, model converted with llama.cpp's convert_hf_to_gguf.py):
  python3 scripts/colab/nvfp4_quality.py --fllm target/release/fllm-rs --gguf Llama-3.1-8B-Instruct-NVFP4.gguf \\
      --tokens 4096 --vllm-serve nvidia/Llama-3.1-8B-Instruct-NVFP4 --out /content/nvfp4_quality
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_TEXT_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"

# name -> environment additions. Everything else is inherited, so FLLM_NVFP4_TILE etc. can be exported around the script.
CONFIGS = [
    ("fast", {}),
    ("exact", {"FLLM_CUDA_EXACT": "1"}),
    ("fast-act-off", {"FLLM_NVFP4_ACT_SCALE": "off"}),
    ("fast-act-dynamic", {"FLLM_NVFP4_ACT_SCALE": "dynamic"}),
    ("nocutlass", {"FLLM_CUDA_NO_CUTLASS": "1"}),
    ("nocutlass-act-off", {"FLLM_CUDA_NO_CUTLASS": "1", "FLLM_NVFP4_ACT_SCALE": "off"}),
    ("fast-fp32-intermediates", {"FLLM_NO_F16_NVFP4": "1", "FLLM_NO_F16_FFN": "1", "FLLM_NO_FUSED_ADD": "1"}),
]
EXTRA_TILE_CONFIGS = [("tile-auto", {"FLLM_NVFP4_TILE": "auto"}), ("tile-128x128x256-sk", {"FLLM_NVFP4_TILE": "128x128x256-sk"})]


def log(msg):
    print(msg, flush=True)


def fetch_text(url, path):
    if os.path.exists(path):
        return
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fllm-nvfp4-quality/1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)


def cut_text(src, dst, tokens):
    """About 6 characters per token for English prose, with margin; fllm-rs ppl keeps the first --ctx tokens."""
    with open(src, encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = re.search(r"\*\*\* START OF .*?\*\*\*", text)
    if m:
        text = text[m.end():]
    text = text.strip()
    text = text[: max(2000, tokens * 7)]
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)


def run_fllm(fllm, gguf, text, ctx, device, env_add, out_nll, ids_in=None, ids_out=None, timeout=3600):
    cmd = [fllm, "ppl", "--model", gguf, "--file", text, "--ctx", str(ctx), "--device", device, "--dump-nll", out_nll]
    if ids_in:
        cmd += ["--token-ids", ids_in]
    if ids_out:
        cmd += ["--dump-tokens", ids_out]
    env = dict(os.environ)
    env.update(env_add)
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({p.returncode}):\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return dt


def read_nll(path):
    """index token_id logprob per line -> (tokens, logprobs); index t scores token t given tokens[:t]."""
    toks, lps = [], []
    with open(path) as f:
        for line in f:
            a = line.split()
            if len(a) == 3:
                toks.append(int(a[1]))
                lps.append(float(a[2]))
    return toks, lps


def http_json(url, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def vllm_logprobs(base_url, model, ids):
    """Logprob of every token after the first, from /v1/completions with prompt_logprobs (the ids go in as a token prompt)."""
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    resp = http_json(base_url.rstrip("/") + "/v1/completions", body)
    pl = resp["choices"][0].get("prompt_logprobs")
    if not pl:
        raise RuntimeError("the vLLM server returned no prompt_logprobs (needs a recent vLLM)")
    out = []
    for t in range(1, len(ids)):
        entry = pl[t]
        if entry is None:
            raise RuntimeError(f"no prompt logprob at position {t}")
        v = entry.get(str(ids[t]))
        if v is None:
            raise RuntimeError(f"position {t}: the token itself is missing from prompt_logprobs")
        out.append(v["logprob"] if isinstance(v, dict) else float(v))
    return out


def start_vllm(model, port, extra):
    exe = shutil.which("vllm")
    cmd = [exe, "serve", model, "--port", str(port), "--max-model-len", "8192"] + extra if exe else [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", str(port), "--max-model-len", "8192"] + extra
    log("starting vLLM: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=open("vllm_server.log", "w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(900):
        if proc.poll() is not None:
            raise RuntimeError("vLLM exited, see vllm_server.log")
        try:
            http_json(base + "/v1/models", timeout=5)
            return proc, base
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2)
    proc.kill()
    raise RuntimeError("vLLM did not come up in 30 minutes")


def stats(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if not n:
        return (float("nan"),) * 3
    return sum(abs(x) for x in d) / n, math.sqrt(sum(x * x for x in d) / n), max(abs(x) for x in d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fllm", default="target/release/fllm-rs", help="fllm-rs binary built with --features cuda")
    ap.add_argument("--gguf", required=True, help="the NVFP4 GGUF (type 40)")
    ap.add_argument("--text", help="text file (default: downloaded)")
    ap.add_argument("--text-url", default=DEFAULT_TEXT_URL)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="nvfp4_quality")
    ap.add_argument("--only", help="comma separated configuration names")
    ap.add_argument("--tiles", action="store_true", help="also score with FLLM_NVFP4_TILE=auto and a stream-K tile (must match `fast`)")
    ap.add_argument("--vllm-url", help="running vLLM server, e.g. http://127.0.0.1:8000")
    ap.add_argument("--vllm-model", help="model name of the running server (default: first of /v1/models)")
    ap.add_argument("--vllm-serve", metavar="MODEL", help="start `vllm serve MODEL` for the run")
    ap.add_argument("--vllm-port", type=int, default=8011)
    ap.add_argument("--vllm-arg", action="append", default=[], help="extra argument for vllm serve (repeatable)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    text_full = args.text
    if not text_full:
        text_full = os.path.join(args.out, "source.txt")
        fetch_text(args.text_url, text_full)
    text = os.path.join(args.out, "text.txt")
    cut_text(text_full, text, args.tokens)

    configs = list(CONFIGS) + (EXTRA_TILE_CONFIGS if args.tiles else [])
    if args.only:
        keep = set(args.only.split(","))
        configs = [c for c in configs if c[0] in keep]
    ids_path = os.path.join(args.out, "token_ids.txt")
    results = {}
    tokens = None
    first = True
    for name, env_add in configs:
        nll_path = os.path.join(args.out, f"{name}.nll")
        log(f"== fLLM {name} {env_add or ''}")
        try:
            dt = run_fllm(args.fllm, args.gguf, text, args.tokens, args.device, env_add, nll_path,
                          ids_in=None if first else ids_path, ids_out=ids_path if first else None)
        except Exception as e:  # keep going: a configuration that does not exist in this build must not hide the others
            log(f"   failed: {e}")
            continue
        first = False
        toks, lps = read_nll(nll_path)
        if tokens is None:
            tokens = toks
        results[name] = {"logprobs": lps, "seconds": dt}
        log(f"   {len(lps)} tokens, mean logprob {sum(lps) / len(lps):.5f}, {dt:.0f} s")

    if tokens is None:
        sys.exit("no fLLM run succeeded")
    with open(ids_path) as f:
        all_ids = [int(x) for x in f.read().split()]

    vproc = None
    base = args.vllm_url
    try:
        if args.vllm_serve:
            vproc, base = start_vllm(args.vllm_serve, args.vllm_port, args.vllm_arg)
        if base:
            model = args.vllm_model or args.vllm_serve
            if not model:
                model = http_json(base.rstrip("/") + "/v1/models")["data"][0]["id"]
            log(f"== vLLM {model}")
            lps = vllm_logprobs(base, model, all_ids)
            results["vllm"] = {"logprobs": lps, "seconds": 0.0}
            log(f"   {len(lps)} tokens, mean logprob {sum(lps) / len(lps):.5f}")
    except Exception as e:
        log(f"vLLM part failed: {e}")
    finally:
        if vproc:
            vproc.terminate()

    n = min(len(r["logprobs"]) for r in results.values())
    ref_exact = results.get("exact", {}).get("logprobs")
    ref_vllm = results.get("vllm", {}).get("logprobs")
    head = f"{'configuration':<26}{'tokens':>7}{'mean lp':>10}{'ppl':>9}"
    if ref_exact:
        head += f"{'|d|~exact':>11}{'rms':>8}{'max':>7}"
    if ref_vllm:
        head += f"{'|d|~vllm':>11}{'rms':>8}{'max':>7}"
    log("\n" + head)
    log("-" * len(head))
    summary = {}
    for name, r in results.items():
        lp = r["logprobs"][:n]
        mean = sum(lp) / n
        line = f"{name:<26}{n:>7}{mean:>10.4f}{math.exp(-mean):>9.4f}"
        row = {"tokens": n, "mean_logprob": mean, "ppl": math.exp(-mean)}
        if ref_exact:
            s = stats(lp, ref_exact[:n])
            line += f"{s[0]:>11.4f}{s[1]:>8.4f}{s[2]:>7.2f}"
            row["vs_exact"] = {"mean_abs": s[0], "rms": s[1], "max": s[2]}
        if ref_vllm:
            s = stats(lp, ref_vllm[:n])
            line += f"{s[0]:>11.4f}{s[1]:>8.4f}{s[2]:>7.2f}"
            row["vs_vllm"] = {"mean_abs": s[0], "rms": s[1], "max": s[2]}
        log(line)
        summary[name] = row
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"gguf": args.gguf, "tokens": n, "summary": summary,
                   "logprobs": {k: v["logprobs"][:n] for k, v in results.items()}}, f)
    log(f"\nwritten to {args.out}/results.json")


if __name__ == "__main__":
    main()
