#!/usr/bin/env python3
"""Compare fLLM with vLLM, SGLang, llama.cpp, ik_llama.cpp and Ollama on the same Colab GPU (or, with --cpu, on the processor), with one and with several users.

Run it in a Colab cell on a GPU runtime (T4, L4, A100, H100, G4), from a checkout of the repository:

    !git clone -q https://github.com/Marco-Lu2508/fLLM && python3 fLLM/scripts/colab/compare.py

Two groups of models, because the engines read different formats:

  * GGUF models (Q4_K_M): fLLM, llama.cpp and Ollama serve the same file.
  * Hugging Face 4-bit models (AWQ INT4): vLLM and SGLang serve the checkpoint, fLLM serves the same checkpoint converted with
    `fllm-rs convert` (the same weights and the same 4-bit scheme).

Steps (each is skipped with a reason when it cannot run, none stops the others):

  1. fLLM       the CUDA binary of the latest release (sha256-checked), or --fllm PATH, or --build-from-source (main, 5-25 minutes)
  2. vLLM       installed into its own environment (10+ minutes; --no-vllm skips it)
  3. SGLang     installed into its own environment (10+ minutes; --no-sglang skips it)
  4. llama.cpp  built from source with CUDA (llama-server, 5-15 minutes; --no-llama skips it)
     ik_llama.cpp  the same build of ikawrakow's fork (10-30 minutes with CUDA; --no-ik skips it)
  5. Ollama     the official Linux install (a few minutes; --no-ollama skips it), started on a private port and model folder
  6. models     --models, default llama8b (GGUF), llama8b-awq (AWQ) and qwen2b (GGUF)
  7. measure    every engine serves the model through /v1/chat/completions with streaming; for each approximate prompt size
                (100, 500, 2000) and number of simultaneous users (1, 4, 16) requests start together with distinct, reproducible
                prompt nonces and seeds. Generate up to 64 tokens. Server usage supplies actual token counts, never SSE chunks.
                Report aggregate and per-user end-to-end tok/s and time to first nonempty output delta. Without usage, tok/s
                is unavailable. fLLM runs with prefix/output caches and speculation explicitly disabled as a control.

The result is a table per model on stdout and compare_result.json in the work directory (default /content/fllm_compare).
Only the Python standard library is used; the network is used for the downloads.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import shlex
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

REPO = "Marco-Lu2508/fLLM"
# key: (label, GGUF URL or Hugging Face repo, smallest GPU memory in MiB)
MODELS = {
    "llama8b": ("Llama-3.1-8B-Instruct Q4_K_M (GGUF)", "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 7000),
    "llama8b-awq": ("Llama-3.1-8B-Instruct AWQ INT4", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", 7000),
    "qwen2b": ("Qwen3.5-2B Q4_K_M (GGUF)", "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf", 3000),
    "qwen3b-awq": ("Qwen2.5-3B-Instruct AWQ", "Qwen/Qwen2.5-3B-Instruct-AWQ", 4000),
    "gemma4": ("Gemma 4 E4B Q4_K_M (GGUF)", "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf", 7000),
    # the quant matrix: one model, every common GGUF quant (llama.cpp, ik_llama.cpp and fLLM read the same file)
    "llama8b-q2k": ("Llama-3.1-8B-Instruct Q2_K (GGUF)", "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q2_K.gguf", 5500),
    "llama8b-iq4xs": ("Llama-3.1-8B-Instruct IQ4_XS (GGUF)", "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf", 7000),
    "llama8b-q5km": ("Llama-3.1-8B-Instruct Q5_K_M (GGUF)", "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf", 8500),
    "llama8b-q6k": ("Llama-3.1-8B-Instruct Q6_K (GGUF)", "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf", 9500),
    "llama8b-q8": ("Llama-3.1-8B-Instruct Q8_0 (GGUF)", "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", 12000),
    "qwen3-8b": ("Qwen3-8B Q4_K_M (GGUF)", "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf", 8000),
    "qwen3-8b-q8": ("Qwen3-8B Q8_0 (GGUF)", "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf", 13000),
    "llama3b": ("Llama-3.2-3B-Instruct Q4_K_M (GGUF)", "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf", 4000),
    "llama3b-q8": ("Llama-3.2-3B-Instruct Q8_0 (GGUF)", "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf", 5500),
    "qwen3-30b-a3b": ("Qwen3-30B-A3B Q4_K_M (GGUF, MoE)", "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf", 22000),
}
GGUF_ENGINES = ("fllm", "llama.cpp", "ik_llama.cpp", "ollama")
HF_ENGINES = ("fllm", "vllm", "sglang")
SENTENCE = "The quick brown fox jumps over the lazy dog. "  # about 10 tokens
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def sh(cmd, timeout=None, env=None):
    e = dict(os.environ)
    e["PATH"] = os.path.expanduser("~/.cargo/bin") + ":/usr/local/cuda/bin:" + e["PATH"]
    if env:
        e.update(env)
    try:
        p = subprocess.run(["bash", "-lc", cmd], env=e, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def download(url, path):
    if not os.path.exists(path):
        log(f"download {os.path.basename(path)}")
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


# ---- engines -------------------------------------------------------------------------------------------------------------

def get_fllm(args, work):
    if args.fllm:
        return args.fllm
    if args.build_from_source:
        src = os.path.join(work, "fllm-src")
        if not os.path.exists(src):
            sh(f"git clone -q --depth 1 https://github.com/{REPO} {src}")
        if shutil.which("cargo") is None and not os.path.exists(os.path.expanduser("~/.cargo/bin/cargo")):
            sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
        rc, cc = sh("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
        arch = cc.strip().splitlines()[0].replace(".", "") if rc == 0 and cc.strip() else "89"
        if arch in ("90", "100", "120") and not os.path.exists(f"{src}/third_party/cutlass"):
            sh(f"git clone -q --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass {src}/third_party/cutlass")
        log(f"building fLLM from source for sm_{arch} ...")
        rc, out = sh(f"cd {src} && FLLM_CUDA_ARCHS={arch} cargo build --release --locked --features cuda -p fllm-engine --bin fllm-rs 2>&1 | tail -3", timeout=3 * 3600)
        binary = f"{src}/target/release/fllm-rs"
        return binary if os.path.exists(binary) else None
    # `releases/latest` skips pre-releases and all 0.x releases are pre-releases: take the newest one that is not a draft
    rel = next(r for r in json.load(urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/releases?per_page=5")) if not r.get("draft") and not str(r.get("tag_name", "")).startswith("colab-"))
    assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    name = "fllm-linux-x86_64-cuda.tar.gz"
    if name not in assets:
        log("the latest release has no CUDA archive")
        return None
    tgz = download(assets[name], os.path.join(work, name))
    if name + ".sha256" in assets:
        want = urllib.request.urlopen(assets[name + ".sha256"]).read().decode().split()[0]
        got = hashlib.sha256(open(tgz, "rb").read()).hexdigest()
        if want != got:
            log("sha256 of the fLLM archive does not match")
            return None
    out = os.path.join(work, "fllm-bin")
    with tarfile.open(tgz) as t:
        t.extractall(out)
    for root, _, files in os.walk(out):
        if "fllm-rs" in files:
            return os.path.join(root, "fllm-rs")
    return None


def get_llama(work, cpu=False, fork="llama.cpp"):
    """llama-server of llama.cpp (fork="llama.cpp") or of ik_llama.cpp (fork="ik_llama.cpp"), built with CUDA for this card or CPU only."""
    url = {"llama.cpp": "https://github.com/ggml-org/llama.cpp", "ik_llama.cpp": "https://github.com/ikawrakow/ik_llama.cpp"}[fork]
    tag = "cpu" if cpu else "cuda"
    src = os.path.join(work, f"{fork}-{tag}")
    binary = os.path.join(src, "build/bin/llama-server")
    if os.path.exists(binary):
        return binary
    log(f"building {fork} ({tag}) ...")
    sh(f"git clone -q --depth 1 {url} {src}")
    flags = "-DGGML_CUDA=OFF" if cpu else "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native"
    rc, out = sh(f"cd {src} && cmake -B build {flags} -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null "
                 f"&& cmake --build build --config Release -j {os.cpu_count() or 4} --target llama-server 2>&1 | tail -3", timeout=3 * 3600)
    if not os.path.exists(binary):
        log(f"{fork} did not build: " + out[-300:].replace("\n", " | "))
    return binary if os.path.exists(binary) else None


def get_ollama():
    if shutil.which("ollama"):
        return shutil.which("ollama")
    log("installing Ollama ...")
    sh("apt-get install -y -qq zstd > /dev/null 2>&1; curl -fsSL https://ollama.com/install.sh | sh", timeout=1800)
    return shutil.which("ollama")


def get_vllm(work):
    vbin = os.path.join(work, "vllm_env/bin/vllm")
    if not os.path.exists(vbin):
        log("installing vLLM ...")
        # uv instead of venv: Colab lacks python3-venv
        sh(f"pip install -q uv && uv venv --clear --python 3.12 {work}/vllm_env && uv pip install -q --python {work}/vllm_env/bin/python vllm", timeout=3600)
    return vbin if os.path.exists(vbin) else None


def get_sglang(work):
    py = os.path.join(work, "sglang_env/bin/python")
    if not os.path.exists(py) or sh(f"{py} -c 'import sglang'")[0] != 0:
        log("installing SGLang ...")
        sh(f"pip install -q uv && uv venv --clear --python 3.12 {work}/sglang_env && uv pip install -q --python {py} 'sglang[all]'", timeout=3600)
    return py if sh(f"{py} -c 'import sglang'")[0] == 0 else None


def hf_download(repo, dest):
    """All files of a Hugging Face model repository (weights, config, tokenizer)."""
    os.makedirs(dest, exist_ok=True)
    info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}"))
    for f in info["siblings"]:
        n = f["rfilename"]
        if n.endswith((".md", ".png", ".jpg")) or n.startswith(".") or "/" in n:
            continue
        download(f"https://huggingface.co/{repo}/resolve/main/{n}", os.path.join(dest, n))
    return dest


def wait_health(url, proc, seconds=900):
    for _ in range(seconds // 2):
        time.sleep(2)
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            if proc.poll() is not None:
                return False
    return False


class Server:
    """One engine serving one model on 127.0.0.1:port; `model` is the name to put in requests."""

    def __init__(self, cmd, health, model, env=None, logfile="server.log", after=None):
        e = dict(os.environ)
        e["PATH"] = "/usr/local/cuda/bin:" + e["PATH"]
        if env:
            e.update(env)
        self.proc = subprocess.Popen(cmd, stdout=open(logfile, "w"), stderr=subprocess.STDOUT, env=e)
        self.model = model
        self.ok = wait_health(health, self.proc)
        if self.ok and after:
            self.ok = after()
        self.logfile = logfile

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except Exception:
                self.proc.kill()


# ---- load test -----------------------------------------------------------------------------------------------------------

def sse_data(response):
    """Decode SSE framing (including comments, CRLF and multi-line data)."""
    data = []
    for line in response:
        line = line.rstrip(b"\r\n")
        if not line:
            if data:
                yield b"\n".join(data)
                data = []
        elif line.startswith(b"data:"):
            data.append(line[5:].removeprefix(b" "))
    if data:
        yield b"\n".join(data)


def usage_count(usage, name):
    value = usage.get(name)
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"invalid usage count: {name}")
    return value


def one_request(url, model, size, tokens, extra, out, seed=None):
    seed = 0 if seed is None else seed
    text = SENTENCE * max(1, size // 10)
    body = {"model": model, "stream": True, "max_tokens": tokens, "temperature": 0,
            "messages": [{"role": "user", "content": f"Run {seed}: count the words of this text and then count to twenty. {text}"}]}
    body.update(extra)
    # These instrument settings must not be accidentally replaced by engine extras.
    body["seed"] = seed
    body["stream"] = True
    body["stream_options"] = {**body.get("stream_options", {}), "include_usage": True}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    first, chunks, sample, usage, done = None, 0, "", {}, False
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for data in sse_data(r):
                if data.strip() == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise ValueError(f"SSE error: {event['error']}")
                if isinstance(event.get("usage"), dict):
                    # Usage is cumulative, not a per-event increment; the final counts win.
                    usage.update(event["usage"])
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    parts = [delta.get(k) for k in ("content", "reasoning_content", "reasoning")]
                    visible = "".join(p for p in parts if isinstance(p, str) and p)
                    if visible:
                        if first is None:
                            first = time.perf_counter() - t
                        chunks += 1  # output-bearing deltas, NOT tokenizer tokens
                        sample = (sample + visible)[:160]
        if not done:
            raise ValueError("incomplete SSE stream: missing [DONE]")
        prompt = usage_count(usage, "prompt_tokens")
        completion = usage_count(usage, "completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        cached = usage_count(details, "cached_tokens")
        if cached is not None and prompt is not None and cached > prompt:
            raise ValueError("cached_tokens exceeds prompt_tokens")
        result = {"first": first, "total": time.perf_counter() - t, "chunks": chunks, "sample": sample,
                  "seed": seed, "prompt_tokens": prompt, "completion_tokens": completion, "cached_tokens": cached,
                  "token_source": "usage" if completion is not None else "sse_chunks_unverified"}
        if completion is None:
            result["measurement_warning"] = "No completion usage: chunks are not tokens; tok/s unavailable."
        out.append(result)
    except Exception as ex:
        out.append({"error": str(ex)[:120], "seed": seed})


def request_seed(seed, size, users, index):
    """Stable, engine-independent prompt nonce AND sampler seed, unique within a batch."""
    key = f"{seed}:{size}:{users}".encode()
    base = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return (base + index) % (2**31)


def summarize_requests(size, users, out, wall):
    # Preserve legacy JSON keys, but never populate tok/s with a chunk estimate.
    out = sorted(out, key=lambda o: o["seed"])
    row = {"prompt": size, "users": users, "requests": out}
    if len(out) != users or users < 1:
        row["error"] = f"received {len(out)} request results for {users} users"
        return row
    errors = [o["error"] for o in out if "error" in o]
    if errors:
        row["error"] = errors[0]
        return row
    sources = {o["token_source"] for o in out}
    measured = sources == {"usage"}
    row["token_source"] = next(iter(sources)) if len(sources) == 1 else "mixed"
    for name in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        values = [o[name] for o in out]
        row[name] = sum(values) if all(v is not None for v in values) else None
    row["actual_prompt_tokens"] = [o["prompt_tokens"] for o in out]
    row["total_tok_s"] = round(row["completion_tokens"] / wall, 1) if measured and wall > 0 else None
    # The first delta can contain multiple tokens. (completion - 1)/(total - first)
    # would still invent a decode rate. Report the honest per-user E2E rate instead.
    each = [o["completion_tokens"] / o["total"] for o in out if measured and o["total"] > 0]
    row["each_tok_s"] = round(sum(each) / len(each), 1) if len(each) == len(out) else None
    row["each_tok_s_basis"] = "end_to_end_including_ttft"
    row["first_s"] = round(sum(o["first"] for o in out) / len(out), 2) if all(o["first"] is not None for o in out) else None
    row["chunks"] = sum(o["chunks"] for o in out)
    row["total_chunks_s"] = round(row["chunks"] / wall, 1) if wall > 0 else None
    if not measured:
        row["measurement_warning"] = "Missing completion usage for at least one request: tok/s unavailable; total_chunks_s is not tok/s."
    return row


def measure(server, port, sizes, users, tokens, extra, seed=0):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    warm = []
    one_request(url, server.model, 50, 8, extra, warm, seed=request_seed(seed, 50, 0, 0))
    fixed = []
    one_request(url, server.model, 100, 40, extra, fixed, seed=7)  # same answer probe on every engine
    rows = [{"prompt": 100, "users": 0, **fixed[0]}]
    for size in sizes:
        for n in users:
            out = []
            threads = [threading.Thread(target=one_request, args=(url, server.model, size, tokens, extra, out,
                        request_seed(seed, size, n, i))) for i in range(n)]
            t = time.perf_counter()
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            wall = time.perf_counter() - t
            rows.append(summarize_requests(size, n, out, wall))
            log(f"  prompt {size:5} users {n}: " + str({k: v for k, v in rows[-1].items() if k != "requests"}))
    return rows


def fllm_control_options(extra_args):
    """The plain-decoding control must not silently enable a different workload."""
    args = shlex.split(extra_args)
    if any(a.startswith(("--speculative", "--num-speculative-tokens")) or a in ("-sp", "-sw") for a in args):
        raise ValueError("--fllm-args cannot override speculation in the control measurement")
    return args + ["--speculative", "off"], {"FLLM_PREFIX_CACHE": "off", "FLLM_OUTPUT_CACHE": "off",
                                             "FLLM_SPECULATIVE_DEFAULT": "off"}


# ---- main ----------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", default="/content/fllm_compare" if os.path.exists("/content") else os.path.abspath("fllm_compare"))
    ap.add_argument("--fllm", help="an fllm-rs binary to use instead of the release")
    ap.add_argument("--build-from-source", action="store_true", help="build fLLM from main (newest changes, takes longer)")
    ap.add_argument("--models", default="llama8b,llama8b-awq,qwen2b", help=f"comma-separated, of {','.join(MODELS)}")
    ap.add_argument("--sizes", default="100,500,2000")
    ap.add_argument("--users", default="1,4,16")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="reproducible prompt nonces and request seeds, identical across engines/A-B runs")
    ap.add_argument("--cpu", action="store_true", help="CPU only: fLLM, llama.cpp and ik_llama.cpp on the processor (no GPU needed; GGUF models only)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for --cpu (default: all cores)")
    ap.add_argument("--no-ik", action="store_true", help="skip ik_llama.cpp")
    ap.add_argument("--no-llama", action="store_true")
    ap.add_argument("--no-ollama", action="store_true")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--no-sglang", action="store_true")
    ap.add_argument("--fllm-args", default="", help="extra arguments for fllm-rs serve, e.g. \"--chunked-prefill-policy prefill-first\"")
    ap.add_argument("--only", help="comma-separated engines to measure (the others are still installed if needed), e.g. fllm")
    args = ap.parse_args()
    try:
        control_args, control_env = fllm_control_options(args.fllm_args)
    except ValueError as ex:
        ap.error(str(ex))
    os.makedirs(args.work, exist_ok=True)
    os.chdir(args.work)
    sizes = [int(x) for x in args.sizes.split(",")]
    users = [int(x) for x in args.users.split(",")]
    maxu = max(users)
    ctx = (max(sizes) + args.tokens + 400) * maxu
    result = {"schema_version": 2, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": args.seed,
              "measurement": {"tokens": "server_usage_only", "first_s": "first_nonempty_output_delta",
                              "each_tok_s": "end_to_end_including_ttft"},
              "fllm_control": {"env": control_env, "args": control_args}}
    if args.cpu:
        cpu_name = sh("lscpu | grep 'Model name' | head -1")[1].split(":", 1)[-1].strip()
        gpu, vram = "CPU: " + cpu_name, 10**9
        args.no_ollama = args.no_vllm = args.no_sglang = True
        args.threads = args.threads or (os.cpu_count() or 4)
        result["cpu_threads"] = args.threads
    else:
        rc, out = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        if rc != 0 or not out.strip():
            sys.exit("No GPU: Runtime > Change runtime type > a GPU, then run this again (or use --cpu).")
        gpu = out.strip().splitlines()[0]
        vram = int("".join(ch for ch in gpu.split(",")[1] if ch.isdigit()) or 0)
    result["gpu"] = gpu
    log(f"{'CPU' if args.cpu else 'GPU'}: {gpu}")

    engines = {}
    engines["fllm"] = get_fllm(args, args.work)
    if not args.no_llama:
        engines["llama.cpp"] = get_llama(args.work, cpu=args.cpu)
    if not args.no_ik:
        engines["ik_llama.cpp"] = get_llama(args.work, cpu=args.cpu, fork="ik_llama.cpp")
    if not args.no_ollama:
        engines["ollama"] = get_ollama()
    if not args.no_vllm:
        engines["vllm"] = get_vllm(args.work)
    if not args.no_sglang:
        engines["sglang"] = get_sglang(args.work)
    result["engines"] = {k: (v or "not available") for k, v in engines.items()}
    log(f"engines: {result['engines']}")
    if engines.get("fllm"):
        result["fllm_version"] = sh(f"{engines['fllm']} --version")[1].strip()[:200]

    result["models"] = {}
    for key in args.models.split(","):
        if key not in MODELS:
            continue
        label, url, need = MODELS[key]
        if vram < need:
            log(f"skip {label}: the GPU has less than {need} MiB")
            continue
        hf = not url.startswith("http")
        if hf:
            # vLLM and SGLang read the checkpoint, fLLM a GGUF converted from it
            hf_dir = hf_download(url, os.path.join(args.work, url.split("/")[1]))
            path = hf_dir + ".gguf"
            if engines.get("fllm") and not os.path.exists(path):
                log(f"converting {url} for fLLM")
                rc, out = sh(f"{engines['fllm']} convert {hf_dir} {path} 2>&1 | tail -3", timeout=3600)
                if not os.path.exists(path):
                    log("conversion failed: " + out[-300:])
        else:
            path = download(url, os.path.join(args.work, os.path.basename(url)))
        per = {}
        no_think = {"chat_template_kwargs": {"enable_thinking": False}}
        ctx_one = max(sizes) + args.tokens + 400
        for name, binary in engines.items():
            if not binary or name not in (HF_ENGINES if hf else GGUF_ENGINES) or (args.only and name not in args.only.split(",")):
                continue
            if name == "fllm" and not os.path.exists(path):
                per[name] = {"error": "no GGUF for fLLM (conversion failed)"}
                continue
            port = random.randint(20000, 30000)
            log(f"== {label} on {name}")
            server = None
            extra = {}
            if name == "fllm":
                device = ["--device", "cpu", "--threads", str(args.threads)] if args.cpu else ["--device", "cuda"]
                server = Server([binary, "serve", "--model", path] + device + ["--port", str(port), "--ctx", str(ctx), "--max-seqs", str(maxu), "--model-name", "m"] + control_args,
                                f"http://127.0.0.1:{port}/health", "m", env=control_env, logfile=f"fllm-{key}.log")
                extra = no_think
            elif name in ("llama.cpp", "ik_llama.cpp"):
                gpu_flags = ["-t", str(args.threads), "-ngl", "0"] if args.cpu else ["-ngl", "99"]
                server = Server([binary, "-m", path] + gpu_flags + ["-c", str(ctx), "-np", str(maxu), "--host", "127.0.0.1", "--port", str(port), "--alias", "m"],
                                f"http://127.0.0.1:{port}/health", "m", logfile=f"{name.replace('.', '_')}-{key}.log",
                                env={"LD_LIBRARY_PATH": os.path.dirname(binary) + ":" + os.environ.get("LD_LIBRARY_PATH", "")})
                extra = no_think
            elif name == "ollama":
                env = {"OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_MODELS": os.path.join(args.work, "ollama-models"),
                       "OLLAMA_NUM_PARALLEL": str(maxu), "OLLAMA_CONTEXT_LENGTH": str(max(sizes) + args.tokens + 400), "OLLAMA_MAX_QUEUE": "512"}
                open("Modelfile", "w").write(f"FROM {path}\n")

                def create(env=env):
                    rc, _ = sh("ollama create bench -f Modelfile", env=env, timeout=1800)
                    return rc == 0
                server = Server([binary, "serve"], f"http://127.0.0.1:{port}/api/version", "bench", env=env, logfile=f"ollama-{key}.log", after=create)
                extra = {"reasoning_effort": "none"}
            elif name == "vllm":
                server = Server([binary, "serve", hf_dir, "--port", str(port), "--max-model-len", str(ctx_one), "--max-num-seqs", str(maxu),
                                 "--gpu-memory-utilization", "0.85", "--served-model-name", "m"],
                                f"http://127.0.0.1:{port}/health", "m", env={"CUDA_HOME": "/usr/local/cuda", "PATH": f"{args.work}/vllm_env/bin:/usr/local/cuda/bin:" + os.environ["PATH"]},
                                logfile=f"vllm-{key}.log")
                extra = no_think
            elif name == "sglang":
                server = Server([binary, "-m", "sglang.launch_server", "--model-path", hf_dir, "--port", str(port), "--host", "127.0.0.1", "--context-length", str(ctx_one),
                                 "--max-running-requests", str(maxu), "--mem-fraction-static", "0.8", "--served-model-name", "m"],
                                f"http://127.0.0.1:{port}/health", "m", env={"CUDA_HOME": "/usr/local/cuda", "PATH": f"{args.work}/sglang_env/bin:/usr/local/cuda/bin:" + os.environ["PATH"]},
                                logfile=f"sglang-{key}.log")
                extra = no_think
            if not server or not server.ok:
                tail = open(server.logfile).read()[-600:] if server else ""
                per[name] = {"error": "did not start: " + tail.replace("\n", " | ")}
                log(per[name]["error"][:300])
                if server:
                    server.stop()
                continue
            try:
                per[name] = measure(server, port, sizes, users, args.tokens, extra, seed=args.seed)
            except Exception as ex:
                per[name] = {"error": str(ex)[:200]}
            if server.proc.poll() is not None or any("error" in r or r.get("each_tok_s") == 0.0 for r in per[name] if isinstance(r, dict)):
                # the server failed during the run: keep the end of its log in the result
                tail = open(server.logfile, errors="replace").read()[-3000:]
                result.setdefault("failures", {})[f"{label} / {name}"] = tail
                log(f"{name} failed, end of {server.logfile}: " + tail[-1500:])
            server.stop()
            time.sleep(3)
        result["models"][label] = per

    print("\n=== RESULT ===")
    for label, per in result["models"].items():
        print(f"\n{label}  (usage tok/s all users | first output delta in s | per-user E2E tok/s; '-' = unavailable)")
        names = [n for n in per if isinstance(per[n], list)]
        print(f"{'prompt':>6} {'users':>5}  " + "  ".join(f"{n:>24}" for n in names))
        table = {n: [r for r in per[n] if r.get("users")] for n in names}
        for i, row in enumerate(table[names[0]] if names else []):
            cells = []
            for n in names:
                r = table[n][i] if i < len(table[n]) else {}
                values = ["-" if r.get(k) is None else r[k] for k in ("total_tok_s", "first_s", "each_tok_s")]
                cells.append(f"{values[0]:>7} | {values[1]:>6} | {values[2]:>5}" if "error" not in r else f"{'error':>24}")
                if r.get("measurement_warning"):
                    print(f"  warning {n}, prompt {r['prompt']}, users {r['users']}: {r['measurement_warning']}")
            print(f"{row['prompt']:>6} {row['users']:>5}  " + "  ".join(f"{c:>24}" for c in cells))
        for n in per:
            if isinstance(per[n], dict) and "error" in per[n]:
                print(f"  {n}: {per[n]['error'][:200]}")
        for n in names:
            sample = next((r.get("sample") for r in per[n] if r.get("users") == 0), "")
            print(f"  answer {n:>13}: {sample[:90]!r}")
    result["seconds"] = round(time.time() - T0)
    json.dump(result, open(os.path.join(args.work, "compare_result.json"), "w"), indent=1)
    print(f"\nwritten: {os.path.join(args.work, 'compare_result.json')}")


if __name__ == "__main__":
    main()
