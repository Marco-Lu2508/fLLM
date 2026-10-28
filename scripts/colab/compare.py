#!/usr/bin/env python3
"""Compare fLLM with vLLM, SGLang, llama.cpp and Ollama on the same Colab GPU, with one and with several users.

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
  5. Ollama     the official Linux install (a few minutes; --no-ollama skips it), started on a private port and model folder
  6. models     --models, default llama8b (GGUF), llama8b-awq (AWQ) and qwen2b (GGUF)
  7. measure    every engine serves the model through /v1/chat/completions with streaming; for each prompt length (100, 500, 2000
                tokens) and number of simultaneous users (1, 4, 16) all requests start at once, every prompt is different (no cache
                reuse), and the answer is 64 tokens long. Reported: generated tokens per second of all users together, the
                average time to the first token, and the speed one user sees after its first token.

The result is a table per model on stdout and compare_result.json in the work directory (default /content/fllm_compare).
Only the Python standard library is used; the network is used for the downloads.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
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
}
GGUF_ENGINES = ("fllm", "llama.cpp", "ollama")
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
    rel = next(r for r in json.load(urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/releases?per_page=5")) if not r.get("draft"))
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


def get_llama(work):
    binary = os.path.join(work, "llama.cpp/build/bin/llama-server")
    if os.path.exists(binary):
        return binary
    log("building llama.cpp with CUDA ...")
    sh(f"git clone -q --depth 1 https://github.com/ggml-org/llama.cpp {work}/llama.cpp")
    rc, out = sh(f"cd {work}/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null "
                 f"&& cmake --build build --config Release -j {os.cpu_count() or 4} --target llama-server 2>&1 | tail -3", timeout=3600)
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

def one_request(url, model, size, tokens, extra, out):
    text = SENTENCE * max(1, size // 10)
    body = {"model": model, "stream": True, "max_tokens": tokens, "temperature": 0,
            "messages": [{"role": "user", "content": f"Run {random.randint(0, 10**9)}: count the words of this text and then count to twenty. {text}"}]}
    body.update(extra)
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    first, chunks = None, 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                d = line[5:].strip()
                if d == b"[DONE]":
                    break
                if b'"content":"' in d or b'"reasoning_content":"' in d or b'"reasoning":"' in d:
                    first = first or time.time() - t
                    chunks += 1
        out.append({"first": first or 0.0, "total": time.time() - t, "chunks": chunks})
    except Exception as ex:
        out.append({"error": str(ex)[:120]})


def measure(server, port, sizes, users, tokens, extra):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    warm = []
    one_request(url, server.model, 50, 8, extra, warm)
    rows = []
    for size in sizes:
        for n in users:
            out = []
            threads = [threading.Thread(target=one_request, args=(url, server.model, size, tokens, extra, out)) for _ in range(n)]
            t = time.time()
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            wall = time.time() - t
            errors = [o["error"] for o in out if "error" in o]
            if errors:
                rows.append({"prompt": size, "users": n, "error": errors[0]})
                continue
            made = sum(o["chunks"] for o in out)
            each = [(o["chunks"] - 1) / (o["total"] - o["first"]) for o in out if o["chunks"] > 1 and o["total"] > o["first"]]
            rows.append({"prompt": size, "users": n, "total_tok_s": round(made / wall, 1), "first_s": round(sum(o["first"] for o in out) / len(out), 2),
                         "each_tok_s": round(sum(each) / max(1, len(each)), 1)})
            log(f"  prompt {size:5} users {n}: {rows[-1]}")
    return rows


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
    ap.add_argument("--no-llama", action="store_true")
    ap.add_argument("--no-ollama", action="store_true")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--no-sglang", action="store_true")
    ap.add_argument("--fllm-args", default="", help="extra arguments for fllm-rs serve, e.g. \"--chunked-prefill-policy prefill-first\"")
    ap.add_argument("--only", help="comma-separated engines to measure (the others are still installed if needed), e.g. fllm")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    os.chdir(args.work)
    sizes = [int(x) for x in args.sizes.split(",")]
    users = [int(x) for x in args.users.split(",")]
    maxu = max(users)
    ctx = (max(sizes) + args.tokens + 400) * maxu
    result = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
    rc, out = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if rc != 0 or not out.strip():
        sys.exit("No GPU: Runtime > Change runtime type > a GPU, then run this again.")
    gpu = out.strip().splitlines()[0]
    vram = int("".join(ch for ch in gpu.split(",")[1] if ch.isdigit()) or 0)
    result["gpu"] = gpu
    log(f"GPU: {gpu}")

    engines = {}
    engines["fllm"] = get_fllm(args, args.work)
    if not args.no_llama:
        engines["llama.cpp"] = get_llama(args.work)
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
            if name == "fllm":
                server = Server([binary, "serve", "--model", path, "--device", "cuda", "--port", str(port), "--ctx", str(ctx), "--max-seqs", str(maxu), "--model-name", "m"] + args.fllm_args.split(),
                                f"http://127.0.0.1:{port}/health", "m", logfile=f"fllm-{key}.log")
                extra = no_think
            elif name == "llama.cpp":
                server = Server([binary, "-m", path, "-ngl", "99", "-c", str(ctx), "-np", str(maxu), "--host", "127.0.0.1", "--port", str(port), "--alias", "m", "--no-webui"],
                                f"http://127.0.0.1:{port}/health", "m", logfile=f"llama-{key}.log")
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
                per[name] = measure(server, port, sizes, users, args.tokens, extra)
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
        print(f"\n{label}  (total tok/s of all users | average first token in s | tok/s one user sees)")
        names = [n for n in per if isinstance(per[n], list)]
        print(f"{'prompt':>6} {'users':>5}  " + "  ".join(f"{n:>24}" for n in names))
        for i, row in enumerate(per[names[0]] if names else []):
            cells = []
            for n in names:
                r = per[n][i] if i < len(per[n]) else {}
                cells.append(f"{r.get('total_tok_s', '-'):>7} | {r.get('first_s', '-'):>6} | {r.get('each_tok_s', '-'):>5}" if "error" not in r else f"{'error':>24}")
            print(f"{row['prompt']:>6} {row['users']:>5}  " + "  ".join(f"{c:>24}" for c in cells))
        for n in per:
            if isinstance(per[n], dict) and "error" in per[n]:
                print(f"  {n}: {per[n]['error'][:200]}")
    result["seconds"] = round(time.time() - T0)
    json.dump(result, open(os.path.join(args.work, "compare_result.json"), "w"), indent=1)
    print(f"\nwritten: {os.path.join(args.work, 'compare_result.json')}")


if __name__ == "__main__":
    main()
