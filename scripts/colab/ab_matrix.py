# fLLM A/B matrix on a Colab GPU (decode, prefill and NVFP4 changes together). One cell:
#   python3 -u /content/ab_matrix.py --quick          # run this first: variants V0, V1, V8 (old / graphs only / new defaults), about 30-45 minutes
#   python3 -u /content/ab_matrix.py                  # then everything (several hours; narrow it with --only-model / --only-variants)
# Requirements: a runtime with a GPU (T4 / L4 / A100 / G4 = RTX PRO 6000 Blackwell sm_120 / H100) and /content/fllm.tgz (source tree, unpacked to fllm/).
# Flow: detect the card -> build (or reuse an existing build) -> selftest-gpu (all, fuse, prefill, nvfp4 on sm_120)
#   -> all variants per model (one server start each, with different FLLM_* variables) -> optionally vLLM -> "=== AB RESULT ===".
# Good means: selftest "0 failed"; per variant text_same = true (the fusions are designed to be bit-identical) and logprob_diff <= 0.05
# (NVFP4: <= 0.5, because V0 computes without an activation scale there); the marker "check" means: have a look. Python standard library only.
import argparse, json, os, re, shutil, subprocess, sys, tarfile, threading, time, urllib.request

W = "/content/fllm_work"
SRC = f"{W}/fllm"
B = f"{SRC}/target/release/fllm-rs"
OUT_JSON = "/content/fllm_ab.json"
PORT = 8970  # fLLM
VPORT = 8971  # vLLM
T0 = time.time()
LENGTHS = [128, 355, 1000, 1863, 4000, 8000]
NVFP4_REPO = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
AWQ_REPO = "Qwen/Qwen2.5-3B-Instruct-AWQ"

# ---------------------------------------------------------------------------------------------------------------------------------
# Variants. A variant is a set of FLLM_* variables (all other FLLM_* variables of the process are removed).
# Notation: "0" switches off (the decode switches also accept off|false|no), unset = default; FLLM_FUSE_NVFP4_* are on only with a value other than 0.

OLD = {  # V0: the behaviour before the decode, prefill and NVFP4 changes
    "FLLM_CUDA_GRAPH": "0",
    "FLLM_FUSE": "0",
    "FLLM_PREFILL_STEP_CACHE": "0",
    "FLLM_PREFILL_PAR_EMBED": "0",
    "FLLM_FUSE_PREFILL_ADD_NORM": "0",
    "FLLM_FUSE_PREFILL_NORM_F16": "0",
    "FLLM_FUSE_PREFILL_ATTN_F16": "0",
    "FLLM_MARLIN_PREFILL_CFG": "vllm",
    "FLLM_NVFP4_ACT_SCALE": "off",
    "FLLM_NVFP4_TILE": "legacy",
    "FLLM_NVFP4_TUNE": "0",
    "FLLM_FUSE_NVFP4_NORM": "0",
    "FLLM_FUSE_NVFP4_ATTN_OUT": "0",
}
PREFILL_FUSIONS = {"FLLM_FUSE_PREFILL_ADD_NORM": "1", "FLLM_FUSE_PREFILL_NORM_F16": "1", "FLLM_FUSE_PREFILL_ATTN_F16": "1"}
NVFP4_TILES = ["legacy", "auto", "256x128x128", "128x256x128", "128x128x128", "128x128x256", "128x128x128-sk", "128x128x256-sk",
               "128x128x128-pp", "128x64x256"]
MARLIN_CFGS = ["128,64,128", "64,128,128", "128,128,256", "64,256,256"]


def with_(base, **kw):
    e = dict(base)
    e.update(kw)
    return e


def all_variants():
    """[(id, description, env, applies_to)] ; applies_to: set of model kinds 'marlin' (AWQ), 'kquant', 'nvfp4'."""
    ALL = {"marlin", "kquant", "nvfp4"}
    V = []
    V.append(("V0", "old: graphs, fusions, prefill changes, NVFP4 activation scale off", dict(OLD), ALL))
    V.append(("V1", "graphs only", with_(OLD, FLLM_CUDA_GRAPH="1"), ALL))
    v1 = with_(OLD, FLLM_CUDA_GRAPH="1")
    v2 = dict(v1)
    del v2["FLLM_FUSE"]
    V.append(("V2", "graphs + decode fusions (FLLM_FUSE unset = on)", v2, ALL))
    v3 = with_(OLD, FLLM_PREFILL_STEP_CACHE="1", FLLM_PREFILL_PAR_EMBED="1")
    V.append(("V3", "prefill: host work (step cache, parallel embeddings)", v3, ALL))
    v4 = with_(v3, **PREFILL_FUSIONS)
    V.append(("V4", "V3 + prefill fusions (add+norm, norm+f16, attn+f16)", v4, ALL))
    V.append(("V5", "V4 + FLLM_MARLIN_PREFILL_CFG=fllm", with_(v4, FLLM_MARLIN_PREFILL_CFG="fllm"), ALL))
    v6 = with_(OLD, FLLM_NVFP4_ACT_SCALE="static")
    V.append(("V6", "NVFP4: activation scale static (input_scale of the checkpoint)", v6, {"nvfp4"}))
    V.append(("V7", "NVFP4: activation scale dynamic (amax per tensor)", with_(OLD, FLLM_NVFP4_ACT_SCALE="dynamic"), {"nvfp4"}))
    for t in NVFP4_TILES:
        V.append((f"V6.tile={t}", f"NVFP4: V6 with FLLM_NVFP4_TILE={t}", with_(v6, FLLM_NVFP4_TILE=t), {"nvfp4"}))
    V.append(("V6.tune", "NVFP4: V6 + FLLM_NVFP4_TUNE=1", with_(v6, FLLM_NVFP4_TUNE="1"), {"nvfp4"}))
    V.append(("V6.norm", "NVFP4: V6 + FLLM_FUSE_NVFP4_NORM=1", with_(v6, FLLM_FUSE_NVFP4_NORM="1"), {"nvfp4"}))
    V.append(("V6.attn", "NVFP4: V6 + FLLM_FUSE_NVFP4_ATTN_OUT=1", with_(v6, FLLM_FUSE_NVFP4_ATTN_OUT="1"), {"nvfp4"}))
    V.append(("V8", "ALL NEW DEFAULTS (no FLLM_* variable)", {}, ALL))
    off = {"FLLM_FUSE": "0"}
    V.append(("V9.add_norm", "only FLLM_FUSE_ADD_NORM=1", with_(off, FLLM_FUSE_ADD_NORM="1"), ALL))
    V.append(("V9.add_norm+norm_quant", "FLLM_FUSE_ADD_NORM=1 + FLLM_FUSE_NORM_QUANT=1", with_(off, FLLM_FUSE_ADD_NORM="1", FLLM_FUSE_NORM_QUANT="1"), ALL))
    V.append(("V9.rope_kv", "only FLLM_FUSE_ROPE_KV=1", with_(off, FLLM_FUSE_ROPE_KV="1"), ALL))
    V.append(("V9.glu_q8", "only FLLM_FUSE_GLU_Q8=1", with_(off, FLLM_FUSE_GLU_Q8="1"), {"kquant"}))
    V.append(("V9.head_copy", "only FLLM_FUSE_HEAD_COPY=1", with_(off, FLLM_FUSE_HEAD_COPY="1"), ALL))
    for c in MARLIN_CFGS:
        V.append((f"V10.cfg={c.replace(',', '-')}", f"AWQ: V8 + FLLM_MARLIN_CFG={c}", {"FLLM_MARLIN_CFG": c}, {"marlin"}))
    return V


# Order of the models: (key, name, kind, minimum VRAM MiB, smallest compute capability, vLLM repo)
MODELS = [
    ("l32", "Llama-3.2-3B Q4_K_M", "kquant", 4000, 75, None),
    ("awq", "Qwen2.5-3B AWQ", "marlin", 4000, 75, AWQ_REPO),
    ("l31", "Llama-3.1-8B Q4_K_M", "kquant", 7500, 75, None),
    ("nvfp4", "Llama-3.1-8B NVFP4 (ModelOpt)", "nvfp4", 7500, 120, NVFP4_REPO),
]


def select_variants(kind, only, quick):
    want = None
    if quick:
        want = ["V0", "V1", "V8"]
    if only:
        want = only
    out = []
    for vid, desc, env, kinds in all_variants():
        if kind not in kinds:
            continue
        if want is not None and not any(vid == w or vid.startswith(w + ".") for w in want):
            continue
        out.append((vid, desc, env))
    # measure an identical environment only once (e.g. V6.tile=legacy is V6)
    seen, plan = {}, []
    for vid, desc, env in out:
        key = tuple(sorted(env.items()))
        plan.append({"id": vid, "desc": desc, "env": env, "same_as": seen.get(key)})
        seen.setdefault(key, vid)
    return plan


def server_cmd(binary, model, port, threads):
    return [binary, "serve", "--model", model, "--port", str(port), "--ctx", "16384", "--max-seqs", "2", "--threads", str(threads),
            "--device", "cuda", "--max-batched-tokens", "1024"]


# ---------------------------------------------------------------------------------------------------------------------------------
def log(msg):
    print(f"[{time.time() - T0:4.0f}s] {msg}", flush=True)


def sh(cmd, env=None, timeout=None, show=True):
    e = dict(os.environ)
    e["PATH"] = os.path.expanduser("~/.cargo/bin") + ":/usr/local/cuda/bin:" + e["PATH"]
    if env:
        e.update(env)
    p = subprocess.run(["bash", "-lc", cmd], env=e, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    if show and out.strip():
        print(out[-3000:])
    return p.returncode, out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="fLLM A/B matrix on a Colab GPU")
    ap.add_argument("--quick", action="store_true", help="only V0, V1, V8")
    ap.add_argument("--only-model", default="", help="comma-separated list of: " + ",".join(m[0] for m in MODELS))
    ap.add_argument("--only-variants", default="", help="comma-separated list, e.g. V0,V8 (V9 = all V9.*)")
    ap.add_argument("--skip-vllm", action="store_true", help="no vLLM comparison")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)), help="prompt lengths for the prefill speed")
    ap.add_argument("--dry-run", action="store_true", help="only print the variant list and server commands, start nothing")
    ap.add_argument("--assume-sm", default="120", help="--dry-run: compute capability without the dot (75, 80, 89, 120 ...)")
    ap.add_argument("--assume-vram-mib", type=int, default=96000, help="--dry-run: memory of the card")
    ap.add_argument("--client-test", action="store_true", help="check the measuring client against a built-in mock server (no GPU)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------------------------------------------------------
# Measuring client (with a model name for vLLM)
class Client:
    def __init__(self, port, model_name=None):
        self.port = port
        self.model_name = model_name
        self.counter = int(time.time()) % 10**6 * 1000

    def post(self, body):
        if self.model_name:
            body = {**body, "model": self.model_name}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=900)

    def unique(self):
        self.counter += 1
        return f"{self.counter} "  # unique start per request: no prefix-cache hit

    def ttft(self, n):
        p = self.unique() + "hello " * n
        t = time.time()
        for line in self.post({"prompt": p, "max_tokens": 1, "temperature": 0, "stream": True, "ignore_eos": True}):
            if line.startswith(b"data: {"):
                return time.time() - t
        raise RuntimeError("no response")

    def decode_rate(self):
        t, c = None, 0
        for line in self.post({"prompt": self.unique() + "hello " * 8, "max_tokens": 128, "temperature": 0, "stream": True, "ignore_eos": True}):
            if line.startswith(b"data: {"):
                c += 1
                t = t or time.time()
        return (c - 1) / (time.time() - t) if c > 1 and t else 0.0

    def text_and_logprobs(self):
        d = json.load(self.post({"prompt": "The capital of France is", "max_tokens": 24, "temperature": 0, "logprobs": 5}))
        ch = d["choices"][0]
        top = (ch.get("logprobs") or {}).get("top_logprobs") or [{}]
        return ch["text"], {k: float(v) for k, v in (top[0] or {}).items()}

    def wait_health(self, proc, seconds=1800):
        end = time.time() + seconds
        while time.time() < end:
            time.sleep(2)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)
                return True
            except Exception:
                if proc is not None and proc.poll() is not None:
                    return False
        return False

    def measure(self, lengths):
        """Warm up (one request per length), then decode (best of 3), prefill per length (best of 3), text and top-5 logprobs."""
        r = {}
        for n in lengths:
            self.ttft(n)
        self.decode_rate()
        r["decode"] = round(max(self.decode_rate() for _ in range(3)), 1)
        r["prefill"] = {str(n): round((n + 3) / min(self.ttft(n) for _ in range(3))) for n in lengths}
        text, top = self.text_and_logprobs()
        r["text"] = text.replace("\n", " ")[:120]
        r["_text_full"] = text
        r["top5"] = top
        return r


def compare(ref, cur, nvfp4):
    """text_same and the largest logprob difference against the reference (V0); marker 'check' above 0.05 (NVFP4: 0.5)."""
    if not ref or "decode" not in ref or "decode" not in cur:
        return {}
    out = {"text_same": cur["_text_full"] == ref["_text_full"]}
    common = [t for t in cur["top5"] if t in ref["top5"]]
    if common:
        out["logprob_diff"] = round(max(abs(cur["top5"][t] - ref["top5"][t]) for t in common), 4)
        limit = 0.5 if nvfp4 else 0.05
        out["check"] = out["logprob_diff"] > limit
    else:
        out["logprob_diff"] = None
        out["check"] = True
    if max(cur["top5"], key=cur["top5"].get, default=None) != max(ref["top5"], key=ref["top5"].get, default=None):
        out["top1_same"] = False
        out["check"] = True
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
def gpu_info():
    rc, out = sh("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader", show=False)
    if rc != 0 or not out.strip():
        sys.exit("No GPU found: choose Runtime -> Change runtime type -> T4 / L4 / A100 / G4 and restart.")
    gpu, cc, mem = [x.strip() for x in out.strip().splitlines()[0].split(",")]
    return gpu, cc, mem, int(re.sub(r"[^0-9]", "", mem.split()[0]))


def sources_mtime():
    newest = 0.0
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d not in ("target", "third_party", ".git")]
        for f in files:
            newest = max(newest, os.path.getmtime(os.path.join(root, f)))
    return newest


def prepare_source_and_build(arch, ncpu, result):
    tgz = "/content/fllm.tgz"
    if not os.path.exists(tgz):
        sys.exit("/content/fllm.tgz is missing: upload the source archive and save it as /content/fllm.tgz.")
    os.makedirs(W, exist_ok=True)
    stamp_file = f"{W}/.tgz_stamp"
    stamp = f"{os.path.getsize(tgz)}:{int(os.path.getmtime(tgz))}"
    old = open(stamp_file).read().strip() if os.path.exists(stamp_file) else ""
    if not os.path.exists(f"{SRC}/Cargo.toml") or old != stamp:
        log("unpacking fllm.tgz")
        with tarfile.open(tgz) as t:
            t.extractall(W)
        # unpacked files carry old timestamps: touch them, otherwise cargo does not rebuild
        sh(f"find {SRC} -path {SRC}/target -prune -o -path {SRC}/third_party -prune -o -type f -exec touch {{}} +", show=False)
        open(stamp_file, "w").write(stamp)
    if shutil.which("cargo") is None and not os.path.exists(os.path.expanduser("~/.cargo/bin/cargo")):
        log("installing Rust")
        sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal", show=False)
    tp = f"{SRC}/third_party"
    os.makedirs(tp, exist_ok=True)
    if arch == "120" and not os.path.exists(f"{tp}/cutlass"):
        sh(f"git clone -q --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass {tp}/cutlass", show=False)
    if int(arch) >= 75 and not os.path.exists(f"{tp}/flashinfer"):
        sh(f"git clone -q --depth 1 --branch v0.6.5 https://github.com/flashinfer-ai/flashinfer {tp}/flashinfer", show=False)
    arch_file = f"{SRC}/target/.ab_arch"
    fresh = (os.path.exists(B) and os.path.getmtime(B) > sources_mtime() and os.path.exists(arch_file) and open(arch_file).read().strip() == arch)
    if fresh:
        log("existing build is newer than the sources: reused")
        result["build_seconds"] = 0
        return
    log(f"building fllm-rs for sm_{arch} (10-30 minutes) ...")
    t = time.time()
    sh(f"cd {SRC} && FLLM_CUDA_ARCHS={arch} CARGO_BUILD_JOBS={ncpu} cargo build --release --locked --features cuda -p fllm-engine --bin fllm-rs > {W}/build.log 2>&1; tail -5 {W}/build.log",
       timeout=3 * 3600)
    result["build_seconds"] = round(time.time() - t)
    if not os.path.exists(B):
        sh(f"grep -E 'error' {W}/build.log | head -20")
        sys.exit("Build failed, see above.")
    open(arch_file, "w").write(arch)


def run_selftests(arch, result):
    plan = [("all", []), ("fuse", ["--only", "fuse"]), ("prefill", ["--only", "prefill"])]
    if arch == "120":
        plan.append(("nvfp4", ["--only", "linear", "--quants", "nvfp4"]))
    res = {}
    for name, extra in plan:
        jf = f"{W}/selftest_{name}.json"
        if os.path.exists(jf):
            os.remove(jf)
        t = time.time()
        rc, out = sh(f"{B} selftest-gpu --device cuda {' '.join(extra)} --json {jf} > {W}/selftest_{name}.log 2>&1", timeout=3 * 3600, show=False)
        r = {"rc": rc, "seconds": round(time.time() - t)}
        try:
            d = json.load(open(jf))
            r.update(checks=d["checks"], failed=d["failed"], skipped=d["skipped"])
            r["failing"] = [f"{x['group']}/{x['op']} {x['shape']}: {(x.get('message') or '')[:100]}" for x in d["results"] if x["status"] == "failed"][:40]
            r["groups"] = {k: v["checks"] for k, v in d.get("groups", {}).items()}
        except Exception as ex:
            tail = open(f"{W}/selftest_{name}.log").read()[-400:] if os.path.exists(f"{W}/selftest_{name}.log") else ""
            r["error"] = f"{ex}; {tail}"[:500]
        res[name] = r
        log(f"selftest {name}: checks={r.get('checks')} failed={r.get('failed')} skipped={r.get('skipped')} {r.get('failing', '')[:3] if r.get('failing') else ''}")
    result["selftest"] = res


# ---------------------------------------------------------------------------------------------------------------------------------
def hf_dir(repo, dest):
    os.makedirs(dest, exist_ok=True)
    info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}"))
    for f in info["siblings"]:
        n = f["rfilename"]
        if n.endswith((".md", ".png", ".jpg")) or n.startswith("."):
            continue
        p = os.path.join(dest, n)
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/{n}", p)


def gguf_download(name, url):
    p = f"{W}/{name}"
    if not os.path.exists(p):
        log(f"downloading {name}")
        urllib.request.urlretrieve(url, p)
    return p


def has_input_scale(path):
    """Does the GGUF contain the input_scale tensors? (Older llama.cpp conversions leave them out; then 'static' is not meaningful.)"""
    try:
        with open(path, "rb") as f:
            return b".input_scale" in f.read(48 << 20)
    except OSError:
        return False


def prepare_model(key):
    """Return the GGUF path (downloads / converts when needed; existing files are reused)."""
    if key == "l32":
        return gguf_download("Llama-3.2-3B-Instruct-Q4_K_M.gguf", "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    if key == "l31":
        return gguf_download("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    if key == "awq":
        d, g = f"{W}/Qwen2.5-3B-Instruct-AWQ", f"{W}/Qwen2.5-3B-Instruct-AWQ.gguf"
        if not os.path.exists(g):
            log(f"downloading and converting {AWQ_REPO}")
            hf_dir(AWQ_REPO, d)
            sh(f"{B} convert {d} {g} 2>&1 | tail -2", show=False)
        if not os.path.exists(g):
            raise RuntimeError("AWQ conversion failed")
        return g
    if key == "nvfp4":
        d = f"{W}/Llama-3.1-8B-Instruct-NVFP4"
        g = d + ".gguf"
        if os.path.exists(g) and not has_input_scale(g):
            log("existing NVFP4 GGUF has no input_scale tensors: converting again with the current llama.cpp")
            g = d + "-v2.gguf"
        if not os.path.exists(g):
            log(f"downloading {NVFP4_REPO} and converting to GGUF (external tool llama.cpp, used at run time only)")
            hf_dir(NVFP4_REPO, d)
            if not os.path.exists(f"{W}/llama.cpp"):
                sh(f"git clone -q --depth 1 https://github.com/ggml-org/llama.cpp {W}/llama.cpp", show=False)
            sh("pip install -q numpy sentencepiece transformers safetensors protobuf", show=False)
            sh(f"cd {W}/llama.cpp && PYTHONPATH={W}/llama.cpp/gguf-py python3 convert_hf_to_gguf.py {d} --outfile {g} --outtype auto 2>&1 | tail -4", timeout=3600)
        if not os.path.exists(g):
            raise RuntimeError("NVFP4 conversion failed")
        if not has_input_scale(g):
            log("WARNING: the NVFP4 GGUF has no input_scale tensors; V6/V7 (static) are then the same as V0")
        return g
    raise KeyError(key)


# ---------------------------------------------------------------------------------------------------------------------------------
def stop(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(60)
    except Exception:
        proc.kill()
        proc.wait()
    time.sleep(3)


def start_server(model, env, logfile, ncpu):
    e = {k: v for k, v in os.environ.items() if not k.startswith("FLLM_")}
    e.update(env)
    cmd = server_cmd(B, model, PORT, ncpu)
    return subprocess.Popen(cmd, stdout=open(logfile, "w"), stderr=subprocess.STDOUT, env=e)


def log_notes(logfile):
    try:
        txt = open(logfile, errors="replace").read()
    except OSError:
        return []
    keep = [l.strip()[:160] for l in txt.splitlines() if re.search(r"CUDA graphs for decode steps|Marlin: trying|without an input_scale|panicked|error", l, re.I)]
    return keep[:6]


class Collector:
    def __init__(self, result):
        self.result = result
        self.result.setdefault("models", {})

    def save(self):
        self.result["duration_seconds"] = round(time.time() - T0)
        with open(OUT_JSON, "w") as f:
            json.dump(strip_private(self.result), f, indent=1, ensure_ascii=False)


def strip_private(o):
    if isinstance(o, dict):
        return {k: strip_private(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, list):
        return [strip_private(v) for v in o]
    return o


def progress_line(model, vid, r):
    if "error" in r:
        return f"{model} {vid}: ERROR {r['error'][:120]}"
    pf = r.get("prefill", {})
    extra = ""
    if "text_same" in r:
        extra = f" text_same={r['text_same']} lp_diff={r.get('logprob_diff')}" + (" CHECK" if r.get("check") else "")
    return (f"{model} {vid}: decode={r.get('decode')} " + " ".join(f"prefill@{n}={v}" for n, v in pf.items()) + extra)


def run_model(key, name, kind, plan, ncpu, lengths, coll):
    rows = coll.result["models"].setdefault(key, {"name": name, "variants": {}})
    log(f"=== Model {name}: {len(plan)} variants")
    try:
        gguf = prepare_model(key)
    except Exception as ex:
        rows["error"] = str(ex)[:300]
        log(f"{name}: {ex}")
        coll.save()
        return
    ref = None
    for p in plan:
        vid = p["id"]
        if p["same_as"]:
            rows["variants"][vid] = {"description": p["desc"], "same_as_variant": p["same_as"]}
            log(f"{name} {vid}: same environment as {p['same_as']}, not measured again")
            continue
        r = {"description": p["desc"], "env": p["env"]}
        proc = None
        t = time.time()
        try:
            logfile = f"{W}/serve_{key}_{vid.replace('/', '_').replace('=', '_')}.log"
            proc = start_server(gguf, p["env"], logfile, ncpu)
            c = Client(PORT)
            if not c.wait_health(proc):
                raise RuntimeError("server does not start: " + open(logfile, errors="replace").read()[-300:].replace("\n", " | "))
            r.update(c.measure(lengths))
            r["log"] = log_notes(logfile)
            if vid == "V0":
                ref = r
            r.update(compare(ref, r, kind == "nvfp4") if vid != "V0" else {})
        except Exception as ex:
            r["error"] = str(ex)[:300]
        finally:
            stop(proc)
        r["seconds"] = round(time.time() - t)
        rows["variants"][vid] = r
        log(progress_line(name, vid, r))
        coll.save()


def run_vllm(cases, ncpu, lengths, coll, ref_by_model):
    if not cases:
        return
    log("=== vLLM comparison: installing vLLM (own environment, 5-15 minutes)")
    vpy = f"{W}/vllm_env/bin/python"
    vbin = f"{W}/vllm_env/bin/vllm"
    if not os.path.exists(vbin):
        sh(f"pip install -q uv && uv venv --clear --python 3.12 {W}/vllm_env && uv pip install -q --python {vpy} vllm 2>&1 | tail -3", timeout=3600)
    if not os.path.exists(vbin):
        coll.result["vllm"] = "installation failed"
        coll.save()
        return
    for key, repo, kind in cases:
        row = {"description": f"vLLM {repo}"}
        proc = None
        try:
            venv = dict(os.environ)
            venv["PATH"] = f"{W}/vllm_env/bin:/usr/local/cuda/bin:" + venv["PATH"]  # FlashInfer compiles at the first start: ninja and nvcc on the path
            venv["CUDA_HOME"] = "/usr/local/cuda"
            venv["MAX_JOBS"] = str(ncpu)
            cmd = [vbin, "serve", repo, "--port", str(VPORT), "--max-model-len", "16384", "--max-num-seqs", "2", "--gpu-memory-utilization", "0.6"]
            proc = subprocess.Popen(cmd, stdout=open(f"{W}/vllm_{key}.log", "w"), stderr=subprocess.STDOUT, env=venv)
            c = Client(VPORT, repo)
            if not c.wait_health(proc, 2400):
                raise RuntimeError("vLLM does not start: " + open(f"{W}/vllm_{key}.log", errors="replace").read()[-300:].replace("\n", " | "))
            row.update(c.measure(lengths))
            row.update(compare(ref_by_model.get(key), row, kind == "nvfp4"))
        except Exception as ex:
            row["error"] = str(ex)[:300]
        finally:
            stop(proc)
        coll.result["models"].setdefault(key, {"name": key, "variants": {}})["variants"]["vLLM"] = row
        log(progress_line(key, "vLLM", row))
        coll.save()


# ---------------------------------------------------------------------------------------------------------------------------------
def table(result):
    lines = []
    head = f"{'model':<30} {'variant':<26} {'decode':>8} {'pf@355':>8} {'pf@1863':>8} {'pf@8000':>8} {'text=':>6} {'lp_diff':>8}  note"
    lines.append(head)
    for key, m in result.get("models", {}).items():
        if "error" in m and not m.get("variants"):
            lines.append(f"{m.get('name', key):<30} ERROR {m['error'][:100]}")
        for vid, r in m.get("variants", {}).items():
            if "same_as_variant" in r:
                lines.append(f"{m['name'][:30]:<30} {vid[:26]:<26} (= {r['same_as_variant']})")
                continue
            if "error" in r:
                lines.append(f"{m['name'][:30]:<30} {vid[:26]:<26} ERROR {r['error'][:90]}")
                continue
            pf = r.get("prefill", {})
            note = "check" if r.get("check") else ""
            lines.append(f"{m['name'][:30]:<30} {vid[:26]:<26} {r.get('decode', '-'):>8} {pf.get('355', '-'):>8} {pf.get('1863', '-'):>8} {pf.get('8000', '-'):>8} "
                         f"{str(r.get('text_same', '-')):>6} {str(r.get('logprob_diff', '-')):>8}  {note}")
    return "\n".join(lines)


def print_result(result):
    print("\n=== AB RESULT ===")
    print(json.dumps(strip_private(result), ensure_ascii=False, separators=(",", ":")))
    print()
    print(table(result))
    print("=== END ===")


# ---------------------------------------------------------------------------------------------------------------------------------
def selected_models(args, arch_n, vram):
    only = [x.strip() for x in args.only_model.split(",") if x.strip()]
    bad = [x for x in only if x not in [m[0] for m in MODELS]]
    if bad:
        sys.exit(f"unknown model {bad}; allowed: {','.join(m[0] for m in MODELS)}")
    out = []
    for key, name, kind, need, min_cc, repo in MODELS:
        if only and key not in only:
            continue
        if arch_n < min_cc or (kind == "nvfp4" and arch_n != 120):
            log(f"skipped (card sm_{arch_n} is not sufficient): {name}")
            continue
        if vram < need:
            log(f"skipped (not enough VRAM): {name}")
            continue
        out.append((key, name, kind, repo))
    return out


def dry_run(args):
    arch_n = int(args.assume_sm)
    only_v = [x.strip() for x in args.only_variants.split(",") if x.strip()]
    ncpu = os.cpu_count() or 2
    print(f"# Dry run: assumed card sm_{arch_n}, {args.assume_vram_mib} MiB; build with FLLM_CUDA_ARCHS={arch_n}"
          f"{'; CUTLASS v4.8.0' if arch_n == 120 else ''}{'; FlashInfer v0.6.5' if arch_n >= 75 else ''}")
    names = set()
    for key, name, kind, repo in selected_models(args, arch_n, args.assume_vram_mib):
        plan = select_variants(kind, only_v, args.quick)
        print(f"\n## {key}: {name} ({kind}), {len(plan)} variants, prefill lengths {args.lengths}")
        for p in plan:
            names.update(p["env"])
            envs = " ".join(f"{k}={v}" for k, v in sorted(p["env"].items())) or "(no FLLM_* variable)"
            same = f"   [same environment as {p['same_as']}: not measured again]" if p["same_as"] else ""
            print(f"  {p['id']:<26} {p['desc']}{same}")
            print(f"      env: {envs}")
            print(f"      cmd: {' '.join(server_cmd(B, '<' + key + '.gguf>', PORT, ncpu))}")
        if repo and not args.skip_vllm and arch_n >= 80 and (kind != "nvfp4" or arch_n == 120) and key in ("awq", "nvfp4"):
            print(f"  vLLM                       {repo}")
            print(f"      cmd: PATH={W}/vllm_env/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda MAX_JOBS={ncpu} {W}/vllm_env/bin/vllm serve {repo} "
                  f"--port {VPORT} --max-model-len 16384 --max-num-seqs 2 --gpu-memory-utilization 0.6")
    print("\n# variable names used: " + " ".join(sorted(names)))
    return sorted(names)


def client_test():
    """Mock server with /health and /v1/completions (streaming, logprobs): checks that the measuring client runs through."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i in range(body["max_tokens"]):
                    time.sleep(0.001)
                    self.wfile.write(b'data: {"choices":[{"text":"x"}]}\n\n')
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                out = json.dumps({"choices": [{"text": " Paris.", "logprobs": {"top_logprobs": [{" Paris": -0.1, " the": -3.0}]}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 18970), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = Client(18970)
    assert c.wait_health(None, 10)
    r = c.measure([128, 355])
    cmp_ = compare(r, dict(r, top5={" Paris": -0.12, " the": -3.0}), False)
    srv.shutdown()
    print(json.dumps(strip_private(r)), json.dumps(cmp_))
    assert r["decode"] > 0 and set(r["prefill"]) == {"128", "355"} and cmp_["text_same"] and cmp_["logprob_diff"] == 0.02
    print("measuring client ok")


def main():
    args = parse_args()
    if args.client_test:
        client_test()
        return
    if args.dry_run:
        dry_run(args)
        return
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    only_v = [x.strip() for x in args.only_variants.split(",") if x.strip()]
    os.makedirs(W, exist_ok=True)
    gpu, cc, mem, vram = gpu_info()
    arch = cc.replace(".", "")
    arch_n = int(arch)
    ncpu = os.cpu_count() or 2
    result = {"suite": "ab_matrix", "gpu": gpu, "compute_cap": cc, "memory": mem, "cpu_cores": ncpu, "command_line": " ".join(sys.argv[1:])}
    result["nvcc"] = sh("nvcc --version | tail -1", show=False)[1].strip()
    log(f"GPU: {gpu}, sm_{arch}, {mem}; {ncpu} CPU cores")
    coll = Collector(result)
    prepare_source_and_build(arch, ncpu, result)
    coll.save()
    log("build done")
    if not args.skip_selftest:
        run_selftests(arch, result)
        coll.save()
    refs = {}
    done = []
    for key, name, kind, repo in selected_models(args, arch_n, vram):
        plan = select_variants(kind, only_v, args.quick)
        if not plan:
            continue
        run_model(key, name, kind, plan, ncpu, lengths, coll)
        done.append((key, repo, kind))
        refs[key] = coll.result["models"].get(key, {}).get("variants", {}).get("V0")
    if not args.skip_vllm and arch_n >= 80:
        cases = [(k, repo, kind) for k, repo, kind in done if k in ("awq", "nvfp4") and repo]
        try:
            run_vllm(cases, ncpu, lengths, coll, refs)
        except Exception as ex:
            result["vllm"] = "error: " + str(ex)[:200]
    coll.save()
    print_result(result)


if __name__ == "__main__":
    main()
