# fLLM CPU confirmation run for the Colab machine (x86, AVX-512, e.g. the G4 with EPYC 9B45 / Zen 5). Answers these questions:
#   1. every kernel of every tier against the f64 reference: `selftest-cpu --tiers all` (ragged shapes, saturation stress,
#      16-row VNNI GEMM called directly) and the same on the real tensors of Llama-3.2-3B Q4_K_M (outlier activations);
#   2. do the tiers agree? Per-text mean NLL (`fllm-rs ppl`, 10 texts of 160 tokens) per tier, next to the spread of
#      the SAME tier with other summation orders (FLLM_NO_R8=1, FLLM_NO_REPACK=1): that spread is the noise floor;
#   3. speed: prefill (355 tokens) and decode per tier for Llama-3.2-3B Q4_K_M and Qwen2.5-3B AWQ, with the build that
#      has the `avx512` feature and the build without it (to decide whether the feature should be on by default).
# Start:  python3 -u /content/cpu_confirm.py [--no-build] [--no-speed] [--threads N]       Result: the block "=== CPU2 ===" and /content/fllm_cpu2.json
# Needs: the source tree in /content/fllm_work/fllm (unpacked from fllm.tgz), the two models in /content/fllm_work
# (Llama-3.2-3B-Instruct-Q4_K_M.gguf is downloaded when missing; the AWQ gguf is made by `fllm-rs convert`).
import json, os, re, statistics, subprocess, sys, time, urllib.request

W = os.environ.get("FLLM_WORK", "/content/fllm_work")
SRC = os.environ.get("FLLM_SRC", f"{W}/fllm")
OUT = os.environ.get("FLLM_OUT", "/content/fllm_cpu2.json")
PORT = 8973
T0 = time.time()
NCPU = os.cpu_count() or 2
THREADS = max(1, NCPU // 2)  # vCPUs are hyperthreads
BUILD = "--no-build" not in sys.argv
SPEED_RUN = "--no-speed" not in sys.argv
if "--threads" in sys.argv:
    THREADS = int(sys.argv[sys.argv.index("--threads") + 1])
R = {}
LLAMA = f"{W}/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
AWQ = f"{W}/Qwen2.5-3B-Instruct-AWQ.gguf"
LLAMA_URL = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
SMALL = os.environ.get("FLLM_PPL_MODEL", LLAMA)  # model for the NLL comparison (any GGUF the CPU path runs)


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def sh(cmd, env=None, timeout=None):
    e = dict(os.environ)
    e["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + e["PATH"]
    if env:
        e.update(env)
    p = subprocess.run(["bash", "-lc", cmd], env=e, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


rc, cpuinfo = sh("grep -m1 flags /proc/cpuinfo")
flags = set(cpuinfo.split())
rc, out = sh("lscpu | grep 'Model name'")
R["cpu"] = out.strip()
R["threads"] = THREADS
R["flags"] = {f: (f in flags) for f in ["avx2", "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512_bf16", "amx_tile", "amx_int8", "avx_vnni"]}
log(f"{R['cpu']}; {NCPU} vCPUs, {THREADS} threads; {R['flags']}")

# ---- builds: with and without the avx512 feature -------------------------------------------------------------------------
BINS = {
    "feature": os.environ.get("FLLM_BIN_FEATURE", f"{W}/target-cpu2-feature/release/fllm-rs"),
    "default": os.environ.get("FLLM_BIN_DEFAULT", f"{W}/target-cpu2-default/release/fllm-rs"),
}
if BUILD:
    for name, flags_ in (("feature", "--no-default-features --features avx512"), ("default", "--no-default-features")):
        b = BINS[name]
        tgt = os.path.dirname(os.path.dirname(b))
        log(f"build {name} ({flags_}) ...")
        t = time.time()
        rc, o = sh(f"cd {SRC} && CARGO_TARGET_DIR={tgt} CARGO_BUILD_JOBS={NCPU} cargo build --release --locked {flags_} -p fllm-engine --bin fllm-rs 2>&1 | tail -n 3", timeout=3 * 3600)
        R.setdefault("build_seconds", {})[name] = round(time.time() - t)
        if not os.path.exists(b):
            sys.exit(f"build {name} failed:\n{o}")
if not os.path.exists(LLAMA):
    log("downloading Llama-3.2-3B-Instruct Q4_K_M ...")
    sh(f"mkdir -p {W} && curl -sL -o {LLAMA} {LLAMA_URL}", timeout=3600)
F = BINS["feature"]

# ---- tiers the CPU can run -----------------------------------------------------------------------------------------------
rc, o = sh(f"{F} selftest-cpu --tier scalar 2>&1 | grep '^supported tiers'")
TIERS = [t for t in o.split(":", 1)[1].split() if t not in ("scalar", "amx")] if ":" in o else ["avx2"]
R["tiers"] = TIERS
log(f"tiers: {TIERS}")


def tier_lines(out):
    return {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r"^tier (\S+)\s+(\d+) checks, (\d+) failed", out, re.M)}


# ---- 1. selftest ---------------------------------------------------------------------------------------------------------
for name, b in BINS.items():
    rc, out = sh(f"{b} selftest-cpu --tiers all 2>&1", timeout=3600)
    R.setdefault("selftest_all", {})[name] = {"exit": rc, "tiers": tier_lines(out), "fail_lines": [l[:160] for l in out.splitlines() if "FAIL" in l][:8]}
    log(f"selftest --tiers all ({name}): {R['selftest_all'][name]['tiers']} exit {rc}")
rc, out = sh(f"{F} selftest-cpu --tiers all --gguf {LLAMA} --tensors 400 2>&1", timeout=7200)
worst = {}
for m in re.finditer(r"^\[(\S+)\s*\] .* max err\s+(\S+)", out, re.M):
    worst[m.group(1)] = max(worst.get(m.group(1), 0.0), float(m.group(2)))
R["selftest_gguf"] = {"exit": rc, "tiers": tier_lines(out), "worst_error": worst, "fail_lines": [l[:160] for l in out.splitlines() if "FAIL" in l][:8]}
log(f"selftest --gguf: {R['selftest_gguf']}")

# ---- 2. agreement: mean NLL per text -------------------------------------------------------------------------------------
docs = sorted(f for f in os.listdir(f"{SRC}/docs") if f.endswith(".md") and os.path.getsize(f"{SRC}/docs/{f}") > 5000)[:10]
os.makedirs(f"{W}/ppl_texts", exist_ok=True)
texts = []
for i, f in enumerate(docs):
    data = open(f"{SRC}/docs/{f}", "rb").read()[:3200][-700:]
    p = f"{W}/ppl_texts/t{i + 1}.txt"
    open(p, "wb").write(data)
    texts.append(p)


def nll(tier, text, extra=None, binary=F):
    env = {"FLLM_CPU_ISA": tier}
    if extra:
        env.update(extra)
    rc, o = sh(f"{binary} ppl --model {SMALL} --file {text} --ctx 160 2>&1 | tail -n 1", env=env, timeout=1800)
    m = re.search(r"avg_nll=([0-9.]+)", o)
    return float(m.group(1)) if m else None


NLL = {}
for tier in TIERS:
    NLL[tier] = [nll(tier, t) for t in texts]
    log(f"nll {tier}: {NLL[tier]}")
base = TIERS[0]
for label, extra in (("avx2 NO_R8", {"FLLM_NO_R8": "1"}), ("avx2 NO_REPACK", {"FLLM_NO_REPACK": "1"})):
    NLL[label] = [nll("avx2", t, extra) for t in texts]
    log(f"nll {label}: {NLL[label]}")
rows = {}
for label, v in NLL.items():
    if label == base or None in v or None in NLL[base]:
        continue
    d = [a - b for a, b in zip(v, NLL[base])]
    rows[label] = {"mean_diff_vs_" + base: round(statistics.mean(d), 4), "sd_per_text": round(statistics.pstdev(d), 4), "max_abs": round(max(abs(x) for x in d), 4)}
R["nll"] = {"model": os.path.basename(SMALL), "per_text": NLL, "difference_to_" + base: rows}
log(f"nll differences: {rows}")

# ---- 3. speed ------------------------------------------------------------------------------------------------------------


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


def ttft(n):
    p = f"{int(time.time() * 1000) % 10**9} " + "hello " * n
    t = time.time()
    for line in post({"prompt": p, "max_tokens": 1, "temperature": 0, "stream": True, "ignore_eos": True}):
        if line.startswith(b"data: {"):
            return time.time() - t


def decode_rate(n=48):
    t, c = None, 0
    for line in post({"prompt": "hello " * 8, "max_tokens": n, "temperature": 0, "stream": True, "ignore_eos": True}):
        if line.startswith(b"data: {"):
            c += 1
            t = t or time.time()
    return (c - 1) / (time.time() - t) if c > 1 else 0


def kill_servers():
    """Stop servers of this script (they listen on PORT); other fllm-rs processes are left alone."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\x00")
        except OSError:
            continue
        if cmd and cmd[0].endswith(b"fllm-rs") and b"serve" in cmd and str(PORT).encode() in cmd:
            os.kill(int(pid), 15)


def serve(binary, model, tier):
    kill_servers()
    time.sleep(2)
    e = dict(os.environ)
    e["FLLM_CPU_ISA"] = tier
    cmd = [binary, "serve", "--model", model, "--port", str(PORT), "--ctx", "2048", "--max-seqs", "1", "--threads", str(THREADS), "--device", "cpu"]
    p = subprocess.Popen(cmd, stdout=open(f"{W}/cpu2_serve.log", "w"), stderr=subprocess.STDOUT, env=e)
    for _ in range(900):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return p
        except Exception:
            if p.poll() is not None:
                break
    raise RuntimeError("server does not start: " + open(f"{W}/cpu2_serve.log").read()[-300:])


SPEED = []
for mname, model in (("Llama-3.2-3B Q4_K_M", LLAMA), ("Qwen2.5-3B AWQ", AWQ)):
    if not SPEED_RUN or not os.path.exists(model):
        continue
    for bname, b in BINS.items():
        for tier in TIERS:
            row = {"model": mname, "build": bname, "tier": tier}
            proc = None
            try:
                proc = serve(b, model, tier)
                ttft(20)
                row["prefill_355_tok_s"] = round(358 / min(ttft(355) for _ in range(3)), 1)
                row["decode_tok_s"] = round(max(decode_rate() for _ in range(3)), 2)
                d = json.load(post({"prompt": "The capital of France is", "max_tokens": 16, "temperature": 0, "logprobs": 5}))
                row["text"] = d["choices"][0]["text"].replace("\n", " ")[:60]
            except Exception as ex:
                row["error"] = str(ex)[:200]
            if proc:
                proc.terminate()
                try:
                    proc.wait(30)
                except Exception:
                    proc.kill()
            SPEED.append(row)
            log(str(row))
R["speed"] = SPEED
kill_servers()
R["seconds"] = round(time.time() - T0)
print("\n=== CPU2 ===")
print(json.dumps(R, indent=1, ensure_ascii=False))
open(OUT, "w").write(json.dumps(R, indent=1, ensure_ascii=False))
print("=== END ===")
