# fLLM GPU test for Google Colab: paste this whole file into ONE code cell and run it (or: !python3 colab_gpu_test.py).
# It downloads the newest test build, runs every self-test group in its own process (one crash cannot spoil the others),
# runs a real model with and without each kernel switch, and prints one report. The report is also saved as /content/fllm_report.txt.
# Edit the CONFIG block if you want something else; everything else is automatic.
import json, os, re, subprocess, sys, time, urllib.request

# ---------------------------------------------------------------- CONFIG
RELEASE_URL = "https://github.com/Marco-Lu2508/fLLM/releases/download/colab-latest/fllm-colab-cuda.tar.gz"
WORKDIR = "/content/fl"
DEVICE = "cuda"
GROUPS = ["linear", "sampling", "host", "moe", "fuse", "prefill", "actdtype", "smallm", "w8a8", "fp8blk", "hattn", "hmla", "mlx", "hgemm", "kv"]
MODELS = [  # (hugging face repo, file) - the first one is also used for the switch matrix
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    # ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
    # ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q8_0.gguf"),
]
# environment switches tried one by one on the first model ("" = nothing changed)
SWITCHES = ["", "FLLM_CUDA_EXACT=1", "FLLM_CUDA_NO_TC=1", "FLLM_CUDA_NO_QGEMM=1", "FLLM_CUDA_NO_MMVQ=1", "FLLM_CUDA_NO_FLASH=1",
            "FLLM_CUDA_OLD_ATTN=1", "FLLM_CUDA_NO_GLU_FUSE=1", "FLLM_CUDA_NO_X16_REUSE=1", "FLLM_FUSE=0", "FLLM_CUDA_GRAPH=0"]
PROMPTS = ["Name three colors of the rainbow.",
           "Count from 1 to 30 separated by commas, then explain in two sentences why the sky is blue."]
RUN_SELFTEST = True
RUN_MODELS = True
RUN_SWITCH_MATRIX = True   # needs RUN_MODELS
# ----------------------------------------------------------------

report = []
def say(line=""):
    print(line, flush=True)
    report.append(line)

def sh(cmd, timeout=None, env=None):
    e = dict(os.environ); e.update(env or {})
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=e)
    return p.returncode, (p.stdout or "") + (p.stderr or "")

os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

# ---------------------------------------------------------------- GPU + binary
rc, out = sh("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader")
say("GPU: " + out.strip())
if not os.path.exists("fllm-rs"):
    say("downloading the test build ...")
    rc, out = sh(f"curl -sL --retry 5 -C - -o b.tgz {RELEASE_URL} && tar xzf b.tgz && chmod +x fllm-rs")
    if rc: say(out); sys.exit(1)
rc, out = sh("./fllm-rs --version 2>&1 | tail -n 2")
say("binary: " + out.strip().replace("\n", " | "))
rc, out = sh("ls -la --time-style=long-iso fllm-rs | awk '{print $6, $7}'")
say("binary built/downloaded: " + out.strip())

# ---------------------------------------------------------------- self test, group by group
if RUN_SELFTEST:
    say("\n=== SELF TEST (one process per group) ===")
    say(f"{'group':10} {'checks':>6} {'ok':>5} {'failed':>6} {'skipped':>7}   first failures")
    for g in GROUPS:
        try:
            rc, out = sh(f"./fllm-rs selftest-gpu --device {DEVICE} --failures-only --only {g} 2>&1", timeout=1800)
        except subprocess.TimeoutExpired:
            say(f"{g:10} TIMEOUT"); continue
        m = re.search(rf"^{g}\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", out, re.M)
        fails = [l for l in out.splitlines() if "FAIL" in l]
        if m:
            say(f"{g:10} {m.group(1):>6} {m.group(2):>5} {m.group(3):>6} {m.group(4):>7}")
        else:
            say(f"{g:10} (no table) " + out.strip().splitlines()[-1][:150] if out.strip() else f"{g:10} (no output)")
        for l in fails[:4]:
            say("      " + re.sub(r"\s+", " ", l)[:230])
        if len(fails) > 4:
            say(f"      ... {len(fails) - 4} more")

# ---------------------------------------------------------------- real models
def fetch(repo, name):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sh("pip -q install -U huggingface_hub")
        from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, name)

def serve_and_ask(model_path, env_switch, prompts, max_tokens=60):
    envs = {}
    if env_switch:
        k, v = env_switch.split("=", 1); envs[k] = v
    sh("pkill fllm-rs; sleep 2")
    log = "/tmp/fllm_serve.log"
    e = dict(os.environ); e.update(envs)
    proc = subprocess.Popen(f"./fllm-rs serve --model '{model_path}' --model-name m --device {DEVICE} --port 8000 > {log} 2>&1", shell=True, env=e)
    ok = False
    for _ in range(90):
        time.sleep(2)
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/ready", timeout=2); ok = True; break
        except Exception:
            if proc.poll() is not None: break
    results = []
    if not ok:
        tail = open(log).read()[-400:].replace("\n", " | ") if os.path.exists(log) else ""
        results.append(("(server did not start)", 0, tail))
    else:
        for p in prompts:
            body = json.dumps({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens, "temperature": 0}).encode()
            req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", body, {"content-type": "application/json"})
            try:
                r = json.load(urllib.request.urlopen(req, timeout=300))
                txt = r["choices"][0]["message"]["content"].replace("\n", " ")
                tps = r.get("timings", {}).get("predicted_per_second", 0)
                results.append((txt, tps, ""))
            except Exception as ex:
                results.append(("(request failed)", 0, str(ex)[:200]))
    sh("pkill fllm-rs; sleep 3")
    return results

if RUN_MODELS:
    say("\n=== MODELS ===")
    first_path = None
    for repo, name in MODELS:
        path = fetch(repo, name)
        if first_path is None: first_path = path
        say(f"-- {name}")
        for txt, tps, err in serve_and_ask(path, "", PROMPTS):
            say(f"   [{tps:5.1f} tok/s] {txt[:230]}" + (f"  {err}" if err else ""))
    if RUN_SWITCH_MATRIX and first_path:
        say(f"\n=== SWITCH MATRIX on {MODELS[0][1]} (first prompt only) ===")
        for sw in SWITCHES:
            res = serve_and_ask(first_path, sw, PROMPTS[:1], max_tokens=30)
            txt, tps, err = res[0]
            say(f"   {sw or '(default)':28} [{tps:5.1f} tok/s] {txt[:110]}" + (f"  {err}" if err else ""))

say("\nDONE. Send me everything above (or /content/fllm_report.txt).")
open("/content/fllm_report.txt", "w").write("\n".join(report))
