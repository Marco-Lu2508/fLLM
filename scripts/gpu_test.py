#!/usr/bin/env python3
# fLLM GPU test for a Linux machine with Python 3 and curl (a CUDA or Vulkan GPU; the CPU is the reference).
#   python3 scripts/gpu_test.py --device cuda|vulkan [--workdir ~/fl] [--tag NAME] [--models small|big|all] [--no-selftest] [--no-cpu-ref] [--bin ./fllm-rs]
# It runs the self test, serves real models on the device and on the CPU (temperature 0), compares the texts and writes
# <workdir>/report_<tag>.txt. Everything runs in the foreground; use nohup/tmux for a long run.
import argparse, json, os, re, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--workdir", default=os.path.expanduser("~/fl"))
ap.add_argument("--tag", default=None)
ap.add_argument("--models", default="small", choices=["small", "big", "all"])
ap.add_argument("--no-selftest", action="store_true")
ap.add_argument("--no-cpu-ref", action="store_true")
ap.add_argument("--binary-url", default="https://github.com/Marco-Lu2508/fLLM/releases/download/colab-latest/fllm-colab-cuda.tar.gz")
ap.add_argument("--bin", default="./fllm-rs", help="binary to test (default: the downloaded colab build)")
ap.add_argument("--extra-env", default="", help="KEY=VAL,KEY=VAL for the GPU server runs")
args = ap.parse_args()
DEV = args.device
BIN = args.bin
TAG = args.tag or DEV
W = os.path.abspath(args.workdir)
os.makedirs(W, exist_ok=True); os.chdir(W)

MODELS_SMALL = [("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")]
MODELS_BIG = [("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
              ("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf")]
MODELS = {"small": MODELS_SMALL, "big": MODELS_BIG, "all": MODELS_SMALL + MODELS_BIG}[args.models]
PROMPTS = ["Name three colors of the rainbow.",
           "Count from 1 to 30 separated by commas, then explain in two sentences why the sky is blue.",
           "Write a Python function that returns the n-th Fibonacci number, with a one-line docstring."]
LONG = " ".join(["The quick brown fox jumps over the lazy dog."] * 90)   # ~900 tokens, measures prompt speed

report = []
def say(line=""):
    print(line, flush=True); report.append(line)
def sh(cmd, timeout=None, env=None):
    e = dict(os.environ); e.update(env or {})
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=e)
    return p.returncode, (p.stdout or "") + (p.stderr or "")
def save():
    open(f"{W}/report_{TAG}.txt", "w").write("\n".join(report))

say(f"=== fLLM test  device={DEV}  host={os.uname().nodename}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
say(sh("lscpu | grep 'Model name' | sed 's/  */ /g'")[1].strip())
say(sh("free -h | sed -n 2p")[1].strip())
if DEV == "cuda": say(sh("nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader")[1].strip())
else: say(sh("vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverInfo' | head -4")[1].strip())

if args.bin == "./fllm-rs" and not os.path.exists("fllm-rs"):
    say("downloading the build ...")
    rc, out = sh(f"curl -sL --retry 5 -C - -o b.tgz {args.binary_url} && tar xzf b.tgz && chmod +x fllm-rs")
    if rc: say(out); save(); sys.exit(1)
say(sh(f"{BIN} --version 2>&1 | tail -n 1")[1].strip())
say("devices: " + sh(f"{BIN} devices 2>&1 | grep -v -i openvino | head -n 12")[1].strip().replace("\n", " | "))

# ------------------------------------------------------------------ self test
if not args.no_selftest:
    say("\n=== SELF TEST ===")
    if DEV == "cuda":
        groups = ["linear", "sampling", "host", "moe", "fuse", "prefill", "actdtype", "smallm", "w8a8", "fp8blk", "hattn", "hmla", "mlx", "hgemm", "kv"]
        say(f"{'group':10} summary")
        for g in groups:
            try:
                rc, out = sh(f"{BIN} selftest-gpu --device {DEV} --failures-only --only {g} 2>&1", timeout=1800)
            except subprocess.TimeoutExpired:
                say(f"{g:10} TIMEOUT"); continue
            ms = re.findall(r"(\d+) checks, (\d+) failed, (\d+) skipped \(unsupported\), (\d+) ok", out)
            say(f"{g:10} " + (" | ".join(f"{a} checks {b} failed {c} skipped {d} ok" for a, b, c, d in ms if int(a)) or (out.strip().splitlines()[-1][:150] if out.strip() else "(no output)")))
            for l in [l for l in out.splitlines() if "FAIL" in l][:4]:
                say("      " + re.sub(r"\s+", " ", l)[:230])
    else:
        try:
            rc, out = sh(f"{BIN} selftest-gpu --device {DEV} --failures-only 2>&1", timeout=7200)
        except subprocess.TimeoutExpired:
            out = "TIMEOUT"
        lines = out.splitlines()
        fails = [l for l in lines if "FAIL" in l]
        say("\n".join(lines[-22:]))
        for l in fails[:25]:
            say("   " + re.sub(r"\s+", " ", l)[:230])
        if len(fails) > 25: say(f"   ... {len(fails) - 25} more failures")
    save()

# ------------------------------------------------------------------ models
def fetch(repo, name):
    os.makedirs(f"{W}/models", exist_ok=True)
    path = f"{W}/models/{name}"
    rc, out = sh(f"curl -sL --retry 8 --retry-delay 3 -C - -o '{path}' 'https://huggingface.co/{repo}/resolve/main/{name}'")
    if rc: say(f"   download failed: {out[-200:]}")
    return path

def serve(model, device, env, port=8000):
    sh("pkill -f '[f]llm-rs.* serve'; sleep 2")
    log = f"{W}/serve_{TAG}.log"
    e = dict(os.environ); e.update(env)
    proc = subprocess.Popen(f"{BIN} serve --model '{model}' --model-name m --device {device} --port {port} > {log} 2>&1", shell=True, env=e)
    for _ in range(300):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=2); return proc, log
        except Exception:
            if proc.poll() is not None: break
    return None, log

def ask(prompt, max_tokens=80, port=8000):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    t = r.get("timings", {})
    return r["choices"][0]["message"]["content"], t.get("prompt_per_second", 0), t.get("predicted_per_second", 0)

def run_model(model, device, env):
    proc, log = serve(model, device, env)
    if proc is None:
        return None, open(log).read()[-500:].replace("\n", " | ")
    res = []
    try:
        for p in PROMPTS:
            res.append(ask(p))
        _, pps, _ = ask(LONG, max_tokens=8)
        res.append(("(long prompt)", pps, 0))
    except Exception as ex:
        res.append((f"(request failed: {ex})", 0, 0))
    sh("pkill -f '[f]llm-rs.* serve'; sleep 3")
    return res, ""

def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

genv = dict(kv.split("=", 1) for kv in args.extra_env.split(",") if "=" in kv)
say("\n=== MODELS ===")
for repo, name in MODELS:
    path = fetch(repo, name)
    say(f"-- {name}")
    gpu, err = run_model(path, DEV, genv)
    if gpu is None:
        say("   GPU server did not start: " + err); save(); continue
    cpu = None
    if not args.no_cpu_ref:
        cpu, err = run_model(path, "cpu", {})
    for i, p in enumerate(PROMPTS):
        t, pps, tps = gpu[i]
        line = f"   [{tps:5.1f} tok/s] {t.replace(chr(10), ' ')[:150]}"
        if cpu and cpu[i][0] and not cpu[i][0].startswith("(request"):
            ct = cpu[i][0]
            n = common_prefix(t, ct)
            line += f"\n      vs CPU: " + ("IDENTICAL" if t == ct else f"first {n} of {len(ct)} chars equal  (cpu: {ct.replace(chr(10), ' ')[:110]})")
        say(line)
    say(f"   prompt speed (~900 tokens): {gpu[-1][1]:.0f} tok/s" + (f"   (cpu: {cpu[-1][1]:.0f} tok/s, decode {cpu[0][2]:.1f} tok/s)" if cpu else ""))
    save()

say("\nDONE")
save()
